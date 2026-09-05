#!/usr/bin/env python
"""Verify LiberoEnv.reset(seed, perturbation) on a GPU node: pose actually changes, sim stays stable.

    python scripts/verify_perturb.py --suite libero_10 --task 0 --seeds 0,1,2 --settle-steps 60

For each seed and each perturbation spec, compares object positions against the unperturbed reset,
then runs no-op steps and checks that moved objects stay at (roughly) table height (no fall-through,
|dz| < 3 cm) and do not drift more than 2 cm in xy. Prints PERTURB_OK / PERTURB_FAIL.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

SPECS = [
    {"shift_xy": [0.03, 0.03]},
    {"shift_xy": [-0.05, 0.0]},
    {"jitter_m": 0.02},
    {"distractor": True},
    {"shift_xy": [0.02, -0.02], "jitter_m": 0.01, "distractor": True},
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--task", default="0")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--settle-steps", type=int, default=60)
    args = ap.parse_args()
    from fleet_memory.envs.libero_env import LiberoEnv, DUMMY_ACTION

    env = LiberoEnv(args.suite, args.task)
    ti = env.task_info()
    print(f"[verify] {ti.suite}/{env.task_idx} {ti.language!r}")
    print(f"[verify] objects (ooi first): {ti.objects}")
    print(f"[verify] obj_of_interest: {env.objects_of_interest}")
    print(f"[verify] envelope: {env.envelope()}")
    assert ti.objects[: len(env.objects_of_interest)] == env.objects_of_interest, "ooi must come first"

    failures = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        base = env.reset(seed)
        base_pos = {k: v[0].copy() for k, v in base.object_poses.items()}
        # determinism of the unperturbed reset
        again = env.reset(seed)
        for k, v in again.object_poses.items():
            if np.abs(v[0] - base_pos[k]).max() > 1e-5:
                failures.append(f"seed={seed} reset not deterministic for {k}")
        for spec in SPECS:
            obs = env.reset(seed, perturbation=spec)
            applied = env.last_perturbation
            moved = {}
            for k, v in obs.object_poses.items():
                d = v[0] - base_pos[k]
                if np.abs(d[:2]).max() > 1e-3:
                    moved[k] = d
            # settle and check stability
            p0 = {k: v[0].copy() for k, v in obs.object_poses.items()}
            for _ in range(args.settle_steps):
                obs, _, _ = env.step(DUMMY_ACTION.copy())
            drift = {}
            for k in moved:
                d = obs.object_poses[k][0] - p0[k]
                drift[k] = d
                if abs(d[2]) > 0.03 or np.linalg.norm(d[:2]) > 0.02:
                    failures.append(f"seed={seed} spec={spec} {k} unstable after settle: d={np.round(d,4)}")
            if not moved and spec:
                failures.append(f"seed={seed} spec={spec} moved nothing (applied={applied})")
            if "shift_xy" in spec and "jitter_m" not in spec:
                tgt = env.objects_of_interest[0]
                d = moved.get(tgt)
                if d is None or np.abs(d[:2] - np.array(spec["shift_xy"])).max() > 0.005:
                    failures.append(f"seed={seed} shift_xy on {tgt}: expected {spec['shift_xy']} got {None if d is None else np.round(d[:2],4)}")
            print(f"[verify] seed={seed} spec={json.dumps(spec)} moved={{{', '.join(f'{k}: {np.round(v,3).tolist()}' for k,v in moved.items())}}} "
                  f"drift_after_{args.settle_steps}={{{', '.join(f'{k}: {np.round(v,4).tolist()}' for k,v in drift.items())}}} "
                  f"success={env.success_flag()}")
            print(f"[verify]   applied={json.dumps(applied)}")
    env.close()
    if failures:
        print("PERTURB_FAIL")
        for f in failures:
            print("  -", f)
        return 1
    print("PERTURB_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
