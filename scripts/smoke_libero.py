#!/usr/bin/env python
"""Smoke test: one LIBERO episode with the zero policy or SmolVLA.

    python scripts/smoke_libero.py --suite libero_10 --task 0 --seed 0 --steps 100 --policy zero
    python scripts/smoke_libero.py --suite libero_10 --task 0 --seed 0 --steps 100 --policy smolvla

Prints obs keys / shapes, per-step timing, success_flag, object names; saves first/last
agentview frames to logs/smoke_<suite>_<task>_<policy>_{first,last}.png.
Run on a GPU node only (never on the login node).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402


def save_png(path: Path, img: np.ndarray) -> None:
    try:
        from PIL import Image
        Image.fromarray(img).save(path)
    except Exception as e:  # pragma: no cover
        np.save(path.with_suffix(".npy"), img)
        print(f"[warn] PIL save failed ({e}); wrote {path.with_suffix('.npy')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--task", default="0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--policy", choices=["smolvla", "zero"], default="zero")
    ap.add_argument("--policy-path", default="HuggingFaceVLA/smolvla_libero")
    ap.add_argument("--n-action-steps", type=int, default=None)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--render-gl", default=os.environ.get("MUJOCO_GL", "egl"))
    ap.add_argument("--jitter", type=float, default=0.0, help="pose jitter (m) via perturb.Perturbation")
    ap.add_argument("--logdir", default=str(ROOT / "logs"))
    args = ap.parse_args()

    os.environ["MUJOCO_GL"] = args.render_gl
    from fleet_memory.envs.libero_env import LiberoEnv, list_tasks
    from fleet_memory.policies.smolvla import make_policy

    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[smoke] MUJOCO_GL={os.environ['MUJOCO_GL']} suite={args.suite} task={args.task}")
    tasks = list_tasks(args.suite)
    print(f"[smoke] {len(tasks)} tasks in {args.suite}:")
    for i, slug, lang in tasks:
        print(f"   {i:2d} {slug}  |  {lang}")

    env = LiberoEnv(args.suite, args.task, image_size=args.image_size, render_gl=args.render_gl)
    obs = env.reset(args.seed)
    print(f"[smoke] env ready in {time.time()-t0:.1f}s")
    if args.jitter > 0:
        from fleet_memory.envs.perturb import Perturbation
        p = Perturbation(distractor=False, pose_jitter_m=args.jitter, seed=args.seed)
        obs = p.apply(env)
        print(f"[smoke] perturbation applied: {p.applied}")

    ti = env.task_info()
    print(f"[smoke] task_info: suite={ti.suite} task_id={ti.task_id} family={ti.task_family} max_steps={ti.max_steps}")
    print(f"[smoke] language: {ti.language!r}")
    print(f"[smoke] objects ({len(ti.objects)}): {ti.objects}")
    print(f"[smoke] obj_of_interest: {env.objects_of_interest}")
    print(f"[smoke] raw obs keys: {sorted(obs.raw.keys())}")
    print(f"[smoke] images: " + ", ".join(f"{k}={v.shape}/{v.dtype}" for k, v in obs.images.items()))
    print(f"[smoke] state dim={obs.state.shape} {obs.state}")
    print(f"[smoke] ee_pos={obs.ee_pos} ee_quat={obs.ee_quat} gripper={obs.gripper_qpos}")
    print(f"[smoke] object_poses: " + json.dumps({k: [round(float(x), 3) for x in v[0]] for k, v in obs.object_poses.items()}))
    print(f"[smoke] success_flag at reset: {env.success_flag()}  subconds: {env.success_subconditions()}")
    save_png(logdir / f"smoke_{args.suite}_{env.task_idx}_{args.policy}_first.png", obs.images["agentview"])
    save_png(logdir / f"smoke_{args.suite}_{env.task_idx}_{args.policy}_wrist_first.png", obs.images["eye_in_hand"])

    tp = time.time()
    pol_kw = {}
    if args.policy == "smolvla":
        pol_kw = dict(path=args.policy_path, device="cuda", n_action_steps=args.n_action_steps, image_size=args.image_size)
    policy = make_policy(args.policy, **pol_kw)
    policy.reset(ti.language)
    print(f"[smoke] policy {policy.name} ready in {time.time()-tp:.1f}s")
    if args.policy == "smolvla":
        print(f"[smoke] n_action_steps={policy.n_action_steps} chunk_size={policy.chunk_size} device={policy.device}")

    step_times, act_times = [], []
    done, success, t = False, False, 0
    while t < args.steps and not done:
        ta = time.time()
        chunk = policy.act(obs)
        act_times.append(time.time() - ta)
        if t == 0:
            print(f"[smoke] first chunk shape={chunk.shape} dtype={chunk.dtype} row0={np.round(chunk[0], 3)}")
        for a in chunk:
            ts = time.time()
            obs, done, info = env.step(a)
            step_times.append(time.time() - ts)
            t += 1
            if t % 20 == 0:
                print(f"[smoke] t={t} act={np.round(a,2)} ee={np.round(obs.ee_pos,3)} succ={info['is_success']} "
                      f"env_step={np.mean(step_times[-20:])*1000:.1f}ms")
            if done or t >= args.steps:
                break
    success = env.success_flag()
    save_png(logdir / f"smoke_{args.suite}_{env.task_idx}_{args.policy}_last.png", obs.images["agentview"])
    print(f"[smoke] DONE steps={t} success_flag={success} subconds={env.success_subconditions()}")
    print(f"[smoke] env.step mean={np.mean(step_times)*1000:.1f}ms  policy.act mean={np.mean(act_times)*1000:.1f}ms "
          f"(n={len(act_times)}), total wall={time.time()-t0:.1f}s")
    env.close()
    print("SMOKE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
