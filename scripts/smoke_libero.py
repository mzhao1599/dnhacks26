#!/usr/bin/env python
"""Smoke test: LIBERO episodes with the zero policy or SmolVLA.

    python scripts/smoke_libero.py --suite libero_10 --task 0 --seed 0 --steps 100 --policy zero
    python scripts/smoke_libero.py --suite libero_10 --task 0 --seed 0 --steps 520 --policy smolvla
    python scripts/smoke_libero.py --suite libero_goal --task 0 --seeds 0,1,2,3 --steps 300 --policy smolvla

Prints obs keys / shapes, per-step timing, success_flag, object names, per-episode wall time and
action statistics; saves first/last agentview frames to logs/smoke_<suite>_<task>_<policy>_{first,last}.png.
Ends with one `SMOKE_RESULT {...json...}` line per episode and `SMOKE_OK`.
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


def action_stats(acts: np.ndarray) -> dict:
    a = np.asarray(acts, dtype=np.float32).reshape(-1, 7)
    if len(a) == 0:
        return {}
    d = np.diff(a[:, :6], axis=0) if len(a) > 1 else np.zeros((1, 6), dtype=np.float32)
    return {
        "n": int(len(a)),
        "mean": np.round(a.mean(0), 3).tolist(),
        "std": np.round(a.std(0), 3).tolist(),
        "min": np.round(a.min(0), 3).tolist(),
        "max": np.round(a.max(0), 3).tolist(),
        "abs_mean_xyz": round(float(np.abs(a[:, :3]).mean()), 3),
        "abs_mean_rot": round(float(np.abs(a[:, 3:6]).mean()), 3),
        "frac_saturated": round(float((np.abs(a[:, :6]) > 0.99).mean()), 3),
        "frac_gripper_closed": round(float((a[:, 6] > 0).mean()), 3),
        "n_gripper_toggles": int((np.diff(np.sign(a[:, 6])) != 0).sum()),
        "jerk_proxy": round(float(np.abs(d).mean()), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--task", default="0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default=None, help="comma-separated seeds; overrides --seed")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--policy", choices=["smolvla", "zero"], default="zero")
    ap.add_argument("--policy-path", default="HuggingFaceVLA/smolvla_libero")
    ap.add_argument("--n-action-steps", type=int, default=None)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--render-gl", default=os.environ.get("MUJOCO_GL", "egl"))
    ap.add_argument("--jitter", type=float, default=0.0, help="pose jitter (m) via reset(perturbation)")
    ap.add_argument("--shift-xy", default=None, help="dx,dy (m) shift of the first object of interest")
    ap.add_argument("--distractor", action="store_true")
    ap.add_argument("--logdir", default=str(ROOT / "logs"))
    args = ap.parse_args()

    os.environ["MUJOCO_GL"] = args.render_gl
    from fleet_memory.envs.libero_env import LiberoEnv, list_tasks
    from fleet_memory.policies.smolvla import make_policy

    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]

    perturbation = None
    if args.jitter > 0 or args.shift_xy or args.distractor:
        perturbation = {}
        if args.jitter > 0:
            perturbation["jitter_m"] = args.jitter
        if args.shift_xy:
            perturbation["shift_xy"] = [float(x) for x in args.shift_xy.split(",")]
        if args.distractor:
            perturbation["distractor"] = True

    t0 = time.time()
    print(f"[smoke] MUJOCO_GL={os.environ['MUJOCO_GL']} suite={args.suite} task={args.task} seeds={seeds}")
    tasks = list_tasks(args.suite)
    print(f"[smoke] {len(tasks)} tasks in {args.suite}:")
    for i, slug, lang in tasks:
        print(f"   {i:2d} {slug}  |  {lang}")

    env = LiberoEnv(args.suite, args.task, image_size=args.image_size, render_gl=args.render_gl)
    obs = env.reset(seeds[0])
    print(f"[smoke] env ready in {time.time()-t0:.1f}s")
    if perturbation:
        before = {k: v[0].copy() for k, v in obs.object_poses.items()}
        obs = env.reset(seeds[0], perturbation=perturbation)
        print(f"[smoke] perturbation {perturbation} applied: {env.last_perturbation}")
        for k, v in obs.object_poses.items():
            d = v[0] - before[k]
            if np.abs(d).max() > 1e-4:
                print(f"[smoke]   moved {k}: {np.round(before[k],3)} -> {np.round(v[0],3)} (d={np.round(d,3)})")

    ti = env.task_info()
    print(f"[smoke] task_info: suite={ti.suite} task_id={ti.task_id} family={ti.task_family} max_steps={ti.max_steps}")
    print(f"[smoke] language: {ti.language!r}")
    print(f"[smoke] objects ({len(ti.objects)}): {ti.objects}")
    print(f"[smoke] obj_of_interest: {env.objects_of_interest}")
    print(f"[smoke] envelope: {env.envelope()}")
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
    print(f"[smoke] policy {policy.name} ready in {time.time()-tp:.1f}s")
    if args.policy == "smolvla":
        print(f"[smoke] n_action_steps={policy.n_action_steps} chunk_size={policy.chunk_size} device={policy.device}")

    results = []
    for ep_i, seed in enumerate(seeds):
        if ep_i > 0 or perturbation is None:
            obs = env.reset(seed, perturbation=perturbation)
        policy.reset(ti.language)
        te = time.time()
        step_times, act_times, acts = [], [], []
        done, t, first_success_t = False, 0, None
        while t < args.steps and not done:
            ta = time.time()
            chunk = policy.act(obs)
            act_times.append(time.time() - ta)
            if t == 0 and ep_i == 0:
                print(f"[smoke] first chunk shape={chunk.shape} dtype={chunk.dtype} row0={np.round(chunk[0], 3)}")
            for a in chunk:
                ts = time.time()
                obs, done, info = env.step(a)
                step_times.append(time.time() - ts)
                acts.append(np.asarray(a, dtype=np.float32))
                t += 1
                if info["is_success"] and first_success_t is None:
                    first_success_t = t
                if t % 40 == 0:
                    print(f"[smoke] seed={seed} t={t} act={np.round(a,2)} ee={np.round(obs.ee_pos,3)} succ={info['is_success']} "
                          f"sub={[int(v) for v in env.success_subconditions().values()]} env_step={np.mean(step_times[-40:])*1000:.1f}ms")
                if done or t >= args.steps:
                    break
        success = env.success_flag()
        wall = time.time() - te
        save_png(logdir / f"smoke_{args.suite}_{env.task_idx}_{args.policy}_last.png", obs.images["agentview"])
        st = action_stats(np.stack(acts)) if acts else {}
        res = {
            "suite": args.suite, "task": env.task_idx, "slug": env.slug, "seed": seed, "steps": t,
            "success": bool(success), "first_success_t": first_success_t,
            "subconds": env.success_subconditions(),
            "wall_s": round(wall, 1), "env_step_ms": round(float(np.mean(step_times)) * 1000, 1),
            "policy_act_ms": round(float(np.mean(act_times)) * 1000, 1),
            "per_step_ms": round(wall / max(t, 1) * 1000, 1),
            "action_stats": st,
        }
        results.append(res)
        print(f"[smoke] DONE seed={seed} steps={t} success_flag={success} first_success_t={first_success_t} "
              f"subconds={res['subconds']}")
        print(f"[smoke] env.step mean={res['env_step_ms']}ms policy.act mean={res['policy_act_ms']}ms "
              f"(n={len(act_times)}) episode wall={wall:.1f}s ({res['per_step_ms']} ms/step)")
        print(f"[smoke] action_stats: {json.dumps(st)}")
        print("SMOKE_RESULT " + json.dumps(res))

    n_ok = sum(r["success"] for r in results)
    print(f"[smoke] SUMMARY {args.suite}/{env.task_idx} success {n_ok}/{len(results)} "
          f"mean_wall={np.mean([r['wall_s'] for r in results]):.1f}s total={time.time()-t0:.1f}s")
    env.close()
    print("SMOKE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
