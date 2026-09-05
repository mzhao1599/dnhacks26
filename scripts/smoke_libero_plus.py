#!/usr/bin/env python
"""LIBERO-Plus smoke: N seeds of one task, standard reset vs a perturbed config, any policy.

    python scripts/smoke_libero_plus.py --suite libero_spatial --task 0 --seeds 0,1,2,3,4 \
        --policy smolvla --config standard,robot_init:0 --steps 220
    python scripts/smoke_libero_plus.py --policy zero --config robot_init:init_state=131 --steps 30
    python scripts/smoke_libero_plus.py --policy pi05 --config standard --seeds 0,1,2 --steps 220

--config is a comma-separated list; each entry is ``standard`` | ``robot_init[:idx]`` (idx-th
LIBERO-Plus robot-init config of this task, default 0) | ``robot_init:init_state=N`` (MountedPanda N)
| ``robot_init:radius=R`` (fresh seeded sample of joint-norm R) | ``layout[:idx]``.
Prints, per episode: success, steps, initial ee_pos/ee_quat (standard reset and after the
perturbation), per-step policy latency; per config: mean success. Machine-readable lines:
``PLUS_RESULT {json}`` per episode, ``PLUS_SUMMARY {json}`` per config, ``PLUS_OK`` at the end.
GPU node only (never the login node).
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


def parse_config(spec: str, suite: str, task_idx: int) -> dict | str:
    from fleet_memory.envs.libero_plus import list_configs
    if spec == "standard":
        return "standard"
    dim, _, arg = spec.partition(":")
    if "=" in arg:
        k, v = arg.split("=", 1)
        return {"dimension": dim, k: (int(v) if k == "init_state" else float(v)), "label": spec}
    cfgs = [c for c in list_configs(dim, suite) if c["base_task_idx"] == task_idx]
    if not cfgs:
        raise SystemExit(f"no {dim} configs for {suite} task {task_idx}")
    cfg = dict(cfgs[int(arg or 0)])
    cfg["label"] = spec
    return cfg


def gpu_mem() -> dict:
    try:
        import torch
        if not torch.cuda.is_available():
            return {}
        return {"gpu_alloc_gb": round(torch.cuda.memory_allocated() / 2**30, 2),
                "gpu_max_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "gpu_reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 2)}
    except Exception:
        return {}


def make_policy(name: str, args):
    if name == "zero":
        from fleet_memory.policies.smolvla import ZeroPolicy
        return ZeroPolicy()
    if name == "smolvla":
        from fleet_memory.policies.smolvla import SmolVLAPolicy
        return SmolVLAPolicy(path=args.policy_path or "HuggingFaceVLA/smolvla_libero", device="cuda",
                             n_action_steps=args.n_action_steps, image_size=args.image_size)
    if name == "pi05":
        from fleet_memory.policies.pi05 import Pi05Policy
        return Pi05Policy(path=args.policy_path or "lerobot/pi05_libero", device="cuda",
                          n_action_steps=args.n_action_steps, image_size=args.image_size, dtype=args.dtype)
    raise ValueError(name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task", default="0")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--steps", type=int, default=220)
    ap.add_argument("--policy", choices=["zero", "smolvla", "pi05"], default="zero")
    ap.add_argument("--policy-path", default=None)
    ap.add_argument("--n-action-steps", type=int, default=None)
    ap.add_argument("--dtype", default=None, help="pi05 only: float32 (checkpoint default) | bfloat16")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--config", default="standard,robot_init:0")
    ap.add_argument("--print-every", type=int, default=20)
    ap.add_argument("--render-gl", default=os.environ.get("MUJOCO_GL", "egl"))
    ap.add_argument("--logdir", default=str(ROOT / "logs" / "plus"))
    args = ap.parse_args()
    os.environ["MUJOCO_GL"] = args.render_gl

    from fleet_memory.envs.libero_plus import base_task_names, is_plus_backend, make_perturbed_env
    seeds = [int(s) for s in args.seeds.split(",")]
    bases = base_task_names(args.suite)
    task_idx = int(args.task) if str(args.task).isdigit() else next(
        i for i, n in enumerate(bases) if n.startswith(args.task))
    print(f"[plus] backend={'plus' if is_plus_backend() else 'native'} MUJOCO_GL={os.environ['MUJOCO_GL']} "
          f"suite={args.suite} task={task_idx} ({bases[task_idx]}) seeds={seeds} steps={args.steps} policy={args.policy}")
    Path(args.logdir).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    policy = make_policy(args.policy, args)
    print(f"[plus] policy {policy.name} ready in {time.time()-t0:.1f}s "
          f"n_action_steps={getattr(policy, 'n_action_steps', 1)} {gpu_mem()}")

    summaries = []
    for spec in args.config.split(","):
        cfg = parse_config(spec.strip(), args.suite, task_idx)
        env = make_perturbed_env(args.suite, task_idx, cfg, image_size=args.image_size, render_gl=args.render_gl)
        ti = env.task_info()
        if isinstance(cfg, dict):
            print(f"[plus] config {spec}: " + json.dumps({k: v for k, v in cfg.items() if k != "init_qpos"}))
        results = []
        for seed in seeds:
            obs = env.reset(seed)
            rp = env.last_perturbation.get("robot_init") if isinstance(cfg, dict) else None
            if rp:
                print(f"[plus] seed={seed} init_state={rp['init_state']} standard ee_pos={rp['ee_pos_before']} "
                      f"ee_quat={rp['ee_quat_before']} -> perturbed ee_pos={rp['ee_pos_after']} "
                      f"ee_quat={rp['ee_quat_after']} (shift {rp['ee_shift_m']} m, qpos {rp['qpos_before']} -> {rp['qpos_after']})")
            else:
                print(f"[plus] seed={seed} initial ee_pos={np.round(obs.ee_pos, 4).tolist()} "
                      f"ee_quat={np.round(obs.ee_quat, 4).tolist()}")
            init_pos, init_quat = obs.ee_pos.copy(), obs.ee_quat.copy()
            policy.reset(ti.language)
            te, act_ms, t, done, first_t = time.time(), [], 0, False, None
            while t < args.steps and not done:
                ta = time.time()
                chunk = policy.act(obs)
                act_ms.append((time.time() - ta) * 1000)
                for a in chunk:
                    obs, done, info = env.step(a)
                    t += 1
                    if info["is_success"] and first_t is None:
                        first_t = t
                    if t % args.print_every == 0:
                        print(f"[plus]   {spec} seed={seed} t={t} policy_ms={act_ms[-1]:.0f} "
                              f"ee={np.round(obs.ee_pos, 3).tolist()} succ={info['is_success']}")
                    if done or t >= args.steps:
                        break
            success = bool(env.success_flag())
            res = {"config": spec, "suite": args.suite, "task": task_idx, "seed": seed, "policy": policy.name,
                   "success": success, "steps": t, "first_success_t": first_t,
                   "init_ee_pos": np.round(init_pos, 4).tolist(), "init_ee_quat": np.round(init_quat, 4).tolist(),
                   "init_state": (rp or {}).get("init_state"), "ee_shift_m": (rp or {}).get("ee_shift_m"),
                   "policy_ms_mean": round(float(np.mean(act_ms)), 1), "policy_ms_p50": round(float(np.median(act_ms)), 1),
                   "policy_ms_p95": round(float(np.percentile(act_ms, 95)), 1), "wall_s": round(time.time() - te, 1),
                   **gpu_mem()}
            results.append(res)
            print(f"[plus] DONE {spec} seed={seed} success={success} steps={t} first_success_t={first_t} "
                  f"policy_ms mean={res['policy_ms_mean']} p95={res['policy_ms_p95']} wall={res['wall_s']}s")
            print("PLUS_RESULT " + json.dumps(res))
        env.close()
        n_ok = sum(r["success"] for r in results)
        summ = {"config": spec, "suite": args.suite, "task": task_idx, "policy": policy.name, "n": len(results),
                "n_success": n_ok, "mean_success": round(n_ok / len(results), 3),
                "mean_steps": round(float(np.mean([r["steps"] for r in results])), 1),
                "policy_ms_mean": round(float(np.mean([r["policy_ms_mean"] for r in results])), 1),
                "init_ee_pos": [r["init_ee_pos"] for r in results], **gpu_mem()}
        summaries.append(summ)
        print(f"[plus] SUMMARY {spec}: success {n_ok}/{len(results)} = {summ['mean_success']}")
        print("PLUS_SUMMARY " + json.dumps(summ))

    print("[plus] " + " | ".join(f"{s['config']}: {s['n_success']}/{s['n']}" for s in summaries)
          + f"  total={time.time()-t0:.0f}s")
    print("PLUS_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
