"""Camera-viewpoint collapse probe (LIBERO-Plus 'Camera Viewpoints', libero_spatial task 0): BM-0 (stock camera) and
BM-1 (raw VLA, camera moved) for EVERY camera variant of the task on the 10 held-out eval layouts (init states 40-49)
x R policy-noise draws. Also dumps the first agentview frame per variant to $FM_LOGS/bench_camera/frames/ so the
perturbation can be eyeballed. Prints `PROBE ...` lines; appends one `camera_probe` event.

    python scripts/exp/camera_probe.py [task=0] [reps=1] [workers=4]
"""
import dataclasses, json, os, sys
import numpy as np
from fleet_memory.envs.libero_plus import list_configs, make_perturbed_env
from fleet_memory.memory.store import EventStore
from fleet_memory.runner.pool import run_many, close_pool
from fleet_memory.runner.worker import RunConfig
from fleet_memory.analysis.metrics import wilson_ci


def dump_frames(task, cfgs, out_dir, seed=5040):
    """One PNG per variant (+ the stock camera), rendered from the same init state, before any policy runs."""
    os.makedirs(out_dir, exist_ok=True)
    from PIL import Image
    env = make_perturbed_env("libero_spatial", task, None)
    obs = env.reset(seed)
    Image.fromarray(obs.images["agentview"]).save(f"{out_dir}/stock.png")
    env.close()
    for i, c in enumerate(cfgs):
        env = make_perturbed_env("libero_spatial", task, c)
        obs = env.reset(seed)
        Image.fromarray(obs.images["agentview"]).save(f"{out_dir}/cfg{i}_view_{c['view']}.png")
        print(f"FRAME cfg={i} view={c['view']} camera={json.dumps(env.last_perturbation.get('camera'))}", flush=True)
        env.close()


def main(task="0", reps=1, workers=4):
    L = os.environ["FM_LOGS"] + "/bench_camera"
    os.makedirs(L, exist_ok=True)
    store = EventStore(L + "/probe.jsonl")
    cfgs = [c for c in list_configs("camera", "libero_spatial") if str(c.get("base_task_idx")) == task]
    print(f"{len(cfgs)} camera variants for task {task}: {[c['view'] for c in cfgs]}", flush=True)
    try:
        dump_frames(task, cfgs, L + "/frames")
    except Exception as ex:                 # frames are a convenience, never block the numbers
        print(f"frame dump failed: {ex!r}", flush=True)
    base = RunConfig(suite="libero_spatial", task_id=task, seed=0, arm="A", env_kind="libero", policy_kind="smolvla",
                     log_path=L + "/probe.jsonl")
    seeds = [5040 + i + 50 * r for r in range(reps) for i in range(10)]     # eval layouts 40-49, reps vary policy noise
    res = {}

    def run(name, cfg, view="stock"):
        eps = run_many([dataclasses.replace(cfg, seed=s) for s in seeds], workers)
        k, n = sum(e.outcome.env_success for e in eps), len(eps)
        lo, hi = wilson_ci(k, n)
        res[name] = {"k": k, "n": n, "rate": k / n, "ci": [lo, hi], "mean_steps": float(np.mean([e.outcome.steps for e in eps])),
                     "per_seed": {str(e.seed): bool(e.outcome.env_success) for e in eps}}
        print(f"PROBE {name}: {k}/{n} = {100 * k / n:.0f}% [{100 * lo:.0f},{100 * hi:.0f}] steps {res[name]['mean_steps']:.0f}", flush=True)
        store.append({"type": "camera_probe_arm", "task": task, "reps": reps, "arm": name, "view": view, **res[name]})   # survives a crash later on

    run("BM-0 stock", base)
    for i, c in enumerate(cfgs):
        run(f"BM-1 cfg={i} view={c['view']}", dataclasses.replace(base, env_kind="libero_plus", env_kwargs={"config": c},
                                                                  environment_tag=f"plus_camera_{i}"), view=c["view"])
    store.append({"type": "camera_probe", "task": task, "reps": reps, "configs": cfgs, "results": res})
    close_pool()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "0", int(sys.argv[2]) if len(sys.argv) > 2 else 1,
         int(sys.argv[3]) if len(sys.argv) > 3 else 4)
