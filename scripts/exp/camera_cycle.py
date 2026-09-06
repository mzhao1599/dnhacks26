"""One unattended sleep cycle on a camera-perturbed skill instance (strong gate: 24 gate layouts x 4 draws), then the
held-out benchmark on the 10 eval layouts x R fresh policy-noise draws: BM-1 (raw VLA), BM-2 (identity shim),
BM-3 (the promoted incumbent). Two variants share nothing but the env:
    default   the full 21-dim vector (action dims + camera calibration dims)
    --act     the 17 action dims only (calibration frozen at identity) -> attribution: is it the calibration?
Prints CYCLE / REPS lines; events go to $FM_LOGS/bench_camera/events.jsonl (per-instance skill_instance records).

    python scripts/exp/camera_cycle.py <config_index|-1 --view V> [--act] [--bm0] [--reps R] [--task 0] [--workers 4] [--cycles 1]

The same experiment may be queued on several cluster tiers: the first job to start claims
$FM_LOGS/bench_camera/locks/<tag>.lock and later duplicates exit immediately.
"""
import argparse, dataclasses, json, os
import numpy as np
from fleet_memory.envs.libero_plus import list_configs
from fleet_memory.execution import params as P
from fleet_memory.execution.params import S3Params
from fleet_memory.memory import house_model
from fleet_memory.memory.store import EventStore
from fleet_memory.runner.consolidate import ConsolidationConfig, consolidate
from fleet_memory.runner.pool import run_many, close_pool
from fleet_memory.runner.worker import RunConfig
from fleet_memory.analysis.metrics import wilson_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config_index", type=int, help="index into this task's camera configs, or -1 with --view")
    ap.add_argument("--view", default=None, help="LIBERO-Plus view string, e.g. 0_0_100_2_354 (overrides config_index)")
    ap.add_argument("--bm0", action="store_true", help="also run BM-0 (stock camera, raw VLA) on the same seeds")
    ap.add_argument("--act", action="store_true", help="freeze the camera-calibration dims (action-only vector)")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--task", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--rep-offset", type=int, default=0)
    ap.add_argument("--skip-eval", action="store_true")
    a = ap.parse_args()
    L = os.environ["FM_LOGS"] + "/bench_camera"
    os.makedirs(L, exist_ok=True)
    store = EventStore(L + "/events.jsonl")
    cfgs = [c for c in list_configs("camera", "libero_spatial") if str(c.get("base_task_idx")) == a.task]
    if a.view:
        a.config_index = next((i for i, c in enumerate(cfgs) if c["view"] == a.view), None)
        cfg = cfgs[a.config_index] if a.config_index is not None else {"dimension": "camera", "view": a.view, "base_task_idx": int(a.task)}
    else:
        cfg = cfgs[a.config_index]
    tag = f"plus_camera_{cfg['view']}" + ("_act" if a.act else "")
    lock_dir = L + "/locks"; os.makedirs(lock_dir, exist_ok=True)
    lock = f"{lock_dir}/t{a.task}_{tag}.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, f"{os.environ.get('SLURM_JOB_ID', '?')}\n".encode()); os.close(fd)
    except FileExistsError:
        print(f"LOCKED {lock} held by job {open(lock).read().strip()} -> this duplicate exits", flush=True)
        return
    tmpl = RunConfig(suite="libero_spatial", task_id=a.task, seed=0, arm="B", env_kind="libero_plus", policy_kind="smolvla",
                     log_path=L + "/events.jsonl", env_kwargs={"config": cfg}, environment_tag=tag)
    print(f"skill instance {tmpl.skill_instance_id} view={cfg['view']} act_only={a.act}", flush=True)
    ccfg = ConsolidationConfig(population=24, elites=6, iterations=4, seeds_per_candidate=4, gate_seeds=24,
                               workers=a.workers, max_wallclock_s=7200.0,
                               frozen_dims=list(P.CALIB_NAMES) if a.act else [])
    for _ in range(a.cycles):
        c = consolidate(store, tmpl.skill_instance_id, template=tmpl, cfg=ccfg, trigger="manual")
        g = c.gate.to_dict() if c.gate else None
        print(f"CYCLE t{a.task} {tag} cons={c.consolidation_id} rollouts={c.rollouts} wall={c.wallclock_s:.0f}s "
              f"validation={json.dumps(c.best_candidate['validation'] if c.best_candidate else None)} gate={json.dumps(g)} "
              f"promoted={c.promoted_version} best={json.dumps(c.best_candidate['params'] if c.best_candidate else None)}", flush=True)
    inc = house_model.incumbent(store, tmpl.skill_instance_id)
    print(f"INCUMBENT t{a.task} {tag} v{inc.version} {json.dumps(inc.params)}", flush=True)
    if a.skip_eval:
        close_pool(); return
    seeds = [5040 + i + 50 * (r + a.rep_offset) for r in range(a.reps) for i in range(10)]
    arms = {}
    if a.bm0:
        arms["BM-0"] = dataclasses.replace(tmpl, arm="A", env_kind="libero", env_kwargs={}, environment_tag="")
    arms.update({"BM-1": dataclasses.replace(tmpl, arm="A"),
                 "BM-2": dataclasses.replace(tmpl, arm="B", s3_params=S3Params.identity().to_dict()),
                 "BM-3": dataclasses.replace(tmpl, arm="B", s3_params=dict(inc.params))})
    res = {}
    for name, c in arms.items():
        eps = run_many([dataclasses.replace(c, seed=s) for s in seeds], a.workers)
        k, n = sum(e.outcome.env_success for e in eps), len(eps)
        lo, hi = wilson_ci(k, n)
        res[name] = {"k": k, "n": n, "rate": k / n, "ci": [lo, hi], "mean_steps": float(np.mean([e.outcome.steps for e in eps])),
                     "s3_params": c.s3_params, "version": inc.version if name == "BM-3" else None}
        print(f"REPS t{a.task} {tag} {name}: {k}/{n} = {100 * k / n:.0f}% [{100 * lo:.0f},{100 * hi:.0f}] steps {res[name]['mean_steps']:.0f}", flush=True)
    store.append({"type": "benchmark_reps", "dimension": "camera", "tag": tag, "task": a.task, "config_index": a.config_index,
                  "view": cfg["view"], "reps": a.reps, "rep_offset": a.rep_offset, "act_only": a.act, "results": res})
    close_pool()


if __name__ == "__main__":
    main()
