"""Held-out benchmark with real n: the 10 eval layouts (init states 40-49) x R policy-noise repetitions per arm,
task 0. Layouts stay held-out (never seen by optimizer/gate); only the policy's RNG seed varies across reps."""
import dataclasses, json, os, sys
from fleet_memory.envs.libero_plus import list_configs
from fleet_memory.execution.params import S3Params
from fleet_memory.memory.store import EventStore
from fleet_memory.runner.pool import run_many, summarize, close_pool
from fleet_memory.runner.worker import RunConfig
from fleet_memory.analysis.metrics import wilson_ci


def incumbent(log, si_id):
    best = None
    for line in open(log):
        try: d = json.loads(line)
        except Exception: continue
        if d.get("type") == "skill_instance" and d["skill_instance_id"] == si_id and d["status"] == "incumbent":
            if best is None or d["version"] > best["version"]: best = d
    return best["params"] if best else None


def main(task="0", reps=5, workers=4, rep_offset=0, with_bm0=False):
    L = os.environ["FM_LOGS"] + "/benchmark"
    store = EventStore(L + "/reps.jsonl")
    cfg = [c for c in list_configs("robot_init", "libero_spatial") if str(c.get("base_task_idx")) == task][0]
    base = RunConfig(suite="libero_spatial", task_id=task, seed=0, arm="A", env_kind="libero_plus", policy_kind="smolvla",
                     log_path=L + "/reps.jsonl", env_kwargs={"config": cfg}, environment_tag="plus_robot_init_0_reps")
    hom = S3Params.identity(); hom.homing_enable = 1.0
    arms = {}
    if with_bm0:
        arms["BM-0"] = dataclasses.replace(base, arm="A", env_kind="libero", env_kwargs={}, environment_tag="")
    arms.update({"BM-1": dataclasses.replace(base, arm="A"),
            "BM-2": dataclasses.replace(base, arm="B", s3_params=S3Params.identity().to_dict()),
            "BM-4": dataclasses.replace(base, arm="B", s3_params=hom.to_dict())})
    v2 = incumbent(L + "/events.jsonl", f"si_{task}__libero_spatial_{task}_plus_robot_init_0")
    if v2: arms["BM-3"] = dataclasses.replace(base, arm="B", s3_params=v2)
    if os.path.exists(L + "/events_wide.jsonl"):
        w = incumbent(L + "/events_wide.jsonl", f"si_{task}__libero_spatial_{task}_plus_robot_init_0_wide")
        if w: arms["BM-3w"] = dataclasses.replace(base, arm="B", s3_params=w)
    seeds = [5040 + i + 50 * (r + rep_offset) for r in range(reps) for i in range(10)]   # init states 40-49; reps vary the RNG seed
    res = {}
    for name, c in arms.items():
        eps = run_many([dataclasses.replace(c, seed=s) for s in seeds], workers)
        k = sum(e.outcome.env_success for e in eps); lo, hi = wilson_ci(k, len(eps))
        res[name] = {"k": k, "n": len(eps), "rate": k / len(eps), "ci": [lo, hi],
                     "mean_steps": sum(e.outcome.steps for e in eps) / len(eps), "s3_params": c.s3_params}
        print(f"REPS {name}: {k}/{len(eps)} = {100 * k / len(eps):.0f}% [{100 * lo:.0f},{100 * hi:.0f}] steps {res[name]['mean_steps']:.0f}", flush=True)
    store.append({"type": "benchmark_reps", "task": task, "reps": reps, "rep_offset": rep_offset, "results": res})
    close_pool()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "0", int(sys.argv[2]) if len(sys.argv) > 2 else 5,
         rep_offset=int(sys.argv[3]) if len(sys.argv) > 3 else 0, with_bm0="--bm0" in sys.argv)
