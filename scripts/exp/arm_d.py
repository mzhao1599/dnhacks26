"""Arm D on the perturbed benchmark task: planner + S1 memory + inner coach (live LLM) on top of the optimized S3
vector. Does the coach add anything beyond the optimizer? n = 10 held-out layouts x 2 noise draws per arm."""
import dataclasses, json, os, sys
from fleet_memory.envs.libero_plus import list_configs
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


def main(task="0", reps=2, workers=4):
    L = os.environ["FM_LOGS"] + "/benchmark"
    store = EventStore(L + "/armD.jsonl")
    cfg = [c for c in list_configs("robot_init", "libero_spatial") if str(c.get("base_task_idx")) == task][0]
    v = incumbent(L + "/events.jsonl", f"si_{task}__libero_spatial_{task}_plus_robot_init_0")
    base = RunConfig(suite="libero_spatial", task_id=task, seed=0, arm="B", env_kind="libero_plus", policy_kind="smolvla",
                     log_path=L + "/armD.jsonl", env_kwargs={"config": cfg}, environment_tag="plus_robot_init_0_armD", s3_params=v)
    seeds = [5040 + i + 50 * r for r in range(reps) for i in range(10)]
    res = {}
    for name, c in {"B (vector only)": base, "D (vector + planner + coach)": dataclasses.replace(base, arm="D")}.items():
        eps = run_many([dataclasses.replace(c, seed=s) for s in seeds], workers)
        k = sum(e.outcome.env_success for e in eps); lo, hi = wilson_ci(k, len(eps))
        iv = sum(len(e.interventions) for e in eps)
        res[name] = {"k": k, "n": len(eps), "ci": [lo, hi], "steps": sum(e.outcome.steps for e in eps) / len(eps), "interventions": iv}
        print(f"ARMD {name}: {k}/{len(eps)} = {100*k/len(eps):.0f}% [{100*lo:.0f},{100*hi:.0f}] steps {res[name]['steps']:.0f} interventions {iv}", flush=True)
    store.append({"type": "arm_d_result", "task": task, "results": res})
    close_pool()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "0")
