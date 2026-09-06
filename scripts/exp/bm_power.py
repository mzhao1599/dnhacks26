"""BM-1 vs BM-4 (hand-set homing; no optimisation involved -> no held-out concern) on ALL 50 init states of given tasks."""
import dataclasses, os, sys
from fleet_memory.envs.libero_plus import list_configs
from fleet_memory.memory.store import EventStore
from fleet_memory.execution.params import S3Params
from fleet_memory.runner.pool import run_many, summarize, close_pool
from fleet_memory.runner.worker import RunConfig
from fleet_memory.analysis.metrics import wilson_ci


def main(tasks=("0", "3"), n_init=50, workers=4):
    L = os.environ["FM_LOGS"] + "/benchmark"
    store = EventStore(L + "/power.jsonl")
    cfgs_all = list_configs("robot_init", "libero_spatial")
    res = {}
    for t in tasks:
        cfg = [c for c in cfgs_all if str(c.get("base_task_idx")) == t][0]
        base = RunConfig(suite="libero_spatial", task_id=t, seed=0, arm="A", env_kind="libero_plus", policy_kind="smolvla",
                         log_path=L + "/power.jsonl", env_kwargs={"config": cfg}, environment_tag="plus_robot_init_0_power")
        hom = S3Params.identity(); hom.homing_enable = 1.0
        arms = {"BM-1": dataclasses.replace(base, arm="A"), "BM-4": dataclasses.replace(base, arm="B", s3_params=hom.to_dict())}
        for name, c in arms.items():
            eps = run_many([dataclasses.replace(c, seed=7000 + i) for i in range(n_init)], workers)  # 7000+i -> init state i
            k = sum(e.outcome.env_success for e in eps)
            res.setdefault(name, [0, 0]); res[name][0] += k; res[name][1] += len(eps)
            print(f"task {t} {name}: {k}/{len(eps)}  {summarize(eps, name)}", flush=True)
    for name, (k, n) in res.items():
        lo, hi = wilson_ci(k, n)
        print(f"POOLED {name}: {k}/{n} = {100 * k / n:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]", flush=True)
    store.append({"type": "benchmark_power", "tasks": list(tasks), "n_init_states": n_init,
                  "results": {k: {"k": v[0], "n": v[1]} for k, v in res.items()}})
    close_pool()


if __name__ == "__main__":
    main(tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("0", "3"))
