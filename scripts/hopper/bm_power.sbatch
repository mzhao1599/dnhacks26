#!/bin/bash
#SBATCH -p gpuq -q gpu --gres=gpu:A100.40gb:1 -c 8 --mem=48G -t 02:00:00 -J fm-bmpower
#SBATCH -o /scratch/ezhao2/fleet-memory/logs/slurm/bmpower_%j.out
# BM-1 vs BM-4 (hand-set homing, no optimisation involved -> no held-out concern) on ALL 50 init states of tasks 0 and 3.
set -uo pipefail
source /scratch/ezhao2/fleet-memory/dnhacks26/scripts/hopper/env.sh
export FM_LLM=mock FM_POLICY_KWARGS='{"n_action_steps":10}' FM_LIBERO_PLUS=/scratch/ezhao2/fleet-memory/LIBERO-plus
cd "$FM_REPO"; L=$FM_LOGS/benchmark
python - <<'PY'
import dataclasses, json, os, sys
from fleet_memory.envs.libero_plus import list_configs
from fleet_memory.memory.store import EventStore
from fleet_memory.execution.params import S3Params
from fleet_memory.runner.pool import run_many, summarize, close_pool
from fleet_memory.runner.worker import RunConfig
from fleet_memory.analysis.metrics import wilson_ci
L = os.environ["FM_LOGS"] + "/benchmark"
store = EventStore(L + "/power.jsonl")
cfgs_all = list_configs("robot_init", "libero_spatial")
res = {}
for t in ("0", "3"):
    cfg = [c for c in cfgs_all if str(c.get("base_task_idx")) == t][0]
    base = RunConfig(suite="libero_spatial", task_id=t, seed=0, arm="A", env_kind="libero_plus", policy_kind="smolvla",
                     log_path=L + "/power.jsonl", env_kwargs={"config": cfg}, environment_tag="plus_robot_init_0_power")
    hom = S3Params.identity(); hom.homing_enable = 1.0
    arms = {"BM-1": dataclasses.replace(base, arm="A"), "BM-4": dataclasses.replace(base, arm="B", s3_params=hom.to_dict())}
    for name, c in arms.items():
        eps = run_many([dataclasses.replace(c, seed=7000 + i) for i in range(50)], 4)   # 7000+i -> init state i, seeds disjoint from everything else
        k = sum(e.outcome.env_success for e in eps); res.setdefault(name, [0, 0]); res[name][0] += k; res[name][1] += len(eps)
        print(f"task {t} {name}: {k}/{len(eps)}  {summarize(eps, name)}", flush=True)
for name, (k, n) in res.items():
    lo, hi = wilson_ci(k, n); print(f"POOLED {name}: {k}/{n} = {100*k/n:.1f}% [{100*lo:.1f}, {100*hi:.1f}]")
store.append({"type": "benchmark_power", "tasks": ["0", "3"], "n_init_states": 50, "results": {k: {"k": v[0], "n": v[1]} for k, v in res.items()}})
close_pool()
PY
echo BMPOWER_DONE
