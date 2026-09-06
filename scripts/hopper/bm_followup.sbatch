#!/bin/bash
#SBATCH -p gpuq -q gpu --gres=gpu:A100.40gb:1 -c 8 --mem=48G -t 03:00:00 -J fm-bmfollow
#SBATCH -o /scratch/ezhao2/fleet-memory/logs/slurm/bmfollow_%j.out
# Runs after all per-task benchmark jobs: for any task whose perturbed skill instance is still v1 (gate refused),
# run a SECOND, wider-search sleep cycle (logged as cycle 2, never hand-tuned) and its BM-3 eval; then pool results.
set -uo pipefail
source /scratch/ezhao2/fleet-memory/dnhacks26/scripts/hopper/env.sh
export FM_LLM=mock FM_POLICY_KWARGS='{"n_action_steps":10}' FM_LIBERO_PLUS=/scratch/ezhao2/fleet-memory/LIBERO-plus
cd "$FM_REPO"; L=$FM_LOGS/benchmark
for t in 0 1 2 3 4; do
  v=$(python - <<PY
import json
v=1
for l in open("$L/events.jsonl"):
    try: d=json.loads(l)
    except Exception: continue
    if d.get("type")=="skill_instance" and d["skill_instance_id"]=="si_${t}__libero_spatial_${t}_plus_robot_init_0" and d["status"]=="incumbent": v=max(v,d["version"])
print(v)
PY
)
  echo "=== task $t incumbent version $v"
  if [ "$v" -lt 2 ]; then
    echo "=== task $t: gate refused in cycle 1 -> cycle 2 with wider initial sigma"
    python -m fleet_memory.runner.benchmark --policy smolvla --suite libero_spatial --tasks $t --dimension robot_init --config-index 0 \
      --arms BM-3 --consolidate small --cem '{"population":32,"elites":8,"iterations":4,"seeds_per_candidate":2,"gate_seeds":12,"sigma_init_frac":0.5}' \
      --n-per-task 10 --workers 4 --log $L/events.jsonl || echo "task $t: still no incumbent after cycle 2"
  fi
done
echo "=== pooled result"
python -m fleet_memory.runner.benchmark --aggregate --log $L/events.jsonl --png $L/bm_pooled.png
echo BMFOLLOW_DONE
