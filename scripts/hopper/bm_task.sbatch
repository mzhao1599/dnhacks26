#!/bin/bash
#SBATCH -p gpuq -q gpu --gres=gpu:A100.40gb:1 -c 8 --mem=48G -t 03:00:00 -J fm-bm
#SBATCH -o /scratch/ezhao2/fleet-memory/logs/slurm/bm_%j.out
set -uo pipefail
TASK=${1:?task index}
source /scratch/ezhao2/fleet-memory/dnhacks26/scripts/hopper/env.sh
export FM_LLM=mock FM_POLICY_KWARGS='{"n_action_steps":10}' FM_LIBERO_PLUS=/scratch/ezhao2/fleet-memory/LIBERO-plus
cd "$FM_REPO"; mkdir -p $FM_LOGS/benchmark
echo "=== BM task $TASK: consolidate (CEM pop24 x K2 x 4 iters, gate 12) then arms BM-0..4 x 10 eval seeds"
python -m fleet_memory.runner.benchmark --policy smolvla --suite libero_spatial --tasks $TASK --dimension robot_init --config-index 0 \
  --arms BM-0,BM-1,BM-2,BM-4 --consolidate small --cem '{"population":24,"elites":6,"iterations":4,"seeds_per_candidate":2,"gate_seeds":12}' \
  --n-per-task 10 --workers 4 --log $FM_LOGS/benchmark/events.jsonl --png $FM_LOGS/benchmark/bm_task$TASK.png
echo "BM_TASK_DONE $TASK"
