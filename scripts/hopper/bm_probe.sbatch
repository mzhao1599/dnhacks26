#!/bin/bash
#SBATCH -p gpuq -q gpu --gres=gpu:A100.40gb:1 -c 8 --mem=48G -t 02:00:00 -J fm-bmprobe
#SBATCH -o /scratch/ezhao2/fleet-memory/logs/slurm/bmprobe_%j.out
set -uo pipefail
source /scratch/ezhao2/fleet-memory/dnhacks26/scripts/hopper/env.sh
export FM_LLM=mock FM_POLICY_KWARGS='{"n_action_steps":10}' FM_LIBERO_PLUS=/scratch/ezhao2/fleet-memory/LIBERO-plus
cd "$FM_REPO"; mkdir -p $FM_LOGS/benchmark
python -m fleet_memory.runner.benchmark --policy smolvla --suite libero_spatial --tasks 0 --dimension robot_init --config-index 0 \
  --arms BM-0,BM-1,BM-4,BM-2 --n-per-task 10 --workers 4 --log $FM_LOGS/benchmark/probe_task0.jsonl --png $FM_LOGS/benchmark/probe_task0.png
echo BMPROBE_DONE
