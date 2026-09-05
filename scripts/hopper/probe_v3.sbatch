#!/bin/bash
#SBATCH --job-name=fm-probev3
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/ezhao2/fleet-memory/logs/slurm/probev3_%j.out
#SBATCH --error=/scratch/ezhao2/fleet-memory/logs/slurm/probev3_%j.out
#
# Phase 1 (v3): hand-set S3 probes on the SAME train seeds as phase0_v3 (matched episodes).
#   good : {"time_scale":1.3}                                          -> logs/probe/events_good.jsonl
#   bad  : {"grasp_offset_z":-0.02,"approach_offset_xyz":[0.03,0.03,0]} -> logs/probe/events_bad.jsonl
# Extra probes: PROBES=probes/s3.jsonl (one JSON S3-params dict per line) -> events_probe<k>.jsonl
# Usage:  sbatch scripts/hopper/probe_v3.sbatch
#         SUITE=libero_goal TASK=0 N=20 sbatch scripts/hopper/probe_v3.sbatch
set -euo pipefail
mkdir -p /scratch/ezhao2/fleet-memory/logs/slurm
source /scratch/ezhao2/fleet-memory/dnhacks26/scripts/hopper/env.sh

SUITE=${SUITE:-libero_10}
TASK=${TASK:-3}
N=${N:-20}
WORKERS=${WORKERS:-4}
ARM=${ARM:-A}
LOGDIR=$FM_LOGS/probe; mkdir -p "$LOGDIR"

echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) MUJOCO_GL=$MUJOCO_GL suite=$SUITE task=$TASK N=$N workers=$WORKERS"
cd "$FM_REPO"

run_probe () {  # name, s3-params json
  echo "=== probe $1: $2 ==="
  python -m fleet_memory.runner.pool \
    --env libero --policy smolvla --suite "$SUITE" --task "$TASK" --arm "$ARM" \
    --n "$N" --workers "$WORKERS" --seed-set train \
    --s3-params "$2" \
    --log "$LOGDIR/events_$1.jsonl" ${FM_POOL_EXTRA:-}
}

run_probe good '{"time_scale":1.3}'
run_probe bad  '{"grasp_offset_z":-0.02,"approach_offset_xyz":[0.03,0.03,0]}'

if [ -n "${PROBES:-}" ] && [ -f "$PROBES" ]; then
  k=0
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    k=$((k+1))
    run_probe "probe$k" "$line"
  done < "$PROBES"
fi

echo "PROBE_V3_DONE suite=$SUITE task=$TASK dir=$LOGDIR"
