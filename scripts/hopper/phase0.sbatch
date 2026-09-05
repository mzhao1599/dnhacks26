#!/bin/bash
#SBATCH --job-name=fm-phase0
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-9
#SBATCH --output=/scratch/ezhao2/fleet-memory/logs/slurm/phase0_%A_%a.out
#SBATCH --error=/scratch/ezhao2/fleet-memory/logs/slurm/phase0_%A_%a.err
#
# Phase 0: baseline (arm A, no memory) over libero_10 with SmolVLA.
# One A100 per array task; 4 worker processes share the GPU (runner.pool --workers 4).
# Partitioning (env PARTITION, default "task"):
#   task : array index = libero_10 task index (0..9); each runs N seeds of that task.
#   seed : every array task runs all tasks; seeds [ID*N, (ID+1)*N) via --seed-offset.
# Usage:  sbatch scripts/hopper/phase0.sbatch            # N=50 seeds per task
#         N=30 sbatch --array=0-4 scripts/hopper/phase0.sbatch
#         PARTITION=seed N=10 sbatch --array=0-4 scripts/hopper/phase0.sbatch
set -euo pipefail
mkdir -p /scratch/ezhao2/fleet-memory/logs/slurm
source /scratch/ezhao2/fleet-memory/dnhacks26/scripts/hopper/env.sh

ID=${SLURM_ARRAY_TASK_ID:-0}
N=${N:-50}
WORKERS=${WORKERS:-4}
ARM=${ARM:-A}
SUITE=${SUITE:-libero_10}
PARTITION=${PARTITION:-task}
LOGDIR=$FM_LOGS/phase0; mkdir -p "$LOGDIR"
LOG=$LOGDIR/events_${ID}.jsonl

echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) MUJOCO_GL=$MUJOCO_GL id=$ID N=$N"
cd "$FM_REPO"

EXTRA=()
if [ "$PARTITION" = "task" ]; then
  EXTRA+=(--tasks "$ID")
else
  EXTRA+=(--seed-offset $((ID * N)))
fi

python -m fleet_memory.runner.pool \
  --env libero --policy smolvla --suite "$SUITE" --arm "$ARM" \
  --n "$N" --workers "$WORKERS" --seed-set train \
  --log "$LOG" "${EXTRA[@]}" ${FM_POOL_EXTRA:-}

echo "PHASE0_DONE id=$ID log=$LOG"
