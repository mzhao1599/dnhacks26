#!/bin/bash
#SBATCH --job-name=fm-phase0v3
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/ezhao2/fleet-memory/logs/slurm/phase0v3_%j.out
#SBATCH --error=/scratch/ezhao2/fleet-memory/logs/slurm/phase0v3_%j.out
#
# Phase 0 (v3): arm A baseline on ONE fixed task, 20 train seeds, 4 worker processes sharing one
# A100.40gb (starts in seconds; A100.80gb is queue-blocked). Writes the frozen cost_reference
# events (--make-reference) that every later arm is scored against.
# Usage:  sbatch scripts/hopper/phase0_v3.sbatch
#         SUITE=libero_goal TASK=0 N=20 sbatch scripts/hopper/phase0_v3.sbatch
set -euo pipefail
mkdir -p /scratch/ezhao2/fleet-memory/logs/slurm
source /scratch/ezhao2/fleet-memory/dnhacks26/scripts/hopper/env.sh

SUITE=${SUITE:-libero_10}
TASK=${TASK:-3}
N=${N:-20}
WORKERS=${WORKERS:-4}
ARM=${ARM:-A}
LOGDIR=$FM_LOGS/phase0; mkdir -p "$LOGDIR"
LOG=${LOG:-$LOGDIR/events.jsonl}

echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) MUJOCO_GL=$MUJOCO_GL suite=$SUITE task=$TASK N=$N workers=$WORKERS llm=${FM_LLM:-auto}"
cd "$FM_REPO"

python -m fleet_memory.runner.pool \
  --env libero --policy smolvla --suite "$SUITE" --task "$TASK" --arm "$ARM" \
  --n "$N" --workers "$WORKERS" --seed-set train \
  --log "$LOG" --make-reference ${FM_POOL_EXTRA:-}

echo "PHASE0_V3_DONE suite=$SUITE task=$TASK log=$LOG"
