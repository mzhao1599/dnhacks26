#!/bin/bash
#SBATCH --job-name=fm-smoke
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/fleet-memory/logs/slurm/smoke_%j.out
#SBATCH --error=/scratch/%u/fleet-memory/logs/slurm/smoke_%j.err
#
# Usage:  sbatch scripts/hopper/smoke.sbatch [suite] [task] [steps]
#         (MIG slice instead:  sbatch --gres=gpu:1g.10gb:1 --export=ALL,MUJOCO_GL=osmesa scripts/hopper/smoke.sbatch)
set -euo pipefail
mkdir -p /scratch/$USER/fleet-memory/logs/slurm
source /scratch/$USER/fleet-memory/dnhacks26/scripts/hopper/env.sh
SUITE=${1:-libero_10}; TASK=${2:-0}; STEPS=${3:-100}

echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) MUJOCO_GL=$MUJOCO_GL"
cd "$FM_REPO"
python scripts/smoke_libero.py --suite "$SUITE" --task "$TASK" --seed 0 --steps 50  --policy zero    --logdir "$FM_LOGS/smoke"
python scripts/smoke_libero.py --suite "$SUITE" --task "$TASK" --seed 0 --steps "$STEPS" --policy smolvla --logdir "$FM_LOGS/smoke"
