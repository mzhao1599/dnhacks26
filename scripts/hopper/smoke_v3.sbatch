#!/bin/bash
#SBATCH --job-name=fm-smoke3
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/ezhao2/fleet-memory/logs/slurm/smoke3_%j.out
#SBATCH --error=/scratch/ezhao2/fleet-memory/logs/slurm/smoke3_%j.out
#
# Full-episode SmolVLA smoke on one A100.40gb (starts in seconds; A100.80gb is queue-blocked).
# Usage:  sbatch scripts/hopper/smoke_v3.sbatch [suite] [task] [seeds] [steps]
#         sbatch scripts/hopper/smoke_v3.sbatch libero_10 0 0,1,2 520
#         sbatch scripts/hopper/smoke_v3.sbatch libero_goal 0 0,1,2 300
# Several (suite task seeds steps) tuples: SWEEP="libero_10 1 0,1 520;libero_10 2 0,1 520" sbatch scripts/hopper/smoke_v3.sbatch
set -euo pipefail
mkdir -p /scratch/ezhao2/fleet-memory/logs/slurm
source /scratch/ezhao2/fleet-memory/dnhacks26/scripts/hopper/env.sh
SUITE=${1:-libero_10}; TASK=${2:-0}; SEEDS=${3:-0}; STEPS=${4:-520}
SWEEP=${SWEEP:-"$SUITE $TASK $SEEDS $STEPS"}

echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) MUJOCO_GL=$MUJOCO_GL sweep='$SWEEP'"
cd "$FM_REPO"
IFS=';' read -ra RUNS <<< "$SWEEP"
for run in "${RUNS[@]}"; do
  read -r s t sd st <<< "$run"
  echo "=== RUN suite=$s task=$t seeds=$sd steps=$st ==="
  python scripts/smoke_libero.py --suite "$s" --task "$t" --seeds "$sd" --steps "$st" --policy smolvla \
    --logdir "$FM_LOGS/smoke" ${SMOKE_EXTRA:-}
done
echo "SMOKE3_DONE"
