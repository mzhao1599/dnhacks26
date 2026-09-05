#!/bin/bash
#SBATCH --job-name=fm-perturb
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/ezhao2/fleet-memory/logs/slurm/perturb_%j.out
#SBATCH --error=/scratch/ezhao2/fleet-memory/logs/slurm/perturb_%j.out
#
# Verifies LiberoEnv.reset(seed, perturbation=...) : object pose changes and the sim stays stable
# (object rests on the table; no fall-through) over 60 no-op steps. Zero policy only (no VLA).
# Usage: sbatch scripts/hopper/perturb_verify.sbatch [suite] [task]
set -euo pipefail
mkdir -p /scratch/ezhao2/fleet-memory/logs/slurm
source /scratch/ezhao2/fleet-memory/dnhacks26/scripts/hopper/env.sh
SUITE=${1:-libero_10}; TASK=${2:-0}
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) MUJOCO_GL=$MUJOCO_GL"
cd "$FM_REPO"
python scripts/verify_perturb.py --suite "$SUITE" --task "$TASK" --seeds 0,1,2 --settle-steps 60
echo "PERTURB_DONE"
