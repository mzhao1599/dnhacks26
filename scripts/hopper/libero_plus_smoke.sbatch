#!/bin/bash
#SBATCH --job-name=fm-plus
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ezhao2/fleet-memory/logs/slurm/plus_%j.out
#SBATCH --error=/scratch/ezhao2/fleet-memory/logs/slurm/plus_%j.out
#
# LIBERO-Plus robot-initial-state smoke: standard reset vs perturbed reset on one task.
# Usage:  sbatch scripts/hopper/libero_plus_smoke.sbatch [suite] [task] [seeds] [steps] [policy] [configs]
#         sbatch scripts/hopper/libero_plus_smoke.sbatch libero_spatial 0 0,1,2,3,4 220 smolvla standard,robot_init:0
# Env:    BACKEND=plus   (default) -> venv_plus + LIBERO-plus package (MountedPanda{N} env, their eval path)
#         BACKEND=native           -> shared venv + hf-libero, joint qpos rewritten after reset
#         EXTRA="--n-action-steps 5"  extra args for scripts/smoke_libero_plus.py
set -euo pipefail
mkdir -p /scratch/ezhao2/fleet-memory/logs/slurm /scratch/ezhao2/fleet-memory/logs/plus
source /scratch/ezhao2/fleet-memory/dnhacks26/scripts/hopper/env.sh
SUITE=${1:-libero_spatial}; TASK=${2:-0}; SEEDS=${3:-0,1,2,3,4}; STEPS=${4:-220}
POLICY=${5:-smolvla}; CONFIGS=${6:-standard,robot_init:0}
BACKEND=${BACKEND:-plus}
export FM_LIBERO_PLUS=/scratch/ezhao2/fleet-memory/LIBERO-plus
if [ "$BACKEND" = "plus" ]; then
  # LIBERO-plus is installed (editable) as the `libero` package in venv_plus; its own config.yaml
  # points bddl_files/init_states at the LIBERO-plus checkout (assets symlinked there).
  source /scratch/ezhao2/fleet-memory/venv_plus/bin/activate
  export LIBERO_CONFIG_PATH=/scratch/ezhao2/fleet-memory/libero_plus_config
fi
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) MUJOCO_GL=$MUJOCO_GL backend=$BACKEND venv=$VIRTUAL_ENV"
echo "suite=$SUITE task=$TASK seeds=$SEEDS steps=$STEPS policy=$POLICY configs=$CONFIGS"
python -c "import libero, os; print('libero from', os.path.dirname(libero.__file__))"
cd "$FM_REPO"
python scripts/smoke_libero_plus.py --suite "$SUITE" --task "$TASK" --seeds "$SEEDS" --steps "$STEPS" \
  --policy "$POLICY" --config "$CONFIGS" --logdir "$FM_LOGS/plus" ${EXTRA:-}
echo "PLUS_SMOKE_DONE"
