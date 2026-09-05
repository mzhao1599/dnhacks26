#!/bin/bash
#SBATCH --job-name=fm-pi05
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ezhao2/fleet-memory/logs/slurm/pi05_%j.out
#SBATCH --error=/scratch/ezhao2/fleet-memory/logs/slurm/pi05_%j.out
#
# pi0.5 (lerobot/pi05_libero, 3B) smoke on standard LIBERO: per-step policy latency, GPU memory, success.
# Usage:  sbatch scripts/hopper/pi05_smoke.sbatch [suite] [task] [seeds] [steps] [configs]
#         sbatch scripts/hopper/pi05_smoke.sbatch libero_spatial 0 0,1,2 220 standard
# Env:    VENV=shared (default, /scratch/ezhao2/fleet-memory/venv) | plus (venv_plus, lerobot[pi] + LIBERO-plus)
#         DTYPE=float32 (checkpoint default) | bfloat16 ;  N_ACTION_STEPS=10 ;  STATS=dataset ;  EXTRA="..."
# Needs (prefetched on the login node): hf download lerobot/pi05_libero ; hf download leo009/paligemma-3b-pt-224
set -euo pipefail
mkdir -p /scratch/ezhao2/fleet-memory/logs/slurm /scratch/ezhao2/fleet-memory/logs/plus
source /scratch/ezhao2/fleet-memory/dnhacks26/scripts/hopper/env.sh
SUITE=${1:-libero_spatial}; TASK=${2:-0}; SEEDS=${3:-0,1,2}; STEPS=${4:-220}; CONFIGS=${5:-standard}
VENV=${VENV:-shared}; DTYPE=${DTYPE:-float32}; N_ACTION_STEPS=${N_ACTION_STEPS:-10}
STATS=${STATS:-}   # STATS=dataset -> normalise state/action with HuggingFaceVLA/libero stats (Hub ckpt ships none)
export FM_LIBERO_PLUS=/scratch/ezhao2/fleet-memory/LIBERO-plus
export FM_PALIGEMMA_TOKENIZER=${FM_PALIGEMMA_TOKENIZER:-leo009/paligemma-3b-pt-224}
if [ "$VENV" = "plus" ]; then
  source /scratch/ezhao2/fleet-memory/venv_plus/bin/activate
  export LIBERO_CONFIG_PATH=/scratch/ezhao2/fleet-memory/libero_plus_config
fi
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) MUJOCO_GL=$MUJOCO_GL venv=$VIRTUAL_ENV dtype=$DTYPE n_action_steps=$N_ACTION_STEPS"
cd "$FM_REPO"
python scripts/smoke_libero_plus.py --suite "$SUITE" --task "$TASK" --seeds "$SEEDS" --steps "$STEPS" \
  --policy pi05 --dtype "$DTYPE" --n-action-steps "$N_ACTION_STEPS" --config "$CONFIGS" \
  --print-every 10 --logdir "$FM_LOGS/plus" ${STATS:+--stats $STATS} ${EXTRA:-}
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
echo "PI05_SMOKE_DONE"
