#!/bin/bash
#SBATCH -p normal -q normal -c 32 --mem=96G -t 00:40:00 -J fm-cpu
#SBATCH -o /scratch/ezhao2/fleet-memory/logs/slurm/cpu_%j.out
# CPU-only SmolVLA + LIBERO throughput smoke (no GPU: osmesa rendering, torch on CPU).
# Usage: sbatch scripts/hopper/cpu_smoke.sbatch [workers=8] [omp=4] [n=8]
set -uo pipefail
WORKERS=${1:-8}; OMP=${2:-4}; N=${3:-8}
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=$OMP MKL_NUM_THREADS=$OMP
source /scratch/ezhao2/fleet-memory/dnhacks26/scripts/hopper/env.sh
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=$OMP MKL_NUM_THREADS=$OMP
export FM_LLM=mock FM_POLICY_KWARGS='{"n_action_steps":10}' FM_LIBERO_PLUS=/scratch/ezhao2/fleet-memory/LIBERO-plus
unset MUJOCO_EGL_DEVICE_ID
cd "$FM_REPO"; mkdir -p "$FM_LOGS/cpu_smoke"
echo "=== CPU smoke: workers=$WORKERS omp=$OMP n=$N node=$(hostname) cpus=$(nproc) start=$(date +%s)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'threads', torch.get_num_threads())"
T0=$(date +%s)
python -m fleet_memory.runner.pool --env libero --policy smolvla --suite libero_spatial --task 0 --arm A \
  --n "$N" --workers "$WORKERS" --seed-set train --log "$FM_LOGS/cpu_smoke/events_w${WORKERS}_o${OMP}.jsonl"
T1=$(date +%s)
echo "=== wall $((T1-T0)) s for $N episodes on $WORKERS workers"
echo "CPU_SMOKE_DONE"
