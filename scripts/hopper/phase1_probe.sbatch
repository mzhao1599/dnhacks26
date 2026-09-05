#!/bin/bash
#SBATCH --job-name=fm-phase1
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-9
#SBATCH --output=/scratch/ezhao2/fleet-memory/logs/slurm/phase1_%A_%a.out
#SBATCH --error=/scratch/ezhao2/fleet-memory/logs/slurm/phase1_%A_%a.err
#
# Phase 1 probes on arm A over the SAME seeds as phase0 (seed-set train, same N, same task
# partition), so every probe episode has a matched baseline episode:
#   S2 : instruction paraphrases   (--probe-instruction "<text>")   one run per paraphrase
#   S3 : constraint edits          (--probe-edit '<json Edit>')     one run per edit
# Array index = libero_10 task index. Configure probes via the PROBES_S2 / PROBES_S3 files
# (one paraphrase / one JSON edit per line) or the defaults below.
# Usage: sbatch scripts/hopper/phase1_probe.sbatch
#        PROBES_S2=probes/s2.txt PROBES_S3=probes/s3.jsonl N=30 sbatch scripts/hopper/phase1_probe.sbatch
set -euo pipefail
mkdir -p /scratch/ezhao2/fleet-memory/logs/slurm
source /scratch/ezhao2/fleet-memory/dnhacks26/scripts/hopper/env.sh

ID=${SLURM_ARRAY_TASK_ID:-0}
N=${N:-50}
WORKERS=${WORKERS:-4}
ARM=${ARM:-A}
SUITE=${SUITE:-libero_10}
LOGDIR=$FM_LOGS/phase1; mkdir -p "$LOGDIR"
cd "$FM_REPO"
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) MUJOCO_GL=$MUJOCO_GL id=$ID N=$N"

# ---- S2: paraphrase probes -------------------------------------------------------
S2_DEFAULT=$(mktemp)
cat > "$S2_DEFAULT" <<'EOF'
__canonical__
__lowercase_no_punct__
__imperative_short__
EOF
S2_FILE=${PROBES_S2:-$S2_DEFAULT}
i=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  i=$((i+1))
  python -m fleet_memory.runner.pool \
    --env libero --policy smolvla --suite "$SUITE" --arm "$ARM" \
    --n "$N" --workers "$WORKERS" --seed-set train --tasks "$ID" \
    --probe-instruction "$line" \
    --log "$LOGDIR/s2_probe${i}_events_${ID}.jsonl" ${FM_POOL_EXTRA:-}
done < "$S2_FILE"

# ---- S3: constraint-edit probes ---------------------------------------------------
S3_DEFAULT=$(mktemp)
cat > "$S3_DEFAULT" <<'EOF'
{"op":"set_velocity_cap","params":{"max_pos_delta":0.6,"max_rot_delta":0.6}}
{"op":"set_gripper_aperture","params":{"value":0.8}}
{"op":"insert_pre_grasp_waypoint","params":{"dx":0.0,"dy":0.0,"dz":0.06,"tol_m":0.015}}
{"op":"set_grasp_offset","params":{"dx":0.0,"dy":0.0,"dz":-0.01}}
EOF
S3_FILE=${PROBES_S3:-$S3_DEFAULT}
j=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  j=$((j+1))
  python -m fleet_memory.runner.pool \
    --env libero --policy smolvla --suite "$SUITE" --arm "$ARM" \
    --n "$N" --workers "$WORKERS" --seed-set train --tasks "$ID" \
    --probe-edit "$line" \
    --log "$LOGDIR/s3_probe${j}_events_${ID}.jsonl" ${FM_POOL_EXTRA:-}
done < "$S3_FILE"

echo "PHASE1_DONE id=$ID s2=$i s3=$j"
