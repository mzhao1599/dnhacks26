#!/bin/bash
#SBATCH -p gpuq -q gpu --gres=gpu:A100.40gb:1 -c 8 --mem=48G -t 02:00:00 -J fm-p1b
#SBATCH -o /scratch/ezhao2/fleet-memory/logs/slurm/p1b_%j.out
set -uo pipefail
source /scratch/ezhao2/fleet-memory/dnhacks26/scripts/hopper/env.sh
export FM_LLM=mock FM_POLICY_KWARGS='{"n_action_steps":10}'
cd "$FM_REPO"; mkdir -p $FM_LOGS/probe_v31
run () { echo "=== probe $1 $2"; python -m fleet_memory.runner.pool --env libero --policy smolvla --suite libero_10 --task 3 --arm B --n 20 --workers 4 --seed-set train --log $FM_LOGS/probe_v31/$1.jsonl --s3-params "$2"; }
run identity     '{}'
run timescale13  '{"time_scale":1.3}'
run homing_on    '{"homing_enable":1.0}'
run blend05      '{"blend_alpha":0.5}'
run bad_offset   '{"blend_alpha":0.5,"grasp_offset_z":-0.02,"approach_offset_xyz":[0.03,0.03,0.0]}'
echo P1B_DONE
