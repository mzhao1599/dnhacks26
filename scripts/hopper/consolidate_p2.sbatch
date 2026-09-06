#!/bin/bash
#SBATCH -p gpuq -q gpu --gres=gpu:A100.40gb:1 -c 8 --mem=48G -t 03:00:00 -J fm-p2sleep
#SBATCH -o /scratch/ezhao2/fleet-memory/logs/slurm/p2sleep_%j.out
set -uo pipefail
source /scratch/ezhao2/fleet-memory/dnhacks26/scripts/hopper/env.sh
export FM_LLM=mock FM_POLICY_KWARGS='{"n_action_steps":10}'
cd "$FM_REPO"
python -m fleet_memory.runner.consolidate --env libero --policy smolvla --suite libero_10 --task 3 --log $FM_LOGS/phase0/events.jsonl --small --workers 4 --trigger manual
echo "=== arm B after sleep (incumbent)"
python -m fleet_memory.runner.pool --env libero --policy smolvla --suite libero_10 --task 3 --arm B --n 20 --workers 4 --seed-set train --log $FM_LOGS/phase0/events.jsonl
echo P2_DONE
