#!/bin/bash
#SBATCH -p gpuq -q gpu --gres=gpu:A100.40gb:1 -c 8 --mem=48G -t 04:00:00 -J fm-sleep31
#SBATCH -o /scratch/ezhao2/fleet-memory/logs/slurm/sleep31_%j.out
set -uo pipefail
source /scratch/ezhao2/fleet-memory/dnhacks26/scripts/hopper/env.sh
export FM_LLM=mock FM_POLICY_KWARGS='{"n_action_steps":10}'
cd "$FM_REPO"; mkdir -p $FM_LOGS/v31
LOG=$FM_LOGS/v31/events.jsonl
python - <<PY
import json
src="$FM_LOGS/phase0/events.jsonl"; keep=[]; ids=set()
for l in open(src):
    d=json.loads(l)
    if d.get("type")=="episode" and d.get("condition")=="A" and not d.get("perturbation",{}).get("consolidation_id"): keep.append(l); ids.add(d["episode_id"])
    elif d.get("type")=="snapshot" and d.get("episode_id") in ids: keep.append(l)
    elif d.get("type")=="cost_reference": keep.append(l)
open("$LOG","w").writelines(keep); print("seeded v31 log with", len(keep), "records")
PY
for i in 1 2 3; do
  echo "=== sleep cycle $i"
  python -m fleet_memory.runner.consolidate --env libero --policy smolvla --suite libero_10 --task 3 --log $LOG --small --workers 4 --trigger manual
  echo "=== arm B eval after cycle $i"
  python -m fleet_memory.runner.pool --env libero --policy smolvla --suite libero_10 --task 3 --arm B --n 20 --workers 4 --seed-set train --log $LOG
done
echo SLEEP31_DONE
