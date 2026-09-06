#!/bin/bash
#SBATCH -p gpuq -q gpu --gres=gpu:A100.40gb:1 -c 8 --mem=48G -t 02:00:00 -J fm-mastery
#SBATCH -o /scratch/ezhao2/fleet-memory/logs/slurm/mastery_%j.out
# Held-out mastery eval: LIBERO-10 task 3, layouts 20-39 (unseen by the train arms), two noise reps -> n=40/arm.
set -uo pipefail
source /scratch/ezhao2/fleet-memory/dnhacks26/scripts/hopper/env.sh
export FM_LLM=mock FM_POLICY_KWARGS='{"n_action_steps":10}'
cd "$FM_REPO"; L=$FM_LOGS/mastery; mkdir -p $L
python - <<'PY'
import dataclasses, json, os
from fleet_memory.memory.store import EventStore
from fleet_memory.runner.pool import run_many, summarize, close_pool
from fleet_memory.runner.worker import RunConfig
from fleet_memory.analysis.metrics import wilson_ci
L = os.environ["FM_LOGS"] + "/mastery"
# copy the frozen cost reference so costs are comparable to the mastery run
src = os.environ["FM_LOGS"] + "/v31/events.jsonl"; store = EventStore(L + "/events.jsonl")
for line in open(src):
    if '"type":"cost_reference"' in line: store.append(json.loads(line)); break
V2 = json.loads('''{"approach_offset_xyz": [-0.0010608549763337152, 0.005532985016776307, 0.006068577779957035], "pregrasp_height": 0.05001680252868872, "grasp_offset_z": 0.0015934970062568936, "time_scale": 1.1477872504289357, "velocity_cap": 1.0, "gripper_cmd": 1.0, "approach_cone_deg": 53.576076552185505, "blend_alpha": 0.0, "homing_enable": 0.4151101399983246, "homing_pos_delta": [0.05, -0.009094765110368462, 0.010440750342220237], "homing_rot_delta": [-0.10477461320948336, -0.002192818652752504, 0.020901521799950945]}'''); V3 = json.loads('''{"approach_offset_xyz": [0.03, 0.003432377645559738, -0.013233494144099509], "pregrasp_height": 0.05308281769812562, "grasp_offset_z": -0.02, "time_scale": 0.872891611780205, "velocity_cap": 0.8603792601272338, "gripper_cmd": 0.8611672722899777, "approach_cone_deg": 33.38458193346288, "blend_alpha": 0.0, "homing_enable": 0.0, "homing_pos_delta": [0.04657390431807079, -0.037391304207733636, 0.011424685585671168], "homing_rot_delta": [-0.05198360446661303, 0.009265725145156167, -0.07046360613288444]}''')
base = RunConfig(suite="libero_10", task_id="3", seed=0, arm="A", env_kind="libero", policy_kind="smolvla", log_path=L + "/events.jsonl")
seeds = [1020 + i for i in range(20)] + [1070 + i for i in range(20)]   # init states 20-39, two policy-noise reps
arms = {"A": base, "B_v2": dataclasses.replace(base, arm="B", s3_params=V2), "B_v3": dataclasses.replace(base, arm="B", s3_params=V3)}
for name, c in arms.items():
    eps = run_many([dataclasses.replace(c, seed=s) for s in seeds], 4)
    k = sum(e.outcome.env_success for e in eps); lo, hi = wilson_ci(k, len(eps))
    costs = [e.metrics.cost for e in eps if e.metrics and e.metrics.cost is not None]
    print(f"HELDOUT {name}: {k}/{len(eps)} = {100*k/len(eps):.0f}% [{100*lo:.0f},{100*hi:.0f}] steps {sum(e.outcome.steps for e in eps)/len(eps):.0f} cost {sum(costs)/len(costs):.2f}", flush=True)
close_pool()
PY
echo MASTERY_DONE
