"""Held-out mastery eval: LIBERO-10 task 3, init states 20-39 (unseen by the train arms), two policy-noise reps -> n=40/arm."""
import dataclasses, json, os
from fleet_memory.memory.store import EventStore
from fleet_memory.runner.pool import run_many, close_pool
from fleet_memory.runner.worker import RunConfig
from fleet_memory.analysis.metrics import wilson_ci

V2 = json.loads('''{"approach_offset_xyz": [-0.0010608549763337152, 0.005532985016776307, 0.006068577779957035], "pregrasp_height": 0.05001680252868872, "grasp_offset_z": 0.0015934970062568936, "time_scale": 1.1477872504289357, "velocity_cap": 1.0, "gripper_cmd": 1.0, "approach_cone_deg": 53.576076552185505, "blend_alpha": 0.0, "homing_enable": 0.4151101399983246, "homing_pos_delta": [0.05, -0.009094765110368462, 0.010440750342220237], "homing_rot_delta": [-0.10477461320948336, -0.002192818652752504, 0.020901521799950945]}''')
V3 = json.loads('''{"approach_offset_xyz": [0.03, 0.003432377645559738, -0.013233494144099509], "pregrasp_height": 0.05308281769812562, "grasp_offset_z": -0.02, "time_scale": 0.872891611780205, "velocity_cap": 0.8603792601272338, "gripper_cmd": 0.8611672722899777, "approach_cone_deg": 33.38458193346288, "blend_alpha": 0.0, "homing_enable": 0.0, "homing_pos_delta": [0.04657390431807079, -0.037391304207733636, 0.011424685585671168], "homing_rot_delta": [-0.05198360446661303, 0.009265725145156167, -0.07046360613288444]}''')


def main(workers=4):
    L = os.environ["FM_LOGS"] + "/mastery"
    os.makedirs(L, exist_ok=True)
    store = EventStore(L + "/events.jsonl")
    for line in open(os.environ["FM_LOGS"] + "/v31/events.jsonl"):   # frozen cost reference -> comparable costs
        if '"type":"cost_reference"' in line:
            store.append(json.loads(line)); break
    base = RunConfig(suite="libero_10", task_id="3", seed=0, arm="A", env_kind="libero", policy_kind="smolvla", log_path=L + "/events.jsonl")
    seeds = [1020 + i for i in range(20)] + [1070 + i for i in range(20)]
    arms = {"A": base, "B_v2": dataclasses.replace(base, arm="B", s3_params=V2), "B_v3": dataclasses.replace(base, arm="B", s3_params=V3)}
    for name, c in arms.items():
        eps = run_many([dataclasses.replace(c, seed=s) for s in seeds], workers)
        k = sum(e.outcome.env_success for e in eps); lo, hi = wilson_ci(k, len(eps))
        costs = [e.metrics.cost for e in eps if e.metrics and e.metrics.cost is not None]
        print(f"HELDOUT {name}: {k}/{len(eps)} = {100 * k / len(eps):.0f}% [{100 * lo:.0f},{100 * hi:.0f}] steps "
              f"{sum(e.outcome.steps for e in eps) / len(eps):.0f} cost {sum(costs) / max(len(costs), 1):.2f}", flush=True)
    close_pool()


if __name__ == "__main__":
    main()
