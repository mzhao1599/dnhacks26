"""Does a stronger gate (24 seeds, K=4) refuse regressions the 12-seed gate let through? Start a fresh skill
instance at the promoted v2 vector, run one CEM cycle, then evaluate the resulting incumbent held-out (n=40)."""
import dataclasses, json, os
from fleet_memory.memory import house_model
from fleet_memory.memory.store import EventStore
from fleet_memory.runner.consolidate import ConsolidationConfig, consolidate
from fleet_memory.runner.pool import run_many, close_pool
from fleet_memory.runner.worker import RunConfig
from fleet_memory.analysis.metrics import wilson_ci


def main():
    L = os.environ["FM_LOGS"]
    LOG = f"{L}/mastery_g24/events.jsonl"
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    store = EventStore(LOG)
    v2 = None
    for line in open(f"{L}/v31/events.jsonl"):
        try: d = json.loads(line)
        except Exception: continue
        if d.get("type") == "cost_reference": store.append({**d, "skill_instance_id": "si_3__libero_10_3_g24", "environment_id": "libero_10_3_g24"})
        if d.get("type") == "skill_instance" and d["version"] == 2: v2 = d["params"]
    tmpl = RunConfig(suite="libero_10", task_id="3", seed=0, arm="B", env_kind="libero", policy_kind="smolvla", log_path=LOG, environment_tag="g24")
    house_model.init_incumbent(store, tmpl.skill_instance_id, tmpl.environment_id, "3", "akita_black_bowl_1", v2)
    cfg = ConsolidationConfig(population=16, elites=4, iterations=3, seeds_per_candidate=4, gate_seeds=24, workers=4)
    c = consolidate(store, tmpl.skill_instance_id, template=tmpl, cfg=cfg, trigger="manual")
    print("GATE24", json.dumps(c.gate.to_dict() if c.gate else None), "promoted", c.promoted_version, flush=True)
    inc = house_model.incumbent(store, tmpl.skill_instance_id)
    seeds = [1020 + i for i in range(20)] + [1070 + i for i in range(20)]
    eps = run_many([dataclasses.replace(tmpl, seed=s, s3_params=dict(inc.params)) for s in seeds], 4)
    k = sum(e.outcome.env_success for e in eps); lo, hi = wilson_ci(k, len(eps))
    costs = [e.metrics.cost for e in eps if e.metrics and e.metrics.cost is not None]
    print(f"HELDOUT g24 incumbent v{inc.version}: {k}/{len(eps)} = {100*k/len(eps):.0f}% [{100*lo:.0f},{100*hi:.0f}] steps {sum(e.outcome.steps for e in eps)/len(eps):.0f} cost {sum(costs)/max(len(costs),1):.2f}", flush=True)
    close_pool()


if __name__ == "__main__":
    main()
