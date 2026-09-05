"""memory/drift.py: EWMA drift trigger — needs a baseline, fires on a cost spike / success drop,
respects the cooldown after a trigger or consolidation."""
from __future__ import annotations

import pytest

from fleet_memory.execution.params import S3Params
from fleet_memory.memory import drift, house_model as hm
from fleet_memory.memory.schema import (Consolidation, Episode, EpisodeMetrics, GateResult, Outcome,
                                        now_iso, skill_instance_id)
from fleet_memory.memory.store import EventStore

ENV = "mock_pick_bowl_to_plate"
SI = skill_instance_id("pick_bowl_to_plate", ENV)
CFG = drift.DriftConfig(alpha=0.3, cost_ratio=1.25, success_drop=0.15, window=10, cooldown_episodes=20)


@pytest.fixture
def store(tmp_path):
    s = EventStore(str(tmp_path / "events.jsonl"))
    hm.init_incumbent(s, SI, ENV, "pick_bowl_to_plate", "black_bowl", S3Params.identity().to_dict())
    return s


def mk_episode(i, version=1, cost=1.0, success=True, rollout=False):
    return Episode(episode_id=f"ep_{i}", suite="mock", task_id="pick_bowl_to_plate", seed=i, condition="B",
                   policy="mock", surfaces_enabled=["S3"], retrieved_lesson_ids=[], applied_lesson_ids=[],
                   retrieval_frozen_at=now_iso(), environment_id=ENV,
                   outcome=Outcome(env_success=success, steps=100, termination="success" if success else "timeout"),
                   skill_instance_versions={SI: version},
                   metrics=None if cost is None else EpisodeMetrics(100, success, 0.5, 0.1, cost=cost),
                   perturbation={"consolidation_id": "con_r"} if rollout else {})


def run(store, i, cfg=CFG, **kw):
    """Log one episode the way the worker does: append, scorecard, drift check."""
    ep = mk_episode(i, **kw)
    store.append(ep)
    hm.update_scorecard(store, SI, ep)
    return drift.update_and_check(store, SI, ep, cfg)


def n_triggers(store):
    return len(list(store.iter_type("drift_trigger")))


# ---------------------------------------------------------------------- no baseline
def test_no_incumbent_no_fire(tmp_path):
    s = EventStore(str(tmp_path / "e.jsonl"))
    assert drift.update_and_check(s, SI, mk_episode(0, cost=99.0), CFG) is None


def test_does_not_fire_before_window(store):
    for i in range(CFG.window - 1):
        assert run(store, i, cost=1.0 if i < 4 else 50.0) is None
    assert n_triggers(store) == 0


def test_does_not_fire_without_costs(store):
    for i in range(CFG.window + 5):
        assert run(store, i, cost=None) is None      # no cost reference -> metrics.cost is None
    assert n_triggers(store) == 0


def test_rollouts_and_other_versions_do_not_count(store):
    for i in range(30):
        store.append(mk_episode(i, cost=50.0, rollout=True))
        store.append(mk_episode(100 + i, version=7, cost=50.0))
    assert run(store, 200, cost=1.0) is None
    assert n_triggers(store) == 0


# ---------------------------------------------------------------------- cost spike
def test_fires_on_cost_spike_after_window(store):
    for i in range(CFG.window):
        assert run(store, i, cost=1.0) is None
    # baseline = mean of first window = 1.0; ewma crosses 1.25 after a few spiked episodes
    fired = None
    for i in range(CFG.window, CFG.window + 6):
        t = run(store, i, cost=3.0)
        if t is not None:
            fired = (i, t)
            break
    assert fired is not None
    i, t = fired
    assert t.reason == "ewma_cost_exceeds_baseline" and t.skill_instance_id == SI
    assert t.baseline_cost == pytest.approx(1.0) and t.ratio == pytest.approx(t.ewma_cost / t.baseline_cost)
    assert t.ratio > CFG.cost_ratio and t.episode_id == f"ep_{i}"
    ev = list(store.iter_type("drift_trigger"))
    assert len(ev) == 1 and ev[0]["episode_id"] == f"ep_{i}" and ev[0]["reason"] == t.reason
    # the scorecard's EWMA (same alpha) is what the trigger reports
    assert hm.incumbent(store, SI).scorecard.ewma_cost == pytest.approx(t.ewma_cost)
    assert drift.pending(store, SI) is not None and drift.pending(store, SI).episode_id == t.episode_id


def test_stable_cost_never_fires(store):
    for i in range(40):
        assert run(store, i, cost=1.0 + 0.2 * ((-1) ** i)) is None     # 0.8 / 1.2 alternating
    assert n_triggers(store) == 0


# ---------------------------------------------------------------------- cooldown
def test_cooldown_after_trigger(store):
    for i in range(CFG.window):
        run(store, i, cost=1.0)
    i = CFG.window
    while n_triggers(store) == 0:
        run(store, i, cost=3.0)
        i += 1
    first_i = i
    # keep spiking: nothing for the next cooldown_episodes - 1 episodes (the trigger's own episode counts)
    for k in range(CFG.cooldown_episodes - 1):
        assert run(store, i + k, cost=3.0) is None, f"fired during cooldown at offset {k}"
    assert n_triggers(store) == 1
    t = run(store, first_i + CFG.cooldown_episodes - 1, cost=3.0)
    assert t is not None and n_triggers(store) == 2


def test_cooldown_after_consolidation_event(store):
    for i in range(CFG.window):
        run(store, i, cost=1.0)
    store.append(Consolidation("con_1", "start", SI, "drift", 1))
    for k in range(CFG.cooldown_episodes - 1):
        assert run(store, 50 + k, cost=3.0) is None
    assert run(store, 99, cost=3.0) is not None
    assert drift.episodes_since_last_marker(store, "si_never") is None


def test_consolidation_end_clears_pending(store):
    for i in range(CFG.window):
        run(store, i, cost=1.0)
    i = CFG.window
    while n_triggers(store) == 0:
        run(store, i, cost=3.0)
        i += 1
    assert drift.pending(store, SI) is not None
    store.append(Consolidation("con_1", "end", SI, "drift", 1, promoted_version=2))
    assert drift.pending(store, SI) is None


# ---------------------------------------------------------------------- success drop
def test_fires_on_success_drop(store):
    for i in range(CFG.window):
        assert run(store, i, cost=1.0, success=True) is None
    fired = None
    for i in range(CFG.window, CFG.window + 10):
        t = run(store, i, cost=1.0, success=False)          # cost held flat on purpose
        if t is not None:
            fired = t
            break
    assert fired is not None and fired.reason == "success_rate_drop"
    assert fired.baseline_success_rate == pytest.approx(1.0)
    assert fired.recent_success_rate < 1.0 - CFG.success_drop


# ---------------------------------------------------------------------- promoted baseline
def test_promoted_version_uses_gate_baseline(store):
    g = GateResult(passed=True, incumbent_cost=2.0, candidate_cost=1.0, n_seeds=16,
                   incumbent_success=0.8, candidate_success=0.9)
    hm.promote(store, SI, S3Params(grasp_offset_z=-0.01).to_dict(), g, "con_1")
    for i in range(30):                                     # episodes on v2, cost 1.2 < 1.25 * 1.0
        assert run(store, i, version=2, cost=1.2, success=True) is None
    assert n_triggers(store) == 0
    fired = None
    for i in range(30, 45):
        t = run(store, i, version=2, cost=1.4, success=True)
        if t is not None:
            fired = t
            break
    assert fired is not None and fired.baseline_cost == 1.0 and fired.baseline_success_rate == 0.9
    assert fired.reason == "ewma_cost_exceeds_baseline"


def test_check_is_pure(store):
    for i in range(CFG.window):
        run(store, i, cost=1.0)
    for i in range(CFG.window, CFG.window + 6):
        store.append(mk_episode(i, cost=3.0))
    t = drift.check(store, SI, mk_episode(99, cost=3.0), CFG)
    assert t is not None and n_triggers(store) == 0
