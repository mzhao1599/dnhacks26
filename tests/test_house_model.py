"""memory/house_model.py: versions, single incumbent, promotion chain, cost reference, scorecard, view."""
from __future__ import annotations

import pytest

from fleet_memory.execution.params import S3Params
from fleet_memory.memory import house_model as hm
from fleet_memory.memory.schema import (Consolidation, CostReference, Edit, Episode, EpisodeMetrics,
                                        GateResult, Lesson, Outcome, Trigger, now_iso, skill_instance_id)
from fleet_memory.memory.store import EventStore

ENV = "mock_pick_bowl_to_plate"
SI = skill_instance_id("pick_bowl_to_plate", ENV)


@pytest.fixture
def store(tmp_path):
    return EventStore(str(tmp_path / "events.jsonl"))


def gate(passed=True, inc=2.0, cand=1.5, inc_s=0.8, cand_s=0.85):
    return GateResult(passed=passed, incumbent_cost=inc, candidate_cost=cand, n_seeds=16,
                      incumbent_success=inc_s, candidate_success=cand_s, success_delta_pp=(cand_s - inc_s) * 100)


def mk_episode(i, version=None, cost=1.0, success=True, env=ENV, condition="D", rollout=False, si_id=SI):
    return Episode(episode_id=f"ep_{i}", suite="mock", task_id="pick_bowl_to_plate", seed=i, condition=condition,
                   policy="mock", surfaces_enabled=["S1", "S3"], retrieved_lesson_ids=[], applied_lesson_ids=[],
                   retrieval_frozen_at=now_iso(), environment_id=env,
                   outcome=Outcome(env_success=success, steps=100 + i, termination="success" if success else "timeout"),
                   skill_instance_versions={} if version is None else {si_id: version},
                   metrics=EpisodeMetrics(steps=100 + i, success=success, jerk=0.5, force_proxy=0.1, cost=cost),
                   perturbation={"consolidation_id": "con_x"} if rollout else {})


def init(store):
    return hm.init_incumbent(store, SI, ENV, "pick_bowl_to_plate", "black_bowl", S3Params.identity().to_dict())


# ---------------------------------------------------------------------- init / incumbent
def test_empty_store(store):
    assert hm.incumbent(store, SI) is None
    assert hm.versions(store, SI) == []
    assert hm.cost_reference(store, SI) is None
    assert hm.all_environments(store) == []


def test_init_incumbent_is_idempotent(store):
    v1 = init(store)
    again = init(store)
    assert v1.version == again.version == 1 and v1.status == "incumbent"
    assert v1.produced_by == {"loop": "init"} and v1.parent_version is None
    assert len(list(store.iter_type("skill_instance"))) == 1
    assert hm.incumbent(store, SI).params == S3Params.identity().to_dict()
    assert hm.all_environments(store) == [ENV]


# ---------------------------------------------------------------------- promotion chain
def test_promotion_chain_keeps_exactly_one_incumbent(store):
    init(store)
    p2 = S3Params(grasp_offset_z=-0.01).to_dict()
    v2 = hm.promote(store, SI, p2, gate(cand=1.5, cand_s=0.9), "con_1")
    assert (v2.version, v2.parent_version, v2.status) == (2, 1, "incumbent")
    assert v2.produced_by == {"loop": "sleep", "consolidation_id": "con_1"}
    assert v2.scorecard.baseline_cost == 1.5 and v2.scorecard.baseline_success_rate == 0.9
    v3 = hm.promote(store, SI, S3Params(time_scale=1.3).to_dict(), gate(inc=1.5, cand=1.2), "con_2")
    assert (v3.version, v3.parent_version) == (3, 2)
    vs = hm.versions(store, SI)
    assert [v.version for v in vs] == [1, 2, 3]
    assert [v.status for v in vs] == ["retired", "retired", "incumbent"]
    assert sum(v.status == "incumbent" for v in vs) == 1
    inc = hm.incumbent(store, SI)
    assert inc.version == 3 and inc.params == S3Params(time_scale=1.3).to_dict()
    assert inc.gate is not None and inc.gate.candidate_cost == 1.2
    status = list(store.iter_type("skill_instance_status"))
    assert [(s["version"], s["to_status"]) for s in status] == [(1, "retired"), (2, "retired")]
    # every prefix of the log holds at most one incumbent (retire precedes the new emission)
    rows = store.read_all()
    for k in range(len(rows) + 1):
        inc_n = 0
        st = {}
        for d in rows[:k]:
            if d["type"] == "skill_instance":
                st[d["version"]] = d["status"]
            elif d["type"] == "skill_instance_status":
                st[d["version"]] = d["to_status"]
        inc_n = sum(s == "incumbent" for s in st.values())
        assert inc_n <= 1


def test_promote_refuses_failed_gate(store):
    init(store)
    with pytest.raises(ValueError):
        hm.promote(store, SI, S3Params().to_dict(), gate(passed=False), "con_bad")
    assert hm.incumbent(store, SI).version == 1


def test_promote_without_incumbent_starts_at_v1(store):
    v1 = hm.promote(store, SI, S3Params().to_dict(), gate(), "con_0")
    assert (v1.version, v1.parent_version, v1.environment_id, v1.skill) == (1, None, ENV, "pick_bowl_to_plate")
    assert hm.incumbent(store, SI).version == 1


def test_retire_and_reinit(store):
    init(store)
    ch = hm.retire(store, SI, 1, "manual")
    assert ch is not None and ch.to_status == "retired"
    assert hm.incumbent(store, SI) is None
    assert hm.retire(store, SI, 1, "again") is None          # already retired
    assert hm.retire(store, SI, 9, "nope") is None           # unknown version
    v = init(store)
    assert v.version == 2 and v.parent_version is None and hm.incumbent(store, SI).version == 2


# ---------------------------------------------------------------------- cost reference
def test_cost_reference_latest_wins(store):
    hm.set_cost_reference(store, CostReference(SI, ENV, steps_ref=120.0, jerk_ref=0.01, n_episodes=20))
    hm.set_cost_reference(store, CostReference("si_other", ENV, steps_ref=1.0, jerk_ref=1.0, n_episodes=1))
    hm.set_cost_reference(store, CostReference(SI, ENV, steps_ref=110.0, jerk_ref=0.02, n_episodes=40))
    ref = hm.cost_reference(store, SI)
    assert isinstance(ref, CostReference) and (ref.steps_ref, ref.n_episodes) == (110.0, 40)
    assert hm.cost_reference(store, "si_missing") is None


# ---------------------------------------------------------------------- scorecard
def test_update_scorecard_tracks_incumbent_version_only(store):
    init(store)
    with pytest.raises(ValueError):
        hm.update_scorecard(store, "si_missing", mk_episode(0, 1))
    costs = [1.0, 2.0, 1.5]
    sc = None
    for i, c in enumerate(costs):
        ep = mk_episode(i, 1, cost=c, success=(i != 1))
        store.append(ep)
        sc = hm.update_scorecard(store, SI, ep)
    assert (sc.n, sc.successes) == (3, 2)
    e = 1.0
    for c in costs[1:]:
        e = 0.3 * c + 0.7 * e
    assert sc.ewma_cost == pytest.approx(e)
    assert sc.mean_steps == pytest.approx(101.0)
    assert sc.baseline_cost is None
    # not-yet-logged episode is counted once; rollouts and other versions are ignored
    ep10 = mk_episode(10, 1, cost=1.0)
    sc2 = hm.update_scorecard(store, SI, ep10)
    assert sc2.n == 4
    store.append(ep10)
    assert hm.update_scorecard(store, SI, ep10).n == 4          # deduped by episode_id once logged
    store.append(mk_episode(11, 1, cost=9.0, rollout=True))
    store.append(mk_episode(12, 2, cost=9.0))
    store.append(mk_episode(13, None, cost=9.0))
    sc3 = hm.update_scorecard(store, SI, mk_episode(11, 1, cost=9.0, rollout=True))
    assert sc3.n == 4 and sc3.ewma_cost == pytest.approx(sc2.ewma_cost)
    events = list(store.iter_type("scorecard"))
    assert len(events) == 6 and events[-1]["version"] == 1 and events[-1]["episode_id"] == "ep_11"
    assert hm.incumbent(store, SI).scorecard.n == 4               # latest scorecard event is authoritative


def test_scorecard_baselines_carried_after_promotion(store):
    init(store)
    hm.promote(store, SI, S3Params().to_dict(), gate(cand=1.4, cand_s=0.75), "con_1")
    ep = mk_episode(0, 2, cost=1.3)
    store.append(ep)
    sc = hm.update_scorecard(store, SI, ep)
    assert (sc.baseline_cost, sc.baseline_success_rate, sc.n) == (1.4, 0.75, 1)
    sc = hm.update_scorecard(store, SI, mk_episode(1, 2, cost=1.1))
    assert sc.baseline_cost == 1.4 and sc.n == 2
    assert hm.versions(store, SI)[1].scorecard.ewma_cost == pytest.approx(0.3 * 1.1 + 0.7 * 1.3)


# ---------------------------------------------------------------------- view
def test_view_and_all_environments(store):
    init(store)
    hm.promote(store, SI, S3Params().to_dict(), gate(), "con_1")
    hm.set_cost_reference(store, CostReference(SI, ENV, steps_ref=100.0, jerk_ref=0.01, n_episodes=10))
    for i, (c, ok) in enumerate([(1.0, True), (2.0, False), (1.5, True)]):
        store.append(mk_episode(i, 2, cost=c, success=ok))
    store.append(mk_episode(7, 2, cost=50.0, rollout=True))
    store.append(mk_episode(8, None, cost=50.0, env="mock_other", condition="A"))
    for lid, env, status in [("les_any", None, "candidate"), ("les_here", ENV, "validated"),
                             ("les_there", "mock_other", "candidate"), ("les_dead", None, "retired")]:
        store.append(Lesson(lesson_id=lid, surface="S1", trigger=Trigger("pick_place", "bowl", "grasp", "always", env),
                            edit=Edit("add_precondition", {"subtask_id": "s1", "predicate": "gripper_open"}),
                            status=status))
    store.append(Consolidation("con_1", "start", SI, "manual", 1))
    store.append(Consolidation("con_1", "end", SI, "manual", 1, promoted_version=2))
    store.append(Consolidation("con_z", "end", "si_other", "manual", 1))
    store.append({"type": "drift_trigger", "skill_instance_id": SI, "reason": "ewma_cost_exceeds_baseline"})
    v = hm.view(store, ENV)
    assert v["environment_id"] == ENV and v["tasks"] == ["pick_bowl_to_plate"] and v["objects"] == ["black_bowl"]
    si = v["skill_instances"]
    assert len(si) == 1 and si[0]["skill_instance_id"] == SI and si[0]["incumbent_version"] == 2
    assert si[0]["n_versions"] == 2 and si[0]["cost_reference"]["steps_ref"] == 100.0
    assert [x["status"] for x in si[0]["versions"]] == ["retired", "incumbent"]
    assert si[0]["versions"][1]["gate"]["passed"] is True
    assert sorted(l["lesson_id"] for l in v["lessons"]) == ["les_any", "les_here"]
    assert [c["consolidation_id"] for c in v["consolidations"]] == ["con_1"]
    assert len(v["drift_triggers"]) == 1
    assert v["episodes"]["n"] == 3 and v["episodes"]["success_rate"] == pytest.approx(2 / 3)
    assert v["episodes"]["mean_cost"] == pytest.approx(1.5)
    assert hm.all_environments(store) == sorted([ENV, "mock_other"])
    other = hm.view(store, "mock_other")
    assert other["skill_instances"] == [] and other["episodes"]["n"] == 1
    assert sorted(l["lesson_id"] for l in other["lessons"]) == ["les_any", "les_there"]
