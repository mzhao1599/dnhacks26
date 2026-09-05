"""agents/: llm mock, planner + lesson application, inner/outer coaches. All offline (FM_LLM=mock)."""
import os

os.environ["FM_LLM"] = "mock"

import numpy as np
import pytest

from fleet_memory.agents import coach_inner, coach_outer, llm, planner
from fleet_memory.agents.coach_inner import InnerCoach, scrub
from fleet_memory.agents.coach_outer import OuterCoach
from fleet_memory.agents.llm import LLM, MOCK_RESPONDERS, image_block, parse_context, validate_lesson, with_context
from fleet_memory.agents.planner import Planner, apply_plan_edit, identity_plan, instruction_vocab
from fleet_memory.envs.base import Obs, TaskInfo
from fleet_memory.execution.constraints import ConstraintSet
from fleet_memory.memory.schema import (EDIT_OPS, PREDICATES, Edit, Episode, Intervention, Lesson, Outcome,
                                        Trigger)

TASK = TaskInfo(suite="mock", task_id="pick_bowl_to_plate", language="pick up the black bowl and place it on the plate",
                task_family="pick_place", objects=["black_bowl", "plate", "mug"], max_steps=200)


def make_obs(t=0, ee=(0.0, 0.0, 0.2)):
    objs = {"black_bowl": (0.1, 0.0, 0.05), "plate": (0.3, 0.1, 0.02), "mug": (-0.1, 0.1, 0.05)}
    q = np.array([0, 0, 0, 1], np.float32)
    return Obs(t=t, images={"agentview": np.zeros((16, 16, 3), np.uint8)}, state=np.zeros(8, np.float32),
               ee_pos=np.array(ee, np.float32), ee_quat=q, gripper_qpos=np.array([0.04, 0.04], np.float32),
               object_poses={k: (np.array(v, np.float32), q) for k, v in objs.items()})


class FakeTracker:
    def __init__(self, stalled=False, stall_steps=0):
        self.stalled, self.stall_steps, self.target = stalled, stall_steps, "black_bowl"

    def no_progress(self):
        return self.stalled

    def initial_z(self, name):
        return 0.05


def lesson(surface, op, params, predicate="always", phase="*"):
    return Lesson(lesson_id="les_x", surface=surface, trigger=Trigger("pick_place", "bowl", phase, predicate),
                  edit=Edit(op, params))


# --------------------------------------------------------------------------- llm

def test_llm_mock_selection_and_roundtrip(monkeypatch):
    assert LLM("m").mock is True
    monkeypatch.setenv("FM_LLM", "real")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert LLM("m").mock is True                     # no key -> mock
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert LLM("m").mock is False
    assert LLM("m", mock=True).mock is True
    out, meta = LLM("m", mock=True).complete_json("s", with_context("u", {"objects": ["a", "b"], "language": "a to b"}),
                                                  {"title": "plan"})
    assert out["subtasks"] and set(meta) == {"tokens_in", "tokens_out", "latency_s", "model"}
    with pytest.raises(KeyError):
        LLM("m", mock=True).complete_json("s", "u", {"title": "nope"})


def test_context_roundtrip_and_images():
    ctx = {"a": np.float32(1.5), "b": np.arange(3), "c": {"d": [1, "x"]}}
    assert parse_context(with_context("hello", ctx)) == {"a": 1.5, "b": [0, 1, 2], "c": {"d": [1, "x"]}}
    assert parse_context("no context here") == {}
    blk = image_block(np.zeros((8, 6, 3), np.uint8))
    assert blk["type"] == "image" and blk["source"]["media_type"] in ("image/png", "image/jpeg")
    assert llm._png_bytes(np.zeros((4, 4, 3), np.uint8)).startswith(b"\x89PNG")


def test_validate_lesson_rules():
    assert validate_lesson(lesson("S3", "set_grasp_offset", {"dz": -0.015}))[0]
    assert not validate_lesson(lesson("S3", "set_grasp_offset", {"dz": 0.5}))[0]          # >10 cm
    assert not validate_lesson(lesson("S3", "set_grasp_offset", {"dq": 0.1}))[0]          # unknown param
    assert not validate_lesson(lesson("S3", "fly", {}))[0]
    assert not validate_lesson(lesson("S4", "set_grasp_offset", {}))[0]
    assert not validate_lesson(lesson("S3", "set_grasp_offset", {}, predicate="bogus"))[0]
    assert not validate_lesson(lesson("S3", "set_abort_retry", {"predicate": "bogus", "max_retries": 1}))[0]
    assert validate_lesson(lesson("S1", "set_abort_condition", {"subtask_id": "grasp_1", "predicate": "no_progress", "max_steps": 40}))[0]
    assert validate_lesson(lesson("S2", "set_instruction", {"instruction": "x"}).to_dict())[0]


# --------------------------------------------------------------------------- planner

def test_vocab_and_identity_plan():
    v = instruction_vocab(TASK)
    assert v[0] == TASK.language and len(v) == 4 and len(set(v)) == 4      # canonical already lowercase -> dedupe
    t2 = TaskInfo("mock", "t", "Pick up the black bowl, then place it on the plate.", "pick_place", TASK.objects, 200)
    assert len(instruction_vocab(t2)) == 5
    p = identity_plan(TASK)
    assert p.source == "identity" and len(p.subtasks) == 1
    assert p.subtasks[0].skill == "custom" and p.subtasks[0].target_object == "black_bowl"
    assert p.subtasks[0].instruction == TASK.language and p.instruction_vocab == v


def test_planner_mock_decomposition():
    plan = Planner().plan(TASK, make_obs(), [])
    assert [s.skill for s in plan.subtasks] == ["reach", "grasp", "lift", "move", "place"]
    assert all(s.target_object == "black_bowl" for s in plan.subtasks)
    assert plan.subtasks[-1].destination == "plate" and plan.subtasks[0].destination is None
    assert all(s.instruction == TASK.language for s in plan.subtasks)
    assert plan.source == "planner" and plan.task_id == TASK.task_id
    assert len({s.subtask_id for s in plan.subtasks}) == 5
    # plan record serialises
    assert plan.to_dict()["subtasks"][1]["skill"] == "grasp"


def test_planner_sanitises_bad_llm_output():
    class Bad:
        mock = True
        def complete_json(self, *a, **k):
            return {"subtasks": [{"subtask_id": "x", "skill": "teleport", "target_object": "unicorn",
                                  "instruction": "not in vocab", "preconditions": ["bogus", "gripper_open"],
                                  "abort_predicate": "nope", "max_steps": 99999}]}, {"tokens_in": 1, "tokens_out": 1, "latency_s": 0, "model": "m"}
    plan = Planner(Bad()).plan(TASK, None, [])
    st = plan.subtasks[0]
    assert st.skill == "custom" and st.target_object == "black_bowl" and st.instruction == TASK.language
    assert st.preconditions == ["gripper_open"] and st.abort_predicate is None and st.max_steps == TASK.max_steps

    class Boom:
        mock = True
        def complete_json(self, *a, **k):
            raise RuntimeError("network")
    plan = Planner(Boom()).plan(TASK, None, [])
    assert plan.source == "identity" and "planner_error" in plan.rationale_text


def test_planner_applies_s1_s2_lessons():
    vocab = instruction_vocab(TASK)
    ls = [
        lesson("S2", "set_instruction", {"instruction": vocab[2]}, phase="grasp"),
        lesson("S2", "set_instruction", {"instruction": "definitely not in vocab"}),
        lesson("S1", "rebind_object", {"subtask_id": "reach_0", "object": "mug"}),
        lesson("S1", "rebind_object", {"subtask_id": "lift_2", "object": "unicorn"}),
        lesson("S1", "add_precondition", {"subtask_id": "grasp_1", "predicate": "ee_above_target"}),
        lesson("S1", "set_abort_condition", {"subtask_id": "grasp_99", "predicate": "grasp_slipped", "max_steps": 33}),
        lesson("S3", "set_grasp_offset", {"dz": -0.015}),        # ignored by the planner
    ]
    plan = Planner().plan(TASK, make_obs(), ls)
    by = {s.skill: s for s in plan.subtasks}
    assert by["grasp"].instruction == vocab[2] and by["reach"].instruction == TASK.language
    assert by["reach"].target_object == "mug" and by["lift"].target_object == "black_bowl"
    assert by["grasp"].preconditions == ["ee_above_target"]
    assert by["grasp"].abort_predicate == "grasp_slipped" and by["grasp"].max_steps == 33   # matched by skill prefix
    # replan replaces everything
    ok = apply_plan_edit(plan, "S1", Edit("replan", {"subtasks": [
        {"subtask_id": "push_0", "skill": "push", "target_object": "mug", "max_steps": 50}]}), TASK)
    assert ok and [s.skill for s in plan.subtasks] == ["push"]
    assert not apply_plan_edit(plan, "S1", Edit("replan", {"subtasks": []}), TASK)
    assert not apply_plan_edit(plan, "S3", Edit("set_grasp_offset", {"dz": 0.0}), TASK)


# --------------------------------------------------------------------------- inner coach

def test_inner_coach_triggers_and_mock_sequence():
    c = InnerCoach(check_interval=60, stall_k=40)
    obs, plan = make_obs(), Planner().plan(TASK, make_obs(), [])
    grasp = plan.subtasks[1]
    args = dict(subtask=grasp, plan=plan, constraints=ConstraintSet(), keyframes=[obs.images["agentview"]],
                trace_summary={"success_subconditions": {"grasped": False}})
    assert c.maybe_intervene("ep1", 5, obs, tracker=FakeTracker(False), **args) is None      # no trigger
    assert c.maybe_intervene("ep1", 60, obs, tracker=FakeTracker(False), **args) is None     # interval -> mock says nothing
    iv = c.maybe_intervene("ep1", 101, obs, tracker=FakeTracker(True, 41), **args)
    assert isinstance(iv, Intervention) and iv.trigger == "stall" and iv.surface == "S3"
    assert iv.edit.op == "set_abort_retry" and iv.edit.params["predicate"] == "grasp_slipped"
    assert iv.step == 101 and iv.episode_id == "ep1" and iv.coach_model.startswith("mock:")
    assert c.maybe_intervene("ep1", 102, obs, tracker=FakeTracker(True, 42), **args) is None   # cooldown
    iv2 = c.maybe_intervene("ep1", 141, obs, tracker=FakeTracker(True, 81), **args)
    assert iv2.edit.op == "set_grasp_offset" and iv2.edit.params["dz"] == pytest.approx(-0.01)
    # new episode resets history -> abort_retry again
    iv3 = c.maybe_intervene("ep2", 50, obs, tracker=FakeTracker(True, 40), **args)
    assert iv3.edit.op == "set_abort_retry"
    assert c.queries == 4 and c.rejected == 0       # steps 60, 101, 141, ep2/50


def test_inner_coach_never_sees_success(monkeypatch):
    seen = {}

    def spy(system, user, schema, images=None, max_tokens=2000):
        seen["system"], seen["user"] = system, user
        return {"surface": None, "rationale": ""}, {"tokens_in": 0, "tokens_out": 0, "latency_s": 0, "model": "m"}
    c = InnerCoach()
    monkeypatch.setattr(c.llm, "complete_json", spy)
    trace = {"success_subconditions": {"grasped": True}, "env_success": True, "outcome": {"x": 1},
             "nested": {"success_flag": 1, "keep": 2}, "steps": 10}
    assert c.maybe_intervene("ep", 70, make_obs(), None, None, ConstraintSet(), [], FakeTracker(True, 40), trace) is None
    assert "success" not in seen["user"].lower() and "success" not in seen["system"].lower()
    ctx = parse_context(seen["user"])
    assert ctx["trace"] == {"progress_signals": {"grasped": True}, "nested": {"keep": 2}, "steps": 10}
    assert scrub([{"Success": 1}, {"ok": 1}]) == [{}, {"ok": 1}]


def test_inner_coach_rejects_invalid_edits(monkeypatch):
    c = InnerCoach()
    bad = iter([
        {"surface": "S3", "op": "set_grasp_offset", "params": {"dz": 0.9}, "rationale": ""},        # >10 cm
        {"surface": "S2", "op": "set_instruction", "params": {"instruction": "x"}, "rationale": ""},  # not an intervention surface
        {"surface": "S1", "op": "add_precondition", "params": {"subtask_id": "g", "predicate": "bogus"}, "rationale": ""},
        {"surface": "S1", "op": "set_abort_condition", "params": {"subtask_id": "g", "predicate": "no_progress", "max_steps": 20}, "rationale": ""},
    ])
    monkeypatch.setattr(c.llm, "complete_json", lambda *a, **k: (next(bad), {"tokens_in": 0, "tokens_out": 0, "latency_s": 0, "model": "m"}))
    common = (make_obs(), None, None, ConstraintSet(), [])
    assert c.maybe_intervene("e", 40, *common, FakeTracker(True, 40), {}) is None
    assert c.maybe_intervene("e", 80, *common, FakeTracker(True, 80), {}) is None
    assert c.maybe_intervene("e", 120, *common, FakeTracker(True, 120), {}) is None
    iv = c.maybe_intervene("e", 160, *common, FakeTracker(True, 160), {})
    assert iv is not None and iv.surface == "S1" and c.rejected == 3


# --------------------------------------------------------------------------- outer coach

def episode(success, plan=None, held_out=False):
    return Episode(episode_id="ep_1", suite="mock", task_id=TASK.task_id, seed=0, condition="D", policy="mock",
                   surfaces_enabled=["S1", "S2", "S3"], retrieved_lesson_ids=[], applied_lesson_ids=[],
                   retrieval_frozen_at="t", plan=plan, instruction_used=TASK.language,
                   outcome=Outcome(env_success=success, steps=150, termination="success" if success else "timeout"),
                   held_out=held_out)


def test_outer_coach_mock_lesson_on_failure():
    plan = Planner().plan(TASK, make_obs(), [])
    oc = OuterCoach()
    ls = oc.assign_credit(episode(False, plan), TASK, [np.zeros((8, 8, 3), np.uint8)] * 3,
                          {"subtask_starts": {"grasp_1": 37}}, [])
    assert len(ls) == 1
    l = ls[0]
    assert l.surface == "S3" and l.edit.op == "set_grasp_offset" and l.edit.params["dz"] == pytest.approx(-0.015)
    assert l.trigger.task_family == "pick_place" and l.trigger.object_class == "bowl"
    assert l.trigger.phase == "grasp" and l.trigger.predicate == "object_height_below_rim"
    assert l.status == "candidate" and l.provenance["from_episodes"] == ["ep_1"]
    assert l.provenance["coach_model"].startswith("mock:") and l.provenance["localisation"]["step"] == 37
    assert "bowl" in l.provenance["situation_text"] and l.lesson_id.startswith("les_")
    assert validate_lesson(l)[0] and l.to_dict()["type"] == "lesson"
    assert oc.rejected == 0
    # success -> nothing; identity plan (no grasp subtask) -> nothing; held-out -> nothing; no outcome -> nothing
    assert oc.assign_credit(episode(True, plan), TASK, [], {}, []) == []
    assert oc.assign_credit(episode(False, identity_plan(TASK)), TASK, [], {}, []) == []
    assert oc.assign_credit(episode(False, plan, held_out=True), TASK, [], {}, []) == []
    ep = episode(False, plan)
    ep.outcome = None
    assert oc.assign_credit(ep, TASK, [], {}, []) == []


def test_outer_coach_rejects_invalid_and_unlocalised(monkeypatch):
    oc = OuterCoach()
    plan = Planner().plan(TASK, make_obs(), [])
    good = {"surface": "S3", "trigger": {"task_family": "*", "object_class": "*", "phase": "*", "predicate": "always"},
            "op": "set_velocity_cap", "params": {"max_pos_delta": 0.5}, "rationale": "slow"}
    resp = {"localisation": {"step": 12, "description": "d"}, "lessons": [
        good,
        {**good, "op": "set_approach_vector", "params": {"vector": [0, 0, 0]}},                 # zero vector
        {**good, "surface": "S2", "op": "set_instruction", "params": {"instruction": "nope"}},   # not in vocab
        {**good, "surface": "S2", "op": "set_instruction", "params": {"instruction": plan.instruction_vocab[1]}},
        {**good, "trigger": {**good["trigger"], "predicate": "bogus"}},
        "garbage",
    ]}
    monkeypatch.setattr(oc.llm, "complete_json", lambda *a, **k: (resp, {"tokens_in": 1, "tokens_out": 1, "latency_s": 0, "model": "m"}))
    ls = oc.assign_credit(episode(False, plan), TASK, [], {}, [])
    assert [l.edit.op for l in ls] == ["set_velocity_cap", "set_instruction"] and oc.rejected == 3
    assert all(l.provenance["localisation"]["step"] == 12 for l in ls)
    resp["localisation"]["step"] = None
    assert oc.assign_credit(episode(False, plan), TASK, [], {}, []) == [] and oc.rejected == 8

    def boom(*a, **k):
        raise RuntimeError("api down")
    monkeypatch.setattr(oc.llm, "complete_json", boom)
    assert oc.assign_credit(episode(False, plan), TASK, [], {}, []) == []


# --------------------------------------------------------------------------- prompts

def test_prompt_contracts():
    for text in (coach_inner.INNER_SYSTEM, coach_outer.OUTER_SYSTEM):
        for s in ("S1", "S2", "S3"):
            for op, params in EDIT_OPS[s].items():
                assert op in text and all(k in text for k in params)
        assert all(p in text for p in PREDICATES)
        assert "valid" in text and "reward" in text and "step index" in text
    assert "success" not in coach_inner.INNER_SYSTEM.lower()
    assert "env_success" in coach_outer.OUTER_SYSTEM
    assert all(p in planner.PLANNER_SYSTEM for p in PREDICATES)
    assert set(MOCK_RESPONDERS) >= {"plan", "intervention", "lessons"}
    assert llm.PLANNER_MODEL == "claude-haiku-4-5-20251001" and llm.COACH_MODEL == "claude-sonnet-5"
