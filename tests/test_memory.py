"""memory/: store, retrieval, lifecycle. Runs offline (hash embedder, no env/policy needed)."""
from __future__ import annotations

import json
import multiprocessing as mp
import os

import numpy as np
import pytest

from fleet_memory.envs.base import TaskInfo
from fleet_memory.memory import lifecycle
from fleet_memory.memory.retrieval import Embedder, Retriever, cosine, situation_text
from fleet_memory.memory.schema import (Edit, Episode, Evidence, Lesson, LessonStatusChange, Outcome,
                                        Trigger, now_iso)
from fleet_memory.memory.store import EventStore


@pytest.fixture(autouse=True)
def _hash_embedder(monkeypatch):
    monkeypatch.setenv("FM_EMBEDDER", "hash")


@pytest.fixture
def store(tmp_path):
    return EventStore(str(tmp_path / "events.jsonl"))


TASK = TaskInfo(suite="mock", task_id="pick_bowl_to_plate", language="put the black bowl on the plate",
                task_family="pick_place", objects=["black_bowl", "plate", "mug"], max_steps=200)


def mk_lesson(lid, surface="S3", op="set_grasp_offset", params=None, family="pick_place", obj="bowl",
              phase="grasp", predicate="object_height_below_rim", status="candidate", text="", rationale=""):
    return Lesson(lesson_id=lid, surface=surface, trigger=Trigger(family, obj, phase, predicate),
                  edit=Edit(op, params if params is not None else {"dz": -0.015}), rationale_text=rationale,
                  status=status, provenance={"situation_text": text} if text else {})


def mk_episode(eid, seed, retrieved, applied, success, task_id=TASK.task_id):
    return Episode(episode_id=eid, suite="mock", task_id=task_id, seed=seed, condition="D", policy="mock",
                   surfaces_enabled=["S1", "S2", "S3"], retrieved_lesson_ids=list(retrieved),
                   applied_lesson_ids=list(applied), retrieval_frozen_at=now_iso(),
                   outcome=Outcome(env_success=success, steps=100, termination="success" if success else "timeout"))


def add_ab_episodes(store, lid, treated_ok, treated_n, control_ok, control_n, start=0):
    """Append treated_n episodes with the lesson applied (treated_ok successes) and control_n withheld."""
    i = start
    for j in range(treated_n):
        store.append(mk_episode(f"ep_{i}", i, [lid], [lid], j < treated_ok)); i += 1
    for j in range(control_n):
        store.append(mk_episode(f"ep_{i}", i, [lid], [], j < control_ok)); i += 1
    return i


# ------------------------------------------------------------------------------ store
def test_store_roundtrip_and_types(store):
    l = mk_lesson("les_1")
    store.append(l)
    store.append({"type": "custom", "x": np.float32(1.5), "arr": np.arange(2)})
    rows = store.read_all()
    assert [r["type"] for r in rows] == ["lesson", "custom"]
    assert rows[1] == {"type": "custom", "x": 1.5, "arr": [0, 1]}
    with open(store.path) as f:
        assert len(f.read().splitlines()) == 2      # one line per record
    assert list(store.iter_type("lesson"))[0]["lesson_id"] == "les_1"
    assert store.lessons()["les_1"].edit.params == {"dz": -0.015}
    assert store.episodes() == [] and store.snapshots() == []


def test_store_missing_file_and_torn_line(tmp_path):
    s = EventStore(str(tmp_path / "nope" / "events.jsonl"))
    assert s.read_all() == []
    s.append({"type": "a"})
    with open(s.path, "a") as f:
        f.write('{"type": "b", "trunc')            # crashed writer
    s.append({"type": "c"})
    assert [r["type"] for r in s.read_all()] == ["a", "c"]


def test_store_lessons_materialise_latest_status(store):
    store.append(mk_lesson("les_1"))
    store.append(mk_lesson("les_2"))
    ev = Evidence(trials=40, treated_n=20, treated_successes=18, control_n=20, control_successes=6, p_value=0.001, lift_pp=60.0)
    store.append(LessonStatusChange("les_1", "candidate", "validated", "promoted", ev))
    store.append(LessonStatusChange("les_1", "validated", "retired", "regression", ev))
    store.append(LessonStatusChange("les_ghost", "candidate", "retired", "no lesson", Evidence()))
    ls = store.lessons()
    assert set(ls) == {"les_1", "les_2"}
    assert ls["les_1"].status == "retired" and ls["les_1"].evidence.treated_n == 20
    assert ls["les_2"].status == "candidate"


def _writer(path, wid, n):
    s = EventStore(path)
    for i in range(n):
        s.append({"type": "ping", "w": wid, "i": i, "pad": "x" * (300 + 7 * wid)})


def test_store_concurrent_appenders(tmp_path):
    path = str(tmp_path / "events.jsonl")
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_writer, args=(path, w, 150)) for w in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    with open(path, "rb") as f:
        raw = f.read()
    assert raw.endswith(b"\n")
    # append() may prepend an extra "\n" when its torn-tail check races another writer (documented as
    # harmless; readers skip blank lines), so count records, not raw lines.
    lines = [x for x in raw.split(b"\n") if x]
    assert len(lines) == 600
    rows = [json.loads(x) for x in lines]            # every line parses => no interleaving
    assert len({(r["w"], r["i"]) for r in rows}) == 600
    assert len(EventStore(path).read_all()) == 600


# -------------------------------------------------------------------------- retrieval
def test_situation_text_and_embedder():
    txt = situation_text(TASK, TASK.language, "grasp")
    for piece in ("pick_bowl_to_plate", "black_bowl", "plate", "mug", "grasp", TASK.language):
        assert piece in txt
    e = Embedder()
    assert e.backend == "hash" and e.dim == 256
    v = e.embed(txt)
    assert v.shape == (256,) and v.dtype == np.float32 and abs(np.linalg.norm(v) - 1) < 1e-5
    assert np.array_equal(v, Embedder().embed(txt))   # deterministic across instances
    assert not np.any(e.embed(""))
    near = e.embed(situation_text(TASK, "put the black bowl on the plate please", "grasp"))
    far = e.embed("open the top drawer of the cabinet | objects: drawer, cabinet | phase: open")
    assert cosine(v, near) > cosine(v, far)


def test_retriever_eligibility_and_ranking(store):
    txt = situation_text(TASK, TASK.language)
    store.append(mk_lesson("ok_exact", text=txt))
    store.append(mk_lesson("ok_wild", family="*", obj="*", text="some unrelated words about drawers"))
    store.append(mk_lesson("bad_family", family="drawer", text=txt))
    store.append(mk_lesson("bad_object", obj="cup", text=txt))
    store.append(mk_lesson("bad_surface", surface="S2", op="set_instruction", params={"instruction": "x"}, text=txt))
    store.append(mk_lesson("bad_status", status="retired", text=txt))
    store.append(mk_lesson("ok_rationale", rationale="lower the grasp on the bowl"))
    store.append(LessonStatusChange("ok_exact", "candidate", "validated", "promoted", Evidence()))

    r = Retriever(store, k=5)
    got = [l.lesson_id for l in r.retrieve(TASK, ["S3"])]
    assert set(got) == {"ok_exact", "ok_wild", "ok_rationale"}
    assert got[0] == "ok_exact" and r.last_scores["ok_exact"] == pytest.approx(1.0)
    assert [l.lesson_id for l in Retriever(store, k=1).retrieve(TASK, ["S3"])] == ["ok_exact"]
    assert "bad_surface" in {l.lesson_id for l in r.retrieve(TASK, ["S2", "S3"])}
    assert [l.lesson_id for l in r.retrieve(TASK, ["S3"], statuses=("validated",))] == ["ok_exact"]
    assert r.retrieve(TASK, []) == []


def test_retriever_randomize_ignores_eligibility(store):
    for i in range(10):
        store.append(mk_lesson(f"far_{i}", family="drawer", obj="cup"))
    assert Retriever(store, k=5).retrieve(TASK, ["S3"]) == []
    a = [l.lesson_id for l in Retriever(store, k=5, randomize=True, seed=1).retrieve(TASK, ["S3"])]
    b = [l.lesson_id for l in Retriever(store, k=5, randomize=True, seed=1).retrieve(TASK, ["S3"])]
    assert len(a) == 5 and len(set(a)) == 5 and a == b
    picks = {tuple(l.lesson_id for l in Retriever(store, k=5, randomize=True, seed=s).retrieve(TASK, ["S3"]))
             for s in range(20)}
    assert len(picks) > 1
    assert Retriever(store, k=5, randomize=True).retrieve(TASK, ["S3"], statuses=("validated",)) == []


# -------------------------------------------------------------------------- lifecycle
def test_validate_lesson_intake():
    ok, why = lifecycle.validate_lesson(mk_lesson("a"))
    assert ok, why
    assert lifecycle.validate_lesson(mk_lesson("a").to_dict())[0]
    # prose only
    ok, why = lifecycle.validate_lesson({"surface": "S3", "rationale_text": "grasp the bowl lower"})
    assert not ok and "prose" in why
    assert not lifecycle.validate_lesson({"lesson_id": "x", "surface": "S3", "trigger": {}, "edit": "lower grasp"})[0]
    # unknown op / op from another surface
    assert not lifecycle.validate_lesson(mk_lesson("a", op="lower_grasp"))[0]
    assert not lifecycle.validate_lesson(mk_lesson("a", surface="S1", op="set_grasp_offset"))[0]
    # unknown predicate
    assert not lifecycle.validate_lesson(mk_lesson("a", predicate="bowl_is_wobbly"))[0]
    assert not lifecycle.validate_lesson(mk_lesson("a", op="set_abort_retry", params={"predicate": "nope", "max_retries": 1}))[0]
    assert not lifecycle.validate_lesson(mk_lesson("a", surface="S1", op="add_precondition",
                                                   params={"subtask_id": "s1", "predicate": "nope"}))[0]
    # unknown surface / params / empty params / bad S3 values
    assert not lifecycle.validate_lesson(mk_lesson("a", surface="S5"))[0]
    assert not lifecycle.validate_lesson(mk_lesson("a", params={"dz": -0.01, "height": 1}))[0]
    assert not lifecycle.validate_lesson(mk_lesson("a", params={}))[0]
    assert not lifecycle.validate_lesson(mk_lesson("a", params={"dz": -0.5}))[0]          # >10 cm
    assert not lifecycle.validate_lesson(mk_lesson("a", op="set_approach_vector", params={"vector": [0, 0, 0]}))[0]
    assert not lifecycle.validate_lesson(mk_lesson("a", op="set_approach_vector", params={"half_angle_deg": 30}))[0]
    # good ones on every surface
    assert lifecycle.validate_lesson(mk_lesson("a", op="set_abort_retry", params={"predicate": "grasp_slipped", "max_retries": 1}))[0]
    assert lifecycle.validate_lesson(mk_lesson("a", op="set_velocity_cap", params={"max_pos_delta": 0.5}))[0]
    assert lifecycle.validate_lesson(mk_lesson("a", surface="S2", op="set_instruction", params={"instruction": "please x"}))[0]
    assert not lifecycle.validate_lesson(mk_lesson("a", surface="S2", op="set_instruction", params={"instruction": ""}))[0]
    assert lifecycle.validate_lesson(mk_lesson("a", surface="S1", op="rebind_object", params={"subtask_id": "s1", "object": "mug"}))[0]
    assert lifecycle.validate_lesson(mk_lesson("a", surface="S1", op="replan", params={"subtasks": [
        {"subtask_id": "s1", "skill": "grasp", "target_object": "black_bowl"}]}))[0]
    assert not lifecycle.validate_lesson(mk_lesson("a", surface="S1", op="replan", params={"subtasks": [{"skill": "grasp"}]}))[0]


def test_assign_arm_deterministic_and_balanced():
    a = lifecycle.assign_arm("les_1", 7, "pick_bowl_to_plate")
    assert a in ("treated", "control")
    assert all(lifecycle.assign_arm("les_1", 7, "pick_bowl_to_plate") == a for _ in range(5))
    arms = [lifecycle.assign_arm("les_1", s, "pick_bowl_to_plate") for s in range(400)]
    assert 140 < arms.count("treated") < 260
    assert [lifecycle.assign_arm("les_2", s, "pick_bowl_to_plate") for s in range(400)] != arms
    assert [lifecycle.assign_arm("les_1", s, "other_task") for s in range(400)] != arms
    assert all(lifecycle.assign_arm("les_1", s, "t", treat_frac=1.0) == "treated" for s in range(20))
    assert all(lifecycle.assign_arm("les_1", s, "t", treat_frac=0.0) == "control" for s in range(20))


def test_update_evidence_counts(store):
    store.append(mk_lesson("les_1"))
    add_ab_episodes(store, "les_1", treated_ok=3, treated_n=4, control_ok=1, control_n=4)
    store.append(mk_episode("other", 99, ["les_2"], ["les_2"], True))       # different lesson: ignored
    noout = mk_episode("noout", 98, ["les_1"], ["les_1"], True)
    noout.outcome = None                                                    # no outcome: ignored
    store.append(noout)
    ev = lifecycle.update_evidence(store, "les_1")
    assert (ev.treated_n, ev.treated_successes, ev.control_n, ev.control_successes, ev.trials) == (4, 3, 4, 1, 8)
    assert ev.lift_pp == pytest.approx(50.0) and 0 < ev.p_value < 0.5
    empty = lifecycle.update_evidence(store, "les_none")
    assert empty.trials == 0 and empty.p_value is None and empty.lift_pp is None


def test_two_proportion_test():
    p, lift = lifecycle.two_proportion_test(18, 20, 6, 20)
    assert lift == pytest.approx(60.0) and p < 1e-3
    p, lift = lifecycle.two_proportion_test(6, 20, 18, 20)
    assert lift == pytest.approx(-60.0) and p > 0.999
    assert lifecycle.two_proportion_test(20, 20, 20, 20) == (0.5, 0.0)
    assert lifecycle.two_proportion_test(3, 5, 0, 0) == (None, None)


def test_evaluate_promotes_with_real_lift(store):
    store.append(mk_lesson("les_1"))
    add_ab_episodes(store, "les_1", treated_ok=13, treated_n=14, control_ok=4, control_n=14)   # n=28 < 30
    assert lifecycle.evaluate(store, "les_1") is None
    assert store.lessons()["les_1"].status == "candidate"
    add_ab_episodes(store, "les_1", treated_ok=1, treated_n=1, control_ok=0, control_n=1, start=28)  # n=30
    ch = lifecycle.evaluate(store, "les_1")
    assert ch is not None and (ch.from_status, ch.to_status) == ("candidate", "validated")
    assert ch.evidence.trials == 30 and ch.evidence.lift_pp > 2 and ch.evidence.p_value < 0.05
    assert store.lessons()["les_1"].status == "validated"
    assert store.lessons()["les_1"].evidence.treated_n == 15
    assert lifecycle.evaluate(store, "les_1") is None            # validated with positive lift: unchanged
    assert lifecycle.evaluate(store, "les_missing") is None


def test_evaluate_retires_without_lift(store):
    store.append(mk_lesson("no_lift"))
    add_ab_episodes(store, "no_lift", treated_ok=8, treated_n=16, control_ok=8, control_n=16)
    ch = lifecycle.evaluate(store, "no_lift")
    assert ch is not None and ch.to_status == "retired" and ch.evidence.lift_pp == 0.0
    # small, non-significant lift also retires
    store.append(mk_lesson("tiny_lift"))
    add_ab_episodes(store, "tiny_lift", treated_ok=9, treated_n=16, control_ok=8, control_n=16, start=100)
    ch = lifecycle.evaluate(store, "tiny_lift")
    assert ch.to_status == "retired" and ch.evidence.lift_pp > 2 and ch.evidence.p_value >= 0.05
    assert lifecycle.evaluate(store, "tiny_lift") is None        # retired is terminal
    assert lifecycle.lesson_precision(store) == {"candidates": 0, "validated": 0, "retired": 2, "precision": 0.0}


def test_evaluate_regression_retires_validated(store):
    store.append(mk_lesson("les_1"))
    add_ab_episodes(store, "les_1", treated_ok=13, treated_n=15, control_ok=4, control_n=15)
    assert lifecycle.evaluate(store, "les_1").to_status == "validated"
    # post-validation every retrieval is applied; treated success collapses below the A/B control rate
    add_ab_episodes(store, "les_1", treated_ok=2, treated_n=50, control_ok=0, control_n=0, start=30)
    ch = lifecycle.evaluate(store, "les_1")
    assert ch is not None and (ch.from_status, ch.to_status) == ("validated", "retired")
    assert ch.evidence.lift_pp < 0 and ch.evidence.control_n == 15 and ch.evidence.treated_n == 65
    assert store.lessons()["les_1"].status == "retired"
    assert [r["to_status"] for r in store.iter_type("lesson_status")] == ["validated", "retired"]


def test_gate_all_and_precision(store):
    store.append(mk_lesson("good"))
    store.append(mk_lesson("bad"))
    store.append(mk_lesson("young"))
    store.append(mk_lesson("dead", status="retired"))
    n = add_ab_episodes(store, "good", 14, 16, 5, 16)
    n = add_ab_episodes(store, "bad", 5, 16, 6, 16, start=n)
    add_ab_episodes(store, "young", 5, 5, 0, 5, start=n)
    changes = lifecycle.gate_all(store)
    assert {(c.lesson_id, c.to_status) for c in changes} == {("good", "validated"), ("bad", "retired")}
    st = {k: v.status for k, v in store.lessons().items()}
    assert st == {"good": "validated", "bad": "retired", "young": "candidate", "dead": "retired"}
    assert lifecycle.lesson_precision(store) == {"candidates": 1, "validated": 1, "retired": 2, "precision": 1 / 3}
    assert lifecycle.gate_all(store) == []                        # idempotent
    assert lifecycle.lesson_precision(EventStore(store.path + ".empty"))["precision"] is None
