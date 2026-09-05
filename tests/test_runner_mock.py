"""MockEnv + MockPolicy + runner (worker / pool / conditions) with the real shim. FM_LLM=mock, no network."""
from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest

os.environ.setdefault("FM_LLM", "mock")
os.environ.setdefault("FM_EMBEDDER", "hash")

from fleet_memory.envs.mock_env import GRASP_R_M, TASKS, MockEnv, MockSuite  # noqa: E402
from fleet_memory.memory.store import EventStore  # noqa: E402
from fleet_memory.policies.mock import MockPolicy  # noqa: E402
from fleet_memory.runner import conditions  # noqa: E402
from fleet_memory.runner.pool import run_many, run_protocol, summarize  # noqa: E402
from fleet_memory.runner.worker import RunConfig, make_env, make_policy, run_episode  # noqa: E402

TASK = "pick_bowl_to_plate"
SEEDS = list(range(40))


def cfgs(log, arm, seeds=SEEDS, **kw):
    return [RunConfig("mock", TASK, s, arm, log_path=log, **kw) for s in seeds]


def batch(log, arm, seeds=SEEDS, **kw):
    """Inline, sharing one env/policy/store (fast): one Episode per seed."""
    store = EventStore(log)
    cs = cfgs(log, arm, seeds, **kw)
    env, pol = make_env(cs[0]), make_policy(cs[0])
    return [run_episode(c, env, pol, store) for c in cs]


def rate(eps):
    return float(np.mean([e.outcome.env_success for e in eps]))


def steps(eps):
    return float(np.mean([e.outcome.steps for e in eps]))


@pytest.fixture
def log(tmp_path):
    return str(tmp_path / "events.jsonl")


# ------------------------------------------------------------------ env
def test_mock_env_protocol_and_determinism():
    assert MockSuite.tasks() == list(TASKS) and len(MockSuite.tasks()) == 4
    env = MockEnv(TASK)
    info = env.task_info()
    assert info.suite == "mock" and info.objects == ["black_bowl", "plate", "mug"] and info.max_steps == 200
    o1, o2 = env.reset(3), MockEnv(TASK).reset(3)
    for k in o1.object_poses:
        np.testing.assert_allclose(o1.object_poses[k][0], o2.object_poses[k][0])
    assert o1.ee_pos.shape == (3,) and o1.state.dtype == np.float32 and set(env.success_subconditions()) == {
        "grasped", "above_dest", "placed"}
    assert not env.success_flag()
    assert env.envelope().workspace_min[2] < 0     # MOCK_ENVELOPE


def test_mock_env_perturbations():
    base = MockEnv(TASK).reset(5).object_pos("black_bowl")
    shifted = MockEnv(TASK).reset(5, {"shift_xy": [0.06, 0.0]}).object_pos("black_bowl")
    np.testing.assert_allclose(shifted - base, [0.06, 0.0, 0.0], atol=1e-6)
    jit = MockEnv(TASK).reset(5, {"jitter_m": 0.01}).object_pos("plate")
    assert np.linalg.norm(jit - MockEnv(TASK).reset(5).object_pos("plate")) > 0
    env = MockEnv(TASK)
    obs = env.reset(5, {"distractor": True})
    assert "mug_2" in obs.object_poses and len(env.task_info().objects) == 4
    assert np.linalg.norm((obs.object_pos("mug_2") - obs.object_pos("black_bowl"))[:2]) < 0.08


def test_mock_env_grasp_window_and_success():
    env = MockEnv(TASK)
    env.reset(0)
    bowl = env.objects["black_bowl"].copy()
    env.ee = bowl + np.array([0.0, 0.0, 0.02])          # too high: no grasp
    env.step(np.array([0, 0, 0, 0, 0, 0, 1.0], np.float32))
    assert env.held is None
    env.step(np.array([0, 0, 0, 0, 0, 0, -1.0], np.float32))
    env.ee = bowl + np.array([0.0, 0.0, 0.005])
    env.step(np.array([0, 0, 0, 0, 0, 0, 1.0], np.float32))
    assert env.held == "black_bowl"
    plate = env.objects["plate"]
    env.ee = plate + np.array([0.0, 0.0, 0.10])          # carried: the object follows the EE
    env.step(np.array([0, 0, 0, 0, 0, 0, 1.0], np.float32))
    assert np.linalg.norm(env.objects["black_bowl"] - env.ee) < GRASP_R_M + 0.01
    _, done, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0], np.float32))   # release: drops onto the plate
    assert env.success_flag() and done and env.success_subconditions()["placed"]


# ------------------------------------------------------------------ policy + arms (priority 1)
def test_arm_a_success_and_slack(log):
    eps = batch(log, "A")
    assert 0.40 <= rate(eps) <= 0.55, rate(eps)
    assert steps(eps) >= 150
    assert all(e.metrics is not None and e.metrics.force_proxy > 0 for e in eps)
    assert all(e.s3_params["grasp_offset_z"] == 0.0 and e.condition == "A" and e.surfaces_enabled == [] for e in eps)


def test_grasp_offset_raises_success(log):
    eps = batch(log, "B", s3_params={"grasp_offset_z": -0.015})
    assert rate(eps) > 0.80
    assert all(e.constraints_active["grasp_offset"][2] == pytest.approx(-0.015) for e in eps)


def test_time_scale_cuts_steps_without_losing_success(log):
    a = batch(log, "A")
    b = batch(log, "B", s3_params={"time_scale": 1.3, "velocity_cap": 1.0})
    assert rate(b) >= rate(a) - 1e-9
    both = [i for i in range(len(SEEDS)) if a[i].outcome.env_success and b[i].outcome.env_success]
    assert len(both) >= 10
    sa, sb = np.mean([a[i].outcome.steps for i in both]), np.mean([b[i].outcome.steps for i in both])
    assert sb < 0.85 * sa, (sa, sb)
    assert steps(b) < steps(a)


def test_shift_perturbation_breaks_policy_and_s3_recovers(log):
    pert = {"shift_xy": [0.06, 0.0]}
    a = batch(log, "A", perturbation=pert)
    assert rate(a) < 0.20
    assert all(e.perturbation == pert for e in a)
    fixed = batch(log, "B", perturbation=pert, s3_params={"approach_offset_xyz": [0.03, 0, 0], "grasp_offset_z": -0.015})
    assert rate(fixed) > 0.5


def test_policy_instruction_and_tasks(log):
    env, pol = MockEnv(TASK), MockPolicy()
    obs = env.reset(1)
    pol.reset("put the black bowl on the plate carefully")
    a = pol.act(obs)
    assert a.shape == (5, 7) and np.abs(a[:, :3]).max() <= 0.5 * 0.7 + 1e-6
    for t in MockSuite.tasks():
        eps = [run_episode(c) for c in [RunConfig("mock", t, s, "A", log_path=log) for s in range(6)]]
        assert eps and all(e.task_id == t for e in eps)


# ------------------------------------------------------------------ worker invariants
def test_run_episode_record_shape(log):
    ep = run_episode(RunConfig("mock", TASK, 7, "A", log_path=log, environment_tag="lab"))
    assert ep.environment_id == "mock_pick_bowl_to_plate_lab" and ep.retrieval_frozen_at <= ep.ts
    assert ep.retrieved_lesson_ids == [] and ep.applied_lesson_ids == [] and ep.plan.source == "identity"
    assert set(ep.s3_params) == {"approach_offset_xyz", "pregrasp_height", "grasp_offset_z", "time_scale",
                                 "velocity_cap", "gripper_cmd", "approach_cone_deg"}
    assert ep.metrics.steps == ep.outcome.steps and ep.metrics.cost is None    # no cost_reference yet
    store = EventStore(log)
    types = [d["type"] for d in store.read_all()]
    assert types.count("episode") == 1 and types.count("snapshot") == 1


def test_envelope_clamps_even_in_arm_a(log):
    class Escape:
        name = "escape"

        def reset(self, instruction):
            pass

        def act(self, obs):
            a = np.zeros((5, 7), np.float32)
            a[:, 2] = 1.0            # straight up, forever
            a[:, 6] = -1.0
            return a

    ep = run_episode(RunConfig("mock", TASK, 0, "A", log_path=log, record_trace=True), policy=Escape())
    tr = [d for d in EventStore(log).read_all() if d["type"] == "trace"][-1]
    assert ep.outcome.steps == 200 and max(p[2] for p in tr["ee_positions"]) <= 1.0 + 1e-6


def test_arm_d_generates_lessons_and_held_out_does_not(log):
    eps = batch(log, "D", seeds=range(8))
    store = EventStore(log)
    assert all(e.plan.source == "planner" and len(e.plan.subtasks) >= 3 for e in eps)
    lessons = store.lessons()
    assert lessons and any(e.lessons_generated for e in eps)
    assert all(l.trigger.environment_id == "mock_pick_bowl_to_plate" for l in lessons.values())
    later = batch(log, "D", seeds=range(8, 12))
    assert any(e.retrieved_lesson_ids for e in later)          # frozen retrieval picked earlier lessons up
    held = batch(log, "D", seeds=range(2), held_out=True)
    assert all(e.held_out and not e.lessons_generated for e in held)


def test_incumbent_and_scorecard_path(log):
    hm = pytest.importorskip("fleet_memory.memory.house_model")
    store = EventStore(log)
    c = RunConfig("mock", TASK, 0, "B", log_path=log)
    hm.init_incumbent(store, c.skill_instance_id, c.environment_id, TASK, "black_bowl", {"grasp_offset_z": -0.015})
    eps = batch(log, "B", seeds=range(6))
    assert all(e.skill_instance_versions == {c.skill_instance_id: 1} for e in eps)
    assert all(e.s3_params["grasp_offset_z"] == pytest.approx(-0.015) for e in eps) and rate(eps) > 0.8
    assert any(d["type"] == "scorecard" for d in store.read_all())


# ------------------------------------------------------------------ conditions
def test_conditions():
    A, B, C, D = (conditions.ARMS[k] for k in "ABCD")
    assert not A.shim and not A.planner and A.surfaces == []
    assert B.shim and B.use_incumbent_s3 and not B.planner
    assert C.planner and C.surfaces == ["S1"] and C.outer_loop and not C.use_incumbent_s3
    assert D.surfaces == ["S1", "S3"] and D.use_incumbent_s3 and conditions.ARMS["P"].surfaces == D.surfaces
    assert conditions.ARMS["E"].deferred and conditions.ARMS["F"].deferred
    assert conditions.ABLATIONS["D_S1"].surfaces == ["S1"] and conditions.ABLATIONS["D_S3"].surfaces == ["S3"]
    sets = {n: set(conditions.seed_set(n, 100)) for n in ("train", "heldout", "probe", "opt", "gate")}
    assert conditions.seed_set("train", 3) == [0, 1, 2] and conditions.seed_set("gate", 2) == [4000, 4001]
    assert not any(s1 & s2 for a, s1 in sets.items() for b, s2 in sets.items() if a < b)
    assert conditions.is_held_out("libero_goal") and not conditions.is_held_out("mock")
    with pytest.raises(KeyError):
        conditions.get_arm("Z")


# ------------------------------------------------------------------ pool
def test_run_many_inline_and_spawn(log):
    cs = cfgs(log, "A", seeds=range(4))
    inline = run_many(cs, workers=1)
    assert [e.seed for e in inline] == [0, 1, 2, 3]
    spawned = run_many(cs, workers=2)
    assert [e.seed for e in spawned] == [0, 1, 2, 3]
    assert [e.outcome.env_success for e in spawned] == [e.outcome.env_success for e in inline]
    assert len(EventStore(log).episodes()) == 8
    line = summarize(inline, "t")
    assert "arm=A" in line and "n=4" in line and "mean_steps" in line


def test_protocol_p_stages(log):
    pytest.importorskip("fleet_memory.memory.house_model")
    store = EventStore(log)
    base = RunConfig("mock", TASK, 0, "P", log_path=log)
    out = run_protocol(store, base, {"shift_xy": [0.06, 0.0]}, n_stage=3, workers=1, gate_every=3)
    assert set(out) >= {"baseline", "perturbed", "recovery"} and all(len(v) == 3 for v in out.values())
    evs = [d for d in store.read_all() if d["type"] == "protocol_stage"]
    assert [e["stage"] for e in evs] == ["baseline", "perturbed", "recovery"]
    assert all(len(e["episode_ids"]) == 3 and e["si_id"] == base.skill_instance_id for e in evs)
    assert any(d["type"] == "cost_reference" for d in store.read_all())
    assert all(e.metrics.cost is not None for e in out["baseline"])
    assert all(e.perturbation.get("shift_xy") for e in out["perturbed"] + out["recovery"])
    assert dataclasses.asdict(base)["arm"] == "P"
