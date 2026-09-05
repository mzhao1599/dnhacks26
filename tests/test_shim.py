"""ExecutionShim + detectors against a tiny kinematic stub env/policy (no dependency on envs/mock_env)."""
from __future__ import annotations

import numpy as np

from fleet_memory.envs.base import Obs, TaskInfo
from fleet_memory.execution import detectors
from fleet_memory.execution.constraints import ConstraintSet, apply_edit
from fleet_memory.execution.detectors import Tracker, evaluate_predicate
from fleet_memory.execution.probes import S3_PROBE_EDITS, paraphrases
from fleet_memory.execution.shim import ExecutionShim
from fleet_memory.memory.schema import EDIT_OPS, PREDICATES, Edit

STEP_M = 0.02   # metres per unit action per step
TASK = TaskInfo(suite="stub", task_id="pick_bowl", language="pick up the bowl", task_family="pick_place",
                objects=["bowl", "plate"], max_steps=300)


class StubEnv:
    """EE point + gripper bool. Grasp succeeds iff the gripper closes within 0.02 m of the bowl and
    ee_z - bowl_z in [-0.005, 0.012]. A held object follows the EE."""

    def __init__(self):
        self.t = 0
        self.ee = np.array([0.0, 0.0, 0.20])
        self.closed = False
        self.held = False
        self.obj = {"bowl": np.array([0.10, 0.05, 0.05]), "plate": np.array([-0.10, 0.0, 0.02])}
        self.close_rel_z: list[float] = []      # ee_z - bowl_z at every close transition
        self.ee_log: list[np.ndarray] = []

    def obs(self) -> Obs:
        q = np.array([0.0, 0.0]) if self.closed else np.array([0.04, -0.04])
        return Obs(t=self.t, images={}, state=np.zeros(8, np.float32), ee_pos=self.ee.copy(),
                   ee_quat=np.array([0, 0, 0, 1.0]), gripper_qpos=q,
                   object_poses={k: (v.copy(), np.array([0, 0, 0, 1.0])) for k, v in self.obj.items()})

    def step(self, a: np.ndarray) -> Obs:
        self.ee = self.ee + STEP_M * np.asarray(a[:3], float)
        if self.held:
            self.obj["bowl"] = self.ee.copy()
        close = float(a[6]) > 0
        if close and not self.closed:
            rel = self.ee[2] - self.obj["bowl"][2]
            self.close_rel_z.append(float(rel))
            self.held = np.linalg.norm(self.ee - self.obj["bowl"]) < 0.02 and -0.005 <= rel <= 0.012
        if not close:
            self.held = False
        self.closed = close
        self.t += 1
        self.ee_log.append(self.ee.copy())
        return self.obs()

    def subconds(self) -> dict[str, bool]:
        return {"grasped": bool(self.held), "lifted": bool(self.held and self.obj["bowl"][2] > 0.08)}


class ScriptedPolicy:
    """reach above bowl -> descend to bowl_z + bias_z -> close -> lift. Chunks of k rows."""
    name = "scripted"

    def __init__(self, bias_z=0.015, k=3):
        self.bias_z, self.k = bias_z, k
        self.resets = 0
        self.acts = 0
        self.phase = "reach"

    def reset(self, instruction):
        self.resets += 1
        self.phase = "reach"

    def act(self, obs):
        self.acts += 1
        bowl = obs.object_pos("bowl")
        a = np.zeros((self.k, 7), np.float32)
        a[:, 6] = -1.0
        if self.phase == "reach":
            goal = bowl + np.array([0, 0, 0.05])
            if np.linalg.norm(goal - obs.ee_pos) < 0.003:
                self.phase = "descend"
            else:
                a[:, :3] = np.clip((goal - obs.ee_pos) / (STEP_M * self.k), -1, 1)
                return a
        if self.phase == "descend":
            goal = bowl + np.array([0, 0, self.bias_z])
            if np.linalg.norm(goal - obs.ee_pos) < 0.002:
                self.phase = "close"
            else:
                a[:, :3] = np.clip((goal - obs.ee_pos) / (STEP_M * self.k), -1, 1)
                return a
        if self.phase == "close":
            a[:, 6] = 1.0
            self.phase = "lift"
            return a
        a[:, 2] = 1.0
        a[:, 6] = 1.0
        return a


class FixedPolicy:
    name = "fixed"

    def __init__(self, chunk):
        self.chunk = np.asarray(chunk, np.float32)
        self.resets = 0

    def reset(self, instruction):
        self.resets += 1

    def act(self, obs):
        return self.chunk.copy()


def run(env, shim, tracker, steps=120):
    obs = env.obs()
    tracker.update(obs, env.subconds())
    shim.reset(TASK.language, "bowl")
    t = 0
    while t < steps:
        chunk = shim.act(obs)
        assert chunk.dtype == np.float32 and chunk.ndim == 2 and chunk.shape[1] == 7
        for row in chunk:
            obs = env.step(row)
            tracker.update(obs, env.subconds())
            t += 1
    return obs


def make(policy, cs=None, enabled=True):
    env, tracker = StubEnv(), Tracker(TASK, stall_k=40)
    shim = ExecutionShim(policy, cs or ConstraintSet(), TASK, tracker, enabled=enabled)
    return env, tracker, shim


# ------------------------------------------------------------------ tests
def test_disabled_is_pass_through():
    cs = apply_edit(apply_edit(ConstraintSet(), S3_PROBE_EDITS[0]), Edit("set_gripper_aperture", {"value": 0.2}))
    pol = FixedPolicy([[1, -1, 1, 1, -1, 1, 1]])
    env, tracker, shim = make(pol, cs, enabled=False)
    shim.reset("x")
    out = shim.act(env.obs())
    assert np.array_equal(out, pol.chunk) and shim.events == []
    assert tracker.gripper_cmd == 1.0   # state is still recorded for the detectors


def test_velocity_clamp_and_gripper_remap():
    cs = apply_edit(apply_edit(ConstraintSet(), S3_PROBE_EDITS[0]), Edit("set_gripper_aperture", {"value": 0.3}))
    pol = FixedPolicy([[1, -1, 0.2, 1, -1, 0.1, 1], [0.4, 0, 0, 0, 0, 0, -1]])
    env, _, shim = make(pol, cs)
    shim.reset("x")
    out = shim.act(env.obs())
    np.testing.assert_allclose(out[0], [0.5, -0.5, 0.2, 0.5, -0.5, 0.1, 0.3], atol=1e-6)
    np.testing.assert_allclose(out[1], [0.4, 0, 0, 0, 0, 0, -1], atol=1e-6)
    kinds = [e["kind"] for e in shim.events]
    assert kinds == ["velocity_clamped", "gripper_remapped"]
    assert shim.events[0]["rows"] == 1 and shim.events[1]["rows"] == 1


def test_approach_cone_only_near_target_and_open():
    cs = apply_edit(ConstraintSet(), S3_PROBE_EDITS[4])          # v = -z, 30 deg
    pol = FixedPolicy([[0.8, 0, 0, 0, 0, 0, -1]])
    env, tracker, shim = make(pol, cs)
    shim.reset("x")
    far = shim.act(env.obs())                                    # EE 0.2 m above the bowl: untouched
    np.testing.assert_allclose(far[0, :3], [0.8, 0, 0])
    env.ee = env.obj["bowl"] + np.array([0, 0, 0.03])
    near = shim.act(env.obs())
    np.testing.assert_allclose(near[0, :3], [0, 0, -0.8], atol=1e-6)
    assert shim.events[-1]["kind"] == "cone_projected"
    tracker.gripper_cmd = 1.0                                    # closed: cone must not fight the lift
    n = len(shim.events)
    lift = shim.act(env.obs())
    np.testing.assert_allclose(lift[0, :3], [0.8, 0, 0])
    assert len(shim.events) == n


def test_waypoint_controller_reaches_then_hands_back():
    cs = apply_edit(ConstraintSet(), S3_PROBE_EDITS[3])          # dz=0.06, tol 0.015
    pol = ScriptedPolicy()
    env, tracker, shim = make(pol, cs)
    obs = env.obs()
    tracker.update(obs, env.subconds())
    shim.reset(TASK.language, "bowl")
    wp = env.obj["bowl"] + np.array([0, 0, 0.06])
    n_ctrl = 0
    while shim.phase != "policy" or n_ctrl == 0:
        chunk = shim.act(obs)
        if shim.phase == "waypoint":
            assert chunk.shape == (1, 7) and chunk[0, 6] == -1 and pol.acts == 0
            n_ctrl += 1
        for row in chunk:
            obs = env.step(row)
            tracker.update(obs, env.subconds())
        assert n_ctrl < 100
    assert n_ctrl > 0 and pol.acts >= 1
    assert min(np.linalg.norm(e - wp) for e in env.ee_log) <= 0.015
    kinds = [e["kind"] for e in shim.events]
    assert kinds[0] == "waypoint_active" and "waypoint_reached" in kinds
    # once reached this subtask, the controller does not re-engage
    for _ in range(5):
        shim.act(obs)
        assert shim.phase == "policy"
    # a new subtask that starts with the gripper closed never engages the waypoint
    shim.reset("lift it", "bowl")
    tracker.gripper_cmd = 1.0
    shim.act(obs)
    assert shim.phase == "policy"


def test_grasp_offset_shifts_closing_point():
    def closing_rel(cs):
        env, tracker, shim = make(ScriptedPolicy(bias_z=0.015), cs)
        run(env, shim, tracker, steps=60)
        assert len(env.close_rel_z) == 1
        return env.close_rel_z[0], env, shim

    base, env0, _ = closing_rel(ConstraintSet())
    assert abs(base - 0.015) < 0.003 and not env0.held               # biased grasp misses
    for dz in (-0.015, 0.015):
        rel, env, shim = closing_rel(apply_edit(ConstraintSet(), Edit("set_grasp_offset", {"dz": dz})))
        assert abs((rel - base) - dz) < 0.005, (rel, base, dz)
        assert [e["kind"] for e in shim.events].count("offset_applied") == 1
        assert env.held == (dz < 0)                                  # -0.015 corrects the bias, +0.015 does not


def test_abort_retry_fires_once_and_resets_policy():
    cs = apply_edit(ConstraintSet(), Edit("set_abort_retry", {"predicate": "grasp_slipped", "max_retries": 1, "lift_m": 0.08}))
    pol = ScriptedPolicy(bias_z=0.015)                               # always misses -> slips
    env, tracker, shim = make(pol, cs)
    run(env, shim, tracker, steps=150)
    retries = [e for e in shim.events if e["kind"] == "abort_retry"]
    assert len(retries) == 1 and shim.retries_used == 1
    assert retries[0]["predicate"] == "grasp_slipped"
    assert pol.resets == 2                                           # shim.reset + one retry
    done = [e for e in shim.events if e["kind"] == "retry_done"][0]
    assert abs(done["lifted_m"] - 0.08) < 0.01
    assert len(env.close_rel_z) == 2                                 # grasp attempted again after the retry
    assert tracker.stall_steps == env.t - done["t"]                  # stall counter was reset at retry_done


def test_intervention_mid_episode_via_set_constraints():
    env, tracker, shim = make(ScriptedPolicy(bias_z=0.015))
    obs = env.obs()
    tracker.update(obs, env.subconds())
    shim.reset(TASK.language, "bowl")
    for t in range(60):
        if t == 3:
            shim.set_constraints(apply_edit(ConstraintSet(), Edit("set_grasp_offset", {"dz": -0.015})))
        for row in shim.act(obs):
            obs = env.step(row)
            tracker.update(obs, env.subconds())
    assert env.held and abs(env.close_rel_z[0]) < 0.005


def test_predicates_and_tracker():
    env, tracker = StubEnv(), Tracker(TASK, stall_k=5)
    obs = env.obs()
    tracker.update(obs, env.subconds())
    assert tracker.initial_z("bowl") == 0.05
    for p in PREDICATES:
        assert isinstance(evaluate_predicate(p, obs, "bowl", tracker), bool)
    assert evaluate_predicate("always", obs, None, tracker)
    assert evaluate_predicate("gripper_open", obs, None, None) and not evaluate_predicate("gripper_closed", obs, None, None)
    assert not evaluate_predicate("ee_near_target", obs, "bowl", tracker)
    assert evaluate_predicate("object_height_below_rim", obs, "bowl", tracker)
    assert not evaluate_predicate("ee_near_target", obs, "missing", tracker)
    env.ee = env.obj["bowl"] + np.array([0.01, 0.0, 0.04])
    obs = env.obs()
    assert evaluate_predicate("ee_near_target", obs, "bowl", tracker)
    assert evaluate_predicate("ee_above_target", obs, "bowl", tracker)
    env.obj["bowl"][2] += 0.04
    assert evaluate_predicate("object_lifted", env.obs(), "bowl", tracker)
    for _ in range(5):
        tracker.update(env.obs(), env.subconds())
    assert tracker.no_progress() and evaluate_predicate("no_progress", env.obs(), None, tracker)
    tracker.update(env.obs(), {"grasped": True})
    assert not tracker.no_progress()
    # grasp_slipped: closed gripper, EE travels, object does not
    tracker.gripper_cmd = 1.0
    for i in range(detectors.SLIP_WINDOW + 1):
        env.ee = env.ee + np.array([0, 0, 0.005])
        tracker.update(env.obs(), env.subconds())
    assert evaluate_predicate("grasp_slipped", env.obs(), "bowl", tracker)
    tracker.gripper_cmd = -1.0
    assert not evaluate_predicate("grasp_slipped", env.obs(), "bowl", tracker)


def test_probes():
    assert len(S3_PROBE_EDITS) == 5
    for e in S3_PROBE_EDITS:
        assert e.op in EDIT_OPS["S3"] and set(e.params) <= set(EDIT_OPS["S3"][e.op])
        apply_edit(ConstraintSet(), e)
    v = paraphrases("Put the black bowl on the plate.")
    assert len(v) == 5 and v[0] == "Put the black bowl on the plate." and v[1].startswith("please ")
    assert v[-1] == "put the black bowl on the plate" and len(set(v)) == 5
    assert "carefully" in v[2] and "handle" in v[3]
    assert len(paraphrases("put the bowl on the plate")) == 4     # lowercase/no-punct variant collapses
