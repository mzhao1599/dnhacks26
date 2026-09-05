"""ScriptedPolicy against a tiny kinematic pick-and-place stub (mirrors the MockEnv contract:
0.02 m per unit action, grasp window, place predicate). Does not import envs/mock_env."""
from __future__ import annotations

import numpy as np

from fleet_memory.envs.base import Obs, TaskInfo
from fleet_memory.execution.constraints import ConstraintSet
from fleet_memory.execution.detectors import Tracker
from fleet_memory.execution.envelope import MOCK_ENVELOPE
from fleet_memory.execution.params import S3Params
from fleet_memory.execution.shim import ExecutionShim
from fleet_memory.policies.scripted import PHASES, ScriptedPolicy, guess_targets

STEP_M = 0.02
TASK = TaskInfo(suite="stub", task_id="pick_bowl_to_plate", language="put the black bowl on the plate",
                task_family="place_on_plate", objects=["black_bowl", "plate", "mug"], max_steps=300)


class StubPickPlace:
    def __init__(self, ee=(0.0, 0.0, 0.25)):
        self.t, self.closed, self.held = 0, False, False
        self.ee = np.array(ee, float)
        self.obj = {"black_bowl": np.array([0.12, 0.06, 0.05]), "plate": np.array([-0.15, -0.05, 0.02]),
                    "mug": np.array([0.25, -0.20, 0.06])}
        self.closes: list[float] = []
        self.actions: list[np.ndarray] = []

    def obs(self) -> Obs:
        q = np.array([0.0, 0.0]) if self.closed else np.array([0.04, -0.04])
        return Obs(t=self.t, images={}, state=np.zeros(8, np.float32), ee_pos=self.ee.copy(),
                   ee_quat=np.array([0, 0, 0, 1.0]), gripper_qpos=q,
                   object_poses={k: (v.copy(), np.array([0, 0, 0, 1.0])) for k, v in self.obj.items()})

    def step(self, a: np.ndarray) -> Obs:
        self.actions.append(np.asarray(a, np.float32).copy())
        self.ee = self.ee + STEP_M * np.asarray(a[:3], float)
        if self.held:
            self.obj["black_bowl"] = self.ee.copy()
        close = float(a[6]) > 0
        if close and not self.closed:
            b = self.obj["black_bowl"]
            rel = self.ee[2] - b[2]
            self.closes.append(float(rel))
            self.held = np.linalg.norm(self.ee - b) < 0.02 and -0.005 <= rel <= 0.012
        if not close:
            self.held = False
        self.closed = close
        self.t += 1
        return self.obs()

    def success(self) -> bool:
        b, p = self.obj["black_bowl"], self.obj["plate"]
        return bool(np.linalg.norm((b - p)[:2]) < 0.05 and b[2] < p[2] + 0.03 and not self.closed)

    def subconds(self) -> dict[str, bool]:
        return {"grasped": bool(self.held), "placed": self.success()}


def rollout(policy, env=None, shim=None, steps=300):
    env = env or StubPickPlace()
    actor = shim or policy
    obs = env.obs()
    if shim is not None:
        shim.tracker.update(obs, env.subconds())
        shim.reset(TASK.language, "black_bowl")
    else:
        policy.reset(TASK.language)
    phases = []
    while env.t < steps and not env.success():
        chunk = actor.act(obs)
        assert chunk.dtype == np.float32 and chunk.ndim == 2 and chunk.shape[1] == 7 and 1 <= len(chunk) <= policy.k
        assert np.all(np.abs(chunk) <= 1.0)
        phases.append(policy.phase)
        for row in chunk:
            obs = env.step(row)
            if shim is not None:
                shim.tracker.update(obs, env.subconds())
    return env, phases


# ------------------------------------------------------------------ tests
def test_pick_and_place_succeeds_on_stub():
    for scale in (0.05, 0.02):
        pol = ScriptedPolicy(pos_scale_m=scale)
        pol.set_targets("black_bowl", "plate")
        env, phases = rollout(pol)
        assert env.success(), (scale, env.t, pol.phase)
        assert len(env.closes) == 1 and -0.005 <= env.closes[0] <= 0.012
        seen = [p for i, p in enumerate(phases) if i == 0 or p != phases[i - 1]]
        assert seen == [p for p in PHASES if p in seen] and seen[0] == "reach" and "place" in seen
        a = np.stack(env.actions)
        closes = np.flatnonzero(a[:, 6] > 0)
        assert len(closes) >= pol.close_steps and np.all(np.diff(closes)[:pol.close_steps - 1] == 1)
        assert np.all(a[closes[:pol.close_steps], :3] == 0)                    # close is stationary
        assert env.t < 200


def test_default_scale_faster_on_matching_world():
    slow = ScriptedPolicy(pos_scale_m=0.05)
    slow.set_targets("black_bowl", "plate")
    fast = ScriptedPolicy(pos_scale_m=0.02)
    fast.set_targets("black_bowl", "plate")
    assert rollout(fast)[0].t < rollout(slow)[0].t


def test_guess_targets_from_instruction():
    objs = ["plate_1", "akita_black_bowl_1", "cookies_1"]
    assert guess_targets("put the black bowl on the plate", objs) == ("akita_black_bowl_1", "plate_1")
    assert guess_targets("place the cookies on the plate", objs) == ("cookies_1", "plate_1")
    assert guess_targets("", objs) == ("plate_1", "akita_black_bowl_1")     # no mention: scene order
    assert guess_targets("x", []) == (None, None)
    pol = ScriptedPolicy()
    env, _ = rollout(pol)                                                   # targets guessed from the language
    assert pol.target == "black_bowl" and pol.destination == "plate" and env.success()


def test_missing_target_is_a_noop_and_stuck_phases_time_out():
    pol = ScriptedPolicy()
    pol.set_targets("ghost", None)
    env = StubPickPlace()
    del env.obj["mug"]
    pol.reset("lift the ghost")                                              # guess falls back to the scene
    assert pol.act(env.obs()).shape[1] == 7
    bad = ScriptedPolicy(grasp_dz_m=0.03, max_regrasps=1, max_phase_steps=40)   # grasps above the window
    bad.set_targets("black_bowl", "plate")
    env, phases = rollout(bad, steps=400)
    assert not env.success() and len(env.closes) == 2 and bad.phase == "done"   # regrasped once, then gave up
    assert phases.index("done") * bad.k < 300                                # ... well before the horizon
    # contact rule: an EE that stops moving in descend/place moves on instead of pushing for max_phase_steps
    stuck = ScriptedPolicy()
    stuck.set_targets("black_bowl", "plate")
    stuck.reset(TASK.language)
    env = StubPickPlace(ee=(0.12, 0.06, 0.13))                               # already at the hover point
    obs = env.obs()
    stuck.act(obs)
    assert stuck.phase == "descend"
    n = 0
    while stuck.phase == "descend" and n < 4:
        stuck.act(obs)                                                       # EE never moves -> contact
        n += 1
    assert stuck.phase == "close" and n <= 3


def test_through_shim_identity_blend_and_time_scale():
    def run_with(vec: S3Params):
        pol = ScriptedPolicy()
        pol.set_targets("black_bowl", "plate")
        env = StubPickPlace()
        tracker = Tracker(TASK, stall_k=40)
        shim = ExecutionShim(pol, vec.to_constraint_set(), TASK, tracker, enabled=True)
        shim.apply_envelope(MOCK_ENVELOPE)
        env, _ = rollout(pol, env, shim)
        return env, shim

    env0, shim0 = run_with(S3Params(blend_alpha=0.5))                       # blend on: waypoint 0.05 vs hover 0.08: stall latch
    assert env0.success()
    kinds = [e["kind"] for e in shim0.events]
    assert "blended" in kinds and "waypoint_reached" in kinds
    env1, shim1 = run_with(S3Params(blend_alpha=0.5, time_scale=1.5))
    assert env1.success() and env1.t < env0.t
    assert any(e["kind"] == "time_scaled" for e in shim1.events)
    env2, shim2 = run_with(S3Params(blend_alpha=0.5, pregrasp_height=0.08, velocity_cap=0.5))
    assert env2.success() and env2.t > env0.t
    assert any(e["kind"] == "velocity_clamped" for e in shim2.events)


def test_disabled_shim_with_envelope_only():
    pol = ScriptedPolicy()
    pol.set_targets("black_bowl", "plate")
    env = StubPickPlace()
    shim = ExecutionShim(pol, ConstraintSet(), TASK, Tracker(TASK), enabled=False)
    shim.apply_envelope(MOCK_ENVELOPE)
    env, _ = rollout(pol, env, shim)
    assert env.success() and shim.events == []
