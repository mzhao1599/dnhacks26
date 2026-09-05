"""v3 shim: time_scale resampling, phase-gated blending, unconditional envelope. Reuses the stub
env / policies from tests/test_shim.py (no dependency on envs/mock_env)."""
from __future__ import annotations

import numpy as np

from fleet_memory.execution import params as s3
from fleet_memory.execution.constraints import ConstraintSet, apply_edit
from fleet_memory.execution.envelope import Envelope
from fleet_memory.execution.params import S3Params
from fleet_memory.execution.shim import ExecutionShim, resample_chunk
from fleet_memory.memory.schema import Edit
from tests.test_shim import TASK, FixedPolicy, ScriptedPolicy, make, run


def test_resample_chunk_preserves_displacement():
    chunk = np.array([[0.3, 0.2, -0.1, 0.1, 0.0, 0.05, -1],
                      [0.3, 0.2, -0.1, 0.1, 0.0, 0.05, -1],
                      [0.2, -0.1, 0.0, 0.0, 0.1, 0.0, 1],
                      [0.1, 0.0, 0.2, 0.0, 0.1, 0.0, 1]], np.float32)
    fast = resample_chunk(chunk, 2.0)
    assert fast.shape == (2, 7) and fast.dtype == np.float32
    np.testing.assert_allclose(fast[:, :6].sum(0), chunk[:, :6].sum(0), atol=1e-6)
    assert list(fast[:, 6]) == [-1, 1]                              # gripper by nearest original row
    slow = resample_chunk(chunk, 0.5)
    assert slow.shape == (8, 7)
    np.testing.assert_allclose(slow[:, :6].sum(0), chunk[:, :6].sum(0), atol=1e-6)
    np.testing.assert_allclose(slow[0, :6], chunk[0, :6] / 2, atol=1e-6)
    assert list(slow[:, 6]) == [-1, -1, -1, -1, 1, 1, 1, 1]
    same = resample_chunk(chunk, 1.0)
    assert same is chunk or np.array_equal(same, chunk)
    assert resample_chunk(chunk[:1], 3.0).shape == (1, 7)           # never fewer than one row


def test_time_scale_halves_chunk_in_shim():
    cs = S3Params(blend_alpha=0.5, time_scale=2.0).to_constraint_set()
    pol = FixedPolicy(np.tile([0.3, -0.2, 0.1, 0.0, 0.0, 0.0, -1.0], (4, 1)))
    env, _, shim = make(pol, cs)
    shim.reset("x")
    out = shim.act(env.obs())
    assert out.shape == (2, 7)
    np.testing.assert_allclose(out[:, :6].sum(0), pol.chunk[:, :6].sum(0), atol=1e-6)
    np.testing.assert_allclose(out[0, :3], [0.6, -0.4, 0.2], atol=1e-6)
    ev = [e for e in shim.events if e["kind"] == "time_scaled"]
    assert len(ev) == 1 and ev[0]["rows_in"] == 4 and ev[0]["rows_out"] == 2 and ev[0]["time_scale"] == 2.0
    # identity vector: untouched, no event
    env, _, shim = make(pol, S3Params.identity().to_constraint_set())
    shim.reset("x")
    out = shim.act(env.obs())
    assert out.shape == (4, 7) and not [e for e in shim.events if e["kind"] == "time_scaled"]


def test_time_scale_shortens_scripted_episode():
    """The stub policy commands proportional (unsaturated) rows near its goals: a faster time base
    lifts the bowl earlier, a slower one later."""
    def lift_step(ts):
        env, tracker, shim = make(ScriptedPolicy(bias_z=0.0), S3Params(blend_alpha=0.5, time_scale=ts).to_constraint_set())
        run(env, shim, tracker, steps=120)
        assert env.held
        down = next(i for i, e in enumerate(env.ee_log) if e[2] < 0.06)          # arrived at the bowl
        return next(j for j, e in enumerate(env.ee_log) if j > down and e[2] > 0.10)   # lifted it
    assert lift_step(1.5) < lift_step(1.0) < lift_step(0.6)


def test_blend_grasp_offset_shifts_closing_point():
    def closing(cs):
        env, tracker, shim = make(ScriptedPolicy(bias_z=0.015), cs)
        run(env, shim, tracker, steps=80)
        assert len(env.close_rel_z) == 1
        return env.close_rel_z[0], env, shim

    base, env0, shim0 = closing(S3Params(blend_alpha=0.5).to_constraint_set())   # blend on, no offset
    assert abs(base - 0.015) < 0.003 and not env0.held                    # biased grasp misses
    kinds0 = [e["kind"] for e in shim0.events]
    assert "blended" in kinds0 and "waypoint_reached" in kinds0 and "waypoint_active" in kinds0
    assert not any(e.get("mode") == "override" for e in shim0.events)
    rel, env, shim = closing(S3Params(blend_alpha=0.5, grasp_offset_z=-0.015).to_constraint_set())
    assert abs((rel - base) + 0.015) < 0.005, (rel, base)
    assert env.held
    kinds = [e["kind"] for e in shim.events]
    assert kinds.count("offset_applied") == 1 and "blended" in kinds


def test_blend_never_overrides_and_only_in_approach():
    cs = S3Params(blend_alpha=0.5, pregrasp_height=0.08).to_constraint_set()             # disagrees with the stub's 0.05 hover
    pol = ScriptedPolicy(bias_z=0.0)
    env, tracker, shim = make(pol, cs)
    obs = env.obs()
    tracker.update(obs, env.subconds())
    shim.reset(TASK.language, "bowl")
    far = shim.act(obs)                                                  # 0.2 m above: outside the approach radius
    assert far.shape == (3, 7) and shim.phase == "policy" and shim.events == []
    n_blend, n_policy_rows = 0, 0
    while env.t < 120 and not env.held:
        chunk = shim.act(obs)
        if shim.phase == "blend":
            n_blend += 1
            assert chunk.shape[0] == pol.k                               # full policy chunk, not 1-row override
        for row in chunk:
            obs = env.step(row)
            tracker.update(obs, env.subconds())
    assert n_blend > 0 and env.held                                      # nudged, then handed back; still grasps
    kinds = [e["kind"] for e in shim.events]
    assert "waypoint_reached" in kinds and kinds.index("blended") < kinds.index("waypoint_reached")
    assert max(e["t"] for e in shim.events if e["kind"] == "blended") <= [e for e in shim.events if e["kind"] == "waypoint_reached"][0]["t"]
    # closed gripper: no blending on a new subtask
    shim.reset("lift it", "bowl")
    tracker.gripper_cmd = 1.0
    n = len(shim.events)
    shim.act(obs)
    assert shim.phase == "policy" and len(shim.events) == n


def test_blend_direction_pulls_toward_waypoint():
    cs = S3Params(blend_alpha=0.5, approach_offset_xyz=np.array([0.03, 0.0, 0.0]), pregrasp_height=0.05).to_constraint_set()
    pol = FixedPolicy([[0, 0, 0, 0, 0, 0, -1]] * 2)                       # a VLA that does nothing
    env, tracker, shim = make(pol, cs)
    env.ee = env.obj["bowl"] + np.array([0.0, 0.0, 0.05])                 # at the unshifted hover point
    tracker.update(env.obs(), env.subconds())
    shim.reset("x", "bowl")
    out = shim.act(env.obs())
    assert shim.phase == "blend" and out.shape == (2, 7)
    np.testing.assert_allclose(out[0, :3], [s3.BLEND_ALPHA * 0.6, 0, 0], atol=1e-6)   # alpha * (0.03/0.05)
    assert out[1, 0] < out[0, 0]                                          # in-chunk EE estimate reduces the pull


def test_cone_in_blend_mode_after_waypoint_only():
    cs = S3Params(blend_alpha=0.5, approach_cone_deg=20.0).to_constraint_set()
    pol = FixedPolicy([[0.8, 0, 0, 0, 0, 0, -1]])
    env, tracker, shim = make(pol, cs)
    env.ee = env.obj["bowl"] + np.array([0, 0, 0.05])                    # at the waypoint -> reached at once
    tracker.update(env.obs(), env.subconds())
    shim.reset("x", "bowl")
    out = shim.act(env.obs())
    np.testing.assert_allclose(out[0, :3], [0, 0, -0.8], atol=1e-6)
    assert [e["kind"] for e in shim.events] == ["waypoint_reached", "cone_projected"]
    up = FixedPolicy([[0.1, 0, 0.9, 0, 0, 0, -1]])                        # retreat rows are never flipped into a dive
    env, tracker, shim = make(up, cs)
    env.ee = env.obj["bowl"] + np.array([0, 0, 0.05])
    shim.reset("x", "bowl")
    np.testing.assert_allclose(shim.act(env.obs())[0, :3], [0.1, 0, 0.9], atol=1e-6)


def test_envelope_clamps_even_when_disabled():
    envlp = Envelope(max_pos_delta=0.5, workspace_min=np.array([-1, -1, 0.0], np.float32),
                     workspace_max=np.array([1, 1, 0.21], np.float32))
    pol = FixedPolicy([[1, -1, 1, 1, -1, 1, 1], [0.2, 0, -0.3, 0, 0, 0, -1]])
    env, tracker, shim = make(pol, ConstraintSet(), enabled=False)
    shim.apply_envelope(envlp)
    shim.reset("x")
    out = shim.act(env.obs())                                            # EE at z=0.20, ceiling 0.21
    np.testing.assert_allclose(out[0], [0.5, -0.5, 0.0, 1, -1, 1, 1], atol=1e-6)   # dz up would leave the box
    np.testing.assert_allclose(out[1], [0.2, 0, -0.3, 0, 0, 0, -1], atol=1e-6)
    assert shim.events == [] and shim.envelope_clamps == 1               # not a surface: no event, counted only
    assert tracker.gripper_cmd == -1.0
    # enabled shim: envelope still applied last, after the S3 caps
    cs = apply_edit(ConstraintSet(), Edit("set_velocity_cap", {"max_pos_delta": 0.8, "max_rot_delta": 0.8}))
    env, _, shim = make(pol, cs)
    shim.apply_envelope(envlp)
    shim.reset("x")
    out = shim.act(env.obs())
    np.testing.assert_allclose(out[0, :3], [0.5, -0.5, 0.0], atol=1e-6)
    np.testing.assert_allclose(out[0, 3:6], [0.8, -0.8, 0.8], atol=1e-6)
    # a controller step (offset hold) is clamped too: 1-row chunk through the envelope
    env, tracker, shim = make(ScriptedPolicy(bias_z=0.015), S3Params(blend_alpha=0.5, grasp_offset_z=-0.015).to_constraint_set())
    shim.apply_envelope(Envelope(max_pos_delta=0.25, workspace_min=np.array([-1, -1, -0.05], np.float32),
                                 workspace_max=np.array([1, 1, 1.0], np.float32)))
    run(env, shim, tracker, steps=120)
    assert env.held and shim.envelope_clamps > 0
