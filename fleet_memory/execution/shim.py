"""S3 enforcement. Wraps a Policy; every action chunk passes through the active ConstraintSet.

Transforms, in order per act():
  (6) abort/retry     abort_predicate fires -> open gripper, lift `abort_lift_m`, policy.reset, continue
  (1) pre-grasp       blend_mode="override" (v2): gripper open and target+pre_grasp_waypoint not yet
                      reached this subtask -> proportional controller overrides the policy (1-row chunks)
                      blend_mode="blend" (v3 default): during the APPROACH PHASE only (gripper open and
                      ||ee-target|| < params.APPROACH_RADIUS_M) and until the waypoint is reached,
                      dpos = (1-alpha)*dpos_vla + alpha*dpos_toward(target+pre_grasp_waypoint)
  (-1) camera calib   v3.2: constraints.image_calib (roll/zoom/shift) is warped onto obs.images["agentview"]
                      before the policy sees it (execution/calib.py). Only the policy's copy changes; the
                      tracker, phase detector and envelope keep the env's observation.
  (0) time_scale      the policy chunk is resampled in time: cumsum of the 6 delta dims, linear
                      interpolation onto round(k/time_scale) rows (>=1), diff back; gripper by nearest
  (2) grasp offset    the first close the policy commands near the target is held while a controller
                      translates the EE by grasp_offset, then the close goes through -> the grasp
                      lands at (policy's closing point + offset). Same in both modes.
  (3) approach cone   approach phase (blend: after the waypoint; override: ee_near_target), gripper
                      open: dpos leaving the cone is replaced by |dpos| * v (retreat rows, dot<0, untouched)
  (4) velocity caps   |dpos| <= max_pos_delta, |drot| <= max_rot_delta (also on controller actions)
  (5) gripper remap   gripper > 0 -> gripper_close_value
  (7) envelope        apply_envelope(env): the outgoing chunk is clamped LAST, always, even when
                      enabled=False. Never an event (it is not a surface); counted in `envelope_clamps`.
enabled=False is a pure pass-through (except the envelope). Every transform is appended to `events`.
"""
from __future__ import annotations

import numpy as np

from fleet_memory.envs.base import Obs, TaskInfo
from fleet_memory.execution import params as s3
from fleet_memory.execution.calib import calibrate_obs
from fleet_memory.execution.constraints import ConstraintSet
from fleet_memory.execution.detectors import Tracker, evaluate_predicate, gripper_closed
from fleet_memory.execution.envelope import Envelope
from fleet_memory.policies.base import ActionChunk, Policy, clip_action

K_P = 1.0               # controller gain: dpos = clip(K_P * err_m / CTRL_SCALE_M, -1, 1)
CTRL_SCALE_M = 0.05     # error (m) that saturates the controller
OFFSET_TOL_M = 0.004    # convergence tolerance of the offset / lift controllers
MAX_CTRL_STEPS = 60     # safety: a controller / blend that has not converged by then hands back
BLEND_STALL_CHUNKS = 3  # blend: chunks without getting closer to the waypoint before handing back
RETRY_OPEN_STEPS = 3    # steps spent opening the gripper before lifting in a retry


def _ctrl(err: np.ndarray, gripper: float = -1.0) -> ActionChunk:
    a = np.zeros((1, 7), dtype=np.float32)
    a[0, :3] = _toward(err)
    a[0, 6] = gripper
    return a


def _toward(err: np.ndarray) -> np.ndarray:
    return np.clip(K_P * np.asarray(err, dtype=np.float32) / CTRL_SCALE_M, -1.0, 1.0)


def resample_chunk(chunk: ActionChunk, time_scale: float) -> ActionChunk:
    """Resample a (k,7) chunk onto round(k/time_scale) rows preserving total 6-D displacement.
    Gripper rows are taken by nearest original row. time_scale=1 (or a no-op rounding) returns the input."""
    chunk = np.asarray(chunk, np.float32).reshape(-1, 7)
    k = len(chunk)
    n = max(1, int(round(k / float(time_scale))))
    if n == k or k == 0:
        return chunk
    c = np.vstack([np.zeros((1, 6), np.float64), np.cumsum(chunk[:, :6].astype(np.float64), axis=0)])
    t_old, t_new = np.arange(k + 1, dtype=np.float64), np.linspace(0.0, float(k), n + 1)
    c_new = np.stack([np.interp(t_new, t_old, c[:, j]) for j in range(6)], axis=1)
    out = np.zeros((n, 7), np.float32)
    out[:, :6] = np.diff(c_new, axis=0)
    idx = np.minimum(k - 1, np.floor((np.arange(n) + 0.5) * k / n).astype(int))
    out[:, 6] = chunk[idx, 6]
    return out


class ExecutionShim:
    """Implements the Policy protocol around `policy`."""

    def __init__(self, policy: Policy, constraints: ConstraintSet, task: TaskInfo, tracker: Tracker,
                 enabled: bool = True, blend_mode: str = "blend", alpha: float = s3.BLEND_ALPHA,
                 envelope: Envelope | None = None, pos_scale_m: float = CTRL_SCALE_M):
        if blend_mode not in ("blend", "override"):
            raise ValueError(f"blend_mode must be 'blend' or 'override', got {blend_mode!r}")
        self.policy = policy
        self.constraints = constraints
        self.task = task
        self.tracker = tracker
        self.enabled = enabled
        self.blend_mode = blend_mode
        self.alpha = float(alpha)
        self.envelope = envelope
        self.pos_scale_m = float(pos_scale_m)     # metres per unit dpos, for the in-chunk EE estimate
        self.envelope_clamps = 0
        self.name = f"shim[{getattr(policy, 'name', 'policy')}]"
        self.events: list[dict] = []
        self.retries_used = 0
        self.phase = "policy"                    # policy | waypoint | blend | offset | retry
        self.instruction = task.language
        self.target: str | None = tracker.target
        self._t = 0
        self._ee: np.ndarray | None = None
        self._last_gripper: float | None = None
        self._new_attempt()

    # ------------------------------------------------------------------ Policy protocol
    def reset(self, instruction: str, target_object: str | None = None,
              constraints: ConstraintSet | None = None) -> None:
        self.instruction = instruction
        if target_object is not None:
            self.set_target(target_object)
        if constraints is not None:
            self.constraints = constraints
        self._new_attempt()
        self.policy.reset(instruction)

    def act(self, obs: Obs) -> ActionChunk:
        self._t = int(obs.t)
        self._ee = np.asarray(obs.ee_pos, dtype=float)
        if not self.enabled:
            return self._emit(self.policy.act(obs))
        cs = self.constraints
        ee = self._ee
        tgt = obs.object_pos(self.target) if self.target else None

        # (6) abort / retry
        if self._retry is not None:
            a = self._retry_step(ee, cs)
            if a is not None:
                return self._emit(a)
        elif (cs.abort_predicate and self.retries_used < cs.abort_max_retries
              and evaluate_predicate(cs.abort_predicate, obs, self.target, self.tracker)):
            self.retries_used += 1
            self._retry = {"stage": "open", "n": 0, "z_goal": float(ee[2]) + cs.abort_lift_m}
            self.phase = "retry"
            self._event("abort_retry", predicate=cs.abort_predicate, retry=self.retries_used,
                        lift_m=cs.abort_lift_m)
            return self._emit(self._retry_step(ee, cs))
        closed = gripper_closed(obs, self.tracker)

        # (1) pre-grasp waypoint controller (override mode only; blend mode handles it below)
        if self.blend_mode == "override" and cs.pre_grasp_waypoint is not None and tgt is not None:
            a = self._waypoint_override(ee, tgt, closed, cs)
            if a is not None:
                return self._emit(a)

        # (2) grasp-offset controller: the policy's closing chunk is held (not discarded, so a
        #     scripted policy that advances on emit keeps its close) until the EE has moved by the offset
        if self._offset_goal is not None:
            err = self._offset_goal - ee
            if float(np.linalg.norm(err)) <= OFFSET_TOL_M or self._ctrl_steps >= MAX_CTRL_STEPS:
                self._event("offset_applied", offset=cs.grasp_offset.tolist(), residual_m=float(np.linalg.norm(err)))
                self._offset_goal, self._offset_done, self._ctrl_steps, self.phase = None, True, 0, "policy"
            else:
                self._ctrl_steps += 1
                self.phase = "offset"
                return self._emit(self._cap(_ctrl(err), cs))

        if self._pending is not None:
            chunk, self._pending = self._pending, None
        else:
            chunk = self._time_scale(clip_action(self.policy.act(self._calibrated(obs, cs))).reshape(-1, 7).copy(), cs)   # (0)
        near = evaluate_predicate("ee_near_target", obs, self.target, self.tracker)
        if near and not closed and not self._offset_done and np.any(cs.grasp_offset) and tgt is not None:
            close_rows = np.flatnonzero(chunk[:, 6] > 0)
            if close_rows.size:
                i = int(close_rows[0])
                if i == 0:
                    self._pending = chunk        # replayed once the offset is applied
                    self._offset_goal = ee + cs.grasp_offset.astype(float)
                    self._ctrl_steps = 1
                    self.phase = "offset"
                    return self._emit(self._cap(_ctrl(self._offset_goal - ee), cs))
                self._pending, chunk = chunk[i:], chunk[:i]   # pre-close rows pass now, the close next call
        # (1 blend) (3) (4) (5)
        self.phase = "policy"
        if self.blend_mode == "blend":
            approach = (not closed and tgt is not None
                        and float(np.linalg.norm(ee - np.asarray(tgt, dtype=float))) < s3.APPROACH_RADIUS_M)
            if approach and cs.pre_grasp_waypoint is not None:
                chunk = self._blend(chunk, ee, np.asarray(tgt, dtype=float), cs)
            cone_ok = approach and self._wp != "pending"
        else:
            cone_ok = near and not closed
        if cone_ok and cs.approach_vector is not None:
            chunk = self._cone(chunk, cs)
        chunk = self._cap(chunk, cs)
        chunk = self._remap_gripper(chunk, cs)
        return self._emit(chunk)

    def set_constraints(self, cs: ConstraintSet) -> None:
        self.constraints = cs

    def set_target(self, name: str) -> None:
        self.target = name
        self.tracker.target = name

    def apply_envelope(self, envelope: Envelope | None) -> None:
        """Immutable safety envelope, clamped onto every outgoing chunk (also when enabled=False)."""
        self.envelope = envelope

    # ------------------------------------------------------------------ internals
    def _new_attempt(self) -> None:
        self._wp: str | None = None              # None = undecided, "pending", "done"
        self._wp_best: float | None = None       # blend: best real distance to the waypoint so far
        self._wp_stall = 0                       # blend: chunks without improvement
        self._offset_goal: np.ndarray | None = None
        self._offset_done = False
        self._pending: np.ndarray | None = None  # policy rows held back by the offset controller
        self._retry: dict | None = None
        self._ctrl_steps = 0
        self._calib_logged = False
        self.phase = "policy"

    def _event(self, kind: str, **kw) -> None:
        self.events.append({"t": self._t, "kind": kind, **kw})

    def _calibrated(self, obs: Obs, cs: ConstraintSet) -> Obs:
        """(-1) the policy's view of the external camera, re-calibrated by cs.image_calib (identity: obs itself)."""
        calib = getattr(cs, "image_calib", None)
        if not calib:
            return obs
        if not self._calib_logged:
            self._calib_logged = True
            self._event("image_calib_active", **{k: v for k, v in calib.items()})
        return calibrate_obs(obs, calib)

    def _emit(self, chunk: ActionChunk) -> ActionChunk:
        chunk = clip_action(chunk).reshape(-1, 7)
        if self.envelope is not None:                                   # (7) always last
            chunk, n = self.envelope.clamp_chunk(chunk, self._ee)
            self.envelope_clamps += int(n)
        g = float(chunk[-1, 6])
        if self._last_gripper is not None and self._last_gripper > 0 >= g:
            self._offset_done = False            # re-opened: the next grasp gets the offset again
        self._last_gripper = g
        self.tracker.gripper_cmd = g
        return chunk

    def _time_scale(self, chunk: ActionChunk, cs: ConstraintSet) -> ActionChunk:
        ts = float(getattr(cs, "time_scale", 1.0) or 1.0)
        if ts == 1.0:
            return chunk
        out = resample_chunk(chunk, ts)
        if len(out) != len(chunk):
            self._event("time_scaled", time_scale=ts, rows_in=int(len(chunk)), rows_out=int(len(out)))
        return out

    def _waypoint_override(self, ee: np.ndarray, tgt: np.ndarray, closed: bool, cs: ConstraintSet) -> ActionChunk | None:
        if self._wp is None:
            self._wp = "done" if closed else "pending"
        if self._wp != "pending":
            return None
        wp = np.asarray(tgt, dtype=float) + cs.pre_grasp_waypoint
        err = wp - ee
        d = float(np.linalg.norm(err))
        if d <= cs.pre_grasp_tol_m or self._ctrl_steps >= MAX_CTRL_STEPS:
            self._wp, self._ctrl_steps, self.phase = "done", 0, "policy"
            self._event("waypoint_reached", dist_m=d, reason="tol" if d <= cs.pre_grasp_tol_m else "timeout")
            return None
        if self._ctrl_steps == 0:
            self._event("waypoint_active", waypoint=wp.tolist(), tol_m=cs.pre_grasp_tol_m, mode="override")
        self._ctrl_steps += 1
        self.phase = "waypoint"
        return self._cap(_ctrl(err), cs)

    def _blend(self, chunk: ActionChunk, ee: np.ndarray, tgt: np.ndarray, cs: ConstraintSet) -> ActionChunk:
        """Approach phase, waypoint pending: pull each row's dpos toward target+pre_grasp_waypoint with
        weight alpha, tracking the EE through the chunk. Latches 'done' within tol or after MAX_CTRL_STEPS
        rows so a policy that disagrees with the waypoint is never held hostage."""
        if self._wp is None:
            self._wp = "pending"
        if self._wp != "pending":
            return chunk
        wp = tgt + cs.pre_grasp_waypoint.astype(float)
        d0 = float(np.linalg.norm(wp - ee))                 # real distance at chunk start: stall check
        if self._wp_best is None or d0 < self._wp_best - OFFSET_TOL_M:
            self._wp_best, self._wp_stall = d0, 0
        else:
            self._wp_stall += 1
        if chunk[0, 6] > 0:                                  # the policy is closing: the approach is over
            self._wp, self._ctrl_steps = "done", 0
            self._event("waypoint_reached", dist_m=d0, reason="closing")
            return chunk
        out, est, n, reached = chunk.copy(), ee.copy(), 0, None
        for row in out:
            err = wp - est
            d = float(np.linalg.norm(err))
            reason = ("tol" if d <= cs.pre_grasp_tol_m else "timeout" if self._ctrl_steps >= MAX_CTRL_STEPS
                      else "stall" if self._wp_stall >= BLEND_STALL_CHUNKS else None)
            if reason:
                self._wp, self._ctrl_steps, reached = "done", 0, (d, reason)
                break
            if self._ctrl_steps == 0:
                self._event("waypoint_active", waypoint=wp.tolist(), tol_m=cs.pre_grasp_tol_m, mode="blend")
            self._ctrl_steps += 1
            a = getattr(cs, "blend_alpha", 0.0) or self.alpha   # vector-controlled strength (v3.1); ctor alpha as fallback
            row[:3] = (1.0 - a) * row[:3] + a * _toward(err)
            est = est + row[:3].astype(float) * self.pos_scale_m
            n += 1
        if n:
            self.phase = "blend"
            self._event("blended", rows=n, alpha=self.alpha, goal=wp.tolist())
        if reached is not None:
            self._event("waypoint_reached", dist_m=reached[0], reason=reached[1])
        return out

    def _retry_step(self, ee: np.ndarray, cs: ConstraintSet) -> ActionChunk | None:
        r = self._retry
        r["n"] += 1
        if r["stage"] == "open":
            if r["n"] >= RETRY_OPEN_STEPS:
                r["stage"] = "lift"
            return self._cap(_ctrl(np.zeros(3)), cs)
        err = np.array([0.0, 0.0, r["z_goal"] - float(ee[2])])
        if err[2] <= OFFSET_TOL_M or r["n"] >= RETRY_OPEN_STEPS + MAX_CTRL_STEPS:
            self._event("retry_done", retry=self.retries_used, lifted_m=float(ee[2] - (r["z_goal"] - cs.abort_lift_m)))
            self._new_attempt()
            self.tracker.reset_stall()
            self.policy.reset(self.instruction)
            return None
        return self._cap(_ctrl(err), cs)

    def _cone(self, chunk: ActionChunk, cs: ConstraintSet) -> ActionChunk:
        v = np.asarray(cs.approach_vector, dtype=np.float32)
        cos_lim = float(np.cos(np.radians(cs.approach_half_angle_deg)))
        n = 0
        for row in chunk:
            m = float(np.linalg.norm(row[:3]))
            if m <= 1e-6:
                continue
            c = float(np.dot(row[:3], v)) / m
            if c >= 0.0 and c < cos_lim:          # retreat rows (c < 0) are not an approach: untouched
                row[:3] = m * v
                n += 1
        if n:
            self._event("cone_projected", rows=n, vector=v.tolist(), half_angle_deg=cs.approach_half_angle_deg)
        return chunk

    def _cap(self, chunk: ActionChunk, cs: ConstraintSet) -> ActionChunk:
        if cs.max_pos_delta >= 1.0 and cs.max_rot_delta >= 1.0:
            return chunk
        out = chunk.copy()
        out[:, :3] = np.clip(out[:, :3], -cs.max_pos_delta, cs.max_pos_delta)
        out[:, 3:6] = np.clip(out[:, 3:6], -cs.max_rot_delta, cs.max_rot_delta)
        n = int(np.sum(np.any(out[:, :6] != chunk[:, :6], axis=1)))
        if n:
            self._event("velocity_clamped", rows=n, max_pos_delta=cs.max_pos_delta, max_rot_delta=cs.max_rot_delta)
        return out

    def _remap_gripper(self, chunk: ActionChunk, cs: ConstraintSet) -> ActionChunk:
        closing = chunk[:, 6] > 0
        if not closing.any():
            return chunk
        out = chunk.copy()
        out[closing, 6] = cs.gripper_close_value
        n = int(np.sum(out[:, 6] != chunk[:, 6]))
        if n:
            self._event("gripper_remapped", rows=n, value=cs.gripper_close_value)
        return out
