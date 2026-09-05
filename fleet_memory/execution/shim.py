"""S3 enforcement. Wraps a Policy; every action chunk passes through the active ConstraintSet.

Transforms, in order per act():
  (6) abort/retry     abort_predicate fires -> open gripper, lift `abort_lift_m`, policy.reset, continue
  (1) waypoint        gripper open and target+pre_grasp_waypoint not yet reached this subtask ->
                      proportional controller overrides the policy (1-row chunks) until within tol
  (2) grasp offset    the first close the policy commands near the target is held while a controller
                      translates the EE by grasp_offset, then the close goes through -> the grasp
                      lands at (policy's closing point + offset)
  (3) approach cone   near target and gripper open: dpos outside the cone is replaced by |dpos| * v
  (4) velocity caps   |dpos| <= max_pos_delta, |drot| <= max_rot_delta (also on controller actions)
  (5) gripper remap   gripper > 0 -> gripper_close_value
enabled=False is a pure pass-through. Every transform is appended to `events`.
"""
from __future__ import annotations

import numpy as np

from fleet_memory.envs.base import Obs, TaskInfo
from fleet_memory.execution.constraints import ConstraintSet
from fleet_memory.execution.detectors import Tracker, evaluate_predicate, gripper_closed
from fleet_memory.policies.base import ActionChunk, Policy, clip_action

K_P = 1.0               # controller gain: dpos = clip(K_P * err_m / CTRL_SCALE_M, -1, 1)
CTRL_SCALE_M = 0.05     # error (m) that saturates the controller
OFFSET_TOL_M = 0.004    # convergence tolerance of the offset / lift controllers
MAX_CTRL_STEPS = 60     # safety: a controller that has not converged by then hands back
RETRY_OPEN_STEPS = 3    # steps spent opening the gripper before lifting in a retry


def _ctrl(err: np.ndarray, gripper: float = -1.0) -> ActionChunk:
    a = np.zeros((1, 7), dtype=np.float32)
    a[0, :3] = np.clip(K_P * np.asarray(err, dtype=np.float32) / CTRL_SCALE_M, -1.0, 1.0)
    a[0, 6] = gripper
    return a


class ExecutionShim:
    """Implements the Policy protocol around `policy`."""

    def __init__(self, policy: Policy, constraints: ConstraintSet, task: TaskInfo, tracker: Tracker,
                 enabled: bool = True):
        self.policy = policy
        self.constraints = constraints
        self.task = task
        self.tracker = tracker
        self.enabled = enabled
        self.name = f"shim[{getattr(policy, 'name', 'policy')}]"
        self.events: list[dict] = []
        self.retries_used = 0
        self.phase = "policy"                    # policy | waypoint | offset | retry
        self.instruction = task.language
        self.target: str | None = tracker.target
        self._t = 0
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
        if not self.enabled:
            return self._emit(self.policy.act(obs))
        cs = self.constraints
        ee = np.asarray(obs.ee_pos, dtype=float)
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

        # (1) pre-grasp waypoint controller
        if cs.pre_grasp_waypoint is not None and tgt is not None:
            if self._wp is None:
                self._wp = "done" if closed else "pending"
            if self._wp == "pending":
                wp = np.asarray(tgt, dtype=float) + cs.pre_grasp_waypoint
                err = wp - ee
                if float(np.linalg.norm(err)) <= cs.pre_grasp_tol_m or self._ctrl_steps >= MAX_CTRL_STEPS:
                    self._wp, self._ctrl_steps, self.phase = "done", 0, "policy"
                    self._event("waypoint_reached", dist_m=float(np.linalg.norm(err)))
                else:
                    if self._ctrl_steps == 0:
                        self._event("waypoint_active", waypoint=wp.tolist(), tol_m=cs.pre_grasp_tol_m)
                    self._ctrl_steps += 1
                    self.phase = "waypoint"
                    return self._emit(self._cap(_ctrl(err), cs))

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

        chunk = self._pending if self._pending is not None else clip_action(self.policy.act(obs)).reshape(-1, 7).copy()
        self._pending = None
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
        # (3) (4) (5)
        if near and not closed and cs.approach_vector is not None:
            chunk = self._cone(chunk, cs)
        chunk = self._cap(chunk, cs)
        chunk = self._remap_gripper(chunk, cs)
        return self._emit(chunk)

    def set_constraints(self, cs: ConstraintSet) -> None:
        self.constraints = cs

    def set_target(self, name: str) -> None:
        self.target = name
        self.tracker.target = name

    # ------------------------------------------------------------------ internals
    def _new_attempt(self) -> None:
        self._wp: str | None = None              # None = undecided, "pending", "done"
        self._offset_goal: np.ndarray | None = None
        self._offset_done = False
        self._pending: np.ndarray | None = None  # policy rows held back by the offset controller
        self._retry: dict | None = None
        self._ctrl_steps = 0
        self.phase = "policy"

    def _event(self, kind: str, **kw) -> None:
        self.events.append({"t": self._t, "kind": kind, **kw})

    def _emit(self, chunk: ActionChunk) -> ActionChunk:
        chunk = clip_action(chunk).reshape(-1, 7)
        g = float(chunk[-1, 6])
        if self._last_gripper is not None and self._last_gripper > 0 >= g:
            self._offset_done = False            # re-opened: the next grasp gets the offset again
        self._last_gripper = g
        self.tracker.gripper_cmd = g
        return chunk

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
            if m > 1e-6 and float(np.dot(row[:3], v)) / m < cos_lim:
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
