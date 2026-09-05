"""Predicates over an Obs (schema.PREDICATES) plus the per-episode Tracker that supplies the
state they need: initial object heights, the stall counter and a short EE/object history.

Nothing here touches an env, a policy or an LLM. The runner calls `tracker.update` every step;
the shim / coach call `evaluate_predicate`.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from fleet_memory.envs.base import Obs, TaskInfo
from fleet_memory.memory.schema import PREDICATES

# Thresholds, as documented next to schema.PREDICATES (metres).
NEAR_M = 0.06            # ee_near_target
ABOVE_XY_M = 0.03        # ee_above_target
BELOW_RIM_M = 0.01       # object_height_below_rim: z < z0 + 0.01
LIFTED_M = 0.03          # object_lifted: z > z0 + 0.03
SLIP_WINDOW = 10         # grasp_slipped: gripper closed for the last 10 steps ...
SLIP_OBJ_M = 0.005       # ... object path length < 0.005 m ...
SLIP_EE_M = 0.02         # ... while EE path length > 0.02 m
# Panda two-finger qpos: open ≈ [0.04, -0.04] (aperture 0.08), closed on a small object < ~0.02.
GRIPPER_CLOSED_APERTURE_M = 0.03


class Tracker:
    """Per-episode state for the predicates. `target` is settable (S1 rebind); `gripper_cmd`
    is the last commanded gripper value, written by the shim, and takes precedence over the
    finger-joint fallback when deciding open/closed."""

    def __init__(self, task: TaskInfo, stall_k: int = 40):
        self.task = task
        self.stall_k = stall_k
        self._target: str | None = task.objects[0] if task.objects else None
        self.gripper_cmd: float | None = None
        self.stall_steps = 0
        self.t = 0
        self._initial_z: dict[str, float] = {}
        self._last_subconds: dict[str, bool] | None = None
        self._hist: deque = deque(maxlen=SLIP_WINDOW + 1)   # (ee_pos, target_pos|None, gripper_closed)

    @property
    def target(self) -> str | None:
        return self._target

    @target.setter
    def target(self, name: str | None) -> None:
        if name != self._target:
            self._hist.clear()
        self._target = name

    def update(self, obs: Obs, subconds: dict[str, bool]) -> None:
        self.t = int(obs.t)
        for name, (pos, _) in obs.object_poses.items():
            self._initial_z.setdefault(name, float(pos[2]))
        sc = dict(subconds or {})
        self.stall_steps = self.stall_steps + 1 if sc == self._last_subconds else 0
        self._last_subconds = sc
        tgt = obs.object_pos(self._target) if self._target else None
        self._hist.append((np.asarray(obs.ee_pos, dtype=float).copy(),
                           None if tgt is None else np.asarray(tgt, dtype=float).copy(),
                           gripper_closed(obs, self)))

    def initial_z(self, name: str) -> float | None:
        return self._initial_z.get(name)

    def no_progress(self) -> bool:
        return self.stall_steps >= self.stall_k

    def reset_stall(self) -> None:
        """Called by the shim after an abort/retry so the stall does not re-fire immediately."""
        self.stall_steps = 0

    def slipped(self) -> bool:
        """Gripper closed for the whole window, EE travelled > SLIP_EE_M, target travelled < SLIP_OBJ_M."""
        if len(self._hist) < SLIP_WINDOW + 1:
            return False
        h = list(self._hist)
        if not all(c for _, _, c in h) or any(o is None for _, o, _ in h):
            return False
        ee_path = sum(float(np.linalg.norm(h[i + 1][0] - h[i][0])) for i in range(len(h) - 1))
        obj_path = sum(float(np.linalg.norm(h[i + 1][1] - h[i][1])) for i in range(len(h) - 1))
        return ee_path > SLIP_EE_M and obj_path < SLIP_OBJ_M


def gripper_closed(obs: Obs, tracker: Tracker | None = None) -> bool:
    cmd = getattr(tracker, "gripper_cmd", None)
    if cmd is not None:
        return float(cmd) > 0
    q = np.asarray(obs.gripper_qpos, dtype=float).ravel()
    if q.size == 0:
        return False
    aperture = abs(q[0] - q[1]) if q.size >= 2 else abs(q[0])
    return bool(aperture < GRIPPER_CLOSED_APERTURE_M)


def evaluate_predicate(name: str, obs: Obs, target: str | None, tracker: Tracker | None) -> bool:
    """Evaluate one of schema.PREDICATES. `target` falls back to tracker.target; predicates that
    need a target return False when it is absent from obs.object_poses. Tolerates tracker=None
    (or a duck-typed one) by treating missing state as "not observed"."""
    if name not in PREDICATES:
        raise ValueError(f"unknown predicate {name!r}")
    if name == "always":
        return True
    if name == "gripper_open":
        return not gripper_closed(obs, tracker)
    if name == "gripper_closed":
        return gripper_closed(obs, tracker)
    if name == "no_progress":
        return bool(tracker is not None and tracker.no_progress())
    tname = target or getattr(tracker, "target", None)
    tpos = obs.object_pos(tname) if tname else None
    if tpos is None:
        return False
    ee = np.asarray(obs.ee_pos, dtype=float)
    tpos = np.asarray(tpos, dtype=float)
    if name == "ee_near_target":
        return float(np.linalg.norm(ee - tpos)) < NEAR_M
    if name == "ee_above_target":
        return float(np.linalg.norm((ee - tpos)[:2])) < ABOVE_XY_M and float(ee[2]) > float(tpos[2])
    z0 = tracker.initial_z(tname) if hasattr(tracker, "initial_z") else None
    z0 = float(tpos[2]) if z0 is None else float(z0)
    if name == "object_height_below_rim":
        return float(tpos[2]) < z0 + BELOW_RIM_M
    if name == "object_lifted":
        return float(tpos[2]) > z0 + LIFTED_M
    if name == "grasp_slipped":
        return bool(hasattr(tracker, "slipped") and gripper_closed(obs, tracker) and tracker.slipped())
    return False
