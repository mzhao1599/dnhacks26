"""Immutable safety envelope. Applied AFTER the shim, unconditionally, on every action.
Not optimisable, not a lesson, never logged as a surface. The optimizer can only get faster by
finding shorter paths inside it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# LIBERO/robosuite OSC_POSE: |dpos|=1 -> 0.05 m target step, |drot|=1 -> 0.5 rad. Franka over a table.
POS_SCALE_M = 0.05


@dataclass(frozen=True)
class Envelope:
    max_pos_delta: float = 1.0                      # absolute cap on |dx|,|dy|,|dz| (action units)
    max_rot_delta: float = 1.0                      # absolute cap on |droll|,|dpitch|,|dyaw|
    max_gripper: float = 1.0                        # absolute cap on |gripper|
    # LIBERO world frame (verified on libero_10 task 0): EE starts at z≈0.70, table objects z≈0.43-0.48,
    # shelf objects up to z≈1.05, x∈[-0.27, 0.08], y∈[-0.26, 0.27]. Loose box — a safety net, not a controller.
    workspace_min: np.ndarray = field(default_factory=lambda: np.array([-0.60, -0.55, 0.38], np.float32))
    workspace_max: np.ndarray = field(default_factory=lambda: np.array([0.45, 0.55, 1.35], np.float32))

    def clamp(self, action: np.ndarray, ee_pos: np.ndarray | None = None) -> tuple[np.ndarray, bool]:
        """Return (clamped action, changed). Zeroes any positional delta that would leave the workspace AABB."""
        a = np.array(action, dtype=np.float32).reshape(-1)
        out = a.copy()
        out[0:3] = np.clip(out[0:3], -self.max_pos_delta, self.max_pos_delta)
        out[3:6] = np.clip(out[3:6], -self.max_rot_delta, self.max_rot_delta)
        out[6] = float(np.clip(out[6], -self.max_gripper, self.max_gripper))
        if ee_pos is not None:
            nxt = np.asarray(ee_pos, np.float32) + out[0:3] * POS_SCALE_M
            leaving = (nxt < self.workspace_min) & (out[0:3] < 0) | (nxt > self.workspace_max) & (out[0:3] > 0)
            out[0:3][leaving] = 0.0
        return out, bool(np.any(out != a))

    def clamp_chunk(self, chunk: np.ndarray, ee_pos: np.ndarray | None = None) -> tuple[np.ndarray, int]:
        """Clamp every row; the AABB check uses the running EE estimate. Returns (chunk, rows changed)."""
        rows, changed, pos = [], 0, None if ee_pos is None else np.asarray(ee_pos, np.float32).copy()
        for a in np.asarray(chunk, np.float32).reshape(-1, 7):
            c, ch = self.clamp(a, pos)
            rows.append(c)
            changed += int(ch)
            if pos is not None:
                pos = pos + c[0:3] * POS_SCALE_M
        return np.stack(rows).astype(np.float32), changed


DEFAULT_ENVELOPE = Envelope()
MOCK_ENVELOPE = Envelope(workspace_min=np.array([-1.0, -1.0, -0.05], np.float32),
                         workspace_max=np.array([1.0, 1.0, 1.0], np.float32))
