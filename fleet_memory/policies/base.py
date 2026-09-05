"""Policy protocol. The VLA is frozen; this is the only way anything talks to it."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from fleet_memory.envs.base import Obs

# ActionChunk: float32[k, 7]; k >= 1 actions to execute in order before re-querying the policy.
# Columns: [dx, dy, dz, droll, dpitch, dyaw, gripper], each in [-1, 1]. gripper > 0 closes.
ActionChunk = np.ndarray


class Policy(Protocol):
    name: str

    def reset(self, instruction: str) -> None:
        """Start a new (sub)episode conditioned on `instruction`. Clears any internal action queue."""
        ...

    def act(self, obs: Obs) -> ActionChunk:
        """Return the next action chunk. The caller executes all k rows before calling again
        unless a shim intervenes."""
        ...


def clip_action(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    return np.clip(a, -1.0, 1.0)
