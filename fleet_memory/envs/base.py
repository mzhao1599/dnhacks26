"""Env protocol. The env predicate is the ONLY source of truth for success."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class Obs:
    t: int                                              # step index within episode
    images: dict[str, np.ndarray]                       # {"agentview": HxWx3 uint8, "eye_in_hand": HxWx3 uint8} — already upright
    state: np.ndarray                                   # raw proprio vector the policy adapter expects (float32)
    ee_pos: np.ndarray                                  # (3,) world-frame end-effector position, metres
    ee_quat: np.ndarray                                 # (4,) xyzw
    gripper_qpos: np.ndarray                            # (2,) finger joint positions
    object_poses: dict[str, tuple[np.ndarray, np.ndarray]]   # name -> (pos(3,), quat(4,)) for every scene object
    raw: dict[str, Any] = field(default_factory=dict)   # untouched underlying obs dict (may be large; never logged)

    def object_pos(self, name: str) -> np.ndarray | None:
        p = self.object_poses.get(name)
        return None if p is None else p[0]


@dataclass
class TaskInfo:
    suite: str                     # "libero_10" | "libero_object" | "libero_goal" | "libero_spatial" | "mock"
    task_id: str                   # stable slug, e.g. "put_the_black_bowl_in_the_drawer"
    language: str                  # canonical instruction string from the benchmark
    task_family: str               # coarse family used by Trigger.task_family
    objects: list[str]             # object names present in Obs.object_poses
    max_steps: int                 # episode horizon


class Env(Protocol):
    """One task, one seed, one episode at a time. Implementations: libero_env.LiberoEnv, mock_env.MockEnv."""

    def task_info(self) -> TaskInfo: ...

    def reset(self, seed: int) -> Obs:
        """Deterministic reset. Same (task, seed) => same initial state."""
        ...

    def step(self, action: np.ndarray) -> tuple[Obs, bool, dict[str, Any]]:
        """action: float32[7] in [-1,1]. Returns (obs, done, info). done is True on success OR horizon."""
        ...

    def success_flag(self) -> bool:
        """The benchmark's programmatic success predicate. Nothing else counts."""
        ...

    def success_subconditions(self) -> dict[str, bool]:
        """Finer-grained progress signals for the stall detector (may be empty). Never used as success."""
        ...

    def close(self) -> None: ...
