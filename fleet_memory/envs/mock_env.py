"""Kinematic 3-D mock env (no physics, no rendering). Implements the Env protocol.

World: an end-effector point + a gripper bool, and three named objects (target / destination /
distractor) with seed-jittered poses on a table at z=0. Actions are OSC-style float32[7]:
dpos * STEP_M metres per step, rotation ignored, gripper > 0 closes.

Grasp: on a close transition the nearest object is grasped iff ||ee - obj|| < GRASP_R_M and
(ee_z - obj_z) in `grasp_z_window`. NOTE: the contract quoted [-0.005, 0.012]; the lower bound is
widened to -0.015 (a ctor arg) because a 15 mm grasp offset must be able to fix a "grasp too high"
policy without breaking the seeds that already succeed (the two windows would overlap by 2 mm).
A held object follows the EE; on release it drops to its rest height.
Success: target within PLACE_R_M (xy) of the destination, z < dest_z + 0.03, gripper open.

reset(seed, perturbation): {"shift_xy": [dx, dy]} shifts the TARGET's initial pose,
{"jitter_m": s} adds seeded gaussian xy jitter to every object, {"distractor": true} adds one
more object next to the target. Nothing here is a success predicate except success_flag().
"""
from __future__ import annotations

from typing import Any

import numpy as np

from fleet_memory.envs.base import Obs, TaskInfo
from fleet_memory.execution.envelope import MOCK_ENVELOPE, Envelope

STEP_M = 0.02            # metres per unit dpos per step
GRASP_R_M = 0.02
PLACE_R_M = 0.05
PLACE_Z_M = 0.03
NOMINAL_JITTER_M = 0.015  # seed jitter (uniform, xy) of the nominal poses: the "training distribution"
EE_START = np.array([0.0, 0.0, 0.20])

# name -> nominal rest pose (x, y, z_center). z is the height the object rests at on the table.
NOMINAL: dict[str, list[float]] = {
    "black_bowl": [0.10, 0.05, 0.030], "plate": [-0.10, 0.00, 0.010], "mug": [0.05, -0.12, 0.035],
    "tray": [-0.12, 0.10, 0.010], "cup": [0.12, -0.02, 0.030], "bottle": [0.02, 0.12, 0.050],
    "basket": [-0.08, -0.12, 0.030],
}

TASKS: dict[str, dict[str, str]] = {
    "pick_bowl_to_plate": dict(target="black_bowl", destination="plate", distractor="mug",
                               language="put the black bowl on the plate", family="place_on_plate"),
    "pick_mug_to_tray": dict(target="mug", destination="tray", distractor="black_bowl",
                             language="put the mug on the tray", family="pick_place_mug"),
    "pick_cup_to_plate": dict(target="cup", destination="plate", distractor="bottle",
                              language="put the cup on the plate", family="place_on_plate"),
    "pick_bottle_to_basket": dict(target="bottle", destination="basket", distractor="mug",
                                  language="put the bottle in the basket", family="place_in_basket"),
}

_QUAT = np.array([0.0, 0.0, 0.0, 1.0], np.float32)
_OPEN = np.array([0.04, -0.04], np.float32)
_CLOSED = np.array([0.0, 0.0], np.float32)


class MockSuite:
    @staticmethod
    def tasks() -> list[str]:
        return list(TASKS)


class MockEnv:
    pos_scale_m = STEP_M     # metres per unit dpos per step (the shim's in-chunk EE estimate reads this)

    def __init__(self, task_id: str = "pick_bowl_to_plate", horizon: int = 200, failure_mode: str = "grasp_high",
                 grasp_z_window: tuple[float, float] | None = None):
        if task_id not in TASKS:
            raise ValueError(f"unknown mock task {task_id!r} (want one of {list(TASKS)})")
        self.task_id, self.horizon, self.failure_mode = task_id, int(horizon), failure_mode
        spec = TASKS[task_id]
        self.target, self.destination, self.distractor = spec["target"], spec["destination"], spec["distractor"]
        self.language, self.family = spec["language"], spec["family"]
        self.grasp_z_window = tuple(grasp_z_window) if grasp_z_window else \
            ((-0.03, 0.03) if failure_mode == "none" else (-0.015, 0.012))
        self.t = 0
        self.seed: int | None = None
        self.perturbation: dict[str, Any] = {}
        self.ee = EE_START.copy()
        self.closed = False
        self.held: str | None = None
        self._held_rel = np.zeros(3)
        self.objects: dict[str, np.ndarray] = {}
        self.rest_z: dict[str, float] = {}
        self._reset_objects(np.random.default_rng(0), None)

    # ------------------------------------------------------------------ protocol
    def task_info(self) -> TaskInfo:
        return TaskInfo(suite="mock", task_id=self.task_id, language=self.language, task_family=self.family,
                        objects=list(self.objects), max_steps=self.horizon)

    def envelope(self) -> Envelope:
        return MOCK_ENVELOPE

    def reset(self, seed: int, perturbation: dict[str, Any] | None = None) -> Obs:
        self.seed, self.t = int(seed), 0
        self.perturbation = dict(perturbation or {})
        rng = np.random.default_rng(self.seed)
        self._reset_objects(rng, self.perturbation)
        self.ee = EE_START + np.array([*rng.uniform(-0.01, 0.01, 2), 0.0])
        self.closed, self.held = False, None
        return self._obs()

    def step(self, action: np.ndarray) -> tuple[Obs, bool, dict[str, Any]]:
        a = np.clip(np.asarray(action, np.float64).reshape(-1)[:7], -1.0, 1.0)
        self.ee = self.ee + a[:3] * STEP_M
        self.ee[2] = max(self.ee[2], 0.0)
        close = bool(a[6] > 0)
        info: dict[str, Any] = {}
        if self.held is not None:
            self.objects[self.held] = self.ee + self._held_rel
        if close and not self.closed:
            info["grasp_attempt"] = self._try_grasp()
        if not close and self.held is not None:
            p = self.objects[self.held]
            self.objects[self.held] = np.array([p[0], p[1], self.rest_z[self.held]])
            self.held = None
        self.closed = close
        self.t += 1
        done = self.success_flag() or self.t >= self.horizon
        info["success"] = self.success_flag()
        return self._obs(), done, info

    def success_flag(self) -> bool:
        tp, dp = self.objects[self.target], self.objects[self.destination]
        return (not self.closed and self.held is None
                and float(np.linalg.norm((tp - dp)[:2])) < PLACE_R_M and float(tp[2]) < float(dp[2]) + PLACE_Z_M)

    def success_subconditions(self) -> dict[str, bool]:
        tp, dp = self.objects[self.target], self.objects[self.destination]
        grasped = self.held == self.target
        return {"grasped": grasped,
                "above_dest": bool(grasped and float(np.linalg.norm((tp - dp)[:2])) < PLACE_R_M),
                "placed": self.success_flag()}

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------ helpers
    def current_obs(self) -> Obs:
        return self._obs()

    @property
    def object_poses(self) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.objects.items()}

    def _reset_objects(self, rng: np.random.Generator, pert: dict[str, Any] | None) -> None:
        pert = pert or {}
        self.objects, self.rest_z = {}, {}
        for name in (self.target, self.destination, self.distractor):
            nom = np.asarray(NOMINAL[name], np.float64)
            self.objects[name] = nom + np.array([*rng.uniform(-NOMINAL_JITTER_M, NOMINAL_JITTER_M, 2), 0.0])
            self.rest_z[name] = float(nom[2])
        shift = pert.get("shift_xy")
        if shift:
            self.objects[self.target] = self.objects[self.target] + np.array([float(shift[0]), float(shift[1]), 0.0])
        prng = np.random.default_rng((self.seed or 0) * 7919 + 17)
        jit = float(pert.get("jitter_m") or 0.0)
        if jit > 0:
            for name in list(self.objects):
                self.objects[name] = self.objects[name] + np.array([*prng.normal(0.0, jit, 2), 0.0])
        if pert.get("distractor"):
            extra = f"{self.distractor}_2"
            ang = prng.uniform(0.0, 2.0 * np.pi)
            tp = self.objects[self.target]
            self.rest_z[extra] = float(NOMINAL[self.distractor][2])
            self.objects[extra] = np.array([tp[0] + 0.06 * np.cos(ang), tp[1] + 0.06 * np.sin(ang), self.rest_z[extra]])

    def _try_grasp(self) -> str | None:
        lo, hi = self.grasp_z_window
        best, best_d = None, GRASP_R_M
        for name, p in self.objects.items():
            if name == self.destination:
                continue
            d, rel = float(np.linalg.norm(self.ee - p)), float(self.ee[2] - p[2])
            if d < best_d and lo <= rel <= hi:
                best, best_d = name, d
        if best is not None:
            self.held, self._held_rel = best, self.objects[best] - self.ee
        return best

    def _obs(self) -> Obs:
        q = _CLOSED if self.closed else _OPEN
        state = np.concatenate([self.ee, np.zeros(3), q]).astype(np.float32)
        poses = {k: (v.astype(np.float32).copy(), _QUAT.copy()) for k, v in self.objects.items()}
        return Obs(t=self.t, images={}, state=state, ee_pos=self.ee.astype(np.float32).copy(), ee_quat=_QUAT.copy(),
                   gripper_qpos=q.copy(), object_poses=poses,
                   raw={"seed": self.seed, "held": self.held, "target": self.target,
                        "destination": self.destination, "perturbation": dict(self.perturbation)})
