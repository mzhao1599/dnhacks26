"""Scripted stand-in for the VLA on MockEnv. Implements the Policy protocol (chunks of k=5).

reach above target -> descend to obj_z + bias_z (+ a seed-dependent depth style) -> close -> lift ->
move over destination -> descend -> open. Deliberately imperfect so the S3 vector has something to fix:
  * slow: ~50% of max speed, hesitant (small steps) near each goal -> time_scale > 1 shortens episodes;
  * bias_z=+0.015 closes too high on a seed-stratified 55% of episodes ("shallow" style); the other
    45% descend ~9 mm deeper and succeed -> grasp_offset_z=-0.015 fixes the shallow ones;
  * out-of-distribution shrinkage: object positions further than OOD_RADIUS from the nominal pose are
    read with error (the policy trusts its training distribution) -> a 6 cm shift_xy breaks it;
  * instruction-insensitive except 'carefully' -> 30% slower;
  * a saturated wrist twitch at lift so force_proxy is nonzero.
The target/destination are bound from the instruction text (first/second object mentioned).
"""
from __future__ import annotations

import re

import numpy as np

from fleet_memory.envs.base import Obs
from fleet_memory.envs.mock_env import NOMINAL, STEP_M
from fleet_memory.policies.base import ActionChunk

ABOVE_M = 0.06          # pre-grasp height above the target
LIFT_Z = 0.15           # absolute carry height
PLACE_ABOVE_M = 0.03    # release height above the destination centre
HESITANT_R_M = 0.04     # inside this radius of a goal, use the hesitant speed
OOD_RADIUS_M = 0.03     # deviations from nominal beyond this are mis-perceived during fine alignment
OOD_GAIN = 1.2          # perceived error = OOD_GAIN * (deviation - OOD_RADIUS_M), toward the nominal pose
REACH_Z_TOL_M = 0.01    # the coarse approach ends at pre-grasp height ...
REACH_XY_TOL_M = 0.04   # ... once roughly above the target (a VLA-like coarse commit)
XY_CORRECTION = 0.5     # fraction of the (perceived) xy error corrected during the descent
STALL_ACTS = 3          # phase advances if the EE has not moved for this many act() calls


def _class_word(name: str) -> str:
    toks = [t for t in re.split(r"[_\s]+", name.lower()) if t and not t.isdigit()]
    return toks[-1] if toks else name


class MockPolicy:
    def __init__(self, name: str = "mock", bias_z: float = 0.015, seed: int = 0, k: int = 5,
                 speed: float = 0.5, hesitant_speed: float = 0.12, ood_gain: float = OOD_GAIN):
        self.name, self.bias_z, self.seed, self.k = name, float(bias_z), int(seed), int(k)
        self.speed, self.hesitant_speed, self.ood_gain = float(speed), float(hesitant_speed), float(ood_gain)
        self.instruction = ""
        self.phase: str | None = None
        self.acts = self.resets = 0
        self._delta_z = 0.0
        self._style = "shallow"
        self._last_ee: np.ndarray | None = None
        self._still = 0
        self._descend_xy = np.zeros(2)
        self._twitched = False

    # ------------------------------------------------------------------ protocol
    def reset(self, instruction: str) -> None:
        self.instruction = instruction or ""
        self.phase, self._last_ee, self._still, self.resets = None, None, 0, self.resets + 1

    def act(self, obs: Obs) -> ActionChunk:
        self.acts += 1
        ee = np.asarray(obs.ee_pos, np.float64)
        target, dest = self._bind(obs)
        tp_true, dp = np.asarray(obs.object_pos(target), np.float64), np.asarray(obs.object_pos(dest), np.float64)
        tp = self._estimate(obs, target)
        if self.phase is None:
            self._start_episode(obs, ee, tp_true)
        self._still = self._still + 1 if (self._last_ee is not None and np.linalg.norm(ee - self._last_ee) < 1e-5) else 0
        self._last_ee = ee.copy()
        stalled = self._stalled()

        if self.phase == "reach":                       # coarse approach from above (true pose); commits to the
            goal = tp_true + np.array([0.0, 0.0, ABOVE_M])   # descent at pre-grasp height, wherever it is in xy
            if (abs(ee[2] - goal[2]) < REACH_Z_TOL_M and np.linalg.norm((goal - ee)[:2]) < REACH_XY_TOL_M) or stalled:
                self._advance("descend")
                self._descend_xy = ee[:2] + XY_CORRECTION * (tp[:2] - ee[:2])
            else:
                return self._move(ee, goal, -1.0)
        if self.phase == "descend":                     # fine alignment: partial xy correction toward the estimate
            goal = np.array([self._descend_xy[0], self._descend_xy[1], tp_true[2] + self.bias_z + self._delta_z])
            if self._at(ee, goal, 0.002) or self._stalled():
                self._advance("close")
            else:
                return self._move(ee, goal, -1.0)
        if self.phase == "close":
            self._advance("lift")
            return self._hold(1.0)
        if self.phase == "lift":
            goal = np.array([ee[0], ee[1], LIFT_Z])
            if abs(ee[2] - LIFT_Z) < 0.01 or self._stalled():
                self._advance("move")
            else:
                return self._move(ee, goal, 1.0, twitch=not self._twitched)
        if self.phase == "move":
            goal = np.array([dp[0], dp[1], LIFT_Z])
            if self._at(ee, goal, 0.004) or self._stalled():
                self._advance("place")
            else:
                return self._move(ee, goal, 1.0)
        if self.phase == "place":
            goal = np.array([dp[0], dp[1], dp[2] + PLACE_ABOVE_M])
            if self._at(ee, goal, 0.004) or self._stalled():
                self._advance("open")
            else:
                return self._move(ee, goal, 1.0)
        if self.phase == "open":
            self.phase = "done"
        return self._hold(-1.0)

    # ------------------------------------------------------------------ internals
    def _stalled(self) -> bool:
        return self._still >= STALL_ACTS

    def _advance(self, phase: str) -> None:
        self.phase, self._still = phase, 0

    def _start_episode(self, obs: Obs, ee: np.ndarray, tp: np.ndarray) -> None:
        seed = (obs.raw or {}).get("seed")
        if seed is None:
            seed = int(abs(np.sum(np.round(np.asarray(tp) * 1e4))))
        rng = np.random.default_rng(int(seed) * 1000003 + self.seed)
        # low-discrepancy split: any block of consecutive seeds is ~45% "deep" (golden-ratio hashing)
        self._style = "deep" if (((int(seed) + self.seed) * 0.6180339887) % 1.0) < 0.45 else "shallow"
        self._delta_z = (-0.009 if self._style == "deep" else 0.004) + float(rng.uniform(-0.003, 0.003))
        self._twitched = False
        q = np.asarray(obs.gripper_qpos, np.float64).ravel()
        closed = q.size >= 2 and abs(q[0] - q[1]) < 0.03
        holding = closed and float(np.linalg.norm(np.asarray(obs.object_pos(self._bind(obs)[0]), np.float64) - ee)) < 0.03
        self.phase = ("move" if ee[2] > LIFT_Z - 0.02 else "lift") if holding else "reach"

    def _bind(self, obs: Obs) -> tuple[str, str]:
        names = list(obs.object_poses)
        lang = self.instruction.lower()
        hits = sorted((pos, i, n) for i, n in enumerate(names)
                      if (pos := max(lang.find(n.replace("_", " ")), lang.find(_class_word(n)))) >= 0)
        raw = obs.raw or {}
        target = hits[0][2] if hits else (raw.get("target") if raw.get("target") in names else names[0])
        rest = [n for n in names if n != target]
        dest = hits[1][2] if len(hits) > 1 else (raw.get("destination") if raw.get("destination") in names else rest[0])
        return target, dest

    def _estimate(self, obs: Obs, name: str) -> np.ndarray:
        """Perceived position: exact inside the training distribution, shrunk toward nominal outside it."""
        p = np.asarray(obs.object_pos(name), np.float64)
        nom = NOMINAL.get(name) or NOMINAL.get(re.sub(r"_\d+$", "", name))
        if nom is None or self.ood_gain <= 0:
            return p
        dev = np.asarray(nom, np.float64)[:2] - p[:2]
        d = float(np.linalg.norm(dev))
        if d <= OOD_RADIUS_M:
            return p
        est = p.copy()
        est[:2] += self.ood_gain * (d - OOD_RADIUS_M) * dev / d
        return est

    def _vmax(self, err_norm: float) -> float:
        v = self.hesitant_speed if err_norm < HESITANT_R_M else self.speed
        return v * (0.7 if "carefully" in self.instruction.lower() else 1.0)

    @staticmethod
    def _at(ee: np.ndarray, goal: np.ndarray, tol: float) -> bool:
        return float(np.linalg.norm(goal - ee)) < tol

    def _move(self, ee: np.ndarray, goal: np.ndarray, gripper: float, twitch: bool = False) -> ActionChunk:
        a = np.zeros((self.k, 7), np.float32)
        a[:, 6] = gripper
        sim = ee.copy()
        for i in range(self.k):
            err = goal - sim
            n = float(np.linalg.norm(err))
            if n < 1e-4:
                break
            step = err / STEP_M                                     # gain 1: one step reaches the goal
            m = float(np.linalg.norm(step))
            if m > self._vmax(n):
                step = step * (self._vmax(n) / m)                   # direction-preserving speed limit
            a[i, :3] = step
            sim = sim + step * STEP_M
        if twitch:
            a[:, 3:6] = np.array([1.0, -1.0, 1.0], np.float32)   # saturated wrist twitch (env ignores rot)
            self._twitched = True
        return a

    def _hold(self, gripper: float) -> ActionChunk:
        a = np.zeros((self.k, 7), np.float32)
        a[:, 6] = gripper
        return a
