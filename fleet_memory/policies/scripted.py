"""IK-free scripted pick-and-place behind the Policy protocol: the fallback for LiberoEnv when no
VLA is available (and a sanity baseline on MockEnv). Reads obs.object_poses only; the worker sets
the target/destination from the plan's first subtask via set_targets(), otherwise they are guessed
from the instruction. OSC delta-pose proportional controller: dpos = clip(err_m / pos_scale_m, -1, 1)
(1.0 == 0.05 m on LIBERO). Phases: reach above (z+hover) -> descend -> close (8 steps) -> lift ->
transit above destination -> descend -> open -> retreat. Chunks of k rows; phase transitions are
decided on the real observation at chunk start, an in-chunk EE estimate only shortens chunks.
"""
from __future__ import annotations

import re

import numpy as np

from fleet_memory.envs.base import Obs
from fleet_memory.policies.base import ActionChunk, clip_action

PHASES = ("reach", "descend", "close", "lift", "transit", "place", "open", "retreat", "done")
OPEN, CLOSE = -1.0, 1.0
STALL_M_PER_ROW = 0.0005   # descend/place: EE moving less than this per executed row = contact -> next phase


class ScriptedPolicy:
    name = "scripted"

    def __init__(self, name: str = "scripted", k: int = 5, pos_scale_m: float = 0.05, hover_m: float = 0.08,
                 grasp_dz_m: float = 0.004, lift_m: float = 0.10, place_dz_m: float = 0.01,
                 close_steps: int = 8, open_steps: int = 4, max_phase_steps: int = 80, max_regrasps: int = 1):
        self.name = name
        self.k = int(k)
        self.pos_scale_m = float(pos_scale_m)
        self.hover_m, self.grasp_dz_m, self.lift_m, self.place_dz_m = hover_m, grasp_dz_m, lift_m, place_dz_m
        self.close_steps, self.open_steps = int(close_steps), int(open_steps)
        self.max_phase_steps, self.max_regrasps = int(max_phase_steps), int(max_regrasps)
        self.target: str | None = None
        self.destination: str | None = None
        self.instruction = ""
        self.resets = 0
        self.reset("")

    # ------------------------------------------------------------------ Policy protocol
    def reset(self, instruction: str) -> None:
        self.instruction = instruction or ""
        self.resets += 1
        self.phase = "reach"
        self._phase_steps = 0
        self._grip_steps = 0
        self._regrasps = 0
        self._grasp_ee: np.ndarray | None = None      # EE position when the close began
        self._grasp_obj_z: float | None = None        # target z when the close began (lift check)
        self._last_ee: np.ndarray | None = None       # EE at the previous act() and rows emitted then
        self._last_rows = 0

    def act(self, obs: Obs) -> ActionChunk:
        tgt, dest = self._resolve(obs)
        ee = np.asarray(obs.ee_pos, dtype=float)
        if tgt is None:
            return self._rows([np.zeros(7, np.float32)])
        self._advance(obs, ee, tgt, dest)
        rows: list[np.ndarray] = []
        est = ee.copy()
        while len(rows) < self.k:
            if self.phase in ("close", "open"):
                rows.append(self._row(np.zeros(3), CLOSE if self.phase == "close" else OPEN))
                self._grip_steps += 1
                if self._grip_steps >= (self.close_steps if self.phase == "close" else self.open_steps):
                    self._set_phase("lift" if self.phase == "close" else "retreat")
                    if self.phase == "retreat":      # let the next act() see the release
                        break
                continue
            if self.phase == "done":
                rows.append(self._row(np.zeros(3), OPEN))
                continue
            goal, tol, grip = self._goal(ee, tgt, dest)
            err = goal - est
            if rows and float(np.linalg.norm(err)) <= tol:
                break                                  # estimate says arrived: hand back for a real look
            dpos = np.clip(err / self.pos_scale_m, -1.0, 1.0)
            rows.append(self._row(dpos, grip))
            est = est + dpos * self.pos_scale_m
        self._phase_steps += len(rows)
        self._last_ee, self._last_rows = ee.copy(), len(rows)
        return self._rows(rows)

    def set_targets(self, target: str | None, destination: str | None = None) -> None:
        self.target, self.destination = target, destination

    # ------------------------------------------------------------------ internals
    def _set_phase(self, phase: str) -> None:
        self.phase, self._phase_steps, self._grip_steps = phase, 0, 0

    def _goal(self, ee: np.ndarray, tgt: np.ndarray, dest: np.ndarray | None) -> tuple[np.ndarray, float, float]:
        """(goal position, tolerance, gripper) for the motion phases."""
        up = np.array([0.0, 0.0, 1.0])
        if self.phase == "reach":
            return tgt + up * self.hover_m, 0.01, OPEN
        if self.phase == "descend":
            return tgt + up * self.grasp_dz_m, 0.005, OPEN
        g = self._grasp_ee if self._grasp_ee is not None else ee
        if self.phase == "lift":
            return np.array([g[0], g[1], g[2] + self.lift_m]), 0.01, CLOSE
        d = dest if dest is not None else tgt
        if self.phase == "transit":
            z = max(g[2] + self.lift_m, float(d[2]) + self.hover_m + self.place_dz_m)
            return np.array([d[0], d[1], z]), 0.01, CLOSE
        if self.phase == "place":
            return d + up * (self.place_dz_m + self.grasp_dz_m), 0.008, CLOSE
        return np.array([ee[0], ee[1], float(d[2]) + self.hover_m]), 0.01, OPEN   # retreat

    def _advance(self, obs: Obs, ee: np.ndarray, tgt: np.ndarray, dest: np.ndarray | None) -> None:
        """Phase transitions on the real observation."""
        if self.phase in ("close", "open", "done"):
            return
        if self._phase_steps >= self.max_phase_steps:      # stuck (e.g. goal outside the envelope): move on
            self._set_phase(PHASES[PHASES.index(self.phase) + 1])
            return
        goal, tol, _ = self._goal(ee, tgt, dest)
        stalled = (self.phase in ("descend", "place") and self._last_ee is not None and self._last_rows >= 2
                   and float(np.linalg.norm(ee - self._last_ee)) < STALL_M_PER_ROW * self._last_rows)
        if float(np.linalg.norm(goal - ee)) > tol and not stalled:
            return
        if self.phase == "reach":
            self._set_phase("descend")
        elif self.phase == "descend":
            self._grasp_ee, self._grasp_obj_z = ee.copy(), float(tgt[2])
            self._set_phase("close")
        elif self.phase == "lift":
            lifted = self._grasp_obj_z is None or float(tgt[2]) > self._grasp_obj_z + 0.02
            if not lifted and self._regrasps < self.max_regrasps:
                self._regrasps += 1
                self._set_phase("reach")
            else:
                self._set_phase("transit")
        elif self.phase == "transit":
            self._set_phase("place")
        elif self.phase == "place":
            self._set_phase("open")
        elif self.phase == "retreat":
            self._set_phase("done")

    def _resolve(self, obs: Obs) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self.target is None or self.target not in obs.object_poses:
            t, d = guess_targets(self.instruction, list(obs.object_poses))
            self.target = t if self.target is None else self.target
            self.destination = d if self.destination is None else self.destination
        tgt = obs.object_pos(self.target) if self.target else None
        dest = obs.object_pos(self.destination) if self.destination else None
        return (None if tgt is None else np.asarray(tgt, dtype=float),
                None if dest is None else np.asarray(dest, dtype=float))

    @staticmethod
    def _row(dpos: np.ndarray, gripper: float) -> np.ndarray:
        a = np.zeros(7, np.float32)
        a[:3] = dpos
        a[6] = gripper
        return a

    @staticmethod
    def _rows(rows: list[np.ndarray]) -> ActionChunk:
        return clip_action(np.stack(rows)).reshape(-1, 7)


def guess_targets(instruction: str, objects: list[str]) -> tuple[str | None, str | None]:
    """Order the scene objects by where their name tokens first appear in the instruction:
    first match = target, second = destination. Objects not mentioned come last; with no
    instruction match at all the first two scene objects are used."""
    text = " " + re.sub(r"[^a-z0-9 ]", " ", instruction.lower()) + " "
    scored: list[tuple[int, int, str]] = []
    for i, name in enumerate(objects):
        toks = [t for t in re.sub(r"_\d+$", "", name.lower()).split("_") if t and not t.isdigit()]
        hits = [text.find(" " + t + " ") for t in toks]
        hits = [h for h in hits if h >= 0]
        scored.append((min(hits) if hits else 10 ** 6, i, name))
    scored.sort()
    names = [n for _, _, n in scored]
    return (names[0] if names else None), (names[1] if len(names) > 1 else None)
