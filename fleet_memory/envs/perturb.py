"""Post-reset perturbations of a LIBERO initial state (robustness probes).

Applied *after* ``env.reset(seed)`` by editing the MuJoCo state through the robosuite sim,
then re-settling for a few no-op steps. Deterministic given (seed, perturbation params).

Dict form (what ``LiberoEnv.reset(seed, perturbation=...)`` takes; keys are optional, combinable):

    {"shift_xy": [dx, dy]}   deterministic xy shift (metres) of the FIRST object of interest
    {"jitter_m": s}          uniform xy jitter (± s metres) of ALL objects of interest
    {"distractor": true}     best-effort: move one non-target free object ~7 cm beside the target
    {"seed": k}              rng seed for jitter/distractor (default: the episode seed)

* objects of interest = ``env.obj_of_interest`` from the BDDL (target objects first, container last);
  falls back to every free-joint object if the BDDL has no obj_of_interest. Fixtures (no free joint)
  are never moved.
* distractor: we cannot add bodies to a compiled MuJoCo model without rebuilding it, so
  "distractor" moves one *non-target* free object to ~7 cm beside the target, which is the
  clutter case that matters for grasp selection. Placement is retried (up to 8 angles) so the
  distractor does not overlap another free object (>5 cm from every other object).
* Objects are lifted 1 mm off their resting height when moved so the settle steps drop them onto the
  table instead of starting inside it.

The ``Perturbation`` dataclass is the older object form; it now delegates to ``apply_perturbation``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fleet_memory.envs.base import Obs

SETTLE_STEPS = 5
DISTRACTOR_RADIUS_M = 0.07
LIFT_M = 0.001


def apply_perturbation(env, perturbation: dict[str, Any] | None, default_seed: int = 0) -> tuple[Obs, dict[str, Any]]:
    """Mutate the sim state of a LiberoEnv that was just reset(); return (new Obs, applied dict).

    ``applied`` records, per moved object, the joint xyz before and after the edit + settle,
    e.g. {"shift_xy:alphabet_soup_1": {"before": [...], "target": [...], "after": [...]}}.
    """
    p = dict(perturbation or {})
    shift = p.get("shift_xy")
    jitter = float(p.get("jitter_m", 0.0) or 0.0)
    distractor = bool(p.get("distractor", False))
    if not shift and jitter <= 0 and not distractor:
        return env.current_obs(), {}

    rng = np.random.default_rng(int(p.get("seed", default_seed)))
    lib = env.raw_env            # libero OffScreenRenderEnv
    pe = lib.env                 # robosuite problem env
    sim = pe.sim
    free = _free_objects(pe)     # name -> free joint name
    targets = [o for o in pe.obj_of_interest if o in free] or list(free)
    applied: dict[str, Any] = {}
    moved: dict[str, np.ndarray] = {}   # name -> qpos before

    def _get(name: str) -> np.ndarray:
        return np.array(sim.data.get_joint_qpos(free[name]), dtype=np.float64)

    def _set(name: str, q: np.ndarray) -> None:
        q = np.array(q, dtype=np.float64)
        q[2] += LIFT_M
        sim.data.set_joint_qpos(free[name], q)

    if shift and targets:
        dx, dy = float(shift[0]), float(shift[1])
        name = targets[0]
        q0 = _get(name)
        q = q0.copy()
        q[0] += dx
        q[1] += dy
        _set(name, q)
        moved.setdefault(name, q0)
        applied[f"shift_xy:{name}"] = {"before": q0[:3].round(4).tolist(), "target": q[:3].round(4).tolist()}

    if jitter > 0:
        for name in targets:
            d = rng.uniform(-jitter, jitter, size=2).astype(np.float64)
            q0 = _get(name)
            q = q0.copy()
            q[0] += d[0]
            q[1] += d[1]
            _set(name, q)
            moved.setdefault(name, q0)
            applied[f"jitter:{name}"] = {"before": q0[:3].round(4).tolist(), "target": q[:3].round(4).tolist(),
                                         "delta": d.round(4).tolist()}

    if distractor and targets:
        tgt = targets[0]
        others = [n for n in free if n not in targets]
        if others:
            dn = others[int(rng.integers(len(others)))]
            tq = _get(tgt)
            dq0 = _get(dn)
            dq = dq0.copy()
            base_ang = rng.uniform(0, 2 * np.pi)
            placed = False
            for k in range(8):
                ang = base_ang + k * (np.pi / 4)
                cand = np.array([tq[0] + DISTRACTOR_RADIUS_M * np.cos(ang), tq[1] + DISTRACTOR_RADIUS_M * np.sin(ang)])
                ok = True
                for n in free:
                    if n in (dn, tgt):
                        continue
                    oq = _get(n)
                    if np.linalg.norm(oq[:2] - cand) < 0.05:
                        ok = False
                        break
                if ok:
                    dq[0], dq[1] = cand
                    placed = True
                    break
            if placed:
                # Drop from just above the target's height (shelf objects would otherwise fall 40+ cm).
                dq[2] = tq[2] + 0.03
                _set(dn, dq)
                moved.setdefault(dn, dq0)
                applied[f"distractor:{dn}"] = {"before": dq0[:3].round(4).tolist(), "target": dq[:3].round(4).tolist(),
                                               "near": tgt}
            else:
                applied["distractor"] = {"skipped": "no collision-free slot", "near": tgt}
        else:
            applied["distractor"] = {"skipped": "no non-target free object"}

    sim.forward()
    from fleet_memory.envs.libero_env import DUMMY_ACTION
    raw = None
    for _ in range(SETTLE_STEPS):
        raw, _, _, _ = lib.step(DUMMY_ACTION.copy())
    for name, q0 in moved.items():
        q1 = _get(name)
        key = next((k for k in applied if k.endswith(":" + name)), None)
        if key is not None:
            applied[key]["after"] = q1[:3].round(4).tolist()
            applied[key]["dz_settle"] = round(float(q1[2] - q0[2]), 4)
    env._t = 0
    obs = env._make_obs(raw)
    env._last_obs = obs
    return obs, applied


@dataclass
class Perturbation:
    distractor: bool = False
    pose_jitter_m: float = 0.0
    seed: int = 0
    shift_xy: list[float] | None = None
    applied: dict[str, Any] = field(default_factory=dict)

    def is_identity(self) -> bool:
        return not self.distractor and self.pose_jitter_m <= 0.0 and not self.shift_xy

    def to_dict(self) -> dict[str, Any]:
        return {"distractor": self.distractor, "pose_jitter_m": self.pose_jitter_m, "seed": self.seed,
                "shift_xy": self.shift_xy, "applied": dict(self.applied)}

    def as_spec(self) -> dict[str, Any]:
        d: dict[str, Any] = {"seed": self.seed}
        if self.shift_xy:
            d["shift_xy"] = list(self.shift_xy)
        if self.pose_jitter_m > 0:
            d["jitter_m"] = self.pose_jitter_m
        if self.distractor:
            d["distractor"] = True
        return d

    # ------------------------------------------------------------------ #
    def apply(self, env) -> Obs:
        """Mutate the sim state of a LiberoEnv that was just reset(); return the new Obs."""
        obs, self.applied = apply_perturbation(env, self.as_spec(), default_seed=self.seed)
        return obs


def _free_objects(pe) -> dict[str, str]:
    """name -> free-joint name for movable BDDL objects (fixtures have no free joint)."""
    out = {}
    for name, obj in pe.objects_dict.items():
        joints = getattr(obj, "joints", None) or []
        for j in joints:
            try:
                if len(np.atleast_1d(pe.sim.data.get_joint_qpos(j))) == 7:   # free joint: pos(3)+quat(4)
                    out[name] = j
                    break
            except Exception:
                continue
    return out


__all__ = ["apply_perturbation", "Perturbation", "SETTLE_STEPS"]
