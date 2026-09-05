"""Post-reset perturbations of a LIBERO initial state (robustness probes).

Applied *after* ``env.reset(seed)`` by editing the MuJoCo state through the robosuite sim,
then re-settling for a few no-op steps. Deterministic given (seed, perturbation params).

    pert = Perturbation(distractor=False, pose_jitter_m=0.02, seed=123)
    obs = env.reset(seed)
    obs = pert.apply(env)            # -> new Obs

* pose_jitter_m: uniform xy jitter (± value, metres) of the target object(s) = env.obj_of_interest
  (falls back to every free-joint object if the BDDL has no obj_of_interest).
* distractor: we cannot add bodies to a compiled MuJoCo model without rebuilding it, so
  "distractor" moves one *non-target* free object to ~7 cm beside the target, which is the
  clutter case that matters for grasp selection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fleet_memory.envs.base import Obs

SETTLE_STEPS = 5


@dataclass
class Perturbation:
    distractor: bool = False
    pose_jitter_m: float = 0.0
    seed: int = 0
    applied: dict[str, Any] = field(default_factory=dict)

    def is_identity(self) -> bool:
        return not self.distractor and self.pose_jitter_m <= 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"distractor": self.distractor, "pose_jitter_m": self.pose_jitter_m, "seed": self.seed,
                "applied": dict(self.applied)}

    # ------------------------------------------------------------------ #
    def apply(self, env) -> Obs:
        """Mutate the sim state of a LiberoEnv that was just reset(); return the new Obs."""
        if self.is_identity():
            return env.current_obs() if hasattr(env, "current_obs") else None
        rng = np.random.default_rng(self.seed)
        lib = env.raw_env            # libero OffScreenRenderEnv
        pe = lib.env                 # robosuite problem env
        sim = pe.sim
        free = _free_objects(pe)     # name -> free joint name
        targets = [o for o in pe.obj_of_interest if o in free] or list(free)
        self.applied = {}

        if self.pose_jitter_m > 0:
            for name in targets:
                d = rng.uniform(-self.pose_jitter_m, self.pose_jitter_m, size=2).astype(np.float64)
                q = np.array(sim.data.get_joint_qpos(free[name]), dtype=np.float64)
                q[0] += d[0]
                q[1] += d[1]
                sim.data.set_joint_qpos(free[name], q)
                self.applied[f"jitter:{name}"] = d.tolist()

        if self.distractor and targets:
            tgt = targets[0]
            others = [n for n in free if n not in targets]
            if others:
                dn = others[int(rng.integers(len(others)))]
                tq = np.array(sim.data.get_joint_qpos(free[tgt]), dtype=np.float64)
                dq = np.array(sim.data.get_joint_qpos(free[dn]), dtype=np.float64)
                ang = rng.uniform(0, 2 * np.pi)
                dq[0] = tq[0] + 0.07 * np.cos(ang)
                dq[1] = tq[1] + 0.07 * np.sin(ang)
                dq[2] = max(dq[2], tq[2])
                sim.data.set_joint_qpos(free[dn], dq)
                self.applied[f"distractor:{dn}"] = dq[:3].tolist()

        sim.forward()
        from fleet_memory.envs.libero_env import DUMMY_ACTION
        raw = None
        for _ in range(SETTLE_STEPS):
            raw, _, _, _ = lib.step(DUMMY_ACTION.copy())
        env._t = 0
        obs = env._make_obs(raw)
        env._last_obs = obs
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
