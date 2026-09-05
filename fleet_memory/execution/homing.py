"""Homing: before the VLA gets control, drive the end-effector back to the canonical LIBERO start pose
(+ an optimizable delta) with a scripted controller. This is the S3 edit that targets the LIBERO-Plus
"robot initial state" collapse: the VLA memorised trajectories from one start distribution, so we put
it back there first.

Deviation from the spec (which says joint-space): LIBERO's action interface is OSC delta-pose and the
VLA's proprio input is EE pose (pos + axis-angle + gripper), not joints. Homing therefore runs in EE
space — same mechanism, and it restores exactly the state the policy conditions on. Homing steps count
toward the episode's steps (and therefore its cost): homing is not free.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fleet_memory.envs.base import Env, Obs
from fleet_memory.execution.envelope import DEFAULT_ENVELOPE, POS_SCALE_M, Envelope

ROT_SCALE_RAD = 0.5   # |drot| = 1 -> 0.5 rad per step (robosuite OSC_POSE default)

# FALLBACK ONLY (envs provide canonical_ee_pose()). Measured at an unperturbed reset of the current LiberoEnv
# (libero_10 task 3 / libero_spatial task 0 agree to ~1 cm: the arm starts the same way in every suite).
# An earlier constant from a pre-rewrite adapter said z=0.70 and sent the arm 47 cm into the table — hence
# the rule: never home to a constant when the env can tell you where it started.
CANONICAL_EE_POS = np.array([-0.2171, -0.0148, 1.1700], np.float32)
CANONICAL_EE_QUAT_XYZW = np.array([0.9996, -0.0011, -0.0293, -0.0002], np.float32)


def quat_to_rotvec(q_xyzw: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R
    return R.from_quat(np.asarray(q_xyzw, np.float64)).as_rotvec()


def rotation_error(q_cur_xyzw: np.ndarray, q_tgt_xyzw: np.ndarray) -> np.ndarray:
    """Axis-angle (rad) that rotates current into target, expressed in the world frame."""
    from scipy.spatial.transform import Rotation as R
    r_cur = R.from_quat(np.asarray(q_cur_xyzw, np.float64))
    r_tgt = R.from_quat(np.asarray(q_tgt_xyzw, np.float64))
    return (r_tgt * r_cur.inv()).as_rotvec()


@dataclass
class HomingTarget:
    pos: np.ndarray                          # (3,) world
    quat_xyzw: np.ndarray                    # (4,)

    @classmethod
    def canonical(cls, delta: np.ndarray | None = None, env=None) -> "HomingTarget":
        """Canonical start pose plus an optional 6-D delta [dx, dy, dz, drx, dry, drz] (m, rad).
        The pose comes from env.canonical_ee_pose() when the env provides it (table heights differ per
        LIBERO suite: LIBERO-10 starts at z~0.70, LIBERO-Spatial at z~1.17); else the LIBERO-10 constant."""
        from scipy.spatial.transform import Rotation as R
        base_pos, base_quat = CANONICAL_EE_POS, CANONICAL_EE_QUAT_XYZW
        if env is not None and hasattr(env, "canonical_ee_pose"):
            base_pos, base_quat = env.canonical_ee_pose()
        d = np.zeros(6) if delta is None else np.asarray(delta, np.float64).reshape(6)
        pos = np.asarray(base_pos, np.float64) + d[:3]
        quat = (R.from_rotvec(d[3:]) * R.from_quat(np.asarray(base_quat, np.float64))).as_quat()
        return cls(pos=pos.astype(np.float32), quat_xyzw=quat.astype(np.float32))


@dataclass
class HomingResult:
    steps: int
    reached: bool
    final_pos_err_m: float
    final_rot_err_rad: float
    ee_positions: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)


class Homing:
    """Proportional EE-space controller. Gripper stays open. Every action passes the envelope."""

    def __init__(self, target: HomingTarget, envelope: Envelope | None = None, k_pos: float = 0.6,
                 k_rot: float = 0.6, tol_pos_m: float = 0.01, tol_rot_rad: float = 0.05, max_steps: int = 40):
        self.target, self.envelope = target, envelope or DEFAULT_ENVELOPE
        self.k_pos, self.k_rot = k_pos, k_rot
        self.tol_pos, self.tol_rot, self.max_steps = tol_pos_m, tol_rot_rad, max_steps

    def action_for(self, obs: Obs) -> tuple[np.ndarray, float, float]:
        pos_err = self.target.pos - np.asarray(obs.ee_pos, np.float64)
        rot_err = rotation_error(obs.ee_quat, self.target.quat_xyzw)
        a = np.zeros(7, np.float32)
        a[0:3] = np.clip(self.k_pos * pos_err / POS_SCALE_M, -1.0, 1.0)
        a[3:6] = np.clip(self.k_rot * rot_err / ROT_SCALE_RAD, -1.0, 1.0)
        a[6] = -1.0
        a, _ = self.envelope.clamp(a, obs.ee_pos)
        return a, float(np.linalg.norm(pos_err)), float(np.linalg.norm(rot_err))

    def run(self, env: Env, obs: Obs) -> tuple[Obs, HomingResult, bool]:
        """Step the env until within tolerance or max_steps. Returns (obs, result, done)."""
        res = HomingResult(steps=0, reached=False, final_pos_err_m=0.0, final_rot_err_rad=0.0)
        done = False
        for _ in range(self.max_steps):
            a, pe, re = self.action_for(obs)
            res.final_pos_err_m, res.final_rot_err_rad = pe, re
            if pe < self.tol_pos and re < self.tol_rot:
                res.reached = True
                break
            res.ee_positions.append(np.asarray(obs.ee_pos, np.float32).copy())
            res.actions.append(a.copy())
            obs, done, _ = env.step(a)
            res.steps += 1
            if done:
                break
        return obs, res, done
