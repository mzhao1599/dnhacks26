"""v3 S3 control surface as ONE bounded continuous vector (9 dims). All optimisation happens over
this and nothing else. Bounds are hard; the shim clips. The optimizer writes these; the coach may
only propose them as candidates that go through the same perturbation gate.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np

from fleet_memory.execution.constraints import ConstraintSet

# name -> (dims, low, high). Order here IS the vector order.
PARAM_SPEC: list[tuple[str, int, float, float]] = [
    ("approach_offset_xyz", 3, -0.03, 0.03),   # m, added to the pre-grasp target pose
    ("pregrasp_height",     1,  0.02, 0.10),   # m, height of the inserted pre-grasp waypoint above the grasp target
    ("grasp_offset_z",      1, -0.02, 0.02),   # m, added to grasp target z
    ("time_scale",          1,  0.60, 1.50),   # action chunk resampled in time by this factor (>1 = faster)
    ("velocity_cap",        1,  0.30, 1.00),   # fraction of max delta-pose per step; shim clips
    ("gripper_cmd",         1,  0.30, 1.00),   # gripper close command scale
    ("approach_cone_deg",   1, 10.0, 60.0),    # half-angle; approach-phase actions leaving the cone are projected back
    ("homing_enable",       1,  0.0, 1.0),     # optimised as continuous, thresholded at 0.5: home the arm before the VLA acts
    ("homing_pos_delta",    3, -0.05, 0.05),   # m, offset from the canonical LIBERO start EE position (homing target)
    ("homing_rot_delta",    3, -0.15, 0.15),   # rad axis-angle offset from the canonical start orientation
]
DIM = sum(d for _, d, _, _ in PARAM_SPEC)  # 16
NAMES: list[str] = [n for n, _, _, _ in PARAM_SPEC]
LOW = np.concatenate([np.full(d, lo, np.float64) for _, d, lo, _ in PARAM_SPEC])
HIGH = np.concatenate([np.full(d, hi, np.float64) for _, d, _, hi in PARAM_SPEC])
RANGE = HIGH - LOW

VEC_FIELDS = ("approach_offset_xyz", "homing_pos_delta", "homing_rot_delta")
APPROACH_RADIUS_M = 0.12   # phase detector: approach = gripper open AND ||ee - target|| < this
BLEND_ALPHA = 0.5          # a = (1-α)·a_vla + α·a_toward_waypoint during approach only


@dataclass
class S3Params:
    approach_offset_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    pregrasp_height: float = 0.05
    grasp_offset_z: float = 0.0
    time_scale: float = 1.0
    velocity_cap: float = 1.0
    gripper_cmd: float = 1.0
    approach_cone_deg: float = 45.0
    homing_enable: float = 0.0
    homing_pos_delta: np.ndarray = field(default_factory=lambda: np.zeros(3))
    homing_rot_delta: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # --- vector view ---------------------------------------------------------
    def to_array(self) -> np.ndarray:
        return np.concatenate([np.asarray(self.approach_offset_xyz, np.float64).reshape(3),
                               [self.pregrasp_height, self.grasp_offset_z, self.time_scale,
                                self.velocity_cap, self.gripper_cmd, self.approach_cone_deg, self.homing_enable],
                               np.asarray(self.homing_pos_delta, np.float64).reshape(3),
                               np.asarray(self.homing_rot_delta, np.float64).reshape(3)])

    @classmethod
    def from_array(cls, x: np.ndarray) -> "S3Params":
        x = np.asarray(x, np.float64).reshape(-1)
        if x.shape[0] == 9:                      # v3.0 vectors (pre-homing) stay loadable
            x = np.concatenate([x, np.zeros(DIM - 9)])
        x = clip(x.reshape(DIM))
        return cls(approach_offset_xyz=x[0:3].copy(), pregrasp_height=float(x[3]), grasp_offset_z=float(x[4]),
                   time_scale=float(x[5]), velocity_cap=float(x[6]), gripper_cmd=float(x[7]),
                   approach_cone_deg=float(x[8]), homing_enable=float(x[9]),
                   homing_pos_delta=x[10:13].copy(), homing_rot_delta=x[13:16].copy())

    # --- homing (executed by the runner before the policy loop, not by the shim) -----------------
    @property
    def homing_on(self) -> bool:
        return self.homing_enable >= 0.5

    def homing_delta(self) -> np.ndarray:
        """6-D [dx, dy, dz, drx, dry, drz] for execution/homing.HomingTarget.canonical(delta)."""
        return np.concatenate([np.asarray(self.homing_pos_delta, np.float64).reshape(3),
                               np.asarray(self.homing_rot_delta, np.float64).reshape(3)])

    # --- dict view (what the log stores) -------------------------------------
    def to_dict(self) -> dict[str, float | list[float]]:
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        for k in VEC_FIELDS:
            d[k] = [float(v) for v in np.asarray(d[k]).reshape(3)]
        return {k: (v if isinstance(v, list) else float(v)) for k, v in d.items()}

    @classmethod
    def from_dict(cls, d: dict | None) -> "S3Params":
        if not d:
            return cls()
        p = cls()
        for f in fields(cls):
            if f.name in d and d[f.name] is not None:
                setattr(p, f.name, np.asarray(d[f.name], np.float64) if f.name in VEC_FIELDS else float(d[f.name]))
        return cls.from_array(p.to_array())   # re-clip

    @classmethod
    def identity(cls) -> "S3Params":
        return cls()

    # --- bridge to the v2 shim -----------------------------------------------
    def to_constraint_set(self, base: ConstraintSet | None = None) -> ConstraintSet:
        """Express the vector as the ConstraintSet the execution shim already enforces."""
        c = ConstraintSet.from_dict(base.to_dict() if base is not None else None)
        off = np.asarray(self.approach_offset_xyz, np.float32).reshape(3)
        c.pre_grasp_waypoint = off + np.array([0.0, 0.0, self.pregrasp_height], np.float32)
        c.grasp_offset = np.array([0.0, 0.0, self.grasp_offset_z], np.float32)
        c.max_pos_delta = float(self.velocity_cap)
        c.max_rot_delta = float(self.velocity_cap)
        c.gripper_close_value = float(self.gripper_cmd)
        c.approach_vector = np.array([0.0, 0.0, -1.0], np.float32)
        c.approach_half_angle_deg = float(self.approach_cone_deg)
        c.time_scale = float(self.time_scale)
        c.applied_edits = list(c.applied_edits) + [{"op": "set_s3_params", "params": self.to_dict()}]
        return c


def clip(x: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(np.asarray(x, np.float64), LOW), HIGH)


def normalize(x: np.ndarray) -> np.ndarray:
    """[low, high] -> [0, 1] per dim (the optimizer searches in this space)."""
    return (clip(x) - LOW) / RANGE


def denormalize(u: np.ndarray) -> np.ndarray:
    return clip(LOW + np.clip(np.asarray(u, np.float64), 0.0, 1.0) * RANGE)
