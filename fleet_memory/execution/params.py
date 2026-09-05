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
]
DIM = sum(d for _, d, _, _ in PARAM_SPEC)  # 9
NAMES: list[str] = [n for n, _, _, _ in PARAM_SPEC]
LOW = np.concatenate([np.full(d, lo, np.float64) for _, d, lo, _ in PARAM_SPEC])
HIGH = np.concatenate([np.full(d, hi, np.float64) for _, d, _, hi in PARAM_SPEC])
RANGE = HIGH - LOW

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

    # --- vector view ---------------------------------------------------------
    def to_array(self) -> np.ndarray:
        return np.concatenate([np.asarray(self.approach_offset_xyz, np.float64).reshape(3),
                               [self.pregrasp_height, self.grasp_offset_z, self.time_scale,
                                self.velocity_cap, self.gripper_cmd, self.approach_cone_deg]])

    @classmethod
    def from_array(cls, x: np.ndarray) -> "S3Params":
        x = clip(np.asarray(x, np.float64).reshape(DIM))
        return cls(approach_offset_xyz=x[0:3].copy(), pregrasp_height=float(x[3]), grasp_offset_z=float(x[4]),
                   time_scale=float(x[5]), velocity_cap=float(x[6]), gripper_cmd=float(x[7]),
                   approach_cone_deg=float(x[8]))

    # --- dict view (what the log stores) -------------------------------------
    def to_dict(self) -> dict[str, float | list[float]]:
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["approach_offset_xyz"] = [float(v) for v in np.asarray(self.approach_offset_xyz).reshape(3)]
        return {k: (v if isinstance(v, list) else float(v)) for k, v in d.items()}

    @classmethod
    def from_dict(cls, d: dict | None) -> "S3Params":
        if not d:
            return cls()
        p = cls()
        for f in fields(cls):
            if f.name in d and d[f.name] is not None:
                setattr(p, f.name, np.asarray(d[f.name], np.float64) if f.name == "approach_offset_xyz" else float(d[f.name]))
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
