"""S3 control surface: the constraint set the execution shim enforces.

An S3 lesson's `edit` is applied to a ConstraintSet via `apply_edit`. The shim reads the
ConstraintSet every step. Everything here is plain data + a few pure functions.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from fleet_memory.memory.schema import EDIT_OPS, Edit


@dataclass
class ConstraintSet:
    # --- grasp geometry (world frame, metres, relative to target object position) ---
    grasp_offset: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    # --- approach cone: unit vector the EE should travel along when near the target; None = off ---
    approach_vector: np.ndarray | None = None
    approach_half_angle_deg: float = 45.0
    # --- gripper ---
    gripper_close_value: float = 1.0          # action-space value used when the policy commands a close (>0)
    # --- velocity caps in action units (1.0 = policy max) ---
    max_pos_delta: float = 1.0
    max_rot_delta: float = 1.0
    # --- pre-grasp waypoint relative to target object; None = off ---
    pre_grasp_waypoint: np.ndarray | None = None
    pre_grasp_tol_m: float = 0.015
    # --- abort / retry ---
    abort_predicate: str | None = None
    abort_max_retries: int = 1
    abort_lift_m: float = 0.08
    # --- v3: temporal resampling of the action chunk (>1 = faster) ---
    time_scale: float = 1.0
    # --- bookkeeping ---
    applied_edits: list[dict[str, Any]] = field(default_factory=list)

    def is_identity(self) -> bool:
        return (
            not np.any(self.grasp_offset)
            and self.approach_vector is None
            and self.gripper_close_value == 1.0
            and self.max_pos_delta >= 1.0
            and self.max_rot_delta >= 1.0
            and self.pre_grasp_waypoint is None
            and self.abort_predicate is None
            and self.time_scale == 1.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "grasp_offset": self.grasp_offset.tolist(),
            "approach_vector": None if self.approach_vector is None else self.approach_vector.tolist(),
            "approach_half_angle_deg": self.approach_half_angle_deg,
            "gripper_close_value": self.gripper_close_value,
            "max_pos_delta": self.max_pos_delta,
            "max_rot_delta": self.max_rot_delta,
            "pre_grasp_waypoint": None if self.pre_grasp_waypoint is None else self.pre_grasp_waypoint.tolist(),
            "pre_grasp_tol_m": self.pre_grasp_tol_m,
            "abort_predicate": self.abort_predicate,
            "abort_max_retries": self.abort_max_retries,
            "abort_lift_m": self.abort_lift_m,
            "time_scale": self.time_scale,
            "applied_edits": list(self.applied_edits),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "ConstraintSet":
        if not d:
            return cls()
        c = cls()
        if d.get("grasp_offset") is not None:
            c.grasp_offset = np.asarray(d["grasp_offset"], dtype=np.float32)
        if d.get("approach_vector") is not None:
            c.approach_vector = np.asarray(d["approach_vector"], dtype=np.float32)
        for k in ("approach_half_angle_deg", "gripper_close_value", "max_pos_delta", "max_rot_delta",
                  "pre_grasp_tol_m", "abort_predicate", "abort_max_retries", "abort_lift_m", "time_scale"):
            if k in d and d[k] is not None:
                setattr(c, k, d[k])
        if d.get("pre_grasp_waypoint") is not None:
            c.pre_grasp_waypoint = np.asarray(d["pre_grasp_waypoint"], dtype=np.float32)
        c.applied_edits = list(d.get("applied_edits", []))
        return c


def apply_edit(cs: ConstraintSet, edit: Edit) -> ConstraintSet:
    """Return a new ConstraintSet with the S3 edit applied. Raises ValueError on unknown op / bad params."""
    if edit.op not in EDIT_OPS["S3"]:
        raise ValueError(f"unknown S3 op {edit.op!r}")
    p = edit.params
    c = replace(cs, grasp_offset=cs.grasp_offset.copy(), applied_edits=list(cs.applied_edits))
    if edit.op == "set_grasp_offset":
        c.grasp_offset = np.asarray([p.get("dx", 0.0), p.get("dy", 0.0), p.get("dz", 0.0)], dtype=np.float32)
        if np.linalg.norm(c.grasp_offset) > 0.10:
            raise ValueError("grasp offset > 10 cm rejected")
    elif edit.op == "set_approach_vector":
        v = np.asarray(p["vector"], dtype=np.float32)
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            raise ValueError("zero approach vector")
        c.approach_vector = v / n
        c.approach_half_angle_deg = float(p.get("half_angle_deg", 45.0))
    elif edit.op == "set_gripper_aperture":
        c.gripper_close_value = float(np.clip(p["value"], -1.0, 1.0))
    elif edit.op == "set_velocity_cap":
        c.max_pos_delta = float(np.clip(p.get("max_pos_delta", 1.0), 0.05, 1.0))
        c.max_rot_delta = float(np.clip(p.get("max_rot_delta", 1.0), 0.05, 1.0))
    elif edit.op == "insert_pre_grasp_waypoint":
        c.pre_grasp_waypoint = np.asarray([p.get("dx", 0.0), p.get("dy", 0.0), p.get("dz", 0.05)], dtype=np.float32)
        c.pre_grasp_tol_m = float(p.get("tol_m", 0.015))
    elif edit.op == "set_abort_retry":
        c.abort_predicate = str(p["predicate"])
        c.abort_max_retries = int(p.get("max_retries", 1))
        c.abort_lift_m = float(p.get("lift_m", 0.08))
    c.applied_edits.append({"op": edit.op, "params": dict(p)})
    return c
