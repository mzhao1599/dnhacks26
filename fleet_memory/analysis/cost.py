"""v3 per-episode metrics and the scalar cost the sleep loop minimises.

    cost = 1.0 * steps/steps_ref + 0.5 * jerk + 0.5 * force_proxy + 3.0 * (1 - success)

steps_ref / jerk_ref are Arm A means on the same skill instance (Phase 0), frozen as a
`cost_reference` event. LIBERO has no force sensor: force_proxy is an action-magnitude proxy
(mean max(0, ||a_t|| - 0.7 a_max)) and the writeup says so.
"""
from __future__ import annotations

import numpy as np

from fleet_memory.memory.schema import CostReference, EpisodeMetrics

W_STEPS, W_JERK, W_FORCE, W_FAIL = 1.0, 0.5, 0.5, 3.0
A_MAX = np.sqrt(6.0)          # ||a|| of a saturated 6-D delta-pose command
FORCE_THRESH = 0.7 * A_MAX


def raw_jerk(ee_positions: np.ndarray) -> float:
    """mean ||Δ²(ee_pos)|| over the trajectory (metres / step²). 0 for < 3 samples."""
    p = np.asarray(ee_positions, np.float64).reshape(-1, 3)
    if len(p) < 3:
        return 0.0
    return float(np.linalg.norm(np.diff(p, n=2, axis=0), axis=1).mean())


def force_proxy(actions: np.ndarray) -> float:
    a = np.asarray(actions, np.float64).reshape(-1, 7)[:, :6]
    if len(a) == 0:
        return 0.0
    return float(np.maximum(0.0, np.linalg.norm(a, axis=1) - FORCE_THRESH).mean())


def episode_metrics(ee_positions: np.ndarray, actions: np.ndarray, success: bool, steps: int,
                    ref: CostReference | None) -> EpisodeMetrics:
    """Compute from the in-memory trace at episode end. jerk is normalised if a reference exists."""
    rj = raw_jerk(ee_positions)
    a = np.asarray(actions, np.float64).reshape(-1, 7)
    m = EpisodeMetrics(
        steps=int(steps), success=bool(success),
        jerk=rj / ref.jerk_ref if (ref is not None and ref.jerk_ref > 0) else rj,
        force_proxy=force_proxy(a),
        mean_pos_delta=float(np.linalg.norm(a[:, :3], axis=1).mean()) if len(a) else 0.0,
    )
    m.cost = cost(m, ref) if ref is not None else None
    return m


def cost(m: EpisodeMetrics, ref: CostReference) -> float:
    return (W_STEPS * m.steps / max(ref.steps_ref, 1.0)
            + W_JERK * m.jerk
            + W_FORCE * m.force_proxy
            + W_FAIL * (0.0 if m.success else 1.0))


def make_reference(skill_instance_id: str, environment_id: str, metrics: list[EpisodeMetrics],
                   raw_jerks: list[float], condition: str = "A") -> CostReference:
    """Arm A means -> frozen reference. `raw_jerks` are un-normalised (metrics.jerk is raw when ref is None)."""
    if not metrics:
        raise ValueError("no episodes to build a reference from")
    return CostReference(skill_instance_id=skill_instance_id, environment_id=environment_id,
                         steps_ref=float(np.mean([m.steps for m in metrics])),
                         jerk_ref=float(max(np.mean(raw_jerks), 1e-9)),
                         n_episodes=len(metrics), source_condition=condition)


def summarize(metrics: list[EpisodeMetrics]) -> dict:
    if not metrics:
        return {"n": 0}
    c = [m.cost for m in metrics if m.cost is not None]
    return {"n": len(metrics),
            "success_rate": float(np.mean([m.success for m in metrics])),
            "mean_steps": float(np.mean([m.steps for m in metrics])),
            "mean_jerk": float(np.mean([m.jerk for m in metrics])),
            "mean_force_proxy": float(np.mean([m.force_proxy for m in metrics])),
            "mean_cost": float(np.mean(c)) if c else None}
