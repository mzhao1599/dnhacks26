"""The per-episode subtask loop (split out of worker.py). Steps the env through the plan's subtasks,
feeds the tracker every step, applies inner-coach interventions, records the EE/action trace.
Success is never decided here: the caller reads env.success_flag()."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fleet_memory.agents.planner import apply_plan_edit
from fleet_memory.envs.base import Obs, TaskInfo
from fleet_memory.execution.constraints import ConstraintSet, apply_edit
from fleet_memory.execution.detectors import Tracker, evaluate_predicate
from fleet_memory.memory.schema import Intervention, Plan, Subtask

MOVE_XY_M = 0.05      # move/place: EE (or target) within this xy radius of the destination


@dataclass
class LoopResult:
    steps: int = 0
    termination: str = "timeout"
    ee_positions: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    interventions: list[Intervention] = field(default_factory=list)
    keyframes: list[np.ndarray] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)
    instruction_used: str = ""
    error: str | None = None


def subtask_done(st: Subtask, obs: Obs, tracker: Tracker) -> bool:
    """Per-skill completion signal (progress only, never success)."""
    sk = st.skill
    try:
        if sk == "reach":
            return evaluate_predicate("ee_near_target", obs, st.target_object, tracker)
        if sk in ("grasp", "close"):
            return evaluate_predicate("gripper_closed", obs, st.target_object, tracker)
        if sk == "lift":
            return evaluate_predicate("object_lifted", obs, st.target_object, tracker)
        if sk in ("release", "open"):
            return evaluate_predicate("gripper_open", obs, st.target_object, tracker)
        if sk in ("move", "place") and st.destination:
            dp = obs.object_pos(st.destination)
            if dp is None:
                return False
            ref = obs.ee_pos if sk == "move" else (obs.object_pos(st.target_object) if sk == "place" else None)
            if ref is None:
                return False
            near = float(np.linalg.norm((np.asarray(ref, float) - np.asarray(dp, float))[:2])) < MOVE_XY_M
            return near if sk == "move" else (near and evaluate_predicate("gripper_open", obs, None, tracker))
    except Exception:
        return False
    return False


def trace_summary(res: LoopResult, shim, t: int) -> dict[str, Any]:
    """Outcome-free summary for the coaches (the inner coach scrubs again on its side)."""
    kinds: dict[str, int] = {}
    for e in getattr(shim, "events", []) or []:
        kinds[e.get("kind", "?")] = kinds.get(e.get("kind", "?"), 0) + 1
    return {"n_steps": t, "subtask_starts": dict(res.trace.get("subtask_starts", {})),
            "subtask_ends": dict(res.trace.get("subtask_ends", {})), "shim_events": kinds,
            "aborted": list(res.trace.get("aborted", [])), "precondition_unmet": list(res.trace.get("precondition_unmet", [])),
            "final_ee": [float(x) for x in res.ee_positions[-1]] if res.ee_positions else None}


def _apply_intervention(iv: Intervention, plan: Plan, st: Subtask, shim, task: TaskInfo, surfaces: list[str]) -> bool:
    if iv.surface not in surfaces:
        return False
    if iv.surface == "S1":
        ok = apply_plan_edit(plan, "S1", iv.edit, task, phase=st.skill)
        if ok and iv.edit.op == "rebind_object":
            shim.set_target(st.target_object)
        return ok
    if iv.surface == "S3":
        try:
            shim.set_constraints(apply_edit(shim.constraints, iv.edit))
            return True
        except Exception:
            return False
    return False


def run_subtasks(env, obs: Obs, plan: Plan, shim, tracker: Tracker, task: TaskInfo, *, episode_id: str,
                 surfaces: list[str], coach=None, max_steps: int | None = None, keyframe_every: int = 25,
                 envelope=None, base_constraints: ConstraintSet | None = None) -> LoopResult:
    """`envelope` is only used when the shim has no apply_envelope (older shim): clamped here instead."""
    res = LoopResult()
    res.trace = {"subtask_starts": {}, "subtask_ends": {}, "aborted": [], "precondition_unmet": [], "applied_interventions": []}
    res.ee_positions.append(np.asarray(obs.ee_pos, np.float64).copy())
    budget = int(max_steps or task.max_steps)
    t, done, i = 0, False, 0
    while i < len(plan.subtasks) and not done and t < budget:
        st = plan.subtasks[i]
        instr = st.instruction or task.language
        res.instruction_used = res.instruction_used or instr
        cs = None
        if st.constraints and base_constraints is not None:
            cs = ConstraintSet.from_dict({**base_constraints.to_dict(), **st.constraints})
        shim.reset(instr, target_object=st.target_object, constraints=cs)
        for p in st.preconditions:
            if not evaluate_predicate(p, obs, st.target_object, tracker):
                res.trace["precondition_unmet"].append({"subtask_id": st.subtask_id, "predicate": p, "t": t})
        res.trace["subtask_starts"][st.subtask_id] = t
        st_steps = 0
        while st_steps < st.max_steps and t < budget and not done:
            chunk = np.asarray(shim.act(obs), np.float32).reshape(-1, 7)
            if envelope is not None:
                chunk, _ = envelope.clamp_chunk(chunk, obs.ee_pos)
            for row in chunk:
                obs, done, _info = env.step(row)
                tracker.update(obs, env.success_subconditions())
                res.ee_positions.append(np.asarray(obs.ee_pos, np.float64).copy())
                res.actions.append(np.asarray(row, np.float32).copy())
                t += 1
                st_steps += 1
                if keyframe_every and t % keyframe_every == 0:
                    img = (obs.images or {}).get("agentview")
                    if img is not None:
                        res.keyframes.append(img)
                if done or st_steps >= st.max_steps or t >= budget:
                    break
            if done:
                break
            if coach is not None:
                iv = coach.maybe_intervene(episode_id, t, obs, st, plan, shim.constraints, res.keyframes, tracker,
                                           trace_summary(res, shim, t))
                if iv is not None:
                    res.interventions.append(iv)
                    if _apply_intervention(iv, plan, st, shim, task, surfaces):
                        res.trace["applied_interventions"].append({"t": t, "surface": iv.surface, "op": iv.edit.op})
                    if i < len(plan.subtasks) and plan.subtasks[i] is not st:   # replan: restart at this index
                        break
            if st.abort_predicate and evaluate_predicate(st.abort_predicate, obs, st.target_object, tracker):
                res.trace["aborted"].append({"subtask_id": st.subtask_id, "t": t})
                break
            if subtask_done(st, obs, tracker):
                break
        res.trace["subtask_ends"][st.subtask_id] = t
        if i < len(plan.subtasks) and plan.subtasks[i] is st:
            i += 1
    res.steps = t
    res.termination = "success" if env.success_flag() else "timeout"
    return res
