"""Inner-loop coach. Called every step during an episode; queries the LLM only on a stall or at a
periodic checkpoint and may return ONE S1/S3 Intervention. Hard invariant 4: it never receives the
success flag — the context it builds is scrubbed of anything outcome-shaped before it leaves here.
"""
from __future__ import annotations

import logging

import numpy as np

from fleet_memory.agents.llm import (COACH_MODEL, EDIT_OPS_TEXT, PREDICATES_TEXT, LLM, parse_context,
                                     register_mock, subsample, validate_lesson, with_context)
from fleet_memory.envs.base import Obs
from fleet_memory.execution.constraints import ConstraintSet
from fleet_memory.memory.schema import EDIT_OPS, PREDICATES, Edit, Intervention, Plan, Subtask, Trigger

log = logging.getLogger(__name__)

INNER_SYSTEM = f"""You are the inner-loop coach watching a frozen robot policy DURING an episode. You are called either
because progress stalled ("stall") or at a periodic checkpoint ("interval"). You may emit at most ONE intervention:
a machine-applicable edit to the plan (S1) or to the execution constraints (S3). You never write instructions or
prose for the policy; `rationale` is shown on a dashboard only.

You do not know how the episode will end and must not speculate about it. Judge only what the images, poses and
progress signals show right now.

Closed edit vocabulary (surface -> op -> params). Only S1 and S3 ops are valid mid-episode:
{EDIT_OPS_TEXT}

Closed predicate vocabulary (for any `predicate` parameter):
{PREDICATES_TEXT}

Procedure:
1. Localise first: name the step index at which the problem began (`localisation_step`) and what physically went
   wrong. If you cannot localise a concrete problem, emit surface null.
2. Pick the single smallest edit that addresses that problem. Do not repeat an edit already listed in
   prior_interventions unless the situation has changed.
3. Emitting no intervention (surface null) is a valid and rewarded answer. Unnecessary edits cost more than they
   gain and are penalised by the A/B gate downstream.
Units: metres, world frame, z up. Grasp offsets and waypoints are relative to the target object; velocity caps are
fractions of the policy maximum."""

INNER_USER = "Trigger fired. Decide whether one intervention is warranted."

INTERVENTION_SCHEMA = {
    "title": "intervention", "type": "object",
    "properties": {
        "surface": {"type": ["string", "null"], "enum": ["S1", "S3", None]},
        "op": {"type": ["string", "null"], "enum": [*EDIT_OPS["S1"], *EDIT_OPS["S3"], None]},
        "params": {"type": "object"},
        "localisation_step": {"type": ["integer", "null"]},
        "rationale": {"type": "string"},
    },
    "required": ["surface", "rationale"],
}


def scrub(o):
    """Drop anything outcome-shaped from a nested dict; keep progress sub-conditions under a neutral name."""
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            kl = str(k).lower()
            if kl in ("success_subconditions", "subconditions"):
                out["progress_signals"] = scrub(v)
            elif "success" in kl or "outcome" in kl:
                continue
            else:
                out[k] = scrub(v)
        return out
    if isinstance(o, (list, tuple)):
        return [scrub(v) for v in o]
    return o


def _predicates(obs: Obs, target: str | None, tracker) -> dict[str, bool | None]:
    try:
        from fleet_memory.execution.detectors import evaluate_predicate
    except Exception:
        return {}
    out = {}
    for name in PREDICATES:
        try:
            out[name] = bool(evaluate_predicate(name, obs, target, tracker))
        except Exception:
            out[name] = None
    return out


class InnerCoach:
    def __init__(self, llm: LLM | None = None, check_interval: int = 60, stall_k: int = 40):
        self.llm = llm or LLM(COACH_MODEL)
        self.check_interval, self.stall_k = check_interval, stall_k
        self.queries = self.rejected = 0
        self._episode_id: str | None = None
        self._emitted: list[Intervention] = []
        self._cooldown_until = 0

    def _start_episode(self, episode_id: str) -> None:
        self._episode_id, self._emitted, self._cooldown_until = episode_id, [], 0

    def maybe_intervene(self, episode_id: str, step: int, obs: Obs, subtask: Subtask | None, plan: Plan | None,
                        constraints: ConstraintSet, keyframes: list[np.ndarray], tracker,
                        trace_summary: dict) -> Intervention | None:
        if episode_id != self._episode_id:
            self._start_episode(episode_id)
        trigger = "stall" if tracker.no_progress() else ("interval" if step > 0 and step % self.check_interval == 0 else None)
        if trigger is None or step < self._cooldown_until:
            return None
        self._cooldown_until = step + self.stall_k     # one query per stall window
        self.queries += 1

        target = subtask.target_object if subtask is not None else getattr(tracker, "target", None)
        ctx = {
            "episode_id": episode_id, "step": step, "trigger": trigger,
            "stall_steps": int(getattr(tracker, "stall_steps", 0)),
            "subtask": subtask.to_dict() if subtask is not None else None,
            "plan": [{"subtask_id": s.subtask_id, "skill": s.skill, "target_object": s.target_object,
                      "destination": s.destination} for s in (plan.subtasks if plan else [])],
            "constraints": constraints.to_dict(),
            "ee_pos": obs.ee_pos, "gripper_qpos": obs.gripper_qpos,
            "object_positions": {k: v[0] for k, v in obs.object_poses.items()},
            "predicates": _predicates(obs, target, tracker),
            "prior_interventions": [{"step": i.step, "surface": i.surface, "op": i.edit.op, "params": i.edit.params}
                                    for i in self._emitted],
            "trace": scrub(trace_summary or {}),
        }
        try:
            out, meta = self.llm.complete_json(INNER_SYSTEM, with_context(INNER_USER, ctx), INTERVENTION_SCHEMA,
                                               images=subsample(keyframes or [], 2))
        except Exception as ex:
            log.warning("inner coach query failed: %s", ex)
            return None

        surface = out.get("surface")
        if surface in (None, "", "null"):
            return None
        edit = Edit(op=str(out.get("op")), params=dict(out.get("params") or {}))
        probe = {"surface": surface, "edit": edit.to_dict(),
                 "trigger": Trigger("*", "*", subtask.skill if subtask else "*", "always").to_dict()}
        ok, why = validate_lesson(probe) if surface in ("S1", "S3") else (False, "S2 is not an intervention surface")
        if not ok:
            self.rejected += 1
            log.info("inner coach edit rejected: %s", why)
            return None
        iv = Intervention(episode_id=episode_id, step=step, trigger=trigger, surface=surface, edit=edit,
                          rationale_text=str(out.get("rationale", "")), coach_model=meta["model"],
                          latency_s=float(meta["latency_s"]), tokens_in=int(meta["tokens_in"]),
                          tokens_out=int(meta["tokens_out"]))
        self._emitted.append(iv)
        return iv


@register_mock("intervention")
def _mock_intervention(user: str) -> dict:
    """Stall: set_abort_retry(grasp_slipped) once, then set_grasp_offset dz=-0.01. Interval: nothing."""
    ctx = parse_context(user)
    if ctx.get("trigger") != "stall":
        return {"surface": None, "rationale": "interval check: no anomaly localised"}
    loc = int(ctx.get("step", 0)) - int(ctx.get("stall_steps", 0))
    if not any(p.get("op") == "set_abort_retry" for p in ctx.get("prior_interventions") or []):
        return {"surface": "S3", "op": "set_abort_retry", "params": {"predicate": "grasp_slipped", "max_retries": 1},
                "localisation_step": loc, "rationale": "stall after grasp: allow one retry if the grasp slips"}
    return {"surface": "S3", "op": "set_grasp_offset", "params": {"dx": 0.0, "dy": 0.0, "dz": -0.01},
            "localisation_step": loc, "rationale": "still stalled: grasp lower"}
