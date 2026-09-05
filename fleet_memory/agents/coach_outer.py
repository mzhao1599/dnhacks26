"""Outer-loop coach. Runs after an episode with the env success flag in hand and assigns credit:
it may emit candidate Lessons (hypotheses; the A/B gate in memory/lifecycle decides if they stick).
Every emitted lesson is validated against the closed edit vocabulary; rejects are dropped and counted.
"""
from __future__ import annotations

import logging

import numpy as np

from fleet_memory.agents.llm import (COACH_MODEL, EDIT_OPS_TEXT, PREDICATES_TEXT, LLM, parse_context,
                                     register_mock, subsample, validate_lesson, with_context)
from fleet_memory.agents.planner import class_word, instruction_vocab
from fleet_memory.envs.base import TaskInfo
from fleet_memory.memory.schema import EDIT_OPS, PREDICATES, Edit, Episode, Lesson, Trigger, new_id

log = logging.getLogger(__name__)

_ALL_OPS = [op for s in ("S1", "S2", "S3") for op in EDIT_OPS[s]]

OUTER_SYSTEM = f"""You are the outer-loop coach for a fleet of frozen robot policies. After each episode you receive the
environment's success flag (env_success), the plan, the constraints and interventions that were active, and
keyframes. Your job is credit assignment: decide whether this episode teaches a reusable, machine-applicable
lesson, and if so emit it as a structured edit on exactly one control surface. You never write prose for the
policy; `rationale` is dashboard-only.

Closed edit vocabulary (surface -> op -> params):
{EDIT_OPS_TEXT}

Closed predicate vocabulary (trigger.predicate and any `predicate` parameter):
{PREDICATES_TEXT}

Trigger fields: task_family (from context, or "*"), object_class (one class word such as "bowl" that is a
substring of the target object's name, or "*"), phase (a subtask skill, or "*"), predicate (when the lesson
applies; "always" if unconditional). S2 `instruction` must be copied verbatim from instruction_vocab.

Procedure:
1. Localise first. `localisation.step` must be the concrete step index where the episode went wrong (or where an
   intervention rescued it) and `localisation.description` what physically happened there. Lessons without a step
   index are discarded.
2. Emit a lesson only when one mechanism explains the outcome and one edit would plausibly change it. Prefer S3
   for geometric or timing errors, S1 for wrong object bindings or missing aborts, S2 only when the instruction
   wording is the likely cause.
3. Emitting no lesson (lessons: []) is valid and rewarded: expected on success (unless an intervention listed in
   the context demonstrably rescued the episode, in which case you may promote that exact edit), and on ambiguous
   failure. Every lesson is A/B tested at fleet scale; false lessons are costly.
Units: metres, world frame, z up."""

OUTER_USER = "Assign credit for this episode."

LESSONS_SCHEMA = {
    "title": "lessons", "type": "object",
    "properties": {
        "localisation": {"type": "object",
                         "properties": {"step": {"type": ["integer", "null"]}, "description": {"type": "string"}},
                         "required": ["step", "description"]},
        "lessons": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "surface": {"type": "string", "enum": ["S1", "S2", "S3"]},
                "trigger": {"type": "object",
                            "properties": {"task_family": {"type": "string"}, "object_class": {"type": "string"},
                                           "phase": {"type": "string"},
                                           "predicate": {"type": "string", "enum": list(PREDICATES)}},
                            "required": ["task_family", "object_class", "phase", "predicate"]},
                "op": {"type": "string", "enum": _ALL_OPS},
                "params": {"type": "object"},
                "rationale": {"type": "string"},
            },
            "required": ["surface", "trigger", "op", "params", "rationale"],
        }},
    },
    "required": ["localisation", "lessons"],
}


def situation_text(task: TaskInfo, instruction: str, phase: str = "*") -> str:
    """memory.retrieval.situation_text if importable (keeps lesson/query embeddings aligned), else a local mirror."""
    try:
        from fleet_memory.memory.retrieval import situation_text as _st
        return _st(task, instruction, phase)
    except Exception:
        return f"{task.suite} {task.task_family} {task.task_id} objects: {' '.join(task.objects)} phase: {phase} instruction: {instruction}"


class OuterCoach:
    def __init__(self, llm: LLM | None = None):
        self.llm = llm or LLM(COACH_MODEL)
        self.rejected = 0
        self.last_meta: dict = {}

    def assign_credit(self, episode: Episode, task: TaskInfo, keyframes: list[np.ndarray], trace_summary: dict,
                      applied_lessons: list[Lesson]) -> list[Lesson]:
        if episode.outcome is None or episode.held_out:      # nothing to credit / invariant 5
            return []
        vocab = episode.plan.instruction_vocab if episode.plan else instruction_vocab(task)
        ctx = {
            "episode_id": episode.episode_id, "condition": episode.condition,
            "task": {"suite": task.suite, "task_id": task.task_id, "language": task.language,
                     "task_family": task.task_family, "objects": list(task.objects), "max_steps": task.max_steps},
            "env_success": bool(episode.outcome.env_success), "steps": episode.outcome.steps,
            "termination": episode.outcome.termination, "error": episode.outcome.error,
            "instruction_used": episode.instruction_used, "instruction_vocab": vocab,
            "constraints_active": episode.constraints_active,
            "plan": [s.to_dict() for s in (episode.plan.subtasks if episode.plan else [])],
            "interventions": [{"step": i.step, "trigger": i.trigger, "surface": i.surface, "op": i.edit.op,
                               "params": i.edit.params} for i in episode.interventions],
            "applied_lessons": [{"lesson_id": l.lesson_id, "surface": l.surface, "op": l.edit.op,
                                 "params": l.edit.params, "trigger": l.trigger.to_dict(), "status": l.status}
                                for l in applied_lessons],
            "n_keyframes": len(keyframes or []), "trace": trace_summary or {},
        }
        try:
            out, meta = self.llm.complete_json(OUTER_SYSTEM, with_context(OUTER_USER, ctx), LESSONS_SCHEMA,
                                               images=subsample(keyframes or [], 8), max_tokens=3000)
        except Exception as ex:
            log.warning("outer coach query failed: %s", ex)
            return []
        self.last_meta = meta
        loc = out.get("localisation") or {}
        step = loc.get("step")
        lessons: list[Lesson] = []
        for d in out.get("lessons") or []:
            if not isinstance(d, dict):
                continue
            if step is None:                                  # localisation is mandatory
                self.rejected += 1
                log.info("lesson rejected: no localisation step")
                continue
            t = d.get("trigger") or {}
            trig = Trigger(task_family=str(t.get("task_family") or task.task_family),
                           object_class=str(t.get("object_class") or "*"), phase=str(t.get("phase") or "*"),
                           predicate=str(t.get("predicate") or "always"))
            edit = Edit(op=str(d.get("op")), params=dict(d.get("params") or {}))
            lesson = Lesson(
                lesson_id=new_id("les"), surface=str(d.get("surface")), trigger=trig, edit=edit,
                rationale_text=str(d.get("rationale", "")), status="candidate",
                provenance={"from_episodes": [episode.episode_id], "coach_model": meta["model"],
                            "situation_text": situation_text(task, episode.instruction_used or task.language, trig.phase),
                            "localisation": {"step": int(step), "description": str(loc.get("description", ""))},
                            "tokens_in": meta["tokens_in"], "tokens_out": meta["tokens_out"],
                            "latency_s": meta["latency_s"]})
            ok, why = validate_lesson(lesson)
            if ok and lesson.surface == "S2" and edit.params.get("instruction") not in vocab:
                ok, why = False, "S2 instruction not in vocabulary"
            if not ok:
                self.rejected += 1
                log.info("lesson rejected: %s", why)
                continue
            lessons.append(lesson)
        return lessons


@register_mock("lessons")
def _mock_lessons(user: str) -> dict:
    """Failure with a grasp subtask -> one S3 set_grasp_offset dz=-0.015 lesson; otherwise nothing."""
    ctx = parse_context(user)
    grasp = next((s for s in ctx.get("plan") or [] if s.get("skill") == "grasp"), None)
    if ctx.get("env_success") or grasp is None:
        return {"localisation": {"step": None, "description": "no failure to localise"}, "lessons": []}
    task = ctx.get("task") or {}
    objects = task.get("objects") or [grasp.get("target_object", "object")]
    starts = (ctx.get("trace") or {}).get("subtask_starts") or {}
    step = int(starts.get(grasp.get("subtask_id"), 0))
    return {
        "localisation": {"step": step, "description": "grasp closed above the rim; object not lifted"},
        "lessons": [{
            "surface": "S3",
            "trigger": {"task_family": task.get("task_family", "*"), "object_class": class_word(objects[0]),
                        "phase": "grasp", "predicate": "object_height_below_rim"},
            "op": "set_grasp_offset", "params": {"dx": 0.0, "dy": 0.0, "dz": -0.015},
            "rationale": "policy grasps too high on this object class; bias the grasp 1.5 cm lower",
        }],
    }
