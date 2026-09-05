"""S1 planner. The LLM (Haiku) proposes a subtask decomposition; S1/S2 lessons are then applied to
the plan deterministically in code. The planner never chooses the instruction string itself (S2);
it only inherits the canonical one, and lessons may swap it for a vocabulary entry.
"""
from __future__ import annotations

import logging
import re

from fleet_memory.agents.llm import LLM, PLANNER_MODEL, PREDICATES_TEXT, parse_context, register_mock, with_context
from fleet_memory.envs.base import Obs, TaskInfo
from fleet_memory.memory.schema import EDIT_OPS, PREDICATES, Edit, Lesson, Plan, Subtask, new_id

log = logging.getLogger(__name__)

SKILLS = ("reach", "grasp", "lift", "move", "place", "release", "open", "close", "push", "custom")

PLANNER_SYSTEM = f"""You are the task planner for a frozen vision-language-action robot policy (control surface S1).
Decompose the task into a short sequence of subtasks. The policy executes each subtask on its own; your plan
only decides, per subtask, which scene object is the target, the destination (if any), preconditions, when to
abort, and a step budget.

Rules:
- skill is one of: {", ".join(SKILLS)}.
- target_object and destination must be object names copied verbatim from `objects` in the context.
- preconditions and abort_predicate are names from this closed list (or null):
{PREDICATES_TEXT}
- Leave `instruction` null. The instruction string handed to the policy is chosen by memory (S2), not by you.
- 3 to 6 subtasks. A pick-and-place is reach, grasp, lift, move, place. max_steps bounds each subtask; the
  sum may exceed the episode horizon.
- subtask_id must be unique and stable: "<skill>_<index>", e.g. "grasp_1".
Answer only through the tool call."""

PLANNER_USER = "Plan this task. Use only the objects listed."

PLAN_SCHEMA = {
    "title": "plan", "type": "object",
    "properties": {
        "subtasks": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "subtask_id": {"type": "string"},
                "skill": {"type": "string", "enum": list(SKILLS)},
                "target_object": {"type": "string"},
                "destination": {"type": ["string", "null"]},
                "instruction": {"type": ["string", "null"]},
                "preconditions": {"type": "array", "items": {"type": "string", "enum": list(PREDICATES)}},
                "abort_predicate": {"type": ["string", "null"]},
                "max_steps": {"type": "integer"},
            },
            "required": ["subtask_id", "skill", "target_object", "max_steps"],
        }},
        "rationale": {"type": "string"},
    },
    "required": ["subtasks"],
}


# --------------------------------------------------------------------------- #
# Vocabulary and identity plan
# --------------------------------------------------------------------------- #

def _paraphrases_fallback(x: str) -> list[str]:
    """Mirror of execution.probes.paraphrases; used only if that module is unavailable."""
    return [x, "please " + x, x + " carefully", x + ", grasping it by the handle",
            re.sub(r"[^\w\s]", "", x).lower()]


def instruction_vocab(task: TaskInfo) -> list[str]:
    """Canonical instruction + the S2 probe paraphrases, deduplicated, canonical first."""
    try:
        from fleet_memory.execution.probes import paraphrases
    except Exception:
        paraphrases = _paraphrases_fallback
    out: list[str] = []
    for s in [task.language, *paraphrases(task.language)]:
        if s not in out:
            out.append(s)
    return out


def identity_plan(task: TaskInfo) -> Plan:
    """Arm A/B fallback: one opaque subtask, canonical instruction, first object as target."""
    st = Subtask(subtask_id="custom_0", skill="custom", target_object=task.objects[0],
                 instruction=task.language, max_steps=task.max_steps)
    return Plan(plan_id=new_id("plan"), task_id=task.task_id, subtasks=[st],
                instruction_vocab=instruction_vocab(task), source="identity")


# --------------------------------------------------------------------------- #
# Applying lessons / interventions to a plan (pure, in place)
# --------------------------------------------------------------------------- #

def _resolve_object(name, objects: list[str]) -> str | None:
    if not name:
        return None
    if name in objects:
        return name
    n = str(name).lower().replace(" ", "_")
    for o in objects:
        if o.lower() == n or n in o.lower() or o.lower() in n:
            return o
    return None


def _find_subtask(plan: Plan, subtask_id, phase: str = "*") -> Subtask | None:
    """Exact id first, else first subtask whose skill matches the id's skill prefix, else the phase."""
    for s in plan.subtasks:
        if s.subtask_id == subtask_id:
            return s
    for want in (str(subtask_id or "").split("_")[0], phase):
        if want and want != "*":
            for s in plan.subtasks:
                if s.skill == want:
                    return s
    return None


def _sanitize(d: dict, i: int, task: TaskInfo, vocab: list[str]) -> Subtask | None:
    if not isinstance(d, dict):
        return None
    skill = d.get("skill") if d.get("skill") in SKILLS else "custom"
    target = _resolve_object(d.get("target_object"), task.objects) or task.objects[0]
    instr = d.get("instruction")
    ab = d.get("abort_predicate")
    try:
        ms = int(d.get("max_steps") or 150)
    except (TypeError, ValueError):
        ms = 150
    return Subtask(
        subtask_id=str(d.get("subtask_id") or f"{skill}_{i}"), skill=skill, target_object=target,
        destination=_resolve_object(d.get("destination"), task.objects),
        instruction=instr if instr in vocab else task.language,
        preconditions=[p for p in (d.get("preconditions") or []) if p in PREDICATES],
        abort_predicate=ab if ab in PREDICATES else None,
        max_steps=max(5, min(ms, task.max_steps)),
    )


def apply_plan_edit(plan: Plan, surface: str, edit: Edit, task: TaskInfo, phase: str = "*") -> bool:
    """Apply one S1/S2 edit to `plan` in place. Returns False (no-op) when it cannot be applied."""
    p = edit.params or {}
    if surface == "S1" and edit.op == "replan":
        subs = [_sanitize(d, i, task, plan.instruction_vocab) for i, d in enumerate(p.get("subtasks") or [])]
        subs = [s for s in subs if s]
        if not subs:
            return False
        plan.subtasks = subs
        return True
    if surface == "S1" and edit.op in ("rebind_object", "add_precondition", "set_abort_condition"):
        st = _find_subtask(plan, p.get("subtask_id"), phase)
        if st is None:
            return False
        if edit.op == "rebind_object":
            obj = _resolve_object(p.get("object"), task.objects)
            if obj is None:
                return False
            st.target_object = obj
        elif edit.op == "add_precondition":
            if p.get("predicate") not in PREDICATES:
                return False
            if p["predicate"] not in st.preconditions:
                st.preconditions.append(p["predicate"])
        else:
            if p.get("predicate") not in PREDICATES:
                return False
            st.abort_predicate = p["predicate"]
            if p.get("max_steps") is not None:
                st.max_steps = max(5, int(p["max_steps"]))
        return True
    if surface == "S2" and edit.op == "set_instruction":
        instr = p.get("instruction")
        if instr not in plan.instruction_vocab:   # closed vocabulary; anything else is silently dropped
            return False
        hit = False
        for st in plan.subtasks:
            if phase in ("*", st.skill):
                st.instruction, hit = instr, True
        return hit
    return False


def apply_lesson(plan: Plan, lesson: Lesson, task: TaskInfo) -> bool:
    if lesson.surface not in ("S1", "S2") or lesson.edit.op not in EDIT_OPS.get(lesson.surface, {}):
        return False
    return apply_plan_edit(plan, lesson.surface, lesson.edit, task, phase=lesson.trigger.phase or "*")


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #

class Planner:
    def __init__(self, llm: LLM | None = None):
        self.llm = llm or LLM(PLANNER_MODEL)
        self.last_meta: dict = {}

    def plan(self, task: TaskInfo, obs: Obs | None, lessons: list[Lesson]) -> Plan:
        vocab = instruction_vocab(task)
        ctx = {"task_id": task.task_id, "suite": task.suite, "language": task.language,
               "task_family": task.task_family, "objects": list(task.objects), "max_steps": task.max_steps,
               "object_positions": {k: v[0] for k, v in obs.object_poses.items()} if obs is not None else {},
               "ee_pos": obs.ee_pos if obs is not None else None}
        subtasks: list[Subtask] = []
        try:
            out, self.last_meta = self.llm.complete_json(PLANNER_SYSTEM, with_context(PLANNER_USER, ctx), PLAN_SCHEMA)
            subtasks = [s for s in (_sanitize(d, i, task, vocab) for i, d in enumerate(out.get("subtasks") or [])) if s]
            rationale = str(out.get("rationale", ""))
        except Exception as ex:                     # network / bad output: fall back to identity
            log.warning("planner failed, using identity plan: %s", ex)
            rationale = f"planner_error: {ex}"
        source = "planner"
        if not subtasks:
            subtasks, source = identity_plan(task).subtasks, "identity"
        plan = Plan(plan_id=new_id("plan"), task_id=task.task_id, subtasks=subtasks,
                    instruction_vocab=vocab, source=source, rationale_text=rationale)
        for l in lessons:
            apply_lesson(plan, l, task)
        return plan


# --------------------------------------------------------------------------- #
# Mock
# --------------------------------------------------------------------------- #

def class_word(name: str) -> str:
    """'akita_black_bowl_1' -> 'bowl'."""
    toks = [t for t in re.split(r"[_\s]+", str(name).lower()) if t and not t.isdigit()]
    return toks[-1] if toks else str(name)


def _mention_pos(name: str, lang: str) -> int:
    pos = lang.find(name.replace("_", " ").lower())
    return pos if pos >= 0 else lang.find(class_word(name))


@register_mock("plan")
def _mock_plan(user: str) -> dict:
    """reach/grasp/lift/move/place over the objects mentioned in the instruction (in order)."""
    ctx = parse_context(user)
    objects = list(ctx.get("objects") or ["object"])
    lang = str(ctx.get("language") or "").lower()
    mentioned = sorted((pos, i, o) for i, o in enumerate(objects) if (pos := _mention_pos(o, lang)) >= 0)
    target = mentioned[0][2] if mentioned else objects[0]
    dest = mentioned[1][2] if len(mentioned) > 1 else next((o for o in objects if o != target), None)
    n = int(ctx.get("max_steps") or 200)
    legs = [("reach", target, None, n // 5), ("grasp", target, None, 3 * n // 10), ("lift", target, None, n // 5)]
    if dest:
        legs += [("move", target, dest, 3 * n // 10), ("place", target, dest, 3 * n // 10)]
    subs = [{"subtask_id": f"{sk}_{i}", "skill": sk, "target_object": t, "destination": d, "instruction": None,
             "preconditions": [], "abort_predicate": None, "max_steps": max(5, ms)}
            for i, (sk, t, d, ms) in enumerate(legs)]
    return {"subtasks": subs, "rationale": f"mock decomposition: {target} -> {dest}"}
