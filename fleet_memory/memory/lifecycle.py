"""Lesson lifecycle: intake validation, A/B arm assignment, evidence, and the candidate -> validated
-> retired gate. Lessons are hypotheses; only this file promotes one (CLAUDE.md invariant 7).

Evidence for a lesson comes from episodes that *retrieved* it: treated if it was applied,
control if the A/B assignment withheld it. Success is Outcome.env_success, nothing else.
"""
from __future__ import annotations

import hashlib
import math
from typing import Iterable

from fleet_memory.execution.constraints import ConstraintSet, apply_edit
from fleet_memory.memory.schema import (EDIT_OPS, PREDICATES, Edit, Episode, Evidence, Lesson,
                                        LessonStatusChange, Subtask, _from)

SURFACES = ("S1", "S2", "S3")


# ------------------------------------------------------------------------- intake
def validate_lesson(l: Lesson | dict) -> tuple[bool, str]:
    """Structural gate at intake. A lesson must carry a machine-applicable edit on one surface from the
    closed op vocabulary, with known param keys and a known trigger predicate. Prose alone is rejected."""
    d = l.to_dict() if isinstance(l, Lesson) else l
    if not isinstance(d, dict):
        return False, f"lesson must be a Lesson or dict, got {type(l).__name__}"
    surface = d.get("surface")
    surface = getattr(surface, "value", surface)
    if surface not in SURFACES:
        return False, f"unknown surface {surface!r} (want one of {SURFACES})"
    edit = d.get("edit")
    if not isinstance(edit, dict) or not isinstance(edit.get("op"), str):
        return False, "missing machine-applicable edit (prose-only lessons are rejected)"
    op, ops = edit["op"], EDIT_OPS[surface]
    if op not in ops:
        return False, f"unknown op {op!r} for {surface} (want one of {sorted(ops)})"
    params = edit.get("params")
    if not isinstance(params, dict) or not params:
        return False, f"edit.params for {op} must be a non-empty dict"
    extra = set(params) - set(ops[op])
    if extra:
        return False, f"unknown params {sorted(extra)} for {op} (want subset of {sorted(ops[op])})"
    trig = d.get("trigger")
    if not isinstance(trig, dict):
        return False, "missing trigger"
    for k in ("task_family", "object_class", "phase"):
        if not isinstance(trig.get(k), str) or not trig[k]:
            return False, f"trigger.{k} must be a non-empty str"
    if trig.get("predicate") not in PREDICATES:
        return False, f"unknown trigger predicate {trig.get('predicate')!r}"
    if surface == "S3":
        if op == "set_abort_retry" and params.get("predicate") not in PREDICATES:
            return False, f"unknown abort predicate {params.get('predicate')!r}"
        try:
            apply_edit(ConstraintSet(), Edit(op=op, params=dict(params)))
        except Exception as e:                      # ValueError / KeyError / TypeError from bad params
            return False, f"S3 edit rejected: {e}"
    elif surface == "S2":
        ins = params.get("instruction")
        if not isinstance(ins, str) or not ins.strip():
            return False, "S2 instruction must be a non-empty str"
    else:  # S1
        if op in ("add_precondition", "set_abort_condition") and params.get("predicate") not in PREDICATES:
            return False, f"unknown S1 predicate {params.get('predicate')!r}"
        if op == "replan":
            subs = params.get("subtasks")
            if not isinstance(subs, list) or not subs:
                return False, "replan needs a non-empty subtasks list"
            try:
                for s in subs:
                    _from(Subtask, s)
            except Exception as e:
                return False, f"replan subtask malformed: {e}"
        if op == "rebind_object" and not isinstance(params.get("object"), str):
            return False, "rebind_object needs a str object"
    return True, "ok"


# ------------------------------------------------------------------- A/B assignment
def assign_arm(lesson_id: str, seed: int, task_id: str, treat_frac: float = 0.5) -> str:
    """Deterministic: same (lesson, seed, task) always lands in the same arm."""
    h = hashlib.sha256(f"{lesson_id}|{task_id}|{int(seed)}".encode("utf-8")).digest()
    u = int.from_bytes(h[:8], "big") / 2.0 ** 64
    return "treated" if u < treat_frac else "control"


# ------------------------------------------------------------------------ evidence
def two_proportion_test(s1: int, n1: int, s2: int, n2: int) -> tuple[float | None, float | None]:
    """One-sided pooled z-test, H1: rate1 > rate2. Returns (p_value, lift_pp); (None, None) if an arm is empty."""
    if n1 == 0 or n2 == 0:
        return None, None
    r1, r2 = s1 / n1, s2 / n2
    p = (s1 + s2) / (n1 + n2)
    se = math.sqrt(p * (1.0 - p) * (1.0 / n1 + 1.0 / n2))
    z = (r1 - r2) / se if se > 0 else 0.0
    p_value = 0.5 * math.erfc(z / math.sqrt(2.0))   # normal survival function
    return float(p_value), float(100.0 * (r1 - r2))


def evidence_from_episodes(episodes: Iterable[Episode], lesson_id: str) -> Evidence:
    ev = Evidence()
    for ep in episodes:
        if lesson_id not in ep.retrieved_lesson_ids or ep.outcome is None:
            continue
        ok = 1 if ep.outcome.env_success else 0
        if lesson_id in ep.applied_lesson_ids:
            ev.treated_n += 1
            ev.treated_successes += ok
        else:
            ev.control_n += 1
            ev.control_successes += ok
    ev.trials = ev.treated_n + ev.control_n
    ev.p_value, ev.lift_pp = two_proportion_test(ev.treated_successes, ev.treated_n,
                                                 ev.control_successes, ev.control_n)
    return ev


def update_evidence(store, lesson_id: str) -> Evidence:
    return evidence_from_episodes(store.episodes(), lesson_id)


# ---------------------------------------------------------------------------- gate
def _decide(lesson: Lesson, ev: Evidence, min_n: int, alpha: float, min_lift_pp: float) -> LessonStatusChange | None:
    n = ev.treated_n + ev.control_n
    if lesson.status == "candidate":
        if n < min_n:
            return None
        if ev.p_value is not None and ev.p_value < alpha and ev.lift_pp >= min_lift_pp:
            to, why = "validated", "promoted"
        else:
            to, why = "retired", "no significant lift"
    elif lesson.status == "validated":
        if n < min_n or ev.lift_pp is None or ev.lift_pp >= 0:
            return None
        to, why = "retired", "regression"
    else:
        return None
    lift = "n/a" if ev.lift_pp is None else f"{ev.lift_pp:+.1f}pp"
    pv = "n/a" if ev.p_value is None else f"{ev.p_value:.4f}"
    reason = f"{why}: n={n} (treated {ev.treated_successes}/{ev.treated_n}, control " \
             f"{ev.control_successes}/{ev.control_n}) lift={lift} p={pv}"
    return LessonStatusChange(lesson_id=lesson.lesson_id, from_status=lesson.status, to_status=to,
                              reason=reason, evidence=ev)


def evaluate(store, lesson_id: str, min_n: int = 30, alpha: float = 0.05,
             min_lift_pp: float = 2.0) -> LessonStatusChange | None:
    """candidate -> validated if n >= min_n, one-sided p < alpha and lift >= min_lift_pp; candidate -> retired
    if n >= min_n otherwise; validated -> retired if n >= min_n and cumulative lift < 0 (control counts freeze
    once validated since every retrieval is applied, so this compares later treated episodes against the
    A/B-era control rate). Appends the status event to the store when something changes."""
    lesson = store.lessons().get(lesson_id)
    if lesson is None:
        return None
    ch = _decide(lesson, update_evidence(store, lesson_id), min_n, alpha, min_lift_pp)
    if ch is not None:
        store.append(ch)
    return ch


def gate_all(store, min_n: int = 30, alpha: float = 0.05, min_lift_pp: float = 2.0) -> list[LessonStatusChange]:
    """Run the gate over every live (candidate/validated) lesson. Reads episodes once."""
    episodes = store.episodes()
    changes: list[LessonStatusChange] = []
    for lesson in store.lessons().values():
        if lesson.status == "retired":
            continue
        ch = _decide(lesson, evidence_from_episodes(episodes, lesson.lesson_id), min_n, alpha, min_lift_pp)
        if ch is not None:
            store.append(ch)
            changes.append(ch)
    return changes


def lesson_precision(store) -> dict:
    """Of lessons that reached a verdict, the fraction validated. None until any verdict exists."""
    counts = {"candidates": 0, "validated": 0, "retired": 0}
    for l in store.lessons().values():
        key = "candidates" if l.status == "candidate" else l.status
        counts[key] = counts.get(key, 0) + 1
    resolved = counts["validated"] + counts["retired"]
    counts["precision"] = counts["validated"] / resolved if resolved else None
    return counts
