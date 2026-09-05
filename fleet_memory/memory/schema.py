"""The contract. Every record type in the event log, the control-surface vocabulary,
and the closed set of machine-applicable edit ops the coach may emit.

Nothing here touches an env, a policy, or an LLM. Pure data.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np

# --------------------------------------------------------------------------- #
# Control surfaces
# --------------------------------------------------------------------------- #

class Surface(str, Enum):
    S1_PLAN = "S1"          # subtask sequence, object bindings, preconditions, abort conditions  (planner)
    S2_INSTRUCTION = "S2"   # which instruction string the VLA receives, closed vocabulary      (policy cond.)
    S3_CONSTRAINT = "S3"    # geometric / kinematic constraints enforced by the execution shim
    S4_SELECTION = "S4"     # action rescoring / rejection sampling — deferred to v2.5
    S5_WEIGHTS = "S5"       # LoRA — baseline only, never a lesson


# Closed vocabulary of edit ops, keyed by surface. The coach must pick one of these.
# Anything outside this table is rejected at lesson intake (memory/lifecycle.py:validate_lesson).
EDIT_OPS: dict[str, dict[str, dict[str, str]]] = {
    "S1": {
        "replan":            {"subtasks": "list[Subtask dict]"},
        "rebind_object":     {"subtask_id": "str", "object": "str"},
        "add_precondition":  {"subtask_id": "str", "predicate": "str"},
        "set_abort_condition": {"subtask_id": "str", "predicate": "str", "max_steps": "int"},
    },
    "S2": {
        "set_instruction":   {"instruction": "str  # must be in the task's instruction vocabulary"},
    },
    "S3": {
        "set_grasp_offset":         {"dx": "float", "dy": "float", "dz": "float"},          # metres, in world frame
        "set_approach_vector":      {"vector": "list[float] len 3", "half_angle_deg": "float"},
        "set_gripper_aperture":     {"value": "float in [-1, 1]  # action-space gripper command when closing"},
        "set_velocity_cap":         {"max_pos_delta": "float in (0, 1]", "max_rot_delta": "float in (0, 1]"},
        "insert_pre_grasp_waypoint": {"dx": "float", "dy": "float", "dz": "float", "tol_m": "float"},  # rel. to target object
        "set_abort_retry":          {"predicate": "str  # detectors.py predicate name", "max_retries": "int", "lift_m": "float"},
    },
}

# Predicates the shim / retrieval can evaluate from an Obs (implemented in execution/detectors.py).
PREDICATES: tuple[str, ...] = (
    "always",
    "ee_near_target",          # ||ee_pos - target_pos|| < 0.06 m
    "gripper_open",
    "gripper_closed",
    "object_height_below_rim", # target object z < its initial z + 0.01 (not lifted)
    "grasp_slipped",           # gripper closed but object not moving with ee for k steps
    "no_progress",             # stall: success sub-conditions unchanged for k steps
    "object_lifted",           # target z > initial z + 0.03
    "ee_above_target",         # xy distance < 0.03 and ee z > target z
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _clean(o: Any) -> Any:
    """Make dataclasses / numpy / enums JSON-serialisable."""
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return {k: _clean(v) for k, v in dataclasses.asdict(o).items()}
    if isinstance(o, Enum):
        return o.value
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


class Record:
    """Mixin for all log records. Subclasses are dataclasses with a `type` field."""

    def to_dict(self) -> dict:
        return _clean(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))


# --------------------------------------------------------------------------- #
# S1: plan
# --------------------------------------------------------------------------- #

@dataclass
class Subtask(Record):
    subtask_id: str
    skill: Literal["reach", "grasp", "lift", "move", "place", "release", "open", "close", "push", "custom"]
    target_object: str                       # object name as it appears in Obs.object_poses
    destination: str | None = None           # object / site name, for move/place
    instruction: str | None = None           # S2: the string handed to the VLA for this subtask
    preconditions: list[str] = field(default_factory=list)   # predicate names
    abort_predicate: str | None = None       # predicate name
    max_steps: int = 150
    constraints: dict[str, Any] = field(default_factory=dict)  # S3 ConstraintSet.to_dict() overrides


@dataclass
class Plan(Record):
    plan_id: str
    task_id: str
    subtasks: list[Subtask]
    instruction_vocab: list[str]             # closed S2 vocabulary for this task
    source: Literal["planner", "identity", "intervention"] = "planner"
    rationale_text: str = ""


# --------------------------------------------------------------------------- #
# Lessons
# --------------------------------------------------------------------------- #

@dataclass
class Trigger(Record):
    task_family: str                         # e.g. "drawer_manipulation", "pick_place", "*"
    object_class: str                        # e.g. "bowl", "*"
    phase: str                               # subtask skill name or "*"
    predicate: str                           # one of PREDICATES
    environment_id: str | None = None        # v3: if set, the lesson only fires in this environment


@dataclass
class Edit(Record):
    op: str                                  # key in EDIT_OPS[surface]
    params: dict[str, Any]


@dataclass
class Evidence(Record):
    trials: int = 0                          # matched episodes where lesson was applicable
    treated_n: int = 0
    treated_successes: int = 0
    control_n: int = 0
    control_successes: int = 0
    p_value: float | None = None
    lift_pp: float | None = None             # treated_rate - control_rate, in percentage points


@dataclass
class Lesson(Record):
    lesson_id: str
    surface: str                             # Surface value: "S1" | "S2" | "S3"
    trigger: Trigger
    edit: Edit
    rationale_text: str = ""                 # dashboard only — NEVER fed to the policy
    status: Literal["candidate", "validated", "retired"] = "candidate"
    provenance: dict[str, Any] = field(default_factory=dict)   # {"from_episodes": [...], "coach_model": "..."}
    evidence: Evidence = field(default_factory=Evidence)
    ts: str = field(default_factory=now_iso)
    type: str = "lesson"


@dataclass
class LessonStatusChange(Record):
    lesson_id: str
    from_status: str
    to_status: str
    reason: str
    evidence: Evidence
    ts: str = field(default_factory=now_iso)
    type: str = "lesson_status"


# --------------------------------------------------------------------------- #
# Episodes, interventions, snapshots
# --------------------------------------------------------------------------- #

@dataclass
class Intervention(Record):
    episode_id: str
    step: int
    trigger: str                             # "stall" | "constraint_violation" | "interval" | "abort_retry"
    surface: str
    edit: Edit
    rationale_text: str = ""
    coach_model: str = ""
    latency_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    ts: str = field(default_factory=now_iso)
    type: str = "intervention"


@dataclass
class Outcome(Record):
    env_success: bool
    steps: int
    termination: Literal["success", "timeout", "abort", "error"]
    wall_s: float = 0.0
    error: str | None = None


@dataclass
class EpisodeMetrics(Record):
    """v3 per-episode cost inputs (analysis/cost.py). jerk is normalised by the Arm A reference."""
    steps: int
    success: bool
    jerk: float                              # mean ||Δ²(ee_pos)|| / jerk_ref
    force_proxy: float                       # mean max(0, ||a_t|| - 0.7 * a_max)
    cost: float | None = None                # None until a cost_reference exists for the skill instance
    mean_pos_delta: float = 0.0              # mean ||dpos|| per step, for the mastery story


@dataclass
class Episode(Record):
    episode_id: str
    suite: str
    task_id: str
    seed: int
    condition: str                           # arm name from runner/conditions.py
    policy: str
    surfaces_enabled: list[str]              # subset of ["S1","S2","S3"]
    retrieved_lesson_ids: list[str]          # FROZEN before reset(); never edited
    applied_lesson_ids: list[str]            # subset actually applied (candidate A/B may withhold some)
    retrieval_frozen_at: str
    plan: Plan | None = None
    instruction_used: str = ""
    constraints_active: dict[str, Any] = field(default_factory=dict)
    interventions: list[Intervention] = field(default_factory=list)
    outcome: Outcome | None = None
    lessons_generated: list[str] = field(default_factory=list)   # lesson_ids from outer loop
    held_out: bool = False                   # held-out suites never generate lessons
    # --- v3 ---
    environment_id: str = ""                 # f"{suite}_{task_id}" + perturbation tag; keys the house model
    skill_instance_versions: dict[str, int] = field(default_factory=dict)   # {skill_instance_id: version used}
    s3_params: dict[str, float] = field(default_factory=dict)              # the S3 vector actually applied (params.py)
    metrics: EpisodeMetrics | None = None
    perturbation: dict[str, Any] = field(default_factory=dict)             # envs/perturb.py config, {} if none
    ts: str = field(default_factory=now_iso)
    type: str = "episode"


@dataclass
class Snapshot(Record):
    episode_id: str
    task_id: str
    situation_embedding: list[float]
    situation_text: str                      # what was embedded: task + scene objects + phase + instruction
    instruction_used: str
    constraints_active: dict[str, Any]
    plan_taken: list[dict]
    outcome_flag: bool
    keyframe_indices: list[int]
    ts: str = field(default_factory=now_iso)
    type: str = "snapshot"


# --------------------------------------------------------------------------- #
# v3: sleep loop — skill instances, consolidations, drift, cost reference
# --------------------------------------------------------------------------- #

def skill_instance_id(skill: str, environment_id: str) -> str:
    return f"si_{skill}__{environment_id}"


@dataclass
class GateResult(Record):
    passed: bool
    incumbent_cost: float
    candidate_cost: float
    n_seeds: int
    incumbent_success: float = 0.0
    candidate_success: float = 0.0
    success_delta_pp: float = 0.0            # (candidate - incumbent) * 100


@dataclass
class Scorecard(Record):
    n: int = 0
    successes: int = 0
    mean_steps: float | None = None
    mean_jerk: float | None = None
    mean_force_proxy: float | None = None
    ewma_cost: float | None = None
    baseline_cost: float | None = None       # gate cost at promotion; drift compares ewma against this
    baseline_success_rate: float | None = None


@dataclass
class SkillInstance(Record):
    """One versioned S3 parameter vector per (skill, object instance, environment). New version = new event.
    Exactly one `incumbent` per skill_instance_id at any time (memory/house_model.py enforces the view)."""
    skill_instance_id: str
    environment_id: str
    skill: str
    object_instance_id: str
    version: int
    parent_version: int | None
    status: Literal["incumbent", "candidate", "retired"]
    params: dict[str, float]                 # execution/params.py S3Params.to_dict()
    produced_by: dict[str, Any] = field(default_factory=dict)   # {"loop": "sleep"|"init"|"coach", "consolidation_id": ...}
    gate: GateResult | None = None
    scorecard: Scorecard = field(default_factory=Scorecard)
    ts: str = field(default_factory=now_iso)
    type: str = "skill_instance"


@dataclass
class SkillInstanceStatusChange(Record):
    """Retire / re-instate without re-emitting params (append-only)."""
    skill_instance_id: str
    version: int
    from_status: str
    to_status: str
    reason: str
    ts: str = field(default_factory=now_iso)
    type: str = "skill_instance_status"


@dataclass
class Consolidation(Record):
    consolidation_id: str
    phase: Literal["start", "end"]
    skill_instance_id: str
    trigger: Literal["manual", "scheduled", "drift"]
    incumbent_version: int
    optimizer: dict[str, Any] = field(default_factory=dict)     # {method, population, elites, iterations, seeds_per_candidate, pose_jitter_m}
    rollouts: int = 0
    wallclock_s: float = 0.0
    best_candidate: dict[str, Any] | None = None                 # {"params": {...}, "opt_cost": float}
    gate: GateResult | None = None
    promoted_version: int | None = None
    history: list[dict[str, Any]] = field(default_factory=list)  # per-iteration {iter, mean_cost, best_cost, sigma}
    ts: str = field(default_factory=now_iso)
    type: str = "consolidation"


@dataclass
class DriftTrigger(Record):
    skill_instance_id: str
    reason: Literal["ewma_cost_exceeds_baseline", "success_rate_drop"]
    ewma_cost: float
    baseline_cost: float
    ratio: float
    recent_success_rate: float
    baseline_success_rate: float
    episode_id: str = ""
    ts: str = field(default_factory=now_iso)
    type: str = "drift_trigger"


@dataclass
class CostReference(Record):
    """Arm A references for cost normalisation, computed once in Phase 0 and frozen."""
    skill_instance_id: str
    environment_id: str
    steps_ref: float
    jerk_ref: float
    n_episodes: int
    source_condition: str = "A"
    ts: str = field(default_factory=now_iso)
    type: str = "cost_reference"


# --------------------------------------------------------------------------- #
# Deserialisation
# --------------------------------------------------------------------------- #

def _from(cls, d: dict):
    """Recursive dataclass hydration for the record types above."""
    if d is None:
        return None
    kw = {}
    for f in dataclasses.fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        if f.name == "trigger":
            v = _from(Trigger, v)
        elif f.name == "edit":
            v = _from(Edit, v)
        elif f.name == "evidence":
            v = _from(Evidence, v)
        elif f.name == "plan":
            v = _from(Plan, v)
        elif f.name == "subtasks":
            v = [_from(Subtask, s) for s in v]
        elif f.name == "interventions":
            v = [_from(Intervention, i) for i in v]
        elif f.name == "outcome":
            v = _from(Outcome, v)
        elif f.name == "metrics":
            v = _from(EpisodeMetrics, v)
        elif f.name == "gate":
            v = _from(GateResult, v)
        elif f.name == "scorecard":
            v = _from(Scorecard, v)
        kw[f.name] = v
    return cls(**kw)


RECORD_TYPES = {
    "lesson": Lesson,
    "lesson_status": LessonStatusChange,
    "intervention": Intervention,
    "episode": Episode,
    "snapshot": Snapshot,
    "skill_instance": SkillInstance,
    "skill_instance_status": SkillInstanceStatusChange,
    "consolidation": Consolidation,
    "drift_trigger": DriftTrigger,
    "cost_reference": CostReference,
}


def record_from_dict(d: dict):
    cls = RECORD_TYPES.get(d.get("type"))
    if cls is None:
        return d
    return _from(cls, d)
