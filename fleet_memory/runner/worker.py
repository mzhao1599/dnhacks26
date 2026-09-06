"""One episode, end to end. The order in run_episode is the contract (CLAUDE.md invariants):
retrieval frozen BEFORE reset, success from env.success_flag() only, envelope always applied,
held-out suites never generate lessons, everything appended to the event log."""
from __future__ import annotations

import json

import inspect
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fleet_memory.analysis import cost as cost_mod
from fleet_memory.envs.base import Obs, TaskInfo
from fleet_memory.execution.constraints import ConstraintSet, apply_edit
from fleet_memory.execution.detectors import Tracker
from fleet_memory.execution.envelope import DEFAULT_ENVELOPE
from fleet_memory.execution.params import S3Params
from fleet_memory.execution.shim import ExecutionShim
from fleet_memory.memory import lifecycle, schema
from fleet_memory.memory.retrieval import Embedder, Retriever, situation_text
from fleet_memory.memory.schema import CostReference, Episode, Lesson, Outcome, Plan, Snapshot, new_id, now_iso
from fleet_memory.memory.store import EventStore
from fleet_memory.runner import loop
from fleet_memory.runner.conditions import Condition, get_arm, is_held_out

log = logging.getLogger(__name__)
RETRIEVE_K = 5


@dataclass
class RunConfig:
    suite: str
    task_id: str
    seed: int
    arm: str
    env_kind: str = "mock"
    policy_kind: str = "mock"
    log_path: str = "logs/events.jsonl"
    probe_instruction: str | None = None
    probe_edits: list[dict] | None = None
    held_out: bool = False
    max_steps: int | None = None
    keyframe_every: int = 25
    s3_params: dict | None = None
    perturbation: dict | None = None
    skill: str | None = None
    environment_tag: str = ""
    record_trace: bool = False
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    policy_kwargs: dict[str, Any] = field(default_factory=dict)
    record_video: str | None = None          # directory: write <episode_id>.mp4 (agentview | wrist, overlay)
    video_label: str = ""                    # overlay label, e.g. "BM-4 perturbed + homing"

    @property
    def environment_id(self) -> str:
        return f"{self.suite}_{self.task_id}" + (f"_{self.environment_tag}" if self.environment_tag else "")

    @property
    def skill_instance_id(self) -> str:
        return schema.skill_instance_id(self.skill or self.task_id, self.environment_id)


# --------------------------------------------------------------------------- factories
def make_env(cfg: RunConfig):
    if cfg.env_kind == "mock":
        from fleet_memory.envs.mock_env import MockEnv
        kw = dict(cfg.env_kwargs)
        if cfg.max_steps:
            kw.setdefault("horizon", cfg.max_steps)
        return MockEnv(cfg.task_id, **kw)
    if cfg.env_kind == "libero":
        from fleet_memory.envs.libero_env import LiberoEnv
        from fleet_memory.envs.libero_plus import is_plus_backend
        if is_plus_backend():   # under the LIBERO-plus package the suite tables index 2402+ tasks: go via the base bddl
            from fleet_memory.envs.libero_plus import make_perturbed_env
            return make_perturbed_env(cfg.suite, cfg.task_id, "standard", **cfg.env_kwargs)
        return LiberoEnv(cfg.suite, cfg.task_id, **cfg.env_kwargs)
    if cfg.env_kind == "libero_plus":              # LIBERO-Plus perturbation config (envs/libero_plus.py)
        from fleet_memory.envs.libero_plus import make_perturbed_env
        kw = dict(cfg.env_kwargs)
        return make_perturbed_env(cfg.suite, cfg.task_id, kw.pop("config"), **kw)
    raise ValueError(f"unknown env_kind {cfg.env_kind!r}")


def make_policy(cfg: RunConfig):
    if cfg.policy_kind == "mock":
        from fleet_memory.policies.mock import MockPolicy
        return MockPolicy(**cfg.policy_kwargs)
    if cfg.policy_kind == "scripted":
        from fleet_memory.policies.scripted import ScriptedPolicy
        kw = dict(cfg.policy_kwargs)
        if cfg.env_kind == "mock":
            kw.setdefault("pos_scale_m", 0.02)
        return ScriptedPolicy(**kw)
    if cfg.policy_kind in ("smolvla", "pi05"):
        # FM_POLICY_KWARGS='{"n_action_steps": 10}' — chunk execution length etc., without a CLI flag per policy.
        kw = {**json.loads(os.environ.get("FM_POLICY_KWARGS", "{}")), **cfg.policy_kwargs}
        if cfg.policy_kind == "smolvla":
            from fleet_memory.policies.smolvla import SmolVLAPolicy
            return SmolVLAPolicy(**kw)
        from fleet_memory.policies.pi05 import Pi05Policy
        return Pi05Policy(**kw)
    raise ValueError(f"unknown policy_kind {cfg.policy_kind!r}")


# --------------------------------------------------------------------------- lazy v3 modules
def _house_model():
    try:
        from fleet_memory.memory import house_model
        return house_model
    except ImportError:
        return None


def _drift():
    try:
        from fleet_memory.memory import drift
        return drift
    except ImportError:
        return None


def incumbent_of(store: EventStore, si_id: str):
    hm = _house_model()
    if hm is None:
        return None
    try:
        return hm.incumbent(store, si_id)
    except Exception as ex:   # house model still being written: degrade to identity
        log.warning("house_model.incumbent failed: %s", ex)
        return None


def cost_reference_of(store: EventStore, si_id: str) -> CostReference | None:
    hm = _house_model()
    if hm is not None and hasattr(hm, "cost_reference"):
        try:
            return hm.cost_reference(store, si_id)
        except Exception as ex:
            log.warning("house_model.cost_reference failed: %s", ex)
    ref = None
    for d in store.iter_type("cost_reference"):
        if d.get("skill_instance_id") == si_id:
            ref = schema.record_from_dict(d)
    return ref


# --------------------------------------------------------------------------- per-process agents
_AGENTS: dict[str, Any] = {}


def _agent(kind: str):
    if kind not in _AGENTS:
        if kind == "planner":
            from fleet_memory.agents.planner import Planner
            _AGENTS[kind] = Planner()
        elif kind == "inner":
            from fleet_memory.agents.coach_inner import InnerCoach
            _AGENTS[kind] = InnerCoach()
        elif kind == "outer":
            from fleet_memory.agents.coach_outer import OuterCoach
            _AGENTS[kind] = OuterCoach()
        elif kind == "embedder":
            _AGENTS[kind] = Embedder()
    return _AGENTS[kind]


# --------------------------------------------------------------------------- pieces of run_episode
def retrieve_lessons(store: EventStore, task: TaskInfo, arm: Condition, environment_id: str) -> list[Lesson]:
    if not arm.memory or not arm.surfaces:
        return []
    r = Retriever(store, embedder=_agent("embedder"), k=None)
    out = [l for l in r.retrieve(task, list(arm.surfaces)) if l.trigger.environment_id in (None, environment_id)]
    return out[:RETRIEVE_K]


def seed_policy_rng(seed: int) -> None:
    np.random.seed(seed % (2 ** 32))
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def reset_env(env, cfg: RunConfig) -> Obs:
    params = inspect.signature(env.reset).parameters
    if "perturbation" in params or any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return env.reset(cfg.seed, perturbation=cfg.perturbation)
    obs = env.reset(cfg.seed)
    if cfg.perturbation:
        try:
            from fleet_memory.envs import perturb
            if hasattr(perturb, "apply"):
                obs = perturb.apply(env, cfg.perturbation, cfg.seed) or obs
            elif hasattr(perturb, "Perturbation"):
                p = perturb.Perturbation(distractor=bool(cfg.perturbation.get("distractor")),
                                         pose_jitter_m=float(cfg.perturbation.get("jitter_m") or 0.0), seed=cfg.seed)
                obs = p.apply(env) or obs
        except Exception as ex:
            log.warning("perturbation not applied: %s", ex)
    return obs


def s3_vector(cfg: RunConfig, arm: Condition, store: EventStore):
    """(S3Params, incumbent|None): explicit cfg.s3_params > incumbent (arm.use_incumbent_s3) > identity.
    The incumbent is only returned when the episode actually runs on its vector: an explicit-params episode
    (benchmark BM-2/BM-4, --s3-params) must not be attributed to its version, scorecard or drift detector."""
    inc = incumbent_of(store, cfg.skill_instance_id) if arm.use_incumbent_s3 else None
    if cfg.s3_params:
        vec = S3Params.from_dict(cfg.s3_params)
        same = inc is not None and np.allclose(vec.to_array(), S3Params.from_dict(inc.params).to_array())
        return vec, (inc if same else None)
    if inc is not None:
        return S3Params.from_dict(inc.params), inc
    return S3Params.identity(), None


def build_constraints(cfg: RunConfig, arm: Condition, vec: S3Params, applied: list[Lesson]) -> ConstraintSet:
    base = ConstraintSet()
    if "S3" in arm.surfaces:
        for l in applied:
            if l.surface == "S3" and l.edit.op in loop.S3_COACH_OPS:   # continuous S3 lessons never reach the shim (inv. 9)
                try:
                    base = apply_edit(base, l.edit)
                except Exception as ex:
                    log.warning("S3 lesson %s not applicable: %s", l.lesson_id, ex)
    for e in cfg.probe_edits or []:
        base = apply_edit(base, schema.Edit(op=e["op"], params=dict(e.get("params") or {})))
    return vec.to_constraint_set(base) if arm.shim else ConstraintSet()


def shim_kwargs(env) -> dict[str, Any]:
    """The env's metres-per-unit-dpos (MockEnv: 0.02, LIBERO: 0.05) for the shim's in-chunk EE estimate."""
    scale = getattr(env, "pos_scale_m", None)
    if scale is None or "pos_scale_m" not in inspect.signature(ExecutionShim.__init__).parameters:
        return {}
    return {"pos_scale_m": float(scale)}


def make_plan(cfg: RunConfig, arm: Condition, task: TaskInfo, obs: Obs, applied: list[Lesson]) -> Plan:
    from fleet_memory.agents.planner import identity_plan
    plan = _agent("planner").plan(task, obs, [l for l in applied if l.surface in ("S1", "S2")]) if arm.planner \
        else identity_plan(task)
    if cfg.probe_instruction:
        for st in plan.subtasks:
            st.instruction = cfg.probe_instruction
    return plan


def make_snapshot(ep: Episode, task: TaskInfo, plan: Plan, res: loop.LoopResult, cfg: RunConfig) -> Snapshot:
    text = situation_text(task, ep.instruction_used or task.language)
    emb = _agent("embedder").embed(text)
    return Snapshot(episode_id=ep.episode_id, task_id=task.task_id, situation_embedding=[float(x) for x in emb],
                    situation_text=text, instruction_used=ep.instruction_used, constraints_active=ep.constraints_active,
                    plan_taken=[s.to_dict() for s in plan.subtasks], outcome_flag=bool(ep.outcome.env_success),
                    keyframe_indices=[cfg.keyframe_every * (i + 1) for i in range(len(res.keyframes))])


# --------------------------------------------------------------------------- the episode
def run_episode(cfg: RunConfig, env=None, policy=None, store: EventStore | None = None) -> Episode:
    arm = get_arm(cfg.arm)
    store = store or EventStore(cfg.log_path)
    env = env or make_env(cfg)
    policy = policy or make_policy(cfg)
    task = env.task_info()
    env_id, si_id = cfg.environment_id, cfg.skill_instance_id
    held_out = bool(cfg.held_out or is_held_out(cfg.suite))
    episode_id = new_id("ep")
    t0 = time.perf_counter()

    # (1) retrieval, frozen before reset
    retrieved = retrieve_lessons(store, task, arm, env_id)
    frozen_at = now_iso()
    applied = [l for l in retrieved
               if l.status == "validated" or lifecycle.assign_arm(l.lesson_id, cfg.seed, cfg.task_id) == "treated"]
    # (2) reset (+ perturbation). Also seed the policy's RNG: flow-matching VLAs draw noise per inference, so
    # without this "same seed" only fixes the scene. With it, arms on the same seed share the policy's draws
    # (common random numbers) — a pass-through shim reproduces arm A exactly, and the gate compares like with like.
    seed_policy_rng(cfg.seed)
    obs = reset_env(env, cfg)
    task = env.task_info()                      # objects may change after a perturbation (distractor)
    # (3) plan
    plan = make_plan(cfg, arm, task, obs, applied)
    # (4) S3 vector -> constraints -> shim (+ envelope, always)
    vec, inc = s3_vector(cfg, arm, store)
    constraints = build_constraints(cfg, arm, vec, applied)
    tracker = Tracker(task, stall_k=40)
    shim = ExecutionShim(policy, constraints, task, tracker, enabled=arm.shim, **shim_kwargs(env))
    envelope = env.envelope() if hasattr(env, "envelope") else DEFAULT_ENVELOPE
    loop_envelope = None
    if hasattr(shim, "apply_envelope"):
        shim.apply_envelope(envelope)
    else:                                       # older shim: the loop clamps instead (never skipped)
        loop_envelope = envelope
    # (4b) v3.1 homing: an S3 parameter, so only when the shim is on. Scripted EE-space controller drives the arm
    # back to the canonical start (+delta) BEFORE the VLA acts; its steps/actions count toward the episode's cost.
    rec = None
    if cfg.record_video:
        from fleet_memory.runner.video import FrameRecorder, s3_summary
        rec = FrameRecorder(label=cfg.video_label or f"arm {cfg.arm} · {cfg.suite} {cfg.task_id} · seed {cfg.seed}",
                            subtitle=task.language, s3_summary=s3_summary(vec.to_dict() if arm.shim else None))
        rec.on_step(obs, 0, "start")
    on_step = rec.on_step if rec is not None else None
    homing_res = None
    if arm.shim and vec.homing_on and cfg.env_kind != "mock":
        from fleet_memory.execution.homing import Homing, HomingTarget
        obs, homing_res, _ = Homing(HomingTarget.canonical(vec.homing_delta(), env=env), envelope).run(env, obs, on_step=on_step)
    # (5) subtask loop
    coach = _agent("inner") if arm.inner_loop else None
    error = None
    try:
        res = loop.run_subtasks(env, obs, plan, shim, tracker, task, episode_id=episode_id, surfaces=list(arm.surfaces),
                                coach=coach, max_steps=cfg.max_steps, keyframe_every=cfg.keyframe_every,
                                envelope=loop_envelope, base_constraints=constraints, on_step=on_step)
    except Exception as ex:                     # a crashed episode is still an episode (termination="error")
        log.exception("episode %s crashed", episode_id)
        res, error = loop.LoopResult(), f"{type(ex).__name__}: {ex}"
    if homing_res is not None:                  # prepend the homing trace so metrics/cost include it
        res.steps += homing_res.steps
        res.ee_positions = list(homing_res.ee_positions) + list(res.ee_positions)
        res.actions = list(homing_res.actions) + list(res.actions)
    # (6) outcome from the env predicate ONLY
    success = bool(env.success_flag())
    if rec is not None:
        rec.finish(success, res.steps)
        out = rec.write(os.path.join(cfg.record_video, f"{episode_id}.mp4"))
        log.info("video written: %s", out)
    outcome = Outcome(env_success=success, steps=res.steps, termination="error" if error else res.termination,
                      wall_s=time.perf_counter() - t0, error=error)
    # (7) metrics
    ref = cost_reference_of(store, si_id)
    ee = np.asarray(res.ee_positions, np.float64).reshape(-1, 3) if res.ee_positions else np.zeros((0, 3))
    acts = np.asarray(res.actions, np.float32).reshape(-1, 7) if res.actions else np.zeros((0, 7), np.float32)
    metrics = cost_mod.episode_metrics(ee, acts, success, res.steps, ref)
    # (9) episode record (built before (8) so the outer coach sees the finished record)
    ep = Episode(episode_id=episode_id, suite=cfg.suite, task_id=cfg.task_id, seed=cfg.seed, condition=cfg.arm,
                 policy=getattr(policy, "name", cfg.policy_kind), surfaces_enabled=list(arm.surfaces),
                 retrieved_lesson_ids=[l.lesson_id for l in retrieved], applied_lesson_ids=[l.lesson_id for l in applied],
                 retrieval_frozen_at=frozen_at, plan=plan, instruction_used=res.instruction_used or task.language,
                 constraints_active=shim.constraints.to_dict(), interventions=list(res.interventions), outcome=outcome,
                 held_out=held_out, environment_id=env_id,
                 skill_instance_versions={si_id: int(inc.version)} if inc is not None else {},
                 s3_params=vec.to_dict(), metrics=metrics, perturbation=dict(cfg.perturbation or {}))
    # (8) outer coach: lessons only on non-held-out suites
    if arm.outer_loop and arm.generates_lessons and not held_out:
        summary = {**loop.trace_summary(res, shim, res.steps), "env_success": success}
        for l in _agent("outer").assign_credit(ep, task, res.keyframes, summary, applied):
            if l.trigger.environment_id is None:
                l.trigger.environment_id = env_id
            store.append(l)
            ep.lessons_generated.append(l.lesson_id)
    store.append(ep)
    store.append(make_snapshot(ep, task, plan, res, cfg))
    if cfg.record_trace:
        store.append({"type": "trace", "episode_id": episode_id, "ee_positions": ee.tolist(),
                      "actions": acts.tolist(), "shim_events": list(getattr(shim, "events", []))})
    # (10) scorecard + drift on the incumbent
    if arm.use_incumbent_s3 and inc is not None:
        hm, dr = _house_model(), _drift()
        try:
            if hm is not None:
                hm.update_scorecard(store, si_id, ep)
            if dr is not None:
                dr.update_and_check(store, si_id, ep)
        except Exception as ex:
            log.warning("scorecard/drift update failed: %s", ex)
    return ep


__all__ = ["RunConfig", "make_env", "make_policy", "run_episode", "incumbent_of", "cost_reference_of"]
