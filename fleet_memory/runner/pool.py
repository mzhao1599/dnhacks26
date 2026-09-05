"""Run many episodes (multiprocessing spawn, per-process env/policy cache) + the batch CLI.

python -m fleet_memory.runner.pool --env mock --policy mock --suite mock --task pick_bowl_to_plate --arm A \
    --n 40 --workers 4 --seed-set train --log logs/x.jsonl [--tasks a,b] [--probe-instruction S] [--probe-edit JSON]
    [--s3-params JSON] [--perturb JSON] [--tag ENVTAG] [--skill NAME] [--gate-every 10] [--auto-sleep]
    [--make-reference] [--protocol P --n-stage 20 --perturb JSON]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing as mp
import sys
from typing import Any

import numpy as np

from fleet_memory.analysis import cost as cost_mod
from fleet_memory.analysis.metrics import wilson_ci
from fleet_memory.memory import lifecycle
from fleet_memory.memory.schema import Edit, Episode, Intervention, now_iso, record_from_dict
from fleet_memory.memory.store import EventStore
from fleet_memory.runner.conditions import get_arm, seed_set
from fleet_memory.runner.worker import RunConfig, cost_reference_of, incumbent_of, make_env, make_policy, run_episode

_CACHE: dict[tuple, tuple[Any, Any]] = {}


def _cached(cfg: RunConfig) -> tuple[Any, Any]:
    key = (cfg.env_kind, cfg.policy_kind, cfg.suite, cfg.task_id, cfg.max_steps,
           json.dumps(cfg.env_kwargs, sort_keys=True), json.dumps(cfg.policy_kwargs, sort_keys=True))
    if key not in _CACHE:
        _CACHE[key] = (make_env(cfg), make_policy(cfg))
    return _CACHE[key]


def _run_one(d: dict) -> dict:
    cfg = RunConfig(**d)
    env, pol = _cached(cfg)
    return run_episode(cfg, env, pol).to_dict()


def episode_from_dict(d: dict) -> Episode:
    """schema._from hydrates every field named `trigger` as a Trigger, which breaks Intervention.trigger
    (a str). Hydrate interventions by hand until the contract is fixed."""
    d = dict(d)
    ivs = d.pop("interventions", None) or []
    ep = record_from_dict(d)
    ep.interventions = [Intervention(**{**i, "edit": Edit(**i["edit"])}) for i in ivs]
    return ep


_POOL: tuple[int, Any] | None = None   # (workers, Pool) — persistent so worker processes keep their env/policy cache


def _pool(workers: int):
    """One long-lived spawn pool per process; recreated only if the worker count changes."""
    global _POOL
    if _POOL is None or _POOL[0] != workers:
        close_pool()
        _POOL = (workers, mp.get_context("spawn").Pool(int(workers)))
    return _POOL[1]


def close_pool() -> None:
    global _POOL
    if _POOL is not None:
        _POOL[1].close()
        _POOL[1].join()
        _POOL = None


def run_many(cfgs: list[RunConfig], workers: int = 1) -> list[Episode]:
    """One Episode per cfg, in order. workers<=1 runs inline (same process cache). Multi-worker runs
    reuse a persistent pool: CEM calls this hundreds of times and a fresh Pool per call re-spawned
    every interpreter and threw away the loaded env/policy each time."""
    dicts = [dataclasses.asdict(c) for c in cfgs]
    if workers <= 1 or len(cfgs) <= 1:
        out = [_run_one(d) for d in dicts]
    else:
        out = _pool(int(workers)).map(_run_one, dicts, chunksize=1)
    return [episode_from_dict(d) for d in out]


# --------------------------------------------------------------------------- reporting
def summarize(eps: list[Episode], label: str = "") -> str:
    n = len(eps)
    k = sum(1 for e in eps if e.outcome and e.outcome.env_success)
    lo, hi = wilson_ci(k, n)
    steps = float(np.mean([e.outcome.steps for e in eps if e.outcome])) if n else 0.0
    costs = [e.metrics.cost for e in eps if e.metrics and e.metrics.cost is not None]
    c = f"{np.mean(costs):.3f}" if costs else "-"
    arms = ",".join(sorted({e.condition for e in eps}))
    return (f"[{label or arms}] arm={arms} n={n} success={k / n if n else 0:.2f} ci=[{lo:.2f},{hi:.2f}] "
            f"mean_steps={steps:.1f} mean_cost={c}")


# --------------------------------------------------------------------------- batch hooks
def make_reference(store: EventStore, eps: list[Episode], si_id: str, env_id: str, condition: str = "A"):
    """Freeze Arm-A means as the cost reference for si_id (metrics.jerk is raw when no ref existed)."""
    ms = [e.metrics for e in eps if e.metrics is not None and e.condition == condition]
    if not ms:
        return None
    ref = cost_mod.make_reference(si_id, env_id, ms, [m.jerk for m in ms], condition=condition)
    try:
        from fleet_memory.memory import house_model
        house_model.set_cost_reference(store, ref)
    except ImportError:
        store.append(ref)
    return ref


def ensure_incumbent(store: EventStore, cfg: RunConfig):
    inc = incumbent_of(store, cfg.skill_instance_id)
    if inc is not None:
        return inc
    from fleet_memory.execution.params import S3Params
    from fleet_memory.memory import house_model
    env = _cached(cfg)[0]
    obj = env.task_info().objects[0]
    return house_model.init_incumbent(store, cfg.skill_instance_id, cfg.environment_id, cfg.skill or cfg.task_id,
                                      obj, S3Params.identity().to_dict())


def drift_pending(store: EventStore, si_id: str) -> bool:
    """A drift_trigger for si_id with no consolidation 'end' after it."""
    pending = False
    for d in store.read_all():
        if d.get("skill_instance_id") != si_id:
            continue
        if d.get("type") == "drift_trigger":
            pending = True
        elif d.get("type") == "consolidation" and d.get("phase") == "end":
            pending = False
    return pending


def maybe_sleep(store: EventStore, cfg: RunConfig, workers: int, trigger: str = "drift"):
    if not drift_pending(store, cfg.skill_instance_id):
        return None
    from fleet_memory.runner.consolidate import ConsolidationConfig, consolidate
    print(f"  drift pending for {cfg.skill_instance_id}: consolidating (medium)", flush=True)
    return consolidate(store, cfg.skill_instance_id, template=cfg, cfg=ConsolidationConfig.medium(workers=workers),
                       trigger=trigger)


def run_batches(store: EventStore, cfgs: list[RunConfig], workers: int, gate_every: int = 0, auto_sleep: bool = False,
                label: str = "") -> list[Episode]:
    size = gate_every if gate_every and gate_every > 0 else len(cfgs)
    eps: list[Episode] = []
    for i in range(0, len(cfgs), max(size, 1)):
        batch = cfgs[i:i + size]
        got = run_many(batch, workers)
        eps.extend(got)
        print(summarize(got, label), flush=True)
        changes = lifecycle.gate_all(store)
        if changes:
            print(f"  lesson gate: {[(c.lesson_id, c.to_status) for c in changes]}", flush=True)
        if auto_sleep:
            maybe_sleep(store, batch[0], workers)
    return eps


# --------------------------------------------------------------------------- protocol P
def stage_event(store: EventStore, cfg: RunConfig, stage: str, eps: list[Episode]) -> dict:
    ev = {"type": "protocol_stage", "stage": stage, "environment_id": cfg.environment_id,
          "si_id": cfg.skill_instance_id, "ts": now_iso(), "episode_ids": [e.episode_id for e in eps],
          "perturbation": dict(cfg.perturbation or {}), "condition": cfg.arm}
    store.append(ev)
    return ev


def run_protocol(store: EventStore, base: RunConfig, perturb: dict, n_stage: int, workers: int, gate_every: int,
                 seed_name: str = "train") -> dict[str, list[Episode]]:
    """baseline (no perturb) -> perturbed (+auto-sleep) -> recovery (perturb persists, +auto-sleep)."""
    seeds = seed_set(seed_name, 3 * n_stage)
    out: dict[str, list[Episode]] = {}
    if cost_reference_of(store, base.skill_instance_id) is None:
        ref_cfgs = [dataclasses.replace(base, arm="A", seed=s, perturbation=None) for s in seed_set("probe", n_stage)]
        ref_eps = run_batches(store, ref_cfgs, workers, label="reference")
        make_reference(store, ref_eps, base.skill_instance_id, base.environment_id)
        out["reference"] = ref_eps
    ensure_incumbent(store, base)
    stages = [("baseline", None, False), ("perturbed", perturb, True), ("recovery", perturb, True)]
    for i, (name, pert, sleep) in enumerate(stages):
        cfgs = [dataclasses.replace(base, seed=s, perturbation=pert) for s in seeds[i * n_stage:(i + 1) * n_stage]]
        eps = run_batches(store, cfgs, workers, gate_every=gate_every or 10, auto_sleep=sleep, label=name)
        stage_event(store, cfgs[0], name, eps)
        out[name] = eps
    return out


# --------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Fleet Memory batch runner")
    ap.add_argument("--env", default="mock")
    ap.add_argument("--policy", default="mock")
    ap.add_argument("--suite", default="mock")
    ap.add_argument("--task", default="pick_bowl_to_plate")
    ap.add_argument("--tasks", default=None, help="comma-separated task ids (overrides --task)")
    ap.add_argument("--arm", default="A")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed-set", default="train")
    ap.add_argument("--log", default="logs/events.jsonl")
    ap.add_argument("--probe-instruction", default=None)
    ap.add_argument("--probe-edit", default=None, help="JSON edit or list of edits")
    ap.add_argument("--s3-params", default=None, help="JSON S3Params dict")
    ap.add_argument("--perturb", default=None, help="JSON perturbation dict")
    ap.add_argument("--tag", default="", help="environment tag (suffix of environment_id)")
    ap.add_argument("--skill", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--held-out", action="store_true")
    ap.add_argument("--gate-every", type=int, default=0)
    ap.add_argument("--auto-sleep", action="store_true")
    ap.add_argument("--make-reference", action="store_true")
    ap.add_argument("--protocol", default=None, choices=[None, "P"])
    ap.add_argument("--n-stage", type=int, default=20)
    ap.add_argument("--record-trace", action="store_true")
    return ap


def _json(s: str | None):
    return json.loads(s) if s else None


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    store = EventStore(a.log)
    edits = _json(a.probe_edit)
    edits = [edits] if isinstance(edits, dict) else edits
    tasks = [t for t in (a.tasks.split(",") if a.tasks else [a.task]) if t]
    for task in tasks:
        base = RunConfig(suite=a.suite, task_id=task, seed=0, arm=a.arm, env_kind=a.env, policy_kind=a.policy,
                         log_path=a.log, probe_instruction=a.probe_instruction, probe_edits=edits, held_out=a.held_out,
                         max_steps=a.max_steps, s3_params=_json(a.s3_params), perturbation=_json(a.perturb),
                         skill=a.skill, environment_tag=a.tag, record_trace=a.record_trace)
        if a.protocol == "P":
            if not base.perturbation:
                raise SystemExit("--protocol P needs --perturb JSON")
            run_protocol(store, base, base.perturbation, a.n_stage, a.workers, a.gate_every, a.seed_set)
            continue
        if get_arm(a.arm).use_incumbent_s3:          # B/D: v1 identity incumbent so episodes carry a version
            ensure_incumbent(store, base)
        cfgs = [dataclasses.replace(base, seed=s) for s in seed_set(a.seed_set, a.n)]
        eps = run_batches(store, cfgs, a.workers, a.gate_every, a.auto_sleep, label=f"{a.arm}:{task}")
        if a.make_reference:
            ref = make_reference(store, eps, base.skill_instance_id, base.environment_id, condition=a.arm)
            print(f"  cost_reference: {ref.to_dict() if ref else None}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        close_pool()
