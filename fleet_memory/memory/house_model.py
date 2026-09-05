"""Per-environment house model: versioned S3 skill instances (exactly ONE incumbent per id),
promotion / retirement, frozen cost references, running scorecards, and a read-only view for the
dashboard and the planner.

Everything is materialised by replaying the append-only log (memory/store.py). Nothing here runs a
rollout or touches a policy; promotion only happens with a passed GateResult (CLAUDE.md invariant 7).
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from fleet_memory.memory.schema import (CostReference, Episode, GateResult, Scorecard, SkillInstance,
                                        SkillInstanceStatusChange, _from, now_iso, record_from_dict)

EWMA_ALPHA = 0.3


# --------------------------------------------------------------------- materialisation
def si_id_parts(si_id: str) -> tuple[str, str]:
    """'si_{skill}__{environment_id}' -> (skill, environment_id). Best effort for odd ids."""
    body = si_id[3:] if si_id.startswith("si_") else si_id
    skill, sep, env = body.partition("__")
    return (skill, env) if sep else (body, "")


def versions(store, si_id: str) -> list[SkillInstance]:
    """All versions of a skill instance, sorted by version. Base `skill_instance` events (last emission
    of a version wins) with the latest `skill_instance_status` and `scorecard` events applied."""
    base: dict[int, SkillInstance] = {}
    for d in store.iter_type("skill_instance"):
        if d.get("skill_instance_id") != si_id:
            continue
        try:
            si = record_from_dict(d)
        except (KeyError, TypeError):
            continue
        base[int(si.version)] = si
    for d in store.iter_type("skill_instance_status"):
        if d.get("skill_instance_id") == si_id and d.get("version") in base and d.get("to_status"):
            base[d["version"]].status = d["to_status"]            # file order => last wins
    for d in store.iter_type("scorecard"):
        if d.get("skill_instance_id") == si_id and d.get("version") in base and isinstance(d.get("scorecard"), dict):
            base[d["version"]].scorecard = _from(Scorecard, d["scorecard"])
    return [base[v] for v in sorted(base)]


def incumbent(store, si_id: str) -> SkillInstance | None:
    """The single incumbent. If the log were ever inconsistent the highest incumbent version wins."""
    inc = [v for v in versions(store, si_id) if v.status == "incumbent"]
    return inc[-1] if inc else None


def skill_instance_ids(store, environment_id: str | None = None) -> list[str]:
    ids: list[str] = []
    for d in store.iter_type("skill_instance"):
        sid = d.get("skill_instance_id")
        if sid and sid not in ids and (environment_id is None or d.get("environment_id") == environment_id):
            ids.append(sid)
    return ids


def all_environments(store) -> list[str]:
    envs: set[str] = set()
    for d in store.read_all():
        if d.get("type") in ("skill_instance", "cost_reference", "episode") and d.get("environment_id"):
            envs.add(d["environment_id"])
    return sorted(envs)


# --------------------------------------------------------------------- writes
def init_incumbent(store, si_id: str, environment_id: str, skill: str, object_instance_id: str,
                   params: dict) -> SkillInstance:
    """Create the incumbent (v1, or max+1 if every earlier version was retired) only if none exists.
    Idempotent: an existing incumbent is returned untouched."""
    cur = incumbent(store, si_id)
    if cur is not None:
        return cur
    vs = versions(store, si_id)
    si = SkillInstance(skill_instance_id=si_id, environment_id=environment_id, skill=skill,
                       object_instance_id=object_instance_id, version=(vs[-1].version + 1) if vs else 1,
                       parent_version=None, status="incumbent", params=dict(params),
                       produced_by={"loop": "init"})
    store.append(si)
    return si


def promote(store, si_id: str, params: dict, gate: GateResult, consolidation_id: str) -> SkillInstance:
    """New version (max+1) becomes incumbent; the old incumbent is retired first so the log never
    holds two incumbents. Refuses a failed gate."""
    if not gate.passed:
        raise ValueError(f"refusing to promote {si_id}: gate did not pass")
    vs = versions(store, si_id)
    old = incumbent(store, si_id)
    ref = old or (vs[-1] if vs else None)
    skill, env = si_id_parts(si_id)
    new = SkillInstance(
        skill_instance_id=si_id,
        environment_id=ref.environment_id if ref else env,
        skill=ref.skill if ref else skill,
        object_instance_id=ref.object_instance_id if ref else "",
        version=(vs[-1].version + 1) if vs else 1,
        parent_version=old.version if old else None,
        status="incumbent", params=dict(params),
        produced_by={"loop": "sleep", "consolidation_id": consolidation_id},
        gate=gate,
        scorecard=Scorecard(baseline_cost=float(gate.candidate_cost),
                            baseline_success_rate=float(gate.candidate_success)))
    if old is not None:
        store.append(SkillInstanceStatusChange(skill_instance_id=si_id, version=old.version,
                                               from_status="incumbent", to_status="retired",
                                               reason=f"superseded by v{new.version} ({consolidation_id})"))
    store.append(new)
    return new


def retire(store, si_id: str, version: int, reason: str) -> SkillInstanceStatusChange | None:
    for v in versions(store, si_id):
        if v.version == version and v.status != "retired":
            ch = SkillInstanceStatusChange(skill_instance_id=si_id, version=version, from_status=v.status,
                                           to_status="retired", reason=reason)
            store.append(ch)
            return ch
    return None


# --------------------------------------------------------------------- cost reference
def cost_reference(store, si_id: str) -> CostReference | None:
    last = None
    for d in store.iter_type("cost_reference"):
        if d.get("skill_instance_id") == si_id:
            last = d
    return record_from_dict(last) if last else None


def set_cost_reference(store, ref: CostReference) -> None:
    store.append(ref)


# --------------------------------------------------------------------- episodes & scorecard
def is_rollout(ep: Episode | dict) -> bool:
    """Consolidation rollouts are tagged in Episode.perturbation; they never feed scorecards or curves."""
    p = ep.perturbation if isinstance(ep, Episode) else (ep.get("perturbation") or {})
    return isinstance(p, dict) and "consolidation_id" in p


def belongs(ep: Episode | dict, si_id: str, version: int | None = None) -> bool:
    versions_used = (ep.skill_instance_versions if isinstance(ep, Episode) else ep.get("skill_instance_versions")) or {}
    v = versions_used.get(si_id)
    return v is not None and (version is None or int(v) == int(version)) and not is_rollout(ep)


def version_episodes(store, si_id: str, version: int, extra: Episode | None = None) -> list[Episode]:
    """Episodes that ran on this version, in log order; `extra` (not yet logged) is appended if it belongs."""
    eps = [e for e in store.episodes() if belongs(e, si_id, version)]
    if extra is not None and belongs(extra, si_id, version) and all(e.episode_id != extra.episode_id for e in eps):
        eps.append(extra)
    return eps


def episode_success(ep: Episode) -> bool | None:
    if ep.outcome is not None:
        return bool(ep.outcome.env_success)
    return bool(ep.metrics.success) if ep.metrics is not None else None


def episode_cost(ep: Episode) -> float | None:
    return None if ep.metrics is None or ep.metrics.cost is None else float(ep.metrics.cost)


def ewma(values: Iterable[float], alpha: float = EWMA_ALPHA) -> float | None:
    out = None
    for v in values:
        out = float(v) if out is None else alpha * float(v) + (1.0 - alpha) * out
    return out


def compute_scorecard(episodes: list[Episode], base: Scorecard | None = None,
                      alpha: float = EWMA_ALPHA) -> Scorecard:
    """Running stats over one version's episodes; baselines are carried from `base` (set at promotion)."""
    sc = Scorecard(baseline_cost=base.baseline_cost if base else None,
                   baseline_success_rate=base.baseline_success_rate if base else None)
    oks = [s for s in (episode_success(e) for e in episodes) if s is not None]
    sc.n, sc.successes = len(oks), int(sum(oks))
    ms = [e.metrics for e in episodes if e.metrics is not None]
    if ms:
        sc.mean_steps = float(np.mean([m.steps for m in ms]))
        sc.mean_jerk = float(np.mean([m.jerk for m in ms]))
        sc.mean_force_proxy = float(np.mean([m.force_proxy for m in ms]))
    sc.ewma_cost = ewma([m.cost for m in ms if m.cost is not None], alpha)
    return sc


def update_scorecard(store, si_id: str, episode: Episode, alpha: float = EWMA_ALPHA) -> Scorecard:
    """Recompute the incumbent's scorecard including `episode` and append a `scorecard` event
    (the latest one is authoritative). Raises if there is no incumbent."""
    inc = incumbent(store, si_id)
    if inc is None:
        raise ValueError(f"no incumbent skill instance for {si_id}")
    eps = version_episodes(store, si_id, inc.version, extra=episode)
    sc = compute_scorecard(eps, inc.scorecard, alpha)
    store.append({"type": "scorecard", "skill_instance_id": si_id, "version": inc.version,
                  "episode_id": episode.episode_id, "scorecard": sc.to_dict(), "ts": now_iso()})
    return sc


# --------------------------------------------------------------------- view
def _si_summary(store, sid: str) -> dict[str, Any]:
    vs = versions(store, sid)
    inc = incumbent(store, sid)
    ref = cost_reference(store, sid)
    any_v = inc or (vs[-1] if vs else None)
    return {"skill_instance_id": sid,
            "skill": any_v.skill if any_v else si_id_parts(sid)[0],
            "object_instance_id": any_v.object_instance_id if any_v else "",
            "incumbent_version": inc.version if inc else None,
            "params": dict(inc.params) if inc else None,
            "scorecard": inc.scorecard.to_dict() if inc else None,
            "n_versions": len(vs),
            "versions": [{"version": v.version, "status": v.status, "parent_version": v.parent_version,
                          "produced_by": v.produced_by, "gate": v.gate.to_dict() if v.gate else None}
                         for v in vs],
            "cost_reference": ref.to_dict() if ref else None}


def view(store, environment_id: str) -> dict[str, Any]:
    """Everything the dashboard / planner needs to know about one environment. Pure read."""
    ids = skill_instance_ids(store, environment_id)
    idset = set(ids)
    lessons = [l.to_dict() for l in store.lessons().values()
               if l.status != "retired" and l.trigger.environment_id in (None, environment_id)]
    cons = [d for d in store.iter_type("consolidation")
            if d.get("skill_instance_id") in idset and d.get("phase") == "end"]
    drifts = [d for d in store.iter_type("drift_trigger") if d.get("skill_instance_id") in idset]
    eps = [e for e in store.episodes() if e.environment_id == environment_id and not is_rollout(e)]
    oks = [s for s in (episode_success(e) for e in eps) if s is not None]
    costs = [c for c in (episode_cost(e) for e in eps) if c is not None]
    return {
        "environment_id": environment_id,
        "tasks": sorted({e.task_id for e in eps}),
        "objects": sorted({s.object_instance_id for sid in ids for s in versions(store, sid) if s.object_instance_id}),
        "skill_instances": [_si_summary(store, sid) for sid in ids],
        "lessons": lessons,
        "consolidations": cons,
        "drift_triggers": drifts,
        "episodes": {"n": len(eps),
                     "success_rate": float(np.mean(oks)) if oks else None,
                     "mean_cost": float(np.mean(costs)) if costs else None},
    }
