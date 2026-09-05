"""v3 mastery / sleep-loop / readaptation metrics over the event log. Read-only.

Re-exported by analysis/metrics.py (`metrics.mastery_curve` etc.); kept separate so metrics.py
stays small. Imports of metrics.py are lazy to avoid a circular import.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

STAGES = ("baseline", "perturbed", "recovery")


def _recs(store) -> list[dict]:
    from fleet_memory.analysis.metrics import _records
    return _records(store)


def _episodes(recs: list[dict]) -> list[dict]:
    return [r for r in recs if r.get("type") == "episode" and r.get("outcome")]


def is_rollout(e: dict) -> bool:
    """Consolidation rollouts carry {"consolidation_id": ...} in Episode.perturbation."""
    return bool((e.get("perturbation") or {}).get("consolidation_id"))


def _refs(recs: list[dict]) -> dict[str, dict]:
    """environment_id -> latest cost_reference record."""
    return {r["environment_id"]: r for r in recs if r.get("type") == "cost_reference" and r.get("environment_id")}


def _cost(e: dict, refs: dict[str, dict] | None = None) -> float | None:
    """metrics.cost, or the same formula applied post hoc (analysis/cost.py weights) for episodes logged
    before their environment's cost_reference existed (Phase 0 arm A: metrics.jerk is then raw)."""
    m = e.get("metrics") or {}
    if m.get("cost") is not None:
        return m["cost"]
    ref = (refs or {}).get(e.get("environment_id"))
    if not ref or not m or not e.get("outcome"):
        return None
    jerk = m.get("jerk", 0.0) / ref["jerk_ref"] if ref.get("jerk_ref") else m.get("jerk", 0.0)
    return (1.0 * m.get("steps", 0) / max(ref.get("steps_ref", 1.0), 1.0) + 0.5 * jerk
            + 0.5 * m.get("force_proxy", 0.0) + 3.0 * (0.0 if e["outcome"].get("env_success") else 1.0))


def _mean(xs: list[float]) -> float | None:
    return (sum(xs) / len(xs)) if xs else None


def _sorted_arms(arms):
    from fleet_memory.analysis.metrics import _sorted_arms
    return _sorted_arms(arms)


# --------------------------------------------------------------------------- #
# Mastery
# --------------------------------------------------------------------------- #

def environments(store) -> list[str]:
    """environment_ids seen in episodes, sorted."""
    return sorted({e.get("environment_id") for e in _episodes(_recs(store)) if e.get("environment_id")})


def mastery_curve(store, environment_id: str, arm: str | None = None) -> list[dict[str, Any]]:
    """Per-episode cost/steps/success/version for one environment in log order, consolidation
    rollouts excluded. `i` is the 0-based index among the returned episodes."""
    out: list[dict] = []
    recs = _recs(store)
    refs = _refs(recs)
    for e in _episodes(recs):
        if e.get("environment_id") != environment_id or is_rollout(e):
            continue
        if arm is not None and e.get("condition") != arm:
            continue
        vers = e.get("skill_instance_versions") or {}
        out.append({"i": len(out), "episode_id": e["episode_id"], "arm": e.get("condition"), "cost": _cost(e, refs),
                    "steps": e["outcome"].get("steps"), "success": bool(e["outcome"].get("env_success")),
                    "version": max(vers.values()) if vers else None})
    return out


def gate_pass_rate(store) -> dict[str, Any]:
    """Consolidation 'end' events: attempted vs promoted (a promoted_version, or gate.passed)."""
    ends = [r for r in _recs(store) if r.get("type") == "consolidation" and r.get("phase") == "end"]
    promoted = sum(1 for r in ends if r.get("promoted_version") is not None or (r.get("gate") or {}).get("passed"))
    return {"attempted": len(ends), "promoted": promoted, "rate": (promoted / len(ends)) if ends else None}


def versions(store, environment_id: str | None = None) -> list[dict[str, Any]]:
    """skill_instance events (optionally for one environment) with the latest status applied."""
    rows: dict[tuple, dict] = {}
    for r in _recs(store):
        if r.get("type") == "skill_instance" and (environment_id is None or r.get("environment_id") == environment_id):
            rows[(r["skill_instance_id"], r["version"])] = dict(r)
        elif r.get("type") == "skill_instance_status" and (r.get("skill_instance_id"), r.get("version")) in rows:
            rows[(r["skill_instance_id"], r["version"])]["status"] = r.get("to_status")
    return sorted(rows.values(), key=lambda r: (r["skill_instance_id"], r["version"]))


# --------------------------------------------------------------------------- #
# Perturbation protocol
# --------------------------------------------------------------------------- #

def _rolling(xs: list[float], w: int) -> list[float]:
    return [sum(xs[max(0, i - w + 1): i + 1]) / len(xs[max(0, i - w + 1): i + 1]) for i in range(len(xs))]


def _stage_events(recs: list[dict], environment_id: str) -> dict[str, tuple[int, dict]]:
    """Latest protocol run for the environment: stage -> (log index, event). A new 'baseline' resets."""
    out: dict[str, tuple[int, dict]] = {}
    for i, r in enumerate(recs):
        if r.get("type") != "protocol_stage" or r.get("environment_id") != environment_id:
            continue
        if r.get("stage") == "baseline":
            out = {}
        out[r.get("stage")] = (i, r)
    return out


def readaptation(store, environment_id: str, window: int = 5, tol: float = 0.10) -> dict[str, Any]:
    """Recovery after the P protocol's perturbation. pre_cost = mean cost of the baseline stage;
    the post-perturbation series (perturbed + recovery stages) is smoothed with a rolling-`window`
    mean; recovered = first full window at/after the peak within `tol` of pre_cost. wallclock_s =
    episode wall time up to recovery plus consolidations that ended in between."""
    res: dict[str, Any] = {"pre_cost": None, "perturbed_peak_cost": None, "recovered_at_episode": None,
                           "episodes_to_recover": None, "wallclock_s": None, "n_post": 0, "consolidations": 0}
    recs = _recs(store)
    st = _stage_events(recs, environment_id)
    if "baseline" not in st or "perturbed" not in st:
        return res
    pos = {r["episode_id"]: i for i, r in enumerate(recs) if r.get("type") == "episode" and r.get("outcome")}
    refs = _refs(recs)
    ids = lambda s: [i for i in ((st.get(s) or (0, {}))[1].get("episode_ids") or []) if i in pos]  # noqa: E731
    pre = [c for c in (_cost(recs[pos[i]], refs) for i in ids("baseline")) if c is not None]
    post = [recs[pos[i]] for i in ids("perturbed") + ids("recovery") if _cost(recs[pos[i]], refs) is not None]
    res["n_post"] = len(post)
    if not pre or not post:
        return res
    pre_cost = _mean(pre)
    roll = _rolling([_cost(e, refs) for e in post], window)
    peak = max(range(len(roll)), key=roll.__getitem__)
    res.update(pre_cost=pre_cost, perturbed_peak_cost=roll[peak])
    rec = next((i for i in range(peak, len(roll)) if i >= window - 1 and roll[i] <= (1.0 + tol) * pre_cost), None)
    b_idx, si_id = st["baseline"][0], st["baseline"][1].get("si_id")
    end = pos[post[rec if rec is not None else -1]["episode_id"]] + 1     # window: baseline event .. recovery (or last post episode)
    cons = [r for r in recs[b_idx:end] if r.get("type") == "consolidation" and r.get("phase") == "end"
            and (r.get("skill_instance_id") == si_id if si_id else str(r.get("skill_instance_id", "")).endswith("__" + environment_id))]
    res["consolidations"] = len(cons)
    if rec is not None:
        res.update(recovered_at_episode=post[rec]["episode_id"], episodes_to_recover=rec + 1,
                   wallclock_s=sum((e["outcome"].get("wall_s") or 0.0) for e in post[: rec + 1])
                   + sum((r.get("wallclock_s") or 0.0) for r in cons))
    return res


def perturbed_success(store, environment_id: str) -> dict[str, float]:
    """{arm: success rate} over perturbed (non-rollout) episodes of the environment."""
    acc: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for e in _episodes(_recs(store)):
        if e.get("environment_id") != environment_id or is_rollout(e):
            continue
        if not any(v not in (None, False, 0, {}, []) for v in (e.get("perturbation") or {}).values()):
            continue
        acc[e["condition"]][0] += 1
        acc[e["condition"]][1] += int(bool(e["outcome"].get("env_success")))
    return {a: acc[a][1] / acc[a][0] for a in _sorted_arms(acc)}


# --------------------------------------------------------------------------- #
# Per-environment report
# --------------------------------------------------------------------------- #

def environment_report(store) -> dict[str, dict[str, Any]]:
    out = {}
    for env in environments(store):
        curve = mastery_curve(store, env)
        rollouts = sum(1 for e in _episodes(_recs(store)) if e.get("environment_id") == env and is_rollout(e))
        arms: dict[str, dict] = {}
        for a in _sorted_arms({p["arm"] for p in curve}):
            pts = [p for p in curve if p["arm"] == a]
            costs = [p["cost"] for p in pts if p["cost"] is not None]
            arms[a] = {"n": len(pts), "success_rate": _mean([float(p["success"]) for p in pts]),
                       "mean_cost": _mean(costs), "first10_cost": _mean(costs[:10]), "last10_cost": _mean(costs[-10:]),
                       "mean_steps": _mean([float(p["steps"] or 0) for p in pts])}
        out[env] = {"n": len(curve), "rollouts": rollouts, "arms": arms, "versions": len(versions(store, env)),
                    "readaptation": readaptation(store, env), "perturbed_success": perturbed_success(store, env)}
    return out


def environment_lines(store) -> list[str]:
    """Markdown lines for results_table: sleep-loop gate rate, then one block per environment."""
    from fleet_memory.analysis.metrics import _num, _pct
    g = gate_pass_rate(store)
    L = ["", "## Sleep loop", f"consolidations {g['attempted']}, promoted {g['promoted']}, gate pass rate {_pct(g['rate'])}"]
    for env, r in environment_report(store).items():
        L += ["", f"### {env}", f"episodes {r['n']} (+{r['rollouts']} consolidation rollouts) · skill-instance versions {r['versions']}",
              "| arm | n | success | mean cost | first 10 | last 10 | mean steps |", "|---|---|---|---|---|---|---|"]
        L += [f"| {a} | {x['n']} | {_pct(x['success_rate'])} | {_num(x['mean_cost'], 3)} | {_num(x['first10_cost'], 3)} | "
              f"{_num(x['last10_cost'], 3)} | {_num(x['mean_steps'])} |" for a, x in r["arms"].items()]
        ra = r["readaptation"]
        if ra["pre_cost"] is not None:
            L.append(f"readaptation: pre {_num(ra['pre_cost'], 3)} -> peak {_num(ra['perturbed_peak_cost'], 3)}, "
                     + (f"recovered at {ra['recovered_at_episode']} after {ra['episodes_to_recover']} episodes "
                        f"({_num(ra['wallclock_s'])} s, {ra['consolidations']} consolidations)"
                        if ra["recovered_at_episode"] else f"not recovered in {ra['n_post']} episodes"))
        if r["perturbed_success"]:
            L.append("perturbed success: " + ", ".join(f"{a} {_pct(v)}" for a, v in r["perturbed_success"].items()))
    return L
