"""Metrics over the append-only event log. Read-only: nothing here feeds back into a run.

`store` is anything with read_all() -> list[dict] (memory/store.py:EventStore), a path to an
events.jsonl, or an in-memory list of records/dicts. Success is read from Outcome.env_success only.

CLI: python -m fleet_memory.analysis.metrics --log logs/events.jsonl [--json]
The latest `benchmark_result` (runner/benchmark.py) is reported first when the log has one.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from typing import Any

from fleet_memory.memory.schema import Record
# v3 (analysis/mastery.py): mastery_curve, gate_pass_rate, readaptation, perturbed_success, versions, ...
from fleet_memory.analysis.mastery import (environment_lines, environment_report, environments,  # noqa: F401
                                           gate_pass_rate, is_rollout, mastery_curve, perturbed_success,
                                           readaptation, versions)

ARM_ORDER = ["A", "B", "C", "D", "P", "E", "F", "D_S1", "D_S2", "D_S3"]
BENCH_ARMS = ["BM-0", "BM-1", "BM-2", "BM-3", "BM-4"]   # runner/benchmark.py (spec §13)
STATUSES = ("candidate", "validated", "retired")


# --------------------------------------------------------------------------- #
# Log access
# --------------------------------------------------------------------------- #

def load_store(path: str):
    """EventStore when memory/store.py is importable, else the path itself (read directly)."""
    try:
        from fleet_memory.memory.store import EventStore
        return EventStore(path)
    except ImportError:
        return path


def _records(store) -> list[dict]:
    if hasattr(store, "read_all"):
        return [r.to_dict() if isinstance(r, Record) else r for r in store.read_all()]
    if isinstance(store, (str, os.PathLike)):
        out: list[dict] = []
        if os.path.exists(store):
            with open(store, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # torn line from a concurrent writer: skip, never fail
        return out
    return [r.to_dict() if isinstance(r, Record) else r for r in store]


def episode_records(store) -> list[dict]:
    """Episode dicts that have an outcome, in log order."""
    return [r for r in _records(store) if r.get("type") == "episode" and r.get("outcome")]


def lesson_records(store) -> dict[str, dict]:
    """Materialised lessons: base 'lesson' events with the latest 'lesson_status' applied."""
    out: dict[str, dict] = {}
    for r in _records(store):
        if r.get("type") == "lesson":
            out[r["lesson_id"]] = dict(r)
        elif r.get("type") == "lesson_status" and r.get("lesson_id") in out:
            lesson = out[r["lesson_id"]]
            lesson["status"] = r["to_status"]
            if r.get("evidence"):
                lesson["evidence"] = r["evidence"]
    return out


def _sorted_arms(arms) -> list[str]:
    return sorted(arms, key=lambda a: (ARM_ORDER.index(a) if a in ARM_ORDER else len(ARM_ORDER), a))


def _ok(e: dict) -> bool:
    return bool(e["outcome"]["env_success"])


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials. n == 0 -> (0, 1)."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def two_proportion_p(k1: int, n1: int, k2: int, n2: int) -> float | None:
    """One-sided pooled z-test P(treated > control) under H0, normal approximation. None if a group is empty."""
    if n1 <= 0 or n2 <= 0:
        return None
    p1, p2, p = k1 / n1, k2 / n2, (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0.0:
        return 1.0  # all-success or all-failure in both groups: no evidence either way
    return 0.5 * math.erfc(((p1 - p2) / se) / math.sqrt(2))


# --------------------------------------------------------------------------- #
# Success
# --------------------------------------------------------------------------- #

def success_by_arm(store) -> dict[str, dict[str, Any]]:
    """{arm: {n, k, rate, ci}} from Outcome.env_success, arms in ARM_ORDER."""
    acc: dict[str, dict] = defaultdict(lambda: {"n": 0, "k": 0})
    for e in episode_records(store):
        a = acc[e["condition"]]
        a["n"] += 1
        a["k"] += int(_ok(e))
    out = {}
    for arm in _sorted_arms(acc):
        a = acc[arm]
        out[arm] = {"n": a["n"], "k": a["k"], "rate": a["k"] / a["n"], "ci": wilson_ci(a["k"], a["n"])}
    return out


def success_curve(store, arm: str, window: int = 20) -> list[tuple[int, float]]:
    """Trailing-window success rate for `arm` in log order. Index is 1-based (episodes of `arm` so far)."""
    flags = [int(_ok(e)) for e in episode_records(store) if e["condition"] == arm]
    pts = []
    for i in range(len(flags)):
        w = flags[max(0, i - window + 1): i + 1]
        pts.append((i + 1, sum(w) / len(w)))
    return pts


# --------------------------------------------------------------------------- #
# Benchmark (runner/benchmark.py: BM-0..BM-4 on LIBERO-Plus)
# --------------------------------------------------------------------------- #

def benchmark_records(store) -> list[dict]:
    """All 'benchmark_result' events in log order."""
    return [r for r in _records(store) if r.get("type") == "benchmark_result"]


def benchmark_summary(store) -> dict[str, Any] | None:
    """The LATEST benchmark_result (last in the log) with its header fields, one flat row per arm
    (BM-0..BM-4 order) and the BM-3/BM-1 risk ratio. None when the log has no benchmark yet."""
    recs = benchmark_records(store)
    if not recs:
        return None
    b = recs[-1]
    rows = []
    for r in sorted(b.get("results") or [],
                    key=lambda r: (BENCH_ARMS.index(r["arm"]) if r["arm"] in BENCH_ARMS else len(BENCH_ARMS), r["arm"])):
        n, k = int(r.get("n") or 0), int(r.get("k") or 0)
        ci = r.get("ci") or wilson_ci(k, n)
        rows.append({"arm": r["arm"], "n": n, "k": k, "rate": r.get("rate", (k / n) if n else 0.0),
                     "ci": [float(ci[0]), float(ci[1])], "mean_steps": r.get("mean_steps"),
                     "episodes": len(r.get("episode_ids") or []), "s3_params": r.get("s3_params")})
    ratio = b.get("ratio") or None
    return {"ts": b.get("ts"), "suite": b.get("suite"), "tasks": list(b.get("tasks") or []),
            "dimension": b.get("dimension"), "config_index": b.get("config_index"), "config": b.get("config"),
            "policy": b.get("policy"), "tag": b.get("tag"), "n_per_task": b.get("n_per_task"),
            "n_runs": len(recs), "rows": rows,
            "ratio": None if ratio is None else {"BM-3/BM-1": ratio.get("BM-3/BM-1"),
                                                 "ci": list(ratio.get("ci") or [None, None]),
                                                 "verdict": ratio.get("verdict")}}


def benchmark_lines(store) -> list[str]:
    """Markdown block for the latest benchmark; [] when there is none."""
    b = benchmark_summary(store)
    if b is None:
        return []
    L = ["## Benchmark (LIBERO-Plus)",
         f"{b['policy']} on {b['suite']} · tasks {','.join(str(t) for t in b['tasks'])} · {b['dimension']} "
         f"#{b['config_index']} · {b['n_per_task']} eval seeds/task · tag {b['tag']} · {b['ts']}"
         + (f" · latest of {b['n_runs']} runs" if b["n_runs"] > 1 else ""),
         "| arm | n | success | 95% CI | mean steps |", "|---|---|---|---|---|"]
    L += [f"| {r['arm']} | {r['n']} | {_pct(r['rate'])} ({r['k']}) | {_ci(r['ci'])} | {_num(r['mean_steps'], 0)} |"
          for r in b["rows"]]
    rr = b["ratio"]
    if rr and rr["BM-3/BM-1"] is not None:
        L.append(f"BM-3 / BM-1 = {rr['BM-3/BM-1']:.2f} [{_num(rr['ci'][0], 2)}, {_num(rr['ci'][1], 2)}] "
                 f"-> {rr['verdict']}")
    return L + [""]


# --------------------------------------------------------------------------- #
# Lessons
# --------------------------------------------------------------------------- #

def _ab_counts(eps: list[dict]) -> dict[str, list[int]]:
    """lesson_id -> [treated_n, treated_successes, control_n, control_successes]. Retrieved = trial;
    applied = treated, withheld = control (the same split lifecycle.update_evidence uses)."""
    ab: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for e in eps:
        applied = set(e.get("applied_lesson_ids") or [])
        ok = int(_ok(e))
        for lid in e.get("retrieved_lesson_ids") or []:
            c = ab[lid]
            if lid in applied:
                c[0] += 1
                c[1] += ok
            else:
                c[2] += 1
                c[3] += ok
    return ab


def _evidence(tn: int, ts: int, cn: int, cs: int) -> dict[str, Any]:
    tr = ts / tn if tn else None
    cr = cs / cn if cn else None
    return {"trials": tn + cn, "treated_n": tn, "treated_successes": ts, "control_n": cn,
            "control_successes": cs, "treated_rate": tr, "control_rate": cr,
            "lift_pp": None if tr is None or cr is None else (tr - cr) * 100.0,
            "p_value": two_proportion_p(ts, tn, cs, cn)}


def lesson_evidence(store, lesson_id: str) -> dict[str, Any]:
    """Matched A/B evidence for one lesson, recomputed from episodes."""
    return _evidence(*_ab_counts(episode_records(store)).get(lesson_id, [0, 0, 0, 0]))


def lesson_rows(store) -> list[dict[str, Any]]:
    """One flat row per lesson (table shape). Evidence is recomputed from episodes; the stored
    evidence is used only when no episode in this log references the lesson."""
    ab = _ab_counts(episode_records(store))
    rows = []
    for lid, l in lesson_records(store).items():
        t, ed = l.get("trigger") or {}, l.get("edit") or {}
        ev = _evidence(*ab.get(lid, [0, 0, 0, 0]))
        stored = l.get("evidence") or {}
        if ev["trials"] == 0 and (stored.get("treated_n", 0) or stored.get("control_n", 0)):
            ev = _evidence(stored.get("treated_n", 0), stored.get("treated_successes", 0),
                           stored.get("control_n", 0), stored.get("control_successes", 0))
        rows.append({"lesson_id": lid, "surface": l.get("surface"), "op": ed.get("op"),
                     "params": ed.get("params") or {},
                     "trigger": "/".join(str(t.get(k, "*")) for k in ("task_family", "object_class", "phase", "predicate")),
                     "status": l.get("status", "candidate"), "rationale": l.get("rationale_text", ""),
                     "ts": l.get("ts", ""), **ev})
    order = {s: i for i, s in enumerate(("validated", "candidate", "retired"))}
    rows.sort(key=lambda r: (order.get(r["status"], 9), r["lesson_id"]))
    return rows


def lesson_precision(store) -> dict[str, Any]:
    """Counts by status; precision = validated / (validated + retired), None before any gate decision."""
    c = Counter(l.get("status", "candidate") for l in lesson_records(store).values())
    decided = c["validated"] + c["retired"]
    return {"candidates": c["candidate"], "validated": c["validated"], "retired": c["retired"],
            "precision": (c["validated"] / decided) if decided else None}


# --------------------------------------------------------------------------- #
# Interventions
# --------------------------------------------------------------------------- #

def rescue_rate(store) -> dict[str, Any]:
    """Success among episodes that had >= 1 inner-loop intervention, vs episodes in the same arms without one."""
    eps = episode_records(store)
    with_iv = [e for e in eps if e.get("interventions")]
    arms = {e["condition"] for e in with_iv}
    without = [e for e in eps if e["condition"] in arms and not e.get("interventions")]
    n, k = len(with_iv), sum(_ok(e) for e in with_iv)
    by_arm: dict[str, dict] = defaultdict(lambda: {"n": 0, "k": 0})
    by_trigger: dict[str, dict] = defaultdict(lambda: {"n": 0, "k": 0})
    for e in with_iv:
        by_arm[e["condition"]]["n"] += 1
        by_arm[e["condition"]]["k"] += int(_ok(e))
        for trig in {iv.get("trigger", "?") for iv in e["interventions"]}:
            by_trigger[trig]["n"] += 1
            by_trigger[trig]["k"] += int(_ok(e))
    for d in list(by_arm.values()) + list(by_trigger.values()):
        d["rate"] = d["k"] / d["n"]
    return {"n": n, "k": k, "rate": (k / n) if n else None, "ci": wilson_ci(k, n),
            "baseline_n": len(without),
            "baseline_rate": (sum(_ok(e) for e in without) / len(without)) if without else None,
            "by_arm": {a: by_arm[a] for a in _sorted_arms(by_arm)}, "by_trigger": dict(by_trigger)}


def intervention_efficiency(store, treated: str = "D", baseline: str = "B") -> dict[str, Any]:
    """Coach tokens spent in `treated` per percentage point of success gained over `baseline`.
    Inner-coach tokens come from Intervention records; outer-coach tokens from Lesson.provenance
    (tokens_in/tokens_out) when the coach recorded them."""
    sba = success_by_arm(store)
    eps = [e for e in episode_records(store) if e["condition"] == treated]
    ids = {e["episode_id"] for e in eps}
    ivs = [iv for e in eps for iv in e.get("interventions") or []]
    inner = sum((iv.get("tokens_in") or 0) + (iv.get("tokens_out") or 0) for iv in ivs)
    outer = sum((l.get("provenance") or {}).get("tokens_in", 0) + (l.get("provenance") or {}).get("tokens_out", 0)
                for l in lesson_records(store).values()
                if ids & set((l.get("provenance") or {}).get("from_episodes") or []))
    rt = sba.get(treated, {}).get("rate")
    rb = sba.get(baseline, {}).get("rate")
    gain = None if rt is None or rb is None else (rt - rb) * 100.0
    total = inner + outer
    return {"treated_arm": treated, "baseline_arm": baseline, "treated_rate": rt, "baseline_rate": rb,
            "gain_pp": gain, "episodes": len(eps), "interventions": len(ivs),
            "inner_tokens": inner, "outer_tokens": outer, "tokens": total,
            "tokens_per_episode": (total / len(eps)) if eps else None,
            "tokens_per_pp": (total / gain) if gain and gain > 0 else None}


def latency_cost(store) -> dict[str, dict[str, Any]]:
    """Per arm: mean steps and wall-clock per episode, and the coach's share (interventions, latency, tokens)."""
    acc: dict[str, dict] = defaultdict(lambda: {"n": 0, "steps": 0, "wall_s": 0.0, "iv": 0, "coach_s": 0.0, "tok": 0})
    for e in episode_records(store):
        a, o = acc[e["condition"]], e["outcome"]
        a["n"] += 1
        a["steps"] += o.get("steps") or 0
        a["wall_s"] += o.get("wall_s") or 0.0
        for iv in e.get("interventions") or []:
            a["iv"] += 1
            a["coach_s"] += iv.get("latency_s") or 0.0
            a["tok"] += (iv.get("tokens_in") or 0) + (iv.get("tokens_out") or 0)
    out = {}
    for arm in _sorted_arms(acc):
        a, n = acc[arm], acc[arm]["n"]
        out[arm] = {"n": n, "mean_steps": a["steps"] / n, "mean_wall_s": a["wall_s"] / n,
                    "interventions_per_episode": a["iv"] / n, "coach_s_per_episode": a["coach_s"] / n,
                    "tokens_per_episode": a["tok"] / n,
                    "coach_share": (a["coach_s"] / a["wall_s"]) if a["wall_s"] > 0 else None}
    return out


def surface_ablation(store) -> dict[str, Any]:
    """D restricted to one surface (D_S1/D_S2/D_S3) against full D and planner-only B, plus lessons per surface."""
    sba = success_by_arm(store)
    ref = sba.get("D")
    arms = {}
    for arm in ("B", "D", "D_S1", "D_S2", "D_S3"):
        if arm in sba:
            row = dict(sba[arm])
            row["delta_vs_D_pp"] = None if ref is None else (row["rate"] - ref["rate"]) * 100.0
            arms[arm] = row
    per: dict[str, Counter] = {s: Counter() for s in ("S1", "S2", "S3")}
    for l in lesson_records(store).values():
        per.setdefault(l.get("surface"), Counter())[l.get("status", "candidate")] += 1
    return {"arms": arms, "lessons_by_surface": {s: {k: c.get(k, 0) for k in STATUSES} for s, c in per.items()}}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def all_metrics(store) -> dict[str, Any]:
    return {"benchmark": benchmark_summary(store),
            "success_by_arm": success_by_arm(store), "lesson_precision": lesson_precision(store),
            "lessons": lesson_rows(store), "rescue_rate": rescue_rate(store),
            "intervention_efficiency": intervention_efficiency(store), "latency_cost": latency_cost(store),
            "surface_ablation": surface_ablation(store), "gate_pass_rate": gate_pass_rate(store),
            "environments": environment_report(store)}


def _pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def _ci(ci) -> str:
    return f"[{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"


def _num(x, nd=1) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def results_table(store) -> str:
    """Markdown summary: benchmark (first, when present), arms, ablation, lessons, interventions, latency,
    sleep loop, environments."""
    sba, prec, resc, eff, lat, abl = (success_by_arm(store), lesson_precision(store), rescue_rate(store),
                                      intervention_efficiency(store), latency_cost(store), surface_ablation(store))
    L = benchmark_lines(store)   # the headline result goes first
    L += ["## Success by arm", "| arm | n | successes | rate | 95% CI |", "|---|---|---|---|---|"]
    L += [f"| {a} | {r['n']} | {r['k']} | {_pct(r['rate'])} | {_ci(r['ci'])} |" for a, r in sba.items()]
    if abl["arms"]:
        L += ["", "## Surface ablation", "| arm | n | rate | 95% CI | delta vs D (pp) |", "|---|---|---|---|---|"]
        L += [f"| {a} | {r['n']} | {_pct(r['rate'])} | {_ci(r['ci'])} | {_num(r['delta_vs_D_pp'])} |"
              for a, r in abl["arms"].items()]
    L += ["", "## Lessons",
          f"candidates {prec['candidates']}, validated {prec['validated']}, retired {prec['retired']}, "
          f"precision {_pct(prec['precision'])}"]
    rows = lesson_rows(store)
    if rows:
        L += ["", "| lesson | surface | op | trigger | status | treated | control | lift (pp) | p |",
              "|---|---|---|---|---|---|---|---|---|"]
        L += [f"| {r['lesson_id']} | {r['surface']} | {r['op']} | {r['trigger']} | {r['status']} | "
              f"{r['treated_successes']}/{r['treated_n']} | {r['control_successes']}/{r['control_n']} | "
              f"{_num(r['lift_pp'])} | {_num(r['p_value'], 3)} |" for r in rows]
    L += ["", "## Interventions",
          f"rescue rate {resc['k']}/{resc['n']} = {_pct(resc['rate'])} {_ci(resc['ci'])}; "
          f"same arms without intervention {_pct(resc['baseline_rate'])} (n={resc['baseline_n']})",
          f"efficiency {eff['treated_arm']} vs {eff['baseline_arm']}: gain {_num(eff['gain_pp'])} pp, "
          f"{eff['tokens']} coach tokens over {eff['interventions']} interventions, "
          f"tokens/pp {_num(eff['tokens_per_pp'], 0)}"]
    if lat:
        L += ["", "## Latency", "| arm | n | mean steps | mean wall (s) | interventions/ep | coach s/ep | tokens/ep |",
              "|---|---|---|---|---|---|---|"]
        L += [f"| {a} | {r['n']} | {_num(r['mean_steps'])} | {_num(r['mean_wall_s'], 2)} | "
              f"{_num(r['interventions_per_episode'], 2)} | {_num(r['coach_s_per_episode'], 2)} | "
              f"{_num(r['tokens_per_episode'], 0)} |" for a, r in lat.items()]
    L += environment_lines(store)   # v3: sleep loop + per-environment mastery / readaptation
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fleet Memory metrics over an events.jsonl")
    ap.add_argument("--log", default="logs/events.jsonl")
    ap.add_argument("--json", action="store_true", help="print all metrics as JSON instead of markdown")
    args = ap.parse_args(argv)
    store = load_store(args.log)
    if args.json:
        print(json.dumps(all_metrics(store), indent=1, default=str))
    else:
        print(results_table(store))
    return 0


if __name__ == "__main__":
    sys.exit(main())
