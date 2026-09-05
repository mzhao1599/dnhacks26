"""analysis/metrics.py, analysis/plots.py and dashboard/index.html over a synthetic events.jsonl."""
from __future__ import annotations

import json
import os
import random
from collections import Counter

import pytest

from fleet_memory.analysis import metrics as M
from fleet_memory.analysis import plots as P
from fleet_memory.memory.schema import (Edit, Episode, Evidence, Intervention, Lesson, LessonStatusChange,
                                        Outcome, Plan, Snapshot, Subtask, Trigger, now_iso)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P_SUCCESS = {"A": 0.35, "B": 0.45, "C": 0.5, "D": 0.65, "E": 0.45, "D_S3": 0.6}
SURFACES = {"A": [], "B": ["S1"], "C": ["S1", "S2", "S3"], "D": ["S1", "S2", "S3"], "E": ["S1", "S2", "S3"],
            "D_S3": ["S3"]}
MEMORY_ARMS = {"C", "D", "E", "D_S3"}


def make_log(path: str, seed: int = 0, per_arm: int = 40) -> list[dict]:
    """Synthetic log built from the schema dataclasses: 4 lessons, episodes across arms with A/B
    retrieval splits and inner-loop interventions, then two gate decisions. Returns the records."""
    rng = random.Random(seed)
    recs: list[dict] = []
    lessons = [
        Lesson("les_valid", "S3", Trigger("pick_place", "bowl", "grasp", "object_height_below_rim"),
               Edit("set_grasp_offset", {"dx": 0, "dy": 0, "dz": -0.015}), rationale_text="grasp lower",
               provenance={"from_episodes": ["ep_seed"], "coach_model": "mock", "tokens_in": 100, "tokens_out": 10}),
        Lesson("les_retired", "S3", Trigger("pick_place", "*", "place", "ee_near_target"),
               Edit("set_velocity_cap", {"max_pos_delta": 0.5, "max_rot_delta": 0.5})),
        Lesson("les_cand", "S1", Trigger("pick_place", "mug", "grasp", "grasp_slipped"),
               Edit("set_abort_condition", {"subtask_id": "s2", "predicate": "grasp_slipped", "max_steps": 80})),
        Lesson("les_s2", "S2", Trigger("*", "*", "*", "always"),
               Edit("set_instruction", {"instruction": "pick up the bowl carefully"}),
               evidence=Evidence(trials=5, treated_n=3, treated_successes=2, control_n=2, control_successes=0)),
    ]
    recs += [l.to_dict() for l in lessons]
    i = 0
    for r in range(per_arm):
        for arm, p in P_SUCCESS.items():
            i += 1
            eid = f"ep_{i:04d}"
            retrieved = ["les_valid", "les_retired", "les_cand"] if arm in MEMORY_ARMS else []
            applied = [l for l in retrieved if rng.random() < 0.5]
            succ = rng.random() < p + (0.15 if "les_valid" in applied else 0.0)
            ivs = []
            if "S3" in SURFACES[arm] and rng.random() < 0.4:
                ivs.append(Intervention(episode_id=eid, step=rng.randrange(20, 150),
                                        trigger=rng.choice(["stall", "interval"]), surface="S3",
                                        edit=Edit("set_abort_retry", {"predicate": "grasp_slipped", "max_retries": 1}),
                                        coach_model="mock", latency_s=1.2, tokens_in=800, tokens_out=120))
            plan = Plan("plan_" + eid, "pick_bowl_to_plate",
                        [Subtask("s1", "grasp", "black_bowl", instruction="pick up the bowl"),
                         Subtask("s2", "place", "black_bowl", destination="plate")], ["pick up the bowl"])
            ep = Episode(episode_id=eid, suite="mock", task_id="pick_bowl_to_plate", seed=r, condition=arm,
                         policy="mock", surfaces_enabled=SURFACES[arm], retrieved_lesson_ids=retrieved,
                         applied_lesson_ids=applied, retrieval_frozen_at=now_iso(), plan=plan,
                         instruction_used="pick up the bowl", interventions=ivs,
                         outcome=Outcome(env_success=succ, steps=rng.randrange(60, 200),
                                         termination="success" if succ else "timeout", wall_s=rng.uniform(2, 5)))
            recs.append(ep.to_dict())
            recs.append(Snapshot(eid, "pick_bowl_to_plate", [0.0] * 4, "mock situation", "pick up the bowl", {},
                                 [s.to_dict() for s in plan.subtasks], succ, [0, 25]).to_dict())
    ev = lambda tn, ts, cn, cs: Evidence(tn + cn, tn, ts, cn, cs)  # noqa: E731
    recs.append(LessonStatusChange("les_valid", "candidate", "validated", "gate", ev(30, 22, 30, 12)).to_dict())
    recs.append(LessonStatusChange("les_retired", "candidate", "retired", "gate", ev(30, 12, 30, 15)).to_dict())
    recs.append(LessonStatusChange("les_nonexistent", "candidate", "retired", "orphan", ev(0, 0, 0, 0)).to_dict())
    with open(path, "w", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        f.write('{"type": "episode", "truncated')  # torn trailing line must be skipped, not fatal
    return recs


@pytest.fixture(scope="module")
def log(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("logs") / "events.jsonl")
    recs = make_log(path)
    return path, recs


def _eps(recs):
    return [r for r in recs if r["type"] == "episode"]


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #

def test_wilson_ci():
    assert M.wilson_ci(0, 0) == (0.0, 1.0)
    lo, hi = M.wilson_ci(5, 10)
    assert abs(lo - 0.2366) < 1e-3 and abs(hi - 0.7634) < 1e-3
    assert M.wilson_ci(0, 20)[0] == 0.0 and M.wilson_ci(20, 20)[1] == 1.0
    a, b = M.wilson_ci(50, 100), M.wilson_ci(500, 1000)
    assert (b[1] - b[0]) < (a[1] - a[0])  # tighter with more data


def test_two_proportion_p():
    assert M.two_proportion_p(0, 0, 5, 10) is None
    assert M.two_proportion_p(10, 10, 10, 10) == 1.0        # identical, degenerate
    strong = M.two_proportion_p(28, 30, 8, 30)
    weak = M.two_proportion_p(16, 30, 15, 30)
    assert 0.0 <= strong < 0.001 and 0.3 < weak < 0.5
    assert M.two_proportion_p(5, 30, 25, 30) > 0.99         # one-sided: treated worse -> p near 1
    try:
        from scipy.stats import norm
        p1, p2, p = 28 / 30, 8 / 30, 36 / 60
        z = (p1 - p2) / (p * (1 - p) * (2 / 30)) ** 0.5
        assert abs(strong - norm.sf(z)) < 1e-9
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# metrics over the synthetic log (via path) — expectations recomputed independently from the records
# --------------------------------------------------------------------------- #

def test_records_and_lessons_materialised(log):
    path, recs = log
    assert len(M.episode_records(path)) == len(_eps(recs))       # torn line skipped
    ls = M.lesson_records(path)
    assert set(ls) == {"les_valid", "les_retired", "les_cand", "les_s2"}
    assert ls["les_valid"]["status"] == "validated" and ls["les_retired"]["status"] == "retired"
    assert ls["les_cand"]["status"] == "candidate"
    assert ls["les_valid"]["evidence"]["treated_successes"] == 22  # evidence from the status event


def test_success_by_arm(log):
    path, recs = log
    sba = M.success_by_arm(path)
    assert list(sba) == ["A", "B", "C", "D", "E", "D_S3"]          # ARM_ORDER, ablations last
    for arm, row in sba.items():
        eps = [e for e in _eps(recs) if e["condition"] == arm]
        k = sum(e["outcome"]["env_success"] for e in eps)
        assert (row["n"], row["k"]) == (len(eps), k)
        assert row["rate"] == pytest.approx(k / len(eps)) and row["ci"] == M.wilson_ci(k, len(eps))
    assert M.success_by_arm([]) == {}


def test_success_curve(log):
    path, recs = log
    flags = [int(e["outcome"]["env_success"]) for e in _eps(recs) if e["condition"] == "D"]
    curve = M.success_curve(path, "D", window=20)
    assert [i for i, _ in curve] == list(range(1, len(flags) + 1))
    assert curve[0][1] == flags[0] and curve[-1][1] == pytest.approx(sum(flags[-20:]) / 20)
    assert [r for _, r in M.success_curve(path, "D", window=1)] == flags
    assert M.success_curve(path, "F") == []


def test_lesson_precision(log):
    path, _ = log
    assert M.lesson_precision(path) == {"candidates": 2, "validated": 1, "retired": 1, "precision": 0.5}
    assert M.lesson_precision([])["precision"] is None


def test_lesson_evidence_and_rows(log):
    path, recs = log
    ev = M.lesson_evidence(path, "les_valid")
    tn = ts = cn = cs = 0
    for e in _eps(recs):
        if "les_valid" in e["retrieved_lesson_ids"]:
            if "les_valid" in e["applied_lesson_ids"]:
                tn, ts = tn + 1, ts + e["outcome"]["env_success"]
            else:
                cn, cs = cn + 1, cs + e["outcome"]["env_success"]
    assert (ev["treated_n"], ev["treated_successes"], ev["control_n"], ev["control_successes"]) == (tn, ts, cn, cs)
    assert ev["trials"] == tn + cn and ev["lift_pp"] == pytest.approx((ts / tn - cs / cn) * 100)
    assert 0.0 <= ev["p_value"] <= 1.0 and ev["lift_pp"] > 0   # the +0.15 treatment effect shows up
    rows = {r["lesson_id"]: r for r in M.lesson_rows(path)}
    assert [r["status"] for r in M.lesson_rows(path)] == ["validated", "candidate", "candidate", "retired"]
    assert rows["les_valid"]["trigger"] == "pick_place/bowl/grasp/object_height_below_rim"
    assert rows["les_valid"]["op"] == "set_grasp_offset" and rows["les_valid"]["treated_n"] == tn
    assert rows["les_s2"]["treated_n"] == 3 and rows["les_s2"]["control_n"] == 2  # stored evidence fallback
    assert M.lesson_evidence(path, "nope")["p_value"] is None


def test_rescue_rate(log):
    path, recs = log
    r = M.rescue_rate(path)
    with_iv = [e for e in _eps(recs) if e["interventions"]]
    assert r["n"] == len(with_iv) and r["k"] == sum(e["outcome"]["env_success"] for e in with_iv)
    assert r["rate"] == pytest.approx(r["k"] / r["n"]) and r["ci"] == M.wilson_ci(r["k"], r["n"])
    assert set(r["by_arm"]) == {"C", "D", "E", "D_S3"} and set(r["by_trigger"]) <= {"stall", "interval"}
    assert sum(v["n"] for v in r["by_arm"].values()) == r["n"]
    without = [e for e in _eps(recs) if e["condition"] in r["by_arm"] and not e["interventions"]]
    assert r["baseline_n"] == len(without)
    assert M.rescue_rate([])["rate"] is None


def test_intervention_efficiency(log):
    path, recs = log
    e = M.intervention_efficiency(path)
    sba = M.success_by_arm(path)
    inner = sum(iv["tokens_in"] + iv["tokens_out"] for ep in _eps(recs) if ep["condition"] == "D"
                for iv in ep["interventions"])
    assert e["inner_tokens"] == inner and e["outer_tokens"] == 0 and e["tokens"] == inner
    assert e["gain_pp"] == pytest.approx((sba["D"]["rate"] - sba["B"]["rate"]) * 100)
    assert e["tokens_per_pp"] == pytest.approx(inner / e["gain_pp"]) if e["gain_pp"] > 0 else e["tokens_per_pp"] is None
    assert e["episodes"] == 40 and e["tokens_per_episode"] == pytest.approx(inner / 40)
    worse = [{"type": "episode", "episode_id": f"e{i}", "condition": c, "outcome": {"env_success": s}}
             for i, (c, s) in enumerate([("D", False), ("D", False), ("B", True), ("B", False)])]
    assert M.intervention_efficiency(worse)["gain_pp"] == -50.0
    assert M.intervention_efficiency(worse)["tokens_per_pp"] is None    # no gain -> undefined
    assert M.intervention_efficiency([])["gain_pp"] is None


def test_latency_cost(log):
    path, recs = log
    lat = M.latency_cost(path)
    assert set(lat) == set(P_SUCCESS)
    assert lat["A"]["interventions_per_episode"] == 0 and lat["A"]["coach_share"] == 0.0
    assert M.latency_cost([{"type": "episode", "condition": "A", "outcome": {"env_success": True, "steps": 3}}]) \
        == {"A": {"n": 1, "mean_steps": 3.0, "mean_wall_s": 0.0, "interventions_per_episode": 0.0,
                  "coach_s_per_episode": 0.0, "tokens_per_episode": 0.0, "coach_share": None}}
    d = [e for e in _eps(recs) if e["condition"] == "D"]
    assert lat["D"]["mean_steps"] == pytest.approx(sum(e["outcome"]["steps"] for e in d) / len(d))
    assert lat["D"]["interventions_per_episode"] == pytest.approx(sum(len(e["interventions"]) for e in d) / len(d))
    assert 0 < lat["D"]["coach_share"] < 1


def test_surface_ablation(log):
    path, _ = log
    abl = M.surface_ablation(path)
    assert set(abl["arms"]) == {"B", "D", "D_S3"} and abl["arms"]["D"]["delta_vs_D_pp"] == 0.0
    assert abl["arms"]["B"]["delta_vs_D_pp"] == pytest.approx((abl["arms"]["B"]["rate"] - abl["arms"]["D"]["rate"]) * 100)
    assert abl["lessons_by_surface"]["S3"] == {"candidate": 0, "validated": 1, "retired": 1}
    assert abl["lessons_by_surface"]["S1"]["candidate"] == 1


def test_results_table_and_all_metrics(log):
    path, _ = log
    md = M.results_table(path)
    for s in ("## Success by arm", "| D |", "## Lessons", "les_valid", "validated", "## Interventions", "## Latency"):
        assert s in md
    js = json.dumps(M.all_metrics(path), default=str)   # JSON-serialisable for the CLI
    assert "success_by_arm" in js and "surface_ablation" in js
    assert "## Success by arm" in M.results_table([])     # empty log is not an error


def test_store_inputs_agree(log):
    """Path, in-memory dicts, dataclass records and (when present) EventStore all give the same numbers."""
    path, recs = log
    via_path, via_list = M.success_by_arm(path), M.success_by_arm(recs)
    assert via_path == via_list
    objs = [Episode(**{k: v for k, v in r.items() if k in Episode.__dataclass_fields__ and k not in
                       ("plan", "interventions", "outcome")}, outcome=Outcome(**r["outcome"]))
            for r in _eps(recs)]
    assert M.success_by_arm(objs) == via_path
    try:
        from fleet_memory.memory.store import EventStore
    except ImportError:
        return
    assert M.success_by_arm(EventStore(path)) == via_path
    assert M.lesson_precision(EventStore(path)) == M.lesson_precision(path)


def test_cli(log, capsys):
    path, _ = log
    assert M.main(["--log", path]) == 0
    assert "## Success by arm" in capsys.readouterr().out
    assert M.main(["--log", path, "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["lesson_precision"]["validated"] == 1
    assert M.main(["--log", path + ".missing"]) == 0         # absent log -> empty report, no crash


# --------------------------------------------------------------------------- #
# plots + dashboard
# --------------------------------------------------------------------------- #

def test_plots(log, tmp_path):
    path, _ = log
    out1 = P.plot_success_curves(path, str(tmp_path / "curves.png"))
    out2 = P.plot_arms(path, str(tmp_path / "arms.png"))
    if P.plt is None:
        assert out1 is None and out2 is None
    else:
        assert os.path.getsize(out1) > 0 and os.path.getsize(out2) > 0
        assert P.plot_arms([], str(tmp_path / "empty.png")) is not None
    assert P.main(["--log", path, "--out-dir", str(tmp_path)]) == 0


def test_dashboard_static_file():
    html = open(os.path.join(ROOT, "dashboard", "index.html"), encoding="utf-8").read()
    assert html.count("\n") <= 900
    assert "<script src" not in html and "<link" not in html and "https://" not in html   # no network deps
    assert 'type="file"' in html and "prefers-color-scheme" in html and "function sample()" in html
    for panel in ("#arms", "#curves", "#lessons", "#timeline", "#episodes", "#tiles",
                  "#mastery", "#house", "#versions", "#consolidations", "#envsel"):          # v3 panels
        assert panel in html
    for kind in ("skill_instance", "consolidation", "drift_trigger", "protocol_stage", "consolidation_id"):
        assert kind in html                                        # v3 record types are read and rollouts excluded
