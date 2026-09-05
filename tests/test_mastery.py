"""v3 metrics (analysis/mastery.py via analysis/metrics.py) over a synthetic sleep-loop / protocol-P log."""
from __future__ import annotations

import json

import pytest

from fleet_memory.analysis import metrics as M
from fleet_memory.memory.schema import (Consolidation, CostReference, DriftTrigger, Episode, EpisodeMetrics,
                                        GateResult, Outcome, SkillInstance, SkillInstanceStatusChange, now_iso)

ENV = "mock_pick_bowl_to_plate"
SI = f"si_pick_bowl_to_plate__{ENV}"
PARAMS = {"approach_offset_xyz": [0, 0, 0], "pregrasp_height": 0.05, "grasp_offset_z": 0.0, "time_scale": 1.0,
          "velocity_cap": 1.0, "gripper_cmd": 1.0, "approach_cone_deg": 45.0}
PRE = [1.0] * 10
PERTURBED = [2.0] * 10
RECOVERY = [1.6, 1.3, 1.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
SHIFT = {"shift_xy": [0.06, 0.0]}


def _ep(i: int, arm: str, cost: float | None, version: int | None, perturbation: dict) -> dict:
    succ = cost is not None and cost < 1.5
    m = None if cost is None else EpisodeMetrics(steps=100, success=succ, jerk=1.0, force_proxy=0.1, cost=cost)
    return Episode(episode_id=f"ep_{i:04d}", suite="mock", task_id="pick_bowl_to_plate", seed=i, condition=arm,
                   policy="mock", surfaces_enabled=[], retrieved_lesson_ids=[], applied_lesson_ids=[],
                   retrieval_frozen_at=now_iso(), outcome=Outcome(env_success=succ, steps=100, termination="success",
                                                                  wall_s=1.0),
                   environment_id=ENV, skill_instance_versions={SI: version} if version else {}, metrics=m,
                   perturbation=perturbation).to_dict()


def _stage(stage: str, ids: list[str]) -> dict:
    return {"type": "protocol_stage", "stage": stage, "environment_id": ENV, "si_id": SI, "ts": now_iso(), "episode_ids": ids}


def make_v3_log(path: str) -> list[dict]:
    recs: list[dict] = [CostReference(SI, ENV, steps_ref=100.0, jerk_ref=0.01, n_episodes=5).to_dict(),
                        SkillInstance(SI, ENV, "pick_bowl_to_plate", "black_bowl", 1, None, "incumbent", PARAMS,
                                      produced_by={"loop": "init"}).to_dict()]
    n = 0
    for _ in range(5):                                            # arm A without a cost reference yet
        n += 1
        recs.append(_ep(n, "A", None, None, {}))
    base = []
    for c in PRE:
        n += 1
        recs.append(_ep(n, "P", c, 1, {}))
        base.append(f"ep_{n:04d}")
    recs.append(_stage("baseline", base))
    pert = []
    for j, c in enumerate(PERTURBED):
        n += 1
        recs.append(_ep(n, "P", c, 1, SHIFT))
        pert.append(f"ep_{n:04d}")
        if j == 4:
            recs.append(DriftTrigger(SI, "ewma_cost_exceeds_baseline", 1.9, 1.0, 1.9, 0.0, 1.0, episode_id=pert[-1]).to_dict())
            recs.append(Consolidation("con_1", "start", SI, "drift", 1).to_dict())
            for _ in range(8):                                    # optimizer rollouts: excluded from the curve
                n += 1
                recs.append(_ep(n, "B", 1.5, 1, {**SHIFT, "jitter_m": 0.015, "consolidation_id": "con_1"}))
            gate = GateResult(True, 2.0, 1.1, 8, 0.0, 0.9, 90.0)
            recs.append(SkillInstance(SI, ENV, "pick_bowl_to_plate", "black_bowl", 2, 1, "incumbent",
                                      {**PARAMS, "grasp_offset_z": -0.012}, {"loop": "sleep", "consolidation_id": "con_1"},
                                      gate=gate).to_dict())
            recs.append(SkillInstanceStatusChange(SI, 1, "incumbent", "retired", "promoted v2").to_dict())
            recs.append(Consolidation("con_1", "end", SI, "drift", 1, rollouts=8, wallclock_s=12.5, gate=gate,
                                      promoted_version=2, history=[{"iter": k, "mean_cost": 2.0 - 0.2 * k,
                                                                    "best_cost": 1.5 - 0.1 * k, "sigma_mean": 0.3 * 0.8 ** k}
                                                                   for k in range(4)]).to_dict())
    recs.append(_stage("perturbed", pert))
    rec = []
    for c in RECOVERY:
        n += 1
        recs.append(_ep(n, "P", c, 2, SHIFT))
        rec.append(f"ep_{n:04d}")
    recs.append(_stage("recovery", rec))
    recs.append(Consolidation("con_2", "start", SI, "manual", 2).to_dict())          # a second, failed gate
    recs.append(Consolidation("con_2", "end", SI, "manual", 2, rollouts=8, wallclock_s=3.0,
                              gate=GateResult(False, 1.0, 1.2, 8), promoted_version=None).to_dict())
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, separators=(",", ":")) + "\n" for r in recs)
    return recs


@pytest.fixture(scope="module")
def v3log(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("logs") / "events.jsonl")
    return path, make_v3_log(path)


def test_mastery_curve_excludes_rollouts(v3log):
    path, recs = v3log
    curve = M.mastery_curve(path, ENV)
    assert len(curve) == 5 + 30 and [p["i"] for p in curve] == list(range(35))
    assert not any(p["arm"] == "B" for p in curve)                          # the 8 rollouts are excluded
    assert sum(1 for r in recs if r["type"] == "episode" and M.is_rollout(r)) == 8
    assert curve[0]["cost"] is None and curve[0]["version"] is None          # arm A before any reference
    assert [p["version"] for p in curve][5:15] == [1] * 10 and curve[-1]["version"] == 2
    p_only = M.mastery_curve(path, ENV, arm="P")
    assert [p["cost"] for p in p_only] == PRE + PERTURBED + RECOVERY and p_only[-1]["success"] is True
    assert M.mastery_curve(path, "nope") == [] and M.environments(path) == [ENV]


def test_gate_pass_rate_and_versions(v3log):
    path, _ = v3log
    assert M.gate_pass_rate(path) == {"attempted": 2, "promoted": 1, "rate": 0.5}
    assert M.gate_pass_rate([])["rate"] is None
    vs = M.versions(path, ENV)
    assert [(v["version"], v["status"]) for v in vs] == [(1, "retired"), (2, "incumbent")]
    assert vs[1]["gate"]["passed"] is True and vs[1]["produced_by"]["consolidation_id"] == "con_1"


def test_readaptation_finds_recovery(v3log):
    path, _ = v3log
    r = M.readaptation(path, ENV)
    assert r["pre_cost"] == pytest.approx(1.0) and r["perturbed_peak_cost"] == pytest.approx(2.0)
    # rolling-5 over PERTURBED + RECOVERY first drops within 10% of pre at post index 15 (mean 1.08)
    assert r["episodes_to_recover"] == 16 and r["recovered_at_episode"] == "ep_0039"
    assert r["consolidations"] == 1 and r["wallclock_s"] == pytest.approx(16 * 1.0 + 12.5)
    assert r["n_post"] == 20
    empty = M.readaptation(path, "nope")
    assert empty["pre_cost"] is None and empty["recovered_at_episode"] is None
    # no recovery when post-perturbation costs stay high
    hi = [dict(x, metrics={**x["metrics"], "cost": 3.0}) if x.get("type") == "episode" and x.get("perturbation")
          and x["condition"] == "P" else x for x in M._records(path)]
    stuck = M.readaptation(hi, ENV)
    assert stuck["recovered_at_episode"] is None and stuck["episodes_to_recover"] is None
    assert stuck["wallclock_s"] is None and stuck["consolidations"] == 1


def test_perturbed_success(v3log):
    path, _ = v3log
    ps = M.perturbed_success(path, ENV)
    assert ps == {"P": pytest.approx(9 / 20)}                               # rollouts (arm B) excluded


def test_report_and_cli(v3log, capsys):
    path, _ = v3log
    rep = M.environment_report(path)[ENV]
    assert rep["n"] == 35 and rep["rollouts"] == 8 and rep["versions"] == 2
    assert rep["arms"]["P"]["last10_cost"] == pytest.approx(sum(RECOVERY) / 10)
    md = M.results_table(path)
    for s in ("## Sleep loop", "gate pass rate 50.0%", f"### {ENV}", "recovered at ep_0039 after 16 episodes", "perturbed success: P 45.0%"):
        assert s in md
    assert M.main(["--log", path, "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["gate_pass_rate"]["promoted"] == 1 and out["environments"][ENV]["readaptation"]["episodes_to_recover"] == 16
