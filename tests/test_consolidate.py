"""runner/consolidate.py: CEM core on a synthetic quadratic, the gate rule on fake episodes, and the
full consolidate() flow with an injected run_many (house_model is required for that last part)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from fleet_memory.execution import params as P
from fleet_memory.execution.params import S3Params
from fleet_memory.memory.schema import Episode, EpisodeMetrics, Outcome, new_id, now_iso, skill_instance_id
from fleet_memory.memory.store import EventStore
from fleet_memory.runner import consolidate as C

U_STAR = np.concatenate([[0.5, 0.5, 0.5, 0.2, 0.25, 0.9, 0.8, 0.7, 0.4], np.full(P.DIM - 9, 0.5)])   # optimum in normalized space (generic in DIM)


def quad(X: np.ndarray) -> np.ndarray:
    U = np.stack([P.normalize(x) for x in np.asarray(X).reshape(-1, P.DIM)])
    return ((U - U_STAR) ** 2).sum(axis=1)


# ------------------------------------------------------------------------------ CEM core
def test_cem_converges_on_quadratic():
    cfg = C.ConsolidationConfig(population=32, elites=6, iterations=12, sigma_init_frac=0.3, sigma_decay=0.8)
    theta0 = S3Params.identity().to_array()
    c0 = float(quad(theta0[None])[0])
    res = C.cem_search(quad, theta0, cfg, np.random.default_rng(0))
    assert res.evaluations == 32 * 12
    assert len(res.history) == 12 and res.history[0]["iter"] == 0
    assert res.best_cost < c0 / 10
    assert np.linalg.norm(P.normalize(res.theta_star) - U_STAR) < 0.15
    assert np.all(res.theta_star >= P.LOW) and np.all(res.theta_star <= P.HIGH)
    # best_ever is monotone non-increasing and the last iteration's mean beat the first's
    be = [h["best_ever"] for h in res.history]
    assert all(b <= a for a, b in zip(be, be[1:]))
    assert res.history[-1]["mean_cost"] < res.history[0]["mean_cost"]


def test_iteration_zero_includes_theta0():
    seen = []

    def ev(X):
        seen.append(np.asarray(X).copy())
        return quad(X)

    theta0 = S3Params.identity().to_array()
    C.cem_search(ev, theta0, C.ConsolidationConfig(population=4, iterations=1), np.random.default_rng(1))
    assert np.allclose(seen[0][0], theta0)


def test_random_method_same_budget_and_bounds():
    cfg = C.ConsolidationConfig(method="random", population=20, iterations=5)
    seen = []

    def ev(X):
        seen.append(np.asarray(X))
        return quad(X)

    res = C.cem_search(ev, S3Params.identity().to_array(), cfg, np.random.default_rng(3))
    assert res.evaluations == 100 and len(seen) == 5
    allx = np.concatenate(seen)
    assert np.all(allx >= P.LOW - 1e-9) and np.all(allx <= P.HIGH + 1e-9)
    assert res.best_cost <= min(float(quad(x[None])[0]) for x in allx) + 1e-12


def test_nan_and_short_costs_become_failed():
    def ev(X):
        c = quad(X)
        c[::2] = np.nan
        return c[:-1]                                 # one short, several NaN

    cfg = C.ConsolidationConfig(population=8, elites=2, iterations=2)
    res = C.cem_search(ev, S3Params.identity().to_array(), cfg, np.random.default_rng(0))
    assert res.best_cost < C.FAILED_COST
    assert C.sanitize_costs([np.nan, 1.0, np.inf], 4).tolist() == [10.0, 1.0, 10.0, 10.0]
    assert C.sanitize_costs(None, 2).tolist() == [10.0, 10.0]


def test_wallclock_deadline_stops_after_first_iteration():
    cfg = C.ConsolidationConfig(population=4, iterations=6)
    import time
    res = C.cem_search(quad, S3Params.identity().to_array(), cfg, np.random.default_rng(0), deadline=time.time() - 1)
    assert res.evaluations == 4 and res.stopped_early and len(res.history) == 1


def test_small_config():
    s = C.ConsolidationConfig.small()
    assert (s.population, s.elites, s.iterations, s.seeds_per_candidate, s.gate_seeds) == (16, 4, 4, 2, 8)
    assert C.ConsolidationConfig.small(method="random").method == "random"


# ---------------------------------------------------------------------------------- gate
def ep(cost, success, seed=0, arm="B", s3=None, pert=None, env_id="mock_pick_bowl_to_plate"):
    return Episode(episode_id=new_id("ep"), suite="mock", task_id="pick_bowl_to_plate", seed=seed, condition=arm,
                   policy="mock", surfaces_enabled=["S3"], retrieved_lesson_ids=[], applied_lesson_ids=[],
                   retrieval_frozen_at=now_iso(), environment_id=env_id, s3_params=s3 or {},
                   outcome=Outcome(env_success=success, steps=50, termination="success" if success else "timeout"),
                   metrics=EpisodeMetrics(steps=50, success=success, jerk=0.5, force_proxy=0.1, cost=cost),
                   perturbation=pert or {})


def test_gate_rule():
    inc = [ep(1.0, True) for _ in range(8)] + [ep(1.2, True) for _ in range(8)]      # cost 1.1, success 1.0
    good = [ep(0.9, True) for _ in range(16)]
    assert C.gate_decision(good, inc).passed
    g = C.gate_decision([ep(0.5, True)] * 15 + [ep(0.5, False)], inc)           # -6.25pp success
    assert not g.passed and g.success_delta_pp == pytest.approx(-6.25) and g.n_seeds == 16
    assert not C.gate_decision([ep(1.2, True)] * 16, inc).passed                 # costlier
    assert not C.gate_decision([ep(1.1, True)] * 16, inc).passed                 # equal cost is not lower
    ok = [ep(0.9, True)] * 50 + [ep(0.9, False)]                                 # -1.96pp: within slack
    assert C.gate_decision(ok, inc).passed
    assert not C.gate_decision([], inc).passed and not C.gate_decision(good, []).passed
    assert C.episode_cost(ep(None, True)) == C.FAILED_COST and C.episode_cost(None) == C.FAILED_COST
    assert C.summarize_rollouts([ep(None, True), ep(2.0, False)]) == (6.0, 0.5)


# ---------------------------------------------------------------------- consolidate() flow
@dataclass
class FakeCfg:
    """Shape-compatible with runner.worker.RunConfig for the fields consolidate() touches."""
    suite: str = "mock"
    task_id: str = "pick_bowl_to_plate"
    seed: int = 0
    arm: str = "B"
    env_kind: str = "mock"
    policy_kind: str = "mock"
    log_path: str = "logs/events.jsonl"
    s3_params: dict | None = None
    perturbation: dict | None = None
    skill: str | None = None
    environment_tag: str = ""


class FakeRunner:
    """Deterministic stand-in for pool.run_many: cost = 1 + 5*||u - U_STAR||^2 (+ seed noise), success
    iff close to U_STAR. Appends Episodes to the store like the worker does; cost is None until a
    cost_reference exists (mirrors worker order)."""

    def __init__(self, store, si_id, hm):
        self.store, self.si_id, self.hm, self.calls, self.seen_arms = store, si_id, hm, [], set()

    def __call__(self, cfgs, workers):
        self.calls.append(len(cfgs))
        has_ref = self.hm.cost_reference(self.store, self.si_id) is not None
        out = []
        for c in cfgs:
            self.seen_arms.add(c.arm)
            u = P.normalize(S3Params.from_dict(c.s3_params).to_array()) if c.arm == "B" else P.normalize(S3Params.identity().to_array())
            d2 = float(((u - U_STAR) ** 2).sum())
            noise = 0.05 * np.sin(c.seed)
            succ = d2 < 1.0
            cost = (1.0 + 5.0 * d2 + noise + (0.0 if succ else 3.0)) if has_ref else None
            e = ep(cost, succ, seed=c.seed, arm=c.arm, s3=c.s3_params, pert=c.perturbation,
                   env_id=C.environment_id_of(c))
            e.metrics.jerk, e.metrics.steps = 0.002, 40 + int(20 * d2)
            self.store.append(e)
            out.append(e)
        return out


def test_consolidate_promotes_when_gate_passes(tmp_path):
    hm = pytest.importorskip("fleet_memory.memory.house_model")
    store = EventStore(str(tmp_path / "events.jsonl"))
    tpl = FakeCfg(log_path=store.path, perturbation={"shift_xy": [0.02, 0.0]})
    si_id = skill_instance_id("pick_bowl_to_plate", C.environment_id_of(tpl))
    runner = FakeRunner(store, si_id, hm)
    cfg = C.ConsolidationConfig.small(iterations=6, population=24, workers=1)
    end = C.consolidate(store, si_id, tpl, cfg, trigger="drift", run_many=runner, rng=np.random.default_rng(0))

    assert end.phase == "end" and end.trigger == "drift" and end.incumbent_version == 1
    assert end.gate is not None and end.gate.passed and end.promoted_version == 2
    assert end.gate.n_seeds == cfg.gate_seeds
    assert len(end.history) == 6 and end.best_candidate["opt_cost"] < 2.0   # 17-dim quadratic, same budget
    n_ref, n_search, n_gate = max(4, cfg.gate_seeds), 6 * 24 * 2, 2 * cfg.gate_seeds
    val = end.best_candidate["validation"]
    n_val = len(val["candidates"]) * cfg.validation_seeds        # {incumbent, final mean, best of each iter}
    assert 2 <= len(val["candidates"]) <= 8 and val["chosen"] in val["candidates"]
    assert end.rollouts == n_ref + n_search + n_val + n_gate == sum(runner.calls)
    assert runner.seen_arms == {"A", "B"}

    inc = hm.incumbent(store, si_id)
    assert inc is not None and inc.version == 2 and inc.status == "incumbent"
    assert np.allclose(S3Params.from_dict(inc.params).to_array(), S3Params.from_dict(end.best_candidate["params"]).to_array())
    assert hm.cost_reference(store, si_id) is not None
    evs = store.read_all()
    cons = [d for d in evs if d.get("type") == "consolidation"]
    assert [d["phase"] for d in cons] == ["start", "end"] and cons[0]["consolidation_id"] == end.consolidation_id
    assert cons[1]["gate"]["passed"] is True and cons[1]["optimizer"]["method"] == "cem"
    eps = store.episodes()
    assert eps and all(e.perturbation.get("consolidation_id") == end.consolidation_id for e in eps)
    assert all(e.perturbation.get("jitter_m") == cfg.pose_jitter_m for e in eps)
    assert all(e.perturbation.get("shift_xy") == [0.02, 0.0] for e in eps)       # template perturbation kept
    assert {e.condition for e in eps} == {"A", "B"}
    gate_seeds = {e.seed for e in eps if e.condition == "B" and e.seed >= 4000}
    opt_seeds = {e.seed for e in eps if e.seed < 4000}
    assert len(gate_seeds) == cfg.gate_seeds and opt_seeds.isdisjoint(gate_seeds)


def test_consolidate_no_promotion_when_incumbent_is_optimal(tmp_path):
    hm = pytest.importorskip("fleet_memory.memory.house_model")
    store = EventStore(str(tmp_path / "events.jsonl"))
    tpl = FakeCfg(log_path=store.path, environment_tag="shifted")
    env_id = C.environment_id_of(tpl)
    assert env_id == "mock_pick_bowl_to_plate_shifted"
    si_id = skill_instance_id("pick_bowl_to_plate", env_id)
    hm.init_incumbent(store, si_id, env_id, "pick_bowl_to_plate", "black_bowl",
                      S3Params.from_array(P.denormalize(U_STAR)).to_dict())
    runner = FakeRunner(store, si_id, hm)
    cfg = C.ConsolidationConfig.small(iterations=2, population=6, workers=1)
    end = C.consolidate(store, si_id, tpl, cfg, run_many=runner, rng=np.random.default_rng(5))
    assert end.promoted_version is None and end.incumbent_version == 1
    assert hm.incumbent(store, si_id).version == 1
    if end.gate is not None:                     # a different candidate was tried and lost the gate
        assert not end.gate.passed and end.gate.candidate_cost >= end.gate.incumbent_cost or \
            end.gate.candidate_success < end.gate.incumbent_success - C.GATE_SUCCESS_SLACK


def test_seed_set_fallback_and_rollout_cfgs():
    s = C.seed_set("gate", 3)
    assert len(s) == 3 and all(isinstance(x, int) for x in s) and s[0] >= 4000
    tpl = FakeCfg()
    cfgs = C.rollout_cfgs(tpl, "B", {"blend_alpha": 0.5, "grasp_offset_z": -0.01}, [7, 8], {"jitter_m": 0.01})
    assert [c.seed for c in cfgs] == [7, 8] and all(c.arm == "B" for c in cfgs)
    assert cfgs[0].perturbation == {"jitter_m": 0.01} and cfgs[0].perturbation is not cfgs[1].perturbation
    assert tpl.arm == "B" and tpl.seed == 0 and tpl.s3_params is None            # template untouched
