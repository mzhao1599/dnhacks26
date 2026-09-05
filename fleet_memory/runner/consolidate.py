"""Sleep loop (v3): CEM over the 9-dim S3 vector of ONE skill instance, gated by a perturbation test on
disjoint seeds, then promotion through memory/house_model.py (CLAUDE.md invariant 7).

    theta0 = incumbent -> cem_search (normalized [0,1]^9, jittered `opt` seeds) -> theta_star
    gate on `gate` seeds (same jitter): mean cost lower AND success >= incumbent - 0.02 -> promote

`cem_search` is a pure function over an injected evaluate(candidates) -> costs, so the optimizer is
testable without the runner. runner.pool / house_model / conditions are imported at call time.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from fleet_memory.analysis import cost as costlib
from fleet_memory.execution import params as P
from fleet_memory.execution.params import S3Params
from fleet_memory.memory.schema import Consolidation, Episode, GateResult, new_id, skill_instance_id
from fleet_memory.memory.store import EventStore

FAILED_COST = 10.0             # NaN / errored / cost-less rollouts
GATE_SUCCESS_SLACK = 0.02      # candidate success may trail the incumbent by at most this
OPT_POOL = 200                 # size of the rotating opt-seed pool

Evaluate = Callable[[np.ndarray], "Sequence[float] | np.ndarray"]
RunMany = Callable[[list, int], list]


@dataclass
class ConsolidationConfig:
    method: str = "cem"                 # "cem" | "random"
    population: int = 64
    elites: int = 8
    iterations: int = 8
    seeds_per_candidate: int = 4
    gate_seeds: int = 16
    validation_seeds: int = 8           # final selection re-evaluates candidates on this many fresh opt seeds (0: off)
    pose_jitter_m: float = 0.015
    sigma_init_frac: float = 0.3
    sigma_decay: float = 0.8
    max_wallclock_s: float = 1500.0
    workers: int = 4
    select: str = "mean"                # "mean": final elite mean (noise-robust) | "best": best-ever sample
                                        # (best-ever compares costs across iterations that used different seed
                                        # pairs, so it picks the luckiest draw: on the mock it gated at 1.50-2.61
                                        # vs 1.46-1.52 for the elite mean, same budget, 5 seeds each)

    @classmethod
    def small(cls, **kw) -> "ConsolidationConfig":
        base = dict(population=16, elites=4, iterations=4, seeds_per_candidate=2, gate_seeds=8, workers=4)
        base.update(kw)
        return cls(**base)

    @classmethod
    def medium(cls, **kw) -> "ConsolidationConfig":
        """Drift-triggered sleep (pool.py --auto-sleep): 4 common seeds per candidate so a 2-sample
        success-rate estimate does not dominate elite selection; gate on the full 16 seeds."""
        base = dict(population=24, elites=6, iterations=5, seeds_per_candidate=4, gate_seeds=16, workers=4)
        base.update(kw)
        return cls(**base)

    def optimizer_dict(self) -> dict[str, Any]:
        return {"method": self.method, "population": self.population, "elites": self.elites,
                "iterations": self.iterations, "seeds_per_candidate": self.seeds_per_candidate,
                "pose_jitter_m": self.pose_jitter_m, "sigma_init_frac": self.sigma_init_frac,
                "sigma_decay": self.sigma_decay, "gate_seeds": self.gate_seeds, "select": self.select,
                "validation_seeds": self.validation_seeds}


@dataclass
class SearchResult:
    theta_star: np.ndarray                      # raw parameter space (params.LOW..HIGH), best ever
    best_cost: float
    history: list[dict[str, Any]] = field(default_factory=list)
    evaluations: int = 0
    wallclock_s: float = 0.0
    stopped_early: bool = False
    validation: dict[str, Any] | None = None    # {candidates, costs, chosen} when a validate() was given


# --------------------------------------------------------------------- optimizer core (pure)
def sanitize_costs(costs, n: int) -> np.ndarray:
    """Coerce whatever evaluate returned into n finite costs; anything missing/NaN/inf -> FAILED_COST."""
    out = np.full(n, FAILED_COST, np.float64)
    try:
        arr = np.asarray(list(costs), np.float64).reshape(-1)
    except (TypeError, ValueError):
        return out
    m = min(n, len(arr))
    out[:m] = arr[:m]
    out[~np.isfinite(out)] = FAILED_COST
    return out


def cem_search(evaluate: Evaluate, theta0: np.ndarray, cfg: ConsolidationConfig,
               rng: np.random.Generator | None = None, deadline: float | None = None,
               validate: Evaluate | None = None) -> SearchResult:
    """Cross-entropy search in normalized [0,1]^DIM. `evaluate(X)` gets (n, DIM) raw-space candidates and
    returns n costs (lower is better). sigma follows sigma_init_frac * sigma_decay**iter per dim; the mean
    moves to the elite mean. Elitism: theta0 (the incumbent) is row 0 of EVERY iteration, so it competes
    on the same seeds as the samples and the mean cannot drift below it without evidence.
    Final selection: `validate` (if given) re-evaluates {theta0, final mean, best sample of each iteration}
    on one fresh common seed set and returns the lowest — a noisy 'best ever' across rotating seed
    pairs picks the luckiest draw, a noisy elite mean can drift; validation on shared seeds does neither.
    Without `validate`: cfg.select "mean" -> final elite mean, "best" -> best-ever sample.
    method="random": uniform samples, same budget. Stops before an iteration (never before the first)
    once time.time() >= deadline."""
    rng = np.random.default_rng(0) if rng is None else rng
    t0 = time.time()
    u0 = P.normalize(np.asarray(theta0, np.float64).reshape(P.DIM))
    mean = u0.copy()
    n_el = max(1, min(cfg.elites, cfg.population))
    best_u, best_cost = mean.copy(), float("inf")
    res = SearchResult(theta_star=P.denormalize(mean), best_cost=best_cost)
    iter_best: list[np.ndarray] = []
    for it in range(cfg.iterations):
        if deadline is not None and it > 0 and time.time() >= deadline:
            res.stopped_early = True
            break
        sigma = float(cfg.sigma_init_frac * cfg.sigma_decay ** it)
        if cfg.method == "random":
            U = rng.uniform(0.0, 1.0, size=(cfg.population, P.DIM))
        else:
            U = np.clip(mean + sigma * rng.standard_normal((cfg.population, P.DIM)), 0.0, 1.0)
            U[0] = u0                                       # elitism
        X = np.stack([P.denormalize(u) for u in U])
        costs = sanitize_costs(evaluate(X), cfg.population)
        res.evaluations += cfg.population
        order = np.argsort(costs, kind="stable")
        if costs[order[0]] < best_cost:
            best_cost, best_u = float(costs[order[0]]), U[order[0]].copy()
        iter_best.append(U[order[0]].copy())
        if cfg.method != "random":
            mean = U[order[:n_el]].mean(axis=0)
        res.history.append({"iter": it, "mean_cost": float(costs.mean()), "best_cost": float(costs[order[0]]),
                            "best_ever": best_cost, "elite_mean_cost": float(costs[order[:n_el]].mean()),
                            "sigma_mean": sigma, "n_evals": res.evaluations})
    if validate is not None and res.history:
        cands = [u0, mean] + iter_best
        keep = [i for i, u in enumerate(cands) if not any(np.allclose(u, cands[j]) for j in range(i))]
        cands = [cands[i] for i in keep]
        labels = [["incumbent", "final_mean"][i] if i < 2 else f"iter{i - 2}_best" for i in keep]
        vc = sanitize_costs(validate(np.stack([P.denormalize(u) for u in cands])), len(cands))
        res.evaluations += len(cands)
        k = int(np.argmin(vc))
        best_u, best_cost = cands[k].copy(), float(vc[k])
        res.validation = {"candidates": labels, "costs": [float(c) for c in vc], "chosen": labels[k]}
    elif cfg.select == "mean" and cfg.method != "random" and res.history:
        best_u, best_cost = mean.copy(), float(res.history[-1]["elite_mean_cost"])
    res.theta_star, res.best_cost, res.wallclock_s = P.denormalize(best_u), best_cost, time.time() - t0
    return res


# ----------------------------------------------------------------------------- gate (pure)
def episode_cost(ep: Episode | None) -> float:
    m = None if ep is None else ep.metrics
    if m is None or m.cost is None or not np.isfinite(m.cost):
        return FAILED_COST
    return float(m.cost)


def episode_success(ep: Episode | None) -> bool:
    """Success is the env flag recorded in Outcome; metrics.success mirrors it (never an LLM verdict)."""
    if ep is None:
        return False
    if ep.outcome is not None:
        return bool(ep.outcome.env_success)
    return bool(ep.metrics.success) if ep.metrics is not None else False


def summarize_rollouts(eps: Sequence[Episode | None]) -> tuple[float, float]:
    """(mean cost, success rate); an empty list is (FAILED_COST, 0.0)."""
    if not eps:
        return FAILED_COST, 0.0
    return (float(np.mean([episode_cost(e) for e in eps])),
            float(np.mean([episode_success(e) for e in eps])))


def gate_decision(candidate_eps: Sequence[Episode | None], incumbent_eps: Sequence[Episode | None],
                  slack: float = GATE_SUCCESS_SLACK) -> GateResult:
    """passed iff mean_cost(candidate) < mean_cost(incumbent) and success(candidate) >= success(incumbent) - slack."""
    cc, cs = summarize_rollouts(candidate_eps)
    ic, isr = summarize_rollouts(incumbent_eps)
    passed = bool(len(candidate_eps) > 0 and len(incumbent_eps) > 0 and cc < ic and cs >= isr - slack)
    return GateResult(passed=passed, incumbent_cost=ic, candidate_cost=cc, n_seeds=len(candidate_eps),
                      incumbent_success=isr, candidate_success=cs, success_delta_pp=100.0 * (cs - isr))


# --------------------------------------------------------------------------- runner glue
def seed_set(name: str, n: int) -> list[int]:
    try:
        from fleet_memory.runner.conditions import seed_set as _ss
        return [int(s) for s in _ss(name, n)]
    except ImportError:
        return [SEED_BASE[name] + i for i in range(n)]


def environment_id_of(template) -> str:
    tag = getattr(template, "environment_tag", "") or ""
    return f"{template.suite}_{template.task_id}" + (f"_{tag}" if tag else "")


def rollout_cfgs(template, arm: str, params: dict | None, seeds: Sequence[int], perturbation: dict) -> list:
    """Derive one RunConfig per seed from the template (arm, seed, s3_params, perturbation overridden)."""
    return [dataclasses.replace(template, arm=arm, seed=int(s), s3_params=params, perturbation=dict(perturbation))
            for s in seeds]


def _default_run_many(cfgs: list, workers: int) -> list:
    from fleet_memory.runner.pool import run_many
    return run_many(cfgs, workers)


class _Rollouts:
    """Counts rollouts; run_many is expected to return one Episode per cfg, in order."""

    def __init__(self, run_many: RunMany | None, workers: int):
        self.run_many, self.workers, self.n = run_many or _default_run_many, workers, 0

    def __call__(self, cfgs: list) -> list:
        if not cfgs:
            return []
        self.n += len(cfgs)
        eps = list(self.run_many(cfgs, self.workers) or [])
        return eps + [None] * (len(cfgs) - len(eps)) if len(eps) < len(cfgs) else eps[:len(cfgs)]


def make_evaluate(template, rollouts: _Rollouts, cfg: ConsolidationConfig, perturbation: dict,
                  opt_seeds: Sequence[int]) -> Evaluate:
    """evaluate(X) for cem_search: each call is one iteration; every candidate in it shares the same
    seeds_per_candidate seeds (common random numbers), rotating through opt_seeds by iteration."""
    state = {"it": 0}

    def evaluate(X: np.ndarray) -> np.ndarray:
        it, k = state["it"], max(1, cfg.seeds_per_candidate)
        state["it"] += 1
        seeds = [opt_seeds[(it * k + j) % len(opt_seeds)] for j in range(k)]
        X = np.asarray(X, np.float64).reshape(-1, P.DIM)
        cfgs, owner = [], []
        for i, x in enumerate(X):
            d = S3Params.from_array(x).to_dict()
            cfgs.extend(rollout_cfgs(template, "B", d, seeds, perturbation))
            owner.extend([i] * len(seeds))
        eps = rollouts(cfgs)
        sums, cnt = np.zeros(len(X)), np.zeros(len(X))
        for i, ep in zip(owner, eps):
            sums[i] += episode_cost(ep)
            cnt[i] += 1
        return np.where(cnt > 0, sums / np.maximum(cnt, 1.0), FAILED_COST)

    return evaluate


def _ensure_reference(store: EventStore, hm, si_id: str, env_id: str, template, rollouts: _Rollouts,
                      cfg: ConsolidationConfig, perturbation: dict) -> None:
    """Phase 0 if missing: Arm A rollouts on opt seeds -> frozen cost_reference (raw jerks, no ref yet)."""
    if hm.cost_reference(store, si_id) is not None:
        return
    eps = rollouts(rollout_cfgs(template, "A", None, seed_set("opt", max(4, cfg.gate_seeds)), perturbation))
    ms = [e.metrics for e in eps if e is not None and e.metrics is not None]
    if not ms:
        raise RuntimeError(f"no metrics from reference rollouts for {si_id}; cannot build a cost_reference")
    ref = costlib.make_reference(si_id, env_id, ms, [m.jerk for m in ms], condition="A")
    hm.set_cost_reference(store, ref)


def consolidate(store: EventStore, si_id: str, template, cfg: ConsolidationConfig | None = None,
                trigger: str = "manual", run_many: RunMany | None = None,
                rng: np.random.Generator | None = None) -> Consolidation:
    """One sleep cycle for `si_id`. Appends Consolidation(start) and Consolidation(end); promotes through
    house_model only if the gate passes. Rollouts go to template.log_path (arm B, environment_tag kept) and
    carry {"consolidation_id": ...} in Episode.perturbation so analysis can exclude them."""
    from fleet_memory.memory import house_model as hm
    cfg = cfg or ConsolidationConfig()
    t_start = time.time()
    env_id = environment_id_of(template)
    inc = hm.incumbent(store, si_id)
    if inc is None:
        skill = getattr(template, "skill", None) or template.task_id
        inc = hm.init_incumbent(store, si_id, env_id, skill, "target", S3Params.identity().to_dict())
    cid = new_id("cons")
    store.append(Consolidation(consolidation_id=cid, phase="start", skill_instance_id=si_id, trigger=trigger,
                               incumbent_version=inc.version, optimizer=cfg.optimizer_dict()))
    theta0 = S3Params.from_dict(inc.params).to_array()
    pert = {**(getattr(template, "perturbation", None) or {}), "jitter_m": cfg.pose_jitter_m,
            "consolidation_id": cid}
    rollouts = _Rollouts(run_many, cfg.workers)
    _ensure_reference(store, hm, si_id, env_id, template, rollouts, cfg, pert)

    rng = rng or np.random.default_rng(int(cid.rsplit("_", 1)[-1], 16) % (2 ** 32))
    opt_seeds = seed_set("opt", OPT_POOL)
    evaluate = make_evaluate(template, rollouts, cfg, pert, opt_seeds[: OPT_POOL // 2])
    validate = None
    if cfg.validation_seeds > 0:            # fresh common seeds from the other half of the opt pool
        vcfg = dataclasses.replace(cfg, seeds_per_candidate=cfg.validation_seeds)
        validate = make_evaluate(template, rollouts, vcfg, pert, opt_seeds[OPT_POOL // 2:])
    sr = cem_search(evaluate, theta0, cfg, rng, deadline=t_start + cfg.max_wallclock_s, validate=validate)
    best = S3Params.from_array(sr.theta_star)

    gate, promoted = None, None
    if not np.allclose(best.to_array(), theta0):            # identical vector cannot beat itself: skip the gate
        gseeds = seed_set("gate", cfg.gate_seeds)
        cand_cfgs = rollout_cfgs(template, "B", best.to_dict(), gseeds, pert)
        inc_cfgs = rollout_cfgs(template, "B", S3Params.from_array(theta0).to_dict(), gseeds, pert)
        eps = rollouts(cand_cfgs + inc_cfgs)
        gate = gate_decision(eps[:len(cand_cfgs)], eps[len(cand_cfgs):])
        if gate.passed:
            promoted = hm.promote(store, si_id, best.to_dict(), gate, cid).version
    end = Consolidation(consolidation_id=cid, phase="end", skill_instance_id=si_id, trigger=trigger,
                        incumbent_version=inc.version, optimizer=cfg.optimizer_dict(), rollouts=rollouts.n,
                        wallclock_s=time.time() - t_start,
                        best_candidate={"params": best.to_dict(), "opt_cost": sr.best_cost,
                                        "stopped_early": sr.stopped_early, "evaluations": sr.evaluations,
                                        "validation": sr.validation},
                        gate=gate, promoted_version=promoted, history=sr.history)
    store.append(end)
    return end


# ----------------------------------------------------------------------------------- CLI
def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Fleet Memory v3 sleep loop: CEM + gate + promotion for one skill instance")
    ap.add_argument("--env", default="mock")
    ap.add_argument("--policy", default="mock")
    ap.add_argument("--suite", default="mock")
    ap.add_argument("--task", default="pick_bowl_to_plate")
    ap.add_argument("--log", default="logs/events.jsonl")
    ap.add_argument("--tag", default="", help="environment_tag appended to the environment_id")
    ap.add_argument("--skill", default=None)
    ap.add_argument("--perturb", default=None, help="JSON perturbation dict applied to every rollout")
    ap.add_argument("--small", action="store_true", help="ConsolidationConfig.small() budget")
    for k in ("population", "elites", "iterations", "seeds-per-candidate", "gate-seeds", "workers"):
        ap.add_argument(f"--{k}", type=int, default=None)
    ap.add_argument("--pose-jitter-m", type=float, default=None)
    ap.add_argument("--max-wallclock-s", type=float, default=None)
    ap.add_argument("--method", choices=["cem", "random"], default=None)
    ap.add_argument("--trigger", choices=["manual", "scheduled", "drift"], default="manual")
    ap.add_argument("--rng-seed", type=int, default=None)
    return ap


def config_from_args(a: argparse.Namespace) -> ConsolidationConfig:
    cfg = ConsolidationConfig.small() if a.small else ConsolidationConfig()
    for k in ("population", "elites", "iterations", "seeds_per_candidate", "gate_seeds", "workers",
              "pose_jitter_m", "max_wallclock_s", "method"):
        v = getattr(a, k, None)
        if v is not None:
            setattr(cfg, k, v)
    return cfg


def main(argv: list[str] | None = None) -> Consolidation:
    a = _build_parser().parse_args(argv)
    from fleet_memory.runner.worker import RunConfig
    template = RunConfig(suite=a.suite, task_id=a.task, seed=0, arm="B", env_kind=a.env, policy_kind=a.policy,
                         log_path=a.log, environment_tag=a.tag, skill=a.skill,
                         perturbation=json.loads(a.perturb) if a.perturb else None)
    si_id = skill_instance_id(a.skill or a.task, environment_id_of(template))
    rng = None if a.rng_seed is None else np.random.default_rng(a.rng_seed)
    end = consolidate(EventStore(a.log), si_id, template, config_from_args(a), trigger=a.trigger, rng=rng)
    g = end.gate
    print(json.dumps({"consolidation_id": end.consolidation_id, "skill_instance_id": si_id,
                      "incumbent_version": end.incumbent_version, "promoted_version": end.promoted_version,
                      "rollouts": end.rollouts, "wallclock_s": round(end.wallclock_s, 1),
                      "opt_cost": None if end.best_candidate is None else end.best_candidate["opt_cost"],
                      "gate": None if g is None else {"passed": g.passed, "incumbent_cost": g.incumbent_cost,
                                                      "candidate_cost": g.candidate_cost,
                                                      "success_delta_pp": g.success_delta_pp}}))
    return end


if __name__ == "__main__":
    main()
