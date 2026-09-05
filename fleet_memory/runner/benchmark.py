"""Benchmark arms BM-0..BM-4 on LIBERO-Plus perturbation configs (spec §13).

    BM-0  frozen VLA + envelope, standard LIBERO          (sanity)
    BM-1  frozen VLA + envelope, perturbed                (the collapse)
    BM-2  + shim with default params (homing off)         (the layer adds nothing until it learns)
    BM-3  + shim with the consolidated incumbent          (the claim: after one unattended sleep)
    BM-4  + shim with hand-set homing_enable=1, delta=0   (did the optimizer find what a human would?)

All arms share the eval seed set from runner/seeds.json (init states 40-49 per task), which the
optimizer (0-29) and the gate (30-39) never see. Same tasks, same configs, same success predicate.

python -m fleet_memory.runner.benchmark --policy smolvla --suite libero_spatial --tasks 0,1,2,3,4 \
    --dimension robot_init --config-index 0 --arms BM-0,BM-1,BM-2,BM-4 --n-per-task 10 --workers 4 \
    --log logs/benchmark/events.jsonl [--consolidate small|medium] [--png logs/benchmark/bm.png]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from typing import Any

import numpy as np

from fleet_memory.analysis.metrics import wilson_ci
from fleet_memory.execution.params import S3Params
from fleet_memory.memory.schema import now_iso
from fleet_memory.memory.store import EventStore
from fleet_memory.runner import seeds as seedmod
from fleet_memory.runner.pool import run_many, summarize
from fleet_memory.runner.worker import RunConfig, incumbent_of

ARMS = ["BM-0", "BM-1", "BM-2", "BM-3", "BM-4"]


def arm_config(arm: str, base: RunConfig, perturbed_kwargs: dict, tag: str, store: EventStore) -> RunConfig:
    """Translate a BM arm into a RunConfig. BM-3 needs an incumbent for the perturbed skill instance."""
    if arm == "BM-0":
        return dataclasses.replace(base, arm="A", env_kind="libero", env_kwargs={}, environment_tag="")
    pert = dict(env_kind="libero_plus", env_kwargs=perturbed_kwargs, environment_tag=tag)
    if arm == "BM-1":
        return dataclasses.replace(base, arm="A", **pert)
    if arm == "BM-2":
        return dataclasses.replace(base, arm="B", s3_params=S3Params.identity().to_dict(), **pert)
    if arm == "BM-4":
        p = S3Params.identity(); p.homing_enable = 1.0
        return dataclasses.replace(base, arm="B", s3_params=p.to_dict(), **pert)
    if arm == "BM-3":
        cfg = dataclasses.replace(base, arm="B", **pert)
        inc = incumbent_of(store, cfg.skill_instance_id)
        if inc is None or int(inc.version) < 2:
            raise RuntimeError(f"BM-3 needs a consolidated incumbent (version>=2) for {cfg.skill_instance_id}; "
                               f"run --consolidate first")
        return dataclasses.replace(cfg, s3_params=dict(inc.params))
    raise ValueError(arm)


def ratio_ci(k_num: int, n_num: int, k_den: int, n_den: int, z: float = 1.96) -> tuple[float, float, float]:
    """Risk ratio with a log-normal CI (0.5 continuity correction when a cell is zero)."""
    a, b = (k_num + 0.5, k_den + 0.5) if (k_num == 0 or k_den == 0) else (k_num, k_den)
    rr = (a / n_num) / (b / n_den)
    se = math.sqrt(max(1 / a - 1 / n_num, 0) + max(1 / b - 1 / n_den, 0))
    return rr, rr * math.exp(-z * se), rr * math.exp(z * se)


def run_arm(arm: str, base: RunConfig, tasks: list[str], perturbed_kwargs: dict, tag: str, n_per_task: int,
            workers: int, store: EventStore) -> dict:
    cfgs = []
    for t in tasks:
        cfg = arm_config(arm, dataclasses.replace(base, task_id=t), perturbed_kwargs, tag, store)
        cfgs += [dataclasses.replace(cfg, seed=s) for s in seedmod.seeds("eval", n_per_task)]
    eps = run_many(cfgs, workers)
    print(summarize(eps, arm), flush=True)
    k, n = sum(1 for e in eps if e.outcome and e.outcome.env_success), len(eps)
    lo, hi = wilson_ci(k, n)
    return {"arm": arm, "n": n, "k": k, "rate": k / n if n else 0.0, "ci": [lo, hi],
            "mean_steps": float(np.mean([e.outcome.steps for e in eps if e.outcome])) if n else None,
            "episode_ids": [e.episode_id for e in eps],
            "s3_params": cfgs[0].s3_params if cfgs else None}


def bar_chart(results: list[dict], out: str, title: str) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    arms = [r["arm"] for r in results]
    rates = [100 * r["rate"] for r in results]
    err = [[100 * (r["rate"] - r["ci"][0]) for r in results], [100 * (r["ci"][1] - r["rate"]) for r in results]]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(arms, rates, yerr=err, capsize=4, color=["#888", "#c33", "#c93", "#393", "#39c"][:len(arms)])
    for i, r in enumerate(results):
        ax.text(i, rates[i] + 2, f"{r['k']}/{r['n']}", ha="center", fontsize=9)
    ax.set_ylabel("success rate (%) — 95% Wilson CI")
    ax.set_ylim(0, 105)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="smolvla")
    ap.add_argument("--suite", default=seedmod.BENCHMARK["suite"])
    ap.add_argument("--tasks", default=",".join(str(t) for t in seedmod.BENCHMARK["tasks"]))
    ap.add_argument("--dimension", default=seedmod.BENCHMARK["perturbation_dimension"])
    ap.add_argument("--config-index", type=int, default=0)
    ap.add_argument("--arms", default="BM-0,BM-1,BM-2,BM-4")
    ap.add_argument("--n-per-task", type=int, default=seedmod.BENCHMARK["n_eval_per_task"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log", default="logs/benchmark/events.jsonl")
    ap.add_argument("--tag", default=None, help="environment tag for the perturbed skill instance")
    ap.add_argument("--consolidate", default=None, choices=[None, "small", "medium", "full"],
                    help="run ONE sleep cycle on the perturbed skill instance (opt+gate seeds) before BM-3")
    ap.add_argument("--png", default=None)
    ap.add_argument("--cem", default=None, help='JSON overrides for ConsolidationConfig, e.g. {"population":24,"iterations":4}')
    a = ap.parse_args(argv)

    from fleet_memory.envs.libero_plus import list_configs
    configs = list_configs(a.dimension, a.suite)
    if not configs:
        print(f"no LIBERO-Plus configs for dimension={a.dimension} suite={a.suite}", file=sys.stderr)
        return 2
    config = configs[a.config_index]
    tag = a.tag or f"plus_{a.dimension}_{a.config_index}"
    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    store = EventStore(a.log)
    base = RunConfig(suite=a.suite, task_id=tasks[0], seed=0, arm="A", env_kind="libero", policy_kind=a.policy,
                     log_path=a.log)
    perturbed_kwargs = {"config": config}
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]

    if a.consolidate:
        from fleet_memory.runner.consolidate import ConsolidationConfig, consolidate
        ccfg = {"small": ConsolidationConfig.small, "medium": ConsolidationConfig.medium,
                "full": ConsolidationConfig}[a.consolidate]()
        ccfg.workers = a.workers
        for k, v in (json.loads(a.cem) if a.cem else {}).items():
            setattr(ccfg, k, v)
        for t in tasks:                       # one skill instance per (task, perturbed env)
            tmpl = dataclasses.replace(base, task_id=t, arm="B", env_kind="libero_plus", env_kwargs=perturbed_kwargs,
                                       environment_tag=tag)
            c = consolidate(store, tmpl.skill_instance_id, template=tmpl, cfg=ccfg, trigger="manual")
            print(f"consolidation {c.consolidation_id} task={t} gate={c.gate.to_dict() if c.gate else None} "
                  f"promoted={c.promoted_version}", flush=True)
        if "BM-3" not in arms:
            arms.append("BM-3")

    results = [run_arm(arm, base, tasks, perturbed_kwargs, tag, a.n_per_task, a.workers, store) for arm in arms]
    by = {r["arm"]: r for r in results}
    ratio = None
    if "BM-3" in by and "BM-1" in by:
        rr, lo, hi = ratio_ci(by["BM-3"]["k"], by["BM-3"]["n"], by["BM-1"]["k"], by["BM-1"]["n"])
        ratio = {"BM-3/BM-1": rr, "ci": [lo, hi],
                 "verdict": ("pass" if by["BM-3"]["ci"][0] > by["BM-1"]["ci"][1] else
                             "partial" if by["BM-3"]["rate"] > by["BM-1"]["rate"] else "fail")}
    event = {"type": "benchmark_result", "ts": now_iso(), "suite": a.suite, "tasks": tasks, "dimension": a.dimension,
             "config_index": a.config_index, "config": config, "policy": a.policy, "tag": tag,
             "n_per_task": a.n_per_task, "results": results, "ratio": ratio}
    store.append(event)
    print("\n| arm | n | success | 95% CI | mean steps |\n|---|---|---|---|---|")
    for r in results:
        print(f"| {r['arm']} | {r['n']} | {100 * r['rate']:.1f}% ({r['k']}) | [{100 * r['ci'][0]:.1f}, "
              f"{100 * r['ci'][1]:.1f}] | {r['mean_steps']:.0f} |")
    if ratio:
        print(f"\nBM-3 / BM-1 = {ratio['BM-3/BM-1']:.2f} [{ratio['ci'][0]:.2f}, {ratio['ci'][1]:.2f}] -> {ratio['verdict']}")
    if a.png:
        print("chart:", bar_chart(results, a.png, f"{a.policy} on {a.suite} — LIBERO-Plus {a.dimension}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
