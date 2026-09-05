"""EWMA drift detector. Appends a `drift_trigger` event when the incumbent skill instance's cost
drifts above its promotion-time baseline, or its recent success rate drops. The sleep loop
(runner/consolidate.py, via pool.py --auto-sleep) reacts to the trigger; this file never runs a rollout.

Rules (DriftConfig):
  * only the incumbent version's own episodes count (consolidation rollouts excluded);
  * nothing fires until >= `window` episodes exist on the version;
  * baseline_cost / baseline_success_rate come from the latest scorecard (set at promotion); when the
    version was never gated (init v1) the mean of its first `window` episodes is the baseline;
  * the EWMA is recomputed from the version's episode costs with `alpha` (identical to the scorecard's
    ewma_cost when house_model.update_scorecard ran with the same alpha);
  * fire iff ewma > cost_ratio * baseline_cost  OR  recent_success < baseline_success - success_drop;
  * noise floor (`noise_sigmas`): both thresholds are widened to at least `noise_sigmas` standard errors
    of the statistic, estimated from the version's first `window` episodes (EWMA std = sd * sqrt(a/(2-a));
    success-rate difference std = sqrt(2 p (1-p) / window)). On the mock, a 45%-success v1 incumbent has
    a bimodal cost (~1 vs ~4.3, sd 1.6) whose EWMA swings 2.3..4.8 on noise alone, so the bare 1.25x
    ratio fired on unperturbed episodes; a gated version (sd ~0.02) keeps the bare spec thresholds;
  * cooldown: no fire while fewer than `cooldown_episodes` episodes of this skill instance were logged
    after the most recent consolidation (any phase) or drift_trigger for it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fleet_memory.memory import house_model as hm
from fleet_memory.memory.schema import DriftTrigger, Episode, record_from_dict


@dataclass
class DriftConfig:
    alpha: float = 0.3
    cost_ratio: float = 1.25
    success_drop: float = 0.15
    window: int = 10
    cooldown_episodes: int = 20
    noise_sigmas: float = 2.5


def noise_floors(first: list[Episode], cfg: DriftConfig) -> tuple[float, float]:
    """(cost margin, success-rate margin) = noise_sigmas standard errors, from the version's first episodes."""
    costs = [c for c in (hm.episode_cost(e) for e in first) if c is not None]
    oks = [s for s in (hm.episode_success(e) for e in first) if s is not None]
    sd = float(np.std(costs)) if len(costs) > 1 else 0.0
    cost_margin = cfg.noise_sigmas * sd * float(np.sqrt(cfg.alpha / (2.0 - cfg.alpha)))
    p = float(np.mean(oks)) if oks else 0.0
    sr_margin = cfg.noise_sigmas * float(np.sqrt(2.0 * p * (1.0 - p) / max(cfg.window, 1)))
    return cost_margin, sr_margin


def episodes_since_last_marker(store, si_id: str, episode: Episode | None = None) -> int | None:
    """Episodes of `si_id` (any version, rollouts excluded) logged after its most recent consolidation or
    drift_trigger event. None when no marker exists. `episode` counts as the newest if not logged yet."""
    marker, count, seen = False, 0, False
    for d in store.read_all():
        t = d.get("type")
        if t in ("consolidation", "drift_trigger") and d.get("skill_instance_id") == si_id:
            marker, count, seen = True, 0, False
        elif t == "episode" and hm.belongs(d, si_id):
            count += 1
            if episode is not None and d.get("episode_id") == episode.episode_id:
                seen = True
    if not marker:
        return None
    if episode is not None and not seen and hm.belongs(episode, si_id):
        count += 1
    return count


def _rate(vals: list[bool]) -> float | None:
    return float(np.mean(vals)) if vals else None


def check(store, si_id: str, episode: Episode, cfg: DriftConfig = DriftConfig()) -> DriftTrigger | None:
    """Evaluate the rules without appending anything. Returns the would-be trigger or None."""
    inc = hm.incumbent(store, si_id)
    if inc is None:
        return None
    eps = hm.version_episodes(store, si_id, inc.version, extra=episode)
    if len(eps) < cfg.window:
        return None
    costs = [c for c in (hm.episode_cost(e) for e in eps) if c is not None]
    if not costs:
        return None                                          # no cost reference yet -> no baseline
    sc = inc.scorecard
    ewma = hm.ewma(costs, cfg.alpha)
    baseline = sc.baseline_cost
    if baseline is None:
        first = [c for c in (hm.episode_cost(e) for e in eps[: cfg.window]) if c is not None]
        if not first:
            return None
        baseline = float(np.mean(first))
    base_sr = sc.baseline_success_rate
    if base_sr is None:
        base_sr = _rate([s for s in (hm.episode_success(e) for e in eps[: cfg.window]) if s is not None])
    recent_sr = _rate([s for s in (hm.episode_success(e) for e in eps[-cfg.window:]) if s is not None])

    cost_margin, sr_margin = noise_floors(eps[: cfg.window], cfg)
    reason = None
    if baseline > 0 and ewma > max(cfg.cost_ratio * baseline, baseline + cost_margin):
        reason = "ewma_cost_exceeds_baseline"
    elif (base_sr is not None and recent_sr is not None
          and recent_sr < base_sr - max(cfg.success_drop, sr_margin)):
        reason = "success_rate_drop"
    if reason is None:
        return None
    since = episodes_since_last_marker(store, si_id, episode)
    if since is not None and since < cfg.cooldown_episodes:
        return None                                          # cooldown
    ratio = ewma / baseline if baseline > 0 else float("inf")
    return DriftTrigger(skill_instance_id=si_id, reason=reason, ewma_cost=float(ewma),
                        baseline_cost=float(baseline), ratio=float(ratio),
                        recent_success_rate=float(recent_sr if recent_sr is not None else 0.0),
                        baseline_success_rate=float(base_sr if base_sr is not None else 0.0),
                        episode_id=episode.episode_id)


def update_and_check(store, si_id: str, episode: Episode, cfg: DriftConfig = DriftConfig()) -> DriftTrigger | None:
    """Run the detector after `episode` (already or about to be logged); append and return the trigger."""
    trig = check(store, si_id, episode, cfg)
    if trig is not None:
        store.append(trig)
    return trig


def triggers(store, si_id: str | None = None) -> list[DriftTrigger]:
    return [record_from_dict(d) for d in store.iter_type("drift_trigger")
            if si_id is None or d.get("skill_instance_id") == si_id]


def pending(store, si_id: str) -> DriftTrigger | None:
    """The latest drift_trigger for si_id with no later consolidation 'end' event (pool.py --auto-sleep)."""
    last = None
    for d in store.read_all():
        if d.get("skill_instance_id") != si_id:
            continue
        if d.get("type") == "drift_trigger":
            last = d
        elif d.get("type") == "consolidation" and d.get("phase") == "end":
            last = None
    return record_from_dict(last) if last else None
