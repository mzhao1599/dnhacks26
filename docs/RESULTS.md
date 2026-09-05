# Fleet Memory v3.1 — Results (2026-09-05, GMU Hopper)

Frozen **SmolVLA** (`HuggingFaceVLA/smolvla_libero`, 10-action chunks, policy RNG seeded per episode so arms on
the same seed share the policy's noise) on **LIBERO** (robosuite/MuJoCo, EGL on A100.40GB, OSMesa on MIG).
Success = the benchmark's programmatic predicate, always. Every number below is in `logs/hopper/**/events.jsonl`
(merged: `scripts/pull_logs.sh` → `logs/demo.jsonl` → `dashboard/index.html`).

## 1. Phase 0 — base rate and cost reference (LIBERO-10 task 3, "put the black bowl in the bottom drawer of the cabinet and close it")
Arm A (raw VLA + envelope), 20 train seeds: **55% [34, 74]**, mean **378 steps**. `steps_ref=377.95`, `jerk_ref` frozen
as a `cost_reference` event. ~20 s per successful episode, 38 s per 520-step timeout.
Chunk-1 vs chunk-10 execution: 4/4 vs 4/4 on seeds 0–3 (chunking not visibly costing success; everything runs chunk-10).

## 2. Phase 1 — does the S3 surface steer this policy? (same 20 seeds, same policy noise; arm B = shim on)
| S3 vector | success | mean steps |
|---|---|---|
| identity (true pass-through) | 80% [58, 92] | 294 |
| `time_scale = 1.3` | 80% | **266** (−9%) |
| `blend_alpha = 0.5` (approach-shaping on, v3.0 default) | 55% | 395 |
| bad offset (+3 cm xy, −2 cm z, blend on) | 45% | 419 |
| homing to a **wrong** constant pose (bug, fixed) | 5% | 518 |

Findings: the shim bites in both directions; approach-phase blending (waypoint + cone + α=0.5) *hurts* this VLA
(first measured as 40% vs 55% for arm A with the v3.0 default — so `blend_alpha` became an optimisable dim with
identity = pure pass-through). Arm A vs identity-B disagree per seed in both directions → policy stochasticity is large;
hence per-episode policy seeding. Homing must target the env's own reset pose (LIBERO-10 starts at z≈1.17, not the
0.70 an early adapter reported).

## 3. Sleep loop on the real env (LIBERO-10 task 3)
- v3.0 semantics (blending on by default): CEM 16×2×4 + gate 16 seeds, 192 rollouts, 25 min → **gate passed, v1→v2
  promoted** (incumbent cost 3.56 → lower); arm B after sleep 55%/380 steps — i.e. it merely undid the harmful
  default (v1 was handicapped), not an improvement over the raw VLA.
- v3.1 semantics (identity = pass-through), cycle 1: opt-cost 1.75 on its 2 seeds/candidate, but on the 12 fresh
  gate seeds **3.53 vs incumbent 2.09, −37 pp** → **gate refused**. Winner's curse at K=2 seeds; the gate is the whole
  defence and it worked. Cycle 2/3: PENDING.
- Gate resolution at n=12–16 seeds and ~50% success is ±20 pp: fine for large effects (homing), not for
  time-scale-sized ones. Report gate pass rate honestly.

## 4. LIBERO-Plus robot-initial-state benchmark (LIBERO-Spatial, our own base numbers — π₀.₅ adapter unverified)
Perturbation = LIBERO-Plus `initstate_N` (joint qpos offset r=0.1 rad, seed-42 table), re-applied after
`set_init_state` (their loop's `set_init_state` overwrites it; only the OSC nullspace target survives — see
`scripts/hopper/LIBERO_PLUS.md`). Eval seeds = init states 40–49, never seen by optimizer (0–29) or gate (30–39).

Task 0 probe (10 eval seeds each):
| arm | success | steps |
|---|---|---|
| BM-0 standard | 8/10 | 102 |
| BM-1 perturbed | 3/10 | 178 |
| BM-2 perturbed + untrained shim | 4/10 | 161 |
| BM-4 perturbed + hand-set homing | 6/10 | 136 |

BM-3 (perturbed + consolidated incumbent after ONE unattended sleep on opt/gate seeds): PENDING per task 0–4; pooled
n=50 via `python -m fleet_memory.runner.benchmark --aggregate`. During CEM on task 0, homing-on rollouts succeeded
50% vs 29% homing-off and the population drifted toward homing (the optimizer is finding the switch). Tasks 1–2 are
hard for this policy even with EE-space homing (≤6%): the joint configuration still differs, which changes the
wrist-camera view — a limit of EE-space homing that joint-space homing on a real controller would not have.

## 5. Arm C (Gemini planner `gemini-3.7-flash` + inner coach `gemini-3.1-pro-preview`), LIBERO-10 task 3
First run (12 seeds): 17%, 173 mean steps — plans were sensible (reach/grasp/lift/place/close, canonical instruction,
no prose leaks, 0 interventions) but the planner's per-subtask step budgets (~460 total) truncated the VLA.
Fixed (plan exhaustion no longer ends the episode); rerun on 20 seeds: PENDING.

## 6. Protocol P (baseline → shift bowl 6 cm → drift → auto-sleep → recovery), LIBERO-Spatial task 0: PENDING.

## Honesty lines
- Base numbers are ours (SmolVLA), on LIBERO-Plus's exact robot-init perturbation; the CVPR table is π₀/OpenVLA.
- The shim reads object pose from simulator state as a stand-in for a detector.
- Frozen VLA in every arm, zero demonstrations; the only thing that changes between BM-1 and BM-3 is one
  versioned, gated parameter file (`skill_instance` events).
- LIBERO has no force sensor: `force_proxy` is an action-magnitude proxy.
- Homing runs in EE space, not joint space (LIBERO's action interface is OSC delta-pose; the VLA's proprio is EE pose).
