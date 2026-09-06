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

## 3. Sleep loop on the real env — mastery (LIBERO-10 task 3, arm B = shim + incumbent, 20 train seeds after each cycle)
| cycle | gate | incumbent after | arm B success | steps | cost |
|---|---|---|---|---|---|
| — (identity v1) | — | v1 | 65% [43, 82] | 341 | 2.46 |
| 1 | refused (cand 3.53 vs 2.09 on gate seeds, −37 pp) | v1 | — | — | — |
| 2 | passed | **v2** | **80% [58, 92]** | **275** | **1.95** |
| 3 | passed | v3 | 60% [39, 78] | 354 | 2.57 |

The gate is the whole defence and it works (cycle 1: a lucky-on-2-seeds candidate refused), but its resolution at
n=12–16 seeds and ~50% success is ±20 pp: cycle 3's promotion regressed on the train seeds. Gate pass rate 2/3.
The mastery curve is v1 2.46 → v2 1.95 → v3 2.57: not monotone; report as measured. (An earlier v3.0-semantics
cycle also promoted, 3.56 → lower, but its v1 was the harmful blending default, so that was undoing damage, not mastery.)
Seeded arm A baseline on the same seeds: PENDING (Phase 0's 55%/378 predates policy seeding).

## 4. LIBERO-Plus robot-initial-state benchmark (LIBERO-Spatial, SmolVLA, our own base numbers)
Perturbation = LIBERO-Plus `initstate_N` joint offset (r=0.1 rad, seed-42 table; task 3 drew r=0.2), re-applied
after `set_init_state` (their loop's `set_init_state` overwrites it — `scripts/hopper/LIBERO_PLUS.md`). Eval seeds =
init states 40–49 per task, never seen by the optimizer (0–29) or the gate (30–39). Policy noise seeded per episode.

| task | BM-0 standard | BM-1 perturbed | BM-2 + untrained shim | BM-4 + hand-set homing | BM-3 after ONE unattended sleep |
|---|---|---|---|---|---|
| 0 | 9/10 | 5/10 | 4/10 | 4/10 | 5/10 (gate: 17%→67% on gate seeds; eval: no change) |
| 1 | 6/10 | 0/10 | 0/10 | 2/10 | cycle 1 refused; cycle 2 refused (17%→0%) — stays v1 |
| 2 | 10/10 | 0/10 | 0/10 | 0/10 | 0/10 (gate passed on cost only, 0%→0%) |
| 3 | 9/10 | 2/10 | 2/10 | 3/10 | cycle 1 refused; cycle 2 (σ₀=0.5) passed 4.11→3.65 → **2/10** (= BM-1) |
| 4 | 6/10 | 0/10 | 0/10 | 1/10 | cycle 1 refused; cycle 2: validation kept the incumbent — stays v1 |
| **pooled** | **40/50 = 80%** | **7/50 = 14% [7, 26]** | 6/50 = 12% | **10/50 = 20% [11, 33]** | 7/30 = 23% on tasks 0/2/3 vs BM-1 7/30 there → 1.00× [0.40, 2.50], **fail at n=10** |

Wide-search variant (task 0, σ₀=0.5, separate skill instance, MIG/OSMesa renderer): BM-1 3/10 → BM-4 5/10 →
**BM-3 7/10** (gate passed 2.94 → 2.72). Same 10 eval layouts; renderer differs from the A100 run, so BM-1 differs too.

**Headline with real n (task 0, the 10 held-out layouts × 5 policy-noise draws, n=50 per arm, all on layouts the optimizer
and gate never saw):**
| arm | success | 95% CI | steps |
|---|---|---|---|
| BM-1 perturbed | 10/50 = **20%** | [11, 33] | 196 |
| BM-2 perturbed + untrained shim | 14/50 = 28% | [17, 42] | 185 |
| BM-4 perturbed + hand-set homing | 17/50 = 34% | [22, 48] | 173 |
| **BM-3 perturbed + ONE unattended sleep** | **25/50 = 50%** | **[37, 63]** | 162 |
| BM-3w perturbed + wide-search sleep (separate instance) | 21/50 = 42% | [29, 56] | 179 |

BM-3 vs BM-1: 2.5×, intervals disjoint → **pass** by §13.5 on task 0. The optimizer's vector (homing + time_scale 0.85 +
grasp offset −1.8 cm + cone 34°) beats hand-set homing alone (50% vs 34%): it found more than the switch.

Verdict by the spec's own rule (§13.5), all five tasks at n=10: **collapse reproduced (80% → 14%)**; the untrained shim adds nothing
(BM-2 ≈ BM-1); hand-set homing recovers 6 points pooled (**partial**, CIs overlap); one unattended sleep found homing
on task 0 (gate +50 pp) but the eval layouts did not confirm it (**fail on the main run, partial on the wide run**).
Per-seed view: on task 0 BM-1 succeeds on layouts 40–44 and fails 45–49; homing flips that (fails 40–42, rescues
47–49) — it moves which layouts succeed rather than expanding the set, and 10 layouts per task cannot resolve that.
BM-1 vs BM-4 on all 50 init states of tasks 0 and 3 (n=100/arm; hand-set → no held-out concern): PENDING.
Tasks 1–2 are hard even with EE-space homing: the joint configuration still differs (wrist-camera view), a limit
joint-space homing on a real controller would not have.

Gate tally, all real-env sleep cycles tonight: 11 attempted → 4 promoted (task 0, task 2 cost-only, task 3 cycle 2, mastery v2/v3 = 5 incl. mastery), 7 refused or validation-kept-incumbent. Every refusal held up on later held-out evidence; one promotion (mastery v3) did not.

## 5. Arm C (Gemini planner `gemini-3.7-flash` + inner coach `gemini-3.1-pro-preview`), LIBERO-10 task 3
First run (12 seeds): 17%, 173 mean steps — plans sensible (reach/grasp/lift/place/close, canonical instruction,
no prose leaks, 0 interventions) but per-subtask step budgets (~460 total) truncated the VLA. Fixed (plan exhaustion
no longer ends the episode). Rerun, 20 seeds: **75% [53, 89], 295 steps** — i.e. ≈ the identity-shim arm (80%/294):
the S1 planner neither helps nor hurts this single-skill task once it stops truncating.

## 6. Protocol P (LIBERO-Spatial task 0, real env, unattended): an honest negative
baseline 8/15 (cost 3.00, 144 steps) → bowl shifted 6 cm → perturbed **1/15** (cost 4.87) → **drift_trigger fired**
(EWMA 4.43 vs baseline 2.37) → auto-sleep #1 (176 rollouts, 44 min on a MIG slice): CEM ran 4 iterations, the
fresh-seed validation chose the **incumbent** (4.74) over every candidate (4.88–5.05) → nothing to gate, no promotion →
drift fired again (5.18) → auto-sleep #2: same outcome → recovery stage **0/15** (cost 5.19).
Reading: the loop did everything it should without a human — detect, sleep, refuse. But a 6 cm object shift is outside
what this S3 vector can express (approach offsets are bounded ±3 cm; the VLA never finds the moved bowl), so there was
nothing legitimate to promote. Re-adaptation: not achieved; reported as such. (On the mock env the same protocol
recovered.)

## 7. Demo assets
- Clips (same seed across arms, `scripts/exp/record_demo.py`, `logs/hopper/videos/`): seed 5047 on LIBERO-Spatial task 0 —
  BM-0 standard **success 69 steps**, BM-1 perturbed **fail**, BM-4 hand-set homing **fail on this seed**, BM-3 consolidated
  vector **success 99 steps**; LIBERO-10 task 3 seed 0 — raw **fail at horizon**, v2 **success 379 steps**.
- Storyboard with embedded clips: https://claude.ai/code/artifact/622fb72e-c0aa-4a88-a23f-5ab8ea2b7957 ·
  analyst dashboard: https://claude.ai/code/artifact/5f1ec73c-fd21-4b6b-845b-a4b1f8432c2e · rebuild: `scripts/pull_logs.sh`,
  `scripts/build_demo_page.py`, then publish `dashboard/demo_artifact.html` / `dashboard/artifact.html`.
- Held-out mastery (LIBERO-10 task 3, layouts 20–39, n=40/arm): A 78% [62, 88] / 302 steps / cost 2.03 → **v2 80% [65, 90] /
  278 steps / cost 1.95** → v3 68% [52, 80] / 335 steps / cost 2.31 (a gate false-positive at n=12; regressed). Homing alone, all 50 layouts of tasks 0+3 (n=100/arm, no optimizer): **BM-1 27% [19, 36] → BM-4 33% [25, 43]** — +6 pp, CIs overlap: real but small.

## Honesty lines
- Base numbers are ours (SmolVLA), on LIBERO-Plus's exact robot-init perturbation; the CVPR table is π₀/OpenVLA.
- The shim reads object pose from simulator state as a stand-in for a detector.
- Frozen VLA in every arm, zero demonstrations; the only thing that changes between BM-1 and BM-3 is one
  versioned, gated parameter file (`skill_instance` events).
- LIBERO has no force sensor: `force_proxy` is an action-magnitude proxy.
- Homing runs in EE space, not joint space (LIBERO's action interface is OSC delta-pose; the VLA's proprio is EE pose).
