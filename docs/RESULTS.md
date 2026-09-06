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

BM-3 vs BM-1: 2.5×, intervals disjoint → **pass** by §13.5 on task 0.

**Replication** (same 10 held-out layouts, 5 *new* policy-noise draws, n=50): BM-0 40/50 = 80% · BM-1 12/50 = 24% [14, 37] ·
BM-2 13/50 = 26% · BM-4 17/50 = 34% · **BM-3 29/50 = 58% [44, 71]** · BM-3w 21/50 = 42%. **Pooled n=100 per arm:**
BM-1 **22%** [15, 31] → BM-4 34% [25, 44] → **BM-3 54% [44, 64]**, ratio **2.45×**, intervals disjoint. The headline replicates.

**Second unattended sleep cycle (strong gate: 24 gate seeds × 4 draws), from v2:** CEM candidate passed the gate,
cost 1.946 → 1.626, success 70.8% → 79.2% on the gate layouts (+8.3 pp), promoted as **v3**. Held-out re-eval on a
third independent set of 5 noise draws (n=50): BM-1 11/50 = 22% · BM-2 13/50 = 26% · BM-4 21/50 = 42% ·
**BM-3 (v3) 35/50 = 70% [56, 81], 141 steps.** The perturbed-environment mastery curve is therefore
**v1 22% → v2 54% → v3 70%** (identity → one cycle → two cycles), each step through a gate on layouts the
optimizer never saw, evaluated on layouts neither saw. v3 = homing on, time_scale 0.755, velocity_cap 0.94,
gripper_cmd 0.91, grasp offset −1.3 cm, cone 40°: the second cycle mostly slowed the chunks further.
BM-1 over all three runs: 33/150 = 22%; BM-4 hand-set homing: 55/150 = 37%.

**Same protocol on task 3** (r=0.2 perturbation; its BM-3 came from a cycle-2 promotion gated at n=12):
BM-1 16/50 = 32% [21, 46] · BM-2 17/50 = 34% · BM-4 hand-set homing 10/50 = **20%** (homing *hurts* here) ·
BM-3 9/50 = **18%** [10, 31] — a **false-positive promotion**: the n=12 gate accepted a vector that is worse held-out.
Replication with 5 new noise draws: BM-0 30/50 = 60% · BM-1 14/50 = 28% · BM-2 17/50 = 34% · BM-4 14/50 = 28% · BM-3 17/50 = 34%.
Strong-gate cycle 2 on task 3 (24 gate seeds × 4 draws, from the n=12-gate incumbent): fresh-seed validation **kept the
incumbent** — nothing promoted. Third independent run (n=50): BM-1 11/50 = 22% · BM-2 12/50 = 24% · BM-4 12/50 = 24% ·
BM-3 15/50 = 30%. **Pooled task 3, n=150:** BM-1 41/150 = **27%** [21, 35] · BM-2 41/150 = 27% · BM-4 36/150 = 24% ·
BM-3 41/150 = **27%** [21, 35] — the n=12-gate vector is exactly baseline (no gain, no harm; the first run's "harm" was
noise), and the strong gate was right to refuse another. Task 3's perturbation is r=0.2 rad (twice task 0's) and even
the standard arm is at 60%: harder scene, weaker policy, and nothing in this 17-dim file fixes it.
With mastery v3 that is two harmful promotions from the 12-seed gate; the strong gate (24 seeds, K=4) has so far
promoted nothing false (mastery cycle from v2: validation kept the incumbent). The strong-gate re-run on task 3 is queued. The optimizer's vector (homing + time_scale 0.85 +
grasp offset −1.8 cm + cone 34°) beats hand-set homing alone (50% vs 34%): it found more than the switch.

Verdict by the spec's own rule (§13.5), all five tasks at n=10: **collapse reproduced (80% → 14%)**; the untrained shim adds nothing
(BM-2 ≈ BM-1); hand-set homing recovers 6 points pooled (**partial**, CIs overlap); one unattended sleep found homing
on task 0 (gate +50 pp) but the eval layouts did not confirm it (**fail on the main run, partial on the wide run**).
Per-seed view: on task 0 BM-1 succeeds on layouts 40–44 and fails 45–49; homing flips that (fails 40–42, rescues
47–49) — it moves which layouts succeed rather than expanding the set, and 10 layouts per task cannot resolve that.
BM-1 vs BM-4 on all 50 init states of tasks 0 and 3 (n=100/arm; hand-set → no held-out concern): PENDING.
Tasks 1–2 are hard even with EE-space homing: the joint configuration still differs (wrist-camera view), a limit
joint-space homing on a real controller would not have.

Gate tally, all real-env sleep cycles tonight: 13 attempted → 6 promoted (task 0 cycles 1 and 2, task 2 cost-only, task 3 cycle 2 [12-seed gate], mastery v2, mastery v3 [12-seed gate]), 7 refused or validation-kept-incumbent (incl. the strong-gate re-runs on task 3 and mastery, which both kept the incumbent). Every refusal held up on later held-out evidence; the two 12-seed-gate promotions did not (≈ baseline at n=100); every 24-seed-gate decision did.

## 5. Arm C (Gemini planner `gemini-3.7-flash` + inner coach `gemini-3.1-pro-preview`), LIBERO-10 task 3
First run (12 seeds): 17%, 173 mean steps — plans sensible (reach/grasp/lift/place/close, canonical instruction,
no prose leaks, 0 interventions) but per-subtask step budgets (~460 total) truncated the VLA. Fixed (plan exhaustion
no longer ends the episode). Rerun, 20 seeds: **75% [53, 89], 295 steps** — i.e. ≈ the identity-shim arm (80%/294):
the S1 planner neither helps nor hurts this single-skill task once it stops truncating.

Arm D on the perturbed benchmark task 0 (optimized vector + Gemini planner + inner coach), 10 held-out layouts × 2
noise draws: **B vector-only 10/20 = 50% [30, 70], 162 steps; D 12/20 = 60% [39, 78], 151 steps; 0 coach interventions**
(episodes are ~160 steps, the stall detector never fires). The S1 layer is inert here; the recovery is the vector's.
Arm C alone on the same 20 perturbed episodes (planner + coach, **no** vector): 10/20 = 50% [30, 70], 155 steps, 0
interventions — indistinguishable from B and D at n=20. Note: the Gemini coach's JSON replies were truncated on most
queries in this run (`inner coach query failed: Unterminated string`), so "0 interventions" is partly a parsing failure,
not only a quiet detector; see `docs/HANDOFF.md`.

## 6. Protocol P (LIBERO-Spatial task 0, real env, unattended): an honest negative
baseline 8/15 (cost 3.00, 144 steps) → bowl shifted 6 cm → perturbed **1/15** (cost 4.87) → **drift_trigger fired**
(EWMA 4.43 vs baseline 2.37) → auto-sleep #1 (176 rollouts, 44 min on a MIG slice): CEM ran 4 iterations, the
fresh-seed validation chose the **incumbent** (4.74) over every candidate (4.88–5.05) → nothing to gate, no promotion →
drift fired again (5.18) → auto-sleep #2: same outcome → recovery stage **0/15** (cost 5.19).
Reading: the loop did everything it should without a human — detect, sleep, refuse. But a 6 cm object shift is outside
what this S3 vector can express (approach offsets are bounded ±3 cm; the VLA never finds the moved bowl), so there was
nothing legitimate to promote. Re-adaptation: not achieved; reported as such. (On the mock env the same protocol
recovered.)

**Overnight follow-ups (queue of 2026-09-05 late, read 2026-09-06 morning) — all honest negatives:**
- Protocol P with a **2.5 cm** bowl shift (inside the ±3 cm approach-offset bound; `small` auto-sleep, 8-seed gate; arm B, n=15/stage):
  baseline 10/15 = 67% (cost 2.46–2.88) → a *false* drift alarm already in the baseline stage (arm B ran below the arm-A reference)
  → sleep #1 refused (cand 4.38 vs 3.87) → perturbed 6/15 = 40% (cost 4.17) → drift → sleep #2 **passed** the 8-seed gate
  (3.86 → 3.42, 37.5% → 50%, promoted v2) → recovery **4/15 = 27%** (cost 4.20–4.64). Detection and refusal worked; the
  8-seed promotion did not hold up on the recovery episodes (another weak-gate false positive, consistent with §4).
- Second mastery task (LIBERO-10 task 0, "put both the alphabet soup and the cream cheese box in the basket", 20 train seeds):
  arm A **30%** [15, 52], 455 steps; one `small` sleep: validation picked the incumbent's neighbourhood, gate refused
  (cand 3.80 vs 3.38, 37.5% → 25%); arm B on the same seeds 30% / 455 / cost 3.60. No change, nothing promoted.
- Arm D with the coach JSON fix, perturbed task 0, n=20 (10 layouts × 2 draws): B vector-only 15/20 = **75%** [53, 89] /
  135 steps · C planner + coach, no vector 7/20 = **35%** / 172 · D vector + planner + coach 13/20 = 65% [43, 82] / 146 ·
  **0 interventions in every arm** (the stall detector never fires in 150-step episodes). The S1/coach layer remains inert on
  this task; the vector carries the recovery. Arm C's 35% vs 50% in the earlier run is n=20 noise.
- Tasks 2 and 4 at n=30 (10 layouts × 3 draws, MIG renderer): task 2 **0/30 in every arm** (BM-1/2/3/4); task 4 BM-1 1/30,
  BM-2/3/4 2/30. Robot-init at r=0.1 on those scenes is below what any execution-side file can fix.

## 7. Demo assets
- Clips (same seed across arms, `scripts/exp/record_demo.py`, `logs/hopper/videos/`): seed 5047 on LIBERO-Spatial task 0 —
  BM-0 standard **success 69 steps**, BM-1 perturbed **fail**, BM-4 hand-set homing **fail on this seed**, BM-3 consolidated
  vector **success 99 steps**; LIBERO-10 task 3 seed 0 — raw **fail at horizon**, v2 **success 379 steps**.
- Storyboard with embedded clips: https://claude.ai/code/artifact/622fb72e-c0aa-4a88-a23f-5ab8ea2b7957 ·
  analyst dashboard: https://claude.ai/code/artifact/5f1ec73c-fd21-4b6b-845b-a4b1f8432c2e · rebuild: `scripts/pull_logs.sh`,
  `scripts/build_demo_page.py`, then publish `dashboard/demo_artifact.html` / `dashboard/artifact.html`.
- Held-out mastery (LIBERO-10 task 3, layouts 20–39, n=40/arm): A 78% [62, 88] / 302 steps / cost 2.03 → **v2 80% [65, 90] /
  278 steps / cost 1.95** → v3 68% [52, 80] / 335 steps / cost 2.31 (a gate false-positive at n=12; regressed).
  Same-seed rerun (determinism check): A 31/40 again (bit-identical), v2 32/40 / 270 steps / 1.94, v3 23/40 = 58% / 367 / 2.67 —
  arm A is reproducible under common random numbers; the shim arms wobble ±10 pp at n=40, v3's regression is real.
  Strong-gate cycle from v2 (24 gate seeds, K=4 reps): fresh-seed validation **kept the incumbent**; its held-out re-eval
  70% [55, 82] / 312 steps / cost 2.33 (n=40) is the same v2 vector measured a third time — i.e. v2 lands 70–80% held-out. Homing alone, all 50 layouts of tasks 0+3 (n=100/arm, no optimizer): **BM-1 27% [19, 36] → BM-4 33% [25, 43]** — +6 pp, CIs overlap: real but small.

## 8. Second perturbation family — LIBERO-Plus **Camera Viewpoints** (2026-09-06, v3.2)
Native port of LIBERO-Plus's camera perturbation (`envs/libero_plus.py: camera_pose_for_view`, verified against their own
helper functions to 1e-4 in `tests/test_camera.py`). Two kinds of view on libero_spatial task 0: **tilt-only**
(`0_0_100_2_352|354`: camera stays put, optical axis turns 2° about z and 8°/6° down — the scene shifts up by ~0.13-0.18 of
the frame) and **moved** (`11..15_15_100_0_0`: camera position rotated 11-15° about z and 15° up, ~30 cm away; a new
perspective). Frames: `logs/hopper/bench_camera/frames/`.

What changed in the layer for this family: the S3 vector gained four **camera-calibration dims** (`cam_roll_deg`,
`cam_zoom`, `cam_shift_xy` — a similarity warp of the agentview frame, applied by the shim to the policy's copy of the
observation only; identity by default, so every robot-init incumbent loads unchanged). The optimizer sees only cost; it is
never told what the perturbation was. `--act` runs freeze those four dims (17 action dims only) for attribution.

**Probe (10 held-out layouts, 1 noise draw each, MIG 2g.20gb / OSMesa, raw policy):** BM-0 stock camera **9/10** (89 steps) ·
BM-1 `0_0_100_2_352` (tilt −8°) **3/10** [11, 60], 203 steps · BM-1 `0_0_100_2_354` (tilt −6°) **2/10** [6, 51], 203 steps.
A 6–8° pointing change of a fixed camera collapses the policy as hard as the robot-init perturbation did (90% → 20–30%).
The four moved-camera views (11–15°) are PENDING (the first probe OOMed at the third view: a per-env policy copy in the worker
cache, fixed in `runner/pool.py`).

**Sleep cycle + held-out reps (10 eval layouts × 5 noise draws, n=50/arm), strong gate (24 gate layouts × 4 draws):**
| task | view | BM-0 stock | BM-1 camera moved | BM-2 identity shim | BM-3 after one sleep | gate | promoted vector |
|---|---|---|---|---|---|---|---|
| 0 | `0_0_100_2_354` (tilt −6°) | 44/50 = **88%** [76, 94], 91 steps | 7/50 = **14%** [7, 26], 208 steps | 3/50 = 6% [2, 16], 214 steps | **29/50 = 58%** [44, 71], 156 steps — **pass** (4.1×, intervals disjoint) | **passed**: cost 4.04 → 1.89, success 12.5% → 66.7% (+54 pp) on 24 gate layouts; 504 rollouts, 31 min on an A100 | v2: `cam_shift_xy` = (−0.17, −0.18) (frame moved up/left by ~17% — the direction that undoes the tilt), `cam_zoom` 1.09, `cam_roll` 2.0°, `time_scale` 0.64, `gripper_cmd` 0.71; blend off, homing off |
| 0 | same, action dims only (`--act`, calibration frozen at identity) | — | — | — | PENDING | passed: cost 3.77 → 3.15, success 20.8% → 33.3% (+12.5 pp) on the same 24 gate layouts; 504 rollouts, 31 min | v2: `time_scale` 0.60, `velocity_cap` 0.71, `gripper_cmd` 0.74, `blend_alpha` 0.14 (cone 30°), homing on (δ ≤ 2 cm) — the robot-init recipe, worth a quarter of the calibration's gain here |
| 0 | `11_15_100_0_0` (moved) | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| 1 | `0_0_100_4_6` | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| 2 | `0_0_100_6_6` | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| 3 | `0_0_100_8_6` | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| 4 | `0_0_100_10_6` | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

Protocol P with a camera bump (baseline → camera tilted → drift → auto-sleep → recovery, unattended): PENDING.

## Honesty lines
- Base numbers are ours (SmolVLA), on LIBERO-Plus's exact robot-init perturbation; the CVPR table is π₀/OpenVLA.
- The shim reads object pose from simulator state as a stand-in for a detector.
- Frozen VLA in every arm, zero demonstrations; the only thing that changes between BM-1 and BM-3 is one
  versioned, gated parameter file (`skill_instance` events).
- LIBERO has no force sensor: `force_proxy` is an action-magnitude proxy.
- Homing runs in EE space, not joint space (LIBERO's action interface is OSC delta-pose; the VLA's proprio is EE pose).
