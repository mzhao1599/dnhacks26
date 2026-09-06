# Fleet Memory v3.1 — Handoff (operational state as of 2026-09-05 ~21:30 EDT)

Read `AGENTS.md` first (what the project is, invariants, conventions). This file is the *state of the world*:
where things run, where the numbers live, what is pending, and how to pick the work back up cold.
Numbers here are copied from `docs/RESULTS.md`; **PENDING means not yet measured — do not invent a value.**

## 1. Cluster: GMU Hopper

- Login: `ssh hopper` (user `ezhao2`, host `hopper.orc.gmu.edu`; account `ezhao`, QOS `gpu`, partition `gpuq`).
  Off-campus logins need Duo. The user's `~/.ssh/config` has `ControlMaster auto` / `ControlPersist 72h`, so
  **once per session** run `ssh -fN hopper`, approve the Duo push, and every later `ssh`/`rsync` reuses the master.
- Paths under `/scratch/ezhao2/fleet-memory/`:
  `venv` (Python 3.12, `lerobot[smolvla,libero]==0.6.1`, torch 2.11 cu130) · `dnhacks26` (repo clone) ·
  `logs` (all event logs, per job dir) · `hf_cache` (`HF_HOME`; SmolVLA, SmolVLM2 backbone, pi05, LIBERO assets
  prefetched) · `LIBERO-plus` (checkout `4976dc3`) · `libero_plus_assets` (6.4 GB assets.zip extraction) ·
  `venv_plus` (LIBERO-Plus's own `libero` package; `BACKEND=plus` only).
- Sync the repo from the laptop (what we do; the cluster clone is not a git remote):
  `rsync -aq --exclude .venv --exclude .git --exclude logs --exclude __pycache__ ./ hopper:/scratch/ezhao2/fleet-memory/dnhacks26/`
- Secrets: `~/.fm_secrets` on Hopper holds `export GEMINI_API_KEY=...` (chmod 600). `scripts/hopper/env.sh`
  sources it; with no key it exports `FM_LLM=mock`. `env.sh` also activates the venv, sets `HF_HOME`,
  `HF_HUB_OFFLINE=1`, `MUJOCO_GL` (default egl), `PYTHONPATH`, `LIBERO_CONFIG_PATH`, `FM_ROOT/FM_REPO/FM_LOGS`.
- GPU tiers:
  - Full A100: `-p gpuq -q gpu --gres=gpu:A100.40gb:1 -c 8 --mem=32G` → lands on `dgx001`, EGL works,
    **starts in seconds**. Use this by default.
  - MIG: `--gres=gpu:3g.40gb:1` (or `1g.10gb`). No EGL. You must
    `export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa` **before AND after** sourcing `env.sh` (it stages
    libOSMesa from the login node into `$FM_ROOT/lib`; source it once on the login node with `MUJOCO_GL=osmesa`
    before the first MIG job). Rendering is ~3x slower; results differ slightly from EGL (see wide-σ run).
  - `A100.80gb` is **queue-blocked** (jobs sit in Resources/Priority indefinitely). Do not request it.
- Never run MuJoCo or the policy on the login node. Compute nodes have outbound internet.
- The GPU queue is shared; other agents have `scancel`ed our jobs before. Check `squeue -u ezhao2` before submitting.

## 2. Policy

- SmolVLA `HuggingFaceVLA/smolvla_libero` via lerobot 0.6.1 (`fleet_memory/policies/smolvla.py`, mirrors
  `lerobot_eval.py` preprocessing exactly: 256x256 images flipped 180°, state = eef_pos(3)+axis-angle(3)+gripper(2)).
- Always run with `FM_POLICY_KWARGS='{"n_action_steps":10}'` — 10-action chunks, ~50 ms/step amortised.
  1-action (checkpoint default) is ~470 ms/step and ~4.5 min per 520-step episode. Chunk-1 vs chunk-10 on seeds
  0-3 was 4/4 vs 4/4, so chunking is not visibly costing success.
- π₀.₅: adapter exists (`fleet_memory/policies/pi05.py`, `lerobot/pi05_libero`, ungated PaliGemma tokenizer mirror)
  but produced **0/3 on standard LIBERO** and was never validated against `lerobot-eval`. **Unverified — do not
  cite any π₀.₅ number.** Details and next knobs in `scripts/hopper/LIBERO_PLUS.md` §4-5.

## 3. Logs on Hopper (`/scratch/ezhao2/fleet-memory/logs/<name>/events.jsonl`)

| dir | contents |
|---|---|
| `phase0/events.jsonl` | arm A unseeded (LIBERO-10 task 3, 20 train seeds) + the frozen `cost_reference` |
| `probe_v31/*.jsonl` | P1 probes: `identity`, `timescale13`, `blend05`, `bad_offset`, `homing_on` |
| `v31/events.jsonl` | 3-cycle mastery sleep loop (v1 identity → v2 → v3) + arm-B evals |
| `benchmark/events.jsonl` | BM tasks 0-4 + follow-up cycle-2s + pooled `benchmark_result aggregated:true` (once bm_followup finishes) |
| `benchmark/events_wide.jsonl` | wide-σ (σ₀=0.5) task-0 variant, MIG/OSMesa renderer |
| `benchmark/power.jsonl` | BM-1 vs BM-4 on all 50 init states of tasks 0 and 3 (n=100/arm) — not yet in the local mirror |
| `protocol/events.jsonl` | protocol P (LIBERO-Spatial task 0: baseline → 6 cm bowl shift → drift → auto-sleep) |
| `armA/events.jsonl` | seeded arm A + identity-B on the same seeds |
| `armC/events.jsonl`, `events_v2.jsonl` | arm C with Gemini; `events.jsonl` = truncating-planner run (17%), `events_v2.jsonl` = fixed rerun (75%) |
| `mastery/events.jsonl` | held-out A vs v2 vs v3 |
| `slurm/*.out` | job stdout (`plus_*`, `pi05_*`, `phase0_*`, ...) |

- Local mirror: `scripts/pull_logs.sh` rsyncs `hopper:/scratch/ezhao2/fleet-memory/logs/` → `logs/hopper/**`
  (jsonl + png only) and merges phase0 / probe_v31 / v31 / benchmark / protocol / armC into `logs/demo.jsonl`
  (dashboard input: open `dashboard/index.html`, drop the file). armA, mastery, events_wide and power are
  mirrored but **not** merged into demo.jsonl — edit the `parts` list in `pull_logs.sh` if you want them shown.
- NFS append from several nodes can tear ~1 line per 1000 (large snapshot lines). Readers are tolerant of a bad
  line; still, **prefer one log file per job**.

## 4. Job scripts (`scripts/hopper/`)

In the local checkout at this commit: `phase0_v3.sbatch` (arm A + cost reference), `probe_v3.sbatch`,
`smoke_v3.sbatch`, `perturb_verify.sbatch`, `libero_plus_smoke.sbatch`, `pi05_smoke.sbatch`, plus v2-era
`phase0.sbatch`, `phase1_probe.sbatch`, `smoke.sbatch`, and `env.sh`.

The coordinator also ran these; they were written on / for Hopper and **are not in this checkout — look in
`hopper:/scratch/ezhao2/fleet-memory/dnhacks26/scripts/hopper/` and rsync them back before relying on them**:
`probe_p1b.sbatch` (P1 probes), `sleep_v31.sbatch` (3 sleep cycles + arm-B evals), `bm_task.sbatch <task>`
(per-task benchmark: consolidate + arms), `bm_followup.sbatch` (cycle-2 for refused gates + `benchmark --aggregate`),
`bm_power.sbatch`, `mastery_heldout.sbatch`, `consolidate_p2.sbatch` / `probe_p1.sbatch` (v3.0, superseded),
and `scripts/homing_diag.py` (homing diagnostic, CPU + osmesa).

## 5. Results so far (tables copied verbatim from `docs/RESULTS.md`)

**Phase 0 — base rate (LIBERO-10 task 3).** Arm A, 20 train seeds: **55% [34, 74]**, mean **378 steps**.
`steps_ref=377.95`, `jerk_ref` frozen as a `cost_reference` event.

**Phase 1 — does the S3 surface steer this policy? (same 20 seeds, same policy noise; arm B = shim on)**

| S3 vector | success | mean steps |
|---|---|---|
| identity (true pass-through) | 80% [58, 92] | 294 |
| `time_scale = 1.3` | 80% | **266** (−9%) |
| `blend_alpha = 0.5` (approach-shaping on, v3.0 default) | 55% | 395 |
| bad offset (+3 cm xy, −2 cm z, blend on) | 45% | 419 |
| homing to a **wrong** constant pose (bug, fixed) | 5% | 518 |

**Sleep loop on the real env — mastery (LIBERO-10 task 3, arm B = shim + incumbent, 20 train seeds after each cycle)**

| cycle | gate | incumbent after | arm B success | steps | cost |
|---|---|---|---|---|---|
| — (identity v1) | — | v1 | 65% [43, 82] | 341 | 2.46 |
| 1 | refused (cand 3.53 vs 2.09 on gate seeds, −37 pp) | v1 | — | — | — |
| 2 | passed | **v2** | **80% [58, 92]** | **275** | **1.95** |
| 3 | passed | v3 | 60% [39, 78] | 354 | 2.57 |

Mastery curve v1 2.46 → v2 1.95 → v3 2.57: not monotone; report as measured. Gate pass rate 2/3.
Seeded arm A baseline on the same seeds: PENDING (Phase 0's 55%/378 predates policy seeding).

**LIBERO-Plus robot-initial-state benchmark (LIBERO-Spatial, SmolVLA, our own base numbers)**

| task | BM-0 standard | BM-1 perturbed | BM-2 + untrained shim | BM-4 + hand-set homing | BM-3 after ONE unattended sleep |
|---|---|---|---|---|---|
| 0 | 9/10 | 5/10 | 4/10 | 4/10 | 5/10 (gate: 17%→67% on gate seeds; eval: no change) |
| 1 | 6/10 | 0/10 | 0/10 | 2/10 | gate refused → cycle 2: PENDING |
| 2 | 10/10 | 0/10 | 0/10 | 0/10 | 0/10 (gate passed on cost only, 0%→0%) |
| 3 | 9/10 | 2/10 | 2/10 | 3/10 | gate refused → cycle 2: PENDING |
| 4 | 6/10 | 0/10 | 0/10 | 1/10 | gate refused → cycle 2: PENDING |
| **pooled** | **40/50 = 80%** | **7/50 = 14% [7, 26]** | 6/50 = 12% | **10/50 = 20% [11, 33]** | 5/20 on tasks 0+2 (= BM-1 there) |

Wide-search variant (task 0, σ₀=0.5, separate skill instance, MIG/OSMesa renderer): BM-1 3/10 → BM-4 5/10 →
**BM-3 7/10** (gate passed 2.94 → 2.72). Verdict by the spec's own rule (§13.5): **collapse reproduced (80% → 14%)**;
untrained shim adds nothing; hand-set homing recovers 6 points pooled (**partial**, CIs overlap); one unattended
sleep found homing on task 0 (gate +50 pp) but eval layouts did not confirm it (**fail on the main run, partial on
the wide run**). BM-1 vs BM-4 on all 50 init states of tasks 0 and 3 (n=100/arm): PENDING.

**Arm C (Gemini planner `gemini-3.7-flash` + inner coach `gemini-3.1-pro-preview`), LIBERO-10 task 3.**
First run (12 seeds): 17%, 173 mean steps — per-subtask step budgets truncated the VLA. Fixed. Rerun, 20 seeds:
**75% [53, 89], 295 steps** — ≈ the identity-shim arm (80%/294): the S1 planner neither helps nor hurts here.

**Protocol P (LIBERO-Spatial task 0):** baseline 8/15 (cost 3.00, 144 steps) → bowl shifted 6 cm → 1/5 →
**drift_trigger fired unattended** (EWMA cost 4.43 vs baseline 2.37) → auto-sleep (small) running; recovery stage PENDING.

Honesty lines (keep them in any write-up): base numbers are ours (SmolVLA), not the CVPR π₀/OpenVLA table; the shim
reads object pose from simulator state as a stand-in for a detector; frozen VLA in every arm, zero demonstrations;
`force_proxy` is an action-magnitude proxy (LIBERO has no force sensor); homing is EE-space, not joint-space.

## 6. Known issues / open work

- Gate resolution is ±20 pp at n=12-16 gate seeds and ~50% success; need ≥16 gate seeds and K≥4 rollouts per seed to see small effects.
- Policy noise is seeded per episode, but GPU nondeterminism still diverges within an episode (A vs identity-B: 80% vs 65% at n=20) → use n≥40 for any A-vs-B claim.
- EE-space homing cannot restore the joint configuration (wrist-camera view differs); LIBERO-Spatial tasks 1-2 stay ≤6% under robot-init perturbation.
- Benchmark is n=10 eval layouts per task; on task 0 homing moves *which* layouts succeed rather than expanding the set — 10 layouts cannot resolve that.
- Protocol-P recovery on the mock env never reached the 10% recovery band; real-env recovery stage is PENDING.
- Arm E/F, the S2 (instruction paraphrase) probe, and cross-suite transfer are deferred.
- Dashboard benchmark panel shows only the *latest* `benchmark_result` event (the aggregated one once `bm_followup` finishes).
- `benchmark/power.jsonl` and the bm_followup cycle-2 results are on Hopper (or still running) and not yet in `docs/RESULTS.md`.
- Sensor-noise dimension of LIBERO-Plus cannot run on GPU nodes (no ImageMagick); Objects-Layout tasks need the asset extraction to finish.

## 7. Resume checklist

1. `ssh -fN hopper` → approve the Duo push → `ssh hopper 'echo ok'` confirms the ControlMaster is up.
2. `ssh hopper 'squeue -u ezhao2'` — note which of bm_followup / bm_power / protocol / mastery jobs are still running or finished; `ls /scratch/ezhao2/fleet-memory/logs/slurm/` for their stdout.
3. `rsync -aq hopper:/scratch/ezhao2/fleet-memory/dnhacks26/scripts/ scripts_hopper_snapshot/` (or diff) to recover any sbatch/py files that exist only on the cluster, then copy the ones you need into `scripts/hopper/` and commit them.
4. `bash scripts/pull_logs.sh` → refreshes `logs/hopper/**` and `logs/demo.jsonl`. Check `logs/hopper/benchmark/power.jsonl` and the tail of `benchmark/events.jsonl` for `benchmark_result` with `aggregated:true`.
5. `source .venv/bin/activate && FM_LLM=mock pytest -q` → 120 passed (sanity that the local tree is intact).
6. `python -m fleet_memory.analysis.metrics --log logs/hopper/benchmark/events.jsonl` (and `--log logs/hopper/protocol/events.jsonl`, `logs/hopper/mastery/events.jsonl`, `logs/hopper/armA/events.jsonl`, `logs/hopper/benchmark/power.jsonl`) — read the tables.
7. Fill each PENDING in `docs/RESULTS.md` from those tables only: seeded arm A baseline (armA), BM-3 cycle-2 for tasks 1/3/4 (benchmark), BM-1 vs BM-4 power (power.jsonl), protocol-P recovery (protocol). If a job did not finish, leave PENDING and say so.
8. Open `dashboard/index.html`, drop `logs/demo.jsonl`, confirm the mastery curve, house model and the aggregated benchmark panel render.
9. If any run must be redone, rsync the repo to Hopper first (command in §1), submit with `A100.40gb`, `FM_POLICY_KWARGS='{"n_action_steps":10}'`, one log file per job, and never on the login node.
10. `git add docs/RESULTS.md scripts/hopper && git commit` — then push only when the user asks.
