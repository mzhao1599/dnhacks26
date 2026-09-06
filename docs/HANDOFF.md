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

## 4. Job scripts and the autonomous queue (`scripts/hopper/`)

- `env.sh` (source it; sets venv, HF cache, MUJOCO_GL, LIBERO config, secrets), one sbatch per experiment
  (`phase0_v3`, `probe_p1b`, `sleep_v31`, `bm_task`, `bm_followup`, `bm_power`, `mastery_heldout`, `libero_plus_smoke`,
  `pi05_smoke`, `unzip_plus_assets`, …), `LIBERO_PLUS.md` (perturbation mechanics + plus backend).
- **Queue runner** (keeps GPUs busy unattended): `queue_runner.sh` polls `queue.txt` every 60 s on the login node
  (`setsid nohup bash scripts/hopper/queue_runner.sh >> $FM_ROOT/logs/queue_runner.log &`; restart with
  `pkill -f "^bash scripts/hopper/queue_runner.sh"` — the anchored pattern matters, a bare `pkill -f queue_runner`
  kills your own ssh shell). One task per line `name | tier [after=<job-or-task-name>] | command`; tier `a100`
  (cap `MAX_A100`, default 6) or `mig` (cap 2, osmesa prefix); `after=` becomes a slurm `afterany` dependency if
  that job is still queued/running. A name is submitted once (`$FM_ROOT/logs/queue.done`; delete the line to resubmit).
  Job names `q-a100-<name>` / `q-mig-<name>`, stdout `logs/slurm/q_<name>_<jid>.out`, ends with `QTASK_DONE <name>`.
- Experiment drivers in `scripts/exp/`: `bm_reps.py <task> <reps> [rep_offset] [--bm0]` (held-out layouts × noise
  reps, prints `REPS <arm>: k/n`), `mastery_heldout.py`, `arm_d.py <task> [B,C,D]`, `bm_power.py`, `record_demo.py`.

## 5. Results

All measured numbers, with n and intervals, are in `docs/RESULTS.md` (single source; do not copy tables here).
Headline as of 2026-09-05 22:30: LIBERO-Spatial task 0 under LIBERO-Plus robot-init perturbation, 10 held-out layouts
× 10 noise draws (n=100/arm): BM-1 22% → BM-4 hand-set homing 34% → **BM-3 one unattended sleep 54%** (2.45×, CIs
disjoint; replicated across two independent 50-episode runs). Task 3: the n=12 gate produced a false positive
(BM-3 18% vs BM-1 32%); strong-gate re-run queued (`cycle2-task3`). Mastery (LIBERO-10 task 3, held-out n=40): A 78%
→ v2 80% (70–80% over three measurements) → v3 58–68% (false positive; strong gate from v2 kept the incumbent).

## 6. Known issues / open work

- Gate resolution: ±20 pp at 12–16 gate seeds; two false-positive promotions came from n=12 gates (mastery v3,
  benchmark task 3). The strong gate (24 seeds, K=4) has promoted nothing false so far. Use it.
- GPU nondeterminism: arm A is bit-reproducible under common random numbers, shim arms wobble ±10 pp at n=40 → n≥40
  for any A-vs-B claim, n=100 for a headline.
- Gemini coach replies are often truncated JSON (`inner/outer coach query failed: Unterminated string`) → 0
  interventions in arms C/D partly for that reason. Fix: raise the coach `max_output_tokens` in `agents/llm.py`
  (`_gemini`) and/or ask for compact JSON; then re-run `scripts/exp/arm_d.py 0 C,D`.
- EE-space homing cannot restore the joint configuration (wrist-camera view differs); LIBERO-Spatial tasks 1–2 stay
  ≤6% under robot-init perturbation. Homing hurts on task 3 (r=0.2).
- Protocol P on the real env: drift detection and refusal worked; recovery 0/15 (6 cm object shift is outside S3's ±3 cm).
- The benchmark is the robot-init family only (spec §13, `seeds.json`). Other LIBERO-Plus families are wired
  (`--dimension camera|light|background|layout`, LIBERO-plus package backend `venv_plus`) but out of plan and unmeasured.
- Arm E/F, S2 probe, cross-suite transfer: deferred.

## 7. Resume checklist

1. `ssh -fN hopper` (Duo) → `ssh hopper 'echo ok'`.
2. `ssh hopper 'squeue -u ezhao2; tail /scratch/ezhao2/fleet-memory/logs/queue_runner.log; pgrep -fa "^bash scripts/hopper/queue_runner.sh"'`
   — if the runner is dead, restart it (§4). Append new work to `scripts/hopper/queue.txt`, rsync, done.
3. Finished jobs: `grep -h "REPS\|HELDOUT\|ARMD\|consolidation\|benchmark_result" /scratch/ezhao2/fleet-memory/logs/slurm/q_*.out`
   or `python -m fleet_memory.runner.benchmark --aggregate --log <log>`; fold into `docs/RESULTS.md` with n and CI.
4. `bash scripts/pull_logs.sh` → `python scripts/build_demo_page.py` → republish `dashboard/demo_artifact.html`
   (storyboard) and `dashboard/artifact.html` (dashboard) to their existing artifact URLs (`docs/RESULTS.md` §7).
5. `source .venv/bin/activate && FM_LLM=mock pytest -q` before any code change lands on Hopper; rsync command in §1.
6. Commit `docs/`, `scripts/hopper/`, `scripts/exp/`; push only when the user asks.
