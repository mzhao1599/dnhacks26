# AGENTS.md — Fleet Memory v3.1 (vendor-neutral agent guide)

This is the vendor-neutral twin of `CLAUDE.md`. Any agent (Claude, Gemini, GPT, human) resuming this repo
should read this file first, then `docs/HANDOFF.md` (operational state) and `docs/DECISIONS.md` (why things
are the way they are). `CLAUDE.md` remains the canonical source for the invariants; this file copies them.

## What the project is

Fleet Memory is weight-frozen improvement of a vision-language-action policy (SmolVLA `HuggingFaceVLA/smolvla_libero`
on LIBERO, via lerobot 0.6.1), keyed to a specific environment and versioned. The VLA's weights never change and
no demonstrations are collected; everything that improves is a small, versioned, gated parameter file
(`skill_instance` events) plus discrete lessons. Two writers exist. The **coach** (an LLM, sparse) writes discrete
knowledge on **S1** (plans, bindings, preferences) as lessons and may *propose* S3 edits only as candidates. The
**sleep-loop optimizer** (CEM, dense) writes the continuous **S3 vector** (`fleet_memory/execution/params.py`,
17 dims: approach shaping, time scale, velocity cap, gripper, blend alpha, homing) per skill instance, gated by a
perturbation test on seeds the optimizer never saw (`fleet_memory/runner/consolidate.py`).

The headline objective is **mastery on a fixed environment** (cost and steps go down across versions) plus
**recovery after perturbation** (EWMA drift detector fires, consolidation runs unattended, cost recovers). The
v3.1 benchmark applies this to the LIBERO-Plus "robot initial state" collapse (arms BM-0..BM-4 in
`fleet_memory/runner/benchmark.py`). Everything is an append-only JSONL event log; the environment's programmatic
success predicate is the only source of truth. Transfer, the S2 probe, LoRA, and arm E/F are deferred.

## Hard invariants (never violate) — copied verbatim from CLAUDE.md

1. **Success comes from the env predicate only.** `Env.success_flag()` is the sole source of truth. No LLM ever decides success.
2. **Retrieval is frozen pre-episode.** `Episode.retrieved_lesson_ids` is written before `reset()` and never edited.
3. **The coach never writes prose into the policy.** Lessons carry a `surface` and a machine-applicable `edit`; `rationale_text` is dashboard-only.
4. **Inner-loop coach never sees the success flag.** Outer-loop coach does.
5. **Held-out task suites never generate lessons.**
6. **Append-only JSONL event log.** State changes are new events; nothing is mutated in place.
7. **Lessons are hypotheses.** Coach-written lessons: `candidate → validated` only via the A/B gate in `memory/lifecycle.py` (n≥30 matched seeds, one-sided p<0.05, lift ≥2pp). Optimizer-written S3 vectors: promoted only via the perturbation gate (incumbent vs candidate on N=16 disjoint seeds: mean cost lower AND success ≥ incumbent − 0.02). Exactly one `incumbent` skill_instance per id.
8. **The safety envelope (`execution/envelope.py`) is applied after the shim, unconditionally, and is never optimised.**
9. **The coach never guesses a grasp offset.** Continuous S3 values come from the optimizer.

## Repo layout

```
fleet_memory/
  envs/        base.py (Env protocol, Obs), libero_env.py (lerobot-0.6.1-exact adapter), libero_plus.py
               (LIBERO-Plus robot-init perturbation), mock_env.py (kinematic 3-D mock), perturb.py (pose shift / distractor)
  policies/    base.py (Policy protocol, ActionChunk), smolvla.py, pi05.py (unverified), mock.py, scripted.py (IK-free fallback)
  execution/   shim.py (S3 enforcement + phase-gated blending + time_scale), constraints.py, params.py (17-dim S3 vector),
               envelope.py (immutable clamp), homing.py (EE-space homing before the VLA acts), detectors.py, probes.py
  agents/      planner.py, coach_inner.py, coach_outer.py, llm.py (anthropic | gemini | mock; one call shape: complete_json)
  memory/      schema.py (ALL record types — the contract), store.py (append-only JSONL), retrieval.py,
               lifecycle.py (lesson A/B gate), drift.py (EWMA → drift_trigger), house_model.py (per-env view; ONE incumbent per id)
  runner/      worker.py (one episode; the order in run_episode is the contract), loop.py (subtask loop), pool.py
               (persistent multiprocessing pool + batch CLI), conditions.py (arms A/B/C/D/P), consolidate.py (sleep loop:
               CEM + gate + promotion), benchmark.py (BM-0..4), seeds.py + seeds.json (disjoint opt/gate/eval seed sets)
  analysis/    cost.py (metrics + scalar cost), metrics.py (CLI results table), mastery.py, plots.py
dashboard/index.html      single static file; open it and drop a logs/*.jsonl on it
scripts/hopper/           sbatch + env.sh for GMU Hopper (README.md, LIBERO_PLUS.md)
scripts/pull_logs.sh      rsync Hopper logs → logs/hopper/** and merge → logs/demo.jsonl
tests/                    pytest, all runnable locally with mock env/policy/LLM (no GPU)
docs/                     RESULTS.md (measured numbers), HANDOFF.md (operational state), DECISIONS.md (decision log)
logs/                     event logs (gitignored; logs/hopper/** is the local mirror of the cluster)
```

## Conventions

- Plain Python 3.12, numpy. No framework, no DB, no orchestration layer. Install locally with
  `uv venv .venv --python 3.12 && source .venv/bin/activate && uv pip install -e . pytest google-genai`.
- **Mock-first.** Everything must run end-to-end with `--env mock --policy mock --llm mock` on a laptop.
  `FM_LLM=mock pytest -q` must pass: **120 tests**. Real sim runs happen only on Hopper (see `docs/HANDOFF.md`).
- **LLM backend selection** (`fleet_memory/agents/llm.py`). Picked by which key is present:
  `ANTHROPIC_API_KEY` → Claude (planner `claude-haiku-4-5-20251001`, coach `claude-sonnet-5`);
  `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) → Gemini (defaults: planner `gemini-3.7-flash`, coach `gemini-3.1-pro-preview`;
  `gemini-2.5-pro` is 404 for new keys); neither → deterministic offline mock.
  `FM_LLM=mock|anthropic|gemini` forces a backend; `FM_PLANNER_MODEL` / `FM_COACH_MODEL` override model ids.
  On Hopper the keys live in `~/.fm_secrets`, sourced by `scripts/hopper/env.sh`.
- Actions are LIBERO/robosuite OSC_POSE: `float32[7]` = `[dx, dy, dz, droll, dpitch, dyaw, gripper]`, each in
  [-1, 1]; gripper > 0 = close. 1.0 of translation == 0.05 m on LIBERO.
- Log every event via `memory/store.py:EventStore.append(record)`; records are dataclasses from `schema.py`
  serialised with `to_dict()`. Readers replay the log to materialise state. Readers must tolerate a torn line.
- Cost (`analysis/cost.py`): `1.0*steps/steps_ref + 0.5*jerk + 0.5*force_proxy + 3.0*(1-success)`; refs are
  Arm A means frozen once as a `cost_reference` event.
- Arms (`runner/conditions.py`): A raw VLA + envelope only · B = A + sleep-optimized S3 (no planner/coach) ·
  C = A + planner + S1 memory + inner coach · D = B + C (the system) · P = perturbation protocol on A vs D.
  Benchmark arms (`runner/benchmark.py`): BM-0 standard · BM-1 perturbed · BM-2 + untrained shim ·
  BM-3 + consolidated incumbent · BM-4 + hand-set homing.
- Seed sets (`runner/seeds.json`): LIBERO init state = seed % 50; opt uses init states 0-29 (seed base 3000),
  gate 30-39 (4000), eval 40-49 (5000). Eval states are never seen by the optimizer or the gate.

## Shared-API rules

- `fleet_memory/memory/schema.py` **is the contract.** Every record type, the control-surface vocabulary and the
  closed set of edit ops live there. Change it only deliberately; everything imports from it; update tests with it.
- `fleet_memory/execution/params.py` defines the S3 surface as ONE bounded continuous vector of **17 dims**
  (`PARAM_SPEC`, `DIM`, `NAMES`). All optimisation happens over this vector and nothing else. Bounds are hard; the
  shim clips. The optimizer writes it; the coach may only propose candidates that go through the same gate.
  Identity (all defaults, `blend_alpha = 0`) must remain a true pass-through of the VLA.
- `fleet_memory/execution/envelope.py` is applied after the shim on every action and is **never optimised**,
  never a lesson, never logged as a surface.
- `Env` (`envs/base.py`) and `Policy` (`policies/base.py`) protocols are the only way anything talks to the
  simulator or the VLA. New adapters implement them; nothing bypasses them.
- `runner/worker.py:run_episode` order is load-bearing: retrieval → reset → (homing) → subtask loop → success
  from `env.success_flag()` → outer coach → append. Do not reorder.

## How to verify a change

Run all of these locally (laptop, no GPU, offline) before touching the cluster:

```
FM_LLM=mock pytest -q                                                # must report 120 passed
FM_LLM=mock python -m fleet_memory.runner.pool --env mock --policy mock --suite mock --task pick_bowl_to_plate \
    --arm A --n 40 --workers 4 --seed-set train --log logs/v3.jsonl --make-reference
# then the same with --arm B, --arm C, --arm D (mock arms A-D)
FM_LLM=mock python -m fleet_memory.runner.consolidate --env mock --policy mock --suite mock \
    --task pick_bowl_to_plate --log logs/v3.jsonl --small           # one sleep cycle (CEM + gate + promotion)
FM_LLM=mock python -m fleet_memory.runner.pool --env mock --policy mock --suite mock --task pick_bowl_to_plate \
    --arm P --protocol P --n-stage 20 --perturb '{"shift_xy":[0.06,0]}' --auto-sleep --log logs/v3.jsonl
python -m fleet_memory.analysis.metrics --log logs/v3.jsonl          # results table; open dashboard/index.html
```

Checklist for a change: tests still 120/120; mock arms A-D run without exceptions and produce `outcome` events;
`consolidate --small` produces a `gate_result` and, on pass, exactly one new `incumbent`; protocol P produces a
`drift_trigger` and an auto-sleep; no invariant above is weakened; `schema.py` diffs are intentional and
documented in `docs/DECISIONS.md`. Never cite a cluster number that is not in `docs/RESULTS.md` or a log.
