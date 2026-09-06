# Fleet Memory v3

Weight-frozen improvement for a vision-language-action policy (SmolVLA on LIBERO), keyed to a specific
environment and versioned. Two writers, two kinds of knowledge:
- **Coach (LLM, sparse)** writes discrete knowledge on **S1** (plans, bindings, preferences) as lessons; may
  *propose* S3 edits only as candidates that must pass the optimizer's gate.
- **Sleep-loop optimizer (CEM, dense)** writes the continuous **S3 vector** (`execution/params.py`, 17 dims)
  per skill instance, gated by a perturbation test on disjoint seeds (`runner/consolidate.py`).
Headline objective: **mastery on a fixed environment** (cost/steps ↓ across versions) plus **recovery after
perturbation** (drift detector → consolidation → cost recovers, unattended). Transfer/S2 probe/LoRA/arm E are
post-hackathon.

## Hard invariants (never violate)
1. **Success comes from the env predicate only.** `Env.success_flag()` is the sole source of truth. No LLM ever decides success.
2. **Retrieval is frozen pre-episode.** `Episode.retrieved_lesson_ids` is written before `reset()` and never edited.
3. **The coach never writes prose into the policy.** Lessons carry a `surface` and a machine-applicable `edit`; `rationale_text` is dashboard-only.
4. **Inner-loop coach never sees the success flag.** Outer-loop coach does.
5. **Held-out task suites never generate lessons.**
6. **Append-only JSONL event log.** State changes are new events; nothing is mutated in place.
7. **Lessons are hypotheses.** Coach-written lessons: `candidate → validated` only via the A/B gate in `memory/lifecycle.py` (n≥30 matched seeds, one-sided p<0.05, lift ≥2pp). Optimizer-written S3 vectors: promoted only via the perturbation gate (incumbent vs candidate on N=16 disjoint seeds: mean cost lower AND success ≥ incumbent − 0.02). Exactly one `incumbent` skill_instance per id.
8. **The safety envelope (`execution/envelope.py`) is applied after the shim, unconditionally, and is never optimised.**
9. **The coach never guesses a grasp offset.** Continuous S3 values come from the optimizer.

## Layout
```
fleet_memory/
  envs/        base.py (Env protocol, Obs), libero_env.py, mock_env.py, perturb.py (pose shift / distractor)
  policies/    base.py (Policy protocol, ActionChunk), smolvla.py, mock.py, scripted.py (IK fallback)
  execution/   shim.py (S3 enforcement + phase-gated blending + time_scale), constraints.py, params.py (17-dim S3 vector),
               envelope.py (immutable clamp), detectors.py
  agents/      planner.py (Haiku), coach_inner.py, coach_outer.py, llm.py (anthropic | gemini | mock)
  memory/      schema.py (ALL record types), store.py, retrieval.py, lifecycle.py (lesson A/B gate),
               drift.py (EWMA → drift_trigger), house_model.py (per-environment view: objects, incumbents, lessons)
  runner/      worker.py, pool.py, conditions.py (arms A/B/C/D/P), consolidate.py (sleep loop: CEM + gate + promotion)
  analysis/    cost.py (metrics + cost), metrics.py, plots.py
dashboard/index.html      single static file over logs/events.jsonl (mastery curve, house model, versions, consolidations)
scripts/hopper/           sbatch + env setup for GMU Hopper
tests/                    pytest, all runnable locally with mock env/policy (no GPU)
logs/events.jsonl         the event log
```

## v3 arms (runner/conditions.py)
A raw VLA + envelope only · B = A + sleep-optimized S3 (no planner/coach) · C = A + planner + S1 memory + inner coach ·
D = B + C (the system) · P = perturbation protocol on A vs D (baseline → perturb → recovery).
Cost (analysis/cost.py): `1.0*steps/steps_ref + 0.5*jerk + 0.5*force_proxy + 3.0*(1-success)`; refs are Arm A means, frozen as `cost_reference` events.

## Conventions
- Plain Python 3.12, numpy. No framework, no DB, no orchestration layer.
- `fleet_memory/memory/schema.py` is the contract. Change it only deliberately; everything imports from it.
- Everything must run end-to-end with `--env mock --policy mock --llm mock` on a laptop. Real runs happen on Hopper.
- LLM calls go through `agents/llm.py`. Backend is picked by key: `ANTHROPIC_API_KEY` → Claude (planner `claude-haiku-4-5-20251001`, coach `claude-sonnet-5`); `GEMINI_API_KEY` → Gemini (planner `gemini-3.7-flash`, coach `gemini-3.1-pro-preview`; `gemini-2.5-pro` is 404 for new keys, override via `FM_PLANNER_MODEL`/`FM_COACH_MODEL`); neither → deterministic offline mock. `FM_LLM=mock|anthropic|gemini` forces one. On Hopper, keys live in `~/.fm_secrets` (sourced by `scripts/hopper/env.sh`).
- Actions are LIBERO/robosuite OSC_POSE: `float32[7]` = `[dx, dy, dz, droll, dpitch, dyaw, gripper]`, each in [-1, 1]; gripper>0 = close.
- Log every event via `memory/store.py:EventStore.append(record)`; records are dataclasses from `schema.py` serialised with `to_dict()`.

## Hopper (GMU ORC)
- Login: `ssh hopper`. Project root on cluster: `/scratch/ezhao2/fleet-memory` (repo clone at `/scratch/ezhao2/fleet-memory/dnhacks26`).
- Env: `source /scratch/ezhao2/fleet-memory/venv/bin/activate`; `export HF_HOME=/scratch/ezhao2/hf_cache MUJOCO_GL=egl`.
- GPU jobs: `sbatch -p gpuq -q gpu --gres=gpu:A100.80gb:1 ...` (full A100 → EGL works; MIG slices do NOT support EGL, use `MUJOCO_GL=osmesa` there).
- Never run episodes on the login node.
