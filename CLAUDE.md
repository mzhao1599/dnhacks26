# Fleet Memory v2

Coaching-driven, weight-frozen improvement for a vision-language-action policy (SmolVLA on LIBERO).
Every lesson in memory is a structured edit to exactly one **control surface** (S1 plan, S2 instruction,
S3 execution constraints). No weight updates in the critical path.

## Hard invariants (never violate)
1. **Success comes from the env predicate only.** `Env.success_flag()` is the sole source of truth. No LLM ever decides success.
2. **Retrieval is frozen pre-episode.** `Episode.retrieved_lesson_ids` is written before `reset()` and never edited.
3. **The coach never writes prose into the policy.** Lessons carry a `surface` and a machine-applicable `edit`; `rationale_text` is dashboard-only.
4. **Inner-loop coach never sees the success flag.** Outer-loop coach does.
5. **Held-out task suites never generate lessons.**
6. **Append-only JSONL event log.** State changes are new events; nothing is mutated in place.
7. **Lessons are hypotheses.** `candidate → validated` only via the A/B gate in `memory/lifecycle.py` (n≥30 matched seeds, one-sided p<0.05, lift ≥2pp).

## Layout
```
fleet_memory/
  envs/        base.py (Env protocol, Obs), libero_env.py, mock_env.py, perturb.py
  policies/    base.py (Policy protocol, ActionChunk), smolvla.py, mock.py
  execution/   shim.py (S3 enforcement), constraints.py, detectors.py
  agents/      planner.py (Haiku), coach_inner.py, coach_outer.py, llm.py (client + offline mock)
  memory/      schema.py (ALL record types + edit-op vocabulary), store.py, retrieval.py, lifecycle.py
  runner/      worker.py, pool.py, conditions.py (arms A–F, seed sets)
  analysis/    metrics.py, plots.py
dashboard/index.html      single static file over logs/events.jsonl
scripts/hopper/           sbatch + env setup for GMU Hopper
tests/                    pytest, all runnable locally with mock env/policy (no GPU)
logs/events.jsonl         the event log
```

## Conventions
- Plain Python 3.12, numpy. No framework, no DB, no orchestration layer.
- `fleet_memory/memory/schema.py` is the contract. Change it only deliberately; everything imports from it.
- Everything must run end-to-end with `--env mock --policy mock --llm mock` on a laptop. Real runs happen on Hopper.
- LLM calls go through `agents/llm.py` (`anthropic` SDK; planner = `claude-haiku-4-5-20251001`, coach = `claude-sonnet-5`). `FM_LLM=mock` selects the deterministic offline mock.
- Actions are LIBERO/robosuite OSC_POSE: `float32[7]` = `[dx, dy, dz, droll, dpitch, dyaw, gripper]`, each in [-1, 1]; gripper>0 = close.
- Log every event via `memory/store.py:EventStore.append(record)`; records are dataclasses from `schema.py` serialised with `to_dict()`.

## Hopper (GMU ORC)
- Login: `ssh hopper`. Project root on cluster: `/scratch/ezhao2/fleet-memory` (repo clone at `/scratch/ezhao2/fleet-memory/dnhacks26`).
- Env: `source /scratch/ezhao2/fleet-memory/venv/bin/activate`; `export HF_HOME=/scratch/ezhao2/hf_cache MUJOCO_GL=egl`.
- GPU jobs: `sbatch -p gpuq -q gpu --gres=gpu:A100.80gb:1 ...` (full A100 → EGL works; MIG slices do NOT support EGL, use `MUJOCO_GL=osmesa` there).
- Never run episodes on the login node.
