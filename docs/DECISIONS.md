# Fleet Memory v3.1 — Decision log

One line per decision, with the reason. Append; do not rewrite history. Dated by the session in which the
decision was made or confirmed. Anything that changes `fleet_memory/memory/schema.py` or
`fleet_memory/execution/params.py` must get a line here.

## 2026-09-05

- **Python 3.12 venv via `uv`** (laptop and Hopper `/scratch/ezhao2/fleet-memory/venv`): the cluster's system Python is 3.6; lerobot 0.6.1 needs ≥3.10 and `uv` builds the env in seconds without touching modules.
- **lerobot 0.6.1 `[smolvla,libero]` instead of raw LIBERO**: the SmolVLA-LIBERO checkpoint was evaluated with lerobot's exact env wrapper and preprocessing (image flip, 8-D state, task string); mirroring it (`envs/libero_env.py`) avoids silently evaluating a different distribution.
- **`A100.40gb` over `A100.80gb`**: the 80 GB pool is queue-blocked (jobs sit in Resources/Priority indefinitely); the 40 GB node `dgx001` starts in seconds, supports EGL, and SmolVLA needs ~1.2 GB.
- **10-action chunks (`n_action_steps=10`)**: 1-action re-query is ~470 ms/step; chunk-10 is ~50 ms/step amortised with 4/4 vs 4/4 success on seeds 0-3, so all real runs use chunk-10 and Phase 0 references were frozen on it.
- **Gemini instead of Claude for the real runs**: the user's available key is `GEMINI_API_KEY`; `gemini-2.5-pro` returns 404 for new keys, so defaults are planner `gemini-3.7-flash` / coach `gemini-3.1-pro-preview` (override via `FM_PLANNER_MODEL` / `FM_COACH_MODEL`); Claude ids stay as the `ANTHROPIC_API_KEY` path.
- **v3.1 S3 vector is 17 dims** (`execution/params.py`): 9 v3.0 dims + `blend_alpha` (1) + `homing_enable` (1) + `homing_pos_delta` (3) + `homing_rot_delta` (3); the optimizer can now switch shaping and homing on or off rather than having them baked in.
- **`blend_alpha` added as an optimisable dim with identity = 0**: the v3.0 default approach-shaping (waypoint + cone + α=0.5) cost this VLA 15-25 pp (55% vs 80% identity on the same seeds), so "shim on" must be a true pass-through unless the optimizer proves shaping helps.
- **Homing in EE space, not joint space** (`execution/homing.py`): LIBERO's action interface is OSC delta-pose and the VLA's proprio is EE pose, so a joint-space controller has no channel; accepted limitation — EE homing cannot restore the joint configuration (LIBERO-Spatial tasks 1-2 stay ≤6%).
- **Canonical homing pose from `env.canonical_ee_pose()`**, not a constant: an early adapter reported z≈0.70 while LIBERO-10 actually starts at z≈1.17; the constant sent the arm into the table (5% success probe). The env owns its reset pose.
- **Policy RNG seeded per episode**: arm A vs identity-B disagreed per seed in both directions, i.e. policy sampling noise dominated; seeding per (seed, episode) makes arms on the same seed share the policy's noise so paired comparisons mean something.
- **Per-task LIBERO-Plus config selection** (`envs/libero_plus.py:list_configs`): LIBERO-Plus has no per-dimension config files; configs are rebuilt from `task_classification.json` + the seed-42 qpos table so each base task picks its own `initstate_N` (task 3 drew r=0.2, the others r=0.1).
- **Re-apply the perturbed qpos after `set_init_state`**: LIBERO-Plus's own loop overwrites the perturbed joints with the base init state (only the OSC nullspace target survives); writing qpos afterwards is the perturbation the paper describes and is deterministic per N.
- **Disjoint init-state ranges opt 0-29 / gate 30-39 / eval 40-49** (`runner/seeds.json`): LiberoEnv maps seed % 50 → init state, so seed-level disjointness was not enough; eval layouts are never seen by the optimizer or the gate, and hand-rolled jitter is applied only to opt/gate rollouts.
- **Persistent multiprocessing pool + incremental JSONL reader** (`runner/pool.py`, `memory/store.py`): previous versions re-spawned workers (reloading SmolVLA each time) and re-parsed the whole log per episode; both were the dominant wall-clock cost once logs passed a few thousand events.
- **Plan exhaustion no longer ends the episode** (`runner/loop.py`): arm C's per-subtask step budgets (~460 total) truncated the VLA and produced 17% vs 80%; the plan now only structures the episode, the env horizon ends it.
- **Envelope respects `MUJOCO_GL`**: nothing in `execution/envelope.py` depends on the renderer, but its bounds were validated under both EGL and OSMesa because MIG (OSMesa) runs produce slightly different trajectories; the clamp itself remains un-optimised and unconditional.
- **Coach S3 edits are proposals only** (audit finding): the coach may emit an S3 candidate, but it enters the same perturbation gate as a CEM candidate and never writes the incumbent directly (invariants 7 and 9).
- **`cost_reference` frozen once** (audit finding): steps_ref / jerk_ref are Arm A means written a single time by `--make-reference`; later runs read the first `cost_reference` for the skill instance and never overwrite it, so costs across versions stay comparable.
- **One log file per job on Hopper**: NFS append from several nodes tears roughly 1 line per 1000 (large snapshot lines); readers tolerate a torn line, but separate files per job remove the problem entirely.
- **π₀.₅ kept as an unverified adapter, not a result**: 0/3 on standard LIBERO with an identity normaliser in the Hub checkpoint and no `lerobot-eval` ground truth; the adapter stays in-tree for the next attempt but no number is reported.
