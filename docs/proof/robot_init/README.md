# Proof of the robot-start results (2026-09-05, GMU Hopper)

Evidence for `docs/RESULTS.md` §4 and the README headline chart (LIBERO-Spatial task 0 under LIBERO-Plus robot-initial-state
perturbation: 80% → 22% → 54% → 70%).

| what | file |
|---|---|
| Slurm accounting for every 2026-09-05 job (ids, nodes, GPUs, start/end) | `sacct_2026-09-05.txt` |
| raw job stdout with the `REPS` / gate lines (`q_reps-*`, `q_cycle2-task3*`, `bm_*`, `bmfollow_*`) | `slurm_out/` |
| per-episode records of every held-out evaluation episode (seed, arm, `s3_params`, `outcome.env_success`, steps, timestamp) plus every `skill_instance` promotion, `consolidation` gate and `cost_reference` | `events_evaluation.jsonl` (extracted from `/scratch/ezhao2/fleet-memory/logs/benchmark/{reps,events,events_wide,power}.jsonl`; CEM/gate rollouts dropped for size) |
| recount from the raw records | `verify_output.txt` = `python scripts/verify_robot_init.py` |

`verify_robot_init.py` labels each arm from the record itself: arm A = raw policy; arm B with the identity file = BM-2; identity +
`homing_enable=1` = BM-4; any other file = BM-3, labelled with the version whose `skill_instance` record carries that exact vector.
It reproduces task 0: BM-0 40/50 · BM-1 33/150 · BM-2 40/150 · BM-4 55/150 · **BM-3 v2 54/100 · BM-3 v3 35/50** (v3 = the second
cycle through the 24-seed gate), the wide-search instance 60/150, and task 3 (27% in every arm, n=150). All evaluation seeds satisfy
`seed % 50 in 40..49` (init states the optimizer and the gate never see; `fleet_memory/runner/seeds.json`).
