# Fleet Memory v3.1

Weight-frozen improvement of a VLA (SmolVLA-LIBERO / π₀.₅) on LIBERO, keyed per environment and versioned.
Two writers: an LLM coach for discrete knowledge (S1 plans, lessons; Gemini or Claude) and a sleep-loop CEM
optimizer for a 17-dim continuous S3 vector (approach shaping, time scale, velocity cap, gripper, homing),
promoted only through a perturbation gate on seeds the optimizer never saw. Everything is an append-only
event log; the env success predicate is the only source of truth. See `CLAUDE.md` for the invariants.

## Commands (laptop, no GPU — mock env/policy, offline LLM)
```
uv venv .venv --python 3.12 && source .venv/bin/activate && uv pip install -e . pytest google-genai
FM_LLM=mock pytest -q                                                                   # 120 tests
FM_LLM=mock python -m fleet_memory.runner.pool --env mock --policy mock --suite mock --task pick_bowl_to_plate \
    --arm A --n 40 --workers 4 --seed-set train --log logs/v3.jsonl --make-reference      # then --arm B/C/D
FM_LLM=mock python -m fleet_memory.runner.consolidate --env mock --policy mock --suite mock \
    --task pick_bowl_to_plate --log logs/v3.jsonl --small                                # one sleep cycle
FM_LLM=mock python -m fleet_memory.runner.pool ... --arm P --protocol P --n-stage 20 \
    --perturb '{"shift_xy":[0.06,0]}' --auto-sleep --log logs/v3.jsonl                  # perturb → drift → re-sleep
python -m fleet_memory.analysis.metrics --log logs/v3.jsonl                             # results table
open dashboard/index.html                                                                # drop logs/*.jsonl on it
```

## Hopper (GMU ORC, real sim + policy)
`scripts/hopper/README.md` — env setup, `phase0_v3.sbatch` (arm A + cost reference), `probe_p1b.sbatch` (P1 probes),
`sleep_v31.sbatch` (3 sleep cycles + arm-B evals), `LIBERO_PLUS.md` + `python -m fleet_memory.runner.benchmark`
(arms BM-0..4 on LIBERO-Plus robot-initial-state perturbation, eval seeds from `fleet_memory/runner/seeds.json`).
