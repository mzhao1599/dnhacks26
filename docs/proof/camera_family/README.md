# Proof of the camera-viewpoint results (2026-09-06, GMU Hopper)

Everything in `docs/RESULTS.md` §8 and the README's camera section is derived from the files in this folder.
Nothing was typed by hand; every success flag comes from LIBERO's own goal predicate (`env.check_success()`), written to an
append-only JSONL log by the cluster job that ran the episode.

## Chain of evidence

| what | file | what it proves |
|---|---|---|
| Slurm accounting | `sacct_2026-09-06.txt` | job ids, nodes (gpu012/018/024 = A100.80gb, gpu021 = MIG, hop/amd = CPU), start/end, elapsed, GPUs allocated |
| raw job stdout | `slurm_out/q_*.out` | the exact `CYCLE` / `INCUMBENT` / `REPS` lines each job printed, with job ids matching sacct |
| event log (evaluation subset) | `events_evaluation.jsonl` | one JSON record per held-out episode: seed, arm, S3 parameter file, success, steps, timestamp; plus every `consolidation` (gate) and `skill_instance` (promotion) record, and protocol P's `protocol_stage` / `drift_trigger` records. Extracted from `/scratch/ezhao2/fleet-memory/logs/{bench_camera,protocol_cam}/events.jsonl` on Hopper (40 MB + 8 MB; CEM/gate rollouts and coach snapshots dropped for size, `_source` says which file) |
| recomputation | `verify_output.txt` = `python scripts/verify_camera.py` | recounts every arm from the raw episodes (not from any summary) and prints the gates from the consolidation records |
| the perturbation | `frame_stock_camera.png`, `frame_camera_tilted_0_0_100_2_354.png` | first agentview frame the policy sees, same init state, stock vs tilted camera |
| videos | `cam_BM-1_s5047_fail.mp4`, `cam_BM-3_s5047_ok.mp4` (`cam_BM-0_s5047_fail.mp4`) | same seed 5047, tilted camera: raw policy fails, the consolidated file succeeds (117 steps). Stock camera also failed on this particular seed (CPU-node render; the stock policy is 88% on the A100 reps). Overlay shows arm, seed and active S3 values |

## Verify in one command

```
python scripts/verify_camera.py          # reads logs/hopper/... ; or point it at this folder's copy:
python - <<'PY'
import json, collections
acc = collections.defaultdict(lambda: [0, 0])
for l in open("docs/proof/camera_family/events_evaluation.jsonl"):
    d = json.loads(l)
    if d.get("type") != "episode" or int(d["seed"]) < 5040 or not (40 <= int(d["seed"]) % 50 <= 49): continue
    arm = "BM-0" if "plus_camera" not in d["environment_id"] else "BM-1" if d["condition"] == "A" else ("BM-2" if abs(d["s3_params"]["time_scale"] - 1) < 1e-9 and d["s3_params"]["cam_shift_xy"] == [0.0, 0.0] else "BM-3")
    a = acc[(d["environment_id"], arm)]; a[0] += int(d["outcome"]["env_success"]); a[1] += 1
for k, (s, n) in sorted(acc.items()): print(f"{k[0]:50} {k[1]}  {s}/{n} = {100*s/n:.0f}%")
PY
```

Expected (task 0, view `0_0_100_2_354`, seeds 5040–5249 = init states 40–49 × 5 policy-noise draws):
BM-0 44/50 · BM-1 7/50 · BM-2 3/50 · **BM-3 29/50**; action-dims-only instance (`_act`): BM-1 4/50 · BM-2 6/50 · BM-3 7/49
(the job itself counted 8/50 — one record was torn by NFS when the log was mirrored; `slurm_out/q_cam-t0-tilt-act-a80_9581993.out`
line 70 has the job's own count). Task 1: 35/50 · 3/50 · 2/50 · 4/49. Task 3: 33/50 · 4/50 · 4/50 · 9/50.

## Why the numbers cannot be gamed by the loop

* The optimizer (CEM) only ever runs on init states 0–29 (seeds 3xxx) and the gate on 30–39 (seeds 4xxx); all evaluation
  episodes above use init states 40–49 (seed % 50 in 40–49). `runner/seeds.json` fixes the split; `verify_camera.py` filters on it.
* Success is `outcome.env_success`, copied from the simulator's predicate; no LLM or human judgement anywhere.
* The camera perturbation is a port of LIBERO-Plus's `_setup_camera`; `tests/test_camera.py` checks our camera poses against
  values computed with their own helper functions.
* The only difference between BM-1 and BM-3 is the `s3_params` dict in each episode record (identity vs the promoted file);
  the promoted file and the gate that promoted it are in the `skill_instance` / `consolidation` records of the same log.
