# Fleet Memory on GMU Hopper

Cluster facts: `ssh hopper` (user `ezhao2`, account `ezhao`, QOS `gpu`). Partition `gpuq`.
Full A100 (`--gres=gpu:A100.80gb:1`) supports EGL rendering; MIG slices (`--gres=gpu:1g.10gb:1`)
do **not** — use `MUJOCO_GL=osmesa` there. Compute nodes have outbound internet. Never run
MuJoCo or the policy on the login node.

Paths:
- venv: `/scratch/ezhao2/fleet-memory/venv` (Python 3.12, `lerobot[smolvla,libero]==0.6.1`, torch 2.11 cu130)
- HF cache: `/scratch/ezhao2/hf_cache` (`HuggingFaceVLA/smolvla_libero`, `lerobot/smolvla_libero` prefetched)
- repo: `/scratch/ezhao2/fleet-memory/dnhacks26`
- logs: `/scratch/ezhao2/fleet-memory/logs/{slurm,smoke,phase0,phase1}`

## 1. Put the repo on the cluster

Either clone:
```bash
ssh hopper
mkdir -p /scratch/ezhao2/fleet-memory && cd /scratch/ezhao2/fleet-memory
git clone <REPO_URL> dnhacks26
```
or rsync from the laptop (what we do during development):
```bash
rsync -a /Users/max/Documents/dnhacks/ hopper:/scratch/ezhao2/fleet-memory/dnhacks26/ \
  --exclude .venv --exclude .git --exclude __pycache__ --exclude logs/
```

(Re)create the venv if needed: `bash /scratch/ezhao2/setup_env.sh` (log: `/scratch/ezhao2/setup_env.log`, ends with `SETUP_DONE`).

**LIBERO one-time gotchas** (both handled once; `env.sh` keeps the first one working):
1. `import libero` blocks on an interactive `input()` if `$LIBERO_CONFIG_PATH/config.yaml` is missing.
   `env.sh` writes it non-interactively to `/scratch/ezhao2/fleet-memory/libero_config/config.yaml`.
2. The `hf-libero` wheel ships **no mesh assets**; they are pulled from the HF Hub on first env
   construction (fails with `HF_HUB_OFFLINE=1` on a compute node). Prefetch once on the login node
   (network is fine there, no sim is run):
   ```bash
   source /scratch/ezhao2/fleet-memory/venv/bin/activate; export HF_HOME=/scratch/ezhao2/hf_cache
   python -c "from libero.libero.utils.download_utils import download_assets_from_huggingface as d; import libero.libero as l, os; print(d(download_dir=os.path.join(os.path.dirname(l.__file__),'assets')))"
   ```
3. SmolVLA loads its VLM backbone from the Hub at construction time (`config.vlm_model_name =
   HuggingFaceTB/SmolVLM2-500M-Instruct`, `load_vlm_weights=True`) — it is *not* part of the policy
   checkpoint. Prefetch once (login node, same env as above):
   ```bash
   python -c "from huggingface_hub import snapshot_download as s; print(s('HuggingFaceTB/SmolVLM2-500M-Instruct'))"
   ```
   With those cached, jobs run with `HF_HUB_OFFLINE=1` (env.sh default); set `HF_HUB_OFFLINE=0` to let
   a compute node fetch missing files itself (works, nodes have internet, but 4 workers may race).

## 2. Environment

```bash
source /scratch/ezhao2/fleet-memory/dnhacks26/scripts/hopper/env.sh
```
Activates the venv, sets `HF_HOME`, `HF_HUB_OFFLINE=1`, `MUJOCO_GL` (default `egl`), `PYTHONPATH`,
and sources `~/.fm_secrets` (put `export ANTHROPIC_API_KEY=...` there, `chmod 600`). Without a key
`FM_LLM=mock` is exported.

## 3. Smoke test (interactive, ~5 min)

```bash
srun -p gpuq -q gpu --gres=gpu:A100.80gb:1 -c 8 --mem=32G -t 00:30:00 --pty bash
source /scratch/ezhao2/fleet-memory/dnhacks26/scripts/hopper/env.sh
python scripts/smoke_libero.py --suite libero_10 --task 0 --seed 0 --steps 50  --policy zero
python scripts/smoke_libero.py --suite libero_10 --task 0 --seed 0 --steps 100 --policy smolvla
```
or as a batch job: `sbatch scripts/hopper/smoke.sbatch [suite] [task] [steps]`.
MIG fallback (if the A100 queue is long, check with `squeue -p gpuq | wc -l`):
```bash
srun -p gpuq -q gpu --gres=gpu:1g.10gb:1 -c 4 --mem=24G -t 00:30:00 --pty bash
MUJOCO_GL=osmesa source scripts/hopper/env.sh
python scripts/smoke_libero.py --policy zero --steps 20 --render-gl osmesa
```
Non-interactive one-liner:
```bash
srun -p gpuq -q gpu --gres=gpu:A100.80gb:1 -c 8 --mem=32G -t 00:30:00 \
  bash -lc 'source /scratch/ezhao2/fleet-memory/dnhacks26/scripts/hopper/env.sh && python scripts/smoke_libero.py --policy smolvla --steps 100'
```
The smoke prints `SMOKE_OK` and writes `logs/smoke_libero_10_0_<policy>_{first,last}.png`.

## 4. Phase 0 — baseline (arm A) on libero_10

```bash
cd /scratch/ezhao2/fleet-memory/dnhacks26
sbatch scripts/hopper/phase0.sbatch                 # array 0-9 = task index, N=50 seeds each, 4 workers/GPU
N=30 sbatch --array=0-9 scripts/hopper/phase0.sbatch
PARTITION=seed N=10 sbatch --array=0-4 scripts/hopper/phase0.sbatch   # all tasks, seed slices via --seed-offset
```
Underlying command (per array task):
```
python -m fleet_memory.runner.pool --env libero --policy smolvla --suite libero_10 --arm A \
   --n $N --workers 4 --seed-set train --tasks $SLURM_ARRAY_TASK_ID \
   --log /scratch/ezhao2/fleet-memory/logs/phase0/events_$SLURM_ARRAY_TASK_ID.jsonl
```

## 5. Phase 1 — S2 / S3 probes over identical seeds

```bash
sbatch scripts/hopper/phase1_probe.sbatch
PROBES_S2=probes/s2.txt PROBES_S3=probes/s3.jsonl N=30 sbatch scripts/hopper/phase1_probe.sbatch
```
`PROBES_S2`: one instruction paraphrase per line (`--probe-instruction`). `PROBES_S3`: one JSON
`Edit` per line (`--probe-edit`), ops from `EDIT_OPS["S3"]` in `memory/schema.py`. Logs land in
`logs/phase1/s{2,3}_probe<k>_events_<task>.jsonl`.

## 6. Monitor / pull logs back

```bash
squeue -u ezhao2
tail -f /scratch/ezhao2/fleet-memory/logs/slurm/phase0_<jobid>_0.out
scancel -u ezhao2 --name=fm-phase0
# laptop:
rsync -a hopper:/scratch/ezhao2/fleet-memory/logs/ /Users/max/Documents/dnhacks/logs/hopper/
cat logs/hopper/phase0/events_*.jsonl > logs/events.jsonl
```

## osmesa on MIG slices
GPU compute nodes have **no** `libOSMesa` installed (the login node does: `mesa-libOSMesa-23.1.4`).
`env.sh` (when `MUJOCO_GL=osmesa`) stages `libOSMesa.so.8`, `libglapi.so.0`, `libLLVM-17.so`, `libdrm`,
`libffi`, `libzstd` from the login node into `/scratch/ezhao2/fleet-memory/lib` the first time it is
sourced *on the login node*, and prepends that dir to `LD_LIBRARY_PATH`. So source `env.sh` once on
the login node with `MUJOCO_GL=osmesa` before the first MIG job. Software rendering is ~10x slower
than EGL; use it only for smoke tests / when the A100 queue is long.

## Notes
- Observation contract matches lerobot 0.6.1 eval exactly: images 256x256 flipped 180° (upright),
  state = eef_pos(3) + axis-angle(3) + gripper_qpos(2) -> float32[8], task string = benchmark language.
- Checkpoint config: chunk_size=50, n_action_steps=1 (lerobot_eval re-queries every step). Set
  `SmolVLAPolicy(n_action_steps=k)` to execute k actions per query (much faster, slightly different behaviour).
- Reset = `env.seed(seed)`, `env.reset()`, `set_init_state(init_states[seed % n_init])`, 10 no-op settle steps.
