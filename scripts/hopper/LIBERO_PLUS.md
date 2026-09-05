# LIBERO-Plus (robot-initial-state perturbation) and π₀.₅ on Hopper

LIBERO-Plus: Fei et al., *In-depth Robustness Analysis of Vision-Language-Action Models*, arXiv 2510.13626.
Code `github.com/sylvestf/LIBERO-plus` (clone: `/scratch/ezhao2/fleet-memory/LIBERO-plus`, commit
`4976dc3`, 2026-01-21), assets `huggingface.co/datasets/Sylvest/LIBERO-plus` (one 6.4 GB `assets.zip`).

## 1. What LIBERO-Plus is, mechanically

LIBERO-Plus is a **fork of LIBERO that replaces the `libero` package** (`pip install -e .`; "simply replace
the originally installed LIBERO"). It pins `robosuite==1.4.0`, `bddl==1.0.1`, `gym==0.25.2` (our shared venv
has robosuite 1.4.0 / mujoco 3.8.1 / hf-libero 0.1.4 — the same LIBERO code base, patched by lerobot for
gymnasium + Hub assets; `diff -r` shows only benchmark tables, env_wrapper, robots, problems, objects differ).

Every perturbation is a **new task name** in `libero/libero/benchmark/libero_suite_task_map.py`
(libero_spatial 2402, libero_object 2518, libero_goal 2591, libero_10 2519, libero_90 90 tasks) plus
`benchmark/task_classification.json` = `{suite: [{id, name, category, difficulty_level}]}` with the seven
categories (Objects Layout, Camera Viewpoints, Robot Initial States, Language Instructions, Light Conditions,
Background Textures, Sensor Noise). The eval loop is unchanged (`OffScreenRenderEnv(bddl_file_name=...)`,
`env.reset()`, `env.set_init_state(init_states[k])`, no-op settle steps, `num_trials_per_task` 50 -> 1).
The perturbation is decoded **from the file name** inside `libero/libero/envs/env_wrapper.py::ControlEnv.__init__`:

```
<base>_view_<h>_<v>_<scale%>_<end_rot>_<end_vert>_initstate_<N>[_noise_<k>].bddl
```
* `view_*` -> camera pose (`Problem._setup_camera` rotates/scales the agentview camera; `0_0_100_0_0` = identity)
* `initstate_N` -> **`robots = ["Panda"+N]`** -> the problem class prefixes `Mounted` -> **`MountedPanda{N}`**
  (`libero/libero/envs/robots/mounted_panda.py`, N = 1..500, registered in robosuite's `ROBOT_CLASS_MAPPING`).
  `MountedPanda{N}` differs from `MountedPanda` **only in `init_qpos`**:
  `init_qpos = base_qpos + r * u`, `u` a random unit vector in R^7 (`np.random.seed(42)`, sequential draws,
  generator script `robots/new_init.py`), `r = 0.1 * ceil(N/100)` -> 0.1 rad (N 1-100) ... 0.5 rad (N 401-500).
  `base_qpos = [0, -0.161, 0, -2.4446, 0, 2.2268, pi/4]`. The paper (App. A.5): "random perturbations are
  applied to the robot arm's initial joint positions (qpos); magnitudes 0.1 to 0.5".
* `noise_k` -> image corruption applied to `agentview_image` in `step/reset` (needs ImageMagick `wand`).
* `_add_N` (Objects Layout) -> separate bddl files with extra/moved objects + their own init state in
  `init_files/libero_newobj/<suite>/<name>.pruned_init` (1-D, reshaped to (1, D)); `_table_N`/`_tb_N`
  (textures), `_light_N`, `_language_N` are separate bddl files sharing the base init states.

There are **no per-dimension config files**: `list_configs()` in `fleet_memory/envs/libero_plus.py` builds
config dicts from `task_classification.json` + the qpos table parsed from `mounted_panda.py`
(the seed-42 regeneration reproduces all 500 rows to 1e-7). For `robot_init` a config looks like
```
{"dimension":"robot_init","suite":"libero_spatial","plus_id":259,
 "name":"pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_1",
 "base_task":"pick_up_the_black_bowl_between_..._on_the_plate","base_task_idx":0,"difficulty":1,
 "variant":"view_0_0_100_0_0_initstate_1","init_state":1,"view":"0_0_100_0_0","radius":0.1,"init_qpos":[7 floats]}
```
Robot-init tasks: 350 in libero_spatial (32 for task 0), 409 in libero_goal (43 for task 0); all have the identity
view. Robot-init tasks need **no new assets** (base scenes); layout tasks need `new_objects/` from `assets.zip`.

**Subtlety that matters (found by running it):** `get_task_init_states()` for an `_initstate_` task loads the
*base* `.pruned_init`, and `set_init_state()` writes the full flattened MuJoCo state, i.e. it **overwrites the
perturbed joints again**. What survives in their loop is the OSC controller's nullspace target
(`Controller.initial_joint`, set from `MountedPanda{N}.init_qpos` at reset). Under delta OSC control the
goal is re-anchored to the current EE pose every step, so the nullspace pull makes the arm *drift* toward the
perturbed configuration: in the MountedPanda1 env the EE had already moved 1.4 cm during the 10 settle steps
(`[-0.2112,-0.0111,1.1742]` -> `[-0.2027,-0.0058,1.1844]`) and keeps drifting during the episode. Our
`apply_robot_init_state()` instead writes the perturbed `qpos` **after** `set_init_state` (+ updates the
nullspace target, zero qvel, `sim.forward`, 10 no-op settle steps) = the perturbation the paper describes,
deterministic per N. The smoke supports both (`robot_init:0` vs `robot_init:0/asis`).

## 2. Install / venvs

* Shared venv `/scratch/ezhao2/fleet-memory/venv` is untouched (hf-libero). **`native` backend**: plain
  `LiberoEnv` + joint rewrite; only needs the LIBERO-plus checkout for the tables (`FM_LIBERO_PLUS`).
* **`venv_plus`** (`/scratch/ezhao2/fleet-memory/venv_plus`, built by `/scratch/ezhao2/setup_venv_plus.sh`
  with uv, cache `/scratch/ezhao2/.uv_cache`): python 3.12, `lerobot[smolvla,pi,scipy-dep,dataset]==0.6.1`
  (no `[libero]` extra), `robosuite==1.4.0 mujoco==3.8.1 bddl==1.0.1 gym==0.26.2 wand scikit-image ...`,
  then `uv pip install -e /scratch/ezhao2/fleet-memory/LIBERO-plus --no-deps`. Two one-time fixes:
  1. the clone lacks `libero/__init__.py` (setuptools `find_packages` found nothing) -> `touch libero/__init__.py`;
  2. LIBERO-plus resolves assets as `libero/libero/assets` (hard-coded relative path) ->
     `ln -s <venv>/site-packages/libero/libero/assets LIBERO-plus/libero/libero/assets` (base assets, 404 MB,
     enough for robot-init); full `assets.zip` (6.4 GB, 457k files, nested under
     `inspire/hdd/.../LIBERO-plus-0/assets/{new_objects,scenes,...}`) is being extracted in
     `/scratch/ezhao2/fleet-memory/libero_plus_assets/` for layout tasks.
  3. `import libero` prompts on stdin if `$LIBERO_CONFIG_PATH/config.yaml` is missing ->
     `/scratch/ezhao2/fleet-memory/libero_plus_config/config.yaml` points bddl_files/init_states at the checkout.
  4. Compute nodes have no ImageMagick (`wand` import fails; login node has it) -> `libero_plus.py` installs a
     no-op `wand` stub at import time (sensor-noise dimension unusable on those nodes, nothing else needs it).
* π₀.₅ runs in the **shared venv** (lerobot 0.6.1 ships `lerobot/policies/pi05`; transformers 5.5.4 is enough).
  Prefetched on the login node: `hf download lerobot/pi05_libero` (14.5 GB, float32) and the tokenizer.
  `google/paligemma-3b-pt-224` is **gated** -> `fleet_memory/policies/pi05.py` substitutes the ungated mirror
  `leo009/paligemma-3b-pt-224` (GemmaTokenizer, vocab 257152 + `<image>`, `<loc0000>`=256000, `<seg000>`=257024,
  identical ids for text) via `preprocessor_overrides={"tokenizer_processor": {"tokenizer_name": ...}}`
  (`FM_PALIGEMMA_TOKENIZER` overrides).

## 3. Commands

```bash
# LIBERO-plus smoke (A100.40gb, ~20 min for 10 SmolVLA episodes)
sbatch scripts/hopper/libero_plus_smoke.sbatch libero_spatial 0 0,1,2,3,4 220 smolvla standard,robot_init:0
BACKEND=native sbatch scripts/hopper/libero_plus_smoke.sbatch ...            # shared venv
BACKEND=plus   sbatch scripts/hopper/libero_plus_smoke.sbatch libero_spatial 0 0,1,2,3,4 220 smolvla \
                      "robot_init:0/asis,robot_init:init_state=450"           # their loop as-is / r=0.5
# zero-policy sanity (20 steps, prints standard vs perturbed EE pose)
BACKEND=plus sbatch -t 00:20:00 scripts/hopper/libero_plus_smoke.sbatch libero_spatial 0 0,1 20 zero standard,robot_init:0
# pi0.5
sbatch scripts/hopper/pi05_smoke.sbatch libero_spatial 0 0,1,2 220 standard   # DTYPE=bfloat16 N_ACTION_STEPS=10
```
Python API: `list_configs("robot_init", "libero_spatial")`, `make_perturbed_env(suite, task, cfg)`,
`apply_robot_init_state(env, cfg, seed)`; config shortcuts `{"dimension":"robot_init","init_state":N}` or
`{"dimension":"robot_init","radius":0.3}` (fresh seeded direction). Logs: `logs/slurm/plus_<job>.out`,
`logs/slurm/pi05_<job>.out`; per-episode `PLUS_RESULT {json}`, per-config `PLUS_SUMMARY {json}`.

## 4. Measured (libero_spatial task 0, "pick up the black bowl between the plate and the ramekin ...", 220 steps)

RESULTS_PLACEHOLDER

## 5. Open issues

ISSUES_PLACEHOLDER
