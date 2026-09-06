"""LIBERO-Plus (Fei et al., arXiv 2510.13626) perturbation adapter for the Env protocol.

Mechanism of the "Robot Initial States" dimension (read from github.com/sylvestf/LIBERO-plus):
* a benchmark task is named ``<base>_view_<h>_<v>_<scale%>_<rot>_<vert>_initstate_<N>[_noise_<k>]``;
  ``ControlEnv.__init__`` (libero/libero/envs/env_wrapper.py) parses that name, maps it back to the
  base ``<base>.bddl`` and builds the problem with ``robots=["Panda"+N]`` -> ``MountedPanda{N}``;
* ``MountedPanda{N}`` (libero/libero/envs/robots/mounted_panda.py, N=1..500) only overrides
  ``init_qpos`` = base Panda qpos + r * u, u a random unit vector in R^7 (np.random.seed(42)),
  r = 0.1 * ceil(N/100)  (0.1 rad for N<=100 ... 0.5 rad for N>400).  No init-state files change:
  ``get_task_init_states`` loads the base ``.pruned_init``.
* NOTE: in the reference eval loop (``env.reset(); env.set_init_state(init_states[k])``) the
  flattened MuJoCo state overwrites the perturbed joints again; only the OSC controller's nullspace
  target keeps the perturbed qpos.  ``apply_robot_init_state`` therefore writes the perturbed joint
  qpos *after* ``set_init_state`` (then re-settles), which is the perturbation the paper describes.

"Camera Viewpoints" dimension (``camera_pose_for_view``, an exact port of LIBERO-Plus's
``Libero_Tabletop_Manipulation._setup_camera`` incl. its 4-decimal rounding): the task name's
``view_<h>_<v>_<scale%>_<end_rot>_<end_vert>`` rotates the agentview camera *position+orientation* about z by
``h`` deg and about a y-parallel axis through (0, 0, 0.8) by ``v`` deg, scales its distance from that pivot,
then turns the optical axis by ``end_rot`` (z) / ``end_vert`` (y). ``0_0_100_0_0`` = the stock camera.
``apply_camera_view`` writes the pose into the compiled model's ``cam_pos/cam_quat`` after every reset
(hard resets rebuild the model) and re-renders the observation.

Two backends, picked automatically from the imported ``libero`` package:
* native  (shared venv, hf-libero): plain LiberoEnv + joint-qpos rewrite; needs only the LIBERO-Plus
  repo checkout (``FM_LIBERO_PLUS``) for the qpos table / task_classification.json (or regenerates
  the table with the seed-42 recipe when the checkout is absent);
* plus    (venv_plus, LIBERO-plus installed as ``libero``): the env is built from the encoded task
  name exactly as their eval does (MountedPanda{N} + controller nullspace target), then the joint
  qpos is re-applied unless ``config["reapply_qpos"] is False``.
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from fleet_memory.envs.base import Obs
from fleet_memory.envs.libero_env import DUMMY_ACTION, NUM_STEPS_WAIT, SUITE_MAX_STEPS, LiberoEnv, slugify

PLUS_ROOT = Path(os.environ.get("FM_LIBERO_PLUS", "/scratch/ezhao2/fleet-memory/LIBERO-plus"))
ORIG_TASKMAP = os.environ.get("FM_LIBERO_TASKMAP",
    "/scratch/ezhao2/fleet-memory/venv/lib/python3.12/site-packages/libero/libero/benchmark/libero_suite_task_map.py")
BASE_QPOS = np.array([0.0, -1.61037389e-01, 0.0, -2.44459747e00, 0.0, 2.22675220e00, np.pi / 4])
DIMENSIONS = {"robot_init": "Robot Initial States", "layout": "Objects Layout", "camera": "Camera Viewpoints",
              "language": "Language Instructions", "light": "Light Conditions",
              "background": "Background Textures", "noise": "Sensor Noise"}
_SUFFIX = re.compile(r"_(view_.*|add_\d+|table_\d+|tb_\d+|language_.*|light_.*|level.*|noise_.*)$")
# stock agentview camera of every LIBERO tabletop scene (hf-libero == LIBERO-Plus ``pos_av`` / ``quat_av``, wxyz)
CAM_POS_AV = [0.6586131746834771, 0.0, 1.6103500240372423]
CAM_QUAT_AV = [0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]
CAM_PIVOT = np.array([0.0, 0.0, 0.8])
IDENTITY_VIEW = "0_0_100_0_0"


# --------------------------------------------------------------------------- #
# LIBERO-Plus tables
# --------------------------------------------------------------------------- #
def _ensure_wand() -> None:
    """LIBERO-plus's env_wrapper imports ImageMagick's ``wand`` at import time (sensor-noise dimension
    only). Compute nodes may lack libMagickWand -> install a no-op stand-in so other dimensions load."""
    try:
        import wand.api, wand.image  # noqa: F401
    except Exception:
        import sys
        import types
        api = types.ModuleType("wand.api")
        api.library = types.SimpleNamespace(MagickMotionBlurImage=types.SimpleNamespace())
        image = types.ModuleType("wand.image")
        image.Image = type("Image", (), {})
        pkg = types.ModuleType("wand")
        pkg.api, pkg.image = api, image
        sys.modules.update({"wand": pkg, "wand.api": api, "wand.image": image})


_ensure_wand()   # before any libero import (LIBERO-plus's env package imports wand at module level)


def is_plus_backend() -> bool:
    try:
        from libero.libero.envs.robots import mounted_panda
        return hasattr(mounted_panda, "MountedPanda1")
    except Exception:
        return False


def plus_root() -> Path:
    if PLUS_ROOT.exists():
        return PLUS_ROOT
    try:
        import libero
        return Path(libero.__file__).resolve().parents[1]
    except Exception:
        return PLUS_ROOT


def _regen_qpos_table() -> dict[int, np.ndarray]:
    """Exact recipe of libero/libero/envs/robots/new_init.py (np.random.seed(42), sequential draws)."""
    np.random.seed(42)
    out = {}
    for n in range(1, 501):
        u = np.random.randn(7)
        out[n] = BASE_QPOS + u / np.linalg.norm(u) * (0.1 * math.ceil(n / 100))
    return out


_QPOS_CACHE: dict[int, np.ndarray] | None = None


def robot_init_qpos_table() -> dict[int, np.ndarray]:
    """{N: init_qpos(7)} parsed from LIBERO-Plus's mounted_panda.py (fallback: regenerated)."""
    global _QPOS_CACHE
    if _QPOS_CACHE is None:
        src = plus_root() / "libero/libero/envs/robots/mounted_panda.py"
        table: dict[int, np.ndarray] = {}
        if src.exists():
            pat = re.compile(r"class MountedPanda(\d+)\(MountedPanda\):.*?np\.array\(\[(.*?)\]\)", re.S)
            for n, arr in pat.findall(src.read_text()):
                table[int(n)] = np.array([float(x) for x in arr.replace("\n", " ").split(",")])
        _QPOS_CACHE = table or _regen_qpos_table()
    return _QPOS_CACHE


def _exec_taskmap(path: Path, suite: str) -> list[str]:
    ns: dict[str, Any] = {}
    exec(path.read_text(), ns)          # plain dict literal; exec avoids importing the benchmark package
    return list(ns["libero_task_map"][suite])


def base_task_names(suite: str) -> list[str]:
    """Canonical (original LIBERO) task order of a suite, e.g. index 0 of libero_spatial."""
    try:
        import libero.libero as L
        names = _exec_taskmap(Path(L.__file__).parent / "benchmark/libero_suite_task_map.py", suite)
        if all(_SUFFIX.search(n) is None for n in names):
            return names
    except Exception:
        pass
    return _exec_taskmap(Path(ORIG_TASKMAP), suite)   # hf-libero's table (bare names, canonical order)


def split_task_name(name: str) -> tuple[str, str]:
    m = _SUFFIX.search(name)
    return (name[: m.start()], m.group(1)) if m else (name, "")


def list_configs(dimension: str, suite: str) -> list[dict]:
    """One dict per LIBERO-Plus task of ``dimension`` ('robot_init', 'layout', ... or the category
    name) in ``suite``; ``base_task_idx`` indexes the original 10-task suite."""
    cat = DIMENSIONS.get(dimension, dimension)
    key = next((k for k, v in DIMENSIONS.items() if v == cat), dimension)
    items = json.loads((plus_root() / "libero/libero/benchmark/task_classification.json").read_text())[suite]
    bases = base_task_names(suite)
    table = robot_init_qpos_table() if key == "robot_init" else {}
    out = []
    for it in items:
        if it["category"] != cat:
            continue
        base, variant = split_task_name(it["name"])
        cfg = {"dimension": key, "suite": suite, "plus_id": it["id"], "name": it["name"], "base_task": base,
               "base_task_idx": bases.index(base) if base in bases else -1,
               "difficulty": it.get("difficulty_level"), "variant": variant}
        if key == "robot_init":
            view, n = variant.split("_initstate_")
            n = int(n.split("_noise_")[0])
            cfg.update(init_state=n, view=view[len("view_"):], radius=round(0.1 * math.ceil(n / 100), 1),
                       init_qpos=table[n].tolist())
        elif key == "layout":
            cfg.update(bddl=f"{suite}/{it['name']}.bddl", init_states_file=f"libero_newobj/{suite}/{it['name']}.pruned_init")
        elif key == "camera":
            view, n = variant.split("_initstate_")
            cfg.update(view=view[len("view_"):], init_state=int(n.split("_noise_")[0]), **parse_view(view[len("view_"):]))
        out.append(cfg)
    return out


# --------------------------------------------------------------------------- #
# Camera viewpoints (port of LIBERO-Plus libero_tabletop_manipulation.py: rotate_around_y/z, scale_distance_from_pivot)
# --------------------------------------------------------------------------- #
def parse_view(view: str) -> dict[str, float]:
    """'11_15_100_0_0' -> {h_deg, v_deg, scale, end_rot_deg, end_vert_deg} (their ints; scale = percent / 100)."""
    h, v, sc, er, ev = view.split("_")
    return {"h_deg": int(h), "v_deg": int(v), "scale": int(sc) / 100.0, "end_rot_deg": int(er), "end_vert_deg": int(ev)}


def _rot_of(q_wxyz):
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])


def _wxyz(rot) -> list[float]:
    x, y, z, w = rot.as_quat()
    return [float(w), float(x), float(y), float(z)]


def _rotate_around_y(quat=None, pos=None, degrees=0):
    """LIBERO-Plus rotate_around_y: about the y-parallel axis through (0, 0, 0.8); positive turns x toward z."""
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_rotvec(np.radians(-degrees) * np.array([0.0, 1.0, 0.0]))
    out = {}
    if quat is not None:
        out["quat"] = _wxyz(rot * _rot_of(quat))
    if pos is not None:
        out["pos"] = (rot.apply(np.asarray(pos, np.float64) - CAM_PIVOT) + CAM_PIVOT).tolist()
    return out


def _rotate_around_z(quat=None, pos=None, degrees=0):
    """LIBERO-Plus rotate_around_z: about the world z axis (positive = counter-clockwise from above)."""
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_euler("z", degrees, degrees=True)
    out = {}
    if quat is not None:
        out["quat"] = _wxyz(rot * _rot_of(quat))
    if pos is not None:
        out["pos"] = rot.apply(np.asarray(pos, np.float64)).tolist()
    return out


def _r4(x) -> list[float]:
    return [round(float(v), 4) for v in x]


def camera_pose_for_view(view: str) -> tuple[list[float], list[float]]:
    """(pos, quat_wxyz) of the agentview camera for a LIBERO-Plus view string, bit-for-bit their _setup_camera
    (same operation order and the same round(x, 4) after every step)."""
    p = parse_view(view)
    pos, quat = list(CAM_POS_AV), list(CAM_QUAT_AV)
    if p["v_deg"] != 0:
        r = _rotate_around_y(quat, pos, p["v_deg"]);       pos, quat = _r4(r["pos"]), _r4(r["quat"])
        r = _rotate_around_z(quat, pos, p["h_deg"]);       pos, quat = _r4(r["pos"]), _r4(r["quat"])
    else:
        r = _rotate_around_z(quat, pos, p["h_deg"]);       pos, quat = _r4(r["pos"]), _r4(r["quat"])
    if p["scale"] != 1.0:
        pos = _r4(CAM_PIVOT + (np.asarray(pos) - CAM_PIVOT) * p["scale"]);  quat = _r4(quat)
    if p["end_rot_deg"] != 0:
        quat = _r4(_rotate_around_z(quat, degrees=p["end_rot_deg"])["quat"])
    if p["end_vert_deg"] != 0:
        quat = _r4(_rotate_around_y(quat, degrees=p["end_vert_deg"])["quat"])
    return pos, quat


def apply_camera_view(env: LiberoEnv, config: dict) -> Obs:
    """Move the compiled model's agentview camera to the config's view (``config["view"]``, e.g. '11_15_100_0_0')
    and re-render the observation. Must run after every reset (hard_reset rebuilds the model). Records
    before/after in ``env.last_perturbation['camera']``."""
    import mujoco
    view = str(config.get("view") or IDENTITY_VIEW)
    pos, quat = camera_pose_for_view(view)
    sim = env.raw_env.env.sim
    model = getattr(sim.model, "_model", sim.model)
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "agentview")
    if cid < 0:
        raise RuntimeError("no 'agentview' camera in the model")
    before_pos, before_quat = model.cam_pos[cid].copy(), model.cam_quat[cid].copy()
    model.cam_pos[cid] = np.asarray(pos, np.float64)
    model.cam_quat[cid] = np.asarray(quat, np.float64)
    sim.forward()
    obs = env.current_obs()
    env.last_perturbation["camera"] = {"view": view, **parse_view(view), "cam_pos_before": _r(before_pos),
                                       "cam_quat_before": _r(before_quat), "cam_pos": _r(pos), "cam_quat": _r(quat),
                                       "cam_shift_m": round(float(np.linalg.norm(np.asarray(pos) - before_pos)), 4)}
    return obs


# --------------------------------------------------------------------------- #
# Perturbation
# --------------------------------------------------------------------------- #
def robot_init_qpos(config: dict, seed: int) -> np.ndarray:
    """Target joint qpos for a robot_init config: explicit ``init_qpos`` > table[``init_state``] >
    fresh LIBERO-Plus-style sample of norm ``radius`` (seeded)."""
    if config.get("init_qpos") is not None:
        return np.asarray(config["init_qpos"], dtype=np.float64)
    if config.get("init_state"):
        return robot_init_qpos_table()[int(config["init_state"])]
    u = np.random.default_rng(seed).standard_normal(7)
    return BASE_QPOS + u / np.linalg.norm(u) * float(config.get("radius", 0.1))


def _r(x, nd: int = 4) -> list:
    return np.round(np.asarray(x, dtype=np.float64), nd).tolist()


def apply_robot_init_state(env: LiberoEnv, config: dict, seed: int) -> Obs:
    """Write the perturbed arm joint qpos into the (already reset) sim, update the OSC nullspace
    target like robosuite's reset does, re-settle NUM_STEPS_WAIT no-op steps. Records before/after
    in ``env.last_perturbation['robot_init']`` and returns the new Obs (ee pose differs)."""
    raw_env = env.raw_env
    pe, robot = raw_env.env, raw_env.env.robots[0]
    q = robot_init_qpos(config, seed)
    before = env.current_obs()
    q_before = np.array(pe.sim.data.qpos[robot._ref_joint_pos_indexes])
    pe.sim.data.qpos[robot._ref_joint_pos_indexes] = q
    pe.sim.data.qvel[robot._ref_joint_vel_indexes] = 0.0
    pe.sim.forward()
    robot.controller.update_initial_joints(q)
    raw = None
    for _ in range(NUM_STEPS_WAIT):
        raw, _, _, _ = raw_env.step(DUMMY_ACTION.copy())
    for r in raw_env.robots:
        r.controller.use_delta = True
    env._t = 0
    obs = env._make_obs(raw)
    env._last_obs = obs
    env.last_perturbation["robot_init"] = {
        "init_state": config.get("init_state"), "radius": config.get("radius"),
        "qpos_before": _r(q_before), "qpos_target": _r(q), "qpos_after": _r(pe.sim.data.qpos[robot._ref_joint_pos_indexes]),
        "ee_pos_before": _r(before.ee_pos), "ee_pos_after": _r(obs.ee_pos),
        "ee_quat_before": _r(before.ee_quat), "ee_quat_after": _r(obs.ee_quat),
        "ee_shift_m": round(float(np.linalg.norm(obs.ee_pos - before.ee_pos)), 4),
    }
    return obs


# --------------------------------------------------------------------------- #
# Env
# --------------------------------------------------------------------------- #
class LiberoPlusEnv(LiberoEnv):
    """LiberoEnv whose reset applies a LIBERO-Plus config. Reuses every LiberoEnv method (obs
    construction, success predicate, step); only the constructor is replaced because under the
    LIBERO-plus package the benchmark tables index 2402+ tasks and lack the base init files."""

    def __init__(self, suite: str, task_id: str | int, config: dict, image_size: int = 256,
                 max_steps: int | None = None, render_gl: str | None = None):
        render_gl = render_gl or os.environ.get("MUJOCO_GL") or "egl"   # respect MUJOCO_GL (osmesa on MIG slices)
        if os.environ.get("MUJOCO_GL") != render_gl:
            os.environ["MUJOCO_GL"] = render_gl
        from libero.libero import get_libero_path
        import libero.libero.envs.bddl_utils as BDDLUtils

        bases = base_task_names(suite)
        if isinstance(task_id, str) and not task_id.isdigit():
            want = slugify(task_id)
            task_id = next(i for i, n in enumerate(bases) if slugify(n).startswith(want))
        self.suite, self.task_idx, self.config = suite, int(task_id), dict(config)
        base = bases[self.task_idx]
        if self.config.get("base_task") not in (None, base):
            raise ValueError(f"config is for {self.config['base_task']!r}, task {task_id} is {base!r}")
        self.plus = is_plus_backend()
        if self.config.get("reapply_qpos") is False and not (self.plus and self.config.get("name")):
            raise ValueError("reapply_qpos=False needs the LIBERO-plus backend and a benchmark task name")
        bddl_dir, init_dir = Path(get_libero_path("bddl_files")), Path(get_libero_path("init_states"))
        self._bddl = str(bddl_dir / suite / f"{base}.bddl")
        init_file = init_dir / suite / f"{base}.pruned_init"
        if self.plus and self.config.get("name"):            # encoded name -> MountedPanda{N} etc.
            self._bddl = str(bddl_dir / suite / f"{self.config['name']}.bddl")
            if self.config.get("dimension") == "layout":
                init_file = init_dir / self.config["init_states_file"]
        self.language: str = BDDLUtils.get_problem_info(str(bddl_dir / suite / f"{base}.bddl"))["language_instruction"]
        self.slug = slugify(self.language)
        self.image_size, self.max_steps = int(image_size), int(max_steps or SUITE_MAX_STEPS.get(suite, 500))
        import torch
        states = np.asarray(torch.load(init_file, weights_only=False))
        self._init_states = states[None] if states.ndim == 1 else states   # libero_newobj files are 1-D
        self.n_init_states = int(len(self._init_states))
        self._bench = self._task = None
        self._env, self._t, self._last_raw, self._last_obs, self._objects = None, 0, {}, None, None
        self.last_seed = self.init_state_index = None
        self.last_perturbation: dict[str, Any] = {}

    def reset(self, seed: int, perturbation: dict[str, Any] | None = None) -> Obs:
        obs = super().reset(seed, perturbation)
        dim = self.config.get("dimension")
        if dim == "robot_init" and self.config.get("reapply_qpos", True):
            obs = apply_robot_init_state(self, self.config, seed)
        elif dim == "camera":
            obs = apply_camera_view(self, self.config)
        self.last_perturbation["plus"] = {k: self.config.get(k) for k in ("dimension", "name", "init_state", "radius", "view")}
        return obs

    def canonical_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """(pos, quat_xyzw) of the EE at the UNPERTURBED reset of this env+seed — the homing target.
        Recorded by apply_robot_init_state before it rewrites the joints; falls back to the current reset pose."""
        ri = self.last_perturbation.get("robot_init")
        if ri and ri.get("ee_pos_before") is not None:
            return np.asarray(ri["ee_pos_before"], np.float32), np.asarray(ri["ee_quat_before"], np.float32)
        obs = self._last_obs
        return np.asarray(obs.ee_pos, np.float32), np.asarray(obs.ee_quat, np.float32)


def make_perturbed_env(suite: str, task_id: str | int, config: dict | str | None, image_size: int = 256,
                       **kw) -> LiberoEnv:
    """``config``: dict from ``list_configs`` (or ``{"dimension":"robot_init","init_state":N}`` /
    ``{"dimension":"robot_init","radius":0.3}``), or None/"standard" for the plain LiberoEnv."""
    if not config or config == "standard":
        if is_plus_backend():   # LIBERO-plus's benchmark tables index 2402+ tasks -> use the base bddl directly
            return LiberoPlusEnv(suite, task_id, {"dimension": "standard"}, image_size=image_size, **kw)
        return LiberoEnv(suite, task_id, image_size=image_size, **kw)
    if isinstance(config, str):
        raise ValueError(f"unknown config {config!r}")
    cfg = dict(config)
    cfg.setdefault("dimension", "robot_init")
    return LiberoPlusEnv(suite, task_id, cfg, image_size=image_size, **kw)


__all__ = ["list_configs", "make_perturbed_env", "apply_robot_init_state", "robot_init_qpos", "LiberoPlusEnv",
           "robot_init_qpos_table", "base_task_names", "is_plus_backend", "DIMENSIONS", "BASE_QPOS",
           "camera_pose_for_view", "apply_camera_view", "parse_view", "IDENTITY_VIEW"]
