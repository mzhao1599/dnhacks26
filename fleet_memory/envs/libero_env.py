"""LIBERO adapter implementing the Env protocol.

Mirrors lerobot 0.6.1's ``lerobot/envs/libero.py`` + ``LiberoProcessorStep`` exactly so the
SmolVLA LIBERO checkpoint sees the same observations it was evaluated with:

* OffScreenRenderEnv(bddl_file_name, camera_heights=camera_widths=image_size, control_freq=20)
* reset: env.seed(seed) -> env.reset() -> env.set_init_state(init_states[k]) -> 10 no-op steps
* images: robosuite renders 180-degree rotated; we flip both H and W so Obs.images are upright
  (this is what LiberoProcessorStep does before the policy sees them).
* state float32[8] = eef_pos(3) + quat2axisangle(eef_quat xyzw)(3) + gripper_qpos(2)
* success_flag() = env.check_success() (the BDDL goal predicate), nothing else.

MUJOCO_GL must be set BEFORE mujoco/robosuite are imported; we do that at module import
time from ``FM_MUJOCO_GL``/``MUJOCO_GL`` (default egl) and again in the constructor if a
different ``render_gl`` is requested (only effective if mujoco was not imported yet).
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np

# --- set the GL backend before any mujoco / robosuite import ---------------------------
os.environ.setdefault("MUJOCO_GL", os.environ.get("FM_MUJOCO_GL", "egl"))
if os.environ["MUJOCO_GL"] == "egl":
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from fleet_memory.envs.base import Obs, TaskInfo  # noqa: E402

# Horizons: lerobot's TASK_SUITE_MAX_STEPS uses 280 for libero_spatial; the project contract
# asks for 220 there. Everything else matches lerobot.
SUITE_MAX_STEPS: dict[str, int] = {
    "libero_10": 520,
    "libero_goal": 300,
    "libero_object": 280,
    "libero_spatial": 220,
    "libero_90": 400,
}
NUM_STEPS_WAIT = 10                     # settle steps after set_init_state (lerobot default)
DUMMY_ACTION = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
CONTROL_FREQ = 20

_FAMILY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("drawer_manipulation", ("drawer",)),
    ("appliance_toggle", ("stove", "microwave", "turn on", "turn off")),
    ("cabinet_manipulation", ("cabinet",)),
    ("stack", ("stack", "on top of")),
    ("pour", ("pour",)),
    ("place_in_basket", ("basket",)),
    ("place_on_plate", ("plate",)),
    ("place_in_bowl", ("bowl",)),
    ("pick_place_mug", ("mug",)),
    ("pick_place_pot", ("pot", "moka")),
    ("pick_place_bottle", ("bottle", "ketchup", "cream cheese", "butter")),
    ("pick_place_book", ("book",)),
    ("pick_place", ("pick", "put", "place", "move")),
]


def task_family_from_language(language: str) -> str:
    s = language.lower()
    for fam, keys in _FAMILY_RULES:
        if any(k in s for k in keys):
            return fam
    return "other"


def slugify(language: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", language.lower()).strip("_")
    return s[:120] or "task"


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Same math as lerobot LiberoProcessorStep._quat2axisangle (xyzw in, (3,) out)."""
    q = np.asarray(quat, dtype=np.float32)
    w = float(np.clip(q[3], -1.0, 1.0))
    den = float(np.sqrt(max(1.0 - w * w, 0.0)))
    if den <= 1e-10:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arccos(w)
    return (q[:3] / den * angle).astype(np.float32)


# --------------------------------------------------------------------------- #
# Benchmark helpers (importing libero pulls in robosuite/mujoco -> keep lazy)
# --------------------------------------------------------------------------- #

def _get_suite(suite: str):
    from libero.libero import benchmark

    bench = benchmark.get_benchmark_dict()
    if suite not in bench:
        raise ValueError(f"unknown LIBERO suite {suite!r}; available: {sorted(bench)}")
    return bench[suite]()


def list_tasks(suite: str) -> list[tuple[int, str, str]]:
    """[(idx, slug, language)] for every task in the suite."""
    b = _get_suite(suite)
    return [(i, slugify(t.language), t.language) for i, t in enumerate(b.tasks)]


def resolve_task_id(suite: str, task_id: str | int) -> int:
    """Accept an int index, a numeric string, a slug, or (a prefix of) the language string."""
    if isinstance(task_id, int):
        return task_id
    if task_id.isdigit():
        return int(task_id)
    tasks = list_tasks(suite)
    want = slugify(task_id)
    for i, slug, _ in tasks:
        if slug == want:
            return i
    for i, slug, _ in tasks:
        if slug.startswith(want):
            return i
    raise ValueError(f"task {task_id!r} not found in {suite}; have {[s for _, s, _ in tasks]}")


def _load_init_states(task, root: Path) -> np.ndarray:
    import torch

    p = root / task.problem_folder / task.init_states_file
    return torch.load(p, weights_only=False)  # numpy pickle (lerobot does the same)


# --------------------------------------------------------------------------- #
# Env
# --------------------------------------------------------------------------- #

class LiberoEnv:
    """One LIBERO task; one episode at a time. See module docstring for the exact protocol."""

    def __init__(
        self,
        suite: str,
        task_id: str | int,
        image_size: int = 256,
        max_steps: int | None = None,
        render_gl: str = "egl",
    ):
        if render_gl and os.environ.get("MUJOCO_GL") != render_gl:
            os.environ["MUJOCO_GL"] = render_gl  # only effective if mujoco not imported yet
            if render_gl == "egl":
                os.environ["PYOPENGL_PLATFORM"] = "egl"
        from libero.libero import get_libero_path

        self.suite = suite
        self.task_idx = resolve_task_id(suite, task_id)
        self._bench = _get_suite(suite)
        self._task = self._bench.get_task(self.task_idx)
        self.language: str = self._task.language
        self.slug: str = slugify(self.language)
        self.image_size = int(image_size)
        self.max_steps = int(max_steps) if max_steps else SUITE_MAX_STEPS.get(suite, 500)
        self._bddl = os.path.join(get_libero_path("bddl_files"), self._task.problem_folder, self._task.bddl_file)
        self._init_states = _load_init_states(self._task, Path(get_libero_path("init_states")))
        self.n_init_states = int(len(self._init_states))

        self._env = None          # libero OffScreenRenderEnv (created lazily; safe before fork)
        self._t = 0
        self._last_raw: dict[str, Any] = {}
        self._last_obs: Obs | None = None
        self._objects: list[str] | None = None
        self.last_seed: int | None = None
        self.init_state_index: int | None = None
        self.last_perturbation: dict[str, Any] = {}

    # ---- lifecycle ----------------------------------------------------------------
    @property
    def raw_env(self):
        """The libero OffScreenRenderEnv (``.env`` is the robosuite problem env)."""
        self._ensure_env()
        return self._env

    def _ensure_env(self) -> None:
        if self._env is not None:
            return
        from libero.libero.envs import OffScreenRenderEnv

        env = OffScreenRenderEnv(
            bddl_file_name=self._bddl,
            camera_heights=self.image_size,
            camera_widths=self.image_size,
            control_freq=CONTROL_FREQ,
            hard_reset=True,
        )
        env.reset()
        self._env = env

    def task_info(self) -> TaskInfo:
        return TaskInfo(
            suite=self.suite,
            task_id=self.slug,
            language=self.language,
            task_family=task_family_from_language(self.language),
            objects=self.object_names(),
            max_steps=self.max_steps,
        )

    def object_names(self) -> list[str]:
        """Scene objects, objects of interest FIRST (target object(s) first, destination/container
        last, as ordered in the BDDL ``obj_of_interest``), then the remaining objects in sim order."""
        if self._objects is None:
            self._ensure_env()
            all_names = list(self._env.env.obj_body_id.keys())
            ooi = [n for n in self._env.env.obj_of_interest if n in all_names]
            self._objects = ooi + [n for n in all_names if n not in ooi]
        return list(self._objects)

    @property
    def objects_of_interest(self) -> list[str]:
        self._ensure_env()
        return list(self._env.env.obj_of_interest)

    def envelope(self):
        """The immutable safety envelope (execution.envelope.DEFAULT_ENVELOPE); never optimised."""
        from fleet_memory.execution.envelope import DEFAULT_ENVELOPE
        return DEFAULT_ENVELOPE

    def reset(self, seed: int, perturbation: dict[str, Any] | None = None) -> Obs:
        """Deterministic reset; optional post-reset perturbation (see ``envs/perturb.py``):
        ``{"shift_xy":[dx,dy]}`` (first object of interest), ``{"jitter_m": s}`` (all objects of
        interest), ``{"distractor": true}`` (best-effort). What was actually applied is recorded in
        ``self.last_perturbation``."""
        self._ensure_env()
        self.last_seed = int(seed)
        self.init_state_index = int(seed) % self.n_init_states
        self._env.seed(int(seed))
        self._env.reset()
        raw = self._env.set_init_state(self._init_states[self.init_state_index])
        for _ in range(NUM_STEPS_WAIT):
            raw, _, _, _ = self._env.step(DUMMY_ACTION.copy())
        for robot in self._env.robots:
            robot.controller.use_delta = True
        self._t = 0
        self._objects = None
        obs = self._make_obs(raw)
        self._last_obs = obs
        self.last_perturbation = {}
        if perturbation:
            from fleet_memory.envs.perturb import apply_perturbation
            obs, self.last_perturbation = apply_perturbation(self, perturbation, default_seed=int(seed))
        return obs

    def step(self, action: np.ndarray) -> tuple[Obs, bool, dict[str, Any]]:
        assert self._env is not None, "reset() before step()"
        a = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        if a.shape != (7,):
            raise ValueError(f"action must be float32[7], got {a.shape}")
        raw, reward, done_sim, info = self._env.step(a)
        self._t += 1
        success = bool(self._env.check_success())
        horizon = self._t >= self.max_steps
        obs = self._make_obs(raw)
        self._last_obs = obs
        info = dict(info or {})
        info.update({"is_success": success, "horizon": horizon, "sim_done": bool(done_sim),
                     "reward": float(reward), "t": self._t})
        return obs, bool(success or horizon), info

    def success_flag(self) -> bool:
        if self._env is None:
            return False
        return bool(self._env.check_success())

    def success_subconditions(self) -> dict[str, bool]:
        """Per-predicate booleans of the BDDL goal conjunction, e.g. {'on(bowl_1, plate_1)': False}."""
        if self._env is None:
            return {}
        pe = self._env.env
        try:
            goal = pe.parsed_problem["goal_state"]
            out = {}
            for state in goal:
                key = f"{state[0]}({', '.join(state[1:])})"
                out[key] = bool(pe._eval_predicate(state))
            return out
        except Exception:
            return {}

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            finally:
                self._env = None

    # ---- observation ---------------------------------------------------------------
    def _object_poses(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        pe = self._env.env
        sim = pe.sim
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, bid in pe.obj_body_id.items():
            pos = np.asarray(sim.data.body_xpos[bid], dtype=np.float32).copy()
            q = np.asarray(sim.data.body_xquat[bid], dtype=np.float32)  # wxyz
            quat = np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)  # -> xyzw
            out[name] = (pos, quat)
        return out

    def _make_obs(self, raw: dict[str, Any]) -> Obs:
        self._last_raw = raw
        agent = np.ascontiguousarray(raw["agentview_image"][::-1, ::-1])       # 180° -> upright
        wrist = np.ascontiguousarray(raw["robot0_eye_in_hand_image"][::-1, ::-1])
        ee_pos = np.asarray(raw["robot0_eef_pos"], dtype=np.float32)
        ee_quat = np.asarray(raw["robot0_eef_quat"], dtype=np.float32)         # xyzw
        grip = np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32)
        state = np.concatenate([ee_pos, quat2axisangle(ee_quat), grip]).astype(np.float32)
        return Obs(
            t=self._t,
            images={"agentview": agent, "eye_in_hand": wrist},
            state=state,
            ee_pos=ee_pos,
            ee_quat=ee_quat,
            gripper_qpos=grip,
            object_poses=self._object_poses(),
            raw=raw,
        )

    def current_obs(self) -> Obs:
        """Re-read observation from the current sim state (after an external state edit)."""
        assert self._env is not None
        raw = self._env.regenerate_obs_from_state(self._env.get_sim_state())
        obs = self._make_obs(raw)
        self._last_obs = obs
        return obs


__all__ = ["LiberoEnv", "list_tasks", "resolve_task_id", "SUITE_MAX_STEPS", "task_family_from_language", "slugify"]
