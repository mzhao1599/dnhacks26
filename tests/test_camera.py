"""v3.2 camera viewpoint perturbation (LIBERO-Plus port) + camera-calibration S3 dims. No simulator needed."""
import json
import dataclasses

import numpy as np
import pytest

from fleet_memory.envs.base import Obs, TaskInfo
from fleet_memory.envs.libero_plus import IDENTITY_VIEW, camera_pose_for_view, parse_view, CAM_POS_AV, CAM_QUAT_AV
from fleet_memory.execution import params as P
from fleet_memory.execution.calib import calibrate_obs, is_identity, warp_image
from fleet_memory.execution.params import S3Params
from fleet_memory.runner import consolidate as C

# (pos, quat_wxyz) computed with LIBERO-Plus's own rotate_around_y / rotate_around_z / scale_distance_from_pivot
# (libero/libero/envs/problems/libero_tabletop_manipulation.py @4976dc3), same call order + round(x, 4) as their
# _setup_camera. The six views are every Camera-Viewpoints variant of libero_spatial task 0.
REF = {
    "0_0_100_0_0": ([0.6586, 0.0, 1.6104], [0.638, 0.3048, 0.3048, 0.638]),
    "0_0_100_2_352": ([0.6586, 0.0, 1.6104], [-0.6036, -0.3439, -0.3531, -0.6266]),
    "0_0_100_2_354": ([0.6586, 0.0, 1.6104], [-0.6097, -0.333, -0.3425, -0.6325]),
    "11_15_100_0_0": ([0.4186, 0.0814, 1.7532], [0.6048, 0.197, 0.239, 0.7337]),
    "13_15_100_0_0": ([0.4155, 0.0959, 1.7532], [0.5919, 0.1928, 0.2424, 0.7441]),
    "14_15_100_0_0": ([0.4137, 0.1032, 1.7532], [0.5854, 0.1907, 0.244, 0.7493]),
    "15_15_100_0_0": ([0.4119, 0.1104, 1.7532], [0.5788, 0.1885, 0.2457, 0.7544]),
    "0_0_130_0_0": ([0.8562, 0.0, 1.8535], [0.638, 0.3048, 0.3048, 0.638]),
    "20_0_100_0_0": ([0.6189, 0.2253, 1.6104], [0.5175, 0.2473, 0.3532, 0.7391]),
}


@pytest.mark.parametrize("view", sorted(REF))
def test_camera_pose_matches_libero_plus(view):
    pos, quat = camera_pose_for_view(view)
    rp, rq = REF[view]
    assert np.allclose(pos, rp, atol=1e-4) and np.allclose(quat, rq, atol=1e-4), (view, pos, quat)


def test_identity_view_is_the_stock_camera():
    pos, quat = camera_pose_for_view(IDENTITY_VIEW)
    assert np.allclose(pos, CAM_POS_AV, atol=1e-4) and np.allclose(quat, CAM_QUAT_AV, atol=1e-4)
    assert parse_view("11_15_100_0_0") == {"h_deg": 11, "v_deg": 15, "scale": 1.0, "end_rot_deg": 0, "end_vert_deg": 0}


# ------------------------------------------------------------------ calibration warp
def _img(h=32, w=32):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_warp_identity_returns_input():
    img = _img()
    assert warp_image(img) is img
    assert is_identity() and not is_identity(roll_deg=1.0)


def test_warp_roll_90_is_rot90_ccw():
    img = _img()
    out = warp_image(img, roll_deg=90.0)
    assert np.array_equal(out, np.rot90(img, 1))       # square, centre (H-1)/2: samples land on integer pixels


def test_warp_shift_moves_content_right_and_down():
    img = np.zeros((40, 40, 3), np.uint8)
    img[10, 10] = 255
    out = warp_image(img, shift_xy=(0.25, 0.25))       # 10 px right and down
    assert out[20, 20, 0] == 255 and out[10, 10, 0] == 0


def test_warp_zoom_magnifies_about_centre():
    img = np.zeros((41, 41), np.uint8)
    img[20, 30] = 255                                   # 10 px right of the centre
    out = warp_image(img, zoom=2.0)
    assert out[20, 40] == 255 and out[20, 30] == 0      # now 20 px right of the centre


def test_calibrate_obs_only_touches_the_external_camera():
    obs = Obs(t=0, images={"agentview": _img(), "eye_in_hand": _img()}, state=np.zeros(8), ee_pos=np.zeros(3),
              ee_quat=np.array([0, 0, 0, 1.0]), gripper_qpos=np.zeros(2), object_poses={})
    same = calibrate_obs(obs, {"roll_deg": 0.0, "zoom": 1.0, "shift_xy": [0.0, 0.0]})
    assert same is obs
    out = calibrate_obs(obs, {"roll_deg": 5.0, "zoom": 1.0, "shift_xy": [0.0, 0.0]})
    assert out is not obs and out.images["eye_in_hand"] is obs.images["eye_in_hand"]
    assert not np.array_equal(out.images["agentview"], obs.images["agentview"])
    assert out.state is obs.state and out.object_poses is obs.object_poses


# ------------------------------------------------------------------ S3 vector
def test_vector_is_21_dims_and_17_dim_incumbents_load_as_identity_calibration():
    assert P.DIM == 21 and P.NAMES[-3:] == ["cam_roll_deg", "cam_zoom", "cam_shift_xy"]
    v17 = S3Params.identity().to_array()[:17].copy()
    v17[5] = 0.85; v17[10] = 1.0                        # a robot-init incumbent (time_scale + homing)
    p = S3Params.from_array(v17)
    assert p.time_scale == pytest.approx(0.85) and p.homing_on
    assert p.cam_roll_deg == 0.0 and p.cam_zoom == 1.0 and np.all(p.cam_shift_xy == 0)
    assert p.calib_dict() is None and p.to_constraint_set().image_calib is None
    d = json.loads(json.dumps(p.to_dict()))
    assert S3Params.from_dict({k: v for k, v in d.items() if not k.startswith("cam_")}).to_array().shape == (21,)
    assert np.allclose(S3Params.from_dict(d).to_array(), p.to_array())


def test_calibration_reaches_the_constraint_set_and_round_trips():
    p = S3Params.identity(); p.cam_roll_deg = -3.0; p.cam_zoom = 1.1; p.cam_shift_xy = np.array([0.05, -0.02])
    cs = p.to_constraint_set()
    assert cs.image_calib == {"roll_deg": -3.0, "zoom": 1.1, "shift_xy": [0.05, -0.02]} and not cs.is_identity()
    from fleet_memory.execution.constraints import ConstraintSet
    assert ConstraintSet.from_dict(cs.to_dict()).image_calib == cs.image_calib
    assert P.dim_indices(P.CALIB_NAMES) == [17, 18, 19, 20]
    with pytest.raises(KeyError):
        P.dim_indices(["nope"])


class _Recorder:
    """Policy stand-in that records the agentview frame it is shown."""
    name = "rec"

    def __init__(self):
        self.seen = []

    def reset(self, instruction, *a, **k):
        pass

    def act(self, obs):
        self.seen.append(obs.images["agentview"])
        return np.zeros((1, 7), np.float32)


def test_shim_applies_calibration_only_when_enabled():
    from fleet_memory.execution.detectors import Tracker
    from fleet_memory.execution.shim import ExecutionShim
    task = TaskInfo(suite="mock", task_id="t", language="pick", task_family="pick_place", objects=["bowl"], max_steps=10)
    img = _img()
    obs = Obs(t=0, images={"agentview": img, "eye_in_hand": img}, state=np.zeros(8), ee_pos=np.zeros(3),
              ee_quat=np.array([0, 0, 0, 1.0]), gripper_qpos=np.zeros(2), object_poses={"bowl": (np.ones(3), np.array([0, 0, 0, 1.0]))})
    p = S3Params.identity(); p.cam_roll_deg = 10.0
    for enabled in (True, False):
        pol = _Recorder()
        shim = ExecutionShim(pol, p.to_constraint_set(), task, Tracker(task), enabled=enabled)
        shim.act(obs); shim.act(obs)
        warped = not np.array_equal(pol.seen[0], img)
        assert warped == enabled
        assert (sum(e["kind"] == "image_calib_active" for e in shim.events) == 1) == enabled   # logged once


def test_cem_frozen_dims_never_move():
    cfg = C.ConsolidationConfig(population=12, elites=3, iterations=3, seeds_per_candidate=1, frozen_dims=list(P.CALIB_NAMES))
    theta0 = S3Params.identity().to_array()
    seen = []

    def ev(X):
        seen.append(np.asarray(X).reshape(-1, P.DIM).copy())
        return [float(np.sum((P.normalize(x) - 0.3) ** 2)) for x in X]

    res = C.cem_search(ev, theta0, cfg, np.random.default_rng(1))
    idx = P.dim_indices(P.CALIB_NAMES)
    for X in seen:
        assert np.allclose(X[:, idx], theta0[idx])
    assert np.allclose(res.theta_star[idx], theta0[idx])
    assert cfg.optimizer_dict()["frozen_dims"] == list(P.CALIB_NAMES)
    # and without the freeze the same search does move them
    free = C.cem_search(ev, theta0, dataclasses.replace(cfg, frozen_dims=[]), np.random.default_rng(1))
    assert not np.allclose(free.theta_star[idx], theta0[idx])
