"""v3.2 camera calibration: a similarity warp (roll / zoom / shift) of the external camera frame, applied by the
execution shim to the policy's copy of ``obs.images["agentview"]`` only. Pure numpy (bilinear, edge-replicate).

Convention (image as displayed, row 0 at the top): ``roll_deg`` > 0 turns the content counter-clockwise,
``zoom`` > 1 magnifies about the centre, ``shift_xy`` = (dx, dy) moves the content right / down by that fraction
of the width / height. The identity (0, 1, (0, 0)) returns the input array untouched.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

from fleet_memory.envs.base import Obs

CALIB_CAMERA = "agentview"       # the external camera; the wrist camera is never warped
_GRID_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def is_identity(roll_deg: float = 0.0, zoom: float = 1.0, shift_xy=(0.0, 0.0), eps: float = 1e-9) -> bool:
    dx, dy = (float(shift_xy[0]), float(shift_xy[1])) if shift_xy is not None else (0.0, 0.0)
    return abs(float(roll_deg)) < eps and abs(float(zoom) - 1.0) < eps and abs(dx) < eps and abs(dy) < eps


def _source_grid(h: int, w: int, roll_deg: float, zoom: float, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """(sy, sx) float source coordinates for every output pixel (inverse mapping), cached per (h, w, params)."""
    key = (h, w, round(roll_deg, 4), round(zoom, 5), round(dx, 5), round(dy, 5))
    hit = _GRID_CACHE.get(key)
    if hit is not None:
        return hit
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    ox, oy = xs - cx - dx * w, ys - cy - dy * h          # output offset from the (shifted) centre
    th = math.radians(-float(roll_deg))                  # inverse rotation; negative = ccw as displayed
    c, s = math.cos(th), math.sin(th)
    sx = (c * ox + s * oy) / float(zoom) + cx
    sy = (-s * ox + c * oy) / float(zoom) + cy
    if len(_GRID_CACHE) > 64:
        _GRID_CACHE.clear()
    _GRID_CACHE[key] = (sy, sx)
    return sy, sx


def warp_image(img: np.ndarray, roll_deg: float = 0.0, zoom: float = 1.0, shift_xy=(0.0, 0.0)) -> np.ndarray:
    """Similarity-warp an HxWxC (or HxW) uint8/float image; bilinear sampling, edge pixels replicated."""
    if is_identity(roll_deg, zoom, shift_xy):
        return img
    zoom = float(zoom)
    if not (zoom > 0):
        raise ValueError(f"zoom must be > 0, got {zoom}")
    dx, dy = float(shift_xy[0]), float(shift_xy[1])
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    sy, sx = _source_grid(h, w, float(roll_deg), zoom, dx, dy)
    sx, sy = np.round(sx, 9), np.round(sy, 9)            # kill 1e-16 residue from exact-angle trig before floor()
    fx0, fy0 = np.floor(sx), np.floor(sy)
    fx = (sx - fx0)[..., None] if arr.ndim == 3 else sx - fx0
    fy = (sy - fy0)[..., None] if arr.ndim == 3 else sy - fy0
    ix, iy = fx0.astype(np.int64), fy0.astype(np.int64)
    x0, x1 = np.clip(ix, 0, w - 1), np.clip(ix + 1, 0, w - 1)   # clip both neighbours of the UNclipped floor:
    y0, y1 = np.clip(iy, 0, h - 1), np.clip(iy + 1, 0, h - 1)   # outside the frame both collapse onto the edge pixel
    a = arr.astype(np.float32)
    top = a[y0, x0] * (1 - fx) + a[y0, x1] * fx
    bot = a[y1, x0] * (1 - fx) + a[y1, x1] * fx
    out = top * (1 - fy) + bot * fy
    if np.issubdtype(arr.dtype, np.integer):
        return np.clip(np.rint(out), 0, 255).astype(arr.dtype)
    return out.astype(arr.dtype)


def calibrate_obs(obs: Obs, calib: dict[str, Any] | None) -> Obs:
    """A copy of ``obs`` whose external-camera image is warped by ``calib`` ({roll_deg, zoom, shift_xy}).
    Everything else (state, poses, raw) is shared with the original; identity calibration returns ``obs``."""
    if not calib:
        return obs
    roll, zoom = float(calib.get("roll_deg", 0.0)), float(calib.get("zoom", 1.0))
    shift = calib.get("shift_xy") or (0.0, 0.0)
    if is_identity(roll, zoom, shift) or CALIB_CAMERA not in obs.images:
        return obs
    images = dict(obs.images)
    images[CALIB_CAMERA] = warp_image(images[CALIB_CAMERA], roll, zoom, shift)
    return dataclasses.replace(obs, images=images)


__all__ = ["warp_image", "calibrate_obs", "is_identity", "CALIB_CAMERA"]
