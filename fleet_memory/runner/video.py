"""Episode video recording for demos. Frames = agentview | wrist cam side by side, with an overlay
(arm, step, subtask, S3 summary, final SUCCESS/FAIL hold). Written as MP4 via imageio-ffmpeg
(pip: imageio imageio-ffmpeg); falls back to a PNG sequence if ffmpeg is unavailable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FrameRecorder:
    label: str                                   # e.g. "BM-1 perturbed start"
    subtitle: str = ""                           # e.g. task language
    s3_summary: str = ""                         # e.g. "homing on · time_scale 0.85"
    fps: int = 20
    hold_frames: int = 30                        # final-state hold (1.5 s at 20 fps)
    frames: list[np.ndarray] = field(default_factory=list)
    phase: str = ""

    def on_step(self, obs, t: int, phase: str | None = None) -> None:
        if phase is not None:
            self.phase = phase
        imgs = obs.images or {}
        a = imgs.get("agentview")
        if a is None:
            return
        w = imgs.get("eye_in_hand")
        frame = np.concatenate([a, w], axis=1) if w is not None and w.shape[0] == a.shape[0] else a
        self.frames.append(self._overlay(np.ascontiguousarray(frame), f"t={t:>3}  {self.phase}"))

    def finish(self, success: bool, steps: int) -> None:
        if not self.frames:
            return
        last = self.frames[-1]
        banner = self._overlay(last.copy(), f"{'SUCCESS' if success else 'FAIL'}  ·  {steps} steps",
                               color=(40, 200, 90) if success else (230, 70, 60), big=True)
        self.frames.extend([banner] * self.hold_frames)

    # ------------------------------------------------------------------ drawing
    def _overlay(self, frame: np.ndarray, status: str, color=(255, 255, 255), big: bool = False) -> np.ndarray:
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:                       # no PIL: raw frames
            return frame
        im = Image.fromarray(frame)
        d = ImageDraw.Draw(im, "RGBA")
        h = im.height
        d.rectangle([0, 0, im.width, 34], fill=(0, 0, 0, 150))
        d.rectangle([0, h - 22, im.width, h], fill=(0, 0, 0, 150))
        font = ImageFont.load_default()
        d.text((6, 3), self.label, fill=(255, 255, 255), font=font)
        d.text((6, 17), (self.subtitle + ("   " + self.s3_summary if self.s3_summary else ""))[:90], fill=(200, 200, 200), font=font)
        if big:
            d.rectangle([0, h // 2 - 16, im.width, h // 2 + 16], fill=(0, 0, 0, 170))
            d.text((im.width // 2 - 4 * len(status), h // 2 - 6), status, fill=color, font=font)
        else:
            d.text((6, h - 18), status, fill=color, font=font)
        return np.asarray(im)

    # ------------------------------------------------------------------ output
    def write(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not self.frames:
            return ""
        try:
            import imageio.v2 as imageio
            with imageio.get_writer(path, fps=self.fps, codec="libx264", quality=7, macro_block_size=1) as w:
                for f in self.frames:
                    w.append_data(f)
            return path
        except Exception as ex:                   # no ffmpeg: PNG sequence next to the intended path
            seq = path.rsplit(".", 1)[0]
            os.makedirs(seq, exist_ok=True)
            try:
                from PIL import Image
                for i, f in enumerate(self.frames):
                    Image.fromarray(f).save(f"{seq}/{i:05d}.png")
                return seq
            except ImportError:
                np.save(seq + ".npy", np.stack(self.frames))
                return seq + ".npy"


def s3_summary(params: dict | None) -> str:
    """Short human-readable S3 vector summary for overlays."""
    if not params:
        return "raw VLA"
    bits = []
    if params.get("homing_enable", 0) >= 0.5:
        bits.append("homing on")
    ts = params.get("time_scale", 1.0)
    if abs(ts - 1.0) > 0.02:
        bits.append(f"time_scale {ts:.2f}")
    vc = params.get("velocity_cap", 1.0)
    if vc < 0.98:
        bits.append(f"vel_cap {vc:.2f}")
    if params.get("blend_alpha", 0) >= 0.05:
        bits.append(f"blend {params['blend_alpha']:.2f}")
    gz = params.get("grasp_offset_z", 0.0)
    if abs(gz) > 0.003 and params.get("blend_alpha", 0) >= 0.05:
        bits.append(f"grasp_z {gz * 100:+.1f}cm")
    return " · ".join(bits) if bits else "identity vector"
