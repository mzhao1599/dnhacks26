"""Phase-1 probes: the hand-written S3 edits and S2 instruction paraphrases used to measure
whether each control surface actually moves the frozen policy before any coach is involved."""
from __future__ import annotations

import re
from typing import Callable

from fleet_memory.memory.schema import Edit

# Exactly five S3 edits, each applicable via constraints.apply_edit.
S3_PROBE_EDITS: list[Edit] = [
    Edit("set_velocity_cap", {"max_pos_delta": 0.5, "max_rot_delta": 0.5}),
    Edit("set_grasp_offset", {"dx": 0.0, "dy": 0.0, "dz": 0.015}),
    Edit("set_grasp_offset", {"dx": 0.0, "dy": 0.0, "dz": -0.015}),
    Edit("insert_pre_grasp_waypoint", {"dx": 0.0, "dy": 0.0, "dz": 0.06, "tol_m": 0.015}),
    Edit("set_approach_vector", {"vector": [0.0, 0.0, -1.0], "half_angle_deg": 30.0}),
]


def _lower_nopunct(x: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", x)).strip().lower()


# S2 paraphrase templates, variant name -> function of the canonical instruction.
S2_TEMPLATES: dict[str, Callable[[str], str]] = {
    "verbatim": lambda x: x,
    "polite": lambda x: "please " + x,
    "careful": lambda x: x + " carefully",
    "detail": lambda x: x + ", grasping it by the handle",
    "lower_nopunct": _lower_nopunct,
}

# canonical instruction -> its paraphrase list; filled by paraphrases() so callers can inspect the vocab.
S2_PARAPHRASES: dict[str, list[str]] = {}


def paraphrases(canonical: str) -> list[str]:
    """Up to 5 variants of `canonical` (verbatim first); duplicates (e.g. an already lowercase,
    punctuation-free instruction) are dropped so the vocabulary has no repeated strings."""
    if canonical not in S2_PARAPHRASES:
        out: list[str] = []
        for f in S2_TEMPLATES.values():
            s = f(canonical)
            if s not in out:
                out.append(s)
        S2_PARAPHRASES[canonical] = out
    return list(S2_PARAPHRASES[canonical])
