"""The committed seed sets (runner/seeds.json). Disjoint at the seed level and at the LIBERO
init-state level (LiberoEnv uses seed % 50 to pick the init state)."""
from __future__ import annotations

import json
from pathlib import Path

SEEDS_PATH = Path(__file__).with_name("seeds.json")
_SPEC = json.loads(SEEDS_PATH.read_text())
N_INIT = int(_SPEC["init_states_per_task"])
SETS = _SPEC["sets"]
BENCHMARK = _SPEC["benchmark"]


def seeds(set_name: str, n: int, rep_offset: int = 0) -> list[int]:
    """n seeds from the named set. Seeds cycle over the set's init-state range, moving to a new
    repetition (a different RNG seed for the same init state) every len(range) seeds."""
    s = SETS[set_name]
    lo, hi = s["init_state_range"]
    base, width = int(s["seed_base"]), hi - lo
    out = []
    for k in range(n):
        rep = rep_offset + k // width
        out.append(base + N_INIT * rep + lo + (k % width))
    return out


def init_state_index(seed: int) -> int:
    return seed % N_INIT


def set_of(seed: int) -> str | None:
    """Which set a seed belongs to, by its init-state index (None if it is in no set)."""
    idx = init_state_index(seed)
    for name, s in SETS.items():
        lo, hi = s["init_state_range"]
        if lo <= idx < hi:
            return name
    return None


def assert_disjoint() -> None:
    ranges = [tuple(s["init_state_range"]) for s in SETS.values()]
    for i, (a0, a1) in enumerate(ranges):
        for b0, b1 in ranges[i + 1:]:
            assert a1 <= b0 or b1 <= a0, f"overlapping init-state ranges {ranges}"


assert_disjoint()
