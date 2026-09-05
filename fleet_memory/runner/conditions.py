"""v3 experimental arms, ablations, seed sets and the held-out suites. Pure data."""
from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass
class Condition:
    name: str
    planner: bool
    memory: bool
    inner_loop: bool
    outer_loop: bool
    surfaces: list[str]
    shim: bool
    use_incumbent_s3: bool
    generates_lessons: bool
    deferred: bool = False
    description: str = ""


ARMS: dict[str, Condition] = {
    "A": Condition("A", planner=False, memory=False, inner_loop=False, outer_loop=False, surfaces=[], shim=False,
                   use_incumbent_s3=False, generates_lessons=False, description="raw VLA + safety envelope only"),
    "B": Condition("B", planner=False, memory=False, inner_loop=False, outer_loop=False, surfaces=[], shim=True,
                   use_incumbent_s3=True, generates_lessons=False,
                   description="A + sleep-optimised incumbent S3 vector (no planner, no coach)"),
    "C": Condition("C", planner=True, memory=True, inner_loop=True, outer_loop=True, surfaces=["S1"], shim=True,
                   use_incumbent_s3=False, generates_lessons=True,
                   description="A + planner + S1 memory + inner coach (shim for S1 abort/rebind only)"),
    "D": Condition("D", planner=True, memory=True, inner_loop=True, outer_loop=True, surfaces=["S1", "S3"], shim=True,
                   use_incumbent_s3=True, generates_lessons=True, description="B + C: the full system"),
    "P": Condition("P", planner=True, memory=True, inner_loop=True, outer_loop=True, surfaces=["S1", "S3"], shim=True,
                   use_incumbent_s3=True, generates_lessons=True,
                   description="perturbation protocol (baseline -> perturb -> recovery), driven by pool.py --protocol P"),
    "E": Condition("E", planner=True, memory=True, inner_loop=True, outer_loop=True, surfaces=["S1", "S3"], shim=True,
                   use_incumbent_s3=True, generates_lessons=True, deferred=True,
                   description="D with random retrieval (deferred)"),
    "F": Condition("F", planner=False, memory=False, inner_loop=False, outer_loop=False, surfaces=[], shim=False,
                   use_incumbent_s3=False, generates_lessons=False, deferred=True,
                   description="LoRA fine-tune baseline (deferred)"),
}

ABLATIONS: dict[str, Condition] = {
    "D_S1": replace(ARMS["D"], name="D_S1", surfaces=["S1"], use_incumbent_s3=False,
                    description="D restricted to the S1 surface"),
    "D_S3": replace(ARMS["D"], name="D_S3", surfaces=["S3"], description="D restricted to the S3 surface"),
}

SEED_OFFSETS: dict[str, int] = {"train": 0, "heldout": 1000, "probe": 2000, "opt": 3000, "gate": 4000}
HELD_OUT_SUITES: set[str] = {"libero_goal"}


def get_arm(name: str) -> Condition:
    c = ARMS.get(name) or ABLATIONS.get(name)
    if c is None:
        raise KeyError(f"unknown arm {name!r} (want one of {sorted(ARMS) + sorted(ABLATIONS)})")
    return c


def seed_set(name: str, n: int) -> list[int]:
    """Disjoint seed ranges: train 0.., heldout 1000.., probe 2000.., opt 3000.., gate 4000.."""
    if name not in SEED_OFFSETS:
        raise KeyError(f"unknown seed set {name!r} (want one of {sorted(SEED_OFFSETS)})")
    base = SEED_OFFSETS[name]
    return [base + i for i in range(int(n))]


def is_held_out(suite: str) -> bool:
    return suite in HELD_OUT_SUITES


__all__ = ["Condition", "ARMS", "ABLATIONS", "SEED_OFFSETS", "HELD_OUT_SUITES", "get_arm", "seed_set", "is_held_out"]
