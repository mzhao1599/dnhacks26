"""Static plots over the event log. matplotlib is optional (the `sim` extra); each function
returns the written path, or None with a warning when matplotlib is not installed.

CLI: python -m fleet_memory.analysis.plots --log logs/events.jsonl --out-dir logs
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # laptop without the sim extra
    plt = None

from fleet_memory.analysis import metrics as M

# Fixed hue per arm (never cycled). Ablations share D's hue; linestyle carries their identity.
ARM_COLOURS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100", "E": "#e87ba4", "F": "#008300"}
ABLATION_STYLES = {"D_S1": "--", "D_S2": ":", "D_S3": "-."}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def _ready(out: str) -> bool:
    if plt is None:
        warnings.warn("matplotlib not installed; skipping plot")
        return False
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    return True


def _chrome(ax, ylabel: str) -> None:
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_success_curves(store, out: str = "logs/success_curves.png", window: int = 20) -> str | None:
    """Rolling success rate per arm over that arm's episodes (log order)."""
    if not _ready(out):
        return None
    sba = M.success_by_arm(store)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for arm in sba:
        pts = M.success_curve(store, arm, window)
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=ARM_COLOURS.get(arm, ARM_COLOURS["D"]), linestyle=ABLATION_STYLES.get(arm, "-"),
                linewidth=2, label=f"{arm} (n={sba[arm]['n']})")
        ax.annotate(arm, (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8, va="center", color=INK)
    ax.set_xlabel("episode", color=MUTED)
    _chrome(ax, f"success rate (rolling {window})")
    if sba:  # legend below the axis so it never covers the end labels
        ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=min(len(sba), 5))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_arms(store, out: str = "logs/arms.png") -> str | None:
    """Success rate per arm with 95% Wilson CI. One series -> one colour."""
    if not _ready(out):
        return None
    sba = M.success_by_arm(store)
    arms = list(sba)
    rates = [sba[a]["rate"] for a in arms]
    lo = [r - sba[a]["ci"][0] for a, r in zip(arms, rates)]
    hi = [sba[a]["ci"][1] - r for a, r in zip(arms, rates)]
    fig, ax = plt.subplots(figsize=(max(4, 0.9 * len(arms) + 2), 4))
    if arms:
        ax.bar(arms, rates, color=ARM_COLOURS["A"], width=0.6, yerr=[lo, hi], capsize=3, ecolor=MUTED, linewidth=0)
        for i, (a, r) in enumerate(zip(arms, rates)):
            ax.text(i, min(0.97, sba[a]["ci"][1] + 0.02), f"{100 * r:.0f}%", ha="center", fontsize=8, color=INK)
    _chrome(ax, "success rate (95% Wilson CI)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render success curves and per-arm bars from an events.jsonl")
    ap.add_argument("--log", default="logs/events.jsonl")
    ap.add_argument("--out-dir", default="logs")
    args = ap.parse_args(argv)
    store = M.load_store(args.log)
    for fn, name in ((plot_success_curves, "success_curves.png"), (plot_arms, "arms.png")):
        path = fn(store, os.path.join(args.out_dir, name))
        print(path or f"skipped {name} (matplotlib missing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
