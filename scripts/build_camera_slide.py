import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# task 0, LIBERO-Plus camera viewpoint 0_0_100_2_354 (camera in place, axis turned 2° sideways / 6° down), n=50 per arm, 95% Wilson CI
rows = [("stock\ncamera",             44, 50, "#a9a9a9"),
        ("camera\ntilted 6°",          7, 50, "#c8433b"),
        ("+untrained\nlayer",          3, 50, "#4b3aa8"),
        ("+sleep, camera\ndims frozen", 8, 50, "#4b3aa8"),
        ("+1 sleep\ncycle",           29, 50, "#3d7fd8")]

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z*z/n; c = p + z*z/(2*n); h = z * ((p*(1-p) + z*z/(4*n)) / n) ** 0.5
    return (c - h) / d, (c + h) / d

fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=200)
fig.patch.set_facecolor("white")
for i, (lab, k, n, col) in enumerate(rows):
    r = 100 * k / n; lo, hi = wilson(k, n)
    ax.bar(i, r, width=0.62, color=col, zorder=3)
    ax.errorbar(i, r, yerr=[[r - 100*lo], [100*hi - r]], fmt="none", ecolor="#222", elinewidth=1.8, capsize=6, capthick=1.8, zorder=4)
    ax.text(i, 100*hi + 3, f"{r:.0f}%", ha="center", va="bottom", fontsize=15, fontweight="bold", color="#111")
ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[0] for r in rows], fontsize=13, color="#333")
ax.set_ylim(0, 105); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels([f"{v}%" for v in (0, 25, 50, 75, 100)], fontsize=12, color="#555")
ax.set_ylabel("task success rate", fontsize=13, color="#333")
ax.grid(axis="y", color="#e6e6e6", zorder=0); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#cfcfcf")
ax.tick_params(axis="both", length=0)
fig.text(0.115, 0.945, "Ours", fontsize=15, fontweight="bold", color="#2f6fd1")
fig.text(0.19, 0.945, "frozen SmolVLA", fontsize=13, color="#666")
fig.subplots_adjust(left=0.11, right=0.97, top=0.88, bottom=0.17)
bb = ax.get_position()
fig.patches.append(FancyBboxPatch((bb.x0 - 0.01, bb.y0 - 0.10), bb.width + 0.02, bb.height + 0.135, boxstyle="round,pad=0.01,rounding_size=0.02",
                                  transform=fig.transFigure, fill=False, edgecolor="#3d7fd8", linewidth=1.8))
for out in ("docs/media/camera_slide.png", "docs/media/camera_slide.svg"):
    fig.savefig(out, facecolor="white")
print("saved")
