"""docs/media/headline.svg — the n=100 task-0 benchmark chart for the README (standalone SVG, fixed colours,
no CSS variables so GitHub renders it in both themes). Data: logs/hopper/benchmark/reps.jsonl pooled per task
exactly as scripts/build_demo_page.py does. The GIF next to it is built with ffmpeg (see README "Demo assets")."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from build_demo_page import ARM_NAMES, pooled_reps, read_jsonl  # noqa: E402

COL = {"BM-0": "#8a939e", "BM-1": "#c73e3e", "BM-2": "#c98a12", "BM-4": "#1f8a4c", "BM-3": "#3b5bdb"}
LABEL = {"BM-0": "standard start", "BM-1": "perturbed start", "BM-2": "+ untrained layer",
         "BM-4": "+ hand-set homing", "BM-3": "+ one sleep cycle"}


def chart(rows, title, sub, path):
    W, H, L, R, T, B = 720, 330, 56, 20, 64, 70
    ph = H - T - B
    n = len(rows)
    bw = (W - L - R) / n
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif">',
           f'<rect width="{W}" height="{H}" rx="10" fill="#ffffff" stroke="#e1e4e8"/>',
           f'<text x="{L}" y="26" font-size="16" font-weight="600" fill="#141a22">{title}</text>',
           f'<text x="{L}" y="46" font-size="12" fill="#5c6670">{sub}</text>']
    for y in (0, 25, 50, 75, 100):
        yy = T + ph - ph * y / 100
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="#eceef1"/>'
                   f'<text x="{L - 8}" y="{yy + 4:.1f}" font-size="11" fill="#8a939e" text-anchor="end">{y}%</text>')
    for i, r in enumerate(rows):
        x = L + i * bw + bw * 0.22
        w = bw * 0.56
        rate = 100 * r["rate"]
        y = T + ph - ph * rate / 100
        lo, hi = 100 * r["ci"][0], 100 * r["ci"][1]
        ylo, yhi = T + ph - ph * lo / 100, T + ph - ph * hi / 100
        cx = x + w / 2
        out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{T + ph - y:.1f}" rx="3" fill="{COL[r["arm"]]}"/>')
        out.append(f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{ylo:.1f}" y2="{yhi:.1f}" stroke="#141a22" stroke-width="1.5"/>')
        out.append(f'<line x1="{cx - 5:.1f}" x2="{cx + 5:.1f}" y1="{ylo:.1f}" y2="{ylo:.1f}" stroke="#141a22" stroke-width="1.5"/>'
                   f'<line x1="{cx - 5:.1f}" x2="{cx + 5:.1f}" y1="{yhi:.1f}" y2="{yhi:.1f}" stroke="#141a22" stroke-width="1.5"/>')
        out.append(f'<text x="{cx:.1f}" y="{yhi - 8:.1f}" font-size="13" font-weight="600" fill="#141a22" text-anchor="middle">{rate:.0f}%</text>')
        out.append(f'<text x="{cx:.1f}" y="{T + ph + 18}" font-size="12" font-weight="600" fill="#141a22" text-anchor="middle">{r["arm"]}</text>')
        out.append(f'<text x="{cx:.1f}" y="{T + ph + 34}" font-size="11" fill="#5c6670" text-anchor="middle">{LABEL[r["arm"]]}</text>')
        out.append(f'<text x="{cx:.1f}" y="{T + ph + 50}" font-size="11" fill="#8a939e" text-anchor="middle">{r["k"]}/{r["n"]}</text>')
    out.append("</svg>")
    open(path, "w").write("\n".join(out))
    return path


if __name__ == "__main__":
    reps = pooled_reps(read_jsonl(os.path.join(ROOT, "logs", "hopper", "benchmark", "reps.jsonl")))
    t0 = reps["0"]
    rows = [{"arm": a, **t0["results"][a]} for a in ("BM-0", "BM-1", "BM-2", "BM-4", "BM-3") if a in t0["results"]]
    os.makedirs(os.path.join(ROOT, "docs", "media"), exist_ok=True)
    p = chart(rows, "LIBERO-Spatial task 0 under LIBERO-Plus robot-init perturbation (frozen SmolVLA)",
              f"10 held-out layouts × {t0['reps']} policy-noise draws, {t0['n_runs']} independent runs · bars = success, whiskers = 95% Wilson CI",
              os.path.join(ROOT, "docs", "media", "headline.svg"))
    print(p, [(r["arm"], r["k"], r["n"]) for r in rows])
