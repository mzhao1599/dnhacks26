"""Build the demo storyboard: dashboard/demo.html (local, videos by relative path) and
dashboard/demo_artifact.html (videos inlined as data URIs for publishing). Numbers come from the
pooled benchmark_result / power / mastery events in logs/hopper; beats come from logs/hopper/videos.

    python scripts/build_demo_page.py            # after scripts/pull_logs.sh
"""
from __future__ import annotations

import base64
import glob
import html
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs", "hopper")
VID = os.path.join(LOGS, "videos")


def read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path):
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def latest(recs, **match):
    hits = [r for r in recs if all(r.get(k) == v for k, v in match.items())]
    return hits[-1] if hits else None


BEATS = [  # (video stem prefix, eyebrow, title, claim)
    ("BM-0", "Beat 1 · the baseline", "Same scene, standard start",
     "Frozen SmolVLA, untouched. It knows this task."),
    ("BM-1", "Beat 2 · the collapse", "Shift the arm's starting joints by 0.1 rad",
     "LIBERO-Plus's robot-initial-state perturbation. Nothing else changes and the policy falls apart."),
    ("BM-4", "Beat 3 · the mechanism", "Home the arm before the policy acts",
     "One S3 parameter: a scripted controller returns the end-effector to the start pose it was trained from."),
    ("BM-3", "Beat 4 · found by sleeping", "The optimizer picked homing on its own",
     "One unattended CEM cycle on layouts the evaluation never sees; promoted through the gate; nobody typed a number."),
    ("MASTERY-A", "Beat 5 · mastery, before", "LIBERO-10: bowl into the drawer, close it",
     "Raw policy on the harder long-horizon task."),
    ("MASTERY-B", "Beat 5 · mastery, after", "Same seed, promoted vector v2",
     "Slower chunks, capped velocity: fewer steps at the same success, on held-out layouts."),
]


def collect():
    bench = read_jsonl(os.path.join(LOGS, "benchmark", "events.jsonl"))
    pooled = latest(bench, type="benchmark_result", aggregated=True) or latest(bench, type="benchmark_result")
    power = read_jsonl(os.path.join(LOGS, "benchmark", "power.jsonl"))
    power = latest(power, type="benchmark_power")
    videos = {}
    for p in sorted(glob.glob(os.path.join(VID, "*.mp4"))):
        stem = os.path.basename(p)
        for prefix, *_ in BEATS:
            if stem.startswith(prefix + "_"):
                videos[prefix] = p
    return pooled, power, videos


def video_tag(path, inline):
    if not path:
        return '<div class="novideo">clip not rendered yet</div>'
    if inline:
        data = base64.b64encode(open(path, "rb").read()).decode("ascii")
        src = f"data:video/mp4;base64,{data}"
    else:
        src = "../logs/hopper/videos/" + os.path.basename(path)
    return f'<video controls muted playsinline preload="metadata" src="{src}"></video>'


def bars_svg(results):
    if not results:
        return ""
    W, H, pad = 640, 220, 36
    n = len(results)
    bw = (W - 2 * pad) / n
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="success rate per arm with 95% confidence intervals">']
    for y in (0, 25, 50, 75, 100):
        yy = H - pad - (H - 2 * pad) * y / 100
        parts.append(f'<line x1="{pad}" x2="{W - pad}" y1="{yy:.1f}" y2="{yy:.1f}" class="grid"/>'
                     f'<text x="{pad - 6}" y="{yy + 4:.1f}" class="tick" text-anchor="end">{y}%</text>')
    colors = {"BM-0": "var(--muted)", "BM-1": "var(--bad)", "BM-2": "var(--warn)", "BM-3": "var(--accent)", "BM-4": "var(--good)"}
    for i, r in enumerate(results):
        x = pad + i * bw + bw * 0.2
        rate = 100 * r["rate"]
        y = H - pad - (H - 2 * pad) * rate / 100
        lo, hi = (100 * r["ci"][0], 100 * r["ci"][1])
        ylo, yhi = (H - pad - (H - 2 * pad) * lo / 100, H - pad - (H - 2 * pad) * hi / 100)
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw * 0.6:.1f}" height="{H - pad - y:.1f}" fill="{colors.get(r["arm"], "var(--accent)")}" rx="2"/>')
        cx = x + bw * 0.3
        parts.append(f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{ylo:.1f}" y2="{yhi:.1f}" class="ci"/>')
        parts.append(f'<text x="{cx:.1f}" y="{yhi - 6:.1f}" class="lab" text-anchor="middle">{r["k"]}/{r["n"]}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{H - pad + 16}" class="tick" text-anchor="middle">{r["arm"]}</text>')
    parts.append("</svg>")
    return "".join(parts)


ARM_NAMES = {"BM-0": "standard start", "BM-1": "perturbed start", "BM-2": "perturbed + untrained layer",
             "BM-3": "perturbed + one sleep cycle", "BM-4": "perturbed + hand-set homing"}


def build(inline: bool) -> str:
    pooled, power, videos = collect()
    res = (pooled or {}).get("results", [])
    by = {r["arm"]: r for r in res}
    ratio = (pooled or {}).get("ratio")
    tasks = ", ".join(str(t) for t in (pooled or {}).get("tasks", []))
    beats_html = []
    for prefix, eyebrow, title, claim in BEATS:
        arm = prefix if prefix in by else None
        stat = ""
        if arm:
            r = by[arm]
            stat = (f'<div class="stat"><span class="num">{100 * r["rate"]:.0f}%</span><span class="unit">success · {r["k"]}/{r["n"]} · '
                    f'CI [{100 * r["ci"][0]:.0f}, {100 * r["ci"][1]:.0f}]</span></div>'
                    f'<div class="stat"><span class="num">{r["mean_steps"]:.0f}</span><span class="unit">mean steps</span></div>')
        vid = videos.get(prefix)
        outcome = ""
        if vid:
            outcome = '<span class="chip ok">this seed: success</span>' if vid.endswith("_ok.mp4") else '<span class="chip fail">this seed: fail</span>'
        beats_html.append(f'''
<section class="beat">
  <div class="media">{video_tag(vid, inline)}</div>
  <div class="text">
    <p class="eyebrow">{html.escape(eyebrow)}</p>
    <h2>{html.escape(title)}</h2>
    <p class="claim">{html.escape(claim)}</p>
    {outcome}
    <div class="stats">{stat}</div>
  </div>
</section>''')
    power_html = ""
    if power:
        r1, r4 = power["results"].get("BM-1", {}), power["results"].get("BM-4", {})
        if r1 and r4:
            power_html = (f'<p class="note">Homing alone, all {power["n_init_states"]} layouts of tasks {", ".join(power["tasks"])} '
                          f'(n={r1["n"]} per arm, no optimizer involved): perturbed <b>{100 * r1["k"] / r1["n"]:.0f}%</b> → '
                          f'homing <b>{100 * r4["k"] / r4["n"]:.0f}%</b>.</p>')
    verdict = ""
    if ratio:
        verdict = (f'<p class="verdict">One sleep cycle vs the collapse: <b>{ratio["BM-3/BM-1"]:.2f}×</b> '
                   f'[{ratio["ci"][0]:.2f}, {ratio["ci"][1]:.2f}] — <span class="chip {ratio["verdict"]}">{ratio["verdict"]}</span> by the pre-registered rule.</p>')
    legend = " · ".join(f"<b>{a}</b> {ARM_NAMES[a]}" for a in ARM_NAMES if a in by)
    return f'''<title>Fleet Memory Demo</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{color-scheme:light dark;--paper:#f3f4f6;--card:#ffffff;--ink:#141a22;--ink2:#465060;--muted:#7d8794;--line:#d9dde3;
  --accent:#3b5bdb;--good:#1f8a4c;--bad:#c73e3e;--warn:#c98a12;--chip-ok:#e3f4ea;--chip-fail:#fbe6e6}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--paper:#0f1216;--card:#171b21;--ink:#eef1f4;--ink2:#b8c0ca;--muted:#7f8994;--line:#2a3039;
  --accent:#7c93f0;--good:#4cc27a;--bad:#ef6b6b;--warn:#e0a83a;--chip-ok:#173324;--chip-fail:#3a1d1d}}}}
:root[data-theme="dark"]{{--paper:#0f1216;--card:#171b21;--ink:#eef1f4;--ink2:#b8c0ca;--muted:#7f8994;--line:#2a3039;
  --accent:#7c93f0;--good:#4cc27a;--bad:#ef6b6b;--warn:#e0a83a;--chip-ok:#173324;--chip-fail:#3a1d1d}}
body{{background:var(--paper);color:var(--ink);font:16px/1.5 "IBM Plex Sans",system-ui,sans-serif;margin:0}}
.wrap{{max-width:1080px;margin:0 auto;padding:40px 24px 64px;display:grid;gap:40px}}
header h1{{font-size:34px;line-height:1.15;margin:0 0 10px;font-weight:600;letter-spacing:-.01em;text-wrap:balance}}
header p{{max-width:64ch;color:var(--ink2);margin:0}}
.eyebrow{{font:500 12px/1 "IBM Plex Mono",ui-monospace,monospace;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);margin:0 0 8px}}
.beat{{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:24px;align-items:start;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px}}
@media (max-width:760px){{.beat{{grid-template-columns:1fr}}}}
.media video{{width:100%;aspect-ratio:2/1;background:#000;border-radius:6px;display:block}}
.novideo{{aspect-ratio:2/1;display:grid;place-items:center;border:1px dashed var(--line);border-radius:6px;color:var(--muted);font-size:14px}}
.beat h2{{font-size:22px;margin:0 0 8px;font-weight:600;text-wrap:balance}}
.claim{{color:var(--ink2);margin:0 0 12px;max-width:44ch}}
.stats{{display:flex;gap:28px;flex-wrap:wrap;margin-top:12px}}
.stat{{display:grid}}.num{{font:500 30px/1.1 "IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}}.unit{{font-size:12px;color:var(--muted)}}
.chip{{display:inline-block;font:500 12px/1 "IBM Plex Mono",monospace;padding:5px 9px;border-radius:999px;border:1px solid var(--line)}}
.chip.ok,.chip.pass{{background:var(--chip-ok);color:var(--good);border-color:transparent}}.chip.fail{{background:var(--chip-fail);color:var(--bad);border-color:transparent}}
.chip.partial{{color:var(--warn)}}
.chart{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px}}
.chart h2{{font-size:20px;margin:0 0 4px}}.chart .sub{{color:var(--muted);font-size:13px;margin:0 0 12px}}
svg{{width:100%;height:auto;max-width:720px;display:block}}.grid{{stroke:var(--line);stroke-width:1}}.ci{{stroke:var(--ink);stroke-width:1.5}}
.tick{{fill:var(--muted);font:11px "IBM Plex Mono",monospace}}.lab{{fill:var(--ink);font:500 12px "IBM Plex Mono",monospace}}
.legend{{font-size:13px;color:var(--ink2);margin:10px 0 0}}.note{{margin:12px 0 0;color:var(--ink2)}}.verdict{{margin:12px 0 0;font-size:17px}}
.honest{{border-top:1px solid var(--line);padding-top:16px;color:var(--muted);font-size:13px;display:grid;gap:4px;max-width:80ch}}
a{{color:var(--accent)}}
</style>
<div class="wrap">
<header>
  <p class="eyebrow">Fleet Memory · frozen VLA · LIBERO-Plus robot-initial-state · 2026-09-05</p>
  <h1>It sleeps, and it gets its start back.</h1>
  <p>A frozen SmolVLA policy, an execution shim with a 17-number parameter file, and an offline optimizer that
  rewrites that file only when a held-out gate says it is better. Every clip below is the same seed across arms:
  the only thing that changes is the file.</p>
</header>
{"".join(beats_html)}
<section class="chart">
  <h2>Pooled result on LIBERO-Spatial tasks {tasks or "—"}</h2>
  <p class="sub">10 evaluation layouts per task, never seen by the optimizer or the gate · 95% Wilson intervals · SmolVLA, our own base numbers</p>
  {bars_svg(res)}
  <p class="legend">{legend}</p>
  {verdict}
  {power_html}
</section>
<div class="honest">
  <span>Base numbers are ours (SmolVLA) on LIBERO-Plus's exact robot-init perturbation; the CVPR table is π₀/OpenVLA.</span>
  <span>The shim reads object pose from simulator state as a stand-in for a detector. Homing runs in end-effector space.</span>
  <span>Frozen VLA in every arm, zero demonstrations. The only thing that changed between the collapse and the recovery is one versioned, gated parameter file.</span>
</div>
</div>
'''


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "dashboard"), exist_ok=True)
    local = build(inline=False)
    open(os.path.join(ROOT, "dashboard", "demo.html"), "w").write("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">" + local.replace("<div class=\"wrap\">", "</head><body><div class=\"wrap\">", 1) + "</body></html>")
    art = build(inline=True)
    open(os.path.join(ROOT, "dashboard", "demo_artifact.html"), "w").write(art)
    print("demo.html", len(local), "bytes; demo_artifact.html", len(art), "bytes")
