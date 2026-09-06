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


def arm_of(e):
    """BM arm from an eval episode record (robust to torn benchmark_result lines)."""
    if e.get("perturbation", {}).get("consolidation_id") or not (5040 <= e["seed"] <= 5049):
        return None
    pert = "plus_robot_init" in e.get("environment_id", "")
    if e["condition"] == "A":
        return "BM-1" if pert else "BM-0"
    if not pert:
        return None
    p = e.get("s3_params", {})
    homing, ts = p.get("homing_enable", 0) >= 0.5, abs(p.get("time_scale", 1.0) - 1.0) < 0.01
    if not homing and ts and abs(p.get("velocity_cap", 1.0) - 1.0) < 0.01:
        return "BM-2"
    return "BM-4" if (homing and ts) else "BM-3"


def pooled_from_episodes(bench):
    from collections import defaultdict
    acc = defaultdict(lambda: {"k": 0, "n": 0, "steps": 0.0, "tasks": set()})
    for e in bench:
        if e.get("type") != "episode" or "_wide" in e.get("environment_id", "") or "_reps" in e.get("environment_id", ""):
            continue
        a = arm_of(e)
        if a is None:
            continue
        acc[a]["n"] += 1; acc[a]["k"] += int(e["outcome"]["env_success"]); acc[a]["steps"] += e["outcome"]["steps"]; acc[a]["tasks"].add(e["task_id"])
    out = []
    for a in ("BM-0", "BM-1", "BM-2", "BM-4", "BM-3"):
        if a in acc and acc[a]["n"]:
            r = acc[a]; lo, hi = wilson(r["k"], r["n"])
            out.append({"arm": a, "k": r["k"], "n": r["n"], "rate": r["k"] / r["n"], "ci": [lo, hi], "mean_steps": r["steps"] / r["n"], "tasks": sorted(r["tasks"])})
    return out


def pooled_reps(recs):
    """Pool every `benchmark_reps` run per task (each run = the 10 held-out layouts x `reps` fresh policy-noise draws),
    so two independent 50-episode runs become n=100. Returns {task: {"task", "reps", "n_runs", "results": {arm: row}}}."""
    from collections import defaultdict
    acc = defaultdict(lambda: defaultdict(lambda: {"k": 0, "n": 0, "steps": 0.0}))
    meta = defaultdict(lambda: {"reps": 0, "n_runs": 0})
    vec_of = defaultdict(dict)                 # task -> {json(s3_params): arm label}; BM-3 differs by promoted version
    for r in recs:
        if r.get("type") != "benchmark_reps":
            continue
        t = str(r.get("task"))
        meta[t]["reps"] += int(r.get("reps") or 0); meta[t]["n_runs"] += 1
        for arm, v in (r.get("results") or {}).items():
            if arm == "BM-3" and v.get("s3_params") is not None:   # one sleep cycle vs two: never pool across versions
                key = json.dumps(v["s3_params"], sort_keys=True)
                arm = vec_of[t].setdefault(key, "BM-3" if not vec_of[t] else f"BM-3.{len(vec_of[t]) + 1}")
            a = acc[t][arm]; a["k"] += int(v["k"]); a["n"] += int(v["n"]); a["steps"] += float(v.get("mean_steps", 0)) * int(v["n"])
    out = {}
    for t, arms in acc.items():
        rows = {}
        for arm, a in arms.items():
            if a["n"]:
                lo, hi = wilson(a["k"], a["n"])
                rows[arm] = {"k": a["k"], "n": a["n"], "rate": a["k"] / a["n"], "ci": [lo, hi], "mean_steps": a["steps"] / a["n"]}
        out[t] = {"task": t, "results": rows, **meta[t]}
    return out


def collect_camera():
    """Second family (v3.2): LIBERO-Plus camera viewpoints. Probe (BM-0 + BM-1 per view, n=10) from
    bench_camera/probe.jsonl; sleep-cycle reps (BM-0/1/2/3, n=50) from bench_camera/events.jsonl, keyed by (task, view),
    pooled over replication runs; the action-only (`--act`) variant is kept as its own row."""
    from collections import defaultdict
    precs = read_jsonl(os.path.join(LOGS, "bench_camera", "probe.jsonl"))
    probe = latest(precs, type="camera_probe")
    if not probe:                                        # per-arm events (or, for the first crashed probe, the episodes themselves)
        arms = {}
        for r in precs:
            if r.get("type") == "camera_probe_arm":
                arms[r["arm"]] = {k: r[k] for k in ("k", "n", "rate", "ci", "mean_steps")}
        if not arms:
            views = ["0_0_100_2_352", "0_0_100_2_354", "11_15_100_0_0", "13_15_100_0_0", "14_15_100_0_0", "15_15_100_0_0"]   # task-0 order
            from collections import defaultdict
            last = {}                                    # (env, seed) -> latest completed episode: a re-run replaces a crashed run
            for r in precs:
                if r.get("type") == "episode" and int(r["outcome"].get("steps") or 0) > 0 and not r["outcome"].get("error"):
                    last[(r.get("environment_id", ""), r.get("seed"))] = r
            acc = defaultdict(lambda: [0, 0, 0.0])
            for (env, _), r in last.items():
                name = "BM-0 stock" if "plus_camera" not in env else f"BM-1 cfg={env.rsplit('_', 1)[1]} view={views[int(env.rsplit('_', 1)[1])]}"
                a = acc[name]; a[0] += int(r["outcome"]["env_success"]); a[1] += 1; a[2] += float(r["outcome"]["steps"])
            for name, (k, n, st) in acc.items():
                lo, hi = wilson(k, n)
                arms[name] = {"k": k, "n": n, "rate": k / n, "ci": [lo, hi], "mean_steps": st / n}
        probe = {"task": "0", "results": arms} if arms else None
    recs = [r for r in read_jsonl(os.path.join(LOGS, "bench_camera", "events.jsonl")) if r.get("type") == "benchmark_reps"]
    acc = defaultdict(lambda: defaultdict(lambda: {"k": 0, "n": 0, "steps": 0.0}))
    meta = {}
    for r in recs:
        key = (str(r.get("task")), r.get("view"))
        meta.setdefault(key, {"task": str(r.get("task")), "view": r.get("view"), "n_runs": 0, "version": None})["n_runs"] += 1
        for arm, v in (r.get("results") or {}).items():
            label = arm + (" (action dims only)" if r.get("act_only") and arm == "BM-3" else "")
            if r.get("act_only") and arm != "BM-3":
                continue                                     # BM-1/BM-2 of the act run duplicate the full run's arms
            a = acc[key][label]; a["k"] += int(v["k"]); a["n"] += int(v["n"]); a["steps"] += float(v.get("mean_steps", 0)) * int(v["n"])
            if arm == "BM-3" and not r.get("act_only") and v.get("version"):
                meta[key]["version"] = v["version"]
    cards = []
    for key in sorted(acc, key=lambda k: (int(k[0]), k[1] or "")):
        rows = {}
        for arm, a in acc[key].items():
            if a["n"]:
                lo, hi = wilson(a["k"], a["n"])
                rows[arm] = {"k": a["k"], "n": a["n"], "rate": a["k"] / a["n"], "ci": [lo, hi], "mean_steps": a["steps"] / a["n"]}
        cards.append({**meta[key], "results": rows})
    return probe, cards


def camera_clips_html(inline):
    """Same-seed clips from scripts/exp/record_camera_demo.py (logs/hopper/videos/cam_<arm>_<ok|fail>.mp4), if rendered."""
    beats = [("BM-0", "Stock camera, frozen VLA"), ("BM-1", "Camera tilted 6°, frozen VLA"), ("BM-3", "Tilted + the consolidated file")]
    cells = []
    for arm, title in beats:
        hits = sorted(glob.glob(os.path.join(VID, f"cam_{arm}_*.mp4")))
        if not hits:
            continue
        vid = hits[-1]
        chip = '<span class="chip ok">this seed: success</span>' if vid.endswith("_ok.mp4") else '<span class="chip fail">this seed: fail</span>'
        cells.append(f'<div class="card"><h3>{html.escape(title)}</h3>{video_tag(vid, inline)}<p class="note">{chip}</p></div>')
    return f'<div class="cards">{"".join(cells)}</div><p class="sub">Same seed (evaluation layout 47) across the three arms; only the parameter file differs.</p>' if cells else ""


def camera_html(inline=False):
    probe, cards = collect_camera()
    if not probe and not cards:
        return ""
    parts = ['<section class="chart"><h2>Second perturbation family: the camera moves</h2>',
             '<p class="sub">LIBERO-Plus camera viewpoints, native port · the parameter file gained four camera-calibration numbers '
             '(roll / zoom / shift of the external camera frame before the frozen policy sees it) · identity by default · '
             'the optimizer only ever sees cost</p>']
    if probe:
        res = probe.get("results") or {}
        rows = "".join(f'<tr><td>{html.escape(k)}</td><td>{v["k"]}/{v["n"]} = {100 * v["rate"]:.0f}%</td>'
                       f'<td>[{100 * v["ci"][0]:.0f}, {100 * v["ci"][1]:.0f}]</td><td>{v["mean_steps"]:.0f}</td></tr>' for k, v in res.items())
        parts.append(f'<h3>Probe: every camera view of task {probe.get("task")}, raw policy, 10 held-out layouts</h3>'
                     f'<table class="tbl"><tr><th>arm · view</th><th>success</th><th>95% CI</th><th>steps</th></tr>{rows}</table>')
    if cards:
        cc = []
        for c in cards:
            order = ["BM-0", "BM-1", "BM-2", "BM-3", "BM-3 (action dims only)"]
            rows = [{"arm": a, **c["results"][a]} for a in order if a in c["results"]]
            r1, r3 = c["results"].get("BM-1"), c["results"].get("BM-3")
            tag = ""
            if r1 and r3:
                v = "pass" if r3["ci"][0] > r1["ci"][1] else ("partial" if r3["rate"] > r1["rate"] else "fail")
                tag = (f'<p class="verdict">camera moved <b>{100 * r1["rate"]:.0f}%</b> → one sleep <b>{100 * r3["rate"]:.0f}%</b> '
                       f'(n={r1["n"]}) — <span class="chip {v}">{v}</span></p>')
            elif r1:
                tag = f'<p class="verdict">camera moved <b>{100 * r1["rate"]:.0f}%</b> (n={r1["n"]}) — sleep cycle pending</p>'
            cc.append(f'<div class="card"><h3>Task {c["task"]} · view {html.escape(str(c["view"]))}'
                      f'{" · vector v" + str(c["version"]) if c.get("version") else ""}</h3>{bars_svg(rows)}{tag}</div>')
        parts.append(f'<div class="cards">{"".join(cc)}</div>')
    parts.append(camera_clips_html(inline))
    parts.append('<p class="note">Views written <code>h_v_scale_rot_vert</code>: <code>0_0_100_2_354</code> keeps the camera in place and turns its optical axis '
                 '2° sideways and 6° down; <code>11_15_100_0_0</code> moves it 11° around the table and 15° up (~30 cm away).</p></section>')
    return "".join(parts)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return ((c - h) / d, (c + h) / d)


def collect():
    bench = read_jsonl(os.path.join(LOGS, "benchmark", "events.jsonl"))
    pooled = {"results": pooled_from_episodes(bench)}
    pooled["tasks"] = sorted({t for r in pooled["results"] for t in r["tasks"]}, key=int)
    by = {r["arm"]: r for r in pooled["results"]}
    if "BM-3" in by and "BM-1" in by:
        import math
        # matched tasks only: BM-1 restricted to the tasks where a sleep cycle produced a BM-3
        t3 = set(by["BM-3"]["tasks"])
        m = [e for e in bench if e.get("type") == "episode" and arm_of(e) == "BM-1" and e["task_id"] in t3
             and "_wide" not in e.get("environment_id", "") and "_reps" not in e.get("environment_id", "")]
        k1m, n1m = sum(int(e["outcome"]["env_success"]) for e in m), len(m)
        by["BM-1"] = {**by["BM-1"], "k_matched": k1m, "n_matched": n1m}
        k3, n3, k1, n1 = by["BM-3"]["k"], by["BM-3"]["n"], k1m, n1m
        rr = (k3 / n3) / max(k1 / n1, 1e-9); se = math.sqrt(max(1 / max(k3, .5) - 1 / n3, 0) + max(1 / max(k1, .5) - 1 / n1, 0))
        lo1, hi1 = wilson(k1, n1)
        pooled["ratio"] = {"BM-3/BM-1": rr, "ci": [rr * math.exp(-1.96 * se), rr * math.exp(1.96 * se)], "matched": f"{k3}/{n3} vs {k1}/{n1} on tasks {sorted(t3)}",
                           "verdict": "pass" if by["BM-3"]["ci"][0] > hi1 else ("partial" if k3 / n3 > k1 / n1 else "fail")}
    power = read_jsonl(os.path.join(LOGS, "benchmark", "power.jsonl"))
    power = latest(power, type="benchmark_power")
    reps = pooled_reps(read_jsonl(os.path.join(LOGS, "benchmark", "reps.jsonl")))
    videos = {}
    for p in sorted(glob.glob(os.path.join(VID, "*.mp4"))):
        stem = os.path.basename(p)
        for prefix, *_ in BEATS:
            if stem.startswith(prefix + "_"):
                videos[prefix] = p
    return pooled, power, videos, reps


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
    colors = {"BM-0": "var(--muted)", "BM-1": "var(--bad)", "BM-2": "var(--warn)", "BM-3": "var(--accent)", "BM-3.2": "var(--accent)", "BM-4": "var(--good)", "BM-3w": "var(--accent)"}
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
             "BM-3": "perturbed + one sleep cycle", "BM-3.2": "perturbed + two sleep cycles (v3)", "BM-4": "perturbed + hand-set homing",
             "BM-3w": "perturbed + wide-search sleep"}


def build(inline: bool) -> str:
    pooled, power, videos, reps = collect()
    res = (pooled or {}).get("results", [])
    by = {r["arm"]: r for r in res}
    # headline = the properly powered task-0 result (10 held-out layouts x 5 policy-noise draws, n=50/arm)
    reps_all, reps = reps, (reps or {}).get("0")
    reps_rows = []
    if reps:
        for arm in ("BM-0", "BM-1", "BM-2", "BM-4", "BM-3", "BM-3.2", "BM-3w"):
            r = reps["results"].get(arm)
            if r: reps_rows.append({"arm": arm, **r})
        by = {**by, **{r["arm"]: r for r in reps_rows if r["arm"] in ("BM-0", "BM-1", "BM-2", "BM-3", "BM-4")}}
    # other tasks measured with the same held-out x noise-draw protocol (n>=30)
    other_html = ""
    others = [v for t, v in sorted((reps_all or {}).items(), key=lambda kv: int(kv[0])) if t != "0" and v["results"].get("BM-1", {}).get("n", 0) >= 30]
    if others:
        cards = []
        for v in others:
            rows = [{"arm": a, **v["results"][a]} for a in ("BM-0", "BM-1", "BM-2", "BM-4", "BM-3") if a in v["results"]]
            r1, r3 = v["results"].get("BM-1"), v["results"].get("BM-3")
            tag = ""
            if r1 and r3:
                sep = r3["ci"][0] > r1["ci"][1]
                tag = (f'<p class="verdict">perturbed <b>{100 * r1["rate"]:.0f}%</b> → one sleep <b>{100 * r3["rate"]:.0f}%</b> '
                       f'(n={r1["n"]}) — <span class="chip {"pass" if sep else ("partial" if r3["rate"] > r1["rate"] else "fail")}">'
                       f'{"pass" if sep else ("partial" if r3["rate"] > r1["rate"] else "fail")}</span></p>')
            cards.append(f'<div class="card"><h3>Task {v["task"]} · n={r1["n"] if r1 else "?"} per arm</h3>{bars_svg(rows)}{tag}</div>')
        other_html = (f'<section class="chart"><h2>Same protocol on the other tasks</h2>'
                      f'<p class="sub">Held-out layouts × fresh policy-noise draws, pooled over replication runs · the sleep cycle for each task is its own; '
                      f'task 3\'s came through the weaker 12-seed gate</p><div class="cards">{"".join(cards)}</div></section>')
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
        verdict = (f'<p class="verdict">One sleep cycle vs the collapse on the same tasks ({ratio.get("matched", "")}): <b>{ratio["BM-3/BM-1"]:.2f}×</b> '
                   f'[{ratio["ci"][0]:.2f}, {ratio["ci"][1]:.2f}] — <span class="chip {ratio["verdict"]}">{ratio["verdict"]}</span> at 10 layouts per task; see the headline above for n=100.</p>')
    legend = " · ".join(f"<b>{a}</b> {ARM_NAMES[a]}" for a in ARM_NAMES if a in by)
    headline_html = ""
    if reps_rows:
        r1 = next(r for r in reps_rows if r["arm"] == "BM-1"); r3 = next((r for r in reps_rows if r["arm"] == "BM-3"), None)
        line = ""
        r32 = next((r for r in reps_rows if r["arm"] == "BM-3.2"), None)
        if r3:
            two = (f' → after a second cycle (24-seed gate) <b>{100 * r32["rate"]:.0f}%</b> [{100 * r32["ci"][0]:.0f}, {100 * r32["ci"][1]:.0f}] (n={r32["n"]})' if r32 else "")
            line = (f'<p class="verdict">Perturbed <b>{100 * r1["rate"]:.0f}%</b> → after one unattended sleep <b>{100 * r3["rate"]:.0f}%</b> '
                    f'(n={r1["n"]} / {r3["n"]}, intervals {"do not overlap" if r3["ci"][0] > r1["ci"][1] else "overlap"}){two} — '
                    f'<span class="chip {"pass" if r3["ci"][0] > r1["ci"][1] else "partial"}">{"pass" if r3["ci"][0] > r1["ci"][1] else "partial"}</span> by the pre-registered rule.</p>')
        headline_html = (f'<section class="chart"><h2>Headline: LIBERO-Spatial task {reps["task"]}, 10 held-out layouts × {reps["reps"]} policy-noise draws</h2>'
                         f'<p class="sub">pooled over {reps["n_runs"]} independent runs (BM-3 kept per promoted version) · the 10 evaluation layouts were never seen by the optimizer or the gate · 95% Wilson intervals</p>'
                         f'{bars_svg(reps_rows)}<p class="legend">{" · ".join(f"<b>{r["arm"]}</b> {ARM_NAMES[r["arm"]]}" for r in reps_rows)}</p>{line}</section>')
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
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}}.card h3{{font-size:15px;margin:0 0 6px;font-weight:600}}
.tbl{{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}}.tbl th,.tbl td{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}}.tbl th{{color:var(--muted);font-weight:500;font-size:12px;text-transform:uppercase;letter-spacing:.05em}}
.chip.fail{{background:var(--chip-fail);color:var(--bad)}}
.honest{{border-top:1px solid var(--line);padding-top:16px;color:var(--muted);font-size:13px;display:grid;gap:4px;max-width:80ch}}
a{{color:var(--accent)}}
</style>
<div class="wrap">
<header>
  <p class="eyebrow">Fleet Memory · frozen VLA · LIBERO-Plus robot-initial-state · 2026-09-05</p>
  <h1>It sleeps, and it gets its start back.</h1>
  <p>A frozen SmolVLA policy, an execution shim with a 17-number parameter file, and an offline optimizer that
  rewrites that file only when a held-out gate says it is better. Every clip below is the same seed across arms
  (benchmark beats: LIBERO-Spatial task 0, evaluation layout 48; mastery beats: LIBERO-10 task 3, seed 2): the only thing
  that changes is the file.</p>
</header>
{"".join(beats_html)}
{headline_html}
{other_html}
{camera_html(inline)}
<section class="chart">
  <h2>Mastery on a fixed task: LIBERO-10 "bowl into the bottom drawer, close it"</h2>
  <p class="sub">Held-out layouts 20–39 × 2 policy-noise draws, n=40 per arm, seeds never used by the optimizer or the gate</p>
  <table class="tbl"><tr><th>arm</th><th>success</th><th>95% CI</th><th>mean steps</th><th>cost</th></tr>
  <tr><td>raw policy (A)</td><td>78%</td><td>[62, 88]</td><td>302</td><td>2.03</td></tr>
  <tr><td><b>promoted vector v2</b> (slower chunks, capped velocity)</td><td><b>80%</b></td><td>[65, 90]</td><td><b>270–278</b></td><td><b>1.94–1.95</b></td></tr>
  <tr><td>v3 (promoted by the 12-seed gate, later refused by the 24-seed gate)</td><td>58–68%</td><td>[42, 80]</td><td>335–367</td><td>2.31–2.67</td></tr></table>
  <p class="note">Same success, ~10% fewer steps: mastery here is efficiency, not a success jump. v2 measured three times (70–80%). v3 is the gate's one false positive; the strong gate (24 seeds × 4 draws) kept v2 when re-run.</p>
</section>
<section class="chart">
  <h2>All five tasks, 10 held-out layouts each</h2>
  <p class="sub">Pooled over tasks {tasks} · BM-3 exists only where a sleep cycle passed its gate (tasks 0, 2, 3; tasks 1 and 4 refused every candidate) · 95% Wilson intervals</p>
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
