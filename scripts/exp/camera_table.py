"""Harvest the camera family from logs/hopper/bench_camera/{probe,events}.jsonl into the RESULTS.md §8 table rows
(markdown) + a JSON dump. Pure reader; run after scripts/pull_logs.sh.

    python scripts/exp/camera_table.py [logs/hopper/bench_camera]
"""
import json, os, sys
from collections import defaultdict


def read(p):
    if not os.path.exists(p):
        return []
    out = []
    for l in open(p):
        try: out.append(json.loads(l))
        except Exception: pass
    return out


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return ((c - h) / d, (c + h) / d)


def cell(r):
    if not r: return "PENDING"
    lo, hi = wilson(r["k"], r["n"])
    return f'{r["k"]}/{r["n"]} = **{100 * r["rate"]:.0f}%** [{100 * lo:.0f}, {100 * hi:.0f}], {r["mean_steps"]:.0f} steps'


def main(d="logs/hopper/bench_camera"):
    ev = read(os.path.join(d, "events.jsonl"))
    cons = {}
    for r in ev:
        if r.get("type") == "consolidation" and r.get("phase") == "end":
            cons.setdefault(r["skill_instance_id"], []).append(r)
    reps = [r for r in ev if r.get("type") == "benchmark_reps"]
    pooled = defaultdict(lambda: defaultdict(lambda: {"k": 0, "n": 0, "steps": 0.0, "version": None, "s3": None}))
    for r in reps:
        key = (str(r["task"]), r["view"], bool(r.get("act_only")))
        for arm, v in r["results"].items():
            a = pooled[key][arm]; a["k"] += v["k"]; a["n"] += v["n"]; a["steps"] += v["mean_steps"] * v["n"]
            a["version"] = v.get("version") or a["version"]; a["s3"] = v.get("s3_params") or a["s3"]
    rows, dump = [], {}
    for (task, view, act), arms in sorted(pooled.items()):
        res = {a: {"k": v["k"], "n": v["n"], "rate": v["k"] / v["n"], "mean_steps": v["steps"] / v["n"]} for a, v in arms.items() if v["n"]}
        si = f"si_{task}__libero_spatial_{task}_plus_camera_{view}" + ("_act" if act else "")
        g = (cons.get(si) or [{}])[-1]
        gate = g.get("gate") or {}
        gtxt = ("PENDING" if not g else "no candidate ≠ incumbent" if not gate else
                f'{"passed" if gate["passed"] else "refused"}: cost {gate["incumbent_cost"]:.2f} → {gate["candidate_cost"]:.2f}, '
                f'success {100 * gate["incumbent_success"]:.0f}% → {100 * gate["candidate_success"]:.0f}% on {gate["n_seeds"]} gate layouts')
        vec = arms.get("BM-3", {}).get("s3") or {}
        vtxt = ("PENDING" if not vec else ", ".join(f"{k}={round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v]}" for k, v in vec.items()
                if (isinstance(v, float) and abs(v - {"pregrasp_height": 0.05, "time_scale": 1.0, "velocity_cap": 1.0, "gripper_cmd": 1.0, "approach_cone_deg": 45.0, "cam_zoom": 1.0}.get(k, 0.0)) > 1e-6)
                or (isinstance(v, list) and any(abs(x) > 1e-6 for x in v))))
        r1, r3 = res.get("BM-1"), res.get("BM-3")
        verdict = ""
        if r1 and r3:
            lo3, _ = wilson(r3["k"], r3["n"]); _, hi1 = wilson(r1["k"], r1["n"])
            verdict = " **pass**" if lo3 > hi1 else (" partial" if r3["rate"] > r1["rate"] else " fail")
        rows.append(f'| {task} | `{view}`{" (action dims only)" if act else ""} | {cell(res.get("BM-0"))} | {cell(r1)} | {cell(res.get("BM-2"))} | '
                    f'{cell(r3)}{verdict} | {gtxt} | v{arms.get("BM-3", {}).get("version") or "?"}: {vtxt} |')
        dump[f"t{task}_{view}{'_act' if act else ''}"] = {"results": res, "gate": gate, "vector": vec, "cons": {k: g.get(k) for k in ("consolidation_id", "rollouts", "wallclock_s", "promoted_version")}}
    print("| task | view | BM-0 stock | BM-1 camera moved | BM-2 identity shim | BM-3 after one sleep | gate | promoted vector |")
    print("|---|---|---|---|---|---|---|---|")
    print("\n".join(rows) if rows else "| (no reps yet) |")
    for si, cs in cons.items():
        for g in cs:
            print(f"\nCONS {si}: rollouts={g.get('rollouts')} wall={g.get('wallclock_s', 0):.0f}s promoted={g.get('promoted_version')} "
                  f"validation={json.dumps((g.get('best_candidate') or {}).get('validation'))}")
    json.dump(dump, open(os.path.join(d, "camera_table.json"), "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
