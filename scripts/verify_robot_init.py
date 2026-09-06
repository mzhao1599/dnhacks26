"""Recompute the robot-init headline (docs/RESULTS.md §4, README "The result") from the RAW episode records in
logs/hopper/benchmark/reps.jsonl: every held-out evaluation episode of the replication runs (init states 40-49 x fresh
policy-noise draws), grouped by task, arm and the S3 parameter file that ran. Versions are labelled by matching the file to
the `skill_instance` promotion records in logs/hopper/benchmark/events.jsonl (or events_wide.jsonl for the wide-search instance).

    python scripts/verify_robot_init.py [logs/hopper/benchmark]
"""
import json, os, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "logs", "hopper", "benchmark")


def read(p):
    out = []
    if not os.path.exists(p): return out
    for l in open(p):
        try: out.append(json.loads(l))
        except Exception: pass
    return out


def norm(p):
    return json.dumps({k: ([round(float(x), 6) for x in v] if isinstance(v, list) else round(float(v), 6)) for k, v in sorted(p.items()) if not k.startswith("cam_")}, sort_keys=True)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return (c - h) / d, (c + h) / d


def main():
    versions = {}
    for f in ("events.jsonl", "events_wide.jsonl"):
        for r in read(os.path.join(D, f)):
            if r.get("type") == "skill_instance":
                versions[norm(r["params"])] = f"v{r['version']} of {r['skill_instance_id']}"
    identity = norm({"approach_offset_xyz": [0, 0, 0], "pregrasp_height": 0.05, "grasp_offset_z": 0, "time_scale": 1, "velocity_cap": 1,
                     "gripper_cmd": 1, "approach_cone_deg": 45, "blend_alpha": 0, "homing_enable": 0, "homing_pos_delta": [0, 0, 0], "homing_rot_delta": [0, 0, 0]})
    homing = norm({**json.loads(identity), "homing_enable": 1.0})
    eps = [r for r in read(os.path.join(D, "reps.jsonl")) if r.get("type") == "episode" and r.get("outcome")]
    acc = defaultdict(lambda: [0, 0, 0.0, set()])
    for e in eps:
        if int(e["seed"]) < 5040 or not (40 <= int(e["seed"]) % 50 <= 49): continue
        task = e.get("environment_id", "").split("_")[2] if "libero_spatial" in e.get("environment_id", "") else "?"
        env = e.get("environment_id", "")
        if "plus_robot_init" not in env:
            arm = "BM-0 standard start, raw policy"
        elif e["condition"] == "A":
            arm = "BM-1 perturbed start, raw policy"
        else:
            key = norm(e.get("s3_params") or {})
            arm = ("BM-2 identity file" if key == identity else "BM-4 hand-set homing" if key == homing
                   else f"BM-3 promoted file ({versions.get(key, 'unlabelled vector')})")
        a = acc[(task, arm)]; a[0] += int(e["outcome"]["env_success"]); a[1] += 1; a[2] += float(e["outcome"]["steps"]); a[3].add(int(e["seed"]))
    print(f"{len(eps)} episode records in reps.jsonl\n")
    print(f"{'task':5} {'arm':78} {'k/n':>8} {'rate':>5} {'95% CI':>12} {'steps':>6}  seeds")
    for key in sorted(acc):
        k, n, st, seeds = acc[key]; lo, hi = wilson(k, n)
        print(f"{key[0]:5} {key[1]:78} {k:>3}/{n:<4} {100*k/n:4.0f}% [{100*lo:4.0f}, {100*hi:4.0f}] {st/n:6.0f}  {min(seeds)}-{max(seeds)} ({len(seeds)} distinct)")
    print("\nGates (consolidation records, benchmark/events.jsonl; gate rollouts = init states 30-39):")
    for r in read(os.path.join(D, "events.jsonl")):
        if r.get("type") == "consolidation" and r.get("phase") == "end":
            g = r.get("gate")
            if g: print(f"  {r['skill_instance_id']:52} {'PASSED ' if g['passed'] else 'refused'} cost {g['incumbent_cost']:.2f} -> {g['candidate_cost']:.2f}, success {100*g['incumbent_success']:.0f}% -> {100*g['candidate_success']:.0f}% on {g['n_seeds']} seeds; promoted v{r.get('promoted_version')}")
            else: print(f"  {r['skill_instance_id']:52} no gate (validation kept the incumbent)")


if __name__ == "__main__":
    main()
