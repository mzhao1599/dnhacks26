"""Recompute every camera-family number in docs/RESULTS.md §8 from the RAW episode records (no summaries involved).

Evidence chain: each episode is one JSON line in logs/hopper/bench_camera/events.jsonl (append-only, written by the
worker on the cluster) with its seed, arm, S3 parameter file, and the env's success predicate. Held-out reps use
seeds 5040 + i + 50*r (init state 40+i, policy-noise draw r; so seed % 50 is 40-49); CEM/gate rollouts carry
perturbation.consolidation_id and use seeds < 5000 (opt: init states 0-29, gate: 30-39). BM-1 = arm A; BM-2 = arm B with the identity file; BM-3 = arm B with the promoted file.

    python scripts/verify_camera.py [logs/hopper]
"""
import json, os, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "logs", "hopper")
IDENTITY = {"approach_offset_xyz": [0, 0, 0], "pregrasp_height": 0.05, "grasp_offset_z": 0, "time_scale": 1, "velocity_cap": 1,
            "gripper_cmd": 1, "approach_cone_deg": 45, "blend_alpha": 0, "homing_enable": 0, "homing_pos_delta": [0, 0, 0],
            "homing_rot_delta": [0, 0, 0], "cam_roll_deg": 0, "cam_zoom": 1, "cam_shift_xy": [0, 0]}


def read(p):
    out = []
    for l in open(p):
        try: out.append(json.loads(l))
        except Exception: pass
    return out


def is_identity(p):
    if not p: return True
    for k, v in IDENTITY.items():
        x = p.get(k, v)
        if isinstance(v, list):
            if any(abs(float(a) - float(b)) > 1e-9 for a, b in zip(x, v)): return False
        elif abs(float(x) - float(v)) > 1e-9: return False
    return True


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return (c - h) / d, (c + h) / d


def main():
    recs = read(os.path.join(LOGS, "bench_camera", "events.jsonl"))
    eps = [r for r in recs if r.get("type") == "episode" and r.get("outcome")]
    held = [e for e in eps if int(e["seed"]) >= 5040 and 40 <= int(e["seed"]) % 50 <= 49 and not (e.get("perturbation") or {}).get("consolidation_id")]
    acc = defaultdict(lambda: [0, 0, 0.0, set()])
    for e in held:
        env = e.get("environment_id", "")
        if "plus_camera" not in env:
            arm = "BM-0 stock camera"
        elif e["condition"] == "A":
            arm = "BM-1 camera moved, raw policy"
        elif is_identity(e.get("s3_params")):
            arm = "BM-2 identity file"
        else:
            arm = "BM-3 promoted file"
        draw = "draws 1-5" if int(e["seed"]) < 5290 else "draws 6-8 (cycle 2)"
        key = (env if "plus_camera" in env else env + " (stock)", arm, draw)
        a = acc[key]; a[0] += int(e["outcome"]["env_success"]); a[1] += 1; a[2] += float(e["outcome"]["steps"]); a[3].add(int(e["seed"]))
    print(f"{len(eps)} episode records, {len(held)} held-out evaluation episodes (init states 40-49, no consolidation_id)\n")
    print(f"{'environment':58} {'arm':34} {'draws':10} {'k/n':>7} {'rate':>6} {'95% CI':>12} {'steps':>6} seeds")
    for key in sorted(acc):
        k, n, st, seeds = acc[key]; lo, hi = wilson(k, n)
        print(f"{key[0]:58} {key[1]:34} {key[2]:10} {k:>3}/{n:<3} {100*k/n:5.0f}% [{100*lo:4.0f}, {100*hi:4.0f}] {st/n:6.0f} {min(seeds)}-{max(seeds)}")
    print("\nGates (from consolidation records; gate rollouts are seeds 4030-4069 = init states 30-39, never in the evaluation set):")
    for r in recs:
        if r.get("type") == "consolidation" and r.get("phase") == "end" and r.get("gate"):
            g = r["gate"]
            print(f"  {r['skill_instance_id']:60} {'PASSED ' if g['passed'] else 'refused'} cost {g['incumbent_cost']:.2f} -> {g['candidate_cost']:.2f}, "
                  f"success {100*g['incumbent_success']:.0f}% -> {100*g['candidate_success']:.0f}% on {g['n_seeds']} layouts; {r.get('rollouts')} rollouts, {r.get('wallclock_s',0)/60:.0f} min; promoted v{r.get('promoted_version')}")
    pp = os.path.join(LOGS, "protocol_cam", "events.jsonl")
    if os.path.exists(pp):
        prec = read(pp); by = {r["episode_id"]: r for r in prec if r.get("type") == "episode"}
        print("\nProtocol P (camera bump, unattended): per stage from the stage records' episode ids")
        for r in prec:
            if r.get("type") == "protocol_stage":
                ids = [i for i in r.get("episode_ids", []) if i in by]
                k = sum(int(by[i]["outcome"]["env_success"]) for i in ids)
                print(f"  {r['stage']:10} {k}/{len(ids)}")
            if r.get("type") == "drift_trigger":
                print(f"  drift_trigger: ewma {r.get('ewma_cost'):.2f} vs baseline {r.get('baseline_cost'):.2f}")
            if r.get("type") == "consolidation" and r.get("phase") == "end" and r.get("gate"):
                g = r["gate"]; print(f"  auto-sleep gate: {'PASSED' if g['passed'] else 'refused'} {100*g['incumbent_success']:.0f}% -> {100*g['candidate_success']:.0f}% ({r.get('rollouts')} rollouts) promoted v{r.get('promoted_version')}")


if __name__ == "__main__":
    main()
