"""Same-seed clips for the camera family: stock camera (BM-0), camera bumped (BM-1), bumped + consolidated vector (BM-3).
Inline (workers=1) so frames stream into the recorder. Output: $FM_LOGS/videos/cam_<beat>_<ok|fail>.mp4

    python scripts/exp/record_camera_demo.py [--task 0] [--view 0_0_100_2_354] [--seed 5047]
"""
import argparse, dataclasses, json, os
from fleet_memory.envs.libero_plus import list_configs
from fleet_memory.runner.worker import RunConfig, run_episode, make_env, make_policy

L = os.environ["FM_LOGS"]
VID = f"{L}/videos"
os.makedirs(VID, exist_ok=True)


def incumbent_params(log, si_id):
    best = None
    for line in open(log):
        try: d = json.loads(line)
        except Exception: continue
        if d.get("type") == "skill_instance" and d["skill_instance_id"] == si_id and d["status"] == "incumbent":
            if best is None or d["version"] > best["version"]: best = d
    return best


def beat(name, cfg, cache, seed=0):
    key = (cfg.env_kind, cfg.suite, cfg.task_id, json.dumps(cfg.env_kwargs, sort_keys=True))
    if key not in cache:
        cache[key] = (make_env(cfg), make_policy(cfg))
    env, pol = cache[key]
    ep = run_episode(dataclasses.replace(cfg, record_video=VID, video_label=name, log_path=f"{VID}/events.jsonl"), env, pol)
    src = f"{VID}/{ep.episode_id}.mp4"
    dst = f"{VID}/cam_{name.split(' ')[0]}_s{seed}_{'ok' if ep.outcome.env_success else 'fail'}.mp4"
    if os.path.exists(src): os.replace(src, dst)
    print(f"{name}: success={ep.outcome.env_success} steps={ep.outcome.steps} -> {dst}", flush=True)
    return ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="0"); ap.add_argument("--view", default="0_0_100_2_354"); ap.add_argument("--seed", type=int, default=5047)
    a = ap.parse_args()
    cfg = next(c for c in list_configs("camera", "libero_spatial") if str(c["base_task_idx"]) == a.task and c["view"] == a.view)
    tag = f"plus_camera_{a.view}"
    base = RunConfig(suite="libero_spatial", task_id=a.task, seed=a.seed, arm="A", env_kind="libero", policy_kind="smolvla")
    pert = dataclasses.replace(base, env_kind="libero_plus", env_kwargs={"config": cfg}, environment_tag=tag)
    inc = incumbent_params(f"{L}/bench_camera/events.jsonl", pert.skill_instance_id)
    cache = {}
    beat("BM-0 stock camera · raw VLA", base, cache, a.seed)
    beat("BM-1 camera bumped · raw VLA", pert, cache, a.seed)
    if inc and inc["version"] >= 2:
        beat(f"BM-3 camera bumped · consolidated v{inc['version']}", dataclasses.replace(pert, arm="B", s3_params=inc["params"]), cache, a.seed)
    else:
        print("no consolidated incumbent (version>=2) for", pert.skill_instance_id)


if __name__ == "__main__":
    main()
