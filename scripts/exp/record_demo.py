"""Render the demo story beats as MP4s (same seed across arms so the only difference is the layer).
Runs inline (workers=1) so frames stream straight into the recorder. Output: $FM_LOGS/videos/<beat>.mp4"""
import dataclasses, json, os, sys
from fleet_memory.envs.libero_plus import list_configs
from fleet_memory.execution.params import S3Params
from fleet_memory.memory.store import EventStore
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
    return best["params"] if best else None


def beat(name, cfg, env_cache):
    key = (cfg.env_kind, cfg.suite, cfg.task_id, json.dumps(cfg.env_kwargs, sort_keys=True))
    if key not in env_cache:
        env_cache[key] = (make_env(cfg), make_policy(cfg))
    env, pol = env_cache[key]
    ep = run_episode(dataclasses.replace(cfg, record_video=VID, video_label=name, log_path=f"{VID}/events.jsonl"), env, pol)
    src = f"{VID}/{ep.episode_id}.mp4"
    dst = f"{VID}/{name.split(' ')[0]}_{'ok' if ep.outcome.env_success else 'fail'}.mp4"
    if os.path.exists(src): os.replace(src, dst)
    print(f"{name}: success={ep.outcome.env_success} steps={ep.outcome.steps} -> {dst}", flush=True)
    return ep


def main(seed_pert=5047, seed_mastery=0):
    cache = {}
    cfg0 = [c for c in list_configs("robot_init", "libero_spatial") if str(c.get("base_task_idx")) == "0"][0]
    std = RunConfig(suite="libero_spatial", task_id="0", seed=seed_pert, arm="A", env_kind="libero", policy_kind="smolvla")
    pert = dataclasses.replace(std, env_kind="libero_plus", env_kwargs={"config": cfg0}, environment_tag="plus_robot_init_0")
    hom = S3Params.identity(); hom.homing_enable = 1.0
    beat("BM-0 standard start · raw VLA", std, cache)
    beat("BM-1 perturbed start (LIBERO-Plus r=0.1) · raw VLA", pert, cache)
    beat("BM-4 perturbed start · shim with hand-set homing", dataclasses.replace(pert, arm="B", s3_params=hom.to_dict()), cache)
    v2 = incumbent_params(f"{L}/benchmark/events.jsonl", "si_0__libero_spatial_0_plus_robot_init_0")
    if v2:
        beat("BM-3 perturbed start · consolidated vector (one unattended sleep)", dataclasses.replace(pert, arm="B", s3_params=v2), cache)
    # mastery: LIBERO-10 task 3, raw vs promoted v2
    m = RunConfig(suite="libero_10", task_id="3", seed=seed_mastery, arm="A", env_kind="libero", policy_kind="smolvla")
    beat("MASTERY-A raw VLA · LIBERO-10 bowl→drawer", m, cache)
    v2m = None
    for line in open(f"{L}/v31/events.jsonl"):
        try: d = json.loads(line)
        except Exception: continue
        if d.get("type") == "skill_instance" and d["version"] == 2: v2m = d["params"]
    if v2m:
        beat("MASTERY-B v2 · after sleep cycle 2", dataclasses.replace(m, arm="B", s3_params=v2m), cache)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5047, int(sys.argv[2]) if len(sys.argv) > 2 else 0)
