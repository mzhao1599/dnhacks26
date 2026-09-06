"""Protocol P with a LIBERO-Plus env perturbation (--perturb-env): stage RunConfig construction only, no simulator."""
from fleet_memory.runner.pool import perturbed_env_fields, stage_configs
from fleet_memory.runner.worker import RunConfig

CAM = {"dimension": "camera", "view": "0_0_100_2_354"}


def _base(**kw):
    d = dict(suite="libero_spatial", task_id="0", seed=0, arm="B", env_kind="libero", policy_kind="smolvla",
             log_path="x.jsonl", environment_tag="cam")
    d.update(kw)
    return RunConfig(**d)


def test_stage_configs_env_perturbation_only():
    seeds = list(range(100, 112))
    stages = stage_configs(_base(), None, CAM, seeds, 4)
    assert [s[0] for s in stages] == ["baseline", "perturbed", "recovery"]
    assert [s[2] for s in stages] == [False, True, True]                       # auto-sleep after the bump only
    base_cfgs, pert_cfgs, rec_cfgs = (s[1] for s in stages)
    assert all(c.env_kind == "libero" and c.env_kwargs == {} and c.perturbation is None for c in base_cfgs)
    for cfgs in (pert_cfgs, rec_cfgs):
        assert all(c.env_kind == "libero_plus" and c.env_kwargs == {"config": CAM} and c.perturbation is None for c in cfgs)
    assert [c.seed for c in base_cfgs] == seeds[0:4] and [c.seed for c in pert_cfgs] == seeds[4:8] \
        and [c.seed for c in rec_cfgs] == seeds[8:12]
    ids = {c.environment_id for _, cfgs, _ in stages for c in cfgs}
    sis = {c.skill_instance_id for _, cfgs, _ in stages for c in cfgs}
    assert ids == {"libero_spatial_0_cam"} and len(sis) == 1                    # one instance across all stages


def test_stage_configs_object_perturbation_and_both():
    shift = {"shift_xy": [0.02, 0.0]}
    only_obj = stage_configs(_base(), shift, None, list(range(6)), 2)
    assert only_obj[0][1][0].perturbation is None and only_obj[0][1][0].env_kind == "libero"
    assert all(c.perturbation == shift and c.env_kind == "libero" for _, cfgs, _ in only_obj[1:] for c in cfgs)
    both = stage_configs(_base(), shift, CAM, list(range(6)), 2)
    assert all(c.perturbation == shift and c.env_kind == "libero_plus" and c.env_kwargs == {"config": CAM}
               for _, cfgs, _ in both[1:] for c in cfgs)
    assert both[0][1][0].env_kwargs == {} and both[0][1][0].perturbation is None


def test_perturbed_env_fields_and_config_isolation():
    assert perturbed_env_fields(None) == {} and perturbed_env_fields({}) == {}
    f = perturbed_env_fields(CAM)
    assert f == {"env_kind": "libero_plus", "env_kwargs": {"config": CAM}} and f["env_kwargs"]["config"] is not CAM


def test_cli_accepts_perturb_env_for_protocol():
    from fleet_memory.runner.pool import build_parser
    a = build_parser().parse_args(["--protocol", "P", "--perturb-env", '{"dimension":"camera","view":"0_0_100_2_354"}'])
    assert a.protocol == "P" and a.perturb is None and a.perturb_env
