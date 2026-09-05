"""SmolVLA (lerobot 0.6.x) adapter implementing the Policy protocol.

Loading and preprocessing follow ``lerobot/scripts/lerobot_eval.py`` exactly:

    cfg    = PreTrainedConfig.from_pretrained(path)              # SmolVLAConfig from config.json
    policy = SmolVLAPolicy.from_pretrained(path, config=cfg)
    pre, post = make_pre_post_processors(cfg, pretrained_path=path,
                                         preprocessor_overrides={"device_processor": {"device": ...}})

The preprocessor pipeline stored with the checkpoint is: rename -> to_batch -> smolvla newline ->
tokenizer(task) -> device -> normalizer(MEAN_STD state/action). The postprocessor is device(cpu)
-> unnormalizer(action). We never hand-roll normalisation.

Observation contract (what ``LiberoEnv`` already gives us, i.e. the output of lerobot's
``LiberoProcessorStep``): upright images, state = eef_pos(3)+axis_angle(3)+gripper_qpos(2).
Batch fed to the pipeline:
    observation.images.image   float32 [1,3,H,W] in [0,1]   (agentview)
    observation.images.image2  float32 [1,3,H,W] in [0,1]   (eye_in_hand)
    observation.state          float32 [1,8]
    task                       [instruction]

``act`` calls ``predict_action_chunk`` (one flow-matching sample -> [1, chunk_size, 7]) and returns
the first ``n_action_steps`` rows as float32[k,7] clipped to [-1,1]. With the checkpoint default
n_action_steps=1 that is exactly what lerobot_eval's ``select_action`` executes per step.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from fleet_memory.envs.base import Obs
from fleet_memory.policies.base import ActionChunk, clip_action

DEFAULT_PATH = "HuggingFaceVLA/smolvla_libero"


class SmolVLAPolicy:
    name = "smolvla-450m"

    def __init__(
        self,
        path: str = DEFAULT_PATH,
        device: str = "cuda",
        n_action_steps: int | None = None,
        image_size: int = 256,
        num_steps: int | None = None,
    ):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as _LRSmolVLA

        self.torch = torch
        self.path = path
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.image_size = int(image_size)

        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.device = device
        cfg.pretrained_path = path
        if n_action_steps is not None:
            cfg.n_action_steps = int(min(max(1, n_action_steps), cfg.chunk_size))
        if num_steps is not None:
            cfg.num_steps = int(num_steps)          # flow-matching denoising steps (default 10)
        self.cfg = cfg
        self.n_action_steps = int(cfg.n_action_steps)
        self.chunk_size = int(cfg.chunk_size)

        self.policy = _LRSmolVLA.from_pretrained(path, config=cfg)
        self.policy.to(device)
        self.policy.eval()

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=path,
            preprocessor_overrides={
                "device_processor": {"device": str(device)},
                "rename_observations_processor": {"rename_map": {}},
            },
        )
        self.image_keys = [k for k in cfg.input_features if k.startswith("observation.images.")]
        self.state_key = "observation.state"
        self.state_dim = int(cfg.input_features[self.state_key].shape[0])
        self.instruction: str = ""
        self.last_chunk: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    def reset(self, instruction: str) -> None:
        self.instruction = str(instruction)
        self.policy.reset()
        self.last_chunk = None

    def _img(self, arr: np.ndarray):
        torch = self.torch
        t = torch.from_numpy(np.ascontiguousarray(arr))
        if t.ndim == 3:
            t = t.unsqueeze(0)
        assert t.dtype == torch.uint8, f"expected uint8 HWC image, got {t.dtype}"
        t = t.permute(0, 3, 1, 2).contiguous().float() / 255.0
        return t

    def build_batch(self, obs: Obs) -> dict[str, Any]:
        torch = self.torch
        cams = {"observation.images.image": obs.images["agentview"],
                "observation.images.image2": obs.images["eye_in_hand"]}
        batch: dict[str, Any] = {}
        for k in self.image_keys:
            if k not in cams:
                raise KeyError(f"checkpoint expects {k}; env provides {list(cams)}")
            batch[k] = self._img(cams[k])
        state = np.asarray(obs.state, dtype=np.float32).reshape(-1)
        if state.shape[0] != self.state_dim:
            raise ValueError(f"state dim {state.shape[0]} != checkpoint {self.state_dim}")
        batch[self.state_key] = torch.from_numpy(state).unsqueeze(0)
        batch["task"] = [self.instruction]
        return batch

    def act(self, obs: Obs) -> ActionChunk:
        torch = self.torch
        batch = self.preprocessor(self.build_batch(obs))
        with torch.inference_mode():
            if hasattr(self.policy, "predict_action_chunk"):
                actions = self.policy.predict_action_chunk(batch)       # [1, chunk, 7] normalised
                actions = self.postprocessor(actions)                   # unnormalise -> cpu
                chunk = actions[0, : self.n_action_steps]
            else:
                a = self.postprocessor(self.policy.select_action(batch))
                chunk = a.reshape(1, -1)
        out = clip_action(chunk.detach().float().cpu().numpy())
        self.last_chunk = out
        return out

    def close(self) -> None:
        self.policy = None
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class ZeroPolicy:
    """No-op policy (gripper open) for env / rendering smoke tests."""
    name = "zero"

    def __init__(self, n_action_steps: int = 1):
        self.k = n_action_steps

    def reset(self, instruction: str) -> None:
        self.instruction = instruction

    def act(self, obs: Obs) -> ActionChunk:
        a = np.zeros((self.k, 7), dtype=np.float32)
        a[:, 6] = -1.0
        return a


def make_policy(name: str, **kw):
    if name == "smolvla":
        return SmolVLAPolicy(**kw)
    if name == "zero":
        return ZeroPolicy()
    raise ValueError(f"unknown policy {name!r}")


__all__ = ["SmolVLAPolicy", "ZeroPolicy", "make_policy", "DEFAULT_PATH"]
