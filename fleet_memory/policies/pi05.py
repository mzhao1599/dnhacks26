"""π₀.₅ (lerobot 0.6.1 ``lerobot/policies/pi05``) adapter implementing the Policy protocol.

Same recipe as ``policies/smolvla.py`` / ``lerobot_eval.py``:

    cfg    = PreTrainedConfig.from_pretrained(path)              # PI05Config from config.json
    policy = PI05Policy.from_pretrained(path, config=cfg)        # builds PaliGemma(2B)+expert(300M) from
                                                                 # config, loads model.safetensors (no VLM download)
    pre, post = make_pre_post_processors(cfg, pretrained_path=path, preprocessor_overrides=...)

Stored preprocessor (``lerobot/pi05_libero``): rename -> to_batch -> normalizer(MEAN_STD state/action)
-> pi05_prepare_state_tokenizer (discretises the normalised state into the prompt
"Task: <task>, State: ...;\\nAction: ") -> tokenizer(google/paligemma-3b-pt-224, max_length 200)
-> device.  Postprocessor: unnormalizer -> cpu.

The PaliGemma tokenizer repo is *gated* on the Hub; ``FM_PALIGEMMA_TOKENIZER`` (default: the
ungated mirror ``leo009/paligemma-3b-pt-224``, same Gemma sentencepiece model + <loc>/<seg> tokens)
is substituted when the original cannot be loaded. Checkpoint defaults: chunk_size=50,
n_action_steps=10, num_inference_steps=10, dtype float32 (~14 GB weights), images 256x256
(resized to 224 inside the model), state 8 = eef_pos(3)+axis_angle(3)+gripper(2) — exactly what
``LiberoEnv`` provides. ``act`` returns the first ``n_action_steps`` rows of one flow-matching
sample as float32[k,7] clipped to [-1,1].
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from fleet_memory.envs.base import Obs
from fleet_memory.policies.base import ActionChunk, clip_action

DEFAULT_PATH = "lerobot/pi05_libero"
TOKENIZER_MIRROR = "leo009/paligemma-3b-pt-224"


def resolve_tokenizer(name: str = "google/paligemma-3b-pt-224") -> str:
    """The original repo is gated; fall back to FM_PALIGEMMA_TOKENIZER / an ungated mirror."""
    from transformers import AutoTokenizer
    for cand in (name, os.environ.get("FM_PALIGEMMA_TOKENIZER", TOKENIZER_MIRROR), TOKENIZER_MIRROR):
        try:
            AutoTokenizer.from_pretrained(cand)
            return cand
        except Exception:
            continue
    raise RuntimeError("no PaliGemma tokenizer available (prefetch leo009/paligemma-3b-pt-224 into HF_HOME)")


class Pi05Policy:
    name = "pi05-3b"

    def __init__(
        self,
        path: str = DEFAULT_PATH,
        device: str = "cuda",
        n_action_steps: int | None = 10,
        image_size: int = 256,
        num_inference_steps: int | None = None,
        dtype: str | None = None,
    ):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy as _LRPi05

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
        if num_inference_steps is not None:
            cfg.num_inference_steps = int(num_inference_steps)
        if dtype:
            cfg.dtype = dtype                       # "bfloat16" halves memory / ~2x faster; default float32
        self.cfg = cfg
        self.n_action_steps = int(cfg.n_action_steps)
        self.chunk_size = int(cfg.chunk_size)

        self.policy = _LRPi05.from_pretrained(path, config=cfg)
        self.policy.to(device)
        self.policy.eval()

        tok = resolve_tokenizer()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=path,
            preprocessor_overrides={
                "device_processor": {"device": str(device)},
                "rename_observations_processor": {"rename_map": {}},
                "tokenizer_processor": {"tokenizer_name": tok},
            },
        )
        self.tokenizer_name = tok
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
        return t.permute(0, 3, 1, 2).contiguous().float() / 255.0

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
            actions = self.policy.predict_action_chunk(batch)          # [1, chunk, 7] normalised
            actions = self.postprocessor(actions)                      # unnormalise -> cpu
            chunk = actions[0, : self.n_action_steps]
        out = clip_action(chunk.detach().float().cpu().numpy())
        self.last_chunk = out
        return out

    def close(self) -> None:
        self.policy = None
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def make_policy(name: str, **kw):
    if name in ("pi05", "pi05-3b"):
        return Pi05Policy(**kw)
    raise ValueError(f"unknown policy {name!r}")


__all__ = ["Pi05Policy", "make_policy", "DEFAULT_PATH", "resolve_tokenizer"]
