# Copyright 2026 The Evo-RL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Load a native PI05 checkpoint behind VLA-Precision's infer interface."""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config

IMAGE_KEYS = {
    "observation/image": "observation.images.exterior_image",
    "observation/wrist_image": "observation.images.wrist_image",
}


def warmup_policy(policy: Any, observation: dict, **kwargs: Any) -> dict:
    """Run first inference with a live elapsed-time display and stop it on exit."""
    stream = sys.stderr
    start = time.monotonic()
    stopped = threading.Event()

    def display() -> None:
        elapsed = time.monotonic() - start
        stream.write(f"\r[WARMUP] First inference / compilation: {elapsed:.1f}s elapsed")
        stream.flush()

    def update() -> None:
        while not stopped.wait(1.0):
            display()

    display()
    timer = threading.Thread(target=update, name="ur-warmup-timer", daemon=True)
    timer.start()
    completed = False
    try:
        result = policy.infer(observation, **kwargs)
        completed = True
        return result
    finally:
        stopped.set()
        timer.join()
        elapsed = time.monotonic() - start
        status = "Completed" if completed else "Stopped before completion"
        stream.write(f"\r[WARMUP] {status} after {elapsed:.2f}s".ljust(90) + "\n")
        stream.flush()


def validate_checkpoint(config: dict[str, Any]) -> PI05Config:
    """Validate the local 19-state/7-action checkpoint and saved normalization."""
    checkpoint = Path(config["model"]["checkpoint_dir"]).expanduser()
    for name in ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(f"Required LeRobot checkpoint file is missing: {checkpoint / name}")
    cfg = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    if cfg.type != "pi05":
        raise ValueError(f"Expected a PI05 checkpoint, got {cfg.type}")
    if cfg.input_features["observation.state"].shape != (19,) or cfg.output_features["action"].shape != (7,):
        raise ValueError("UR inference requires a checkpoint with 19 state channels and 7 action channels")
    if set(cfg.image_features) != set(IMAGE_KEYS.values()):
        raise ValueError(
            f"Expected checkpoint image features {set(IMAGE_KEYS.values())}, got {set(cfg.image_features)}"
        )
    if cfg.rtc_config is not None and cfg.rtc_config.enabled:
        raise ValueError("This synchronous UR entry requires RTC to be disabled")
    if cfg.use_peft:
        raise ValueError("This entry requires full PI05 weights; merge PEFT adapters before deployment")
    if config.get("policy", {}).get("location", "local") != "local":
        raise ValueError("LeRobot UR inference currently requires policy.location=local")
    if config["robot"].get("fix_gripper", False):
        raise ValueError("This 19-state/7-action checkpoint requires robot.fix_gripper=false")
    if config["robot"].get("direct_action_horizon", False):
        raise ValueError("Set robot.direct_action_horizon=false to preserve the trained PI05 chunk size")
    horizon = int(config["robot"]["action_horizon"])
    if not 1 <= horizon <= cfg.chunk_size:
        raise ValueError(f"Expected execution horizon between 1 and {cfg.chunk_size}, got {horizon}")
    if (
        cfg.normalization_mapping["STATE"].value != "QUANTILES"
        or cfg.normalization_mapping["ACTION"].value != "QUANTILES"
    ):
        raise ValueError("This checkpoint must use QUANTILES normalization for state and action")
    saved_stats = {}
    for pipeline, normalizer in (("pre", "normalizer_processor"), ("post", "unnormalizer_processor")):
        spec = json.loads((checkpoint / f"policy_{pipeline}processor.json").read_text())
        for step in spec["steps"]:
            if step.get("state_file"):
                state_path = checkpoint / step["state_file"]
                if not state_path.is_file():
                    raise FileNotFoundError(f"Required processor state is missing: {state_path}")
            if step.get("registry_name") == normalizer:
                stats = load_file(str(checkpoint / step["state_file"]))
                expected = {"observation.state": 19, "action": 7} if pipeline == "pre" else {"action": 7}
                for key, dim in expected.items():
                    for quantile in ("q01", "q99"):
                        value = stats.get(f"{key}.{quantile}")
                        if value is None or value.shape != (dim,) or not torch.isfinite(value).all():
                            raise ValueError(
                                f"Expected finite {dim}-D {key}.{quantile} in {pipeline}processor"
                            )
                saved_stats[pipeline] = stats
    if set(saved_stats) != {"pre", "post"}:
        raise ValueError("Checkpoint must contain saved normalizer and unnormalizer states")
    for quantile in ("q01", "q99"):
        if not torch.equal(
            saved_stats["pre"][f"action.{quantile}"], saved_stats["post"][f"action.{quantile}"]
        ):
            raise ValueError(f"Action {quantile} differs between checkpoint preprocessor and postprocessor")
    return cfg


def observation_to_batch(observation: dict[str, Any]) -> dict[str, Any]:
    """Map raw UR RGB frames and physical state to unbatched LeRobot inputs."""
    state = np.asarray(observation["observation/state"], dtype=np.float32)
    if state.shape != (19,) or not np.isfinite(state).all():
        raise ValueError(f"Expected finite 19-D UR state, got shape {state.shape}")
    batch = {"observation.state": torch.from_numpy(state.copy()), "task": observation["prompt"]}
    for source, target in IMAGE_KEYS.items():
        image = np.asarray(observation[source])
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f"Expected HWC uint8 RGB image for {source}, got {image.shape}, {image.dtype}")
        batch[target] = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div(255)
    return batch


class URPolicy:
    """Expose full fresh PI05 chunks as unnormalized physical robot actions."""

    def __init__(self, model, preprocessor, postprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    @torch.inference_mode()
    def infer(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        batch = self.preprocessor(observation_to_batch(observation))
        # UR executes only a leading horizon, so never retain a LeRobot action queue.
        actions = self.model.predict_action_chunk(batch)
        actions = self.postprocessor(actions).squeeze(0).cpu().numpy()
        expected = (self.model.config.chunk_size, 7)
        if actions.shape != expected or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite actions with shape {expected}, got {actions.shape}")
        return {"actions": actions}


def create_policy(config: dict[str, Any]) -> URPolicy:
    """Restore native model weights and processors without OpenPI transforms."""
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.pretrained import PreTrainedPolicy

    cfg = validate_checkpoint(config)
    options = config["model"]
    checkpoint = Path(options["checkpoint_dir"]).expanduser()
    cfg.device = str(options.get("device", cfg.device))
    cfg.pretrained_path = checkpoint
    cfg.gradient_checkpointing = False
    cfg.compile_model = bool(options.get("compile_model", cfg.compile_model))
    cfg.compile_mode = options.get("compile_mode", cfg.compile_mode)
    cfg.num_inference_steps = int(options.get("sample_steps", cfg.num_inference_steps))
    if cfg.num_inference_steps < 1:
        raise ValueError(f"Expected positive sample_steps, got {cfg.num_inference_steps}")
    torch.set_num_threads(int(options.get("torch_num_threads", 4)))
    overrides = {"device_processor": {"device": cfg.device}}
    if options.get("tokenizer_path"):
        overrides["tokenizer_processor"] = {
            "tokenizer_name": str(Path(options["tokenizer_path"]).expanduser())
        }
    preprocessor, postprocessor = make_pre_post_processors(
        cfg,
        pretrained_path=checkpoint,
        preprocessor_overrides=overrides,
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    # Native saved keys need no OpenPI remapping. The base loader propagates
    # weight errors instead of the PI05 importer returning an uninitialized model.
    model = PreTrainedPolicy.from_pretrained.__func__(
        PI05Policy,
        checkpoint,
        config=cfg,
        strict=True,
        local_files_only=True,
    )
    model.eval()
    model.reset()
    logging.info(
        "Loaded PI05 checkpoint %s: chunk=%s, steps=%s, compile=%s (%s)",
        checkpoint,
        cfg.chunk_size,
        cfg.num_inference_steps,
        cfg.compile_model,
        cfg.compile_mode,
    )
    logging.info("Normalization restored from checkpoint processor states (QUANTILES)")
    return URPolicy(model, preprocessor, postprocessor)


def check_model(config: dict[str, Any]) -> None:
    """Run one synthetic observation through the actual weights, without hardware."""
    policy = create_policy(config)
    observation = {
        "observation/state": np.zeros(19, dtype=np.float32),
        "observation/image": np.zeros((480, 640, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((480, 640, 3), dtype=np.uint8),
        "prompt": config["task"]["description"],
    }
    result = warmup_policy(policy, observation)
    logging.info(
        "Model-only check passed: finite physical actions %s; no hardware connected", result["actions"].shape
    )
