"""Native Evo-RL PI05 adapter for the original UR observation/action contract."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.rl.acp_tags import build_acp_tagged_task


def inference_task(cfg: dict) -> str:
    """Detect saved ACP training mode; never silently deploy ACP without its tag."""
    mode = cfg["model"].get("conditioning", "auto")
    if mode not in ("auto", "sft", "acp"):
        raise ValueError("model.conditioning must be auto, sft or acp")
    saved = Path(cfg["model"]["checkpoint_dir"]).expanduser() / "train_config.json"
    trained_acp = None
    if saved.is_file():
        trained_acp = bool(json.loads(saved.read_text()).get("acp", {}).get("enable", False))
    if mode == "auto":
        if trained_acp is None:
            raise ValueError("No train_config.json: explicitly set model.conditioning to sft or acp")
        enabled = trained_acp
    else:
        enabled = mode == "acp"
        if trained_acp is not None and enabled != trained_acp:
            raise ValueError("Requested conditioning conflicts with saved ACP training configuration")
    task = cfg["record"]["task"]["description"]
    return build_acp_tagged_task(task, True) if enabled else task


def validate_model_config(cfg: dict) -> None:
    from lerobot.scripts.ur5e_inference.policy import validate_checkpoint

    from .run_rollout import inference_config

    validate_checkpoint(inference_config(cfg))
    inference_task(cfg)
    indices = cfg["model"]["obs_indices"]
    # This checkpoint was trained on these exact first 19 raw UR state channels.
    if indices != list(range(1, 20)):
        raise ValueError("This UR checkpoint requires obs_indices 1..19 in original collection order")


class URRolloutPolicy:
    def __init__(self, policy: Any, state_names: list[str], obs_indices: list[int], task: str):
        self.policy = policy
        self.state_names = tuple(state_names)
        self.indices = np.asarray(obs_indices, dtype=int) - 1
        if (
            len(self.indices) != 19
            or len(set(self.indices)) != 19
            or np.any(self.indices < 0)
            or np.any(self.indices >= len(state_names))
        ):
            raise ValueError("Expected 19 unique valid one-based UR state indices")
        self.task = task

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        state = np.asarray([observation[name] for name in self.state_names], dtype=np.float32)
        # Raw RGB goes into native processors/model. No OpenPI resize or normalization.
        return self.policy.infer(
            {
                "observation/state": state[self.indices],
                "observation/image": observation["exterior_image"],
                "observation/wrist_image": observation["wrist_image"],
                "prompt": self.task,
            }
        )
