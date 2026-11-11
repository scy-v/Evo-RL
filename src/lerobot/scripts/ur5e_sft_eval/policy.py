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

"""Native Evo-RL model I/O with RLinf UR execution-chunk semantics."""

import logging
import time

import numpy as np

from lerobot.scripts.ur5e_inference.policy import (
    create_policy as create_native_policy,
    validate_checkpoint,
    warmup_policy as native_warmup,
)

__all__ = ["check_model", "create_policy", "validate_checkpoint", "warmup_policy"]


class EvaluationPolicy:
    def __init__(self, native, action_horizon):
        self.native = native
        self.action_horizon = action_horizon

    def infer(self, observation):
        started = time.perf_counter()
        result = self.native.infer(observation)
        return {
            "actions": result["actions"][: self.action_horizon],
            "controller": "vla",
            "policy_timing": {"infer_ms": (time.perf_counter() - started) * 1000},
        }


def create_policy(config):
    return EvaluationPolicy(create_native_policy(config), int(config["robot"]["action_horizon"]))


def warmup_policy(policy, observation, *, timing_recorder=None):
    started = time.perf_counter()
    result = native_warmup(policy, observation)
    if timing_recorder is not None:
        timing_recorder("warmup", duration_sec=time.perf_counter() - started)
    return result


def check_model(config):
    """Check real native inference and execution slicing without hardware."""
    policy = create_policy(config)
    observation = {
        "observation/state": np.zeros(19, dtype=np.float32),
        "observation/image": np.zeros((480, 640, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((480, 640, 3), dtype=np.uint8),
        "prompt": config["task"]["description"],
    }
    result = warmup_policy(policy, observation)
    expected = (int(config["robot"]["action_horizon"]), 7)
    if result["actions"].shape != expected or not np.isfinite(result["actions"]).all():
        raise ValueError(f"Expected finite execution actions {expected}")
    logging.info("Model-only check passed: execution actions %s; no hardware connected", expected)
