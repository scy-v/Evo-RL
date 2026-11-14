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

"""Contracts for the LeRobot policy adapter used by native UR inference."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.processor import NormalizerProcessorStep, UnnormalizerProcessorStep
from lerobot.processor.converters import batch_to_transition, transition_to_batch
from lerobot.scripts.ur5e_inference.policy import URPolicy, observation_to_batch, validate_checkpoint


@pytest.fixture
def observation():
    return {
        "observation/state": np.arange(19, dtype=np.float32),
        "observation/image": np.full((48, 64, 3), 255, dtype=np.uint8),
        "observation/wrist_image": np.zeros((48, 64, 3), dtype=np.uint8),
        "prompt": "Insert the black rack into the blue slot",
    }


@pytest.fixture
def checkpoint_config(tmp_path):
    cfg = PI05Config(
        device="cpu",
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (19,)),
            "observation.images.exterior_image": PolicyFeature(FeatureType.VISUAL, (3, 480, 640)),
            "observation.images.wrist_image": PolicyFeature(FeatureType.VISUAL, (3, 480, 640)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
    )
    cfg.save_pretrained(tmp_path)
    save_file({"test": torch.zeros(1)}, tmp_path / "model.safetensors")
    stats = {
        "observation.state.q01": torch.zeros(19),
        "observation.state.q99": torch.full((19,), 2.0),
        "action.q01": torch.zeros(7),
        "action.q99": torch.full((7,), 2.0),
    }
    for prefix, registry in (("pre", "normalizer_processor"), ("post", "unnormalizer_processor")):
        state = f"{prefix}_stats.safetensors"
        save_file(stats, tmp_path / state)
        (tmp_path / f"policy_{prefix}processor.json").write_text(
            json.dumps({"steps": [{"registry_name": registry, "state_file": state}]})
        )
    return {
        "model": {"checkpoint_dir": str(tmp_path)},
        "robot": {"action_horizon": 3, "fix_gripper": False, "direct_action_horizon": False},
    }


def test_observation_mapping_preserves_camera_resolution_and_order(observation):
    original = copy.deepcopy(observation)
    batch = observation_to_batch(observation)
    torch.testing.assert_close(batch["observation.state"], torch.arange(19, dtype=torch.float32))
    assert batch["observation.images.exterior_image"].shape == (3, 48, 64)
    assert batch["observation.images.exterior_image"].eq(1).all()
    assert batch["observation.images.wrist_image"].eq(0).all()
    assert batch["task"] == observation["prompt"]
    for key in ("observation/state", "observation/image", "observation/wrist_image"):
        np.testing.assert_array_equal(observation[key], original[key])


@pytest.mark.parametrize("bad_state", [np.zeros(18), np.full(19, np.nan)])
def test_invalid_state_is_rejected(observation, bad_state):
    observation["observation/state"] = bad_state
    with pytest.raises(ValueError, match="finite 19-D"):
        observation_to_batch(observation)


def test_float_images_are_rejected_instead_of_dividing_twice(observation):
    observation["observation/image"] = observation["observation/image"].astype(np.float32) / 255
    with pytest.raises(ValueError, match="HWC uint8 RGB"):
        observation_to_batch(observation)


def test_checkpoint_statistics_are_required(checkpoint_config, tmp_path):
    assert validate_checkpoint(checkpoint_config).chunk_size == 50
    (tmp_path / "post_stats.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="processor state"):
        validate_checkpoint(checkpoint_config)


def test_inconsistent_action_statistics_are_rejected(checkpoint_config, tmp_path):
    save_file(
        {"action.q01": torch.ones(7), "action.q99": torch.full((7,), 2.0)},
        tmp_path / "post_stats.safetensors",
    )
    with pytest.raises(ValueError, match="differs"):
        validate_checkpoint(checkpoint_config)


@pytest.mark.parametrize(
    "key,value", [("fix_gripper", True), ("direct_action_horizon", True), ("action_horizon", 51)]
)
def test_incompatible_robot_contract_is_rejected(checkpoint_config, key, value):
    checkpoint_config["robot"][key] = value
    with pytest.raises(ValueError):
        validate_checkpoint(checkpoint_config)


def test_checkpoint_quantiles_normalize_state_and_restore_actions(checkpoint_config):
    cfg = validate_checkpoint(checkpoint_config)
    from safetensors.torch import load_file

    path = checkpoint_config["model"]["checkpoint_dir"]
    normalizer = NormalizerProcessorStep(features=cfg.input_features, norm_map=cfg.normalization_mapping)
    unnormalizer = UnnormalizerProcessorStep(features=cfg.output_features, norm_map=cfg.normalization_mapping)
    normalizer.load_state_dict(load_file(f"{path}/pre_stats.safetensors"))
    unnormalizer.load_state_dict(load_file(f"{path}/post_stats.safetensors"))
    state = torch.ones(1, 19)
    normalized = transition_to_batch(normalizer(batch_to_transition({"observation.state": state})))
    torch.testing.assert_close(normalized["observation.state"], torch.zeros_like(state))
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action

    action = transition_to_policy_action(unnormalizer(policy_action_to_transition(torch.zeros(1, 50, 7))))
    torch.testing.assert_close(action, torch.ones(1, 50, 7))


def test_each_infer_requests_a_fresh_full_chunk(observation):
    class ChunkModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(chunk_size=50)
            self.calls = 0

        def predict_action_chunk(self, batch):
            self.calls += 1
            assert not torch.is_grad_enabled()
            return torch.full((1, 50, 7), float(self.calls))

    policy = URPolicy(ChunkModel(), lambda batch: batch, lambda action: action * 2)
    first, second = policy.infer(observation), policy.infer(observation)
    assert first["actions"].shape == (50, 7)
    assert np.all(first["actions"] == 2)
    assert np.all(second["actions"] == 4)


@pytest.mark.parametrize("enabled,use_cfg,beta", [(False, False, 1), (True, False, 1), (True, True, 0), (True, True, 1), (True, True, 2)])
def test_acp_prompt_and_cfg_share_noise(observation, enabled, use_cfg, beta):
    from lerobot.rl.acp_tags import build_acp_tagged_task

    prompt = observation["prompt"]
    positive = build_acp_tagged_task(prompt, is_positive=True)

    class Model:
        config = SimpleNamespace(chunk_size=50, device="cpu")

        def __init__(self):
            self.tasks = []
            self.noises = []

        def predict_action_chunk(self, batch):
            self.tasks.append(batch["task"])
            noise = torch.rand(1, 50, 7)
            self.noises.append(noise)
            return noise + (2 if batch["task"] == positive else 0)

    model = Model()
    policy = URPolicy(model, lambda batch: batch, lambda action: action.square(),
                      {"enable": enabled, "use_cfg": use_cfg, "cfg_beta": beta})
    result = policy.infer(observation)["actions"]
    assert observation["prompt"] == prompt
    if use_cfg:
        assert model.tasks == [positive, prompt]
        torch.testing.assert_close(model.noises[0], model.noises[1])
        unconditional = model.noises[0].square()
        conditional = (model.noises[0] + 2).square()
        expected = unconditional + beta * (conditional - unconditional)
    else:
        assert model.tasks == [positive if enabled else prompt]
        expected = (model.noises[0] + (2 if enabled else 0)).square()
    np.testing.assert_allclose(result, expected.squeeze(0).numpy(), rtol=1e-6)


@pytest.mark.parametrize("options", [
    {"use_cfg": True}, {"enable": "false"}, {"cfg_beta": -1}, {"cfg_beta": float("nan")},
])
def test_invalid_acp_options_are_rejected(options):
    with pytest.raises(ValueError):
        URPolicy(None, None, None, options)
