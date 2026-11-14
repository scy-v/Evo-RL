"""CPU-only stage-1 preflight; never launch training or allocate model storage."""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"

import json
import shlex
import sys
import uuid
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch

import draccus
import torch
from safetensors import safe_open

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch
from lerobot.utils.constants import ACTION


def compare_sft_reference(cfg, work):
    """Compare parsed configurations, including defaults omitted from the commands."""
    reference_doc = work.parent / "docs/source/pi05.mdx"
    block = next(part.split("```", 1)[0] for part in reference_doc.read_text().split("```bash\n")[1:]
                 if "--policy.type=pi05" in part)
    arguments = shlex.split(block.replace("\\\n", " "))[2:]
    arguments = ["--policy.device=cpu" if arg.startswith("--policy.device=") else arg for arg in arguments]
    reference = draccus.parse(config_class=TrainPipelineConfig, args=arguments)
    # No validation side effects or remote requests: presets are pure config objects.
    reference.optimizer = reference.policy.get_optimizer_preset()
    reference.scheduler = reference.policy.get_scheduler_preset()
    allowed_policy = {"pretrained_path", "push_to_hub", "repo_id"}
    allowed_training = {"dataset", "policy", "batch_size", "steps", "save_freq", "output_dir", "job_name", "wandb"}
    for field in fields(cfg.policy):
        if field.name not in allowed_policy:
            assert getattr(cfg.policy, field.name) == getattr(reference.policy, field.name), f"policy.{field.name}"
    for field in fields(cfg):
        if field.name not in allowed_training:
            assert getattr(cfg, field.name) == getattr(reference, field.name), field.name
    for field in fields(cfg.wandb):
        if field.name != "disable_artifact":
            assert getattr(cfg.wandb, field.name) == getattr(reference.wandb, field.name), f"wandb.{field.name}"
    for field in fields(cfg.dataset):
        if field.name not in {"repo_id", "root", "video_backend", "state_indices"}:
            assert getattr(cfg.dataset, field.name) == getattr(reference.dataset, field.name), f"dataset.{field.name}"
    core = ["compile_model", "gradient_checkpointing", "dtype", "freeze_vision_encoder",
            "train_expert_only", "use_peft", "use_amp", "chunk_size", "n_action_steps",
            "max_state_dim", "max_action_dim", "tokenizer_max_length", "normalization_mapping",
            "rtc_config", "mask_action_padding_loss", "optimizer_lr", "optimizer_betas",
            "optimizer_eps", "optimizer_weight_decay", "optimizer_grad_clip_norm"]
    report = {"reference": str(reference_doc), "all_core_fields_match": True,
              "policy": {name: getattr(cfg.policy, name) for name in core},
              "acp_enable": cfg.acp.enable, "use_rabc": cfg.use_rabc,
              "image_transforms_enable": cfg.dataset.image_transforms.enable,
              "optimizer": repr(cfg.optimizer), "scheduler": repr(cfg.scheduler)}
    print("Parsed config versus repository PI05 SFT example: all core fields and default switches MATCH", flush=True)
    return report


def main():
    torch.set_num_threads(2)
    work = Path(__file__).parent
    blocks = (work / "训练命令.md").read_text().split("```bash\n")
    command = next(block.split("```", 1)[0] for block in blocks[1:] if "accelerate launch" in block and "--acp.enable=false" in block)
    words = shlex.split(command.replace("\\\n", " "))
    env_indices = next(word.split("=", 1)[1] for word in words if word.startswith("OBS_INDICES="))
    os.environ["OBS_INDICES"] = env_indices
    module_index = words.index("lerobot.scripts.lerobot_train")
    arguments = words[module_index + 1:]
    # Parse the actual command; replace only the device to keep the check CPU-only.
    arguments = ["--policy.device=cpu" if arg.startswith("--policy.device=") else arg for arg in arguments]
    sys.argv = ["cpu-preflight", *arguments]
    cfg = draccus.parse(config_class=TrainPipelineConfig, args=arguments)
    requested_output = cfg.output_dir
    output_available = not requested_output.exists()
    if not output_available:
        print(f"OUTPUT EXISTS: {requested_output}; a new training run must use another output directory.", flush=True)
        # Audit an existing experiment without overwriting it or changing the command.
        cfg.output_dir = Path("/tmp") / f"evo-rl-readonly-audit-{uuid.uuid4().hex}"
    try:
        cfg.validate()
    finally:
        cfg.output_dir = requested_output
    assert cfg.policy.type == "pi05" and not cfg.acp.enable and not cfg.use_rabc
    assert not cfg.policy.freeze_vision_encoder and not cfg.policy.train_expert_only and not cfg.policy.use_peft
    assert cfg.policy.compile_model and cfg.policy.gradient_checkpointing
    assert cfg.wandb.enable and cfg.save_freq == 5000
    assert cfg.wandb.disable_artifact and not cfg.policy.push_to_hub
    assert cfg.batch_size == 16 and cfg.steps == 30000
    assert "--num_processes=4" in words and "--multi_gpu" in words
    comparison = compare_sft_reference(cfg, work)
    print("Actual command parses and validates: PI05 plain full SFT, 4 x 16 = 64", flush=True)

    dataset = make_dataset(cfg)
    features = dataset_to_policy_features(dataset.meta.features)
    cfg.policy.input_features = {key: value for key, value in features.items() if key != ACTION}
    cfg.policy.output_features = {ACTION: features[ACTION]}
    assert cfg.policy.input_features["observation.state"].shape == (19,)
    assert cfg.policy.output_features[ACTION].shape == (7,)
    sample = dataset[0]
    from torch.utils.data._utils.collate import default_collate
    batch = default_collate([sample])
    processor_ok, processor_error = False, None
    try:
        pre, post = make_pre_post_processors(
            cfg.policy, pretrained_path=cfg.policy.pretrained_path, dataset_stats=dataset.meta.stats,
            preprocessor_overrides={
                "device_processor": {"device": "cpu"},
                "normalizer_processor": {"stats": dataset.meta.stats,
                                         "features": {**cfg.policy.input_features, **cfg.policy.output_features},
                                         "norm_map": cfg.policy.normalization_mapping},
                "rename_observations_processor": {"rename_map": cfg.rename_map},
            },
            postprocessor_overrides={"unnormalizer_processor": {
                "stats": dataset.meta.stats, "features": cfg.policy.output_features,
                "norm_map": cfg.policy.normalization_mapping}},
        )
        batch = pre(default_collate([sample]))
        assert batch["observation.state"].shape == (1, 19)
        assert batch["action"].shape == (1, 50, 7)
        assert batch["observation.language.tokens"].shape == (1, 200)
        assert torch.isfinite(batch["action"]).all()
        torch.testing.assert_close(post(batch["action"]), sample["action"].unsqueeze(0), atol=1e-6, rtol=1e-5)
        print("Actual base processors + real sample: state19, action50x7, tokens200, inverse stats PASS", flush=True)

        processor_ok = True
    except Exception as error:
        processor_error = f"{type(error).__name__}: {str(error)[:500]}"
        print("PROCESSOR BLOCKED:", processor_error, flush=True)

    # Build only a shape graph on meta; no 14GB parameter allocation and no GPU.
    with torch.device("meta"), patch.object(PI05Pytorch, "to", lambda self, *args, **kwargs: self):
        policy = PI05Policy(cfg.policy)
    assert cfg.policy.gradient_checkpointing and policy.model.gradient_checkpointing_enabled
    source_shapes = {}
    with safe_open(str(cfg.policy.pretrained_path / "model.safetensors"), framework="pt", device="cpu") as weights:
        for name in weights.keys():
            source_shapes[name] = torch.empty(weights.get_slice(name).get_shape(), device="meta")
    fixed = policy._fix_pytorch_state_dict_keys(source_shapes, cfg.policy)
    mapped = {name if name.startswith("model.") else f"model.{name}": value for name, value in fixed.items()}
    target = policy.state_dict()
    missing, extra = sorted(set(target) - set(mapped)), sorted(set(mapped) - set(target))
    mismatched = [name for name in set(target) & set(mapped) if target[name].shape != mapped[name].shape]
    assert not missing and not extra and not mismatched, (missing, extra, mismatched)
    total = sum(param.numel() for param in policy.parameters())
    trainable = sum(param.numel() for param in policy.parameters() if param.requires_grad)
    assert total == trainable
    with patch.object(policy, "parameters", return_value=iter([torch.empty(0, device="cpu")])):
        images, masks = policy._preprocess_images(batch)
    assert len(images) == 2 and all(mask.all().item() for mask in masks)
    assert all(tuple(image.shape) == (1, 3, 224, 224) for image in images)
    assert not torch.cuda.is_initialized()
    report = {"reference_core_fields_match": comparison["all_core_fields_match"], "command_valid": True, "global_batch": 64, "num_processes": 4,
              "new_run_output_available": output_available,
              "compile_model": cfg.policy.compile_model, "wandb_enabled": cfg.wandb.enable,
              "wandb_model_artifact_upload": not cfg.wandb.disable_artifact,
              "hf_model_upload": cfg.policy.push_to_hub, "hf_model_repo": cfg.policy.repo_id,
              "save_freq": cfg.save_freq,
              "state_dimensions": 19, "action_dimensions": 7, "chunk_size": 50,
              "cameras": list(cfg.policy.image_features), "processor_real_sample": processor_ok, "processor_error": processor_error,
              "weight_tensor_names_and_shapes_match": True, "gradient_checkpointing_setup": True,
              "all_parameters_trainable": True, "parameter_count": total,
              "tokenizer_cached": processor_ok, "cuda_initialized": False,
              "training_started": False, "gpu_memory_fit_tested": False}
    print(json.dumps(report, indent=2), flush=True)
    if not processor_ok:
        raise SystemExit("Preflight incomplete: resolve the processor/tokenizer error before starting SFT.")
    if not output_available:
        raise SystemExit("CPU checks passed; choose a new output directory before starting another SFT run.")


if __name__ == "__main__":
    main()
