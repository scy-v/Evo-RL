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

"""Command line entry for LeRobot PI05 inference on the RLinf-derived VLA-Precision SFT evaluation backend."""

import argparse
import logging
import os
from pathlib import Path

import yaml


def main() -> None:
    """Run UR inference or a hardware-free checkpoint check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="UR inference YAML config")
    parser.add_argument(
        "--check-model", action="store_true", help="Run synthetic inference without connecting hardware"
    )
    parser.add_argument(
        "--check-checkpoint",
        action="store_true",
        help="Validate config and processor stats without loading model/tokenizer",
    )
    parser.add_argument("--device", help="Override model.device, e.g. cpu or cuda:0")
    parser.add_argument("--no-compile", action="store_true", help="Disable compilation for this process")
    parser.add_argument("--sample-steps", type=int, help="Override the number of denoising steps")
    parser.add_argument(
        "--no-visualize", action="store_true", help="Disable OpenCV preview for headless environments"
    )
    parser.add_argument(
        "--continuous-tracking",
        choices=("true", "false"),
        help="Override background force tracking for this run",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = yaml.safe_load(args.config.read_text())
    if args.continuous_tracking is not None:
        config["robot"]["force_mode"]["continuous_tracking"] = args.continuous_tracking == "true"
    if args.device:
        config["model"]["device"] = args.device
    if args.no_compile:
        config["model"]["compile_model"] = False
    if args.sample_steps is not None:
        config["model"]["sample_steps"] = args.sample_steps
    if args.no_visualize:
        config["video"]["visualize"] = False
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.get("policy", {}).get("cuda_visible_devices", "0"))
    from .policy import check_model, create_policy, validate_checkpoint

    if args.check_checkpoint:
        cfg = validate_checkpoint(config)
        logging.info(
            "Checkpoint/config/statistics check passed: state=19, action=7, chunk=%s", cfg.chunk_size
        )
    elif args.check_model:
        check_model(config)
    else:
        # Resolve weights, tokenizer and device errors before connecting the robot.
        policy = create_policy(config)
        if config["video"].get("visualize", False):
            import cv2

            if any(
                line.strip().startswith("GUI:") and "NONE" in line
                for line in cv2.getBuildInformation().splitlines()
            ):
                raise RuntimeError(
                    "OpenCV has no GUI backend; use --no-visualize or install a GUI-enabled OpenCV build"
                )
        from .ur import Inference

        inference = Inference(config, policy)
        inference.run()


if __name__ == "__main__":
    main()
