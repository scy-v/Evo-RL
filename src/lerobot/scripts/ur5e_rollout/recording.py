# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import sys
import termios
from typing import Any

from send2trash import send2trash

from lerobot.utils.constants import HF_LEROBOT_HOME

from .robot import UR5e, UR5eConfig
from .teleop import UR5eTeleop, UR5eTeleopConfig

logger = logging.getLogger("lerobot.ur5e_rollout")


class RecordConfig:
    def __init__(self, cfg: dict[str, Any]):
        storage = cfg["storage"]
        task = cfg["task"]
        time = cfg["time"]
        cam = cfg.get("cameras", {})
        robot = cfg["robot"]
        teleop = cfg.get("teleop", {})
        force_cfg = robot.get("force_mode", {})
        position_cfg = robot.get("position_mode", {})
        payload_cfg = robot.get("payload", {})

        # global config
        self.repo_id: str = cfg["repo_id"]
        self.fps: float = cfg.get("fps", 15)
        self.dataset_path: str = HF_LEROBOT_HOME / self.repo_id
        self.user_info: str = cfg.get("user_notes")

        # robot config
        self.robot_ip: str = robot["ip"]
        self.gripper_port: str = robot["gripper_port"]
        self.use_gripper: bool = robot.get("use_gripper", True)
        self.init_gripper: bool = robot.get("init_gripper", True)
        self.close_threshold: float = robot.get("close_threshold", 0.7)
        self.gripper_reverse: bool = robot.get("gripper_reverse", False)
        self.gripper_bin_threshold: float = robot.get("gripper_bin_threshold", 0.98)
        self.gripper_force: int = robot.get("gripper_force", 70)
        self.gripper_speed: int = robot.get("gripper_speed", 60)
        self.control_space: str = robot.get("control_space", "position")
        self.reference_frame: str = robot.get("reference_frame", "base")
        self.control_frame_euler_deg: list = teleop.get(
            "control_frame_euler_deg",
            force_cfg.get("control_frame_euler_deg", [0.0, 0.0, 0.0]),
        )
        self.debug: bool = robot.get("debug", False)
        self.kp: int = force_cfg.get("kp", 2000)
        self.kd: int = force_cfg.get("kd", 200)
        self.kp_rot: int = force_cfg.get("kp_rot", 4000)
        self.kd_rot: int = force_cfg.get("kd_rot", 800)
        self.hold_current_pose_on_idle: bool = force_cfg.get(
            "hold_current_pose_on_idle",
            force_cfg.get("hold_last_target_pose", False),
        )
        self.rtde_freq: int = force_cfg.get("rtde_freq", 125)
        self.select_vector: list = force_cfg.get("select_vector", [1, 1, 1, 1, 1, 1])
        self.force_limit: list = force_cfg.get("force_limit", [2, 2, 2, 2, 2, 2])
        self.pos_delta: float = force_cfg.get("pos_delta", 0.2)
        self.vel_delta: float = force_cfg.get("vel_delta", 0.2)
        self.gain_scale: float = force_cfg.get("gain_scale", 1.5)
        self.speed: float = position_cfg.get("speed", 0.5)
        self.acceleration: float = position_cfg.get("acceleration", 0.5)
        self.servo_time: float = position_cfg.get("servo_time", 0.1)
        self.lookahead_time: float = position_cfg.get("lookahead_time", 0.1)
        self.gain: int = position_cfg.get("gain", 300)
        self.payload_mass: float = payload_cfg.get("mass", 1.601)
        self.payload_cog: list = payload_cfg.get("cog", [0.011, -0.002, 0.052])
        self.init_pose: list[float] = robot["init_pose"]
        legacy_init_pos_range = robot.get("init_pos_range", [0.0, 0.0, 0.0])
        self.init_pose_range: list[float] = robot.get(
            "init_pose_range",
            [*legacy_init_pos_range, 0.0, 0.0, 0.0],
        )

        # teleop config
        self.teleop_position_step_size: float = teleop.get(
            "position_step_size", teleop.get("step_size", 0.01)
        )
        self.teleop_force_step_size: float = teleop.get("force_step_size", teleop.get("step_size", 0.048))
        self.teleop_force_fine_step_size: float = teleop.get(
            "force_fine_step_size", self.teleop_force_step_size
        )
        self.teleop_position_rot_step_size: float = teleop.get(
            "position_rot_step_size", teleop.get("rot_step_size", 0.02)
        )
        self.teleop_force_rot_step_size: float = teleop.get(
            "force_rot_step_size", teleop.get("rot_step_size", 0.05)
        )
        self.teleop_force_fine_rot_step_size: float = teleop.get(
            "force_fine_rot_step_size", self.teleop_force_rot_step_size
        )
        if self.control_space == "position":
            self.teleop_step_size: float = self.teleop_position_step_size
            self.teleop_rot_step_size: float = self.teleop_position_rot_step_size
            self.teleop_alternate_step_size: float | None = None
            self.teleop_alternate_rot_step_size: float | None = None
        elif self.control_space == "force":
            self.teleop_step_size: float = self.teleop_force_step_size
            self.teleop_rot_step_size: float = self.teleop_force_rot_step_size
            self.teleop_alternate_step_size: float | None = self.teleop_force_fine_step_size
            self.teleop_alternate_rot_step_size: float | None = self.teleop_force_fine_rot_step_size
        else:
            raise ValueError(f"Unsupported control_space: {self.control_space}")

        # task config
        self.num_episodes: int = task.get("num_episodes", 1)
        self.display: bool = task.get("display", True)
        self.task_description: str = task.get("description", task.get("task_description", "default task"))
        self.resume: bool = task.get("resume", False)
        self.resume_dataset: str = task.get("resume_dataset", self.repo_id)

        # time config
        self.episode_time_sec: int = time.get("episode_time_sec", 60)
        self.reset_time_sec: int = time.get("reset_time_sec", 10)
        self.save_mera_period: int = time.get("save_meta_period", time.get("save_mera_period", 1))

        # cameras config
        cameras_enabled = cam.get("enabled", True)
        if not isinstance(cameras_enabled, bool):
            raise ValueError("cameras.enabled must be true or false.")
        self.cameras_enabled: bool = cameras_enabled
        self.wrist_cam_serial: str | None = cam.get("wrist_cam_serial")
        self.exterior_cam_serial: str | None = cam.get("exterior_cam_serial")
        self.width: int = cam.get("width", 640)
        self.height: int = cam.get("height", 480)
        self.dummy_value: int = cam.get("dummy_value", 0)

        # storage config
        self.push_to_hub: bool = storage.get("push_to_hub", False)


def make_camera_configs(record_cfg: RecordConfig) -> dict:
    camera_names = ("wrist_image", "exterior_image")
    if not record_cfg.cameras_enabled:
        from .robot import DummyCameraConfig

        logging.info(
            "====== [CAM] RealSense disabled; using dummy frames for %s ======",
            ", ".join(camera_names),
        )
        return {
            name: DummyCameraConfig(
                fps=record_cfg.fps,
                width=record_cfg.width,
                height=record_cfg.height,
                value=record_cfg.dummy_value,
            )
            for name in camera_names
        }

    if not record_cfg.wrist_cam_serial or not record_cfg.exterior_cam_serial:
        raise ValueError("Both camera serial numbers are required when cameras.enabled is true.")

    # Import RealSense only when real cameras are enabled.
    from lerobot.cameras.configs import ColorMode, Cv2Rotation
    from lerobot.cameras.realsense.camera_realsense import RealSenseCameraConfig

    return {
        "wrist_image": RealSenseCameraConfig(
            serial_number_or_name=record_cfg.wrist_cam_serial,
            fps=record_cfg.fps,
            width=record_cfg.width,
            height=record_cfg.height,
            color_mode=ColorMode.RGB,
            use_depth=False,
            rotation=Cv2Rotation.NO_ROTATION,
        ),
        "exterior_image": RealSenseCameraConfig(
            serial_number_or_name=record_cfg.exterior_cam_serial,
            fps=record_cfg.fps,
            width=record_cfg.width,
            height=record_cfg.height,
            color_mode=ColorMode.RGB,
            use_depth=False,
            rotation=Cv2Rotation.NO_ROTATION,
        ),
    }


def handle_incomplete_dataset(dataset_path) -> bool:
    if dataset_path.exists():
        logger.info(f"====== [WARNING] Detected an incomplete dataset folder: {dataset_path} ======")
        if sys.stdin.isatty():
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
        ans = input("Move the entire dataset to Trash? [y/N]: ").strip().lower()
        if ans == "y":
            logger.info(f"====== [TRASH] Moving folder to trash: {dataset_path} ======")
            send2trash(str(dataset_path))
            logger.info("====== [DONE] Incomplete dataset folder moved to trash successfully. ======")
            return False
        else:
            logger.info("====== [KEEP] Incomplete dataset folder retained, please check manually. ======")
            return True
    return False


def make_hardware(record_cfg: RecordConfig):
    # Create the robot and teleoperator configurations
    camera_config = make_camera_configs(record_cfg)
    teleop_config = UR5eTeleopConfig(
        use_gripper=record_cfg.use_gripper,
        init_gripper=record_cfg.init_gripper,
        step_size=record_cfg.teleop_step_size,
        rot_step_size=record_cfg.teleop_rot_step_size,
        alternate_step_size=record_cfg.teleop_alternate_step_size,
        alternate_rot_step_size=record_cfg.teleop_alternate_rot_step_size,
        reference_frame=record_cfg.reference_frame,
        control_frame_euler_deg=record_cfg.control_frame_euler_deg,
        select_vector=record_cfg.select_vector,
    )
    robot_config = UR5eConfig(
        robot_ip=record_cfg.robot_ip,
        gripper_port=record_cfg.gripper_port,
        cameras=camera_config,
        use_gripper=record_cfg.use_gripper,
        init_gripper=record_cfg.init_gripper,
        close_threshold=record_cfg.close_threshold,
        gripper_reverse=record_cfg.gripper_reverse,
        gripper_bin_threshold=record_cfg.gripper_bin_threshold,
        gripper_force=record_cfg.gripper_force,
        gripper_speed=record_cfg.gripper_speed,
        control_space=record_cfg.control_space,
        debug=record_cfg.debug,
        kp=record_cfg.kp,
        kd=record_cfg.kd,
        kp_rot=record_cfg.kp_rot,
        kd_rot=record_cfg.kd_rot,
        hold_current_pose_on_idle=record_cfg.hold_current_pose_on_idle,
        rtde_freq=record_cfg.rtde_freq,
        select_vector=record_cfg.select_vector,
        force_limit=record_cfg.force_limit,
        pos_delta=record_cfg.pos_delta,
        vel_delta=record_cfg.vel_delta,
        gain_scale=record_cfg.gain_scale,
        speed=record_cfg.speed,
        acceleration=record_cfg.acceleration,
        servo_time=record_cfg.servo_time,
        lookahead_time=record_cfg.lookahead_time,
        gain=record_cfg.gain,
        payload_mass=record_cfg.payload_mass,
        payload_cog=record_cfg.payload_cog,
        init_pose=record_cfg.init_pose,
        init_pose_range=record_cfg.init_pose_range,
        reference_frame=record_cfg.reference_frame,
        control_frame_euler_deg=record_cfg.control_frame_euler_deg,
    )
    # Initialize the robot and teleoperator
    robot = UR5e(robot_config)
    teleop = UR5eTeleop(teleop_config)
    teleop.set_robot(robot)

    return robot, teleop
