# Vendor SDK names intentionally match RTDE.
# ruff: noqa: N802
# Copyright 2026 The RLinf Authors.
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

"""RTDE and PGE vendor boundaries used by UR5e contract tests."""

import threading
import time
import types
from typing import Any

import numpy as np
import pytest


def module(name, **attributes):
    result = types.ModuleType(name)
    result.__dict__.update(attributes)
    return result


def modules() -> dict[str, types.ModuleType]:
    """Return stateful RTDE and serial fakes sharing each robot's pose."""
    states: dict[str, dict[str, Any]] = {}

    class Receive:
        def __init__(self, ip: str, frequency: float = -1.0, *args):
            self.state = states.setdefault(
                ip,
                {
                    "pose": [0.4, 0.0, 0.3, 0.0, 0.0, 0.0],
                    "q": [0.0] * 6,
                    "moves": [],
                    "forces": [],
                    "events": [],
                },
            )
            self.state[f"{type(self).__name__.lower()}_frequency"] = frequency
            self.connected = True

        def isConnected(self):
            return self.connected

        def getActualTCPPose(self):
            return self.state["pose"]

        def getActualTCPSpeed(self):
            return self.state.get("speed", [0.0] * 6)

        def getActualQ(self):
            return self.state["q"]

        def getActualQd(self):
            return [0.0] * 6

        def getActualTCPForce(self):
            return [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        def disconnect(self):
            self.connected = False

    class Control(Receive):
        fail_move = False
        fail_force = False
        safe = True

        def isPoseWithinSafetyLimits(self, pose):
            return self.safe

        def getTCPOffset(self):
            return self.state.get("tcp_offset", [0.0] * 6)

        def servoStop(self):
            self.state["events"].append("servoStop")
            return True

        def zeroFtSensor(self):
            self.state["events"].append("zeroFtSensor")
            return True

        def forceModeSetGainScaling(self, gain):
            self.state["gain"] = gain
            return True

        def setPayload(self, mass, cog):
            self.state["events"].append("setPayload")
            self.state["payload"] = (mass, list(cog))
            return True

        def forceMode(self, frame, selection, wrench, mode, limits):
            self.state["forces"].append((list(frame), list(selection), list(wrench), mode, list(limits)))
            return not self.fail_force

        def forceModeStop(self):
            self.state["events"].append("forceModeStop")
            return True

        def moveL(self, pose, speed, acceleration):
            self.state["events"].append("moveL")
            self.state["moves"].append((list(pose), speed, acceleration))
            if self.fail_move:
                return False
            self.state["pose"] = list(pose)
            return True

        def moveJ(self, joints, speed, acceleration):
            self.state["events"].append("moveJ")
            self.state["q"] = list(joints)
            return True

        def stopScript(self):
            self.state["events"].append("stopScript")
            return True

    class PGE:
        instances = []

        def __init__(self, port):
            self.port = port
            self.raw = 1000
            self.closed = False
            self.ser = types.SimpleNamespace(close=self.close_serial)
            self.commands = []
            type(self).instances.append(self)

        def close_serial(self):
            self.closed = True

        def init_feedback(self):
            return True

        def set_force(self, force):
            self.force = force

        def set_vel(self, speed):
            self.speed = speed

        def read_pos(self):
            return self.raw

        def set_pos(self, val, blocking=True):
            self.write_uart(modbus_high_addr=0x01, modbus_low_addr=0x03, val=val)

        def write_uart(self, **command):
            self.commands.append(command)
            self.raw = command["val"]

    return {
        "rtde_control": module("rtde_control", RTDEControlInterface=Control, states=states),
        "rtde_receive": module("rtde_receive", RTDEReceiveInterface=Receive),
        "pyDHgripper": module("pyDHgripper", PGE=PGE),
    }


@pytest.fixture
def native_ur_force_engine(monkeypatch, tmp_path):
    for package in ("lerobot.cameras", "pyDHgripper", "av", "crcmod"):
        pytest.importorskip(package)

    from lerobot.scripts.ur5e_sft_eval import ur
    from lerobot.scripts.ur5e_sft_eval.recording import Recorder
    from lerobot.scripts.ur5e_sft_eval.utils import FpsCounter

    sdk = modules()
    engine = object.__new__(ur.Inference)
    engine.__dict__.update(
        rtde_r=sdk["rtde_receive"].RTDEReceiveInterface("continuous-native"),
        rtde_c=sdk["rtde_control"].RTDEControlInterface("continuous-native"),
        continuous_tracking=True,
        _force_controller=None,
        _last_force_report=0.0,
        control_space="force",
        debug=False,
        fix_gripper=True,
        init_gripper=False,
        reference_frame="base",
        action_fps=50,
        kp=1200,
        kd=600,
        kp_rot=20,
        kd_rot=4,
        rtde_freq=100,
        pos_delta=0.048,
        vel_delta=4.0,
        use_control_frame=False,
        task_frame=[0.0] * 6,
        select_vector=[1, 1, 1, 0, 0, 1],
        force_type=2,
        force_limit=[2] * 6,
        filter_zero_output=False,
        _zero_force_lock_active=False,
        show_target_error=False,
        action_idle_hold_enabled=True,
        action_idle_position_threshold=0.0001,
        action_idle_rotation_threshold=0.001,
        action_timing_enabled=True,
        save_logs=True,
        _timing_path=tmp_path / "native-timing.jsonl",
        _velocity=0.2,
        _acceleration=0.2,
        payload_mass=1.601,
        payload_cog=[0.011, -0.002, 0.052],
        pre_reset_tcp_pose=None,
        init_tcp_pose=[0.4, 0.0, 0.3, 0.0, 0.0, 0.0],
        init_pose_range=[0.0] * 6,
        _episode_time_printer_stop=None,
        _episode_time_printer_thread=None,
        _ep_idx=1,
        num_episodes=1,
        ep_timeout=60,
        _success_episodes=0,
        _failed_episodes=0,
        _reset_requested=threading.Event(),
        _success_requested=threading.Event(),
        use_target_success=False,
        fps_action=FpsCounter("test"),
    )
    engine._clear_action_idle_hold()
    engine.recorder = object.__new__(Recorder)
    engine.recorder.save_logs = False
    engine.recorder.episode_results = []
    monkeypatch.setattr(ur, "PAYLOAD_SETTLE_TIME_S", 0)
    monkeypatch.setattr(ur, "FT_ZERO_SETTLE_TIME_S", 0)
    yield engine, sdk
    engine.close()


@pytest.mark.parametrize("chunk_size", [3, 5])
def test_vla_precision_ur_executes_and_records_entire_prediction_chunk(native_ur_force_engine, chunk_size):
    import queue

    engine, _ = native_ur_force_engine
    engine.continuous_tracking = False
    engine._ep_steps = 0
    engine.recorder.save_logs = True
    engine.recorder.queue_log = queue.Queue()
    engine.recorder.queue_vis = queue.Queue()
    actions = np.full((chunk_size, 7), 0.001)
    obs = {"prompt": "test", "observation/state": np.zeros(19)}

    engine._submit_step({"actions": actions}, obs, infer_idx=7)

    assert len(engine.rtde_c.state["forces"]) == chunk_size
    record = engine.recorder.queue_log.get_nowait()
    assert record["action_horizon"] == chunk_size
    assert record["infer_index"] == 7
    np.testing.assert_allclose(record["actions"], actions[:, :6])
    assert engine.recorder.queue_vis.get_nowait() is obs


def test_native_ur_continuous_force_tracks_state_and_pauses_for_reset(native_ur_force_engine, monkeypatch):
    engine, sdk = native_ur_force_engine
    state = engine.rtde_c.state
    zero_sensor = sdk["rtde_control"].RTDEControlInterface.zeroFtSensor

    def zero_while_paused(control):
        assert engine._force_controller is None or not engine._force_controller.running
        return zero_sensor(control)

    monkeypatch.setattr(sdk["rtde_control"].RTDEControlInterface, "zeroFtSensor", zero_while_paused)
    for _ in range(2):
        before = len(state["forces"])
        engine.reset_episode()
        time.sleep(0.025)
        assert len(state["forces"]) == before
        engine._start_episode()
        engine.execute_actions(np.array([[0.01, 0, 0, 0, 0, 0]]))
        np.testing.assert_allclose(state["forces"][-1][2][:3], [12, 0, 0], atol=1e-8)
        controller = engine._force_controller
        target_sequence = controller._sequence
        with controller.io_lock:
            state["pose"][0] = 0.405
        deadline = time.monotonic() + 1
        while not np.isclose(state["forces"][-1][2][0], 6) and time.monotonic() < deadline:
            time.sleep(0.005)
        np.testing.assert_allclose(state["forces"][-1][2][:3], [6, 0, 0], atol=1e-8)
        assert controller._sequence == target_sequence
        before = len(state["forces"])
        time.sleep(0.035)
        assert len(state["forces"]) > before
        engine.stop_force()
        assert not controller.running
        before = len(state["forces"])
        time.sleep(0.025)
        assert len(state["forces"]) == before
    engine._start_episode()
    engine.close()
    before = len(state["forces"])
    time.sleep(0.025)
    assert len(state["forces"]) == before
    assert not controller.running
    import json

    records = [json.loads(row) for row in engine._timing_path.read_text().splitlines()]
    assert any(row["event"] == "controller" and row["samples"] > 1 for row in records)


@pytest.mark.parametrize("outcome", ["manual_success", "timeout", "keyboard_reset"])
def test_native_ur_continuous_force_stops_at_episode_end(native_ur_force_engine, outcome):
    engine, _ = native_ur_force_engine
    engine._start_episode()
    engine.execute_actions(np.array([[0.01, 0, 0, 0, 0, 0]]))
    if outcome == "manual_success":
        engine._success_requested.set()
    elif outcome == "keyboard_reset":
        engine._reset_requested.set()
    else:
        engine._ep_start -= 61
    assert engine._finish_episode_if_needed()
    assert engine.recorder.episode_results[-1]["status"] == outcome
    before = len(engine.rtde_c.state["forces"])
    time.sleep(0.025)
    assert len(engine.rtde_c.state["forces"]) == before
    assert not engine._force_controller.running


def test_native_ur_continuous_force_preserves_zero_filter(native_ur_force_engine):
    engine, _ = native_ur_force_engine
    engine._start_episode()
    engine.filter_zero_output = True
    engine.action_idle_rotation_threshold = -1
    engine.execute_actions(np.array([[0.01, 0, 0, 0, 0, 0]]))
    engine.execute_actions(np.zeros((1, 6)))
    zero_start = len(engine.rtde_c.state["forces"])
    time.sleep(0.025)
    assert len(engine.rtde_c.state["forces"]) > zero_start
    for command in engine.rtde_c.state["forces"][zero_start:]:
        assert command[1] == [0] * 6
        np.testing.assert_array_equal(command[2], np.zeros(6))
    engine.execute_actions(np.array([[0.02, 0, 0, 0, 0, 0]]))
    assert engine.rtde_c.state["forces"][-1][1] == engine.select_vector


@pytest.mark.parametrize("operation", ["forceMode", "forceModeStop"])
def test_native_ur_continuous_force_failure_blocks_reset(native_ur_force_engine, monkeypatch, operation):
    engine, sdk = native_ur_force_engine
    engine._start_episode()
    state = engine.rtde_c.state
    state["events"].clear()
    with monkeypatch.context() as patch:
        patch.setattr(sdk["rtde_control"].RTDEControlInterface, operation, lambda *args: False)
        if operation == "forceMode":
            with pytest.raises(RuntimeError, match="force controller failed"):
                engine.execute_actions(np.array([[0.01, 0, 0, 0, 0, 0]]))
        with pytest.raises(RuntimeError, match="force controller failed"):
            engine.reset_episode()
    assert "moveL" not in state["events"]
    assert "zeroFtSensor" not in state["events"]


def test_native_model_chunk_and_raw_images():
    from types import SimpleNamespace

    from lerobot.scripts.ur5e_sft_eval.policy import EvaluationPolicy
    from lerobot.scripts.ur5e_sft_eval.ur import Inference

    actions = np.arange(350).reshape(50, 7)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    engine = object.__new__(Inference)
    engine.fix_gripper = False
    observation = engine._transfer_obs_state(
        {
            "tcp_pose": np.zeros(6),
            "tcp_speed": np.zeros(6),
            "tcp_force": np.zeros(6),
            "gripper_position": 1.0,
            "exterior_image": image,
            "wrist_image": image,
            "prompt": "task",
        }
    )
    assert observation["observation/image"] is image
    assert observation["observation/state"].shape == (19,)
    seen = []

    def infer(obs):
        seen.append(obs)
        return {"actions": actions}

    result = EvaluationPolicy(SimpleNamespace(infer=infer), 10).infer(observation)
    np.testing.assert_array_equal(result["actions"], actions[:10])
    assert seen == [observation]
