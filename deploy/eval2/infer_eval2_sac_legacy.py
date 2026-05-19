"""
Standalone inference for the EVAL2 SO101 colour-conditional deliverable
(squint-native-iso, tag eval2_final, ckpt md5 a67cd585384c3b4cf03e038ae47d8461).

EVAL2 = TWO cubes ADJACENT side-by-side (goal + 1 distractor of a different
colour). Tell the policy which colour to pick via --goal_color; it must put
THAT cube in the bowl and leave the distractor out.
Contract verified from the checkpoint weights + the Eval2 env:
  - rgb   : wrist camera -> (1,16,16,3) uint8 (center-crop, 128 then 16)
  - state : (1,18) = [measured_qpos(6), controller_target_qpos(6), goal_colour_onehot(6)]
  - action: (6,) in [-1,1], pd_joint_target_delta_pos, 10 Hz
Same corrected infer.py base as Eval1 (10 Hz not 30, delta caps [0.1*5,0.2],
REST_QPOS wrist_roll = -pi/2, robot driver byte-identical). The only
difference vs Eval1: state is 18 (goal-colour one-hot) ÔÇö auto-detected from
the checkpoint, so --goal_color IS used here.

Goal colours (COLOR_PALETTE order): 0 red  1 blue  2 green  3 yellow  4 purple  5 orange

Everything is in this one file: the policy network, the obs/action contract,
and the robot driver. The only repo file you need is this script + the
checkpoint. Dependencies: torch, numpy, opencv-python, lerobot[feetech].

Usage:
    python infer_eval2.py --checkpoint eval2_ckpt.pt --goal_color 0 --action_scale 0.1
"""
import os
# Windows: lerobot 0.4.3 forces cv2.CAP_MSMF which hangs/fails on this webcam.
# Bias OpenCV away from MSMF before importing cv2.
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_DSHOW"] = "100"

import argparse
import sys
import time
from pathlib import Path

_DEPLOY = Path(__file__).resolve().parent.parent
if str(_DEPLOY) not in sys.path:
    sys.path.insert(0, str(_DEPLOY))
from homes import get_home_rad  # noqa: E402
from robot_calibration import LOCAL_CALIBRATION_DIR, make_so101_follower_config  # noqa: E402
from sac_infer_home import add_sac_home_cli, sac_maybe_home  # noqa: E402

import cv2

# Subclass cv2.VideoCapture so any int-only or CAP_ANY/CAP_MSMF call gets
# rerouted to CAP_DSHOW (verified working on this PC). Subclass keeps the
# isinstance(x, cv2.VideoCapture) check in lerobot working.
_OriginalVideoCapture = cv2.VideoCapture
_BAD_BACKENDS = {int(cv2.CAP_ANY), int(cv2.CAP_MSMF)}

class _DShowVideoCapture(_OriginalVideoCapture):
    def __init__(self, *args, **kwargs):
        if "apiPreference" in kwargs:
            if int(kwargs["apiPreference"]) in _BAD_BACKENDS:
                kwargs["apiPreference"] = cv2.CAP_DSHOW
            super().__init__(*args, **kwargs)
            return
        if len(args) == 1 and isinstance(args[0], int):
            super().__init__(args[0], cv2.CAP_DSHOW)
        elif len(args) == 2 and isinstance(args[0], int) and int(args[1]) in _BAD_BACKENDS:
            super().__init__(args[0], cv2.CAP_DSHOW)
        else:
            super().__init__(*args, **kwargs)

cv2.VideoCapture = _DShowVideoCapture

import numpy as np
import torch
import torch.nn as nn

from lerobot.robots.utils import make_robot_from_config
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.motors.motors_bus import MotorNormMode

try:
    import rerun as rr
except ImportError:
    rr = None

JOINT_NAMES = ["pan", "lift", "elbow", "wrist_flex", "wrist_roll", "gripper"]

# ´┐¢ÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòù
# Ôòæ  EDIT THESE for your robot                                                Ôòæ
# ÔòÜÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòØ
ROBOT_PORT = "COM3"                       # Windows COM port for the SO-101 follower
CAMERA_INDEX = 1                          # OpenCV index for the wrist camera on this PC
# Calibration: deploy/calibration/so101_follower.json

# ÔöÇÔöÇ Contract constants (must match the training env) ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
IMAGE_SIZE = 16          # CNN input H=W
SIM_CAM_SIZE = 128       # sim wrist-camera resolution (intermediate resize)
N_COLORS = 6             # goal-color one-hot length (UNUSED for Eval1 state=12)
# EVAL1 = 10 Hz. The Eval1 deliverable was trained at control_freq=10 with a
# 10 Hz-calibrated actuator (delay=1 step, lag a=0.645). Deploying at 30 Hz
# would 3x-mismatch the modelled latency -> DO NOT change this.
CONTROL_HZ = 30

# Per-joint delta caps (rad/step) at 30 Hz: 1/3 of the 10 Hz training bounds
# (arm 0.1 / gripper 0.2 at 10 Hz -> 0.0333 / 0.0667 at 30 Hz) so per-second
# motion stays equivalent. Actuator-lag mismatch isn't compensated -- the
# teammate's note explicitly warns 30 Hz triples the modelled latency.
DELTA_CAP = np.array([0.0333, 0.0333, 0.0333, 0.0333, 0.0333, 0.0667], dtype=np.float32)
# Joint limits from so101.urdf, order: pan, lift, elbow, wrist_flex, wrist_roll, gripper.
JOINT_LOWER = np.array([-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.174533])
JOINT_UPPER = np.array([1.91986, 1.74533, 1.69, 1.65806, 2.84121, 2.0944])
# SO101 "start" keyframe (envs/robot/so101.py). The Eval1 env seeds rest_qpos
# AND the controller target here, so deployed state[6:12] starts from this.
# NOTE wrist_roll = -pi/2 (generic infer.py wrongly used 0.0 for Eval1).
# Rest poses: deploy/homes.py (default preset eval1_sac_legacy)


# ´┐¢ÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòù
# Ôòæ  Robot driver ÔÇö wraps a LeRobot SO101 follower                            Ôòæ
# ÔòÜÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòØ
def create_real_robot():
    config = make_so101_follower_config(
        ROBOT_PORT,
        use_degrees=True,
        cameras={"base_camera": OpenCVCameraConfig(
            index_or_path=CAMERA_INDEX, fps=30, width=640, height=480,
        )},
        calibration_dir=LOCAL_CALIBRATION_DIR,
    )
    return make_robot_from_config(config)


class RealRobotAgent:
    """Minimal driver. Handles the unit conversions the policy contract needs:
    joint positions sim-radians <-> servo-degrees, and the gripper's separate
    sim range (-10┬░..120┬░) <-> servo range (-62.5┬░..64.62┬░)."""

    def __init__(self, robot):
        self.real_robot = robot
        self._cached_qpos = None
        self._motor_keys = None
        # gripper mapping (measured): sim -10┬░..120┬░  <->  servo -62.5┬░..64.62┬░
        self._g_sim_min, self._g_sim_max = -10.0, 120.0
        self._g_servo_min, self._g_servo_max = -60.13, 66.73
        self._g_sim_range = self._g_sim_max - self._g_sim_min
        self._g_servo_range = self._g_servo_max - self._g_servo_min
        robot.bus.motors["gripper"].norm_mode = MotorNormMode.DEGREES

    def get_qpos(self):
        """Measured joint angles in sim radians, shape (1, 6)."""
        if self._cached_qpos is not None:
            return self._cached_qpos.clone()
        deg = self.real_robot.bus.sync_read("Present_Position")
        servo = deg["gripper"]                                          # gripper: servo deg -> sim deg
        deg["gripper"] = (servo - self._g_servo_min) / self._g_servo_range * self._g_sim_range + self._g_sim_min
        if self._motor_keys is None:
            self._motor_keys = list(deg.keys())
        flat = np.array([deg[k] for k in self._motor_keys], dtype=np.float32)
        self._cached_qpos = torch.deg2rad(torch.from_numpy(flat)).unsqueeze(0)
        return self._cached_qpos.clone()

    def set_target_qpos(self, qpos):
        """Send a joint-angle target (sim radians) to the servos."""
        self._cached_qpos = None
        deg = torch.rad2deg(torch.as_tensor(qpos, dtype=torch.float32).flatten())
        cmd = {f"{self._motor_keys[i]}.pos": float(deg[i]) for i in range(len(deg))}
        sim_deg = cmd["gripper.pos"]                                    # gripper: sim deg -> servo deg
        cmd["gripper.pos"] = (sim_deg - self._g_sim_min) / self._g_sim_range * self._g_servo_range + self._g_servo_min
        self.real_robot.send_action(cmd)

    def reset(self, qpos, freq=30, max_rad_per_step=0.025):
        """Move smoothly to qpos by ramping the target a little each tick."""
        qpos = torch.as_tensor(qpos, dtype=torch.float32).flatten()
        target = self.get_qpos().flatten()
        for _ in range(int(20 * freq)):
            delta = (qpos - target).clamp(-max_rad_per_step, max_rad_per_step)
            if torch.linalg.norm(delta) <= 1e-4:
                break
            target = target + delta
            self.set_target_qpos(target)
            time.sleep(1.0 / freq)

    def capture_sensor_data(self):
        self._sensor_data = {}
        for name, cam in self.real_robot.cameras.items():
            frame = np.asarray(cam.async_read())                        # (H, W, 3) uint8 RGB
            self._sensor_data[name] = {"rgb": torch.from_numpy(frame).unsqueeze(0)}

    def get_sensor_data(self):
        return self._sensor_data


# ´┐¢ÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòù
# Ôòæ  Policy network ÔÇö architecture must match the checkpoint exactly          Ôòæ
# ÔòÜÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòØ
class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=1), nn.ReLU(),
            nn.Flatten(),
        )

    def forward(self, rgb_uint8):           # (B, 16, 16, 3) uint8
        x = rgb_uint8.permute(0, 3, 1, 2).float()
        x = x / 255.0 - 0.5
        return self.conv(x)                 # (B, 1024)


class Projection(nn.Module):
    def __init__(self, n_state):
        super().__init__()
        self.rgb_proj = nn.Sequential(nn.Linear(1024, 50), nn.LayerNorm(50), nn.Tanh())
        self.state_proj = nn.Sequential(nn.Linear(n_state, 256), nn.LayerNorm(256), nn.ReLU())

    def forward(self, rgb_feat, state):
        return torch.cat([self.rgb_proj(rgb_feat), self.state_proj(state)], dim=-1)


class Actor(nn.Module):
    def __init__(self, n_state=18, n_act=6):
        super().__init__()
        self.proj = Projection(n_state)
        self.fc = nn.Sequential(
            nn.Linear(306, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
        )
        self.fc_mean = nn.Linear(256, n_act)
        self.fc_logstd = nn.Linear(256, n_act)
        self.register_buffer("action_scale", torch.ones(n_act))
        self.register_buffer("action_bias", torch.zeros(n_act))

    def forward(self, rgb_feat, state):     # deterministic eval action Ôêê [-1, 1]
        x = self.fc(self.proj(rgb_feat, state))
        return torch.tanh(self.fc_mean(x)) * self.action_scale + self.action_bias


# ÔöÇÔöÇ Observation / action helpers ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
def preprocess_image(rgb):
    """Real camera frame (1,H,W,3) uint8 ÔåÆ (1,16,16,3) uint8 tensor.

    Center-crop to square, resize to the 128px sim resolution, area-downsample
    to the 16px CNN input ÔÇö same two-step path used during training.
    """
    img = rgb[0].cpu().numpy() if torch.is_tensor(rgb) else np.asarray(rgb[0])
    h, w = img.shape[:2]
    c = min(h, w)
    img = img[(h - c) // 2:(h - c) // 2 + c, (w - c) // 2:(w - c) // 2 + c]
    img = cv2.resize(img, (SIM_CAM_SIZE, SIM_CAM_SIZE), interpolation=cv2.INTER_AREA)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(img).unsqueeze(0).to(torch.uint8)


def init_viz():
    """Spawn a Rerun viewer window for live camera + joint plots."""
    if rr is None:
        raise RuntimeError("rerun not installed in this env (pip install rerun-sdk)")
    rr.init("squint_infer", spawn=True)


def log_step(step, raw_rgb, policy_rgb, qpos, target_qpos, action_raw):
    """Push one timestep to the Rerun viewer."""
    if rr is None:
        return
    rr.set_time("step", sequence=step)
    rr.log("camera/raw", rr.Image(raw_rgb))
    rr.log("camera/policy_input_16x16", rr.Image(policy_rgb))
    for i, name in enumerate(JOINT_NAMES):
        rr.log(f"joints/qpos_measured/{name}", rr.Scalars([float(qpos[i])]))
        rr.log(f"joints/qpos_target/{name}", rr.Scalars([float(target_qpos[i])]))
        rr.log(f"action_raw/{name}", rr.Scalars([float(action_raw[i])]))


def build_state(qpos, target_qpos, goal_color, bowl_xyz=None):
    """State vector for the policy.

    EVAL1 (12-d): [measured_qpos(6), controller_target_qpos(6)] ÔÇö pass
    goal_color=None (no colour conditioning).
    18-d: + goal_onehot(6). 21-d: + bowl_xyz(3). (other checkpoints)
    """
    parts = [qpos, target_qpos]
    if goal_color is not None:                       # 18/21-d colour-conditioned
        onehot = np.zeros(N_COLORS, dtype=np.float32)
        onehot[goal_color] = 1.0
        parts.append(onehot)
    if bowl_xyz is not None:
        parts.append(np.asarray(bowl_xyz, dtype=np.float32))
    vec = np.concatenate(parts).astype(np.float32)
    return torch.from_numpy(vec).unsqueeze(0)


# ´┐¢ÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòù
# Ôòæ  Main                                                                     Ôòæ
# ÔòÜÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòØ
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="path to ckpt.pt")
    p.add_argument("--goal_color", type=int, default=0, help="0 red 1 blue 2 green 3 yellow 4 purple 5 orange")
    p.add_argument("--action_scale", type=float, default=0.15, help="safety multiplier on policy action (lower = slower)")
    p.add_argument("--episode_steps", type=int, default=150, help="control steps per episode (Eval2 sim ~50 @ 10 Hz = 5s; default 150 = 15s real-world margin)")
    p.add_argument("--viz", action=argparse.BooleanOptionalAction, default=True, help="open a Rerun viewer with live camera + joint plots (--no-viz to disable)")
    p.add_argument("--n_episodes", type=int, default=0, help="if >0, run this many episodes back-to-back without waiting for Enter")
    p.add_argument("--log_dir", type=str, default=None, help="if set, dump per-step npz logs there (one file per episode)")
    p.add_argument("--bowl_xyz", type=float, nargs=3, default=[0.25, 0.10, 0.00],
                   metavar=("X", "Y", "Z"),
                   help="bowl position fed to the policy when the checkpoint expects a 21-d state (default: 0.25 0.10 0.00)")
    add_sac_home_cli(p, default_home_pose="eval1_sac_legacy")
    args = p.parse_args()

    if args.viz:
        init_viz()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load policy (only encoder + actor are needed; critic/log_alpha are training-only).
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # Detect the state-vector width the checkpoint was trained with.
    n_state_ckpt = ckpt["actor"]["proj.state_proj.0.weight"].shape[1]
    if n_state_ckpt not in (12, 18, 21):
        raise RuntimeError(f"Unsupported state size in checkpoint: {n_state_ckpt} (expected 12=Eval1, 18, or 21)")
    colour_conditioned = n_state_ckpt in (18, 21)   # 12 = Eval1, no goal one-hot
    use_bowl_xyz = n_state_ckpt == 21
    if not colour_conditioned:
        print("Eval1 checkpoint (state=12): single-cube place, NO colour conditioning "
              "(--goal_color ignored).")
    encoder = CNNEncoder().to(device).eval()
    actor = Actor(n_state=n_state_ckpt).to(device).eval()
    encoder.load_state_dict(ckpt["encoder"])
    actor.load_state_dict(ckpt["actor"])
    print(f"Loaded checkpoint (trained to step {ckpt.get('global_step', '?')}), n_state={n_state_ckpt}"
          + (f" ÔåÆ feeding bowl_xyz={args.bowl_xyz}" if use_bowl_xyz else ""))

    # Connect robot, then build the driver (it touches robot.bus on init).
    print("[debug] building robot config ...", flush=True)
    robot = create_real_robot()
    print("[debug] robot config built. Calling robot.connect() ...", flush=True)
    robot.connect()
    print("[debug] robot.connect() done. Building agent ...", flush=True)
    agent = RealRobotAgent(robot)
    print("[debug] agent ready.", flush=True)

    log_dir = Path(args.log_dir) if args.log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

    def episodes():
        if args.n_episodes > 0:
            for i in range(args.n_episodes):
                yield i
        else:
            i = 0
            while True:
                input(f"\n[Enter] start episode (goal color {args.goal_color}), Ctrl+C to quit ")
                yield i
                i += 1

    try:
        for ep in episodes():
            if args.n_episodes > 0:
                print(f"\nÔöÇÔöÇ Episode {ep + 1}/{args.n_episodes} (goal color {args.goal_color}) ÔöÇÔöÇ", flush=True)
            sac_maybe_home(agent, args)
            print(f"[home] ROLLOUT START ({args.episode_steps} steps @ "
                  f"{CONTROL_HZ} Hz target = {args.episode_steps / CONTROL_HZ:.1f}s)", flush=True)
            rollout_t0 = time.perf_counter()
            target_qpos = agent.get_qpos().cpu().numpy().flatten()

            log_qpos, log_target, log_action_raw, log_policy_rgb = [], [], [], []

            for step in range(args.episode_steps):
                t0 = time.perf_counter()

                qpos = agent.get_qpos().cpu().numpy().flatten()
                agent.capture_sensor_data()
                rgb = agent.get_sensor_data()["base_camera"]["rgb"]

                obs_rgb = preprocess_image(rgb).to(device)
                obs_state = build_state(
                    qpos, target_qpos,
                    args.goal_color if colour_conditioned else None,
                    bowl_xyz=args.bowl_xyz if use_bowl_xyz else None,
                ).to(device)

                with torch.no_grad():
                    raw_action = actor(encoder(obs_rgb), obs_state)[0].cpu().numpy()

                action = np.clip(raw_action * args.action_scale, -1.0, 1.0)
                target_qpos = np.clip(target_qpos + action * DELTA_CAP, JOINT_LOWER, JOINT_UPPER)
                agent.set_target_qpos(torch.from_numpy(target_qpos))

                if args.viz:
                    log_step(
                        step=step,
                        raw_rgb=rgb[0].cpu().numpy() if torch.is_tensor(rgb) else np.asarray(rgb[0]),
                        policy_rgb=obs_rgb[0].cpu().numpy(),
                        qpos=qpos,
                        target_qpos=target_qpos,
                        action_raw=raw_action,
                    )
                if log_dir:
                    log_qpos.append(qpos.copy())
                    log_target.append(target_qpos.copy())
                    log_action_raw.append(raw_action.copy())
                    log_policy_rgb.append(obs_rgb[0].cpu().numpy().copy())

                time.sleep(max(0.0, 1.0 / CONTROL_HZ - (time.perf_counter() - t0)))

            if log_dir:
                np.savez(
                    log_dir / f"ep{ep:03d}.npz",
                    qpos=np.stack(log_qpos),
                    target_qpos=np.stack(log_target),
                    action_raw=np.stack(log_action_raw),
                    policy_rgb=np.stack(log_policy_rgb),
                    joint_names=np.array(JOINT_NAMES),
                )
                print(f"  ÔåÆ saved {log_dir / f'ep{ep:03d}.npz'}")
            elapsed = time.perf_counter() - rollout_t0
            achieved_hz = args.episode_steps / elapsed
            print(f"Episode done ({args.episode_steps} steps in {elapsed:.2f}s "
                  f"-> achieved {achieved_hz:.1f} Hz, target {CONTROL_HZ} Hz)")
    except KeyboardInterrupt:
        print("\nQuitting.")
    finally:
        if args.home:
            agent.reset(get_home_rad(args.home_pose))
        robot.disconnect()


if __name__ == "__main__":
    main()
