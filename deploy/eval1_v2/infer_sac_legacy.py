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
IMAGE_SIZE = 16          # Legacy square CNN side (n_conv=2). Overwritten in main().
# Rectangular policy CNN input (H, W) in pixels — set in main() from checkpoint.
POLICY_IMG_H = 16
POLICY_IMG_W = 16
SIM_CAM_SIZE = 128       # sim wrist-camera resolution (intermediate resize)
N_COLORS = 6             # goal-color one-hot length (UNUSED for Eval1 state=12)
# EVAL1 = 10 Hz. The Eval1 deliverable was trained at control_freq=10 with a
# 10 Hz-calibrated actuator (delay=1 step, lag a=0.645). Deploying at 30 Hz
# would 3x-mismatch the modelled latency -> DO NOT change this.
CONTROL_HZ_DEFAULT = 30   # legacy default; runtime uses args.control_hz

# 30 Hz: 1/3 of the 10 Hz training caps so per-second speed stays equivalent.
# Combined with --action_scale 0.3 this approximately matches 0.1*0.1 motion
# per 10 Hz step. Note: actuator-lag mismatch isn't compensated.
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

    def reset(self, qpos, freq=30, max_rad_per_step=0.012):
        """Move smoothly to qpos by ramping the target a little each tick.

        Default max_rad_per_step lowered from 0.025 (~1.4 deg/step at 30Hz =
        43 deg/s) to 0.012 (~0.7 deg/step = 21 deg/s) for gentler homing
        after the motor-3 swap.
        """
        qpos = torch.as_tensor(qpos, dtype=torch.float32).flatten()
        target = self.get_qpos().flatten()
        for _ in range(int(30 * freq)):
            delta = (qpos - target).clamp(-max_rad_per_step, max_rad_per_step)
            if torch.linalg.norm(delta) <= 1e-4:
                break
            target = target + delta
            self.set_target_qpos(target)
            time.sleep(1.0 / freq)

    def servo_deg_to_sim_rad(self, servo_deg):
        """Convert a 6-vec of servo degrees (e.g. from homes.py) into the
        sim-radian space the agent's set_target_qpos expects (arm direct,
        gripper through the inverse remap)."""
        import numpy as _np
        sd = _np.asarray(servo_deg, dtype=_np.float32).flatten()
        sim_rad = _np.deg2rad(sd).astype(_np.float32).copy()
        g_servo = float(sd[5])
        g_sim_deg = (
            (g_servo - self._g_servo_min) / self._g_servo_range
            * self._g_sim_range + self._g_sim_min
        )
        sim_rad[5] = _np.deg2rad(g_sim_deg)
        return sim_rad

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
    """Two architectures supported, auto-selected by ``n_conv``:

      n_conv = 2 (Eval1/Eval2 originals):
          3->32 (4x4 s2) -> 32->64 (4x4 s1) -> Flatten
          Requires 16x16 input  ->  flatten 64 * 4 * 4 = 1024.
      n_conv = 3 (e.g. ckpt_best_*, e1*lat):
          3->32 (4x4 s2) -> 32->64 (4x4 s2) -> 64->64 (3x3 s1) -> Flatten
          Policy input HxW is inferred from ``proj.rgb_proj`` in_features (e.g. 32x32 -> 1024,
          42x32 -> 1792).
    """
    def __init__(self, n_conv: int = 2):
        super().__init__()
        if n_conv == 2:
            layers = [
                nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=1), nn.ReLU(),
                nn.Flatten(),
            ]
        elif n_conv == 3:
            layers = [
                nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
                nn.Flatten(),
            ]
        else:
            raise ValueError(f"CNNEncoder n_conv must be 2 or 3, got {n_conv}")
        self.conv = nn.Sequential(*layers)

    def forward(self, rgb_uint8):           # (B, H, W, 3) uint8
        x = rgb_uint8.permute(0, 3, 1, 2).float()
        x = x / 255.0 - 0.5
        return self.conv(x)


def _detect_encoder_arch(encoder_state: dict) -> int:
    """Return n_conv (2 or 3) from the saved encoder state dict."""
    return 3 if "conv.4.weight" in encoder_state else 2


def _parse_actor_dims(actor_sd: dict) -> tuple[int, int, int, int, int, int]:
    """Return (flatten_dim, rgb_emb_dim, state_emb_dim, fusion_dim, n_state, n_act)."""
    rgb_w = actor_sd["proj.rgb_proj.0.weight"]
    rgb_emb_dim, flat_dim = int(rgb_w.shape[0]), int(rgb_w.shape[1])
    s_w = actor_sd["proj.state_proj.0.weight"]
    state_emb_dim, n_state = int(s_w.shape[0]), int(s_w.shape[1])
    fusion_dim = rgb_emb_dim + state_emb_dim
    if int(actor_sd["fc.0.weight"].shape[1]) != fusion_dim:
        raise RuntimeError(
            f"Actor fc.0 in_features {actor_sd['fc.0.weight'].shape[1]} "
            f"!= rgb_emb+state_emb {fusion_dim}"
        )
    n_act = int(actor_sd["fc_mean.weight"].shape[0])
    return flat_dim, rgb_emb_dim, state_emb_dim, fusion_dim, n_state, n_act


def _infer_policy_hw_3conv(
    encoder: "CNNEncoder",
    target_flatten: int,
    device: str,
    *,
    max_hw: int = 128,
) -> tuple[int, int]:
    """Find (H, W) such that the 3-conv encoder output has ``target_flatten`` elements."""
    enc = encoder.to(device).eval()

    def _check(h: int, w: int) -> bool:
        x = torch.randint(0, 255, (1, h, w, 3), dtype=torch.uint8, device=device)
        try:
            with torch.no_grad():
                y = enc(x)
        except RuntimeError:
            return False
        return y.numel() == target_flatten

    # Fast paths (common Squint / e1 checkpoints — avoids O(n^2) search).
    if target_flatten == 1024 and _check(32, 32):
        return 32, 32
    if target_flatten == 1792 and _check(42, 32):
        return 42, 32

    solutions: list[tuple[int, int]] = []
    for h in range(8, max_hw + 1):
        for w in range(8, max_hw + 1):
            x = torch.randint(0, 255, (1, h, w, 3), dtype=torch.uint8, device=device)
            try:
                with torch.no_grad():
                    y = enc(x)
            except RuntimeError:
                continue
            if y.numel() == target_flatten:
                solutions.append((h, w))
    if not solutions:
        raise RuntimeError(
            f"Could not find policy image H,W for flatten_dim={target_flatten} "
            f"(3-conv). Pass --policy_image_hw H W explicitly."
        )
    if (42, 32) in solutions:
        return 42, 32
    if (32, 32) in solutions and target_flatten == 1024:
        return 32, 32
    # Prefer largest area (typical training crop); deterministic tie-break.
    solutions.sort(key=lambda t: t[0] * t[1], reverse=True)
    return solutions[0]


def _infer_policy_hw(
    encoder: "CNNEncoder",
    n_conv: int,
    target_flatten: int,
    device: str,
) -> tuple[int, int]:
    if n_conv == 2:
        if target_flatten != 1024:
            raise RuntimeError(
                f"2-conv encoder expects flatten 1024, checkpoint has {target_flatten}"
            )
        return 16, 16
    return _infer_policy_hw_3conv(encoder, target_flatten, device)


class Projection(nn.Module):
    def __init__(self, n_state: int, flatten_dim: int, rgb_emb_dim: int, state_emb_dim: int):
        super().__init__()
        self.rgb_proj = nn.Sequential(
            nn.Linear(flatten_dim, rgb_emb_dim),
            nn.LayerNorm(rgb_emb_dim),
            nn.Tanh(),
        )
        self.state_proj = nn.Sequential(
            nn.Linear(n_state, state_emb_dim),
            nn.LayerNorm(state_emb_dim),
            nn.ReLU(),
        )

    def forward(self, rgb_feat, state):
        return torch.cat([self.rgb_proj(rgb_feat), self.state_proj(state)], dim=-1)


class Actor(nn.Module):
    """SAC actor; use ``build_actor`` so projection dims match the checkpoint."""

    def forward(self, rgb_feat, state):
        x = self.fc(self.proj(rgb_feat, state))
        return torch.tanh(self.fc_mean(x)) * self.action_scale + self.action_bias


def build_actor(
    n_state: int,
    n_act: int,
    flat_dim: int,
    rgb_emb_dim: int,
    state_emb_dim: int,
) -> Actor:
    fusion_dim = rgb_emb_dim + state_emb_dim
    actor = Actor.__new__(Actor)
    nn.Module.__init__(actor)
    actor.proj = Projection(n_state, flat_dim, rgb_emb_dim, state_emb_dim)
    actor.fc = nn.Sequential(
        nn.Linear(fusion_dim, 256), nn.LayerNorm(256), nn.ReLU(),
        nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
        nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
    )
    actor.fc_mean = nn.Linear(256, n_act)
    actor.fc_logstd = nn.Linear(256, n_act)
    actor.register_buffer("action_scale", torch.ones(n_act))
    actor.register_buffer("action_bias", torch.zeros(n_act))
    return actor


def _center_crop_to_aspect(img: np.ndarray, tgt_w: int, tgt_h: int) -> np.ndarray:
    """Crop (h,w) image toward center so w/h matches tgt_w/tgt_h."""
    h, w = img.shape[:2]
    ar = tgt_w / tgt_h
    cur = w / h
    if cur > ar:
        new_w = int(round(h * ar))
        x0 = (w - new_w) // 2
        return img[:, x0:x0 + new_w]
    new_h = int(round(w / ar))
    y0 = (h - new_h) // 2
    return img[y0:y0 + new_h, :]


def preprocess_image(rgb):
    """Real camera (1,H,W,3) uint8 -> (1,policy_H,policy_W,3) for the CNN.

    Globals POLICY_IMG_H, POLICY_IMG_W set in main(). Square: legacy 128-then-down;
    rectangular: aspect crop then resize to (W,H) for OpenCV.
    """
    global POLICY_IMG_H, POLICY_IMG_W
    img = rgb[0].cpu().numpy() if torch.is_tensor(rgb) else np.asarray(rgb[0])
    ph, pw = int(POLICY_IMG_H), int(POLICY_IMG_W)
    if ph == pw:
        h, w = img.shape[:2]
        c = min(h, w)
        img = img[(h - c) // 2:(h - c) // 2 + c, (w - c) // 2:(w - c) // 2 + c]
        img = cv2.resize(img, (SIM_CAM_SIZE, SIM_CAM_SIZE), interpolation=cv2.INTER_AREA)
        img = cv2.resize(img, (ph, pw), interpolation=cv2.INTER_AREA)
    else:
        img = _center_crop_to_aspect(img, pw, ph)
        img = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(img).unsqueeze(0).to(torch.uint8)


def init_viz():
    """Spawn a Rerun viewer window for live camera + joint plots."""
    if rr is None:
        raise RuntimeError("rerun not installed in this env (pip install rerun-sdk)")
    rr.init("squint_infer", spawn=True)


def log_step(step, raw_rgb, policy_rgb, qpos, target_qpos, action_raw,
             freq_inst_hz: float | None = None,
             freq_avg_hz: float | None = None,
             target_hz: float | None = None):
    """Push one timestep to the Rerun viewer.

    freq_inst_hz : instantaneous control rate this step (1 / dt vs prev step).
    freq_avg_hz  : smoothed control rate (running average over last N steps).
    target_hz    : desired rate the loop is paced at.
    """
    if rr is None:
        return
    rr.set_time("step", sequence=step)
    rr.log("camera/raw", rr.Image(raw_rgb))
    rr.log("camera/policy_input", rr.Image(policy_rgb))
    for i, name in enumerate(JOINT_NAMES):
        rr.log(f"joints/qpos_measured/{name}", rr.Scalars([float(qpos[i])]))
        rr.log(f"joints/qpos_target/{name}", rr.Scalars([float(target_qpos[i])]))
        rr.log(f"action_raw/{name}", rr.Scalars([float(action_raw[i])]))
    # Policy-inference rate panel
    if freq_inst_hz is not None:
        rr.log("control/freq_hz_inst", rr.Scalars([float(freq_inst_hz)]))
    if freq_avg_hz is not None:
        rr.log("control/freq_hz_avg", rr.Scalars([float(freq_avg_hz)]))
    if target_hz is not None:
        rr.log("control/freq_hz_target", rr.Scalars([float(target_hz)]))


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
    p.add_argument("--episode_steps", type=int, default=450, help="control steps per episode (default 450 ~ 16-17s @ ~27 Hz observed; legacy was 150)")
    p.add_argument("--viz", action=argparse.BooleanOptionalAction, default=True, help="open a Rerun viewer with live camera + joint plots (--no-viz to disable)")
    p.add_argument("--n_episodes", type=int, default=0, help="if >0, run this many episodes back-to-back without waiting for Enter")
    p.add_argument("--log_dir", type=str, default=None, help="if set, dump per-step npz logs there (one file per episode)")
    p.add_argument("--bowl_xyz", type=float, nargs=3, default=[0.25, 0.10, 0.00],
                   metavar=("X", "Y", "Z"),
                   help="bowl position fed to the policy when the checkpoint expects a 21-d state (default: 0.25 0.10 0.00)")
    p.add_argument("--grip_force_steps", type=int, default=30,
                   help="When the policy first commands gripper-close (target < 20 sim deg), "
                        "force the gripper fully closed for this many additional steps "
                        "(arm stays policy-driven). Default 30; wrappers can override.")
    p.add_argument("--grip_pin_after_close",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="After the initial --grip_force_steps elapse, keep the gripper "
                        "pinned fully closed for the rest of the episode (default off).")
    p.add_argument("--control_hz", type=int, default=30,
                   help="Control loop rate (Hz). Default 30. Lower this if the rollout "
                        "is too jerky; raise to match training (10 Hz for original sim).")
    p.add_argument("--park_pose", default="eval1_rest",
                   help="Home preset to fold the arm into AFTER the rollout (servo deg, "
                        "converted to sim rad with gripper remap). Default 'eval1_rest' "
                        "= universal folded pose. Pass an empty string '' to disable.")
    add_sac_home_cli(p, default_home_pose="auto", allow_auto=True)
    p.add_argument("--policy_image_hw", type=int, nargs=2, metavar=("H", "W"),
                   help="Override CNN input height/width in pixels (default: inferred from "
                        "encoder + proj.rgb_proj weights).")
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
    # Auto-detect encoder depth (2 vs 3 conv) and policy tensor layout from weights.
    n_conv = _detect_encoder_arch(ckpt["encoder"])
    flat_dim, rgb_emb_dim, state_emb_dim, fusion_dim, n_state_parsed, n_act = _parse_actor_dims(
        ckpt["actor"],
    )
    if n_state_parsed != n_state_ckpt:
        raise RuntimeError(f"Inconsistent n_state: actor {n_state_parsed} vs ckpt {n_state_ckpt}")

    if args.home_pose == "auto":
        # Weight-shape metadata: friend 42x32 / 1792-dim head uses rgb_emb=75 and was
        # trained from universal physical home. Legacy sim-rest policies use rgb_emb=50.
        if rgb_emb_dim == 75 or flat_dim == 1792:
            args.home_pose = "eval1_sac_universal"
        else:
            args.home_pose = "eval1_sac_legacy"
        print(
            f"[infer] --home_pose auto -> {args.home_pose!r} "
            f"(rgb_emb_dim={rgb_emb_dim}, flatten_dim={flat_dim})",
            flush=True,
        )

    encoder = CNNEncoder(n_conv=n_conv).to(device).eval()
    encoder.load_state_dict(ckpt["encoder"])

    global POLICY_IMG_H, POLICY_IMG_W, IMAGE_SIZE
    if args.policy_image_hw is not None:
        POLICY_IMG_H, POLICY_IMG_W = int(args.policy_image_hw[0]), int(args.policy_image_hw[1])
    else:
        POLICY_IMG_H, POLICY_IMG_W = _infer_policy_hw(encoder, n_conv, flat_dim, device)
    IMAGE_SIZE = POLICY_IMG_H  # compat: square policies use H==W

    print(
        f"Detected encoder: n_conv={n_conv}, flatten_dim={flat_dim}, "
        f"policy_image_hw=({POLICY_IMG_H}, {POLICY_IMG_W}), "
        f"rgb_emb={rgb_emb_dim}, state_emb={state_emb_dim}, fusion={fusion_dim}"
    )

    actor = build_actor(n_state_ckpt, n_act, flat_dim, rgb_emb_dim, state_emb_dim).to(device).eval()
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
                  f"{args.control_hz} Hz target = {args.episode_steps / args.control_hz:.1f}s)", flush=True)
            rollout_t0 = time.perf_counter()
            target_qpos = agent.get_qpos().cpu().numpy().flatten()

            # Per-step timing for the Rerun control/freq_hz_* panel.
            t_prev_step = None
            from collections import deque as _deque
            dt_window = _deque(maxlen=15)   # rolling avg over ~0.5s @ 30Hz

            # Gripper-close grip helper:
            # When the policy commands the gripper target below threshold for the
            # first time, force the gripper fully closed for N extra steps so it
            # actually grips the cube. Arm joints stay under policy control.
            GRIP_TRIGGER_RAD = np.deg2rad(20.0)   # sim deg <20 = "closing" intent
            GRIP_FORCE_STEPS = int(args.grip_force_steps)  # forced-close steps; override via --grip_force_steps
            grip_force_remaining = 0
            grip_already_triggered = False

            log_qpos, log_target, log_action_raw, log_policy_rgb = [], [], [], []

            for step in range(args.episode_steps):
                t_obs_start = time.perf_counter()
                t0 = t_obs_start   # keep legacy name for the sleep at end

                qpos = agent.get_qpos().cpu().numpy().flatten()
                agent.capture_sensor_data()
                rgb = agent.get_sensor_data()["base_camera"]["rgb"]
                t_obs_done = time.perf_counter()

                obs_rgb = preprocess_image(rgb).to(device)
                obs_state = build_state(
                    qpos, target_qpos,
                    args.goal_color if colour_conditioned else None,
                    bowl_xyz=args.bowl_xyz if use_bowl_xyz else None,
                ).to(device)

                with torch.no_grad():
                    raw_action = actor(encoder(obs_rgb), obs_state)[0].cpu().numpy()
                t_policy_done = time.perf_counter()

                action = np.clip(raw_action * args.action_scale, -1.0, 1.0)
                target_qpos = np.clip(target_qpos + action * DELTA_CAP, JOINT_LOWER, JOINT_UPPER)

                # --- Gripper-grip helper -------------------------------------
                # First time the gripper target dips below the trigger, queue
                # GRIP_FORCE_STEPS of forced full-close. During those steps,
                # gripper joint is pinned to JOINT_LOWER[5]; arm stays policy-driven.
                # If --grip_pin_after_close is set, keep pinning forever after
                # the initial countdown elapses.
                if (not grip_already_triggered
                        and target_qpos[5] < GRIP_TRIGGER_RAD):
                    grip_force_remaining = GRIP_FORCE_STEPS
                    grip_already_triggered = True
                    suffix = " (then pin closed for rest of episode)" if args.grip_pin_after_close else ""
                    print(f"  step {step:4d}: gripper close detected, forcing "
                          f"{GRIP_FORCE_STEPS} extra close steps{suffix}", flush=True)
                if grip_force_remaining > 0:
                    target_qpos[5] = JOINT_LOWER[5]
                    grip_force_remaining -= 1
                elif args.grip_pin_after_close and grip_already_triggered:
                    target_qpos[5] = JOINT_LOWER[5]
                # -------------------------------------------------------------

                agent.set_target_qpos(torch.from_numpy(target_qpos))

                # Per-step control-rate accounting (for Rerun panel).
                t_now = time.perf_counter()
                freq_inst = None
                freq_avg = None
                if t_prev_step is not None:
                    dt = t_now - t_prev_step
                    if dt > 1e-6:
                        freq_inst = 1.0 / dt
                        dt_window.append(dt)
                        if dt_window:
                            freq_avg = len(dt_window) / sum(dt_window)
                t_prev_step = t_now

                if args.viz:
                    log_step(
                        step=step,
                        raw_rgb=rgb[0].cpu().numpy() if torch.is_tensor(rgb) else np.asarray(rgb[0]),
                        policy_rgb=obs_rgb[0].cpu().numpy(),
                        qpos=qpos,
                        target_qpos=target_qpos,
                        action_raw=raw_action,
                        freq_inst_hz=freq_inst,
                        freq_avg_hz=freq_avg,
                        target_hz=float(args.control_hz),
                    )
                if log_dir:
                    log_qpos.append(qpos.copy())
                    log_target.append(target_qpos.copy())
                    log_action_raw.append(raw_action.copy())
                    log_policy_rgb.append(obs_rgb[0].cpu().numpy().copy())

                time.sleep(max(0.0, 1.0 / args.control_hz - (time.perf_counter() - t0)))

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
                  f"-> achieved {achieved_hz:.1f} Hz, target {args.control_hz} Hz)")
    except KeyboardInterrupt:
        print("\nQuitting.")
    finally:
        # End-of-rollout park: fold to --park_pose preset (servo deg from
        # homes.py, converted to sim rad with the gripper remap). If
        # --park_pose '', skip and just disconnect.
        park = getattr(args, "park_pose", "") or ""
        if park:
            try:
                from homes import get_home_deg
                park_deg = get_home_deg(park)
                sim_rad = agent.servo_deg_to_sim_rad(park_deg)
                print(f"\n[park] folding to preset {park!r} (servo deg): "
                      f"{park_deg.round(1).tolist()}", flush=True)
                agent.reset(torch.from_numpy(sim_rad))
                print("[park] folded.", flush=True)
            except Exception as e:
                print(f"[park] could not fold to {park!r}: {e}", flush=True)
        elif args.home:
            agent.reset(get_home_rad(args.home_pose))
        robot.disconnect()


if __name__ == "__main__":
    main()
