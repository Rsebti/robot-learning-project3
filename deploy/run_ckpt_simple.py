"""
run_ckpt_simple.py — self-contained SAC inference for friend / Eval-1 checkpoints.

Auto-detects from the checkpoint:
  - encoder depth (2-conv 16x16 | 3-conv 32x32 or 42x32)
  - state width: 12 (Eval1) / 18 (+color) / 21 (+bowl_xyz)

Usage:
    python deploy/run_ckpt_simple.py --checkpoint C:/Users/hugod/ckpt_final.pt --color 3 --bowl_xyz 0.15 0.23 0.0

    # Wrist keyway ~90 deg off (physical CW vs training): try +90 on servo boundary
    python deploy/run_ckpt_simple.py --checkpoint C:/Users/hugod/ckpt_final.pt --color 3 --wrist_roll_offset_deg 90
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_DSHOW"] = "1000"

import cv2
import numpy as np
import torch
import torch.nn as nn

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

from lerobot.robots.utils import make_robot_from_config
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.motors.motors_bus import MotorNormMode

_DEPLOY = Path(__file__).resolve().parent
_PROJECT = _DEPLOY.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

N_COLORS = 6
SIM_CAM_SIZE = 128
# Training env was 10 Hz with delta caps [0.1, ..., 0.2]. Deploying at 30 Hz
# with caps scaled to 1/3 keeps the same per-second motion budget.
CONTROL_HZ = 30
DELTA_CAP = np.array([0.0333, 0.0333, 0.0333, 0.0333, 0.0333, 0.0667],
                     dtype=np.float32)
JOINT_LOWER = np.array([-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.174533])
JOINT_UPPER = np.array([1.91986, 1.74533, 1.69, 1.65806, 2.84121, 2.0944])

REST_QPOS_LEGACY = np.array(
    [0.0, 0.0, 0.0, np.pi / 2, -np.pi / 2, np.deg2rad(60)],
    dtype=np.float32,
)
REST_QPOS_UNIVERSAL = np.array(
    [-0.039131, -1.410071, 0.641358, 1.516708, -1.433855, -0.256321],
    dtype=np.float32,
)

GRIP_TRIGGER_RAD = np.deg2rad(20.0)


def pick_rest_qpos(n_state: int, flat_dim: int, rgb_emb: int, home: str) -> np.ndarray | None:
    if home == "legacy":
        return REST_QPOS_LEGACY.copy()
    if home == "universal":
        return REST_QPOS_UNIVERSAL.copy()
    if home == "auto":
        if n_state == 12:
            return REST_QPOS_LEGACY.copy()
        if n_state in (18, 21) or rgb_emb == 75 or flat_dim == 1792:
            return REST_QPOS_UNIVERSAL.copy()
        return REST_QPOS_LEGACY.copy()
    return REST_QPOS_LEGACY.copy()


class CNNEncoder(nn.Module):
    def __init__(self, n_conv: int):
        super().__init__()
        if n_conv == 2:
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=1), nn.ReLU(),
                nn.Flatten(),
            )
        elif n_conv == 3:
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
                nn.Flatten(),
            )
        else:
            raise ValueError(f"n_conv must be 2 or 3, got {n_conv}")

    def forward(self, rgb_uint8):
        x = rgb_uint8.permute(0, 3, 1, 2).float()
        x = x / 255.0 - 0.5
        return self.conv(x)


class Actor(nn.Module):
    def __init__(self, n_state: int, n_act: int, flat_dim: int, rgb_emb_dim: int, state_emb_dim: int):
        super().__init__()
        fusion_dim = rgb_emb_dim + state_emb_dim
        self.proj = nn.ModuleDict({
            "rgb_proj": nn.Sequential(
                nn.Linear(flat_dim, rgb_emb_dim), nn.LayerNorm(rgb_emb_dim), nn.Tanh(),
            ),
            "state_proj": nn.Sequential(
                nn.Linear(n_state, state_emb_dim), nn.LayerNorm(state_emb_dim), nn.ReLU(),
            ),
        })
        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
        )
        self.fc_mean = nn.Linear(256, n_act)
        self.fc_logstd = nn.Linear(256, n_act)
        self.register_buffer("action_scale", torch.ones(n_act))
        self.register_buffer("action_bias", torch.zeros(n_act))

    def forward(self, rgb_feat, state):
        x = torch.cat([self.proj["rgb_proj"](rgb_feat), self.proj["state_proj"](state)], dim=-1)
        x = self.fc(x)
        return torch.tanh(self.fc_mean(x)) * self.action_scale + self.action_bias


def _infer_image_hw(n_conv: int, flat_dim: int) -> tuple[int, int]:
    if n_conv == 2 and flat_dim == 1024:
        return 16, 16
    if n_conv == 3 and flat_dim == 1024:
        return 32, 32
    if n_conv == 3 and flat_dim == 1792:
        return 42, 32
    if n_conv == 3 and flat_dim == 3840:
        # ckpt_short variant: input 36x64 -> 17x31 -> 7x14 -> 5x12, flatten 64*5*12=3840.
        return 36, 64
    raise RuntimeError(f"Unknown (n_conv={n_conv}, flat_dim={flat_dim})")


def parse_actor_dims(actor_sd: dict) -> tuple[int, int, int, int, int]:
    rgb_w = actor_sd["proj.rgb_proj.0.weight"]
    rgb_emb, flat_dim = int(rgb_w.shape[0]), int(rgb_w.shape[1])
    s_w = actor_sd["proj.state_proj.0.weight"]
    state_emb, n_state = int(s_w.shape[0]), int(s_w.shape[1])
    n_act = int(actor_sd["fc_mean.weight"].shape[0])
    return flat_dim, rgb_emb, state_emb, n_state, n_act


def preprocess_image(rgb_tensor, image_h: int, image_w: int):
    img = rgb_tensor[0].cpu().numpy() if torch.is_tensor(rgb_tensor) else np.asarray(rgb_tensor[0])
    if image_h == image_w:
        h, w = img.shape[:2]
        c = min(h, w)
        img = img[(h - c) // 2:(h - c) // 2 + c, (w - c) // 2:(w - c) // 2 + c]
        img = cv2.resize(img, (SIM_CAM_SIZE, SIM_CAM_SIZE), interpolation=cv2.INTER_AREA)
        img = cv2.resize(img, (image_w, image_h), interpolation=cv2.INTER_AREA)
    else:
        h, w = img.shape[:2]
        ar = image_w / image_h
        if (w / h) > ar:
            new_w = int(round(h * ar))
            x0 = (w - new_w) // 2
            img = img[:, x0:x0 + new_w]
        else:
            new_h = int(round(w / ar))
            y0 = (h - new_h) // 2
            img = img[y0:y0 + new_h, :]
        img = cv2.resize(img, (image_w, image_h), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(img).unsqueeze(0).to(torch.uint8)


def build_state(qpos, target_qpos, goal_color, bowl_xyz=None):
    parts = [qpos, target_qpos]
    if goal_color is not None:
        one = np.zeros(N_COLORS, dtype=np.float32)
        one[int(goal_color)] = 1.0
        parts.append(one)
    if bowl_xyz is not None:
        parts.append(np.asarray(bowl_xyz, dtype=np.float32))
    return torch.from_numpy(np.concatenate(parts).astype(np.float32)).unsqueeze(0)


class RealRobotAgent:
    """Sim rad <-> servo deg; optional wrist_roll / gripper boundary offsets (servo = policy + offset)."""

    def __init__(self, robot, *, wrist_roll_offset_deg: float = 0.0, gripper_offset_deg: float = 0.0):
        self.real_robot = robot
        self._cached_qpos = None
        self._motor_keys: list[str] | None = None
        self._wrist_roll_key: str | None = None
        self._g_sim_min, self._g_sim_max = -10.0, 120.0
        self._g_servo_min, self._g_servo_max = -60.13, 66.73
        self._g_sim_range = self._g_sim_max - self._g_sim_min
        self._g_servo_range = self._g_servo_max - self._g_servo_min
        self.wrist_roll_offset_deg = float(wrist_roll_offset_deg)
        self.gripper_offset_deg = float(gripper_offset_deg)
        robot.bus.motors["gripper"].norm_mode = MotorNormMode.DEGREES
        self._init_motor_keys()

    def _init_motor_keys(self) -> None:
        deg = self.real_robot.bus.sync_read("Present_Position")
        self._motor_keys = list(deg.keys())
        self._wrist_roll_key = next((k for k in self._motor_keys if "wrist_roll" in k), None)

    def _apply_read_offsets(self, deg: dict) -> None:
        if self._wrist_roll_key and self.wrist_roll_offset_deg:
            deg[self._wrist_roll_key] = float(deg[self._wrist_roll_key]) - self.wrist_roll_offset_deg
        if self.gripper_offset_deg and "gripper" in deg:
            deg["gripper"] = float(deg["gripper"]) - self.gripper_offset_deg

    def get_qpos(self):
        if self._cached_qpos is not None:
            return self._cached_qpos.clone()
        deg = dict(self.real_robot.bus.sync_read("Present_Position"))
        self._apply_read_offsets(deg)
        servo = deg["gripper"]
        deg["gripper"] = (
            (servo - self._g_servo_min) / self._g_servo_range * self._g_sim_range + self._g_sim_min
        )
        flat = np.array([deg[k] for k in self._motor_keys], dtype=np.float32)
        self._cached_qpos = torch.deg2rad(torch.from_numpy(flat)).unsqueeze(0)
        return self._cached_qpos.clone()

    def set_target_qpos(self, qpos):
        if self._motor_keys is None:
            self._init_motor_keys()
        self._cached_qpos = None
        deg = torch.rad2deg(torch.as_tensor(qpos, dtype=torch.float32).flatten())
        cmd = {f"{self._motor_keys[i]}.pos": float(deg[i]) for i in range(len(deg))}
        sim_deg = cmd[f"{self._motor_keys[5]}.pos"]
        cmd[f"{self._motor_keys[5]}.pos"] = (
            (sim_deg - self._g_sim_min) / self._g_sim_range * self._g_servo_range + self._g_servo_min
        )
        if self._wrist_roll_key and self.wrist_roll_offset_deg:
            cmd[f"{self._wrist_roll_key}.pos"] += self.wrist_roll_offset_deg
        if self.gripper_offset_deg:
            cmd[f"{self._motor_keys[5]}.pos"] += self.gripper_offset_deg
        self.real_robot.send_action(cmd)

    def reset(self, qpos, freq=30, max_rad_per_step=0.012):
        qpos = torch.as_tensor(qpos, dtype=torch.float32).flatten()
        target = self.get_qpos().flatten()
        for _ in range(int(30 * freq)):
            delta = (qpos - target).clamp(-max_rad_per_step, max_rad_per_step)
            if torch.linalg.norm(delta) <= 1e-4:
                break
            target = target + delta
            self.set_target_qpos(target)
            time.sleep(1.0 / freq)

    def read_wrist_frame(self):
        return self.real_robot.cameras["base_camera"].async_read()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--port", default="COM3")
    p.add_argument("--camera_index", type=int, default=0,
                   help="Wrist USB camera index (default 0 on this setup).")
    p.add_argument("--color", "--goal_color", dest="color", type=int, default=0)
    p.add_argument("--bowl_xyz", type=float, nargs=3, default=[0.25, 0.10, 0.00], metavar=("X", "Y", "Z"))
    p.add_argument("--action_scale", type=float, default=0.1)
    p.add_argument("--episode_steps", type=int, default=150)
    p.add_argument("--n_episodes", type=int, default=1)
    p.add_argument("--control_hz", type=int, default=CONTROL_HZ)
    p.add_argument("--home", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--home_countdown_s", type=float, default=3.0)
    p.add_argument("--home_pose", choices=["auto", "legacy", "universal"], default="auto")
    p.add_argument("--wrist_roll_offset_deg", type=float, default=0.0,
                   help="servo = policy_deg + offset on wrist_roll (try 90 if ~90 deg CW keyway).")
    p.add_argument("--gripper_offset_deg", type=float, default=0.0)
    p.add_argument("--grip_force_steps", type=int, default=30)
    p.add_argument("--grip_pin_after_close", action="store_true")
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint).expanduser()
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ckpt] loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor_sd = ckpt["actor"]
    encoder_sd = ckpt["encoder"]
    flat_dim, rgb_emb, state_emb, n_state, n_act = parse_actor_dims(actor_sd)
    n_conv = 3 if "conv.4.weight" in encoder_sd else 2
    image_h, image_w = _infer_image_hw(n_conv, flat_dim)

    if n_state not in (12, 18, 21):
        raise RuntimeError(f"Unsupported n_state={n_state}")
    colour_conditioned = n_state in (18, 21)
    use_bowl_xyz = n_state == 21
    rest_qpos = pick_rest_qpos(n_state, flat_dim, rgb_emb, args.home_pose) if args.home else None

    print(f"[arch] n_conv={n_conv} image=({image_h},{image_w}) flat={flat_dim} rgb_emb={rgb_emb} "
          f"n_state={n_state} n_act={n_act}")
    print(f"[arch] colour={colour_conditioned} bowl={use_bowl_xyz} control_hz={args.control_hz}")
    if rest_qpos is not None:
        print(f"[home] {args.home_pose} -> rest (deg) {np.rad2deg(rest_qpos).round(1).tolist()}")
    if args.wrist_roll_offset_deg or args.gripper_offset_deg:
        print(f"[offset] wrist_roll {args.wrist_roll_offset_deg:+.1f}  gripper {args.gripper_offset_deg:+.1f}")

    encoder = CNNEncoder(n_conv=n_conv).to(device).eval()
    encoder.load_state_dict(encoder_sd)
    actor = Actor(n_state, n_act, flat_dim, rgb_emb, state_emb).to(device).eval()
    fixed_sd = {}
    for k, v in actor_sd.items():
        if k.startswith("proj.rgb_proj."):
            fixed_sd[f"proj.rgb_proj.{k[len('proj.rgb_proj.'):]}"] = v
        elif k.startswith("proj.state_proj."):
            fixed_sd[f"proj.state_proj.{k[len('proj.state_proj.'):]}"] = v
        else:
            fixed_sd[k] = v
    actor.load_state_dict(fixed_sd)
    print(f"[ckpt] step {ckpt.get('global_step', '?')}")

    robot = make_robot_from_config(SO101FollowerConfig(
        port=args.port,
        use_degrees=True,
        cameras={"base_camera": OpenCVCameraConfig(
            index_or_path=args.camera_index,
            fps=30,
            width=640,
            height=480,
            fourcc="MJPG",
            warmup_s=3,
        )},
    ))
    print(f"[robot] {args.port} cam={args.camera_index}")
    robot.connect()
    agent = RealRobotAgent(
        robot,
        wrist_roll_offset_deg=args.wrist_roll_offset_deg,
        gripper_offset_deg=args.gripper_offset_deg,
    )

    try:
        for ep in range(args.n_episodes):
            print(f"\n-- episode {ep + 1}/{args.n_episodes} color={args.color}"
                  f"{'' if not use_bowl_xyz else ' bowl=' + str(args.bowl_xyz)} --")

            if rest_qpos is not None:
                print("[home] ramping ...")
                agent.reset(torch.from_numpy(rest_qpos))
                if args.home_countdown_s > 0:
                    print(f"[home] start in {args.home_countdown_s:.0f}s")
                    time.sleep(args.home_countdown_s)

            target_qpos = agent.get_qpos().cpu().numpy().flatten()
            grip_force_remaining = 0
            grip_triggered = False
            t0 = time.perf_counter()

            for step in range(args.episode_steps):
                tick = time.perf_counter()
                qpos = agent.get_qpos().cpu().numpy().flatten()
                rgb_np = np.asarray(agent.read_wrist_frame())
                if rgb_np.ndim == 2:
                    rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_GRAY2BGR)
                obs_rgb = preprocess_image(torch.from_numpy(rgb_np).unsqueeze(0), image_h, image_w).to(device)
                obs_state = build_state(
                    qpos, target_qpos,
                    args.color if colour_conditioned else None,
                    bowl_xyz=args.bowl_xyz if use_bowl_xyz else None,
                ).to(device)

                with torch.no_grad():
                    raw_action = actor(encoder(obs_rgb), obs_state)[0].cpu().numpy()

                action = np.clip(raw_action * args.action_scale, -1.0, 1.0)
                target_qpos = np.clip(target_qpos + action * DELTA_CAP, JOINT_LOWER, JOINT_UPPER)

                if not grip_triggered and target_qpos[5] < GRIP_TRIGGER_RAD:
                    grip_force_remaining = int(args.grip_force_steps)
                    grip_triggered = True
                    print(f"  step {step}: grip close -> force {args.grip_force_steps} steps")
                if grip_force_remaining > 0:
                    target_qpos[5] = JOINT_LOWER[5]
                    grip_force_remaining -= 1
                elif args.grip_pin_after_close and grip_triggered:
                    target_qpos[5] = JOINT_LOWER[5]

                agent.set_target_qpos(torch.from_numpy(target_qpos))
                time.sleep(max(0.0, 1.0 / args.control_hz - (time.perf_counter() - tick)))
                if step % args.control_hz == 0:
                    print(f"  step {step:3d}/{args.episode_steps}")

            elapsed = time.perf_counter() - t0
            print(f"done {args.episode_steps} steps in {elapsed:.1f}s ({args.episode_steps / elapsed:.1f} Hz)")

    except KeyboardInterrupt:
        print("\n[stop] Ctrl+C")
    finally:
        try:
            if rest_qpos is not None:
                agent.reset(torch.from_numpy(rest_qpos))
        except Exception:
            pass
        robot.disconnect()
        print("[robot] disconnected")


if __name__ == "__main__":
    main()
