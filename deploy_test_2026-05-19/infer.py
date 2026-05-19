"""
UNIFIED real-robot inference for ALL SO-101 PlaceCube checkpoints
(Eval1 + Eval2, 16x16 px + 64x64 px) — one script, auto-detecting.

Tomorrow's job: deploy every checkpoint in ./checkpoints/ one by one on the
real SO-101 and watch what it does. This script figures out, FROM THE
CHECKPOINT ITSELF, everything that differs between policies:

  * image size  : conv.0 kernel 4 -> 16px (2-conv) | kernel 8 -> 64px (3-conv)
  * task / state: state_proj width 12 -> Eval1 (no colour) | 18 -> Eval2
                   (colour-conditioned, needs --goal_color) | 21 -> +bowl_xyz
  * mapping     : --mapping correct (VALIDATED real<->sim) | native (raw squint)

So you NEVER edit this file per checkpoint. Same command, different --checkpoint.

Contract (verified from the weights of every shipped ckpt):
  rgb    : wrist cam -> (1,S,S,3) uint8  (center-crop -> 128 -> S, S=16 or 64)
  state  : (1,12) Eval1 = [qpos(6), target_qpos(6)]
           (1,18) Eval2 = + goal_color one-hot(6)
  action : (6,) in [-1,1], pd_joint_target_delta_pos, 10 Hz

Dependencies: torch, numpy, opencv-python, lerobot[feetech]==0.4.3
(see README_CLAUDE.md — the 0.4.3 import paths are required).

Usage (Windows PowerShell, at the robot):
  python infer.py --checkpoint checkpoints/eval1/64px/eval1_64px_S2_DRctrl_BEST.pt ^
      --mapping correct --action_scale 0.1 --port COM3 --camera-index 1
  # Eval2 (colour-conditioned): add  --goal_color 0   (0 red 1 blue 2 green
  #                                                     3 yellow 4 purple 5 orange)
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from lerobot.robots.utils import make_robot_from_config
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.motors.motors_bus import MotorNormMode

try:
    import rerun as rr
except ImportError:
    rr = None

JOINT_NAMES = ["pan", "lift", "elbow", "wrist_flex", "wrist_roll", "gripper"]

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  DEFAULTS for your robot — override on the CLI, no need to edit this file  ║
# ║  (CLAUDE.md last-known laptop values: follower COM3, wrist cam index 1)    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
DEF_PORT = "COM3"                       # Windows: COMx | Mac: /dev/cu.usbmodemXXXX
DEF_CAMERA_INDEX = 1                    # laptop built-in webcam = 0, wrist = 1
DEF_CALIBRATION_ID = "so101_follower_arm"   # calibration .json filename (no ext)
DEF_CALIBRATION_DIR = Path(__file__).parent  # folder holding that .json

SIM_CAM_SIZE = 128       # sim wrist-camera resolution (intermediate resize step)
N_COLORS = 6             # goal-color one-hot length (Eval2 only)
CONTROL_HZ = 10          # ALL shipped ckpts trained at control_freq=10 — do NOT change
# pd_joint_target_delta_pos bounds at 10 Hz: arm +-0.1, gripper +-0.2
DELTA_CAP = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.2], dtype=np.float32)
# so101.urdf joint limits, order [pan, lift, elbow, wrist_flex, wrist_roll, gripper]
JOINT_LOWER = np.array([-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.174533])
JOINT_UPPER = np.array([1.91986, 1.74533, 1.69, 1.65806, 2.84121, 2.0944])
# SO101 "start" keyframe; the env seeds rest_qpos AND the controller target here.
REST_QPOS = np.array([0.0, 0.0, 0.0, np.pi / 2, -np.pi / 2, np.deg2rad(60)], dtype=np.float32)


def ramp_gripper_open(agent, arm_target_qpos, *, grip_idx: int = 5, steps: int = 15) -> np.ndarray:
    """Ramp gripper to sim-open while holding arm at last target so the cube can drop before homing."""
    tgt = np.asarray(arm_target_qpos, dtype=np.float64).flatten().copy()
    tgt = np.clip(tgt, JOINT_LOWER, JOINT_UPPER)
    g0 = float(tgt[grip_idx])
    g1 = float(JOINT_UPPER[grip_idx])
    for i in range(1, steps + 1):
        alpha = i / steps
        tgt[grip_idx] = g0 * (1.0 - alpha) + g1 * alpha
        tgt = np.clip(tgt, JOINT_LOWER, JOINT_UPPER)
        agent.set_target_qpos(torch.from_numpy(tgt.astype(np.float32)))
        time.sleep(1.0 / CONTROL_HZ)
    return tgt.astype(np.float32)


# ── The two real<->sim mappings (selectable with --mapping) ────────────────
# CORRECT = the validated mapping (4 ground-truth anchors + step-by-step HF
#   teleop-demo replay, visually validated 2026-05-18). Fixes the false-grasp:
#   the gripper now actually closes, and a constant lift/elbow homing offset
#   puts the TCP at z~=0.010 m at grasp (pure deg2rad left it ~3-4 cm too high).
# NATIVE = the original raw squint mapping (pure deg2rad, OLD gripper range).
#   Kept so you can A/B it on the real robot tomorrow.
MAPPINGS = {
    "correct": dict(
        joint_offset=np.array([0.0, 0.080, 0.220, 0.0, 0.0, 0.0], dtype=np.float32),
        g_servo_min=1.0, g_servo_max=75.0, g_sim_min=-18.0, g_sim_max=120.0,
    ),
    "native": dict(
        joint_offset=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        g_servo_min=-60.13, g_servo_max=66.73, g_sim_min=-10.0, g_sim_max=120.0,
    ),
}


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Robot driver — wraps a LeRobot SO101 follower                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def create_real_robot(port, camera_index, calibration_id, calibration_dir):
    config = SO101FollowerConfig(
        port=port,
        use_degrees=True,
        cameras={"base_camera": OpenCVCameraConfig(
            index_or_path=camera_index, fps=30, width=640, height=480,
        )},
        id=calibration_id,
        calibration_dir=calibration_dir,
    )
    return make_robot_from_config(config)


class RealRobotAgent:
    """Handles sim-radians <-> servo-degrees with a selectable mapping."""

    def __init__(self, robot, mapping_name):
        self.real_robot = robot
        self._cached_qpos = None
        self._motor_keys = None
        m = MAPPINGS[mapping_name]
        self.mapping_name = mapping_name
        self._joint_offset = m["joint_offset"]
        self._g_servo_min, self._g_servo_max = m["g_servo_min"], m["g_servo_max"]
        self._g_sim_min, self._g_sim_max = m["g_sim_min"], m["g_sim_max"]
        self._g_sim_range = self._g_sim_max - self._g_sim_min
        self._g_servo_range = self._g_servo_max - self._g_servo_min
        robot.bus.motors["gripper"].norm_mode = MotorNormMode.DEGREES

    def get_qpos(self):
        """Measured joint angles in sim radians, shape (1, 6)."""
        if self._cached_qpos is not None:
            return self._cached_qpos.clone()
        deg = self.real_robot.bus.sync_read("Present_Position")
        servo = deg["gripper"]                                  # gripper servo deg -> sim deg
        deg["gripper"] = (servo - self._g_servo_min) / self._g_servo_range * self._g_sim_range + self._g_sim_min
        if self._motor_keys is None:
            self._motor_keys = list(deg.keys())
        flat = np.array([deg[k] for k in self._motor_keys], dtype=np.float32)
        self._cached_qpos = (torch.deg2rad(torch.from_numpy(flat))
                             + torch.from_numpy(self._joint_offset)).unsqueeze(0)
        return self._cached_qpos.clone()

    def set_target_qpos(self, qpos):
        """Send a joint-angle target (sim radians) to the servos."""
        self._cached_qpos = None
        _q = torch.as_tensor(qpos, dtype=torch.float32).flatten()
        deg = torch.rad2deg(_q - torch.from_numpy(self._joint_offset))
        cmd = {f"{self._motor_keys[i]}.pos": float(deg[i]) for i in range(len(deg))}
        sim_deg = cmd["gripper.pos"]                             # gripper sim deg -> servo deg
        cmd["gripper.pos"] = (sim_deg - self._g_sim_min) / self._g_sim_range * self._g_servo_range + self._g_servo_min
        self.real_robot.send_action(cmd)

    def reset(self, qpos, freq=30, max_rad_per_step=0.025):
        """Smoothly ramp to qpos (used to go to REST between episodes)."""
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
            frame = np.asarray(cam.async_read())                # (H,W,3) uint8 RGB
            self._sensor_data[name] = {"rgb": torch.from_numpy(frame).unsqueeze(0)}

    def get_sensor_data(self):
        return self._sensor_data


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Policy network — built to match the detected checkpoint architecture     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
class CNNEncoder(nn.Module):
    """16px (2-conv) or 64px (3-conv); both flatten to 1024."""
    def __init__(self, image_size):
        super().__init__()
        if image_size == 16:
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=1), nn.ReLU(),
                nn.Flatten(),
            )
        elif image_size == 64:
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
                nn.Flatten(),
            )
        else:
            raise ValueError(f"unsupported image_size {image_size}")

    def forward(self, rgb_uint8):               # (B,S,S,3) uint8
        x = rgb_uint8.permute(0, 3, 1, 2).float()
        x = x / 255.0 - 0.5
        return self.conv(x)                     # (B,1024)


class Projection(nn.Module):
    def __init__(self, n_state):
        super().__init__()
        self.rgb_proj = nn.Sequential(nn.Linear(1024, 50), nn.LayerNorm(50), nn.Tanh())
        self.state_proj = nn.Sequential(nn.Linear(n_state, 256), nn.LayerNorm(256), nn.ReLU())

    def forward(self, rgb_feat, state):
        return torch.cat([self.rgb_proj(rgb_feat), self.state_proj(state)], dim=-1)


class Actor(nn.Module):
    def __init__(self, n_state, n_act=6):
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

    def forward(self, rgb_feat, state):         # deterministic eval action in [-1,1]
        x = self.fc(self.proj(rgb_feat, state))
        return torch.tanh(self.fc_mean(x)) * self.action_scale + self.action_bias


# ── Observation / action helpers ────────────────────────────────────────────
def preprocess_image(rgb, image_size):
    """Real camera frame (1,H,W,3) uint8 -> (1,S,S,3) uint8.
    Center-crop to square, resize to 128, area-downsample to S — the exact
    two-step path used during training."""
    img = rgb[0].cpu().numpy() if torch.is_tensor(rgb) else np.asarray(rgb[0])
    h, w = img.shape[:2]
    c = min(h, w)
    img = img[(h - c) // 2:(h - c) // 2 + c, (w - c) // 2:(w - c) // 2 + c]
    img = cv2.resize(img, (SIM_CAM_SIZE, SIM_CAM_SIZE), interpolation=cv2.INTER_AREA)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(img).unsqueeze(0).to(torch.uint8)


def build_state(qpos, target_qpos, goal_color, bowl_xyz=None):
    parts = [qpos, target_qpos]
    if goal_color is not None:                  # 18/21-d colour-conditioned (Eval2)
        onehot = np.zeros(N_COLORS, dtype=np.float32)
        onehot[goal_color] = 1.0
        parts.append(onehot)
    if bowl_xyz is not None:
        parts.append(np.asarray(bowl_xyz, dtype=np.float32))
    vec = np.concatenate(parts).astype(np.float32)
    return torch.from_numpy(vec).unsqueeze(0)


def detect_arch(ckpt):
    """Return (image_size, n_state) read straight from the weights."""
    k = ckpt["encoder"]["conv.0.weight"].shape[-1]      # conv kernel size
    image_size = {4: 16, 8: 64}.get(k)
    if image_size is None:
        raise RuntimeError(f"unrecognised conv.0 kernel {k} (expected 4=16px or 8=64px)")
    n_state = ckpt["actor"]["proj.state_proj.0.weight"].shape[1]
    if n_state not in (12, 18, 21):
        raise RuntimeError(f"unrecognised n_state {n_state} (expected 12=Eval1, 18=Eval2, 21)")
    return image_size, n_state


def init_viz():
    if rr is None:
        raise RuntimeError("rerun not installed (pip install rerun-sdk) — or run --no-viz")
    rr.init("squint_infer", spawn=True)


def log_step(step, raw_rgb, policy_rgb, qpos, target_qpos, action_raw):
    if rr is None:
        return
    rr.set_time("step", sequence=step)
    rr.log("camera/raw", rr.Image(raw_rgb))
    rr.log("camera/policy_input", rr.Image(policy_rgb))
    for i, name in enumerate(JOINT_NAMES):
        rr.log(f"joints/qpos_measured/{name}", rr.Scalars([float(qpos[i])]))
        rr.log(f"joints/qpos_target/{name}", rr.Scalars([float(target_qpos[i])]))
        rr.log(f"action_raw/{name}", rr.Scalars([float(action_raw[i])]))


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Main                                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def main():
    p = argparse.ArgumentParser(description="Unified SO-101 deploy (auto 16/64, Eval1/2).")
    p.add_argument("--checkpoint", required=True, help="path to a .pt in ./checkpoints/")
    p.add_argument("--mapping", choices=["correct", "native"], default="correct",
                   help="correct = validated real<->sim (fixes false-grasp); "
                        "native = raw squint (pure deg2rad, old gripper range)")
    p.add_argument("--goal_color", type=int, default=0,
                   help="Eval2 only: 0 red 1 blue 2 green 3 yellow 4 purple 5 orange")
    p.add_argument("--action_scale", type=float, default=0.1,
                   help="safety multiplier on the policy action (START at 0.1)")
    p.add_argument("--episode_steps", type=int, default=150,
                   help="control steps/episode (sim ~50 @10Hz=5s; 150=15s real margin)")
    p.add_argument("--n_episodes", type=int, default=0,
                   help=">0 = run N back-to-back; 0 = wait for Enter each episode")
    p.add_argument("--port", default=DEF_PORT, help="follower serial port (COM3 / /dev/cu.*)")
    p.add_argument("--camera-index", type=int, default=DEF_CAMERA_INDEX,
                   help="wrist cam OpenCV index (laptop built-in=0, wrist often=1)")
    p.add_argument("--calibration-id", default=DEF_CALIBRATION_ID,
                   help="calibration .json filename without extension")
    p.add_argument("--calibration-dir", default=str(DEF_CALIBRATION_DIR),
                   help="folder containing the calibration .json")
    p.add_argument("--bowl_xyz", type=float, nargs=3, default=[0.25, 0.10, 0.00],
                   metavar=("X", "Y", "Z"), help="only if a ckpt expects 21-d state")
    p.add_argument("--viz", action=argparse.BooleanOptionalAction, default=True,
                   help="Rerun viewer (live cam + joints). --no-viz to disable")
    p.add_argument("--log_dir", default=None, help="if set, dump per-step npz per episode")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    image_size, n_state = detect_arch(ckpt)
    eval_name = {12: "Eval1 (single cube, NO colour)",
                 18: "Eval2 (colour-conditioned — uses --goal_color)",
                 21: "21-d (colour + bowl_xyz)"}[n_state]
    colour_conditioned = n_state in (18, 21)
    use_bowl_xyz = n_state == 21

    print("=" * 70)
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  AUTO-DETECT: image {image_size}x{image_size}px  |  n_state {n_state}  |  {eval_name}")
    print(f"  mapping    : {args.mapping.upper()}"
          + ("  (validated real<->sim)" if args.mapping == "correct" else "  (raw squint — A/B baseline)"))
    print(f"  action_scale {args.action_scale}  control {CONTROL_HZ} Hz  steps/ep {args.episode_steps}")
    if colour_conditioned:
        print(f"  goal_color : {args.goal_color}  (0 red 1 blue 2 green 3 yellow 4 purple 5 orange)")
    print(f"  trained to step {ckpt.get('global_step', '?')}")
    print("=" * 70)

    encoder = CNNEncoder(image_size).to(device).eval()
    actor = Actor(n_state=n_state).to(device).eval()
    encoder.load_state_dict(ckpt["encoder"])    # strict: errors loudly on any mismatch
    actor.load_state_dict(ckpt["actor"])

    if args.viz:
        init_viz()

    robot = create_real_robot(args.port, args.camera_index,
                              args.calibration_id, Path(args.calibration_dir))
    robot.connect()
    agent = RealRobotAgent(robot, args.mapping)

    log_dir = Path(args.log_dir) if args.log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

    def episodes():
        if args.n_episodes > 0:
            yield from range(args.n_episodes)
        else:
            i = 0
            while True:
                input(f"\n[Enter] start episode "
                      f"{'(goal color %d) ' % args.goal_color if colour_conditioned else ''}"
                      f"— Ctrl+C to quit ")
                yield i
                i += 1

    try:
        for ep in episodes():
            if args.n_episodes > 0:
                print(f"\n-- Episode {ep + 1}/{args.n_episodes} --")
            agent.reset(REST_QPOS)
            target_qpos = agent.get_qpos().cpu().numpy().flatten()
            lq, lt, la, lr = [], [], [], []

            for step in range(args.episode_steps):
                t0 = time.perf_counter()
                qpos = agent.get_qpos().cpu().numpy().flatten()
                agent.capture_sensor_data()
                rgb = agent.get_sensor_data()["base_camera"]["rgb"]
                obs_rgb = preprocess_image(rgb, image_size).to(device)
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
                    log_step(step,
                             rgb[0].cpu().numpy() if torch.is_tensor(rgb) else np.asarray(rgb[0]),
                             obs_rgb[0].cpu().numpy(), qpos, target_qpos, raw_action)
                if log_dir:
                    lq.append(qpos.copy()); lt.append(target_qpos.copy())
                    la.append(raw_action.copy()); lr.append(obs_rgb[0].cpu().numpy().copy())
                time.sleep(max(0.0, 1.0 / CONTROL_HZ - (time.perf_counter() - t0)))

            if log_dir:
                np.savez(log_dir / f"ep{ep:03d}.npz",
                         qpos=np.stack(lq), target_qpos=np.stack(lt),
                         action_raw=np.stack(la), policy_rgb=np.stack(lr),
                         joint_names=np.array(JOINT_NAMES))
                print(f"  -> saved {log_dir / f'ep{ep:03d}.npz'}")
            # Release / drop: open gripper at full episode length before next REST reset.
            ramp_gripper_open(agent, target_qpos)
            print(f"Episode done ({args.episode_steps} steps).")
    except KeyboardInterrupt:
        print("\nQuitting — opening gripper then ramping back to REST.")
    finally:
        try:
            q = agent.get_qpos().cpu().numpy().flatten()
            ramp_gripper_open(agent, q, steps=20)
        except Exception:
            pass
        agent.reset(REST_QPOS)
        robot.disconnect()


if __name__ == "__main__":
    main()
