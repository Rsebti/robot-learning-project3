#!/usr/bin/env python3
"""
infer_rlpd.py — deploy Squint RLPD checkpoints on the SO-101 follower.

Loads encoder+actor from squint-rlpd ``runs/*/ckpt.pt`` (80×144 RGB, delta actions).
Probes checkpoint metadata (n_state, image size, global_step) on load.

Handoff to scripted place (ik_relative lift_place):
  --handoff_on_space     press SPACE mid-rollout → stop policy, exit 0
  --handoff_on_grasp     stop after gripper closed for N steps (like SAC handoff)
  --handoff_step N       auto-stop at step N

Pair with deploy/run_rlpd_ik_place.py for RLPD pick → IK place.

Usage:
  python deploy/infer_rlpd.py --ckpt C:\\Users\\hugod\\squint-rlpd\\runs\\<run>\\ckpt_best.pt \\
      --goal_color 3 --bowl_xyz 0.16 0.32 0.0 --follower_port COM3

  python deploy/infer_rlpd.py --name rlpd --handoff_on_space --bowl_xyz 0.16 0.32 0.0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_DEPLOY = Path(__file__).resolve().parent
_PROJECT = _DEPLOY.parent
if str(_DEPLOY) not in sys.path:
    sys.path.insert(0, str(_DEPLOY))

from homes import get_home_rad, list_home_poses  # noqa: E402
from robot_calibration import make_so101_follower_config  # noqa: E402
from sac_infer_home import add_sac_home_cli, sac_maybe_home  # noqa: E402
from rlpd_infer_common import (  # noqa: E402
    CONTROL_HZ,
    DELTA_CAP,
    build_state_vector,
    load_rlpd_policy,
    preprocess_rgb,
    probe_rlpd_checkpoint,
    resolve_rlpd_checkpoint,
)

import cv2  # noqa: E402
from lerobot.robots.utils import make_robot_from_config  # noqa: E402
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: E402
from lerobot.motors.motors_bus import MotorNormMode  # noqa: E402

JOINT_NAMES = ["pan", "lift", "elbow", "wrist_flex", "wrist_roll", "gripper"]
JOINT_LOWER = np.array([-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.174533])
JOINT_UPPER = np.array([1.91986, 1.74533, 1.69, 1.65806, 2.84121, 2.0944])

GOAL_COLORS = {
    "red": 0, "blue": 1, "green": 2, "yellow": 3, "purple": 4, "orange": 5,
}


def _space_pressed() -> bool:
    if sys.platform == "win32":
        import msvcrt
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            return ch in (b" ", b"\r")
    return False


class RealRobotAgent:
    """LeRobot SO-101 driver (sim-radian qpos, same gripper mapping as infer_sac_legacy)."""

    def __init__(self, robot, *, camera_index: int = 1):
        self.real_robot = robot
        self._cached_qpos = None
        self._motor_keys = None
        self._g_sim_min, self._g_sim_max = -10.0, 120.0
        self._g_servo_min, self._g_servo_max = -60.13, 66.73
        self._g_sim_range = self._g_sim_max - self._g_sim_min
        self._g_servo_range = self._g_servo_max - self._g_servo_min
        robot.bus.motors["gripper"].norm_mode = MotorNormMode.DEGREES
        self.cameras = {
            "wrist": cv2.VideoCapture(camera_index, cv2.CAP_DSHOW),
        }
        if not self.cameras["wrist"].isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")

    def get_qpos(self) -> torch.Tensor:
        if self._cached_qpos is not None:
            return self._cached_qpos.clone()
        deg = self.real_robot.bus.sync_read("Present_Position")
        servo_g = deg["gripper"]
        deg["gripper"] = (servo_g - self._g_servo_min) / self._g_servo_range * self._g_sim_range + self._g_sim_min
        if self._motor_keys is None:
            self._motor_keys = list(deg.keys())
        flat = np.array([deg[k] for k in self._motor_keys], dtype=np.float32)
        self._cached_qpos = torch.deg2rad(torch.from_numpy(flat)).unsqueeze(0)
        return self._cached_qpos.clone()

    def get_qpos_deg(self) -> np.ndarray:
        return np.rad2deg(self.get_qpos().cpu().numpy().flatten())

    def set_target_qpos(self, qpos: torch.Tensor | np.ndarray) -> None:
        self._cached_qpos = None
        q = torch.as_tensor(qpos, dtype=torch.float32).flatten()
        deg = torch.rad2deg(q)
        cmd = {f"{self._motor_keys[i]}.pos": float(deg[i]) for i in range(len(deg))}
        sim_g = cmd["gripper.pos"]
        cmd["gripper.pos"] = (sim_g - self._g_sim_min) / self._g_sim_range * self._g_servo_range + self._g_servo_min
        self.real_robot.send_action(cmd)

    def reset(self, qpos_rad: np.ndarray, *, freq: float = CONTROL_HZ, max_rad_per_step: float = 0.025) -> None:
        target = torch.as_tensor(qpos_rad, dtype=torch.float32).flatten()
        cur = self.get_qpos().flatten()
        for _ in range(int(20 * freq)):
            delta = (target - cur).clamp(-max_rad_per_step, max_rad_per_step)
            if torch.linalg.norm(delta) <= 1e-4:
                break
            cur = cur + delta
            self.set_target_qpos(cur)
            time.sleep(1.0 / freq)

    def capture_rgb(self) -> np.ndarray:
        ok, frame = self.cameras["wrist"].read()
        if not ok or frame is None:
            raise RuntimeError("wrist camera read failed")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self.cameras["wrist"].release()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=None, help="Path to ckpt.pt / ckpt_best.pt")
    p.add_argument("--name", default=None, help="Run folder name under squint-rlpd/runs/")
    p.add_argument("--follower_port", default="COM3")
    p.add_argument("--camera_index", type=int, default=1)
    p.add_argument("--goal_color", type=int, default=None)
    p.add_argument("--color", default="yellow", choices=list(GOAL_COLORS))
    p.add_argument("--bowl_xyz", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                   help="User-frame bowl position (required if checkpoint n_state>=21 or 58).")
    p.add_argument("--action_scale", type=float, default=0.15)
    p.add_argument("--episode_steps", type=int, default=600)
    p.add_argument("--handoff_on_space", action="store_true",
                   help="Press SPACE during rollout to stop for IK place.")
    p.add_argument("--handoff_on_grasp", action="store_true",
                   help="Stop when gripper stays closed (deg) for --grasp_hold_steps.")
    p.add_argument("--handoff_grasp_immediate", action="store_true",
                   help="With --handoff_on_grasp: first close intent triggers handoff.")
    p.add_argument("--grasp_hold_steps", type=int, default=15,
                   help="Consecutive steps with gripper<=grasp_deg to count as grasp.")
    p.add_argument("--grasp_gripper_max_deg", type=float, default=35.0)
    p.add_argument("--handoff_step", type=int, default=None,
                   help="Stop policy at this step index (0-based) for IK handoff.")
    p.add_argument("--probe_only", action="store_true", help="Print checkpoint metadata and exit.")
    add_sac_home_cli(p, default_home_pose="eval1_rest")
    args = p.parse_args()

    ckpt_path = resolve_rlpd_checkpoint(args.ckpt, name=args.name)
    meta = probe_rlpd_checkpoint(ckpt_path)

    print("[rlpd] checkpoint:", meta.path)
    print(f"[rlpd] global_step={meta.global_step}  n_state={meta.n_state}  "
          f"image={meta.image_h}x{meta.image_w}  repr_dim={meta.encoder_repr_dim}")
    for w in meta.warnings:
        print(f"[rlpd] WARNING: {w}")

    if args.probe_only:
        print(json.dumps({
            "path": meta.path,
            "global_step": meta.global_step,
            "n_state": meta.n_state,
            "image_h": meta.image_h,
            "image_w": meta.image_w,
            "use_bowl_xyz": meta.use_bowl_xyz,
            "is_full_state_58": meta.is_full_state_58,
        }, indent=2))
        return 0

    goal_color = args.goal_color if args.goal_color is not None else GOAL_COLORS[args.color]
    bowl_xyz = np.asarray(args.bowl_xyz, dtype=np.float32) if args.bowl_xyz is not None else None
    if meta.use_bowl_xyz and bowl_xyz is None:
        p.error(f"Checkpoint expects bowl_xyz in state (n_state={meta.n_state}); pass --bowl_xyz X Y Z")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, actor, meta = load_rlpd_policy(ckpt_path, device=device)

    config = make_so101_follower_config(
        args.follower_port,
        use_degrees=True,
        cameras={},
    )
    robot = make_robot_from_config(config)
    robot.connect()
    agent = RealRobotAgent(robot, camera_index=args.camera_index)

    try:
        rest = sac_maybe_home(agent, args)
        target_qpos = agent.get_qpos().cpu().numpy().flatten()
        prev_qpos = target_qpos.copy()
        grasp_streak = 0
        handoff = False
        handoff_reason = ""

        print(f"[rlpd] rollout goal_color={goal_color}  steps={args.episode_steps}  hz={CONTROL_HZ}")
        if bowl_xyz is not None:
            print(f"[rlpd] bowl_xyz (user m): {bowl_xyz.tolist()}")

        for step in range(args.episode_steps):
            t0 = time.perf_counter()

            if args.handoff_on_space and _space_pressed():
                handoff = True
                handoff_reason = "space"
                print(f"[rlpd] SPACE pressed at step {step} — handoff for IK", flush=True)
                break

            if args.handoff_step is not None and step >= args.handoff_step:
                handoff = True
                handoff_reason = f"step>={args.handoff_step}"
                print(f"[rlpd] handoff_step reached at {step}", flush=True)
                break

            qpos = agent.get_qpos().cpu().numpy().flatten()
            rgb = agent.capture_rgb()
            obs_rgb = preprocess_rgb(rgb, h=meta.image_h, w=meta.image_w).to(device)
            state = build_state_vector(
                qpos, target_qpos,
                goal_color=goal_color,
                bowl_xyz=bowl_xyz,
                prev_qpos_rad=prev_qpos if meta.is_full_state_58 else None,
                n_state=meta.n_state,
            ).to(device)

            with torch.no_grad():
                feat = encoder(obs_rgb)
                raw_action = actor.eval_action(feat, state)[0].cpu().numpy()

            action = np.clip(raw_action * args.action_scale, -1.0, 1.0)
            target_qpos = np.clip(target_qpos + action * DELTA_CAP, JOINT_LOWER, JOINT_UPPER)
            agent.set_target_qpos(target_qpos)
            prev_qpos = qpos.copy()

            if args.handoff_on_grasp:
                g_deg = float(np.rad2deg(qpos[5]))
                closed = g_deg <= args.grasp_gripper_max_deg
                if args.handoff_grasp_immediate and closed:
                    handoff = True
                    handoff_reason = "grasp_immediate"
                    print(f"[rlpd] grasp (immediate) at step {step} grip={g_deg:.1f} deg", flush=True)
                    break
                if closed:
                    grasp_streak += 1
                    if grasp_streak >= args.grasp_hold_steps:
                        handoff = True
                        handoff_reason = "grasp_hold"
                        print(f"[rlpd] grasp hold at step {step} ({grasp_streak} steps)", flush=True)
                        break
                else:
                    grasp_streak = 0

            if step % 30 == 0:
                print(f"  step {step:4d}  grip_deg={np.rad2deg(qpos[5]):.1f}  "
                      f"action_norm={np.linalg.norm(action):.3f}")

            time.sleep(max(0.0, 1.0 / CONTROL_HZ - (time.perf_counter() - t0)))

        if handoff or args.handoff_on_space or args.handoff_on_grasp or args.handoff_step is not None:
            code = 0 if handoff else 2
            print(f"[rlpd] done handoff={handoff} reason={handoff_reason or 'none'} exit={code}", flush=True)
            return code

        print("[rlpd] episode finished (no handoff flag)", flush=True)
        return 0

    except KeyboardInterrupt:
        print("\n[rlpd] interrupted", flush=True)
        return 130
    finally:
        agent.close()
        robot.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
