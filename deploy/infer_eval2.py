"""
infer_eval2.py - goal-conditioned inference for Eval-2 / Eval-3.

Loads a trained goal-conditioned ACT policy and runs autonomous rollouts on
the SO-101 follower. At each step it captures the robot's obs (joints + wrist
camera), INJECTS the user-specified `observation.environment_state` (6D
color one-hot + 2D bowl xy in meters) into the obs dict, calls the policy,
and sends the action back to the robot.

This is the deploy-time complement to `add_goal_conditioning.py`:
training-time the dataset has env_state per frame; at deploy we have to
build env_state from CLI args and feed it in ourselves, since the physical
robot does not produce it.

Usage:
    python deploy/infer_eval2.py \\
        --policy_path  hudela390/projet3-act-eval2-v1-goal \\
        --target_color red \\
        --bowl_x -0.155  --bowl_y  0.295 \\
        --follower_port COM3

CLI flags map straight to env_state:
    env_state = [c_yellow, c_orange, c_red, c_blue, c_green, c_violet,
                 bowl_x_m, bowl_y_m]
    color one-hot picked from --target_color, bowl xy from --bowl_x/--bowl_y.

NOTE on lerobot versions: this uses lerobot's Python API (Robot, Policy
classes) rather than the `lerobot-record` CLI because the CLI does not
expose env_state injection. Tested with lerobot 0.5.x. If imports fail on
a newer version, the patterns are still right -- the import paths may have
moved to `lerobot.robots.*` / `lerobot.policies.*`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch


COLORS = ["yellow", "orange", "red", "blue", "green", "violet"]
ENV_DIM = 8


def build_env_state(color: str, bowl_x_m: float, bowl_y_m: float) -> np.ndarray:
    if color not in COLORS:
        raise ValueError(f"color must be one of {COLORS}, got {color!r}")
    vec = np.zeros(ENV_DIM, dtype=np.float32)
    vec[COLORS.index(color)] = 1.0
    vec[6] = bowl_x_m
    vec[7] = bowl_y_m
    return vec


def _load_policy(policy_path: str, device: str):
    """Load the trained ACT policy. Falls back across two lerobot import paths."""
    try:
        from lerobot.common.policies.act.modeling_act import ACTPolicy
    except ImportError:
        from lerobot.policies.act.modeling_act import ACTPolicy   # newer lerobot

    policy = ACTPolicy.from_pretrained(policy_path)
    policy = policy.to(device).eval()
    return policy


def _make_robot(port: str, camera_index: int, fps: int):
    """Connect to the SO-101 follower with wrist camera."""
    try:
        from lerobot.common.robot_devices.robots.utils import make_robot_from_config
        from lerobot.common.robot_devices.robots.configs import So101FollowerConfig
        from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
    except ImportError:
        from lerobot.robots.utils import make_robot_from_config
        from lerobot.robots.so101_follower.configs import So101FollowerConfig
        from lerobot.cameras.opencv.configs import OpenCVCameraConfig

    cfg = So101FollowerConfig(
        port=port,
        id="so101_follower",
        cameras={
            "wrist": OpenCVCameraConfig(
                index_or_path=camera_index,
                fps=fps, width=640, height=480,
            ),
        },
    )
    robot = make_robot_from_config(cfg)
    robot.connect()
    return robot


def _prepare_obs(robot_obs: dict, env_state_np: np.ndarray, device: str) -> dict:
    """Add batch dim, move to device, inject env_state."""
    out = {}
    for k, v in robot_obs.items():
        if isinstance(v, torch.Tensor):
            t = v
        else:
            t = torch.as_tensor(v)
        if t.dim() == 1 or t.dim() == 3:        # add batch dim
            t = t.unsqueeze(0)
        out[k] = t.to(device, non_blocking=True)

    out["observation.environment_state"] = torch.from_numpy(
        env_state_np
    ).unsqueeze(0).to(device)
    return out


def run_episode(robot, policy, env_state_np: np.ndarray,
                 n_steps: int, fps: int, device: str):
    period = 1.0 / fps
    policy.reset()
    for step in range(n_steps):
        t0 = time.time()
        raw_obs = robot.capture_observation()
        obs = _prepare_obs(raw_obs, env_state_np, device)

        with torch.inference_mode():
            action = policy.select_action(obs)
        action_np = action.squeeze(0).cpu().numpy()

        robot.send_action(torch.from_numpy(action_np))

        elapsed = time.time() - t0
        if elapsed < period:
            time.sleep(period - elapsed)
        if step % fps == 0:
            print(f"  step {step:4d}/{n_steps}: action={action_np.round(2).tolist()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy_path",   default="hudela390/projet3-act-eval2-v1-goal")
    parser.add_argument("--target_color",  required=True, choices=COLORS)
    parser.add_argument("--bowl_x",        type=float, required=True,
                         help="Bowl x in meters (right +, left -)")
    parser.add_argument("--bowl_y",        type=float, required=True,
                         help="Bowl y in meters (forward +, back -)")
    parser.add_argument("--follower_port", default="COM3")
    parser.add_argument("--camera_index",  type=int, default=1)
    parser.add_argument("--fps",           type=int, default=30)
    parser.add_argument("--episode_time_s", type=float, default=15.0)
    parser.add_argument("--num_episodes",  type=int, default=1)
    parser.add_argument("--reset_time_s",  type=float, default=10.0)
    parser.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"[infer] policy:        {args.policy_path}")
    print(f"[infer] target color:  {args.target_color}")
    print(f"[infer] bowl xy:       ({args.bowl_x:+.3f}, {args.bowl_y:+.3f}) m")
    print(f"[infer] device:        {args.device}")

    env_state = build_env_state(args.target_color, args.bowl_x, args.bowl_y)
    print(f"[infer] env_state:     {env_state.tolist()}")

    print("[infer] Loading policy ...")
    policy = _load_policy(args.policy_path, args.device)

    print(f"[infer] Connecting to follower on {args.follower_port} ...")
    robot = _make_robot(args.follower_port, args.camera_index, args.fps)

    try:
        n_steps = int(args.episode_time_s * args.fps)
        for ep in range(args.num_episodes):
            print(f"\n[infer] === Episode {ep+1}/{args.num_episodes} ===")
            run_episode(robot, policy, env_state, n_steps, args.fps, args.device)
            if ep + 1 < args.num_episodes:
                print(f"[infer] Reset for {args.reset_time_s:.1f}s (move robot back to home)")
                time.sleep(args.reset_time_s)
    finally:
        print("\n[infer] Disconnecting robot ...")
        robot.disconnect()
        print("[infer] Done")


if __name__ == "__main__":
    main()
