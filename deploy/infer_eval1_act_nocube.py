"""
infer_eval1_act_nocube.py - LeRobot ACT inference for the Eval-1 8D goal-
conditioned policy (no cube_xy in env_state).

env_state = [color_6, bowl_x_m, bowl_y_m]   (8D)

Color and bowl xy come from CLI flags (told by the TA at eval time). No CV
scout, no cube position - the policy must locate the cube from the wrist
image alone. Sister of infer_eval1_act.py (10D variant).

Usage:
    python deploy/infer_eval1_act_nocube.py \\
        --target_color yellow \\
        --bowl_x 0.16 --bowl_y 0.32 \\
        --follower_port COM3 --camera_index 1
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

COLORS  = ["yellow", "orange", "red", "blue", "green", "violet"]
ENV_DIM = 8
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]


def build_env_state(color: str, bowl_x_m: float, bowl_y_m: float) -> np.ndarray:
    if color not in COLORS:
        raise ValueError(f"color must be one of {COLORS}, got {color!r}")
    vec = np.zeros(ENV_DIM, dtype=np.float32)
    vec[COLORS.index(color)] = 1.0
    vec[6] = bowl_x_m
    vec[7] = bowl_y_m
    return vec


def _load_policy_and_processors(policy_path: str, device: str):
    try:
        from lerobot.common.policies.act.modeling_act import ACTPolicy
    except ImportError:
        from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors

    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy = ACTPolicy.from_pretrained(policy_path).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor


def _make_robot(port: str, camera_index: int, fps: int):
    from lerobot.robots.utils import make_robot_from_config
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    cfg = SO101FollowerConfig(
        port=port,
        id="so101_follower",
        cameras={
            "wrist": OpenCVCameraConfig(
                index_or_path=camera_index, fps=fps, width=640, height=480,
            ),
        },
    )
    robot = make_robot_from_config(cfg)
    robot.connect()
    return robot


def _robot_obs_to_policy_obs(raw_obs: dict, env_state_np: np.ndarray,
                              camera_key: str = "wrist") -> dict:
    state = np.array([raw_obs[f"{m}.pos"] for m in MOTOR_NAMES], dtype=np.float32)
    img = raw_obs[camera_key]
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    return {
        "observation.state":             state,
        "observation.images.wrist":      img,
        "observation.environment_state": env_state_np.astype(np.float32),
    }


def _action_to_robot(action_np: np.ndarray) -> dict:
    return {f"{name}.pos": float(action_np[i]) for i, name in enumerate(MOTOR_NAMES)}


def run_episode(robot, policy, preprocessor, postprocessor,
                 env_state_np: np.ndarray, n_steps: int, fps: int,
                 device: str, task_desc: str | None = None,
                 max_delta_deg: float = 8.0, abort_if_normalized: bool = True):
    from lerobot.utils.control_utils import predict_action

    torch_device = torch.device(device)
    period = 1.0 / fps
    policy.reset()
    prev_action = None
    for step in range(n_steps):
        t0 = time.time()
        raw_obs = robot.get_observation()
        obs = _robot_obs_to_policy_obs(raw_obs, env_state_np)

        action_t = predict_action(
            observation=obs,
            policy=policy,
            device=torch_device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=False,
            task=task_desc,
            robot_type="so101_follower",
        )
        action_np = action_t.detach().cpu().numpy()
        if action_np.ndim == 2:
            action_np = action_np.squeeze(0)

        if step == 0 and abort_if_normalized and float(np.max(np.abs(action_np))) < 5.0:
            raise RuntimeError(
                f"First action looks NORMALIZED (max |a|={np.max(np.abs(action_np)):.3f} < 5). "
                f"Refusing to drive robot. Values: {action_np.tolist()}."
            )

        if prev_action is not None:
            delta = action_np - prev_action
            big = np.abs(delta) > max_delta_deg
            if big.any():
                action_np = prev_action + np.clip(delta, -max_delta_deg, max_delta_deg)
                print(f"  step {step:4d}: clamped delta on joints {np.where(big)[0].tolist()}")

        robot.send_action(_action_to_robot(action_np))
        prev_action = action_np

        elapsed = time.time() - t0
        if elapsed < period:
            time.sleep(period - elapsed)
        if step % fps == 0:
            print(f"  step {step:4d}/{n_steps}: action={action_np.round(2).tolist()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy_path",   default="hudela390/projet3-act-eval1-v1-no-cube")
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
    parser.add_argument("--dry_run",       action="store_true",
                         help="Load policy + run ONE inference with synthetic obs, no robot")
    args = parser.parse_args()

    print(f"[infer] policy:        {args.policy_path}")
    print(f"[infer] target color:  {args.target_color}")
    print(f"[infer] bowl xy:       ({args.bowl_x:+.3f}, {args.bowl_y:+.3f}) m")
    print(f"[infer] device:        {args.device}")

    print("[infer] Loading policy + processors ...")
    policy, preprocessor, postprocessor = _load_policy_and_processors(args.policy_path, args.device)

    task_desc = (f"Pick {args.target_color} block and place in bowl at "
                  f"({args.bowl_x*100:.1f},{args.bowl_y*100:.1f}) cm")

    env_state = build_env_state(args.target_color, args.bowl_x, args.bowl_y)
    print(f"[infer] env_state:     {env_state.round(3).tolist()}")

    if args.dry_run:
        from lerobot.utils.control_utils import predict_action
        print(f"\n[infer] DRY RUN -- no robot, synthetic obs.")
        obs = {
            "observation.state":             np.zeros(6, dtype=np.float32),
            "observation.images.wrist":      np.zeros((480, 640, 3), dtype=np.uint8),
            "observation.environment_state": env_state.astype(np.float32),
        }
        policy.reset()
        action_t = predict_action(
            observation=obs, policy=policy, device=torch.device(args.device),
            preprocessor=preprocessor, postprocessor=postprocessor,
            use_amp=False, task=task_desc, robot_type="so101_follower",
        )
        action_np = action_t.detach().cpu().numpy()
        if action_np.ndim == 2:
            action_np = action_np.squeeze(0)
        print(f"[infer] action: shape={tuple(action_t.shape)}, values={action_np.round(3).tolist()}")
        max_abs = float(np.max(np.abs(action_np)))
        print(f"[infer] max |action| = {max_abs:.3f}")
        if max_abs < 5.0:
            print("[infer] !! WARNING: action looks NORMALIZED. DO NOT run on robot.")
        elif max_abs > 200.0:
            print("[infer] !! WARNING: action magnitude exceptionally large.")
        else:
            print("[infer] OK -- action magnitude in plausible joint-angle range.")
        return

    print(f"[infer] Connecting to follower on {args.follower_port} ...")
    robot = _make_robot(args.follower_port, args.camera_index, args.fps)

    try:
        n_steps = int(args.episode_time_s * args.fps)
        for ep in range(args.num_episodes):
            print(f"\n[infer] === Episode {ep+1}/{args.num_episodes} ===")
            run_episode(robot, policy, preprocessor, postprocessor,
                          env_state, n_steps, args.fps, args.device,
                          task_desc=task_desc)
            if ep + 1 < args.num_episodes:
                print(f"[infer] Reset for {args.reset_time_s:.1f}s (move robot back to home)")
                time.sleep(args.reset_time_s)
    finally:
        print("\n[infer] Disconnecting robot ...")
        robot.disconnect()
        print("[infer] Done")


if __name__ == "__main__":
    main()
