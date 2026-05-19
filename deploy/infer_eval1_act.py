"""
infer_eval1_act.py - LeRobot ACT inference for the Eval-1 10D goal-conditioned
policy (osammotg1/projet3-act-eval1-v1-cube-goal).

env_state = [color_6, bowl_x_m, bowl_y_m, cube_x_m, cube_y_m]   (10D)

- Color and bowl xy come from CLI flags (told by the TA at eval time).
- Cube xy is either passed explicitly via --cube_x / --cube_y, OR
  auto-detected from the FIRST wrist frame using toolset/detect_cube_cv.py.

Usage:
    python deploy/infer_eval1_act.py \\
        --target_color yellow \\
        --bowl_x 0.16 --bowl_y 0.32 \\
        --follower_port COM3 --camera_index 1

    python deploy/infer_eval1_act.py ... --cube_x 0.28 --cube_y -0.08
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from act_infer_common import (
    COLORS,
    MOTOR_NAMES,
    add_act_cli_args,
    build_env_state,
    dry_run_check,
    load_policy_and_processors,
    make_robot,
    maybe_home_robot,
    ramp_to_home,
    run_episode,
)
from homes import DEFAULT_HOME_POSE, get_home_deg

sys.path.insert(0, str(Path(__file__).parent.parent / "toolset"))
from detect_cube_cv import (  # noqa: E402
    HSV_RANGES,
    detect_cube_pixel,
    pixel_to_base,
    K,
    T_CAM_IN_WRIST,
    Z_TABLE,
)

HSV_RANGES["red"] = [((0, 120, 60), (4, 255, 255)), ((172, 120, 60), (180, 255, 255))]
HSV_RANGES["orange"] = [((4, 120, 80), (22, 255, 255))]


def _scout_cube_xy(robot, color: str) -> tuple[float, float]:
    """Capture one wrist frame, detect cube of `color`, back-project to base xy."""
    print(f"[scout] capturing frame to find {color} cube ...", flush=True)
    obs = robot.get_observation()
    img = obs["wrist"]
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    scout_path = Path(__file__).parent / "_scout_frame.png"
    cv2.imwrite(str(scout_path), bgr)
    print(f"[scout] saved frame -> {scout_path}", flush=True)

    pixel, _mask, area = detect_cube_pixel(bgr, color)
    if pixel is None:
        raise RuntimeError(
            f"[scout] no {color} cube detected in {scout_path}. "
            f"Inspect the image; pass --cube_x/--cube_y manually "
            f"or move the cube into view."
        )

    q_rad = np.deg2rad([obs[f"{m}.pos"] for m in MOTOR_NAMES[:5]]).tolist()
    base_xyz = pixel_to_base(*pixel, K, T_CAM_IN_WRIST, q_rad, z_table=Z_TABLE)
    if base_xyz is None:
        raise RuntimeError("[scout] back-projection failed (ray parallel to table).")
    print(
        f"[scout] {color} cube at pixel {pixel} (area {int(area)}) -> "
        f"base xy = ({base_xyz[0]:+.3f}, {base_xyz[1]:+.3f}) m",
        flush=True,
    )
    return float(base_xyz[0]), float(base_xyz[1])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--policy_path", default="osammotg1/projet3-act-eval1-v1-cube-goal")
    parser.add_argument("--cube_x", type=float, default=None,
                        help="Cube x in meters (overrides CV scout)")
    parser.add_argument("--cube_y", type=float, default=None,
                        help="Cube y in meters (overrides CV scout)")
    add_act_cli_args(parser, default_home_pose="eval1_rest")
    args = parser.parse_args()

    print(f"[infer] policy:        {args.policy_path}")
    print(f"[infer] target color:  {args.target_color}")
    print(f"[infer] bowl xy:       ({args.bowl_x:+.3f}, {args.bowl_y:+.3f}) m")
    print(f"[infer] device:        {args.device}")

    print("[infer] Loading policy + processors ...")
    policy, preprocessor, postprocessor = load_policy_and_processors(
        args.policy_path, args.device,
    )

    task_desc = (
        f"Pick {args.target_color} block and place in bowl at "
        f"({args.bowl_x*100:.1f},{args.bowl_y*100:.1f}) cm"
    )

    if args.dry_run:
        cube_x = args.cube_x if args.cube_x is not None else 0.25
        cube_y = args.cube_y if args.cube_y is not None else 0.00
        env_state = build_env_state(
            args.target_color, args.bowl_x, args.bowl_y,
            cube_x_m=cube_x, cube_y_m=cube_y,
        )
        print(f"[infer] env_state:     {env_state.tolist()}")
        dry_run_check(policy, preprocessor, postprocessor, env_state, args.device, task_desc)
        return

    print(f"[infer] Connecting to follower on {args.follower_port} ...")
    robot = make_robot(
        args.follower_port, args.camera_index, args.fps,
        calibration_dir=args.calibration_dir,
    )

    if args.cube_x is not None and args.cube_y is not None:
        cube_x, cube_y = args.cube_x, args.cube_y
        print(f"[infer] cube xy (CLI): ({cube_x:+.3f}, {cube_y:+.3f}) m")
    else:
        cube_x, cube_y = _scout_cube_xy(robot, args.target_color)

    env_state = build_env_state(
        args.target_color, args.bowl_x, args.bowl_y,
        cube_x_m=cube_x, cube_y_m=cube_y,
    )
    print(f"[infer] env_state:     {env_state.round(3).tolist()}")

    home_deg = get_home_deg(getattr(args, "home_pose", None) or DEFAULT_HOME_POSE)
    try:
        n_steps = int(args.episode_time_s * args.fps)
        for ep in range(args.num_episodes):
            print(f"\n[infer] === Episode {ep+1}/{args.num_episodes} ===")
            maybe_home_robot(robot, args)
            run_episode(
                robot, policy, preprocessor, postprocessor,
                env_state, n_steps, args.fps, args.device, task_desc=task_desc,
            )
            if ep + 1 < args.num_episodes:
                print(f"[infer] Reset for {args.reset_time_s:.1f}s (manual scene reset)")
                time.sleep(args.reset_time_s)
    finally:
        if getattr(args, "home", True):
            print("\n[infer] Returning to home before disconnect ...", flush=True)
            ramp_to_home(robot, home_deg, fps=args.fps)
        print("\n[infer] Disconnecting robot ...")
        robot.disconnect()
        print("[infer] Done")


if __name__ == "__main__":
    main()
