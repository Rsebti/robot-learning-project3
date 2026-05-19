"""
scout_cube_cli.py - ramp to scout home, detect cube, write JSON, ramp home, disconnect.

Usage:
    python -m toolset.perception.scout_cube_cli --port COM3 --color red `
        --out deploy/_snaps/scout_latest.json
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

DEPLOY = Path(__file__).resolve().parents[2] / "deploy"
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

from homes import get_home_deg
from robot_calibration import make_so101_follower_config
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from toolset.perception.cube_localization import WRIST_CAM_HEIGHT, WRIST_CAM_WIDTH
from lerobot.robots.utils import make_robot_from_config

from toolset.perception.probe_camera_vs_fk import ramp_to_target_deg
from toolset.perception.scout_cube import make_localizer, scout_bowl_xyz_user, write_scout_result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="COM3")
    p.add_argument("--camera_index", type=int, default=0)
    p.add_argument("--home_pose", default="eval1_rest")
    p.add_argument("--color", required=True)
    p.add_argument("--n_frames", type=int, default=10)
    p.add_argument("--obs_hz", type=float, default=5.0)
    p.add_argument("--out", type=Path, default=DEPLOY / "_snaps" / "scout_latest.json")
    p.add_argument("--episode", type=int, default=None)
    p.add_argument("--loosen_hsv", action="store_true")
    args = p.parse_args()

    home_deg = get_home_deg(args.home_pose)
    cfg = make_so101_follower_config(
        args.port,
        cameras={"base_camera": OpenCVCameraConfig(
            index_or_path=args.camera_index, fps=30,
            width=WRIST_CAM_WIDTH, height=WRIST_CAM_HEIGHT)},
        use_degrees=True,
    )
    robot = make_robot_from_config(cfg)
    robot.connect()
    localizer = make_localizer(loosen_color=args.color if args.loosen_hsv else None)
    bowl = None

    try:
        print(f"[scout] ramp -> {args.home_pose!r}", flush=True)
        ramp_to_target_deg(robot, home_deg)
        time.sleep(0.4)
        print(f"[scout] detecting {args.color!r} ({args.n_frames} frames)...", flush=True)
        bowl, log = scout_bowl_xyz_user(
            robot, localizer,
            color=args.color,
            n_frames=args.n_frames,
            obs_hz=args.obs_hz,
        )
        if bowl is None:
            print("[scout] FAILED: no detection", flush=True)
        else:
            print(f"[scout] bowl_xyz user (m): x={bowl[0]:+.3f} y={bowl[1]:+.3f} z={bowl[2]:+.3f}", flush=True)
        write_scout_result(
            args.out, bowl_xyz_user=bowl, color=args.color,
            frame_log=log, home_pose=args.home_pose, episode=args.episode,
        )
        print(f"[scout] -> {args.out}", flush=True)
        print("[scout] ramp home ...", flush=True)
        ramp_to_target_deg(robot, home_deg)
    finally:
        robot.disconnect()
    return 1 if bowl is None else 0


if __name__ == "__main__":
    raise SystemExit(main())
