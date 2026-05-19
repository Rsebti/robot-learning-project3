"""
probe_cli.py - read SO-101 motor angles, run URDF FK, print pose.

Usage:
    conda activate trim
    python -m toolset.kinematics.probe_cli --port COM3 --target gripper_tip

Output format includes BOTH the URDF base frame and the user frame
(right+, forward+, up+) so it can be pasted directly into deploy scripts.

Run `calibration_scripts/calibrate_motor_offsets.py` first if you haven't
populated configs/kinematics/motor_offsets.yaml - otherwise the URDF
angles will be wrong by whatever offset your lerobot calibration set.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from toolset.kinematics.config import KinematicsConfig
from toolset.kinematics.motor_to_urdf import MotorToUrdfConfig, read_motor_deg_from_obs
from toolset.kinematics.urdf_fk import SO101FK


def urdf_to_user(xyz_urdf: np.ndarray) -> np.ndarray:
    """URDF (forward+x, left+y, up+z) -> user (right+x, forward+y, up+z)."""
    return np.array([-xyz_urdf[1], xyz_urdf[0], xyz_urdf[2]], dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--camera_index", type=int, default=1,
                        help="Camera index for SO101FollowerConfig (required even if unused).")
    parser.add_argument("--target", default="gripper_tip",
                        choices=["wrist", "wrist_roll", "gripper_frame", "gripper_tip"])
    parser.add_argument("--config", default=None,
                        help="Path to kinematics/config.yaml (default: project config).")
    parser.add_argument("--offsets", default=None,
                        help="Path to motor_offsets.yaml (default: project config).")
    args = parser.parse_args()

    cfg = KinematicsConfig.load(args.config) if args.config else KinematicsConfig.load()
    motor_cfg = (
        MotorToUrdfConfig.load(args.offsets) if args.offsets else MotorToUrdfConfig.load()
    )
    fk = SO101FK(cfg.urdf_path, gripper_tip_offset=cfg.gripper_tip_offset_m)

    from lerobot.robots.utils import make_robot_from_config
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from toolset.perception.cube_localization import WRIST_CAM_HEIGHT, WRIST_CAM_WIDTH

    rconf = SO101FollowerConfig(
        port=args.port,
        id="so101_follower",
        cameras={"wrist": OpenCVCameraConfig(
            index_or_path=args.camera_index, fps=30,
            width=WRIST_CAM_WIDTH, height=WRIST_CAM_HEIGHT,
        )},
    )
    robot = make_robot_from_config(rconf)
    print(f"[probe] Connecting to {args.port} ...")
    robot.connect()

    try:
        obs = robot.get_observation()
        motor_deg = read_motor_deg_from_obs(obs)
        urdf_rad = motor_cfg.motor_to_urdf_rad(motor_deg)

        print(f"\n[probe] motor angles (deg, lerobot): {motor_deg.round(2).tolist()}")
        print(f"[probe] URDF angles (rad, post-offsets): {np.round(urdf_rad, 4).tolist()}")
        print(f"[probe] URDF angles (deg): {np.rad2deg(urdf_rad).round(2).tolist()}")

        out = fk.fk(urdf_rad, target=args.target)
        urdf_xyz = out["position"]
        user_xyz = urdf_to_user(urdf_xyz)
        euler_deg = np.rad2deg(out["euler_xyz"])

        print(f"\n[probe] {args.target} pose in URDF base frame:")
        print(f"        xyz (m) = ({urdf_xyz[0]:+.4f}, {urdf_xyz[1]:+.4f}, {urdf_xyz[2]:+.4f})")
        print(f"        euler XYZ (deg) = ({euler_deg[0]:+.1f}, {euler_deg[1]:+.1f}, {euler_deg[2]:+.1f})")
        print(f"\n[probe] in user frame (right+, forward+, up+):")
        print(f"        x = {user_xyz[0]:+.4f} m")
        print(f"        y = {user_xyz[1]:+.4f} m")
        print(f"        z = {user_xyz[2]:+.4f} m")
        print(f"\n  --cube_x {user_xyz[0]:.3f} --cube_y {user_xyz[1]:.3f}  # paste into deploy")

        # Sanity: warn if motor offsets look like identity
        if all(abs(v) < 1e-6 for v in motor_cfg.offsets_deg.values()):
            print("\n[WARN] motor_offsets.yaml is identity-valued. FK is almost certainly")
            print("       wrong. Run: python -m toolset.calibration_scripts.calibrate_motor_offsets")
    finally:
        robot.disconnect()
        print("[probe] Disconnected.")


if __name__ == "__main__":
    main()
