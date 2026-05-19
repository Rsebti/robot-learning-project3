"""
measure_gripper_tip.py - measure the offset from gripper_frame_link to
the closed-gripper jaw centerpoint, by:

  1. asking the operator to touch a known-height surface (e.g. the table)
     with the closed gripper tip,
  2. running FK at the current motor angles to get gripper_frame xyz,
  3. comparing to the known surface height.

The recovered offset is the vector (in the gripper_frame_link frame) from
that link origin to the contact point of the closed jaws.

Limitation: a single measurement only constrains the offset along the
gripper-axis direction. To get all three components (x, y, z in the
gripper frame) we need 3 measurements with different gripper orientations.
The default protocol does 3.

Output: prints the recovered offset and writes it to
`toolset/configs/kinematics/config.yaml` (replaces the placeholder
gripper_tip_offset_m). Backs up the old yaml as `.bak`.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from toolset.kinematics.config import DEFAULT_CONFIG_PATH, KinematicsConfig
from toolset.kinematics.motor_to_urdf import MotorToUrdfConfig, read_motor_deg_from_obs
from toolset.kinematics.urdf_fk import SO101FK


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--camera_index", type=int, default=1)
    parser.add_argument("--surface_z_m", type=float, default=None,
                        help="Known z (base frame) of the surface you touch. "
                             "Default: KinematicsConfig.table_z_m.")
    parser.add_argument("--n_poses", type=int, default=3,
                        help="Number of touch poses (with different gripper orientations).")
    args = parser.parse_args()

    kcfg = KinematicsConfig.load()
    mcfg = MotorToUrdfConfig.load()
    surface_z = args.surface_z_m if args.surface_z_m is not None else kcfg.table_z_m

    if all(abs(v) < 1e-6 for v in mcfg.offsets_deg.values()):
        print("[meas] motor_offsets.yaml is identity. Run calibrate_motor_offsets first.")
        sys.exit(2)

    # FK with current (possibly zero) offset so measurements describe the
    # OFFSET CORRECTION relative to gripper_frame_link.
    fk = SO101FK(kcfg.urdf_path, gripper_tip_offset=(0.0, 0.0, 0.0))

    from lerobot.robots.utils import make_robot_from_config
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    rconf = SO101FollowerConfig(
        port=args.port, id="so101_follower",
        cameras={"wrist": OpenCVCameraConfig(
            index_or_path=args.camera_index, fps=30, width=640, height=480,
        )},
    )
    robot = make_robot_from_config(rconf)
    print(f"[meas] Connecting to {args.port} ...")
    robot.connect()

    # Each pose i gives: surface_z = (T_frame @ [offset; 1])[2]
    # i.e. (R_frame[2,:] dot offset_local) + t_frame[2] = surface_z
    # That's one linear equation in 3 unknowns; need >= 3 independent
    # orientations.
    A_rows = []
    b_rows = []
    pose_records = []
    try:
        for i in range(args.n_poses):
            print(f"\n=== Touch pose {i + 1}/{args.n_poses} ===")
            print(f"  Surface z (base frame) = {surface_z:.4f} m")
            print("  Move the arm so the CLOSED gripper just touches the surface.")
            print("  Use a different gripper orientation each pose for best conditioning.")
            input("  Press Enter when in contact, steady. ")
            obs = robot.get_observation()
            motor_deg = read_motor_deg_from_obs(obs)
            urdf = mcfg.motor_to_urdf_rad(motor_deg)
            T_frame = fk.fk(urdf, target="gripper_frame")["T"]
            R = T_frame[:3, :3]
            t = T_frame[:3, 3]
            # row of A is R[2, :]   (i.e. R z-row in base frame)
            # rhs is surface_z - t[2]
            A_rows.append(R[2, :])
            b_rows.append(surface_z - t[2])
            pose_records.append({
                "T_frame": T_frame.tolist(),
                "motor_deg": motor_deg.tolist(),
                "urdf_rad": urdf.tolist(),
            })
            print(f"  gripper_frame xyz = {t.round(4).tolist()} m")
    finally:
        robot.disconnect()
        print("[meas] Disconnected.")

    A = np.array(A_rows, dtype=float)        # (n_poses, 3)
    b = np.array(b_rows, dtype=float)        # (n_poses,)
    if A.shape[0] >= 3:
        offset_local, *_ = np.linalg.lstsq(A, b, rcond=None)
    else:
        # Single pose: solve only along the dominant frame-z component
        offset_local = np.array([0.0, 0.0, b[0] / A[0, 2]])

    print(f"\n[meas] recovered gripper_tip offset in gripper_frame_link frame (m):")
    print(f"  x = {offset_local[0]:+.4f}")
    print(f"  y = {offset_local[1]:+.4f}")
    print(f"  z = {offset_local[2]:+.4f}")
    print(f"  norm = {np.linalg.norm(offset_local) * 1000:.1f} mm")

    # Sanity: check consistency
    residual = A @ offset_local - b
    rms_mm = float(np.sqrt(np.mean(residual ** 2)) * 1000)
    print(f"[meas] RMS residual across poses: {rms_mm:.2f} mm "
          f"(< 5 mm = good; > 10 mm = re-do, your touches were uneven)")

    # Write into config.yaml (backup the old)
    bak = DEFAULT_CONFIG_PATH.with_suffix(".yaml.bak")
    shutil.copy(DEFAULT_CONFIG_PATH, bak)
    with open(DEFAULT_CONFIG_PATH) as f:
        data = yaml.safe_load(f) or {}
    data["gripper_tip_offset_m"] = {
        "x": float(offset_local[0]),
        "y": float(offset_local[1]),
        "z": float(offset_local[2]),
    }
    with open(DEFAULT_CONFIG_PATH, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"\n[meas] wrote new offset into {DEFAULT_CONFIG_PATH}")
    print(f"[meas] backup at {bak}")


if __name__ == "__main__":
    main()
