"""
calibrate_hand_eye.py - solve for T_cam_in_wrist (camera extrinsics
expressed in the wrist frame) via OpenCV's calibrateHandEye.

Procedure:
  1. Tape a chessboard to the table (any pose works; it just needs to be
     stationary).
  2. Run this script. It will guide you through 10-20 poses where the
     wrist camera can see the chessboard.
  3. At each pose: capture frame -> solvePnP -> stash with current wrist
     pose from FK.
  4. cv2.calibrateHandEye -> T_cam_in_wrist.

Requires:
  - configs/kinematics/motor_offsets.yaml (else FK is wrong)
  - configs/kinematics/camera_intrinsics.npz (else PnP is wrong)

Output: configs/kinematics/camera_in_wrist.{npy,yaml}
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from toolset.kinematics.config import KinematicsConfig
from toolset.kinematics.motor_to_urdf import MotorToUrdfConfig, read_motor_deg_from_obs
from toolset.kinematics.urdf_fk import SO101FK


PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "toolset" / "configs" / "kinematics"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--camera_index", type=int, default=1)
    parser.add_argument("--n_poses", type=int, default=15)
    parser.add_argument("--pattern_cols", type=int, default=9)
    parser.add_argument("--pattern_rows", type=int, default=6)
    parser.add_argument("--square_size_m", type=float, default=0.025)
    parser.add_argument("--intrinsics", type=Path,
                        default=DEFAULT_OUTPUT_DIR / "camera_intrinsics.npz")
    parser.add_argument("--method", default="TSAI",
                        choices=["TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS"])
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if not args.intrinsics.exists():
        print(f"[handeye] missing camera intrinsics: {args.intrinsics}")
        print("[handeye] run calibrate_camera_chessboard.py first.")
        sys.exit(2)
    data = np.load(args.intrinsics)
    K = data["K"]
    dist = data["dist"]
    print(f"[handeye] intrinsics K=\n{K}\n  dist={dist.flatten()}")

    pattern_size = (args.pattern_cols, args.pattern_rows)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = (
        np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        * args.square_size_m
    )

    kcfg = KinematicsConfig.load()
    mcfg = MotorToUrdfConfig.load()
    if all(abs(v) < 1e-6 for v in mcfg.offsets_deg.values()):
        print("[handeye] motor offsets are identity. Run calibrate_motor_offsets first.")
        sys.exit(2)
    fk = SO101FK(kcfg.urdf_path, gripper_tip_offset=kcfg.gripper_tip_offset_m)

    # Connect robot
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
    print(f"[handeye] Connecting to {args.port} ...")
    robot.connect()

    # gather pose pairs
    # gripper2base ~= T_wrist (wrist frame in base frame)
    # target2cam ~= T_board_in_camera (board frame in camera frame)
    R_g2b, t_g2b = [], []
    R_t2c, t_t2c = [], []
    captures_dir = args.output_dir / "handeye_captures"
    captures_dir.mkdir(parents=True, exist_ok=True)

    captured = 0
    try:
        while captured < args.n_poses:
            input(f"\n[handeye] Move arm so the wrist cam sees the chessboard. "
                  f"Capture #{captured + 1}/{args.n_poses}. Press Enter. ")
            obs = robot.get_observation()
            img = obs["wrist"]
            if img is None:
                print("  camera returned None; retry.")
                continue
            bgr = cv2.cvtColor(np.asarray(img, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            ts = int(time.time())
            cv2.imwrite(str(captures_dir / f"handeye_{ts}.png"), bgr)

            # find corners
            if hasattr(cv2, "findChessboardCornersSB"):
                ok, corners = cv2.findChessboardCornersSB(gray, pattern_size)
            else:
                ok, corners = cv2.findChessboardCorners(gray, pattern_size)
                if ok:
                    corners = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1),
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3),
                    )
            if not ok:
                print("  chessboard NOT found. Retake.")
                continue

            # PnP -> T_board_in_camera
            ok_pnp, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)
            if not ok_pnp:
                print("  solvePnP failed. Retake.")
                continue
            R_board, _ = cv2.Rodrigues(rvec)
            R_t2c.append(R_board)
            t_t2c.append(tvec.flatten())

            # FK -> T_wrist in base frame
            motor_deg = read_motor_deg_from_obs(obs)
            urdf = mcfg.motor_to_urdf_rad(motor_deg)
            # Use "wrist" target (joint-5 link tip) - the camera is rigidly
            # mounted on wrist_link, so calibrateHandEye recovers cam in wrist.
            wrist = fk.fk(urdf, target="wrist")
            R_g2b.append(wrist["rotation_matrix"])
            t_g2b.append(wrist["position"])

            captured += 1
            print(f"  ok. wrist xyz = {wrist['position'].round(3).tolist()},  "
                  f"board distance = {np.linalg.norm(tvec):.3f} m")
    finally:
        robot.disconnect()
        print("[handeye] Disconnected.")

    method_map = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    R_c2w, t_c2w = cv2.calibrateHandEye(
        R_g2b, t_g2b, R_t2c, t_t2c, method=method_map[args.method],
    )
    T_cam_in_wrist = np.eye(4, dtype=float)
    T_cam_in_wrist[:3, :3] = R_c2w
    T_cam_in_wrist[:3, 3] = t_c2w.flatten()

    # Diagnostic: estimated T_board_in_base should be consistent across poses
    T_board_in_base_all = []
    for i in range(len(R_g2b)):
        T_w_in_b = np.eye(4); T_w_in_b[:3, :3] = R_g2b[i]; T_w_in_b[:3, 3] = t_g2b[i]
        T_b_in_c = np.eye(4); T_b_in_c[:3, :3] = R_t2c[i]; T_b_in_c[:3, 3] = t_t2c[i]
        T_board_in_base = T_w_in_b @ T_cam_in_wrist @ T_b_in_c
        T_board_in_base_all.append(T_board_in_base[:3, 3])
    positions = np.stack(T_board_in_base_all)
    std_mm = positions.std(axis=0) * 1000
    print(f"\n[handeye] board-position std across poses (should be small): "
          f"{std_mm.round(2).tolist()} mm")
    if std_mm.max() > 10:
        print("[handeye] WARNING: std > 1 cm. Calibration is likely poor.")
        print("           - was the board absolutely stationary?")
        print("           - is FK trustworthy (motor offsets correct)?")
        print("           - reprojection error of intrinsics < 1 px?")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "camera_in_wrist.npy", T_cam_in_wrist)
    with open(args.output_dir / "camera_in_wrist.yaml", "w") as f:
        yaml.safe_dump(
            {
                "T_cam_in_wrist": T_cam_in_wrist.tolist(),
                "method": args.method,
                "n_poses": len(R_g2b),
                "board_position_std_mm": std_mm.tolist(),
            },
            f, default_flow_style=False, sort_keys=False,
        )
    print(f"[handeye] wrote camera_in_wrist.npy / .yaml in {args.output_dir}")


if __name__ == "__main__":
    main()
