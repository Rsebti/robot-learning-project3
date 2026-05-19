"""
calibrate_camera_chessboard.py - solve for the wrist camera's intrinsic
matrix K and distortion coefficients via a printed chessboard pattern.

Two capture modes:
  --mode live    - the robot is connected; press Enter at each pose to
                   grab one frame. You manually jog the robot.
  --mode offline - point at a folder of pre-captured PNG/JPG frames.

Output: NPZ + YAML in toolset/configs/kinematics/. The YAML format
matches what `cube_localization.py` expects.

Pattern defaults match OpenCV's `chessboard.png` (9x6 internal corners,
square_size 0.025 m). Override via CLI if your print is different.

Quality gate:
  - per-image reprojection error printed for each frame
  - global reprojection error MUST be < 1.0 pixel; ideally < 0.5
  - if > 1.0 px, the script prints diagnostics but still writes output;
    re-take images and re-run.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "toolset" / "configs" / "kinematics"


def find_corners_sb(gray: np.ndarray, pattern_size: tuple[int, int]):
    """Prefer findChessboardCornersSB (subpixel) when available."""
    if hasattr(cv2, "findChessboardCornersSB"):
        ok, corners = cv2.findChessboardCornersSB(gray, pattern_size)
        if ok:
            return ok, corners
    ok, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if ok:
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3),
        )
    return ok, corners


def build_objp(pattern_size: tuple[int, int], square_size: float) -> np.ndarray:
    cols, rows = pattern_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
    return objp


def run_offline(image_dir: Path, pattern_size, square_size, save_overlay: bool):
    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")
    )
    if not image_paths:
        raise SystemExit(f"No images found in {image_dir}.")
    print(f"[chess] {len(image_paths)} image(s) to process.")
    objp = build_objp(pattern_size, square_size)
    objpoints, imgpoints = [], []
    img_size = None
    found_paths = []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"  skip {p.name}: failed to read.")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = gray.shape[::-1]
        ok, corners = find_corners_sb(gray, pattern_size)
        if not ok:
            print(f"  {p.name}: chessboard NOT found.")
            continue
        objpoints.append(objp.copy())
        imgpoints.append(corners)
        found_paths.append(p)
        print(f"  {p.name}: found {len(corners)} corners.")
        if save_overlay:
            cv2.drawChessboardCorners(img, pattern_size, corners, ok)
            (image_dir / "_overlay").mkdir(exist_ok=True)
            cv2.imwrite(str(image_dir / "_overlay" / p.name), img)
    return objpoints, imgpoints, img_size, found_paths


def run_live(port: str, camera_index: int, n_target: int, pattern_size, square_size,
             save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    from lerobot.robots.utils import make_robot_from_config
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    rconf = SO101FollowerConfig(
        port=port, id="so101_follower",
        cameras={"wrist": OpenCVCameraConfig(
            index_or_path=camera_index, fps=30, width=640, height=480,
        )},
    )
    robot = make_robot_from_config(rconf)
    print(f"[chess] Connecting to {port} (need camera at index {camera_index}).")
    robot.connect()

    objp = build_objp(pattern_size, square_size)
    objpoints, imgpoints = [], []
    img_size = None
    captured = 0
    try:
        while captured < n_target:
            input(f"\n[chess] Move arm so the chessboard fills ~60% of the wrist view, "
                  f"varied angle. Capture #{captured + 1}/{n_target}. Press Enter. ")
            obs = robot.get_observation()
            img = obs["wrist"]
            if img is None:
                print("  camera returned None; retry.")
                continue
            img = np.asarray(img, dtype=np.uint8)
            if img.shape[-1] == 3:
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                bgr = img
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            if img_size is None:
                img_size = gray.shape[::-1]
            ok, corners = find_corners_sb(gray, pattern_size)
            ts = int(time.time())
            cv2.imwrite(str(save_dir / f"chess_{ts}.png"), bgr)
            if not ok:
                print("  chessboard NOT found in this frame - retake.")
                continue
            objpoints.append(objp.copy())
            imgpoints.append(corners)
            captured += 1
            print(f"  found {len(corners)} corners.")
    finally:
        robot.disconnect()
        print("[chess] Disconnected.")
    return objpoints, imgpoints, img_size, []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["live", "offline"], required=True)
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--camera_index", type=int, default=1)
    parser.add_argument("--n_captures", type=int, default=20,
                        help="Live mode: number of frames to capture.")
    parser.add_argument("--image_dir", type=Path,
                        help="Offline mode: folder with PNG/JPG frames.")
    parser.add_argument("--pattern_cols", type=int, default=9,
                        help="Number of INTERNAL corner columns (default 9 for 10x7 board).")
    parser.add_argument("--pattern_rows", type=int, default=6)
    parser.add_argument("--square_size_m", type=float, default=0.025,
                        help="Side length of one square in meters (MEASURE YOUR PRINT).")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--save_overlays", action="store_true")
    args = parser.parse_args()

    pattern_size = (args.pattern_cols, args.pattern_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "live":
        save_dir = args.output_dir / "chess_captures"
        objpoints, imgpoints, img_size, _ = run_live(
            args.port, args.camera_index, args.n_captures,
            pattern_size, args.square_size_m, save_dir,
        )
    else:
        if not args.image_dir or not args.image_dir.exists():
            print("[chess] --image_dir required and must exist for offline mode.")
            sys.exit(2)
        objpoints, imgpoints, img_size, _ = run_offline(
            args.image_dir, pattern_size, args.square_size_m, args.save_overlays,
        )

    if len(objpoints) < 5:
        print(f"[chess] only {len(objpoints)} good frame(s); need >= 5 (ideally 15+).")
        sys.exit(1)

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None,
    )
    # Per-frame reprojection error
    mean_err = 0.0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
        mean_err += err
    mean_err /= max(1, len(objpoints))

    print(f"\n[chess] RMS reprojection error (cv2): {ret:.4f} px")
    print(f"[chess] mean per-point reprojection error: {mean_err:.4f} px")
    if mean_err > 1.0:
        print("[chess] WARNING: reprojection error > 1 px. Recommend retake with")
        print("        more varied poses (tilt, distance, board orientation).")

    # Save
    npz_path = args.output_dir / "camera_intrinsics.npz"
    yaml_path = args.output_dir / "camera_intrinsics.yaml"
    np.savez(npz_path, K=K, dist=dist, image_size=img_size, mean_reproj_err=mean_err)
    with open(yaml_path, "w") as f:
        yaml.safe_dump(
            {
                "K": K.tolist(),
                "dist_coeffs": dist.flatten().tolist(),
                "image_size": list(img_size),
                "mean_reproj_err_px": float(mean_err),
                "n_frames": int(len(objpoints)),
                "pattern": list(pattern_size),
                "square_size_m": float(args.square_size_m),
            },
            f, default_flow_style=False, sort_keys=False,
        )
    print(f"\n[chess] wrote {npz_path}")
    print(f"[chess] wrote {yaml_path}")


if __name__ == "__main__":
    main()
