"""
observe_cubes.py - drive to a 'lookdown' home pose where the wrist cam
sees the table, then continuously run HSV cube detection + back-projection
on each frame. Prints/saves per-color base-frame xyz estimates.

Pipeline per frame:
  agent.get_observation()  -> motor_deg + wrist BGR frame
  CubeLocalizer.locate()   -> (pixel, area, base_xyz) for each requested color
  rolling median           -> stable per-color estimate

Default home preset 'eval1_rest' (universal home, post motor-3 fix). Pass
--home_pose <name> or --target_deg <6 vals> to override.

Usage:
    python -m toolset.perception.observe_cubes --port COM3
    python -m toolset.perception.observe_cubes --port COM3 --colors yellow red orange
    python -m toolset.perception.observe_cubes --port COM3 --hz 5 --max_frames 200 --save_csv

>>> Make sure cubes are placed in front of the robot BEFORE the script
    starts the rollout. The home pose should put the wrist cam over the
    workspace; if you can't see anything, edit deploy/homes.py or pass
    --target_deg manually.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import torch

DEPLOY = Path(__file__).resolve().parents[2] / "deploy"
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

# Windows DSHOW patch for the wrist cam
_OriginalVideoCapture = cv2.VideoCapture


class _DShowVideoCapture(_OriginalVideoCapture):
    def __init__(self, *args, **kwargs):
        if len(args) >= 2 and isinstance(args[0], int) and args[1] in (
            cv2.CAP_ANY, cv2.CAP_MSMF
        ):
            args = (args[0], cv2.CAP_DSHOW)
        elif len(args) == 1 and isinstance(args[0], int):
            args = (args[0], cv2.CAP_DSHOW)
        super().__init__(*args, **kwargs)


cv2.VideoCapture = _DShowVideoCapture

from homes import get_home_deg                                          # noqa: E402
from robot_calibration import make_so101_follower_config                # noqa: E402
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: E402
from toolset.perception.cube_localization import WRIST_CAM_HEIGHT, WRIST_CAM_WIDTH  # noqa: E402
from lerobot.robots.utils import make_robot_from_config                 # noqa: E402

from toolset.perception.cube_localization import CubeLocalizer          # noqa: E402

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                "wrist_flex", "wrist_roll", "gripper"]


def ramp_to_target_deg(robot, target_deg: np.ndarray, fps: int = 30,
                       max_deg_per_step: float = 0.7,
                       abort_diff_deg: float = 30.0,
                       max_total_s: float = 25.0) -> bool:
    """Linear joint-space ramp to target. Returns True on completion."""
    obs = robot.get_observation()
    q_now = np.array([float(obs[f"{n}.pos"]) for n in MOTOR_NAMES], dtype=float)
    q_cmd = q_now.copy()
    period = 1.0 / fps
    start_t = time.time()
    print(f"[ramp] start {q_now.round(1).tolist()}  ->  target {target_deg.round(1).tolist()}")
    while True:
        err = target_deg - q_cmd
        if np.max(np.abs(err)) < 0.5:
            return True
        if time.time() - start_t > max_total_s:
            print(f"[ramp] timeout, stopping at {q_cmd.round(1).tolist()}")
            return False
        step = np.clip(err, -max_deg_per_step, max_deg_per_step)
        q_cmd = q_cmd + step
        action = {f"{n}.pos": float(q_cmd[i]) for i, n in enumerate(MOTOR_NAMES)}
        t0 = time.time()
        robot.send_action(action)
        obs = robot.get_observation()
        q_live = np.array([float(obs[f"{n}.pos"]) for n in MOTOR_NAMES], dtype=float)
        if np.max(np.abs(q_live - q_cmd)) > abort_diff_deg:
            print(f"[ramp] ABORT: joint deviation > {abort_diff_deg} deg")
            return False
        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="COM3")
    p.add_argument("--camera_index", type=int, default=1)
    p.add_argument("--home_pose", default="eval1_rest",
                   help="Named home in deploy/homes.py (default: eval1_rest).")
    p.add_argument("--target_deg", type=float, nargs=6, default=None,
                   help="Override home: explicit 6 servo deg.")
    p.add_argument("--colors", nargs="+",
                   default=["yellow", "red", "orange", "blue", "green", "violet"],
                   help="Which cube colors to detect each frame.")
    p.add_argument("--hz", type=float, default=3.0,
                   help="Frames per second for the observation loop.")
    p.add_argument("--max_frames", type=int, default=0,
                   help="Stop after this many frames (0 = run until Ctrl+C).")
    p.add_argument("--window", type=int, default=20,
                   help="Rolling median window per color (frames).")
    p.add_argument("--save_csv", action="store_true",
                   help="Append per-frame detections to deploy/_snaps/cube_obs_<ts>.csv.")
    p.add_argument("--save_frames", action="store_true",
                   help="Save each captured frame as PNG (heavy).")
    p.add_argument("--no_ramp", action="store_true",
                   help="Skip the ramp-to-home; observe from current pose.")
    args = p.parse_args()

    # Resolve target deg
    if args.target_deg is not None:
        target_deg = np.array(args.target_deg, dtype=float)
        print(f"[obs] using --target_deg = {target_deg.round(1).tolist()}")
    else:
        target_deg = get_home_deg(args.home_pose)
        print(f"[obs] using home preset {args.home_pose!r} = {target_deg.round(1).tolist()}")

    cfg = make_so101_follower_config(
        args.port,
        cameras={"wrist": OpenCVCameraConfig(
            index_or_path=args.camera_index, fps=30,
            width=WRIST_CAM_WIDTH, height=WRIST_CAM_HEIGHT,
        )},
        use_degrees=True,
    )
    robot = make_robot_from_config(cfg)
    print(f"[obs] connecting to {args.port} ...")
    robot.connect()

    localizer = CubeLocalizer()
    print(f"[obs] cube localizer: K[0,0]={localizer.K[0,0]:.1f}  "
          f"T_cam_in_wrist[:3,3]={localizer.T_cam_in_wrist[:3,3].round(3).tolist()}  "
          f"target_z={localizer.target_z:.3f}m")

    # Output buffers
    histories: dict[str, deque] = {c: deque(maxlen=args.window) for c in args.colors}
    csv_path = None
    csv_w = None
    csv_f = None
    if args.save_csv:
        out_dir = DEPLOY / "_snaps"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"cube_obs_{int(time.time())}.csv"
        csv_f = open(csv_path, "w", newline="")
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["frame", "t_s", "color", "px", "py", "area_px",
                        "x_urdf", "y_urdf", "z_urdf",
                        "x_user", "y_user", "z_user",
                        "med_x_user", "med_y_user", "med_z_user", "n_med"])
        print(f"[obs] saving CSV -> {csv_path}")

    frames_dir = DEPLOY / "_snaps" / f"frames_{int(time.time())}"
    if args.save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)
        print(f"[obs] saving frames -> {frames_dir}")

    try:
        # 1. Drive to the home pose (smooth ramp)
        if not args.no_ramp:
            ok = ramp_to_target_deg(robot, target_deg)
            if not ok:
                print("[obs] ramp did not reach target; observing from wherever we stopped.")
            time.sleep(0.5)

        # 2. Observe loop
        period = 1.0 / max(0.1, args.hz)
        frame_idx = 0
        start_t = time.time()
        print(f"\n[obs] starting observation loop at {args.hz:.1f} Hz "
              f"({'forever' if args.max_frames == 0 else f'{args.max_frames} frames'}). "
              f"Ctrl+C to stop.\n")
        while True:
            t0 = time.time()
            obs = robot.get_observation()
            motor_deg = np.array([float(obs[f"{n}.pos"]) for n in MOTOR_NAMES],
                                  dtype=float)
            img = obs.get("wrist")
            if img is None:
                print(f"  [{frame_idx}] no camera frame; skipping.")
                frame_idx += 1
                time.sleep(period)
                continue
            arr = np.asarray(img, dtype=np.uint8)
            bgr = (cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                   if arr.shape[-1] == 3 else arr)

            if args.save_frames:
                cv2.imwrite(str(frames_dir / f"f{frame_idx:05d}.png"), bgr)

            per_frame: list[str] = []
            for color in args.colors:
                det = localizer.locate(bgr, motor_deg, color)
                if det is None:
                    continue
                histories[color].append(det.base_xyz_m)
                med = np.median(np.stack(list(histories[color])), axis=0)
                # user frame
                x_u, y_u, z_u = -det.base_xyz_m[1], +det.base_xyz_m[0], +det.base_xyz_m[2]
                mx_u, my_u, mz_u = -med[1], +med[0], +med[2]
                per_frame.append(
                    f"{color:>7s} pix=({det.pixel[0]:6.1f},{det.pixel[1]:6.1f}) "
                    f"area={int(det.contour_area_px):>5d} "
                    f"xyz_user=({x_u:+.3f},{y_u:+.3f},{z_u:+.3f}) "
                    f"med(n={len(histories[color])})=({mx_u:+.3f},{my_u:+.3f},{mz_u:+.3f})"
                )
                if csv_w is not None:
                    t_s = time.time() - start_t
                    csv_w.writerow([
                        frame_idx, f"{t_s:.3f}", color,
                        f"{det.pixel[0]:.2f}", f"{det.pixel[1]:.2f}",
                        f"{det.contour_area_px:.0f}",
                        f"{det.base_xyz_m[0]:+.4f}", f"{det.base_xyz_m[1]:+.4f}",
                        f"{det.base_xyz_m[2]:+.4f}",
                        f"{x_u:+.4f}", f"{y_u:+.4f}", f"{z_u:+.4f}",
                        f"{mx_u:+.4f}", f"{my_u:+.4f}", f"{mz_u:+.4f}",
                        len(histories[color]),
                    ])

            print(f"frame {frame_idx:4d} ({time.time() - start_t:6.1f}s)  detected: "
                  f"{len(per_frame)}/{len(args.colors)}")
            for line in per_frame:
                print(f"  {line}")
            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        print("\n[obs] interrupted.")
    finally:
        if csv_f is not None:
            csv_f.close()
            print(f"[obs] csv saved -> {csv_path}")

        # Final summary: per-color median over the full session
        print("\n=== Final per-color medians (user frame, right+/forward+/up+) ===")
        for color in args.colors:
            h = histories[color]
            if not h:
                print(f"  {color:>7s}  NEVER detected")
                continue
            arr = np.stack(list(h))
            med = np.median(arr, axis=0)
            x_u, y_u, z_u = -med[1], +med[0], +med[2]
            print(f"  {color:>7s}  n={len(h):3d}  "
                  f"xyz=({x_u:+.3f}, {y_u:+.3f}, {z_u:+.3f}) m  "
                  f"  --bowl_x {x_u:.3f} --bowl_y {y_u:.3f}    # or --cube_x/y")

        robot.disconnect()
        print("[obs] disconnected.")


if __name__ == "__main__":
    main()
