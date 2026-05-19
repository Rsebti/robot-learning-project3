"""
snap_wrist_frame.py - capture a single frame from the SO-101 wrist camera
and save to disk. Use for color calibration, visual inspection, or as a
chessboard frame for offline calibration.

Default output: deploy/_snaps/snap_<timestamp>.png

Usage:
    python -m toolset.control.snap_wrist_frame --port COM3
    python -m toolset.control.snap_wrist_frame --port COM3 --n 5 --interval_s 0.5
    python -m toolset.control.snap_wrist_frame --port COM3 --out scene.png
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

# Windows MSMF -> DSHOW patch
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--camera_index", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path. Defaults to deploy/_snaps/snap_<ts>.png "
                             "(or _NNN suffixed when --n > 1).")
    parser.add_argument("--n", type=int, default=1,
                        help="Number of frames to capture.")
    parser.add_argument("--interval_s", type=float, default=0.5,
                        help="Seconds between frames when --n > 1.")
    parser.add_argument("--warmup_s", type=float, default=1.0,
                        help="Throw away frames for this long before saving (lets "
                             "exposure settle).")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    if args.out is None:
        outdir = project_root / "deploy" / "_snaps"
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)

    from lerobot.robots.utils import make_robot_from_config
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    cfg = SO101FollowerConfig(
        port=args.port, id="so101_follower",
        cameras={"wrist": OpenCVCameraConfig(
            index_or_path=args.camera_index, fps=30, width=640, height=480,
        )},
    )
    robot = make_robot_from_config(cfg)
    print(f"[snap] Connecting to {args.port} ...")
    robot.connect()
    try:
        # Warmup
        t_end = time.time() + args.warmup_s
        while time.time() < t_end:
            robot.get_observation()
        print(f"[snap] warmup done; capturing {args.n} frame(s).")
        saved = []
        for i in range(args.n):
            obs = robot.get_observation()
            img = obs.get("wrist")
            if img is None:
                print(f"  [{i}] no frame; skipping.")
                continue
            arr = np.asarray(img, dtype=np.uint8)
            if arr.shape[-1] == 3:
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            else:
                bgr = arr
            ts = int(time.time())
            if args.out is None:
                path = (project_root / "deploy" / "_snaps"
                        / f"snap_{ts}_{i:03d}.png")
            elif args.n == 1:
                path = args.out
            else:
                path = args.out.with_stem(f"{args.out.stem}_{i:03d}")
            cv2.imwrite(str(path), bgr)
            saved.append(path)
            print(f"  [{i}] saved -> {path}  ({bgr.shape[1]}x{bgr.shape[0]})")
            if i + 1 < args.n:
                time.sleep(args.interval_s)
        print(f"\n[snap] {len(saved)} frame(s) saved.")
        for p in saved:
            print(f"  {p}")
    finally:
        robot.disconnect()
        print("[snap] Disconnected.")


if __name__ == "__main__":
    main()
