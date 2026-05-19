"""
camera_latency_test.py - measure wrist camera latency WITHOUT connecting
to the robot. Pure OpenCV - works even when the arm calibration is broken.

Usage:
    python -m toolset.control.camera_latency_test --camera_index 1
    python -m toolset.control.camera_latency_test --camera_index 1 --preview
    python -m toolset.control.camera_latency_test --camera_index 1 --n 300
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_index", type=int, default=1)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--target_fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--backend", default="dshow", choices=["dshow", "msmf", "any"])
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    backend_map = {
        "dshow": cv2.CAP_DSHOW,
        "msmf":  cv2.CAP_MSMF,
        "any":   cv2.CAP_ANY,
    }
    backend = backend_map[args.backend]

    print(f"[cam-lat] opening camera index {args.camera_index} via {args.backend} ...")
    cap = cv2.VideoCapture(args.camera_index, backend)
    if not cap.isOpened():
        raise SystemExit(f"failed to open camera index {args.camera_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.target_fps)
    actual_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[cam-lat] camera says {actual_w}x{actual_h} @ {actual_fps:.1f} fps")
    print(f"[cam-lat] sampling {args.n} frames "
          f"(target dt = {1000/args.target_fps:.1f} ms)")

    # Warmup
    for _ in range(5):
        cap.read()

    timestamps = []
    capture_durs = []
    win = "cam_latency - press q to stop"
    if args.preview:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    t_prev = time.perf_counter()
    start = t_prev
    try:
        for i in range(args.n):
            t0 = time.perf_counter()
            ok, frame = cap.read()
            t1 = time.perf_counter()
            if not ok:
                print(f"  frame {i}: read FAILED; continuing.")
                continue
            cap_dur = (t1 - t0) * 1000
            inter = (t1 - t_prev) * 1000
            timestamps.append(t1 - start)
            capture_durs.append(cap_dur)
            t_prev = t1
            if i % 30 == 0:
                print(f"  frame {i:4d}: capture={cap_dur:6.2f}ms  inter={inter:6.2f}ms")
            if args.preview:
                disp = frame.copy()
                cv2.putText(disp, f"capture={cap_dur:5.1f}ms  inter={inter:5.1f}ms",
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow(win, disp)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
    finally:
        if args.preview:
            cv2.destroyAllWindows()
        cap.release()
        print("[cam-lat] camera released.")

    if not capture_durs:
        return
    caps = np.asarray(capture_durs)
    inters = np.diff(np.asarray(timestamps)) * 1000
    print(f"\n[cam-lat] n_frames = {len(caps)}")
    print(f"  capture duration (ms): min={caps.min():.2f}  median={np.median(caps):.2f}  "
          f"mean={caps.mean():.2f}  p95={np.percentile(caps, 95):.2f}  max={caps.max():.2f}")
    if len(inters):
        print(f"  inter-frame dt   (ms): min={inters.min():.2f}  median={np.median(inters):.2f}  "
              f"mean={inters.mean():.2f}  p95={np.percentile(inters, 95):.2f}  max={inters.max():.2f}")
        achieved_hz = 1000.0 / np.median(inters)
        print(f"  achieved polling rate (median): {achieved_hz:.1f} Hz")
        if achieved_hz < args.target_fps * 0.9:
            print(f"  WARN: {achieved_hz:.1f} Hz << target {args.target_fps} Hz; "
                  f"camera/USB bottleneck.")

    out_dir = Path(__file__).parent.parent.parent / "deploy" / "_snaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"cam_latency_{int(time.time())}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "timestamp_s", "capture_ms"])
        for i, (ts, cap_ms) in enumerate(zip(timestamps, capture_durs)):
            w.writerow([i, ts, cap_ms])
    print(f"  CSV -> {csv_path}")


if __name__ == "__main__":
    main()
