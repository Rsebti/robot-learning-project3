"""
rehome_only.py - re-set the per-motor homing_offset to make the CURRENT
physical pose read "0 deg" on every joint, while keeping the existing
range_min / range_max from the calibration JSON.

Use this when you've just changed a motor (so its raw-step zero shifted)
but you trust the joint ranges already on file.

Procedure: put the arm into the SAME rest pose you used before, run this
script. It reads the current motor positions, computes new homing_offsets
so present_position becomes the mid-of-range for each joint, and
overwrites the homing_offset values in the calibration JSON.

Usage:
    python -m toolset.calibration_scripts.rehome_only --port COM3
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

# --- Windows camera workaround --------------------------------------------
# On Windows, OpenCV's default MSMF backend hangs on the SO-101 wrist cam.
# DSHOW works. Monkey-patch cv2.VideoCapture to force DSHOW before lerobot
# instantiates its OpenCVCamera.
_OriginalVideoCapture = cv2.VideoCapture


class _DShowVideoCapture(_OriginalVideoCapture):
    def __init__(self, *args, **kwargs):
        # Swap MSMF / CAP_ANY for DSHOW when a (index, backend) form is used
        if len(args) >= 2 and isinstance(args[0], int) and args[1] in (
            cv2.CAP_ANY, cv2.CAP_MSMF
        ):
            args = (args[0], cv2.CAP_DSHOW)
        elif len(args) == 1 and isinstance(args[0], int):
            args = (args[0], cv2.CAP_DSHOW)
        super().__init__(*args, **kwargs)


cv2.VideoCapture = _DShowVideoCapture
# --------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_CALIB = _REPO / "deploy" / "calibration" / "so101_follower.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--camera_index", type=int, default=1)
    parser.add_argument("--calibration", type=Path, default=_DEFAULT_CALIB,
                        help="LeRobot calibration JSON (default: deploy/calibration/)")
    args = parser.parse_args()

    if not args.calibration.exists():
        raise SystemExit(f"calibration not found: {args.calibration}")

    bak = args.calibration.with_suffix(".json.rehome_bak")
    shutil.copy(args.calibration, bak)
    print(f"[rehome] backup -> {bak}")

    with open(args.calibration) as f:
        calib = json.load(f)

    import sys
    if str(_REPO / "deploy") not in sys.path:
        sys.path.insert(0, str(_REPO / "deploy"))
    from robot_calibration import make_so101_follower_config
    from lerobot.robots.utils import make_robot_from_config
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    rconf = make_so101_follower_config(
        args.port,
        cameras={"wrist": OpenCVCameraConfig(
            index_or_path=args.camera_index, fps=30, width=640, height=480,
        )},
        calibration_dir=args.calibration.parent,
        use_degrees=True,
    )
    robot = make_robot_from_config(rconf)
    print(f"[rehome] Connecting to {args.port} ...")
    robot.connect()

    try:
        bus = robot.bus
        # Live preview: show wrist cam, overlay current joint angles,
        # capture when the user presses SPACE.
        print("\n[rehome] Live preview opening.")
        print("  - Move the arm into the desired rest pose.")
        print("  - SPACE: capture this pose and write new homing offsets.")
        print("  - ESC  : cancel (no changes written).")
        win = "rehome - move arm to rest pose, SPACE to capture"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        captured = False
        last_overlay = ""
        while True:
            obs = robot.get_observation()
            img = obs.get("wrist")
            if img is not None:
                frame = np.asarray(img, dtype=np.uint8)
                if frame.shape[-1] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "NO CAMERA FEED", (180, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            # Overlay live joint angles (deg)
            try:
                deg_now = {m: float(obs[f"{m}.pos"]) for m in calib.keys()}
                lines = [f"{m}: {deg_now[m]:+6.1f} deg" for m in calib.keys()]
            except Exception:
                lines = ["(reading joints...)"]
            y = 20
            for ln in lines:
                cv2.putText(frame, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1, cv2.LINE_AA)
                y += 18
            cv2.putText(frame, "SPACE = capture   ESC = cancel",
                        (8, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow(win, frame)
            key = cv2.waitKey(33) & 0xFF
            if key == 27:    # ESC
                print("[rehome] cancelled by user; no changes written.")
                cv2.destroyAllWindows()
                return
            if key == 32:    # SPACE
                captured = True
                break
        cv2.destroyAllWindows()

        if not captured:
            return

        # Pull RAW actual position (NOT normalized, NOT homing-subtracted)
        # by adding the current homing_offset back to Present_Position.
        present = bus.sync_read("Present_Position", normalize=False)
        homing_now = {m: int(calib[m]["homing_offset"]) for m in present.keys()}
        # Lerobot reports Present_Position = Actual_Position - Homing_Offset
        actual = {m: int(present[m]) + homing_now[m] for m in present.keys()}

        print(f"\n[rehome] joint readings (raw):")
        new_homing = {}
        out_of_range = []
        FEETECH_HOMING_LIMIT = 2047    # sign-magnitude 11-bit
        for m, raw in actual.items():
            rmin = int(calib[m]["range_min"])
            rmax = int(calib[m]["range_max"])
            mid = (rmin + rmax) // 2
            # New homing such that (raw - new_homing) == mid -> joint reads 0
            offset = raw - mid
            in_range = rmin <= raw <= rmax
            in_limit = abs(offset) <= FEETECH_HOMING_LIMIT
            marker = "" if (in_range and in_limit) else "  <-- OUT OF CALIBRATED RANGE"
            print(f"  {m:14s}  raw={raw:5d}  range=[{rmin}, {rmax}] mid={mid}  "
                  f"old_homing={homing_now[m]:+5d}  new_homing={offset:+5d}{marker}")
            if not (in_range and in_limit):
                out_of_range.append((m, raw, rmin, rmax, offset))
                continue
            new_homing[m] = offset

        if out_of_range:
            print("\n[rehome] ABORT - one or more joints are OUTSIDE the calibrated range:")
            for m, raw, rmin, rmax, offset in out_of_range:
                print(f"   {m:14s}  raw={raw} not in [{rmin}, {rmax}]  "
                      f"(would need homing offset {offset:+d}, max allowed +/-{FEETECH_HOMING_LIMIT})")
            print("\n[rehome] No writes made (JSON unchanged, no motor writes).")
            print("[rehome] FIX: physically move the offending joint(s) BACK into range")
            print("        first (use the camera preview + live joint readout to verify),")
            print("        then re-run this script. Or do a full lerobot-calibrate if the")
            print("        encoder has wrapped past its expected range.")
            return

        for m, offset in new_homing.items():
            calib[m]["homing_offset"] = int(offset)

        # CRITICAL: write new homing to each motor's onboard register.
        # Feetech motors require Lock=0 (EEPROM unlocked) AND torque disabled
        # before Homing_Offset writes persist across power cycles. lerobot's
        # `disable_torque` does both. Re-enable (which re-locks) at the end
        # so the change is committed to EEPROM.
        print(f"\n[rehome] disabling torque + unlocking EEPROM for persistent write ...")
        bus.disable_torque()
        print(f"[rehome] writing Homing_Offset to each motor register ...")
        for m, new_off in new_homing.items():
            try:
                bus.write("Homing_Offset", m, int(new_off))
                print(f"  {m:14s}  Homing_Offset <- {int(new_off):+d}  OK")
            except Exception as e:
                print(f"  {m:14s}  FAILED: {e}")
        # Re-enable torque (also re-locks EEPROM -> values persist across power cycles)
        print(f"[rehome] re-enabling torque (Lock=1, EEPROM committed).")
        bus.enable_torque()

        # Write JSON copy
        with open(args.calibration, "w") as f:
            json.dump(calib, f, indent=4)
        print(f"\n[rehome] wrote new homing offsets to {args.calibration}")
        print(f"[rehome] revert with: copy {bak.name} {args.calibration.name}")
        print("[rehome] then: python -m toolset.calibration_scripts.manage_calibration install")

        # Verify: re-read Present_Position; with new homing offsets it
        # should now read near the mid of the calibrated range for each.
        verify = bus.sync_read("Present_Position", normalize=False)
        print(f"\n[rehome] verification (each value should be ~0):")
        for m, v in verify.items():
            rmin = int(calib[m]["range_min"])
            rmax = int(calib[m]["range_max"])
            mid = (rmin + rmax) // 2
            print(f"  {m:14s}  Present_Position={int(v):+5d}  (target mid={mid}, "
                  f"so deviation = {int(v) - mid:+d} steps)")
    finally:
        robot.disconnect()
        print("[rehome] Disconnected.")


if __name__ == "__main__":
    main()
