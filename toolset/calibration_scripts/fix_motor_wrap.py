"""
fix_motor_wrap.py - per-motor calibration that survives an encoder wrap.

WHY this exists:
  lerobot-calibrate sweeps every motor in one pass. If a motor's mechanical
  range crosses the Feetech encoder's 0<->4095 wrap point (which happens
  after a motor swap or a remount), the resulting range_min/range_max it
  records will be the *post-wrap* arc only, not the full physical span.
  The calibration JSON then maps "0 deg policy" to the wrong physical
  pose, and any trained policy that homes to "0 deg elbow" jumps ~90 deg.

WHAT this does (mirrors your colleague's fix protocol):
  1. Disable torque on the target motor.
  2. Call set_half_turn_homings([motor]) - puts Present_Position=2047 at
     wherever the joint is right now (the operator-chosen "center").
  3. Operator sweeps the joint through its FULL physical range. We sample
     Present_Position at 20 Hz for `--sweep_s` seconds.
  4. Detect a wrap: a large gap (>500 steps) between sample clusters means
     the joint crossed 4095<->0. Unwrap by shifting the low cluster +4096.
  5. Recompute geometric center = (unwrapped_min + unwrapped_max) / 2 and
     write a final Homing_Offset that puts that geometric center at 2047.
  6. Write Min_Position_Limit and Max_Position_Limit centered around 2047
     in the new Homing_Offset frame.
  7. Update the lerobot calibration JSON to match.

Usage:
    python -m toolset.calibration_scripts.fix_motor_wrap \\
        --port COM3 --motor elbow_flex --sweep_s 15
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2

# Windows MSMF -> DSHOW patch (lerobot may touch cv2 even with empty cameras)
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


CALIB_PATH = (
    Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"
    / "robots" / "so_follower" / "so101_follower.json"
)
FEETECH_MAX_RES = 4095          # 12-bit encoder
FEETECH_HALF_TURN = 2047
WRAP_GAP_THRESHOLD = 500        # steps; >this between clusters = wrap


def detect_and_unwrap(samples: list[int]) -> tuple[list[int], bool]:
    """If samples have a large gap, shift the low cluster by +4096 to unwrap.

    Returns (unwrapped, did_wrap).
    """
    s = sorted(set(samples))
    if len(s) < 2:
        return list(samples), False
    # find largest gap
    gaps = [(s[i + 1] - s[i], i) for i in range(len(s) - 1)]
    max_gap, idx = max(gaps)
    if max_gap < WRAP_GAP_THRESHOLD:
        return list(samples), False
    threshold = s[idx]  # values <= threshold are "low cluster"
    unwrapped = [v + (FEETECH_MAX_RES + 1) if v <= threshold else v for v in samples]
    return unwrapped, True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--motor", required=True,
                        help="Motor name: shoulder_pan / shoulder_lift / elbow_flex / "
                             "wrist_flex / wrist_roll / gripper")
    parser.add_argument("--sweep_s", type=float, default=15.0,
                        help="Seconds to sample during the manual sweep.")
    parser.add_argument("--sample_hz", type=float, default=20.0)
    parser.add_argument("--calibration", type=Path, default=CALIB_PATH)
    args = parser.parse_args()

    if not args.calibration.exists():
        raise SystemExit(f"calibration JSON not found: {args.calibration}")
    bak = args.calibration.with_suffix(".json.fixwrap_bak")
    shutil.copy(args.calibration, bak)
    print(f"[fix] backup -> {bak}")

    # Connect robot with empty cameras (skip camera init)
    deploy_dir = Path(__file__).resolve().parents[2] / "deploy"
    if str(deploy_dir) not in sys.path:
        sys.path.insert(0, str(deploy_dir))
    from robot_calibration import make_so101_follower_config  # noqa: E402
    from lerobot.robots.utils import make_robot_from_config  # noqa: E402

    rconf = make_so101_follower_config(args.port, cameras={}, use_degrees=True)
    robot = make_robot_from_config(rconf)
    print(f"[fix] Connecting to {args.port} ...")
    robot.connect()

    bus = robot.bus
    if args.motor not in bus.motors:
        raise SystemExit(f"motor '{args.motor}' not on the bus; valid: {list(bus.motors.keys())}")

    try:
        # --- Step 1: disable torque so the user can hand-move freely
        print(f"\n[fix] disabling torque (you may need to support the arm) ...")
        input(">>> support the arm, then press Enter. ")
        bus.disable_torque()

        # --- Step 2: half-turn homing at the user-chosen physical center
        print(f"\n[fix] move {args.motor!r} to its rough physical center "
              f"(any pose near mid-range).")
        input(">>> press Enter once positioned. ")
        bus.set_half_turn_homings([args.motor])
        print(f"[fix] set_half_turn_homings done. Present_Position is now 2047 at this pose.")

        # --- Step 3: sweep + sample
        print(f"\n[fix] move {args.motor!r} slowly through its FULL physical range "
              f"(both extremes, multiple times). Sampling for {args.sweep_s:.0f}s.")
        input(">>> press Enter to start sampling. ")
        period = 1.0 / args.sample_hz
        end_t = time.time() + args.sweep_s
        samples_raw = []   # Present_Position values
        last_print = 0
        while time.time() < end_t:
            t0 = time.time()
            pos = bus.sync_read("Present_Position", [args.motor], normalize=False)
            samples_raw.append(int(pos[args.motor]))
            if t0 - last_print > 0.5:
                print(f"  pos = {samples_raw[-1]:4d}   ({len(samples_raw)} samples)",
                      end="\r", flush=True)
                last_print = t0
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
        print(f"\n[fix] collected {len(samples_raw)} samples; raw range "
              f"[{min(samples_raw)}, {max(samples_raw)}].")

        # --- Step 4: detect + unwrap
        unwrapped, did_wrap = detect_and_unwrap(samples_raw)
        u_min, u_max = min(unwrapped), max(unwrapped)
        span = u_max - u_min
        print(f"[fix] wrap detected: {did_wrap}.  unwrapped range [{u_min}, {u_max}]  "
              f"span={span} ({span * 360 / 4096:.1f} deg)")
        if span > FEETECH_MAX_RES + 1:
            print(f"[fix] WARNING: unwrapped span > encoder resolution; sweep likely "
                  f"over-included noise. Re-run with cleaner sweep.")

        # --- Step 5: recompute homing so unwrapped center -> 2047
        center = (u_min + u_max) // 2
        # current Homing_Offset (in EEPROM) was set by set_half_turn_homings;
        # Present_Position = Actual_Position - Homing_Offset = 2047 at the
        # operator's "center". To shift the *geometric* center to 2047 we add
        # (center - 2047) to Homing_Offset.
        current_homing = int(bus.sync_read(
            "Homing_Offset", [args.motor], normalize=False)[args.motor])
        shift = center - FEETECH_HALF_TURN
        new_homing = current_homing + shift
        # Min/Max in the new homing frame, centered around 2047:
        new_min = u_min - center + FEETECH_HALF_TURN
        new_max = u_max - center + FEETECH_HALF_TURN
        print(f"[fix] geometric center (unwrapped) = {center}  shift = {shift:+d}")
        print(f"[fix] new Homing_Offset = {new_homing}")
        print(f"[fix] new Min_Position_Limit / Max_Position_Limit = {new_min} / {new_max}")

        # Sanity on Feetech limits
        if abs(new_homing) > FEETECH_HALF_TURN:
            print(f"[fix] WARNING: new Homing_Offset {new_homing} exceeds Feetech "
                  f"+/-{FEETECH_HALF_TURN} limit. The geometric center is too far from "
                  f"the user-chosen center. Re-run and pick a center closer to the "
                  f"true geometric middle of the sweep.")
            sys.exit(2)

        # --- Step 6: write EEPROM (Lock is already 0 because torque is disabled)
        print(f"\n[fix] writing EEPROM ...")
        bus.write("Homing_Offset",      args.motor, int(new_homing), normalize=False)
        bus.write("Min_Position_Limit", args.motor, int(new_min),    normalize=False)
        bus.write("Max_Position_Limit", args.motor, int(new_max),    normalize=False)
        print(f"[fix] EEPROM written.")

        # Verify by reading back
        rb = bus.sync_read(
            ["Homing_Offset", "Min_Position_Limit", "Max_Position_Limit"],
            [args.motor],
            normalize=False,
        ) if False else None  # sync_read takes a single field; do single reads:
        rb_homing = bus.sync_read("Homing_Offset",      [args.motor], normalize=False)[args.motor]
        rb_min    = bus.sync_read("Min_Position_Limit", [args.motor], normalize=False)[args.motor]
        rb_max    = bus.sync_read("Max_Position_Limit", [args.motor], normalize=False)[args.motor]
        print(f"[fix] readback: Homing_Offset={rb_homing}  Min={rb_min}  Max={rb_max}")

        # --- Step 7: update JSON
        with open(args.calibration) as f:
            calib = json.load(f)
        calib[args.motor]["homing_offset"] = int(rb_homing)
        calib[args.motor]["range_min"]     = int(rb_min)
        calib[args.motor]["range_max"]     = int(rb_max)
        with open(args.calibration, "w") as f:
            json.dump(calib, f, indent=4)
        print(f"[fix] updated {args.calibration}")
        print(f"[fix] revert with: copy {bak.name} {args.calibration.name}")
    finally:
        try:
            bus.enable_torque()
        except Exception as e:
            print(f"[fix] could not re-enable torque: {e}")
        robot.disconnect()
        print("[fix] Disconnected.")


if __name__ == "__main__":
    main()
