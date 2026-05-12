"""
probe_bowl_position.py — read wrist (x, y, z) in robot base frame.

Steps:
  1. Power on the SO-101 follower, connect USB.
  2. Manually position the arm so the WRIST JOINT is directly above the bowl
     center (wrist is more stable than the gripper tip).
  3. Run this script — it reads joint angles and computes FK.
  4. Prints the bowl position and appends it to configs/bowl_positions.yaml.

Usage:
  python teleop/probe_bowl_position.py --port COM3 --label bowl_eval1
  python teleop/probe_bowl_position.py --port COM3 --label bowl_eval2_red
  python teleop/probe_bowl_position.py --port COM3 --list

  On Linux:  --port /dev/ttyACM0
  On Windows: --port COM3  (check Device Manager → Ports if unsure)
"""

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# FK — DH parameters for SO-101 (wrist frame, 5 joints, no gripper)
# ---------------------------------------------------------------------------
# These are the standard SO-101 / SO-ARM100 DH parameters in meters.
# Source: TheRobotStudio/SO-ARM100 URDF (matches the lerobot calibration).
#
# Convention: modified DH (Craig), joints in order:
#   1 = shoulder_pan   (revolute, z-axis)
#   2 = shoulder_lift  (revolute, y-axis)
#   3 = elbow_flex     (revolute, y-axis)
#   4 = wrist_flex     (revolute, y-axis)
#   5 = wrist_roll     (revolute, z-axis)   ← this is the "wrist" frame
#
# Link lengths (m) taken from the SO101 URDF visual meshes:
#   L1 = 0.0   (base to shoulder_pan offset along z)
#   L2 = 0.117 (upper arm)
#   L3 = 0.136 (forearm)
#   L4 = 0.049 (wrist_flex link length)
#   d5 = 0.0   (wrist_roll offset)

_L_BASE_Z   = 0.0595   # height of shoulder_pan joint above base plate (m)
_L_UPPER    = 0.1165   # shoulder_lift → elbow (upper arm, m)
_L_FOREARM  = 0.1360   # elbow → wrist_flex (forearm, m)
_L_WRIST    = 0.0490   # wrist_flex → wrist_roll (m)


def _rot_z(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0, 0],
                     [s,  c, 0, 0],
                     [0,  0, 1, 0],
                     [0,  0, 0, 1]])


def _rot_y(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[ c, 0, s, 0],
                     [ 0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [ 0, 0, 0, 1]])


def _trans(x, y, z):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


def fk_wrist(q: list[float]) -> np.ndarray:
    """Forward kinematics to the wrist_roll frame.

    Args:
        q: [q1, q2, q3, q4, q5] in radians.
           q1 = shoulder_pan, q2 = shoulder_lift, q3 = elbow_flex,
           q4 = wrist_flex,   q5 = wrist_roll

    Returns:
        4×4 homogeneous transform: wrist frame in robot base frame.
        Position xyz = T[:3, 3] in meters.
    """
    q1, q2, q3, q4, q5 = q

    # Base → shoulder_pan
    T = _trans(0, 0, _L_BASE_Z) @ _rot_z(q1)
    # shoulder_pan → shoulder_lift (90° fixed rotation to align y-axis)
    T = T @ _rot_y(q2)
    # shoulder_lift → elbow
    T = T @ _trans(_L_UPPER, 0, 0) @ _rot_y(q3)
    # elbow → wrist_flex
    T = T @ _trans(_L_FOREARM, 0, 0) @ _rot_y(q4)
    # wrist_flex → wrist_roll
    T = T @ _trans(_L_WRIST, 0, 0) @ _rot_z(q5)

    return T


# ---------------------------------------------------------------------------
# Robot connection — read joint positions via lerobot
# ---------------------------------------------------------------------------

def read_joint_angles(port: str) -> list[float]:
    """Connect to SO-101 follower, read joint positions, disconnect.

    Returns 5 joint angles [q1..q5] in radians (gripper excluded).
    lerobot observation.state is [shoulder_pan, shoulder_lift, elbow_flex,
                                   wrist_flex, wrist_roll, gripper] in degrees.
    """
    try:
        from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
    except ImportError:
        print("[ERROR] lerobot not found. Install it first (pip install lerobot).")
        sys.exit(1)

    MOTOR_NAMES = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]
    MOTOR_MODELS = ["sts3215"] * 6
    MOTOR_IDS    = [1, 2, 3, 4, 5, 6]

    print(f"[probe] Connecting to SO-101 on {port} ...")
    bus = FeetechMotorsBus(
        port=port,
        motors={name: (mid, model)
                for name, mid, model in zip(MOTOR_NAMES, MOTOR_IDS, MOTOR_MODELS)},
    )
    bus.connect()

    # Read present positions (raw motor steps)
    positions = bus.read("Present_Position")  # dict name→raw_steps

    bus.disconnect()
    print("[probe] Disconnected.")

    # Convert raw steps → degrees via lerobot calibration if available,
    # otherwise use the standard Feetech 4096 steps/rev mapping.
    # lerobot stores calibration in ~/.cache/huggingface/lerobot/calibration/
    calib = _load_calibration(port)

    angles_rad = []
    for name in MOTOR_NAMES[:5]:  # skip gripper
        raw = positions[name]
        if calib and name in calib:
            deg = _apply_calibration(raw, calib[name])
        else:
            # Fallback: 4096 steps = 360°, zero at step 2048
            deg = (raw - 2048) * 360.0 / 4096.0
        angles_rad.append(math.radians(deg))

    return angles_rad


def _load_calibration(port: str) -> dict | None:
    """Try to load lerobot calibration file for the follower."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"
    # lerobot saves calibration as so101_follower.json
    for candidate in ["so101_follower.json", "so101_follower_calibration.json"]:
        f = cache_dir / candidate
        if f.exists():
            import json
            with open(f) as fp:
                data = json.load(fp)
            print(f"[probe] Loaded calibration from {f}")
            return data
    print("[probe] No calibration file found — using raw step→degree fallback.")
    return None


def _apply_calibration(raw: int, cal: dict) -> float:
    """Convert raw motor steps to degrees using lerobot calibration data."""
    # lerobot calibration format: {homing_offset, drive_mode, ...}
    offset = cal.get("homing_offset", 0)
    drive_mode = cal.get("drive_mode", 0)
    deg = (raw - 2048 - offset) * 360.0 / 4096.0
    if drive_mode:
        deg = -deg
    return deg


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "bowl_positions.yaml"


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


def save_config(cfg: dict):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"[probe] Saved → {CONFIG_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port",  default="COM3",
                        help="Serial port of the SO-101 follower (default: COM3)")
    parser.add_argument("--label", default="bowl",
                        help="Name for this bowl position, e.g. bowl_eval1")
    parser.add_argument("--list",  action="store_true",
                        help="Print saved bowl positions and exit")
    args = parser.parse_args()

    if args.list:
        cfg = load_config()
        if not cfg:
            print("[probe] No bowl positions saved yet.")
        else:
            print(f"\nSaved positions ({CONFIG_PATH}):")
            for label, pos in cfg.items():
                print(f"  {label}: x={pos['x']:.4f}  y={pos['y']:.4f}  z={pos['z']:.4f}")
        return

    # 1. Read joint angles
    q = read_joint_angles(args.port)
    print(f"\n[probe] Joint angles (rad): {[f'{a:.4f}' for a in q]}")
    print(f"[probe] Joint angles (deg): {[f'{math.degrees(a):.1f}' for a in q]}")

    # 2. FK → wrist position in base frame
    T = fk_wrist(q)
    x, y, z = T[:3, 3]

    print(f"\n[probe] Wrist position in robot base frame:")
    print(f"         x = {x:.4f} m")
    print(f"         y = {y:.4f} m")
    print(f"         z = {z:.4f} m")
    print(f"\n  → bowl xy ≈ ({x:.4f}, {y:.4f})  [z from table height = {z:.4f}]")

    # 3. Save
    cfg = load_config()
    cfg[args.label] = {"x": round(float(x), 4),
                       "y": round(float(y), 4),
                       "z": round(float(z), 4)}
    save_config(cfg)
    print(f"\n[probe] Saved as '{args.label}'.")
    print(f"        Use --list to see all saved positions.")


if __name__ == "__main__":
    main()
