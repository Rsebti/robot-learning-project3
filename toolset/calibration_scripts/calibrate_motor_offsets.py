"""
calibrate_motor_offsets.py - figure out the per-joint offset between
lerobot motor degrees and URDF joint radians.

WHY: lerobot's `get_observation()` returns angles where 0 deg == midpoint
of each motor's calibrated range, which is generally NOT the URDF zero
pose. To run URDF-native FK we need:

    urdf_rad = sign * (motor_deg - offset_deg) * pi/180

This script offers TWO calibration modes:

 1. "anchor" mode: you put the robot into one or more URDF-known poses
    (e.g. the home pose `[0, -pi/2, pi/2, 0, 0]`) and we record motor
    angles. With enough anchor poses, we recover the offset per joint
    (assuming sign=+1, which is the SO-101 convention from the URDF).

 2. "ruler" mode: you put the gripper tip on a measured world point
    (e.g. "12 cm right of base center, 25 cm forward, on the table"),
    and we solve for the offsets that make FK match. Uses scipy
    least_squares on multiple measurements.

Anchor mode is faster and more reliable; use that. Ruler mode is the
fallback when you can't reach the URDF-zero or home pose safely.

Output: `toolset/configs/kinematics/motor_offsets.yaml`.

Usage:
    # Anchor mode (recommended, simplest)
    python -m toolset.calibration_scripts.calibrate_motor_offsets \\
        --port COM3 --mode anchor

    # Ruler mode (multiple known-xyz touches)
    python -m toolset.calibration_scripts.calibrate_motor_offsets \\
        --port COM3 --mode ruler
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from toolset.kinematics.config import KinematicsConfig
from toolset.kinematics.motor_to_urdf import MotorToUrdfConfig, read_motor_deg_from_obs
from toolset.kinematics.urdf_fk import CHAIN_JOINTS, SO101FK


# Named URDF anchor poses (radians). These are the poses we can ASK the
# operator to put the arm in. They were chosen for ease of physical
# verification with a flat reference.
ANCHOR_POSES = {
    "home_pointing_down": {
        # Arm folded with gripper pointing straight down at the base center.
        # shoulder_lift = -90 deg (arm up), elbow_flex = +90 deg (forearm
        # forward), wrist_flex = 0, wrist_roll = 0. This is the URDF-frame
        # "home" we use as REST in inference scripts.
        "urdf_rad": [0.0, -1.5707963, 1.5707963, 0.0, 0.0],
        "instructions": (
            "Move the arm to the standard rest pose:\n"
            "  - shoulder_pan: gripper aligned with the forward direction\n"
            "  - shoulder_lift: upper arm vertical (pointing up)\n"
            "  - elbow_flex:    forearm horizontal, pointing forward\n"
            "  - wrist_flex:    gripper roughly horizontal in line with forearm\n"
            "  - wrist_roll:    jaw axis horizontal\n"
            "Then press Enter."
        ),
    },
    "arm_extended_forward": {
        # All joints at 0 in URDF -> arm horizontal forward, gripper points
        # +x. NOT a stable hold pose without support; skip unless safe.
        "urdf_rad": [0.0, 0.0, 0.0, 0.0, 0.0],
        "instructions": (
            "Move arm fully horizontal forward, gripper pointing along base +x.\n"
            "Hold the wrist if needed - this is not a stable pose for most arms.\n"
            "Press Enter when steady."
        ),
    },
}


def calibrate_anchor(robot, anchors: list[str]) -> MotorToUrdfConfig:
    """For each anchor pose, ask the operator to move there and read motors.

    Recovers per-joint offset assuming sign=+1 (URDF axes match lerobot
    physical directions). If a joint's offset is clearly wrong, try sign=-1
    interactively.
    """
    samples: list[tuple[np.ndarray, np.ndarray]] = []  # (motor_deg5, urdf_rad5)
    for name in anchors:
        spec = ANCHOR_POSES[name]
        print(f"\n=== Anchor: {name} ===")
        print(spec["instructions"])
        input(">>> Press Enter when the arm is in position. ")
        obs = robot.get_observation()
        m_deg = read_motor_deg_from_obs(obs)[:5]
        urdf = np.asarray(spec["urdf_rad"], dtype=float)
        samples.append((m_deg, urdf))
        print(f"  recorded motor (deg): {m_deg.round(2).tolist()}")
        print(f"  target  URDF  (rad):  {urdf.round(3).tolist()}")

    # offset_deg[i] = motor_deg[i] - urdf_deg[i] (averaged over anchors)
    offsets_deg = {}
    for i, name in enumerate(CHAIN_JOINTS):
        diffs = [s[0][i] - np.rad2deg(s[1][i]) for s in samples]
        offsets_deg[name] = float(np.mean(diffs))

    cfg = MotorToUrdfConfig()
    cfg.offsets_deg = offsets_deg
    cfg.signs = {n: +1 for n in CHAIN_JOINTS}

    # Sign-flip detector: if at multiple anchors the residual error for a
    # joint flips sign symmetrically, the sign is probably wrong.
    print("\n[calib] per-joint offset (deg) recovered:")
    for n in CHAIN_JOINTS:
        print(f"  {n:14s}  offset = {offsets_deg[n]:+8.2f} deg")
    return cfg


def calibrate_ruler(robot, fk: SO101FK, measurements: list[dict]) -> MotorToUrdfConfig:
    """Least-squares fit of per-joint offsets such that FK at the corrected
    angles matches the measured xyz at each touch.

    measurements: list of {"motor_deg": [..5..], "tip_xyz_m": [x, y, z]}
    """
    try:
        from scipy.optimize import least_squares
    except ImportError:
        print("[calib] scipy not available; install scipy to use ruler mode.")
        sys.exit(2)

    def residuals(offsets_deg5: np.ndarray) -> np.ndarray:
        cfg = MotorToUrdfConfig()
        cfg.offsets_deg = {n: float(offsets_deg5[i]) for i, n in enumerate(CHAIN_JOINTS)}
        cfg.signs = {n: +1 for n in CHAIN_JOINTS}
        errs = []
        for m in measurements:
            urdf = cfg.motor_to_urdf_rad(np.asarray(m["motor_deg"]))
            pred = fk.fk(urdf, target="gripper_tip")["position"]
            errs.extend((pred - np.asarray(m["tip_xyz_m"], dtype=float)).tolist())
        return np.array(errs, dtype=float)

    x0 = np.zeros(5)
    sol = least_squares(residuals, x0, method="lm", max_nfev=200)
    offsets_deg = {n: float(sol.x[i]) for i, n in enumerate(CHAIN_JOINTS)}

    cfg = MotorToUrdfConfig()
    cfg.offsets_deg = offsets_deg
    cfg.signs = {n: +1 for n in CHAIN_JOINTS}
    rms = float(np.sqrt(np.mean(sol.fun ** 2)) * 1000)
    print(f"\n[calib] ruler-mode RMS error: {rms:.2f} mm")
    print("[calib] per-joint offset (deg) recovered:")
    for n in CHAIN_JOINTS:
        print(f"  {n:14s}  offset = {offsets_deg[n]:+8.2f} deg")
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--camera_index", type=int, default=1)
    parser.add_argument("--mode", default="anchor", choices=["anchor", "ruler"])
    parser.add_argument("--anchors", nargs="+",
                        default=["home_pointing_down"],
                        help="Anchor pose names (see ANCHOR_POSES dict).")
    parser.add_argument("--measurements_yaml",
                        help="(ruler mode) YAML with list of "
                             "{motor_deg: [...], tip_xyz_m: [...]} entries.")
    parser.add_argument("--output",
                        default=None,
                        help="Where to write motor_offsets.yaml (default: project config).")
    args = parser.parse_args()

    kcfg = KinematicsConfig.load()
    fk = SO101FK(kcfg.urdf_path, gripper_tip_offset=kcfg.gripper_tip_offset_m)

    repo = Path(__file__).resolve().parents[2]
    if str(repo / "deploy") not in sys.path:
        sys.path.insert(0, str(repo / "deploy"))
    from robot_calibration import make_so101_follower_config
    from lerobot.robots.utils import make_robot_from_config
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    rconf = make_so101_follower_config(
        args.port,
        cameras={"wrist": OpenCVCameraConfig(
            index_or_path=args.camera_index, fps=30, width=640, height=480,
        )},
        use_degrees=True,
    )
    robot = make_robot_from_config(rconf)
    print(f"[calib] Connecting to {args.port} ...")
    robot.connect()

    try:
        if args.mode == "anchor":
            mcfg = calibrate_anchor(robot, args.anchors)
        else:
            if not args.measurements_yaml:
                print("[calib] --measurements_yaml required for ruler mode.")
                sys.exit(2)
            with open(args.measurements_yaml) as f:
                meas = yaml.safe_load(f)
            mcfg = calibrate_ruler(robot, fk, meas)
    finally:
        robot.disconnect()
        print("[calib] Disconnected.")

    # Quick sanity round-trip: FK at recovered offsets at *each* anchor, print error
    if args.mode == "anchor":
        print("\n[calib] verification - FK position at each anchor:")
        for name in args.anchors:
            urdf = np.asarray(ANCHOR_POSES[name]["urdf_rad"])
            pos = fk.fk(urdf, target="gripper_tip")["position"]
            print(f"  {name}: gripper_tip xyz = {pos.round(4).tolist()} m")

    out = Path(args.output) if args.output else None
    mcfg.save(out) if out else mcfg.save()
    print(f"\n[calib] wrote offsets to {out or '(default path)'}.")


if __name__ == "__main__":
    main()
