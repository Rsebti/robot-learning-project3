"""
probe_position.py - read current SO-101 joint angles, compute the gripper
frame pose with URDF-based forward kinematics, print (x, y, z) both in the
URDF base frame and in user-facing (right+, forward+) coordinates.

The URDF parsed here is `so101_new_calib.urdf` from TheRobotStudio's SO-ARM100
repository — the same file lerobot's `RobotKinematics` uses internally. This
replaces the hand-tuned DH approximation in `toolset/probe_bowl_position.py`
which was off by ~12 cm in the forward direction.

Usage:
    conda activate trim
    python deploy/probe_position.py --port COM3
"""
from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

URDF_PATH = Path(__file__).parent / "so101_new_calib.urdf"

# Chain from base_link to gripper_frame_link. All SO-101 revolute joints
# rotate about local z. Order matches lerobot observation.state order.
CHAIN_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper_frame_joint",  # fixed
]
MOTOR_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF rpy convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _origin_transform(xyz: tuple[float, float, float],
                      rpy: tuple[float, float, float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rpy_to_matrix(*rpy)
    T[:3, 3] = xyz
    return T


def _rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    T = np.eye(4)
    T[0, 0] = c;  T[0, 1] = -s
    T[1, 0] = s;  T[1, 1] = c
    return T


def _parse_urdf_chain(urdf_path: Path) -> list[dict]:
    """Extract (xyz, rpy, type, axis) for each joint in CHAIN_JOINTS order."""
    root = ET.parse(urdf_path).getroot()
    by_name = {j.get("name"): j for j in root.findall("joint")}
    out = []
    for jname in CHAIN_JOINTS:
        j = by_name[jname]
        origin = j.find("origin")
        xyz = tuple(float(v) for v in origin.get("xyz").split())
        rpy = tuple(float(v) for v in origin.get("rpy").split())
        jtype = j.get("type")
        axis_el = j.find("axis")
        axis = tuple(float(v) for v in axis_el.get("xyz").split()) if axis_el is not None else (0, 0, 1)
        out.append({"name": jname, "xyz": xyz, "rpy": rpy, "type": jtype, "axis": axis})
    return out


def fk_gripper_frame(q_rad: list[float], chain: list[dict]) -> np.ndarray:
    """FK from base_link to gripper_frame_link.

    Args:
        q_rad: 5 joint angles in radians, order
               [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll].
        chain: parsed joint chain from `_parse_urdf_chain`.

    Returns:
        4x4 homogeneous transform of gripper_frame_link in base_link frame.
    """
    T = np.eye(4)
    rev_idx = 0
    for joint in chain:
        T_origin = _origin_transform(joint["xyz"], joint["rpy"])
        if joint["type"] == "revolute":
            axis = joint["axis"]
            if axis != (0, 0, 1):
                raise NotImplementedError(f"Non-z axis on {joint['name']}: {axis}")
            T_origin = T_origin @ _rot_z(q_rad[rev_idx])
            rev_idx += 1
        T = T @ T_origin
    return T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--camera_index", type=int, default=1,
                        help="Camera index required by the SO101 config; ignored otherwise.")
    parser.add_argument("--urdf", default=str(URDF_PATH))
    args = parser.parse_args()

    urdf_path = Path(args.urdf)
    if not urdf_path.exists():
        print(f"[probe] URDF not found at {urdf_path}", file=sys.stderr)
        sys.exit(2)
    chain = _parse_urdf_chain(urdf_path)

    deploy_dir = Path(__file__).resolve().parent
    if str(deploy_dir) not in sys.path:
        sys.path.insert(0, str(deploy_dir))
    from robot_calibration import make_so101_follower_config
    from lerobot.robots.utils import make_robot_from_config
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    cfg = make_so101_follower_config(
        args.port,
        cameras={"wrist": OpenCVCameraConfig(
            index_or_path=args.camera_index, fps=30, width=640, height=480,
        )},
        use_degrees=True,
    )
    robot = make_robot_from_config(cfg)
    print(f"[probe] Connecting to {args.port} ...")
    robot.connect()

    try:
        obs = robot.get_observation()
        deg = np.array([obs[f"{n}.pos"] for n in MOTOR_NAMES], dtype=float)
        q_rad = np.deg2rad(deg[:5]).tolist()

        print(f"\n[probe] joint angles (deg): {deg.round(1).tolist()}")
        T = fk_gripper_frame(q_rad, chain)
        xu, yu, zu = T[:3, 3]

        # URDF base frame: +x forward, +y left, +z up (ROS).
        # User frame:      +x right,   +y forward, +z up.
        x_user = -yu
        y_user = xu
        z_user = zu

        print(f"\n[probe] gripper_frame_link in URDF base frame:")
        print(f"        x_urdf = {xu:+.4f} m  (forward +)")
        print(f"        y_urdf = {yu:+.4f} m  (left +)")
        print(f"        z_urdf = {zu:+.4f} m")
        print(f"\n[probe] in user frame (right+, forward+):")
        print(f"        x = {x_user:+.4f} m  (right + / left -)")
        print(f"        y = {y_user:+.4f} m  (forward + / back -)")
        print(f"        z = {z_user:+.4f} m")
        print(f"\n  --cube_x {x_user:.3f} --cube_y {y_user:.3f}    # paste into deploy command")
    finally:
        robot.disconnect()
        print("[probe] Disconnected.")


if __name__ == "__main__":
    main()
