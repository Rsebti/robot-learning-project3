"""
urdf_fk.py - URDF-native forward kinematics for the SO-101.

The kinematic chain is parsed directly from `so101_new_calib.urdf` (the
TheRobotStudio SO-ARM100 URDF; same file lerobot's RobotKinematics uses
internally). No hand-derived DH parameters - the URDF is treated as ground
truth for link geometry.

Target frames:
    "wrist"       - origin of `wrist_link` (end of joint 5, before the
                    gripper jaws). Useful for "where is my hand".
    "gripper_tip" - centerpoint between the closed gripper jaws. Offset
                    from `gripper_link` along the gripper's local axes by
                    `gripper_tip_offset` (must be MEASURED on the physical
                    robot, see config.yaml).

Joint angle convention (all 5 revolute joints rotate about local +Z):
    [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]
The angles are URDF-frame radians, NOT lerobot motor degrees. Use
`motor_to_urdf.py` to convert.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_URDF_PATH = Path(__file__).parent.parent.parent / "deploy" / "so101_new_calib.urdf"

CHAIN_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
JOINT_NAMES_FULL = CHAIN_JOINTS + ["gripper"]


@dataclass(frozen=True)
class JointSpec:
    name: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    jtype: str  # "revolute" | "fixed"
    axis: tuple[float, float, float]


def _rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF rpy convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _T(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rpy_to_R(*rpy)
    T[:3, 3] = xyz
    return T


def _Rz(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    T = np.eye(4)
    T[0, 0] = c;  T[0, 1] = -s
    T[1, 0] = s;  T[1, 1] = c
    return T


def parse_urdf(urdf_path: Path) -> dict[str, JointSpec]:
    """Parse all joints from the URDF, keyed by joint name."""
    root = ET.parse(urdf_path).getroot()
    out: dict[str, JointSpec] = {}
    for j in root.findall("joint"):
        name = j.get("name")
        origin = j.find("origin")
        xyz = tuple(float(v) for v in origin.get("xyz", "0 0 0").split())
        rpy = tuple(float(v) for v in origin.get("rpy", "0 0 0").split())
        axis_el = j.find("axis")
        axis = (
            tuple(float(v) for v in axis_el.get("xyz").split())
            if axis_el is not None
            else (0.0, 0.0, 1.0)
        )
        out[name] = JointSpec(
            name=name, xyz=xyz, rpy=rpy,
            jtype=j.get("type"), axis=axis,
        )
    return out


class SO101FK:
    """Forward kinematics solver. Construct once, call fk(...) many times.

    The chain encoded here is base_link -> shoulder_link -> upper_arm_link
    -> lower_arm_link -> wrist_link -> gripper_link -> gripper_frame_link.

    Frames available via `target=`:
        "wrist"           : wrist_link origin (end of joint 5 wrist_flex,
                            before wrist_roll)
        "wrist_roll"      : gripper_link origin (output of joint 5 wrist_roll,
                            same as old `fk_wrist` semantics)
        "gripper_frame"   : gripper_frame_link (URDF's canonical EE frame,
                            ~98 mm below gripper_link)
        "gripper_tip"     : centerpoint between closed jaws. Built from
                            gripper_frame + the configured offset.
    """

    def __init__(
        self,
        urdf_path: Path | str = DEFAULT_URDF_PATH,
        gripper_tip_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        """
        Args:
            urdf_path: Path to so101_new_calib.urdf.
            gripper_tip_offset: (x, y, z) in meters, expressed in the
                gripper_frame_link frame. Default (0,0,0) means tip ==
                gripper_frame. The physical measured offset goes in
                config.yaml; pass it in here.
        """
        self.urdf_path = Path(urdf_path)
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")
        self.joints = parse_urdf(self.urdf_path)
        self.gripper_tip_offset = np.asarray(gripper_tip_offset, dtype=float)

        # Chain: base -> shoulder_pan -> shoulder_lift -> elbow_flex
        #        -> wrist_flex -> wrist_roll -> gripper_frame_joint (fixed)
        self._chain_order = CHAIN_JOINTS + ["gripper_frame_joint"]
        for jname in self._chain_order:
            if jname not in self.joints:
                raise KeyError(f"Joint '{jname}' missing from URDF")
            j = self.joints[jname]
            if j.jtype == "revolute" and tuple(j.axis) != (0.0, 0.0, 1.0):
                raise NotImplementedError(
                    f"Joint {jname} has non-Z axis {j.axis}; not supported."
                )

    def fk(
        self,
        q_rad: np.ndarray | list[float],
        target: str = "gripper_tip",
    ) -> dict:
        """Compute FK at joint angles `q_rad` (5 radians, URDF convention).

        Args:
            q_rad: Length-5 array [shoulder_pan, shoulder_lift, elbow_flex,
                   wrist_flex, wrist_roll] in URDF-frame radians.
            target: One of "wrist", "wrist_roll", "gripper_frame",
                    "gripper_tip".

        Returns:
            dict with:
              position        : (3,) ndarray xyz in base_link frame (m)
              rotation_matrix : (3,3) ndarray
              euler_xyz       : (3,) ndarray of intrinsic Euler [rx, ry, rz] (rad)
              T               : (4,4) homogeneous transform
        """
        q = np.asarray(q_rad, dtype=float).reshape(-1)
        if q.shape[0] != 5:
            raise ValueError(f"q_rad must have 5 elements, got {q.shape[0]}")

        # Walk the chain, capturing intermediate frames we may need
        T = np.eye(4)
        frames: dict[str, np.ndarray] = {}
        rev_idx = 0
        # shoulder_pan -> shoulder_link
        T = T @ _T(self.joints["shoulder_pan"].xyz, self.joints["shoulder_pan"].rpy) @ _Rz(q[0])
        # shoulder_lift -> upper_arm_link
        T = T @ _T(self.joints["shoulder_lift"].xyz, self.joints["shoulder_lift"].rpy) @ _Rz(q[1])
        # elbow_flex -> lower_arm_link
        T = T @ _T(self.joints["elbow_flex"].xyz, self.joints["elbow_flex"].rpy) @ _Rz(q[2])
        # wrist_flex -> wrist_link
        T = T @ _T(self.joints["wrist_flex"].xyz, self.joints["wrist_flex"].rpy) @ _Rz(q[3])
        frames["wrist"] = T.copy()
        # wrist_roll -> gripper_link
        T = T @ _T(self.joints["wrist_roll"].xyz, self.joints["wrist_roll"].rpy) @ _Rz(q[4])
        frames["wrist_roll"] = T.copy()
        # gripper_frame_joint (fixed) -> gripper_frame_link
        T = T @ _T(self.joints["gripper_frame_joint"].xyz, self.joints["gripper_frame_joint"].rpy)
        frames["gripper_frame"] = T.copy()
        # gripper_tip = gripper_frame translated by offset (in its own frame)
        T_tip = T.copy()
        T_tip[:3, 3] = T_tip[:3, 3] + T_tip[:3, :3] @ self.gripper_tip_offset
        frames["gripper_tip"] = T_tip

        if target not in frames:
            raise ValueError(f"Unknown target '{target}'. "
                             f"Choose from {list(frames)}.")

        T_out = frames[target]
        R = T_out[:3, :3]
        return {
            "position": T_out[:3, 3].copy(),
            "rotation_matrix": R.copy(),
            "euler_xyz": _R_to_euler_xyz(R),
            "T": T_out.copy(),
        }

    def fk_chain(self, q_rad: np.ndarray | list[float]) -> dict[str, np.ndarray]:
        """Return all named frame transforms (4x4 each) for diagnostic plots."""
        result: dict[str, np.ndarray] = {}
        for tgt in ["wrist", "wrist_roll", "gripper_frame", "gripper_tip"]:
            result[tgt] = self.fk(q_rad, target=tgt)["T"]
        return result


def _R_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    """Intrinsic XYZ Euler angles (Tait-Bryan), in radians.

    Convention: R = Rx(rx) @ Ry(ry) @ Rz(rz). Singular at ry = +/- pi/2.
    """
    sy = -R[2, 0]
    cy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if cy > 1e-6:
        rx = math.atan2(R[2, 1], R[2, 2])
        ry = math.atan2(sy, cy)
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        rx = math.atan2(-R[1, 2], R[1, 1])
        ry = math.atan2(sy, cy)
        rz = 0.0
    return np.array([rx, ry, rz])


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def _smoke_test():
    fk = SO101FK()
    for q5 in [
        [0, 0, 0, 0, 0],
        [0, -math.pi / 2, math.pi / 2, 0, 0],
        [math.pi / 4, -math.pi / 4, math.pi / 4, 0, 0],
    ]:
        out = fk.fk(q5, target="gripper_frame")
        x, y, z = out["position"]
        print(f"q={q5}  ->  gripper_frame xyz=({x:+.4f}, {y:+.4f}, {z:+.4f})  "
              f"euler={out['euler_xyz'].round(3).tolist()}")


if __name__ == "__main__":
    _smoke_test()
