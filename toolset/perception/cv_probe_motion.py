"""
Move gripper using the same IK conventions as ik_relative.py:

  - Warm-start q_init = motor arm angles in rad (not motor_to_urdf).
  - Target = start_tip_urdf + user_offset_to_urdf(delta_user).
  - Execute delta_urdf added to current motor-rad arm joints.
  - Optional XY-then-Z staging (like lift_place) when one-shot IK fails.
"""
from __future__ import annotations

import numpy as np

from toolset.kinematics.config import KinematicsConfig
from toolset.kinematics.ik import IKSolver
from toolset.kinematics.ik_relative import (
    ramp_joints,
    urdf_xyz_to_user,
    user_offset_to_urdf,
)
from toolset.kinematics.motor_to_urdf import MotorToUrdfConfig
from toolset.kinematics.urdf_fk import CHAIN_JOINTS, SO101FK

MOTOR_ARM = list(CHAIN_JOINTS)


def read_motor_state(robot) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns (q_arm_motor_rad 5, q_arm_pseudo_urdf 5, gripper_deg)."""
    obs = robot.get_observation()
    motor_deg = np.array([float(obs[f"{n}.pos"]) for n in MOTOR_ARM], dtype=float)
    gripper_deg = float(obs["gripper.pos"])
    q_arm_motor_rad = np.deg2rad(motor_deg)
    # ik_relative uses motor->rad as IK/FK joint vector (offset cancels in delta).
    q_pseudo = q_arm_motor_rad.copy()
    return q_arm_motor_rad, q_pseudo, gripper_deg


def _solve_relative_waypoint(
    fk: SO101FK,
    ik: IKSolver,
    q_pseudo: np.ndarray,
    xyz_start_urdf: np.ndarray,
    delta_user: np.ndarray,
    *,
    target: str = "gripper_tip",
) -> tuple[bool, np.ndarray, str]:
    """IK for start_urdf + user_delta; returns (ok, q_target_pseudo, msg)."""
    off_urdf = user_offset_to_urdf(np.asarray(delta_user, dtype=float).reshape(3))
    tgt_urdf = xyz_start_urdf + off_urdf
    res = ik.solve(
        tgt_urdf,
        q_init=q_pseudo.copy(),
        target=target,
        mode="position",
        retry_from_home=True,
    )
    if not res.converged:
        return False, q_pseudo, (
            f"IK failed ({res.reason}): pos_err={res.pos_err_m * 1000:.1f} mm "
            f"delta_user={np.asarray(delta_user).round(3).tolist()} "
            f"tgt_urdf={tgt_urdf.round(3).tolist()}"
        )
    return True, res.joints_rad.copy(), "ok"


def _execute_pseudo_delta(
    robot,
    q_arm_motor_rad: np.ndarray,
    q_pseudo: np.ndarray,
    q_target_pseudo: np.ndarray,
    *,
    gripper_deg: float,
    max_step_rad: float,
    period: float,
    label: str,
) -> None:
    delta = q_target_pseudo - q_pseudo
    q_motor_target = q_arm_motor_rad + delta
    ramp_joints(
        robot,
        q_arm_motor_rad,
        q_motor_target,
        gripper_deg=gripper_deg,
        max_step_rad=max_step_rad,
        period=period,
        ease="smooth",
        label=label,
    )


def move_hover_above_cube(
    robot,
    fk: SO101FK,
    ik: IKSolver,
    cube_xyz_user: np.ndarray,
    *,
    hover_m: float = 0.01,
    gripper_deg: float | None = None,
    max_step_rad: float = np.deg2rad(2.0),
    period: float = 1.0 / 30.0,
    target: str = "gripper_tip",
    staged_xy_z: bool = True,
) -> tuple[bool, str]:
    """
    Move tip to cube_xyz_user + hover_m on +Z (user frame), ik_relative style.

    cube_xyz_user: CV estimate (right+, forward+, up+).
    """
    cube_user = np.asarray(cube_xyz_user, dtype=float).reshape(3)
    hover_user = cube_user + np.array([0.0, 0.0, float(hover_m)], dtype=float)

    q_arm, q_pseudo, g_now = read_motor_state(robot)
    if gripper_deg is None:
        gripper_deg = g_now

    xyz_start = fk.fk(q_pseudo, target=target)["position"]
    tip_user = urdf_xyz_to_user(xyz_start)
    delta_user = hover_user - tip_user

    print(
        f"  [hover-ik] tip_user={tip_user.round(3).tolist()}  "
        f"cube_user={cube_user.round(3).tolist()}  "
        f"hover_user={hover_user.round(3).tolist()}  "
        f"delta_user={delta_user.round(3).tolist()}",
        flush=True,
    )

    if float(np.linalg.norm(delta_user[:2])) < 1e-4 and abs(delta_user[2]) < 1e-4:
        return True, "already at hover (delta ~ 0)"

    # One-shot (same as a single ik_relative --offsets waypoint)
    ok, q_tgt, msg = _solve_relative_waypoint(
        fk, ik, q_pseudo, xyz_start, delta_user, target=target,
    )
    if ok:
        _execute_pseudo_delta(
            robot, q_arm, q_pseudo, q_tgt,
            gripper_deg=gripper_deg, max_step_rad=max_step_rad, period=period,
            label="hover",
        )
        return True, f"hover OK (1-step) delta_mm={np.linalg.norm(delta_user)*1000:.0f}"

    if not staged_xy_z:
        return False, msg

    # Staged: horizontal then Z (lift_place pattern)
    print("  [hover-ik] one-shot failed; trying XY then Z ...", flush=True)
    q_arm, q_pseudo, _ = read_motor_state(robot)
    xyz_start = fk.fk(q_pseudo, target=target)["position"]

    delta_xy = np.array([delta_user[0], delta_user[1], 0.0], dtype=float)
    if np.linalg.norm(delta_xy[:2]) > 1e-4:
        ok1, q1, msg1 = _solve_relative_waypoint(
            fk, ik, q_pseudo, xyz_start, delta_xy, target=target,
        )
        if not ok1:
            return False, f"hover XY failed: {msg1}"
        _execute_pseudo_delta(
            robot, q_arm, q_pseudo, q1,
            gripper_deg=gripper_deg, max_step_rad=max_step_rad, period=period,
            label="hover-xy",
        )
        q_arm, q_pseudo, _ = read_motor_state(robot)
        xyz_start = fk.fk(q_pseudo, target=target)["position"]

    delta_z = np.array([0.0, 0.0, float(delta_user[2])], dtype=float)
    if abs(delta_z[2]) > 1e-4:
        ok2, q2, msg2 = _solve_relative_waypoint(
            fk, ik, q_pseudo, xyz_start, delta_z, target=target,
        )
        if not ok2:
            return False, f"hover Z failed: {msg2}"
        _execute_pseudo_delta(
            robot, q_arm, q_pseudo, q2,
            gripper_deg=gripper_deg, max_step_rad=max_step_rad, period=period,
            label="hover-z",
        )

    return True, "hover OK (staged XY then Z)"


def make_ik_stack(kcfg: KinematicsConfig | None = None) -> tuple[KinematicsConfig, IKSolver, SO101FK, MotorToUrdfConfig]:
    kcfg = kcfg or KinematicsConfig.load()
    fk = SO101FK(kcfg.urdf_path, gripper_tip_offset=kcfg.gripper_tip_offset_m)
    ik = IKSolver(kcfg, fk)
    mcfg = MotorToUrdfConfig.load()
    return kcfg, ik, fk, mcfg
