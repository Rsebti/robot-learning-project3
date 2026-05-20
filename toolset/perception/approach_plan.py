"""
approach_plan.py — staged grasp approach from probe map (user frame).

Stages (absolute targets in user m):
  1. hover over cube XY at table + cube_half + hover_above_m
  2. side: +side_offset_m along fixed-jaw approach direction in XY
  3. down: -descend_m in Z
"""
from __future__ import annotations

import numpy as np

from toolset.kinematics.config import KinematicsConfig
from toolset.kinematics.motor_to_urdf import MotorToUrdfConfig
from toolset.kinematics.urdf_fk import SO101FK


def user_xyz_to_urdf(xyz_user: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(xyz_user, dtype=float).reshape(3)
    return np.array([y, -x, z], dtype=float)


def urdf_xyz_to_user(xyz_urdf: np.ndarray) -> np.ndarray:
    return np.array([-xyz_urdf[1], xyz_urdf[0], xyz_urdf[2]], dtype=float)


def approach_dir_user_xy(
    fk: SO101FK,
    mcfg: MotorToUrdfConfig,
    grasp_motor_deg: list[float] | np.ndarray,
) -> np.ndarray:
    """
    Unit vector in user XY for side offset (fixed-jaw approach axis).

    Prefer frame->tip when tip offset is set; else tool +x from gripper_frame
    (jaw forward / into cube) at the probe grasp pose.
    """
    q = mcfg.motor_to_urdf_rad(np.asarray(grasp_motor_deg, dtype=float))
    p_frame = fk.fk(q, target="gripper_frame")["position"]
    p_tip = fk.fk(q, target="gripper_tip")["position"]
    d_user = urdf_xyz_to_user(p_tip - p_frame)
    d_xy = d_user[:2]
    n = float(np.linalg.norm(d_xy))
    if n >= 1e-4:
        return d_xy / n
    # tip offset ~ 0: use tool +x in user XY from grasp orientation
    R = fk.fk(q, target="gripper_frame")["rotation_matrix"]
    tool_x_user = urdf_xyz_to_user(R[:, 0])
    d_xy = tool_x_user[:2]
    n = float(np.linalg.norm(d_xy))
    if n < 1e-6:
        return np.array([0.0, 1.0], dtype=float)
    return d_xy / n


def plan_approach_waypoints(
    kcfg: KinematicsConfig,
    *,
    cube_xy_user: np.ndarray,
    grasp_motor_deg: list[float] | np.ndarray,
    hover_above_m: float = 0.03,
    side_offset_m: float = 0.03,
    descend_m: float = 0.01,
    side_sign: float = -1.0,
) -> dict:
    """
    Absolute user-frame waypoints. cube_xy from map; Z from table + cube half.

    side_sign: -1 = offset opposite jaw forward (toward fixed-jaw side, default).
    """
    mcfg = MotorToUrdfConfig.load()
    fk = SO101FK(kcfg.urdf_path, gripper_tip_offset=kcfg.gripper_tip_offset_m)
    xy = np.asarray(cube_xy_user, dtype=float).reshape(2)
    z_top = kcfg.table_z_m + kcfg.cube_half_height_m
    z_hover = z_top + float(hover_above_m)

    hover = np.array([xy[0], xy[1], z_hover], dtype=float)
    dir_xy = approach_dir_user_xy(fk, mcfg, grasp_motor_deg)
    side_xy = xy + float(side_sign) * float(side_offset_m) * dir_xy
    side = np.array([side_xy[0], side_xy[1], z_hover], dtype=float)
    down = np.array([side_xy[0], side_xy[1], z_hover - float(descend_m)], dtype=float)

    return {
        "cube_xy_user": xy,
        "approach_dir_xy": dir_xy,
        "hover_user": hover,
        "side_user": side,
        "down_user": down,
        "z_top_m": z_top,
        "z_hover_m": z_hover,
    }


def offsets_from_tip_start(
    tip_start_user: np.ndarray,
    plan: dict,
) -> list[list[float]]:
    """ik_relative offsets (from start pose) for hover -> side -> down."""
    tip = np.asarray(tip_start_user, dtype=float).reshape(3)
    out = []
    for key in ("hover_user", "side_user", "down_user"):
        tgt = np.asarray(plan[key], dtype=float)
        off = tgt - tip
        out.append([float(off[0]), float(off[1]), float(off[2])])
    return out
