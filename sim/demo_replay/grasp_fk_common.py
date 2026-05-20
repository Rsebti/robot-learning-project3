"""Shared FK + frame helpers for LeRobot demo grasp annotation and sim replay."""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

from toolset.kinematics.config import KinematicsConfig
from toolset.kinematics.ik_relative import urdf_xyz_to_user, user_offset_to_urdf
from toolset.kinematics.motor_to_urdf import MotorToUrdfConfig
from toolset.kinematics.urdf_fk import SO101FK

MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def user_xyz_to_urdf(xyz_user: np.ndarray) -> np.ndarray:
    return user_offset_to_urdf(np.asarray(xyz_user, dtype=float).reshape(3))


def make_fk_stack(kcfg: KinematicsConfig | None = None):
    kcfg = kcfg or KinematicsConfig.load()
    fk = SO101FK(kcfg.urdf_path, gripper_tip_offset=kcfg.gripper_tip_offset_m)
    mcfg = MotorToUrdfConfig.load()
    return kcfg, fk, mcfg


def tip_user_from_motor_deg(
    motor_deg: np.ndarray,
    fk: SO101FK,
    mcfg: MotorToUrdfConfig,
) -> np.ndarray:
    q = mcfg.motor_to_urdf_rad(np.asarray(motor_deg, dtype=float).reshape(-1)[:6])
    tip_urdf = fk.fk(q, target="gripper_tip")["position"]
    return urdf_xyz_to_user(tip_urdf)


def cube_center_user_from_tip(
    tip_user: np.ndarray,
    kcfg: KinematicsConfig,
    *,
    z_mode: str = "tip_minus_half",
) -> np.ndarray:
    """Map grasp tip FK to assumed cube centroid (user frame, m)."""
    tip = np.asarray(tip_user, dtype=float).reshape(3)
    out = tip.copy()
    if z_mode == "tip_minus_half":
        out[2] = tip[2] - kcfg.cube_half_height_m
    elif z_mode == "tip_z":
        pass
    elif z_mode == "table_plus_half":
        out[2] = kcfg.table_z_m + kcfg.cube_half_height_m
    else:
        raise ValueError(f"unknown z_mode {z_mode!r}")
    return out


def cube_pose_urdf_world(cube_xyz_user: np.ndarray) -> list[float]:
    """URDF position + identity quat for Isaac/ManiSkill-style spawn [x,y,z,qw,qx,qy,qz]."""
    p = user_xyz_to_urdf(cube_xyz_user)
    return [float(p[0]), float(p[1]), float(p[2]), 1.0, 0.0, 0.0, 0.0]


def hf_snapshot_dir(repo_id: str) -> Path:
    slug = repo_id.replace("/", "--")
    root = Path.home() / ".cache" / "huggingface" / "hub" / f"datasets--{slug}" / "snapshots"
    snaps = sorted(root.glob("*"), key=lambda p: p.stat().st_mtime)
    if not snaps:
        raise FileNotFoundError(
            f"No local snapshot for {repo_id} under {root}. "
            f"Run: python -m sim.demo_replay.annotate_grasp_fk --download"
        )
    return snaps[-1]


def load_parquet_dataset(snapshot: Path) -> "pd.DataFrame":
    import pandas as pd

    files = sorted(glob.glob(str(snapshot / "data/chunk-*/*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet under {snapshot / 'data'}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)


def stack_state_column(series) -> np.ndarray:
    """Parquet column of length-6 state vectors -> (T, 6) float."""
    return np.stack(series.values, axis=0).astype(np.float64)


def pick_grasp_frame_rollout(
    qpos_deg: np.ndarray,
    *,
    fps: float = 30.0,
    max_arm_vel_deg_s: float = 8.0,
    min_static_frames: int = 8,
    gripper_closed_max_deg: float = 35.0,
) -> tuple[int, dict]:
    """
    Grasp frame for real2sim (full rollout scan).

    1. Find runs where the arm stays quasi-static (``min_static_frames`` at <= vel threshold)
       and gripper is closed enough.
    2. Among those holds, pick the one with the **most closed** gripper (min servo deg).
    3. Within that hold, take the **last** frame (end of the clamp / quasi-static plateau).

    Falls back to the most-closed low-velocity frame in the episode if no hold is found.
    """
    q = np.asarray(qpos_deg, dtype=np.float64)
    t = q.shape[0]
    if t < 2:
        return t - 1, {
            "reason": "too_short",
            "max_arm_vel_deg_s": float("nan"),
            "gripper_deg": float(q[-1, 5]),
            "static_run_len": 0,
        }

    vel = np.zeros(t - 1, dtype=float)
    for i in range(1, t):
        vel[i - 1] = float(np.max(np.abs(q[i, :5] - q[i - 1, :5]))) * fps

    def _arm_vel_at(i: int) -> float:
        return float(vel[i - 1]) if i > 0 else float(vel[0])

    static_ok = np.array(
        [
            _arm_vel_at(i) <= max_arm_vel_deg_s and float(q[i, 5]) <= gripper_closed_max_deg
            for i in range(t)
        ],
        dtype=bool,
    )

    runs: list[tuple[int, int, float]] = []
    i = 0
    while i < t:
        if not static_ok[i]:
            i += 1
            continue
        j = i
        while j < t and static_ok[j]:
            j += 1
        if j - i >= min_static_frames:
            seg_g = q[i:j, 5]
            runs.append((i, j - 1, float(seg_g.min())))
        i = j

    if runs:
        min_grip = min(r[2] for r in runs)
        tied = [r for r in runs if r[2] <= min_grip + 0.25]
        start, end, g_min = max(tied, key=lambda r: r[1])
        return end, {
            "reason": "static_hold_most_closed",
            "max_arm_vel_deg_s": _arm_vel_at(end),
            "gripper_deg": float(q[end, 5]),
            "static_run_len": end - start + 1,
            "static_run_start": start,
            "min_gripper_in_run_deg": g_min,
            "n_static_runs": len(runs),
        }

    scored = []
    for i in range(t):
        if float(q[i, 5]) <= gripper_closed_max_deg:
            scored.append((float(q[i, 5]), _arm_vel_at(i), i))
    if scored:
        _, _, idx = min(scored, key=lambda x: (x[0], x[1], -x[2]))
        return idx, {
            "reason": "fallback_most_closed_low_vel",
            "max_arm_vel_deg_s": _arm_vel_at(idx),
            "gripper_deg": float(q[idx, 5]),
            "static_run_len": 0,
            "n_static_runs": 0,
        }

    idx = int(np.argmin(q[:, 5]))
    return idx, {
        "reason": "fallback_global_min_gripper",
        "max_arm_vel_deg_s": _arm_vel_at(idx),
        "gripper_deg": float(q[idx, 5]),
        "static_run_len": 0,
        "n_static_runs": 0,
    }


def episode_record_for_isaac(row: dict) -> dict:
    """Subset of annotation fields consumed by Isaac replay / verify scripts."""
    return {
        "episode_index": row["episode_index"],
        "n_frames": row["n_frames"],
        "grasp_frame_index": row["grasp_frame_index"],
        "grasp_frame_local": row["grasp_frame_local"],
        "grasp_pick_reason": row["grasp_pick_reason"],
        "grasp_gripper_deg": row["grasp_gripper_deg"],
        "grasp_static_run_len": row.get("grasp_static_run_len", 0),
        "tip_xyz_user_m": row["tip_xyz_user_m"],
        "cube_xyz_user_m": row["cube_xyz_user_m"],
        "cube_pose_urdf_world": row["cube_pose_urdf_world"],
        "grasp_motor_deg": row["grasp_motor_deg"],
        "trajectory_observation_state_deg": row["trajectory_observation_state_deg"],
        "trajectory_action_deg": row["trajectory_action_deg"],
        "trajectory_frame_index": row["trajectory_frame_index"],
    }
