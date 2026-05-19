"""
ik.py - inverse kinematics for the SO-101, position-only with optional
gripper-axis vertical constraint.

We have 5 DoF in the manipulator. With 5 joints we can't independently
control all 6 DoF of a pose; the practical compromise (matches box2ai
and most SO-101 pick-and-place code) is:

    "position"          : minimize ||tip - target||, orientation is free
    "position_vertical" : minimize ||tip - target|| AND constrain the
                          gripper z-axis to point "down" along base -z

We use damped least-squares with a numerical Jacobian computed from the
URDF FK. Joint limits are enforced by clamping after each update; if the
solver wants to push past a limit, the corresponding dimension goes to
zero in the Jacobian (poor man's saturation) so it stops trying.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import KinematicsConfig
from .urdf_fk import CHAIN_JOINTS, SO101FK


@dataclass
class IKResult:
    joints_rad: np.ndarray              # 5-vector URDF rad
    converged: bool
    iters: int
    pos_err_m: float
    rot_err_rad: float
    reason: str                         # "converged", "max_iters", "out_of_workspace", ...


def _numerical_jacobian(
    fk: SO101FK, q: np.ndarray, target: str, eps: float = 1e-4
) -> np.ndarray:
    """6 x 5 numerical Jacobian of (pos, omega) wrt joint angles. omega is
    the small-rotation vector that approximates dR via the matrix log.
    """
    out = fk.fk(q, target=target)
    pos0 = out["position"]
    R0 = out["rotation_matrix"]
    J = np.zeros((6, 5), dtype=float)
    for i in range(5):
        dq = np.zeros(5)
        dq[i] = eps
        out_p = fk.fk(q + dq, target=target)
        out_m = fk.fk(q - dq, target=target)
        # Position derivative (central difference)
        J[:3, i] = (out_p["position"] - out_m["position"]) / (2 * eps)
        # Rotation derivative -> angular velocity at q (use omega from R0^T @ dR)
        dR = (out_p["rotation_matrix"] - out_m["rotation_matrix"]) / (2 * eps)
        skew = R0.T @ dR
        J[3, i] = (skew[2, 1] - skew[1, 2]) / 2
        J[4, i] = (skew[0, 2] - skew[2, 0]) / 2
        J[5, i] = (skew[1, 0] - skew[0, 1]) / 2
    _ = pos0  # silence unused; pos0 is implicit in central-diff usage
    return J


def _orientation_error_vertical(R: np.ndarray) -> np.ndarray:
    """Rotation vector to align the tool +x (pointing direction) with world -z.

    For the SO-101 gripper_tip/gripper_frame, the URDF gripper_frame_joint
    applies rpy="0 pi 0" which puts the pointing direction along local +x
    (the closed jaws extend along this axis). To make the arm "point down"
    onto the table, we want tool +x to align with base -z.

    Returns a 3-vector omega such that applying it as a small rotation to
    R brings R[:,0] toward (0,0,-1). Magnitude = sin(angle), so saturates
    for very-misaligned starts but is fine within ~30 deg.
    """
    point_axis = R[:, 0]
    goal = np.array([0.0, 0.0, -1.0])
    return np.cross(point_axis, goal)


class IKSolver:
    def __init__(self, cfg: KinematicsConfig, fk: SO101FK | None = None):
        self.cfg = cfg
        self.fk = fk or SO101FK(cfg.urdf_path, gripper_tip_offset=cfg.gripper_tip_offset_m)
        self.damping = float(cfg.ik["damping"])
        self.max_iters = int(cfg.ik["max_iters"])
        self.pos_tol = float(cfg.ik["pos_tol_m"])
        self.rot_tol = float(cfg.ik["rot_tol_rad"])
        self.step_clip = float(cfg.ik["step_clip_rad"])
        self._limit_lo = np.array(
            [cfg.joint_limits_rad[n][0] for n in CHAIN_JOINTS], dtype=float
        )
        self._limit_hi = np.array(
            [cfg.joint_limits_rad[n][1] for n in CHAIN_JOINTS], dtype=float
        )

    def solve(
        self,
        target_position: np.ndarray | list[float],
        q_init: np.ndarray | list[float],
        target: str = "gripper_tip",
        mode: str = "position_vertical",
        retry_from_home: bool = True,
    ) -> IKResult:
        """When the warm start fails on joint-limit stall, retry from a
        neutral "home" config (arm vertical, forearm forward). Set
        retry_from_home=False to disable.
        """
        target_xyz = np.asarray(target_position, dtype=float).reshape(3)
        q = np.asarray(q_init, dtype=float).reshape(-1).copy()
        if q.shape[0] != 5:
            raise ValueError(f"q_init must have 5 elements, got {q.shape[0]}")
        q = np.clip(q, self._limit_lo, self._limit_hi)

        if not self.cfg.in_workspace(target_xyz):
            return IKResult(
                joints_rad=q, converged=False, iters=0,
                pos_err_m=float("nan"), rot_err_rad=float("nan"),
                reason="out_of_workspace",
            )

        if mode == "position":
            n_rows = 3
        elif mode == "position_vertical":
            n_rows = 6
        else:
            raise ValueError(f"mode must be 'position' or 'position_vertical', got {mode!r}")

        for it in range(self.max_iters):
            out = self.fk.fk(q, target=target)
            pos = out["position"]
            R = out["rotation_matrix"]
            err_pos = target_xyz - pos
            pos_err = float(np.linalg.norm(err_pos))
            if mode == "position_vertical":
                err_rot = _orientation_error_vertical(R)
                rot_err = float(np.linalg.norm(err_rot))
            else:
                err_rot = np.zeros(3)
                rot_err = 0.0

            if pos_err < self.pos_tol and rot_err < self.rot_tol:
                return IKResult(q.copy(), True, it, pos_err, rot_err, "converged")

            J_full = _numerical_jacobian(self.fk, q, target)
            J = J_full[:n_rows, :]
            err = np.concatenate([err_pos, err_rot])[:n_rows]

            # Damped pseudo-inverse: dq = J^T (J J^T + lambda^2 I)^-1 err
            JJT = J @ J.T
            damp = (self.damping ** 2) * np.eye(JJT.shape[0])
            try:
                dq = J.T @ np.linalg.solve(JJT + damp, err)
            except np.linalg.LinAlgError:
                return IKResult(q.copy(), False, it, pos_err, rot_err, "singular")

            # Clip step magnitude per-joint
            dq = np.clip(dq, -self.step_clip, self.step_clip)
            q = np.clip(q + dq, self._limit_lo, self._limit_hi)

        # Max iters
        out = self.fk.fk(q, target=target)
        pos_err = float(np.linalg.norm(target_xyz - out["position"]))
        rot_err = (
            float(np.linalg.norm(_orientation_error_vertical(out["rotation_matrix"])))
            if mode == "position_vertical" else 0.0
        )
        converged = pos_err < self.pos_tol and rot_err < self.rot_tol
        if converged or not retry_from_home:
            return IKResult(
                q.copy(), converged, self.max_iters,
                pos_err, rot_err,
                "converged" if converged else "max_iters",
            )
        # Retry from a neutral home config (arm vertical, forearm forward,
        # wrist neutral). Helps when the warm start hit joint limits.
        home = np.array([0.0, -np.pi / 2, np.pi / 2, 0.0, 0.0], dtype=float)
        home = np.clip(home, self._limit_lo, self._limit_hi)
        retry = self.solve(target_xyz, q_init=home, target=target, mode=mode, retry_from_home=False)
        if retry.converged or retry.pos_err_m < pos_err:
            return IKResult(
                retry.joints_rad, retry.converged, self.max_iters + retry.iters,
                retry.pos_err_m, retry.rot_err_rad,
                retry.reason + "_from_home" if retry.converged else "max_iters_retry",
            )
        return IKResult(
            q.copy(), converged, self.max_iters,
            pos_err, rot_err,
            "max_iters",
        )
