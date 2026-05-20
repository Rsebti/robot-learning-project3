"""
staged_ik.py — two-step IK when a single solve lands far from the target in FK.

Typical use (approach / ik_relative waypoints):
  1. Solve with desired mode (e.g. position_vertical = gripper down).
  2. If FK position error > retry_mm, solve to a midpoint in Cartesian space.
  3. Re-solve from that joint config to the real target.
"""
from __future__ import annotations

import numpy as np

from .ik import IKResult, IKSolver
from .urdf_fk import SO101FK


def fk_pos_err_mm(
    fk: SO101FK,
    q_urdf: np.ndarray,
    target_urdf: np.ndarray,
    *,
    target_frame: str = "gripper_tip",
) -> float:
    pos = fk.fk(q_urdf, target=target_frame)["position"]
    return float(np.linalg.norm(pos - np.asarray(target_urdf, dtype=float).reshape(3))) * 1000.0


def solve_staged(
    ik: IKSolver,
    fk: SO101FK,
    target_urdf: np.ndarray,
    q_init: np.ndarray,
    *,
    target_frame: str = "gripper_tip",
    mode: str = "position_vertical",
    retry_mm: float = 8.0,
    mid_frac: float = 0.5,
) -> tuple[IKResult, str]:
    """
    IK to target_urdf. When FK error after the first solve exceeds retry_mm (or the
    solve did not converge), retry via a Cartesian midpoint then the final target.
    """
    target_urdf = np.asarray(target_urdf, dtype=float).reshape(3)
    q_init = np.asarray(q_init, dtype=float).reshape(5).copy()

    def _solve(q0: np.ndarray, tgt: np.ndarray, m: str) -> IKResult:
        return ik.solve(tgt, q_init=q0.copy(), target=target_frame, mode=m)

    res1 = _solve(q_init, target_urdf, mode)
    err1 = fk_pos_err_mm(fk, res1.joints_rad, target_urdf, target_frame=target_frame)
    if res1.converged and err1 <= retry_mm:
        return res1, "direct"

    p0 = fk.fk(q_init, target=target_frame)["position"]
    mid = p0 + float(mid_frac) * (target_urdf - p0)

    res_mid = _solve(q_init, mid, mode)
    mid_mode = mode
    if not res_mid.converged or fk_pos_err_mm(fk, res_mid.joints_rad, mid, target_frame=target_frame) > retry_mm * 2:
        res_mid_pos = _solve(q_init, mid, "position")
        if res_mid_pos.converged or (
            not res_mid.converged
            and fk_pos_err_mm(fk, res_mid_pos.joints_rad, mid, target_frame=target_frame)
            < fk_pos_err_mm(fk, res_mid.joints_rad, mid, target_frame=target_frame)
        ):
            res_mid = res_mid_pos
            mid_mode = "position"

    q_mid = res_mid.joints_rad.copy() if res_mid.converged else q_init.copy()
    res2 = _solve(q_mid, target_urdf, mode)
    err2 = fk_pos_err_mm(fk, res2.joints_rad, target_urdf, target_frame=target_frame)

    pick = res2 if (err2 < err1 or (res2.converged and not res1.converged)) else res1
    pick_err = err2 if pick is res2 else err1
    note = (
        f"staged mid_frac={mid_frac} via {mid_mode} "
        f"err {err1:.1f} -> {pick_err:.1f} mm"
    )
    return pick, note
