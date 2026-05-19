"""
test_ik.py - IK round-trip and out-of-workspace rejection tests.

Run:
    python -m toolset.kinematics.tests.test_ik
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from toolset.kinematics.config import KinematicsConfig
from toolset.kinematics.ik import IKSolver
from toolset.kinematics.urdf_fk import SO101FK


REACHABLE_Q = [
    [0.0,  -1.0,  1.0,  0.0,  0.0],
    [0.3,  -0.8,  1.2,  0.1, -0.2],
    [-0.4, -1.2,  1.1, -0.3,  0.2],
    [0.6,  -0.5,  0.7,  0.0,  0.0],
    [-0.6, -1.5,  1.5,  0.2,  0.0],
]


def _assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)
    print(f"  PASS - {msg}")


def test_round_trip_position(ik: IKSolver, fk: SO101FK):
    print("\n[ik] round-trip (position-only): FK -> IK -> ||target - recovered_xyz||")
    q_init = [0.0, -math.pi / 4, math.pi / 4, 0.0, 0.0]
    errs = []
    for q_truth in REACHABLE_Q:
        target_xyz = fk.fk(q_truth, target="gripper_tip")["position"]
        res = ik.solve(target_xyz, q_init=q_init, target="gripper_tip", mode="position")
        recovered = fk.fk(res.joints_rad, target="gripper_tip")["position"]
        err_mm = float(np.linalg.norm(target_xyz - recovered) * 1000)
        errs.append(err_mm)
        print(f"  target={target_xyz.round(3).tolist()}  reason={res.reason:11s}  "
              f"iters={res.iters:3d}  pos_err={err_mm:.2f}mm")
    _assert(all(e < 5.0 for e in errs),
            f"all reachable targets converge within 5 mm  (max {max(errs):.2f} mm)")


def test_out_of_workspace_rejection(ik: IKSolver):
    print("\n[ik] out-of-workspace targets should be rejected immediately")
    unreachable = [
        [1.0, 0.0, 0.1],     # way too far forward
        [0.0, 0.0, 1.0],     # ceiling
        [-0.5, 0.0, 0.1],    # behind base
        [0.1, 0.5, 0.1],     # too far left
    ]
    q_init = [0.0, -math.pi / 4, math.pi / 4, 0.0, 0.0]
    for tgt in unreachable:
        res = ik.solve(tgt, q_init=q_init, target="gripper_tip", mode="position")
        ok = (res.reason == "out_of_workspace")
        print(f"  target={tgt}  reason={res.reason!r}  -> rejected={ok}")
        _assert(ok, f"target {tgt} rejected as out-of-workspace")


def test_round_trip_with_orientation(ik: IKSolver, fk: SO101FK):
    print("\n[ik] round-trip (position + vertical-down constraint)")
    q_init = [0.0, -math.pi / 4, math.pi / 4, 0.0, 0.0]
    errs_pos = []
    errs_rot = []
    for q_truth in REACHABLE_Q[:3]:
        target_xyz = fk.fk(q_truth, target="gripper_tip")["position"]
        res = ik.solve(target_xyz, q_init=q_init, target="gripper_tip",
                       mode="position_vertical")
        errs_pos.append(res.pos_err_m * 1000)
        errs_rot.append(res.rot_err_rad)
        print(f"  target={target_xyz.round(3).tolist()}  reason={res.reason}  "
              f"pos={res.pos_err_m * 1000:.2f}mm  rot={res.rot_err_rad:.3f}rad")
    # With only 5 DoF, vertical-down isn't always reachable - just check
    # the pose error stays bounded.
    _assert(all(e < 30.0 for e in errs_pos),
            f"position error stays bounded with orientation constraint (max {max(errs_pos):.1f} mm)")


def main():
    kcfg = KinematicsConfig.load()
    fk = SO101FK(kcfg.urdf_path, gripper_tip_offset=kcfg.gripper_tip_offset_m)
    ik = IKSolver(kcfg, fk)
    test_round_trip_position(ik, fk)
    test_out_of_workspace_rejection(ik)
    test_round_trip_with_orientation(ik, fk)
    print("\n[ik] all IK tests passed.")


if __name__ == "__main__":
    main()
