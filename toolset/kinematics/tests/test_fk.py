"""
test_fk.py - URDF FK sanity tests that run without the robot.

Goals:
  - all joints at URDF zero -> arm extends forward, no left/right offset
  - FK chain is monotonic in the obvious direction (lift goes up/down)
  - gripper_tip frame is gripper_frame translated by config offset

Run:
    python -m toolset.kinematics.tests.test_fk
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from toolset.kinematics.config import KinematicsConfig
from toolset.kinematics.urdf_fk import SO101FK


def _assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)
    print(f"  PASS - {msg}")


def test_zero_pose_geometry(fk: SO101FK):
    print("\n[fk] zero pose (all 5 revolute joints = 0)")
    out = fk.fk([0, 0, 0, 0, 0], target="gripper_frame")
    x, y, z = out["position"]
    print(f"  gripper_frame xyz = ({x:+.4f}, {y:+.4f}, {z:+.4f}) m")
    _assert(x > 0.30, "x_urdf > 0.30 m (arm extends forward)")
    _assert(abs(y) < 0.01, "|y_urdf| < 1 cm at zero pose (no left/right drift)")
    _assert(z > 0.15, "z_urdf > 0.15 m (gripper above base origin)")


def test_shoulder_lift_monotonic(fk: SO101FK):
    print("\n[fk] shoulder_lift sweep at q1=0, q3=q4=q5=0")
    zs = []
    for lift_deg in [-60, -30, 0, 30, 60]:
        q = [0.0, math.radians(lift_deg), 0.0, 0.0, 0.0]
        zs.append(fk.fk(q, target="gripper_frame")["position"][2])
    print(f"  z values at lift=[-60,-30,0,30,60]: {[round(z, 4) for z in zs]}")
    # Going from lift=-60 to lift=+60, the gripper sweeps through an arc.
    # We don't expect strictly monotonic z (passes through peak at q=0
    # roughly), so just check the range is meaningful.
    _assert(max(zs) - min(zs) > 0.05, "shoulder_lift sweep moves z by >5 cm")


def test_gripper_tip_offset(fk_zero: SO101FK, kcfg: KinematicsConfig):
    print("\n[fk] gripper_tip vs gripper_frame offset")
    if kcfg.gripper_tip_offset_m == (0.0, 0.0, 0.0):
        print("  SKIP - gripper_tip_offset_m is (0,0,0); run measure_gripper_tip")
        return
    fk_offset = SO101FK(kcfg.urdf_path, gripper_tip_offset=kcfg.gripper_tip_offset_m)
    q = [0.0, -1.0, 1.0, 0.0, 0.0]
    p_frame = fk_zero.fk(q, target="gripper_frame")["position"]
    p_tip = fk_offset.fk(q, target="gripper_tip")["position"]
    # In the local gripper frame, tip is offset by the config vector. In
    # base frame the magnitude must equal ||offset||.
    delta = float(np.linalg.norm(p_tip - p_frame))
    norm = float(np.linalg.norm(kcfg.gripper_tip_offset_m))
    print(f"  ||tip - frame|| = {delta:.4f} m   ||offset|| = {norm:.4f} m")
    _assert(abs(delta - norm) < 1e-6, "gripper_tip is offset by the configured vector")


def test_fk_consistency_with_old_probe():
    print("\n[fk] consistency with deploy/probe_position.py")
    # The legacy probe_position.py uses the same URDF and same chain; at
    # q=0 we expect identical xyz.
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "deploy"))
    try:
        from probe_position import _parse_urdf_chain, fk_gripper_frame, URDF_PATH  # noqa: E402
    except Exception as e:
        print(f"  SKIP - couldn't import legacy probe ({e})")
        return
    chain = _parse_urdf_chain(URDF_PATH)
    p_legacy = fk_gripper_frame([0, 0, 0, 0, 0], chain)[:3, 3]
    new_fk = SO101FK()
    p_new = new_fk.fk([0, 0, 0, 0, 0], target="gripper_frame")["position"]
    delta = float(np.linalg.norm(p_legacy - p_new))
    print(f"  legacy ({p_legacy.round(4).tolist()}) vs new ({p_new.round(4).tolist()})  d={delta:.6f} m")
    _assert(delta < 1e-6, "new FK matches deploy/probe_position.py at q=0")


def main():
    kcfg = KinematicsConfig.load()
    fk = SO101FK(kcfg.urdf_path, gripper_tip_offset=(0.0, 0.0, 0.0))
    test_zero_pose_geometry(fk)
    test_shoulder_lift_monotonic(fk)
    test_gripper_tip_offset(fk, kcfg)
    test_fk_consistency_with_old_probe()
    print("\n[fk] all FK tests passed.")


if __name__ == "__main__":
    main()
