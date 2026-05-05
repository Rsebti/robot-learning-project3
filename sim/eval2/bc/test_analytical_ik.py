"""Roundtrip test for the analytical IK.

Strategy:
    1. Sample N random joint configurations (5-D) within reasonable
       ranges, with theta4 chosen so the gripper points down
       (phi = theta2 + theta3 + theta4 = -pi/2).
    2. Run forward kinematics to get the resulting EE tip positions.
    3. Run our analytical IK on those positions.
    4. Run forward kinematics on the IK output.
    5. Compare original positions vs FK-on-IK positions.

We do NOT compare joint angles directly because:
    - There can be multiple IK solutions (elbow-up vs elbow-down).
    - Joint wrapping (theta vs theta + 2*pi) is allowed.
    - Only the tip position matters for our use case.

Usage:
    cd C:/Users/user/Desktop/MA2/robot-learning-project3
    python -m sim.eval2.bc.test_analytical_ik

(No Isaac Sim required — pure torch.)

Pass criterion: max position error < 1 mm on all reachable cases.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

from sim.eval2.bc.analytical_ik import (
    analytical_ik_so101,
    forward_kinematics_so101,
)


def main():
    config_path = Path(__file__).parent / "so101_link_lengths.json"
    if not config_path.exists():
        print(
            f"ERROR: {config_path} does not exist.\n"
            "Run sim.eval2.bc.measure_link_lengths first to generate it."
        )
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)
    L1 = config["L1"]
    L2 = config["L2"]
    L3 = config["L3"]
    base_offset_z = config["base_offset_z"]
    base_offset_r = config.get("base_offset_r", 0.0)
    o2 = config.get("offset_theta2", 0.0)
    o3 = config.get("offset_theta3", 0.0)
    o4 = config.get("offset_theta4", 0.0)
    print(
        f"Loaded: L1={L1:.4f} L2={L2:.4f} L3={L3:.4f} "
        f"base_z={base_offset_z:.4f} base_r={base_offset_r:.4f}"
    )
    print(f"        offsets theta2/3/4 = {o2:+.4f} / {o3:+.4f} / {o4:+.4f}")

    torch.manual_seed(42)
    n = 1000
    phi = -math.pi / 2

    # Sample URDF joint configurations within reasonable ranges. We
    # constrain wrist_flex so the gripper points down (phi = -pi/2 in
    # IK convention), accounting for the URDF -> IK offset:
    #   phi = (theta2_urdf + o2) + (theta3_urdf + o3) + (theta4_urdf + o4)
    #   => theta4_urdf = phi - o2 - o3 - o4 - theta2_urdf - theta3_urdf
    joint_pos = torch.zeros(n, 5)
    joint_pos[:, 0] = torch.empty(n).uniform_(-1.5, 1.5)
    joint_pos[:, 1] = torch.empty(n).uniform_(-0.5, 0.5)
    joint_pos[:, 2] = torch.empty(n).uniform_(-0.5, 0.5)
    joint_pos[:, 3] = phi - o2 - o3 - o4 - joint_pos[:, 1] - joint_pos[:, 2]
    joint_pos[:, 4] = 0.0

    # FK to get target tip positions (URDF angles in, position out)
    ee_target = forward_kinematics_so101(
        joint_pos, L1, L2, L3, base_offset_z,
        base_offset_r=base_offset_r,
        offset_theta2=o2, offset_theta3=o3, offset_theta4=o4,
    )

    # IK to recover joint configurations (target in, URDF angles out)
    result = analytical_ik_so101(
        ee_target,
        L1=L1, L2=L2, L3=L3,
        phi=phi, elbow_up=True,
        base_offset_z=base_offset_z,
        base_offset_r=base_offset_r,
        offset_theta2=o2, offset_theta3=o3, offset_theta4=o4,
    )
    n_reachable = int(result.reachable.sum().item())
    print(f"\n{n_reachable}/{n} samples flagged as reachable")

    # FK on the recovered joints (with same offsets)
    ee_recovered = forward_kinematics_so101(
        result.joint_pos, L1, L2, L3, base_offset_z,
        offset_theta2=o2, offset_theta3=o3, offset_theta4=o4,
    )

    # Position error on reachable cases (the only ones where the IK
    # is supposed to give an exact answer).
    err_m = torch.norm(ee_target - ee_recovered, dim=-1)
    err_reachable = err_m[result.reachable]

    err_mm_mean = err_reachable.mean().item() * 1000.0
    err_mm_max = err_reachable.max().item() * 1000.0
    err_mm_p99 = torch.quantile(err_reachable, 0.99).item() * 1000.0

    print(f"\nEE position error on reachable cases:")
    print(f"  mean : {err_mm_mean:.4f} mm")
    print(f"  p99  : {err_mm_p99:.4f} mm")
    print(f"  max  : {err_mm_max:.4f} mm")

    if err_mm_max < 1.0:
        print("\n[PASS] IK roundtrip error < 1 mm on all reachable cases.")
        return 0
    else:
        print("\n[FAIL] IK roundtrip error too high — check sign conventions.")
        # Show the worst sample for debugging
        idx_worst = err_m.argmax().item()
        print(f"\nWorst case (index {idx_worst}, reachable={bool(result.reachable[idx_worst])}):")
        print(f"  joint_pos_orig:  {joint_pos[idx_worst].tolist()}")
        print(f"  ee_target:       {ee_target[idx_worst].tolist()}")
        print(f"  joint_pos_ik:    {result.joint_pos[idx_worst].tolist()}")
        print(f"  ee_recovered:    {ee_recovered[idx_worst].tolist()}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
