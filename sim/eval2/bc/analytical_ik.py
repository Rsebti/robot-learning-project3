"""Analytical closed-form IK for the SO-101 5-DoF arm.

The SO-101 has 5 arm DoF (shoulder_pan, shoulder_lift, elbow_flex,
wrist_flex, wrist_roll) plus 1 gripper DoF. Differential IK on a
5-DoF arm tracking a 6-D goal is mathematically under-determined,
which causes the wrist drift and slack-variable issues we observed
in v1-v3 of scripted_controller.py.

This module solves the IK exactly by exploiting the SO-101's joint
topology under a TOP-DOWN gripper constraint:

    1. shoulder_pan = atan2(y, x)               # 1-DoF azimuth
    2. shoulder_lift, elbow_flex, wrist_flex    # 3-DoF planar with
                                                  orientation phi imposed
    3. wrist_roll = 0                           # 1-DoF, free for
                                                  symmetric grasps

J2, J3, J4 share the same local axis after J1's rotation, so they
collapse to a planar 3R problem with the orientation constraint
phi = theta2 + theta3 + theta4 = -pi/2 (gripper points straight down).

This gives an exact O(1) solution — no iteration, no Jacobian, no
silent failures from singularities.

See ``so101_analytical_ik_implementation.md`` for the full derivation.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch


class IKResult(NamedTuple):
    """Output of analytical_ik_so101.

    joint_pos: (N, 5) tensor of joint angles in radians, in the order
        [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll].
    reachable: (N,) bool tensor — False if the target is geometrically
        unreachable (distance > L1+L2 or < |L1-L2|). When False, the
        joint_pos is the closest feasible solution (clamped), not the
        exact answer.
    """

    joint_pos: torch.Tensor
    reachable: torch.Tensor


def analytical_ik_so101(
    target_xyz_b: torch.Tensor,
    L1: float,
    L2: float,
    L3: float,
    phi: float | torch.Tensor = -math.pi / 2,
    elbow_up: bool = True,
    base_offset_z: float = 0.0,
    base_offset_r: float = 0.0,
    offset_theta2: float = 0.0,
    offset_theta3: float = 0.0,
    offset_theta4: float = 0.0,
) -> IKResult:
    """Closed-form IK for SO-101 with top-down gripper orientation.

    Internally solves in the "planar 3R" convention (theta_n_ik defined
    so that phi_n = theta_2_ik + ... + theta_n_ik = absolute link angle
    from horizontal) then converts to the URDF joint convention via
    additive offsets:

        theta_n_urdf = theta_n_ik - offset_theta_n

    The offsets are computed empirically by measure_link_lengths.py
    from the body positions at home pose. They capture the kinematic
    bias introduced by the URDF rpy at each joint origin (e.g. for
    SO-101 the upper arm is already at +74 deg from horizontal at
    theta2_urdf=0).

    Args:
        target_xyz_b: (N, 3) target gripper TIP positions in base frame.
        L1, L2, L3: link lengths from measure_link_lengths.py.
        phi: gripper orientation angle in the arm plane.
            -pi/2 = pointing straight down.
        elbow_up: True picks the elbow-up branch of the 2-link IK.
        base_offset_z: vertical offset of J2 axis above robot base.
        offset_theta2/3/4: URDF<->IK angular offsets from
            measure_link_lengths.py. Default 0 = pure planar 3R, no
            URDF correction (use only for testing).

    Returns:
        IKResult(joint_pos, reachable). joint_pos is (N, 5) in URDF
        convention, ready to be commanded to the robot.
    """
    device = target_xyz_b.device
    dtype = target_xyz_b.dtype
    n = target_xyz_b.shape[0]

    x = target_xyz_b[:, 0]
    y = target_xyz_b[:, 1]
    z = target_xyz_b[:, 2] - base_offset_z

    # --- Step 1: shoulder_pan from XY of the target. -----------------
    theta1 = torch.atan2(y, x)

    # --- Step 2: project to 2D arm plane (r horizontal, z vertical). -
    # Subtract base_offset_r so r=0 corresponds to J2 (where the planar
    # 2-link IK starts), not to the J1 axis. J2 is offset horizontally
    # from J1 by base_offset_r (= sqrt(J2.x^2 + J2.y^2) at home).
    r = torch.sqrt(x * x + y * y) - base_offset_r

    # --- Step 3-4: subtract the gripper segment to find wrist pos. ---
    # phi can be a Python float (uniform for all envs) or a (N,) tensor
    # (per-env phi, e.g. from adaptive_phi search). Use torch ops so
    # both cases broadcast correctly with r and z.
    if torch.is_tensor(phi):
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
    else:
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
    r_wrist = r - L3 * cos_phi
    z_wrist = z - L3 * sin_phi

    # --- Step 5: 2-link planar IK (J2, J3) for the wrist position. ---
    d_sq = r_wrist * r_wrist + z_wrist * z_wrist

    cos_t3 = (d_sq - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    reachable = cos_t3.abs() <= 1.0
    cos_t3 = cos_t3.clamp(-1.0, 1.0)

    sin_t3_unsigned = torch.sqrt(torch.clamp(1.0 - cos_t3 * cos_t3, min=0.0))
    sin_t3 = sin_t3_unsigned if elbow_up else -sin_t3_unsigned
    theta3_ik = torch.atan2(sin_t3, cos_t3)

    theta2_ik = torch.atan2(z_wrist, r_wrist) - torch.atan2(
        L2 * sin_t3, L1 + L2 * cos_t3
    )

    # --- Step 6: wrist_flex closes the orientation constraint. -------
    theta4_ik = phi - theta2_ik - theta3_ik

    # --- Step 7: convert IK convention -> URDF convention. -----------
    theta2_urdf = theta2_ik - offset_theta2
    theta3_urdf = theta3_ik - offset_theta3
    theta4_urdf = theta4_ik - offset_theta4

    # --- Step 8: wrist_roll free, set to 0 for symmetric grasps. -----
    theta5 = torch.zeros(n, device=device, dtype=dtype)

    joint_pos = torch.stack(
        [theta1, theta2_urdf, theta3_urdf, theta4_urdf, theta5], dim=-1
    )
    return IKResult(joint_pos=joint_pos, reachable=reachable)


def forward_kinematics_so101(
    joint_pos: torch.Tensor,
    L1: float,
    L2: float,
    L3: float,
    base_offset_z: float = 0.0,
    base_offset_r: float = 0.0,
    offset_theta2: float = 0.0,
    offset_theta3: float = 0.0,
    offset_theta4: float = 0.0,
) -> torch.Tensor:
    """Forward kinematics for SO-101 to gripper tip in base frame.

    Mirror of ``analytical_ik_so101`` (uses the same offsets to convert
    URDF -> IK convention).

    Args:
        joint_pos: (N, 5) joint angles in URDF convention
            [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll].
        L1, L2, L3: link lengths.
        base_offset_z: vertical offset of J2 axis above robot base.
        offset_theta2/3/4: URDF<->IK angular offsets.

    Returns:
        (N, 3) tip positions in base frame.
    """
    theta1 = joint_pos[:, 0]
    # Convert URDF -> IK convention (so the planar 3R formula applies).
    theta2_ik = joint_pos[:, 1] + offset_theta2
    theta3_ik = joint_pos[:, 2] + offset_theta3
    theta4_ik = joint_pos[:, 3] + offset_theta4
    # theta5 (wrist_roll) doesn't affect tip POSITION.

    # Cumulative absolute angles from horizontal.
    phi2 = theta2_ik
    phi3 = theta2_ik + theta3_ik
    phi4 = theta2_ik + theta3_ik + theta4_ik

    # In the (r, z) arm plane after J1 rotation. r is measured from J2
    # in the planar formula; we add base_offset_r at the end to recover
    # the world r (distance from the J1 axis).
    r_in_arm = (
        L1 * torch.cos(phi2)
        + L2 * torch.cos(phi3)
        + L3 * torch.cos(phi4)
    )
    z_in_arm = (
        L1 * torch.sin(phi2)
        + L2 * torch.sin(phi3)
        + L3 * torch.sin(phi4)
    )

    r_world = r_in_arm + base_offset_r
    z_world = z_in_arm + base_offset_z

    # Rotate the arm plane by theta1 (azimuth) into 3D base frame.
    x = r_world * torch.cos(theta1)
    y = r_world * torch.sin(theta1)

    return torch.stack([x, y, z_world], dim=-1)


if __name__ == "__main__":
    # Quick smoke test with arbitrary link lengths (real values come from
    # measure_link_lengths.py). Just verifies the math doesn't crash.
    L1, L2, L3 = 0.115, 0.135, 0.090
    base_offset_z = 0.05

    target = torch.tensor([
        [0.20, 0.00, 0.05],
        [0.20, -0.10, 0.05],
        [0.15, 0.05, 0.10],
    ])
    result = analytical_ik_so101(target, L1, L2, L3, base_offset_z=base_offset_z)
    print("Smoke test analytical_ik_so101:")
    print(f"  reachable: {result.reachable.tolist()}")
    print(f"  joint_pos:\n{result.joint_pos}")

    fk = forward_kinematics_so101(result.joint_pos, L1, L2, L3, base_offset_z)
    err = torch.norm(target - fk, dim=-1) * 1000.0
    print(f"  roundtrip error (mm): {err.tolist()}")
