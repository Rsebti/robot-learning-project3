"""Measure SO-101 link lengths from a running Isaac Lab env.

One-off calibration script. Sets the robot to its zero pose, reads the
body positions in the robot base frame, and writes
``so101_link_lengths.json`` with L1, L2, L3, base_offset_z. Consumed by
``analytical_ik.py``.

Run from the isaac_so_arm101 venv:

    cd C:/Users/user/Desktop/MA2/isaac/isaac_so_arm101
    uv run python -m sim.eval2.bc.measure_link_lengths

The script also prints all available joint and body names — handy
reference if the URDF naming differs from what the spec assumes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

# AppLauncher must be invoked before any torch / isaaclab import.
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Eval2-PickInClutter-Play-v1")
parser.add_argument(
    "--output",
    type=str,
    default=str(Path(__file__).parent / "so101_link_lengths.json"),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def main():
    import gymnasium as gym
    import numpy as np
    import torch
    from isaaclab.utils.math import subtract_frame_transforms
    from isaaclab_tasks.utils import parse_env_cfg

    import sim.eval2  # registers gym envs

    env_cfg = parse_env_cfg(args.task, num_envs=1)
    env = gym.make(args.task, cfg=env_cfg)
    env.reset()

    robot = env.unwrapped.scene["robot"]
    device = env.unwrapped.device

    print("\n=== robot.data.joint_names ===")
    for i, n in enumerate(robot.data.joint_names):
        print(f"  [{i}] {n}")
    print("\n=== robot.data.body_names ===")
    for i, n in enumerate(robot.data.body_names):
        print(f"  [{i}] {n}")

    # Build a zero arm action (5 arm joints + 1 gripper) so the PD
    # controllers drive the robot to its default joint pose. Default
    # for wrist_flex is 1.57 in SO_ARM101_CFG, so the arm settles in
    # the gripper-down home pose — exactly what the IK derivation
    # assumes (theta_i = 0 for all i except wrist_flex which we treat
    # as the orientation reference).
    action_dim = env.unwrapped.action_manager.total_action_dim
    print(f"\n[measure] action_dim = {action_dim}")
    zero_action = torch.zeros(1, action_dim, device=device)

    print("[measure] stepping 30 times to settle to home pose...")
    for _ in range(30):
        env.step(zero_action)

    # Read body world poses, convert to base frame.
    root_pos_w = robot.data.root_state_w[0:1, :3]
    root_quat_w = robot.data.root_state_w[0:1, 3:7]

    body_names_to_check = [
        # Common SO-101 / SO-ARM101 body names. The script will skip any
        # missing one and report — adjust the list if needed based on
        # the body_names print above.
        "base_link",
        "shoulder_link",
        "upper_arm_link",
        "lower_arm_link",
        "wrist_link",
        "gripper_link",
        "gripper_frame_link",
    ]

    positions_b = {}
    print("\n=== body positions in base frame at home pose ===")
    for name in body_names_to_check:
        try:
            idx = robot.data.body_names.index(name)
        except ValueError:
            print(f"  ! '{name}' not in body_names — skipping")
            continue
        pos_w = robot.data.body_state_w[0:1, idx, :3]
        pos_b, _ = subtract_frame_transforms(
            root_pos_w, root_quat_w, pos_w
        )
        positions_b[name] = pos_b[0].cpu().numpy().tolist()
        p = positions_b[name]
        print(f"  {name:<22} x={p[0]:+.4f}  y={p[1]:+.4f}  z={p[2]:+.4f}")

    # Compute link lengths. We use the EUCLIDEAN distance between
    # consecutive joint origins, which captures the physical link
    # length regardless of axis conventions.
    def dist(a: list[float], b: list[float]) -> float:
        return float(np.linalg.norm(np.array(a) - np.array(b)))

    required = ["upper_arm_link", "lower_arm_link", "wrist_link", "gripper_frame_link"]
    missing = [n for n in required if n not in positions_b]
    if missing:
        print(f"\n!!! Missing required bodies: {missing}")
        print("    Adjust body_names_to_check based on the body_names list above.")
        env.close()
        return

    # SO-101 link chain (after J1 rotates the arm plane):
    #   J2 (shoulder_lift) at the origin of upper_arm_link
    #   J3 (elbow_flex)    at the origin of lower_arm_link
    #   J4 (wrist_flex)    at the origin of wrist_link
    #   tip                at gripper_frame_link (= the "fingertip" point
    #                      between the two jaws, where the env's
    #                      FrameTransformer also places its target)
    #
    # L1 = J2->J3 (upper arm length)
    # L2 = J3->J4 (forearm length)
    # L3 = J4->tip (wrist + gripper combined, since wrist_roll is locked)
    L1 = dist(positions_b["upper_arm_link"], positions_b["lower_arm_link"])
    L2 = dist(positions_b["lower_arm_link"], positions_b["wrist_link"])
    L3 = dist(positions_b["wrist_link"], positions_b["gripper_frame_link"])

    # base_offset_z = z of the J2 axis above the robot base origin.
    base_offset_z = positions_b["upper_arm_link"][2]
    # base_offset_r = horizontal distance from J1 axis (z-axis through base)
    # to J2 axis. With J1 = 0 at home, this is the (x, y)-norm of J2's
    # world position. The IK needs this to start its planar 2-link solve
    # from the J2 position, not from the base.
    j2_x = positions_b["upper_arm_link"][0]
    j2_y = positions_b["upper_arm_link"][1]
    base_offset_r = float(np.sqrt(j2_x * j2_x + j2_y * j2_y))

    # ---- Angular offsets (URDF <-> IK convention) ------------------
    # The URDF home pose has theta2_urdf = 0, theta3_urdf = 0,
    # theta4_urdf = 1.57. But in the planar-3R IK convention, the
    # "0" of each joint corresponds to "no rotation from the previous
    # link's direction". Due to the rpy at each URDF joint origin,
    # these conventions disagree — we need to compute the offsets
    # empirically.
    #
    # In the IK, theta_n_ik is defined so that:
    #   phi_n = theta_2_ik + theta_3_ik + ... + theta_n_ik
    # where phi_n is the absolute angle of link n with horizontal in
    # the arm plane (positive = link points up in z).
    #
    # At home, we measure phi_n directly from the body positions:
    import math
    def angle_in_arm_plane(p_from: list[float], p_to: list[float]) -> float:
        """Angle (rad) of vector p_from->p_to in the (r, z) arm plane.
        r = horizontal distance from origin, z = world z."""
        dx = p_to[0] - p_from[0]
        dy = p_to[1] - p_from[1]
        dz = p_to[2] - p_from[2]
        # In arm plane: dr is the magnitude of (dx, dy) projected onto
        # the J1=0 plane (which is the XZ plane in world frame).
        dr = math.sqrt(dx * dx + dy * dy)
        # Sign convention: dr > 0 means going AWAY from base. If dx < 0
        # (going toward -x), the link is folded backward — give dr a
        # negative sign so atan2 disambiguates.
        if dx < 0:
            dr = -dr
        return math.atan2(dz, dr)

    phi2_home = angle_in_arm_plane(
        positions_b["upper_arm_link"], positions_b["lower_arm_link"]
    )
    phi3_home = angle_in_arm_plane(
        positions_b["lower_arm_link"], positions_b["wrist_link"]
    )
    phi4_home = angle_in_arm_plane(
        positions_b["wrist_link"], positions_b["gripper_frame_link"]
    )

    # IK angles at home (relative): theta_n_ik = phi_n - phi_(n-1).
    # phi_0 = 0 (horizontal reference).
    theta2_ik_home = phi2_home
    theta3_ik_home = phi3_home - phi2_home
    theta4_ik_home = phi4_home - phi3_home

    # URDF angles at home: (0, 0, 1.57).
    theta2_urdf_home = 0.0
    theta3_urdf_home = 0.0
    theta4_urdf_home = math.pi / 2

    # Offset to convert IK -> URDF: theta_urdf = theta_ik - offset.
    offset_theta2 = theta2_ik_home - theta2_urdf_home
    offset_theta3 = theta3_ik_home - theta3_urdf_home
    offset_theta4 = theta4_ik_home - theta4_urdf_home

    result = {
        "L1": L1,
        "L2": L2,
        "L3": L3,
        "base_offset_z": base_offset_z,
        "base_offset_r": base_offset_r,
        "offset_theta2": offset_theta2,
        "offset_theta3": offset_theta3,
        "offset_theta4": offset_theta4,
        "phi2_home": phi2_home,
        "phi3_home": phi3_home,
        "phi4_home": phi4_home,
        "raw_positions_b": positions_b,
        "task": args.task,
    }

    out_path = Path(args.output)
    out_path.write_text(json.dumps(result, indent=2))

    print("\n" + "=" * 60)
    print(f"  L1 (upper arm)        = {L1:.4f} m")
    print(f"  L2 (forearm)          = {L2:.4f} m")
    print(f"  L3 (J4 -> tip)        = {L3:.4f} m")
    print(f"  base_offset_z (J2.z)  = {base_offset_z:.4f} m")
    print(f"  base_offset_r (J2.r)  = {base_offset_r:.4f} m")
    print(f"  phi2 (home)           = {phi2_home:+.4f} rad ({math.degrees(phi2_home):+.1f} deg)")
    print(f"  phi3 (home)           = {phi3_home:+.4f} rad ({math.degrees(phi3_home):+.1f} deg)")
    print(f"  phi4 (home)           = {phi4_home:+.4f} rad ({math.degrees(phi4_home):+.1f} deg)")
    print(f"  offset_theta2         = {offset_theta2:+.4f} rad")
    print(f"  offset_theta3         = {offset_theta3:+.4f} rad")
    print(f"  offset_theta4         = {offset_theta4:+.4f} rad")
    print(f"  saved to: {out_path}")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    import sys
    import traceback
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        sys.stdout.flush()
        sys.stderr.flush()
        print("\n[measure] !!! UNCAUGHT EXCEPTION !!!", flush=True)
        traceback.print_exc()
    finally:
        simulation_app.close()
