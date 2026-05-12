"""Verify the actual palm-jaw vertical offset across several robot poses.

The V2.18 hover design assumes ``palm-jaw vertical offset ≈ 5 cm in top-down
pose``, so that ``palm at cube_top + 5cm → jaw at cube_top``. This was a
designer assertion in the docstring of `hover_height_gaussian`, never
empirically verified. The only documented dump is in HORIZONTAL home pose
(palm-jaw ≈ 0 mm), which doesn't tell us the top-down offset.

This script forces the SO-101 into several poses and dumps the actual
palm-jaw delta in world Z, so we can confirm (or refute) the 5 cm value
before trusting it for the V2.18b hover sweet spot.

Usage:
    cd C:/Users/user/Desktop/MA2/robot-learning-project3
    C:/Users/user/Desktop/MA2/isaac/isaac_so_arm101/.venv/Scripts/python.exe \
      sim/eval2/scripts/verify_palm_jaw_offset.py
"""
from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    default="Isaac-LeIsaac-SO101-Lift-Visual-V218B-cold-Play-v0",
    help="gym task id (use a *-Play-v0 task for small num_envs).",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--settle_steps",
    type=int,
    default=30,
    help="number of physics steps to settle after writing joint state.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

import sim.eval2  # noqa: F401, E402
import isaac_so_arm101.tasks  # noqa: F401, E402
from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


# Each pose is a (name, dict) where dict maps joint_name → target_pos.
# Joints not listed are left at their reset value.
TEST_POSES: list[tuple[str, dict | None]] = [
    ("HOME (after env.reset, no override)", None),
    (
        "WRIST FLEX -1.5 rad (try gripper-down)",
        {"wrist_flex": -1.5},
    ),
    (
        "WRIST FLEX +1.5 rad (try gripper-down other dir)",
        {"wrist_flex": +1.5},
    ),
    (
        "ARM REACH FORWARD + WRIST DOWN (best top-down attempt)",
        {
            "shoulder_pan": 0.0,
            "shoulder_lift": -1.0,
            "elbow_flex": +1.5,
            "wrist_flex": -1.5,
            "wrist_roll": 0.0,
        },
    ),
    (
        "ALT TOP-DOWN (different elbow sign)",
        {
            "shoulder_pan": 0.0,
            "shoulder_lift": -0.5,
            "elbow_flex": -1.5,
            "wrist_flex": -1.5,
            "wrist_roll": 0.0,
        },
    ),
]


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg) -> None:  # type: ignore[no-untyped-def]
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    u = env.unwrapped
    robot = u.scene["robot"]
    ee_frame = u.scene["ee_frame"]
    cube = u.scene["cube"]
    sim = u.sim

    joint_names = list(robot.data.joint_names)
    body_names = list(robot.data.body_names)
    print()
    print("=" * 78)
    print("PALM-JAW VERTICAL OFFSET VERIFICATION")
    print("=" * 78)
    print(f"Task        : {args_cli.task}")
    print(f"Joint names : {joint_names}")
    print(f"Body  names : {body_names}")

    # Locate the palm body (= 'gripper' per the FrameTransformer config)
    # and the jaw body for sanity-print.
    try:
        palm_body_idx = robot.find_bodies("gripper")[0][0]
    except Exception:
        palm_body_idx = None
    try:
        jaw_body_idx = robot.find_bodies("jaw")[0][0]
    except Exception:
        jaw_body_idx = None

    cube_z = cube.data.root_pos_w[0, 2].item()
    cube_top = cube_z + 0.010
    cube_bottom = cube_z - 0.010
    print(f"Cube spawn z = {cube_z:+.4f}  (top={cube_top:+.4f}, bottom={cube_bottom:+.4f})")

    physics_dt = sim.get_physics_dt()

    for pose_name, pose_dict in TEST_POSES:
        env.reset()

        if pose_dict is not None:
            new_pos = robot.data.joint_pos[0].clone()
            for jname, val in pose_dict.items():
                if jname not in joint_names:
                    print(f"  [WARN] joint '{jname}' not found — skipping")
                    continue
                idx = joint_names.index(jname)
                new_pos[idx] = float(val)
            new_pos = new_pos.unsqueeze(0)
            new_vel = torch.zeros_like(new_pos)
            # Write physics state AND set actuator target — otherwise the PD
            # controller pulls the joints back to the previous target each step.
            robot.write_joint_state_to_sim(new_pos, new_vel)
            robot.set_joint_position_target(new_pos)
            robot.write_data_to_sim()
            # Let the simulator settle.
            for _ in range(args_cli.settle_steps):
                sim.step(render=False)
                u.scene.update(physics_dt)

        # Read positions after settle.
        palm_t_z = ee_frame.data.target_pos_w[0, 0, 2].item()
        jaw_t_z = ee_frame.data.target_pos_w[0, 1, 2].item()
        palm_b_z = (
            robot.data.body_pos_w[0, palm_body_idx, 2].item()
            if palm_body_idx is not None
            else float("nan")
        )
        jaw_b_z = (
            robot.data.body_pos_w[0, jaw_body_idx, 2].item()
            if jaw_body_idx is not None
            else float("nan")
        )

        # Also dump joint pos to confirm what was actually written.
        actual = robot.data.joint_pos[0].cpu().numpy()
        joints_str = ", ".join(
            f"{n}={v:+.3f}" for n, v in zip(joint_names, actual)
        )

        delta_target = palm_t_z - jaw_t_z
        delta_body = palm_b_z - jaw_b_z

        print()
        print(f"--- {pose_name} ---")
        print(f"  joints     : {joints_str}")
        print(f"  palm BODY  z (world) : {palm_b_z:+.4f}")
        print(f"  jaw  BODY  z (world) : {jaw_b_z:+.4f}")
        print(f"  palm-jaw BODY  delta_z    : {delta_body:+.4f}  m")
        print(f"  palm TARGET z (world): {palm_t_z:+.4f}   <- used by hover_height fn")
        print(f"  jaw  TARGET z (world): {jaw_t_z:+.4f}   <- used by strict_grasp predicate")
        print(f"  palm-jaw TARGET delta_z   : {delta_target:+.4f}  m   <-*** KEY MEASUREMENT")
        print(f"  -> if palm sits at cube_top+5cm = {cube_top + 0.05:+.4f},")
        print(f"     then jaw target sits at      {cube_top + 0.05 - delta_target:+.4f}")
        print(f"     vs cube_top = {cube_top:+.4f}, cube_bottom = {cube_bottom:+.4f}")
        if delta_target > 0:
            jaw_pos_in_cube = (cube_top + 0.05 - delta_target) - cube_z
            if jaw_pos_in_cube > 0.010:
                verdict = "ABOVE cube top -> grasp impossible"
            elif jaw_pos_in_cube > -0.010:
                verdict = f"INSIDE cube ({jaw_pos_in_cube*1000:+.0f} mm from cube center) -> graspable"
            else:
                verdict = "BELOW cube bottom -> jaw passes under cube, would hit table"
            print(f"     verdict at hover sweet spot : {verdict}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
