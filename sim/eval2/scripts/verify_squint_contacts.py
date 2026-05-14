"""Verify the Squint-native contact sensors fire when the cube touches the jaw.

Uses the kinematic-hold pattern (cf. ``isaac_lab_gpu_physx_contact_filter``
memory): the cube is teleported each physics step to overlap the jaw body
so PhysX generates a steady contact event. Reads
``sensor.data.net_forces_w`` on ``gripper_contact`` and ``jaw_contact``
and prints magnitudes for three scenarios:

  1. Home pose, cube far away  → expect both forces ≈ 0 N.
  2. Cube teleported on top of the gripper body each step → F_g > 0.
  3. Cube teleported on top of the jaw body each step       → F_j > 0.

Run::

    cd C:/Users/user/Desktop/MA2/isaac/isaac_so_arm101
    .venv/Scripts/python.exe \
        C:/Users/user/Desktop/MA2/robot-learning-project3/sim/eval2/scripts/verify_squint_contacts.py
"""
from __future__ import annotations
import argparse
import torch  # noqa


parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-SquintNative-Place-Play-v0")
parser.add_argument("--n_settle", type=int, default=10,
                    help="Substeps to settle between scenarios.")
parser.add_argument("--n_probe", type=int, default=10,
                    help="Steps to hold the cube + read forces.")
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
args.num_envs = 1
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import sim.eval2  # noqa: F401,E402


def _hold_cube_on_body(env, cube, robot, body_name: str, z_offset: float = 0.0):
    """Write the cube's pose to the body's pose so PhysX sees a continuous overlap."""
    bi = robot.body_names.index(body_name)
    pos = robot.data.body_pos_w[:, bi].clone()
    quat = robot.data.body_quat_w[:, bi].clone()
    pos[:, 2] += z_offset
    pose = torch.cat([pos, quat], dim=-1)
    vel = torch.zeros(env.unwrapped.num_envs, 6, device=env.unwrapped.device)
    cube.write_root_pose_to_sim(pose)
    cube.write_root_velocity_to_sim(vel)


def _print_forces(env, label: str):
    s_g = env.unwrapped.scene.sensors["gripper_contact"]
    s_j = env.unwrapped.scene.sensors["jaw_contact"]
    F_g = s_g.data.net_forces_w
    F_j = s_j.data.net_forces_w
    Fg_mag = torch.linalg.norm(F_g[0, 0]).item() if F_g is not None else float("nan")
    Fj_mag = torch.linalg.norm(F_j[0, 0]).item() if F_j is not None else float("nan")
    print(f"[probe] {label:<55s}  F_g={Fg_mag:7.3f} N   F_j={Fj_mag:7.3f} N")


def main():
    env_cfg = parse_env_cfg(args.task, num_envs=1)
    try: env_cfg.recorders = None
    except Exception: pass
    env = gym.make(args.task, cfg=env_cfg)
    base = env.unwrapped
    obs, _ = env.reset(seed=0)

    cube = base.scene["cube"]
    robot = base.scene["robot"]
    n_act = env.action_space.shape[-1]
    zero = torch.zeros(1, n_act, device=base.device)

    # Settle in the home pose.
    for _ in range(args.n_settle):
        env.step(zero)

    # --- Scenario 1: home pose, cube far away --------------------------
    # Put the cube somewhere far so no contact is possible.
    far_pose = torch.tensor([[0.6, 0.5, 0.05, 1.0, 0.0, 0.0, 0.0]], device=base.device)
    cube.write_root_pose_to_sim(far_pose)
    cube.write_root_velocity_to_sim(torch.zeros(1, 6, device=base.device))
    for _ in range(args.n_probe):
        env.step(zero)
    _print_forces(env, "(1) HOME pose, cube far away (expect ~0 N)")

    # --- Scenario 2: cube held overlapping the GRIPPER body ------------
    for _ in range(args.n_probe):
        _hold_cube_on_body(env, cube, robot, body_name="gripper")
        env.step(zero)
    _print_forces(env, "(2) cube pinned ON gripper body (expect F_g > 0)")

    # --- Scenario 3: cube held overlapping the JAW body ----------------
    # Re-settle the robot before scenario 3 (cube was on gripper above).
    cube.write_root_pose_to_sim(far_pose)
    cube.write_root_velocity_to_sim(torch.zeros(1, 6, device=base.device))
    for _ in range(args.n_settle):
        env.step(zero)
    for _ in range(args.n_probe):
        _hold_cube_on_body(env, cube, robot, body_name="jaw")
        env.step(zero)
    _print_forces(env, "(3) cube pinned ON jaw body (expect F_j > 0)")

    env.close()
    app.close()


if __name__ == "__main__":
    main()
