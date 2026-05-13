"""Replay a Squint-recorded qpos trajectory in Isaac to verify kinematic parity.

Reads two text files saved by the Squint side:

- ``episode_0_scene.txt``:
    robot_qpos x6  (initial qpos)
    cube_pose x y z qw qx qy qz  (world pose)
    bin_pose  x y z qw qx qy qz   (world pose, taken as the bin floor center)

- ``episode_0_qpos.txt``:
    one line per control step, six space-separated joint positions in
    the Squint joint order [shoulder_pan, shoulder_lift, elbow_flex,
    wrist_flex, wrist_roll, gripper].

The script forces the Isaac robot through that exact qpos trajectory using
``write_joint_state_to_sim`` (bypasses the controller — pure kinematic
replay) so we can visually confirm Isaac and ManiSkill agree on:

- joint axis directions (no sign flips)
- joint zero positions
- arm geometry (URDF→USD conversion is faithful)

The cube + bin are placed at the recorded world poses so the visual
context matches Squint's render of the same episode.

By default the trajectory is replayed 3 times back-to-back for clarity.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch  # noqa: F401  (import early to win over Isaac Kit DLLs)

parser = argparse.ArgumentParser(description="Replay a Squint qpos trajectory in Isaac")
parser.add_argument("--task", type=str, default="Isaac-SquintNative-Place-Replay-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--scene_file", type=str,
                    default=r"C:/Users/user/Downloads/episode_0_scene.txt")
parser.add_argument("--qpos_file", type=str,
                    default=r"C:/Users/user/Downloads/episode_0_qpos.txt")
parser.add_argument("--n_replays", type=int, default=3)
parser.add_argument("--step_dt_s", type=float, default=0.033,
                    help="Wall-clock pause between qpos frames so a human can see.")
parser.add_argument("--physics_substeps_per_frame", type=int, default=1,
                    help="How many sim.step() calls between writing successive qpos. "
                         "Set 0 to skip physics (snap-only) — fastest but no settle.")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# Force GUI so the user can see the replay.
args.headless = False
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import sim.eval2  # noqa: F401,E402

from sim.eval2.envs.squint_native.squint_events import (  # noqa: E402
    BIN_PART_LOCAL_OFFSETS,
)


# ----------------------------------------------------------------------
# File parsing
# ----------------------------------------------------------------------


def _parse_scene(path: Path) -> dict:
    """Parse the scene file. Returns a dict with robot_qpos / cube / bin."""
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        toks = line.split()
        key = toks[0]
        vals = [float(t) for t in toks[1:]]
        out[key] = vals
    return out


def _parse_qpos_trajectory(path: Path) -> list[list[float]]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        toks = line.split()
        if len(toks) != 6:
            raise ValueError(f"qpos line expected 6 cols, got {len(toks)}: {line!r}")
        rows.append([float(t) for t in toks])
    return rows


# ----------------------------------------------------------------------
# Pose application
# ----------------------------------------------------------------------


def _set_cube_pose(env, pose: list[float]) -> None:
    """pose = [x, y, z, qw, qx, qy, qz] (world)."""
    cube = env.scene["cube"]
    device = env.device
    p = torch.tensor(pose[:3], device=device, dtype=torch.float32).unsqueeze(0)
    q = torch.tensor(pose[3:], device=device, dtype=torch.float32).unsqueeze(0)
    # Add env_origin offset so it works for multi-env replays too.
    p = p + env.scene.env_origins
    full = torch.cat([p, q], dim=-1)
    vel = torch.zeros(1, 6, device=device)
    cube.write_root_pose_to_sim(full)
    cube.write_root_velocity_to_sim(vel)


def _set_bin_pose(env, pose: list[float]) -> None:
    """pose = [x, y, z (floor center), qw, qx, qy, qz] (world).

    Places the 5 bin parts with their local offsets, rotated by the bin's
    z-yaw so the whole bin moves as a rigid block.
    """
    device = env.device
    p_center = torch.tensor(pose[:3], device=device, dtype=torch.float32).unsqueeze(0)
    q = torch.tensor(pose[3:], device=device, dtype=torch.float32).unsqueeze(0)
    # The user's bin_pose z is the floor CENTER (z = BIN_THICKNESS/2 = 0.0025).
    # We need to convert to the "bin center" we use in squint_events
    # (z_center = 0; floor sits at +BIN_THICKNESS/2 above it).
    # bin center xy = floor center xy; z = floor z - thickness/2.
    p_anchor = p_center.clone()
    p_anchor[:, 2] = 0.0  # bin's reference frame sits on the table
    p_anchor = p_anchor + env.scene.env_origins

    # Extract yaw from quat.
    qw, qx, qy, qz = q[0]
    yaw = 2 * torch.atan2(qz, qw)
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)

    for part_name, (lx, ly, lz) in BIN_PART_LOCAL_OFFSETS.items():
        wx = cos_y * lx - sin_y * ly
        wy = sin_y * lx + cos_y * ly
        part_pos = p_anchor.clone()
        part_pos[:, 0] = part_pos[:, 0] + wx
        part_pos[:, 1] = part_pos[:, 1] + wy
        part_pos[:, 2] = part_pos[:, 2] + lz
        full = torch.cat([part_pos, q], dim=-1)
        vel = torch.zeros(1, 6, device=device)
        part = env.scene[part_name]
        part.write_root_pose_to_sim(full)
        part.write_root_velocity_to_sim(vel)


def _set_robot_qpos(env, qpos: list[float]) -> None:
    robot = env.scene["robot"]
    device = env.device
    qp = torch.tensor(qpos, device=device, dtype=torch.float32).unsqueeze(0)
    qv = torch.zeros_like(qp)
    robot.write_joint_state_to_sim(position=qp, velocity=qv)


def _force_target_to_qpos(env, qpos: list[float]) -> None:
    """Also drive the PD controller's target so the joint doesn't snap back."""
    try:
        action_term = env.action_manager.get_term("arm_and_gripper")
    except Exception:
        return
    device = env.device
    qp = torch.tensor(qpos, device=device, dtype=torch.float32).unsqueeze(0)
    # Reach into the term's _target directly. Bypasses normal action flow.
    if hasattr(action_term, "_target"):
        action_term._target[:] = qp


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> None:
    scene_path = Path(args.scene_file)
    qpos_path = Path(args.qpos_file)
    if not scene_path.exists():
        raise FileNotFoundError(scene_path)
    if not qpos_path.exists():
        raise FileNotFoundError(qpos_path)

    scene_data = _parse_scene(scene_path)
    trajectory = _parse_qpos_trajectory(qpos_path)
    n_steps = len(trajectory)

    print(f"[replay] Loaded scene from {scene_path.name}:")
    for k, v in scene_data.items():
        print(f"  {k:12s} = {v}")
    print(f"[replay] Trajectory has {n_steps} frames")

    env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
    try:
        env_cfg.recorders = None
    except Exception:
        pass
    env = gym.make(args.task, cfg=env_cfg)
    base_env = env.unwrapped

    obs, _ = env.reset(seed=0)
    sim = base_env.sim
    physics_dt = sim.get_physics_dt()

    cube_pose = scene_data["cube_pose"]
    bin_pose = scene_data["bin_pose"]

    print(f"\n[replay] Playing {args.n_replays} loops of {n_steps} frames each.")
    print(f"[replay] Inter-frame pause: {args.step_dt_s} s  (physics_substeps={args.physics_substeps_per_frame})")

    for loop in range(args.n_replays):
        # ---- 1. Reset cube + bin to the recorded poses ----
        _set_cube_pose(base_env, cube_pose)
        _set_bin_pose(base_env, bin_pose)
        # Initial qpos = first frame of trajectory.
        _set_robot_qpos(base_env, trajectory[0])
        _force_target_to_qpos(base_env, trajectory[0])
        # Settle a few sim steps so the renderer catches up.
        for _ in range(5):
            sim.step(render=True)

        print(f"\n[replay] === LOOP {loop + 1}/{args.n_replays} ===", flush=True)

        # ---- 2. Step through the trajectory ----
        robot = base_env.scene["robot"]
        cube = base_env.scene["cube"]
        body_names = list(robot.body_names)
        gi = body_names.index("gripper") if "gripper" in body_names else 0
        f1 = body_names.index("finger1_tip") if "finger1_tip" in body_names else gi
        f2 = body_names.index("finger2_tip") if "finger2_tip" in body_names else gi

        for i, qpos in enumerate(trajectory):
            _set_robot_qpos(base_env, qpos)
            _force_target_to_qpos(base_env, qpos)
            # A few physics sub-steps so the render reflects the new pose
            # AND the cube/bin physics doesn't drift away.
            for _ in range(max(1, args.physics_substeps_per_frame)):
                sim.step(render=True)
            # Wall-clock pause for human eyes.
            if args.step_dt_s > 0:
                time.sleep(args.step_dt_s)
            if i % 5 == 0 or i == n_steps - 1:
                g = robot.data.body_pos_w[0, gi].cpu().numpy()
                p1 = robot.data.body_pos_w[0, f1].cpu().numpy()
                p2 = robot.data.body_pos_w[0, f2].cpu().numpy()
                tcp = (p1 + p2) / 2.0
                c = cube.data.root_pos_w[0].cpu().numpy()
                d_g = ((g - c) ** 2).sum() ** 0.5
                d_tcp = ((tcp - c) ** 2).sum() ** 0.5
                print(
                    f"  L{loop+1} f{i:>3}: "
                    f"gripper=({g[0]:+.3f},{g[1]:+.3f},{g[2]:+.3f})  "
                    f"tcp=({tcp[0]:+.3f},{tcp[1]:+.3f},{tcp[2]:+.3f})  "
                    f"cube=({c[0]:+.3f},{c[1]:+.3f},{c[2]:+.3f})  "
                    f"d_gripper={d_g:.3f}  d_tcp={d_tcp:.3f}",
                    flush=True,
                )

        # Brief pause between loops
        if loop < args.n_replays - 1:
            time.sleep(0.8)

    print("\n[replay] Done. Close the Isaac Sim window to exit.")
    # Keep the viewer open so the user can inspect.
    while simulation_app.is_running():
        sim.step(render=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
