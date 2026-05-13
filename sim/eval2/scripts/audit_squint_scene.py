"""Audit Squint env scene — print all relevant world positions.

Resets the env, lets physics settle for a few steps so the cube lands
on the table, then prints:
- Env origin in world frame
- Robot base position (world + env-local)
- Cube initial vs settled position (settled - half_size = table top z)
- Bin floor / wall positions
- Wrist camera position
- Inferred table top z (the value to use for bin placement)

Launch
------
.. code-block:: powershell

    cd C:\\Users\\user\\Desktop\\MA2\\isaac\\isaac_so_arm101
    C:/Users/user/Desktop/MA2/isaac/isaac_so_arm101/.venv/Scripts/python.exe `
        -m sim.eval2.scripts.audit_squint_scene `
        --task Isaac-SO101-Squint-Place-Play-v0 --num_envs 1
"""
from __future__ import annotations

import argparse

parser = argparse.ArgumentParser(description="Audit Squint env scene")
parser.add_argument("--task", type=str, default="Isaac-SO101-Squint-Place-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--settle_steps", type=int, default=60, help="Physics steps to settle the cube")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True  # No GUI needed for audit; faster

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import sim.eval2  # noqa: F401,E402


def _print_pos(name: str, pos_w, env_origin=None) -> None:
    """Print a position in world + (optionally) env-local frame."""
    p = pos_w.detach().cpu().numpy().ravel()
    if env_origin is not None:
        eo = env_origin.detach().cpu().numpy().ravel()
        p_local = p - eo
        print(f"  {name:30s} world=({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})  "
              f"env_local=({p_local[0]:+.4f}, {p_local[1]:+.4f}, {p_local[2]:+.4f})")
    else:
        print(f"  {name:30s} ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    base_env = env.unwrapped

    # Reset and settle
    obs, _ = env.reset(seed=0)
    n_act = base_env.action_space.shape[-1]
    zero_action = torch.zeros(args.num_envs, n_act, device=device)
    print(f"[audit] Settling physics for {args.settle_steps} steps...")
    for _ in range(args.settle_steps):
        env.step(zero_action)

    # Probe scene
    scene = base_env.scene
    env_origin = scene.env_origins[0]  # first env

    print("\n" + "=" * 78)
    print(f"SCENE AUDIT — task={args.task}")
    print("=" * 78)
    print(f"\n[env_origins]")
    _print_pos("env[0] origin", env_origin)
    if args.num_envs > 1:
        _print_pos("env[1] origin", scene.env_origins[1])

    # Robot articulation
    print(f"\n[articulations]")
    for name in scene.articulations:
        art = scene[name]
        root_pos = art.data.root_pos_w[0]
        _print_pos(f"{name}.root_pos", root_pos, env_origin)
        # Body 'base' (robot base link)
        if hasattr(art.data, "body_pos_w"):
            body_names = art.data.body_names
            for body_idx, body_name in enumerate(body_names):
                pos = art.data.body_pos_w[0, body_idx]
                _print_pos(f"  body[{body_name}]", pos, env_origin)

    # Rigid objects
    print(f"\n[rigid_objects]")
    for name in scene.rigid_objects:
        obj = scene[name]
        pos = obj.data.root_pos_w[0]
        _print_pos(name, pos, env_origin)

    # Sensors (cameras)
    print(f"\n[sensors]")
    for name in scene.sensors:
        sensor = scene[name]
        if hasattr(sensor.data, "pos_w"):
            pos = sensor.data.pos_w[0]
            _print_pos(name, pos, env_origin)
            # Also print rotation + compute the ground projection of the
            # camera optical axis onto the table top (for placing objects
            # in the camera's center view).
            attrs = [a for a in dir(sensor.data) if "quat" in a.lower()]
            print(f"  {name:30s} quat-like attrs on sensor.data = {attrs}")
            quat = None
            for cand in ["quat_w_world", "quat_w_ros", "quat_w_opengl", "quat_w"]:
                if hasattr(sensor.data, cand):
                    quat = getattr(sensor.data, cand)[0]
                    print(f"  {name:30s} using {cand}")
                    break
            if quat is None:
                continue
            qw, qx, qy, qz = quat.detach().cpu().numpy().ravel()
            print(f"  {name:30s} quat (w,x,y,z) = ({qw:+.4f}, {qx:+.4f}, {qy:+.4f}, {qz:+.4f})")
            # Try all 6 possible forward directions to find the one that
            # points down (-Z heavy). The optical axis depends on Isaac
            # Lab's `convention` setting, which is ambiguous between
            # versions.
            for axis_name, axis in [("+X", (1, 0, 0)), ("-X", (-1, 0, 0)),
                                     ("+Y", (0, 1, 0)), ("-Y", (0, -1, 0)),
                                     ("+Z", (0, 0, 1)), ("-Z", (0, 0, -1))]:
                ax, ay, az = axis
                R = (
                    (1 - 2 * (qy*qy + qz*qz)) * ax + 2 * (qx*qy - qw*qz) * ay + 2 * (qx*qz + qw*qy) * az,
                    2 * (qx*qy + qw*qz) * ax + (1 - 2 * (qx*qx + qz*qz)) * ay + 2 * (qy*qz - qw*qx) * az,
                    2 * (qx*qz - qw*qy) * ax + 2 * (qy*qz + qw*qx) * ay + (1 - 2 * (qx*qx + qy*qy)) * az,
                )
                print(f"  {name:30s}   local {axis_name} → world ({R[0]:+.3f}, {R[1]:+.3f}, {R[2]:+.3f})")
            # Isaac Lab `convention="world"`: optical axis = local +X.
            fwd_x = 1 - 2 * (qy * qy + qz * qz)
            fwd_y = 2 * (qx * qy + qw * qz)
            fwd_z = 2 * (qx * qz - qw * qy)
            print(f"  {name:30s} cam fwd (world) = "
                  f"({fwd_x:+.3f}, {fwd_y:+.3f}, {fwd_z:+.3f})")
            # Ray onto table plane (z=0.036 env-local). Convert cam pos to
            # env_local first.
            p = (pos - env_origin).detach().cpu().numpy().ravel()
            table_z_local = 0.036
            if fwd_z < -1e-6:
                t = (table_z_local - p[2]) / fwd_z
                hit_x_local = p[0] + t * fwd_x
                hit_y_local = p[1] + t * fwd_y
                hit_x_world = hit_x_local + env_origin.detach().cpu().numpy().ravel()[0]
                hit_y_world = hit_y_local + env_origin.detach().cpu().numpy().ravel()[1]
                print(f"  {name:30s} ground projection (table z={table_z_local}):")
                print(f"        env_local = ({hit_x_local:+.3f}, {hit_y_local:+.3f})")
                print(f"        WORLD     = ({hit_x_world:+.3f}, {hit_y_world:+.3f})")
                print(f"        ➜ Use this as cube/bin xy anchor.")
        elif hasattr(sensor.data, "target_pos_w"):
            for i in range(sensor.data.target_pos_w.shape[1]):
                _print_pos(f"{name}.target[{i}]", sensor.data.target_pos_w[0, i], env_origin)
        else:
            print(f"  {name:30s} (no .pos_w or .target_pos_w attr)")

    # Inferred table top from settled cube position
    print("\n[inferred geometry]")
    if "cube" in scene.rigid_objects:
        cube = scene["cube"]
        cube_pos = cube.data.root_pos_w[0]
        cube_local_z = (cube_pos - env_origin)[2].item()
        # Cube half-size unknown without USD inspection; assume 0.015 m
        # (LeIsaac default ~3 cm cube). Tell user the assumption.
        cube_half_size = 0.015
        inferred_table_top = cube_local_z - cube_half_size
        print(f"  cube settled z (env_local) = {cube_local_z:.4f} m")
        print(f"  assuming cube_half_size = {cube_half_size:.3f} m")
        print(f"  → INFERRED TABLE TOP z (env_local) = {inferred_table_top:.4f} m")
        print(f"\n  ➜ Use this as `table_top_z` in squint_envs.py SquintPlaceEnvCfg.")
        print(f"     Bin floor center z should be: {inferred_table_top + 0.0025:.4f}")
        print(f"     Bin walls center z should be: {inferred_table_top + 0.005 + 0.01:.4f}")
        print(f"     Goal pose z range should be:  ({inferred_table_top + 0.005 + cube_half_size:.4f}, ...)")

    print("\n" + "=" * 78)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
