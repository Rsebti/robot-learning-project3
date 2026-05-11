"""Dump EVERYTHING about the scene's frames, bodies, joints, and observations.

Goal : single source of truth on coordinate conventions so reward functions
that depend on geometry (gripper orientation, jaw-palm-wrist relations,
object-base distances) can be written without sign/axis ambiguity.

For each frame/body, we project its LOCAL +X / +Y / +Z axes to world frame.
That tells us exactly which local axis is "out of the palm", "side of the
gripper", etc. — no more guessing from quaternion math.

Usage :
    python sim/eval2/scripts/dump_scene_frames.py --headless --enable_cameras

Optional :
    --task <gym-id>     pick a different task (default V213 Play)
    --num_resets 5      number of resets to measure spawn variance
"""
from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str,
                    default="Isaac-LeIsaac-SO101-Lift-Visual-V213-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_resets", type=int, default=5,
                    help="how many resets to measure spawn variance")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import sim.eval2  # noqa: F401
import isaac_so_arm101.tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import quat_apply


def axes_in_world(quat_w):
    """Given a (N, 4) quat in world frame, return the 3 local axes (each Nx3)
    projected into world frame: (x_w, y_w, z_w)."""
    device = quat_w.device
    n = quat_w.shape[0]
    x_loc = torch.zeros((n, 3), device=device); x_loc[..., 0] = 1.0
    y_loc = torch.zeros((n, 3), device=device); y_loc[..., 1] = 1.0
    z_loc = torch.zeros((n, 3), device=device); z_loc[..., 2] = 1.0
    return (quat_apply(quat_w, x_loc),
            quat_apply(quat_w, y_loc),
            quat_apply(quat_w, z_loc))


def axis_to_world_label(axis_w):
    """Return a short label like '+world_z' or '-world_y' for a 3D vector,
    rounded to nearest world cardinal direction if dominant component > 0.85.
    """
    ax, ay, az = axis_w[0].item(), axis_w[1].item(), axis_w[2].item()
    mags = [abs(ax), abs(ay), abs(az)]
    dom = max(range(3), key=lambda i: mags[i])
    if mags[dom] < 0.85:
        return f"({ax:+.2f}, {ay:+.2f}, {az:+.2f})  [mixed]"
    sign = "+" if [ax, ay, az][dom] > 0 else "-"
    label = ["x", "y", "z"][dom]
    return f"({ax:+.2f}, {ay:+.2f}, {az:+.2f})  [{sign}world_{label}]"


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    u = env.unwrapped

    sep = "=" * 78
    print()
    print(sep); print(f"SCENE DUMP — task = {args_cli.task}"); print(sep)

    # ---------- Scene entities ----------
    print()
    print("--- SCENE ENTITIES (env.scene keys) ---")
    try:
        for k, v in u.scene._articulations.items():
            print(f"  Articulation: {k:<15s} type={type(v).__name__}")
        for k, v in u.scene._rigid_objects.items():
            print(f"  RigidObject : {k:<15s} type={type(v).__name__}")
        for k, v in u.scene._sensors.items():
            print(f"  Sensor      : {k:<15s} type={type(v).__name__}")
    except Exception as e:
        print(f"  [WARN] private attr access failed ({e}); falling back to keys():")
        try:
            for k in u.scene.keys():
                print(f"  scene['{k}']  type={type(u.scene[k]).__name__}")
        except Exception as e2:
            print(f"  [WARN] keys() also failed: {e2}")

    robot: Articulation = u.scene["robot"]
    cube: RigidObject = u.scene["cube"]
    ee_frame: FrameTransformer = u.scene["ee_frame"]

    # ---------- Robot bodies ----------
    print()
    print("--- ROBOT BODIES (world frame, env 0, reset pose) ---")
    print(f"  {'idx':>3s} {'name':<20s} {'pos_x':>8s} {'pos_y':>8s} {'pos_z':>8s}"
          f"  {'local +x in world':<30s} {'local +y in world':<30s} {'local +z in world':<30s}")
    for i, name in enumerate(robot.body_names):
        pos = robot.data.body_pos_w[0, i]
        quat = robot.data.body_quat_w[0, i:i + 1]   # keep (1,4)
        xw, yw, zw = axes_in_world(quat)
        print(f"  {i:>3d} {name:<20s} {pos[0]:>8.4f} {pos[1]:>8.4f} {pos[2]:>8.4f}"
              f"  {axis_to_world_label(xw[0]):<30s} {axis_to_world_label(yw[0]):<30s} {axis_to_world_label(zw[0]):<30s}")

    # ---------- Robot joints ----------
    print()
    print("--- ROBOT JOINTS (current state + limits) ---")
    print(f"  {'idx':>3s} {'name':<20s} {'pos':>8s} {'lo':>8s} {'hi':>8s} {'range':>8s} {'is_default':>11s}")
    default_pos = robot.data.default_joint_pos[0]
    cur_pos = robot.data.joint_pos[0]
    limits = robot.data.soft_joint_pos_limits[0]
    for i, name in enumerate(robot.joint_names):
        lo, hi = limits[i, 0].item(), limits[i, 1].item()
        cur, df = cur_pos[i].item(), default_pos[i].item()
        is_def = "YES" if abs(cur - df) < 1e-4 else f"diff={cur-df:+.3f}"
        print(f"  {i:>3d} {name:<20s} {cur:>+8.3f} {lo:>+8.3f} {hi:>+8.3f} {hi - lo:>8.3f} {is_def:>11s}")

    # ---------- ee_frame (FrameTransformer) ----------
    print()
    print("--- EE_FRAME (FrameTransformer) — config ---")
    try:
        cfg = ee_frame.cfg
        print(f"  cfg type      : {type(cfg).__name__}")
        print(f"  prim_path     : {getattr(cfg, 'prim_path', 'N/A')}")
        target_frames = getattr(cfg, "target_frames", None)
        if target_frames is not None:
            print(f"  num targets   : {len(target_frames)}")
            for i, tf in enumerate(target_frames):
                print(f"  target[{i}] = {tf}")
        else:
            print(f"  (no .target_frames attr — full cfg repr:)")
            print(f"  {cfg}")
    except Exception as e:
        print(f"  [WARN] ee_frame.cfg dump failed: {type(e).__name__}: {e}")

    print()
    print("--- EE_FRAME — live targets (reset pose, env 0) ---")
    try:
        n_targets = ee_frame.data.target_pos_w.shape[1]
        target_names = getattr(ee_frame.data, "target_frame_names", None)
        print(f"  {'idx':>3s} {'name':<10s} {'pos_x':>8s} {'pos_y':>8s} {'pos_z':>8s}"
              f"  {'local +x in world':<30s} {'local +y in world':<30s} {'local +z in world':<30s}")
        for i in range(n_targets):
            pos = ee_frame.data.target_pos_w[0, i]
            quat = ee_frame.data.target_quat_w[0, i:i + 1]
            xw, yw, zw = axes_in_world(quat)
            name = target_names[i] if (target_names and i < len(target_names)) else f"t{i}"
            print(f"  {i:>3d} {name:<10s} {pos[0]:>8.4f} {pos[1]:>8.4f} {pos[2]:>8.4f}"
                  f"  {axis_to_world_label(xw[0]):<30s} {axis_to_world_label(yw[0]):<30s} {axis_to_world_label(zw[0]):<30s}")
    except Exception as e:
        print(f"  [WARN] ee_frame live data dump failed: {type(e).__name__}: {e}")

    # ---------- Cube state ----------
    print()
    print("--- CUBE (RigidObject) ---")
    cpos = cube.data.root_pos_w[0]
    cquat = cube.data.root_quat_w[0:1]
    cxw, cyw, czw = axes_in_world(cquat)
    print(f"  prim_path : {cube.cfg.prim_path}")
    print(f"  root_pos_w     = ({cpos[0]:.4f}, {cpos[1]:.4f}, {cpos[2]:.4f})")
    print(f"  root_quat_w    = ({cquat[0,0]:.3f}, {cquat[0,1]:.3f}, {cquat[0,2]:.3f}, {cquat[0,3]:.3f})  [w,x,y,z]")
    print(f"  local +x in world: {axis_to_world_label(cxw[0])}")
    print(f"  local +y in world: {axis_to_world_label(cyw[0])}")
    print(f"  local +z in world: {axis_to_world_label(czw[0])}")

    # ---------- Command state (goal pose) ----------
    print()
    print("--- COMMAND MANAGER ---")
    try:
        cm = u.command_manager
        print(repr(cm))
        cmd = cm.get_command("object_pose")
        print(f"  object_pose shape : {tuple(cmd.shape)}")
        print(f"  object_pose env 0 : pos=({cmd[0, 0]:.4f}, {cmd[0, 1]:.4f}, {cmd[0, 2]:.4f})  quat=({cmd[0, 3]:.3f}, {cmd[0, 4]:.3f}, {cmd[0, 5]:.3f}, {cmd[0, 6]:.3f})  [robot-base frame]")
    except Exception as e:
        print(f"  [WARN] command manager dump failed: {type(e).__name__}: {e}")

    # ---------- Useful pairwise relations ----------
    print()
    print("--- PAIRWISE RELATIONS at reset (env 0) ---")
    base_z = robot.data.body_pos_w[0, 0, 2].item()
    print(f"  base body z (world) : {base_z:+.4f}")
    palm_z = ee_frame.data.target_pos_w[0, 0, 2].item()
    jaw_z = ee_frame.data.target_pos_w[0, 1, 2].item()
    print(f"  palm (target[0]) z (world) : {palm_z:+.4f}")
    print(f"  jaw  (target[1]) z (world) : {jaw_z:+.4f}")
    print(f"  palm - jaw   (>0 means palm above jaw, =gripper down at rest)   : {palm_z - jaw_z:+.4f}")

    # Wrist body z
    wrist_ids, _ = robot.find_bodies("wrist")
    if len(wrist_ids) == 1:
        wrist_z = robot.data.body_pos_w[0, wrist_ids[0], 2].item()
        print(f"  wrist body z (world) : {wrist_z:+.4f}")
        print(f"  palm - wrist (>0 means palm above wrist body)              : {palm_z - wrist_z:+.4f}")

    cube_z = cube.data.root_pos_w[0, 2].item()
    print(f"  cube root z (world) : {cube_z:+.4f}")
    print(f"  palm - cube  (vertical distance EE-cube)                       : {palm_z - cube_z:+.4f}")

    # ---------- Reset variance ----------
    print()
    print(f"--- SPAWN VARIANCE over {args_cli.num_resets} resets (env 0) ---")
    cube_xyz = []; goal_xyz = []
    for k in range(args_cli.num_resets):
        env.reset()
        cube_xyz.append(cube.data.root_pos_w[0].clone())
        try:
            cmd = cm.get_command("object_pose")
            goal_xyz.append(cmd[0, :3].clone())
        except Exception:
            pass
    cube_xyz = torch.stack(cube_xyz)
    print(f"  cube root_pos_w :")
    print(f"    x : min={cube_xyz[:, 0].min():+.4f}  max={cube_xyz[:, 0].max():+.4f}  std={cube_xyz[:, 0].std():.4f}")
    print(f"    y : min={cube_xyz[:, 1].min():+.4f}  max={cube_xyz[:, 1].max():+.4f}  std={cube_xyz[:, 1].std():.4f}")
    print(f"    z : min={cube_xyz[:, 2].min():+.4f}  max={cube_xyz[:, 2].max():+.4f}  std={cube_xyz[:, 2].std():.4f}")
    if goal_xyz:
        goal_xyz = torch.stack(goal_xyz)
        print(f"  goal command (robot-base frame) :")
        print(f"    x : min={goal_xyz[:, 0].min():+.4f}  max={goal_xyz[:, 0].max():+.4f}  std={goal_xyz[:, 0].std():.4f}")
        print(f"    y : min={goal_xyz[:, 1].min():+.4f}  max={goal_xyz[:, 1].max():+.4f}  std={goal_xyz[:, 1].std():.4f}")
        print(f"    z : min={goal_xyz[:, 2].min():+.4f}  max={goal_xyz[:, 2].max():+.4f}  std={goal_xyz[:, 2].std():.4f}")

    # ---------- Observation manager ----------
    print()
    print("--- OBSERVATION MANAGER (policy group) ---")
    try:
        om = u.observation_manager
        # Repr fallback — covers API variants across Isaac Lab versions.
        print(repr(om))
    except Exception as e:
        print(f"  [WARN] obs manager dump failed: {type(e).__name__}: {e}")

    # ---------- Action manager ----------
    print()
    print("--- ACTION MANAGER ---")
    try:
        am = u.action_manager
        print(repr(am))
        print(f"  total action dim : {am.total_action_dim}")
        # Try to introspect term configs through env_cfg (always works).
        actions_cfg = env_cfg.actions
        for attr_name in dir(actions_cfg):
            if attr_name.startswith("_"):
                continue
            val = getattr(actions_cfg, attr_name, None)
            if val is None or callable(val):
                continue
            print(f"  cfg.actions.{attr_name} : {type(val).__name__}")
            for sub in ["asset_name", "joint_names", "scale", "use_zero_offset",
                        "use_default_offset", "clip", "open_command_expr",
                        "close_command_expr"]:
                if hasattr(val, sub):
                    print(f"    .{sub} = {getattr(val, sub)}")
    except Exception as e:
        print(f"  [WARN] action manager dump failed: {type(e).__name__}: {e}")

    # ---------- Termination manager ----------
    print()
    print("--- TERMINATION MANAGER ---")
    try:
        tm = u.termination_manager
        print(repr(tm))
    except Exception as e:
        print(f"  [WARN] termination manager dump failed: {type(e).__name__}: {e}")

    # ---------- Reward manager ----------
    print()
    print("--- REWARD MANAGER ---")
    try:
        rm = u.reward_manager
        print(repr(rm))
    except Exception as e:
        print(f"  [WARN] reward manager dump failed: {type(e).__name__}: {e}")

    print()
    print(sep); print("DUMP COMPLETE"); print(sep)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
