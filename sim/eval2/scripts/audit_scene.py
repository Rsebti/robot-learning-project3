"""Audit script — verify every assumed quantity in our V2.x env code.

Exhaustively probes the LeIsaac V285 PLAY scene to validate (or refute)
the position thresholds, body names, joint ranges, FrameTransformer
target indices, and reward magnitudes that were hardcoded into
``leisaac_lift_env_cfg.py``, ``mdp/rewards.py`` and ``mdp/terminations.py``
without empirical confirmation.

Probes performed (in this order):

  1. Robot body names + indices
  2. Joint names + per-joint ranges (esp. gripper)
  3. Cube spawn xyz distribution at reset (xy randomization sanity)
  4. Goal command distribution in robot root frame and world frame
  5. Reachability sanity: distance from default EE pose to goal range
  6. FrameTransformer ``ee_frame`` targets — names + positions at reset
  7. Distance gripper-target ↔ cube at reset (sanity for diff_threshold)
  8. Episode length and step rate (decimation × dt sanity)

Run from the Isaac venv. Use V285 RL PLAY task (state-only, 50 envs):

.. code-block:: powershell

    cd C:\\Users\\user\\Desktop\\MA2\\isaac\\isaac_so_arm101
    .\\.venv\\Scripts\\Activate.ps1
    python -m sim.eval2.scripts.audit_scene `
        --task Isaac-LeIsaac-SO101-Lift-RL-V285-Play-v0 `
        --num_envs 50 --num_resets 20 --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Audit assumed values in the LeIsaac scene.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-LeIsaac-SO101-Lift-RL-V285-Play-v0",
    help="Task name (defaults to V285 RL PLAY).",
)
parser.add_argument("--num_envs", type=int, default=50)
parser.add_argument("--num_resets", type=int, default=20)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import sim.eval2  # noqa: F401
import isaac_so_arm101.tasks  # noqa: F401
from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.math import combine_frame_transforms
from isaaclab_tasks.utils import parse_env_cfg


def _stat(name: str, values: torch.Tensor, indent: int = 4) -> None:
    flat = values.flatten()
    pad = " " * indent
    print(
        f"{pad}{name:30s}  "
        f"min={flat.min().item():+.4f}  "
        f"max={flat.max().item():+.4f}  "
        f"mean={flat.mean().item():+.4f}  "
        f"std={flat.std().item():.4f}"
    )


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped

    print(f"\n[INFO] task     : {args_cli.task}")
    print(f"[INFO] num_envs : {args_cli.num_envs}")
    print(f"[INFO] num_resets: {args_cli.num_resets}")

    # Reset once to populate fields.
    env.reset()

    cube: RigidObject = unwrapped.scene["cube"]
    robot: Articulation = unwrapped.scene["robot"]
    ee_frame = unwrapped.scene["ee_frame"]

    # =====================================================================
    # 1. ROBOT BODY NAMES
    # =====================================================================
    print("\n=== [1] Robot body names ===")
    print(f"    body_names: {robot.body_names}")
    for assumed in ("base", "gripper", "jaw"):
        try:
            ids = robot.find_bodies(assumed)[0]
            if len(ids) > 0:
                print(f"    ✓ '{assumed}' found at index {ids[0]}")
            else:
                print(f"    ✗ '{assumed}' NOT FOUND")
        except Exception as e:
            print(f"    ✗ '{assumed}' lookup error: {e}")

    # =====================================================================
    # 2. JOINT NAMES + RANGES
    # =====================================================================
    print("\n=== [2] Joint names + ranges ===")
    joint_names = robot.joint_names
    joint_limits = robot.data.joint_pos_limits[0]  # (num_joints, 2)
    print(f"    joint_names: {joint_names}")
    print(f"    {'joint':20s}  {'low':>8s}  {'high':>8s}  {'default':>8s}")
    default_joint_pos = robot.data.default_joint_pos[0]
    for i, n in enumerate(joint_names):
        low = joint_limits[i, 0].item()
        high = joint_limits[i, 1].item()
        default = default_joint_pos[i].item()
        print(f"    {n:20s}  {low:+.4f}  {high:+.4f}  {default:+.4f}")

    # Verify the assumed arm joint names exist
    print("\n    Assumed arm joint names from env_cfg.actions.arm_action:")
    assumed_arm = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    for j in assumed_arm:
        marker = "✓" if j in joint_names else "✗"
        print(f"      {marker} {j}")
    print("\n    Assumed gripper joint name from env_cfg.actions.gripper_action:")
    marker = "✓" if "gripper" in joint_names else "✗"
    print(f"      {marker} gripper")

    # Verify the assumed gripper open/close commands fit the range
    print("\n    Assumed gripper open=0.5, close=0.0:")
    if "gripper" in joint_names:
        g_idx = joint_names.index("gripper")
        g_low = joint_limits[g_idx, 0].item()
        g_high = joint_limits[g_idx, 1].item()
        print(f"      gripper range: [{g_low:+.4f}, {g_high:+.4f}]")
        for label, val in (("open=0.5", 0.5), ("close=0.0", 0.0)):
            ok = g_low <= val <= g_high
            marker = "✓" if ok else "✗"
            print(f"      {marker} {label} {'(in range)' if ok else '(OUT OF RANGE)'}")

    # =====================================================================
    # 3. CUBE SPAWN xyz DISTRIBUTION
    #
    # IMPORTANT: world-frame measurements are polluted by the env grid
    # spacing (50 envs spread over a 7×7 grid with 2.5 m spacing → world
    # std ≈ 5 m even with no per-env randomization). To measure the actual
    # randomization, subtract `env.scene.env_origins` (each env's origin
    # in world frame) and compare in env-local frame.
    # =====================================================================
    print(f"\n=== [3] Cube spawn xyz over {args_cli.num_resets} resets ===")
    env_origins = unwrapped.scene.env_origins  # (num_envs, 3), constant
    cube_xs_w, cube_ys_w, cube_zs_w = [], [], []
    cube_xs_e, cube_ys_e = [], []
    for _ in range(args_cli.num_resets):
        env.reset()
        pos_w = cube.data.root_pos_w.detach().clone()
        pos_e = pos_w - env_origins
        cube_xs_w.append(pos_w[:, 0])
        cube_ys_w.append(pos_w[:, 1])
        cube_zs_w.append(pos_w[:, 2])
        cube_xs_e.append(pos_e[:, 0])
        cube_ys_e.append(pos_e[:, 1])

    print("    World frame (polluted by env grid spacing — std ≈ env_spacing × n):")
    _stat("cube.x (world)", torch.cat(cube_xs_w))
    _stat("cube.y (world)", torch.cat(cube_ys_w))
    _stat("cube.z (world)", torch.cat(cube_zs_w))

    print("\n    Env-local frame (= cube_pos_world - env_origin):")
    _stat("cube.x (env-local)", torch.cat(cube_xs_e))
    _stat("cube.y (env-local)", torch.cat(cube_ys_e))
    print("    Expected randomization from EventsCfg.reset_cube_position:")
    print("      x noise ~ uniform clamped to (-0.05, +0.05) → std ≈ 0.029")
    print("      y noise ~ uniform clamped to (-0.10, +0.10) → std ≈ 0.058")
    print("      z noise = 0 (cube stays on table)")
    print("    Sanity: env-local std must be > 0; if = 0, the cube spawn")
    print("    randomization event isn't firing.")

    # =====================================================================
    # 4. GOAL COMMAND DISTRIBUTION (robot root frame + world frame)
    # =====================================================================
    print(f"\n=== [4] Goal command distribution over {args_cli.num_resets} resets ===")
    print("    Note: goal is generated in robot root frame; world frame")
    print("    depends on the robot's pose in the scene.")
    goal_b_xs, goal_b_ys, goal_b_zs = [], [], []
    goal_w_xs, goal_w_ys, goal_w_zs = [], [], []
    base_zs = []
    for _ in range(args_cli.num_resets):
        env.reset()
        cmd = unwrapped.command_manager.get_command("object_pose")  # (B, 7)
        goal_b = cmd[:, :3]
        goal_w, _ = combine_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w, goal_b
        )
        goal_b_xs.append(goal_b[:, 0].detach().clone())
        goal_b_ys.append(goal_b[:, 1].detach().clone())
        goal_b_zs.append(goal_b[:, 2].detach().clone())
        goal_w_xs.append(goal_w[:, 0].detach().clone())
        goal_w_ys.append(goal_w[:, 1].detach().clone())
        goal_w_zs.append(goal_w[:, 2].detach().clone())
        base_zs.append(robot.data.root_pos_w[:, 2].detach().clone())

    print("    Goal in ROBOT ROOT frame (what UniformPoseCommandCfg samples):")
    _stat("goal_x_b", torch.cat(goal_b_xs))
    _stat("goal_y_b", torch.cat(goal_b_ys))
    _stat("goal_z_b", torch.cat(goal_b_zs))
    print("    Expected: x ∈ (-0.05, +0.05), y ∈ (-0.20, -0.10), z ∈ (+0.10, +0.20)")

    print("\n    Goal in WORLD frame (what the reward distance compares against):")
    _stat("goal_x_w", torch.cat(goal_w_xs))
    _stat("goal_y_w", torch.cat(goal_w_ys))
    _stat("goal_z_w", torch.cat(goal_w_zs))
    _stat("robot.root.z (world)", torch.cat(base_zs))

    # =====================================================================
    # 5. REACHABILITY: EE → GOAL distance at reset
    # =====================================================================
    print("\n=== [5] EE → goal distance at reset (default joint pose) ===")
    # ee_frame.target_pos_w: (num_envs, num_targets, 3)
    target_names = ee_frame.cfg.target_frames if hasattr(ee_frame.cfg, "target_frames") else None
    print(f"    ee_frame target sources (cfg): {target_names}")
    print(f"    ee_frame.data.target_pos_w shape: {ee_frame.data.target_pos_w.shape}")

    env.reset()
    cmd = unwrapped.command_manager.get_command("object_pose")
    goal_b = cmd[:, :3]
    goal_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, goal_b
    )
    # If we have at least one target (gripper), measure ee→goal at reset.
    n_targets = ee_frame.data.target_pos_w.shape[1]
    if n_targets >= 1:
        ee_w_0 = ee_frame.data.target_pos_w[:, 0, :]  # gripper
        d_ee_goal = torch.norm(ee_w_0 - goal_w, dim=-1)
        _stat("|EE_target_0 → goal| (m)", d_ee_goal)
    if n_targets >= 2:
        ee_w_1 = ee_frame.data.target_pos_w[:, 1, :]  # jaw
        d_jaw_cube = torch.norm(ee_w_1 - cube.data.root_pos_w, dim=-1)
        _stat("|EE_target_1 → cube| (m)", d_jaw_cube)
        d_g_j = torch.norm(ee_w_0 - ee_w_1, dim=-1)
        _stat("|target_0 → target_1| (m)", d_g_j)

    # =====================================================================
    # 6. FrameTransformer ee_frame targets — name + position
    # =====================================================================
    print("\n=== [6] FrameTransformer 'ee_frame' targets at reset ===")
    print(f"    num targets : {n_targets}")
    print("    Target index 0 (assumed = 'gripper'):")
    if n_targets >= 1:
        p = ee_frame.data.target_pos_w[0, 0, :]
        print(f"      env 0 world xyz : ({p[0].item():+.4f}, {p[1].item():+.4f}, {p[2].item():+.4f})")
    print("    Target index 1 (assumed = 'jaw' with grasp offset):")
    if n_targets >= 2:
        p = ee_frame.data.target_pos_w[0, 1, :]
        print(f"      env 0 world xyz : ({p[0].item():+.4f}, {p[1].item():+.4f}, {p[2].item():+.4f})")

    # =====================================================================
    # 7. EPISODE LENGTH AND STEP RATE
    # =====================================================================
    print("\n=== [7] Episode timing ===")
    cfg = unwrapped.cfg
    print(f"    episode_length_s : {cfg.episode_length_s}")
    print(f"    decimation       : {cfg.decimation}")
    print(f"    sim.dt           : {cfg.sim.dt}")
    step_dt = cfg.sim.dt * cfg.decimation
    n_steps = int(cfg.episode_length_s / step_dt)
    ctrl_hz = 1.0 / step_dt
    print(f"    -> step dt       : {step_dt:.4f} s ({ctrl_hz:.1f} Hz control)")
    print(f"    -> steps/episode : {n_steps}")

    # =====================================================================
    # 8. REWARD AT RESET (cube on table, EE in default pose, gripper open)
    # =====================================================================
    print("\n=== [8] Per-term rewards at reset (zero action, single step) ===")
    # Take one zero-action step to populate reward buffer.
    actions = torch.zeros(env.action_space.shape, device=unwrapped.device)
    obs, reward, term, trunc, info = env.step(actions)
    rew_extras = info.get("log", info)
    if isinstance(rew_extras, dict):
        # Print each per-term reward under "Episode_Reward/*"
        for k, v in rew_extras.items():
            if "Reward/" in str(k) or "reward" in str(k).lower():
                try:
                    val = float(v) if not hasattr(v, "item") else v.item()
                    print(f"    {k}: {val:+.4f}")
                except Exception:
                    pass
    print(f"    total reward (env 0): {reward[0].item() if hasattr(reward, '__getitem__') else reward:+.4f}")
    print(f"    terminations (sum)   : term={int(term.sum().item())}, trunc={int(trunc.sum().item())}")

    # =====================================================================
    # 8b. CUBE → ROBOT BASE distance (reachability sanity)
    #
    # Env-local frame coords are misleading on their own because the
    # robot itself may be offset within its env (LeIsaac places the
    # robot at the edge of the table, not at env_origin). What matters
    # for reachability is `cube - robot_root` directly.
    # =====================================================================
    print("\n=== [8b] Cube ↔ robot reachability check ===")
    cube_dx_r, cube_dy_r, cube_dz_r, dist_r = [], [], [], []
    for _ in range(args_cli.num_resets):
        env.reset()
        cube_w = cube.data.root_pos_w.detach().clone()
        robot_w = robot.data.root_pos_w.detach().clone()
        diff = cube_w - robot_w
        cube_dx_r.append(diff[:, 0])
        cube_dy_r.append(diff[:, 1])
        cube_dz_r.append(diff[:, 2])
        dist_r.append(torch.norm(diff, dim=-1))

    print("    Cube position relative to robot root (world axes, robot root translated):")
    _stat("cube.x - robot.x", torch.cat(cube_dx_r))
    _stat("cube.y - robot.y", torch.cat(cube_dy_r))
    _stat("cube.z - robot.z", torch.cat(cube_dz_r))
    _stat("|cube - robot| (m)", torch.cat(dist_r))
    print("    Sanity: SO-101 max reach ≈ 0.30-0.35 m. Mean |cube - robot|")
    print("    should be < 0.30 for the cube to be reachable in default pose.")

    # =====================================================================
    # 9. LIFTING THRESHOLD vs SPAWN HEIGHT (the bug we already know about)
    # =====================================================================
    print("\n=== [9] Lifting threshold sanity ===")
    base_idx = robot.find_bodies("base")[0][0]
    base_z_w = robot.data.body_pos_w[:, base_idx, 2]
    cube_z_w = cube.data.root_pos_w[:, 2]
    diff = (cube_z_w - base_z_w)
    print(f"    cube.z - base.z at this reset: mean={diff.mean().item():+.4f}")
    print(f"    height_threshold currently used in rewards: 0.05")
    if diff.mean().item() > 0.05:
        excess = diff.mean().item() - 0.05
        print(f"    ✗  cube_lifted_above_base FIRES at spawn (excess +{excess*100:.1f} cm)")
        print(f"    Recommendation: bump threshold to {diff.mean().item() + 0.025:.2f}–0.10 m")
    else:
        print(f"    ✓  cube_lifted_above_base does not fire at spawn")

    print("\n=== AUDIT COMPLETE ===\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
