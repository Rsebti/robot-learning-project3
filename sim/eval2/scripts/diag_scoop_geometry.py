"""Diagnostic D — verify the SO-101 URDF geometric assumption behind
``scoop_grasp_penalty``.

Claim under test:
    ``scoop_grasp_penalty = clamp(ee_z - wrist_z, min=0)``
fires only when the gripper points UP / scoops horizontally. It should
return 0 when the gripper points straight DOWN (correct top-down pose).

This script samples body positions over many resets AND across a small
random action rollout, and reports the distribution of:
    - ee_z, wrist_z, gripper_z, jaw_z (world frame)
    - dz = ee_z - wrist_z (the scoop penalty argument)
    - clamp(dz, min=0) ≈ scoop penalty per step
    - gripper_down_score = 1 + z_local·world_up  (the orientation reward arg)

If we observe ``dz > 0`` (penalty fires) WHILE ``gripper_down_score > 1.5``
(gripper points down), the penalty is geometrically broken on SO-101.

Usage:
    python sim/eval2/scripts/diag_scoop_geometry.py --headless --enable_cameras
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str,
                    default="Isaac-LeIsaac-SO101-Lift-Visual-V213-Play-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_steps", type=int, default=80,
                    help="random-action rollout length per env")
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
from isaaclab.assets import Articulation
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import quat_apply


def collect_geometry(env, n_steps: int):
    """Run env.step() n_steps with random actions, gather geometry per step."""
    unwrapped = env.unwrapped
    robot: Articulation = unwrapped.scene["robot"]
    ee_frame: FrameTransformer = unwrapped.scene["ee_frame"]

    # Locate 'wrist' body index — matches scoop_grasp_penalty config.
    wrist_ids, wrist_names = robot.find_bodies("wrist")
    assert len(wrist_ids) == 1, f"expected 1 wrist body, got {wrist_names}"
    wrist_id = wrist_ids[0]
    print(f"[INFO] Using wrist body index {wrist_id} = '{wrist_names[0]}'")

    num_envs = unwrapped.num_envs
    action_dim = unwrapped.action_manager.total_action_dim
    device = unwrapped.device

    samples = []
    for step in range(n_steps):
        # small random delta actions
        action = (torch.rand((num_envs, action_dim), device=device) * 2 - 1) * 0.5
        env.step(action)

        wrist_z = robot.data.body_pos_w[:, wrist_id, 2]            # (N_env,)
        wrist_xyz = robot.data.body_pos_w[:, wrist_id]             # (N_env, 3)
        ee_xyz = ee_frame.data.target_pos_w[..., 0, :]             # (N_env, 3) — palm
        jaw_xyz = ee_frame.data.target_pos_w[..., 1, :]            # (N_env, 3) — jaw
        ee_quat = ee_frame.data.target_quat_w[..., 0, :]           # (N_env, 4)

        # gripper down score: rotate local z-axis to world, take world-z component
        z_local = torch.zeros((num_envs, 3), device=device)
        z_local[..., 2] = 1.0
        z_world = quat_apply(ee_quat, z_local)
        gripper_down_score = 1.0 + z_world[..., 2]                 # 2=down, 0=up

        dz = ee_xyz[..., 2] - wrist_z
        scoop_pen = torch.clamp(dz, min=0.0)

        for i in range(num_envs):
            samples.append({
                "step": step,
                "env": i,
                "wrist_x": wrist_xyz[i, 0].item(),
                "wrist_y": wrist_xyz[i, 1].item(),
                "wrist_z": wrist_z[i].item(),
                "ee_x": ee_xyz[i, 0].item(),
                "ee_y": ee_xyz[i, 1].item(),
                "ee_z": ee_xyz[i, 2].item(),
                "jaw_z": jaw_xyz[i, 2].item(),
                "dz_ee_minus_wrist": dz[i].item(),
                "scoop_penalty": scoop_pen[i].item(),
                "gripper_down_score": gripper_down_score[i].item(),
            })

    return samples


def quantiles(values, qs=(0.05, 0.25, 0.50, 0.75, 0.95)):
    import numpy as np
    arr = np.array(values)
    return {q: float(np.quantile(arr, q)) for q in qs}


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    unwrapped = env.unwrapped
    num_envs = unwrapped.num_envs
    print(f"[INFO] num_envs = {num_envs}, num_steps = {args_cli.num_steps}")

    # --- Reset-only snapshot (initial pose, no actions yet) ---
    robot: Articulation = unwrapped.scene["robot"]
    ee_frame: FrameTransformer = unwrapped.scene["ee_frame"]
    wrist_ids, wrist_names = robot.find_bodies("wrist")
    wrist_id = wrist_ids[0]

    print()
    print("=" * 70)
    print("RESET POSE — single env, all bodies (world frame)")
    print("=" * 70)
    for i, name in enumerate(robot.body_names):
        pos = robot.data.body_pos_w[0, i].cpu().numpy()
        print(f"  [{i:2d}] {name:<30s} x={pos[0]:+.4f}  y={pos[1]:+.4f}  z={pos[2]:+.4f}")

    print()
    print("EE_FRAME targets at reset (env 0):")
    for ti in range(ee_frame.data.target_pos_w.shape[1]):
        pos = ee_frame.data.target_pos_w[0, ti].cpu().numpy()
        quat = ee_frame.data.target_quat_w[0, ti].cpu().numpy()
        print(f"  target[{ti}] pos = ({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f})  quat=({quat[0]:+.3f}, {quat[1]:+.3f}, {quat[2]:+.3f}, {quat[3]:+.3f})")

    # Key diagnostic at reset
    print()
    print("SCOOP-PENALTY GEOMETRY AT RESET POSE (across envs):")
    for env_i in range(num_envs):
        wz = robot.data.body_pos_w[env_i, wrist_id, 2].item()
        ez = ee_frame.data.target_pos_w[env_i, 0, 2].item()
        jz = ee_frame.data.target_pos_w[env_i, 1, 2].item()
        dz = ez - wz
        print(f"  env {env_i}: wrist_z={wz:+.4f}  ee_z={ez:+.4f}  jaw_z={jz:+.4f}  dz=ee-wrist={dz:+.4f}  →  scoop_pen=clamp(dz,0)={max(0.0, dz):.4f}")

    # --- Rollout with random actions ---
    print()
    print("=" * 70)
    print(f"ROLLOUT — {args_cli.num_steps} random-delta steps per env")
    print("=" * 70)
    samples = collect_geometry(env, args_cli.num_steps)
    n = len(samples)
    print(f"[INFO] collected {n} sample-step rows ({num_envs} envs × {args_cli.num_steps} steps)")

    # Aggregate
    print()
    print("DISTRIBUTION across all rollout samples:")
    metrics = ["wrist_z", "ee_z", "jaw_z", "dz_ee_minus_wrist",
               "scoop_penalty", "gripper_down_score"]
    for m in metrics:
        vals = [s[m] for s in samples]
        q = quantiles(vals)
        print(f"  {m:<22s}  p05={q[0.05]:+.4f}  p25={q[0.25]:+.4f}  p50={q[0.50]:+.4f}  p75={q[0.75]:+.4f}  p95={q[0.95]:+.4f}")

    # Cross-correlation: when gripper_down_score is HIGH (>1.5 → mostly down),
    # is scoop_penalty ZERO (good) or POSITIVE (BUG)?
    print()
    print("CORRELATION CHECK — scoop_penalty conditional on gripper orientation:")
    print(f"  {'orientation bucket':<35s} {'count':>8s} {'mean scoop_pen':>15s} {'%scoop>0':>10s}")
    buckets = [
        ("gripper UP        (score < 0.5)", 0.0, 0.5),
        ("gripper horizontal(0.5-1.5)    ", 0.5, 1.5),
        ("gripper DOWN      (score > 1.5)", 1.5, 2.1),
    ]
    for label, lo, hi in buckets:
        in_bucket = [s for s in samples if lo <= s["gripper_down_score"] < hi]
        count = len(in_bucket)
        if count == 0:
            print(f"  {label:<35s} {count:>8d} {'--':>15s} {'--':>10s}")
            continue
        mean_pen = sum(s["scoop_penalty"] for s in in_bucket) / count
        pct_fire = 100.0 * sum(1 for s in in_bucket if s["scoop_penalty"] > 0) / count
        print(f"  {label:<35s} {count:>8d} {mean_pen:>15.4f} {pct_fire:>9.1f}%")

    # Verdict
    print()
    print("=" * 70)
    down_samples = [s for s in samples if s["gripper_down_score"] >= 1.5]
    if len(down_samples) == 0:
        print("VERDICT: insufficient gripper-down samples to judge.")
    else:
        pct_bug = 100.0 * sum(1 for s in down_samples if s["scoop_penalty"] > 0.005) / len(down_samples)
        if pct_bug > 30:
            print(f"VERDICT: GEOMETRIC BUG CONFIRMED — {pct_bug:.1f}% of gripper-down samples")
            print(f"         fire the scoop_penalty (dz > 0.005m). The 'wrist' body sits BELOW")
            print(f"         the palm in correct top-down pose, so the penalty punishes the")
            print(f"         right posture. ACTION: drop scoop_grasp_penalty (weight=0) in V2.13 v3.")
        else:
            print(f"VERDICT: NO BUG — only {pct_bug:.1f}% of gripper-down samples fire the")
            print(f"         penalty. Scoop_grasp_penalty is correctly punishing scoop posture.")
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
