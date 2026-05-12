"""Verify V2.13 v2 implementation: resolve gym task, instantiate env_cfg,
check final reward weights and termination state.

Launch with:
    python sim/eval2/scripts/verify_v213_v2.py --headless --enable_cameras
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now safe to import Isaac Lab + env cfgs.
import gymnasium as gym
import sim.eval2  # registers gym ids

from isaaclab_tasks.utils import parse_env_cfg

print("=" * 70)
print("V2.13 v2 IMPLEMENTATION VERIFICATION")
print("=" * 70)

for task_id in [
    "Isaac-LeIsaac-SO101-Lift-RL-V213-v0",
    "Isaac-LeIsaac-SO101-Lift-Visual-V213-v0",
]:
    print(f"\n[{task_id}]")
    spec = gym.spec(task_id)
    print(f"  env_cfg entry  : {spec.kwargs['env_cfg_entry_point']}")
    print(f"  rsl_rl entry   : {spec.kwargs['rsl_rl_cfg_entry_point']}")

    env_cfg = parse_env_cfg(task_id, num_envs=4)

    # action class + clip
    arm = env_cfg.actions.arm_action
    print(f"  action class   : {type(arm).__name__}")
    print(f"  action scale   : {arm.scale}")
    clip = getattr(arm, "clip", None)
    print(f"  action clip    : {clip}")

    # reward weights
    print("  reward weights:")
    for name in [
        "reaching_object",
        "gripper_orientation_penalty",
        "scoop_grasp_penalty",
        "cube_dropped_penalty",
        "success_bonus",
        "lifting_object",
        "object_goal_tracking",
        "object_goal_tracking_fine_grained",
        "ee_to_cube_distance",
    ]:
        term = getattr(env_cfg.rewards, name, None)
        weight = "MISSING" if term is None else term.weight
        print(f"    {name:42s} = {weight}")

    # terminations
    print("  terminations:")
    for tname in ["time_out", "cube_reached_goal", "cube_dropped",
                  "ee_far_from_cube"]:
        term = getattr(env_cfg.terminations, tname, None)
        if term is None:
            print(f"    {tname:25s} = None (DISABLED)")
        else:
            print(f"    {tname:25s} = {type(term).__name__}")

# PPO config
from sim.eval2.agents.rsl_rl_ppo_cfg_v2_13 import LiftCubePPORunnerCfgV213
p = LiftCubePPORunnerCfgV213()
print(f"\n[PPO cfg LiftCubePPORunnerCfgV213]")
print(f"  experiment_name = {p.experiment_name}")
print(f"  max_iterations  = {p.max_iterations}")
print(f"  num_steps_per_env = {p.num_steps_per_env}")
print(f"  init_noise_std    = {p.policy.init_noise_std}")
print(f"  entropy_coef      = {p.algorithm.entropy_coef}")
print(f"  desired_kl        = {p.algorithm.desired_kl}")
print(f"  gamma             = {p.algorithm.gamma}")
print(f"  learning_rate     = {p.algorithm.learning_rate}")

print("\nDone — verification complete.")
simulation_app.close()
