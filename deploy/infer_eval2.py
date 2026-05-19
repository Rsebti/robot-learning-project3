"""
infer_eval2.py - goal-conditioned inference for Eval-2 / Eval-3.

Loads a trained goal-conditioned ACT policy and runs autonomous rollouts on
the SO-101 follower. At each step it captures the robot's obs (joints + wrist
camera), INJECTS the user-specified `observation.environment_state` (6D
color one-hot + 2D bowl xy in meters) into the obs dict, calls the policy,
and sends the action back to the robot.

env_state = [c_yellow, c_orange, c_red, c_blue, c_green, c_violet,
             bowl_x_m, bowl_y_m]   (8D)

Usage:
    python deploy/infer_eval2.py \\
        --policy_path  hudela390/projet3-act-eval2-v1-goal \\
        --target_color red \\
        --bowl_x -0.155  --bowl_y  0.295 \\
        --follower_port COM3
"""

from __future__ import annotations

import argparse

import torch

from act_infer_common import (
    add_act_cli_args,
    build_env_state,
    load_policy_and_processors,
    run_act_rollouts,
)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--policy_path", default="hudela390/projet3-act-eval2-v1-goal")
    add_act_cli_args(parser, default_home_pose="eval2_rest")
    args = parser.parse_args()

    print(f"[infer] policy:        {args.policy_path}")
    print(f"[infer] target color:  {args.target_color}")
    print(f"[infer] bowl xy:       ({args.bowl_x:+.3f}, {args.bowl_y:+.3f}) m")
    print(f"[infer] device:        {args.device}")

    env_state = build_env_state(args.target_color, args.bowl_x, args.bowl_y)
    print(f"[infer] env_state:     {env_state.round(3).tolist()}")

    print("[infer] Loading policy + processors ...")
    policy, preprocessor, postprocessor = load_policy_and_processors(
        args.policy_path, args.device,
    )

    task_desc = (
        f"Pick {args.target_color} block and place in bowl at "
        f"({args.bowl_x*100:.1f},{args.bowl_y*100:.1f}) cm"
    )

    run_act_rollouts(args, policy, preprocessor, postprocessor, env_state, task_desc)


if __name__ == "__main__":
    main()
