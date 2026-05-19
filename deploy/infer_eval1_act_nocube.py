"""
infer_eval1_act_nocube.py - LeRobot ACT inference for the Eval-1 8D goal-
conditioned policy (no cube_xy in env_state).

env_state = [color_6, bowl_x_m, bowl_y_m]   (8D)

Color and bowl xy come from CLI flags (told by the TA at eval time). No CV
scout, no cube position - the policy must locate the cube from the wrist
image alone. Sister of infer_eval1_act.py (10D variant).

Usage:
    python deploy/infer_eval1_act_nocube.py \\
        --target_color yellow \\
        --bowl_x 0.16 --bowl_y 0.32 \\
        --follower_port COM3 --camera_index 1

    --no-home          skip ramp to rest pose (start from current pose)
    --home_pose NAME   preset in deploy/homes.py (default: eval1_rest)
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
    parser.add_argument("--policy_path", default="hudela390/projet3-act-eval1-v1-no-cube")
    add_act_cli_args(parser, default_home_pose="eval1_rest")
    args = parser.parse_args()

    print(f"[infer] policy:        {args.policy_path}")
    print(f"[infer] target color:  {args.target_color}")
    print(f"[infer] bowl xy:       ({args.bowl_x:+.3f}, {args.bowl_y:+.3f}) m")
    print(f"[infer] device:        {args.device}")

    print("[infer] Loading policy + processors ...")
    policy, preprocessor, postprocessor = load_policy_and_processors(
        args.policy_path, args.device,
    )

    task_desc = (
        f"Pick {args.target_color} block and place in bowl at "
        f"({args.bowl_x*100:.1f},{args.bowl_y*100:.1f}) cm"
    )
    env_state = build_env_state(args.target_color, args.bowl_x, args.bowl_y)
    print(f"[infer] env_state:     {env_state.round(3).tolist()}")

    run_act_rollouts(args, policy, preprocessor, postprocessor, env_state, task_desc)


if __name__ == "__main__":
    main()
