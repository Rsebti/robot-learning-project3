"""Dump the SO-101 robot body names + joint names + ee_frame targets.

Minimal init: spawn env, reset once, print body/joint info, exit.
Used to find the right body name for `scoop_grasp_penalty` (wrist body).
"""
import argparse
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-LeIsaac-SO101-Lift-Visual-V212-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

# Force headless + cameras
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import sim.eval2  # noqa: F401
import isaac_so_arm101.tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.assets import Articulation


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    robot: Articulation = env.unwrapped.scene["robot"]

    print("=" * 70, flush=True)
    print("ROBOT BODY NAMES (in order):", flush=True)
    for i, name in enumerate(robot.body_names):
        print(f"  [{i:2d}]  {name}", flush=True)
    print()
    print("ROBOT JOINT NAMES (in order):", flush=True)
    for i, name in enumerate(robot.joint_names):
        print(f"  [{i:2d}]  {name}", flush=True)
    print("=" * 70, flush=True)

    # Try common wrist body names
    print()
    print("Trying common wrist body names:", flush=True)
    candidates = ["wrist", "wrist_roll", "wrist_roll_link", "wrist_pitch",
                  "wrist_pitch_link", "wrist_flex", "wrist_flex_link", "link5", "link_5"]
    for cand in candidates:
        try:
            ids, names = robot.find_bodies(cand)
            if len(ids) > 0:
                print(f"  ✓ '{cand}' → index {ids[0]}, matched name '{names[0]}'", flush=True)
            else:
                print(f"  ✗ '{cand}' (no match)", flush=True)
        except Exception as e:
            print(f"  ✗ '{cand}' (error: {e})", flush=True)

    # Also dump ee_frame targets
    try:
        ee_frame = env.unwrapped.scene["ee_frame"]
        print()
        print("EE FRAME TARGETS:", flush=True)
        if hasattr(ee_frame, "_target_frames"):
            for i, t in enumerate(ee_frame._target_frames):
                print(f"  [{i}]  {t}", flush=True)
        else:
            cfg = ee_frame.cfg
            print(f"  (cfg: {cfg})", flush=True)
    except Exception as e:
        print(f"  ee_frame inspection error: {e}", flush=True)

    print()
    print("DONE — exiting", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
