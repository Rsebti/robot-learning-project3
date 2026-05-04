"""Eval 2 — open the env in interactive 3D mode without training.

Loads the scene, resets the robot to its home pose, then **freezes the
physics simulation** so you can inspect / position cameras / edit prim
transforms without the robot drifting under gravity or wandering.

Usage:
    uv run python -m sim.eval2.scripts.view --task Eval2-PickInClutter-Play-v2 --num_envs 1 --enable_cameras

Once the Isaac Sim window is up:
- The scene is fully loaded but **paused** (no physics ticking).
- You can navigate with the mouse, switch viewport cameras (top-left
  dropdown), and edit prim transforms in the Property panel — your
  edits show up live and nothing in the scene moves on its own.
- Close the window or Ctrl+C in the terminal to quit.

Tip — slow down viewport navigation:
    The default viewport navigation in Isaac Sim is fast. To slow it
    down, click the gear/cog icon in the top-left of the viewport,
    open "Navigation Speed" and reduce it. Or use the keyboard:
    holding Shift makes movement slower.
"""

from __future__ import annotations

import argparse

# We must launch the SimulationApp BEFORE importing isaaclab/torch.
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Inspect the Eval 2 env without training (physics paused)."
)
parser.add_argument("--task", type=str, default="Eval2-PickInClutter-v1")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Force UI on so the user actually sees the window.
args.headless = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


# Hardcoded "scan" pose — gripper hovering above the workspace pointing
# straight down toward the table.
#
# How this works geometrically:
# - In the SO-101 home pose, the arm is horizontal forward and the wrist
#   is bent 90 deg (wrist_flex=1.57) which makes the gripper point straight
#   down. We KEEP wrist_flex=1.57 so the gripper keeps pointing down.
# - To raise the gripper *above* the workspace while preserving its
#   downward orientation, we tilt the upper arm UP by some angle alpha
#   (shoulder_lift=+alpha) and bend the elbow BACK by the same angle
#   (elbow_flex=-alpha). The two cancel out so the lower arm stays
#   horizontal — only the gripper's height/forward position changes.
# - alpha=1.0 rad puts the gripper roughly 12 cm above where home pose
#   has it, which is enough to clear the bowl walls and have a top-down
#   view of the workspace.
#
# Tweak these radians to taste; they only affect this view session, not
# training (the env home pose stays whatever is in SO_ARM101_CFG).
SCAN_POSE_JOINTS = {
    "shoulder_pan":   0.0,    # arm faces forward (+x)
    "shoulder_lift":  1.0,    # tilt upper arm up by ~57 deg
    "elbow_flex":    -1.0,    # bend elbow back by the same amount
    "wrist_flex":     1.57,   # keep the home 90-deg wrist bend = gripper down
    "wrist_roll":     0.0,
    "gripper":        0.0,    # jaws closed
}


def main():
    import gymnasium as gym
    import torch

    # Import side-effect: registers our gym envs.
    import sim.eval2  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)

    # One reset to instantiate the scene at the home pose.
    obs, _ = env.reset()

    # Override the robot to a "scan" pose where the gripper hovers above the
    # workspace pointing down. Useful for visually positioning the wrist
    # camera in a configuration close to what it'll see during a pick.
    # This only affects this view session — the training home pose stays
    # whatever is in SO_ARM101_CFG.
    robot = env.unwrapped.scene["robot"]
    joint_names = robot.data.joint_names
    target_pos = torch.zeros(args.num_envs, len(joint_names), device="cuda:0")
    for i, name in enumerate(joint_names):
        if name in SCAN_POSE_JOINTS:
            target_pos[:, i] = SCAN_POSE_JOINTS[name]
    target_vel = torch.zeros_like(target_pos)
    robot.write_joint_state_to_sim(target_pos, target_vel)

    # Freeze the world: don't step physics. Just render frames so the user
    # can pan around, edit prim transforms in the Property panel, and watch
    # those edits update live without anything else moving.
    print("\n" + "=" * 60)
    print(" Scene loaded. Physics is PAUSED.")
    print(" - Robot is in 'scan' pose: gripper above workspace, pointing down.")
    print(" - To change: edit SCAN_POSE_JOINTS at top of view.py.")
    print(" - Edit prims (e.g. wrist_cam) freely in the Property panel.")
    print(" - Switch viewport camera via the top-left dropdown.")
    print(" Close the window or Ctrl+C to quit.")
    print("=" * 60 + "\n")

    # Render-only loop — no physics step.
    while simulation_app.is_running():
        # ``simulation_app.update()`` advances the kit app one frame
        # (renders, processes UI events) without ticking the physics
        # simulation. This is the "paused but live" state we want.
        simulation_app.update()

    env.close()


if __name__ == "__main__":
    import traceback

    try:
        main()
    except Exception:  # noqa: BLE001
        # Without this, exceptions during gym.make / env.reset get swallowed
        # by Isaac Sim's shutdown sequence and the user only sees a silent
        # exit with no stack trace.
        print("\n=== view.py crashed — Python traceback follows ===")
        traceback.print_exc()
        print("=== end traceback ===\n")
    finally:
        simulation_app.close()
