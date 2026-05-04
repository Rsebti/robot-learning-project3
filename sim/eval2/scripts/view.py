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
# Per-joint overrides (in radians). If unset, the value from SCAN_POSE_JOINTS
# below is used. Convenient for dialing in the scan pose interactively:
#     uv run python -m sim.eval2.scripts.view ... --shoulder_lift 1.4 --elbow_flex -2.0 --wrist_flex 1.0
parser.add_argument("--shoulder_pan",  type=float, default=None)
parser.add_argument("--shoulder_lift", type=float, default=None)
parser.add_argument("--elbow_flex",    type=float, default=None)
parser.add_argument("--wrist_flex",    type=float, default=None)
parser.add_argument("--wrist_roll",    type=float, default=None)
parser.add_argument("--gripper",       type=float, default=None)
parser.add_argument(
    "--no_lock",
    action="store_true",
    help=(
        "Disable the per-frame joint state rewrite. Pair with Isaac Sim's "
        "Pause button (or rely on the auto sim.pause() call) so the robot "
        "stays where you put it while you edit joint targets in the "
        "Property panel ('drive: angular: target position'). When you have "
        "a pose you like, copy the 6 joint values and hardcode them."
    ),
)
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
    print(f"\n[view.py] Robot joint names ({len(joint_names)}): {joint_names}")

    # Build effective scan pose: dict + per-joint CLI overrides.
    effective_pose = dict(SCAN_POSE_JOINTS)
    cli_overrides = {
        "shoulder_pan":  args.shoulder_pan,
        "shoulder_lift": args.shoulder_lift,
        "elbow_flex":    args.elbow_flex,
        "wrist_flex":    args.wrist_flex,
        "wrist_roll":    args.wrist_roll,
        "gripper":       args.gripper,
    }
    for name, val in cli_overrides.items():
        if val is not None:
            effective_pose[name] = val

    target_pos = torch.zeros(args.num_envs, len(joint_names), device="cuda:0")
    for i, name in enumerate(joint_names):
        if name in effective_pose:
            target_pos[:, i] = effective_pose[name]

    print(f"[view.py] effective pose (rad):")
    for name, val in effective_pose.items():
        marker = " <-- CLI" if cli_overrides.get(name) is not None else ""
        print(f"   {name:<14} = {val:+.3f}{marker}")

    target_vel = torch.zeros_like(target_pos)

    sim = env.unwrapped.sim

    # Initial pose: write once before the render loop in both modes so the
    # scene starts at the desired scan pose.
    robot.write_joint_state_to_sim(target_pos, target_vel)
    robot.set_joint_position_target(target_pos)
    robot.write_data_to_sim()

    if args.no_lock:
        # Interactive joint-editing mode: pause physics and let the user
        # tweak joint targets via Isaac Sim's Property panel
        # (Stage > Robot > joints > <joint> > drive: angular: target position).
        # Without sim.pause() the controllers would fight the user's edits
        # and drift the robot back to whatever target was set programmatically.
        try:
            sim.pause()
        except Exception:  # noqa: BLE001
            pass  # if pause isn't supported, fall back to render-only loop
        print("\n" + "=" * 60)
        print(" Interactive mode (--no_lock). Physics PAUSED.")
        print(" - Robot is at the initial scan pose.")
        print(" - Edit joints in Property panel:")
        print("     Stage panel  ->  Robot  ->  joints  ->  <joint>")
        print("     Property panel  ->  drive: angular: target position")
        print(" - When you find a pose you like, note the 6 joint values")
        print("   and pass them via CLI (--shoulder_lift X --elbow_flex Y ...)")
        print("   on the next launch, OR hardcode them in SCAN_POSE_JOINTS.")
        print(" - Bascule sur 'wrist_cam' dans le dropdown camera pour")
        print("   verifier la vue depuis le poignet.")
        print(" Close the window or Ctrl+C to quit.")
        print("=" * 60 + "\n")
        while simulation_app.is_running():
            simulation_app.update()
    else:
        # Force-lock mode (default): re-write the joint state every frame
        # so internal PD controllers can't drift the robot away from the
        # requested pose. Use this when you've already found the pose you
        # like and just want to inspect it / position the camera.
        print("\n" + "=" * 60)
        print(" Lock mode. Robot LOCKED to scan pose every frame.")
        print(" - Robot stays exactly where you put it (re-written each frame).")
        print(" - Edit prims (e.g. wrist_cam) freely in the Property panel.")
        print(" - Switch viewport camera via the top-left dropdown.")
        print(" - To edit joints interactively, relaunch with --no_lock.")
        print(" Close the window or Ctrl+C to quit.")
        print("=" * 60 + "\n")
        while simulation_app.is_running():
            robot.write_joint_state_to_sim(target_pos, target_vel)
            robot.set_joint_position_target(target_pos)
            robot.write_data_to_sim()
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
