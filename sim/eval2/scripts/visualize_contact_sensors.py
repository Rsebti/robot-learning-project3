"""Visual + manual debug for the V2.19 contact sensors.

Opens Isaac Sim with the GUI, puts the SO-101 in a top-down pose with the
gripper CLOSED, enables debug visualization on both contact sensors, and
prints force readings continuously. You can then manually drag the cube
in the viewport to touch each finger and watch the forces respond in the
terminal.

Two visual aids are added on top of Isaac Sim's defaults:
  - RED   sphere  -> body 'gripper' (FIXED top wrist plate, contact_gripper)
  - GREEN sphere  -> body 'jaw'     (MOVING lower finger,  contact_jaw)

Manual interaction workflow:
  1. Once the viewport is up, press SPACEBAR to pause the sim.
  2. Click the cube in the viewport.
  3. Press W to activate the translate gizmo (or right-click > Move).
  4. Drag the cube onto the RED or GREEN sphere.
  5. Press SPACEBAR to resume -> physics resolves the overlap and
     the corresponding force prints non-zero in this terminal.
  6. Ctrl+C here to quit.

Usage:
    cd C:/Users/user/Desktop/MA2/robot-learning-project3
    C:/Users/user/Desktop/MA2/isaac/isaac_so_arm101/.venv/Scripts/python.exe \
      sim/eval2/scripts/visualize_contact_sensors.py
"""
from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    default="Isaac-LeIsaac-SO101-Lift-Visual-V219-cold-Play-v0",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--print_every",
    type=int,
    default=30,
    help="print sensor forces every N physics steps (~0.5 s at 60 Hz)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

args_cli.headless = False         # GUI ON
args_cli.enable_cameras = True    # V219 is the Visual variant
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

import sim.eval2  # noqa: F401, E402
import isaac_so_arm101.tasks  # noqa: F401, E402
from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402


def _scene_get(scene, name):
    """Safe accessor — InteractiveScene has __getitem__ (KeyError) but no .get()."""
    try:
        return scene[name]
    except KeyError:
        return None


TOP_DOWN_JOINTS = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -1.0,
    "elbow_flex": +1.5,
    "wrist_flex": -1.5,
    "wrist_roll": 0.0,
    "gripper": 0.0,   # CLOSED so a placed cube gets pinched
}


def _make_marker(prim_path: str, color: tuple[float, float, float], radius: float = 0.012):
    """Create a small colored sphere marker. Returns None if the markers API fails."""
    try:
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        cfg = VisualizationMarkersCfg(
            prim_path=prim_path,
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                )
            },
        )
        return VisualizationMarkers(cfg)
    except Exception as e:
        print(f"[WARN] could not create marker {prim_path}: {e}")
        return None


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg) -> None:  # type: ignore[no-untyped-def]
    env_cfg.scene.num_envs = args_cli.num_envs

    # Enable debug visualization on contact sensors BEFORE env build.
    if hasattr(env_cfg.scene, "contact_gripper") and env_cfg.scene.contact_gripper is not None:
        env_cfg.scene.contact_gripper.debug_vis = True
    if hasattr(env_cfg.scene, "contact_jaw") and env_cfg.scene.contact_jaw is not None:
        env_cfg.scene.contact_jaw.debug_vis = True

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    u = env.unwrapped
    robot = u.scene["robot"]
    cube = u.scene["cube"]
    sim = u.sim
    contact_gripper = _scene_get(u.scene, "contact_gripper")
    contact_jaw = _scene_get(u.scene, "contact_jaw")

    # Force robot into a top-down pose with the gripper closed
    joint_names = list(robot.data.joint_names)
    new_pos = robot.data.joint_pos[0].clone()
    for jname, val in TOP_DOWN_JOINTS.items():
        if jname in joint_names:
            new_pos[joint_names.index(jname)] = float(val)
    new_pos = new_pos.unsqueeze(0)
    new_vel = torch.zeros_like(new_pos)
    robot.write_joint_state_to_sim(new_pos, new_vel)
    robot.set_joint_position_target(new_pos)
    robot.write_data_to_sim()

    physics_dt = sim.get_physics_dt()
    for _ in range(20):
        sim.step(render=False)
        u.scene.update(physics_dt)

    # Body indices for marker tracking
    try:
        palm_body_idx = robot.find_bodies("gripper")[0][0]
    except Exception as e:
        palm_body_idx = None
        print(f"[WARN] body 'gripper' not found: {e}")
    try:
        jaw_body_idx = robot.find_bodies("jaw")[0][0]
    except Exception as e:
        jaw_body_idx = None
        print(f"[WARN] body 'jaw' not found: {e}")

    # Custom marker spheres at body locations
    palm_marker = _make_marker("/Visuals/palm_sensor_marker", color=(1.0, 0.0, 0.0))
    jaw_marker = _make_marker("/Visuals/jaw_sensor_marker", color=(0.0, 1.0, 0.0))

    print()
    print("=" * 78)
    print("V2.19 CONTACT SENSOR VISUAL DEBUG")
    print("=" * 78)
    print(f"Task         : {args_cli.task}")
    print(f"Robot bodies : {list(robot.data.body_names)}")
    if contact_gripper is not None:
        print(f"contact_gripper  prim_path = {contact_gripper.cfg.prim_path}")
        print(f"                 filter    = {contact_gripper.cfg.filter_prim_paths_expr}")
    else:
        print("!! contact_gripper NOT FOUND in scene")
    if contact_jaw is not None:
        print(f"contact_jaw      prim_path = {contact_jaw.cfg.prim_path}")
        print(f"                 filter    = {contact_jaw.cfg.filter_prim_paths_expr}")
    else:
        print("!! contact_jaw NOT FOUND in scene")
    try:
        print(f"Cube cfg prim_path : {cube.cfg.prim_path}")
        if hasattr(cube, "root_physx_view"):
            print(f"Cube resolved      : {cube.root_physx_view.prim_paths[0]}")
    except Exception as e:
        print(f"Cube path read failed: {e}")
    print()
    print("Markers in viewport:")
    print("  RED   sphere  -> body 'gripper' (FIXED jaw, top wrist plate)")
    print("  GREEN sphere  -> body 'jaw'     (MOVING jaw, lower finger)")
    print()
    print("Manual interaction:")
    print("  - SPACEBAR in viewport         pause / resume physics")
    print("  - click cube + W               translate gizmo")
    print("  - drag cube onto a sphere      overlap with that sensor body")
    print("  - resume                       contact forces appear below")
    print("  - Ctrl+C in this terminal      quit")
    print("=" * 78)
    print()

    step = 0
    try:
        while simulation_app.is_running():
            sim.step(render=True)
            u.scene.update(physics_dt)

            # Move marker spheres to track the bodies in case they shift
            if palm_marker is not None and palm_body_idx is not None:
                palm_pos = robot.data.body_pos_w[:, palm_body_idx, :]
                palm_marker.visualize(translations=palm_pos)
            if jaw_marker is not None and jaw_body_idx is not None:
                jaw_pos = robot.data.body_pos_w[:, jaw_body_idx, :]
                jaw_marker.visualize(translations=jaw_pos)

            step += 1
            if step % args_cli.print_every == 0:
                cube_p = cube.data.root_pos_w[0].cpu().numpy()
                if palm_body_idx is not None:
                    d_palm = (cube.data.root_pos_w[0] - robot.data.body_pos_w[0, palm_body_idx, :]).norm().item()
                else:
                    d_palm = float("nan")
                if jaw_body_idx is not None:
                    d_jaw = (cube.data.root_pos_w[0] - robot.data.body_pos_w[0, jaw_body_idx, :]).norm().item()
                else:
                    d_jaw = float("nan")

                if contact_gripper is not None:
                    f_grip = contact_gripper.data.net_forces_w_history[0].norm(dim=-1).max().item()
                else:
                    f_grip = float("nan")
                if contact_jaw is not None:
                    f_jaw = contact_jaw.data.net_forces_w_history[0].norm(dim=-1).max().item()
                else:
                    f_jaw = float("nan")

                print(
                    f"step={step:5d}  "
                    f"cube=({cube_p[0]:+.3f},{cube_p[1]:+.3f},{cube_p[2]:+.3f})  "
                    f"d(palm)={d_palm*100:5.1f}cm  d(jaw)={d_jaw*100:5.1f}cm  "
                    f"|F_gripper|={f_grip:6.2f}N  |F_jaw|={f_jaw:6.2f}N"
                )
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt -- exiting")
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
