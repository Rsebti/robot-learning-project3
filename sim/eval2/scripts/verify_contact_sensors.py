"""Verify ContactSensor placement and force readings for the V2.19 grasp predicate.

Method 1 of the V2.19 contact-impulse grasp verification: directly read
``sensor.data.net_forces_w_history`` and confirm that:

  1. Both contact sensors are registered in the scene.
       - ``contact_gripper``  bound to body ``gripper`` (FIXED top wrist plate)
       - ``contact_jaw``      bound to body ``jaw``     (MOVING lower jaw)
  2. The ``filter_prim_paths_expr`` actually matches the cube prim path
     (the script prints the resolved path so you can compare).
  3. Sensors read ~0 N when the cube is far away (no contact).
  4. Sensors read >> 0 N when the cube is forced to overlap the body
     (validates that contact filtering against the cube actually works).

Usage:
    cd C:/Users/user/Desktop/MA2/robot-learning-project3
    C:/Users/user/Desktop/MA2/isaac/isaac_so_arm101/.venv/Scripts/python.exe \
      sim/eval2/scripts/verify_contact_sensors.py
"""
from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    default="Isaac-LeIsaac-SO101-Lift-Visual-V219-cold-Play-v0",
    help="V2.19 Play task id (the verifier needs the V219 contact sensors).",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--settle_steps",
    type=int,
    default=20,
    help="physics steps after writing pose / placing cube before reading forces.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

args_cli.headless = True
args_cli.enable_cameras = True   # V219 is the Visual variant
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

import sim.eval2  # noqa: F401, E402
import isaac_so_arm101.tasks  # noqa: F401, E402
from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

from sim.eval2.mdp.rewards import cube_grasped_contact_v219  # noqa: E402


TOP_DOWN_JOINTS = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -1.0,
    "elbow_flex": +1.5,
    "wrist_flex": -1.5,
    "wrist_roll": 0.0,
}


SCENARIOS = [
    dict(
        name="(1) HOME pose, gripper OPEN  -- baseline, no contact",
        joints={"gripper": 0.5},
        cube_target=None,
        expect="both forces ~ 0 N, predicate FALSE",
    ),
    dict(
        name="(2) HOME pose, gripper CLOSED -- baseline, gripper closes in air",
        joints={"gripper": 0.0},
        cube_target=None,
        expect="both forces ~ 0 N, predicate FALSE",
    ),
    dict(
        name="(3) TOP-DOWN, cube held at palm body origin",
        joints={**TOP_DOWN_JOINTS, "gripper": 0.5},
        cube_target="palm",
        expect="contact_gripper > 0 N, predicate TRUE if d_jaw small",
    ),
    dict(
        name="(4) TOP-DOWN, cube held at jaw body origin (touches mesh, near fingertip)",
        joints={**TOP_DOWN_JOINTS, "gripper": 0.5},
        cube_target="jaw",
        expect="both forces > 0 AND d_jaw_tip < 10 cm AND predicate TRUE",
    ),
    dict(
        name="(5) TOP-DOWN with arm pushed flat into table, cube at default spawn (FP test)",
        joints={
            "shoulder_pan": 0.0,
            "shoulder_lift": -1.6,    # max forward to drive arm hard down
            "elbow_flex": +1.6,       # full elbow flex
            "wrist_flex": -1.6,       # wrist points down hard
            "wrist_roll": 0.0,
            "gripper": 0.0,           # closed jaws
        },
        cube_target=None,             # cube stays at default spawn (~20-30 cm away)
        expect="if both forces > 0 from table: predicate MUST stay FALSE (cube far)",
    ),
]


def _force_recent(sensor):
    """Most recent ||f|| on env 0 (handles both (N,T,B,3) and (N,T,3) shapes)."""
    h = sensor.data.net_forces_w_history
    f = h[0, 0]
    if f.ndim == 2:
        return float(f.norm(dim=-1).max().item())
    return float(f.norm().item())


def _force_max_history(sensor):
    """Max ||f|| over the whole history buffer for env 0 (catches transient impulses)."""
    h = sensor.data.net_forces_w_history
    f = h[0]
    return float(f.norm(dim=-1).max().item())


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg) -> None:  # type: ignore[no-untyped-def]
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    u = env.unwrapped
    robot = u.scene["robot"]
    cube = u.scene["cube"]
    sim = u.sim
    def _scene_get(scene, name):
        try:
            return scene[name]
        except KeyError:
            return None
    contact_gripper = _scene_get(u.scene, "contact_gripper")
    contact_jaw = _scene_get(u.scene, "contact_jaw")

    print()
    print("=" * 80)
    print("V2.19 CONTACT SENSOR VERIFICATION")
    print("=" * 80)
    print(f"Task         : {args_cli.task}")
    print(f"Robot bodies : {list(robot.data.body_names)}")
    print(f"Joint names  : {list(robot.data.joint_names)}")
    try:
        print(f"Cube cfg prim_path : {cube.cfg.prim_path}")
    except Exception as e:
        print(f"Cube cfg prim_path : <error: {e}>")
    if hasattr(cube, "root_physx_view"):
        try:
            paths = cube.root_physx_view.prim_paths
            print(f"Cube resolved path : {paths[0]}   <- this must match filter_prim_paths_expr")
        except Exception as e:
            print(f"Cube resolved path : <error: {e}>")
    print()

    for label, sensor in [
        ("contact_gripper (FIXED jaw / top wrist plate)", contact_gripper),
        ("contact_jaw     (MOVING jaw / lower finger)", contact_jaw),
    ]:
        if sensor is None:
            print(f"!! {label}: NOT FOUND in scene -- sensor was never registered.")
            continue
        cfg = sensor.cfg
        print(f"{label}:")
        print(f"  prim_path           : {cfg.prim_path}")
        print(f"  filter_prim_paths   : {cfg.filter_prim_paths_expr}")
        print(f"  history_length      : {cfg.history_length}")
        print(f"  net_forces_w_history shape : {tuple(sensor.data.net_forces_w_history.shape)}")
    print()

    palm_body_idx = robot.find_bodies("gripper")[0][0]
    jaw_body_idx = robot.find_bodies("jaw")[0][0]

    physics_dt = sim.get_physics_dt()
    joint_names = list(robot.data.joint_names)
    results: list[tuple[str, float, float]] = []

    for sc in SCENARIOS:
        env.reset()

        # 1) Force joint positions
        new_pos = robot.data.joint_pos[0].clone()
        for jname, val in sc["joints"].items():
            if jname in joint_names:
                new_pos[joint_names.index(jname)] = float(val)
            else:
                print(f"  [WARN] joint '{jname}' not in robot -- skipping")
        new_pos = new_pos.unsqueeze(0)
        new_vel = torch.zeros_like(new_pos)
        robot.write_joint_state_to_sim(new_pos, new_vel)
        robot.set_joint_position_target(new_pos)
        robot.write_data_to_sim()

        # 2) Settle a few steps so palm/jaw world position is up to date
        for _ in range(5):
            sim.step(render=False)
            u.scene.update(physics_dt)

        # 3) Optionally re-place cube to overlap with a robot-frame target.
        # Compute target position now (palm/jaw already up to date after step 2).
        palm_pos = robot.data.body_pos_w[0, palm_body_idx, :].clone()
        jaw_pos = robot.data.body_pos_w[0, jaw_body_idx, :].clone()
        target = None
        if sc["cube_target"] == "palm":
            target = palm_pos.clone()           # cube on palm body origin (wrist plate)
        elif sc["cube_target"] == "jaw":
            target = jaw_pos.clone()            # cube on jaw body origin (rotation pivot)
        elif sc["cube_target"] == "jaw_fingertip":
            # cube CENTER at jaw FINGERTIP (ee_frame.target[1], with offset).
            # Realistic grasp: cube is where the jaw fingertip touches it.
            try:
                ee_frame = u.scene["ee_frame"]
                target = ee_frame.data.target_pos_w[0, 1, :].clone()
            except Exception as e:
                print(f"  [WARN] could not read ee_frame jaw fingertip: {e}")
                target = jaw_pos.clone()

        # 4) Settle: re-pin cube each step so gravity does not pull it away
        # (kinematic-style hold). PhysX continuously resolves the overlap and
        # registers contact forces every step.
        zero6 = torch.zeros((1, 6), device=robot.data.body_pos_w.device)
        if target is not None:
            quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=target.device)
        for _ in range(args_cli.settle_steps):
            if target is not None:
                # Re-track the body in case it moves slightly (shouldn't, but
                # robot PD may compensate to fight the cube push).
                if sc["cube_target"] == "palm":
                    target = robot.data.body_pos_w[0, palm_body_idx, :].clone()
                elif sc["cube_target"] == "jaw":
                    target = robot.data.body_pos_w[0, jaw_body_idx, :].clone()
                elif sc["cube_target"] == "jaw_fingertip":
                    try:
                        target = u.scene["ee_frame"].data.target_pos_w[0, 1, :].clone()
                    except Exception:
                        pass
                pose = torch.cat([target, quat]).unsqueeze(0)
                cube.write_root_pose_to_sim(pose)
                cube.write_root_velocity_to_sim(zero6)
            sim.step(render=False)
            u.scene.update(physics_dt)

        # 5) Read final state + forces + predicate
        palm_z = robot.data.body_pos_w[0, palm_body_idx, 2].item()
        jaw_z = robot.data.body_pos_w[0, jaw_body_idx, 2].item()
        cube_p = cube.data.root_pos_w[0].cpu().numpy()
        gripper_q = robot.data.joint_pos[0, joint_names.index("gripper")].item()

        f_grip_recent = _force_recent(contact_gripper) if contact_gripper is not None else float("nan")
        f_jaw_recent = _force_recent(contact_jaw) if contact_jaw is not None else float("nan")
        f_grip_max = _force_max_history(contact_gripper) if contact_gripper is not None else float("nan")
        f_jaw_max = _force_max_history(contact_jaw) if contact_jaw is not None else float("nan")

        # Compute distance from cube to jaw fingertip (via ee_frame.target[1])
        try:
            ee_frame = u.scene["ee_frame"]
            jaw_tip = ee_frame.data.target_pos_w[0, 1, :]
            d_jaw_tip = (cube.data.root_pos_w[0] - jaw_tip).norm().item()
        except Exception as e:
            d_jaw_tip = float("nan")
            print(f"  [WARN] could not read ee_frame jaw tip: {e}")

        # Run the actual V2.19 predicate
        try:
            predicate = bool(cube_grasped_contact_v219(u)[0].item())
        except Exception as e:
            predicate = None
            print(f"  [WARN] predicate failed: {e}")

        print(f"--- {sc['name']} ---")
        print(f"  expected            : {sc['expect']}")
        print(f"  palm_z   = {palm_z:+.4f} m   jaw_z = {jaw_z:+.4f} m   gripper_q = {gripper_q:+.3f}")
        print(f"  cube_pos = ({cube_p[0]:+.3f}, {cube_p[1]:+.3f}, {cube_p[2]:+.3f})")
        print(f"  contact_gripper (FIXED) : recent = {f_grip_recent:7.3f} N   max_history = {f_grip_max:7.3f} N")
        print(f"  contact_jaw     (MOBILE): recent = {f_jaw_recent:7.3f} N   max_history = {f_jaw_max:7.3f} N")
        print(f"  d(cube, jaw_fingertip)  = {d_jaw_tip*100:6.2f} cm   (gate threshold: 10.00 cm)")
        print(f"  PREDICATE cube_grasped  = {predicate}")
        print()

        results.append((sc["name"], f_grip_max, f_jaw_max, d_jaw_tip, predicate))

    # Diagnostic verdict
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    # results entries are (name, f_grip_max, f_jaw_max, d_jaw_tip, predicate)
    sc1 = results[0]
    sc4 = results[3]
    sc5 = results[4]

    print(f"  (1) HOME baseline forces        : grip={sc1[1]:.2f} N  jaw={sc1[2]:.2f} N  predicate={sc1[4]}")
    print(f"  (4) cube AT jaw body (true grasp): grip={sc4[1]:.2f} N  jaw={sc4[2]:.2f} N  d_jaw={sc4[3]*100:.1f} cm  predicate={sc4[4]}")
    print(f"  (5) robot crash on table        : grip={sc5[1]:.2f} N  jaw={sc5[2]:.2f} N  d_jaw={sc5[3]*100:.1f} cm  predicate={sc5[4]}")
    print()

    pass_baseline = sc1[1] < 0.1 and sc1[2] < 0.1 and sc1[4] is False
    pass_true_grasp = sc4[1] > 1.0 and sc4[2] > 1.0 and sc4[4] is True
    pass_no_false_pos = sc5[4] is False     # the critical anti-table-contact check

    if pass_baseline and pass_true_grasp and pass_no_false_pos:
        print("  *** PASS *** Sensors fire on cube contact AND predicate rejects table contact.")
        print("              V2.19 grasp_milestone wired correctly + false-positive resistant.")
    else:
        print("  *** FAIL *** at least one check failed:")
        if not pass_baseline:
            print(f"      - baseline non-zero or predicate=True (forces={sc1[1]:.3f}, {sc1[2]:.3f}, predicate={sc1[4]})")
        if not pass_true_grasp:
            print(f"      - true grasp not detected (forces={sc4[1]:.3f}, {sc4[2]:.3f}, predicate={sc4[4]})")
            print(f"        d_jaw={sc4[3]*100:.1f} cm: if > 5 cm, increase cube_proximity_threshold")
        if not pass_no_false_pos:
            print(f"      - FALSE POSITIVE on table crash (predicate={sc5[4]})")
            print(f"        d_jaw={sc5[3]*100:.1f} cm: if < 5 cm, decrease cube_proximity_threshold")
            print(f"        OR cube was inadvertently moved into grip zone by the crash")
    print("=" * 80)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
