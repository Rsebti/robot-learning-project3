"""Full diagnostic dump of the Isaac scene at robot home pose.

Prints EVERYTHING we need to compare against Squint training:

- Robot bodies: world position, world quaternion, local axes (where they
  point in world frame at home pose)
- Robot joints: name, qpos, qvel, soft + hard limits, joint axis direction
  in world frame
- Actuators: stiffness, damping, force/velocity limits per joint
- Cube + bin parts: world position, quaternion
- Wrist camera: world position, quaternion in 3 conventions, optical axis
  direction in world (per convention), ground projection on table
- Sanity: distance gripper→cube, gripper→bin

Launch
------
.. code-block:: powershell

    cd C:\\Users\\user\\Desktop\\MA2\\isaac\\isaac_so_arm101
    C:/Users/user/Desktop/MA2/isaac/isaac_so_arm101/.venv/Scripts/python.exe `
        -m sim.eval2.scripts.debug_full_audit --task Isaac-SO101-Squint-Place-Play-v0 `
        --num_envs 1 --enable_cameras
"""
from __future__ import annotations

import argparse
import math
import sys

# Windows: console redirected to a file defaults to cp1252 and chokes on
# our box-drawing characters (─, →, °, Δ). Force UTF-8 for stdout/stderr.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch  # noqa: F401  (import early to win over Isaac Kit DLLs)

parser = argparse.ArgumentParser(description="Full Isaac scene audit at robot home pose")
parser.add_argument("--task", type=str, default="Isaac-SO101-Squint-Place-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--settle_steps", type=int, default=10,
                    help="Physics steps to settle the cube before reading positions")
parser.add_argument("--dump_wrist_png", type=str, default=None,
                    help="If set, save the wrist cam first frame as PNG to this path")
parser.add_argument("--probe_joint_axes", action="store_true", default=True,
                    help="Probe each joint axis by commanding +0.1 normalized action on each joint")
parser.add_argument("--probe_steps", type=int, default=10,
                    help="Number of physics steps per joint probe")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import sim.eval2  # noqa: F401,E402  (registers the Squint tasks)


# ---------------------------------------------------------------------------
# Quaternion → world axis helpers (Hamilton convention, scalar-first)
# ---------------------------------------------------------------------------


def _quat_rotate(q, v):
    """Rotate vector v by quaternion q (w, x, y, z). Returns (x, y, z) tuple."""
    qw, qx, qy, qz = q
    vx, vy, vz = v
    rx = (1 - 2 * (qy * qy + qz * qz)) * vx + 2 * (qx * qy - qw * qz) * vy + 2 * (qx * qz + qw * qy) * vz
    ry = 2 * (qx * qy + qw * qz) * vx + (1 - 2 * (qx * qx + qz * qz)) * vy + 2 * (qy * qz - qw * qx) * vz
    rz = 2 * (qx * qz - qw * qy) * vx + 2 * (qy * qz + qw * qx) * vy + (1 - 2 * (qx * qx + qy * qy)) * vz
    return (rx, ry, rz)


def _local_axes_in_world(q):
    """Return where local +X, +Y, +Z point in world for given quat (w,x,y,z)."""
    return {
        "+X": _quat_rotate(q, (1, 0, 0)),
        "+Y": _quat_rotate(q, (0, 1, 0)),
        "+Z": _quat_rotate(q, (0, 0, 1)),
        "-X": _quat_rotate(q, (-1, 0, 0)),
        "-Y": _quat_rotate(q, (0, -1, 0)),
        "-Z": _quat_rotate(q, (0, 0, -1)),
    }


def _fmt_vec(v, prec=4):
    return f"({v[0]:+.{prec}f}, {v[1]:+.{prec}f}, {v[2]:+.{prec}f})"


def _fmt_quat(q, prec=4):
    return f"(w={q[0]:+.{prec}f}, x={q[1]:+.{prec}f}, y={q[2]:+.{prec}f}, z={q[3]:+.{prec}f})"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    # Avoid HDF5 file-lock collision when run in parallel with other tools.
    try:
        env_cfg.recorders = None
    except Exception:
        pass
    env = gym.make(args.task, cfg=env_cfg)
    base_env = env.unwrapped
    scene = base_env.scene

    obs, _ = env.reset(seed=0)
    n_act = base_env.action_space.shape[-1]
    zero_action = torch.zeros(args.num_envs, n_act, device=device)
    print(f"\n[audit] Settling physics for {args.settle_steps} steps...")
    for _ in range(args.settle_steps):
        env.step(zero_action)

    env_origin = scene.env_origins[0]
    eo = env_origin.detach().cpu().numpy().tolist()

    # ---- Header ----
    line = "=" * 90
    print(f"\n{line}\n  FULL SCENE AUDIT — task={args.task}\n{line}")
    print(f"\n[env_origin] world = {_fmt_vec(eo)}")

    # ============================================================
    # ROBOT
    # ============================================================
    print(f"\n{'─' * 90}\n  ROBOT\n{'─' * 90}")

    robot = scene["robot"]
    joint_names = list(robot.joint_names)
    body_names = list(robot.body_names)
    n_joints = len(joint_names)
    n_bodies = len(body_names)
    print(f"\n[joint count] {n_joints}    [body count] {n_bodies}")

    # ---- Joints ----
    print(f"\n[joints — qpos / qvel / soft limits]")
    qpos = robot.data.joint_pos[0].detach().cpu().tolist()
    qvel = robot.data.joint_vel[0].detach().cpu().tolist()
    soft_lo = robot.data.soft_joint_pos_limits[0, :, 0].detach().cpu().tolist()
    soft_hi = robot.data.soft_joint_pos_limits[0, :, 1].detach().cpu().tolist()
    for i, jn in enumerate(joint_names):
        print(f"  [{i}] {jn:18s}  qpos={qpos[i]:+8.4f} rad ({math.degrees(qpos[i]):+8.2f}°)  "
              f"qvel={qvel[i]:+8.4f}  soft=[{soft_lo[i]:+6.3f}, {soft_hi[i]:+6.3f}]")

    # ---- Actuator gains ----
    print(f"\n[actuators — stiffness/damping (POST our override should be 1000/100)]")
    try:
        for actuator_name, actuator in robot.actuators.items():
            print(f"  [{actuator_name}]")
            print(f"    joint_indices = {actuator.joint_indices}")
            print(f"    stiffness     = {actuator.stiffness}")
            print(f"    damping       = {actuator.damping}")
            print(f"    effort_limit_sim   = {actuator.effort_limit_sim}")
            print(f"    velocity_limit_sim = {actuator.velocity_limit_sim}")
    except Exception as e:
        print(f"  (failed to read actuators: {e})")

    # ---- Bodies (link world poses + local axes) ----
    print(f"\n[bodies — world position + quat + local axes in world]")
    body_pos_w = robot.data.body_pos_w[0].detach().cpu().numpy()
    body_quat_w = robot.data.body_quat_w[0].detach().cpu().numpy()
    for i, bn in enumerate(body_names):
        p = body_pos_w[i].tolist()
        q = body_quat_w[i].tolist()
        axes = _local_axes_in_world(q)
        print(f"  [{i}] {bn:25s}")
        print(f"        pos_w  = {_fmt_vec(p)}    env_local = {_fmt_vec([p[k]-eo[k] for k in range(3)])}")
        print(f"        quat_w = {_fmt_quat(q)}")
        print(f"        local +X → world {_fmt_vec(axes['+X'], prec=3)}    "
              f"local +Y → world {_fmt_vec(axes['+Y'], prec=3)}    "
              f"local +Z → world {_fmt_vec(axes['+Z'], prec=3)}")

    # ============================================================
    # CUBE + BIN
    # ============================================================
    print(f"\n{'─' * 90}\n  CUBE + BIN\n{'─' * 90}")
    rigid_names = list(scene.rigid_objects.keys())
    print(f"\n[rigid objects: {rigid_names}]")
    for rn in rigid_names:
        obj = scene[rn]
        p = obj.data.root_pos_w[0].detach().cpu().numpy().tolist()
        q = obj.data.root_quat_w[0].detach().cpu().numpy().tolist()
        print(f"  {rn:18s}  pos_w = {_fmt_vec(p)}    quat_w = {_fmt_quat(q)}")

    # ============================================================
    # CAMERA
    # ============================================================
    print(f"\n{'─' * 90}\n  WRIST CAMERA\n{'─' * 90}")
    if "wrist" in scene.sensors:
        cam = scene["wrist"]
        cam_pos = cam.data.pos_w[0].detach().cpu().numpy().tolist()
        print(f"\n[wrist cam] world pos = {_fmt_vec(cam_pos)}")
        for cv in ("quat_w_world", "quat_w_opengl", "quat_w_ros"):
            if hasattr(cam.data, cv):
                q = getattr(cam.data, cv)[0].detach().cpu().numpy().tolist()
                axes = _local_axes_in_world(q)
                print(f"\n  [{cv}] = {_fmt_quat(q)}")
                for axis_name, vec in axes.items():
                    label = ""
                    # Annotate the optical axis under each convention.
                    # Isaac convention semantics:
                    #   "world"  → forward = local +X
                    #   "opengl" → forward = local -Z
                    #   "ros"    → forward = local +Z
                    if cv == "quat_w_world" and axis_name == "+X":
                        label = "  ← optical axis (world/SAPIEN: forward=+X)"
                    elif cv == "quat_w_opengl" and axis_name == "-Z":
                        label = "  ← optical axis (OpenGL)"
                    elif cv == "quat_w_ros" and axis_name == "+Z":
                        label = "  ← optical axis (ROS)"
                    print(f"      local {axis_name} → world {_fmt_vec(vec, prec=3)}{label}")
                # Compute ground projection of optical axis on table z=0.036
                if cv == "quat_w_world":
                    opt = axes["+X"]
                    if opt[2] < -1e-6:
                        t = (0.036 - cam_pos[2]) / opt[2]
                        hit = (cam_pos[0] + t * opt[0], cam_pos[1] + t * opt[1], 0.036)
                        print(f"\n  → ground projection (table z=0.036, world): {_fmt_vec(hit, prec=3)}")
                    else:
                        print(f"\n  → cam optical axis does NOT point downward — ground hit at infinity")

    # ============================================================
    # SANITY DISTANCES
    # ============================================================
    print(f"\n{'─' * 90}\n  KEY DISTANCES\n{'─' * 90}")
    if "gripper" in body_names:
        gp = body_pos_w[body_names.index("gripper")]
        if "cube" in scene.rigid_objects:
            cp = scene["cube"].data.root_pos_w[0].detach().cpu().numpy()
            d = ((gp - cp) ** 2).sum() ** 0.5
            print(f"  gripper → cube     distance = {d:.4f} m  (Δxyz = {_fmt_vec((cp - gp).tolist(), prec=3)})")
        if "bowl" in scene.rigid_objects:
            bp = scene["bowl"].data.root_pos_w[0].detach().cpu().numpy()
            d = ((gp - bp) ** 2).sum() ** 0.5
            print(f"  gripper → bowl     distance = {d:.4f} m  (Δxyz = {_fmt_vec((bp - gp).tolist(), prec=3)})")
    if "wrist" in scene.sensors and "cube" in scene.rigid_objects:
        cam_pos = scene["wrist"].data.pos_w[0].detach().cpu().numpy()
        cp = scene["cube"].data.root_pos_w[0].detach().cpu().numpy()
        d = ((cam_pos - cp) ** 2).sum() ** 0.5
        print(f"  wrist cam → cube   distance = {d:.4f} m  (Δxyz = {_fmt_vec((cp - cam_pos).tolist(), prec=3)})")

    # ============================================================
    # JOINT AXIS PROBE (mirrors ManiSkill side)
    # ============================================================
    if args.probe_joint_axes:
        print(f"\n{'─' * 90}\n  JOINT AXIS PROBE")
        print(f"  Commanding +0.1 (norm action) on ONE joint at a time, observing Δworld pos of jaw.")
        print(f"  This reveals which world-direction each joint rotates the end-effector.\n{'─' * 90}")
        try:
            tip_name = None
            for cand in ("jaw", "moving_jaw_so101_v1_link", "gripper", "gripper_link", "finger1_tip"):
                if cand in body_names:
                    tip_name = cand
                    break
            if tip_name is None:
                print(f"  No tip link found in body_names: {body_names}")
            else:
                print(f"  Tip link = {tip_name}\n  Action space dim = {n_act}\n")
                tip_idx = body_names.index(tip_name)
                # Snapshot the home qpos / qvel so we can restore between probes
                # WITHOUT calling env.reset (which re-composes the whole stage).
                home_qpos = robot.data.joint_pos.clone()
                home_qvel = robot.data.joint_vel.clone()
                joint_indices_all = torch.arange(n_joints, device=device)
                env_ids_all = torch.arange(args.num_envs, device=device)
                for i in range(min(n_act, n_joints)):
                    try:
                        # Restore robot state to home (light-weight, no scene rebuild).
                        robot.write_joint_state_to_sim(
                            position=home_qpos,
                            velocity=home_qvel,
                            joint_ids=joint_indices_all,
                            env_ids=env_ids_all,
                        )
                        # A few zero-action steps to let the controller resync.
                        for _ in range(3):
                            env.step(zero_action)
                        p0 = robot.data.body_pos_w[0, tip_idx].detach().cpu().numpy().tolist()
                        act = torch.zeros(args.num_envs, n_act, device=device)
                        act[0, i] = 0.1
                        for _ in range(args.probe_steps):
                            env.step(act)
                        p1 = robot.data.body_pos_w[0, tip_idx].detach().cpu().numpy().tolist()
                        d = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
                        mag = (d[0] ** 2 + d[1] ** 2 + d[2] ** 2) ** 0.5
                        jn = joint_names[i] if i < n_joints else f"action[{i}]"
                        sys.stdout.write(
                            f"  action[{i}] (joint={jn:18s})  +0.1 x {args.probe_steps}  "
                            f"Δworld_tip = {_fmt_vec(d, prec=4)}  |Δ|={mag:.4f}\n"
                        )
                        sys.stdout.flush()
                    except Exception as e_iter:
                        sys.stdout.write(f"  action[{i}] FAILED: {type(e_iter).__name__}: {e_iter}\n")
                        sys.stdout.flush()
        except Exception as e:
            print(f"  (probe failed: {e})")

    # ============================================================
    # OPTIONAL: dump first wrist RGB
    # ============================================================
    if args.dump_wrist_png and "wrist" in scene.sensors:
        try:
            import os
            from PIL import Image
            rgb = scene["wrist"].data.output["rgb"][0].detach().cpu().numpy()
            if rgb.shape[-1] == 4:
                rgb = rgb[..., :3]
            os.makedirs(os.path.dirname(args.dump_wrist_png) or ".", exist_ok=True)
            Image.fromarray(rgb).save(args.dump_wrist_png)
            print(f"\n[dumped wrist RGB → {args.dump_wrist_png}]")
        except Exception as e:
            print(f"\n[wrist RGB dump failed: {e}]")

    print(f"\n{line}\n  END AUDIT\n{line}\n")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
