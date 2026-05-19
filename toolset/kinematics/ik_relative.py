"""
ik_relative.py - drive the arm to a sequence of points specified as
offsets from the start pose. NO motor-offset calibration needed.

Why this works without calibration:
  The URDF FK + DLS-IK both operate in a coordinate frame where "0 = URDF
  zero". The actual motor angles use a different zero, so absolute IK
  targets don't map to physical meters. BUT relative deltas cancel the
  per-joint constant offset, so:
      target_urdf = ik_solve(start_urdf + delta)
      delta_motor = target_urdf - start_urdf
      final_motor = current_motor + delta_motor
  drives the arm by `delta_motor` regardless of the offset.

User frame for offsets: x_user = right+, y_user = forward+, z_user = up+.

Manual waypoints:
    python -m toolset.kinematics.ik_relative --port COM3 `
        --offsets 0.0 0.0 0.10 `
        --offsets 0.14 0.28 0.10 `
        --final_gripper mid --no-return_to_start

Preset (arm at grasp pose, start as-is): tighten jaw, lift 10 cm, move to bowl, open mid:
    python -m toolset.kinematics.ik_relative --port COM3 `
        --sequence lift_place --bowl_xy_m 0.16 0.32 --lift_m 0.10 `
        --final_gripper mid

Fixed horizontal nudge (meters right+, forward+ from start, at lift height):
    python -m toolset.kinematics.ik_relative --port COM3 `
        --sequence lift_place --delta_xy_m 0.15 0.15 --lift_m 0.10 `
        --final_gripper mid
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

DEPLOY = Path(__file__).resolve().parents[2] / "deploy"
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

from homes import DEFAULT_HOME_POSE, get_home_deg, list_home_poses  # noqa: E402
from robot_calibration import make_so101_follower_config  # noqa: E402
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: E402
from lerobot.robots.utils import make_robot_from_config  # noqa: E402

from toolset.kinematics.config import KinematicsConfig  # noqa: E402
from toolset.kinematics.ik import IKSolver  # noqa: E402
from toolset.kinematics.urdf_fk import CHAIN_JOINTS, SO101FK  # noqa: E402

# Servo degrees (LeRobot convention on this arm): ~0 closed, ~100 open.
GRIPPER_CLOSED_DEG = 0.0
GRIPPER_MID_DEG = 50.0
GRIPPER_OPEN_DEG = 100.0


def user_offset_to_urdf(off_user: np.ndarray) -> np.ndarray:
    """User (right+, forward+, up+) -> URDF (forward+x, left+y, up+z)."""
    x_user, y_user, z_user = off_user
    return np.array([+y_user, -x_user, +z_user], dtype=float)


def urdf_xyz_to_user(xyz_urdf: np.ndarray) -> np.ndarray:
    return np.array([-xyz_urdf[1], +xyz_urdf[0], +xyz_urdf[2]], dtype=float)


def resolve_gripper_deg(
    preset: str | None,
    explicit_deg: float | None,
) -> float | None:
    if explicit_deg is not None:
        return float(explicit_deg)
    if preset is None:
        return None
    key = preset.lower()
    if key == "closed":
        return GRIPPER_CLOSED_DEG
    if key == "mid":
        return GRIPPER_MID_DEG
    if key == "open":
        return GRIPPER_OPEN_DEG
    raise ValueError(f"unknown --final_gripper preset {preset!r} (use closed|mid|open)")


def plan_lift_place(
    fk: SO101FK,
    ik: IKSolver,
    q_urdf_start: np.ndarray,
    xyz_start_urdf: np.ndarray,
    xyz_start_user: np.ndarray,
    lift_m: float,
    *,
    target: str,
    bowl_xy_m: tuple[float, float] | None = None,
    delta_xy_m: tuple[float, float] | None = None,
    phones_rf: tuple[float, float] | None = None,
    phone_m: float = 0.075,
) -> tuple[list[np.ndarray], list[list[float]], dict]:
    """Plan lift then place. WP offsets are from start (not cumulative).

    After IK for the lift waypoint, FK at the lifted joints gives the true
    tip pose; horizontal move to the bowl uses (bowl_xy - xyz_lift_xy).
    """
    lift_m = float(lift_m)
    off0 = np.array([0.0, 0.0, lift_m], dtype=float)
    tgt0_urdf = xyz_start_urdf + user_offset_to_urdf(off0)
    res0 = ik.solve(tgt0_urdf, q_init=q_urdf_start.copy(), target=target, mode="position")
    if not res0.converged:
        raise RuntimeError(
            f"lift waypoint IK failed: {res0.reason} pos_err={res0.pos_err_m * 1000:.2f} mm"
        )

    fk_lift = fk.fk(res0.joints_rad, target=target)
    xyz_lift_urdf = fk_lift["position"]
    xyz_lift_user = urdf_xyz_to_user(xyz_lift_urdf)

    meta: dict = {
        "xyz_lift_user": xyz_lift_user,
        "lift_pos_err_mm": res0.pos_err_m * 1000.0,
        "lift_iters": res0.iters,
    }

    if delta_xy_m is not None:
        dx, dy = float(delta_xy_m[0]), float(delta_xy_m[1])
        meta["horiz_mode"] = "delta_xy_m"
        meta["delta_lift_to_target_xy"] = (dx, dy)
    elif bowl_xy_m is not None:
        bx, by = float(bowl_xy_m[0]), float(bowl_xy_m[1])
        dx = bx - float(xyz_lift_user[0])
        dy = by - float(xyz_lift_user[1])
        meta["horiz_mode"] = "bowl_xy_m"
        meta["bowl_xy_m"] = (bx, by)
        meta["delta_lift_to_bowl_xy"] = (dx, dy)
    elif phones_rf is not None:
        dx = float(phones_rf[0]) * float(phone_m)
        dy = float(phones_rf[1]) * float(phone_m)
        meta["horiz_mode"] = "phones"
        meta["delta_lift_to_target_xy"] = (dx, dy)
    else:
        raise ValueError("lift_place needs --delta_xy_m, --bowl_xy_m, or --phones")

    # IK targets are start + offset (user frame); z uses FK height after lift.
    z_off = float(xyz_lift_user[2] - xyz_start_user[2])
    off1 = np.array(
        [
            float(xyz_lift_user[0] + dx - xyz_start_user[0]),
            float(xyz_lift_user[1] + dy - xyz_start_user[1]),
            z_off,
        ],
        dtype=float,
    )

    tgt1_urdf = xyz_start_urdf + user_offset_to_urdf(off1)
    res1 = ik.solve(tgt1_urdf, q_init=res0.joints_rad.copy(), target=target, mode="position")
    if not res1.converged:
        raise RuntimeError(
            f"place waypoint IK failed: {res1.reason} pos_err={res1.pos_err_m * 1000:.2f} mm"
        )

    offsets = [off0.tolist(), off1.tolist()]
    plans = [res0.joints_rad.copy(), res1.joints_rad.copy()]
    meta["place_pos_err_mm"] = res1.pos_err_m * 1000.0
    meta["place_iters"] = res1.iters
    return plans, offsets, meta


def _joint_traj(
    q_from: np.ndarray,
    q_to: np.ndarray,
    max_step_rad: float,
    *,
    ease: str,
) -> np.ndarray:
    """Joint-space ramp; smoothstep eases accel at start/end."""
    delta = q_to - q_from
    n_steps = max(1, int(np.ceil(np.max(np.abs(delta)) / max_step_rad)))
    if ease == "linear":
        return np.linspace(q_from, q_to, n_steps + 1)[1:]
    # Per-joint smoothstep: zero velocity at endpoints, fewer jerks on hardware.
    t = np.linspace(0.0, 1.0, n_steps + 1, dtype=float)[1:]
    u = t * t * (3.0 - 2.0 * t)
    return q_from + u[:, np.newaxis] * delta


def ramp_joints(
    robot,
    q_from: np.ndarray,
    q_to: np.ndarray,
    *,
    gripper_deg: float,
    max_step_rad: float,
    period: float,
    ease: str = "smooth",
    label: str = "",
) -> None:
    traj = _joint_traj(q_from, q_to, max_step_rad, ease=ease)
    n_steps = len(traj)
    if label:
        print(f"  -> {label}: {n_steps} steps ({ease})")
    for step_q in traj:
        t0 = time.time()
        cmd = {f"{n}.pos": float(np.rad2deg(step_q[j])) for j, n in enumerate(CHAIN_JOINTS)}
        cmd["gripper.pos"] = float(gripper_deg)
        robot.send_action(cmd)
        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)


MOTOR_NAMES = [*CHAIN_JOINTS, "gripper"]


def read_motor_deg(robot) -> np.ndarray:
    raw = robot.get_observation()
    return np.array([float(raw[f"{n}.pos"]) for n in MOTOR_NAMES], dtype=np.float32)


def ramp_to_home_deg(
    robot,
    home_deg: np.ndarray,
    *,
    fps: int,
    max_delta_deg: float,
    tol_deg: float = 0.5,
    max_steps: int = 600,
) -> None:
    """Ramp all six joints to a deploy/homes.py preset."""
    target = np.asarray(home_deg, dtype=np.float32).reshape(len(MOTOR_NAMES))
    current = read_motor_deg(robot)
    for _ in range(max_steps):
        delta = target - current
        if float(np.max(np.abs(delta))) <= tol_deg:
            break
        current = current + np.clip(delta, -max_delta_deg, max_delta_deg)
        robot.send_action({f"{n}.pos": float(current[i]) for i, n in enumerate(MOTOR_NAMES)})
        time.sleep(1.0 / fps)
    else:
        print("[ik-rel] home: warning: did not fully converge within max_steps")
    final = read_motor_deg(robot)
    err = final - target
    print(f"[ik-rel] home target (deg):  {target.round(2).tolist()}")
    print(f"[ik-rel] home reached (deg): {final.round(2).tolist()}")
    print(
        f"[ik-rel] home error (deg):   {err.round(2).tolist()}  "
        f"max={float(np.max(np.abs(err))):.1f}"
    )


def hold_pose(
    robot,
    q_motor_rad: np.ndarray,
    gripper_deg: float,
    hold_s: float,
    period: float,
) -> None:
    """Keep commanding the same pose (fights servo drift / drop on disconnect)."""
    if hold_s <= 0:
        return
    n_steps = max(1, int(hold_s / period))
    cmd = {f"{n}.pos": float(np.rad2deg(q_motor_rad[j])) for j, n in enumerate(CHAIN_JOINTS)}
    cmd["gripper.pos"] = float(gripper_deg)
    for _ in range(n_steps):
        t0 = time.time()
        robot.send_action(cmd)
        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)


def ramp_gripper(
    robot,
    gripper_from_deg: float,
    gripper_to_deg: float,
    *,
    hold_s: float,
    period: float = 1.0 / 30.0,
    ease: str = "smooth",
) -> None:
    """Open/close gripper while arm joints stay fixed."""
    n_steps = max(1, int(hold_s / period))
    if ease == "linear":
        grip_vals = np.linspace(gripper_from_deg, gripper_to_deg, n_steps + 1)[1:]
    else:
        t = np.linspace(0.0, 1.0, n_steps + 1, dtype=float)[1:]
        u = t * t * (3.0 - 2.0 * t)
        grip_vals = gripper_from_deg + u * (gripper_to_deg - gripper_from_deg)
    for g in grip_vals:
        t0 = time.time()
        obs = robot.get_observation()
        cmd = {f"{n}.pos": float(obs[f"{n}.pos"]) for n in CHAIN_JOINTS}
        cmd["gripper.pos"] = float(g)
        robot.send_action(cmd)
        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--port", default="COM3")
    p.add_argument("--camera_index", type=int, default=1)
    p.add_argument("--sequence", choices=["lift_place"], default=None,
                   help="Built-in waypoint list (see module docstring).")
    p.add_argument("--lift_m", type=float, default=0.10,
                   help="lift_place: vertical offset in meters (default 0.10).")
    p.add_argument("--bowl_xy_m", type=float, nargs=2, metavar=("X", "Y"),
                   help="lift_place: bowl position in user frame (right+, forward+).")
    p.add_argument("--delta_xy_m", type=float, nargs=2, metavar=("dX", "dY"),
                   help="lift_place: move right+ and forward+ in meters from start (at lift height).")
    p.add_argument("--phones", type=float, nargs=2, metavar=("RIGHT", "FWD"),
                   help="lift_place: horizontal move as N phone-lengths right and forward.")
    p.add_argument("--phone_m", type=float, default=0.075,
                   help="Meters per 'phone length' (default 0.075).")
    p.add_argument("--offsets", action="append", type=float, nargs=3,
                   metavar=("dX", "dY", "dZ"),
                   help="Waypoint as RELATIVE offset (m) in user frame from START pose. "
                        "Repeat for multiple waypoints (not cumulative).")
    p.add_argument("--speed", choices=["slow", "fluid", "smooth", "normal", "fast", "max"],
                   default="fluid",
                   help="Motion preset (default fluid; use max for fastest).")
    p.add_argument("--ease", choices=["smooth", "linear"], default="smooth",
                   help="Joint/gripper ramp easing (default smooth = smoothstep).")
    p.add_argument("--target", default="gripper_tip",
                   choices=["wrist", "wrist_roll", "gripper_frame", "gripper_tip"])
    p.add_argument("--return_to_start", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="After waypoints, drive arm joints back to start (default: yes, "
                        "no for lift_place / when opening gripper).")
    p.add_argument("--final_gripper", choices=["closed", "mid", "open"], default=None,
                   help="After last waypoint, ramp gripper to preset (arm stays put).")
    p.add_argument("--final_gripper_deg", type=float, default=None,
                   help="Explicit servo degrees for gripper after last waypoint.")
    p.add_argument("--final_gripper_hold_s", type=float, default=None,
                   help="Seconds to ramp final gripper (default 0.6 for lift_place, 1.5 else).")
    p.add_argument("--tighten_grip", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="lift_place: squeeze jaw closed before lifting (default: on).")
    p.add_argument("--tighten_grip_deg", type=float, default=GRIPPER_CLOSED_DEG,
                   help="Servo degrees for jaw tighten (default 0 = closed).")
    p.add_argument("--tighten_grip_hold_s", type=float, default=None,
                   help="Seconds to ramp jaw closed (default 0.5 for lift_place, 1.0 else).")
    p.add_argument("--hold_pose_s", type=float, default=None,
                   help="After place, keep commanding final pose before disconnect "
                        "(default 3.0 for lift_place, 0 else).")
    p.add_argument("--go_home", action=argparse.BooleanOptionalAction, default=None,
                   help="After hold, ramp to deploy/homes.py preset (default on for lift_place).")
    p.add_argument("--home_pose", default=DEFAULT_HOME_POSE, choices=list_home_poses(),
                   help=f"Home preset when --go_home (default {DEFAULT_HOME_POSE}).")
    p.add_argument("--dry_run", action="store_true",
                   help="IK plan only (no motion). With --port, reads the current "
                        "pose as-is; without a robot, use --start_q_deg.")
    p.add_argument("--start_q_deg", type=float, nargs=6, default=None,
                   metavar=("PAN", "LIFT", "ELBOW", "WFLEX", "WROLL", "GRIP"),
                   help="Offline dry_run only: substitute start pose (deg) when "
                        "the arm is not connected.")
    args = p.parse_args()

    final_grip_deg = resolve_gripper_deg(args.final_gripper, args.final_gripper_deg)

    if args.sequence == "lift_place":
        if args.return_to_start is None:
            args.return_to_start = False
        if final_grip_deg is None:
            final_grip_deg = GRIPPER_MID_DEG
        if args.tighten_grip_hold_s is None:
            args.tighten_grip_hold_s = 0.5
        if args.final_gripper_hold_s is None:
            args.final_gripper_hold_s = 0.6
        if args.hold_pose_s is None:
            args.hold_pose_s = 3.0
        if args.go_home is None:
            args.go_home = True
    else:
        if args.tighten_grip_hold_s is None:
            args.tighten_grip_hold_s = 1.0
        if args.final_gripper_hold_s is None:
            args.final_gripper_hold_s = 1.5
        if args.hold_pose_s is None:
            args.hold_pose_s = 0.0
        if args.go_home is None:
            args.go_home = False
        if args.return_to_start is None:
            args.return_to_start = True

    if final_grip_deg is not None and args.return_to_start:
        print("[ik-rel] note: opening gripper at end; use --no-return_to_start to stay over bowl.")

    if args.sequence is None and not args.offsets:
        p.error("Provide --offsets and/or --sequence lift_place")

    kcfg = KinematicsConfig.load()
    fk = SO101FK(kcfg.urdf_path, gripper_tip_offset=kcfg.gripper_tip_offset_m)
    ik = IKSolver(kcfg, fk)
    speed_cfg = kcfg.path["speeds"][args.speed]
    fps = int(speed_cfg["fps"])
    max_step_rad = float(np.deg2rad(speed_cfg["max_delta_deg_per_step"]))
    period = 1.0 / fps

    robot = None
    if args.dry_run and args.start_q_deg is not None:
        q_motor = np.deg2rad(np.array(args.start_q_deg, dtype=float))
        print("[ik-rel] DRY RUN (offline). --start_q_deg = "
              f"{np.rad2deg(q_motor).round(1).tolist()}")
    elif args.dry_run:
        cfg = make_so101_follower_config(
            args.port,
            cameras={"base_camera": OpenCVCameraConfig(
                index_or_path=args.camera_index, fps=30, width=640, height=480)},
            use_degrees=True,
        )
        robot = make_robot_from_config(cfg)
        print(f"[ik-rel] DRY RUN: reading start pose as-is from {args.port} ...")
        robot.connect()
        obs = robot.get_observation()
        q_motor = np.array(
            [float(obs[f"{n}.pos"]) for n in CHAIN_JOINTS] + [float(obs["gripper.pos"])],
            dtype=float,
        )
        q_motor = np.deg2rad(q_motor)
        print(f"[ik-rel] start motor (deg): {np.rad2deg(q_motor).round(2).tolist()}")
        robot.disconnect()
        robot = None
    else:
        cfg = make_so101_follower_config(
            args.port,
            cameras={"base_camera": OpenCVCameraConfig(
                index_or_path=args.camera_index, fps=30, width=640, height=480)},
            use_degrees=True,
        )
        robot = make_robot_from_config(cfg)
        print(f"[ik-rel] connecting to {args.port} ...")
        robot.connect()
        obs = robot.get_observation()
        q_motor = np.array(
            [float(obs[f"{n}.pos"]) for n in CHAIN_JOINTS] + [float(obs["gripper.pos"])],
            dtype=float,
        )
        q_motor = np.deg2rad(q_motor)
        print(f"[ik-rel] start motor (deg): {np.rad2deg(q_motor).round(2).tolist()}")

    q_urdf_now = q_motor[:5].copy()
    gripper_now_deg = float(np.rad2deg(q_motor[5]))
    start_q_motor = q_motor.copy()

    fk_start = fk.fk(q_urdf_now, target=args.target)
    xyz_start_urdf = fk_start["position"]
    xyz_start_user = urdf_xyz_to_user(xyz_start_urdf)
    print(f"[ik-rel] start FK xyz (URDF):  {xyz_start_urdf.round(4).tolist()} m")
    print(f"[ik-rel] start FK xyz (user):  {xyz_start_user.round(4).tolist()} m "
          f"(right+, forward+, up+)")
    print("[ik-rel] waypoints: offsets from this start pose (not cumulative).")

    plans: list[np.ndarray] = []
    if args.sequence == "lift_place":
        bowl = tuple(args.bowl_xy_m) if args.bowl_xy_m is not None else None
        delta_xy = tuple(args.delta_xy_m) if args.delta_xy_m is not None else None
        phones = tuple(args.phones) if args.phones is not None else None
        if args.tighten_grip:
            print(f"[ik-rel] tighten jaw -> {args.tighten_grip_deg:.1f} deg "
                  f"({args.tighten_grip_hold_s:.1f}s) before lift")
        try:
            plans, offset_list, meta = plan_lift_place(
                fk, ik, q_urdf_now, xyz_start_urdf, xyz_start_user, args.lift_m,
                target=args.target,
                bowl_xy_m=bowl, delta_xy_m=delta_xy, phones_rf=phones, phone_m=args.phone_m,
            )
        except RuntimeError as exc:
            print(f"[ik-rel] {exc}")
            if robot is not None:
                robot.disconnect()
            return
        xyz_lift = meta["xyz_lift_user"]
        print(f"[ik-rel] sequence lift_place: lift={args.lift_m} m")
        print(f"[ik-rel]   FK after lift (user): {xyz_lift.round(4).tolist()} m")
        if meta.get("horiz_mode") == "bowl_xy_m":
            dlb = meta["delta_lift_to_bowl_xy"]
            print(f"[ik-rel]   bowl_xy_m={meta['bowl_xy_m']}  "
                  f"horiz from lifted tip -> bowl: dX={dlb[0]:+.3f} dY={dlb[1]:+.3f} m")
        elif "delta_lift_to_target_xy" in meta:
            dlt = meta["delta_lift_to_target_xy"]
            print(f"[ik-rel]   horiz from lifted tip: dX={dlt[0]:+.3f} dY={dlt[1]:+.3f} m")
        print(f"[ik-rel]   offset_wp1 (from start)={offset_list[1]}")
        if final_grip_deg is not None:
            print(f"[ik-rel]   final gripper -> {final_grip_deg:.1f} deg")
        if args.hold_pose_s and args.hold_pose_s > 0:
            print(f"[ik-rel]   hold final pose {args.hold_pose_s:.1f}s before disconnect")
        if args.go_home:
            print(f"[ik-rel]   then go_home -> {args.home_pose!r}")
        q_prev = q_urdf_now.copy()
        for i, (off_user, q_tgt) in enumerate(zip(offset_list, plans)):
            off_user = np.array(off_user, dtype=float)
            tgt_urdf = xyz_start_urdf + user_offset_to_urdf(off_user)
            tgt_user = urdf_xyz_to_user(tgt_urdf)
            delta_rad = q_tgt - q_prev
            err_mm = meta["lift_pos_err_mm"] if i == 0 else meta["place_pos_err_mm"]
            iters = meta["lift_iters"] if i == 0 else meta["place_iters"]
            print(f"[ik-rel] WP {i}: offset_user={off_user.tolist()}  "
                  f"target_user={tgt_user.round(3).tolist()}  iters={iters}  "
                  f"pos_err={err_mm:.2f} mm  "
                  f"delta_deg={np.rad2deg(delta_rad).round(2).tolist()}")
            q_prev = q_tgt.copy()
    else:
        offset_list = [list(o) for o in args.offsets]
        q_init = q_urdf_now.copy()
        for i, off_user in enumerate(offset_list):
            off_user = np.array(off_user, dtype=float)
            off_urdf = user_offset_to_urdf(off_user)
            tgt_urdf = xyz_start_urdf + off_urdf
            res = ik.solve(tgt_urdf, q_init=q_init, target=args.target, mode="position")
            if not res.converged:
                print(f"[ik-rel] WP {i} target_urdf={tgt_urdf.round(3).tolist()} "
                      f"FAILED: reason={res.reason}  pos_err={res.pos_err_m * 1000:.2f} mm")
                if robot is not None:
                    robot.disconnect()
                return
            delta_rad = res.joints_rad - q_init
            tgt_user = urdf_xyz_to_user(tgt_urdf)
            print(f"[ik-rel] WP {i}: offset_user={off_user.tolist()}  "
                  f"target_user={tgt_user.round(3).tolist()}  iters={res.iters}  "
                  f"pos_err={res.pos_err_m * 1000:.2f} mm  "
                  f"delta_deg={np.rad2deg(delta_rad).round(2).tolist()}")
            plans.append(res.joints_rad.copy())
            q_init = res.joints_rad.copy()

    if args.dry_run or robot is None:
        print("[ik-rel] dry run complete.")
        return

    print(f"\n[ik-rel] executing {len(plans)} waypoint(s) at speed={args.speed} "
          f"({fps} Hz, max {speed_cfg['max_delta_deg_per_step']:.1f} deg/step).")

    obs = robot.get_observation()
    q_motor_now = np.deg2rad(
        np.array([float(obs[f"{n}.pos"]) for n in CHAIN_JOINTS], dtype=float))
    gripper_exec_deg = gripper_now_deg

    if args.sequence == "lift_place" and args.tighten_grip:
        print(f"[ik-rel] tightening jaw {gripper_now_deg:.1f} -> "
              f"{args.tighten_grip_deg:.1f} deg ...")
        ramp_gripper(
            robot, gripper_now_deg, args.tighten_grip_deg,
            hold_s=args.tighten_grip_hold_s,
            period=period,
            ease=args.ease,
        )
        gripper_exec_deg = float(args.tighten_grip_deg)
        gripper_now_deg = gripper_exec_deg

    print(f"[ik-rel] gripper held at {gripper_exec_deg:.1f} deg during arm motion.")

    for i, q_urdf_target in enumerate(plans):
        delta = q_urdf_target - q_urdf_now
        q_motor_target = q_motor_now + delta
        ramp_joints(
            robot, q_motor_now, q_motor_target,
            gripper_deg=gripper_exec_deg,
            max_step_rad=max_step_rad,
            period=period,
            ease=args.ease,
            label=f"WP {i}",
        )
        q_motor_now = q_motor_target.copy()
        q_urdf_now = q_urdf_target.copy()

    if final_grip_deg is not None:
        print(f"[ik-rel] ramping gripper {gripper_now_deg:.1f} -> {final_grip_deg:.1f} deg "
              f"over {args.final_gripper_hold_s:.1f}s (arm fixed).")
        ramp_gripper(
            robot, gripper_now_deg, final_grip_deg,
            hold_s=args.final_gripper_hold_s,
            period=period,
            ease=args.ease,
        )
        gripper_now_deg = final_grip_deg

    if args.return_to_start:
        print("[ik-rel] returning arm to start pose (gripper unchanged).")
        ramp_joints(
            robot, q_motor_now, start_q_motor[:5],
            gripper_deg=gripper_now_deg,
            max_step_rad=max_step_rad,
            period=period,
            ease=args.ease,
            label="return",
        )

    if args.hold_pose_s and args.hold_pose_s > 0:
        print(f"[ik-rel] holding pose {args.hold_pose_s:.1f}s (arm + gripper) ...")
        hold_pose(
            robot, q_motor_now, gripper_now_deg,
            hold_s=args.hold_pose_s,
            period=period,
        )

    if args.go_home:
        home_deg = get_home_deg(args.home_pose)
        print(f"[ik-rel] ramping to home preset {args.home_pose!r} ...")
        ramp_to_home_deg(
            robot,
            home_deg,
            fps=fps,
            max_delta_deg=float(speed_cfg["max_delta_deg_per_step"]),
        )

    robot.disconnect()
    print("[ik-rel] disconnected.")


if __name__ == "__main__":
    main()
