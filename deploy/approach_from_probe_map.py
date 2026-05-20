"""
approach_from_probe_map.py — map pixel -> hover -> side offset -> down (then grasp).

Uses probe session map + grasp_motor_deg from nearest trial for approach angle and
optional IK warm-start. Default IK uses position_vertical + staged retry if FK miss.
Detection: session learned_hsv → bright-top mask → wall-line square fit (same as probe postprocess).
Map pixels: mapping_samples_refined.csv (from frame_*_square.json centroids).

Typical workflow:
  1) python -m toolset.perception.probe_map_feasibility --session_dir deploy/_snaps/probe_XXX
  2) python deploy/approach_from_probe_map.py --session_dir ... --dry_run [--du U] [--dv V]
  3) python deploy/approach_from_probe_map.py --session_dir ... --execute --from_home

Usage:
    python deploy/approach_from_probe_map.py `
        --session_dir deploy/_snaps/probe_1779205245 `
        --dry_run --du -108 --dv 15

    python deploy/approach_from_probe_map.py `
        --session_dir deploy/_snaps/probe_1779205245 `
        --execute --from_home
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_DEPLOY = Path(__file__).resolve().parent
_PROJECT = _DEPLOY.parent
if str(_DEPLOY) not in sys.path:
    sys.path.insert(0, str(_DEPLOY))
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from homes import get_home_deg  # noqa: E402
from robot_calibration import make_so101_follower_config  # noqa: E402
from wrist_camera_config import make_wrist_opencv_camera_config  # noqa: E402
from lerobot.robots.utils import make_robot_from_config  # noqa: E402

from toolset.kinematics.config import KinematicsConfig  # noqa: E402
from toolset.kinematics.motor_to_urdf import MotorToUrdfConfig  # noqa: E402
from toolset.kinematics.urdf_fk import SO101FK  # noqa: E402
from toolset.kinematics.ik_relative import urdf_xyz_to_user  # noqa: E402
from toolset.perception.approach_plan import (  # noqa: E402
    offsets_from_tip_start,
    plan_approach_waypoints,
    user_xyz_to_urdf,
)
from toolset.perception.probe_map_data import estimate_xyz_from_pixel  # noqa: E402
from toolset.perception.refined_cube_detect import (  # noqa: E402
    detect_refined_pixel,
    pixel_to_du_dv,
)
from toolset.perception.probe_camera_vs_fk import ramp_to_target_deg, MOTOR_NAMES  # noqa: E402


def _bgr_from_obs(obs) -> np.ndarray | None:
    bgr = obs.get("observation.images.base_camera")
    if bgr is None:
        for k, v in obs.items():
            if "base_camera" in str(k) and hasattr(v, "shape"):
                bgr = v
                break
    if bgr is None:
        return None
    if hasattr(bgr, "cpu"):
        bgr = bgr.cpu().numpy()
    if bgr.ndim == 3 and bgr.shape[0] == 3:
        bgr = np.transpose(bgr, (1, 2, 0))
    if bgr.dtype != np.uint8:
        bgr = (np.clip(bgr, 0, 1) * 255).astype(np.uint8)
    if bgr.shape[2] == 3:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
    return bgr


def detect_du_dv_from_camera(
    robot,
    color: str,
    session_dir: Path,
    *,
    n_frames: int = 5,
    save_debug: Path | None = None,
) -> tuple[float, float] | None:
    """learned HSV bright-top + wall square fit (same as probe postprocess)."""
    pixels = []
    for i in range(n_frames):
        obs = robot.get_observation()
        bgr = _bgr_from_obs(obs)
        if bgr is None:
            time.sleep(0.1)
            continue
        det = detect_refined_pixel(bgr, color, session_dir=session_dir)
        if det.pixel is not None:
            pixels.append(det.pixel)
            if save_debug is not None and i == 0 and det.final_mask is not None:
                save_debug.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_debug / "live_bright_mask.png"), det.bright_mask or det.final_mask)
                cv2.imwrite(str(save_debug / "live_final_mask.png"), det.final_mask)
        time.sleep(0.15)
    if not pixels:
        return None
    med = np.median(np.asarray(pixels), axis=0)
    return pixel_to_du_dv(float(med[0]), float(med[1]))


def tip_user_from_robot(robot, fk: SO101FK) -> np.ndarray:
    obs = robot.get_observation()
    motor = np.array([float(obs[f"{n}.pos"]) for n in MOTOR_NAMES], dtype=float)
    mcfg = MotorToUrdfConfig.load()
    q = mcfg.motor_to_urdf_rad(motor)
    return urdf_xyz_to_user(fk.fk(q, target="gripper_tip")["position"])


def tip_user_from_home_deg(fk: SO101FK, home_deg: list[float] | np.ndarray) -> np.ndarray:
    mcfg = MotorToUrdfConfig.load()
    q = mcfg.motor_to_urdf_rad(np.asarray(home_deg, dtype=float))
    return urdf_xyz_to_user(fk.fk(q, target="gripper_tip")["position"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session_dir", type=Path, required=True)
    p.add_argument("--port", default="COM3")
    p.add_argument("--camera_index", type=int, default=0)
    p.add_argument("--color", default="yellow")
    p.add_argument("--home_pose", default="eval1_rest")
    p.add_argument("--du", type=float, default=None)
    p.add_argument("--dv", type=float, default=None)
    p.add_argument("--pixel", nargs=2, type=float, metavar=("U", "V"))
    p.add_argument("--k", type=int, default=3, help="kNN for map lookup")
    p.add_argument("--hover_above_m", type=float, default=0.03)
    p.add_argument("--side_offset_m", type=float, default=0.03)
    p.add_argument("--descend_m", type=float, default=0.01)
    p.add_argument("--side_sign", type=float, default=-1.0,
                   help="Flip approach side if offset wrong way (default -1).")
    p.add_argument("--dry_run", action="store_true", default=True,
                   help="Plan only, print ik_relative command (default).")
    p.add_argument("--execute", action="store_true",
                   help="Run ik_relative after planning.")
    p.add_argument("--from_home", action="store_true",
                   help="Ramp to home before capture / move.")
    p.add_argument("--speed", default="fluid")
    p.add_argument("--no_return", action="store_true",
                   help="Pass --no-return_to_start to ik_relative.")
    p.add_argument("--raw_map", action="store_true",
                   help="Use original mapping_samples.csv pixels (not refined).")
    p.add_argument("--save_debug", type=Path, default=None,
                   help="Save live_final_mask.png on first detect frame.")
    p.add_argument("--go_home", action=argparse.BooleanOptionalAction, default=True,
                   help="After IK waypoints, ramp to home preset (default: on).")
    p.add_argument("--prefer_vertical_grasp", action=argparse.BooleanOptionalAction, default=True,
                   help="Choose grasp exemplar from kNN neighbors with vertical wrist bias.")
    p.add_argument("--vertical_wrist_target_deg", type=float, default=90.0,
                   help="Desired wrist_flex deg when prefer_vertical_grasp is on (default 90).")
    p.add_argument("--vertical_bias_px_per_deg", type=float, default=0.8,
                   help="Penalty scale for wrist_flex distance from target when selecting grasp.")
    p.add_argument("--ik_mode", choices=["position", "position_vertical"], default="position_vertical",
                   help="Passed to ik_relative: position only vs position + gripper down.")
    p.add_argument("--no_ik_seed_grasp", action="store_true",
                   help="Do not pass map grasp_motor_deg as IK warm-start to ik_relative.")
    args = p.parse_args()
    if args.execute:
        args.dry_run = False

    session_dir = args.session_dir.resolve()
    kcfg = KinematicsConfig.load()
    fk = SO101FK(kcfg.urdf_path, gripper_tip_offset=kcfg.gripper_tip_offset_m)

    du, dv = args.du, args.dv
    if args.pixel is not None:
        du, dv = pixel_to_du_dv(args.pixel[0], args.pixel[1])

    tip_start = None
    home_tip = tip_user_from_home_deg(fk, get_home_deg(args.home_pose))
    robot = None
    if du is None or dv is None:
        if args.dry_run and not args.execute:
            raise SystemExit("Offline dry_run needs --du/--dv or --pixel (run camera path with --execute).")
        cfg = make_so101_follower_config(
            args.port,
            cameras={"base_camera": make_wrist_opencv_camera_config(args.camera_index)},
            use_degrees=True,
        )
        robot = make_robot_from_config(cfg)
        robot.connect()
        if args.from_home:
            ramp_to_target_deg(robot, get_home_deg(args.home_pose))
            time.sleep(0.3)
        out = detect_du_dv_from_camera(
            robot, args.color, session_dir,
            save_debug=args.save_debug,
        )
        if out is None:
            robot.disconnect()
            robot = None
            raise SystemExit("No cube pixel detected at home.")
        du, dv = out
        tip_start = tip_user_from_robot(robot, fk)
        # If we intentionally start from home, plan from FK(home) so reconnect drift
        # before ik_relative does not create giant wrong offsets.
        if args.from_home and args.execute:
            tip_start = home_tip.copy()
        if not args.execute:
            robot.disconnect()
    elif args.execute:
        cfg = make_so101_follower_config(args.port, use_degrees=True)
        robot = make_robot_from_config(cfg)
        robot.connect()
        if args.from_home:
            ramp_to_target_deg(robot, get_home_deg(args.home_pose))
            time.sleep(0.3)
        tip_start = tip_user_from_robot(robot, fk)
        if args.from_home:
            tip_start = home_tip.copy()

    use_refined = not args.raw_map
    est = estimate_xyz_from_pixel(session_dir, du, dv, k=args.k, use_refined=use_refined)
    xyz = est["fk_xyz_user_m"]
    grasp = None
    grasp_src_trial = None
    if est["neighbors"]:
        if args.prefer_vertical_grasp:
            scored = []
            dists = est.get("neighbor_dist_px", [])
            for i, n in enumerate(est["neighbors"]):
                g = n.get("grasp_motor_deg")
                if g is None:
                    continue
                dist_px = float(dists[i]) if i < len(dists) else 1e6
                wrist_flex = float(g[3]) if len(g) >= 4 else args.vertical_wrist_target_deg
                vertical_pen = abs(wrist_flex - args.vertical_wrist_target_deg)
                score = dist_px + args.vertical_bias_px_per_deg * vertical_pen
                scored.append((score, g, n.get("trial"), dist_px, wrist_flex))
            if scored:
                scored.sort(key=lambda x: x[0])
                _, grasp, grasp_src_trial, src_dist_px, src_wflex = scored[0]
                print(
                    f"[approach] grasp source trial={grasp_src_trial} "
                    f"dist={src_dist_px:.1f}px wrist_flex={src_wflex:.1f}deg "
                    f"(target {args.vertical_wrist_target_deg:.1f})",
                )
        if grasp is None:
            n0 = est["neighbors"][0]
            grasp = n0.get("grasp_motor_deg")
            grasp_src_trial = n0.get("trial")
    if grasp is None:
        raise SystemExit("No grasp_motor_deg in map neighbors.")

    plan = plan_approach_waypoints(
        kcfg,
        cube_xy_user=xyz[:2],
        grasp_motor_deg=grasp,
        hover_above_m=args.hover_above_m,
        side_offset_m=args.side_offset_m,
        descend_m=args.descend_m,
        side_sign=args.side_sign,
    )

    offline = tip_start is None
    if offline:
        home_deg = get_home_deg(args.home_pose)
        tip_start = tip_user_from_home_deg(fk, home_deg)
        print(f"[approach] offline — tip_start from FK({args.home_pose}), "
              "ik offsets from that pose (not cumulative).")

    offsets = offsets_from_tip_start(tip_start, plan)
    stage_deltas = [
        (plan["hover_user"] - tip_start).tolist(),
        (plan["side_user"] - plan["hover_user"]).tolist(),
        (plan["down_user"] - plan["side_user"]).tolist(),
    ]

    meta = {
        "session_dir": str(session_dir),
        "du_dv": [du, dv],
        "neighbor_trials": est["neighbor_trials"],
        "fk_xyz_user_m": xyz.tolist(),
        "tip_start_user_m": tip_start.tolist(),
        "plan": {k: np.asarray(v).tolist() for k, v in plan.items() if isinstance(v, np.ndarray)},
        "offsets_user_from_start": offsets,
        "stage_deltas_user": stage_deltas,
        "stages": {
            "hover": plan["hover_user"].tolist(),
            "side": plan["side_user"].tolist(),
            "down": plan["down_user"].tolist(),
        },
    }
    plan_path = session_dir / "approach_plan.json"
    plan_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    ws = [
        kcfg.in_workspace(user_xyz_to_urdf(plan["hover_user"])),
        kcfg.in_workspace(user_xyz_to_urdf(plan["side_user"])),
        kcfg.in_workspace(user_xyz_to_urdf(plan["down_user"])),
    ]

    print(f"[approach] session {session_dir}  map={'refined' if use_refined else 'raw'}")
    print(f"[approach] pixel du={du:+.0f} dv={dv:+.0f}  neighbors={est['neighbor_trials']}")
    print(f"[approach] map fk (user m): {xyz.round(4).tolist()}")
    print(f"[approach] hover (user m):  {plan['hover_user'].round(4).tolist()}  ws={ws[0]}")
    print(f"[approach] side  (user m):  {plan['side_user'].round(4).tolist()}  ws={ws[1]}")
    print(f"[approach] down  (user m):  {plan['down_user'].round(4).tolist()}  ws={ws[2]}")
    print(f"[approach] approach_dir_xy: {plan['approach_dir_xy'].round(3).tolist()}")
    print(f"[approach] wrote {plan_path}")
    print(f"[approach] tip_start (user m): {tip_start.round(4).tolist()}")
    print("[approach] stage motion (for intuition): "
          f"to_hover {np.array(stage_deltas[0]).round(3).tolist()}  "
          f"to_side {np.array(stage_deltas[1]).round(3).tolist()}  "
          f"to_down {np.array(stage_deltas[2]).round(3).tolist()}")

    cmd = ["python", "-m", "toolset.kinematics.ik_relative", "--port", args.port,
           "--speed", args.speed, "--ik_mode", args.ik_mode, "--ik_staged"]
    if grasp is not None and not args.no_ik_seed_grasp:
        g6 = [float(x) for x in grasp[:6]]
        cmd.extend(["--ik_seed_q_deg", *[f"{v:.2f}" for v in g6]])
    for off in offsets:
        cmd.extend(["--offsets", f"{off[0]:.5f}", f"{off[1]:.5f}", f"{off[2]:.5f}"])
    if args.no_return:
        cmd.append("--no-return_to_start")
    if args.go_home:
        cmd.extend(["--go_home", "--home_pose", args.home_pose])
    else:
        cmd.append("--no-go_home")
    print("\n[approach] ik_relative command:")
    print("  " + " ".join(cmd))

    if args.execute:
        # Release serial port before spawning ik_relative, which opens the same port.
        if robot is not None:
            robot.disconnect()
            robot = None
        # Normalize unknown arm state: when current pose drifted, relative offsets may
        # project to impossible targets. Force a quick home reset first.
        if args.from_home:
            home_cmd = [
                "python", "-m", "toolset.kinematics.ik_relative",
                "--port", args.port,
                "--speed", args.speed,
                "--offsets", "0", "0", "0",
                "--go_home",
                "--home_pose", args.home_pose,
                "--no-return_to_start",
            ]
            print("\n[approach] pre-reset: force home before target move ...")
            rc_home = subprocess.call(home_cmd, cwd=str(_PROJECT))
            if rc_home != 0:
                raise SystemExit(f"pre-home reset failed with code {rc_home}")
        if args.from_home:
            print(f"[approach] planning assumes start at home FK({args.home_pose}).")
        print("\n[approach] executing ik_relative ...")
        rc = subprocess.call(cmd, cwd=str(_PROJECT))
        if rc != 0:
            raise SystemExit(f"ik_relative failed with code {rc}")
        print("[approach] done — close gripper manually or run grasp pose from probe trial.")
    else:
        print("\n[approach] dry_run only. Check probe_map_feasibility first, then --execute.")


if __name__ == "__main__":
    main()
