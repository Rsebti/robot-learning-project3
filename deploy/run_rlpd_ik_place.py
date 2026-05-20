#!/usr/bin/env python3
"""
run_rlpd_ik_place.py — RLPD grasp rollout, then ik_relative lift_place to bowl.

Phase 1: deploy/infer_rlpd.py until handoff (SPACE, grasp, or --handoff_step).
Phase 2: toolset.kinematics.ik_relative --sequence lift_place

Usage:
  python deploy/run_rlpd_ik_place.py \\
      --ckpt C:\\Users\\hugod\\squint-rlpd\\runs\\<run>\\ckpt_best.pt \\
      --bowl_xyz 0.16 0.32 0.0 --goal_color 3

  # Mid-rollout: press SPACE when holding cube, then IK runs automatically:
  python deploy/run_rlpd_ik_place.py --name rlpd --handoff_on_space \\
      --bowl_xy_m 0.16 0.32 --goal_color 3

  # IK only (arm already at grasp):
  python deploy/run_rlpd_ik_place.py --ik_only --bowl_xy_m 0.16 0.32
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_DEPLOY = Path(__file__).resolve().parent
_PROJECT = _DEPLOY.parent
if str(_DEPLOY) not in sys.path:
    sys.path.insert(0, str(_DEPLOY))

from rlpd_infer_common import probe_rlpd_checkpoint, resolve_rlpd_checkpoint  # noqa: E402

INFER_RLPD = _DEPLOY / "infer_rlpd.py"
SESSION_DIR = _DEPLOY / "_snaps" / f"rlpd_ik_{int(time.time())}"


def run_lift_place(
    *,
    port: str,
    bowl_xy: tuple[float, float],
    lift_m: float,
    home_pose: str,
    speed: str,
    dry_run: bool,
) -> int:
    cmd = [
        sys.executable, "-m", "toolset.kinematics.ik_relative",
        "--port", port,
        "--sequence", "lift_place",
        "--bowl_xy_m", f"{bowl_xy[0]:.5f}", f"{bowl_xy[1]:.5f}",
        "--lift_m", f"{lift_m:.4f}",
        "--speed", speed,
        "--home_pose", home_pose,
    ]
    if dry_run:
        cmd.append("--dry_run")
    print("[rlpd-ik] IK lift_place:", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(_PROJECT)).returncode


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--name", default=None, help="squint-rlpd/runs/<name>/")
    p.add_argument("--bowl_xyz", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    p.add_argument("--bowl_xy_m", type=float, nargs=2, default=None, metavar=("X", "Y"))
    p.add_argument("--goal_color", type=int, default=None)
    p.add_argument("--color", default="yellow")
    p.add_argument("--follower_port", default="COM3")
    p.add_argument("--home_pose", default="eval1_rest")
    p.add_argument("--lift_m", type=float, default=0.10)
    p.add_argument("--ik_speed", default="fluid")
    p.add_argument("--episode_steps", type=int, default=600)
    p.add_argument("--action_scale", type=float, default=None)
    p.add_argument("--handoff_on_space", action="store_true")
    p.add_argument("--handoff_on_grasp", action="store_true")
    p.add_argument("--handoff_grasp_immediate", action="store_true")
    p.add_argument("--handoff_step", type=int, default=None)
    p.add_argument("--grasp_hold_steps", type=int, default=15)
    p.add_argument("--rlpd_only", action="store_true")
    p.add_argument("--ik_only", action="store_true")
    p.add_argument("--ik_dry_run", action="store_true")
    p.add_argument("--force_ik", action="store_true", help="Run IK even if RLPD exit != 0.")
    p.add_argument("--probe_only", action="store_true")
    args, extra = p.parse_known_args()

    if args.bowl_xyz is None and not args.ik_only:
        p.error("--bowl_xyz required unless --ik_only")
    if not args.ik_only and not (
        args.handoff_on_space or args.handoff_on_grasp or args.handoff_step is not None
    ):
        p.error("Phase 1 needs a handoff trigger: --handoff_on_space, --handoff_on_grasp, or --handoff_step")
    if args.bowl_xyz is None and args.bowl_xy_m is None and args.ik_only:
        p.error("--ik_only needs --bowl_xy_m or --bowl_xyz")

    bowl_xyz = list(args.bowl_xyz) if args.bowl_xyz is not None else [0.0, 0.0, 0.0]
    bowl_xy = (
        (float(args.bowl_xy_m[0]), float(args.bowl_xy_m[1]))
        if args.bowl_xy_m is not None
        else (float(bowl_xyz[0]), float(bowl_xyz[1]))
    )

    ckpt_path = resolve_rlpd_checkpoint(args.ckpt, name=args.name)
    meta = probe_rlpd_checkpoint(ckpt_path)
    print("[rlpd-ik] checkpoint metadata:")
    print(f"  path={meta.path}")
    print(f"  step={meta.global_step}  n_state={meta.n_state}  image={meta.image_h}x{meta.image_w}")

    if args.probe_only:
        return 0

    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    log: dict = {"ckpt": meta.path, "bowl_xyz": bowl_xyz, "bowl_xy_m": list(bowl_xy)}

    rlpd_rc = 0
    if not args.ik_only:
        infer_cmd = [
            sys.executable, str(INFER_RLPD),
            "--ckpt", str(ckpt_path),
            "--follower_port", args.follower_port,
            "--episode_steps", str(args.episode_steps),
            "--home", "--home_pose", args.home_pose,
        ]
        if args.goal_color is not None:
            infer_cmd += ["--goal_color", str(args.goal_color)]
        else:
            infer_cmd += ["--color", args.color]
        infer_cmd += ["--bowl_xyz", str(bowl_xyz[0]), str(bowl_xyz[1]), str(bowl_xyz[2])]
        if args.action_scale is not None:
            infer_cmd += ["--action_scale", str(args.action_scale)]
        if args.handoff_on_space:
            infer_cmd.append("--handoff_on_space")
        if args.handoff_on_grasp:
            infer_cmd.append("--handoff_on_grasp")
        if args.handoff_grasp_immediate:
            infer_cmd.append("--handoff_grasp_immediate")
        if args.handoff_step is not None:
            infer_cmd += ["--handoff_step", str(args.handoff_step)]
        infer_cmd += ["--grasp_hold_steps", str(args.grasp_hold_steps)]
        infer_cmd += extra

        log["infer_cmd"] = infer_cmd
        print("[rlpd-ik] phase 1: RLPD until handoff ...", flush=True)
        rlpd_rc = subprocess.run(infer_cmd, cwd=str(_PROJECT)).returncode
        log["rlpd_rc"] = rlpd_rc
        print(f"[rlpd-ik] RLPD exit {rlpd_rc} (0=handoff OK, 2=no handoff)", flush=True)

        if args.rlpd_only:
            (SESSION_DIR / "run.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
            return rlpd_rc

        if rlpd_rc != 0 and not args.force_ik:
            print("[rlpd-ik] skipping IK (no handoff). Use --force_ik to place anyway.", flush=True)
            (SESSION_DIR / "run.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
            return rlpd_rc

    print("[rlpd-ik] phase 2: IK lift_place ...", flush=True)
    time.sleep(0.5)
    ik_rc = run_lift_place(
        port=args.follower_port,
        bowl_xy=bowl_xy,
        lift_m=args.lift_m,
        home_pose=args.home_pose,
        speed=args.ik_speed,
        dry_run=args.ik_dry_run,
    )
    log["ik_rc"] = ik_rc
    (SESSION_DIR / "run.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"[rlpd-ik] log -> {SESSION_DIR / 'run.json'}", flush=True)
    return ik_rc if ik_rc != 0 else (0 if rlpd_rc in (0, None) else rlpd_rc)


if __name__ == "__main__":
    raise SystemExit(main())
