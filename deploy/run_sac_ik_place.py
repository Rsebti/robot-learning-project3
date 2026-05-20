"""
run_sac_ik_place.py — SAC pick until grasp, then ik_relative lift_place to bowl.

No scouting: pass --bowl_xyz (user frame, same as policy) and --bowl_xy_m or use xyz[:2].

Phases:
  1. SAC rollout with --handoff_on_grasp (stops after grip-force close completes)
  2. ik_relative --sequence lift_place (tighten, lift 10 cm, move to bowl XY, open, home)

Does NOT modify toolset/kinematics/ik_relative.py — calls it as subprocess.

Usage:
    python deploy/run_sac_ik_place.py `
        --ckpt C:\\Users\\hugod\\e1100lat.pt `
        --goal_color 3 `
        --bowl_xyz 0.16 0.32 0.00

    # Dry-run IK only (arm already holding cube at grasp pose):
    python deploy/run_sac_ik_place.py --ik_only `
        --bowl_xy_m 0.16 0.32 --port COM3

    # SAC only (no place):
    python deploy/run_sac_ik_place.py --sac_only --ckpt ... --bowl_xyz ...
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

from checkpoint_probe import probe  # noqa: E402
from run_checkpoint import (  # noqa: E402
    _inject_sac_defaults,
    resolve_checkpoint,
    SAC_GOAL_COLORS,
)


def _strip_cli_flags(argv: list[str], *names: str) -> list[str]:
    drop = set(names)
    out: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] in drop:
            if i + 1 < len(argv) and not str(argv[i + 1]).startswith("-"):
                i += 2
                continue
            i += 1
            continue
        out.append(argv[i])
        i += 1
    return out

INFER_SAC = _DEPLOY / "eval1_v2" / "infer_sac_legacy.py"
SESSION_DIR = _DEPLOY / "_snaps" / f"sac_ik_{int(time.time())}"


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
    print("[sac-ik] IK lift_place:", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(_PROJECT)).returncode


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--bowl_xyz", type=float, nargs=3, default=None,
                   metavar=("X", "Y", "Z"),
                   help="Bowl in user frame (policy 21-d state). Required unless --ik_only.")
    p.add_argument("--bowl_xy_m", type=float, nargs=2, default=None,
                   metavar=("X", "Y"),
                   help="Override bowl XY for IK (default: first two of --bowl_xyz).")
    p.add_argument("--goal_color", type=int, default=None)
    p.add_argument("--color", default="yellow",
                   help="If --goal_color omitted, map from SAC_GOAL_COLORS.")
    p.add_argument("--port", default="COM3")
    p.add_argument("--home_pose", default="eval1_rest",
                   help="SAC start home + IK go_home preset.")
    p.add_argument("--lift_m", type=float, default=0.10)
    p.add_argument("--ik_speed", default="fluid")
    p.add_argument("--episode_steps", type=int, default=450)
    p.add_argument("--action_scale", type=float, default=None)
    p.add_argument("--grip_force_steps", type=int, default=30)
    p.add_argument("--handoff_grasp_immediate", action="store_true",
                   help="Hand off on first close intent (default: after force-close steps).")
    p.add_argument("--sac_only", action="store_true", help="Run SAC handoff only.")
    p.add_argument("--ik_only", action="store_true", help="Skip SAC; run lift_place now.")
    p.add_argument("--ik_dry_run", action="store_true", help="IK plan only.")
    p.add_argument("--no_viz", action="store_true", default=True,
                   help="Disable SAC Rerun viz (default on).")
    p.add_argument("--viz", action="store_true", help="Enable SAC Rerun viz.")
    p.add_argument("--skip_ik_if_no_grasp", action="store_true", default=True,
                   help="If SAC exits without grasp, do not run IK (default).")
    p.add_argument("--force_ik", action="store_true",
                   help="Run IK even if grasp was not detected.")
    args, extra = p.parse_known_args()

    if args.viz:
        args.no_viz = False

    if args.bowl_xyz is None and not args.ik_only:
        p.error("--bowl_xyz required unless --ik_only")
    if args.bowl_xyz is None and args.bowl_xy_m is None and args.ik_only:
        p.error("--ik_only needs --bowl_xy_m or --bowl_xyz")

    bowl_xyz = list(args.bowl_xyz) if args.bowl_xyz is not None else [0.0, 0.0, 0.0]
    if args.bowl_xy_m is not None:
        bowl_xy = (float(args.bowl_xy_m[0]), float(args.bowl_xy_m[1]))
    else:
        bowl_xy = (float(bowl_xyz[0]), float(bowl_xyz[1]))

    goal_color = args.goal_color
    if goal_color is None:
        inv = {v: k for k, v in SAC_GOAL_COLORS.items()}
        if args.color not in inv:
            p.error(f"Unknown --color {args.color!r}; set --goal_color 0..5")
        goal_color = inv[args.color]

    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    log = {
        "bowl_xyz": list(bowl_xyz),
        "bowl_xy_m": list(bowl_xy),
        "goal_color": goal_color,
    }

    sac_rc = 0
    if not args.ik_only:
        ckpt = resolve_checkpoint(args.ckpt, args.name)
        info = probe(ckpt)
        if info.backend != "sac":
            raise SystemExit(f"Expected SAC .pt, got {info.backend!r}")

        infer_extra = list(extra)
        infer_extra += [
            "--checkpoint", str(ckpt),
            "--goal_color", str(goal_color),
            "--bowl_xyz", str(bowl_xyz[0]), str(bowl_xyz[1]), str(bowl_xyz[2]),
            "--n_episodes", "1",
            "--handoff_on_grasp",
            "--park_pose", "",
            "--home",
            "--home_countdown_s", "2",
            "--grip_force_steps", str(args.grip_force_steps),
            "--episode_steps", str(args.episode_steps),
        ]
        if args.home_pose != "eval1_rest":
            infer_extra += ["--home_pose", args.home_pose]
        if args.handoff_grasp_immediate:
            infer_extra.append("--handoff_grasp_immediate")
        if args.no_viz:
            infer_extra.append("--no-viz")
        else:
            infer_extra.append("--viz")
        if args.action_scale is not None:
            infer_extra += ["--action_scale", str(args.action_scale)]
        infer_extra = _inject_sac_defaults(infer_extra, info, info.manifest or {})
        infer_extra = _strip_cli_flags(
            infer_extra, "--no-home", "--park_pose", "--home", "--home_pose",
        )
        infer_extra += ["--home", "--home_pose", args.home_pose]
        if args.home_pose == "eval1_rest":
            infer_extra = _strip_cli_flags(infer_extra, "--home_pose")
            infer_extra += ["--home_pose", "auto"]

        sac_cmd = [sys.executable, str(INFER_SAC)] + infer_extra
        log["sac_cmd"] = sac_cmd
        print("[sac-ik] phase 1: SAC until grasp ...", flush=True)
        print("[sac-ik] cmd:", " ".join(sac_cmd[:12]), "...", flush=True)
        sac_rc = subprocess.run(sac_cmd, cwd=str(_PROJECT)).returncode
        log["sac_rc"] = sac_rc
        print(f"[sac-ik] SAC exit code {sac_rc} (0=grasp handoff, 2=no grasp)", flush=True)

        if args.sac_only:
            (SESSION_DIR / "run.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
            print(f"[sac-ik] sac_only done. log -> {SESSION_DIR / 'run.json'}")
            raise SystemExit(sac_rc)

        if sac_rc != 0 and args.skip_ik_if_no_grasp and not args.force_ik:
            print("[sac-ik] skipping IK (no grasp). Use --force_ik to place anyway.", flush=True)
            (SESSION_DIR / "run.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
            raise SystemExit(sac_rc)

    if args.sac_only:
        raise SystemExit(0)

    print("[sac-ik] phase 2: IK lift_place (tighten → lift → bowl → open → home) ...", flush=True)
    time.sleep(0.5)  # brief gap so COM port releases after SAC disconnect
    ik_rc = run_lift_place(
        port=args.port,
        bowl_xy=bowl_xy,
        lift_m=args.lift_m,
        home_pose=args.home_pose,
        speed=args.ik_speed,
        dry_run=args.ik_dry_run,
    )
    log["ik_rc"] = ik_rc
    (SESSION_DIR / "run.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"[sac-ik] done. log -> {SESSION_DIR / 'run.json'}", flush=True)
    raise SystemExit(ik_rc if ik_rc != 0 else (0 if sac_rc in (0, None) else sac_rc))


if __name__ == "__main__":
    main()
