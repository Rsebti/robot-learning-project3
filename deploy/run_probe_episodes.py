"""
run_probe_episodes.py — automated episodes: home → scout cube → SAC rollout → home.

Between episodes waits ``--setup_pause_s`` (default 5) so you can move the cube.

Each episode:
  1. Scout at ``--scout_home_pose`` (default eval1_rest): camera → bowl_xyz
  2. Policy rollout via infer_sac_legacy (one episode) with probed ``--bowl_xyz``
  3. Park back to ``--scout_home_pose``

Usage:
    python deploy/run_probe_episodes.py `
        --ckpt C:\\Users\\hugod\\e1100lat.pt `
        --color red --goal_color 0 `
        --n_episodes 5 --setup_pause_s 5

    # Scout only (no policy), 5 placements:
    python deploy/run_probe_episodes.py --probe_only --color red --n_episodes 5
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

from checkpoint_probe import probe
from run_checkpoint import resolve_checkpoint, _inject_sac_defaults, SAC_GOAL_COLORS

SCOUT_JSON = _DEPLOY / "_snaps" / "scout_latest.json"
SESSION_DIR = _DEPLOY / "_snaps" / f"probe_infer_{int(time.time())}"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=None, help="SAC .pt (required unless --probe_only)")
    p.add_argument("--name", default=None, help="Checkpoint stem under ~/ or eval1_v2/")
    p.add_argument("--color", default="red", help="Cube color for HSV scout.")
    p.add_argument("--goal_color", type=int, default=None,
                   help="Policy goal index 0..5 (default: from --color).")
    p.add_argument("--n_episodes", type=int, default=5)
    p.add_argument("--setup_pause_s", type=float, default=5.0,
                   help="Seconds between episodes to reposition the cube.")
    p.add_argument("--scout_home_pose", default="eval1_rest",
                   help="Physical home for scouting (wrist sees table).")
    p.add_argument("--park_pose", default="eval1_rest",
                   help="Pose after each rollout (default same as scout home).")
    p.add_argument("--port", default="COM3")
    p.add_argument("--camera_index", type=int, default=0)
    p.add_argument("--scout_frames", type=int, default=10)
    p.add_argument("--scout_hz", type=float, default=5.0)
    p.add_argument("--loosen_hsv", action="store_true")
    p.add_argument("--probe_only", action="store_true",
                   help="Only scout + log positions; do not run the policy.")
    p.add_argument("--episode_steps", type=int, default=None)
    p.add_argument("--action_scale", type=float, default=None)
    p.add_argument("--no_viz", action="store_true")
    p.add_argument("--skip_on_scout_fail", action="store_true",
                   help="Skip rollout if scout fails (default: abort).")
    p.add_argument("--fallback_bowl_xyz", type=float, nargs=3,
                   metavar=("X", "Y", "Z"),
                   help="User-frame bowl xyz if scout fails.")
    args, extra = p.parse_known_args()

    goal_color = args.goal_color
    if goal_color is None:
        inv = {v: k for k, v in SAC_GOAL_COLORS.items()}
        if args.color not in inv:
            p.error(f"Unknown --color {args.color!r}; set --goal_color 0..5")
        goal_color = inv[args.color]

    ckpt_path = None
    if not args.probe_only:
        ckpt_path = resolve_checkpoint(args.ckpt, args.name)
        info = probe(ckpt_path)
        if info.backend != "sac":
            raise SystemExit(f"Expected SAC .pt, got {info.backend!r}")

    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    log_path = SESSION_DIR / "episodes.jsonl"
    print(f"[probe-ep] session log -> {SESSION_DIR}")

    python = sys.executable
    scout_cmd_base = [
        python, "-m", "toolset.perception.scout_cube_cli",
        "--port", args.port,
        "--camera_index", str(args.camera_index),
        "--home_pose", args.scout_home_pose,
        "--color", args.color,
        "--n_frames", str(args.scout_frames),
        "--obs_hz", str(args.scout_hz),
    ]
    if args.loosen_hsv:
        scout_cmd_base.append("--loosen_hsv")

    for ep in range(args.n_episodes):
        if ep > 0 and args.setup_pause_s > 0:
            print(f"\n[probe-ep] Place the {args.color} cube, then wait "
                  f"{args.setup_pause_s:.0f}s ...", flush=True)
            time.sleep(args.setup_pause_s)

        print(f"\n{'=' * 60}\n[probe-ep] Episode {ep + 1}/{args.n_episodes}\n{'=' * 60}", flush=True)

        scout_out = SESSION_DIR / f"ep{ep:03d}_scout.json"
        scout_cmd = scout_cmd_base + ["--out", str(scout_out), "--episode", str(ep)]
        print("[probe-ep] scouting ...", flush=True)
        rc = subprocess.run(scout_cmd, cwd=str(_PROJECT))
        bowl_xyz = None
        if scout_out.is_file():
            data = json.loads(scout_out.read_text())
            bowl_xyz = data.get("bowl_xyz_policy") or data.get("bowl_xyz_user_m")

        record = {
            "episode": ep,
            "scout_file": str(scout_out),
            "bowl_xyz_user": bowl_xyz,
            "scout_rc": rc.returncode,
        }

        if bowl_xyz is None:
            print("[probe-ep] scout failed — no bowl_xyz", flush=True)
            if args.fallback_bowl_xyz:
                bowl_xyz = list(args.fallback_bowl_xyz)
                print(f"[probe-ep] using fallback {bowl_xyz}", flush=True)
            elif args.skip_on_scout_fail:
                record["rollout"] = "skipped_scout_fail"
                with open(log_path, "a") as lf:
                    lf.write(json.dumps(record) + "\n")
                continue
            else:
                raise SystemExit("Scout failed. Fix HSV/lighting, use --fallback_bowl_xyz, or --skip_on_scout_fail")

        if args.probe_only:
            record["rollout"] = "probe_only"
            with open(log_path, "a") as lf:
                lf.write(json.dumps(record) + "\n")
            print(f"[probe-ep] probe_only: bowl_xyz={bowl_xyz}", flush=True)
            continue

        infer_extra = list(extra)
        infer_extra += [
            "--checkpoint", str(ckpt_path),
            "--goal_color", str(goal_color),
            "--bowl_xyz", str(bowl_xyz[0]), str(bowl_xyz[1]), str(bowl_xyz[2]),
            "--n_episodes", "1",
            "--home_countdown_s", "2",
            "--park_pose", args.park_pose,
        ]
        if args.no_viz:
            infer_extra.append("--no-viz")
        else:
            infer_extra.append("--viz")
        if args.episode_steps is not None:
            infer_extra += ["--episode_steps", str(args.episode_steps)]
        if args.action_scale is not None:
            infer_extra += ["--action_scale", str(args.action_scale)]

        info = probe(ckpt_path)
        infer_extra = _inject_sac_defaults(infer_extra, info, info.manifest or {})
        # One episode per subprocess; scout already homed — policy still does sac_maybe_home.
        infer_script = _DEPLOY / "eval1_v2" / "infer_sac_legacy.py"
        infer_cmd = [python, str(infer_script)] + infer_extra
        print(f"[probe-ep] rollout bowl_xyz={bowl_xyz}", flush=True)
        print(f"[probe-ep] cmd: {' '.join(infer_cmd[:8])} ...", flush=True)
        rc_infer = subprocess.run(infer_cmd, cwd=str(_PROJECT))
        record["rollout_rc"] = rc_infer.returncode
        record["infer_cmd"] = infer_cmd
        with open(log_path, "a") as lf:
            lf.write(json.dumps(record) + "\n")

        if rc_infer.returncode != 0:
            print(f"[probe-ep] warning: infer exited {rc_infer.returncode}", flush=True)

    print(f"\n[probe-ep] done. Log: {log_path}", flush=True)


if __name__ == "__main__":
    main()
