"""SAC legacy infer scripts: --home / --home_pose wired to deploy/homes.py."""

from __future__ import annotations

import argparse
import time

import numpy as np

from homes import get_home_rad, list_home_poses


def add_sac_home_cli(
    parser: argparse.ArgumentParser,
    *,
    default_home_pose: str = "eval1_sac_legacy",
) -> None:
    parser.add_argument(
        "--home",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ramp to rest pose from deploy/homes.py before rollout (default: on)",
    )
    parser.add_argument(
        "--home_pose",
        default=default_home_pose,
        choices=list_home_poses(),
        help="Named preset in deploy/homes.py",
    )
    parser.add_argument(
        "--home_countdown_s",
        type=float,
        default=5.0,
        help="Pause after homing before rollout (0 = skip)",
    )


def sac_maybe_home(agent, args) -> np.ndarray:
    rest = get_home_rad(args.home_pose)
    if not args.home:
        print("[home] --no-home: skipping ramp; rollout starts from current pose.", flush=True)
        return rest
    start_qpos_deg = np.rad2deg(agent.get_qpos().cpu().numpy().flatten())
    print(f"[home] starting qpos (deg): {start_qpos_deg.round(1).tolist()}", flush=True)
    print(f"[home] preset {args.home_pose!r} target (deg): "
          f"{np.rad2deg(rest).round(1).tolist()}", flush=True)
    print("[home] ramping to rest pose (up to 20s) ...", flush=True)
    agent.reset(rest)
    home_qpos_deg = np.rad2deg(agent.get_qpos().cpu().numpy().flatten())
    print(f"[home] homed qpos    (deg): {home_qpos_deg.round(1).tolist()}", flush=True)
    err_deg = home_qpos_deg - np.rad2deg(rest)
    print(f"[home] home error    (deg): {err_deg.round(1).tolist()}  "
          f"max={float(np.max(np.abs(err_deg))):.1f}", flush=True)
    countdown = float(args.home_countdown_s)
    if countdown > 0:
        print(f"\n[home] HOMED. Rollout starts in {countdown:.0f}s (Ctrl+C to abort) ...", flush=True)
        steps = max(1, int(round(countdown)))
        for k in range(steps, 0, -1):
            print(f"  {k} ...", flush=True)
            time.sleep(countdown / steps)
    return rest
