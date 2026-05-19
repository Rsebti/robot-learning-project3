"""
measure_action_latency.py - measure how long it takes from
``robot.send_action(...)`` returning to the motor encoder actually moving.

How it works:
  1. Read Present_Position at rest, repeatedly, for a "noise floor" baseline.
  2. Issue a small step command (default +3 deg) on one joint.
  3. Poll Present_Position as fast as possible.
  4. Time from the moment send_action() returned to the first Present_Position
     sample more than `threshold_deg` away from baseline = "send-to-motion".

Outputs per-trial stats: send latency (send call duration), poll-then-detect
latency (motion start), full latency (send_start -> motion).

Usage:
    python -m toolset.control.measure_action_latency `
        --port COM3 --motor shoulder_pan --step_deg 3 --n_trials 5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from statistics import mean, stdev

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="COM3")
    p.add_argument("--motor", default="shoulder_pan",
                   help="Motor to nudge. Pick one with little load and gravity "
                        "tendency (shoulder_pan or wrist_roll work well).")
    p.add_argument("--step_deg", type=float, default=3.0,
                   help="Step command magnitude (deg). Keep small (1-5).")
    p.add_argument("--n_trials", type=int, default=5)
    p.add_argument("--threshold_deg", type=float, default=0.3,
                   help="Position change considered 'motion started' (deg).")
    p.add_argument("--poll_timeout_s", type=float, default=0.5)
    p.add_argument("--return_settle_s", type=float, default=1.0)
    args = p.parse_args()

    deploy = Path(__file__).resolve().parents[2] / "deploy"
    if str(deploy) not in sys.path:
        sys.path.insert(0, str(deploy))
    from robot_calibration import make_so101_follower_config
    from lerobot.robots.utils import make_robot_from_config

    cfg = make_so101_follower_config(args.port, cameras={}, use_degrees=True)
    robot = make_robot_from_config(cfg)
    print(f"[lat] Connecting to {args.port} ...")
    robot.connect()

    joints = list(robot.bus.motors.keys())
    if args.motor not in joints:
        raise SystemExit(f"unknown motor {args.motor!r}; pick from {joints}")

    try:
        send_durs, motion_lags, total_lags = [], [], []
        for trial in range(args.n_trials):
            # Settle: read baseline current position
            time.sleep(0.2)
            base = robot.get_observation()[f"{args.motor}.pos"]
            target = base + args.step_deg
            cmd = {f"{n}.pos": robot.get_observation()[f"{n}.pos"] for n in joints}
            cmd[f"{args.motor}.pos"] = target

            # --- send + poll -------------------------------------------------
            t_send_start = time.perf_counter()
            robot.send_action(cmd)
            t_send_end = time.perf_counter()

            t_motion_start = None
            t_deadline = t_send_end + args.poll_timeout_s
            samples = 0
            poll_dts = []
            t_last_poll = t_send_end
            while time.perf_counter() < t_deadline:
                # Fast single-motor read, NOT full get_observation (which reads
                # all 6 motors + camera and dominates the poll period).
                d = robot.bus.sync_read("Present_Position", [args.motor])
                t_now = time.perf_counter()
                poll_dts.append(t_now - t_last_poll)
                t_last_poll = t_now
                samples += 1
                pos = float(d[args.motor])
                if abs(pos - base) >= args.threshold_deg:
                    t_motion_start = t_now
                    break

            send_dur_ms = (t_send_end - t_send_start) * 1000
            if t_motion_start is None:
                print(f"  trial {trial}: NO motion detected within {args.poll_timeout_s*1000:.0f} ms "
                      f"(send_dur={send_dur_ms:.2f}ms, samples={samples}). Try larger --step_deg or "
                      f"different motor.")
            else:
                motion_lag_ms = (t_motion_start - t_send_end) * 1000
                total_ms = (t_motion_start - t_send_start) * 1000
                send_durs.append(send_dur_ms)
                motion_lags.append(motion_lag_ms)
                total_lags.append(total_ms)
                poll_period_ms = (np.mean(poll_dts) * 1000) if poll_dts else float("nan")
                print(f"  trial {trial}: send_dur={send_dur_ms:6.2f} ms  "
                      f"send_to_motion={motion_lag_ms:6.2f} ms  "
                      f"total_send_start_to_motion={total_ms:6.2f} ms  "
                      f"(polls={samples}, poll_period={poll_period_ms:.2f} ms)")

            # Return joint to its starting position before next trial
            cmd_back = {f"{n}.pos": robot.get_observation()[f"{n}.pos"] for n in joints}
            cmd_back[f"{args.motor}.pos"] = base
            robot.send_action(cmd_back)
            time.sleep(args.return_settle_s)

        if total_lags:
            print(f"\n[lat] n={len(total_lags)} successful trials on {args.motor!r}:")
            print(f"  send call duration       : mean {mean(send_durs):6.2f} ms  "
                  f"std {stdev(send_durs) if len(send_durs) > 1 else 0:5.2f}")
            print(f"  send return -> motion    : mean {mean(motion_lags):6.2f} ms  "
                  f"std {stdev(motion_lags) if len(motion_lags) > 1 else 0:5.2f}")
            print(f"  send_start -> motion     : mean {mean(total_lags):6.2f} ms  "
                  f"std {stdev(total_lags) if len(total_lags) > 1 else 0:5.2f}")
        else:
            print(f"\n[lat] no successful trials; nothing to report.")
    finally:
        robot.disconnect()
        print("[lat] Disconnected.")


if __name__ == "__main__":
    main()
