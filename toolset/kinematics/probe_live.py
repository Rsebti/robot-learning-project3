"""
probe_live.py - move the arm by hand (torque OFF) and watch the FK-computed
gripper tip xyz update in real time. Compare against a ruler to find out
whether FK matches reality or if motor-offset calibration is needed.

Output (refreshed in place):
    joint angles (deg)              | xyz URDF base (m) | xyz user (right+, fwd+)
    pan / lift / elbow / wf / wr     |  x   y   z         |  x   y   z

Usage:
    python -m toolset.kinematics.probe_live --port COM3
    python -m toolset.kinematics.probe_live --port COM3 --hz 20

>>> SUPPORT THE ARM before pressing Enter; torque is disabled so it goes limp. <<<
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

from robot_calibration import make_so101_follower_config  # noqa: E402
from lerobot.robots.utils import make_robot_from_config  # noqa: E402

from toolset.kinematics.config import KinematicsConfig  # noqa: E402
from toolset.kinematics.urdf_fk import CHAIN_JOINTS, SO101FK  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="COM3")
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--target", default="gripper_tip",
                   choices=["wrist", "wrist_roll", "gripper_frame", "gripper_tip"])
    p.add_argument("--no_disable_torque", action="store_true",
                   help="Keep torque on (e.g. if driven by leader arm).")
    args = p.parse_args()

    kcfg = KinematicsConfig.load()
    fk = SO101FK(kcfg.urdf_path, gripper_tip_offset=kcfg.gripper_tip_offset_m)

    cfg = make_so101_follower_config(args.port, cameras={}, use_degrees=True)
    robot = make_robot_from_config(cfg)
    print(f"[probe-live] connecting to {args.port} ...")
    robot.connect()

    bus = robot.bus
    try:
        if not args.no_disable_torque:
            print("\n>>> SUPPORT THE ARM <<<  (torque will be disabled)")
            input("Press Enter when ready: ")
            bus.disable_torque()
            print("[probe-live] torque OFF. Move the gripper by hand.")
        print(f"[probe-live] reading at {args.hz:.0f} Hz, target={args.target}. Ctrl+C to stop.\n")

        # Header
        hdr = (
            f"{'pan':>7} {'lift':>7} {'elbow':>7} {'wflex':>7} {'wroll':>7}  "
            f"|  {'x_urdf':>7} {'y_urdf':>7} {'z_urdf':>7}  "
            f"|  {'x_user':>7} {'y_user':>7} {'z_user':>7}"
        )
        print(hdr)
        print("-" * len(hdr))

        period = 1.0 / max(0.1, args.hz)
        while True:
            t0 = time.time()
            obs = robot.get_observation()
            deg = np.array([obs[f"{n}.pos"] for n in CHAIN_JOINTS], dtype=float)
            q_rad = np.deg2rad(deg)
            out = fk.fk(q_rad, target=args.target)
            xu, yu, zu = out["position"]
            x_user, y_user, z_user = -yu, +xu, +zu
            line = (
                f"{deg[0]:>+7.1f} {deg[1]:>+7.1f} {deg[2]:>+7.1f} "
                f"{deg[3]:>+7.1f} {deg[4]:>+7.1f}  "
                f"|  {xu:>+7.3f} {yu:>+7.3f} {zu:>+7.3f}  "
                f"|  {x_user:>+7.3f} {y_user:>+7.3f} {z_user:>+7.3f}"
            )
            print(line, end="\r", flush=True)
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        print("\n[probe-live] stopped.")
    finally:
        if not args.no_disable_torque:
            try:
                bus.enable_torque()
                print("[probe-live] torque re-enabled.")
            except Exception as e:
                print(f"[probe-live] could not re-enable torque: {e}")
        robot.disconnect()
        print("[probe-live] disconnected.")


if __name__ == "__main__":
    main()
