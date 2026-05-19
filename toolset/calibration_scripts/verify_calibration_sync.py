"""
verify_calibration_sync.py - read motor EEPROM and compare against the
deploy/calibration/so101_follower.json. Prints a match column per motor.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DEPLOY = Path(__file__).resolve().parents[2] / "deploy"
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

from robot_calibration import (  # noqa: E402
    LOCAL_CALIBRATION_FILE,
    make_so101_follower_config,
)
from lerobot.robots.utils import make_robot_from_config  # noqa: E402


def main():
    cfg = make_so101_follower_config("COM3", cameras={}, use_degrees=True)
    robot = make_robot_from_config(cfg)
    robot.connect()
    bus = robot.bus
    calib = json.load(open(LOCAL_CALIBRATION_FILE))

    header = (
        f"{'motor':<14} {'m_h':>6} {'j_h':>6} "
        f"{'m_min':>6} {'j_min':>6} "
        f"{'m_max':>6} {'j_max':>6}  match"
    )
    print(header)
    all_ok = True
    for m in bus.motors:
        h = int(bus.sync_read("Homing_Offset",      [m], normalize=False)[m])
        mn = int(bus.sync_read("Min_Position_Limit", [m], normalize=False)[m])
        mx = int(bus.sync_read("Max_Position_Limit", [m], normalize=False)[m])
        j = calib[m]
        ok = (
            h == int(j["homing_offset"])
            and mn == int(j["range_min"])
            and mx == int(j["range_max"])
        )
        all_ok = all_ok and ok
        print(
            f"{m:<14} "
            f"{h:>+6d} {int(j['homing_offset']):>+6d} "
            f"{mn:>6d} {int(j['range_min']):>6d} "
            f"{mx:>6d} {int(j['range_max']):>6d}  {ok}"
        )

    print(f"\nall match: {all_ok}")
    robot.disconnect()


if __name__ == "__main__":
    main()
