"""
probe_cubes.py - guided protocol: place cube at a known world position,
touch it with the gripper tip, record FK to validate kinematics.

Output (per cube):
  - truth_xyz_m       : where you put the cube (operator input)
  - probe_xyz_m       : FK gripper_tip position when touching the cube
  - error_xyz_m       : truth - probe (per axis)
  - error_norm_mm     : ||truth - probe||
  - joints_motor_deg  : motor angles at the probe moment
  - joints_urdf_rad   : URDF-corrected angles

Writes a CSV summary and a YAML with all per-probe details. Use the CSV
to check for systematic offsets (e.g. consistent -5 mm in y means the
hand-eye / FK is biased by 5 mm).

Usage:
    python -m toolset.calibration_scripts.probe_cubes --port COM3
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from toolset.kinematics.config import KinematicsConfig
from toolset.kinematics.motor_to_urdf import MotorToUrdfConfig, read_motor_deg_from_obs
from toolset.kinematics.urdf_fk import SO101FK


PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "toolset" / "configs" / "kinematics"


def _ask_xyz(prompt: str) -> np.ndarray:
    while True:
        s = input(prompt).strip()
        try:
            parts = [float(x) for x in s.replace(",", " ").split()]
            if len(parts) != 3:
                raise ValueError
            return np.array(parts, dtype=float)
        except ValueError:
            print("  Please enter three numbers separated by spaces or commas (e.g. 0.18 0.31 0.025).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--camera_index", type=int, default=1)
    parser.add_argument("--n", type=int, default=4, help="Number of cubes to probe.")
    parser.add_argument("--frame", choices=["user", "urdf"], default="user",
                        help="Frame for the truth xyz. 'user' = (right+, forward+, up+).")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    kcfg = KinematicsConfig.load()
    mcfg = MotorToUrdfConfig.load()
    fk = SO101FK(kcfg.urdf_path, gripper_tip_offset=kcfg.gripper_tip_offset_m)

    from lerobot.robots.utils import make_robot_from_config
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    rconf = SO101FollowerConfig(
        port=args.port, id="so101_follower",
        cameras={"wrist": OpenCVCameraConfig(
            index_or_path=args.camera_index, fps=30, width=640, height=480,
        )},
    )
    robot = make_robot_from_config(rconf)
    print(f"[probe-cubes] Connecting to {args.port} ...")
    robot.connect()

    probes = []
    try:
        for i in range(args.n):
            print(f"\n=== Probe {i + 1}/{args.n} ===")
            print(f"  Frame: {args.frame}. Enter cube xyz in METERS.")
            truth = _ask_xyz("  truth xyz (x y z): ")
            input("  Place cube there. Teleop the gripper tip to TOUCH the cube top-center.\n"
                  "  Hold steady, then press Enter to record. ")
            obs = robot.get_observation()
            motor_deg = read_motor_deg_from_obs(obs)
            urdf = mcfg.motor_to_urdf_rad(motor_deg)
            tip = fk.fk(urdf, target="gripper_tip")["position"]
            # If truth is in user frame, convert to URDF for the comparison
            if args.frame == "user":
                truth_urdf = np.array([+truth[1], -truth[0], truth[2]], dtype=float)
            else:
                truth_urdf = truth
            err = truth_urdf - tip
            err_norm_mm = float(np.linalg.norm(err) * 1000)
            print(f"  motor (deg): {motor_deg.round(2).tolist()}")
            print(f"  urdf (rad):  {urdf.round(4).tolist()}")
            print(f"  probe xyz (m, URDF): {tip.round(4).tolist()}")
            print(f"  truth xyz (m, URDF): {truth_urdf.round(4).tolist()}")
            print(f"  error xyz (mm): {(err * 1000).round(1).tolist()}  ||.||={err_norm_mm:.1f} mm")
            probes.append({
                "truth_xyz_m_urdf": truth_urdf.tolist(),
                "probe_xyz_m_urdf": tip.tolist(),
                "error_xyz_m_urdf": err.tolist(),
                "error_norm_mm": err_norm_mm,
                "joints_motor_deg": motor_deg.tolist(),
                "joints_urdf_rad": urdf.tolist(),
                "truth_frame_input": args.frame,
                "timestamp": time.time(),
            })
    finally:
        robot.disconnect()
        print("[probe-cubes] Disconnected.")

    if not probes:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "probe_cubes.csv"
    yaml_path = args.output_dir / "probe_cubes.yaml"

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["probe_idx", "truth_x", "truth_y", "truth_z",
                    "probe_x", "probe_y", "probe_z",
                    "err_x_mm", "err_y_mm", "err_z_mm", "err_norm_mm"])
        for i, p in enumerate(probes):
            tx, ty, tz = p["truth_xyz_m_urdf"]
            px, py, pz = p["probe_xyz_m_urdf"]
            ex, ey, ez = p["error_xyz_m_urdf"]
            w.writerow([i, tx, ty, tz, px, py, pz,
                        ex * 1000, ey * 1000, ez * 1000, p["error_norm_mm"]])
    with open(yaml_path, "w") as f:
        yaml.safe_dump({"probes": probes}, f, default_flow_style=False, sort_keys=False)

    errs_mm = np.array([p["error_norm_mm"] for p in probes])
    print(f"\n[probe-cubes] {len(probes)} probes recorded.")
    print(f"  mean error: {errs_mm.mean():.1f} mm")
    print(f"  max  error: {errs_mm.max():.1f} mm")
    print(f"  rms  error: {np.sqrt((errs_mm ** 2).mean()):.1f} mm")
    print(f"  -> {csv_path}")
    print(f"  -> {yaml_path}")
    if errs_mm.max() > 10:
        print("\n[probe-cubes] WARNING: max error > 1 cm. Possible causes:")
        print("   - motor_offsets are wrong (re-run calibrate_motor_offsets)")
        print("   - gripper_tip_offset_m in config.yaml is 0 (measure & set)")
        print("   - mechanical calibration of the arm has drifted")


if __name__ == "__main__":
    main()
