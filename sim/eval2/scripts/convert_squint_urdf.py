"""Convert the Squint SO-101 URDF to USD for use in Isaac Sim.

The output USD will replace LeIsaac's `so101_follower.usd` so that the
robot's link transforms, joint axes, and kinematic chain match Squint's
training URDF EXACTLY (eliminates the gripper π/2 rotation issue).

Launch
------
.. code-block:: powershell

    cd C:\\Users\\user\\Desktop\\MA2\\isaac\\isaac_so_arm101
    C:/Users/user/Desktop/MA2/isaac/isaac_so_arm101/.venv/Scripts/python.exe `
        -m sim.eval2.scripts.convert_squint_urdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Convert Squint URDF → USD")
parser.add_argument(
    "--urdf",
    type=str,
    default=r"C:/Users/user/Desktop/MA2/robot-learning-project3/squint/envs/robot/so101.urdf",
)
parser.add_argument(
    "--out_dir",
    type=str,
    default=r"C:/Users/user/Desktop/MA2/robot-learning-project3/squint/converted_usd",
)
parser.add_argument("--out_name", type=str, default="so101_squint.usd")
parser.add_argument("--force", action="store_true", help="Re-convert even if USD exists")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True  # no GUI needed

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402


def main() -> None:
    urdf_path = Path(args.urdf).resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[convert] URDF source: {urdf_path}")
    print(f"[convert] Output dir : {out_dir}")
    print(f"[convert] Output USD : {out_dir / args.out_name}")

    cfg = UrdfConverterCfg(
        asset_path=str(urdf_path),
        usd_dir=str(out_dir),
        usd_file_name=args.out_name,
        # Force a full reconvert if the USD exists already.
        force_usd_conversion=bool(args.force),
        # Topology / physics options
        fix_base=True,
        merge_fixed_joints=False,
        # Joint drive defaults — match ManiSkill SO-101 controller (1000/100).
        # Per-joint can be tuned later via ImplicitActuatorCfg.
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=1000.0,
                damping=100.0,
            ),
        ),
    )
    converter = UrdfConverter(cfg)
    print(f"[convert] Done. USD path = {converter.usd_path}")

    simulation_app.close()


if __name__ == "__main__":
    main()
