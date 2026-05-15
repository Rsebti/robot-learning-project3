"""Dump per-link mass / COM / inertia of the robot in our converted USD and
compare against Squint's URDF spec (mass/com/inertia table from teammate).

Verifies whether the URDF→USD conversion preserved inertial parameters.
Any mismatch beyond ~1% is suspicious and could explain sim-to-sim drift.
"""
from __future__ import annotations
import argparse, torch  # noqa

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-SquintNative-Place-Play-v0")
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
args.num_envs = 1
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import sim.eval2  # noqa: F401,E402


# Squint URDF spec (from teammate's table).
SQUINT_SPEC = {
    "base":              dict(mass=0.147,    com=(-0.00636,  0.00000, -0.00240), I=(1.147e-4, 1.361e-4, 1.304e-4)),
    "shoulder":          dict(mass=0.100006, com=(-0.03040,  0.000422, -0.04170), I=(8.376e-5, 8.104e-5, 2.398e-5)),
    "upper_arm":         dict(mass=0.103,    com=(-0.11257, -0.01550,  0.01870), I=(4.080e-5, 1.473e-4, 1.425e-4)),
    "lower_arm":         dict(mass=0.104,    com=(-0.06485, -0.03200,  0.01820), I=(2.874e-5, 1.598e-4, 1.453e-4)),
    "wrist":             dict(mass=0.079,    com=( 0.00000, -0.04240,  0.03060), I=(3.683e-5, 2.539e-5, 2.100e-5)),
    "gripper":           dict(mass=0.087,    com=( 0.00770,  0.00010, -0.02340), I=(2.751e-5, 4.337e-5, 3.451e-5)),
    "gripper_frame_link":dict(mass=1e-9,     com=( 0.0,      0.0,      0.0    ), I=(0.0, 0.0, 0.0)),
    "jaw":               dict(mass=0.012,    com=( 0.00000,  0.00000,  0.01890), I=(6.614e-6, 1.890e-6, 5.287e-6)),
    "finger1_tip":       dict(mass=0.0,      com=(0,0,0),                     I=(0,0,0)),
    "finger2_tip":       dict(mass=0.0,      com=(0,0,0),                     I=(0,0,0)),
}


def main():
    env_cfg = parse_env_cfg(args.task, num_envs=1)
    try: env_cfg.recorders = None
    except Exception: pass
    env = gym.make(args.task, cfg=env_cfg)
    base = env.unwrapped
    env.reset(seed=0)
    robot = base.scene["robot"]

    body_names = robot.body_names
    masses = robot.root_physx_view.get_masses()[0]                  # (n_bodies,)
    inertias = robot.root_physx_view.get_inertias()[0]              # (n_bodies, 9) for articulation
    coms = robot.root_physx_view.get_coms()[0]                       # (n_bodies, 7) pos(3) + quat(4)

    print(f"\n{'Body':<22s}  {'Isaac mass':>10s}  {'Squint mass':>11s}  {'Δ%':>7s}   "
          f"{'Isaac COM (x,y,z)':>30s}   {'Squint COM (x,y,z)':>30s}")
    print("-" * 130)
    total_isaac = 0.0
    total_squint = 0.0
    for i, name in enumerate(body_names):
        m_isaac = float(masses[i].item())
        com_isaac = tuple(float(x.item()) for x in coms[i, :3])
        ixx, iyy, izz = float(inertias[i, 0]), float(inertias[i, 4]), float(inertias[i, 8])
        ixy, ixz, iyz = float(inertias[i, 1]), float(inertias[i, 2]), float(inertias[i, 5])

        spec = SQUINT_SPEC.get(name)
        if spec is None:
            print(f"{name:<22s}  {m_isaac:>10.6f}  {'(no spec)':>11s}")
            continue
        m_sq = spec["mass"]
        com_sq = spec["com"]
        I_sq = spec["I"]
        d_mass = (m_isaac - m_sq) / max(m_sq, 1e-9) * 100 if m_sq > 1e-8 else 0.0
        print(f"{name:<22s}  {m_isaac:>10.6f}  {m_sq:>11.6f}  {d_mass:+6.1f}%   "
              f"({com_isaac[0]:+.5f}, {com_isaac[1]:+.5f}, {com_isaac[2]:+.5f})   "
              f"({com_sq[0]:+.5f}, {com_sq[1]:+.5f}, {com_sq[2]:+.5f})")
        print(f"{'':>22s}  I_isaac (ixx,iyy,izz, ixy,ixz,iyz) = "
              f"({ixx:.3e}, {iyy:.3e}, {izz:.3e}, {ixy:+.2e}, {ixz:+.2e}, {iyz:+.2e})")
        print(f"{'':>22s}  I_squint (ixx,iyy,izz)             = "
              f"({I_sq[0]:.3e}, {I_sq[1]:.3e}, {I_sq[2]:.3e})")
        total_isaac += m_isaac
        if m_sq > 1e-8:
            total_squint += m_sq

    print("-" * 130)
    print(f"Total mass: Isaac = {total_isaac:.6f} kg   Squint = {total_squint:.6f} kg  "
          f"Δ% = {(total_isaac - total_squint) / total_squint * 100:+.2f}")

    env.close()
    app.close()


if __name__ == "__main__":
    main()
