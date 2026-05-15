"""Probe: send a "close gripper" action and watch the gripper qpos / target.

If the gripper doesn't close even with a deterministic ``action = -1`` on
the gripper channel, the issue is in the env (joint limits, action term,
effort limit). If it DOES close here but the trained ckpt fails to close
it during deploy, the issue is in the policy.
"""
from __future__ import annotations
import argparse, torch  # noqa

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-SquintNative-Place-Play-v0")
parser.add_argument("--n_steps", type=int, default=30)
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


def main():
    env_cfg = parse_env_cfg(args.task, num_envs=1)
    try: env_cfg.recorders = None
    except Exception: pass
    env = gym.make(args.task, cfg=env_cfg)
    base = env.unwrapped
    obs, _ = env.reset(seed=0)
    robot = base.scene["robot"]

    print(f"[probe] joint_names = {robot.joint_names}")
    g_idx = robot.joint_names.index("gripper")
    lo = robot.data.soft_joint_pos_limits[0, g_idx, 0].item()
    hi = robot.data.soft_joint_pos_limits[0, g_idx, 1].item()
    print(f"[probe] gripper soft limits = ({lo:+.4f}, {hi:+.4f}) rad  "
          f"= ({lo*180/3.14159:+.1f}°, {hi*180/3.14159:+.1f}°)")
    qpos_home = robot.data.joint_pos[0, g_idx].item()
    print(f"[probe] gripper qpos @ home (after settle 0) = {qpos_home:+.4f} rad "
          f"= {qpos_home*180/3.14159:+.1f}°")

    # Get action term to read target.
    term = base.action_manager.get_term("arm_and_gripper")
    target_home = term.target_qpos[0, g_idx].item()
    print(f"[probe] gripper target @ home = {target_home:+.4f} rad")

    # Action: zeros on arm, -1 on gripper → wants to close at max rate.
    n_act = env.action_space.shape[-1]
    close_action = torch.zeros(1, n_act, device=base.device)
    close_action[0, g_idx] = -1.0   # full negative → close
    print(f"[probe] sending action[gripper] = -1.0 for {args.n_steps} steps")

    for t in range(args.n_steps):
        env.step(close_action)
        q = robot.data.joint_pos[0, g_idx].item()
        tgt = term.target_qpos[0, g_idx].item()
        print(f"  step {t:2d}: target = {tgt:+.4f}  qpos = {q:+.4f}  Δqpos = {q - qpos_home:+.4f}")
        qpos_home_track = q

    env.close()
    app.close()


if __name__ == "__main__":
    main()
