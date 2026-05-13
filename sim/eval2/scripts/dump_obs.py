"""Dump obs['policy'] (12-d state) and obs['rgb'] stats at home pose."""
from __future__ import annotations

import argparse
import torch  # noqa: F401

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-SquintNative-Place-Play-v0")
parser.add_argument("--settle_steps", type=int, default=5,
                    help="Match Squint's 5 settle steps for a fair comparison.")
parser.add_argument("--per_step_dump", action="store_true", default=False,
                    help="Print qpos/qvel each settle step (useful to compare convergence).")
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
args.num_envs = 1
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import sim.eval2  # noqa: F401,E402


def main():
    env_cfg = parse_env_cfg(args.task, num_envs=1)
    try:
        env_cfg.recorders = None
    except Exception:
        pass
    env = gym.make(args.task, cfg=env_cfg)
    obs, _ = env.reset(seed=0)
    n_act = env.action_space.shape[-1]
    zero = torch.zeros(1, n_act, device=env.unwrapped.device)
    robot = env.unwrapped.scene["robot"]
    for s in range(args.settle_steps):
        obs, *_ = env.step(zero)
        if args.per_step_dump:
            qp = robot.data.joint_pos[0].cpu().tolist()
            qv = robot.data.joint_vel[0].cpu().tolist()
            print(f"  settle step {s:2d}: qpos[1]={qp[1]:+.5f}  qvel[1]={qv[1]:+.5f}  "
                  f"max|qvel|={max(abs(v) for v in qv):.5f}", flush=True)

    out_path = r"C:/Users/user/Desktop/MA2/robot-learning-project3/notes/obs_dump.txt"
    with open(out_path, "w") as f:
        f.write("=== OBS DUMP after reset(seed=0) + {} settle steps ===\n".format(args.settle_steps))
        f.write("\nobs keys: {}\n".format(list(obs.keys())))
        for k, v in obs.items():
            if isinstance(v, torch.Tensor):
                f.write(f"\nobs['{k}'].shape = {tuple(v.shape)}, dtype = {v.dtype}\n")
                f.write(f"  values: {v.cpu().numpy().flatten().tolist()}\n")
            elif isinstance(v, dict):
                f.write(f"\nobs['{k}'] (dict, keys={list(v.keys())})\n")
                for k2, v2 in v.items():
                    if isinstance(v2, torch.Tensor):
                        f.write(f"  obs['{k}']['{k2}'].shape = {tuple(v2.shape)}, dtype = {v2.dtype}\n")
                        if v2.numel() <= 100:
                            f.write(f"    values: {v2.cpu().numpy().flatten().tolist()}\n")
                        else:
                            arr = v2.cpu().float().numpy()
                            f.write(f"    min/mean/max = {arr.min():.3f} / {arr.mean():.3f} / {arr.max():.3f}\n")
                            if arr.shape[-1] == 3:
                                f.write(f"    R mean = {arr[..., 0].mean():.3f}, G = {arr[..., 1].mean():.3f}, B = {arr[..., 2].mean():.3f}\n")

        # Also print qpos directly from robot for ground-truth cross-check
        f.write("\nGROUND TRUTH (direct robot read):\n")
        f.write(f"  qpos = {robot.data.joint_pos[0].cpu().tolist()}\n")
        f.write(f"  qvel = {robot.data.joint_vel[0].cpu().tolist()}\n")
        # Cube + bin world positions
        scene = env.unwrapped.scene
        for name in ("cube", "bin_floor"):
            if name in scene.rigid_objects:
                p = scene[name].data.root_pos_w[0].cpu().tolist()
                f.write(f"  {name:12s} pos_w = {p}\n")

    print(f"[dump] wrote {out_path}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
