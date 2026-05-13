"""Debug the wrist cam seg output: what keys, what labels."""
from __future__ import annotations
import argparse
import torch  # noqa: F401

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-SquintNative-Place-Play-v0")
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
    base_env = env.unwrapped
    obs, _ = env.reset(seed=0)
    for _ in range(5):
        env.step(torch.zeros(1, 6, device=base_env.device))

    cam = base_env.scene.sensors["wrist"]
    out_path = r"C:/Users/user/Desktop/MA2/robot-learning-project3/notes/seg_dump.txt"
    with open(out_path, "w") as f:
        f.write(f"cam.data.output keys: {list(cam.data.output.keys())}\n\n")
        for k, v in cam.data.output.items():
            f.write(f"--- '{k}' ---\n")
            if isinstance(v, torch.Tensor):
                f.write(f"  shape={tuple(v.shape)}, dtype={v.dtype}\n")
                f.write(f"  min={v.min().item()}, max={v.max().item()}\n")
                if v.numel() < 50:
                    f.write(f"  values: {v.cpu().numpy().tolist()}\n")
                else:
                    flat = v.flatten()
                    uniq = torch.unique(flat).cpu().numpy()
                    f.write(f"  unique values (n={len(uniq)}): {uniq[:30].tolist()}{' ...' if len(uniq) > 30 else ''}\n")
            else:
                f.write(f"  type={type(v)}, value={v}\n")
        f.write(f"\ncam.data.info[0] = {cam.data.info[0]}\n")
    print(f"[dump] wrote {out_path}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
