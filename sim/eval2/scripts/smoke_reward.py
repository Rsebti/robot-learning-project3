"""Quick smoke test: build env + step + check reward shape / finiteness / range."""
from __future__ import annotations
import argparse
import torch  # noqa

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-SquintNative-Place-Play-v0")
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
    base_env = env.unwrapped
    obs, _ = env.reset(seed=0)
    n_act = env.action_space.shape[-1]
    zero = torch.zeros(1, n_act, device=base_env.device)

    rewards = []
    for s in range(args.n_steps):
        obs, r, term, trunc, info = env.step(zero)
        rewards.append(r.item())

    rewards_t = torch.tensor(rewards)
    print(f"[smoke] reward stats over {args.n_steps} zero-action steps:")
    print(f"        mean = {rewards_t.mean().item():+.4f}")
    print(f"        std  = {rewards_t.std().item():+.4f}")
    print(f"        min  = {rewards_t.min().item():+.4f}")
    print(f"        max  = {rewards_t.max().item():+.4f}")
    print(f"        any nan/inf = {(~torch.isfinite(rewards_t)).any().item()}")
    print(f"        first 5: {[f'{x:+.3f}' for x in rewards[:5]]}")
    print(f"        last 5:  {[f'{x:+.3f}' for x in rewards[-5:]]}")
    env.close()
    app.close()


if __name__ == "__main__":
    main()
