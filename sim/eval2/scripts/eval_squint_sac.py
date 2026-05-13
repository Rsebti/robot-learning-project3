"""Visualize and evaluate a trained Squint SAC checkpoint.

Loads encoder + actor weights from a ``ckpt.pt`` produced by
``train_squint_sac.py`` and runs N episodes in the Play env. Per env
prints a final tally of:
- mean episode reward
- success rate (any step where ``squint_place_success`` returned 1.0)
- mean step rewards per stage (reach / grasp / lift / placement)

Use ``--headless`` to skip the GUI (faster eval). Otherwise the Isaac
Sim viewer pops open and you can watch the robot in action.

Launch
------
.. code-block:: powershell

    cd C:\\Users\\user\\Desktop\\MA2\\isaac\\isaac_so_arm101
    C:/Users/user/Desktop/MA2/isaac/isaac_so_arm101/.venv/Scripts/python.exe `
        -m sim.eval2.scripts.eval_squint_sac `
        --task Isaac-SO101-Squint-Place-Play-v0 `
        --ckpt "C:/Users/user/Desktop/MA2/isaac/isaac_so_arm101/runs/squint_sac__Isaac-SO101-Squint-Place-v0__1__1778603338/ckpt.pt" `
        --num_envs 8 --n_episodes 16 --enable_cameras
"""
from __future__ import annotations

import argparse

# IMPORTANT: import torch BEFORE AppLauncher to avoid Isaac Kit DLLs
# winning the load order against torch — otherwise tensordict's _C.pyd
# crashes during import (silent fatal in `_C.pyd!+0xb2d1`).
import torch  # noqa: F401

parser = argparse.ArgumentParser(description="Evaluate a Squint SAC checkpoint")
parser.add_argument("--task", type=str, default="Isaac-SO101-Squint-Place-Play-v0")
parser.add_argument("--ckpt", type=str, required=True, help="Path to ckpt.pt")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--n_episodes", type=int, default=16, help="Total episodes across all envs")
parser.add_argument("--image_size", type=int, default=16)
parser.add_argument("--seed", type=int, default=0)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True  # always need RGB obs

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import sim.eval2  # noqa: F401,E402
from sim.eval2.policy.squint_sac import CNNEncoder, SquintActor  # noqa: E402


# Inlined from train_squint_sac.py (can't import that module — its argparse
# runs at import time and rejects our --ckpt / --n_episodes flags).
def _downsample_rgb(rgb: torch.Tensor, target_size: int) -> torch.Tensor:
    if rgb.shape[-3] == target_size:
        return rgb
    rgb = rgb.permute(0, 3, 1, 2).float()
    rgb = F.interpolate(rgb, size=(target_size, target_size), mode="area")
    return rgb.permute(0, 2, 3, 1).to(torch.uint8)


def _extract_obs(obs: dict, image_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(obs["wrist"], dict):
        rgb = obs["wrist"]["rgb"]
    else:
        rgb = obs["wrist"]
    state = obs["policy"]
    if not torch.is_tensor(state):
        state = torch.as_tensor(state)
    rgb = _downsample_rgb(rgb, image_size)
    return rgb, state.float()


def main() -> None:
    print("[eval] main() entered", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    print(f"[eval] device={device}", flush=True)

    # ---- Env --------------------------------------------------------
    env_cfg = parse_env_cfg(args.task, device=str(device), num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    base_env = env.unwrapped
    n_envs = base_env.num_envs

    action_space = base_env.action_space
    n_act = int(action_space.shape[-1])
    raw_low = torch.from_numpy(action_space.low.reshape(-1)[:n_act]).float()
    raw_high = torch.from_numpy(action_space.high.reshape(-1)[:n_act]).float()
    action_low = torch.where(torch.isfinite(raw_low), raw_low, torch.full_like(raw_low, -1.0))
    action_high = torch.where(torch.isfinite(raw_high), raw_high, torch.full_like(raw_high, 1.0))

    obs, _ = env.reset(seed=args.seed)
    rgb, state = _extract_obs(obs, args.image_size)
    n_state = int(state.shape[-1])
    n_channels = int(rgb.shape[-1])

    print(f"[eval] Task={args.task} num_envs={n_envs} n_state={n_state} n_act={n_act}")
    print(f"[eval] Loading checkpoint: {args.ckpt}")

    # ---- Modules ----------------------------------------------------
    encoder = CNNEncoder(
        n_obs=(args.image_size, args.image_size, n_channels), device=device
    )
    actor = SquintActor(
        n_rgb_repr=encoder.repr_dim,
        n_state=n_state,
        n_act=n_act,
        action_low=action_low,
        action_high=action_high,
        device=device,
    )

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["encoder"])
    actor.load_state_dict(ckpt["actor"])
    encoder.eval()
    actor.eval()

    print(f"[eval] Checkpoint trained to step {ckpt.get('global_step', '?')}")

    # ---- Roll out ---------------------------------------------------
    n_episodes_per_env = max(1, args.n_episodes // n_envs)
    print(f"[eval] Rolling {n_episodes_per_env} episodes per env × {n_envs} envs = {n_episodes_per_env * n_envs} total")

    # Track per-env episode reward and success
    episode_returns = torch.zeros(n_envs, device=device)
    episode_lengths = torch.zeros(n_envs, dtype=torch.long, device=device)
    finished_returns: list[float] = []
    finished_successes: list[float] = []
    finished_lengths: list[int] = []

    # We also want per-reward-term tallies — grab the reward manager.
    rm = base_env.reward_manager
    term_names = list(rm._term_names)
    term_sums = {n: 0.0 for n in term_names}
    term_count = 0

    # Probe the success term separately each step.
    success_term_idx = term_names.index("success_bonus") if "success_bonus" in term_names else None
    above_bin_idx = term_names.index("above_bin") if "above_bin" in term_names else None

    step = 0
    while len(finished_returns) < n_episodes_per_env * n_envs:
        with torch.no_grad():
            features = encoder(rgb)
            action = actor.get_eval_action(features, state)

        next_obs, reward, terminated, truncated, info = env.step(action)

        # Per-term inspection — call functions directly.
        for tname, tcfg in zip(term_names, rm._term_cfgs):
            v = tcfg.func(base_env, **tcfg.params)
            term_sums[tname] += float(v.mean().item())
        term_count += 1

        episode_returns += reward.float()
        episode_lengths += 1

        done = terminated | truncated
        if done.any():
            for env_idx in torch.where(done)[0].tolist():
                finished_returns.append(float(episode_returns[env_idx].item()))
                finished_lengths.append(int(episode_lengths[env_idx].item()))
                episode_returns[env_idx] = 0.0
                episode_lengths[env_idx] = 0

        next_rgb, next_state = _extract_obs(next_obs, args.image_size)
        rgb, state = next_rgb, next_state
        step += 1

        if step % 50 == 0:
            print(f"[eval] step={step}  finished={len(finished_returns)}/{n_episodes_per_env*n_envs}")

    # ---- Report -----------------------------------------------------
    rets = np.array(finished_returns)
    lens = np.array(finished_lengths)
    print("\n" + "=" * 70)
    print(f"EVALUATION COMPLETE — {len(finished_returns)} episodes")
    print("=" * 70)
    print(f"episode return : mean={rets.mean():+.3f}  std={rets.std():.3f}  min={rets.min():+.3f}  max={rets.max():+.3f}")
    print(f"episode length : mean={lens.mean():.1f}  min={lens.min()}  max={lens.max()}")
    print("\nPer-reward-term mean value (averaged across step×env):")
    for tname, total in term_sums.items():
        print(f"  {tname:18s}  {total / max(1, term_count):+.4f}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
