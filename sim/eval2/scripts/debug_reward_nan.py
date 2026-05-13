"""Find which reward term produces NaN in SquintPlaceEnvCfg.

Builds the env, resets, steps a few times with random actions, and after
each step prints **every reward term value** independently so we can see
which one returns NaN.

Launch
------
.. code-block:: powershell

    cd C:\\Users\\user\\Desktop\\MA2\\isaac\\isaac_so_arm101
    C:/Users/user/Desktop/MA2/isaac/isaac_so_arm101/.venv/Scripts/python.exe `
        -m sim.eval2.scripts.debug_reward_nan `
        --task Isaac-SO101-Squint-Place-v0 --num_envs 4 --headless
"""
from __future__ import annotations

import argparse

parser = argparse.ArgumentParser(description="Debug reward NaN in Squint envs")
parser.add_argument("--task", type=str, default="Isaac-SO101-Squint-Place-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--n_steps", type=int, default=8)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import sim.eval2  # noqa: F401,E402


def _stats(name: str, t: torch.Tensor) -> str:
    t = t.detach().float()
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    flag = " <<< NaN!" if has_nan else (" <<< Inf!" if has_inf else "")
    return f"  {name:25s} shape={tuple(t.shape)}  min={t.min().item():+.4f}  max={t.max().item():+.4f}  mean={t.mean().item():+.4f}{flag}"


def _probe_reward_terms(env, label: str) -> None:
    """Print every reward term independently — bypasses the reward manager
    to call the raw function on the live env state."""
    base = env.unwrapped
    rm = base.reward_manager

    print(f"\n=== {label} ===")
    print(f"  total reward step = {rm._reward_buf.detach().float()}")
    print(f"  step_dt = {base.step_dt}")

    # Iterate by term — Isaac Lab stores per-term cfg under rm._term_cfgs
    # (private, but stable across recent versions).
    for term_name, term_cfg in zip(rm._term_names, rm._term_cfgs):
        try:
            val = term_cfg.func(base, **term_cfg.params)
        except Exception as e:
            print(f"  {term_name:25s} EXCEPTION: {type(e).__name__}: {e}")
            continue
        print(_stats(term_name, val))


def _probe_scene_state(env) -> None:
    """Print key scene tensors that feed into rewards: cube/robot/ee poses."""
    base = env.unwrapped
    scene = base.scene

    print("\n--- scene state ---")
    if "robot" in scene.articulations:
        robot = scene["robot"]
        print(_stats("robot.root_pos_w", robot.data.root_pos_w))
        print(_stats("robot.root_quat_w", robot.data.root_quat_w))
        print(_stats("robot.joint_pos", robot.data.joint_pos))
        print(_stats("robot.joint_vel", robot.data.joint_vel))
    if "cube" in scene.rigid_objects:
        cube = scene["cube"]
        print(_stats("cube.root_pos_w", cube.data.root_pos_w))
        print(_stats("cube.root_quat_w", cube.data.root_quat_w))
    if "ee_frame" in scene.sensors:
        ee = scene["ee_frame"]
        print(_stats("ee_frame.target_pos_w", ee.data.target_pos_w.reshape(-1, 3)))
    cmd = base.command_manager.get_command("object_pose")
    print(_stats("command.object_pose", cmd))


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    base = env.unwrapped

    print(f"\n[debug] Task={args.task} num_envs={args.num_envs}")
    print(f"[debug] action_space={env.action_space}")
    print(f"[debug] reward terms = {base.reward_manager._term_names}")

    obs, _ = env.reset(seed=42)
    _probe_scene_state(env)
    _probe_reward_terms(env, "After reset (no step yet)")

    n_act = env.action_space.shape[-1]
    for step in range(args.n_steps):
        # Small random action — magnitudes 1% so we don't fling joints.
        action = torch.empty(args.num_envs, n_act, device=device).uniform_(-0.05, 0.05)
        obs, reward, term, trunc, info = env.step(action)

        nan_any = torch.isnan(reward).any().item()
        print(f"\n[step {step+1}] reward.mean()={reward.float().mean().item():+.4f}  any_nan={nan_any}")
        if nan_any or step in (0, 1, 4, 7):
            _probe_scene_state(env)
            _probe_reward_terms(env, f"After step {step+1}")
            if nan_any:
                print(f"\n!!! NaN detected at step {step+1}; stopping early.")
                break

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
