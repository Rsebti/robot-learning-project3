"""Collect expert demonstrations for DAPG warmstart (Voie B).

Runs ScriptedPickController in 4096 parallel envs, captures (obs, action)
pairs for every step of every episode, then saves only the SUCCESS episodes
to demos.pt. Output is consumed by ``bc_pretrain.py`` and ``ppo_dapg.py``.

Usage (from the isaac_so_arm101 venv with this repo installed editable):

    uv run python -m sim.eval2.bc.generate_demos --num_envs 4096 --num_episodes 5

Output files (next to this script):
    demos.pt           ← {"obs": (M, 29), "actions": (M, 6)} concatenated
                          across all successful episodes (M ~10k-30k pairs)
    demos_meta.json    ← stats: n_episodes_total, n_success, mean_episode_len

Strategy
--------
- Use the Eval2-PickInClutter-v1 task (state-based, 29-D obs, no camera)
  so we collect 4096 envs at full speed (no rendering).
- Pre-roll N episodes worth of steps. Each env step writes (obs, action)
  to a per-env trajectory buffer.
- At episode termination, check the env's success criterion. If success,
  flush that env's trajectory into the master list. Otherwise discard.
- Auto-reset on done is fine: each env can contribute multiple episodes
  per `num_episodes` rollout cycle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

# Argparse and AppLauncher MUST come before isaaclab/torch imports.
parser = argparse.ArgumentParser(description="Collect scripted demos for DAPG.")
parser.add_argument("--task", type=str, default="Eval2-PickInClutter-v1")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument(
    "--num_episodes",
    type=int,
    default=5,
    help="How many episode rollouts to do across the whole batch. With 4096 "
    "envs and 5 episodes, you get 20k tentatives total; with ~30%% scripted "
    "success rate, ~6k success episodes are collected.",
)
parser.add_argument(
    "--output",
    type=str,
    default=str(Path(__file__).parent / "demos.pt"),
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--max_steps_per_episode",
    type=int,
    default=500,
    help="Hard cap on steps per episode (env episode_length_s override too).",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def main():
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    # Side-effect: registers our gym envs.
    import sim.eval2  # noqa: F401
    from sim.eval2.bc.scripted_controller import ScriptedPickController
    from sim.eval2.pick_in_clutter_env_cfg import BOWL_INNER_HALF

    env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    # Override episode_length to fit the scripted phase budget.
    env_cfg.episode_length_s = 10.0

    print(f"[demos] gym.make({args.task!r}, num_envs={args.num_envs})", flush=True)
    env = gym.make(args.task, cfg=env_cfg)

    # First reset to populate robot data buffers.
    obs, _ = env.reset()
    n_envs = env.unwrapped.num_envs
    device = env.unwrapped.device

    controller = ScriptedPickController(env.unwrapped)
    print(
        f"[demos] arm joints = {controller.arm_joint_names}  "
        f"ee_body_idx={controller.ee_body_idx}",
        flush=True,
    )

    # Per-env trajectory buffers (cleared on done). Stored as lists of
    # 1-D CPU tensors that we'll stack at the end.
    obs_traj: list[list[torch.Tensor]] = [[] for _ in range(n_envs)]
    act_traj: list[list[torch.Tensor]] = [[] for _ in range(n_envs)]

    # Per-env "did we succeed at any step this episode?" flag.
    episode_success_seen = torch.zeros(n_envs, dtype=torch.bool, device=device)
    episode_count = torch.zeros(n_envs, dtype=torch.long, device=device)

    # Master accumulator for SUCCESS episodes only.
    all_obs_chunks: list[torch.Tensor] = []
    all_act_chunks: list[torch.Tensor] = []
    n_total_finished = 0
    n_total_success = 0
    target_total_episodes = args.num_episodes * n_envs

    print(
        f"[demos] target episodes = {target_total_episodes} "
        f"({args.num_episodes} per env x {n_envs} envs)", flush=True
    )

    def _extract_policy_obs(o):
        """rsl_rl-style obs: dict {"policy": (N, 29)} or plain tensor."""
        if isinstance(o, tuple):
            o = o[0]
        if hasattr(o, "keys"):
            try:
                if "policy" in o.keys():
                    return o["policy"]
            except Exception:
                pass
        return o

    obs_t = _extract_policy_obs(obs)

    while n_total_finished < target_total_episodes:
        action = controller.compute_action()  # (N, 6)

        # Compute "is success NOW" for this step (before env.step) using the
        # env's terminated success criterion (block in bowl, geometric).
        scene = env.unwrapped.scene
        red = scene["block_red"]
        blue = scene["block_blue"]
        target_idx = env.unwrapped.target_color
        is_red = (target_idx == 0).unsqueeze(-1)
        target_pos_w = torch.where(is_red, red.data.root_pos_w, blue.data.root_pos_w)
        bowl_pos_w = scene["bowl_floor"].data.root_pos_w
        xy_dist = torch.norm(target_pos_w[:, :2] - bowl_pos_w[:, :2], dim=1)
        dz = target_pos_w[:, 2] - bowl_pos_w[:, 2]
        success_now = (xy_dist < BOWL_INNER_HALF) & (dz > -0.01) & (dz < 0.10)
        episode_success_seen |= success_now

        # Save (obs, action) into the per-env trajectory.
        # Keep on CPU to avoid OOM with N=4096 * 500 steps in GPU buffers.
        obs_cpu = obs_t.detach().cpu()
        act_cpu = action.detach().cpu()
        for i in range(n_envs):
            obs_traj[i].append(obs_cpu[i].clone())
            act_traj[i].append(act_cpu[i].clone())

        # Step the env.
        obs, _, terminated, truncated, _ = env.step(action)
        obs_t = _extract_policy_obs(obs)
        dones = terminated | truncated

        # Process done episodes
        done_envs = dones.nonzero(as_tuple=False).flatten().tolist()
        for env_idx in done_envs:
            if bool(episode_success_seen[env_idx]):
                # Flush success trajectory
                if len(obs_traj[env_idx]) >= 10:  # filter trivially short
                    obs_chunk = torch.stack(obs_traj[env_idx])  # (T, 29)
                    act_chunk = torch.stack(act_traj[env_idx])  # (T, 6)
                    all_obs_chunks.append(obs_chunk)
                    all_act_chunks.append(act_chunk)
                    n_total_success += 1
            # Reset buffers for this env (for the next auto-reset episode)
            obs_traj[env_idx] = []
            act_traj[env_idx] = []
            episode_success_seen[env_idx] = False
            episode_count[env_idx] += 1
            n_total_finished += 1

        # Reset controller state for done envs
        if done_envs:
            controller.reset(torch.tensor(done_envs, device=device))

        # Periodic logging
        if n_total_finished > 0 and n_total_finished % (n_envs // 2) == 0:
            rate = n_total_success / max(n_total_finished, 1) * 100.0
            n_pairs = sum(c.shape[0] for c in all_obs_chunks)
            print(
                f"[demos] {n_total_finished:>6d}/{target_total_episodes} episodes "
                f"| success {n_total_success} ({rate:.1f}%) "
                f"| {n_pairs} (s, a) pairs",
                flush=True,
            )

    # Concat everything and save
    if all_obs_chunks:
        all_obs = torch.cat(all_obs_chunks, dim=0)   # (M, 29)
        all_act = torch.cat(all_act_chunks, dim=0)   # (M, 6)
    else:
        all_obs = torch.zeros(0, 29)
        all_act = torch.zeros(0, 6)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"obs": all_obs, "actions": all_act}, out_path)

    meta = {
        "n_episodes_total": n_total_finished,
        "n_episodes_success": n_total_success,
        "success_rate": n_total_success / max(n_total_finished, 1),
        "n_pairs": int(all_obs.shape[0]),
        "obs_dim": int(all_obs.shape[1]) if all_obs.numel() > 0 else 0,
        "action_dim": int(all_act.shape[1]) if all_act.numel() > 0 else 0,
        "task": args.task,
        "num_envs": args.num_envs,
        "seed": args.seed,
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    print("\n" + "=" * 60)
    print(f"[demos] FINAL: {n_total_success}/{n_total_finished} episodes "
          f"= {meta['success_rate'] * 100:.2f}% scripted success rate")
    print(f"[demos] Saved {meta['n_pairs']} (obs, action) pairs to {out_path}")
    print(f"[demos] Meta -> {meta_path}")

    env.close()


if __name__ == "__main__":
    import sys
    import traceback
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        sys.stdout.flush()
        sys.stderr.flush()
        print("\n[demos] !!! UNCAUGHT EXCEPTION !!!", flush=True)
        traceback.print_exc()
    finally:
        simulation_app.close()
