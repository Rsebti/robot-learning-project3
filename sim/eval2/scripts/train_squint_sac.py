"""Squint SAC+C51 training loop adapted to Isaac Lab — full architecture.

Faithful port of ``train_squint.py`` from
https://github.com/aalmuzairee/squint:

- CNNEncoder + Projection + Actor + Critic (in
  ``sim/eval2/policy/squint_sac.py``)
- vmap'd Q ensemble via ``tensordict.from_modules`` (in Critic)
- Update functions take a single ``TensorDict`` and return a
  ``TensorDict`` — compatible with ``torch.compile`` and
  ``CudaGraphModule``
- Encoder weight-sharing via ``tensordict.from_module(encoder).data.to_module(encoder_eval)``
  for rollout / eval inference
- C51 distributional projection
- Bootstrap-always mode
- Polyak averaging on target critic
- Auto-tuned entropy

Adaptations from Squint that DO change behavior (env-level only —
sim engine and robot model differ):
- Source env is Isaac Lab ``ManagerBasedRLEnv`` instead of ManiSkill
  ``ManiSkillVectorEnv``. Obs comes out as ``obs["policy"]`` (state
  vector) and ``obs["wrist"]`` (raw RGB 64x64 from ``TiledCamera``).
- We downsample the rgb 64 → 16 in the trainer (Squint downsamples
  128 → 16 — same final input shape, just one less render pixel).
- Color jitter is applied here (Squint applies via wrapper).

Launch
------
.. code-block:: powershell

    cd C:\\Users\\user\\Desktop\\MA2\\isaac\\isaac_so_arm101
    C:/Users/user/Desktop/MA2/isaac/isaac_so_arm101/.venv/Scripts/python.exe `
        -m sim.eval2.scripts.train_squint_sac `
        --task Isaac-SO101-Squint-Lift-v0 `
        --headless --enable_cameras --num_envs 256 --total_timesteps 1500000

Tip: 256 envs × 256 updates per env step + camera rendering at 64x64 +
C51 atoms (101) is heavy. If RTX 5070 OOMs, drop ``--num_envs 128``
or ``--num_updates 64``.
"""
from __future__ import annotations

import argparse
import os
import random
import time
from collections import deque

import torch
import torch.nn.functional as F
import torch.optim as optim


# ---------------------------------------------------------------------------
# CLI args + Isaac Lab AppLauncher (must come before any isaaclab imports).
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Squint SAC+C51 trainer for Isaac Lab")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--total_timesteps", type=int, default=1_500_000)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--learning_starts", type=int, default=5_000)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--buffer_size", type=int, default=500_000)
parser.add_argument("--num_updates", type=int, default=128, help="Updates per env step (Squint default 256)")
parser.add_argument("--policy_lr", type=float, default=3e-4)
parser.add_argument("--q_lr", type=float, default=3e-4)
parser.add_argument("--alpha_lr", type=float, default=3e-4)
parser.add_argument("--policy_frequency", type=int, default=4)
parser.add_argument("--target_network_frequency", type=int, default=1)
parser.add_argument("--gamma", type=float, default=0.9)
parser.add_argument("--tau", type=float, default=0.01)
parser.add_argument("--num_q", type=int, default=2)
parser.add_argument("--num_atoms", type=int, default=101)
parser.add_argument("--v_min", type=float, default=-20.0)
parser.add_argument("--v_max", type=float, default=20.0)
parser.add_argument("--alpha", type=float, default=0.2)
parser.add_argument("--image_size", type=int, default=16, help="CNN encoder input size (after downsample)")
parser.add_argument("--apply_jitter", action="store_true", default=True)
parser.add_argument("--compile", action="store_true", default=True, help="torch.compile update fns + actor")
parser.add_argument("--no-compile", dest="compile", action="store_false")
parser.add_argument("--cudagraphs", action="store_true", default=False, help="CudaGraphModule on top of compile (risky on tensordict 0.12)")
parser.add_argument("--save_interval", type=int, default=50_000)
parser.add_argument("--exp_name", type=str, default="squint_sac")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


# ---------------------------------------------------------------------------
# Now safe to import everything else.
# ---------------------------------------------------------------------------

import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from tensordict import TensorDict, from_module  # noqa: E402
from torchrl.data import LazyTensorStorage, ReplayBuffer  # noqa: E402

import sim.eval2  # noqa: F401,E402  (registers our gym tasks)
from sim.eval2.policy.squint_sac import (  # noqa: E402
    CNNEncoder,
    SquintActor,
    SquintCritic,
)


# ---------------------------------------------------------------------------
# Helpers — image preprocessing (downsample + jitter)
# ---------------------------------------------------------------------------


def _downsample_rgb(rgb: torch.Tensor, target_size: int) -> torch.Tensor:
    """Bilinear downsample HWC uint8 → HWC uint8 at target_size."""
    if rgb.shape[-3] == target_size:
        return rgb
    rgb = rgb.permute(0, 3, 1, 2).float()
    rgb = F.interpolate(rgb, size=(target_size, target_size), mode="area")
    return rgb.permute(0, 2, 3, 1).to(torch.uint8)


def _color_jitter(rgb: torch.Tensor, brightness: float = 0.2, contrast: float = 0.2) -> torch.Tensor:
    """Per-batch random brightness + contrast jitter on uint8 HWC images."""
    rgb_f = rgb.float()
    B = rgb_f.shape[0]
    device = rgb_f.device
    b = (1.0 - brightness) + 2.0 * brightness * torch.rand(B, 1, 1, 1, device=device)
    c = (1.0 - contrast) + 2.0 * contrast * torch.rand(B, 1, 1, 1, device=device)
    mean = rgb_f.mean(dim=(1, 2, 3), keepdim=True)
    rgb_f = (rgb_f - mean) * c + mean * b
    return rgb_f.clamp(0, 255).to(torch.uint8)


def _extract_obs(obs: dict, image_size: int, apply_jitter: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract (rgb, state) from Isaac Lab obs dict.

    Expects ``obs["wrist"]`` (B, H, W, 3) uint8 and ``obs["policy"]``
    (B, n_state) float.
    """
    if isinstance(obs["wrist"], dict):
        # WristRgbObsCfg with concatenate_terms=False returns a dict
        # {"rgb": tensor}
        rgb = obs["wrist"]["rgb"]
    else:
        rgb = obs["wrist"]
    state = obs["policy"]
    if not torch.is_tensor(state):
        state = torch.as_tensor(state)
    rgb = _downsample_rgb(rgb, image_size)
    if apply_jitter:
        rgb = _color_jitter(rgb)
    return rgb, state.float()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = f"{args.exp_name}__{args.task}__{args.seed}__{int(time.time())}"
    save_dir = os.path.abspath(f"runs/{run_name}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"[squint_sac] Run dir: {save_dir}")

    # ---- Env --------------------------------------------------------
    env_cfg = parse_env_cfg(args.task, device=str(device), num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    base_env = env.unwrapped
    n_envs = base_env.num_envs

    action_space = base_env.action_space
    n_act = int(action_space.shape[-1])
    raw_low = torch.from_numpy(action_space.low.reshape(-1)[:n_act]).float()
    raw_high = torch.from_numpy(action_space.high.reshape(-1)[:n_act]).float()
    # Isaac Lab's JointPositionAction reports unbounded action_space (-inf,
    # inf) because the joint pos target itself is unbounded. SAC's tanh
    # squashes to [-1, 1] so we clip the bounds to that range; otherwise
    # action_scale = (inf - (-inf))/2 = NaN and all actions become NaN.
    action_low = torch.where(torch.isfinite(raw_low), raw_low, torch.full_like(raw_low, -1.0))
    action_high = torch.where(torch.isfinite(raw_high), raw_high, torch.full_like(raw_high, 1.0))
    print(f"[squint_sac] action bounds: low={action_low.tolist()}  high={action_high.tolist()}")

    obs, _ = env.reset(seed=args.seed)
    rgb, state = _extract_obs(obs, args.image_size, apply_jitter=False)  # no jitter for shape probe
    n_state = int(state.shape[-1])
    n_channels = int(rgb.shape[-1])
    print(
        f"[squint_sac] Env: {args.task} | num_envs={n_envs} "
        f"| n_state={n_state} | n_act={n_act} | rgb={rgb.shape}"
    )

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
    critic = SquintCritic(
        n_rgb_repr=encoder.repr_dim,
        n_state=n_state,
        n_act=n_act,
        num_atoms=args.num_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
        num_q=args.num_q,
        device=device,
    )
    critic_target = SquintCritic(
        n_rgb_repr=encoder.repr_dim,
        n_state=n_state,
        n_act=n_act,
        num_atoms=args.num_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
        num_q=args.num_q,
        device=device,
    )
    critic_target.load_state_dict(critic.state_dict())
    for p in critic_target.parameters():
        p.requires_grad_(False)

    # Inference encoder copies — Squint trick: weight-sharing via tensordict.
    encoder_detach = CNNEncoder(
        n_obs=(args.image_size, args.image_size, n_channels), device=device
    )
    encoder_eval = CNNEncoder(
        n_obs=(args.image_size, args.image_size, n_channels), device=device
    ).eval()
    from_module(encoder).data.to_module(encoder_detach)
    from_module(encoder).data.to_module(encoder_eval)

    actor_detach = SquintActor(
        n_rgb_repr=encoder.repr_dim,
        n_state=n_state, n_act=n_act,
        action_low=action_low, action_high=action_high,
        device=device,
    )
    from_module(actor).data.to_module(actor_detach)

    # Entropy auto-tune
    target_entropy = -float(n_act)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha = log_alpha.detach().exp()

    # Optimizers
    actor_opt = optim.Adam(actor.parameters(), lr=args.policy_lr)
    critic_opt = optim.Adam(
        list(critic.parameters()) + list(encoder.parameters()),
        lr=args.q_lr,
    )
    alpha_opt = optim.Adam([log_alpha], lr=args.alpha_lr)

    # Replay buffer
    rb = ReplayBuffer(storage=LazyTensorStorage(args.buffer_size, device=device))

    # ---- Update closures (TensorDict in, TensorDict out) ------------

    def get_rollout_action(rgb_in: torch.Tensor, state_in: torch.Tensor) -> torch.Tensor:
        rgb_feat = encoder_detach(rgb_in)
        action, _, _ = actor_detach.get_action(rgb_feat, state_in)
        return action

    def update_main(data: TensorDict) -> TensorDict:
        with torch.no_grad():
            next_rgb_feat = encoder(data["next_rgb"])
            next_state = data["next_state"]
            next_action, next_logp, _ = actor.get_action(next_rgb_feat, next_state)

            done = data["done"]
            bootstrap = (~done).float()
            rewards = data["reward"].flatten()
            entropy_bonus = alpha * next_logp.flatten()
            rewards_with_entropy = rewards - bootstrap.flatten() * args.gamma * entropy_bonus

            target_dist = critic_target.categorical(
                next_rgb_feat, next_state, next_action,
                rewards_with_entropy, bootstrap, args.gamma,
            )

        rgb_feat = encoder(data["rgb"])
        state = data["state"]
        action = data["action"]
        logits = critic(rgb_feat, state, action)
        log_probs = F.log_softmax(logits, dim=-1)
        losses = -(target_dist * log_probs).sum(dim=-1).mean(dim=-1)
        critic_loss = losses.sum()

        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        # Alpha update (autotune)
        with torch.no_grad():
            _, logp_alpha, _ = actor.get_action(rgb_feat, state)
        alpha_loss = (-log_alpha.exp() * (logp_alpha + target_entropy)).mean()
        alpha_opt.zero_grad()
        alpha_loss.backward()
        alpha_opt.step()

        # Track current alpha (mutated in-place via the captured variable
        # below — caller updates it after each call).
        return TensorDict(
            {
                "critic_loss": critic_loss.detach(),
                "alpha_loss": alpha_loss.detach(),
                "encoded_rgb": rgb_feat.detach(),
            },
            batch_size=[],
        )

    def update_actor(data: TensorDict, encoded_rgb: torch.Tensor) -> TensorDict:
        state = data["state"]
        pi, logp, _ = actor.get_action(encoded_rgb, state)
        q_values = critic.get_q_values(encoded_rgb, state, pi, detach_critic=True)
        critic_value = q_values.mean(dim=0)
        actor_loss = (alpha * logp - critic_value).mean()

        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()
        return TensorDict({"actor_loss": actor_loss.detach()}, batch_size=[])

    # ---- torch.compile + CudaGraphModule wrapping --------------------

    if args.compile:
        print("[squint_sac] torch.compile ON")
        update_main = torch.compile(update_main)
        update_actor = torch.compile(update_actor)
        get_rollout_action = torch.compile(get_rollout_action)

    if args.cudagraphs:
        print("[squint_sac] CudaGraphModule ON (risky on tensordict 0.12)")
        try:
            from tensordict.nn import CudaGraphModule
            update_main = CudaGraphModule(update_main)
            update_actor = CudaGraphModule(update_actor)
        except Exception as exc:  # noqa: BLE001
            print(f"[squint_sac] CudaGraphModule wrap failed ({exc}); continuing without")

    # ---- Training loop ----------------------------------------------

    rgb, state = _extract_obs(obs, args.image_size, args.apply_jitter)
    global_step = 0
    n_total_iterations = args.total_timesteps // n_envs
    print(f"[squint_sac] {n_total_iterations} iterations × {n_envs} envs = {args.total_timesteps} steps")

    last_print = time.time()
    avg_reward = deque(maxlen=20)

    for iteration in range(n_total_iterations):
        # Action selection
        if global_step < args.learning_starts:
            action = torch.empty(n_envs, n_act, device=device).uniform_(-1.0, 1.0)
            action = action * actor.action_scale + actor.action_bias
        else:
            with torch.no_grad():
                action = get_rollout_action(rgb, state)

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_rgb, next_state = _extract_obs(next_obs, args.image_size, args.apply_jitter)

        # Bootstrap-always: don't propagate done into target (Squint default)
        done = torch.zeros_like(terminated, dtype=torch.bool)

        transition = TensorDict(
            {
                "rgb": rgb,
                "state": state,
                "action": action,
                "reward": reward.float(),
                "next_rgb": next_rgb,
                "next_state": next_state,
                "done": done,
            },
            batch_size=[n_envs],
            device=device,
        )
        rb.extend(transition)

        avg_reward.append(reward.mean().item())

        rgb, state = next_rgb, next_state
        global_step += n_envs

        # ---- SAC updates ----
        if global_step > args.learning_starts:
            for grad_step in range(args.num_updates):
                batch = rb.sample(args.batch_size)
                out_main = update_main(batch)
                encoded_rgb = out_main.pop("encoded_rgb", None)
                if grad_step % args.policy_frequency == 0:
                    update_actor(batch, encoded_rgb)
                # Polyak averaging on target critic.
                if grad_step % args.target_network_frequency == 0:
                    with torch.no_grad():
                        for p, p_t in zip(critic.parameters(), critic_target.parameters()):
                            p_t.data.lerp_(p.data, args.tau)

                # Refresh inference copies (free if weights are shared via from_module)
            # Sync inference copies after each iteration's updates.
            from_module(encoder).data.to_module(encoder_detach)
            from_module(actor).data.to_module(actor_detach)
            alpha = log_alpha.detach().exp()

        # ---- Logging ----
        now = time.time()
        if now - last_print > 5.0:
            sps = (n_envs * (iteration + 1)) / max(0.01, now - last_print) if iteration < 5 else n_envs / max(0.01, now - last_print)
            r_mean = sum(avg_reward) / max(1, len(avg_reward)) if avg_reward else 0.0
            print(
                f"[squint_sac] step={global_step} "
                f"| step_reward_mean(20iter)={r_mean:+.3f} "
                f"| alpha={float(alpha):.4f} "
                f"| sps~{sps:.0f}"
            )
            last_print = now

        # ---- Checkpoint ----
        if args.save_interval > 0 and (
            (global_step // args.save_interval)
            > ((global_step - n_envs) // args.save_interval)
        ):
            ckpt_path = os.path.join(save_dir, "ckpt.pt")
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "critic_target": critic_target.state_dict(),
                    "log_alpha": log_alpha.detach().clone(),
                    "global_step": global_step,
                },
                ckpt_path,
            )
            print(f"[squint_sac] Saved checkpoint at step {global_step}: {ckpt_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
