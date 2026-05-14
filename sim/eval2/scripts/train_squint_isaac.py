"""Train Squint's SAC+C51 policy on the Squint-native Isaac Lab env.

PHASE C — full SAC + C51 training loop
=====================================
Adds on top of the Phase B skeleton:
- ``LazyTensorStorage`` replay buffer keyed on (obs, next_obs, action, reward, done).
- ``update_main``  — critic + encoder C51 cross-entropy loss + autotune-α step.
- ``update_actor`` — SAC policy loss with ``detach_critic=True``.
- Soft target update via ``torch._foreach_lerp_(target_params, online_params, tau)``.
- Bootstrap-at-done=``always`` (Squint default — dones is always False;
  we always bootstrap from the next-obs Q-value).

Launch
------
.. code-block:: powershell

    cd C:\\Users\\user\\Desktop\\MA2\\isaac\\isaac_so_arm101
    $env:PYTHONUNBUFFERED='1'; $env:PYTHONIOENCODING='utf-8'
    C:/Users/user/Desktop/MA2/isaac/isaac_so_arm101/.venv/Scripts/python.exe `
        C:/Users/user/Desktop/MA2/robot-learning-project3/sim/eval2/scripts/train_squint_isaac.py `
        --num_envs 64 --total_steps 100000 --headless
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import torch  # noqa: F401 (early import for DLL load order)


parser = argparse.ArgumentParser(description="Squint SAC+C51 trainer on Isaac")
parser.add_argument("--task", type=str, default="Isaac-SquintNative-Place-v0",
                    help="Multi-env training variant (Play variant has num_envs=1).")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--total_steps", type=int, default=10_000,
                    help="Total env interactions (global_step counter).")
parser.add_argument("--learning_starts", type=int, default=5_000,
                    help="Pure-random rollout steps before policy takes over (Phase C).")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--policy_lr", type=float, default=3e-4)
parser.add_argument("--q_lr", type=float, default=3e-4)
parser.add_argument("--alpha_lr", type=float, default=3e-4)
parser.add_argument("--gamma", type=float, default=0.9)
parser.add_argument("--tau", type=float, default=0.01)
parser.add_argument("--num_q", type=int, default=2)
parser.add_argument("--num_atoms", type=int, default=101)
parser.add_argument("--v_min", type=float, default=-20.0)
parser.add_argument("--v_max", type=float, default=20.0)
parser.add_argument("--save_dir", type=str,
                    default=r"C:/Users/user/Desktop/MA2/robot-learning-project3/squint/runs")
parser.add_argument("--log_every", type=int, default=100)

# Replay / training schedule.
parser.add_argument("--buffer_size", type=int, default=1_000_000,
                    help="Capacity of the replay buffer (transitions).")
parser.add_argument("--batch_size", type=int, default=512,
                    help="Minibatch size sampled from the replay buffer.")
parser.add_argument("--num_updates", type=int, default=16,
                    help="Gradient updates per env step set. Squint uses 256; "
                         "we start low to keep the smoke test fast.")
parser.add_argument("--policy_frequency", type=int, default=4,
                    help="Update the actor every N critic updates (Squint=4).")
parser.add_argument("--target_network_frequency", type=int, default=1,
                    help="Target soft-update every N critic updates (Squint=1).")
parser.add_argument("--checkpoint_every", type=int, default=20_000,
                    help="Save checkpoint every N global_steps. 0 disables.")
parser.add_argument("--resume_from", type=str, default=None,
                    help="Path to a previous ckpt_isaac.pt to resume from "
                         "(loads encoder, actor, critic_target, log_alpha, global_step).")
parser.add_argument("--warmstart_from", type=str, default=None,
                    help="Path to an external ckpt (e.g. Squint ckpt 8) to warmstart "
                         "encoder + actor + critic (no global_step / log_alpha). "
                         "Automatically zero-pads state-projection layers when the "
                         "ckpt's state dim is smaller than the current env's (e.g. "
                         "ckpt8 has 18-d state, our 21-d env adds bowl_xyz).")
parser.add_argument("--target_entropy_scale", type=float, default=1.0,
                    help="Multiplier on the SAC target entropy. Default 1.0 → "
                         "target_entropy = -n_act = -6 (standard SAC). For warmstart "
                         "set 2.0 → -12 (tolerate near-deterministic actor). Auto-set "
                         "to 2.0 when --warmstart_from is given.")
parser.add_argument("--min_buffer_for_update", type=int, default=0,
                    help="Minimum number of transitions in the replay buffer before "
                         "any SAC gradient update fires. Default 0 (start updates as "
                         "soon as buffer has --batch_size). For warmstart we want a "
                         "diverse buffer first to avoid overfitting the critic on the "
                         "first few transitions — auto-set to 5000 with --warmstart_from.")
parser.add_argument("--freeze_actor_steps", type=int, default=0,
                    help="For the first N global_steps after training starts, ONLY "
                         "update the critic — actor remains frozen. This lets the "
                         "Q-function re-align to the new env's reward landscape "
                         "before the actor follows it. Critical for offline→online "
                         "fine-tuning where the warmstart critic was trained on a "
                         "slightly different reward function (cf. IBRL, RLPD, CalQL). "
                         "Auto-set to 20000 with --warmstart_from.")
parser.add_argument("--bc_loss_coef_start", type=float, default=0.0,
                    help="Initial weight λ of the BC regularization loss on the actor "
                         "update. The BC loss anchors the current actor's mean output "
                         "to a frozen teacher actor (= warmstart actor) so SAC drift "
                         "during the early training phase is bounded. Auto-set to 1.0 "
                         "with --warmstart_from. Set 0 to disable BC reg.")
parser.add_argument("--bc_loss_coef_decay_steps", type=int, default=80_000,
                    help="Linear decay horizon for the BC loss coefficient — λ goes "
                         "from --bc_loss_coef_start to 0 over this many global_steps. "
                         "After this point pure SAC training takes over.")

# Wandb (optional — only used if --track is passed).
parser.add_argument("--track", action="store_true", default=False,
                    help="Log to wandb (requires WANDB_API_KEY env var).")
parser.add_argument("--wandb_project", type=str, default="squint-isaac")
parser.add_argument("--wandb_entity", type=str, default=None)
parser.add_argument("--exp_name", type=str, default=None,
                    help="Run name on wandb. Defaults to a timestamp.")


from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

# ---------------------------------------------------------------------------
# Warmstart safety overrides
# ---------------------------------------------------------------------------
# When ``--warmstart_from`` is set we are starting from a pre-trained
# policy (e.g. Squint ckpt 8). The default SAC schedule destroys this
# warmstart for two reasons:
#   1) ``learning_starts > 0`` uses PURE RANDOM actions for the first N
#      steps, flooding the replay buffer with garbage transitions. SAC
#      then trains the critic on this garbage → critic predicts low Q
#      everywhere → the actor is pulled away from the warmstart toward
#      the entropy-maximising "random" policy.
#   2) The default LR (3e-4) is calibrated for from-scratch SAC. For
#      fine-tuning a competent policy we want a much smaller step so the
#      ckpt's representation is preserved while the critic catches up.
#
# Both are auto-overridden here unless the user explicitly passed the
# corresponding flags on the CLI.
if args.warmstart_from is not None:
    import sys as _sys
    cli_flags = set(_sys.argv[1:])
    overrides: list[str] = []
    if "--learning_starts" not in cli_flags and args.learning_starts != 0:
        args.learning_starts = 0
        overrides.append("learning_starts -> 0 (use warmstart actor from step 0)")
    # Asymmetric LR: critic learns the new env's reward landscape faster
    # than the actor should drift from the warmstart. This is the standard
    # offline→online recipe (RLPD, IBRL).
    if "--q_lr" not in cli_flags and args.q_lr != 3e-5:
        args.q_lr = 3e-5
        overrides.append("q_lr -> 3e-5 (critic fine-tune LR, 10× lower than from-scratch)")
    if "--alpha_lr" not in cli_flags and args.alpha_lr != 3e-5:
        args.alpha_lr = 3e-5
        overrides.append("alpha_lr -> 3e-5 (entropy auto-tune LR matched to critic)")
    if "--target_entropy_scale" not in cli_flags and args.target_entropy_scale != 2.0:
        args.target_entropy_scale = 2.0
        overrides.append("target_entropy_scale -> 2.0 (target_entropy = -2·n_act, tolerate deterministic actor)")
    if "--min_buffer_for_update" not in cli_flags and args.min_buffer_for_update != 20000:
        args.min_buffer_for_update = 20000
        overrides.append("min_buffer_for_update -> 20000 (buffer diversification before SAC kicks in)")
    if "--freeze_actor_steps" not in cli_flags and args.freeze_actor_steps != 20000:
        args.freeze_actor_steps = 20000
        overrides.append("freeze_actor_steps -> 20000 (critic-only phase to re-align Q to our env)")
    if "--bc_loss_coef_start" not in cli_flags and args.bc_loss_coef_start != 1.0:
        args.bc_loss_coef_start = 1.0
        overrides.append("bc_loss_coef_start -> 1.0 (BC anchor on warmstart teacher, decays linearly)")
    # Lower the policy LR further: critic adapts faster than actor in fine-tuning
    if "--policy_lr" not in cli_flags and args.policy_lr != 1e-5:
        args.policy_lr = 1e-5
        overrides.append("policy_lr -> 1e-5 (asymmetric: actor LR 1/3 of critic LR for stability)")
    if "--tau" not in cli_flags and args.tau != 0.005:
        args.tau = 0.005
        overrides.append("tau -> 0.005 (slower target-Q updates, standard SAC default)")
    if overrides:
        print(f"[warmstart] auto-overrides for warmstart safety:")
        for o in overrides:
            print(f"  - {o}")
        print(f"[warmstart] pass the flag explicitly to opt out.")

app = AppLauncher(args).app

import math  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from tensordict import TensorDict, from_module  # noqa: E402
from torchrl.data import LazyTensorStorage, ReplayBuffer  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import sim.eval2  # noqa: F401,E402  (task registration)
from sim.eval2.envs.squint_native import squint_terminations  # noqa: E402
from sim.eval2.policy import CNNEncoder, SquintActor, SquintCritic  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Episode bookkeeping (Isaac → ManiSkill-style metrics)
# ─────────────────────────────────────────────────────────────────────────────


class EpisodeTracker:
    """Per-env running episodic return + success_once flag.

    Isaac Lab doesn't ship ``infos['final_info']['episode']`` like ManiSkill.
    We compute it here: on every step, accumulate reward + success; on env
    reset (terminated | truncated), copy current values into ``last_*``
    buffers for logging, then zero the running counters.
    """

    def __init__(self, num_envs: int, device: torch.device):
        z = torch.zeros(num_envs, device=device)
        self.return_running = z.clone()
        self.success_once_running = z.clone().bool()
        self.lift_once_running = z.clone().bool()
        self.max_cube_z_running = z.clone() - 1.0  # init below any possible cube z
        self.length_running = z.clone().long()
        self.last_return = z.clone()
        self.last_success_once = z.clone().bool()
        self.last_success_at_end = z.clone().bool()
        self.last_lift_once = z.clone().bool()
        self.last_max_cube_z = z.clone()
        self.last_length = z.clone().long()
        self.n_episodes_done = 0
        self._dev = device

    def step(self, rewards: torch.Tensor, success_now: torch.Tensor,
             dones: torch.Tensor, lift_now: torch.Tensor | None = None,
             cube_z: torch.Tensor | None = None) -> None:
        self.return_running += rewards
        self.success_once_running = self.success_once_running | success_now
        if lift_now is not None:
            self.lift_once_running = self.lift_once_running | lift_now
        if cube_z is not None:
            self.max_cube_z_running = torch.maximum(self.max_cube_z_running, cube_z)
        self.length_running += 1

        if not dones.any():
            return
        idx = dones.nonzero(as_tuple=False).squeeze(-1)
        self.last_return[idx] = self.return_running[idx]
        self.last_success_once[idx] = self.success_once_running[idx]
        self.last_success_at_end[idx] = success_now[idx]
        self.last_lift_once[idx] = self.lift_once_running[idx]
        self.last_max_cube_z[idx] = self.max_cube_z_running[idx]
        self.last_length[idx] = self.length_running[idx]
        self.n_episodes_done += int(idx.numel())

        # Reset running counters for the envs that ended.
        self.return_running[idx] = 0.0
        self.success_once_running[idx] = False
        self.lift_once_running[idx] = False
        self.max_cube_z_running[idx] = -1.0
        self.length_running[idx] = 0

    def metrics(self) -> dict[str, float]:
        if self.n_episodes_done == 0:
            return {}
        return {
            "train/return": float(self.last_return.mean().item()),
            "train/success_once": float(self.last_success_once.float().mean().item()),
            "train/success_at_end": float(self.last_success_at_end.float().mean().item()),
            "train/lift_once": float(self.last_lift_once.float().mean().item()),
            "train/max_cube_z": float(self.last_max_cube_z.mean().item()),
            "train/episode_length": float(self.last_length.float().mean().item()),
            "train/episodes_done": int(self.n_episodes_done),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _obs_to_tensors(obs):
    """Isaac obs dict → (rgb, state) tensors that Squint nets expect.

    obs['policy'] is the 18-d state (qpos | target_qpos | goal_one_hot).
    obs['rgb']['rgb'] is the (N, 16, 16, 3) uint8 wrist image.

    Sanitises NaN/Inf in the state (degenerate physics during random rollout
    can push qpos to NaN; we replace those entries with zeros so the replay
    buffer never sees an unrecoverable batch).
    """
    state = obs["policy"]
    state = torch.where(torch.isfinite(state), state, torch.zeros_like(state))
    rgb_group = obs["rgb"]
    rgb = rgb_group["rgb"] if isinstance(rgb_group, dict) else rgb_group
    return rgb, state


def _action_bounds(env):
    """Extract per-action low/high tensors.

    Isaac Lab's gym Box presents ``±inf`` because the actual range is
    enforced by the action term, not the space. We read the configured
    ``bounds`` off the env's action term so the Squint actor sees a finite
    action_scale (otherwise tanh-scale-bias produces NaN).
    """
    base = env.unwrapped
    try:
        term = base.action_manager.get_term("arm_and_gripper")
        bounds = getattr(term.cfg, "bounds", None)
        if bounds is not None:
            b = torch.tensor(list(bounds), dtype=torch.float32)
            return -b, b
    except Exception:
        pass
    # Fallback: ±1 (still finite, lets training proceed).
    n_act = env.action_space.shape[-1]
    return -torch.ones(n_act), torch.ones(n_act)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 0. Optional wandb init -----------------------------------------
    run_name = args.exp_name or f"squint-isaac-{int(time.time())}"
    if args.track:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config=vars(args),
                save_code=True,
            )
            print(f"[train] wandb run: {wandb.run.url}")
        except Exception as e:
            print(f"[train] wandb init failed ({e}); continuing without tracking.")
            args.track = False

    # ---- 1. Build env ---------------------------------------------------
    env_cfg = parse_env_cfg(args.task, device=str(device), num_envs=args.num_envs)
    try:
        env_cfg.recorders = None
    except Exception:
        pass
    env = gym.make(args.task, cfg=env_cfg)
    base_env = env.unwrapped

    obs, _ = env.reset(seed=args.seed)
    rgb, state = _obs_to_tensors(obs)
    n_envs = state.shape[0]
    n_state = state.shape[-1]
    n_act = env.action_space.shape[-1]
    n_obs_hwc = (rgb.shape[1], rgb.shape[2], rgb.shape[3])

    action_low, action_high = _action_bounds(env)
    print(f"[train] action_low    = {action_low.tolist()}")
    print(f"[train] action_high   = {action_high.tolist()}")

    print(f"[train] task          = {args.task}")
    print(f"[train] num_envs      = {n_envs}")
    print(f"[train] state dim     = {n_state}")
    print(f"[train] action dim    = {n_act}")
    print(f"[train] rgb shape     = {n_obs_hwc}")
    print(f"[train] device        = {device}")

    # ---- 2. Build modules ----------------------------------------------
    encoder = CNNEncoder(n_obs=n_obs_hwc, device=device)
    actor = SquintActor(
        n_rgb_repr=encoder.repr_dim, n_state=n_state, n_act=n_act,
        action_low=action_low, action_high=action_high, device=device,
    )
    critic = SquintCritic(
        n_rgb_repr=encoder.repr_dim, n_state=n_state, n_act=n_act,
        num_atoms=args.num_atoms, v_min=args.v_min, v_max=args.v_max,
        num_q=args.num_q, device=device,
    )
    critic_target = SquintCritic(
        n_rgb_repr=encoder.repr_dim, n_state=n_state, n_act=n_act,
        num_atoms=args.num_atoms, v_min=args.v_min, v_max=args.v_max,
        num_q=args.num_q, device=device,
    )
    critic_target.load_state_dict(critic.state_dict())

    # Inference copies via tensordict weight-sharing (Squint pattern):
    encoder_detach = CNNEncoder(n_obs=n_obs_hwc, device=device)
    from_module(encoder).data.to_module(encoder_detach)
    actor_detach = SquintActor(
        n_rgb_repr=encoder.repr_dim, n_state=n_state, n_act=n_act,
        action_low=action_low, action_high=action_high, device=device,
    )
    from_module(actor).data.to_module(actor_detach)

    # ---- 3. Entropy autotune --------------------------------------------
    target_entropy = -float(n_act) * float(args.target_entropy_scale)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha = log_alpha.detach().exp()
    alpha_opt = torch.optim.Adam([log_alpha], lr=args.alpha_lr)

    # ---- 4. Optimizers --------------------------------------------------
    critic_opt = torch.optim.Adam(
        list(critic.parameters()) + list(encoder.parameters()),
        lr=args.q_lr,
    )
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.policy_lr)

    print(f"[train] encoder params   = {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"[train] actor params     = {sum(p.numel() for p in actor.parameters()):,}")
    print(f"[train] critic params    = {sum(p.numel() for p in critic.parameters()):,}")

    # ---- 5. Episode tracker --------------------------------------------
    tracker = EpisodeTracker(num_envs=n_envs, device=device)

    # ---- 6. Replay buffer ----------------------------------------------
    rb = ReplayBuffer(storage=LazyTensorStorage(args.buffer_size, device=device))
    print(f"[train] replay buffer  : LazyTensorStorage(capacity={args.buffer_size:,}, "
          f"device={device})")

    # Cache the online critic params + target params as flat lists so the
    # soft-update step uses ``torch._foreach_lerp_`` (Squint pattern).
    critic_online_params = list(critic.parameters())
    critic_target_params = list(critic_target.parameters())

    # ---- 7. Loss / update functions ------------------------------------
    def update_main(data):
        """Critic + encoder C51 update, plus autotune alpha step.

        Returns a TensorDict with logged scalars + the encoded current obs
        (so update_actor can reuse it without re-running the encoder).
        """
        next_obs_rgb = data["next_observations"]["rgb"]
        next_state = data["next_observations"]["state"]
        actions_b = data["actions"]
        rewards_b = data["rewards"].flatten()
        dones_b = data["dones"]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                next_obs_feat = encoder(next_obs_rgb)
                next_action, next_log_pi, _ = actor.get_action(next_obs_feat, next_state)

                bootstrap = (~dones_b).float()
                discount = args.gamma
                entropy_bonus = alpha * next_log_pi.flatten()
                rewards_with_entropy = rewards_b - bootstrap.flatten() * discount * entropy_bonus

                target_distributions = critic_target.categorical(
                    next_obs_feat, next_state, next_action,
                    rewards_with_entropy, bootstrap, discount,
                )

            obs_rgb = data["observations"]["rgb"]
            state_b = data["observations"]["state"]
            obs_feat = encoder(obs_rgb)

            q_outputs = critic(obs_feat, state_b, actions_b)
            q_log_probs = F.log_softmax(q_outputs, dim=-1)
            q_losses = -torch.sum(target_distributions * q_log_probs, dim=-1).mean(dim=-1)
            critic_loss = q_losses.sum()

            with torch.no_grad():
                q_probs = F.softmax(q_outputs, dim=-1)
                q_values = torch.sum(q_probs * critic.q_support, dim=-1)
                q_max = q_values.max()
                q_min = q_values.min()

        critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_opt.step()

        # Autotune alpha (entropy regularization).
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                _, log_pi_a, _ = actor.get_action(obs_feat, state_b)
            alpha_loss = (-log_alpha.exp() * (log_pi_a + target_entropy)).mean()
        alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        alpha_opt.step()
        alpha_new = log_alpha.detach().exp()

        return TensorDict(
            critic_loss=critic_loss.detach(),
            q_max=q_max, q_min=q_min,
            alpha=alpha_new, alpha_loss=alpha_loss.detach(),
            encoded_rgb=obs_feat.detach(),
            batch_size=[],
            device=device,
        )

    def update_actor(data, encoded_rgb, bc_coef: float = 0.0):
        """SAC policy gradient with ``detach_critic=True`` — Squint pattern.

        Gradient flows through ``actions`` only; critic params + encoder
        are frozen.

        Adds an optional BC regularization term that anchors the current
        actor's mean output to the frozen warmstart "teacher" actor:

            actor_loss = α·log_π - Q(s, π(s)) + bc_coef · MSE(μ_cur, μ_teacher)

        ``bc_coef`` is scheduled externally (linear decay over
        ``--bc_loss_coef_decay_steps``). When bc_coef = 0 or teacher_actor
        is None this collapses to vanilla SAC.
        """
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            state_b = data["observations"]["state"]
            pi, log_pi, _ = actor.get_action(encoded_rgb, state_b)
            q_values = critic.get_q_values(encoded_rgb, state_b, pi, detach_critic=True)
            critic_value = q_values.mean(dim=0)
            sac_loss = (alpha * log_pi - critic_value).mean()

            if teacher_actor is not None and bc_coef > 0.0:
                # MSE between pre-tanh means (mu) — this is the action-space
                # representation the actor outputs internally. Anchoring at
                # the mu level is more stable than at the action level
                # because the tanh squashing is identical between actors.
                mu_cur = actor.forward(encoded_rgb, state_b)
                with torch.no_grad():
                    mu_teacher = teacher_actor.forward(encoded_rgb, state_b)
                bc_loss = torch.nn.functional.mse_loss(mu_cur, mu_teacher)
                actor_loss = sac_loss + bc_coef * bc_loss
            else:
                bc_loss = torch.zeros((), device=device)
                actor_loss = sac_loss

        actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_opt.step()
        return TensorDict(
            actor_loss=actor_loss.detach(),
            sac_actor_loss=sac_loss.detach(),
            bc_loss=bc_loss.detach(),
            bc_coef=torch.tensor(float(bc_coef), device=device),
            batch_size=[], device=device,
        )

    # ---- 8. Training loop ----------------------------------------------
    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"{run_name}.pt")

    global_step = 0

    # Resume from a previous checkpoint if requested.
    if args.resume_from is not None and os.path.exists(args.resume_from):
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        actor.load_state_dict(ckpt["actor"])
        critic.load_state_dict(ckpt["critic"])
        critic_target.load_state_dict(ckpt["critic"])
        if "log_alpha" in ckpt:
            with torch.no_grad():
                log_alpha.copy_(ckpt["log_alpha"])
                alpha = log_alpha.detach().exp()
        global_step = int(ckpt.get("global_step", 0))
        print(f"[train] Resumed from {args.resume_from} @ step {global_step:,}")
    elif args.warmstart_from is not None and os.path.exists(args.warmstart_from):
        ckpt = torch.load(args.warmstart_from, map_location=device, weights_only=False)

        def _pad_state_proj(src_sd: dict, dst_sd: dict) -> dict:
            """For every key whose tensor shape differs ONLY in the last
            (input-feature) dim and the ckpt's dim is smaller, zero-pad on
            the right so the loaded weight handles the new bowl_xyz
            dimensions without disturbing the rest. Returns a new state
            dict ready for ``strict=False`` loading.
            """
            patched: dict = {}
            for k, v_src in src_sd.items():
                if k in dst_sd:
                    v_dst = dst_sd[k]
                    if v_src.shape == v_dst.shape:
                        patched[k] = v_src
                    elif (v_src.ndim == v_dst.ndim
                          and v_src.shape[:-1] == v_dst.shape[:-1]
                          and v_src.shape[-1] < v_dst.shape[-1]):
                        padded = torch.zeros_like(v_dst)
                        padded[..., :v_src.shape[-1]] = v_src.to(padded.dtype).to(padded.device)
                        patched[k] = padded
                        print(f"[warmstart] zero-pad {k}: "
                              f"{tuple(v_src.shape)} -> {tuple(v_dst.shape)}")
                    else:
                        print(f"[warmstart] SKIP shape mismatch {k}: "
                              f"ckpt={tuple(v_src.shape)} env={tuple(v_dst.shape)}")
                else:
                    print(f"[warmstart] SKIP unexpected key {k}")
            return patched

        # Encoder: same RGB input shape — direct load.
        try:
            encoder.load_state_dict(ckpt["encoder"])
            print("[warmstart] encoder loaded")
        except Exception as e:
            print(f"[warmstart] encoder load failed: {e!r}")

        # Actor: state input may have grown — pad state_proj.
        actor_src = ckpt.get("actor", {})
        actor_patched = _pad_state_proj(actor_src, actor.state_dict())
        missing, unexpected = actor.load_state_dict(actor_patched, strict=False)
        print(f"[warmstart] actor missing={list(missing)[:4]} unexpected={list(unexpected)[:4]}")

        # Critic (if present in ckpt): pad similarly.
        if "critic" in ckpt:
            critic_src = ckpt["critic"]
            critic_patched = _pad_state_proj(critic_src, critic.state_dict())
            missing, unexpected = critic.load_state_dict(critic_patched, strict=False)
            critic_target.load_state_dict(critic.state_dict())
            print(f"[warmstart] critic missing={list(missing)[:4]} unexpected={list(unexpected)[:4]}")
        else:
            print("[warmstart] no critic in ckpt — leaving freshly initialised")

        # Critical: preserve the warmstart actor's near-deterministic
        # behaviour. Without this, log_alpha starts at 0 (α=1.0), making
        # the SAC actor loss -(Q - α·log_π) dominated by the entropy
        # term — the optimizer then PUSHES the actor's log_std UP every
        # update, washing out the ckpt's tight action distribution. With
        # 3e-5 LR this still wrecks the warmstart in ~30-60 min.
        #
        # If the ckpt saved log_alpha → use it (true continuation).
        # Otherwise → seed log_alpha with a small value (α≈0.01) so the
        # actor stays near deterministic while the auto-tune refines.
        # Either way the user can override via env var SQUINT_INIT_LOG_ALPHA.
        if "log_alpha" in ckpt:
            with torch.no_grad():
                log_alpha.copy_(ckpt["log_alpha"].to(device).reshape_as(log_alpha))
                alpha = log_alpha.detach().exp()
            print(f"[warmstart] log_alpha loaded from ckpt: "
                  f"log_alpha={log_alpha.item():.4f} → α={alpha.item():.6f}")
        else:
            init_log_alpha = float(os.environ.get("SQUINT_INIT_LOG_ALPHA", -4.6))  # α ≈ 0.01
            with torch.no_grad():
                log_alpha.fill_(init_log_alpha)
                alpha = log_alpha.detach().exp()
            print(f"[warmstart] no log_alpha in ckpt — seeded with "
                  f"log_alpha={init_log_alpha:.4f} → α={alpha.item():.6f} "
                  f"(override via SQUINT_INIT_LOG_ALPHA env var)")

        global_step = 0  # warmstart always restarts the step counter
        print(f"[warmstart] {args.warmstart_from} → global_step reset to 0")

    # Teacher actor for BC regularization — a frozen copy of the actor right
    # after warmstart loading (or right after fresh init if no warmstart).
    # Used to compute the BC anchor loss in update_actor.
    teacher_actor = None
    if args.bc_loss_coef_start > 0.0:
        import copy as _copy
        teacher_actor = _copy.deepcopy(actor).to(device).eval()
        for p in teacher_actor.parameters():
            p.requires_grad_(False)
        n_teacher_params = sum(p.numel() for p in teacher_actor.parameters())
        print(f"[bc] teacher actor snapshot taken ({n_teacher_params:,} params, frozen). "
              f"BC coef: {args.bc_loss_coef_start} → 0 over {args.bc_loss_coef_decay_steps:,} steps.")

    start_time = time.perf_counter()
    print(f"[train] Starting SAC+C51 training. Total steps target: {args.total_steps:,}")
    print(f"[train] learning_starts={args.learning_starts:,}  "
          f"updates/step={args.num_updates}  batch={args.batch_size}  gamma={args.gamma}")

    last_log = {}
    # Best-metric tracking for separate "_best_*.pt" checkpoints.
    # We track two metrics independently because ``success_once`` may stay
    # at 0 for a long time early in training, during which ``lift_once``
    # still provides a useful signal for ranking partial competence.
    best_succ = -1.0
    best_lift = -1.0
    best_succ_path = ckpt_path.replace(".pt", "_best_success.pt")
    best_lift_path = ckpt_path.replace(".pt", "_best_lift.pt")

    def _save_best(metric_name: str, current: float, prev_best: float,
                   target_path: str) -> float:
        if current <= prev_best:
            return prev_best
        torch.save({
            "encoder": encoder.state_dict(),
            "actor": actor.state_dict(),
            "critic": critic_target.state_dict(),
            "log_alpha": log_alpha.detach().clone(),
            "global_step": global_step,
            "best_metric": {"name": metric_name, "value": current},
            "args": vars(args),
        }, target_path)
        print(f"[train] NEW BEST {metric_name}={current:.4f} (prev {prev_best:.4f}) "
              f"@ step {global_step:,} → {target_path}")
        return current

    warmstart_loaded = (args.warmstart_from is not None
                        and os.path.exists(args.warmstart_from))

    while global_step < args.total_steps:
        # Action selection.
        # If warmstart was loaded, NEVER take random actions — we want the
        # buffer filled with policy-driven (= competent) transitions from
        # step 0. Random actions during learning_starts would flood the
        # buffer with garbage and corrupt the critic's targets.
        if global_step < args.learning_starts and not warmstart_loaded:
            actions = torch.empty(n_envs, n_act, device=device).uniform_(-1, +1)
            actions = action_low.to(device) + (actions + 1.0) * 0.5 * (action_high - action_low).to(device)
        else:
            with torch.no_grad():
                # Sync inference copies with online weights.
                from_module(encoder).data.to_module(encoder_detach)
                from_module(actor).data.to_module(actor_detach)
                rgb_feat = encoder_detach(rgb)
                actions, _, _ = actor_detach.get_action(rgb_feat, state)

        next_obs, rewards, term, trunc, info = env.step(actions)
        dones_real = term | trunc

        # Squint "bootstrap_at_done=always": dones used for TD bootstrap is
        # forced to False so the target Q always includes V(next_obs).
        dones_for_buffer = torch.zeros_like(dones_real, dtype=torch.bool)

        # Success is no longer a termination term (Squint partial_reset=False
        # parity) — compute it directly from cube + bowl positions.
        # Lift is "cube ever lifted ≥ 4 cm above the table this episode" —
        # cheap diagnostic that fires before the policy can actually place.
        with torch.no_grad():
            success_now = squint_terminations.success(base_env)
            cube_z = base_env.scene["cube"].data.root_pos_w[:, 2]
            lift_now = cube_z >= 0.04
        tracker.step(rewards, success_now, dones_real, lift_now=lift_now, cube_z=cube_z)

        # Build replay-buffer transition (TensorDict, like Squint).
        next_rgb, next_state_t = _obs_to_tensors(next_obs)

        # Squint pattern: for envs that ended this step, replace the
        # post-reset ``next_obs`` with the pre-reset ``final_observation``
        # so V(next) bootstrap targets the true terminal state.
        # Isaac surfaces this via the extras dict that SquintNativePlaceEnv.step()
        # populates on every reset.
        if "final_observation" in info and info.get("final_observation"):
            final = info["final_observation"]
            final_ids = info.get("final_observation_env_ids", None)
            if final_ids is not None and final_ids.numel() > 0:
                if "policy" in final:
                    state_clean = torch.where(
                        torch.isfinite(final["policy"]),
                        final["policy"], torch.zeros_like(final["policy"]),
                    )
                    next_state_t = next_state_t.clone()
                    next_state_t[final_ids] = state_clean[final_ids]
                if "rgb" in final and isinstance(final["rgb"], dict) and "rgb" in final["rgb"]:
                    next_rgb = next_rgb.clone()
                    next_rgb[final_ids] = final["rgb"]["rgb"][final_ids]

        transition = TensorDict({
            "observations": TensorDict({"rgb": rgb, "state": state}, batch_size=[n_envs]),
            "next_observations": TensorDict({"rgb": next_rgb, "state": next_state_t}, batch_size=[n_envs]),
            "actions": actions.float(),
            "rewards": rewards.float(),
            "dones": dones_for_buffer,
        }, batch_size=[n_envs], device=device)
        rb.extend(transition)

        # Advance current obs.
        obs = next_obs
        rgb, state = next_rgb, next_state_t
        global_step += n_envs

        # Updates.
        # Gate by BOTH learning_starts (for from-scratch random rollout
        # warmup) AND min_buffer_for_update (for warmstart buffer
        # diversification before the critic starts learning).
        if (global_step >= args.learning_starts
                and len(rb) >= max(args.batch_size, args.min_buffer_for_update)):
            # ── BC coef schedule (linear decay) ──────────────────────────────
            # Starts at bc_loss_coef_start, decays linearly to 0 over
            # bc_loss_coef_decay_steps. After that, pure SAC.
            if args.bc_loss_coef_start > 0.0 and args.bc_loss_coef_decay_steps > 0:
                progress = min(1.0, global_step / float(args.bc_loss_coef_decay_steps))
                bc_coef_now = args.bc_loss_coef_start * (1.0 - progress)
            else:
                bc_coef_now = 0.0

            # ── Actor freeze phase ───────────────────────────────────────────
            # While step < freeze_actor_steps, ONLY update the critic. The
            # actor stays at the warmstart weights so the critic can re-align
            # to our env's reward landscape without dragging the actor with it.
            actor_active = global_step >= args.freeze_actor_steps

            for grad_step in range(args.num_updates):
                data = rb.sample(args.batch_size)
                out_main = update_main(data)
                encoded_rgb = out_main.pop("encoded_rgb")
                last_log.update(out_main.to_dict())
                if actor_active and grad_step % args.policy_frequency == 0:
                    out_actor = update_actor(data, encoded_rgb, bc_coef=bc_coef_now)
                    last_log.update(out_actor.to_dict())
                if grad_step % args.target_network_frequency == 0:
                    with torch.no_grad():
                        torch._foreach_lerp_(
                            critic_target_params, critic_online_params, args.tau
                        )

        # Logging.
        if global_step % args.log_every < n_envs:
            metrics = tracker.metrics()
            sps = global_step / (time.perf_counter() - start_time)
            if global_step < args.learning_starts:
                phase = "rollout"  # pure-random data collection
            elif len(rb) < max(args.batch_size, args.min_buffer_for_update):
                phase = "warmup"   # warmstart actor collecting, no SAC updates yet
            elif global_step < args.freeze_actor_steps:
                phase = "Q-only"   # critic re-aligning, actor frozen at warmstart
            else:
                phase = "train"
            critic_loss = float(last_log.get("critic_loss", 0.0))
            actor_loss = float(last_log.get("actor_loss", 0.0))
            alpha_v = float(last_log.get("alpha", 0.0))
            q_max = float(last_log.get("q_max", 0.0))
            q_min = float(last_log.get("q_min", 0.0))
            # ETA: how long until total_steps at current throughput.
            steps_left = max(args.total_steps - global_step, 0)
            eta_s = steps_left / max(sps, 1e-6)
            eta_h = int(eta_s // 3600)
            eta_m = int((eta_s % 3600) // 60)
            pct = 100.0 * global_step / max(args.total_steps, 1)
            bc_coef_v = float(last_log.get("bc_coef", 0.0))
            bc_loss_v = float(last_log.get("bc_loss", 0.0))
            print(f"[train] {phase:7s}  step={global_step:7d}  "
                  f"({pct:5.1f}%)  sps={sps:7.1f}  ETA={eta_h:02d}h{eta_m:02d}m  "
                  f"ret={metrics.get('train/return', 0.0):+.3f}  "
                  f"lift={metrics.get('train/lift_once', 0.0):.3f}  "
                  f"succ={metrics.get('train/success_once', 0.0):.3f}  "
                  f"max_z={metrics.get('train/max_cube_z', 0.0):.3f}  "
                  f"bc={bc_coef_v:.2f}/L={bc_loss_v:.3f}  "
                  f"eps={metrics.get('train/episodes_done', 0):4d}  "
                  f"critic_l={critic_loss:+.3f}  actor_l={actor_loss:+.3f}  "
                  f"α={alpha_v:.4f}  Q[{q_min:+.1f},{q_max:+.1f}]  rb={len(rb)}")
            if args.track:
                import wandb
                wandb.log({
                    "train/sps": sps,
                    "train/global_step": global_step,
                    "train/critic_loss": critic_loss,
                    "train/actor_loss": actor_loss,
                    "train/alpha": alpha_v,
                    "train/q_max": q_max,
                    "train/q_min": q_min,
                    "train/buffer_size": len(rb),
                    **metrics,
                }, step=global_step)

        # Best-metric checkpoint saves — independent of the periodic save.
        # Only consider metrics for which at least one episode finished
        # since the last log window (otherwise tracker.metrics() returns
        # stale ``last_*`` values and we'd re-save the same ckpt forever).
        if (global_step % args.log_every < n_envs
                and metrics.get("train/episodes_done", 0) > 0):
            succ_now = float(metrics.get("train/success_once", 0.0))
            lift_now_metric = float(metrics.get("train/lift_once", 0.0))
            best_succ = _save_best("success_once", succ_now, best_succ, best_succ_path)
            best_lift = _save_best("lift_once", lift_now_metric, best_lift, best_lift_path)

        # Checkpoint save.
        if args.checkpoint_every > 0 and global_step % args.checkpoint_every < n_envs:
            torch.save({
                "encoder": encoder.state_dict(),
                "actor": actor.state_dict(),
                "critic": critic_target.state_dict(),
                "log_alpha": log_alpha.detach().clone(),
                "global_step": global_step,
                "args": vars(args),
            }, ckpt_path)
            print(f"[train] Saved checkpoint @ step {global_step:,} → {ckpt_path}")
            if args.track:
                import wandb
                try:
                    artifact = wandb.Artifact(name=f"model_{run_name}", type="model")
                    artifact.add_file(ckpt_path)
                    wandb.log_artifact(artifact)
                except Exception:
                    pass

    # Final checkpoint save at the end of training.
    torch.save({
        "encoder": encoder.state_dict(),
        "actor": actor.state_dict(),
        "critic": critic_target.state_dict(),
        "log_alpha": log_alpha.detach().clone(),
        "global_step": global_step,
        "args": vars(args),
    }, ckpt_path)
    print(f"[train] Final ckpt → {ckpt_path}")

    print(f"[train] Finished {args.total_steps:,} steps in "
          f"{time.perf_counter() - start_time:.1f} s")
    if args.track:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass
    env.close()
    app.close()


if __name__ == "__main__":
    main()
