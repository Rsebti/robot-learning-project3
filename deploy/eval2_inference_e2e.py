"""Eval 2 — end-to-end vision deploy of an ACT (or DAPG-finetuned) policy.

Architectural pivot from the modular ``eval2_inference.py``: there's no
separate perception CNN. The wrist camera image goes DIRECTLY into the
policy, alongside joint state + goal-conditioning inputs (target color
and bowl xyz). Single model, single inference call.

    [wrist camera RGB] ─┐
                        │
    [joint_pos/vel]    ─┼─── ACT policy ─── action (6-D) ──> sim/robot
    [target_color]     ─┤
    [bowl_xyz]         ─┘

The policy is expected to be a lerobot ACT checkpoint trained on
teleop demos that include image + state + goal-conditioning fields.
Deploy reads ``policy.config.input_features`` to know what to provide,
so the script adapts to whatever feature layout the training used —
no hardcoded slice indices.

Usage (sim mode, fixed PC with Isaac Sim):

    uv run python deploy/eval2_inference_e2e.py \\
        --task Eval2-PickInClutter-Play-v2 \\
        --policy Rsebti/projet3-act-eval2-bc \\
        --target_color red \\
        --bowl_x 0.20 --bowl_y -0.15 --bowl_z 0.02 \\
        --num_episodes 20

When the env randomizes target_color / bowl per reset, override with
``--use_env_goal`` to read these from the env at each reset (matches the
distribution the policy was trained on if demos were recorded with the
same randomization).

Real-robot mode is NOT yet wired here — the final deploy on SO-101 will
go through ``lerobot-record --policy.path=<repo>`` (the same flow as the
sanity check), with target_color and bowl_xyz fed as additional task
metadata. Will be added separately.
"""

# ===========================================================================
# CRITICAL: AppLauncher must be invoked BEFORE any torch / isaaclab import.
# ===========================================================================
from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Eval 2 end-to-end deploy (sim).")
parser.add_argument(
    "--task",
    default="Eval2-PickInClutter-Play-v2",
    help="Gym task ID. Must be a v2 variant (with wrist_cam in the scene).",
)
parser.add_argument(
    "--policy",
    required=True,
    help=(
        "HuggingFace repo ID or local path to the ACT checkpoint. "
        "Example: Rsebti/projet3-act-eval2-bc"
    ),
)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--num_episodes", type=int, default=20)
parser.add_argument(
    "--target_color",
    choices=["red", "blue"],
    default=None,
    help=(
        "Target block color. If unset, --use_env_goal must be on, and we "
        "read whatever the env randomized to."
    ),
)
parser.add_argument("--bowl_x", type=float, default=None)
parser.add_argument("--bowl_y", type=float, default=None)
parser.add_argument("--bowl_z", type=float, default=None)
parser.add_argument(
    "--use_env_goal",
    action="store_true",
    help=(
        "Read target_color and bowl_xyz from the env state at each reset "
        "instead of from CLI args. Matches the training distribution when "
        "demos were recorded with randomized goals."
    ),
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--no_headless",
    action="store_true",
    help="Open the Isaac Sim window for visual inspection. Slower.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Force --enable_cameras on (the v2 task spawns a CameraCfg; without this
# flag Isaac Lab silently disables camera rendering).
args.enable_cameras = True
args.headless = not args.no_headless

if not args.use_env_goal and (
    args.target_color is None
    or args.bowl_x is None
    or args.bowl_y is None
    or args.bowl_z is None
):
    raise SystemExit(
        "[deploy] either pass --target_color + --bowl_x/y/z explicitly, "
        "or use --use_env_goal to read these from each env reset."
    )

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


# ===========================================================================
# Heavy imports: only after AppLauncher.
# ===========================================================================
def main():
    import gymnasium as gym
    import torch
    import torch.nn.functional as F
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab.utils.math import subtract_frame_transforms
    from lerobot.policies.act.modeling_act import ACTPolicy

    # Side-effect: registers our gym envs.
    import sim.eval2  # noqa: F401

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # ---- env ----
    env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    print(f"[deploy] gym.make({args.task!r}, num_envs={args.num_envs})")
    env = gym.make(args.task, cfg=env_cfg)
    env.reset()
    scene = env.unwrapped.scene
    robot = scene["robot"]
    wrist_cam = scene["wrist_cam"]
    block_red = scene["block_red"]
    block_blue = scene["block_blue"]
    bowl_floor = scene["bowl_floor"]

    # ---- policy ----
    print(f"[deploy] loading ACT policy from {args.policy!r}...")
    policy = ACTPolicy.from_pretrained(args.policy).to(device).eval()
    feats = policy.config.input_features
    print(f"[deploy] policy input features: {list(feats.keys())}")
    print(f"[deploy] chunk_size={policy.config.chunk_size}, "
          f"n_action_steps={policy.config.n_action_steps}")

    # The script supports two common layouts. If the actual training uses
    # different feature names, add them to STATE_BUILDERS / IMAGE_KEY below.
    image_key = next(
        (k for k in feats if k.startswith("observation.image")), None
    )
    if image_key is None:
        raise SystemExit(
            f"[deploy] ACT config has no observation.image* feature. Got "
            f"{list(feats.keys())}. End-to-end deploy needs an image input."
        )
    image_h, image_w = feats[image_key].shape[-2:]
    print(f"[deploy] image_key={image_key}, expected size=({image_h}, {image_w})")

    state_key = "observation.state"
    state_dim = feats[state_key].shape[0] if state_key in feats else None
    print(f"[deploy] state_key={state_key}, expected dim={state_dim}")

    # ---- goal-conditioning resolution ----
    if not args.use_env_goal:
        # Static goal from CLI args, broadcast to all envs.
        cli_target_color = 0 if args.target_color == "red" else 1
        cli_bowl_xyz = torch.tensor(
            [args.bowl_x, args.bowl_y, args.bowl_z], device=device
        )

    # ---- rollout ----
    n_envs = env.unwrapped.num_envs
    n_episodes_done = 0
    n_successes = 0
    episode_success_seen = torch.zeros(n_envs, dtype=torch.bool, device=device)

    print(
        f"[deploy] rollout: target {args.num_episodes} episodes across "
        f"{n_envs} envs, mode={'env-goal' if args.use_env_goal else 'CLI-goal'}"
    )

    while n_episodes_done < args.num_episodes:
        with torch.inference_mode():
            # ---- camera ----
            wrist_cam.update(dt=env.unwrapped.step_dt)
            rgb = wrist_cam.data.output["rgb"]  # (N, H, W, 3) uint8 or float
            if rgb is None:
                env.step(torch.zeros((n_envs, 6), device=device))
                continue
            if rgb.shape[-1] == 4:
                rgb = rgb[..., :3]
            if rgb.dtype == torch.uint8:
                img = rgb.float() / 255.0
            else:
                img = rgb.float()
                if img.max() > 1.5:
                    img = img / 255.0
            img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW
            if img.shape[-2:] != (image_h, image_w):
                img = F.interpolate(
                    img, size=(image_h, image_w), mode="bilinear", align_corners=False
                )

            # ---- state vector ----
            joint_pos = robot.data.joint_pos[:, :]  # (N, 6)
            joint_vel = robot.data.joint_vel[:, :]  # (N, 6)

            # Resolve goal per-env.
            if args.use_env_goal:
                target_color_idx = env.unwrapped.target_color  # (N,) long
                bowl_xyz_w = bowl_floor.data.root_pos_w  # (N, 3)
                bowl_xyz_b, _ = subtract_frame_transforms(
                    robot.data.root_state_w[:, :3],
                    robot.data.root_state_w[:, 3:7],
                    bowl_xyz_w,
                )
            else:
                target_color_idx = torch.full(
                    (n_envs,), cli_target_color, device=device, dtype=torch.long
                )
                bowl_xyz_b = cli_bowl_xyz.unsqueeze(0).expand(n_envs, 3)
            target_color_oh = F.one_hot(target_color_idx, num_classes=2).float()

            # State = concat(joint_pos, joint_vel, target_color_oh, bowl_xyz_b).
            # If the trained policy uses a different layout, this is the line
            # to adapt.
            state = torch.cat(
                [joint_pos, joint_vel, target_color_oh, bowl_xyz_b], dim=-1
            )  # (N, 17 by default)
            if state_dim is not None and state.shape[-1] != state_dim:
                raise SystemExit(
                    f"[deploy] built state of dim {state.shape[-1]} but policy "
                    f"expects {state_dim}. Adjust the concat order/contents to "
                    f"match the training dataset's observation.state layout."
                )

            # ---- policy forward ----
            batch = {image_key: img, state_key: state}
            # ACT manages an internal action queue; select_action returns one
            # action at a time. Across N envs, we call once and the same
            # queue is shared — fine for batch-1 deploy. For batch>1, we'd
            # need a queue per env. For now, run one env at a time.
            actions = []
            for env_idx in range(n_envs):
                policy.reset() if env_idx == 0 else None  # share queue across envs
                env_batch = {k: v[env_idx : env_idx + 1] for k, v in batch.items()}
                a = policy.select_action(env_batch)  # (1, 6)
                actions.append(a)
            action = torch.cat(actions, dim=0)  # (N, 6)

            # ---- success criterion (mirror eval2_inference.py) ----
            BOWL_INNER_HALF = 0.06
            target_pos = torch.where(
                (target_color_idx == 0).unsqueeze(-1),
                block_red.data.root_pos_w,
                block_blue.data.root_pos_w,
            )
            xy_dist = torch.norm(target_pos[:, :2] - bowl_floor.data.root_pos_w[:, :2], dim=1)
            dz = target_pos[:, 2] - bowl_floor.data.root_pos_w[:, 2]
            success_now = (xy_dist < BOWL_INNER_HALF) & (dz > -0.01) & (dz < 0.10)
            episode_success_seen |= success_now

            # ---- step ----
            obs, _, dones, _, _ = env.step(action)

            # ---- tally terminations ----
            done_envs = dones.nonzero(as_tuple=False).flatten().tolist()
            for env_idx in done_envs:
                n_episodes_done += 1
                if bool(episode_success_seen[env_idx]):
                    n_successes += 1
                episode_success_seen[env_idx] = False
                if n_episodes_done % 5 == 0 or n_episodes_done >= args.num_episodes:
                    rate = n_successes / n_episodes_done * 100
                    print(
                        f"[deploy] {n_episodes_done:>4d}/{args.num_episodes} "
                        f"episodes | success {n_successes}/{n_episodes_done} = {rate:.1f}%"
                    )
                if n_episodes_done >= args.num_episodes:
                    break

    print("\n" + "=" * 60)
    print(
        f"[deploy] FINAL: {n_successes}/{n_episodes_done} = "
        f"{n_successes / max(n_episodes_done, 1) * 100:.2f}% success rate"
    )
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
        print("\n[deploy] !!! UNCAUGHT EXCEPTION !!!", flush=True)
        traceback.print_exc()
        sys.exit(1)
    finally:
        simulation_app.close()
