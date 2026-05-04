"""Eval 2 — closed-loop deploy of the trained PPO policy with the perception CNN.

This is the script that wires together the three trained components:

    [wrist camera image]
            |
            v
    [perception CNN]  ── (red_xyz, blue_xyz)
                                  \\
    [joint state from sim/robot] ──┴── 29-D obs vector ── [policy] ── action ──> sim/robot
    [bowl_xyz from CLI args     ]
    [target_color from CLI args ]
    [last_action buffer         ]

For now it runs in **sim mode**: Isaac Sim is both the input source (joint
state, camera) and the output sink (motor targets). This validates the full
pipeline before swapping the I/O layer for the real SO-101.

The policy was trained with **ground-truth** block positions from PhysX.
Here we substitute the perception output for those positions, simulating
what will happen at deploy time on the real robot. We log:

    - success rate over N episodes
    - mean absolute error of the perception (vs PhysX GT) during rollout

so we can quantify the perception-induced degradation.

Usage (from the isaac_so_arm101 venv with our package installed editable):

    uv run python deploy/eval2_inference.py \\
        --task Eval2-PickInClutter-Play-v2 \\
        --policy_checkpoint C:/path/to/eval2_pick_in_bowl/<run>/model_999.pt \\
        --num_episodes 20

    # Use ground-truth block positions instead of perception (baseline):
    uv run python deploy/eval2_inference.py ... --use_ground_truth

    # Show the simulation window:
    uv run python deploy/eval2_inference.py ... --no_headless

Real-robot mode is NOT yet implemented — the I/O layer abstraction is in
place but only the SimSource is provided. Adding a RealSource that reads
from lerobot + OpenCV is straightforward and will go here later.
"""

# =============================================================================
# CRITICAL: AppLauncher must be invoked BEFORE any torch / isaaclab import.
# Argparse goes here for the same reason.
# =============================================================================
from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Eval 2 closed-loop deploy (sim mode).")
parser.add_argument(
    "--task",
    type=str,
    default="Eval2-PickInClutter-Play-v2",
    help="Gym task ID. Must be a v2 variant (with wrist_cam in the scene).",
)
parser.add_argument(
    "--policy_checkpoint",
    type=Path,
    required=True,
    help="Path to the trained rsl_rl checkpoint (model_*.pt).",
)
parser.add_argument(
    "--perception_checkpoint",
    type=Path,
    default=Path(__file__).parent.parent / "sim/eval2/perception/checkpoint.pt",
    help="Path to the trained perception CNN checkpoint.",
)
parser.add_argument("--num_envs", type=int, default=8, help="Parallel envs in sim.")
parser.add_argument(
    "--num_episodes",
    type=int,
    default=20,
    help="Total number of episodes to roll out (across all envs).",
)
parser.add_argument(
    "--use_ground_truth",
    action="store_true",
    help="Skip the perception CNN and feed PhysX ground-truth block xyz to the "
    "policy (= matches training conditions). Lets you measure the perception-"
    "induced success-rate gap by comparing two runs.",
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--no_headless",
    action="store_true",
    help="Open the Isaac Sim window for visual inspection. Slower.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Force --enable_cameras on (the v2 task spawns a CameraCfg; without this flag
# Isaac Lab silently disables camera rendering and our perception gets garbage).
args.enable_cameras = True
args.headless = not args.no_headless

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


# =============================================================================
# Heavy imports: only after AppLauncher.
# =============================================================================
def main():
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    # Side-effect: registers our gym envs.
    import sim.eval2  # noqa: F401
    from sim.eval2.agents.rsl_rl_ppo_cfg import Eval2PPORunnerCfg
    from sim.eval2.perception.inference import PerceptionPipeline

    # ---- Build env (Play variant: 50 envs by default, no obs noise) ----
    env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
    env_cfg.seed = args.seed

    # NOTE: we deliberately do NOT lower wrist_cam.update_period here. In some
    # Isaac Lab versions, update_period=0 means "never auto-update" (not "every
    # step"), which is the opposite of what we want. Instead we force a refresh
    # explicitly inside the loop with cam.update(dt) — same trick as
    # sim/eval2/perception/capture_with_policy.py.

    print(f"[deploy] gym.make({args.task!r}, num_envs={args.num_envs})", flush=True)
    env = gym.make(args.task, cfg=env_cfg)
    print("[deploy] wrapping with RslRlVecEnvWrapper ...", flush=True)
    env = RslRlVecEnvWrapper(env, clip_actions=None)

    # ---- Load the policy ----
    agent_cfg = Eval2PPORunnerCfg()
    print("[deploy] building OnPolicyRunner ...", flush=True)
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=str(env.unwrapped.device)
    )
    print(f"[deploy] loading policy checkpoint: {args.policy_checkpoint}", flush=True)
    runner.load(str(args.policy_checkpoint))

    # IMPORTANT: this policy was trained with a HIGH action noise std (~4.8 at
    # iter 999) — it never converged to a deterministic-mean solution. Calling
    # act_inference() returns just the mean, which alone produces saturated
    # actions (~+/-15 in unit-less, well past joint limits after the *0.5
    # action scale). To reproduce the conditions the policy was trained under,
    # we sample stochastic actions: actor mean + N(0, std). This matches what
    # rsl_rl does inside OnPolicyRunner.learn() for collection.
    policy_nn = runner.alg.policy
    print(f"[deploy] policy std mean = {policy_nn.std.mean().item():.3f} "
          f"(>> 0.5 => not converged, will sample with noise)", flush=True)

    def policy(obs_in):
        # rsl_rl ActorCritic.act() updates the distribution then samples.
        return policy_nn.act(obs_in)

    # ---- Load the perception (skip if --use_ground_truth) ----
    perception = None
    if not args.use_ground_truth:
        print(f"[deploy] loading perception checkpoint: {args.perception_checkpoint}", flush=True)
        perception = PerceptionPipeline(
            checkpoint=str(args.perception_checkpoint),
            device=str(env.unwrapped.device),
        )
    print("[deploy] perception ready", flush=True)

    # ---- Verify the obs layout matches what we expect ----
    # The policy was trained with this 29-D layout (see PolicyCfg in
    # pick_in_clutter_env_cfg.py). If you change the obs order there, update
    # these slices accordingly.
    OBS_BLOCK_RED_SLICE = slice(12, 15)
    OBS_BLOCK_BLUE_SLICE = slice(15, 18)
    expected_obs_dim = 29
    print("[deploy] calling env.get_observations() ...", flush=True)
    obs = env.get_observations()
    print(f"[deploy] get_observations returned type={type(obs).__name__}", flush=True)

    def _extract_policy_tensor(o):
        """rsl_rl >= 2.3 returns a TensorDict; older returns a (tensor, extras) tuple
        or a bare tensor. Normalize to the (num_envs, 29) policy obs tensor.
        """
        if isinstance(o, tuple):
            o = o[0]
        # TensorDict has .keys() and supports ["policy"] indexing.
        if hasattr(o, "keys") and not hasattr(o, "shape_dim"):
            try:
                keys = list(o.keys())
                if "policy" in keys:
                    return o["policy"], "tensordict"
            except Exception:
                pass
        # Plain tensor.
        return o, "tensor"

    obs_tensor, obs_kind = _extract_policy_tensor(obs)
    print(
        f"[deploy] obs kind={obs_kind} | policy tensor shape={tuple(obs_tensor.shape)} "
        f"dtype={obs_tensor.dtype} device={obs_tensor.device}",
        flush=True,
    )
    # Print env 0's full 29-D obs to check it has sensible values.
    obs0 = obs_tensor[0].cpu().tolist()
    labels = (
        ["jpos[0..5]"] * 6 + ["jvel[0..5]"] * 6 + ["red_xyz"] * 3 + ["blue_xyz"] * 3
        + ["bowl_xyz"] * 3 + ["target_color"] * 2 + ["last_act"] * 6
    )
    print("[deploy] env0 initial obs (29-D):", flush=True)
    for i, (lab, v) in enumerate(zip(labels, obs0)):
        print(f"           [{i:2d}] {lab:<14} = {v:+.4f}", flush=True)
    if obs_tensor.shape[-1] != expected_obs_dim:
        raise RuntimeError(
            f"Obs dim mismatch: expected {expected_obs_dim}, got {obs_tensor.shape[-1]}. "
            "The policy was trained with the v1 observation layout."
        )

    # ---- Stats accumulators ----
    n_envs = env.unwrapped.num_envs
    n_episodes_done = 0
    n_successes = 0
    perception_err_sum_cm = torch.zeros(6, device=env.unwrapped.device)
    perception_err_count = 0
    step_idx = 0  # used to log per-step diagnostics for the first few steps

    # Per-env "did the target ever land in the bowl during this episode?" flag.
    # Auto-reset on env.step() wipes the success state before we can read it,
    # so we OR-accumulate the per-step success criterion across the episode
    # and check the flag when the env terminates.
    episode_success_seen = torch.zeros(n_envs, dtype=torch.bool, device=env.unwrapped.device)

    # ---- Closed-loop rollout ----
    print(f"[deploy] rollout starting: {args.num_envs} parallel envs, target {args.num_episodes} episodes total.")
    print(f"[deploy] mode = {'GROUND-TRUTH (baseline)' if args.use_ground_truth else 'PERCEPTION CNN'}")

    while n_episodes_done < args.num_episodes:
        with torch.inference_mode():
            # ---- Read GT block positions for logging (always, even in perception mode) ----
            scene = env.unwrapped.scene
            robot = scene["robot"]
            red = scene["block_red"]
            blue = scene["block_blue"]
            from isaaclab.utils.math import subtract_frame_transforms

            gt_red_b, _ = subtract_frame_transforms(
                robot.data.root_state_w[:, :3],
                robot.data.root_state_w[:, 3:7],
                red.data.root_pos_w[:, :3],
            )
            gt_blue_b, _ = subtract_frame_transforms(
                robot.data.root_state_w[:, :3],
                robot.data.root_state_w[:, 3:7],
                blue.data.root_pos_w[:, :3],
            )

            # ---- Build the obs the policy will see ----
            # The env returns a TensorDict (rsl_rl >= 2.3). The underlying
            # 29-D tensor lives at obs["policy"]. We modify it in place so
            # the policy call below sees the perception-substituted version.
            obs_tensor, obs_kind = _extract_policy_tensor(obs)

            if perception is not None:
                # Read camera output: (num_envs, H, W, 3) uint8 or float
                cam = scene["wrist_cam"]
                rgb = cam.data.output["rgb"]  # (N, H, W, 3)
                # Normalize to (N, 3, H, W) float in [0, 1] for the CNN.
                if rgb.dtype != torch.uint8:
                    # Some Isaac Sim versions return float in [0, 1] already.
                    rgb_proc = rgb.float()
                    if rgb_proc.max() > 1.5:
                        rgb_proc = rgb_proc / 255.0
                else:
                    rgb_proc = rgb.float() / 255.0
                rgb_proc = rgb_proc.permute(0, 3, 1, 2)  # NHWC -> NCHW

                # Resize each env's image to the perception input size if needed.
                if rgb_proc.shape[-1] != perception.image_size or rgb_proc.shape[-2] != perception.image_size:
                    rgb_proc = torch.nn.functional.interpolate(
                        rgb_proc,
                        size=(perception.image_size, perception.image_size),
                        mode="bilinear",
                        align_corners=False,
                    )

                # IMPORTANT: round-trip through uint8 to match the training data
                # exactly. The capture script (capture_with_policy.py) quantizes
                # to uint8 before saving to disk, and the dataset loader reads
                # them back as uint8 -> float / 255. Skipping this round-trip
                # in deploy feeds the CNN slightly different floats than it saw
                # during training (the bilinear resize output isn't snapped to
                # 1/255 levels), which a small CNN without BatchNorm can react
                # to disproportionately.
                rgb_proc = (rgb_proc.clamp(0.0, 1.0) * 255.0).to(torch.uint8).float() / 255.0

                # Batched perception forward.
                pred = perception.model(rgb_proc.to(perception.device))  # (N, 6)
                pred_red = pred[:, 0:3]
                pred_blue = pred[:, 3:6]

                # Substitute into the obs vector (in place — the policy reads
                # from the same tensor on the next line).
                obs_tensor[:, OBS_BLOCK_RED_SLICE] = pred_red
                obs_tensor[:, OBS_BLOCK_BLUE_SLICE] = pred_blue

                # Track perception error vs ground-truth (in cm).
                err_cm = torch.cat(
                    [(pred_red - gt_red_b).abs(), (pred_blue - gt_blue_b).abs()],
                    dim=1,
                ) * 100.0  # (N, 6)
                perception_err_sum_cm += err_cm.sum(dim=0)
                perception_err_count += err_cm.shape[0]

                # First-20-step diagnostics: print mean err across envs to see
                # if the perception is bad from the very first frame (likely
                # cause: image format / resize / scene mismatch with training)
                # or only drifts later (likely cause: closed-loop feedback OOD).
                if step_idx < 20:
                    mean_err = err_cm.mean(dim=0).cpu().tolist()
                    avg = sum(mean_err) / len(mean_err)
                    print(
                        f"[deploy] step {step_idx:3d} | "
                        f"err cm  red(x={mean_err[0]:5.2f},y={mean_err[1]:5.2f},z={mean_err[2]:5.2f})  "
                        f"blue(x={mean_err[3]:5.2f},y={mean_err[4]:5.2f},z={mean_err[5]:5.2f})  "
                        f"avg={avg:5.2f}",
                        flush=True,
                    )
                    # On the first step also dump the GT and pred so we can
                    # eyeball whether the predictions are in the right ballpark.
                    if step_idx == 0:
                        print(
                            f"[deploy] step 0 | rgb tensor shape={tuple(rgb.shape)} "
                            f"dtype={rgb.dtype} min={rgb.min().item():.2f} max={rgb.max().item():.2f}",
                            flush=True,
                        )
                        print(
                            f"[deploy] step 0 env 0 | GT red  {(gt_red_b[0] * 100).tolist()} cm | "
                            f"pred {(pred_red[0] * 100).tolist()} cm",
                            flush=True,
                        )
                        print(
                            f"[deploy] step 0 env 0 | GT blue {(gt_blue_b[0] * 100).tolist()} cm | "
                            f"pred {(pred_blue[0] * 100).tolist()} cm",
                            flush=True,
                        )
                        # Save the raw 240x320 frame and the resized 84x84
                        # frame seen by the CNN, with the GT + pred written on
                        # them. We'll diff them against the dataset previews.
                        try:
                            from PIL import Image, ImageDraw
                            preview_dir = Path(args.perception_checkpoint).parent / "deploy_preview"
                            preview_dir.mkdir(exist_ok=True)
                            for env_idx in range(min(rgb.shape[0], 4)):
                                # Raw 240x320 frame straight from the camera.
                                raw_np = rgb[env_idx].cpu().numpy()
                                Image.fromarray(raw_np, mode="RGB").save(
                                    preview_dir / f"step0_env{env_idx}_raw_{tuple(raw_np.shape[:2])}.png"
                                )
                                # The 84x84 actually fed to the CNN.
                                cnn_input = (rgb_proc[env_idx].clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                                pil = Image.fromarray(cnn_input, mode="RGB").resize((84 * 4, 84 * 4), Image.NEAREST)
                                d = ImageDraw.Draw(pil)
                                gtr = gt_red_b[env_idx].cpu().tolist()
                                prr = pred_red[env_idx].cpu().tolist()
                                gtb = gt_blue_b[env_idx].cpu().tolist()
                                prb = pred_blue[env_idx].cpu().tolist()
                                d.text((4, 4), f"GT  red ({gtr[0]*100:.1f},{gtr[1]*100:.1f},{gtr[2]*100:.1f})", fill=(255, 255, 0))
                                d.text((4, 18), f"Pred red ({prr[0]*100:.1f},{prr[1]*100:.1f},{prr[2]*100:.1f})", fill=(255, 80, 80))
                                d.text((4, 32), f"GT  blu ({gtb[0]*100:.1f},{gtb[1]*100:.1f},{gtb[2]*100:.1f})", fill=(255, 255, 0))
                                d.text((4, 46), f"Pred blu ({prb[0]*100:.1f},{prb[1]*100:.1f},{prb[2]*100:.1f})", fill=(80, 80, 255))
                                pil.save(preview_dir / f"step0_env{env_idx}_cnn_input_84x84.png")
                            print(f"[deploy] saved step-0 frames to {preview_dir}", flush=True)
                        except Exception as e:
                            print(f"[deploy] preview save failed: {e}", flush=True)

            # ---- Compute the success criterion at the CURRENT state ----
            BOWL_INNER_HALF = 0.06
            target_idx = env.unwrapped.target_color
            is_red = (target_idx == 0).unsqueeze(-1)
            target_pos = torch.where(is_red, red.data.root_pos_w, blue.data.root_pos_w)
            bowl_pos_w = scene["bowl_floor"].data.root_pos_w
            xy_dist = torch.norm(target_pos[:, :2] - bowl_pos_w[:, :2], dim=1)
            dz = target_pos[:, 2] - bowl_pos_w[:, 2]
            success_now = (xy_dist < BOWL_INNER_HALF) & (dz > -0.01) & (dz < 0.10)
            # OR-accumulate over the episode: a success at ANY step counts. This
            # is necessary because env.step() auto-resets done envs, wiping the
            # success state before we can read it post-step.
            episode_success_seen |= success_now

            # ---- Step the env ----
            # Pass the (possibly modified) TensorDict / tensor straight to the
            # policy — it knows the format that came from get_observations().
            action = policy(obs)
            if step_idx < 3:
                print(
                    f"[deploy] step {step_idx} | action shape={tuple(action.shape)} "
                    f"min={action.min().item():.3f} max={action.max().item():.3f} "
                    f"mean={action.mean().item():.3f} | env0={action[0].cpu().tolist()}",
                    flush=True,
                )
            obs, _, dones, _ = env.step(action)
            if step_idx < 3:
                n_done = int(dones.sum().item())
                print(f"[deploy] step {step_idx} | dones={n_done}/{n_envs}", flush=True)
            # Always increment step_idx so diagnostics fire just for the first
            # few steps (was previously only inside the perception block).
            step_idx += 1

            # Force the wrist camera to render and refresh its data buffer NOW
            # so the next iteration's perception read sees the post-step scene.
            # Without this, cam.data.output["rgb"] holds the previous frame
            # because the camera's auto-update period is much larger than the
            # env step (0.1 s vs 0.02 s by default). This was the same bug
            # that produced "5 identical frames" in capture_with_policy.py.
            if perception is not None:
                scene["wrist_cam"].update(dt=env.unwrapped.step_dt)

            # ---- Tally episode terminations ----
            done_envs = dones.nonzero(as_tuple=False).flatten().tolist()
            for env_idx in done_envs:
                n_episodes_done += 1
                if bool(episode_success_seen[env_idx]):
                    n_successes += 1
                # Reset the per-env success flag for the next episode.
                episode_success_seen[env_idx] = False
                if n_episodes_done % 5 == 0 or n_episodes_done >= args.num_episodes:
                    rate = n_successes / n_episodes_done * 100
                    print(
                        f"[deploy] {n_episodes_done:>4d}/{args.num_episodes} episodes done | "
                        f"success {n_successes}/{n_episodes_done} = {rate:.1f}%",
                        flush=True,
                    )
                if n_episodes_done >= args.num_episodes:
                    break

    # ---- Final report ----
    print("\n" + "=" * 60)
    print(f"[deploy] FINAL: {n_successes}/{n_episodes_done} = "
          f"{n_successes / max(n_episodes_done, 1) * 100:.2f}% success rate")
    print(f"[deploy] mode  : {'GROUND-TRUTH (baseline)' if args.use_ground_truth else 'PERCEPTION CNN'}")

    if perception is not None and perception_err_count > 0:
        avg = (perception_err_sum_cm / perception_err_count).cpu().tolist()
        names = ["x_red", "y_red", "z_red", "x_blue", "y_blue", "z_blue"]
        print(f"\n[deploy] perception MAE during rollout (cm), N={perception_err_count} steps:")
        for n, v in zip(names, avg):
            print(f"           {n:<8} {v:5.2f} cm")
        print(f"           {'avg':<8} {sum(avg) / len(avg):5.2f} cm")

    env.close()


if __name__ == "__main__":
    import sys
    import traceback
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        # Make sure we see the traceback BEFORE Isaac Sim swallows it on close().
        sys.stdout.flush()
        sys.stderr.flush()
        print("\n[deploy] !!! UNCAUGHT EXCEPTION !!!", flush=True)
        traceback.print_exc()
        sys.stderr.flush()
    finally:
        simulation_app.close()
