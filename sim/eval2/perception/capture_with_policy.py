"""Generate a perception training dataset by running a trained v1 policy
in the v2 env (which has the wrist camera).

Why this is better than capture_dataset.py
    capture_dataset.py held the robot in a single fixed scan pose and only
    captured snapshots from that one viewpoint. The CNN trained on that
    data only saw the workspace from one angle, and at deploy time would
    fail when the robot moved into a different pose.

    This script instead drives the robot with a working RL policy (the
    v1.2 checkpoint that succeeds 7.46% of the time) and captures images
    + ground-truth block positions DURING the rollouts. The camera
    naturally sees the workspace from a wide range of viewpoints — the
    same range it'll see at deploy time. The CNN trained on this data
    is genuinely viewpoint-robust.

Output
    A .pt file with:
        images   uint8   (N, 84, 84, 3)
        targets  float32 (N, 6)   meters, robot frame:
                                  [x_red, y_red, z_red, x_blue, y_blue, z_blue]

Usage
    uv run python -m sim.eval2.perception.capture_with_policy \\
        --checkpoint logs/rsl_rl/eval2_pick_in_bowl/2026-04-30_15-53-41/model_999.pt \\
        --num_samples 5000 \\
        --enable_cameras

Notes
- We use the v2 env (Eval2-PickInClutter-v2) so the wrist camera is in
  the scene. The trained v1 checkpoint is compatible because v2 has the
  same observation space as v1 (we only added a sensor, no obs change).
- Single-env capture by default (Blackwell + Camera sensor is slow when
  parallelized; 1 env is robust). Override via --num_envs if you want.
- The env auto-resets on episode termination, so the dataset spans many
  rollouts naturally.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Policy-driven perception data capture.")
parser.add_argument("--task", type=str, default="Eval2-PickInClutter-v2")
parser.add_argument(
    "--checkpoint",
    type=str,
    default=(
        # v1.2 baseline (success 7.46% at iter 999) — full path on the dev box.
        "C:/Users/user/Desktop/MA2/isaac/isaac_so_arm101/logs/rsl_rl/eval2_pick_in_bowl/2026-04-30_15-53-41/model_999.pt"
    ),
)
parser.add_argument("--num_samples", type=int, default=5000)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--image_size", type=int, default=84, help="Output image side length (square).")
parser.add_argument(
    "--output",
    type=str,
    default=str(Path(__file__).parent / "data.pt"),
    help="Path to the output .pt file.",
)
parser.add_argument(
    "--save_preview",
    action="store_true",
    help="Save the first 10 captures as PNG files alongside the .pt for visual inspection.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Camera rendering is required to read RGB from the wrist sensor.
args.enable_cameras = True
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def main():
    import gymnasium as gym
    import torch
    import torch.nn.functional as F
    from rsl_rl.runners import OnPolicyRunner

    import sim.eval2  # noqa: F401  (registers gym envs)
    from sim.eval2.agents.rsl_rl_ppo_cfg import Eval2PPORunnerCfg
    from isaaclab.utils.math import subtract_frame_transforms
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_dir = output_path.parent / "preview"
    if args.save_preview:
        preview_dir.mkdir(parents=True, exist_ok=True)

    # Build env (v2 with camera).
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    env_wrapped = RslRlVecEnvWrapper(env)

    # Load the trained policy.
    agent_cfg = Eval2PPORunnerCfg()
    runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
    print(f"[capture] Loading checkpoint: {args.checkpoint}")
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device="cuda:0")

    # Sensors and assets we'll read from each step.
    wrist_cam = env.unwrapped.scene["wrist_cam"]
    block_red = env.unwrapped.scene["block_red"]
    block_blue = env.unwrapped.scene["block_blue"]
    robot = env.unwrapped.scene["robot"]

    # Output buffers.
    images = torch.zeros(args.num_samples, args.image_size, args.image_size, 3, dtype=torch.uint8)
    targets = torch.zeros(args.num_samples, 6, dtype=torch.float32)

    # Roll out the policy and capture each step.
    obs = env_wrapped.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]
    saved = 0
    print(f"[capture] Rolling out policy, target {args.num_samples} samples...")
    while saved < args.num_samples:
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = env_wrapped.step(actions)

        # The wrist camera has update_period=0.1s in the env config but each
        # env.step only advances ~0.02s of sim time, so by default the
        # camera buffer would only refresh once every 5 steps — meaning
        # 4 of every 5 captured frames would be identical. Force-refresh
        # the camera output here by passing a dt > update_period.
        wrist_cam.update(dt=1.0)

        # --- camera image (resized to image_size square) ---
        rgb = wrist_cam.data.output["rgb"]  # (num_envs, H, W, 3 or 4)
        if rgb is None:
            # Camera may not have produced a frame yet on the first few ticks.
            continue
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        rgb_chw = rgb.permute(0, 3, 1, 2).float() / 255.0
        rgb_resized = F.interpolate(
            rgb_chw,
            size=(args.image_size, args.image_size),
            mode="bilinear",
            align_corners=False,
        )
        rgb_uint8_hwc = (rgb_resized.clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1)

        # --- ground truth block positions (in robot base frame) ---
        red_w = block_red.data.root_pos_w[: args.num_envs]   # (num_envs, 3)
        blue_w = block_blue.data.root_pos_w[: args.num_envs]
        base_pos_w = robot.data.root_state_w[: args.num_envs, :3]
        base_quat_w = robot.data.root_state_w[: args.num_envs, 3:7]
        red_b, _ = subtract_frame_transforms(base_pos_w, base_quat_w, red_w)
        blue_b, _ = subtract_frame_transforms(base_pos_w, base_quat_w, blue_w)

        # Store one sample per env, until we hit num_samples.
        for env_idx in range(rgb_uint8_hwc.shape[0]):
            if saved >= args.num_samples:
                break
            images[saved] = rgb_uint8_hwc[env_idx].cpu()
            targets[saved, 0:3] = red_b[env_idx].cpu()
            targets[saved, 3:6] = blue_b[env_idx].cpu()

            if args.save_preview and saved < 10:
                try:
                    from PIL import Image as _Image
                    img_np = images[saved].numpy()
                    _Image.fromarray(img_np).save(preview_dir / f"preview_{saved:02d}.png")
                    t = targets[saved].tolist()
                    with (preview_dir / f"preview_{saved:02d}.txt").open("w") as f:
                        f.write(
                            f"red  xyz (cm): {t[0]*100:.2f}, {t[1]*100:.2f}, {t[2]*100:.2f}\n"
                            f"blue xyz (cm): {t[3]*100:.2f}, {t[4]*100:.2f}, {t[5]*100:.2f}\n"
                        )
                except ImportError:
                    pass

            saved += 1
            if saved % 100 == 0:
                print(f"  ... {saved} / {args.num_samples}")

    # Save tensors.
    torch.save({"images": images, "targets": targets}, output_path)
    print(f"\n[capture] Saved {saved} samples -> {output_path}")
    print(f"  images.shape  = {tuple(images.shape)}  ({images.dtype})")
    print(f"  targets.shape = {tuple(targets.shape)}  ({targets.dtype})  meters, robot frame")
    print(
        f"  red xyz   range x[{targets[:, 0].min():.3f},{targets[:, 0].max():.3f}]  "
        f"y[{targets[:, 1].min():.3f},{targets[:, 1].max():.3f}]  "
        f"z[{targets[:, 2].min():.3f},{targets[:, 2].max():.3f}]"
    )
    print(
        f"  blue xyz  range x[{targets[:, 3].min():.3f},{targets[:, 3].max():.3f}]  "
        f"y[{targets[:, 4].min():.3f},{targets[:, 4].max():.3f}]  "
        f"z[{targets[:, 5].min():.3f},{targets[:, 5].max():.3f}]"
    )
    env.close()


if __name__ == "__main__":
    import traceback

    try:
        main()
    except Exception:  # noqa: BLE001
        print("\n=== capture_with_policy.py crashed — traceback follows ===")
        traceback.print_exc()
        print("=== end traceback ===\n")
    finally:
        simulation_app.close()
