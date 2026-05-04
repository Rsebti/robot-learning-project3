"""One-off script that generates a training dataset for the perception CNN.

Loads the Eval 2 v2 env (wrist camera enabled), holds the robot in the
'scan' pose used in view.py, and at every reset captures:
- the wrist-camera RGB image (resized to 84x84, uint8 in HWC),
- the (x, y, z) world->robot-frame coordinates of the red and blue
  block centers, computed by transforming PhysX world positions by the
  inverse of the robot base pose.

Outputs ``data.pt`` next to this file with two tensors:
    images   uint8   (N, 84, 84, 3)
    targets  float32 (N, 6)  meters, robot frame:
                              [x_red, y_red, z_red, x_blue, y_blue, z_blue]

Usage:
    uv run python -m sim.eval2.perception.capture_dataset \\
        --num_samples 5000 \\
        --output sim/eval2/perception/data.pt \\
        --enable_cameras

Notes on Approach (B)
- Targets are in the robot's base frame, in meters. The CNN learns the
  geometry directly from the image (block apparent size + position).
- No table-plane calibration is needed at deploy time: same image -> xyz
  out of the CNN -> goes to the policy.
- Reset events already randomize block positions and bowl position.
  Robot stays in fixed scan pose so the camera viewpoint is constant —
  fine for v2 since the deploy will always start from a similar pose.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Capture perception training data.")
parser.add_argument("--task", type=str, default="Eval2-PickInClutter-Play-v2")
parser.add_argument("--num_samples", type=int, default=5000)
parser.add_argument(
    "--output",
    type=str,
    default=str(Path(__file__).parent / "data.pt"),
    help="Path to the output .pt file (PyTorch tensor dict).",
)
parser.add_argument("--image_size", type=int, default=84, help="Output image side length (square).")
parser.add_argument(
    "--save_preview",
    action="store_true",
    help="Save the first 10 captures as PNG files alongside the .pt file, for visual inspection.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Required for Isaac Lab to actually render camera images.
args.enable_cameras = True
args.headless = True  # we don't need a window for data capture

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


# Same scan pose as scripts/view.py: gripper hovering above the workspace
# pointing down. Defined here too so capture is self-contained.
SCAN_POSE_JOINTS = {
    "shoulder_pan":   0.0,
    "shoulder_lift":  1.0,
    "elbow_flex":    -1.0,
    "wrist_flex":     1.57,
    "wrist_roll":     0.0,
    "gripper":        0.0,
}


def main():
    import gymnasium as gym
    import sim.eval2  # noqa: F401  (registers gym envs)
    import torch.nn.functional as F
    from isaaclab.utils.math import subtract_frame_transforms
    from isaaclab_tasks.utils import parse_env_cfg

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use only one parallel env — perception data capture is sequential.
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
    env = gym.make(args.task, cfg=env_cfg)

    obs, _ = env.reset()

    # Apply the scan pose once so the wrist cam frames the workspace.
    robot = env.unwrapped.scene["robot"]
    joint_names = robot.data.joint_names
    scan_pose = torch.zeros(1, len(joint_names), device="cuda:0")
    for i, name in enumerate(joint_names):
        if name in SCAN_POSE_JOINTS:
            scan_pose[:, i] = SCAN_POSE_JOINTS[name]
    zero_vel = torch.zeros_like(scan_pose)
    robot.write_joint_state_to_sim(scan_pose, zero_vel)

    wrist_cam = env.unwrapped.scene["wrist_cam"]
    block_red = env.unwrapped.scene["block_red"]
    block_blue = env.unwrapped.scene["block_blue"]

    images = torch.zeros(args.num_samples, args.image_size, args.image_size, 3, dtype=torch.uint8)
    # 6 outputs per sample: x_red, y_red, z_red, x_blue, y_blue, z_blue (meters, robot frame)
    targets = torch.zeros(args.num_samples, 6, dtype=torch.float32)

    sim = env.unwrapped.sim

    print(f"Capturing {args.num_samples} samples -> {output_path}")
    saved = 0
    while saved < args.num_samples:
        # Reset randomizes block_red and block_blue positions (cluster shift)
        # and resamples the bowl position. We keep the robot in the scan pose.
        obs, _ = env.reset()

        # Two things to handle:
        # (a) the PD controllers try to drive joints back toward home pose
        #     after our teleport, so we re-write the joint state every tick.
        # (b) the wrist camera in v2 has update_period=0.1s, while each
        #     sim.step advances 0.02s sim time (sim.dt=0.01 × decimation=2).
        #     We need at least ~5 ticks to give the camera a chance to
        #     refresh its output buffer, otherwise we read stale (black)
        #     pixels from the previous reset's render.
        for _ in range(6):
            robot.write_joint_state_to_sim(scan_pose, zero_vel)
            robot.set_joint_position_target(scan_pose)
            robot.write_data_to_sim()
            sim.step(render=True)
        # Final lock + force the camera sensor to refresh its buffer
        # regardless of update_period timing (large dt > update_period).
        robot.write_joint_state_to_sim(scan_pose, zero_vel)
        robot.write_data_to_sim()
        wrist_cam.update(dt=1.0)

        # --- read camera image ---
        rgb = wrist_cam.data.output["rgb"]  # (1, H, W, 3 or 4) uint8
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        rgb_chw = rgb.permute(0, 3, 1, 2).float() / 255.0
        rgb_resized = F.interpolate(
            rgb_chw,
            size=(args.image_size, args.image_size),
            mode="bilinear",
            align_corners=False,
        )
        rgb_resized_hwc = (rgb_resized.clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1)
        images[saved] = rgb_resized_hwc[0].cpu()

        # --- read block world positions ---
        red_w = block_red.data.root_pos_w[0:1]    # (1, 3)
        blue_w = block_blue.data.root_pos_w[0:1]  # (1, 3)

        # --- transform world -> robot base frame ---
        # subtract_frame_transforms takes (parent_pos, parent_quat, point_pos)
        # and returns the point expressed in the parent frame. The robot
        # articulation's root_state_w gives us the base pose in world.
        base_pos_w = robot.data.root_state_w[0:1, :3]    # (1, 3)
        base_quat_w = robot.data.root_state_w[0:1, 3:7]  # (1, 4)
        red_b, _ = subtract_frame_transforms(base_pos_w, base_quat_w, red_w)
        blue_b, _ = subtract_frame_transforms(base_pos_w, base_quat_w, blue_w)

        targets[saved, 0:3] = red_b[0].cpu()
        targets[saved, 3:6] = blue_b[0].cpu()

        # Optional: save the first 10 frames as PNG for visual inspection.
        if args.save_preview and saved < 10:
            try:
                from PIL import Image as _Image
                preview_dir = output_path.parent / "preview"
                preview_dir.mkdir(parents=True, exist_ok=True)
                img_np = images[saved].numpy()  # (H, W, 3) uint8
                _Image.fromarray(img_np).save(preview_dir / f"preview_{saved:02d}.png")
                # Also write a small txt with the targets (cm) for cross-check.
                t = targets[saved].tolist()
                with (preview_dir / f"preview_{saved:02d}.txt").open("w") as f:
                    f.write(
                        f"red  xyz (cm): {t[0]*100:.2f}, {t[1]*100:.2f}, {t[2]*100:.2f}\n"
                        f"blue xyz (cm): {t[3]*100:.2f}, {t[4]*100:.2f}, {t[5]*100:.2f}\n"
                    )
            except ImportError:
                # PIL not available; skip preview silently.
                pass

        saved += 1
        if saved % 100 == 0:
            print(f"  ... {saved} / {args.num_samples}")

    # Save tensors to disk.
    torch.save({"images": images, "targets": targets}, output_path)
    print(f"\nSaved {saved} samples to {output_path}")
    print(f"  images.shape  = {tuple(images.shape)}  ({images.dtype})")
    print(f"  targets.shape = {tuple(targets.shape)} ({targets.dtype})  meters, robot frame")
    print(f"  red xyz   range x[{targets[:, 0].min():.3f},{targets[:, 0].max():.3f}]  "
          f"y[{targets[:, 1].min():.3f},{targets[:, 1].max():.3f}]  "
          f"z[{targets[:, 2].min():.3f},{targets[:, 2].max():.3f}]")
    print(f"  blue xyz  range x[{targets[:, 3].min():.3f},{targets[:, 3].max():.3f}]  "
          f"y[{targets[:, 4].min():.3f},{targets[:, 4].max():.3f}]  "
          f"z[{targets[:, 5].min():.3f},{targets[:, 5].max():.3f}]")
    env.close()


if __name__ == "__main__":
    import traceback

    try:
        main()
    except Exception:  # noqa: BLE001
        print("\n=== capture_dataset.py crashed — traceback follows ===")
        traceback.print_exc()
        print("=== end traceback ===\n")
    finally:
        simulation_app.close()
