"""
profile_loop.py - exhaustive per-stage timing of the SAC rollout loop.

For each iteration we time these fragments individually:

  motor_read       : bus.sync_read("Present_Position", all 6)
  camera_read      : agent.capture_sensor_data() (lerobot async_read of wrist)
  image_preproc    : center-crop + resize + permute + to(device)
  state_build      : np concat + to-tensor + to(device)
  encoder_fwd      : CNN forward
  actor_fwd        : Actor MLP forward (incl rgb_proj + state_proj)
  cpu_to_numpy     : action.cpu().numpy()
  action_clip      : action_scale + clip + DELTA_CAP integrate
  send_action      : bus write of 6 motor targets
  step_total       : wall clock t_start -> just before sleep

Plus:
  camera_real_fps  : measured frame-to-frame interval where the BYTES changed,
                     i.e. how often a genuinely new frame is being delivered

At end prints mean / std / p50 / p95 per fragment, the sum of fragments, and
the discrepancy vs step_total (= overhead I cannot attribute).

Usage:
    python -m toolset.control.profile_loop \\
        --port COM3 --camera_index 1 `
        --checkpoint C:/Users/hugod/ckpt_best_1.pt `
        --n_steps 100 --warmup 10
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

DEPLOY = Path(__file__).resolve().parents[2] / "deploy"
EVAL1V2 = DEPLOY / "eval1_v2"
for p in (str(DEPLOY), str(EVAL1V2)):
    if p not in sys.path:
        sys.path.insert(0, p)

from robot_calibration import make_so101_follower_config  # noqa: E402
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: E402
from lerobot.robots.utils import make_robot_from_config  # noqa: E402

import infer_sac_legacy as legacy  # noqa: E402


def _stats(samples):
    a = np.asarray(samples, dtype=float)
    return dict(
        mean=float(np.mean(a)),
        std=float(np.std(a)),
        p50=float(np.percentile(a, 50)),
        p95=float(np.percentile(a, 95)),
        min=float(np.min(a)),
        max=float(np.max(a)),
        n=int(len(a)),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="COM3")
    p.add_argument("--camera_index", type=int, default=1)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n_steps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--target_hz", type=int, default=30,
                   help="Used only to report the per-step budget; loop runs full speed.")
    p.add_argument("--goal_color", type=int, default=3)
    p.add_argument("--bowl_xyz", type=float, nargs=3, default=[0.16, 0.32, 0.0])
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[prof] device = {device}")

    # Load checkpoint and rebuild policy via the legacy code
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    n_state = ckpt["actor"]["proj.state_proj.0.weight"].shape[1]
    n_conv, image_size = legacy._detect_encoder_arch(ckpt["encoder"])
    legacy.IMAGE_SIZE = image_size
    print(f"[prof] policy n_state={n_state}  n_conv={n_conv}  image_size={image_size}")

    encoder = legacy.CNNEncoder(n_conv=n_conv).to(device).eval()
    actor = legacy.Actor(n_state=n_state).to(device).eval()
    encoder.load_state_dict(ckpt["encoder"])
    actor.load_state_dict(ckpt["actor"])

    cfg = make_so101_follower_config(
        args.port,
        cameras={"base_camera": OpenCVCameraConfig(
            index_or_path=args.camera_index, fps=30, width=640, height=480)},
        use_degrees=True,
    )
    robot = make_robot_from_config(cfg)
    print(f"[prof] connecting to {args.port} ...")
    robot.connect()
    agent = legacy.RealRobotAgent(robot)

    colour_conditioned = n_state in (18, 21)
    use_bowl_xyz = n_state == 21
    target_qpos = agent.get_qpos().cpu().numpy().flatten()

    # buckets
    buckets: dict[str, list[float]] = {
        k: [] for k in (
            "motor_read", "camera_read", "image_preproc", "state_build",
            "encoder_fwd", "actor_fwd", "cpu_to_numpy", "action_clip",
            "send_action", "step_total",
        )
    }
    new_frame_dts: list[float] = []
    prev_frame_hash = None
    prev_frame_t = None

    total = args.warmup + args.n_steps
    try:
        for step in range(total):
            t_step = time.perf_counter()

            t = time.perf_counter()
            qpos = agent.get_qpos().cpu().numpy().flatten()
            motor_read_dt = (time.perf_counter() - t) * 1000

            t = time.perf_counter()
            agent.capture_sensor_data()
            rgb = agent.get_sensor_data()["base_camera"]["rgb"]
            camera_read_dt = (time.perf_counter() - t) * 1000

            # frame-novelty check: did the byte content change vs last call?
            t_now = time.perf_counter()
            frame_np = (rgb[0].cpu().numpy() if torch.is_tensor(rgb)
                        else np.asarray(rgb[0]))
            # Cheap hash: sum of pixel values; for jitter detection only.
            h = int(frame_np.sum())
            if prev_frame_hash is not None and h != prev_frame_hash and prev_frame_t is not None:
                new_frame_dts.append((t_now - prev_frame_t) * 1000)
                prev_frame_t = t_now
            elif prev_frame_hash is None:
                prev_frame_t = t_now
            prev_frame_hash = h

            t = time.perf_counter()
            obs_rgb = legacy.preprocess_image(rgb).to(device)
            image_preproc_dt = (time.perf_counter() - t) * 1000

            t = time.perf_counter()
            obs_state = legacy.build_state(
                qpos, target_qpos,
                args.goal_color if colour_conditioned else None,
                bowl_xyz=args.bowl_xyz if use_bowl_xyz else None,
            ).to(device)
            state_build_dt = (time.perf_counter() - t) * 1000

            with torch.no_grad():
                t = time.perf_counter()
                z = encoder(obs_rgb)
                # ensure work is done before timing actor (mostly relevant for CUDA)
                if device == "cuda":
                    torch.cuda.synchronize()
                encoder_fwd_dt = (time.perf_counter() - t) * 1000

                t = time.perf_counter()
                a_tensor, _ = actor(z, obs_state)
                if device == "cuda":
                    torch.cuda.synchronize()
                actor_fwd_dt = (time.perf_counter() - t) * 1000

                t = time.perf_counter()
                raw_action = a_tensor[0].cpu().numpy()
                cpu_to_numpy_dt = (time.perf_counter() - t) * 1000

            t = time.perf_counter()
            action = np.clip(raw_action * 0.15, -1.0, 1.0)
            target_qpos = np.clip(
                target_qpos + action * legacy.DELTA_CAP,
                legacy.JOINT_LOWER, legacy.JOINT_UPPER,
            )
            action_clip_dt = (time.perf_counter() - t) * 1000

            t = time.perf_counter()
            agent.set_target_qpos(torch.from_numpy(target_qpos))
            send_action_dt = (time.perf_counter() - t) * 1000

            step_total_dt = (time.perf_counter() - t_step) * 1000

            if step >= args.warmup:
                buckets["motor_read"].append(motor_read_dt)
                buckets["camera_read"].append(camera_read_dt)
                buckets["image_preproc"].append(image_preproc_dt)
                buckets["state_build"].append(state_build_dt)
                buckets["encoder_fwd"].append(encoder_fwd_dt)
                buckets["actor_fwd"].append(actor_fwd_dt)
                buckets["cpu_to_numpy"].append(cpu_to_numpy_dt)
                buckets["action_clip"].append(action_clip_dt)
                buckets["send_action"].append(send_action_dt)
                buckets["step_total"].append(step_total_dt)
    finally:
        # park back to current pose (no big motion)
        robot.disconnect()
        print("[prof] disconnected.")

    print(f"\n=== Per-stage timing (ms), n={len(buckets['step_total'])} steps "
          f"(warmup {args.warmup} skipped) ===")
    print(f"{'stage':16s} {'mean':>8} {'std':>7} {'p50':>8} {'p95':>8} {'min':>7} {'max':>7}")
    for k in [
        "motor_read", "camera_read", "image_preproc", "state_build",
        "encoder_fwd", "actor_fwd", "cpu_to_numpy", "action_clip",
        "send_action", "step_total",
    ]:
        s = _stats(buckets[k])
        print(f"{k:16s} {s['mean']:8.3f} {s['std']:7.3f} {s['p50']:8.3f} "
              f"{s['p95']:8.3f} {s['min']:7.3f} {s['max']:7.3f}")

    pieces_mean = sum(
        _stats(buckets[k])["mean"]
        for k in ("motor_read", "camera_read", "image_preproc", "state_build",
                  "encoder_fwd", "actor_fwd", "cpu_to_numpy",
                  "action_clip", "send_action")
    )
    total_mean = _stats(buckets["step_total"])["mean"]
    unaccounted = total_mean - pieces_mean
    print(f"\nSum of fragment means   : {pieces_mean:7.3f} ms")
    print(f"Measured step_total mean: {total_mean:7.3f} ms")
    print(f"Unaccounted overhead    : {unaccounted:+7.3f} ms   (= step_total - sum)")
    print(f"=> implied control rate : {1000.0 / total_mean:6.2f} Hz "
          f"(target = {args.target_hz} Hz, budget = {1000.0 / args.target_hz:.2f} ms)")

    if new_frame_dts:
        s = _stats(new_frame_dts)
        print(f"\nCamera frame novelty (interval where pixel-hash changed):")
        print(f"  mean={s['mean']:.2f}  std={s['std']:.2f}  p50={s['p50']:.2f}  "
              f"p95={s['p95']:.2f}  min={s['min']:.2f}  max={s['max']:.2f}  n={s['n']}")
        print(f"=> camera delivering ~{1000.0 / s['mean']:.1f} new frames/s "
              f"(declared 30 fps; if << 30 => USB/driver bottleneck "
              f"OR camera throttled by exposure).")
    else:
        print(f"\n[prof] WARNING: never saw the camera frame bytes change. "
              f"camera may be stuck on a single buffered frame.")


if __name__ == "__main__":
    main()
