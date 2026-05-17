#!/usr/bin/env python3
"""Wrap `lerobot-record` so we can time the components of the inner loop
separately and pinpoint the deploy-time bottleneck.

Monkey-patches (before invoking lerobot-record.main):
  - SO101Follower.get_observation  -> robot side (servo read + camera read)
  - OpenCVCamera.async_read         -> isolate the camera grab
  - SmolVLAPolicy.select_action     -> policy step (queue pop or chunk forward)
  - SO101Follower.send_action       -> motor write
  - DataProcessorPipeline.__call__  -> pre/post-processing (tokenize, normalize)

Reads env vars the same way infer_smolvla.sh does. Defaults are set so that
running this script with no env-var overrides mimics the bash script.

Usage:
  TARGET_COLOR=yellow NUM_EPISODES=1 python tools/timed_infer_smolvla.py
"""
import atexit
import os
import statistics
import sys
import time
from collections import defaultdict

import torch

# ---------- timing -----------------------------------------------------------
stats: dict[str, list[float]] = defaultdict(list)
_step_total: list[float] = []

def _record(label, dt):
    stats[label].append(dt)

def _print_report():
    print("\n\n=========================== TIMING REPORT ==============================")
    print(f"{'stage':32s} {'n':>6s} {'avg(ms)':>10s} {'p50':>8s} {'p95':>8s} {'max':>8s}")
    print("-" * 75)
    for label in sorted(stats.keys()):
        v = stats[label]
        if not v:
            continue
        v_ms = [x * 1000 for x in v]
        v_ms.sort()
        n = len(v_ms)
        avg = sum(v_ms) / n
        p50 = v_ms[n // 2]
        p95 = v_ms[min(int(0.95 * n), n - 1)]
        mx = v_ms[-1]
        print(f"{label:32s} {n:>6d} {avg:>10.1f} {p50:>8.1f} {p95:>8.1f} {mx:>8.1f}")
    if _step_total:
        v_ms = sorted(x * 1000 for x in _step_total)
        n = len(v_ms)
        avg = sum(v_ms) / n
        print("-" * 75)
        print(f"{'derived loop step (sum stages)':32s} {n:>6d} {avg:>10.1f}  =>  {1000.0/avg:.1f} Hz")
    print("=========================================================================\n")

atexit.register(_print_report)


# ---------- monkey-patches ---------------------------------------------------
def install_patches():
    from lerobot.robots.so_follower.so_follower import SO101Follower
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor.pipeline import DataProcessorPipeline
    from lerobot.teleoperators.so_leader.so_leader import SO101Leader

    orig_get_obs = SO101Follower.get_observation
    orig_send = SO101Follower.send_action
    orig_cam_async = OpenCVCamera.async_read
    orig_cam_read = OpenCVCamera.read
    orig_select = SmolVLAPolicy.select_action
    orig_pipeline = DataProcessorPipeline.__call__
    orig_leader_get = SO101Leader.get_action

    def timed_get_obs(self, *a, **kw):
        t = time.perf_counter()
        try:
            return orig_get_obs(self, *a, **kw)
        finally:
            _record("robot.get_observation", time.perf_counter() - t)
            # also count it as a "step" tick
            _step_total.append(time.perf_counter() - t)

    def timed_send(self, *a, **kw):
        t = time.perf_counter()
        try:
            return orig_send(self, *a, **kw)
        finally:
            _record("robot.send_action", time.perf_counter() - t)

    def timed_cam_async(self, *a, **kw):
        t = time.perf_counter()
        try:
            return orig_cam_async(self, *a, **kw)
        finally:
            _record("camera.async_read", time.perf_counter() - t)

    def timed_cam_read(self, *a, **kw):
        t = time.perf_counter()
        try:
            return orig_cam_read(self, *a, **kw)
        finally:
            _record("camera.read", time.perf_counter() - t)

    def timed_select(self, *a, **kw):
        t = time.perf_counter()
        try:
            out = orig_select(self, *a, **kw)
        finally:
            try:
                torch.mps.synchronize()
            except Exception:
                pass
            _record("policy.select_action", time.perf_counter() - t)
        return out

    def timed_pipeline(self, *a, **kw):
        t = time.perf_counter()
        try:
            return orig_pipeline(self, *a, **kw)
        finally:
            # label per-pipeline by name if available
            name = getattr(self, "name", type(self).__name__)
            _record(f"pipeline:{name}", time.perf_counter() - t)

    def timed_leader_get(self, *a, **kw):
        t = time.perf_counter()
        try:
            return orig_leader_get(self, *a, **kw)
        finally:
            _record("leader.get_action", time.perf_counter() - t)

    SO101Follower.get_observation = timed_get_obs
    SO101Follower.send_action = timed_send
    OpenCVCamera.async_read = timed_cam_async
    OpenCVCamera.read = timed_cam_read
    SmolVLAPolicy.select_action = timed_select
    DataProcessorPipeline.__call__ = timed_pipeline
    SO101Leader.get_action = timed_leader_get


# ---------- argv assembly (mirror infer_smolvla.sh) --------------------------
def build_argv():
    target_color = os.environ.get("TARGET_COLOR", "yellow")
    policy = os.environ.get(
        "POLICY_PATH",
        "osammotg1/projet3-smolvla-eval2-v1-dark-noise-step44k",
    )
    policy_device = os.environ.get("POLICY_DEVICE", "mps")
    follower_port = os.environ.get(
        "FOLLOWER_PORT", "/dev/tty.usbmodem5B141129871"
    )
    leader_port = os.environ.get("LEADER_PORT", "/dev/tty.usbmodem5B141128171")
    cam_idx = os.environ.get("CAMERA_INDEX", "0")
    cam_w = os.environ.get("CAMERA_WIDTH", "640")
    cam_h = os.environ.get("CAMERA_HEIGHT", "480")
    cam_fps = os.environ.get("CAMERA_FPS", "30")
    num_eps = os.environ.get("NUM_EPISODES", "1")
    ep_time = os.environ.get("EPISODE_TIME_S", "20")
    reset_time = os.environ.get("RESET_TIME_S", "10")
    display_data = os.environ.get("DISPLAY_DATA", "false")
    push_hub = os.environ.get("PUSH_TO_HUB", "false")

    task = (
        f"Pick {target_color} block and place in bowl at (-15.5,29.5) cm"
    )
    policy_short = os.path.basename(policy).replace("/", "_")
    eval_repo = os.environ.get(
        "EVAL_REPO_ID", f"osammotg1/eval_{policy_short}-{target_color}-timed"
    )

    cameras_json = (
        f'{{"camera1": {{"type": "opencv", "index_or_path": {cam_idx}, '
        f'"width": {cam_w}, "height": {cam_h}, "fps": {cam_fps}}}}}'
    )
    rename_map = '{"observation.images.wrist": "observation.images.camera1"}'

    argv = [
        "lerobot-record",
        f"--robot.type=so101_follower",
        f"--robot.port={follower_port}",
        f"--robot.id=so101_follower",
        f"--robot.cameras={cameras_json}",
        f"--display_data={display_data}",
        f"--policy.path={policy}",
        f"--policy.device={policy_device}",
        f"--teleop.type=so101_leader",
        f"--teleop.port={leader_port}",
        f"--teleop.id=so101_leader",
        f"--dataset.repo_id={eval_repo}",
        f"--dataset.num_episodes={num_eps}",
        f"--dataset.fps={cam_fps}",
        f"--dataset.episode_time_s={ep_time}",
        f"--dataset.reset_time_s={reset_time}",
        f"--dataset.single_task={task}",
        f"--dataset.rename_map={rename_map}",
        f"--dataset.private=true",
        f"--dataset.push_to_hub={push_hub}",
    ]

    # clean stale lerobot cache for this eval repo
    cache = os.path.expanduser(f"~/.cache/huggingface/lerobot/{eval_repo}")
    if os.path.isdir(cache):
        import shutil
        print(f"[timed-infer] removing stale cache: {cache}", flush=True)
        shutil.rmtree(cache, ignore_errors=True)

    return argv


def main():
    install_patches()
    sys.argv = build_argv()
    print(f"[timed-infer] argv = {sys.argv}", flush=True)
    from lerobot.scripts.lerobot_record import main as lr_main
    lr_main()


if __name__ == "__main__":
    main()
