"""
debug_sac_step.py - step-by-step diagnostic for the friend's SAC SO101
checkpoint. Each phase is bracketed by 3-second sleeps so you can watch
the arm and read the prints.

What it does:
  1) Connect to robot.                                   [pause 3 s]
  2) Read motor EEPROM (Homing_Offset, Min, Max).        [pause 3 s]
  3) Compare EEPROM to calibration JSON.                 [pause 3 s]
  4) Read live Present_Position (deg).                   [pause 3 s]
  5) Disable torque.                                     [pause 3 s]
  6) Re-enable torque.                                   [pause 3 s]
  7) Load the SAC checkpoint and run ONE forward pass
     with the CURRENT obs (qpos, qpos-target, wrist cam).
     Print the action and what target_qpos would become.
                                                         [pause 3 s]
  8) Loop: every 3 s, send the action one step, then read
     back what happened. --max_steps N (default 5).      [pause 3 s each]

It does NOT change EEPROM or move the arm large distances; the per-step
action is clamped by DELTA_CAP just like the real rollout.

Usage:
    python -m toolset.control.debug_sac_step --port COM3 \\
        --checkpoint deploy/eval1/eval1_ckpt.pt --max_steps 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# Windows MSMF -> DSHOW patch
_OriginalVideoCapture = cv2.VideoCapture


class _DShowVideoCapture(_OriginalVideoCapture):
    def __init__(self, *args, **kwargs):
        if len(args) >= 2 and isinstance(args[0], int) and args[1] in (
            cv2.CAP_ANY, cv2.CAP_MSMF
        ):
            args = (args[0], cv2.CAP_DSHOW)
        elif len(args) == 1 and isinstance(args[0], int):
            args = (args[0], cv2.CAP_DSHOW)
        super().__init__(*args, **kwargs)


cv2.VideoCapture = _DShowVideoCapture


CALIB_PATH = (
    Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"
    / "robots" / "so_follower" / "so101_follower.json"
)
PAUSE_S = 3.0


def banner(s: str):
    bar = "=" * 70
    print(f"\n{bar}\n  {s}\n{bar}", flush=True)


def pause():
    print(f"  ... ({PAUSE_S:.0f} s pause) ...", flush=True)
    time.sleep(PAUSE_S)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="COM3")
    p.add_argument("--camera_index", type=int, default=1)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--max_steps", type=int, default=5)
    p.add_argument("--action_scale", type=float, default=0.05)
    args = p.parse_args()

    # connect
    banner("STEP 1: connect")
    deploy_dir = Path(__file__).resolve().parents[2] / "deploy"
    if str(deploy_dir) not in sys.path:
        sys.path.insert(0, str(deploy_dir))
    from robot_calibration import make_so101_follower_config  # noqa: E402
    from lerobot.robots.utils import make_robot_from_config  # noqa: E402
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: E402

    rconf = make_so101_follower_config(
        args.port, cameras={"wrist": OpenCVCameraConfig(
            index_or_path=args.camera_index, fps=30, width=640, height=480)},
        use_degrees=True,
    )
    robot = make_robot_from_config(rconf)
    print(f"  connecting to {args.port} ...")
    robot.connect()
    bus = robot.bus
    pause()

    # EEPROM
    banner("STEP 2: read motor EEPROM (Homing_Offset / Min / Max)")
    eep = {}
    for m in bus.motors.keys():
        h = int(bus.sync_read("Homing_Offset", [m], normalize=False)[m])
        mn = int(bus.sync_read("Min_Position_Limit", [m], normalize=False)[m])
        mx = int(bus.sync_read("Max_Position_Limit", [m], normalize=False)[m])
        eep[m] = (h, mn, mx)
        print(f"  {m:<14}  homing={h:+5d}  min={mn:>5d}  max={mx:>5d}")
    pause()

    # compare to JSON
    banner("STEP 3: compare motor EEPROM to calibration JSON")
    with open(CALIB_PATH) as f:
        calib = json.load(f)
    mismatch = False
    for m, (h, mn, mx) in eep.items():
        jh = int(calib[m]["homing_offset"])
        jmn = int(calib[m]["range_min"])
        jmx = int(calib[m]["range_max"])
        ok = (h == jh) and (mn == jmn) and (mx == jmx)
        marker = "" if ok else "  <-- MISMATCH"
        if not ok:
            mismatch = True
        print(f"  {m:<14}  motor=({h:+d},{mn},{mx})  json=({jh:+d},{jmn},{jmx}){marker}")
    print(f"\n  any mismatch: {mismatch}")
    pause()

    # live positions
    banner("STEP 4: read live Present_Position (servo deg)")
    obs0 = robot.get_observation()
    for m in bus.motors.keys():
        print(f"  {m:<14}  {obs0[f'{m}.pos']:+8.2f} deg")
    cam0 = obs0.get("wrist")
    if cam0 is not None:
        cam0 = np.asarray(cam0, dtype=np.uint8)
        print(f"  wrist camera: shape={cam0.shape}  dtype={cam0.dtype}  mean={cam0.mean():.1f}")
    pause()

    # disable / re-enable torque (sanity)
    banner("STEP 5: disable torque")
    print("  >>> support the arm <<<  (will re-enable in 3 s)")
    pause()
    bus.disable_torque()
    print("  torque disabled.")
    pause()

    banner("STEP 6: re-enable torque")
    bus.enable_torque()
    print("  torque re-enabled.")
    pause()

    # load policy
    banner("STEP 7: load SAC checkpoint + ONE forward pass with current obs")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    n_state = ckpt["actor"]["proj.state_proj.0.weight"].shape[1]
    print(f"  checkpoint n_state = {n_state} ({'12=Eval1' if n_state == 12 else 'OTHER'})")

    sys.path.insert(0, str(Path(args.checkpoint).resolve().parent))
    sys.path.insert(0, str(deploy_dir))
    # Re-use friend's policy classes
    import importlib.util
    legacy = deploy_dir / "eval1" / "infer_eval1_sac_legacy.py"
    spec = importlib.util.spec_from_file_location("infer_eval1_sac_legacy", legacy)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["infer_eval1_sac_legacy"] = mod
    spec.loader.exec_module(mod)

    encoder = mod.CNNEncoder().eval()
    actor = mod.Actor(n_state=n_state).eval()
    encoder.load_state_dict(ckpt["encoder"])
    actor.load_state_dict(ckpt["actor"])

    agent = mod.RealRobotAgent(robot)
    qpos = agent.get_qpos().cpu().numpy().flatten()
    target_qpos = qpos.copy()
    print(f"  initial measured qpos (deg): {np.rad2deg(qpos).round(2).tolist()}")
    print(f"  initial target  qpos (deg): {np.rad2deg(target_qpos).round(2).tolist()}")

    # Build one observation, run policy
    agent.capture_sensor_data()
    sensor = agent.get_sensor_data()
    rgb = sensor[list(sensor.keys())[0]]["rgb"]  # (1, H, W, 3) uint8
    print(f"  rgb tensor shape: {tuple(rgb.shape)}  mean={rgb.float().mean():.1f}")
    state = mod.build_state(qpos, target_qpos, goal_color=0)
    with torch.no_grad():
        z = encoder(rgb.float())
        a, _ = actor(z, torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
    a_np = a.cpu().numpy().flatten()
    print(f"  raw action (-1..1):                 {a_np.round(3).tolist()}")
    a_scaled = a_np * args.action_scale
    print(f"  scaled action (x{args.action_scale}):     {a_scaled.round(4).tolist()}")
    delta = a_scaled * mod.DELTA_CAP
    print(f"  joint delta this step (rad):        {delta.round(4).tolist()}")
    print(f"  joint delta this step (deg):        {np.rad2deg(delta).round(2).tolist()}")
    pause()

    # Drive a few steps slowly
    banner(f"STEP 8: send {args.max_steps} actions, 3 s apart, observe motor tracking")
    for step in range(args.max_steps):
        qpos = agent.get_qpos().cpu().numpy().flatten()
        agent.capture_sensor_data()
        rgb = agent.get_sensor_data()[list(sensor.keys())[0]]["rgb"]
        state = mod.build_state(qpos, target_qpos, goal_color=0)
        with torch.no_grad():
            z = encoder(rgb.float())
            a, _ = actor(z, torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
        a_np = a.cpu().numpy().flatten()
        target_qpos = np.clip(
            target_qpos + a_np * args.action_scale * mod.DELTA_CAP,
            mod.JOINT_LOWER, mod.JOINT_UPPER,
        )
        agent.set_target_qpos(torch.from_numpy(target_qpos.astype(np.float32)))

        time.sleep(0.5)  # give motors time to react
        live = agent.get_qpos().cpu().numpy().flatten()
        diff = live - target_qpos
        print(f"\n  step {step}:")
        print(f"    action raw (-1..1):     {a_np.round(2).tolist()}")
        print(f"    target qpos  (deg):     {np.rad2deg(target_qpos).round(1).tolist()}")
        print(f"    measured qpos (deg):    {np.rad2deg(live).round(1).tolist()}")
        print(f"    measured - target (deg): {np.rad2deg(diff).round(1).tolist()}  "
              f"||={np.linalg.norm(np.rad2deg(diff)):.1f} deg")
        if step + 1 < args.max_steps:
            pause()

    banner("DONE")
    print("  Look at STEP 8 prints: does any joint go FAR from its target?")
    print("  - If 'measured - target' grows large: motor isn't tracking command.")
    print("  - If 'target qpos' itself swings 90 deg: policy is outputting saturated")
    print("    actions, and the issue is upstream (obs not matching training).")
    robot.disconnect()
    print("  disconnected.")


if __name__ == "__main__":
    main()
