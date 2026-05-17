#!/usr/bin/env python3
"""Capture canonical SO-101 joint angles at 4 sim-to-real anchor poses.

Drives the follower smoothly through each pose, announces with macOS `say`
(plus stdout), waits a configurable photo window, then writes a markdown
record so a sim engineer can replicate the same configuration in Isaac Lab
/ MuJoCo / URDF and check visual + numerical agreement.

Output:
  notes/sim_anchors_<DATE>.md
  notes/sim_anchors_<DATE>_calibration.json
  notes/sim_anchors_<DATE>_images/<pose>.jpg
"""
import argparse
import hashlib
import json
import math
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2

from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SOFollower


JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Training home pose of osammotg1/projet3-eval1-bowl1-v1 (degrees).
HOME = {
    "shoulder_pan":  -2.0,
    "shoulder_lift": -92.2,
    "elbow_flex":    +97.7,
    "wrist_flex":    +75.2,
    "wrist_roll":     0.0,
    "gripper":        +1.2,
}


@dataclass
class PoseDef:
    slug: str
    description: str
    target: dict  # joint -> degrees


def _with(base, **overrides):
    out = dict(base)
    out.update(overrides)
    return out


POSES = [
    PoseDef(
        slug="home_pose",
        description=(
            "Training home for the eval1 dark-shadow policy. Arm folded, "
            "gripper closed, wrist hovering above workspace. This is the "
            "operational baseline every rollout starts from."
        ),
        target=HOME,
    ),
    PoseDef(
        slug="shoulder_pan_30",
        description=(
            "Home pose with shoulder_pan rotated to +30 deg. Tests the "
            "shoulder_pan axis sign and direction: the gripper should swing "
            "roughly 30 deg about the base from its home position."
        ),
        target=_with(HOME, shoulder_pan=+30.0),
    ),
    PoseDef(
        slug="wrist_rolled_45",
        description=(
            "Home pose with wrist_roll rotated to +45 deg. Tests the wrist_roll "
            "axis (the joint that carries the +1588 homing_offset on this Mac "
            "per project_calibration_shift_2026-05-17). +45 stays within the "
            "documented folded-arm wrist plateau (50/80 deg per "
            "feedback_so101_wrist_camera_collision); +90 caused the wrist cam "
            "to collide with the robot base on 2026-05-18, hence the safer "
            "magnitude here."
        ),
        target=_with(HOME, wrist_roll=+45.0),
    ),
    PoseDef(
        slug="gripper_open",
        description=(
            "Home pose with gripper opened to +30 deg. Tests the gripper axis. "
            "Chosen over an all-zeros 'canonical' pose because (a) the real "
            "arm cannot physically reach elbow_flex=0 from home (self-collision "
            "stalls it at +14.6 deg) and (b) jaws-open vs jaws-closed is a "
            "binary visual signal that survives photo compression and rendering "
            "differences. +30 deg matches the open-gripper outliers seen in the "
            "dark-shadow training data (ep08, ep24, ep32)."
        ),
        target=_with(HOME, gripper=+30.0),
    ),
]


# ---------- I/O helpers ----------

def say_async(text: str) -> None:
    """macOS TTS, non-blocking. Silent fallback if `say` is unavailable."""
    try:
        subprocess.Popen(
            ["say", text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass


def grab_wrist_frame(cam_index: int, out_path: Path) -> bool:
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        return False
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for _ in range(10):  # warmup; AVFoundation needs a few frames to expose
        cap.read()
    ok, frame = cap.read()
    cap.release()
    if not (ok and frame is not None):
        return False
    cv2.imwrite(str(out_path), frame)
    return True


def find_calibration() -> Path | None:
    cache = Path.home() / ".cache/huggingface/lerobot/calibration"
    if not cache.exists():
        return None
    for p in cache.rglob("so101_follower.json"):
        return p
    return None


def snapshot_calibration(dest: Path) -> tuple[Path | None, str | None]:
    src = find_calibration()
    if src is None:
        return None, None
    data = src.read_text()
    dest.write_text(data)
    sha = hashlib.sha256(data.encode()).hexdigest()
    return src, sha


# ---------- motion ----------

def drive_to(robot, target: dict, hz: float = 30.0) -> tuple[dict, float, float]:
    obs = robot.get_observation()
    start = {j: float(obs[f"{j}.pos"]) for j in JOINTS}
    max_delta = max(abs(target[j] - start[j]) for j in JOINTS)
    # Cap angular velocity ~25 deg/s; never faster than the 2.5s baseline.
    duration_s = max(2.5, max_delta / 25.0)
    n_steps = max(2, int(duration_s * hz))
    dt = 1.0 / hz
    for i in range(1, n_steps + 1):
        a = i / n_steps
        wp = {f"{j}.pos": start[j] + a * (target[j] - start[j]) for j in JOINTS}
        robot.send_action(wp)
        time.sleep(dt)
    time.sleep(0.6)  # let the final waypoint settle
    final = robot.get_observation()
    achieved = {j: float(final[f"{j}.pos"]) for j in JOINTS}
    return achieved, max_delta, duration_s


def photo_window(seconds: float) -> None:
    """Block for `seconds`, announcing remaining time at key intervals."""
    t0 = time.time()
    last = None
    print(f"    PHOTO WINDOW {seconds:.0f}s -- take your picture now")
    while True:
        remaining = seconds - (time.time() - t0)
        if remaining <= 0:
            break
        r = int(remaining)
        if r != last and r in (8, 5, 3, 2, 1):
            print(f"    ... {r}s")
            last = r
        time.sleep(0.1)


# ---------- markdown ----------

def render_md(
    records: list[dict],
    calib_src: Path | None,
    calib_sha: str | None,
    calib_dst: Path,
    md_path: Path,
    image_dir: Path,
) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    host = socket.gethostname()
    lines: list[str] = []

    lines.append("# SO-101 sim-to-real joint anchors")
    lines.append("")
    lines.append(f"- Generated: `{ts}`")
    lines.append(f"- Host: `{host}`")
    lines.append(f"- Joint order: `{', '.join(JOINTS)}`")
    lines.append("- Units: degrees (radians in parens). Achieved angles, not commanded.")
    lines.append("- lerobot convention: `use_degrees=True`. Gripper is the servo angle, NOT a jaw-opening percentage.")
    lines.append("")
    lines.append("## Calibration snapshot")
    lines.append("")
    if calib_src is not None:
        lines.append(f"- Source on this machine: `{calib_src}`")
        lines.append(f"- Snapshot file: `{calib_dst.name}` (sibling to this MD)")
        lines.append(f"- SHA256: `{calib_sha}`")
        try:
            calib_obj = json.loads(calib_dst.read_text())
            offsets = {
                m: calib_obj[m].get("homing_offset")
                for m in calib_obj
                if isinstance(calib_obj[m], dict) and "homing_offset" in calib_obj[m]
            }
            if offsets:
                lines.append("- Key homing offsets:")
                for m, v in offsets.items():
                    lines.append(f"  - `{m}.homing_offset = {v}`")
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    else:
        lines.append("- WARNING: no calibration file found in cache when this MD was generated.")
    lines.append("")
    lines.append("## NOT in scope (v1)")
    lines.append("")
    lines.append("- Dynamics / mass / torque / friction comparison")
    lines.append("- Forward-kinematics end-effector xyz (deferred; would depend on the MJCF we are trying to validate)")
    lines.append("- Sim-side validation (Ryan's responsibility)")
    lines.append("- Cross-machine averaging (single-machine snapshot)")
    lines.append("")
    lines.append("## Poses")
    lines.append("")
    for i, r in enumerate(records, 1):
        lines.append(f"### {i}. `{r['slug']}`")
        lines.append("")
        lines.append(r["description"])
        lines.append("")
        lines.append("**Commanded target:**")
        lines.append("")
        lines.append("```")
        for j in JOINTS:
            lines.append(f"  {j:15s} = {r['target'][j]:+8.2f} deg")
        lines.append("```")
        lines.append("")
        lines.append("**Achieved (servo-reported after settle):**")
        lines.append("")
        lines.append("```")
        for j in JOINTS:
            d = r["achieved"][j]
            rad = math.radians(d)
            lines.append(f"  {j:15s} = {d:+8.2f} deg  ({rad:+8.4f} rad)")
        lines.append("```")
        lines.append("")
        if r["image"]:
            lines.append(f"**Wrist cam frame:** `{Path(r['image']).name}`")
            lines.append("")
        if r.get("photo_taken"):
            lines.append(f"**External photo taken by operator:** yes (during {r['photo_window_s']:.0f}s window)")
            lines.append("")
    md_path.write_text("\n".join(lines) + "\n")


# ---------- main ----------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", default="/dev/tty.usbmodem5B141129871")
    p.add_argument("--id", default="so101_follower")
    p.add_argument("--cam-index", type=int, default=0)
    p.add_argument("--photo-window-s", type=float, default=10.0)
    p.add_argument("--pre-photo-pause-s", type=float, default=1.5)
    p.add_argument("--no-voice", action="store_true")
    p.add_argument("--out-dir", default="notes")
    args = p.parse_args()

    date = datetime.now().strftime("%Y-%m-%d")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"sim_anchors_{date}.md"
    calib_dst = out_dir / f"sim_anchors_{date}_calibration.json"
    image_dir = out_dir / f"sim_anchors_{date}_images"
    image_dir.mkdir(exist_ok=True)

    def speak(t: str) -> None:
        print(f"    >>> {t}")
        if not args.no_voice:
            say_async(t)

    cfg = SO101FollowerConfig(
        port=args.port,
        id=args.id,
        use_degrees=True,
        disable_torque_on_disconnect=True,
    )
    robot = SOFollower(cfg)
    print(f"[anchors] connecting to {args.port}")
    robot.connect()

    try:
        calib_src, calib_sha = snapshot_calibration(calib_dst)
        if calib_src is not None:
            print(f"[anchors] calibration: {calib_src}")
            print(f"[anchors]   -> snapshot {calib_dst.name} (sha256 {calib_sha[:16]}...)")
        else:
            print("[anchors] WARNING: no calibration file found in cache")

        records: list[dict] = []
        for i, pose in enumerate(POSES, 1):
            print()
            print(f"=== Pose {i}/{len(POSES)}: {pose.slug} ===")
            print(f"    {pose.description}")
            achieved, mx, dur = drive_to(robot, pose.target)
            print(f"    moved (max delta {mx:.1f} deg over {dur:.1f}s)")
            print("    achieved:")
            for j in JOINTS:
                print(f"      {j:15s} = {achieved[j]:+8.2f} deg")

            time.sleep(args.pre_photo_pause_s)
            spoken = pose.slug.replace("_", " ")
            speak(f"Position {i}, {spoken}. Record position.")
            img_path = image_dir / f"{pose.slug}.jpg"
            grabbed = grab_wrist_frame(args.cam_index, img_path)
            if not grabbed:
                print(f"    WARNING: failed to grab wrist-cam frame (index {args.cam_index})")
            photo_window(args.photo_window_s)
            speak("Moving.")

            records.append({
                "slug": pose.slug,
                "description": pose.description,
                "target": pose.target,
                "achieved": achieved,
                "image": str(img_path) if grabbed else None,
                "photo_taken": True,
                "photo_window_s": args.photo_window_s,
            })

        print()
        print("=== Returning to home for safe disconnect ===")
        drive_to(robot, HOME)

        render_md(records, calib_src, calib_sha, calib_dst, md_path, image_dir)
        print()
        print(f"[anchors] wrote {md_path}")
        print(f"[anchors] wrote {calib_dst}")
        print(f"[anchors] images in {image_dir}/")
        speak("Done.")
    finally:
        print("[anchors] disconnecting (torque off).")
        robot.disconnect()

    return 0


if __name__ == "__main__":
    sys.exit(main())
