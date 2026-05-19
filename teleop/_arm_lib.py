"""Shared helpers for SO-101 follower + leader scripts.

Used by:
  teleop/record_home_pose.py   — captures the home pose
  teleop/verify_home_pose.py   — drives (follower) or guides (leader) back to it
  tools/drive_to_home.py        — refactored to use drive_to() from here

API divergence between follower and leader is hidden behind read_pose() and
drive_to() / guide_to():

  SOFollower: get_observation() + send_action()  → can be driven by motor
  SOLeader:   get_action()      + send_feedback()→ position-write not supported
                                                   by the teleop class; the user
                                                   moves it by hand. We loop and
                                                   display per-joint deltas until
                                                   they are within threshold.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

DEFAULT_FOLLOWER_PORT = "/dev/tty.usbmodem5B141129871"
DEFAULT_LEADER_PORT = "/dev/tty.usbmodem5B141128171"

ANCHOR_PATH = Path(__file__).resolve().parent / "anchors" / "home_pose.json"
DRIFT_LOG_PATH = Path(__file__).resolve().parent / "anchors" / "_drift_log.jsonl"

_CALIB_ROOT = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"
_CALIB_PATH = {
    "follower": _CALIB_ROOT / "robots" / "so_follower" / "so101_follower.json",
    "leader":   _CALIB_ROOT / "teleoperators" / "so_leader" / "so101_leader.json",
}


# ---------- TTS ----------

def say_async(text: str) -> None:
    """macOS TTS, non-blocking. Silent fallback off-mac."""
    try:
        subprocess.Popen(
            ["say", text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass


# ---------- Calibration pinning ----------

def calibration_sha256(side: str) -> str | None:
    """SHA-256 of the on-disk calibration file lerobot will load for `side`."""
    p = _CALIB_PATH[side]
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------- Connect ----------

def connect_arm(
    side: str,
    port: str | None = None,
    robot_id: str | None = None,
    hold_torque_on_exit: bool = False,
):
    """Open the follower (side='follower') or leader (side='leader').

    SAFE DEFAULT: hold_torque_on_exit=False — never leave an arm energized
    unless the caller explicitly opts in. For the leader this is mandatory
    (it must stay backdrivable so teleop can use it after positioning).
    """
    if side == "follower":
        from lerobot.robots.so_follower.so_follower import SOFollower
        from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
        cfg = SO101FollowerConfig(
            port=port or DEFAULT_FOLLOWER_PORT,
            id=robot_id or "so101_follower",
            use_degrees=True,
            disable_torque_on_disconnect=not hold_torque_on_exit,
        )
        arm = SOFollower(cfg)
    elif side == "leader":
        if hold_torque_on_exit:
            raise ValueError(
                "Refusing to hold torque on the leader — it must stay "
                "backdrivable so teleop can use it after this script exits."
            )
        from lerobot.teleoperators.so_leader.so_leader import SOLeader
        from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
        cfg = SO101LeaderConfig(
            port=port or DEFAULT_LEADER_PORT,
            id=robot_id or "so101_leader",
            use_degrees=True,
        )
        arm = SOLeader(cfg)
    else:
        raise ValueError(f"side must be 'follower' or 'leader', got {side!r}")

    arm.connect()
    return arm


# ---------- Unified pose read ----------

def read_pose(arm) -> dict:
    """Read joint positions (degrees) regardless of follower/leader class."""
    if hasattr(arm, "get_observation"):
        obs = arm.get_observation()
    else:
        obs = arm.get_action()
    return {j: float(obs[f"{j}.pos"]) for j in JOINTS}


# ---------- Sample a pose ----------

def sample_pose(
    arm,
    duration_s: float = 10.0,
    hz: float = 30.0,
    say_cues: bool = True,
) -> dict:
    """Sample joint positions while the user holds the arm steady."""
    n_steps = max(2, int(duration_s * hz))
    dt = 1.0 / hz

    if say_cues:
        say_async(f"Hold the arm steady. Registering for {int(duration_s)} seconds.")
    print(f"[sample] hold steady for {duration_s:.0f}s "
          f"(sampling at {hz:.0f} Hz, {n_steps} samples)")

    samples = {j: [] for j in JOINTS}
    t0 = time.monotonic()
    for i in range(n_steps):
        pose = read_pose(arm)
        for j in JOINTS:
            samples[j].append(pose[j])
        if (i + 1) % int(hz) == 0:
            elapsed = time.monotonic() - t0
            print(f"[sample]   t={elapsed:4.1f}s  ({i+1}/{n_steps})")
        time.sleep(dt)

    if say_cues:
        say_async("Done.")

    stats = {}
    saved_pose = {}
    for j in JOINTS:
        xs = sorted(samples[j])
        med = statistics.median(xs)
        mad = statistics.median(abs(x - med) for x in xs)
        stats[j] = {
            "mean":   round(statistics.fmean(xs), 4),
            "median": round(med, 4),
            "mad":    round(mad, 4),
            "range":  round(xs[-1] - xs[0], 4),
            "n":      len(xs),
        }
        saved_pose[j] = round(med, 4)

    return {
        "joints_deg": saved_pose,
        "stats": stats,
        "n": n_steps,
        "hz": hz,
        "duration_s": duration_s,
    }


def stability_report(
    stats: dict,
    mad_threshold: float = 0.5,
    range_threshold: float = 2.0,
) -> tuple[bool, list[str]]:
    warnings = []
    for j, s in stats.items():
        if s["mad"] > mad_threshold:
            warnings.append(f"  {j}: MAD={s['mad']:.3f}° > {mad_threshold}° (shaky hand)")
        if s["range"] > range_threshold:
            warnings.append(f"  {j}: range={s['range']:.3f}° > {range_threshold}° (drifted)")
    return (len(warnings) == 0), warnings


# ---------- Drive (follower only) ----------

def drive_to(
    arm,
    target_deg: dict,
    hz: float = 30.0,
    max_vel_deg_s: float = 25.0,
    min_duration_s: float = 2.5,
    settle_s: float = 0.6,
) -> dict:
    """Smooth-interpolate the follower from its current pose to target_deg.

    Raises NotImplementedError if `arm` is a leader (no send_action).
    """
    if not hasattr(arm, "send_action"):
        raise NotImplementedError(
            "drive_to requires a follower (send_action). Use guide_to() "
            "for the leader — the SOLeader class doesn't expose position writes."
        )

    start = read_pose(arm)
    max_delta = max(abs(target_deg[j] - start[j]) for j in JOINTS)
    duration_s = max(min_duration_s, max_delta / max_vel_deg_s)
    n_steps = max(2, int(duration_s * hz))
    dt = 1.0 / hz

    print(f"[drive] max joint delta: {max_delta:.1f}° → ramp over {duration_s:.1f}s "
          f"({max_delta / duration_s:.1f}°/s, cap {max_vel_deg_s:.0f}°/s)")

    for i in range(1, n_steps + 1):
        a = i / n_steps
        wp = {f"{j}.pos": start[j] + a * (target_deg[j] - start[j]) for j in JOINTS}
        arm.send_action(wp)
        time.sleep(dt)

    time.sleep(settle_s)
    achieved = read_pose(arm)
    residual = {j: round(achieved[j] - target_deg[j], 3) for j in JOINTS}

    return {
        "start":      {j: round(start[j], 3) for j in JOINTS},
        "target":     {j: round(target_deg[j], 3) for j in JOINTS},
        "achieved":   {j: round(achieved[j], 3) for j in JOINTS},
        "residual":   residual,
        "max_delta":  round(max_delta, 3),
        "max_error":  round(max(abs(v) for v in residual.values()), 3),
        "duration_s": round(duration_s, 2),
    }


# ---------- Guide-by-hand (leader) ----------

def guide_to(
    arm,
    target_deg: dict,
    joint_threshold_deg: float = 1.0,
    poll_hz: float = 10.0,
    timeout_s: float | None = 120.0,
) -> dict:
    """Loop until the user has moved the (leader) arm to within
    `joint_threshold_deg` on every joint. Prints a live per-joint delta.
    Returns achieved pose + residual.
    """
    print(f"[guide] Move the leader by hand to match the saved pose.")
    print(f"[guide] Target: " + "  ".join(f"{j}={target_deg[j]:+7.2f}" for j in JOINTS))
    print(f"[guide] Threshold: {joint_threshold_deg:.1f}° on every joint.")
    say_async("Move the leader by hand to match the target pose.")

    dt = 1.0 / poll_hz
    t0 = time.monotonic()
    last_print = 0.0
    while True:
        pose = read_pose(arm)
        deltas = {j: target_deg[j] - pose[j] for j in JOINTS}
        worst_j = max(deltas, key=lambda j: abs(deltas[j]))
        worst = abs(deltas[worst_j])

        now = time.monotonic()
        if now - last_print > 0.2:
            line = "  ".join(f"{j}={deltas[j]:+5.1f}" for j in JOINTS)
            print(f"\r[guide] Δ° {line}  worst:{worst_j}={deltas[worst_j]:+.2f}   ",
                  end="", flush=True)
            last_print = now

        if worst <= joint_threshold_deg:
            print()
            print(f"[guide] ✓ All joints within {joint_threshold_deg}°.")
            say_async("Leader is positioned.")
            achieved = pose
            residual = {j: round(pose[j] - target_deg[j], 3) for j in JOINTS}
            return {
                "achieved": {j: round(pose[j], 3) for j in JOINTS},
                "target":   {j: round(target_deg[j], 3) for j in JOINTS},
                "residual": residual,
                "max_error": round(max(abs(v) for v in residual.values()), 3),
            }

        if timeout_s is not None and (now - t0) > timeout_s:
            print()
            print(f"[guide] Timed out after {timeout_s:.0f}s — worst residual {worst:.1f}°.")
            raise TimeoutError(f"Leader not positioned within {timeout_s}s")

        time.sleep(dt)


# ---------- Anchor persistence ----------

def load_anchor() -> dict | None:
    if not ANCHOR_PATH.exists():
        return None
    return json.loads(ANCHOR_PATH.read_text())


def save_anchor(side: str, sample_result: dict, port: str | None) -> dict:
    """Write/merge the anchor for `side`. Preserves the OTHER side if present."""
    existing = load_anchor() or {"schema_version": 1, "follower": None, "leader": None}
    if existing.get("schema_version") != 1:
        raise RuntimeError(
            f"home_pose.json schema_version != 1 (got {existing.get('schema_version')!r}); "
            "delete it or migrate."
        )

    try:
        import lerobot
        lerobot_version = getattr(lerobot, "__version__", "unknown")
    except Exception:
        lerobot_version = "unknown"

    side_block = {
        "joints_deg": sample_result["joints_deg"],
        "stats": sample_result["stats"],
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "calibration_sha256": calibration_sha256(side),
        "lerobot_version": lerobot_version,
        "port_at_capture": port,
        "sample": {"n": sample_result["n"], "hz": sample_result["hz"],
                   "duration_s": sample_result["duration_s"]},
    }
    existing[side] = side_block
    ANCHOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANCHOR_PATH.write_text(json.dumps(existing, indent=2) + "\n")
    return existing


def append_drift(entry: dict) -> None:
    DRIFT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DRIFT_LOG_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")
