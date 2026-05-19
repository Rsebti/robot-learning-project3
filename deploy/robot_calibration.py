"""
Project-local SO-101 follower calibration (LeRobot 0.5.x).

Canonical file:
    deploy/calibration/so101_follower.json

All deploy inference scripts should load calibration from here via
``make_so101_follower_config()`` so joint degrees match training/demos.

After ``lerobot-calibrate`` (writes to the HF cache), run:
    python -m toolset.calibration_scripts.manage_calibration pull

Before first connect on a fresh PC (cache empty), run:
    python -m toolset.calibration_scripts.manage_calibration install
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

DEPLOY_DIR = Path(__file__).resolve().parent
LOCAL_CALIBRATION_DIR = DEPLOY_DIR / "calibration"
ROBOT_ID = "so101_follower"
LOCAL_CALIBRATION_FILE = LOCAL_CALIBRATION_DIR / f"{ROBOT_ID}.json"


def hf_cache_calibration_candidates() -> list[Path]:
    """Possible LeRobot cache paths (0.5.x and older layouts)."""
    base = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"
    return [
        base / "robots" / "so_follower" / f"{ROBOT_ID}.json",
        base / "robots" / "so_follower" / ROBOT_ID / "calibration.json",
        base / f"{ROBOT_ID}.json",
    ]


def resolve_hf_cache_calibration() -> Path | None:
    for p in hf_cache_calibration_candidates():
        if p.is_file():
            return p
    return None


def ensure_local_calibration_dir() -> Path:
    LOCAL_CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    return LOCAL_CALIBRATION_DIR


def pull_cache_to_local(*, force: bool = False) -> Path:
    """Copy HF-cache calibration → deploy/calibration/ (after lerobot-calibrate)."""
    src = resolve_hf_cache_calibration()
    if src is None:
        raise FileNotFoundError(
            "No calibration in HF cache. Run lerobot-calibrate first, e.g.\n"
            "  lerobot-calibrate --robot.type=so101_follower "
            f"--robot.port=COM3 --robot.id={ROBOT_ID}"
        )
    ensure_local_calibration_dir()
    if LOCAL_CALIBRATION_FILE.exists() and not force:
        bak = LOCAL_CALIBRATION_FILE.with_suffix(".json.bak")
        shutil.copy(LOCAL_CALIBRATION_FILE, bak)
        print(f"[calib] backed up local -> {bak}")
    shutil.copy(src, LOCAL_CALIBRATION_FILE)
    print(f"[calib] pulled {src} -> {LOCAL_CALIBRATION_FILE}")
    return LOCAL_CALIBRATION_FILE


def install_local_to_cache(*, force: bool = False) -> Path:
    """Copy deploy/calibration/ → HF cache so bare lerobot CLI uses the same file."""
    if not LOCAL_CALIBRATION_FILE.is_file():
        raise FileNotFoundError(
            f"Local calibration missing: {LOCAL_CALIBRATION_FILE}\n"
            "Run lerobot-calibrate + manage_calibration pull, or copy an existing JSON there."
        )
    dst = hf_cache_calibration_candidates()[0]
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        bak = dst.with_suffix(".json.bak")
        shutil.copy(dst, bak)
        print(f"[calib] backed up cache -> {bak}")
    shutil.copy(LOCAL_CALIBRATION_FILE, dst)
    print(f"[calib] installed {LOCAL_CALIBRATION_FILE} -> {dst}")
    return dst


def validate_calibration(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    required = ("shoulder_pan", "shoulder_lift", "elbow_flex",
                "wrist_flex", "wrist_roll", "gripper")
    missing = [m for m in required if m not in data]
    if missing:
        raise ValueError(f"{path} missing motors: {missing}")
    for m in required:
        for key in ("id", "homing_offset", "range_min", "range_max"):
            if key not in data[m]:
                raise ValueError(f"{path} motor {m} missing {key}")
    return data


def make_so101_follower_config(
    port: str,
    *,
    cameras: dict | None = None,
    calibration_dir: Path | str | None = None,
    robot_id: str = ROBOT_ID,
    use_degrees: bool = True,
):
    """Build SO101FollowerConfig pointing at project-local calibration."""
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig

    cal_dir = Path(calibration_dir) if calibration_dir is not None else LOCAL_CALIBRATION_DIR
    cal_file = cal_dir / f"{robot_id}.json"
    if not cal_file.is_file():
        raise FileNotFoundError(
            f"Calibration not found: {cal_file}\n"
            "See deploy/calibration/README.md — run manage_calibration pull/install."
        )

    kwargs: dict = {
        "port": port,
        "id": robot_id,
        "calibration_dir": cal_dir,
        "use_degrees": use_degrees,
    }
    if cameras is not None:
        kwargs["cameras"] = cameras
    return SO101FollowerConfig(**kwargs)
