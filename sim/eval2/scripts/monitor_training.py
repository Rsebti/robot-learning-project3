"""Live training monitor — prints a human-readable summary every N seconds.

Reads the most recent TensorBoard event file from the active training run
and prints metrics in human terms (percentages, ratios, normalized values)
so the trainee can understand training progress at a glance.

Usage (in a SECOND terminal, while training runs in the first):

    cd C:\\Users\\user\\Desktop\\MA2\\isaac\\isaac_so_arm101
    .\\.venv\\Scripts\\Activate.ps1
    python -m sim.eval2.scripts.monitor_training

Optional flags:
    --experiment lift_v2_10    (default: latest)
    --interval 30              (seconds between prints, default 30)
"""
from __future__ import annotations

import argparse
import os
import time
from collections import deque
from pathlib import Path

_CANDIDATE_LOG_ROOTS = [
    Path(r"C:\Users\user\Desktop\MA2\robot-learning-project3\logs\rsl_rl"),
    Path(r"C:\Users\user\Desktop\MA2\isaac\isaac_so_arm101\logs\rsl_rl"),
]


def _pick_log_root() -> Path:
    """Pick the most recently modified rsl_rl logs dir.

    Historically logs went to isaac_so_arm101/logs/, but starting V2.13
    runs (2026-05-11) the rsl_rl runner writes into the project repo's
    logs/ dir instead. We auto-detect by mtime.
    """
    existing = [p for p in _CANDIDATE_LOG_ROOTS if p.exists()]
    if not existing:
        return _CANDIDATE_LOG_ROOTS[0]
    return max(existing, key=lambda p: p.stat().st_mtime)


LOG_ROOT = _pick_log_root()


def _latest_run_dir(experiment: str | None, run: str | None = None) -> Path | None:
    if not LOG_ROOT.exists():
        return None
    if experiment:
        exp_dir = LOG_ROOT / experiment
        if not exp_dir.exists():
            return None
    else:
        # pick the most recently modified experiment dir
        candidates = [p for p in LOG_ROOT.iterdir() if p.is_dir()]
        if not candidates:
            return None
        exp_dir = max(candidates, key=lambda p: p.stat().st_mtime)
    if run:
        # explicit run dir name (e.g. "2026-05-11_01-15-42") — exact match
        run_dir = exp_dir / run
        return run_dir if run_dir.exists() else None
    runs = [p for p in exp_dir.iterdir() if p.is_dir()]
    if not runs:
        return None
    return max(runs, key=lambda p: p.stat().st_mtime)


def _read_tb_events(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    """Read all scalar events from the TB file in run_dir."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        return {}

    ea = event_accumulator.EventAccumulator(
        str(run_dir),
        size_guidance={event_accumulator.SCALARS: 0},  # all
    )
    ea.Reload()
    out = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        out[tag] = [(e.step, e.value) for e in events]
    return out


def _latest(values: list[tuple[int, float]]) -> float | None:
    return values[-1][1] if values else None


def _last_n(values: list[tuple[int, float]], n: int) -> list[float]:
    return [v for _, v in values[-n:]] if values else []


def _human(name: str, val: float | None) -> str:
    if val is None:
        return f"{name}: -"
    return f"{name}: {val}"


def render_summary(events: dict[str, list[tuple[int, float]]]) -> str:
    """Format a human-readable training summary."""
    lines: list[str] = []
    iter_n = None

    # Try to detect iteration count from any tag's last step
    if events:
        any_tag = next(iter(events.values()))
        if any_tag:
            iter_n = any_tag[-1][0]

    lines.append("=" * 70)
    lines.append(f"TRAINING MONITOR  iter ≈ {iter_n}    " + time.strftime("%H:%M:%S"))
    lines.append("=" * 70)

    # ---- Outcomes (terminations) ----
    succ = _latest(events.get("Episode_Termination/success", []))
    drop = _latest(events.get("Episode_Termination/cube_dropped", []))
    timeout = _latest(events.get("Episode_Termination/time_out", []))
    ep_len = _latest(events.get("Train/mean_episode_length", []))

    if succ is not None and ep_len:
        # Per-step rate × episode length ≈ fraction of episodes per outcome.
        # Clamped to 100 because the approximation overshoots when episodes
        # terminate non-uniformly (e.g. all-timeout regime in early training).
        succ_pct = min(100.0, 100 * succ * ep_len)
        drop_pct = min(100.0, 100 * drop * ep_len) if drop else 0
        time_pct = min(100.0, 100 * timeout * ep_len) if timeout else 0
        lines.append(f"Outcomes (estimated per-episode, clamped to 100%):")
        lines.append(f"  SUCCESS  : {succ_pct:5.1f}%")
        lines.append(f"  DROPPED  : {drop_pct:5.1f}%")
        lines.append(f"  TIMEOUT  : {time_pct:5.1f}%")
        lines.append(f"  mean episode length: {ep_len:.0f} steps  (max=150)")
    lines.append("")

    # ---- Reward terms (in % of max possible if known) ----
    # Per-step reward / weight = "fraction of frames where this fires"
    # We use rough max values to convert to percentages
    reward_max_step = {
        "Episode_Reward/reaching_object": 1.0,         # max 1.0
        "Episode_Reward/grasping_cube": 5.0,           # binary × weight
        "Episode_Reward/lifting_object": 10.0,
        "Episode_Reward/object_goal_tracking": 16.0,
        "Episode_Reward/object_goal_tracking_fine_grained": 5.0,
        "Episode_Reward/success_bonus": 1500.0,        # sparse
    }
    lines.append("Reward signals (% of max possible per step):")
    for tag, max_v in reward_max_step.items():
        val = _latest(events.get(tag, []))
        if val is None:
            continue
        pct = 100 * val / max_v if max_v else 0
        bar = "▓" * int(pct / 5) + "░" * (20 - int(pct / 5))
        clean_name = tag.split("/")[-1]
        lines.append(f"  {clean_name:35s} {pct:5.2f}% [{bar}]")
    lines.append("")

    # ---- Negative regularizers ----
    neg_tags = [
        "Episode_Reward/action_rate",
        "Episode_Reward/joint_vel",
        "Episode_Reward/joint_acc",
        "Episode_Reward/cube_dropped_penalty",
    ]
    lines.append("Penalty signals (closer to 0 = smoother):")
    for tag in neg_tags:
        val = _latest(events.get(tag, []))
        if val is None:
            continue
        lines.append(f"  {tag.split('/')[-1]:35s} {val:+.4f}")
    lines.append("")

    # ---- Policy health ----
    noise = _latest(events.get("Policy/mean_noise_std", []))
    ent = _latest(events.get("Loss/entropy", []))
    vf = _latest(events.get("Loss/value_function", []))
    lr = _latest(events.get("Loss/learning_rate", []))

    lines.append("Policy/training health:")
    if noise is not None:
        trend = ""
        recent = _last_n(events.get("Policy/mean_noise_std", []), 10)
        if len(recent) >= 5:
            d = recent[-1] - recent[0]
            trend = " ↘ converging" if d < -0.01 else (" ↗ diverging ⚠" if d > 0.01 else " ─ stable")
        lines.append(f"  noise_std         : {noise:.3f}{trend}  (smaller = more committed; <0.5 is good)")
    if ent is not None:
        recent = _last_n(events.get("Loss/entropy", []), 10)
        trend = ""
        if len(recent) >= 5:
            d = recent[-1] - recent[0]
            trend = " ↘ converging" if d < -0.05 else (" ↗ diverging ⚠" if d > 0.05 else " ─ stable")
        lines.append(f"  entropy           : {ent:.2f}{trend}  (smaller = policy committed)")
    if vf is not None:
        lines.append(f"  value_function    : {vf:.3f}  (smaller = critic accurate)")
    if lr is not None:
        lines.append(f"  learning_rate     : {lr:.2e}  (adaptive KL; throttled if too high)")
    lines.append("")

    # ---- Reward total ----
    mean_rew = _latest(events.get("Train/mean_reward", []))
    if mean_rew is not None:
        lines.append(f"Mean total reward / episode : {mean_rew:.2f}")
    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default=None,
                        help="experiment name (default: latest by mtime)")
    parser.add_argument("--run", default=None,
                        help="explicit run dir name within experiment "
                             "(e.g. '2026-05-11_01-15-42'). Defaults to "
                             "latest by mtime.")
    parser.add_argument("--interval", type=int, default=30,
                        help="seconds between prints")
    args = parser.parse_args()

    print("[INFO] Watching training logs. Ctrl+C to stop.")
    print(f"[INFO] Log root: {LOG_ROOT}")
    print(f"[INFO] Interval: {args.interval}s")
    if args.experiment:
        print(f"[INFO] Experiment: {args.experiment}")
    else:
        print("[INFO] Experiment: latest (auto-detect)")
    if args.run:
        print(f"[INFO] Run: {args.run} (pinned)")

    while True:
        run_dir = _latest_run_dir(args.experiment, args.run)
        if run_dir is None:
            print(f"[WARN] No run found under {LOG_ROOT}. Waiting…")
        else:
            events = _read_tb_events(run_dir)
            if not events:
                print(f"[WARN] No TB events yet in {run_dir.name}. Waiting…")
            else:
                print(render_summary(events))
                print(f"[INFO] Source: {run_dir}")
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
