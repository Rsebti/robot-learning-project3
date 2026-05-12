#!/usr/bin/env python3
"""FPS health monitor for `lerobot-record`.

Pipes lerobot stdout/stderr through unchanged, parses the
    "Record loop is running slower (X.X Hz) than the target FPS (N Hz)"
warnings emitted by `lerobot/scripts/lerobot_record.py`, and overlays a
live status + audible alert on stderr so we don't record broken data.

Usage:
    lerobot-record ... 2>&1 | python3 teleop/monitor_fps.py

Or set MONITOR=true when running `bash teleop/record_eval1.sh`.

Note: lerobot only emits one combined loop frequency (camera + joint
read + dataset write + rerun publish, lumped). It does NOT distinguish
camera-FPS from joint-FPS in its log lines, so we surface a single Hz.
"""

import argparse
import re
import sys

SLOW_RE = re.compile(
    r"Record loop is running slower \(([\d.]+) Hz\) than the target FPS \((\d+) Hz\)"
)
EPISODE_RE = re.compile(r"Recording episode\s+(\d+)")

RED = "\033[31m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
BOLD = "\033[1m"
RESET = "\033[0m"
BELL = "\a"


def banner(msg: str) -> None:
    """Red, bell-printing banner — fires on alert decisions."""
    bar = "=" * 64
    sys.stderr.write(f"\n{BELL}{BOLD}{RED}{bar}\n>>> {msg}\n{bar}{RESET}\n\n")
    sys.stderr.flush()


def status(episode: str, slow_count: int, last_hz: float, target_fps: int) -> None:
    """One-line colored status update on stderr."""
    if slow_count == 0:
        color, tag = GREEN, "on target"
    elif slow_count < 5:
        color, tag = YELLOW, f"{slow_count} slow frames (last={last_hz:.1f} Hz)"
    else:
        color, tag = RED, f"{slow_count} SLOW FRAMES (last={last_hz:.1f} Hz)"
    sys.stderr.write(
        f"{color}[FPS-MON] ep={episode}  target={target_fps} Hz  {tag}{RESET}\n"
    )
    sys.stderr.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="Per-episode slow-frame count that should trigger the alert banner.",
    )
    parser.add_argument(
        "--severe-hz",
        type=float,
        default=15.0,
        help="If a single slow frame drops below this Hz, it's a severe drop.",
    )
    args = parser.parse_args()

    episode = "?"
    slow_count = 0
    last_hz = 0.0
    target_fps = 30
    alerted_this_episode = False

    for line in sys.stdin:
        sys.stdout.write(line)
        sys.stdout.flush()

        ep_m = EPISODE_RE.search(line)
        if ep_m:
            episode = ep_m.group(1)
            slow_count = 0
            alerted_this_episode = False
            status(episode, slow_count, last_hz, target_fps)
            continue

        slow_m = SLOW_RE.search(line)
        if slow_m:
            last_hz = float(slow_m.group(1))
            target_fps = int(slow_m.group(2))
            slow_count += 1

            # TODO(human): implement the alert decision logic.
            #
            # Variables you can use:
            #   - slow_count: how many slow frames in this episode so far
            #   - last_hz: Hz of the most recent slow frame (always < target_fps)
            #   - target_fps: what we asked for (usually 30)
            #   - alerted_this_episode: True if banner() already fired for this episode
            #   - args.threshold (default 10): per-episode count knob
            #   - args.severe_hz (default 15.0): "this single frame is really bad" knob
            #
            # Helpers:
            #   - banner(msg): red bell-printing banner
            #   - status(episode, slow_count, last_hz, target_fps): colored one-liner
            #
            # Things to think about:
            #   1. WHEN to fire banner() — every slow frame is too noisy. Common
            #      shapes: fire once per episode when slow_count crosses
            #      args.threshold; OR also fire immediately on a severe single
            #      drop (last_hz < args.severe_hz) even before the count
            #      threshold; OR escalate (banner at threshold, then again at
            #      2x, 5x...).
            #   2. The latch (alerted_this_episode) prevents banner spam.
            #      Decide whether to use it AS-IS or reset it on certain
            #      conditions (e.g., reset on severe drops so they always alert).
            #   3. Always call status(...) at the end of this branch so the
            #      live counter climbs visibly even when no banner fires.
            pass

    if slow_count > 0:
        sys.stderr.write(
            f"\n{YELLOW}[FPS-MON] episode {episode} ended with {slow_count} "
            f"slow frames (most recent {last_hz:.1f} Hz). Review before reusing.{RESET}\n"
        )


if __name__ == "__main__":
    main()
