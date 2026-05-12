"""Live training monitor — V2.18 cold-start ("Precision Landing") architecture.

Reads the most recent TensorBoard event file from the active training run
and prints metrics in human terms (percentages, ratios, normalized values)
so the trainee can understand training progress at a glance.

V2.18 architecture vs older variants:
  - Bounded per-step magnitude budget |r| ≤ +6.3 (vs V2.17's |V|≈4500 that
    crashed). All dense terms are clamped/normalized so the value function
    cannot blow up.
  - Strict 6-condition grasp predicate. `grasping_cube` only fires when
    cube is geometrically BETWEEN the jaws (not just "jaw closed near
    cube" → V2.7→V2.15 false positives ejected 89% of attempts).
  - Multiplicative gating: `lifting_object`, `object_goal_tracking`, fine
    are all × strict_grasp → "lift without true grasp" is mathematically
    impossible.
  - 12 dense terms + 2 terminals (success / drop).
  - V2.16's `cube_height_above_spawn` term is REMOVED (replaced by gated
    `lifting_object`).

LIFT BAR semantics (V2.18):
  - Uses `lifting_object` directly (= lift_height_gated, max +1.5/step).
  - NOISE ZONE: lifting_object < 0.05/step → gate rarely True, no real lift.
  - LIFTING:    lifting_object ≥ 0.05/step → strict_grasp fires AND cube up.
  - Maxed at 1.5/step (sustained perfect lift over the whole episode).

EXPLOIT WATCH section flags two known V2.18 risks:
  - Hover-stall: `hover_height` monotone ↗ past iter 300 → policy is
    hovering above cube without grasping (exploits the +0.5 hover bonus).
  - Sparse-gradient-stuck: `grasping_cube` ≈ 0 + `palm_xy_above_cube` > 0.4
    → policy aligned laterally but strict predicate never trips → 6 conds
    are too tight, may need looser predicate.

Usage (in a SECOND terminal, while training runs in the first):

    cd C:\\Users\\user\\Desktop\\MA2\\robot-learning-project3
    C:/Users/user/Desktop/MA2/isaac/isaac_so_arm101/.venv/Scripts/python.exe `
      -m sim.eval2.scripts.monitor_training --experiment lift_v2_13 --interval 30

Optional flags:
    --experiment lift_v2_13    (default: latest by mtime)
    --run 2026-05-12_14-30-00  (pin a specific run; default: latest by mtime)
    --interval 30              (seconds between prints, default 30)
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from collections import deque
from pathlib import Path

# Force UTF-8 stdout so the unicode bar chars (▓░) render correctly under
# Windows PowerShell (which defaults to cp1252).
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  line_buffering=True)

_CANDIDATE_LOG_ROOTS = [
    Path(r"C:\Users\user\Desktop\MA2\robot-learning-project3\logs\rsl_rl"),
    Path(r"C:\Users\user\Desktop\MA2\isaac\isaac_so_arm101\logs\rsl_rl"),
]


def _pick_log_root() -> Path:
    """Pick the most recently modified rsl_rl logs dir."""
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
        candidates = [p for p in LOG_ROOT.iterdir() if p.is_dir()]
        if not candidates:
            return None
        exp_dir = max(candidates, key=lambda p: p.stat().st_mtime)
    if run:
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
        size_guidance={event_accumulator.SCALARS: 0},
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


def _trend(values: list[tuple[int, float]], n: int = 10,
           up_thresh: float = 0.01, down_thresh: float = -0.01) -> str:
    """Return arrow indicating recent trend over last n samples."""
    recent = _last_n(values, n)
    if len(recent) < 5:
        return ""
    d = recent[-1] - recent[0]
    if d > up_thresh:
        return " ↗"
    if d < down_thresh:
        return " ↘"
    return " ─"


def _bar(pct: float, width: int = 20) -> str:
    """Render a progress bar — pct in [0, 100]."""
    n = min(width, max(0, int(pct * width / 100)))
    return "▓" * n + "░" * (width - n)


def render_summary(events: dict[str, list[tuple[int, float]]]) -> str:
    """Format a human-readable training summary for V2.18."""
    lines: list[str] = []
    iter_n = None
    if events:
        any_tag = next(iter(events.values()))
        if any_tag:
            iter_n = any_tag[-1][0]

    lines.append("=" * 72)
    lines.append(f"V2.18 TRAINING MONITOR  iter ≈ {iter_n}    " + time.strftime("%H:%M:%S"))
    lines.append("=" * 72)

    # ---- Outcomes (terminations) ----
    succ = _latest(events.get("Episode_Termination/success", []))
    drop = _latest(events.get("Episode_Termination/cube_dropped", []))
    timeout = _latest(events.get("Episode_Termination/time_out", []))
    ep_len = _latest(events.get("Train/mean_episode_length", []))

    if succ is not None and ep_len:
        # Per-step rate × episode length ≈ fraction of episodes per outcome.
        succ_pct = min(100.0, 100 * succ * ep_len)
        drop_pct = min(100.0, 100 * drop * ep_len) if drop else 0
        time_pct = min(100.0, 100 * timeout * ep_len) if timeout else 0
        lines.append("Outcomes (estimated per-episode, clamped to 100%):")
        lines.append(f"  SUCCESS  : {succ_pct:5.1f}%")
        lines.append(f"  DROPPED  : {drop_pct:5.1f}%")
        lines.append(f"  TIMEOUT  : {time_pct:5.1f}%")
        lines.append(f"  mean episode length: {ep_len:.0f} steps  (max=300, ep_length_s=10s)")
    lines.append("")

    # ---- V2.18 Positive reward terms ----
    # Per-step max magnitudes from V2.18 design:
    #   reaching_object       weight 1.0  → max +1.0/step
    #   grasping_cube (STRICT) weight 2.0 → max +2.0/step
    #   lifting_object (GATED) weight 1.5 → max +1.5/step
    #   object_goal_tracking   weight 1.0 → max +1.0/step
    #   object_goal_tracking_fine_grained weight 0.5 → max +0.5/step
    #   palm_xy_above_cube     weight 0.8 → max +0.8/step
    #   hover_height           weight 0.5 → max +0.5/step
    #   palm_to_jaw_orient_v218 weight 0.3 → max +0.3/step
    #   success_bonus          weight 2000 (one-shot terminal)
    # NB: TB tag = field name in RewardsCfgV218 dataclass. V218 keeps two
    # *legacy* field names whose semantics changed:
    #   - gripper_orientation_penalty → now a POSITIVE +0.3 reward (sign-fixed)
    #   - jaw_below_cube_penalty      → now jaw_table_impact_penalty fn (gated)
    pos_terms = [
        ("Episode_Reward/reaching_object", 1.0, "approach cube (always on)"),
        ("Episode_Reward/grasping_cube", 2.0, "STRICT predicate (6-cond gate)"),
        ("Episode_Reward/lifting_object", 1.5, "× strict_grasp (gated)"),
        ("Episode_Reward/object_goal_tracking", 1.0, "× strict × cube_z>8cm"),
        ("Episode_Reward/object_goal_tracking_fine_grained", 0.5, "fine, gated"),
        ("Episode_Reward/palm_xy_above_cube", 0.8, "lateral alignment (gated)"),
        ("Episode_Reward/hover_height", 0.5, "Gaussian σ=2.5cm @ h=5cm above"),
        ("Episode_Reward/gripper_orientation_penalty", 0.3,
         "vertical orient (sign-fixed → POSITIVE in V218)"),
    ]
    lines.append("Positive rewards (% of max per-step):")
    grasp_strict_val = None
    palm_xy_val = None
    hover_val = None
    lift_gated_val = None
    for tag, max_v, hint in pos_terms:
        val = _latest(events.get(tag, []))
        if val is None:
            continue
        if "grasping_cube" in tag:
            grasp_strict_val = val
        if "palm_xy_above_cube" in tag:
            palm_xy_val = val
        if "hover_height" in tag:
            hover_val = val
        if "lifting_object" in tag:
            lift_gated_val = val
        pct = 100 * val / max_v if max_v else 0
        clean_name = tag.split("/")[-1]
        lines.append(f"  {clean_name:38s} {val:+6.3f} ({pct:5.1f}%) [{_bar(pct)}]")
        lines.append(f"    └ {hint}")
    # success_bonus shown separately (terminal, sparse)
    succ_bonus = _latest(events.get("Episode_Reward/success_bonus", []))
    if succ_bonus is not None:
        lines.append(f"  {'success_bonus (terminal +2000)':38s} {succ_bonus:+8.2f}")
    lines.append("")

    # ---- LIFT BAR (V2.18) — uses lifting_object (gated) ----
    # `lifting_object` = lift_height_gated × strict_grasp, weight 1.5.
    # Per-step max = 1.5 (perfect lift to 20cm WITH strict_grasp every step).
    # Below 0.05/step, gate is rarely True → no meaningful lift activity.
    if lift_gated_val is not None:
        threshold = 0.05  # per-step ep_reward indicating gate occasionally fires
        max_scale = 1.0   # per-step ep_reward indicating sustained good lift
        if lift_gated_val < threshold:
            lines.append("  LIFT ACTIVITY (activates when strict_grasp fires + cube rises):")
            lines.append(f"    lifting_object (gated)        {lift_gated_val:+.4f}    "
                         f"🟦 NOISE ZONE  [{'░' * 20}]")
            lines.append(f"    threshold = {threshold:.2f}/step (gate must trip ≥ ~3% of timesteps)")
        else:
            progress = min(100.0, 100 * (lift_gated_val - threshold) / (max_scale - threshold))
            lines.append("  LIFT ACTIVITY (strict_grasp fires + cube rises — TRACKING ACTIVE):")
            lines.append(f"    lifting_object (gated)        {lift_gated_val:+.4f}    "
                         f"🟩 LIFTING     [{_bar(progress)}]")
            lines.append(f"    scale: ░ at {threshold:.2f}/step, ▓▓▓▓▓▓▓▓▓▓ at {max_scale:.1f}/step")
    lines.append("")

    # ---- Penalty signals ----
    neg_terms = [
        ("Episode_Reward/ee_to_cube_distance", -1.0, "approach (clipped at 0.30 m)"),
        ("Episode_Reward/jaw_below_cube_penalty", -2.0,
         "jaw-vs-table impact (V218 fn), gated NOT-grasped"),
        ("Episode_Reward/cube_dropped_penalty", -50.0, "terminal drop"),
        ("Episode_Reward/action_rate", None, "joint smoothness"),
        ("Episode_Reward/joint_vel", None, "joint speed"),
    ]
    lines.append("Penalty signals (closer to 0 = smoother / less violation):")
    for tag, min_v, hint in neg_terms:
        val = _latest(events.get(tag, []))
        if val is None:
            continue
        if min_v is not None and min_v != 0:
            pct_used = 100 * abs(val) / abs(min_v)
            pct_used = min(100.0, pct_used)
            bar = _bar(pct_used)
            lines.append(f"  {tag.split('/')[-1]:38s} {val:+8.4f} ({pct_used:5.1f}% of cap) [{bar}]")
        else:
            lines.append(f"  {tag.split('/')[-1]:38s} {val:+8.4f}")
        lines.append(f"    └ {hint}")
    lines.append("")

    # ---- EXPLOIT WATCH (V2.18 known risks) ----
    lines.append("EXPLOIT WATCH:")

    # Hover-stall: hover_height monotonically ↗ past iter 300 = policy
    # hovers above cube without grasping (gets +0.5 hover bonus, ignores grasp).
    hover_recent = _last_n(events.get("Episode_Reward/hover_height", []), 20)
    hover_stall_warn = ""
    if hover_val is not None and iter_n and iter_n > 300 and len(hover_recent) >= 10:
        # check monotone increase over last 20 samples
        first_half = sum(hover_recent[:10]) / 10
        second_half = sum(hover_recent[10:]) / 10
        if (second_half > first_half + 0.05) and (lift_gated_val or 0) < 0.05:
            hover_stall_warn = " ⚠ HOVER-STALL: hover ↗ but no lift activity"
    h_str = f"{hover_val:.3f}" if hover_val is not None else "n/a"
    lines.append(f"  hover_height          : {h_str}{hover_stall_warn}")

    # Sparse-gradient-stuck: grasp_strict ≈ 0 AND palm_xy > 0.4 (= aligned
    # laterally but strict 6-cond predicate never trips → too tight).
    sparse_warn = ""
    if (grasp_strict_val is not None and palm_xy_val is not None
            and grasp_strict_val < 0.05 and palm_xy_val > 0.4
            and iter_n and iter_n > 200):
        sparse_warn = " ⚠ SPARSE-STUCK: aligned but strict predicate never fires"
    g_str = f"{grasp_strict_val:.3f}" if grasp_strict_val is not None else "n/a"
    p_str = f"{palm_xy_val:.3f}" if palm_xy_val is not None else "n/a"
    lines.append(f"  grasp_strict / palm_xy: {g_str} / {p_str}{sparse_warn}")
    lines.append("")

    # ---- Policy / training health ----
    noise = _latest(events.get("Policy/mean_noise_std", []))
    ent = _latest(events.get("Loss/entropy", []))
    vf = _latest(events.get("Loss/value_function", []))
    lr = _latest(events.get("Loss/learning_rate", []))

    lines.append("Policy / training health:")
    if noise is not None:
        trend = _trend(events.get("Policy/mean_noise_std", []),
                        n=10, up_thresh=0.01, down_thresh=-0.01)
        if trend == " ↗":
            trend_str = " ↗ diverging ⚠"
        elif trend == " ↘":
            trend_str = " ↘ converging"
        else:
            trend_str = " ─ stable"
        lines.append(f"  noise_std         : {noise:.3f}{trend_str}  "
                     "(smaller = more committed; <0.5 is good)")
    if ent is not None:
        trend = _trend(events.get("Loss/entropy", []),
                        n=10, up_thresh=0.05, down_thresh=-0.05)
        if trend == " ↗":
            trend_str = " ↗ diverging ⚠"
        elif trend == " ↘":
            trend_str = " ↘ converging"
        else:
            trend_str = " ─ stable"
        lines.append(f"  entropy           : {ent:.2f}{trend_str}  "
                     "(smaller = policy committed)")
    if vf is not None:
        # V2.18 budget: |V| ≤ 600 (vs V2.17 |V|≈4500 that crashed at iter 343).
        # Loss/value_function should stay in [0, 5]. Above 5 = VF blowup risk.
        if vf > 10:
            warn = " ⚠⚠ CRITICAL VF BLOWUP (>10, V2.13v2/V2.17 pattern)"
        elif vf > 5:
            warn = " ⚠ HIGH (>5 risks VF blowup)"
        else:
            warn = ""
        lines.append(f"  value_function    : {vf:.3f}{warn}  "
                     "(V2.18 budget: should stay <5)")
    if lr is not None:
        lines.append(f"  learning_rate     : {lr:.2e}  "
                     "(adaptive KL; throttled if too high)")
    lines.append("")

    # ---- Reward total + design-doc cibles ----
    mean_rew = _latest(events.get("Train/mean_reward", []))
    if mean_rew is not None:
        lines.append(f"Mean total reward / episode : {mean_rew:.2f}")
        # V2.18 design-doc cibles (from notes/v218_design_claude_search.md runbook):
        #   iter 50  : grasp_strict ≥ 0.01, lift ≈ 0     (early signal)
        #   iter 100 : grasp_strict ≈ 0.05, lift ≥ 0.005 (gate firing occasionally)
        #   iter 200 : grasp_strict ≈ 0.15, lift ≥ 0.05  (gate fires regularly)
        #   iter 300 : grasp_strict ≈ 0.20, lift ≥ 0.10  (grasping behavior emerging)
        #   iter 500 : grasp_strict ≈ 0.40, lift ≥ 0.30  (lifting consistent)
        #   iter 1000: grasp_strict ≈ 0.70, lift ≥ 0.50  (goal_tracking emerges)
        #   iter 1500: grasp_strict ≈ 0.85, success_pct > 50%
        if iter_n is not None:
            if iter_n < 75:
                target = "iter 50  cible : grasp_strict ≥ 0.01"
            elif iter_n < 150:
                target = "iter 100 cible : grasp_strict ≈ 0.05, lift ≥ 0.005"
            elif iter_n < 250:
                target = "iter 200 cible : grasp_strict ≈ 0.15, lift ≥ 0.05"
            elif iter_n < 400:
                target = "iter 300 cible : grasp_strict ≈ 0.20, lift ≥ 0.10"
            elif iter_n < 750:
                target = "iter 500 cible : grasp_strict ≈ 0.40, lift ≥ 0.30"
            elif iter_n < 1250:
                target = "iter 1000 cible: grasp_strict ≈ 0.70, lift ≥ 0.50"
            else:
                target = "iter 1500 cible: grasp_strict ≈ 0.85, SUCCESS > 50%"
            lines.append(f"Design-doc {target}")
    lines.append("=" * 72)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default=None,
                        help="experiment name (default: latest by mtime)")
    parser.add_argument("--run", default=None,
                        help="explicit run dir name within experiment "
                             "(e.g. '2026-05-12_14-30-00'). Defaults to "
                             "latest by mtime.")
    parser.add_argument("--interval", type=int, default=30,
                        help="seconds between prints")
    args = parser.parse_args()

    print("[INFO] Watching V2.18 training logs. Ctrl+C to stop.")
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
