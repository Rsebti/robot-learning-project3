#!/usr/bin/env python3
"""
HIL-SERL walkthrough — generates a self-contained HTML page that explains
all components of Human-in-the-Loop SERL, the training process, the 50/50
replay split, where RL enters, and how the human-in-the-loop mechanism works.

Source material: notes/so101_robot_learning_playbook.md §T2 + §3 + §5 (HIL-SERL),
plus the HIL-SERL paper (Luo et al. 2024, arXiv:2410.21845).

Run:
    /Users/admin/miniforge3/bin/python generate_hilserl_walkthrough.py

Output:
    hilserl_walkthrough.html (open in any browser, fully self-contained)
"""

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---- consistent palette (matches SmolVLA walkthrough where applicable) ----
COLOR_INPUT = "#FFD7B5"
COLOR_ROBOT = "#FFE4A8"
COLOR_ACTOR = "#C5DCFF"
COLOR_CRITIC = "#D4C5F9"
COLOR_BUFFER_DEMO = "#C8F0CD"
COLOR_BUFFER_ONLINE = "#F9D5C8"
COLOR_BUFFER_INTV = "#FFC9E0"
COLOR_REWARD = "#FFF1A8"
COLOR_HUMAN = "#FFA5A5"
COLOR_ARROW = "#3A3A3A"
COLOR_RL = "#2E86DE"
COLOR_BC = "#E74C3C"
COLOR_HIL = "#27AE60"
FONT = {"family": "DejaVu Sans"}


def b64_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    s = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return s


def add_box(ax, xy, w, h, label, fc, fontsize=10, fontweight="normal", ec="black", lw=1.4):
    box = FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
        fc=fc, ec=ec, lw=lw,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + w / 2, xy[1] + h / 2, label,
        ha="center", va="center",
        fontsize=fontsize, fontweight=fontweight, **FONT,
    )
    return box


def add_arrow(ax, start, end, color=COLOR_ARROW, lw=1.8, style="->", connectionstyle="arc3,rad=0"):
    ar = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=18,
        color=color, lw=lw, shrinkA=4, shrinkB=4,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(ar)
    return ar


# ---- Figure 1: HIL-SERL system architecture (all components) -------------
def fig_system_arch():
    fig, ax = plt.subplots(figsize=(14, 8.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.5)
    ax.axis("off")
    ax.set_title(
        "HIL-SERL — Full system architecture",
        fontsize=14, fontweight="bold", pad=15,
    )

    # === REAL WORLD layer (top) ===
    add_box(ax, (0.5, 7.2), 3.0, 0.9, "Real SO-101\n(follower arm)", COLOR_ROBOT, fontsize=11, fontweight="bold")
    add_box(ax, (5.0, 7.2), 2.8, 0.9, "Wrist camera\n+ joint encoders", COLOR_INPUT, fontsize=10)
    add_box(ax, (10.5, 7.2), 3.0, 0.9, "Human operator\n(holding gamepad)", COLOR_HUMAN, fontsize=11, fontweight="bold")

    # === OBSERVATION / ACTION layer ===
    add_box(ax, (5.5, 5.5), 3.0, 0.8, "obs = (img 128×128, q_joints)", "#EEEEEE", fontsize=10)
    # arrow from camera/robot to observation
    add_arrow(ax, (6.4, 7.2), (6.6, 6.3))
    add_arrow(ax, (4.0, 7.6), (5.5, 6.0), connectionstyle="arc3,rad=0.2")

    # === ACTOR / POLICY ===
    add_box(ax, (0.5, 4.0), 3.2, 1.0, "Actor π(a|s)\n6-D Cartesian twist\n(SAC policy)", COLOR_ACTOR, fontsize=10, fontweight="bold")
    add_box(ax, (0.5, 2.7), 3.2, 0.8, "Gripper DQN\n(discrete open/close)", COLOR_ACTOR, fontsize=9)

    # arrow from obs to actor
    add_arrow(ax, (5.5, 5.7), (3.7, 4.7))
    add_arrow(ax, (5.5, 5.7), (3.7, 3.2))

    # === HUMAN INTERVENTION ===
    add_box(ax, (10.5, 4.0), 3.0, 1.0,
            "Intervention\nmonitor", COLOR_HUMAN, fontsize=10, fontweight="bold")
    # arrow from human to intervention
    add_arrow(ax, (12.0, 7.2), (12.0, 5.0))

    # === ACTION (decision merger) ===
    add_box(ax, (5.5, 4.0), 3.0, 1.0,
            "action chosen:\nautonomous π(a) OR\nhuman override",
            "#FFFFD0", fontsize=10)
    # actor → action
    add_arrow(ax, (3.7, 4.5), (5.5, 4.5))
    # human intervention → action
    add_arrow(ax, (10.5, 4.5), (8.5, 4.5))
    # action → robot (back up)
    add_arrow(ax, (7.0, 5.0), (2.5, 7.2), color=COLOR_ARROW,
              connectionstyle="arc3,rad=-0.3", lw=1.6)
    ax.text(2.0, 6.2, "act on robot",
            fontsize=8.5, color="#444", fontstyle="italic", rotation=60)

    # === REWARD CLASSIFIER ===
    add_box(ax, (10.5, 2.7), 3.0, 0.8, "Reward classifier\n(ResNet-10 + MLP)\nbinary success",
            COLOR_REWARD, fontsize=9)
    add_arrow(ax, (8.5, 5.5), (10.5, 3.1), connectionstyle="arc3,rad=-0.3")

    # === THREE BUFFERS ===
    add_box(ax, (0.3, 0.7), 2.5, 1.4, "Demo Buffer\n~20–30 trajectories\n(BC teleop)",
            COLOR_BUFFER_DEMO, fontsize=9, fontweight="bold")
    add_box(ax, (3.0, 0.7), 2.5, 1.4, "Intervention Buffer\n(human-corrected\ntransitions)",
            COLOR_BUFFER_INTV, fontsize=9, fontweight="bold")
    add_box(ax, (5.7, 0.7), 2.5, 1.4, "Online RL Buffer\n(autonomous\ntransitions)",
            COLOR_BUFFER_ONLINE, fontsize=9, fontweight="bold")

    # arrows from action+reward → appropriate buffers
    # autonomous → online
    add_arrow(ax, (7.0, 4.0), (7.0, 2.1), color="#777", lw=1.4)
    ax.text(7.15, 3.0, "autonomous", fontsize=8, color="#444", rotation=90, va="center")
    # intervention → both intervention AND demo
    add_arrow(ax, (10.5, 4.0), (4.3, 2.1), color=COLOR_HUMAN, lw=1.6,
              connectionstyle="arc3,rad=0.3")
    add_arrow(ax, (10.5, 4.0), (1.5, 2.1), color=COLOR_HUMAN, lw=1.6,
              connectionstyle="arc3,rad=0.3")
    ax.text(8.5, 2.8, "human override →\nboth buffers",
            fontsize=8, color=COLOR_HUMAN, fontweight="bold", style="italic")

    # === CRITIC ENSEMBLE ===
    add_box(ax, (9.0, 0.7), 4.5, 1.4,
            "Critic ensemble: 10 × Q(s,a)\n(LayerNorm, REDQ subset of 2,\nUTD=10–20)",
            COLOR_CRITIC, fontsize=10, fontweight="bold")

    # 50/50 sampling arrow from buffers to critic + actor
    add_arrow(ax, (4.3, 1.4), (9.0, 1.4), color=COLOR_RL, lw=2.4)
    ax.text(6.5, 1.6, "50/50 minibatch (batch 256)",
            fontsize=9, color=COLOR_RL, fontweight="bold", ha="center")

    # critic ensemble → actor update (TD loss)
    add_arrow(ax, (11.2, 2.1), (2.0, 4.0), color=COLOR_RL, lw=1.8,
              connectionstyle="arc3,rad=0.3")
    ax.text(7.0, 3.3, "TD loss → actor + critic update (SAC)",
            fontsize=8.5, color=COLOR_RL, fontstyle="italic", ha="center")

    # Legend (bottom right)
    legend_y = 0.05
    ax.text(0.3, legend_y, "Green = offline demos · Pink = human-intervention · Orange = autonomous online · Blue arrows = RL update",
            fontsize=8.5, color="#555", fontstyle="italic")

    return fig


# ---- Figure 2: The 50/50 minibatch split ---------------------------------
def fig_replay_split():
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    ax.set_title("The 50/50 minibatch — RLPD's core trick",
                 fontsize=14, fontweight="bold", pad=15)

    # Two buffer columns
    add_box(ax, (0.5, 3.8), 4.5, 2.0,
            "OFFLINE buffer\n(Demo + Intervention)\n\n~20–30 BC trajectories\n+ accumulating\nhuman corrections",
            COLOR_BUFFER_DEMO, fontsize=10.5, fontweight="bold")
    add_box(ax, (8.0, 3.8), 4.5, 2.0,
            "ONLINE buffer\n(autonomous rollouts)\n\nfilled by π acting\non the real robot\nduring training",
            COLOR_BUFFER_ONLINE, fontsize=10.5, fontweight="bold")

    # Sampling: each → 128 samples
    add_box(ax, (1.2, 2.4), 3.1, 0.8, "Sample 128 transitions", "#EEE", fontsize=10)
    add_box(ax, (8.7, 2.4), 3.1, 0.8, "Sample 128 transitions", "#EEE", fontsize=10)
    add_arrow(ax, (2.75, 3.8), (2.75, 3.2))
    add_arrow(ax, (10.25, 3.8), (10.25, 3.2))

    # Merge to minibatch
    add_box(ax, (4.5, 0.8), 4.0, 1.2,
            "Minibatch (256)\n128 offline + 128 online",
            COLOR_RL, fontsize=11, fontweight="bold")
    add_arrow(ax, (2.75, 2.4), (5.3, 2.0))
    add_arrow(ax, (10.25, 2.4), (7.7, 2.0))

    # SAC update arrow
    ax.annotate(
        "→ SAC actor + critic update\n   (10 Q-functions, subset of 2)",
        xy=(8.5, 1.4), xytext=(11.5, 1.4),
        fontsize=10, color=COLOR_RL, fontweight="bold",
        arrowprops=dict(arrowstyle="<-", color=COLOR_RL, lw=1.5),
        ha="left", va="center",
    )

    # Annotation: why 50/50 matters
    ax.text(6.5, 6.1,
            "Every gradient step sees demos AND online experience equally.\n"
            "This prevents catastrophic forgetting of the BC prior AND prevents over-fitting to easy demos.",
            ha="center", fontsize=10, fontstyle="italic", color="#333")

    return fig


# ---- Figure 3: Where RL comes in (the SAC update) ------------------------
def fig_where_rl():
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    ax.set_title("Where does \"RL\" actually happen? — the SAC update",
                 fontsize=14, fontweight="bold", pad=15)

    # Step 1: Sample minibatch
    add_box(ax, (0.3, 4.7), 2.5, 1.4,
            "(s, a, r, s')\nminibatch\n(50/50 split)",
            "#EEE", fontsize=10, fontweight="bold")

    # Step 2: Critic update (Q learning)
    add_box(ax, (3.5, 4.7), 4.5, 1.4,
            "Critic update\n\nminimize: ‖Q(s,a) − (r + γ · min_{i∈subset}\n           Q_target,i(s', π(s')))‖²",
            COLOR_CRITIC, fontsize=9, fontweight="bold")
    add_arrow(ax, (2.8, 5.4), (3.5, 5.4))

    # Step 3: Actor update (SAC policy improvement)
    add_box(ax, (8.7, 4.7), 4.0, 1.4,
            "Actor update\n\nmaximize: E[ min_i Q_i(s, π(s))\n            − α · log π(a|s) ]",
            COLOR_ACTOR, fontsize=9, fontweight="bold")
    add_arrow(ax, (8.0, 5.4), (8.7, 5.4))

    # Step 4: Target network update
    add_box(ax, (4.5, 2.8), 4.0, 1.0,
            "Target networks update\nQ_target ← τ·Q + (1−τ)·Q_target\n(τ = 0.005, slow tracking)",
            "#E8E8FF", fontsize=9)
    add_arrow(ax, (5.5, 4.7), (5.5, 3.8))
    add_arrow(ax, (10.7, 4.7), (8.5, 3.8))

    # Step 5: Loop back
    add_arrow(ax, (6.5, 2.8), (1.5, 4.7), color=COLOR_RL, lw=1.6,
              connectionstyle="arc3,rad=-0.3")
    ax.text(2.5, 3.4, "next minibatch →", fontsize=9, color=COLOR_RL, fontstyle="italic")

    # Annotations on the left
    ax.text(6.5, 1.6,
            "The reward r in the TD target comes from the trained reward classifier,\n"
            "NOT a hand-engineered function. r ∈ {0, 1}.",
            ha="center", fontsize=10, color=COLOR_BC, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.4", fc="#FFEEEA", ec=COLOR_BC))

    # Note about ensemble
    ax.text(6.5, 0.6,
            "min_{i∈subset of 2} over 10 Q-functions = REDQ pessimism.\n"
            "Prevents Q-value over-estimation on out-of-distribution actions.",
            ha="center", fontsize=9, color="#333", fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.4", fc="#F8F8F8", ec="#888"))

    return fig


# ---- Figure 4: Human-in-the-loop control flow ----------------------------
def fig_hil_flow():
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Per-step control flow — how human interventions work",
                 fontsize=14, fontweight="bold", pad=15)

    # Step boxes
    steps = [
        ((0.3, 4.5), 2.2, "Get obs from\nrobot & camera"),
        ((2.9, 4.5), 2.4, "Actor proposes\na = π(obs)"),
        ((5.7, 4.5), 2.6, "Human watching?\nWill cube be missed?"),
    ]
    for (x, y), w, label in steps:
        add_box(ax, (x, y), w, 1.0, label, "#EEEEEE", fontsize=10)

    # Decision diamond (using triangle + text)
    add_box(ax, (8.7, 4.5), 1.8, 1.0, "Intervene\nor not?", "#FFFFD0", fontsize=10, fontweight="bold")

    # Two branches
    # Branch A: no intervention
    add_box(ax, (1.5, 2.4), 4.5, 1.0, "Send a → robot, record transition", COLOR_BUFFER_ONLINE, fontsize=10)
    add_box(ax, (1.5, 0.7), 4.5, 1.0, "Add to ONLINE RL buffer", COLOR_BUFFER_ONLINE, fontsize=10, fontweight="bold")
    # Branch B: intervention
    add_box(ax, (7.5, 2.4), 5.0, 1.0, "Send human's action → robot, record transition", COLOR_BUFFER_INTV, fontsize=10)
    add_box(ax, (7.5, 0.7), 5.0, 1.0, "Add to BOTH demo + intervention buffers",
            COLOR_BUFFER_INTV, fontsize=10, fontweight="bold")

    # Arrows
    add_arrow(ax, (2.5, 5.5), (2.9, 5.0))
    add_arrow(ax, (5.3, 5.0), (5.7, 5.0))
    add_arrow(ax, (8.3, 5.0), (8.7, 5.0))
    # No branch
    add_arrow(ax, (9.0, 4.5), (4.0, 3.4), color="#777", lw=1.5)
    ax.text(6.5, 3.9, "no override", fontsize=9, color="#666", rotation=-15)
    add_arrow(ax, (3.7, 2.4), (3.7, 1.7))
    # Intervention branch
    add_arrow(ax, (10.0, 4.5), (10.0, 3.4), color=COLOR_HUMAN, lw=2)
    ax.text(10.1, 3.9, "override", fontsize=9, color=COLOR_HUMAN, fontweight="bold")
    add_arrow(ax, (10.0, 2.4), (10.0, 1.7), color=COLOR_HUMAN, lw=2)

    # Key insight
    ax.text(6.5, 0.05,
            "Human interventions are double-counted on purpose: they enrich both the offline imitation prior\n"
            "and the RL replay. The policy learns from both the rescue and the lesson.",
            ha="center", fontsize=9.5, fontstyle="italic", color=COLOR_HIL, fontweight="bold")

    return fig


# ---- Figure 5: Reward classifier ----------------------------------------
def fig_reward_classifier():
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    ax.set_title("The reward classifier — sidesteps manual reward shaping",
                 fontsize=14, fontweight="bold", pad=15)

    # Training data (offline phase)
    add_box(ax, (0.3, 3.7), 3.2, 1.3,
            "Labeled frames\n200 success ✓\n1000 failure ✗\n(human-labeled)",
            COLOR_HUMAN, fontsize=10, fontweight="bold")

    # The classifier
    add_box(ax, (4.0, 3.7), 4.0, 1.3,
            "ResNet-10 (ImageNet)\nfrozen backbone\n+ small MLP head",
            COLOR_REWARD, fontsize=10, fontweight="bold")
    add_arrow(ax, (3.5, 4.3), (4.0, 4.3))

    # Output: binary reward
    add_box(ax, (8.5, 3.7), 4.0, 1.3,
            "Output\nbinary r ∈ {0, 1}\n>95% accuracy",
            "#FFFFD0", fontsize=10, fontweight="bold")
    add_arrow(ax, (8.0, 4.3), (8.5, 4.3))

    # Bottom: the inference-time flow
    ax.text(6.5, 3.0, "Training phase (~30 min, before RL starts)",
            ha="center", fontsize=10, fontweight="bold", color="#555", style="italic")

    # Divider line
    ax.plot([0.3, 12.5], [2.5, 2.5], color="#AAA", linestyle="--", lw=1)

    ax.text(6.5, 2.05, "Inference phase (during RL training, every step)",
            ha="center", fontsize=10, fontweight="bold", color="#555", style="italic")

    # Inference flow
    add_box(ax, (0.5, 0.5), 3.0, 1.2, "Current obs\n(image)", COLOR_INPUT, fontsize=10)
    add_box(ax, (4.5, 0.5), 3.5, 1.2, "Reward classifier\n(frozen, inference only)",
            COLOR_REWARD, fontsize=10)
    add_box(ax, (9.0, 0.5), 3.0, 1.2, "r → RL replay buffer\n(fills (s, a, r, s'))",
            COLOR_BUFFER_ONLINE, fontsize=10)
    add_arrow(ax, (3.5, 1.1), (4.5, 1.1))
    add_arrow(ax, (8.0, 1.1), (9.0, 1.1))

    return fig


# ---- Figure 6: Training timeline -----------------------------------------
def fig_training_timeline():
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    ax.set_title("HIL-SERL training — phase-by-phase wall-clock",
                 fontsize=14, fontweight="bold", pad=15)

    # Timeline bar
    ax.plot([0.3, 13.7], [4.0, 4.0], color="#888", lw=2, solid_capstyle="round")
    # Tick marks
    ticks = [0.3, 2.0, 4.0, 13.5]
    tick_labels = ["t = 0", "+30 min", "+35 min", "+~2.5 h"]
    for x, lab in zip(ticks, tick_labels):
        ax.plot([x, x], [3.85, 4.15], color="#888", lw=2)
        ax.text(x, 4.35, lab, ha="center", fontsize=9, color="#555")

    # Phase 1
    add_box(ax, (0.3, 1.5), 1.7, 1.8,
            "Phase 1\nLabel ~1200 frames\n+ train reward\nclassifier\n(~30 min)",
            COLOR_HUMAN, fontsize=9, fontweight="bold")
    # Phase 2
    add_box(ax, (2.0, 1.5), 2.0, 1.8,
            "Phase 2\nPre-fill demo\nbuffer with\n20–30 BC episodes\n(~5 min)",
            COLOR_BUFFER_DEMO, fontsize=9, fontweight="bold")
    # Phase 3 (the big online RL phase)
    add_box(ax, (4.0, 1.5), 9.5, 1.8,
            "Phase 3 — Online RL loop on real robot\n"
            "actor + critic update from 50/50 minibatches\n"
            "human watches; intervenes when policy is about to fail\n"
            "(1–2.5 h wall-clock per task on Franka; 3+ weeks debugging reality on SO-101)",
            COLOR_RL, fontsize=10, fontweight="bold")

    # Bottom: per-iteration zoom
    ax.text(8.7, 0.85,
            "Each online iteration: obs → action → step robot → reward classifier → store transition → "
            "sample minibatch → SAC update → repeat at 10 Hz",
            ha="center", fontsize=9, fontstyle="italic", color="#444",
            bbox=dict(boxstyle="round,pad=0.4", fc="#F8F8F8", ec="#888"))

    return fig


# ---- Figure 7: Performance and where it fits -----------------------------
def fig_performance_and_fit():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: HIL-SERL vs alternatives — published success on real low-cost arms
    ax = axes[0]
    methods = ["Plain BC\n(MLP, 200 ep,\nHIL-SERL Table 1c)",
               "ACT\n(50 demos,\nSO-101)",
               "RLPD\n(no BC pretrain,\nno human)",
               "HIL-SERL\npaper, Franka\n(13 tasks avg)",
               "HIL-SERL\nSO-101 (ggando,\ngrasp only)"]
    success = [47, 70, 75, 100, 70]
    colors = [COLOR_BC, COLOR_BC, COLOR_RL, COLOR_HIL, COLOR_HIL]
    bars = ax.bar(methods, success, color=colors, edgecolor="black", lw=1.2)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Success rate (%)", fontsize=11)
    ax.set_title("Method comparison on real-arm pick-place (published)",
                 fontsize=11, fontweight="bold")
    for bar, v in zip(bars, success):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 2, f"{v}%",
                ha="center", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_axisbelow(True)

    # Right: where HIL-SERL sits in the Task 2 plan
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Where HIL-SERL fits in our Eval 2 plan",
                 fontsize=11, fontweight="bold")

    # Three boxes vertically
    add_box(ax, (1.0, 7.5), 8.0, 1.4,
            "Step 1: color-conditioned BC primitive\n(SmolVLA finetune or FiLM-ACT)",
            COLOR_BC, fontsize=10, fontweight="bold")
    add_box(ax, (1.0, 5.0), 8.0, 1.7,
            "Step 2: HIL-SERL refinement\n· demos = our existing 101 episodes\n"
            "· actor init = the BC primitive from Step 1\n· human-in-loop at the SO-101",
            COLOR_HIL, fontsize=10, fontweight="bold")
    add_box(ax, (1.0, 2.5), 8.0, 1.4,
            "Step 3: robustness sweep\nrun the 5-rollout matrix, retrain on failure modes",
            "#EEE", fontsize=10)
    # arrows
    add_arrow(ax, (5.0, 7.5), (5.0, 6.7))
    add_arrow(ax, (5.0, 5.0), (5.0, 3.9))

    ax.text(5.0, 1.5, "HIL-SERL is the \"RL mandatory\" step.\n"
                       "BC primitive + this = spec-compliant submission.",
            ha="center", fontsize=9.5, fontstyle="italic", color=COLOR_HIL,
            bbox=dict(boxstyle="round,pad=0.4", fc="#E9F8EE", ec=COLOR_HIL))

    plt.tight_layout()
    return fig


# ---- HTML template -------------------------------------------------------
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>HIL-SERL Walkthrough — Robot Learning Project 3</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      max-width: 1100px;
      margin: 0 auto;
      padding: 2rem 2.5rem 4rem 2.5rem;
      color: #1a1a1a;
      line-height: 1.55;
      background: #fafafa;
    }}
    h1 {{
      border-bottom: 3px solid #27AE60;
      padding-bottom: 0.4rem;
      margin-bottom: 0.2rem;
    }}
    h2 {{
      color: #27AE60;
      border-bottom: 1px solid #ddd;
      padding-bottom: 0.2rem;
      margin-top: 2.5rem;
    }}
    h3 {{ color: #444; }}
    p {{ font-size: 15px; }}
    .subtitle {{
      color: #666;
      font-style: italic;
      margin-top: 0;
      margin-bottom: 2rem;
    }}
    figure {{
      margin: 2rem 0;
      text-align: center;
      background: white;
      padding: 1.2rem;
      border-radius: 10px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }}
    figure img {{
      max-width: 100%;
      height: auto;
    }}
    figcaption {{
      font-size: 13px;
      color: #555;
      margin-top: 0.6rem;
      font-style: italic;
    }}
    blockquote {{
      border-left: 4px solid #27AE60;
      background: #ecf9f0;
      margin: 1.2rem 0;
      padding: 0.8rem 1rem;
      font-size: 14px;
      border-radius: 4px;
    }}
    code {{
      background: #f0eee6;
      padding: 0.1rem 0.4rem;
      border-radius: 3px;
      font-size: 0.92em;
    }}
    .keytake {{
      background: #f7f3ff;
      border-left: 4px solid #6b3fa0;
      padding: 0.8rem 1rem;
      margin: 1.5rem 0;
      border-radius: 4px;
    }}
    .keytake strong {{ color: #4b2c70; }}
    .warn {{
      background: #FFF3E0;
      border-left: 4px solid #E67E22;
      padding: 0.8rem 1rem;
      margin: 1.5rem 0;
      border-radius: 4px;
    }}
    .meta {{
      font-size: 12px;
      color: #777;
      text-align: right;
      margin-top: 3rem;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 1rem 0;
      font-size: 14px;
    }}
    th, td {{ border: 1px solid #ddd; padding: 0.4rem 0.6rem; text-align: left; vertical-align: top; }}
    th {{ background: #f0f0f0; }}
  </style>
</head>
<body>

<h1>HIL-SERL — a visual walkthrough</h1>
<p class="subtitle">Why HIL-SERL is the playbook's #1 RL method for SO-101 pick-and-place, what every moving part does, and exactly where the RL update happens.</p>

<h2>1. What HIL-SERL is, in one sentence</h2>
<p>
  HIL-SERL = <strong>RLPD</strong> (a sample-efficient off-policy RL algorithm) <strong>+ a human watching with a gamepad</strong>.
  The human's job is to intervene when the policy is about to make a costly mistake; those interventions
  become extra training data. The "RL" part is standard SAC — soft actor-critic — but it never has to explore
  catastrophically because the human is the safety net.
</p>
<p>
  Why we care: it's the only RL method with a documented reproduction on SO-100/SO-101 hardware (ggando,
  ~70% on grasp-only after three weeks of debugging) and the only RL method with a first-class LeRobot port
  (<code>huggingface.co/docs/lerobot/hilserl</code>). The HIL-SERL paper itself (Luo et al. 2024,
  arXiv:2410.21845) reports <strong>100% success on all 13 tasks in 1–2.5 h wall-clock</strong> on a Franka arm.
</p>

<h2>2. All the components and how they wire together</h2>
<figure>
  <img alt="HIL-SERL system architecture" src="data:image/png;base64,{img_arch}" />
  <figcaption>Figure 1 — every component and every data flow. Pink = human-in-the-loop; green = offline demos;
  orange = autonomous online experience; blue = RL update.</figcaption>
</figure>

<h3>Component-by-component</h3>
<table>
  <tr><th>Component</th><th>What it is</th><th>Concrete spec</th></tr>
  <tr>
    <td><strong>Real robot</strong></td>
    <td>The SO-101 follower. Receives commanded actions, returns observations.</td>
    <td>Wrist cam 128×128 RGB, joint encoders, 10 Hz control loop, impedance ctrl at 1 kHz with Δ-clip.</td>
  </tr>
  <tr>
    <td><strong>Actor π(a|s)</strong></td>
    <td>The policy. Maps observation → 6-D Cartesian twist action (continuous).</td>
    <td>ResNet-10 (frozen, ImageNet) + 2-layer MLP, 256-wide. Initialized from your BC primitive (e.g. SmolVLA).</td>
  </tr>
  <tr>
    <td><strong>Gripper DQN</strong></td>
    <td>Separate policy for the discrete open/close action. Trained as its own MDP.</td>
    <td>Small MLP, ε-greedy exploration.</td>
  </tr>
  <tr>
    <td><strong>Critic ensemble</strong></td>
    <td>10 Q-functions Q(s,a). REDQ subset of 2 used per update for pessimism.</td>
    <td>Each: 2-layer MLP 256-wide, LayerNorm on every hidden layer. UTD=10–20.</td>
  </tr>
  <tr>
    <td><strong>Reward classifier</strong></td>
    <td>Trained binary success/failure detector. Sidesteps manual reward shaping.</td>
    <td>ResNet-10 + MLP head, ~200 positive + ~1000 negative human-labeled frames, &gt;95% acc.</td>
  </tr>
  <tr>
    <td><strong>Demo buffer</strong></td>
    <td>Pre-filled from teleop demonstrations. Never overwritten.</td>
    <td>~20–30 trajectories. For us: a subset of our 101 episodes.</td>
  </tr>
  <tr>
    <td><strong>Intervention buffer</strong></td>
    <td>Stores human-corrected transitions. Routed to <em>both</em> demo and online buffers.</td>
    <td>Grows during training. Per-transition (s, a_human, r, s').</td>
  </tr>
  <tr>
    <td><strong>Online RL buffer</strong></td>
    <td>Autonomous transitions from the actor acting in the environment.</td>
    <td>Capped-size circular buffer. Per-transition (s, a_π, r, s').</td>
  </tr>
  <tr>
    <td><strong>Human + gamepad</strong></td>
    <td>You. Watching the robot. Pressing buttons when you see disaster coming.</td>
    <td>Continuous monitoring, ~1–2 h wall-clock per task per the paper.</td>
  </tr>
</table>

<h2>3. The 50/50 split — the heart of RLPD</h2>
<p>
  Standard off-policy RL has one replay buffer. RLPD (the algorithm under HIL-SERL) has two — and every
  gradient update sees them in equal measure. This is the single most important design choice in the system.
</p>
<figure>
  <img alt="50/50 minibatch split" src="data:image/png;base64,{img_split}" />
  <figcaption>Figure 2 — Every minibatch (256) is exactly 128 offline + 128 online. The model never gets to
  "forget" the demos.</figcaption>
</figure>

<div class="keytake">
  <strong>Why 50/50 and not, say, 90/10 demos?</strong> If you over-weight demos, the policy never breaks
  free of imitating exactly what the human did, and you lose the RL benefits. If you under-weight demos,
  early online exploration is catastrophic because the actor doesn't know what "good behavior" looks like.
  50/50 is the empirically validated balance from the RLPD paper (arXiv:2302.02948).
</div>

<h2>4. Where the "RL" actually happens — the SAC update</h2>
<p>
  Everything above is plumbing. The <em>learning</em> happens in two equations, one for the critic and one
  for the actor, applied per minibatch.
</p>
<figure>
  <img alt="SAC update equations" src="data:image/png;base64,{img_rl}" />
  <figcaption>Figure 3 — The TD update (critic) and the policy improvement (actor). Both run per minibatch,
  multiple times per environment step (UTD = update-to-data ratio of 10–20).</figcaption>
</figure>

<p>
  In plain English:
</p>
<ul>
  <li><strong>Critic update:</strong> "Adjust Q so it correctly predicts the reward you actually got plus the
      discounted Q of where you ended up." Standard temporal-difference learning. The <code>min</code> over
      a random subset of the 10 Q-functions is the REDQ pessimism trick — it prevents the policy from
      exploiting over-optimistic Q-values on actions it hasn't tried yet.</li>
  <li><strong>Actor update:</strong> "Choose actions that maximize Q, but with an entropy bonus so you keep
      exploring." Standard SAC. The <code>α</code> entropy coefficient is auto-tuned (SAC auto-α).</li>
  <li><strong>Target update:</strong> Slow-tracking copies of the Q-functions (τ=0.005) used to compute the TD
      target. Stops the bootstrap from chasing its own tail.</li>
</ul>

<p>
  The <strong>reward r</strong> in those equations comes from the trained classifier, <em>not</em> a
  hand-designed dense reward. It's binary: 0 or 1. That's all the policy needs to know.
</p>

<h2>5. How the human-in-the-loop part actually works</h2>
<p>
  At every step the human watches the rollout. If the policy is about to do something dumb — overshoot the
  grasp, knock the cube off the table, miss the bowl — the human grabs the gamepad and overrides the action.
  That single decision routes the transition into the right buffer(s).
</p>
<figure>
  <img alt="Human-in-the-loop control flow" src="data:image/png;base64,{img_hil}" />
  <figcaption>Figure 4 — Per-step decision tree. Autonomous transitions go to the online buffer only.
  Intervention transitions go to <em>both</em> the demo buffer AND the intervention buffer.</figcaption>
</figure>

<div class="keytake">
  <strong>The double-write is the key mechanism.</strong> Intervention transitions are valuable for two
  different reasons: as imitation targets (they show the actor what the right action <em>was</em>) and as
  reward signal (they're high-quality (s, a, r, s') tuples that the critic learns from). Routing them to
  both buffers means both networks see them on every update.
</div>

<h2>6. The reward classifier — how we get a learnable reward signal</h2>
<p>
  Manual reward shaping for robot manipulation is notoriously difficult (we know from our PPO-from-scratch
  archive — see CLAUDE.md). HIL-SERL replaces this with a learned classifier:
</p>
<figure>
  <img alt="Reward classifier" src="data:image/png;base64,{img_reward}" />
  <figcaption>Figure 5 — Two-phase reward model. Train it once offline from labeled success/failure frames,
  then use it as a frozen black box during RL training.</figcaption>
</figure>
<p>
  Labeling is cheap: you watch your own teleop episodes and tag each frame as "success state" or "failure
  state." ~200 positives + ~1000 negatives gets you >95% accuracy. The frozen ResNet-10 backbone (pretrained
  on ImageNet) means you're effectively learning a small MLP on top of strong visual features — a few hundred
  examples is enough.
</p>

<h2>7. The training process, end-to-end</h2>
<figure>
  <img alt="Training timeline" src="data:image/png;base64,{img_timeline}" />
  <figcaption>Figure 6 — Wall-clock view. The paper's 1–2.5 h is real on a Franka with a well-tuned setup.
  Add 3+ weeks of debugging on SO-101 specifically, per ggando's reproduction.</figcaption>
</figure>

<ol>
  <li><strong>Phase 1 (offline, ~30 min):</strong> Label ~1200 frames from teleop episodes as success/failure.
      Train the reward classifier. Validate >95% accuracy on a held-out set.</li>
  <li><strong>Phase 2 (offline, ~5 min):</strong> Pre-fill the demo buffer with 20–30 teleop trajectories.
      Initialize the actor from your BC primitive (e.g., the SmolVLA finetune from Week 1).</li>
  <li><strong>Phase 3 (online, 1–2.5 h on Franka):</strong> The RL loop runs at 10 Hz. The actor proposes
      actions. The reward classifier scores frames. You watch and intervene. The critic and actor update
      from 50/50 minibatches every step. UTD=10–20 means the networks update 10–20 times per environment
      step — this is what makes HIL-SERL sample-efficient enough for real hardware.</li>
</ol>

<h2>8. What to expect on SO-101 specifically</h2>
<figure>
  <img alt="Performance comparison and where HIL-SERL fits" src="data:image/png;base64,{img_perf}" />
  <figcaption>Figure 7 — Left: published numbers from low-cost arm reproductions. Right: how HIL-SERL fits
  into our Week 1–6 Eval 2 plan.</figcaption>
</figure>

<div class="warn">
  <strong>⚠ Reality check from the playbook:</strong>
  <ul>
    <li><strong>LeRobot issue #1387:</strong> "so101 can't work with HIL-SERL" — only
        <code>so100_follower_end_effector</code> works out of the box. Expect to need a port or fix.</li>
    <li><strong>ggando reproduction:</strong> reached ~70% on grasp-only after 3 weeks of fixes:
        MuJoCo-FK replacement, state caching bug, lighting sensitivity.</li>
    <li><strong>Indraneel Patil:</strong> <em>"performance is similar to IL with the same wall-clock"</em> —
        i.e., HIL-SERL doesn't beat well-tuned BC by huge margins on SO-101.</li>
  </ul>
  Treat HIL-SERL as a robustness booster on top of a working BC primitive, not as a magic substitute for
  demonstration data.
</div>

<h2>9. Why this is the right RL choice for Eval 2 (the playbook's argument)</h2>
<table>
  <tr><th>Property</th><th>What it means for us</th></tr>
  <tr>
    <td>Reuses our 101 teleop episodes</td>
    <td>Demo buffer is pre-filled from our existing dataset. Nothing wasted.</td>
  </tr>
  <tr>
    <td>Builds on a BC primitive</td>
    <td>The SmolVLA (or FiLM-ACT) policy from Week 1 is the actor initialization. Cumulative wins.</td>
  </tr>
  <tr>
    <td>Real-robot training (no sim-to-real gap)</td>
    <td>No Isaac Lab / Blackwell sm_120 pain. We train on the actual SO-101 in the lab.</td>
  </tr>
  <tr>
    <td>Human-in-loop safety net</td>
    <td>You override before disasters happen. The policy never destroys the gripper, the cubes, or itself.</td>
  </tr>
  <tr>
    <td>Reward classifier learnable</td>
    <td>Skips the dense-reward design problem that killed our archived PPO-from-scratch attempt.</td>
  </tr>
  <tr>
    <td>1st-class LeRobot port</td>
    <td>We don't have to reimplement RLPD from scratch. The framework is there.</td>
  </tr>
  <tr>
    <td>Satisfies "RL mandatory"</td>
    <td>SAC + replay buffer + Q-learning is unambiguously RL by the spec's intent.</td>
  </tr>
</table>

<p class="meta">Generated by <code>notes/visualizations/generate_hilserl_walkthrough.py</code> · references
<code>notes/so101_robot_learning_playbook.md</code> §T2, §3, §5 + HIL-SERL paper arXiv:2410.21845</p>

</body>
</html>
"""


def main():
    out_dir = Path(__file__).parent
    print("Generating HIL-SERL figures...", flush=True)
    figs = {
        "img_arch": fig_system_arch(),
        "img_split": fig_replay_split(),
        "img_rl": fig_where_rl(),
        "img_hil": fig_hil_flow(),
        "img_reward": fig_reward_classifier(),
        "img_timeline": fig_training_timeline(),
        "img_perf": fig_performance_and_fit(),
    }
    encoded = {name: b64_png(fig) for name, fig in figs.items()}
    html = HTML_TEMPLATE.format(**encoded)
    out_path = out_dir / "hilserl_walkthrough.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
