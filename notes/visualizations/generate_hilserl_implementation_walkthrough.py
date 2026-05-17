#!/usr/bin/env python3
"""HIL-SERL implementation walkthrough — what WE built today and the plan
to actually train + evaluate the policy on the SO-101.

This is different from `generate_hilserl_walkthrough.py`:
  - That one explains the TEXTBOOK HIL-SERL (Luo et al. paper, RLPD core,
    50/50 minibatch, reward classifier, etc.). It is "what HIL-SERL is."
  - This one explains "what WE simplified, what we built, where we are RIGHT NOW,
    and what the next 3 days look like." It's the project-specific story.

Run:
    /Users/admin/miniforge3/bin/python3.13 generate_hilserl_implementation_walkthrough.py

Output:
    hilserl_implementation_walkthrough.html (open in any browser, fully self-contained)
"""

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---- consistent palette (matches sibling walkthroughs) -------------------
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
COLOR_DONE = "#9BD8A4"
COLOR_INFLIGHT = "#FFD86B"
COLOR_PENDING = "#E0E0E0"
COLOR_DEFERRED = "#D0D0D0"
COLOR_KEPT = "#A8E6C8"
COLOR_CUT = "#FFB8B8"
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


# ---- Figure 1: The Eval 2 task we are actually solving --------------------
def fig_eval2_scene():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title(
        "Eval 2 — Targeted Pick-and-Place (50 pts)",
        fontsize=14, fontweight="bold", pad=15,
    )

    # === Scene panel (left) ===
    add_box(ax, (0.3, 0.5), 7.2, 5.8, "", "#FAFAFA", ec="#888", lw=1)
    ax.text(3.9, 6.0, "The scene the TA evaluator builds",
            ha="center", fontsize=11, fontweight="bold", color="#333")

    # Robot base
    add_box(ax, (0.6, 1.0), 1.2, 1.5, "SO-101\nfollower", COLOR_ROBOT, fontsize=9, fontweight="bold")
    # Arm sketch (segment lines)
    ax.plot([1.2, 2.5], [2.5, 4.2], color="#333", lw=4, solid_capstyle="round")
    ax.plot([2.5, 3.7], [4.2, 3.6], color="#333", lw=4, solid_capstyle="round")
    ax.plot([3.7, 4.3], [3.6, 2.5], color="#333", lw=3.5, solid_capstyle="round")
    # Gripper
    ax.plot([4.25, 4.35], [2.4, 2.1], color="#333", lw=2.5)
    ax.plot([4.30, 4.40], [2.4, 2.1], color="#333", lw=2.5)
    # Wrist cam
    ax.add_patch(FancyBboxPatch((4.1, 2.7), 0.4, 0.4, boxstyle="round,pad=0.03",
                                 fc="#444", ec="black", lw=1))
    ax.text(4.3, 2.93, "cam", ha="center", fontsize=6, color="white")

    # Table (light gray)
    ax.add_patch(FancyBboxPatch((1.8, 0.7), 5.2, 0.5, boxstyle="round,pad=0",
                                 fc="#B8ADA9", ec="black", lw=1))
    ax.text(4.4, 0.95, "Table (#B8ADA9)", ha="center", fontsize=8, color="#fff")

    # 2 cubes adjacent
    ax.add_patch(FancyBboxPatch((4.7, 1.2), 0.45, 0.45, boxstyle="round,pad=0",
                                 fc="#F5C518", ec="black", lw=1.2))
    ax.text(4.93, 1.42, "Y", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.add_patch(FancyBboxPatch((5.18, 1.2), 0.45, 0.45, boxstyle="round,pad=0",
                                 fc="#3498DB", ec="black", lw=1.2))
    ax.text(5.40, 1.42, "B", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(5.15, 1.85, "2 cubes adjacent\n(target color = yellow)", ha="center",
            fontsize=8, fontstyle="italic", color="#222")

    # Bowl (randomized position per TA spec)
    ax.add_patch(FancyBboxPatch((6.2, 1.2), 0.7, 0.4, boxstyle="round,pad=0.02",
                                 fc="#888", ec="black", lw=1.2))
    ax.text(6.55, 1.4, "bowl", ha="center", va="center", fontsize=8, fontweight="bold", color="white")
    ax.text(6.55, 1.85, "(x, y, z)\nRANDOMIZED per spec\n(we'll fix in v1)",
            ha="center", fontsize=7, fontstyle="italic", color=COLOR_BC)

    # Annotations
    ax.text(3.9, 5.5,
            "Task: identify the yellow block,\ngrasp it, place in the bowl,\nRELEASE.",
            ha="center", fontsize=9.5, color="#222",
            bbox=dict(boxstyle="round,pad=0.4", fc="#FFFEEA", ec="#888"))

    # === Evaluation rubric panel (right) ===
    add_box(ax, (8.0, 0.5), 5.7, 5.8, "", "#FAFAFA", ec="#888", lw=1)
    ax.text(10.85, 6.0, "Evaluation rubric (TA-spec)",
            ha="center", fontsize=11, fontweight="bold", color="#333")

    rubric_y = 5.3
    line_h = 0.6
    rubric = [
        ("5 rollouts", "10 pts each = 50 pts total"),
        ("Target color", "provided as input (we pick yellow for v1)"),
        ("Bowl pose", "provided as input, randomized per rollout"),
        ("Block positions", "randomized per rollout"),
        ("Success", "correct block IN bowl AND released"),
        ("RL required", "BC alone does not satisfy"),
    ]
    for i, (key, val) in enumerate(rubric):
        y = rubric_y - i * line_h
        ax.text(8.3, y, "•", fontsize=11, color=COLOR_RL, fontweight="bold")
        ax.text(8.5, y, key + ":", fontsize=9.5, fontweight="bold", color="#222")
        ax.text(10.0, y, val, fontsize=9, color="#333")

    ax.text(10.85, 1.0,
            "Scoring at TA eval time → we submit ONE policy.\n"
            "Our plan: ship TWO (SmolVLA BC + HIL-SERL RL),\n"
            "submit whichever scores higher on the 5-rollout matrix.",
            ha="center", fontsize=9, fontstyle="italic", color=COLOR_HIL,
            bbox=dict(boxstyle="round,pad=0.4", fc="#E9F8EE", ec=COLOR_HIL))

    return fig


# ---- Figure 2: The simplification ladder — what we cut to ship v1 -------
def fig_simplification_ladder():
    fig, ax = plt.subplots(figsize=(14, 8.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.5)
    ax.axis("off")
    ax.set_title(
        "Simplification ladder — what we cut from \"full Eval 2\" to ship v1 tomorrow",
        fontsize=14, fontweight="bold", pad=15,
    )

    # The full spec at top
    add_box(ax, (3.0, 7.4), 8.0, 0.8,
            "FULL Eval 2 spec (the 10-star product)",
            "#FAFAFA", fontsize=11, fontweight="bold", ec="#888")
    ax.text(7.0, 7.05,
            "6 colors × randomized bowl × language-conditioned RL × trained classifier × top+wrist cams",
            ha="center", fontsize=9, color="#444", fontstyle="italic")

    # Cascade of simplifications
    rungs = [
        # (y_top, label, kept_left, cut_right, rationale)
        (5.8, "Decision D8 — SmolVLA as base",
              "SmolVLA stays a parallel BC baseline (already trained)",
              "Use SmolVLA encoder as SAC actor (Option C — fork lerobot, multi-day)",
              "lerobot HIL-SERL hardcodes SAC arch; encoder transplant is research, not ship"),
        (4.2, "Decision D1 — color scope (Option C)",
              "ONE policy for YELLOW only (~15 demos, 1 training run)",
              "6 policies (per-color, 90 demos) OR 1 goal-conditioned (custom plumbing)",
              "6-color demo collection = full evening of teleop; goal-cond is v1.5 plumbing"),
        (2.6, "Decision D2 — bowl pose",
              "Train at fixed bowl (-15.5, 29.5) cm",
              "Bowl xyz in state vector from v1 (requires custom processor step)",
              "No documented hook in lerobot 0.5.2; bowl-OOD probe on Day 3 tells us if v1.5 needs it"),
        (1.0, "Decision D3 — reward signal",
              "Manual keyboard reward (s = success, esc = fail)",
              "Trained ResNet-10 classifier (~200 pos / ~1000 neg labels)",
              "HF doc explicitly endorses manual for round 1; classifier is v1.5"),
    ]
    for y, decision, kept, cut, why in rungs:
        # Decision label centered
        ax.text(7.0, y + 1.05, decision, ha="center", fontsize=10,
                fontweight="bold", color="#333")
        # Kept side (left, green)
        add_box(ax, (0.4, y), 6.0, 0.9, kept, COLOR_KEPT, fontsize=9, fontweight="bold")
        # Cut side (right, red)
        add_box(ax, (7.6, y), 6.0, 0.9, cut, COLOR_CUT, fontsize=9)
        # Arrow from FULL spec down through each decision
        if y == 5.8:
            add_arrow(ax, (7.0, 7.4), (7.0, 6.7), color="#666", lw=1.5)
        # Why annotation
        ax.text(7.0, y - 0.2, f"Why: {why}", ha="center", fontsize=8,
                fontstyle="italic", color="#666")

    # The v1 outcome at bottom
    add_box(ax, (3.0, -0.3), 8.0, 1.0,
            "v1 = single-color (yellow) HIL-SERL on SO-101,\n"
            "fixed bowl, manual reward, wrist cam, SmolVLA in parallel as backup",
            COLOR_HIL, fontsize=11, fontweight="bold", ec=COLOR_HIL, lw=2.5)

    # Legend
    ax.text(0.4, -1.1, "Legend:  ",
            fontsize=9, color="#555", fontweight="bold")
    ax.add_patch(FancyBboxPatch((1.4, -1.2), 0.3, 0.2, fc=COLOR_KEPT, ec="black", lw=1))
    ax.text(1.8, -1.1, "KEPT in v1", fontsize=9, color="#222")
    ax.add_patch(FancyBboxPatch((3.5, -1.2), 0.3, 0.2, fc=COLOR_CUT, ec="black", lw=1))
    ax.text(3.9, -1.1, "DEFERRED to v1.5 / v2 / never", fontsize=9, color="#222")
    ax.set_ylim(-1.5, 8.5)

    return fig


# ---- Figure 3: Two-machine architecture (where things run) ---------------
def fig_architecture():
    fig, ax = plt.subplots(figsize=(14, 7.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    ax.set_title(
        "Two-machine architecture — Mac + 5090 + HF, what runs where",
        fontsize=14, fontweight="bold", pad=15,
    )

    # === HF Hub (top, shared) ===
    add_box(ax, (4.5, 6.2), 5.0, 0.9, "Hugging Face Hub  (osammotg1/*)",
            "#FFF1A8", fontsize=11, fontweight="bold")
    ax.text(7.0, 5.95,
            "datasets · models · checkpoints — read by both machines",
            ha="center", fontsize=8, fontstyle="italic", color="#555")

    # === Mac (left) ===
    add_box(ax, (0.3, 0.4), 6.3, 5.2, "", "#F5F8FA", ec="#4A90E2", lw=2)
    ax.text(3.45, 5.25, "This MacBook  (M-series, MPS)",
            ha="center", fontsize=11, fontweight="bold", color="#4A90E2")

    # SO-101 hardware
    add_box(ax, (0.6, 4.0), 2.4, 0.8, "SO-101 follower\n+ wrist camera\n(USB)",
            COLOR_ROBOT, fontsize=9)
    add_box(ax, (3.4, 4.0), 2.8, 0.8, "SO-101 leader\n(in your hands)",
            COLOR_HUMAN, fontsize=9, fontweight="bold")

    # lerobot env
    add_box(ax, (0.6, 2.7), 5.6, 1.0,
            "lerobot 0.5.2 editable  (base conda Python 3.13)\n"
            "+ [hilserl] extra: placo, gym-hil, pin, mujoco",
            "#EEEEEE", fontsize=9)

    # Actor process
    add_box(ax, (0.6, 1.5), 2.7, 0.9, "ACTOR process\n(lerobot.rl.actor)\nMPS inference",
            COLOR_ACTOR, fontsize=9, fontweight="bold")
    add_box(ax, (3.5, 1.5), 2.7, 0.9, "record + eval\n(gym_manipulator)\nrecord_hilserl_demos.sh",
            COLOR_BUFFER_DEMO, fontsize=9)

    # SmolVLA inference (parallel)
    add_box(ax, (0.6, 0.5), 5.6, 0.8, "SmolVLA inference (parallel BC baseline)  ·  infer_smolvla.sh",
            COLOR_BC, fontsize=9)

    # === 5090 (right) ===
    add_box(ax, (7.4, 0.4), 6.3, 5.2, "", "#FAF0F8", ec="#9B59B6", lw=2)
    ax.text(10.55, 5.25, "5090 workstation  (ethrc-rl-ws1, user tommaso)",
            ha="center", fontsize=11, fontweight="bold", color="#9B59B6")

    add_box(ax, (7.7, 4.0), 5.6, 0.8,
            "RTX 5090  (NVIDIA, CUDA 12.x, ~80 GB VRAM)",
            COLOR_REWARD, fontsize=10, fontweight="bold")

    add_box(ax, (7.7, 2.7), 5.6, 1.0,
            "lerobot 0.5.2 editable (uv .venv)\n+ training stack: torch.cuda, wandb, transformers",
            "#EEEEEE", fontsize=9)

    add_box(ax, (7.7, 1.5), 2.7, 0.9, "LEARNER process\n(lerobot.rl.learner)\nSAC updates",
            COLOR_CRITIC, fontsize=9, fontweight="bold")
    add_box(ax, (10.6, 1.5), 2.7, 0.9,
            "DATASET CONVERSION\n(running NOW, ~20 min)\nfor v1.5 multi-color",
            COLOR_INFLIGHT, fontsize=9, fontweight="bold", lw=2.2)

    add_box(ax, (7.7, 0.5), 5.6, 0.8,
            "Past training: ACT + SmolVLA  (overnight runs)",
            COLOR_BC, fontsize=9)

    # === Cross-machine arrows ===
    # Mac → HF (push demos, pull checkpoints)
    add_arrow(ax, (3.0, 5.6), (5.5, 6.2), color=COLOR_HIL, lw=1.8)
    ax.text(4.0, 5.95, "push demos /\npull checkpoints", fontsize=8, color=COLOR_HIL,
            fontstyle="italic", rotation=15)
    # 5090 → HF (push checkpoints, pull dataset for conversion)
    add_arrow(ax, (11.0, 5.6), (8.5, 6.2), color=COLOR_HIL, lw=1.8)
    ax.text(10.0, 5.95, "push model /\npull dataset", fontsize=8, color=COLOR_HIL,
            fontstyle="italic", rotation=-15)

    # Mac actor ↔ 5090 learner (gRPC during training)
    add_arrow(ax, (3.3, 1.95), (7.7, 1.95), color=COLOR_RL, lw=2.4,
              connectionstyle="arc3,rad=0.0")
    ax.text(5.5, 2.15, "gRPC  (transitions ↔ weight updates @ 1.5 s)",
            ha="center", fontsize=8.5, color=COLOR_RL, fontweight="bold", fontstyle="italic")
    add_arrow(ax, (7.7, 1.75), (3.3, 1.75), color=COLOR_RL, lw=2.4,
              connectionstyle="arc3,rad=0.0")

    # Bottom legend
    ax.text(0.3, -0.2,
            "Blue arrows = RL training loop · Green arrows = HF Hub data flow · "
            "Yellow = in-flight RIGHT NOW (5090 conversion)",
            fontsize=9, color="#555", fontstyle="italic")
    ax.set_ylim(-0.6, 7.5)

    return fig


# ---- Figure 4: Training data flow (offline → online → eval) --------------
def fig_data_flow():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title(
        "v1 training data flow — from teleop demos to a deployed yellow-grasp policy",
        fontsize=14, fontweight="bold", pad=15,
    )

    # === PHASE 1 (left): Offline demo collection ===
    ax.text(2.3, 6.4, "PHASE 1 — offline demos (Day 1, ~30 min at robot)",
            ha="center", fontsize=11, fontweight="bold", color="#333")
    add_box(ax, (0.3, 4.8), 4.0, 1.2,
            "bash teleop/record_hilserl_demos.sh\n\n"
            "Drives gym_manipulator in record mode.\n"
            "15 episodes of yellow + distractor → bowl.\n"
            "Each demo: YOU pilot the leader, follower mirrors.",
            COLOR_BUFFER_DEMO, fontsize=9)
    add_box(ax, (0.3, 3.4), 4.0, 1.1,
            "EE-space dataset on HF\n"
            "osammotg1/projet3-hilserl-yellow-v1",
            "#EEEEEE", fontsize=9, fontweight="bold")
    add_arrow(ax, (2.3, 4.8), (2.3, 4.5))

    add_box(ax, (0.3, 2.1), 4.0, 1.1,
            "DEMO BUFFER  (offline)\nNever empties.\nSampled 50/50 every minibatch.",
            COLOR_BUFFER_DEMO, fontsize=9, fontweight="bold")
    add_arrow(ax, (2.3, 3.4), (2.3, 3.2))

    # === PHASE 2 (middle): Online RL ===
    ax.text(7.0, 6.4, "PHASE 2 — online RL on the real arm (Day 2, ~2.5 h)",
            ha="center", fontsize=11, fontweight="bold", color="#333")
    add_box(ax, (5.0, 4.8), 4.0, 1.2,
            "ACTOR (Mac, MPS)  10 Hz loop:\n"
            "obs → π(a|s) → twist + gripper →\n"
            "IK → joint targets → SO-101\n"
            "(YOU watch with the leader)",
            COLOR_ACTOR, fontsize=9)
    add_box(ax, (5.0, 2.7), 4.0, 0.9,
            "ONLINE BUFFER  (grows)\nAutonomous rollouts:\n(obs, action, r, next_obs)",
            COLOR_BUFFER_ONLINE, fontsize=9)
    add_box(ax, (5.0, 1.3), 4.0, 0.9,
            "INTERVENTION BUFFER\n(your leader actions, double-counted\ninto demo buffer too)",
            COLOR_BUFFER_INTV, fontsize=9)
    add_arrow(ax, (7.0, 4.8), (7.0, 3.6), color="#666", lw=1.4)
    ax.text(7.15, 4.3, "autonomous", fontsize=8, color="#666", rotation=90)
    add_arrow(ax, (5.0, 4.9), (4.3, 1.5), color=COLOR_HUMAN, lw=2,
              connectionstyle="arc3,rad=-0.3")
    ax.text(3.6, 2.5, "intervention\n→ both buffers",
            fontsize=8, color=COLOR_HUMAN, fontweight="bold", fontstyle="italic")

    # Reward (manual)
    add_box(ax, (5.0, 0.3), 4.0, 0.7,
            "Reward = YOU press 's' or 'esc' per episode (binary, sparse)",
            COLOR_REWARD, fontsize=9)

    # === PHASE 3 (right): Learner + eval ===
    ax.text(11.7, 6.4, "PHASE 3 — eval (Day 3, ~45 min)",
            ha="center", fontsize=11, fontweight="bold", color="#333")
    add_box(ax, (9.8, 4.5), 3.9, 1.4,
            "LEARNER  (5090, CUDA)\nSAC updates from\n50/50 minibatches\n10 Q-critics, REDQ subset of 2",
            COLOR_CRITIC, fontsize=9, fontweight="bold")
    add_box(ax, (9.8, 3.0), 3.9, 1.0,
            "Updated π weights →\nactor via gRPC every 1.5 s",
            "#FFFFD0", fontsize=9)
    add_arrow(ax, (9.8, 5.2), (9.0, 5.2), color=COLOR_RL, lw=2)
    add_arrow(ax, (9.8, 3.5), (9.0, 4.8), color=COLOR_RL, lw=1.6,
              connectionstyle="arc3,rad=0.2")

    add_box(ax, (9.8, 1.3), 3.9, 1.3,
            "5-ROLLOUT EVAL\n+ 5-rollout BOWL-OOD probe\n→ submit whichever beats SmolVLA",
            COLOR_HIL, fontsize=9, fontweight="bold")
    add_arrow(ax, (11.7, 3.0), (11.7, 2.6))

    # Cross-phase: demos seed online phase
    add_arrow(ax, (4.3, 2.6), (5.0, 2.6), color=COLOR_HIL, lw=2)
    ax.text(4.65, 2.8, "seed", fontsize=8, color=COLOR_HIL, fontweight="bold")

    return fig


# ---- Figure 5: State machine — where we are NOW --------------------------
def fig_state_machine():
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    ax.set_title(
        "Where we ARE  ([DONE] = green  ·  [NOW] = yellow  ·  [TBD] = grey)",
        fontsize=14, fontweight="bold", pad=15,
    )

    # 9 states in a row, color-coded by status
    states = [
        ("S0", "Plan +\nplan-eng-review", COLOR_DONE, "[DONE]"),
        ("S1", "Probe lerobot\nSO-101 EE class", COLOR_DONE, "[DONE]"),
        ("S2", "Install\n[hilserl] extra", COLOR_DONE, "[DONE]"),
        ("S3", "Vendor URDF\n+ meshes", COLOR_DONE, "[DONE]"),
        ("S4", "Scaffold\nconfigs + scripts", COLOR_DONE, "[DONE]"),
        ("S5", "Pre-flight\nTests 1-3", COLOR_DONE, "[DONE]"),
        ("S6", "Push to git\n(commit 6a2d220)", COLOR_DONE, "[DONE]"),
        ("S7", "5090 dataset\nconversion (v1.5)", COLOR_INFLIGHT, "[NOW]"),
        ("S8", "Pre-flight\nTest 4 (USB)", COLOR_PENDING, "[TBD]"),
    ]
    y_top = 4.5
    y_mid = 3.6
    y_bot = 2.7
    x0 = 0.3
    box_w = 1.45
    gap = 0.10

    for i, (sid, lab, color, mark) in enumerate(states):
        x = x0 + i * (box_w + gap)
        add_box(ax, (x, y_mid), box_w, 0.95, f"{mark}\n{lab}",
                color, fontsize=8.5, fontweight="bold")
        ax.text(x + box_w/2, y_mid + 1.05, sid, ha="center", fontsize=8,
                color="#555", fontweight="bold")
        if i < len(states) - 1:
            add_arrow(ax, (x + box_w + 0.01, y_mid + 0.48),
                          (x + box_w + gap, y_mid + 0.48), color="#888", lw=1.2)

    # Second row — the upcoming steps (at the robot)
    upcoming = [
        ("S9",  "find_joint_limits\n(EE bounds)", COLOR_PENDING),
        ("S10", "Record 15\nyellow demos", COLOR_PENDING),
        ("S11", "replay\nsanity check", COLOR_PENDING),
        ("S12", "crop_dataset_roi\n(128×128 ROI)", COLOR_PENDING),
        ("S13", "5090↔Mac\nlatency probe", COLOR_PENDING),
        ("S14", "Launch learner\n(5090, SSH)", COLOR_PENDING),
        ("S15", "Launch actor\n(Mac) + train ~2.5 h", COLOR_PENDING),
        ("S16", "5-roll eval +\nbowl-OOD probe", COLOR_PENDING),
        ("S17", "Submit best\npolicy", COLOR_PENDING),
    ]
    for i, (sid, lab, color) in enumerate(upcoming):
        x = x0 + i * (box_w + gap)
        add_box(ax, (x, y_bot - 1.05), box_w, 0.95, f"[TBD]\n{lab}",
                color, fontsize=8.5)
        ax.text(x + box_w/2, y_bot - 0.10, sid, ha="center", fontsize=8,
                color="#555", fontweight="bold")
        if i < len(upcoming) - 1:
            add_arrow(ax, (x + box_w + 0.01, y_bot - 0.57),
                          (x + box_w + gap, y_bot - 0.57), color="#888", lw=1.2)

    # Connector between rows
    add_arrow(ax, (x0 + box_w/2, y_mid),
                  (x0 + box_w/2, y_bot - 0.10),
                  color="#666", lw=1.8,
                  connectionstyle="arc3,rad=-0.2")
    ax.text(x0 + box_w + 0.1, y_mid - 0.4, "you are HERE",
            fontsize=9, fontweight="bold", color="#B22222",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFEEEA", ec="#B22222"))

    # Labels
    ax.text(0.3, y_mid + 1.5, "TODAY (off-robot, both machines)",
            fontsize=10, fontweight="bold", color="#333")
    ax.text(0.3, y_bot + 0.2, "NEXT — at the robot",
            fontsize=10, fontweight="bold", color="#333")

    # Title for parallel track (5090)
    ax.text(13.7, y_mid + 1.5, "(S7 runs in parallel on 5090)",
            fontsize=8, fontstyle="italic", color="#666", ha="right")

    ax.set_ylim(-0.5, 6.5)
    return fig


# ---- Figure 6: Two parallel tracks — timeline ----------------------------
def fig_parallel_tracks():
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    ax.set_title(
        "The two parallel tracks — what's running on each machine RIGHT NOW",
        fontsize=14, fontweight="bold", pad=15,
    )

    # Mac track (top)
    ax.text(0.3, 4.7, "MAC TRACK — single-color (yellow) HIL-SERL v1",
            fontsize=11, fontweight="bold", color="#4A90E2")
    add_box(ax, (0.3, 3.7), 1.7, 0.7, "[DONE] Scaffolding\n+ pre-flight 1-3", COLOR_DONE, fontsize=8)
    add_box(ax, (2.2, 3.7), 1.7, 0.7, "Now: pre-flight 4\n(USB enum)", COLOR_INFLIGHT, fontsize=8, fontweight="bold")
    add_box(ax, (4.1, 3.7), 1.7, 0.7, "find_joint_limits\n(at robot)", COLOR_PENDING, fontsize=8)
    add_box(ax, (6.0, 3.7), 1.7, 0.7, "Record 15 demos\n(~25 min)", COLOR_PENDING, fontsize=8)
    add_box(ax, (7.9, 3.7), 1.7, 0.7, "Crop ROI\n(~5 min)", COLOR_PENDING, fontsize=8)
    add_box(ax, (9.8, 3.7), 1.7, 0.7, "Train HIL-SERL\n(~2.5 h)", COLOR_RL, fontsize=8, fontweight="bold", ec="white")
    add_box(ax, (11.7, 3.7), 2.0, 0.7, "5-roll eval +\nbowl-OOD probe", COLOR_HIL, fontsize=8, fontweight="bold", ec="white")
    for i in range(6):
        add_arrow(ax, (2.0 + i*1.9, 4.05), (2.2 + i*1.9, 4.05), color="#888", lw=1.2)

    # 5090 track (bottom)
    ax.text(0.3, 2.6, "5090 TRACK — multi-color dataset conversion (for v1.5)",
            fontsize=11, fontweight="bold", color="#9B59B6")
    add_box(ax, (0.3, 1.6), 3.6, 0.7, "git pull + read\nHANDOFF doc", COLOR_DONE, fontsize=8)
    add_box(ax, (4.0, 1.6), 4.0, 0.7,
            "DATASET CONVERSION  (running ~20 min)\nFK on 101 ep × 6 colors → EE space → push to HF",
            COLOR_INFLIGHT, fontsize=8.5, fontweight="bold", lw=2.2)
    add_box(ax, (8.1, 1.6), 3.0, 0.7,
            "v1.5 dataset on HF\nosammotg1/...-multicolor-v1", COLOR_PENDING, fontsize=8)
    add_box(ax, (11.2, 1.6), 2.5, 0.7,
            "Wait for v1 results;\nuse if v1 succeeds", COLOR_PENDING, fontsize=8)
    for i in range(3):
        add_arrow(ax, (3.9 + i*3.95, 1.95), (4.1 + i*3.95, 1.95), color="#888", lw=1.2)

    # Time axis at the very bottom
    ax.plot([0.3, 13.7], [0.7, 0.7], color="#666", lw=2)
    ax.text(0.3, 0.4, "NOW", fontsize=9, color="#B22222", fontweight="bold")
    ax.text(5.5, 0.4, "+30 min  (start of recording at robot)", fontsize=9, color="#555")
    ax.text(9.8, 0.4, "+3-4 h  (training starts)", fontsize=9, color="#555")
    ax.text(12.5, 0.4, "Day 3  (eval)", fontsize=9, color="#555")
    ax.text(3.0, 0.85, "▼  NOW", fontsize=14, color="#B22222", fontweight="bold", ha="center")

    return fig


# ---- Figure 7: Decision tree — what if v1 fails / succeeds ---------------
def fig_decision_tree():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title(
        "Day-3 decision tree — what happens after the eval matrix",
        fontsize=14, fontweight="bold", pad=15,
    )

    # Root: the eval results
    add_box(ax, (4.5, 5.6), 5.0, 0.9,
            "Day 3 — 5-rollout primary eval (yellow, trained bowl)",
            COLOR_HIL, fontsize=10, fontweight="bold")

    # Three branches based on success rate
    branches = [
        (1.0, "Success ≥ 4 / 5", COLOR_DONE,
            "BEATS SmolVLA?",
            "Submit HIL-SERL\nas Eval 2 deliverable",
            "Submit SmolVLA\n+ keep HIL-SERL as v1.5 base"),
        (6.0, "Success 2-3 / 5", COLOR_INFLIGHT,
            "Bowl-OOD probe results?",
            "If OOD ≥ 2/5 →\nthis IS v1.5 baseline,\nadd multi-color",
            "If OOD = 0/5 →\nretrain with bowl\nrandomization (v1.5)"),
        (10.0, "Success ≤ 1 / 5", COLOR_CUT,
            "Why did it fail?",
            "Demos bad → re-record\nOR use 5090 multicolor",
            "Reward signal bad →\ntrain ResNet-10 classifier\n(v1.5 short path)"),
    ]
    for x, label, color, q, optA, optB in branches:
        # Branch label
        add_box(ax, (x, 4.4), 3.0, 0.6, label, color, fontsize=10, fontweight="bold")
        # Arrow from root
        add_arrow(ax, (7.0, 5.6), (x + 1.5, 5.0), color="#666", lw=1.5)
        # Question
        ax.text(x + 1.5, 4.0, q, ha="center", fontsize=9.5,
                fontweight="bold", fontstyle="italic", color="#333")
        # Two options
        add_box(ax, (x - 0.2, 2.4), 3.4, 1.1, optA, "#F5F5F5", fontsize=8.5)
        add_box(ax, (x - 0.2, 0.9), 3.4, 1.1, optB, "#F5F5F5", fontsize=8.5)
        add_arrow(ax, (x + 1.5, 3.8), (x + 1.5, 3.5), color="#888", lw=1)
        add_arrow(ax, (x + 1.5, 2.4), (x + 1.5, 2.0), color="#888", lw=1)

    # Bottom annotation: the v1.5 trigger
    ax.text(7.0, 0.3,
            "Either way, the 5090 conversion (running NOW) is what unlocks v1.5 "
            "without 90 more minutes of teleop.",
            ha="center", fontsize=9.5, color=COLOR_HIL, fontweight="bold",
            fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.4", fc="#E9F8EE", ec=COLOR_HIL))
    ax.set_ylim(-0.2, 7)
    return fig


# ---- Figure 8: File inventory — what we built today ----------------------
def fig_file_inventory():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_title(
        "File inventory — what landed in commit 6a2d220 (15 files, +2469 lines)",
        fontsize=14, fontweight="bold", pad=15,
    )

    # Group 1: sim/hilserl/
    add_box(ax, (0.3, 5.6), 13.4, 2.7, "", "#FAFAFA", ec="#888", lw=1)
    ax.text(0.5, 8.05, "sim/hilserl/",
            fontsize=11, fontweight="bold", color="#4A90E2")
    files_sim = [
        ("assets/so101/urdf/so_arm101.urdf", "13 KB", "SO-101 robot model (placo IK)", COLOR_KEPT),
        ("assets/so101/urdf/assets/*.stl",  "15 MB", "13 mesh files (gitignored, regenerable)", COLOR_DEFERRED),
        ("assets/so101/README.md",          "small", "How to refetch meshes", COLOR_KEPT),
        ("configs/env_config_so101.json",   "80 lines", "Record-mode env config (so101+leader+IK)", COLOR_KEPT),
        ("configs/train_config_hilserl_so101.json", "134 lines", "SAC + learner/actor + W&B config", COLOR_KEPT),
        ("configs/_upstream/*",             "stale", "Reference HF configs (do not load against 0.5.2)", COLOR_DEFERRED),
        ("scripts/convert_eval2_to_hilserl_yellow.py", "207 lines", "FK conversion (joint → EE), validated", COLOR_KEPT),
        ("README.md",                       "172 lines", "Run order + per-config decision rationale", COLOR_KEPT),
    ]
    y = 7.75
    for path, size, desc, color in files_sim:
        ax.text(0.6, y, path, fontsize=8.5, family="monospace", color="#222")
        ax.text(5.5, y, size, fontsize=8.5, color="#666", style="italic")
        ax.text(6.5, y, desc, fontsize=8.5, color="#333")
        ax.add_patch(FancyBboxPatch((13.0, y - 0.05), 0.3, 0.18,
                                     fc=color, ec="black", lw=0.5))
        y -= 0.28

    # Group 2: teleop/
    add_box(ax, (0.3, 3.5), 13.4, 1.9, "", "#FAFAFA", ec="#888", lw=1)
    ax.text(0.5, 5.15, "teleop/",
            fontsize=11, fontweight="bold", color="#27AE60")
    files_teleop = [
        ("_common.sh",                "74 lines", "Shared port/auth/cache/log plumbing", COLOR_KEPT),
        ("record_hilserl_demos.sh",   "88 lines", "gym_manipulator wrapper (EE-space recorder)", COLOR_KEPT),
    ]
    y = 4.75
    for path, size, desc, color in files_teleop:
        ax.text(0.6, y, path, fontsize=8.5, family="monospace", color="#222")
        ax.text(5.5, y, size, fontsize=8.5, color="#666", style="italic")
        ax.text(6.5, y, desc, fontsize=8.5, color="#333")
        ax.add_patch(FancyBboxPatch((13.0, y - 0.05), 0.3, 0.18,
                                     fc=color, ec="black", lw=0.5))
        y -= 0.28

    # Group 3: notes/
    add_box(ax, (0.3, 0.5), 13.4, 2.7, "", "#FAFAFA", ec="#888", lw=1)
    ax.text(0.5, 2.95, "notes/",
            fontsize=11, fontweight="bold", color="#9B59B6")
    files_notes = [
        ("hilserl_eval2_plan.md",                       "247 lines", "Design doc (plan-eng-review v2 outcomes)", COLOR_KEPT),
        ("hilserl_lab_checklist.md",                    "219 lines", "Day-1 at-the-robot step-by-step", COLOR_KEPT),
        ("HANDOFF_5090_HILSERL_DATASET_CONVERSION.md", "307 lines", "5090-side instructions for v1.5 conversion", COLOR_KEPT),
    ]
    y = 2.55
    for path, size, desc, color in files_notes:
        ax.text(0.6, y, path, fontsize=8.5, family="monospace", color="#222")
        ax.text(5.5, y, size, fontsize=8.5, color="#666", style="italic")
        ax.text(6.5, y, desc, fontsize=8.5, color="#333")
        ax.add_patch(FancyBboxPatch((13.0, y - 0.05), 0.3, 0.18,
                                     fc=color, ec="black", lw=0.5))
        y -= 0.28

    # Legend
    ax.add_patch(FancyBboxPatch((0.3, 0.05), 0.25, 0.15, fc=COLOR_KEPT, ec="black", lw=0.5))
    ax.text(0.6, 0.13, "committed + active", fontsize=8.5, color="#222")
    ax.add_patch(FancyBboxPatch((3.0, 0.05), 0.25, 0.15, fc=COLOR_DEFERRED, ec="black", lw=0.5))
    ax.text(3.3, 0.13, "gitignored or reference only", fontsize=8.5, color="#222")

    return fig


# ---- HTML template -------------------------------------------------------
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>HIL-SERL Implementation Walkthrough — what WE built</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      max-width: 1180px;
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
    h3 {{ color: #444; margin-top: 1.5rem; }}
    p {{ font-size: 15px; }}
    .subtitle {{
      color: #666;
      font-style: italic;
      margin-top: 0;
      margin-bottom: 2rem;
    }}
    .badge {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: bold;
      margin-right: 4px;
    }}
    .badge-done    {{ background: #9BD8A4; color: #1A4D2E; }}
    .badge-inflight{{ background: #FFD86B; color: #6B4F00; }}
    .badge-pending {{ background: #E0E0E0; color: #555; }}
    .badge-deferred{{ background: #D0D0D0; color: #555; }}
    figure {{
      margin: 2rem 0;
      text-align: center;
      background: white;
      padding: 1.2rem;
      border-radius: 10px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }}
    figure img {{ max-width: 100%; height: auto; }}
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
    .youhere {{
      background: #FFEEEA;
      border-left: 4px solid #B22222;
      padding: 0.8rem 1rem;
      margin: 1.5rem 0;
      border-radius: 4px;
      font-weight: bold;
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

<h1>HIL-SERL Implementation — what WE built today (and what we'll do tomorrow)</h1>
<p class="subtitle">
  Sister doc to <code>hilserl_walkthrough.html</code>. That one explains the
  TEXTBOOK HIL-SERL (Luo et al. paper). This one explains <strong>what we
  simplified, what we shipped, where we are right now, and the 3-day plan to
  evaluate the policy.</strong>
</p>

<div class="youhere">
  YOU ARE HERE: <span class="badge badge-done">scaffolding ✓</span>
  <span class="badge badge-done">pre-flight 1-3 ✓</span>
  <span class="badge badge-inflight">5090 dataset conversion (running ~20 min)</span>
  <span class="badge badge-pending">USB enum (Test 4)</span>
  <span class="badge badge-pending">at-robot Day 1</span>
</div>

<h2>1. The task we're actually solving (TA-spec, no embellishment)</h2>
<figure>
  <img alt="Eval 2 task — scene + rubric" src="data:image/png;base64,{img_scene}" />
  <figcaption>Figure 1 — Eval 2 scene + the 50-pt scoring rubric. Two cubes of
  distinct colors, one target color provided as input, bowl at a TA-randomized
  position. We will train on yellow + fixed bowl in v1; the bowl-OOD probe on
  Day 3 tells us whether v1.5 needs goal-conditioning.</figcaption>
</figure>

<p>
  The TA gives us five rollouts at evaluation. Each rollout is 10 points if the
  policy picks the correct-color block and places it in the bowl. Crucially,
  <em>which</em> color and <em>where</em> the bowl sits are randomized per
  rollout — neither is known at policy-design time. We don't get to pick the
  yellow scene every time at TA eval. v1 ignores that; v1.5 addresses it.
</p>

<h2>2. The simplification ladder — what we cut to ship v1 tomorrow</h2>
<figure>
  <img alt="Simplification ladder" src="data:image/png;base64,{img_simplification}" />
  <figcaption>Figure 2 — Four decisions, each cutting scope to make v1 shippable
  in 3 days. Green = kept; red = deferred to v1.5 or beyond. Honest CEO-style
  framing: every "kept" choice is a known compromise.</figcaption>
</figure>

<p>
  The four decisions came out of yesterday's plan-eng-review:
</p>
<ul>
  <li><strong>D8 (SmolVLA's role):</strong> Option A — SmolVLA stays a parallel
      BC baseline (already trained, deployable today). HIL-SERL trains its own
      SAC policy from fresh demos. <em>Why not Option C?</em> lerobot's SAC
      hardcodes a ResNet-10 actor; transplanting SmolVLA's encoder is a multi-day
      fork.</li>
  <li><strong>D1 (color scope):</strong> Option C — ONE policy for yellow only.
      ~15 demos vs ~90 for 6 colors. The 5090 dataset conversion (running NOW)
      unlocks the multi-color v1.5 path without more teleop.</li>
  <li><strong>D2 (bowl pose):</strong> Train at the fixed (-15.5, 29.5) cm pose.
      Goal-conditioning would require a custom <code>GoalConditioningProcessorStep</code>
      because lerobot 0.5.2's <code>ObservationConfig</code> has no documented
      hook for arbitrary goal vectors.</li>
  <li><strong>D3 (reward):</strong> Manual keyboard reward (<code>s</code> /
      <code>esc</code>) for v1. HF doc explicitly endorses this for round 1.
      Trained classifier is v1.5.</li>
</ul>

<div class="keytake">
  <strong>The CEO-review question:</strong> are we cutting too much? Each
  simplification has a quantifiable upgrade path. If v1 succeeds at all, the
  multi-color dataset is already converting on the 5090 → v1.5 is 1 training
  run away, not 1 teleop session. If bowl-OOD fails, we have the data + URDF
  to add the goal-conditioning step in v1.5b. <em>The simplifications are
  cheap to reverse, hard to revisit during the semester clock.</em>
</div>

<h2>3. Where things run — two machines + HF Hub</h2>
<figure>
  <img alt="Two-machine architecture" src="data:image/png;base64,{img_arch}" />
  <figcaption>Figure 3 — The Mac (M-series, MPS) drives the SO-101 and runs
  the SAC actor; the 5090 (RTX 5090, CUDA) runs the SAC learner via gRPC. HF
  Hub is the artifact exchange. The yellow box is the dataset conversion
  RUNNING RIGHT NOW.</figcaption>
</figure>

<p>
  The actor↔learner split is HIL-SERL's signature. Without it, the 10 Hz control
  loop on the Mac would compete with the SAC update for CPU/GPU time and the
  arm would stall mid-episode. With gRPC, the Mac only does inference (MPS is
  enough for a ResNet-10 + MLP at 10 Hz) and the 5090 burns its CUDA on the
  REDQ-10 ensemble + UTD=10-20 SAC updates.
</p>

<h2>4. The training data flow — three phases</h2>
<figure>
  <img alt="Data flow: demos → online RL → eval" src="data:image/png;base64,{img_flow}" />
  <figcaption>Figure 4 — Phase 1 fills the demo buffer (offline). Phase 2 trains
  online while the human watches with the leader; interventions go into BOTH
  the online and demo buffers. Phase 3 evaluates and decides what to submit.</figcaption>
</figure>

<p>
  Two things to internalize about this flow:
</p>
<ol>
  <li><strong>The demo buffer is the imitation prior</strong> — it never empties,
      and every minibatch is 128 demos + 128 online (the RLPD 50/50 split). It's
      what keeps the policy from drifting after thousands of online steps.</li>
  <li><strong>Your interventions are double-counted</strong> — when you grab
      the leader mid-rollout to rescue a doomed grasp, that transition lands in
      the intervention buffer AND gets aliased into the demo buffer. The critic
      learns from it as RL data; the actor learns from it as imitation data.</li>
</ol>

<h2>5. Where we are RIGHT NOW — full state machine</h2>
<figure>
  <img alt="State machine — where we are now" src="data:image/png;base64,{img_states}" />
  <figcaption>Figure 5 — 9 states completed today (S0-S6), 1 in-flight on the
  5090 (S7), 1 pending on this Mac (S8 USB enum). Then 9 more states to go on
  Day 1 - Day 3.</figcaption>
</figure>

<p>
  The "you are here" marker sits right after S6 (git push) and just before S8
  (the USB enumeration test). S7 — the multi-color dataset conversion on the
  5090 — is running in parallel and doesn't block any S8+ step.
</p>

<h2>6. The two parallel tracks — same wall-clock, different machines</h2>
<figure>
  <img alt="Parallel tracks timeline" src="data:image/png;base64,{img_tracks}" />
  <figcaption>Figure 6 — Mac track and 5090 track run independently. Mac
  produces v1 (single-color); 5090 produces the v1.5 multi-color dataset. They
  converge after v1 results are in (Day 3).</figcaption>
</figure>

<h2>7. Day-3 decision tree — what we do based on eval results</h2>
<figure>
  <img alt="Decision tree based on Day-3 results" src="data:image/png;base64,{img_decisions}" />
  <figcaption>Figure 7 — Three branches based on the primary 5-rollout success
  count. The bowl-OOD probe modulates the v1.5 choice. The 5090 conversion
  unlocks every branch's "v1.5" path without more teleop.</figcaption>
</figure>

<div class="warn">
  <strong>The single most important Day-3 question</strong> isn't "did
  HIL-SERL beat SmolVLA?" — it's "does the policy fail catastrophically when
  the bowl moves 3 cm?". If it does, that failure mode is invisible at the
  trained bowl pose but guaranteed to bite at TA eval time (where bowl is
  randomized). That's why the bowl-OOD probe is non-negotiable.
</div>

<h2>8. File inventory — exactly what we built (and what's gitignored)</h2>
<figure>
  <img alt="File inventory of commit 6a2d220" src="data:image/png;base64,{img_files}" />
  <figcaption>Figure 8 — 15 files in commit <code>6a2d220</code>, +2469 lines.
  The grey-marked rows are gitignored (STL meshes, ~15 MB) or stale-reference
  (the upstream HF config examples).</figcaption>
</figure>

<h3>Validated facts (so you don't have to retest)</h3>
<table>
  <tr><th>Item</th><th>Status</th><th>Source</th></tr>
  <tr>
    <td>URDF loads via placo + 6 joints match SO-101</td>
    <td>✅ <code>joints = [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]</code></td>
    <td>FK smoke test, today</td>
  </tr>
  <tr>
    <td>FK at home pose</td>
    <td>✅ <code>(0.391, 0.0, 0.226) m</code></td>
    <td>FK smoke test</td>
  </tr>
  <tr>
    <td>SmolVLA import survives <code>transformers 4.57.6 → 5.3.0</code></td>
    <td>✅ at import level (end-to-end inference UNTESTED)</td>
    <td>pre-flight Test 2</td>
  </tr>
  <tr>
    <td>HF auth as <code>osammotg1</code></td>
    <td>✅</td>
    <td>pre-flight Test 3</td>
  </tr>
  <tr>
    <td>FK conversion of 16 yellow episodes → EE space</td>
    <td>✅ EE range X∈[0.07, 0.36] m, Y∈[-0.14, 0.17] m, Z∈[-0.03, 0.21] m</td>
    <td><code>outputs/datasets/projet3-hilserl-yellow-v1-converted/conversion_report.md</code></td>
  </tr>
  <tr>
    <td>EE deltas at 30 fps stay under 20 mm / frame</td>
    <td>✅ 0% of frames exceed the <code>end_effector_step_sizes</code> clamp</td>
    <td>same report</td>
  </tr>
  <tr>
    <td><code>SO101Follower</code> exists in lerobot 0.5.2</td>
    <td>✅ (unified <code>so_follower</code> module; no separate EE class needed)</td>
    <td>lerobot probe, today</td>
  </tr>
  <tr>
    <td>SO-101 EE robot class (per HF doc)</td>
    <td>⚠️ Class doesn't exist; use <code>SO101Follower</code> + IK processor pipeline</td>
    <td>HF doc is stale; lerobot 0.5.2 reality</td>
  </tr>
</table>

<h3>Open questions (deferred to the lab)</h3>
<ul>
  <li><strong>D4 — Learner location.</strong> 5090 over SSH/gRPC vs. Mac MPS.
      Resolution: run <code>ping</code> / <code>iperf3</code> probe from the
      lab; if median RTT &lt; 5 ms, keep learner on 5090.</li>
  <li><strong>USB enumeration on Mac.</strong> The ports
      <code>/dev/tty.usbmodem5B141128171</code> (leader) and
      <code>/dev/tty.usbmodem5B141129871</code> (follower) baked into our
      configs may swap after a reboot. Test 4 of pre-flight confirms.</li>
  <li><strong>SmolVLA end-to-end after transformers upgrade.</strong> Import
      works; <code>bash deploy/infer_smolvla.sh</code> with the arm is the real
      test, untested yet.</li>
  <li><strong>Top camera.</strong> Playbook claims +10-20 pp on SmolVLA from a
      top cam. Open question for the teammate at the robot.</li>
</ul>

<h2>9. The minimum-viable path from now to a deployed policy</h2>

<p>Once the user types <code>ls /dev/tty.usbmodem*</code> and confirms two
ports show up, the rest is a sequence of well-defined commands from
<code>notes/hilserl_lab_checklist.md</code>:</p>

<pre style="background:#f4f4f4; padding:1rem; border-radius:6px; font-size:13px; line-height:1.4;">
# At the robot
PY=/Users/admin/miniforge3/bin/python3.13

# 1. find EE bounds (drive leader through workspace)
$PY -m lerobot.find_joint_limits \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5B141129871 \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodem5B141128171
# → paste printed bounds into sim/hilserl/configs/env_config_so101.json

# 2. record 15 yellow demos
bash teleop/record_hilserl_demos.sh

# 3. pick the wrist-image ROI
$PY -m lerobot.rl.crop_dataset_roi --repo-id osammotg1/projet3-hilserl-yellow-v1
# → paste printed crop into both env_config + train_config

# 4. on the 5090 (SSH): start the learner
$PY -m lerobot.rl.learner --config_path sim/hilserl/configs/train_config_hilserl_so101.json

# 5. on the Mac (separate terminal): start the actor + intervene
$PY -m lerobot.rl.actor --config_path sim/hilserl/configs/train_config_hilserl_so101.json

# 6. eval (Day 3)
#    5 primary rollouts + 5 bowl-OOD rollouts → submit best policy
</pre>

<div class="keytake">
  <strong>The whole HIL-SERL apparatus comes down to those 6 commands</strong>
  once the configs are in place. Everything we built today existed so that
  these commands would actually work when you run them — the URDF, the
  IK config, the env JSON, the train JSON, the shared teleop plumbing,
  the conversion script, the handoff doc.
</div>

<h2>10. Pointers</h2>
<ul>
  <li><strong>Design doc:</strong> <code>notes/hilserl_eval2_plan.md</code></li>
  <li><strong>Lab checklist:</strong> <code>notes/hilserl_lab_checklist.md</code></li>
  <li><strong>5090 handoff:</strong> <code>notes/HANDOFF_5090_HILSERL_DATASET_CONVERSION.md</code></li>
  <li><strong>Config reference:</strong> <code>sim/hilserl/README.md</code></li>
  <li><strong>HIL-SERL textbook walkthrough:</strong> <code>notes/visualizations/hilserl_walkthrough.html</code> (sister doc)</li>
  <li><strong>SmolVLA walkthrough:</strong> <code>notes/visualizations/smolvla_walkthrough.html</code> (the parallel BC track)</li>
</ul>

<p class="meta">Generated by
<code>notes/visualizations/generate_hilserl_implementation_walkthrough.py</code>
· covers commit <code>6a2d220</code> · session date 2026-05-17</p>

</body>
</html>
"""


def main():
    out_dir = Path(__file__).parent
    print("Generating HIL-SERL implementation walkthrough figures...", flush=True)
    figs = {
        "img_scene":           fig_eval2_scene(),
        "img_simplification":  fig_simplification_ladder(),
        "img_arch":            fig_architecture(),
        "img_flow":            fig_data_flow(),
        "img_states":          fig_state_machine(),
        "img_tracks":          fig_parallel_tracks(),
        "img_decisions":       fig_decision_tree(),
        "img_files":           fig_file_inventory(),
    }
    encoded = {name: b64_png(fig) for name, fig in figs.items()}
    html = HTML_TEMPLATE.format(**encoded)
    out_path = out_dir / "hilserl_implementation_walkthrough.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
