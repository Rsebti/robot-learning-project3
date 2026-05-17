#!/usr/bin/env python3
"""
SmolVLA walkthrough — generates a self-contained HTML page that explains
how SmolVLA solves the Eval 2 targeted pick-and-place task and why it
should beat the stock ACT we trained tonight.

Run:
    /Users/admin/miniforge3/bin/python generate_smolvla_walkthrough.py

Output:
    smolvla_walkthrough.html  (open in any browser, fully self-contained)
"""

import base64
import io
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---- consistent palette ---------------------------------------------------
COLOR_INPUT_IMG = "#FFD7B5"
COLOR_INPUT_PROP = "#FFE4A8"
COLOR_INPUT_TEXT = "#FFC9E0"
COLOR_ENCODER = "#C5DCFF"
COLOR_FUSION = "#D4C5F9"
COLOR_OUTPUT = "#C8F0CD"
COLOR_ARROW = "#3A3A3A"
COLOR_ACT = "#E74C3C"
COLOR_SMOLVLA = "#2E86DE"
FONT = {"family": "DejaVu Sans"}


def b64_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    s = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return s


def add_box(ax, xy, w, h, label, fc, fontsize=10, fontweight="normal"):
    """Rounded box with centered text."""
    box = FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
        fc=fc, ec="black", lw=1.4,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + w / 2, xy[1] + h / 2, label,
        ha="center", va="center",
        fontsize=fontsize, fontweight=fontweight, **FONT,
    )
    return box


def add_arrow(ax, start, end, color=COLOR_ARROW, lw=1.8, style="->"):
    ar = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=18,
        color=color, lw=lw, shrinkA=4, shrinkB=4,
    )
    ax.add_patch(ar)
    return ar


# ---- Figure 1: SmolVLA architecture --------------------------------------
def fig_smolvla_arch():
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    ax.set_title(
        "SmolVLA — Three Input Streams Fused via Cross-Attention",
        fontsize=14, fontweight="bold", pad=15,
    )

    # Inputs (left column)
    add_box(ax, (0.3, 4.7), 2.4, 1.0, "Wrist image\n640 × 480 × 3", COLOR_INPUT_IMG, fontsize=10)
    add_box(ax, (0.3, 3.0), 2.4, 1.0, "Proprioception\n6 joint angles", COLOR_INPUT_PROP, fontsize=10)
    add_box(ax, (0.3, 1.3), 2.4, 1.0, "Task text\n\"Pick yellow block...\"", COLOR_INPUT_TEXT, fontsize=10)

    # Encoders (middle column)
    add_box(ax, (3.6, 4.7), 2.8, 1.0, "Vision encoder\n(SigLIP-style)", COLOR_ENCODER)
    add_box(ax, (3.6, 3.0), 2.8, 1.0, "Proprio MLP", COLOR_ENCODER)
    add_box(ax, (3.6, 1.3), 2.8, 1.0, "Text encoder\n(SmolLM-base)", COLOR_ENCODER)

    # Fusion (cross-attention)
    add_box(ax, (7.4, 2.7), 3.1, 1.6,
            "Cross-attention\ntransformer\n(fuses all 3 streams)",
            COLOR_FUSION, fontsize=11, fontweight="bold")

    # Action head + output
    add_box(ax, (11.0, 3.0), 1.7, 1.0, "Action\nhead", COLOR_OUTPUT)

    # Arrows from inputs to encoders
    add_arrow(ax, (2.7, 5.2), (3.6, 5.2))
    add_arrow(ax, (2.7, 3.5), (3.6, 3.5))
    add_arrow(ax, (2.7, 1.8), (3.6, 1.8))

    # Arrows from encoders to fusion
    add_arrow(ax, (6.4, 5.2), (7.4, 3.9))
    add_arrow(ax, (6.4, 3.5), (7.4, 3.5))
    add_arrow(ax, (6.4, 1.8), (7.4, 3.1))

    # Arrow from fusion to action head
    add_arrow(ax, (10.5, 3.5), (11.0, 3.5))

    # Final output text
    ax.text(11.85, 2.4,
            "Action chunk\n[θ₁..θ₆] × N steps\n(~3.3 s ahead)",
            ha="center", va="top", fontsize=9, fontstyle="italic")

    # Annotation: the key win
    ax.annotate(
        "Text encoder is the channel\nACT does not have.\nThis is why SmolVLA can read\n\"pick yellow\" — and ACT cannot.",
        xy=(5.0, 1.8), xytext=(7.5, 0.5),
        fontsize=9, color=COLOR_SMOLVLA, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COLOR_SMOLVLA, lw=1.3),
        ha="center", bbox=dict(boxstyle="round,pad=0.4", fc="#EAF4FF", ec=COLOR_SMOLVLA),
    )

    # Param count caption
    ax.text(6.5, 0.05, "~450M parameters total · pretrained on multi-task SO-100/SO-101 corpus + open trajectories",
            ha="center", fontsize=9, color="#555", fontstyle="italic")
    return fig


# ---- Figure 2: ACT vs SmolVLA side-by-side -------------------------------
def fig_act_vs_smolvla():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Why ACT plateaus at 50% on 2-cube tasks — and why SmolVLA shouldn't",
                 fontsize=14, fontweight="bold")

    # --- ACT (left) ---
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("ACT (what we trained tonight)", fontsize=12, color=COLOR_ACT, fontweight="bold")

    add_box(ax, (0.3, 4.5), 2.8, 0.9, "Wrist image", COLOR_INPUT_IMG)
    add_box(ax, (0.3, 3.1), 2.8, 0.9, "Proprio", COLOR_INPUT_PROP)
    # ghosted text input (not connected)
    add_box(ax, (0.3, 1.6), 2.8, 0.9, "Task text\n(stored in dataset only)", "#FFEEEE")
    # ACT body
    add_box(ax, (4.4, 3.0), 3.2, 1.6, "ACT transformer\n(vision + proprio)", COLOR_FUSION)
    # Output
    add_box(ax, (8.0, 3.2), 1.6, 1.0, "Action\nchunk", COLOR_OUTPUT)

    add_arrow(ax, (3.1, 4.9), (4.4, 4.0))
    add_arrow(ax, (3.1, 3.5), (4.4, 3.5))
    # Red X on text-to-model arrow
    ax.plot([3.1, 4.4], [2.0, 3.0], color=COLOR_ACT, lw=2.5, linestyle=":", alpha=0.6)
    ax.text(3.75, 2.55, "✗", fontsize=22, color=COLOR_ACT, ha="center", va="center", fontweight="bold")
    ax.text(3.75, 1.05, "NEVER REACHES MODEL",
            fontsize=9, color=COLOR_ACT, ha="center", fontweight="bold")
    add_arrow(ax, (7.6, 3.7), (8.0, 3.7))

    # Result
    ax.text(5.0, 0.2,
            "Result: policy averages over the dataset's color choices.\nIn a 2-cube scene, target-color obedience ≈ 50% (random).",
            ha="center", fontsize=9.5, color="#222",
            bbox=dict(boxstyle="round,pad=0.4", fc="#FFE2E2", ec=COLOR_ACT))

    # --- SmolVLA (right) ---
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("SmolVLA (what we want to train next)",
                 fontsize=12, color=COLOR_SMOLVLA, fontweight="bold")

    add_box(ax, (0.3, 4.5), 2.8, 0.9, "Wrist image", COLOR_INPUT_IMG)
    add_box(ax, (0.3, 3.1), 2.8, 0.9, "Proprio", COLOR_INPUT_PROP)
    add_box(ax, (0.3, 1.6), 2.8, 0.9, "Task text\n\"Pick yellow block...\"", COLOR_INPUT_TEXT)
    add_box(ax, (4.4, 2.6), 3.2, 2.0, "SmolVLA\n(vision + proprio + text\nfused via cross-attn)", COLOR_FUSION)
    add_box(ax, (8.0, 3.2), 1.6, 1.0, "Action\nchunk", COLOR_OUTPUT)

    add_arrow(ax, (3.1, 4.9), (4.4, 4.1))
    add_arrow(ax, (3.1, 3.5), (4.4, 3.5))
    add_arrow(ax, (3.1, 2.0), (4.4, 2.9), color=COLOR_SMOLVLA, lw=2.4)
    ax.text(3.75, 2.45, "✓", fontsize=22, color=COLOR_SMOLVLA, ha="center", va="center", fontweight="bold")
    add_arrow(ax, (7.6, 3.7), (8.0, 3.7))

    ax.text(5.0, 0.2,
            "Result: policy reads the instruction.\nExpected target-color obedience: 70–90% in-distribution.",
            ha="center", fontsize=9.5, color="#222",
            bbox=dict(boxstyle="round,pad=0.4", fc="#E2F0FF", ec=COLOR_SMOLVLA))

    plt.tight_layout()
    return fig


# ---- Figure 3: The two-cube decision -------------------------------------
def fig_two_cube_decision():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("The 2-cube decision: what each policy actually \"sees\"",
                 fontsize=14, fontweight="bold")

    def draw_scene(ax, title, title_color):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=12, color=title_color, fontweight="bold")
        ax.axis("off")
        # table
        table = mpatches.FancyBboxPatch((0.5, 1.0), 9.0, 3.0,
                                         boxstyle="round,pad=0.05",
                                         fc="#E8DDD6", ec="#444", lw=1.2)
        ax.add_patch(table)
        # bowl
        bowl = mpatches.Ellipse((5.0, 1.6), 1.6, 0.6, fc="#DDD", ec="#222", lw=1.3)
        ax.add_patch(bowl)
        ax.text(5.0, 1.6, "bowl", ha="center", va="center", fontsize=8)
        # cubes
        yellow = mpatches.FancyBboxPatch((3.3, 2.7), 0.8, 0.8,
                                          boxstyle="round,pad=0.02,rounding_size=0.05",
                                          fc="#F1C40F", ec="black")
        blue = mpatches.FancyBboxPatch((5.9, 2.7), 0.8, 0.8,
                                        boxstyle="round,pad=0.02,rounding_size=0.05",
                                        fc="#3498DB", ec="black")
        ax.add_patch(yellow)
        ax.add_patch(blue)
        ax.text(3.7, 3.1, "Y", ha="center", va="center", fontweight="bold")
        ax.text(6.3, 3.1, "B", ha="center", va="center", fontweight="bold")
        # camera frustum (gripper above the scene)
        gripper = mpatches.Rectangle((4.6, 5.0), 0.8, 0.5, fc="#666", ec="#222")
        ax.add_patch(gripper)
        ax.text(5.0, 5.25, "wrist cam", ha="center", va="center", fontsize=7, color="white")

    # ACT side
    draw_scene(axes[0], "ACT sees only pixels", COLOR_ACT)
    # both arrows to both cubes with 50/50 label
    axes[0].annotate("", xy=(3.7, 3.5), xytext=(5.0, 5.0),
                     arrowprops=dict(arrowstyle="->", color=COLOR_ACT, lw=2, alpha=0.55))
    axes[0].annotate("", xy=(6.3, 3.5), xytext=(5.0, 5.0),
                     arrowprops=dict(arrowstyle="->", color=COLOR_ACT, lw=2, alpha=0.55))
    axes[0].text(2.5, 4.2, "≈50%", color=COLOR_ACT, fontsize=11, fontweight="bold")
    axes[0].text(7.0, 4.2, "≈50%", color=COLOR_ACT, fontsize=11, fontweight="bold")
    axes[0].text(5.0, 0.4,
                 "No \"pick yellow\" signal → policy guesses",
                 ha="center", fontsize=9.5, fontstyle="italic", color="#444")

    # SmolVLA side
    draw_scene(axes[1], "SmolVLA sees pixels + text", COLOR_SMOLVLA)
    axes[1].annotate("", xy=(3.7, 3.5), xytext=(5.0, 5.0),
                     arrowprops=dict(arrowstyle="->", color=COLOR_SMOLVLA, lw=3))
    axes[1].annotate("", xy=(6.3, 3.5), xytext=(5.0, 5.0),
                     arrowprops=dict(arrowstyle="->", color="#BBB", lw=1, alpha=0.45))
    axes[1].text(2.3, 4.2, "→ yellow", color=COLOR_SMOLVLA, fontsize=11, fontweight="bold")
    # show the text instruction box
    axes[1].text(5.0, 5.7, "Task: \"Pick yellow block and place in bowl at (-15.5,29.5) cm\"",
                 ha="center", fontsize=9, color=COLOR_SMOLVLA, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.4", fc="#E2F0FF", ec=COLOR_SMOLVLA))
    axes[1].text(5.0, 0.4,
                 "Text encoder routes attention to the named cube",
                 ha="center", fontsize=9.5, fontstyle="italic", color="#444")

    plt.tight_layout()
    return fig


# ---- Figure 4: Performance numbers from the playbook ---------------------
def fig_performance_numbers():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ----- Left: T1 in-distribution success on SO-100/SO-101 -----
    ax = axes[0]
    methods = ["ACT\nfrom scratch\n(SO-100 multi-task)",
               "ACT\n50 episodes\n(Karkada SO-101)",
               "ACT\n125 episodes\n(Sherry Chen)",
               "SmolVLA-base\nfinetune\n(paper)"]
    success = [48.3, 70.0, 90.0, 78.3]
    colors = [COLOR_ACT, COLOR_ACT, COLOR_ACT, COLOR_SMOLVLA]
    bars = ax.bar(methods, success, color=colors, edgecolor="black", lw=1.2)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Success rate (%)", fontsize=11)
    ax.set_title("T1 (single-block pick-place) — published numbers", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, success):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.1f}%",
                ha="center", fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_axisbelow(True)

    # ----- Right: Eval 2 (your real result) vs playbook expectations -----
    ax = axes[1]
    labels = ["ACT no-aug\n(yours)", "ACT dark-noise\n(yours)", "ACT dark-shadow\n(yours)",
              "ACT 100k\n(yours)", "SmolVLA finetune\n(expected)\nT2 paper #s"]
    real = [50, 50, 50, 50, 85]
    colors2 = [COLOR_ACT, COLOR_ACT, COLOR_ACT, COLOR_ACT, COLOR_SMOLVLA]
    bars = ax.bar(labels, real, color=colors2, edgecolor="black", lw=1.2)
    # Add error bars / range for SmolVLA expected
    ax.errorbar([4], [85], yerr=[[15], [5]], fmt="none", color="black", capsize=6, lw=1.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Target-color obedience (%)", fontsize=11)
    ax.set_title("Eval 2 — your ACT results vs SmolVLA's expected range",
                 fontsize=12, fontweight="bold")
    ax.axhline(50, color="#888", linestyle="--", lw=1, alpha=0.6)
    ax.text(0.05, 51.5, "random baseline (50%)", fontsize=8, color="#666", fontstyle="italic")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, real):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v}%",
                ha="center", fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    return fig


# ---- Figure 5: Finetune pipeline flowchart -------------------------------
def fig_finetune_pipeline():
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("How we'd actually train SmolVLA on our existing data",
                 fontsize=14, fontweight="bold", pad=15)

    steps = [
        ("1. Load\npretrained\nlerobot/smolvla_base\n(~450M params)", COLOR_INPUT_TEXT),
        ("2. Stream our\ndataset\n(101 episodes,\n6 task strings)", COLOR_INPUT_IMG),
        ("3. Finetune\n20k steps,\nbatch 64,\n~4h A100/5090", COLOR_FUSION),
        ("4. Push to HF\nosammotg1/\nprojet3-smolvla-\neval2-v1", COLOR_OUTPUT),
        ("5. Deploy via\ndeploy/infer.sh\n(POLICY_PATH=...\noverride)", COLOR_ENCODER),
    ]
    x = 0.3
    w, h = 2.5, 2.4
    centers = []
    for label, fc in steps:
        add_box(ax, (x, 0.9), w, h, label, fc, fontsize=10)
        centers.append(x + w / 2)
        x += w + 0.25

    # Connect with arrows
    for i in range(len(steps) - 1):
        add_arrow(ax,
                  (centers[i] + w / 2, 0.9 + h / 2),
                  (centers[i + 1] - w / 2, 0.9 + h / 2),
                  lw=2.2)

    # Footer caveat
    ax.text(7, 0.1,
            "⚠ Critical pre-check: confirm dataset's wrist-camera key matches lerobot/smolvla_base's expected key — "
            "camera-name mismatch is the #1 silent killer per playbook (LeRobot issue #2915).",
            ha="center", fontsize=9, color="#A04040", fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.4", fc="#FFF1E5", ec="#A04040"))

    return fig


# ---- Assemble the HTML ---------------------------------------------------
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>SmolVLA Walkthrough — Robot Learning Project 3</title>
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
      border-bottom: 3px solid #2E86DE;
      padding-bottom: 0.4rem;
      margin-bottom: 0.2rem;
    }}
    h2 {{
      color: #2E86DE;
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
      border-left: 4px solid #2E86DE;
      background: #eaf4ff;
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
    th, td {{ border: 1px solid #ddd; padding: 0.4rem 0.6rem; text-align: left; }}
    th {{ background: #f0f0f0; }}
  </style>
</head>
<body>

<h1>SmolVLA — a visual walkthrough</h1>
<p class="subtitle">Why SmolVLA is the cheapest fix for the 50% target-color ceiling we hit with ACT on Eval 2.</p>

<h2>1. The problem we're solving</h2>
<p>
  We trained 4 ACT models on the same 101-episode dataset, varying only image augmentation
  and training steps. <strong>All four plateaued at the same ~50% target-color obedience</strong>
  on the real SO-101. That's not a training-quality issue — it's random with respect to
  color choice, which is exactly what an architectural gap looks like.
</p>
<p>
  ACT in lerobot is <strong>vision-only</strong>. It receives image + proprioception, nothing
  else. The task-string field <code>single_task</code> is stored as dataset metadata and
  never reaches the model. So even though we labeled every episode with
  <code>"Pick yellow block and place in bowl at (-15.5,29.5) cm"</code>, the model could
  not condition on it.
</p>

<figure>
  <img alt="ACT vs SmolVLA side-by-side" src="data:image/png;base64,{img_act_vs_smolvla}" />
  <figcaption>Figure 1 — the same dataset feeds both models, but only one can read the task string.</figcaption>
</figure>

<div class="keytake">
  <strong>Key takeaway:</strong> the 50% ceiling is the architecture, not the data.
  Same data, different architecture (SmolVLA) is the cheapest test.
</div>

<h2>2. How SmolVLA actually works</h2>
<p>
  SmolVLA (Shukor et al. 2026, arXiv:2506.01844) is a 450M-parameter
  <strong>vision-language-action</strong> model built around three encoders that converge into
  a cross-attention transformer:
</p>
<ul>
  <li><strong>Vision encoder</strong> — SigLIP-style image embeddings. Trained on a huge
      multi-task robot-learning corpus.</li>
  <li><strong>Proprio MLP</strong> — small projection that turns joint angles into a token
      sequence the transformer can attend to.</li>
  <li><strong>Text encoder</strong> — SmolLM-base. Tokenizes natural language task strings
      and turns them into embeddings.</li>
</ul>
<p>
  The three streams meet in a cross-attention transformer that fuses them into a unified
  hidden representation, then a small action head emits chunks of joint targets
  (similar to ACT, but conditioned on all three modalities).
</p>

<figure>
  <img alt="SmolVLA architecture block diagram" src="data:image/png;base64,{img_arch}" />
  <figcaption>Figure 2 — SmolVLA's three input streams. The text-encoder pathway (pink → blue)
  is the channel ACT lacks.</figcaption>
</figure>

<p>
  Crucially, <strong>SmolVLA is pretrained</strong>. The 450M params already encode generic
  robot-learning priors from hundreds of thousands of demonstration trajectories.
  We fine-tune those weights on our 101 episodes — we are <em>not</em> training from scratch.
  This is why the recipe in the playbook is so cheap: 20k steps, batch 64, ~4 hours on an A100.
</p>

<h2>3. What this means for the 2-cube task</h2>
<p>
  In Eval 1 there was one cube. "Pick a cube" was unambiguous. ACT, despite having
  no instruction input, still hit 5/5 because every demonstration in the dataset
  pointed at the same kind of action.
</p>
<p>
  In Eval 2 there are two cubes. "Pick a cube" is now <strong>ambiguous</strong> —
  which one? The instruction is the disambiguating signal. Without it, the policy
  averages, which on a balanced 2-color dataset means ~50% by construction.
</p>

<figure>
  <img alt="Two-cube decision visualization" src="data:image/png;base64,{img_decision}" />
  <figcaption>Figure 3 — ACT splits its attention 50/50. SmolVLA routes attention to
  the named cube via the text token's cross-attention scores.</figcaption>
</figure>

<h2>4. What we should expect — published numbers</h2>
<p>
  The SmolVLA paper (Table 3 in arXiv:2506.01844) benchmarks both architectures on
  SO-100/SO-101 pick-place tasks. Community reproductions exist with substantial variance
  (Habuda, Saroha, Sawane, Kamath all report different numbers — data hygiene dominates).
</p>

<figure>
  <img alt="Performance numbers" src="data:image/png;base64,{img_perf}" />
  <figcaption>Figure 4 — Left: published T1 success on single-block pick-place from
  paper + community. Right: our actual ACT results vs the playbook's expected range for
  SmolVLA on T2 (error bars show community variance).</figcaption>
</figure>

<div class="keytake">
  <strong>Reading the right chart:</strong> our 4 ACT runs are tied at 50% because
  the task is genuinely <em>not learnable</em> for them in clutter — no matter how much
  augmentation we add. SmolVLA's expected range is 70–90% in-distribution, but the
  community variance (visible as the wide error bar) is the risk we accept.
</div>

<h2>5. How we'd actually do it</h2>
<p>
  The training script we'd send to the 5090 box looks like the standard lerobot finetune
  recipe with a different policy type. We don't change the dataset, the camera setup, the
  bowl position, or the deploy script. We change one thing: the model.
</p>

<figure>
  <img alt="Finetune pipeline" src="data:image/png;base64,{img_pipeline}" />
  <figcaption>Figure 5 — five-step pipeline. The deploy command on the lab Mac stays
  <code>bash deploy/infer.sh</code>; we override <code>POLICY_PATH</code> at runtime.</figcaption>
</figure>

<h2>6. Risks worth naming</h2>
<table>
  <tr>
    <th>Risk</th>
    <th>What goes wrong</th>
    <th>Mitigation</th>
  </tr>
  <tr>
    <td>Camera-name mismatch</td>
    <td>SmolVLA pretraining expects specific camera keys (top + wrist). Our dataset uses
        <code>wrist</code> only. If the key naming differs, pretrained visual weights don't
        load and we're effectively training from scratch.</td>
    <td>Check <code>lerobot/smolvla_base</code> model card before training. Rename camera
        in features mapping if needed (no dataset mutation).</td>
  </tr>
  <tr>
    <td>Wrist-only vs top+wrist</td>
    <td>SmolVLA pretraining is dominated by top + wrist views. Wrist-only is documented
        to underperform by 10–20 percentage points.</td>
    <td>Accept this; no top camera is available on our SO-101 mid-project. Compare
        result against the lower end of the published range, not the upper.</td>
  </tr>
  <tr>
    <td>Community reproduction variance</td>
    <td>LeRobot issue #2915: 0% with 120 SO-101 episodes. Same recipe, different result —
        usually a data-hygiene issue.</td>
    <td>Validate single-batch inference output before committing to full training.</td>
  </tr>
  <tr>
    <td>Blackwell sm_120 on the 5090</td>
    <td>PyTorch wheel mismatch can cause CUDA kernel errors on training.</td>
    <td>Pin <code>torch==2.9.1+cu128</code> if needed (CLAUDE.md flags this).</td>
  </tr>
</table>

<h2>7. The decision this visualization is meant to inform</h2>
<p>
  The playbook recommends two ways to add color conditioning to ACT-class policies:
  <strong>SmolVLA finetune (cheap, no surgery)</strong> or
  <strong>FiLM-conditioned ACT (more invasive)</strong>. SmolVLA is week-1 priority
  because the marginal cost is one Brev H100 finetune (~$8) and the upside is a usable
  primitive for HIL-SERL refinement in weeks 2–4.
</p>
<p>
  If SmolVLA still misses target color (down at the wrist-only-failed end of the range),
  the next escalation is FiLM-ACT. That's a separate experiment with separate risk profile —
  but it would happen only after this one definitively rules out language conditioning.
</p>

<p class="meta">Generated by <code>notes/visualizations/generate_smolvla_walkthrough.py</code></p>

</body>
</html>
"""


def main():
    out_dir = Path(__file__).parent
    print("Generating figures...", flush=True)

    figs = {
        "img_arch": fig_smolvla_arch(),
        "img_act_vs_smolvla": fig_act_vs_smolvla(),
        "img_decision": fig_two_cube_decision(),
        "img_perf": fig_performance_numbers(),
        "img_pipeline": fig_finetune_pipeline(),
    }
    encoded = {name: b64_png(fig) for name, fig in figs.items()}

    html = HTML_TEMPLATE.format(**encoded)
    out_path = out_dir / "smolvla_walkthrough.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
