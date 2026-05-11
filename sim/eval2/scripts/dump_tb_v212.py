"""Dump V2.12 TensorBoard scalars to a markdown file consumable by an LLM.

Reads the events file from the training run and writes a comprehensive
dump (per-iteration table + context header + auto-detected milestones)
to ``notes/v212_training_dump.md``. The output is meant to be uploaded
to another LLM as a single text artifact instead of TB screenshots.

Usage:
    cd C:\\Users\\user\\Desktop\\MA2\\isaac\\isaac_so_arm101
    .\\.venv\\Scripts\\Activate.ps1
    python -m sim.eval2.scripts.dump_tb_v212

The script auto-detects the most recent run under
``logs/rsl_rl/lift_v2_12/``.
"""
from __future__ import annotations

from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator


LOG_ROOT = Path(r"C:\Users\user\Desktop\MA2\isaac\isaac_so_arm101\logs\rsl_rl\lift_v2_12")
OUT_FILE = Path(r"C:\Users\user\Desktop\MA2\robot-learning-project3\notes\v212_training_dump.md")


def latest_run_dir() -> Path:
    runs = sorted(LOG_ROOT.iterdir(), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise SystemExit(f"No runs in {LOG_ROOT}")
    return runs[-1]


def main() -> None:
    run_dir = latest_run_dir()
    print(f"[INFO] Reading events from: {run_dir}")

    ea = event_accumulator.EventAccumulator(
        str(run_dir),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()

    all_tags = ea.Tags()["scalars"]

    groups = {
        "Train": ["Train/mean_reward", "Train/mean_episode_length"],
        "Episode_Reward (positives)": [
            "Episode_Reward/reaching_object",
            "Episode_Reward/grasping_cube",
            "Episode_Reward/lifting_object",
            "Episode_Reward/object_goal_tracking",
            "Episode_Reward/object_goal_tracking_fine_grained",
            "Episode_Reward/success_bonus",
        ],
        "Episode_Reward (negatives / regularizers)": [
            "Episode_Reward/action_rate",
            "Episode_Reward/joint_vel",
            "Episode_Reward/ee_to_cube_distance",
            "Episode_Reward/cube_dropped_penalty",
        ],
        "Episode_Termination": [
            "Episode_Termination/success",
            "Episode_Termination/cube_dropped",
            "Episode_Termination/ee_far_from_cube",
            "Episode_Termination/time_out",
        ],
        "Loss": [
            "Loss/learning_rate",
            "Loss/entropy",
            "Loss/value_function",
            "Loss/surrogate",
        ],
        "Policy": ["Policy/mean_noise_std"],
        "Metrics": [
            "Metrics/object_pose/position_error",
            "Metrics/object_pose/orientation_error",
        ],
    }

    # Determine the canonical iter set from `Train/mean_reward` (which is
    # logged against iteration count). Tags ending with `/time` use
    # cumulative timesteps as their step value and would corrupt the iter
    # universe — exclude them.
    iter_tags = [t for t in all_tags if not t.endswith("/time")]
    canonical_iters: set[int] = set()
    if "Train/mean_reward" in iter_tags:
        canonical_iters = {ev.step for ev in ea.Scalars("Train/mean_reward")}
    else:
        # fallback: use any non-/time tag's steps
        canonical_iters = {
            ev.step for t in iter_tags for ev in ea.Scalars(t)
        }
    max_iter_canonical = max(canonical_iters)

    # Build iter -> tag -> value, only keeping iter-aligned tags
    data: dict[int, dict[str, float]] = {}
    for t in iter_tags:
        for ev in ea.Scalars(t):
            if ev.step in canonical_iters:
                data.setdefault(ev.step, {})[t] = ev.value

    iters = sorted(data.keys())
    last_iter = max(iters)
    print(
        f"[INFO] {len(iter_tags)} iter-aligned tags, "
        f"{len(iters)} iters logged, max iter = {last_iter}"
    )
    if last_iter != max_iter_canonical:
        print(
            f"[WARN] last_iter ({last_iter}) differs from canonical "
            f"({max_iter_canonical})"
        )

    # Sample iters: every iter for first 50, every 5 from 50-100,
    # every 10 from 100-200, every 20 from 200+
    sample_iters: list[int] = []
    for it in iters:
        if it <= 50:
            sample_iters.append(it)
        elif it <= 100 and it % 5 == 0:
            sample_iters.append(it)
        elif it <= 200 and it % 10 == 0:
            sample_iters.append(it)
        elif it % 20 == 0:
            sample_iters.append(it)
    if iters[-1] not in sample_iters:
        sample_iters.append(iters[-1])

    def _fmt(v: float | None) -> str:
        if v is None:
            return "-"
        if abs(v) >= 100:
            return f"{v:.1f}"
        if abs(v) >= 1:
            return f"{v:.3f}"
        if abs(v) >= 0.001:
            return f"{v:.4f}"
        return f"{v:.2e}"

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        f.write(f"# V2.12 Training Data Dump - iter 0 to {last_iter}\n\n")
        f.write(
            f"> Run: `lift_v2_12/{run_dir.name}` "
            f"({len(iters)} iterations logged, max_iter={last_iter}/1500)\n"
        )
        f.write(
            "> Generated for LLM consumption - full per-iteration "
            "TensorBoard scalars (no screenshots needed).\n\n"
        )

        f.write("## Training context (for the LLM reading this)\n\n")
        f.write(
            "PPO training of a SO-101 robot arm in Isaac Lab simulation. "
            "Task: lift a 2cm cube to a goal pose specified in robot root frame.\n\n"
        )
        f.write("**Stack V2.12 :**\n\n")
        f.write(
            "- **Action**: `RelativeJointPositionActionCfg(scale=0.20, "
            "use_zero_offset=True)` - DELTA control, vmax mechanical = "
            "0.20 / (1/30s) = 6 rad/s = Feetech STS3215 limit. Per-step "
            "joint delta capped, no chaotic yeet possible.\n"
        )
        f.write(
            "- **Observation (Visual variant, 555D)**: joint_pos(6) + "
            "joint_vel(6) + object_pos_in_root(3) + target_obj_pose(7) + "
            "last_action(6) + wrist_features_resnet18(512) + "
            "target_color_zero_placeholder(6) + bowl_xyz_zero_placeholder(3) "
            "+ ee_to_cube_vec(3) + cube_to_goal_vec(3)\n"
        )
        f.write("- **Reward shaping**:\n")
        f.write(
            "  - reaching_object: w=+1.0, `1 - tanh(d/0.15)` "
            "(d = ||ee_target - cube||)\n"
        )
        f.write(
            "  - grasping_cube: w=+5.0, binary "
            "(jaw<4cm AND gripper<0.26 rad)\n"
        )
        f.write(
            "  - lifting_object: w=+10.0, gated by "
            "`grasp AND (cube_z - base_z > 0.08m)`\n"
        )
        f.write(
            "  - object_goal_tracking: w=+16.0, gated by `grasp AND lift`, "
            "std=0.30\n"
        )
        f.write(
            "  - object_goal_tracking_fine_grained: w=+5.0, same gating, "
            "std=0.05\n"
        )
        f.write(
            "  - success_bonus: w=+2500.0, terminal on cube_reached_goal "
            "(3D dist <5cm to commanded goal). Math: at hover-near-goal, "
            "discounted future dense reward over 100 steps with gamma=0.99 "
            "is approx +1900, so success_bonus must exceed this to "
            "incentivize finishing.\n"
        )
        f.write(
            "  - ee_to_cube_distance: w=-1.0, raw `||ee_target - cube||` "
            "in meters (V2.10c innovation: provides constant -1/m gradient "
            "even at large d where tanh saturates).\n"
        )
        f.write(
            "  - action_rate_l2: w=-1e-4 (Isaac Lab default noise floor; "
            "DELTA action class already caps velocity, so this just "
            "discourages chatter).\n"
        )
        f.write("  - joint_vel_l2: w=-1e-4 (same rationale).\n")
        f.write(
            "  - cube_dropped_penalty: w=-50.0 (V2.12 fix: a typical "
            "Touch-and-Yeet trajectory accumulates ~+40 reward briefly "
            "(reach +5, grasp 3*5, lift 2*10), so penalty must be > 40 "
            "to keep yeet net-negative).\n"
        )
        f.write(
            "- **Termination**: time_out (5s = 150 steps), "
            "cube_reached_goal (success), cube_dropped (world_z<0.04m), "
            "ee_far_from_cube (||ee - cube|| > 0.5m, V2.12 fail-fast).\n"
        )
        f.write(
            "- **PPO**: init_noise_std=1.0, entropy_coef=0.005, "
            "value_loss_coef=1.0, n_epochs=5, n_mini_batches=4, "
            "lr=1e-3, schedule=adaptive, desired_kl=0.02 (relaxed from "
            "0.01 to tolerate small-batch KL noise), gamma=0.99, "
            "max_grad_norm=1.0, hidden_dims=[256,128,128].\n"
        )
        f.write(
            "- **Scene**: 256 envs (Visual), 30 Hz control, episode 5s "
            "= 150 steps, cube spawn random x +/-7.5cm, y +/-7.5cm, "
            "yaw +/-30deg.\n\n"
        )

        f.write("## How to interpret Episode_Reward values\n\n")
        f.write(
            "In rsl_rl logs, `Episode_Reward/<term>` = "
            "`mean(sum_of_weighted_term_per_episode) / max_episode_length_s`. "
            "With `max_episode_length_s = 5`:\n\n"
        )
        f.write("- `sum_per_ep = Episode_Reward * 5`\n")
        f.write(
            "- For reaching_object (w=1, max=1/step): per-step value = "
            "sum_per_ep / 150 = Episode_Reward / 30. Then "
            "`d = atanh(1 - per_step) * 0.15` gives avg EE-cube distance.\n"
        )
        f.write(
            "- For grasping_cube (w=5, binary): "
            "n_grasp_steps_per_ep = sum_per_ep / 5 = Episode_Reward\n"
        )
        f.write(
            "- For lifting_object (w=10, binary): "
            "n_lift_steps_per_ep = sum_per_ep / 10 = Episode_Reward / 2\n"
        )
        f.write(
            "- For ee_to_cube_distance (w=-1, raw distance in meters): "
            "per-step distance = -Episode_Reward / 30 in meters\n"
        )
        f.write(
            "- For success_bonus (w=2500, sparse terminal): "
            "n_success_per_ep = sum_per_ep / 2500 = Episode_Reward / 500. "
            "If Episode_Reward = 1.0, that's 0.002 = 0.2% success rate.\n\n"
        )

        f.write("## Per-iteration metrics\n\n")
        for group_name, tags_in_group in groups.items():
            f.write(f"### {group_name}\n\n")
            f.write("| iter |")
            for t in tags_in_group:
                short = t.split("/")[-1]
                f.write(f" {short} |")
            f.write("\n|---|")
            for _ in tags_in_group:
                f.write("---|")
            f.write("\n")
            for it in sample_iters:
                if it not in data:
                    continue
                f.write(f"| {it} |")
                for t in tags_in_group:
                    f.write(f" {_fmt(data[it].get(t))} |")
                f.write("\n")
            f.write("\n")

        f.write("## Key milestones (auto-detected)\n\n")

        def _first_above(tag: str, threshold: float, fmt: str = ".3f") -> str:
            for it in iters:
                v = data[it].get(tag, 0)
                if v > threshold:
                    return f"iter {it} (value {v:{fmt}})"
            return "not yet"

        f.write(
            f"- **First grasping_cube > 0.1**: "
            f"{_first_above('Episode_Reward/grasping_cube', 0.1)}\n"
        )
        f.write(
            f"- **First grasping_cube > 1.0**: "
            f"{_first_above('Episode_Reward/grasping_cube', 1.0)}\n"
        )
        f.write(
            f"- **First grasping_cube > 4.0** (sustained ~4 grasp steps/ep): "
            f"{_first_above('Episode_Reward/grasping_cube', 4.0)}\n"
        )
        f.write(
            f"- **First lifting_object > 0.05**: "
            f"{_first_above('Episode_Reward/lifting_object', 0.05)}\n"
        )
        f.write(
            f"- **First lifting_object > 0.5**: "
            f"{_first_above('Episode_Reward/lifting_object', 0.5)}\n"
        )
        f.write(
            f"- **First lifting_object > 1.0**: "
            f"{_first_above('Episode_Reward/lifting_object', 1.0)}\n"
        )
        f.write(
            f"- **First object_goal_tracking > 0.5**: "
            f"{_first_above('Episode_Reward/object_goal_tracking', 0.5)}\n"
        )
        f.write(
            f"- **First success_bonus > 0**: "
            f"{_first_above('Episode_Reward/success_bonus', 0.0, '.3e')}\n"
        )
        f.write(
            f"- **First success_bonus > 1.0** (~0.04% success): "
            f"{_first_above('Episode_Reward/success_bonus', 1.0)}\n"
        )
        f.write(
            f"- **mean_reward crosses 0**: "
            f"{_first_above('Train/mean_reward', 0.0)}\n"
        )
        f.write(
            f"- **mean_reward > +10**: "
            f"{_first_above('Train/mean_reward', 10.0)}\n"
        )
        f.write(
            f"- **mean_reward > +20**: "
            f"{_first_above('Train/mean_reward', 20.0)}\n"
        )

        last_data = data[last_iter]
        f.write(f"\n### Snapshot at latest iter ({last_iter}/1500)\n\n")
        for t in [
            "Train/mean_reward",
            "Train/mean_episode_length",
            "Episode_Reward/reaching_object",
            "Episode_Reward/grasping_cube",
            "Episode_Reward/lifting_object",
            "Episode_Reward/object_goal_tracking",
            "Episode_Reward/success_bonus",
            "Episode_Reward/ee_to_cube_distance",
            "Episode_Reward/cube_dropped_penalty",
            "Episode_Termination/success",
            "Episode_Termination/cube_dropped",
            "Policy/mean_noise_std",
            "Loss/learning_rate",
        ]:
            v = last_data.get(t)
            f.write(f"- `{t}` = {_fmt(v)}\n")

        f.write("\n## Available scalar tags (for LLM reference)\n\n")
        f.write("All tags logged in this run:\n\n```\n")
        for t in sorted(all_tags):
            f.write(f"{t}\n")
        f.write("```\n")

    print(f"[OK] Wrote dump to: {OUT_FILE}")
    print(f"[OK] File size: {OUT_FILE.stat().st_size / 1024:.1f} KB")
    print(f"[OK] Iters covered: 0 to {last_iter}")


if __name__ == "__main__":
    main()
