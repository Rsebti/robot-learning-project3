"""BC pretrain — MLP on Federico's teleop dataset (no camera input).

Trains a state-only Gaussian-MLP policy to imitate the teleop demos.
Output checkpoint is structurally compatible with rsl_rl's ActorCritic
so it can later be loaded as the PPO actor warmstart.

Usage (from the conda ``lerobot`` env on the fixed PC):

    & "C:\\Users\\user\\anaconda3\\envs\\lerobot\\python.exe" \
        -m sim.eval2.bc.train_mlp_bc \
        --epochs 30 --batch_size 256 --lr 3e-4

Or with a fresh PowerShell after ``conda init``:
    conda activate lerobot
    cd C:\\Users\\user\\Desktop\\MA2\\robot-learning-project3
    python -m sim.eval2.bc.train_mlp_bc --epochs 30

Output:
    outputs/bc_eval2/<timestamp>/
        ├── policy.pt          full state_dict (model + norm buffers)
        ├── meta.json          hyperparams + final metrics + obs/action layout
        └── train_log.csv      per-epoch loss and per-joint MAE in degrees
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from sim.eval2.bc.dataset_eval2 import (
    ACTION_DIM,
    BOWL_XYZ_TELEOP,
    Eval2BCDataset,
    NUM_COLORS,
    OBS_DIM,
    TASK_INDEX_TO_COLOR,
)
from sim.eval2.bc.policy_mlp import MLPGaussianPolicy

JOINT_NAMES = (
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BC pretrain MLP on Eval 2 teleop demos.")
    p.add_argument("--repo_id", type=str, default="osammotg1/projet3-eval2-v1")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--init_noise_std",
        type=float,
        default=0.5,
        help="log_std init — matches PPO's init_noise_std for warmstart.",
    )
    p.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[256, 128, 64],
        help="MLP hidden sizes — must match rsl_rl actor for warmstart.",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--output_root",
        type=str,
        default="outputs/bc_eval2",
        help="Output directory root (one timestamped subdir per run).",
    )
    return p.parse_args()


def per_joint_mae_deg(model: MLPGaussianPolicy, loader: DataLoader, device: str) -> tuple[float, list[float]]:
    """Returns (overall_mae_deg, per_joint_mae_deg list of 6)."""
    model.eval()
    abs_errors_sum = torch.zeros(ACTION_DIM, device=device)
    n_samples = 0
    with torch.no_grad():
        for obs, action in loader:
            obs = obs.to(device, non_blocking=True)
            action = action.to(device, non_blocking=True)
            pred = model(obs)  # in degrees (data units)
            abs_errors_sum += (pred - action).abs().sum(dim=0)
            n_samples += obs.shape[0]
    per_joint = (abs_errors_sum / max(n_samples, 1)).cpu().tolist()
    overall = sum(per_joint) / len(per_joint)
    model.train()
    return overall, per_joint


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"[bc] device={args.device}")
    print(f"[bc] repo_id={args.repo_id}")
    print(f"[bc] loading dataset + episode-level train/val split (val={args.val_fraction})")
    train_ds, val_ds = Eval2BCDataset.make_train_val_split(
        repo_id=args.repo_id, val_fraction=args.val_fraction, seed=args.seed
    )
    print(f"[bc] train frames={len(train_ds)} | val frames={len(val_ds)}")
    print(
        f"[bc] obs layout: joint_pos[6] + color_oh[{NUM_COLORS}] + bowl_xyz[3] = {OBS_DIM}D"
    )
    print(
        f"[bc] task_index -> color: " + ", ".join(f"{i}={c}" for i, c in TASK_INDEX_TO_COLOR.items())
    )
    print(f"[bc] bowl_xyz placeholder for BC = {BOWL_XYZ_TELEOP}")

    # Normalization stats from the TRAIN split only (no val leakage).
    stats = train_ds.compute_normalization()
    print(f"[bc] obs_mean (joints) = {stats.obs_mean[:6].tolist()}")
    print(f"[bc] obs_std  (joints) = {stats.obs_std[:6].tolist()}")
    print(f"[bc] action_mean       = {stats.action_mean.tolist()}")
    print(f"[bc] action_std        = {stats.action_std.tolist()}")

    model = MLPGaussianPolicy(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dims=list(args.hidden_dims),
        init_noise_std=args.init_noise_std,
        obs_mean=stats.obs_mean,
        obs_std=stats.obs_std,
        action_mean=stats.action_mean,
        action_std=stats.action_std,
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[bc] MLP params: {n_params:,} (hidden_dims={args.hidden_dims})")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = nn.SmoothL1Loss()  # Huber — robust to outliers (e.g. gripper jumps)

    # Output dir.
    timestamp = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.output_root) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[bc] output dir = {out_dir}")

    log_path = out_dir / "train_log.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "epoch",
        "train_loss",
        "train_mae_deg_overall",
        "val_mae_deg_overall",
        *[f"val_mae_deg_{j}" for j in JOINT_NAMES],
    ])
    log_file.flush()

    best_val_mae = float("inf")
    print("[bc] training...")
    for epoch in range(1, args.epochs + 1):
        # ----- train -----
        model.train()
        train_loss_sum = 0.0
        n_train_batches = 0
        for obs, action in train_loader:
            obs = obs.to(args.device, non_blocking=True)
            action = action.to(args.device, non_blocking=True)

            # Regress in NORMALIZED action space — much better gradient
            # scaling across joints with very different ranges (gripper
            # range ~30 deg, shoulder range ~150 deg).
            pred_norm = model.predict_normalized(obs)
            target_norm = (action - model.action_mean) / model.action_std.clamp(min=1e-6)
            loss = loss_fn(pred_norm, target_norm)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            n_train_batches += 1

        train_loss = train_loss_sum / max(n_train_batches, 1)
        train_mae_overall, _ = per_joint_mae_deg(model, train_loader, args.device)
        val_mae_overall, val_mae_per_joint = per_joint_mae_deg(model, val_loader, args.device)

        log_writer.writerow([
            epoch, f"{train_loss:.6f}",
            f"{train_mae_overall:.4f}", f"{val_mae_overall:.4f}",
            *[f"{m:.4f}" for m in val_mae_per_joint],
        ])
        log_file.flush()

        per_joint_str = " ".join(
            f"{name}={m:.2f}" for name, m in zip(JOINT_NAMES, val_mae_per_joint)
        )
        is_best = val_mae_overall < best_val_mae
        best_marker = "  *BEST*" if is_best else ""
        print(
            f"[bc] epoch {epoch:>3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | train_mae={train_mae_overall:.2f}° | "
            f"val_mae={val_mae_overall:.2f}° | {per_joint_str}{best_marker}",
            flush=True,
        )

        if is_best:
            best_val_mae = val_mae_overall
            torch.save({
                "model_state_dict": model.state_dict(),
                "obs_dim": OBS_DIM,
                "action_dim": ACTION_DIM,
                "hidden_dims": list(args.hidden_dims),
                "init_noise_std": args.init_noise_std,
                "epoch": epoch,
                "val_mae_overall_deg": val_mae_overall,
                "val_mae_per_joint_deg": dict(zip(JOINT_NAMES, val_mae_per_joint)),
            }, out_dir / "policy.pt")

    log_file.close()

    # Final meta.json (run config + best metric).
    meta = {
        "repo_id": args.repo_id,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "init_noise_std": args.init_noise_std,
        "hidden_dims": list(args.hidden_dims),
        "device": args.device,
        "obs_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "obs_layout": "joint_pos[0:6] + color_oh[6:12] + bowl_xyz[12:15]",
        "task_index_to_color": TASK_INDEX_TO_COLOR,
        "bowl_xyz_teleop_placeholder": list(BOWL_XYZ_TELEOP),
        "best_val_mae_deg": best_val_mae,
        "n_train_frames": len(train_ds),
        "n_val_frames": len(val_ds),
        "n_params": n_params,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[bc] DONE. best val_mae = {best_val_mae:.3f}° (checkpoint at {out_dir / 'policy.pt'})")


if __name__ == "__main__":
    main()
