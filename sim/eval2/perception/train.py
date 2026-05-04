"""Train the ColorBlockCNN on the dataset produced by capture_dataset.py.

Usage:
    uv run python -m sim.eval2.perception.train \\
        --data sim/eval2/perception/data.pt \\
        --epochs 30 \\
        --batch_size 64 \\
        --output sim/eval2/perception/checkpoint.pt

Loss
    Plain MSE on the 6 outputs (xyz_red + xyz_blue, in meters, robot frame).
    We also report the mean absolute error per coordinate (in cm) so the
    number is interpretable: "the model is wrong by ~1.2 cm on x_red on
    average".

Train / val split
    Random 80/20. Tracks val loss + val MAE (cm), saves the best
    checkpoint, early-stops on plateau.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .dataset import BlockPositionDataset
from .model import ColorBlockCNN, count_parameters


def parse_args():
    p = argparse.ArgumentParser(description="Train the perception CNN.")
    p.add_argument("--data", type=str, default=str(Path(__file__).parent / "data.pt"))
    p.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent / "checkpoint.pt"),
        help="Where to save the best model.",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--patience", type=int, default=8, help="Early stop after N epochs w/o val improvement.")
    return p.parse_args()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, torch.Tensor]:
    """Returns (mean MSE, per-coordinate mean absolute error in meters).

    The per-coord MAE has shape (6,) corresponding to:
    [x_red, y_red, z_red, x_blue, y_blue, z_blue] in meters.
    """
    model.eval()
    mse_sum = 0.0
    abs_err_sum = torch.zeros(6, device=device)
    n = 0
    loss_fn = nn.MSELoss(reduction="sum")
    with torch.no_grad():
        for img, target in loader:
            img = img.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            pred = model(img)
            mse_sum += loss_fn(pred, target).item()
            abs_err_sum += (pred - target).abs().sum(dim=0)
            n += img.size(0)
    mse = mse_sum / max(n * 6, 1)  # mean across all (sample, coordinate) entries
    mae_per_coord = abs_err_sum / max(n, 1)
    return mse, mae_per_coord.cpu()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Device: {device}")

    # --- load dataset and split ---
    # Use SEPARATE Dataset instances for train and val so the ``augment``
    # flag can differ (Subset shares the underlying Dataset, which would
    # leak augmentation into val).
    train_data = BlockPositionDataset(args.data, augment=True)
    val_data = BlockPositionDataset(args.data, augment=False)
    n = len(train_data)
    n_val = int(n * args.val_split)
    n_train = n - n_val
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    train_subset = Subset(train_data, indices[n_val:].tolist())
    val_subset = Subset(val_data, indices[:n_val].tolist())
    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    print(f"Train: {n_train}  Val: {n_val}")

    # --- model + optimizer ---
    model = ColorBlockCNN().to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # --- training loop ---
    best_val_mse = float("inf")
    epochs_since_improve = 0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    coord_names = ["x_red", "y_red", "z_red", "x_blue", "y_blue", "z_blue"]

    for epoch in range(1, args.epochs + 1):
        # Train one epoch
        model.train()
        train_mse_sum = 0.0
        train_n = 0
        for img, target in train_loader:
            img = img.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(img)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            train_mse_sum += loss.item() * img.size(0)
            train_n += img.size(0)
        train_mse = train_mse_sum / max(train_n, 1)

        # Validate
        val_mse, val_mae = evaluate(model, val_loader, device)
        val_mae_cm = val_mae * 100.0  # m -> cm
        avg_mae_cm = val_mae_cm.mean().item()

        msg = (
            f"epoch {epoch:3d}  "
            f"train_mse {train_mse:.5f}  "
            f"val_mse {val_mse:.5f}  "
            f"val_mae {avg_mae_cm:.2f}cm "
            f"({', '.join(f'{n}={v:.2f}' for n, v in zip(coord_names, val_mae_cm.tolist()))})"
        )
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            epochs_since_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_mse": val_mse,
                    "val_mae_cm_per_coord": val_mae_cm.tolist(),
                },
                output_path,
            )
            msg += "  [saved]"
        else:
            epochs_since_improve += 1
        print(msg)

        if epochs_since_improve >= args.patience:
            print(f"No val improvement for {args.patience} epochs — early stopping.")
            break

    print(f"\nBest val MSE: {best_val_mse:.5f}  -> checkpoint at {output_path}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Total wall-clock: {time.time() - t0:.1f}s")
