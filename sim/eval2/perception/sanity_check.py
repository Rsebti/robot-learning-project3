"""Quick sanity check for the trained perception CNN.

Loads the captured dataset and the trained checkpoint, runs batched
inference, and reports:

1. Overall mean absolute error per coordinate (xyz red + xyz blue).
2. MAE split between "lifted" frames (a block is in flight, max(z_red, z_blue)
   > 3 cm above the table) and "on-table" frames. This is the key check —
   the val MAE 0.87 cm headline number could hide a much worse error on
   in-flight frames where depth is harder to estimate.
3. A few sample rows showing GT vs prediction in cm, picked from each
   regime.
4. Optionally, annotated PNGs of those samples.

Usage (from a venv with this package installed, e.g. the isaac_so_arm101 venv):
    uv run python -m sim.eval2.perception.sanity_check
    uv run python -m sim.eval2.perception.sanity_check --num_samples 12 --save_pngs

Output layout (with --save_pngs):
    sim/eval2/perception/preview/sanity_*.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import ColorBlockCNN


HERE = Path(__file__).parent
DEFAULT_DATA = HERE / "data.pt"
DEFAULT_CKPT = HERE / "checkpoint.pt"
PREVIEW_DIR = HERE / "preview"

LIFTED_Z_THRESHOLD = 0.03  # meters above the table center (= 2 cm above table top)
COORD_NAMES = ["x_red", "y_red", "z_red", "x_blue", "y_blue", "z_blue"]


def load_data(path: Path):
    print(f"[sanity] loading dataset from {path} ...")
    data = torch.load(path, map_location="cpu")
    images = data["images"]   # (N, H, W, 3) uint8
    targets = data["targets"] # (N, 6) float32
    print(f"          images  {tuple(images.shape)}  {images.dtype}")
    print(f"          targets {tuple(targets.shape)}  {targets.dtype}")
    return images, targets


def load_model(path: Path, device: torch.device, image_size: int) -> ColorBlockCNN:
    print(f"[sanity] loading checkpoint from {path} ...")
    model = ColorBlockCNN().to(device).eval()
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


@torch.no_grad()
def batched_predict(
    model: ColorBlockCNN,
    images: torch.Tensor,
    image_size: int,
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    """Returns (N, 6) predictions on CPU in meters."""
    n = images.shape[0]
    out = torch.zeros(n, 6, dtype=torch.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        # (B, H, W, 3) uint8 -> (B, 3, H, W) float in [0, 1]
        x = images[start:end].permute(0, 3, 1, 2).float() / 255.0
        # Resize if needed
        if x.shape[-1] != image_size or x.shape[-2] != image_size:
            x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
        x = x.to(device)
        pred = model(x).cpu()
        out[start:end] = pred
    return out


def per_coord_mae_cm(targets: torch.Tensor, preds: torch.Tensor) -> list[float]:
    """Returns per-coordinate MAE in centimeters."""
    err = (preds - targets).abs() * 100.0  # cm
    return err.mean(dim=0).tolist()


def format_mae_table(label: str, n: int, maes_cm: list[float]) -> str:
    parts = [f"  {name}: {v:5.2f} cm" for name, v in zip(COORD_NAMES, maes_cm)]
    avg = sum(maes_cm) / len(maes_cm)
    head = f"\n[{label}]  N = {n}  |  avg MAE = {avg:.2f} cm"
    return head + "\n" + "\n".join(parts)


def print_sample_rows(
    title: str,
    indices: list[int],
    targets: torch.Tensor,
    preds: torch.Tensor,
):
    print(f"\n--- {title} ---")
    header = (
        f"{'idx':>5} | "
        f"{'GT  red xyz (cm)':>22} | {'Pred red xyz (cm)':>22} | {'err (cm)':>14}"
        f"  ||  "
        f"{'GT  blue xyz (cm)':>22} | {'Pred blue xyz (cm)':>22} | {'err (cm)':>14}"
    )
    print(header)
    print("-" * len(header))
    for idx in indices:
        gt = (targets[idx] * 100.0).tolist()
        pr = (preds[idx] * 100.0).tolist()
        err = [abs(g - p) for g, p in zip(gt, pr)]

        gt_red = f"{gt[0]:6.2f},{gt[1]:6.2f},{gt[2]:6.2f}"
        pr_red = f"{pr[0]:6.2f},{pr[1]:6.2f},{pr[2]:6.2f}"
        er_red = f"{err[0]:4.2f},{err[1]:4.2f},{err[2]:4.2f}"

        gt_blu = f"{gt[3]:6.2f},{gt[4]:6.2f},{gt[5]:6.2f}"
        pr_blu = f"{pr[3]:6.2f},{pr[4]:6.2f},{pr[5]:6.2f}"
        er_blu = f"{err[3]:4.2f},{err[4]:4.2f},{err[5]:4.2f}"

        print(
            f"{idx:>5d} | {gt_red:>22} | {pr_red:>22} | {er_red:>14}  ||  "
            f"{gt_blu:>22} | {pr_blu:>22} | {er_blu:>14}"
        )


def maybe_save_pngs(
    indices_lifted: list[int],
    indices_table: list[int],
    images: torch.Tensor,
    targets: torch.Tensor,
    preds: torch.Tensor,
):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("[sanity] PIL not available — skipping PNG export. `pip install Pillow` to enable.")
        return

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    pairs = [("lifted", indices_lifted), ("table", indices_table)]
    saved = 0
    for label, idxs in pairs:
        for k, idx in enumerate(idxs):
            img_hwc = images[idx].numpy()  # (H, W, 3) uint8
            pil = Image.fromarray(img_hwc, mode="RGB")
            # Upscale 4x so the annotation is readable.
            pil = pil.resize((pil.width * 4, pil.height * 4), Image.NEAREST)
            draw = ImageDraw.Draw(pil)

            gt = (targets[idx] * 100.0).tolist()
            pr = (preds[idx] * 100.0).tolist()
            err = [abs(g - p) for g, p in zip(gt, pr)]

            # Just dump the numbers in the top-left.
            lines = [
                f"GT  red  ({gt[0]:.1f}, {gt[1]:.1f}, {gt[2]:.1f}) cm",
                f"Pred red ({pr[0]:.1f}, {pr[1]:.1f}, {pr[2]:.1f}) cm  err {sum(err[:3]):.1f}",
                f"GT  blue ({gt[3]:.1f}, {gt[4]:.1f}, {gt[5]:.1f}) cm",
                f"Pred blue({pr[3]:.1f}, {pr[4]:.1f}, {pr[5]:.1f}) cm  err {sum(err[3:]):.1f}",
            ]
            for i, line in enumerate(lines):
                # Outline so it's readable on any background.
                xy = (4, 4 + 14 * i)
                draw.text(xy, line, fill=(255, 255, 255))

            out_path = PREVIEW_DIR / f"sanity_{label}_{k:02d}.png"
            pil.save(out_path)
            saved += 1
    print(f"\n[sanity] wrote {saved} annotated PNGs to {PREVIEW_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=8,
        help="Number of sample rows to print PER REGIME (lifted / on-table).",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=84,
        help="CNN input size, must match training. Default 84.",
    )
    parser.add_argument(
        "--save_pngs",
        action="store_true",
        help="Save annotated PNGs of the displayed samples to perception/preview/.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sanity] device = {device}")

    images, targets = load_data(args.data)
    model = load_model(args.checkpoint, device, args.image_size)

    print("[sanity] running batched inference ...")
    preds = batched_predict(model, images, args.image_size, device)

    # ---- Aggregate metrics ----
    n = images.shape[0]
    z_max = torch.maximum(targets[:, 2], targets[:, 5])  # max(z_red, z_blue), in meters
    lifted_mask = z_max > LIFTED_Z_THRESHOLD
    table_mask = ~lifted_mask
    n_lift = int(lifted_mask.sum().item())
    n_table = int(table_mask.sum().item())

    mae_all = per_coord_mae_cm(targets, preds)
    mae_lift = per_coord_mae_cm(targets[lifted_mask], preds[lifted_mask]) if n_lift > 0 else None
    mae_table = per_coord_mae_cm(targets[table_mask], preds[table_mask]) if n_table > 0 else None

    print(format_mae_table("ALL", n, mae_all))
    if mae_table is not None:
        print(format_mae_table(f"ON-TABLE  (max(z) <= {LIFTED_Z_THRESHOLD * 100:.0f} cm)", n_table, mae_table))
    if mae_lift is not None:
        print(format_mae_table(f"LIFTED   (max(z) > {LIFTED_Z_THRESHOLD * 100:.0f} cm)", n_lift, mae_lift))

    # ---- Sample rows ----
    g = torch.Generator()
    g.manual_seed(args.seed)
    lifted_indices = torch.where(lifted_mask)[0]
    table_indices = torch.where(table_mask)[0]

    pick_lift = lifted_indices[torch.randperm(len(lifted_indices), generator=g)[: args.num_samples]].tolist()
    pick_table = table_indices[torch.randperm(len(table_indices), generator=g)[: args.num_samples]].tolist()

    print_sample_rows("ON-TABLE samples", pick_table, targets, preds)
    if pick_lift:
        print_sample_rows("LIFTED samples", pick_lift, targets, preds)
    else:
        print("\n[!] No lifted frames in the dataset (max z always <= threshold).")

    # ---- Optional PNG export ----
    if args.save_pngs:
        maybe_save_pngs(pick_lift, pick_table, images, targets, preds)


if __name__ == "__main__":
    main()
