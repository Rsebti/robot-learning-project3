"""Pad V2.9 checkpoint to match V2.10 obs dim (Phase C placeholders).

V2.9 was trained without the Phase C placeholders (`target_color_zero` 6D
+ `bowl_xyz_zero` 3D). V2.10 adds those 9 dims to the observation, so the
actor/critic input layers go from (256, 540) to (256, 549).

Since the placeholders are zeros, padding the input layer with 9 zero
**columns** preserves V2.9's exact behavior at step 0: each new column is
multiplied by 0 in every forward pass.

We also pad the Adam optimizer state (exp_avg, exp_avg_sq) for those
two layers and reset `iter` to 0 so the resumed run starts a fresh
1500-iter budget.

Usage:
    cd C:\\Users\\user\\Desktop\\MA2\\isaac\\isaac_so_arm101
    .\\.venv\\Scripts\\Activate.ps1
    python -m sim.eval2.scripts.pad_v29_for_v210
"""
from __future__ import annotations

import torch
from pathlib import Path

V29_CKPT = Path(
    r"C:\Users\user\Desktop\MA2\isaac\isaac_so_arm101\logs\rsl_rl"
    r"\lift_v2_9\2026-05-08_23-10-05\model_1499.pt"
)
OUT_DIR = Path(
    r"C:\Users\user\Desktop\MA2\isaac\isaac_so_arm101\logs\rsl_rl"
    r"\lift_v2_10\warmstart_from_v29"
)
OUT_NAME = "model_0.pt"
PAD_DIMS = 9  # 6 (target_color) + 3 (bowl_xyz)


def _pad_input_cols(t: torch.Tensor, n: int) -> torch.Tensor:
    """Append n zero columns to a 2D weight tensor (out_dim, in_dim)."""
    assert t.ndim == 2, f"expected 2D tensor, got shape {tuple(t.shape)}"
    out_dim, in_dim = t.shape
    pad = torch.zeros(out_dim, n, dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=1)


def main() -> None:
    print(f"[INFO] Loading V2.9 checkpoint: {V29_CKPT.name}")
    ckpt = torch.load(V29_CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    opt = ckpt["optimizer_state_dict"]

    # ---------------- Pad model state_dict ----------------
    layers_to_pad = ["actor.0.weight", "critic.0.weight"]
    for k in layers_to_pad:
        old_shape = tuple(sd[k].shape)
        sd[k] = _pad_input_cols(sd[k], PAD_DIMS)
        print(f"[PAD]  {k:25s} {old_shape} -> {tuple(sd[k].shape)}")

    # ---------------- Pad optimizer state -----------------
    # PyTorch Adam stores per-param entries keyed by param index. The
    # ordering matches the order params were registered with the
    # optimizer. From the V2.9 state_dict, the actor/critic input-layer
    # weights are at known positions; we identify them by *shape* to
    # avoid hardcoding indices.
    target_old_shape = (256, 540)
    target_new_shape = (256, 549)
    n_padded = 0
    for pid, state in opt["state"].items():
        for stat_key in ("exp_avg", "exp_avg_sq"):
            if stat_key not in state:
                continue
            t = state[stat_key]
            if tuple(t.shape) == target_old_shape:
                state[stat_key] = _pad_input_cols(t, PAD_DIMS)
                n_padded += 1
    print(f"[PAD]  optimizer Adam moments: {n_padded} tensors padded "
          f"{target_old_shape} -> {target_new_shape}")
    assert n_padded == 4, (
        f"expected 4 Adam moment tensors to pad (actor.0.weight + "
        f"critic.0.weight, each with exp_avg + exp_avg_sq), got {n_padded}"
    )

    # ---------------- Reset iter so we get a fresh 1500-iter budget ----------------
    ckpt["iter"] = 0
    ckpt["infos"] = None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / OUT_NAME
    torch.save(ckpt, out_path)
    print(f"[OK]   Saved padded checkpoint -> {out_path}")
    print(f"[OK]   File size: {out_path.stat().st_size / 1024:.1f} KB")
    print()
    print("Next:")
    print("  cd C:\\Users\\user\\Desktop\\MA2\\isaac\\isaac_so_arm101")
    print("  .\\.venv\\Scripts\\Activate.ps1")
    print("  python -m sim.eval2.scripts.train `")
    print("    --task Isaac-LeIsaac-SO101-Lift-Visual-V210-v0 `")
    print("    --headless --enable_cameras `")
    print("    --resume `")
    print(f"    --load_run warmstart_from_v29 `")
    print(f"    --checkpoint {OUT_NAME}")


if __name__ == "__main__":
    main()
