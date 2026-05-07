"""Smoke test the lerobot ACT pipeline locally.

Pulls our existing sanity-check ACT checkpoint from HF, builds a synthetic
observation matching what it expects, and runs a forward pass.

Goal: confirm that the local Python env can:
    1. import lerobot
    2. download a HF checkpoint
    3. instantiate ACTPolicy.from_pretrained
    4. run select_action on a fake batch and get a reasonable action

If this passes, the same code path will work for the Eval 2 ACT checkpoint
once we train it on Federico's demos.

Usage (from any venv with lerobot installed):

    python train/smoke_test_act.py
    # or with a different checkpoint:
    python train/smoke_test_act.py --checkpoint Rsebti/projet3-act-sanity
"""

from __future__ import annotations

import argparse
import sys
import traceback


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="Rsebti/projet3-act-sanity",
        help="HF repo ID or local path to the ACT checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu or cuda. cpu is fine for one forward pass.",
    )
    args = parser.parse_args()

    print(f"[smoke] device   = {args.device}")
    print(f"[smoke] checkpoint = {args.checkpoint}")

    # ---- 1. Imports ----
    print("\n[smoke] importing lerobot...")
    try:
        import torch
        from lerobot.policies.act.modeling_act import ACTPolicy
    except Exception:
        print("[smoke] FAIL — lerobot import errored:")
        traceback.print_exc()
        return 1
    print(f"[smoke] OK — torch {torch.__version__}, lerobot ACTPolicy imported")

    # ---- 2. Load checkpoint ----
    print(f"\n[smoke] loading checkpoint {args.checkpoint!r}...")
    try:
        policy = ACTPolicy.from_pretrained(args.checkpoint)
    except Exception:
        print("[smoke] FAIL — could not load checkpoint:")
        traceback.print_exc()
        return 1
    policy = policy.to(args.device)
    policy.eval()
    print(f"[smoke] OK — policy loaded on {args.device}")
    print(f"[smoke] config.input_features  = {list(policy.config.input_features.keys())}")
    print(f"[smoke] config.output_features = {list(policy.config.output_features.keys())}")
    print(f"[smoke] config.chunk_size      = {policy.config.chunk_size}")
    print(f"[smoke] config.n_action_steps  = {policy.config.n_action_steps}")

    # ---- 3. Build a synthetic batch ----
    # Read the expected shapes from the config.
    print("\n[smoke] building a synthetic observation batch...")
    batch = {}
    for name, feat in policy.config.input_features.items():
        shape = (1, *feat.shape)  # add batch dim
        batch[name] = torch.zeros(shape, device=args.device, dtype=torch.float32)
        print(f"           {name:<35} -> shape {tuple(shape)}")

    # ---- 4. Run select_action ----
    print("\n[smoke] running select_action...")
    try:
        policy.reset()
        action = policy.select_action(batch)
    except Exception:
        print("[smoke] FAIL — select_action errored:")
        traceback.print_exc()
        return 1

    print(f"[smoke] OK — action shape = {tuple(action.shape)}")
    print(f"[smoke] OK — action dtype = {action.dtype}")
    print(f"[smoke] OK — action mean  = {action.mean().item():+.4f}")
    print(f"[smoke] OK — action range = [{action.min().item():+.4f}, {action.max().item():+.4f}]")

    # ---- 5. Sanity check the action ----
    expected_action_dim = next(iter(policy.config.output_features.values())).shape[0]
    if action.shape[-1] != expected_action_dim:
        print(
            f"[smoke] FAIL — action dim {action.shape[-1]} != expected {expected_action_dim}"
        )
        return 1

    print("\n" + "=" * 60)
    print(" [smoke] PASS — lerobot ACT pipeline works end-to-end on this machine.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
