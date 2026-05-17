#!/usr/bin/env python3
"""Profile SmolVLA on this machine to figure out where deploy-time latency
goes and whether action chunking is actually amortizing inference cost.

What it measures (in order):
  1. Cold model load + first forward (warmup).
  2. `predict_action_chunk` — one full chunk forward (VLM + N flow-matching
     steps). This is the per-chunk cost; with chunk_size=50 + 30 Hz target,
     this needs to be < ~1.6 s for real-time control.
  3. `select_action` cold (first call) and warm (50 subsequent calls).
     If chunking works, warm calls are queue pops at ~1 ms; if not,
     warm calls are the same cost as the chunk forward.
  4. Effective Hz the loop would run at given the observed timings.

The script also sweeps `num_steps` (flow-matching iterations) so you can
see the latency/quality trade-off without retraining: drop num_steps
from 10 -> 5 typically halves chunk latency.

Usage:
  python tools/profile_smolvla.py
  python tools/profile_smolvla.py \
      --repo-id osammotg1/projet3-smolvla-eval2-v1-dark-noise-step44k \
      --device mps --num-steps 10 5 3
"""

import argparse
import statistics
import time
from contextlib import contextmanager

import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


@contextmanager
def sync_timer(device):
    """Time a block with device-side sync (so MPS/CUDA async work is included)."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    yield lambda: None
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def time_ms(device, fn, repeats=10):
    """Run `fn` `repeats` times after sync, return list of per-call ms."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        out.append((time.perf_counter() - t0) * 1000)
    return out


def make_dummy_obs(policy, device, image_size=256):
    """Build a deploy-shape observation batch, including the pre-tokenized
    language that SmolVLA expects (the lerobot preprocessor would normally
    add these fields before the policy sees the batch)."""
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    task = "Pick yellow block and place in bowl at (-15.5,29.5) cm"
    enc = tokenizer(
        task,
        padding="max_length",
        max_length=policy.config.tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return {
        "observation.state": torch.zeros(1, 6, device=device),
        "observation.images.camera1": torch.zeros(
            1, 3, image_size, image_size, device=device
        ),
        "observation.language.tokens": enc["input_ids"].to(device),
        # SmolVLA's internal attention path requires a bool mask (torch.where
        # condition); the HF tokenizer returns Long by default.
        "observation.language.attention_mask": enc["attention_mask"].bool().to(device),
        "task": [task],
    }


def report_block(name, ms_list):
    if not ms_list:
        return
    avg = statistics.mean(ms_list)
    mn = min(ms_list)
    mx = max(ms_list)
    print(f"  {name:32s}  avg {avg:7.1f} ms  min {mn:6.1f}  max {mx:6.1f}  (n={len(ms_list)})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default="osammotg1/projet3-smolvla-eval2-v1-dark-noise-step44k",
        help="Pretrained SmolVLA policy on HF Hub or local path.",
    )
    parser.add_argument("--device", default="mps", choices=["cpu", "mps", "cuda"])
    parser.add_argument(
        "--num-steps",
        type=int,
        nargs="+",
        default=[10, 5, 3],
        help="Flow-matching iterations to sweep (smaller = faster, lower quality).",
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Input image resolution; SmolVLA trained at 256.",
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=30,
        help="Robot control target fps; used to compute headroom.",
    )
    args = parser.parse_args()

    print(f"=== loading SmolVLA from {args.repo_id} on {args.device} ===")
    t0 = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(args.repo_id)
    policy = policy.to(args.device).eval()
    load_s = time.perf_counter() - t0
    print(f"  model load + .to({args.device}): {load_s:.2f} s")

    chunk_size = policy.config.chunk_size
    n_action_steps = policy.config.n_action_steps
    default_num_steps = policy.config.num_steps
    print(
        f"  chunk_size={chunk_size}  n_action_steps={n_action_steps}  "
        f"num_steps(default)={default_num_steps}"
    )

    obs = make_dummy_obs(policy, args.device, args.image_size)

    print(f"\n=== warmup (2 chunk forwards) ===")
    with torch.no_grad():
        for _ in range(2):
            policy.reset()
            _ = policy.predict_action_chunk(obs)

    for ns in args.num_steps:
        policy.config.num_steps = ns  # cheap monkey-patch; SmolVLA reads this at call time
        print(f"\n=== num_steps = {ns}  (default = {default_num_steps}) ===")

        # 1. predict_action_chunk — one full forward
        def _chunk():
            policy.reset()
            with torch.no_grad():
                policy.predict_action_chunk(obs)
        chunk_ms = time_ms(args.device, _chunk, repeats=args.repeats)
        report_block("predict_action_chunk (full)", chunk_ms)

        # 2. select_action cold (first call after reset)
        def _select_cold():
            policy.reset()
            with torch.no_grad():
                policy.select_action(obs)
        cold_ms = time_ms(args.device, _select_cold, repeats=args.repeats)
        report_block("select_action cold (1st call)", cold_ms)

        # 3. select_action warm: reset once, then 50 calls (one chunk worth)
        policy.reset()
        with torch.no_grad():
            # prime the queue
            policy.select_action(obs)
        def _select_warm():
            with torch.no_grad():
                policy.select_action(obs)
        # do n_action_steps - 1 warm calls (queue should be primed)
        warm_n = max(1, n_action_steps - 1)
        warm_ms = time_ms(args.device, _select_warm, repeats=min(warm_n, args.repeats * 5))
        report_block(f"select_action warm (queue pop, n={len(warm_ms)})", warm_ms)

        # 4. effective Hz
        chunk_avg = statistics.mean(chunk_ms)
        warm_avg = statistics.mean(warm_ms)
        # one chunk amortized over n_action_steps select_action calls
        eff_step_ms = (chunk_avg + (n_action_steps - 1) * warm_avg) / n_action_steps
        eff_hz = 1000.0 / eff_step_ms
        target_step_ms = 1000.0 / args.target_fps
        headroom = target_step_ms - eff_step_ms

        print(
            f"  effective avg step:               {eff_step_ms:7.1f} ms  "
            f"=> {eff_hz:5.1f} Hz"
        )
        print(
            f"  target step @ {args.target_fps} Hz:               "
            f"{target_step_ms:7.1f} ms  (headroom {headroom:+.1f} ms)"
        )
        if eff_hz < args.target_fps * 0.9:
            print(
                f"  >> bottleneck: chunking would need {chunk_avg / target_step_ms:.1f}x more "
                f"speedup, OR n_action_steps must rise to >= "
                f"{int(chunk_avg / target_step_ms) + 1} to amortize."
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
