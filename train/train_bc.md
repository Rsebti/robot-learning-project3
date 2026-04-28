# Train BC policy (ACT) on sanity-check dataset

## Goal
Overfit an ACT policy on the 20 quasi-identical demos to validate the pipeline.
Target: the policy reproduces the demo motion when deployed on the real SO-101.

## Prerequisites
- Dataset pushed to HF: `Rsebti/projet3-demos-v1bis`
- LeRobot installed (env `lerobot`, Python 3.12)
- HF logged in (`hf auth login`)
- GPU available (`nvidia-smi`)

## Command (local, RTX 5070)

```bash
lerobot-train \
  --dataset.repo_id=Rsebti/projet3-demos-v1bis \
  --policy.type=act \
  --policy.device=cuda \
  --policy.repo_id=Rsebti/projet3-act-sanity \
  --output_dir=outputs/train/act_sanity \
  --job_name=act_sanity \
  --batch_size=8 \
  --steps=20000 \
  --save_freq=5000 \
  --eval_freq=0 \
  --log_freq=200 \
  --wandb.enable=false
```

Notes:
- `--steps=20000` is enough to overfit 20 demos. Bump to 40k if loss is still decreasing.
- `--save_freq=5000` → checkpoints in `outputs/train/act_sanity/checkpoints/`.
- `--eval_freq=0` disables sim eval (we evaluate on the real robot).
- `--policy.repo_id` is the HF model repo where the final policy gets pushed (optional for sanity).

## RTX 5070 (Blackwell sm_120) — known issue
PyTorch wheels shipped with LeRobot 0.5.2 may not support sm_120 yet.
If `RuntimeError: CUDA error: no kernel image is available for execution on the device`:
- Install a PyTorch nightly with cu124+:
  ```bash
  pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124
  ```
- Or fall back to CPU (`--policy.device=cpu`, slow but works) for a quick smoke test.
- If still broken → train on Brev H100.

## Output
- Checkpoints: `outputs/train/act_sanity/checkpoints/<step>/pretrained_model/`
- Last checkpoint symlink: `outputs/train/act_sanity/checkpoints/last/pretrained_model/`
- Use `last/pretrained_model` as `--policy.path` for deployment.

## Sanity training metrics to watch
- `loss` decreasing smoothly to a small value (overfit expected).
- No NaNs.
- Training time on 5070 ≈ 30-60 min for 20k steps.
