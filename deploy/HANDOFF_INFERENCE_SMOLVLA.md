# Handoff: SmolVLA inference on the real SO-101 (eval2)

> Companion to `notes/tommaso_eval2_sim_handoff.md`. That doc is about training
> SmolVLA on the 5090. This doc is the **laptop / robot-PC side**: how to grab
> the trained checkpoint and drive the arm.

## TL;DR

```bash
# 1. (on 5090) push an intermediate checkpoint to HF, e.g. step 30k:
STEP=30000 bash deploy/push_smolvla_checkpoint.sh
# → prints: POLICY_PATH=osammotg1/projet3-smolvla-eval2-v1-step30k bash deploy/infer_smolvla.sh

# 2. (on robot PC) plug arms + camera in, then:
git pull
POLICY_PATH=osammotg1/projet3-smolvla-eval2-v1-step30k bash deploy/infer_smolvla.sh
# → interactive picker asks for target cube color
```

That's the whole loop. The rest of this doc explains the moving parts and the
SmolVLA-specific gotchas you won't find in the ACT inference doc.

## Why this is its own script (vs. `deploy/infer.sh`)

`deploy/infer.sh` runs ACT policies. SmolVLA needs two things ACT didn't:

1. **Camera-key rename at inference.** Our dataset has
   `observation.images.wrist`; the pretrained `lerobot/smolvla_base` keys its
   visual encoder as `camera1/camera2/camera3`. `train/launch_smolvla.sh`
   passes `--rename_map='{"observation.images.wrist": "observation.images.camera1"}'`
   so the dataset frame is fed into camera1's pretrained encoder. The saved
   `config.json` in the checkpoint therefore expects the batch key
   `observation.images.camera1` — but `lerobot-record` on the robot still
   exposes the live wrist camera under `wrist`. So we pass the **same**
   rename map again at inference via `--dataset.rename_map`. Skip it and the
   model sees no image input.

2. **The task string is load-bearing.** With ACT, `single_task` was metadata
   the model never read. With SmolVLA, the text encoder consumes
   `single_task` as the language instruction — that's the whole reason we
   trained it. `infer_smolvla.sh` builds the same exact phrasing the dataset
   was labeled with:
   `Pick {color} block and place in bowl at (-15.5,29.5) cm`.
   Keep this string identical to the dataset's `tasks.jsonl` entries.

Everything else (interactive color picker, leader/follower port handling,
eval-dataset cache cleanup) is copied verbatim from `infer.sh`.

## Three ways to get the checkpoint onto the robot PC

Pick one. HF is the recommended default.

### A. HF Hub (recommended)

Pre-condition: the 5090 has already pushed the checkpoint to HF.

- **Final 50k-step checkpoint** auto-pushes at end of training to
  `osammotg1/projet3-smolvla-eval2-v1` (handled inside the training run).
- **Any intermediate checkpoint** (every 2k steps) can be pushed manually
  from the 5090 with `deploy/push_smolvla_checkpoint.sh`:
  ```bash
  # on the 5090, in this repo:
  STEP=30000 bash deploy/push_smolvla_checkpoint.sh
  # → pushes pretrained_model/ to osammotg1/projet3-smolvla-eval2-v1-step30k
  ```
  Omitting `STEP` uses the most recent checkpoint (`last/` symlink).

On the robot PC, nothing extra to do — just pass the repo id:
```bash
POLICY_PATH=osammotg1/projet3-smolvla-eval2-v1-step30k bash deploy/infer_smolvla.sh
```
`lerobot-record` downloads it into `~/.cache/huggingface/hub/` on first use
and reuses it thereafter.

### B. SCP straight from the 5090 (skip HF)

If HF is being slow, or you want to test without polluting the model namespace.
On the robot PC:

```bash
# pick where you want to keep local checkpoints
mkdir -p ~/checkpoints/smolvla
scp -r ethrc-rl-ws1:/home/ethrc/Desktop/training/checkpoints/projet3/projet3_smolvla_eval2_v1_20260513_190321/checkpoints/030000/pretrained_model \
      ~/checkpoints/smolvla/step30k

POLICY_PATH=~/checkpoints/smolvla/step30k bash deploy/infer_smolvla.sh
```

The script auto-detects "looks like a path" vs. "looks like an HF repo id"
based on the leading `/` or `./`.

### C. Wait for the run to finish

Final repo (`osammotg1/projet3-smolvla-eval2-v1`) is pushed automatically
when the 50k-step run terminates. ETA was ~1h53m from 19:03 local — see
[W&B run 1sxlvpgv](https://wandb.ai/tom-gazzini-ethrc/projet3-smolvla/runs/1sxlvpgv).

```bash
bash deploy/infer_smolvla.sh   # POLICY_PATH defaults to ...projet3-smolvla-eval2-v1
```

## What runs on the robot PC

```bash
# minimal: pick color interactively, defaults for everything else
bash deploy/infer_smolvla.sh

# scripted: skip the picker, run a specific policy
TARGET_COLOR=yellow \
  POLICY_PATH=osammotg1/projet3-smolvla-eval2-v1-step30k \
  bash deploy/infer_smolvla.sh

# different host (Linux/Windows): override ports
TARGET_COLOR=red \
  POLICY_DEVICE=cuda \
  FOLLOWER_PORT=/dev/ttyACM0 LEADER_PORT=/dev/ttyACM1 \
  bash deploy/infer_smolvla.sh
```

### Env-var reference

| Var | Default | Notes |
|---|---|---|
| `POLICY_PATH` | `osammotg1/projet3-smolvla-eval2-v1` | HF repo id or local dir. |
| `POLICY_DEVICE` | `mps` | Use `cuda` on Linux PC, `cpu` if both fail. |
| `TARGET_COLOR` | *(prompt)* | One of yellow/blue/green/violet/red/orange. |
| `SINGLE_TASK` | derived from `TARGET_COLOR` | Override only if you change the labeling. |
| `FOLLOWER_PORT` | `/dev/tty.usbmodem5B141129871` | Mac default; Linux is `/dev/ttyACM*`, Windows `COM*`. |
| `LEADER_PORT` | `/dev/tty.usbmodem5B141128171` | Set to `""` to disable leader-driven reset. |
| `NUM_EPISODES` | `5` | Per TA spec: 5 rollouts × 10 pts. |
| `EPISODE_TIME_S` | `20` | Per-rollout timeout. |
| `RENAME_MAP` | `{"observation.images.wrist": "observation.images.camera1"}` | **Don't unset this** unless you re-train without the rename. |

## Sanity checks before the arm moves

1. **GPU/MPS available?** SmolVLA is ~450M params; CPU inference is unusably
   slow. On macOS: `python -c "import torch; print(torch.backends.mps.is_available())"`.
   On Linux PC: `nvidia-smi`.
2. **Camera key match?** Check the trained config:
   ```bash
   python -c "import json; print(json.load(open('<POLICY_PATH or ~/.cache/.../config.json>'))['input_features'].keys())"
   ```
   Should list `observation.images.camera1`. If it doesn't, the rename map at
   training time was different and `infer_smolvla.sh`'s default
   `RENAME_MAP` will need to be updated to match.
3. **Task string match?** Open the dataset's `meta/tasks.jsonl` or
   `single_task` in episode metadata. The string passed at inference must
   match the labeling exactly (case, spacing, the `(-15.5,29.5)` coords) or
   the text encoder will see an unfamiliar phrase. Don't paraphrase.
4. **Dry-run with `NUM_EPISODES=1 EPISODE_TIME_S=5`** before committing to a
   full 5-rollout eval.

## Quick spot-test (no robot needed)

To confirm the checkpoint loads + produces non-NaN actions before plugging in
the arm, on the robot PC:

```python
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
import torch

policy = SmolVLAPolicy.from_pretrained("osammotg1/projet3-smolvla-eval2-v1-step30k").eval()
# dummy batch matching the checkpoint's expected input keys:
batch = {
    "observation.state": torch.zeros(1, 6),
    "observation.images.camera1": torch.zeros(1, 3, 480, 640),
    "task": "Pick yellow block and place in bowl at (-15.5,29.5) cm",
}
with torch.no_grad():
    action = policy.select_action(batch)
print(action.shape, action.isfinite().all().item())
```

If this prints `torch.Size([1, 6]) True`, the checkpoint is intact. If it
NaNs out, something is wrong with the push or the camera-key rename — check
`config.json` and the keys in `policy_preprocessor.json`.

## Expected behavior on the real arm

Per the strategy playbook (`notes/so101_robot_learning_playbook.md` §T2):

- **Optimistic path (camera-naming gods favor us):** 70–90% in-dist
  target-color obedience. Massive jump over the ~50% ceiling all four ACT
  variants hit, because SmolVLA actually reads `single_task`.
- **Pessimistic path (wrist-only kills us):** 0–50%, indistinguishable from
  ACT. Community variance on SmolVLA reproductions is large; LeRobot issue
  #2915 documents 0% with 120 episodes on the same robot family.

Either outcome is informative — see the "★ Insight" box at the bottom of
`notes/tommaso_eval2_sim_handoff.md`.

## When you're done

If you ran a real 5-episode eval (`NUM_EPISODES=5 PUSH_TO_HUB=true`), the
rollout dataset is at `osammotg1/eval_<policy_short>-<color>`. Drop a row in
`notes/sanity_results.md` (or `notes/smolvla_eval2_v1.md` once it's created)
with: target color, success/fail per rollout, qualitative failure mode
(picked wrong color / missed grasp / overshot bowl / etc.). That's the data
the next "stay on SmolVLA vs. pivot to FiLM-ACT" decision is made on.
