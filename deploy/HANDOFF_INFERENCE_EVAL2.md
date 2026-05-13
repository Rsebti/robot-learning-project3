# Handoff: run Eval-2 ACT inference on the SO-101 (laptop session)

Paste this into a fresh Claude Code session **on your laptop next to the robot**.
The remote 5090 box has finished four eval2 trainings overnight — this is the
recipe to download them and drive the arm.

> Read this **before** the older `HANDOFF_INFERENCE.md` if you intend to test
> eval2 specifically. The eval1 doc is still valid for the single-cube task.

---

## What was trained (eval2)

Four ACT policies, all small-data ACT (`dim_model=256`, 17.1 M params), trained on
**`osammotg1/projet3-eval2-v1-tom-hugo`** — 101 episodes / 43,289 frames, 6 colors,
**2 adjacent cubes + 1 bowl** per scene, target color encoded in the task string.
Bowl is **fixed at x = -15.5 cm, y = 29.5 cm** (robot frame) across the whole dataset.
W&B project: `tom-gazzini-ethrc/projet3-act`.

Task-string format the dataset uses:

```
Pick <color> block and place in bowl at (-15.5,29.5) cm
color ∈ {yellow, blue, green, violet, red, orange}
```

### The four HF policy repos

| run                  | steps   | HF repo                                              | aug profile                                                                                                                |
| -------------------- | ------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **no_aug**           | 50,000  | `osammotg1/projet3-act-eval2-no-aug`                | image_transforms disabled. Clean baseline.                                                                                  |
| **dark_noise**       | 50,000  | `osammotg1/projet3-act-eval2-dark-noise`            | brightness 0.2–1.1, contrast 0.5–1.5, sat 0.2–1.3, hue ±0.03, gaussian noise σ=0.06, autocontrast p=0.35, gaussian blur — max 3 of 7 per sample, random order. |
| **dark_shadow**      | 50,000  | `osammotg1/projet3-act-eval2-dark-shadow`           | brightness 0.2–1.1, contrast 0.5–1.5, gaussian noise σ=0.04, **random_erasing p=0.6** (occlusion), autocontrast p=0.5 — max 3 of 5 per sample, random order. |
| **dark_noise_100k**  | 100,000 | `osammotg1/projet3-act-eval2-dark-noise-100k`       | same as dark_noise but trained for 2× the steps (final loss ≈ 0.060, vs ≈ 0.07 for the 50k variants).                       |

> ⚠️ **ACT is vision + proprioception only — it does NOT condition on the task string.**
> With two cubes of different colors in the scene, the policy cannot read "pick yellow"
> from the language. It has learned *some* default behavior (most likely a position
> bias — pick the closer cube, or a learned color preference). When tomorrow's you
> evaluates: don't expect target-color obedience yet. The "color" knob in `SINGLE_TASK`
> only labels the eval dataset metadata; it does not change which cube the policy grabs.
> A language-conditioned policy (e.g. SmolVLA, OpenVLA, or ACT-with-CLIP) would be the
> next step if target-color obedience matters. Surface this with the human early.

### Which policy to try first?

- **First sanity check**: `no_aug`. If even the clean baseline does sensible motions
  toward one of the cubes, the pipeline is wired up correctly.
- **Most likely to generalize to your real lighting**: `dark_noise` or `dark_noise_100k`.
- **If your gripper occludes the cubes during approach**: `dark_shadow` (random_erasing
  during training simulated occlusion).
- **If you have time to test exhaustively**: A/B all four on the same scene and rank
  by success rate. The 100k variant should be a strict win over its 50k twin if it
  helped at all.

## Hardware prereqs

Same as eval1:
- SO-101 follower on USB (Linux `/dev/ttyACM0`, macOS `/dev/tty.usbmodem...`, Windows `COM5` etc.).
- Wrist camera (OpenCV index 0 by default, 640×480 @ 30 fps).
- **Two cubes** of different colors in the trained palette (yellow/blue/green/violet/red/orange).
- **One bowl, fixed at (-15.5, 29.5) cm** in the robot base frame (the policy
  has only ever seen that position — bowl elsewhere = out of distribution).

## Software prereqs

```bash
# In your local clone of robot-learning-project3, with the lerobot venv active:
hf auth login            # paste your write token (osammotg1)
hf auth whoami           # should print: user: osammotg1
```

## Quick start

Same wrapper as eval1: `deploy/infer.sh`. Override `POLICY_PATH` to one of the
four repos above; set `SINGLE_TASK` to a label that matches the trained
distribution; everything else is optional.

```bash
# Linux example — try dark_noise_100k first (best convergence)
FOLLOWER_PORT=/dev/ttyACM0 \
POLICY_PATH=osammotg1/projet3-act-eval2-dark-noise-100k \
EVAL_REPO_ID=osammotg1/projet3-eval2-test-dark-noise-100k \
NUM_EPISODES=8 \
EPISODE_TIME_S=30 \
RESET_TIME_S=10 \
SINGLE_TASK="Pick yellow block and place in bowl at (-15.5,29.5) cm" \
PUSH_TO_HUB=true \
  bash deploy/infer.sh
```

```bash
# macOS example
FOLLOWER_PORT=/dev/tty.usbmodem5B141129871 \
POLICY_PATH=osammotg1/projet3-act-eval2-dark-noise-100k \
EVAL_REPO_ID=osammotg1/projet3-eval2-test-dark-noise-100k \
NUM_EPISODES=8 \
EPISODE_TIME_S=30 \
RESET_TIME_S=10 \
SINGLE_TASK="Pick yellow block and place in bowl at (-15.5,29.5) cm" \
PUSH_TO_HUB=true \
  bash deploy/infer.sh
```

```powershell
# Windows (PowerShell)
$env:FOLLOWER_PORT="COM5"
$env:POLICY_PATH="osammotg1/projet3-act-eval2-dark-noise-100k"
$env:EVAL_REPO_ID="osammotg1/projet3-eval2-test-dark-noise-100k"
$env:NUM_EPISODES="8"
$env:EPISODE_TIME_S="30"
$env:SINGLE_TASK="Pick yellow block and place in bowl at (-15.5,29.5) cm"
$env:PUSH_TO_HUB="true"
bash deploy/infer.sh
```

## Tunable knobs (env vars)

Same shape as the eval1 handoff; the only thing worth re-stating is that
`SINGLE_TASK` is a label for the recorded eval dataset, not a real policy input
for ACT (see warning above).

| variable          | default                           | what it does                                                                                                          |
| ----------------- | --------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `POLICY_PATH`     | *(must override)*                 | One of the four HF repos in the table above, or a local `pretrained_model/` dir.                                      |
| `FOLLOWER_PORT`   | *(required)*                      | Serial port for the SO-101 follower.                                                                                  |
| `NUM_EPISODES`    | `5`                               | Number of rollouts. Use 8–12 per condition for an honest comparison.                                                  |
| `EPISODE_TIME_S`  | `15`                              | Wall-clock seconds per rollout. Training demos were 30 s (eval2 was recorded at 30 s); 25–35 s is reasonable.         |
| `RESET_TIME_S`    | `10`                              | Seconds between rollouts to rearrange the cubes and bowl.                                                             |
| `SINGLE_TASK`     | `Pick block and place in bowl`    | Label only. ACT can't read it. Use one of the six color strings below if you want metadata to match the training set. |
| `EVAL_REPO_ID`    | `Rsebti/projet3-eval-sanity`      | HF repo for the recorded eval dataset. **Override per condition** so you don't overwrite previous evals.              |
| `PUSH_TO_HUB`     | `false`                           | Set `true` to publish the recorded eval dataset; useful for after-the-fact review.                                    |
| `POLICY_DEVICE`   | `cuda`                            | `cpu` if your laptop has no GPU; `mps` on Apple Silicon.                                                              |
| `CAMERA_INDEX`    | `0`                               | OpenCV index; bump if wrist cam isn't on 0.                                                                           |
| `CAMERA_WIDTH/HEIGHT/FPS` | `640/480/30`              | Match the training resolution / fps.                                                                                  |
| `DISPLAY_DATA`    | `true`                            | Live Rerun viewer.                                                                                                    |

### Trained task strings (use one of these for `SINGLE_TASK`)

```
Pick yellow block and place in bowl at (-15.5,29.5) cm
Pick blue   block and place in bowl at (-15.5,29.5) cm
Pick green  block and place in bowl at (-15.5,29.5) cm
Pick violet block and place in bowl at (-15.5,29.5) cm
Pick red    block and place in bowl at (-15.5,29.5) cm
Pick orange block and place in bowl at (-15.5,29.5) cm
```

## Recommended evaluation matrix

This is the protocol I'd use to actually rank the four policies. ~10 min per
batch, ~40 min total for the basic matrix. Move the cube pair to a fresh layout
during each `RESET_TIME_S` window.

```bash
BASE_FLAGS=(
  --quiet 1
)

for policy in no-aug dark-noise dark-shadow dark-noise-100k; do
  FOLLOWER_PORT=/dev/ttyACM0 \
  POLICY_PATH=osammotg1/projet3-act-eval2-${policy} \
  EVAL_REPO_ID=osammotg1/projet3-eval2-test-${policy} \
  NUM_EPISODES=8 \
  EPISODE_TIME_S=30 \
  RESET_TIME_S=10 \
  SINGLE_TASK="Pick yellow block and place in bowl at (-15.5,29.5) cm" \
  PUSH_TO_HUB=true \
    bash deploy/infer.sh
done
```

Run this once per cube *pair* you care about (yellow+blue, yellow+green, …).
For a tighter ablation, fix the pair and rotate the policy.

Scoring rubric (write this in your notes per run):
1. Did the gripper *approach* either cube? (Y/N)
2. Did it close on *a* cube? (Y/N)
3. Did it close on the *target* cube? (Y/N — expected to be ~50% by chance,
   since ACT can't read the language)
4. Did it place the cube *in* the bowl? (Y/N)
5. Smoothness: 1–5 subjective.

## What I want you (Claude on the laptop) to do

When the human runs this handoff:

1. Verify `hf auth whoami` → `osammotg1` (offer `hf auth login` otherwise).
2. Confirm the bowl is at (-15.5, 29.5) cm and two trained-color cubes are
   on the table. If they ask about a different bowl position, flag that
   this is out of distribution and the policy may behave erratically.
3. Ask which policy they want to test first (default: `dark-noise-100k`).
4. Ask which `SINGLE_TASK` (drop-down of the six color strings) and how many
   `NUM_EPISODES`. Remind them that `SINGLE_TASK` only labels the dataset.
5. Build the env-var block, paste the command, let them run it.
6. Between batches, write per-rollout outcomes to a Markdown table in the
   chat — that's the most useful thing to hand back to the trainer side.

## Troubleshooting

Same failure modes as the eval1 handoff:
- Camera index → `lerobot-find-cameras opencv`.
- Stale HF cache → `rm -rf ~/.cache/huggingface/hub/models--osammotg1--projet3-act-eval2-*`.
- Forbidden push → re-`hf auth login` with the write token.
- Policy from HF cache stale after retraining → same as above.

One eval2-specific gotcha: if the policy reaches for *neither* cube and
just stays still, the bowl might be too far from (-15.5, 29.5) — the
proprioception offset puts you outside the policy's manifold. Move the
bowl back to the trained position.

## Provenance (in case you need to retrain)

- Training script: `train/launch_act.sh` (env-var driven; see `EXCLUDE_EPISODES`,
  `IMAGE_TFS`, `MAX_NUM_TRANSFORMS`, `RANDOM_ORDER`, `IMAGE_TRANSFORMS_ENABLE`,
  `STEPS`, `POLICY_REPO_ID`, etc.).
- Sweep driver used overnight: `/tmp/projet3_eval2_sweep_driver.sh` on the 5090 box
  (sequential 4-run orchestrator, also writes to
  `/home/ethrc/Desktop/training/launch_act_eval2_sweep_*.log`).
- Final losses (training log): no_aug ≈ 0.07, dark_noise ≈ 0.07, dark_shadow ≈ 0.07,
  dark_noise_100k ≈ 0.060.
- One push retry was needed for the 100k run after a transient DNS error overnight;
  manually completed with `hf upload osammotg1/projet3-act-eval2-dark-noise-100k …`.
