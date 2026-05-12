# Handoff: run ACT inference on the SO-101 (laptop session)

Paste this whole file into a fresh Claude Code session **on your laptop next to the robot**.
The remote 5090 box has finished training; everything below is the recipe to download
the policies and drive the arm with them.

---

## What was trained

Two ACT policies, both small-data ACT (`dim_model=256`, 17.1 M params), trained for 30,000 steps
on **`osammotg1/projet3-eval1-bowl1-v1`** (39 of 40 episodes used; episode 30 was excluded as bad).
W&B project: `tom-gazzini-ethrc/projet3-act`. Dataset task strings the policies have seen:

```
Pick yellow block and place in bowl at (16,32) cm [bowl_pos=1]
Pick blue   block and place in bowl at (16,32) cm [bowl_pos=1]
Pick green  block and place in bowl at (16,32) cm [bowl_pos=1]
Pick violet block and place in bowl at (16,32) cm [bowl_pos=1]
Pick red    block and place in bowl at (16,32) cm [bowl_pos=1]
```

> The bowl was **fixed at position 1 (x=16 cm, y=32 cm)** during recording, so don't expect
> the policy to generalize to other bowl positions. The cube color varies; the policy was
> trained on all five colors above.
>
> ACT in lerobot is **vision + proprioception only** — it does NOT condition on the task
> string. Changing `SINGLE_TASK` at inference time only relabels the eval dataset; it does
> not change the policy's behavior.

## The two HF policy repos

| name        | HF repo                                                    | aug profile (training-time)                                                                                                                            |
| ----------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| dark_noise  | `osammotg1/projet3-act-eval1-v1-dark-noise`               | brightness 0.2–1.1, contrast 0.5–1.5, saturation 0.2–1.3, hue ±0.03, gaussian noise σ=0.06, autocontrast p=0.35, gaussian blur (3 of 7 per sample, random order) |
| dark_shadow | `osammotg1/projet3-act-eval1-v1-dark-shadow`              | brightness 0.2–1.1, contrast 0.5–1.5, gaussian noise σ=0.04, **random_erasing p=0.6 (occlusion)**, autocontrast p=0.5 (3 of 5 per sample, random order) |

Both reach loss ≈ 0.07 by step 30k. They should be roughly comparable on clean scenes;
dark_noise is biased toward lighting robustness, dark_shadow toward occlusion robustness.
Try both, see which one survives your real-world conditions better.

## Hardware prereqs

- SO-101 follower arm plugged in (USB serial). On Linux `ls /dev/tty*` will show
  `/dev/ttyACM0` or `/dev/ttyUSB0`; on macOS `/dev/tty.usbmodem...`; on Windows
  it's a `COM`-port in Device Manager.
- Wrist camera (the same OpenCV camera used during recording). Default index 0,
  640×480 @ 30 fps. If you don't know the index: `lerobot-find-cameras opencv`.
- Cube (any of the 5 trained colors) and the bowl at bowl-position-1
  (16 cm right, 32 cm forward from the robot base — same as during recording).

## Software prereqs

```bash
# In your local clone of robot-learning-project3, with the lerobot venv active:
hf auth login                  # paste your write token; verify with `hf auth whoami`
hf auth whoami                 # should print: user: osammotg1
```

If you don't have the venv yet:
```bash
cd robot-learning-project3
uv sync                        # or: pip install -e . in a fresh venv
```

## Quick start (the command you'll actually run)

`deploy/infer.sh` is already in this repo and wraps `lerobot-record` in autonomous (policy-driven) mode.
Every knob is a plain env var. Minimum required: `FOLLOWER_PORT` and `POLICY_PATH`.

```bash
# Linux example (substitute your real port)
FOLLOWER_PORT=/dev/ttyACM0 \
POLICY_PATH=osammotg1/projet3-act-eval1-v1-dark-noise \
  bash deploy/infer.sh
```

```bash
# macOS example
FOLLOWER_PORT=/dev/tty.usbmodem5B141129871 \
POLICY_PATH=osammotg1/projet3-act-eval1-v1-dark-noise \
  bash deploy/infer.sh
```

```powershell
# Windows (PowerShell)
$env:FOLLOWER_PORT="COM5"
$env:POLICY_PATH="osammotg1/projet3-act-eval1-v1-dark-noise"
bash deploy/infer.sh
```

First run will download the policy from HF into the local cache and start episode 1
immediately. Press **Right-Arrow** during a rollout to end the episode early.

## Tunable knobs (env vars on the same line as `bash deploy/infer.sh`)

| variable          | default                                          | what it does                                                                                                          |
| ----------------- | ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| `POLICY_PATH`     | `Rsebti/projet3-act-sanity`                      | **change me.** Either an HF repo id or a local `pretrained_model/` dir. Use one of the two listed above.              |
| `FOLLOWER_PORT`   | *(required)*                                     | Serial port to the SO-101 follower.                                                                                   |
| `NUM_EPISODES`    | `5`                                              | **How many rollouts to record.** Bump to 10–20 for a real comparison between dark_noise and dark_shadow.              |
| `EPISODE_TIME_S`  | `15`                                             | **Wall-clock seconds per rollout.** Training demos were 55s; 15–30s is usually enough for a pick-and-place.           |
| `RESET_TIME_S`    | `10`                                             | Seconds you get between rollouts to reset the scene (move the cube, swap colors, etc.).                               |
| `SINGLE_TASK`     | `Pick block and place in bowl`                   | **Task string written into the eval dataset.** Doesn't change policy behavior, but use one of the 5 trained-on strings if you want the eval metadata to match the training distribution. |
| `EVAL_REPO_ID`    | `Rsebti/projet3-eval-sanity`                     | HF repo the eval dataset is logged to. **You probably want to change this** to e.g. `osammotg1/projet3-eval-dark-noise`. |
| `PUSH_TO_HUB`     | `false`                                          | Set `true` to also push the recorded eval dataset to HF.                                                              |
| `POLICY_DEVICE`   | `cuda`                                           | Set to `cpu` if your laptop has no GPU. Set to `mps` on Apple Silicon.                                                |
| `ROBOT_TYPE`      | `so101_follower`                                 | Don't change unless you swapped robots.                                                                               |
| `CAMERA_INDEX`    | `0`                                              | OpenCV index; bump to 1/2/etc. if your wrist cam isn't on 0.                                                          |
| `CAMERA_WIDTH`    | `640`                                            | Match the training resolution.                                                                                        |
| `CAMERA_HEIGHT`   | `480`                                            | Match the training resolution.                                                                                        |
| `CAMERA_FPS`      | `30`                                             | Match the training FPS.                                                                                               |
| `DISPLAY_DATA`    | `true`                                           | Show the Rerun viewer while rolling out.                                                                              |

### About "tags"

Two different things both get called "tags" in this stack — pick the one you mean:

1. **Task string** — what the human writes on the dataset for each batch (`SINGLE_TASK` above).
   Each rollout writes ONE task string; if you want a mix, run `deploy/infer.sh` multiple times,
   once per color, with the matching `SINGLE_TASK`.

2. **Dataset tags** — free-form labels on the dataset's HF repo, set via the lerobot CLI
   `--dataset.tags` argument. To use these, edit the `lerobot-record` call in
   `deploy/infer.sh` and add `--dataset.tags='[\"dark_noise\",\"eval1\",\"bowl1\"]'`
   before the closing line. They show up as topic chips on the HF dataset page.

## Recommended evaluation runs

This is the protocol I'd use for a clean apples-to-apples comparison. Two batches of 10 each.
Each takes ~5 minutes including resets. Move the cube to a fresh spot during every reset window.

```bash
# Batch 1 — dark_noise policy, 10 rollouts × 20s, yellow cubes
FOLLOWER_PORT=/dev/ttyACM0 \
POLICY_PATH=osammotg1/projet3-act-eval1-v1-dark-noise \
EVAL_REPO_ID=osammotg1/projet3-eval-dark-noise-yellow \
NUM_EPISODES=10 \
EPISODE_TIME_S=20 \
RESET_TIME_S=8 \
SINGLE_TASK="Pick yellow block and place in bowl at (16,32) cm [bowl_pos=1]" \
PUSH_TO_HUB=true \
  bash deploy/infer.sh

# Batch 2 — dark_shadow policy, same conditions
FOLLOWER_PORT=/dev/ttyACM0 \
POLICY_PATH=osammotg1/projet3-act-eval1-v1-dark-shadow \
EVAL_REPO_ID=osammotg1/projet3-eval-dark-shadow-yellow \
NUM_EPISODES=10 \
EPISODE_TIME_S=20 \
RESET_TIME_S=8 \
SINGLE_TASK="Pick yellow block and place in bowl at (16,32) cm [bowl_pos=1]" \
PUSH_TO_HUB=true \
  bash deploy/infer.sh
```

Repeat per cube color you want to test (`yellow`, `blue`, `green`, `violet`, `red`).

## Troubleshooting

- **`Could not open camera at index 0`** → either the wrist cam isn't connected, or a Rerun
  viewer is still holding it. Close any leftover Python processes and try again. `lerobot-find-cameras opencv` lists what's available.
- **`Timeout connecting to follower`** → wrong `FOLLOWER_PORT`, or a previous run didn't release
  the serial port. Wait 5 s and retry, or unplug+replug the arm USB.
- **Policy moves but never grabs the cube** → check that the wrist camera framing matches the
  training data. Same height/angle on the gripper. If it's mounted differently from when the
  demos were recorded, the policy will be confused.
- **`Forbidden` when pushing eval dataset** → `hf auth whoami` should say `osammotg1`. If it
  says someone else, do `hf auth login` again with your write token.
- **Policy from HF cache is stale** → `rm -rf ~/.cache/huggingface/hub/models--osammotg1--projet3-act-eval1-v1-*`
  and rerun (forces a fresh download).

## Files in this folder

- `deploy/infer.sh` — the wrapper you just ran.
- `deploy/inference.md` — original notes from when the script was first written.
- `deploy/deploy_policy.md` — broader notes on policy deployment.

## What I want you (Claude on the laptop) to do

When the human runs this handoff:

1. Confirm `hf auth whoami` shows `osammotg1` (or offer to run `hf auth login`).
2. Ask which policy they want to test first: `dark_noise` or `dark_shadow` (or both).
3. Ask which `SINGLE_TASK` they want (drop-down of the 5 trained-on color strings) and how
   many `NUM_EPISODES` × `EPISODE_TIME_S` they want.
4. Build the env-var block, paste the command, and let them run it.
5. After each batch, optionally help them push the recorded eval dataset to HF and link
   the dataset URL in chat.
