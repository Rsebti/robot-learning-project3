# Eval 1 — record 60 demos (6 colors × 10 episodes)

## What this records

- **Task**: pick a single colored cube and place it in the bowl
- **Colors**: yellow, blue, green, violet, red, orange (recorded in this order)
- **Episodes per color**: 10 (60 total)
- **Randomization**: cube position + bowl position randomized per episode (during the 10 s reset window)
- **Per-episode task string**: `"Pick {color} block and place in bowl"` — keeps color info in the dataset metadata so ACT can later condition on language

## Output

- Single HF dataset: **`osammotg1/projet3-eval1-v1`** (private)
- Pushed to HF after each color (so a crash mid-way doesn't lose progress)

## Pre-flight

1. Both arms plugged in, leader to follower facing each other, robot near the power supply.
2. **Calibration files** for `so101_follower` and `so101_leader` exist on this Mac. If not, the first lerobot-record call will run calibration interactively (move bras to mid-range, press ENTER; full range, press ENTER) — that's fine, do it once and the script will keep going.
3. Wrist camera at OpenCV index 0 (USB UVC), confirmed visually.
4. HF authenticated as `osammotg1` (`hf auth whoami`).
5. Six cubes ready, one of each color.

## Run

```bash
bash teleop/record_eval1.sh
```

### Optional overrides (env vars)

```bash
# Quick 2-episode smoke test before the real 60-episode run:
EPISODES_PER_COLOR=2 bash teleop/record_eval1.sh

# Different repo:
REPO_ID=osammotg1/projet3-eval1-test bash teleop/record_eval1.sh

# Different ports / camera index (re-detect with lerobot-find-port if devices were re-plugged):
FOLLOWER_PORT=/dev/tty.usbmodemXXX LEADER_PORT=/dev/tty.usbmodemYYY CAM_INDEX=0 bash teleop/record_eval1.sh
```

## During recording

For each of the 6 colors:

1. The script prints a banner telling you which color cube to put in the workspace.
2. Press ENTER when the scene is ready.
3. lerobot-record starts the 10 episodes for that color.
4. **Per episode**: 15 s record + 10 s reset (you reposition cube + bowl during the reset window).
5. After the 10th episode, the dataset is encoded and pushed to HF — **do not press any key during the silent encoding phase** (~10–15 s); buffered keypresses fire on the next start and skip the first episode of the next color.
6. Move on to the next color.

## Stop / resume

- `Ctrl+C` mid-run is OK; whatever was already pushed to HF is safe.
- To resume from a specific color, edit `COLORS=(...)` in the script to start from that color, and keep `--resume=true` enabled (already automatic for steps 2–6).

## After all 60 episodes

- Dataset visible at https://huggingface.co/datasets/osammotg1/projet3-eval1-v1
- Spot-check a few episodes:
  ```bash
  lerobot-replay \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5B141129871 \
    --robot.id=so101_follower \
    --dataset.repo_id=osammotg1/projet3-eval1-v1 \
    --dataset.episode=0   # try 0, 10, 20, 30, 40, 50 to cover all 6 colors
  ```
- Then trigger ACT training using the existing `train/launch_act.sh` (update `dataset.repo_id` accordingly).
