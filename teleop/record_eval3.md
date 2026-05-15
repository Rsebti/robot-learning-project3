# Eval 3 — record 100 demos (20 scenarios × 5 episodes)

## What this records

- **Task**: sequential 3-step pick-and-place. Four cubes of distinct colors are
  in the workspace; pick **three** of them in a given order and place each in
  the bowl. The fourth color is a **distractor** that stays on the table.
- **Goal-conditioning input** (encoded in the per-episode task string):
  - the 3 target colors, **in pick order**
  - the bowl position `(x, y)` in cm, robot frame
  - the distractor color
- **Scenarios**: 20 fixed `(4 colors, 3-color sequence)` combos.
- **Episodes per scenario**: 5 (100 total).
- **Bowl**: fixed at **(30, 20) cm** in the robot frame for the entire dataset.
- **Randomization**: the 4 cube positions are re-randomized per episode during
  the reset window; the bowl does not move (matches the TA spec — bowl
  positions fixed within a rollout).
- **Per-episode task string**:
  `"Pick {c1}, {c2}, {c3} blocks in this order and place each in bowl at (x,y) cm [distractor={c4}]"`

## Output

- Single HF dataset: **`kenzy17/projet3-eval3-v1`** (private)
- Pushed to HF after each scenario (so a crash mid-way doesn't lose progress)

## Pre-flight

1. Both arms plugged in, leader to follower facing each other, robot near the
   power supply.
2. **Calibration files** for `so101_follower` and `so101_leader` exist on this
   Mac. If not, the first lerobot-record call runs calibration interactively
   (mid-range → ENTER, full range → ENTER) — do it once and the script continues.
3. Wrist camera at OpenCV index 0 (USB UVC), confirmed visually.
4. HF authenticated as `kenzy17` (`hf auth login` then `hf auth whoami`).
5. **Six cubes** ready, one of each color (yellow, blue, green, violet, red,
   orange) — each scenario uses a different subset of 4.
6. One bowl, tape-marked at a single fixed position **(30, 20) cm** in the
   robot frame — it stays there for all 100 episodes.

(30,20)
## Run

```bash
bash teleop/record_eval3.sh
```

### Optional overrides (env vars)

```bash
# Quick smoke test before the real 100-episode run:
EPISODES_PER_SCENARIO=1 bash teleop/record_eval3.sh

# Different repo:
REPO_ID=kenzy17/projet3-eval3-test bash teleop/record_eval3.sh

# Different ports / camera index (re-detect with lerobot-find-port if re-plugged):
FOLLOWER_PORT=/dev/tty.usbmodemXXX LEADER_PORT=/dev/tty.usbmodemYYY CAM_INDEX=0 \
  bash teleop/record_eval3.sh

# Move the fixed bowl position (cm, robot frame):
BOWL_POS_X=25 BOWL_POS_Y=18 bash teleop/record_eval3.sh

# Custom scenario list (pipe-separated; first 3 colors = pick order, 4th = distractor):
SCENARIOS="yellow blue green violet|red orange yellow blue" \
  bash teleop/record_eval3.sh
```

## During recording

For each of the 20 scenarios:

1. The script prints a banner: which **4 cubes** to place, the **pick order**
   for the 3 targets, the **distractor** color, and the **bowl position**.
2. Place the 4 cubes and the bowl, then press ENTER.
3. lerobot-record starts the 5 episodes for that scenario.
4. **Per episode**: 90 s record + 15 s reset. During the record window, pick
   the 3 targets in order and place each in the bowl; leave the distractor.
   During the 15 s reset window, re-randomize the 4 cube positions (bowl stays).
5. After the 5th episode, the dataset is encoded and pushed to HF — **do not
   press any key during the silent encoding phase** (~10–15 s); buffered
   keypresses fire on the next start and skip the first episode of the next
   scenario.
6. Move on to the next scenario.

## Stop / resume

- `Ctrl+C` mid-run is OK; whatever was already pushed to HF is safe.
- Resume from a specific scenario with `START_FROM_SCENARIO` (1-indexed):
  ```bash
  START_FROM_SCENARIO=8 bash teleop/record_eval3.sh
  ```
  The script skips scenarios 1–7 and keeps `--resume=true` so episodes append
  to the existing dataset.

## After all 100 episodes

- Dataset visible at https://huggingface.co/datasets/kenzy17/projet3-eval3-v1
- Spot-check a few episodes (one per scenario family):
  ```bash
  lerobot-replay \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5B141129871 \
    --robot.id=so101_follower \
    --dataset.repo_id=kenzy17/projet3-eval3-v1 \
    --dataset.episode=0   # try 0, 25, 50, 75, 99 to cover the scenario spread
  ```
- This expert data is used to bootstrap the Eval 3 RL policy (training-efficiency
  warmstart — encouraged by the TA spec, RL still mandatory for the final policy).



##### kenzy's pc
follower port :'/dev/ttyACM0'
leader port : '/dev/ttyACM1'

camera : /dev/video4
# 
newgrp dialout
source /home/kenzy/lerobot-venv/bin/activate

# smoketest
 EPISODES_PER_SCENARIO=1 EPISODE_TIME_S=120 RESET_TIME_S=5 \
  REPO_ID=kenzy17/projet3-eval3-test \
    bash teleop/record_eval3.sh

bash teleop/record_eval3.sh

# 
bowl at (30,20)

#
Before running, in that terminal:
  groups | grep dialout || newgrp dialout    # skip if you rebooted
  source /home/kenzy/lerobot-venv/bin/activate

  During the run:
  - 20 scenario banners; place the 4 cubes + press ENTER for each.
  - 5 episodes per scenario, pushed to HF after each scenario.
  - If interrupted (Ctrl+C, crash, gripper overload), resume from where you stopped: START_FROM_SCENARIO=<n> EPISODE_TIME_S=120 bash teleop/record_eval3.sh — already-pushed
  scenarios are safe.

   GREEN  ->  VIOLET  ->  RED
   VIOLET  ->  RED  ->  ORANGE
   RED  ->  ORANGE  ->  YELLOW
   ORANGE  ->  YELLOW  ->  BLUE
   YELLOW  ->  GREEN  ->  
    BLUE  ->  VIOLET  ->  ORANGE
    GREEN  ->  YELLOW  ->  VIOLET
    VIOLET  ->  BLUE  ->  YELLOW
    RED  ->  GREEN  ->  BLUE

START_FROM_SCENARIO=12 bash teleop/record_eval3.sh
ORANGE  ->  VIOLET  ->  GREEN
YELLOW  ->  RED  ->  ORANGE
blue orange red
GREEN  ->  BLUE  ->  ORANGE
VIOLET  ->  YELLOW  ->  RED
RED  ->  VIOLET  ->  GREEN
ORANGE  ->  GREEN  ->  YELLOW
YELLOW  ->  VIOLET  ->  BLUE
BLUE  ->  RED  ->  GREEN