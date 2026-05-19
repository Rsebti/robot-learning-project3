# infer_eval2.py - quick reference

Uses the same LeRobot ACT deploy stack as `infer_eval1_act_nocube.py`
(shared helpers in `act_infer_common.py`: 8D env_state, DSHOW camera patch,
joint-position actions @ 30 Hz).

Goal-conditioned deploy script for Eval-2 / Eval-3. Wraps the lerobot
Robot + Policy Python APIs and injects `observation.environment_state`
per frame, since `lerobot-record` cannot do that on its own.

## Run

Default policy (`hudela390/projet3-act-eval2-v1-goal`) and Eval-2 bowl:

```bash
# Linux
python deploy/infer_eval2.py \
    --target_color red \
    --bowl_x -0.155 \
    --bowl_y  0.295 \
    --follower_port /dev/ttyACM0

# Windows
python deploy/infer_eval2.py `
    --target_color red `
    --bowl_x -0.155 `
    --bowl_y  0.295 `
    --follower_port COM3
```

## CLI flags

| flag | required | default | meaning |
| --- | --- | --- | --- |
| `--target_color`   | yes | -                                 | one of `yellow, orange, red, blue, green, violet` |
| `--bowl_x`         | yes | -                                 | bowl x (m) in robot base frame, right + / left - |
| `--bowl_y`         | yes | -                                 | bowl y (m) in robot base frame, forward + / back - |
| `--policy_path`    |     | `hudela390/projet3-act-eval2-v1-goal` | HF repo id or local pretrained_model dir |
| `--follower_port`  |     | `COM3`                            | serial port of the SO-101 follower |
| `--camera_index`   |     | `1`                               | OpenCV index of the wrist camera |
| `--fps`            |     | `30`                              | control loop frequency |
| `--episode_time_s` |     | `15.0`                            | duration of one rollout |
| `--num_episodes`   |     | `1`                               | how many sequential rollouts |
| `--reset_time_s`   |     | `10.0`                            | pause between episodes for manual reset |
| `--device`         |     | `cuda` if available else `cpu`    | torch device for the policy |

## What the env_state looks like

```
env_state = [c_yellow, c_orange, c_red, c_blue, c_green, c_violet,
             bowl_x_m, bowl_y_m]
```

`--target_color red --bowl_x -0.155 --bowl_y 0.295` becomes
`[0, 0, 1, 0, 0, 0, -0.155, 0.295]` and is fed to the policy every frame.

## Eval-3 multi-step

For the sequential Eval-3 task: call this script three times in a row, one
per (color, bowl) pair. Between calls, the robot stays where it is - no
reset, just kick off the next color.
