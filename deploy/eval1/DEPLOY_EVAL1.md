# Eval1 deployment — SO101 place-one-cube-in-bowl

Hand this to whoever has the robot. **Self-contained**: you only need
`infer_eval1.py` + the checkpoint + your arm calibration `.json`.

Eval1 = put **one** cube in the bowl. No colour, no sequence. The policy is
sim-trained (squint-native-iso, tag `eval1_final`); genuine ~75% in sim under
full realistic domain randomization, no hover artefact. Real transfer is
imperfect — treat the first runs as a bring-up, hand on the e-stop.

## Files you receive
| File | What | md5 |
|---|---|---|
| `infer_eval1.py` | standalone inference + robot driver | — |
| `eval1_ckpt.pt` | the policy weights (7.4 MB) | `0fbbcecfee4e07f3e78c3cfe36560ff2` |
| your `*.json` | arm calibration (you already have this from earlier infer.py runs) | — |

Verify the ckpt after transfer: `md5sum eval1_ckpt.pt` must equal the value above.

## Setup (Mac, same as before)
```bash
conda create -n squint python=3.10 -y && conda activate squint
pip install torch torchvision numpy opencv-python "lerobot[feetech]==0.4.3"
pip install rerun-sdk            # optional, live viewer (or run --no-viz)
```

## Edit the marked block at the top of `infer_eval1.py`
```
ROBOT_PORT      = "/dev/cu.usbmodemXXXX"   # ls /dev/cu.*
CAMERA_INDEX    = 0                        # try 0 / 1 / 2
CALIBRATION_ID  = "so101_follower_arm"     # your calibration .json name (no ext)
CALIBRATION_DIR = Path(__file__).parent    # folder holding that .json
```
Put your calibration `.json` next to the script (or point `CALIBRATION_DIR` at it).
This is the **same robot driver as the infer.py you already ran** — the gripper
servo mapping (`_g_servo_min/max`) is byte-identical, no re-tuning.

## Run
```bash
python infer_eval1.py --checkpoint eval1_ckpt.pt --action_scale 0.1
```
- `--action_scale` — START at `0.1` (slow/safe). Raise toward `0.15`–`0.25`
  once motion looks sane. It is a pure safety multiplier on the action.
- `--episode_steps` — default 150 (= 15 s @ 10 Hz). Sim episode is ~5 s.
- `--no-viz` — disable the Rerun window if rerun isn't installed.
- `--n_episodes N` — run N back-to-back; default waits for Enter each episode.
- `--goal_color` is **ignored** for Eval1 (no colour conditioning) — leave it.

`Enter` starts an episode; `Ctrl+C` quits and ramps the arm back to rest.

## Scene (must match sim)
- One ~2 cm cube on the table, the bowl in the workspace, light-grey table.
- Wrist RGB camera (the policy sees ONLY the wrist camera, 16×16 after
  cropping — it does not use a third-person view).
- Arm starts at the SO101 **`start`** keyframe; the script ramps there itself
  on each episode (`REST_QPOS`, wrist_roll = −90°).

## Verified contract (why this script, not the generic infer.py)
`infer_eval1.py` is `infer.py` adapted to the Eval1 checkpoint. Differences,
all verified against the checkpoint weights + the Eval1 sim env:

| | generic infer.py | **infer_eval1.py (correct for Eval1)** |
|---|---|---|
| state | 18 = qpos+target+**goal_onehot(6)** | **12 = qpos(6)+target(6)**, no onehot |
| control rate | 30 Hz | **10 Hz** (10 Hz-calibrated actuator) |
| delta cap /step | ±0.0333 / ±0.0667 | **±0.1 arm / ±0.2 gripper** |
| REST wrist_roll | 0.0 | **−π/2** |

These were checked by loading `eval1_ckpt.pt` into the script's network with
`load_state_dict(strict=True)` (passes) and a forward pass (valid 6-d action).
Running the **generic** infer.py on this ckpt would fail / behave wrongly.

State each step = `[ measured_qpos(6), controller_target_qpos(6) ]` in
joint order `[pan, lift, elbow, wrist_flex, wrist_roll, gripper]` (radians).
Action = 6-d ∈ [−1,1] → `clip(a·action_scale)` → `delta = a·[0.1×5,0.2]` →
`target = clip(target+delta, joint_limits)` → sent at 10 Hz.

## Safety
- First runs: `--action_scale 0.1`, hand on the e-stop, clear workspace.
- The arm self-returns to rest on `Ctrl+C`/exit; if it faults first,
  power-cycle then re-run.
- Sim2real is imperfect; lighting/camera placement affect it. If it flails,
  lower `action_scale`, check the camera index and that the wrist view roughly
  matches a top-down-ish cube+bowl framing.

## If you want to sanity-check without the robot
`python infer_eval1.py --checkpoint eval1_ckpt.pt` will still fail at
robot.connect() — that's expected. The network/contract was already validated
offline (strict checkpoint load + forward) before this hand-off.
