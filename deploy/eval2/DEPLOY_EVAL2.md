# Eval2 deployment — SO101 colour-conditional place

Hand this to whoever has the robot. Self-contained: `infer_eval2.py` + the
checkpoint + your arm calibration `.json`.

**Eval2** = TWO cubes ADJACENT side-by-side, different colours (goal +
distractor). You tell the policy which colour to fetch via `--goal_color`; it
must put THAT cube in the bowl and leave the other one out. Sim-trained
(squint-native-iso, tag `eval2_final`); genuine ~68% in sim under realistic
domain randomization (dynamics + camera + realistic lab lighting), no hover
artefact, colour-conditional criterion verified robust. Real transfer is
imperfect — bring-up carefully, hand on the e-stop.

> Honest note: Eval2 ~68% < Eval1 ~75% — it is a harder compound task
> (colour-select + pick + place + avoid distractor). The remaining sim2real
> gap is strong lighting variation degrading colour discrimination at the
> 16×16 input (documented ceiling). Keep lab lighting reasonably steady.

## Files you receive
| File | What | md5 |
|---|---|---|
| `infer_eval2.py` | standalone inference + robot driver | — |
| `eval2_ckpt.pt` | the policy weights (7.4 MB) | `a67cd585384c3b4cf03e038ae47d8461` |
| your `*.json` | arm calibration (your own, as for earlier infer.py runs) | — |

Verify after transfer: `md5sum eval2_ckpt.pt` must equal the value above.

## Setup (Mac, same as Eval1)
```bash
conda create -n squint python=3.10 -y && conda activate squint
pip install torch torchvision numpy opencv-python "lerobot[feetech]==0.4.3"
pip install rerun-sdk            # optional live viewer (or --no-viz)
```

## Edit the marked block at the top of `infer_eval2.py`
```
ROBOT_PORT      = "/dev/cu.usbmodemXXXX"   # ls /dev/cu.*
CAMERA_INDEX    = 0                        # try 0 / 1 / 2
CALIBRATION_ID  = "so101_follower_arm"     # your calibration .json name (no ext)
CALIBRATION_DIR = Path(__file__).parent
```
Same robot driver as the infer.py you already ran (gripper servo mapping
byte-identical, no re-tuning).

## Run
```bash
python infer_eval2.py --checkpoint eval2_ckpt.pt --goal_color 0 --action_scale 0.1
```
- `--goal_color` — **REQUIRED, meaningful**: which colour cube to pick.
  `0 red  1 blue  2 green  3 yellow  4 purple  5 orange` (COLOR_PALETTE order).
  Set it to the colour of the cube you want placed; the other cube is the
  distractor and must be a different one of these colours.
- `--action_scale` — START at `0.1`, raise toward `0.15`–`0.25` once safe.
- `--episode_steps` — default 150 (= 15 s @ 10 Hz).
- `--no-viz` — disable the Rerun window.
- `--n_episodes N` — N back-to-back (else Enter per episode).

`Enter` starts an episode; `Ctrl+C` quits and ramps the arm to rest.

## Scene (must match sim)
- TWO ~2 cm cubes **adjacent, side-by-side** (the distractor sits ~one
  cube-width in +x from the goal cube). Different palette colours.
- The bowl in the workspace, light-grey table, wrist RGB camera (the policy
  sees ONLY the wrist camera, 16×16 after cropping).
- Arm starts at the SO101 `start` keyframe; the script ramps there each
  episode (REST_QPOS, wrist_roll = −90°).
- **Success** = the `--goal_color` cube in the bowl AND the distractor NOT
  in the bowl.

## Verified contract (why this script, not the generic infer.py)
`infer_eval2.py` = the corrected infer.py base. The generic infer.py had the
right state size for Eval2 (18) but the WRONG control rate / delta caps /
rest pose. Fixed + verified by strict checkpoint load + forward pass:

| | generic infer.py | **infer_eval2.py** |
|---|---|---|
| state | 18 (qpos+target+goal_onehot) | 18 — same, **auto-detected** ✓ |
| control rate | 30 Hz | **10 Hz** |
| delta cap /step | ±0.0333 / ±0.0667 | **±0.1 arm / ±0.2 gripper** |
| REST wrist_roll | 0.0 | **−π/2** |

State each step = `[ measured_qpos(6), controller_target_qpos(6),
goal_colour_onehot(6) ]` (radians; joint order pan, lift, elbow, wrist_flex,
wrist_roll, gripper). Action pipeline identical to Eval1, at 10 Hz.

## Safety
- First runs `--action_scale 0.1`, hand on the e-stop, clear workspace.
- Arm self-returns to rest on Ctrl+C/exit; if it faults first, power-cycle.
- If it picks the wrong-colour cube, double-check `--goal_color` matches the
  cube you actually want and that lighting is steady (colour discrimination
  is the hardest part of this task).
