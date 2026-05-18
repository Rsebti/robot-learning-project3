# Eval2 — FINE graded curriculum chain warm-start handoff (2026-05-18)

SO-101 `SO101PlaceCubeEval2-v1` (colour-conditioned: state = 18 =
qpos6+target6+goal_onehot6), raw-rgb **64×64**, SAC+C51. **9 checkpoints**
from a fine graded warm-start curriculum (each warm-started from the previous;
DR ramped gradually; exposure introduced LATE and GENTLY to avoid the
one-shot collapse seen on Eval1's coarse 3-stage run):

| ckpt | stage | DR |
|---|---|---|
| `ckpt_S1_base_noDR.pt` | S1 | base, **no DR** (cold, 4M) |
| `ckpt_S2_ctrl_mild.pt` | S2 | + control DR **mild** (ranges near native) |
| `ckpt_S3_ctrl_full.pt` | S3 | + control DR **full** (lighting off) |
| `ckpt_S4_light_mild.pt` | S4 | + lighting **mild** |
| `ckpt_S5_light_full.pt` | S5 | + lighting **full** |
| `ckpt_S6_jitter.pt` | S6 | + colour jitter |
| `ckpt_S7_expo030.pt` | S7 | + exposure DR **scale 0.30** (very gentle) |
| `ckpt_S8_expo060.pt` | S8 | + exposure DR **scale 0.60** |
| `ckpt_S9_expo100.pt` | S9 | + exposure DR **scale 1.00** (reduced envelope) |

S2–S9 each 1.5M steps, warm-chained. The `ExposureDRWrapper` envelope was
**reduced** (gain 0.5-2.0→0.7-1.5, wb .12→.07, gamma .7-1.5→.85-1.25,
white_lift .10→.05) and `EXPO_DR_SCALE` linearly interpolates it from neutral
(no-op) at scale 0 toward that reduced max at scale 1 — S7/S8/S9 = 0.30/0.60/
1.00. DR knobs only; reward/evaluate/env stay **NATIVE** (keep-native).

## Run
```bash
python infer_eval2_64.py --checkpoint ckpt_S6_jitter.pt --goal_color <color> --action_scale 0.1
```
(set `ROBOT_PORT`/`CAMERA_INDEX`/`CALIBRATION_ID`/`CALIBRATION_DIR` at the top;
`--goal_color` selects the target colour for this colour-conditioned policy;
start `--action_scale 0.1`, hand on e-stop.)

## Real↔sim mapping (validated, identical to deploy_utils/manipulator.py & Eval1)
- Gripper: real servo **[1°,75°] ↔ sim [-18°,120°]** (closed→open; -18 clamps
  the joint limit for a hard grasp).
- Arm: pure `deg2rad` **+ constant offset** `JOINT_OFFSET =
  [0, +0.080, +0.220, 0, 0, 0] rad` (lift/elbow recalibration; TCP z ≈
  0.010 m at grasp). Bidirectional, round-trip exact.
- Preprocess: center-crop → 128 → **64** (area); 64px encoder; strict
  state_dict load verified against the curriculum ckpts.
- Regional compromise (faithful approach/grasp/place, not the curled parked
  home). **Real-robot run is the final check.**

## How to judge (verdict = the planches; the metric is NOT trusted)
`planche_S{1..9}.png` = 50 clean-env rollouts each (final frame).
**Green sphere = goal marker (IGNORE it); the COLORED CUBE is the object.**
Genuine success = colored cube resting IN the bowl (NOT in the gripper /
hovering / outside / wrong-colour-bowl). The env's reported `success_at_end`
over- AND under-counts — **count genuine successes yourself on each planche.**

Honest factual notes (not a success grade — I did not self-count this run):
- The fine ramp is designed so each stage adds only a small DR increment, so
  the policy should keep ~its placement skill while gaining robustness —
  unlike Eval1's coarse S3 which collapsed (full visu+exposure at once).
- `ckpt_S9` (exposure 1.0): its `ckpt_best` never improved past the S8-warm
  start during S9 (full exposure suppresses the training metric). Treat S9 as
  "max exposure-robust but possibly lower genuine quality" — **compare its
  planche against S5–S8 and pick the best by eye.** Typically the
  quality/robustness sweet spot is a mid-late stage (≈ S5–S7).

## Bottom line
9 progressively-more-robust checkpoints + their planches are provided. Pick
the deployable one by visually comparing the planches (genuine cube-in-bowl
of the correct goal colour). The genuine ceiling remains the exploitable
success criterion (deliberately NOT changed — keeps env NATIVE; that fix
needs an explicit decision). Sim numbers are indicative; the real SO-101 run
is the only true gate.
