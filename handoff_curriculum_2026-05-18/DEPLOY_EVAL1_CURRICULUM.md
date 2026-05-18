# Eval1 — curriculum chain warm-start handoff (2026-05-18)

SO-101 `SO101PlaceCube-v1`, raw-rgb **64×64**, SAC+C51. Three checkpoints from
a chained warm-start curriculum (each warm-started from the previous):

| ckpt | stage | training |
|---|---|---|
| `ckpt_S1_base_noDR.pt` | S1 | base env, raw-rgb 64px, **no DR** (cold, 4M) |
| `ckpt_S2_warm_DRcontrol.pt` | S2 | warm S1 + **DR control** (arm/gripper stiffness·damping·delay·lag; lighting off), 3M |
| `ckpt_S3_warm_DRvisu_exposure.pt` | S3 | warm S2 + **DR visual (jitter) + DR exposure**, 3M |

## Run
```bash
python infer_eval1_64.py --checkpoint ckpt_S2_warm_DRcontrol.pt --action_scale 0.1
```
(set `ROBOT_PORT`, `CAMERA_INDEX`, `CALIBRATION_ID`, `CALIBRATION_DIR` at the
top of the script; start `--action_scale 0.1`, hand on e-stop.)

## Real↔sim mapping (validated, identical to deploy_utils/manipulator.py)
- Gripper: real servo **[1°,75°] ↔ sim [-18°,120°]** (closed→open; -18 clamps
  the joint limit for a hard grasp). Old `[-60.13,66.73]→[-10,120]` was wrong
  (mapped real-closed to sim-open).
- Arm: pure `deg2rad` **+ constant offset** `JOINT_OFFSET =
  [0, +0.080, +0.220, 0, 0, 0] rad` (lift/elbow homing-type recalibration so
  the gripper tip reaches the cube; TCP z ≈ 0.010 m at grasp). Added in
  `get_qpos`, subtracted in `set_target_qpos` (round-trip exact, err ~1e-15).
- Image preprocessing: center-crop → 128 → **64** (area), same two-step path
  as training. Encoder is the 64px architecture (strict state_dict load
  verified for S1/S2/S3).
- Caveat: the mapping is a *regional* compromise — faithful in the
  approach/grasp/place region (visually validated), not at the curled parked
  home (outside policy operation). **The real-robot run is the final check.**

## Honest per-stage assessment (verdict = the planches; verify yourself)
Planches `planche_S{1,2,3}.png` = 50 clean-env rollouts each (final frame).
**Green sphere = goal marker (ignore it); the COLORED CUBE is the object.**
Genuine success = colored cube resting IN the bowl (NOT in the gripper /
hovering / outside). The env's reported `success_at_end` is NOT reliable
(over- AND under-counts — verify visually on the planche).

- **S1 (base, no DR):** earlier visual count ≈ **26/50 (52%)**. Reported
  metric 0.88 was massively inflated (hover-with-cube-in-gripper and
  cube-outside-bowl count as success — the documented exploitable
  success-criterion loophole). Not a DR problem (S1 has no DR).
- **S2 (warm + DR control):** earlier visual count ≈ **33/50 (66%)** — best
  honest stage; warm+DR-control improved genuine placement and reduced the
  loophole share. Reported metric 0.44 *under*-counted (clean-env genuine
  is higher). **Recommended checkpoint.**
- **S3 (warm + DR visu + exposure):** ⚠️ **training COLLAPSED** under the
  aggressive visual+exposure DR — `success_at_end` stayed 0.00, return ~3.6,
  ckpt never improved past its S2-warm start. Included only for completeness
  as requested; **do not deploy S3** (matches the documented ceiling: raw-rgb
  policies degrade under aggressive lighting/exposure DR).

## Bottom line
Best deployable here = **S2** (`ckpt_S2_warm_DRcontrol.pt`), ≈66% genuine in
sim (visually). The genuine ceiling is the exploitable success criterion
(touching reward/evaluate to fix it was deliberately NOT done — that needs an
explicit decision; the curriculum keeps the env NATIVE). Treat the sim numbers
as indicative; the real SO-101 run is the only true gate.
