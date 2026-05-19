# SO-101 kinematics & control toolkit

URDF-native FK + IK + waypoint executor + camera pipeline. Built for
zero-shot deployment on the SO-101 follower.

## TL;DR — order of operations on the physical robot

```
1.  conda activate trim
2.  cd C:/Users/hugod/project3

3.  python -m toolset.calibration_scripts.calibrate_motor_offsets \
        --port COM3 --mode anchor
        # ~30 s. Aligns lerobot motor degrees -> URDF radians.

4.  python -m toolset.calibration_scripts.measure_gripper_tip \
        --port COM3 --n_poses 3
        # ~1 min. Touch the table 3x with different gripper orientations.

5.  python -m toolset.kinematics.probe_cli --port COM3 --target gripper_tip
        # sanity-check: probe the gripper tip in user frame and compare to ruler.

6.  python -m toolset.calibration_scripts.calibrate_camera_chessboard \
        --mode live --port COM3 --n_captures 20
        # ~5 min. Take 20 chessboard frames; outputs camera_intrinsics.npz.
        # SKIP if you trust the legacy camera_calibration.yaml.

7.  python -m toolset.calibration_scripts.calibrate_hand_eye \
        --port COM3 --n_poses 15
        # ~3 min. Recovers T_cam_in_wrist via OpenCV calibrateHandEye.
        # Depends on steps 3 and 6.

8.  python -m toolset.calibration_scripts.probe_cubes \
        --port COM3 --n 4
        # Validate: place a cube at known xy, touch with gripper, check FK error.
        # If max error > 1 cm, go back to step 3 or 4.

9.  Deploy:
    a) ACT policy:    use deploy/infer_eval1_act_nocube.py (8D, no cube_xy)
    b) Hand-coded:    python -m toolset.kinematics.execute_path \
                          --port COM3 --waypoints_yaml <your.yaml>
```

## Files

```
toolset/kinematics/
  urdf_fk.py            URDF-native FK; targets {wrist, wrist_roll, gripper_frame, gripper_tip}
  motor_to_urdf.py      lerobot motor deg <-> URDF rad (loads motor_offsets.yaml)
  ik.py                 damped least-squares IK; position-only or vertical-down
  execute_path.py       joint-space waypoint executor with speed presets
  probe_cli.py          read current pose, print xyz (URDF + user frames)
  config.py             loads config.yaml
  tests/
    test_fk.py          offline FK sanity (q=0 forward, sweep monotonic, ...)
    test_ik.py          offline IK round-trip + workspace rejection

toolset/calibration_scripts/
  calibrate_motor_offsets.py     anchor- or ruler-mode offset recovery
  measure_gripper_tip.py         touch table x3 -> gripper-tip offset
  calibrate_camera_chessboard.py 9x6 chessboard intrinsics, live or offline
  calibrate_hand_eye.py          cv2.calibrateHandEye -> T_cam_in_wrist
  probe_cubes.py                 ground-truth FK error survey

toolset/perception/
  cube_localization.py    end-to-end pipeline using new FK + new calibration
  hsv_config.yaml         per-color HSV ranges (tunable)

toolset/configs/kinematics/
  config.yaml                       master config (workspace, joint limits, IK tuning)
  motor_offsets.yaml                written by calibrate_motor_offsets
  camera_intrinsics.{npz,yaml}      written by calibrate_camera_chessboard
  camera_in_wrist.{npy,yaml}        written by calibrate_hand_eye
  example_pick_and_place.yaml       waypoint demo (full pick+place sequence)
```

## Frame conventions

- **URDF base**: +x forward, +y left, +z up (standard ROS).
- **User**: +x right, +y forward, +z up. Used in CLI flags and CSV files.
- Conversion: `user = (-y_urdf, +x_urdf, +z_urdf)`,
              `urdf = (+y_user, -x_user, +z_user)`.

The CLI tools accept whichever frame is most natural and convert internally.

## Verification gates (DO before declaring deploy-ready)

- [ ] `python -m toolset.kinematics.tests.test_fk` passes (offline, no robot)
- [ ] `python -m toolset.kinematics.tests.test_ik` passes
- [ ] `probe_cli` at the standard rest pose returns xyz within 5 mm of ruler
- [ ] `probe_cubes` over 4 ground-truth positions: mean error < 1 cm, max < 1.5 cm
- [ ] `calibrate_hand_eye` reports board-position std < 5 mm across poses
- [ ] `cube_localization.locate` on a static frame: xyz within 1 cm of ruler

If any of these fail, recalibrate that step's dependencies before moving on.

## Gotchas

- **5-DoF arm + strict vertical-down**: some xy positions (especially near the
  base, or off to the side) can't satisfy both `position` AND `vertical-down`
  simultaneously. Use `mode: position` for transits, `position_vertical` only
  for grasp + release.
- **box2ai vs URDF convention**: box2ai's chain skips shoulder_pan and uses
  Rx for wrist_roll; ours follows the URDF literally with Rz on every revolute
  joint. Don't mix the two.
- **Legacy fallback**: if no chessboard/hand-eye output exists, `cube_localization`
  falls back to `toolset/configs/camera_calibration.yaml` (self-cal from demos,
  mean_err ~1.2 cm). Run chessboard + hand-eye to beat that.
