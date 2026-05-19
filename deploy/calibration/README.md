# SO-101 follower calibration (project-local)

**Single file used by all deploy scripts:**

```
deploy/calibration/so101_follower.json
```

Do not edit copies under `deploy/eval1/` etc. — those are legacy duplicates; use this folder.

## When to recalibrate

- You replaced or remounted a motor (encoder zero shifted).
- `robot.connect()` fails on homing_offset / range errors.
- Joint angles in inference look wrong vs physical pose (after motor work).

## Full calibration (new motor or first time on this PC)

```powershell
conda activate lerobot
cd C:\Users\hugod\project3

# Interactive: follow on-screen poses (mid-range, gripper open/close, etc.)
lerobot-calibrate --robot.type=so101_follower --robot.port=COM3 --robot.id=so101_follower

# Copy result into the repo (all infer_* scripts read this path)
python -m toolset.calibration_scripts.manage_calibration pull

# So lerobot-record / teleop use the same file as deploy
python -m toolset.calibration_scripts.manage_calibration install
```

## Motor remounted only (ranges still OK)

Put the arm in your rest pose (same as `deploy/homes.py` → `eval1_rest`), then:

```powershell
python -m toolset.calibration_scripts.rehome_only --port COM3 `
    --calibration deploy/calibration/so101_follower.json
```

Press **SPACE** in the preview when the pose looks right. Then:

```powershell
python -m toolset.calibration_scripts.manage_calibration install
```

## If connect fails: homing_offset out of range (+/-2047)

```powershell
python -m toolset.calibration_scripts.manage_calibration clip
python -m toolset.calibration_scripts.manage_calibration install
```

Clipping only unblocks connect; joints that were clipped may still need a full `lerobot-calibrate`.

## Inspect current file

```powershell
python -m toolset.calibration_scripts.manage_calibration show
python -m toolset.calibration_scripts.manage_calibration validate
```

## URDF / FK offsets (separate from LeRobot JSON)

For `toolset/kinematics` and cube CV back-projection:

```powershell
python -m toolset.calibration_scripts.calibrate_motor_offsets --port COM3 --mode anchor
```

That writes `toolset/configs/kinematics/motor_offsets.yaml` — not the LeRobot bus calibration.
