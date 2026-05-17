# SO-101 sim-to-real joint anchors

- Generated: `2026-05-17T22:42:53+00:00`
- Host: `student-net-als-3849.intern.ethz.ch`
- Joint order: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`
- Units: degrees (radians in parens). Achieved angles, not commanded.
- lerobot convention: `use_degrees=True`. Gripper is the servo angle, NOT a jaw-opening percentage.

## Calibration snapshot

- Source on this machine: `/Users/admin/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower.json`
- Snapshot file: `sim_anchors_2026-05-18_calibration.json` (sibling to this MD)
- SHA256: `d96939ac2d265cd8d5e409e5ca07700cf1f5d6607a56cbb0efbdaa93d83d2b5d`
- Key homing offsets:
  - `shoulder_pan.homing_offset = 1586`
  - `shoulder_lift.homing_offset = -1075`
  - `elbow_flex.homing_offset = 853`
  - `wrist_flex.homing_offset = -847`
  - `wrist_roll.homing_offset = 1588`
  - `gripper.homing_offset = 1846`

## NOT in scope (v1)

- Dynamics / mass / torque / friction comparison
- Forward-kinematics end-effector xyz (deferred; would depend on the MJCF we are trying to validate)
- Sim-side validation (Ryan's responsibility)
- Cross-machine averaging (single-machine snapshot)

## Poses

### 1. `home_pose`

Training home for the eval1 dark-shadow policy. Arm folded, gripper closed, wrist hovering above workspace. This is the operational baseline every rollout starts from.

**Commanded target:**

```
  shoulder_pan    =    -2.00 deg
  shoulder_lift   =   -92.20 deg
  elbow_flex      =   +97.70 deg
  wrist_flex      =   +75.20 deg
  wrist_roll      =    +0.00 deg
  gripper         =    +1.20 deg
```

**Achieved (servo-reported after settle):**

```
  shoulder_pan    =    -1.80 deg  ( -0.0315 rad)
  shoulder_lift   =   -92.09 deg  ( -1.6072 rad)
  elbow_flex      =   +97.23 deg  ( +1.6970 rad)
  wrist_flex      =   +74.11 deg  ( +1.2935 rad)
  wrist_roll      =    +0.48 deg  ( +0.0084 rad)
  gripper         =    +0.74 deg  ( +0.0129 rad)
```

**Wrist cam frame:** `home_pose.jpg`

**External photo taken by operator:** yes (during 10s window)

### 2. `shoulder_pan_30`

Home pose with shoulder_pan rotated to +30 deg. Tests the shoulder_pan axis sign and direction: the gripper should swing roughly 30 deg about the base from its home position.

**Commanded target:**

```
  shoulder_pan    =   +30.00 deg
  shoulder_lift   =   -92.20 deg
  elbow_flex      =   +97.70 deg
  wrist_flex      =   +75.20 deg
  wrist_roll      =    +0.00 deg
  gripper         =    +1.20 deg
```

**Achieved (servo-reported after settle):**

```
  shoulder_pan    =   +29.49 deg  ( +0.5148 rad)
  shoulder_lift   =   -92.09 deg  ( -1.6072 rad)
  elbow_flex      =   +97.23 deg  ( +1.6970 rad)
  wrist_flex      =   +74.11 deg  ( +1.2935 rad)
  wrist_roll      =    +0.40 deg  ( +0.0069 rad)
  gripper         =    +0.74 deg  ( +0.0129 rad)
```

**Wrist cam frame:** `shoulder_pan_30.jpg`

**External photo taken by operator:** yes (during 10s window)

### 3. `wrist_rolled_45`

Home pose with wrist_roll rotated to +45 deg. Tests the wrist_roll axis (the joint that carries the +1588 homing_offset on this Mac per project_calibration_shift_2026-05-17). +45 stays within the documented folded-arm wrist plateau (50/80 deg per feedback_so101_wrist_camera_collision); +90 caused the wrist cam to collide with the robot base on 2026-05-18, hence the safer magnitude here.

**Commanded target:**

```
  shoulder_pan    =    -2.00 deg
  shoulder_lift   =   -92.20 deg
  elbow_flex      =   +97.70 deg
  wrist_flex      =   +75.20 deg
  wrist_roll      =   +45.00 deg
  gripper         =    +1.20 deg
```

**Achieved (servo-reported after settle):**

```
  shoulder_pan    =    -1.80 deg  ( -0.0315 rad)
  shoulder_lift   =   -92.09 deg  ( -1.6072 rad)
  elbow_flex      =   +97.49 deg  ( +1.7016 rad)
  wrist_flex      =   +74.64 deg  ( +1.3027 rad)
  wrist_roll      =   +44.09 deg  ( +0.7695 rad)
  gripper         =    +0.74 deg  ( +0.0129 rad)
```

**Wrist cam frame:** `wrist_rolled_45.jpg`

**External photo taken by operator:** yes (during 10s window)

### 4. `gripper_open`

Home pose with gripper opened to +30 deg. Tests the gripper axis. Chosen over an all-zeros 'canonical' pose because (a) the real arm cannot physically reach elbow_flex=0 from home (self-collision stalls it at +14.6 deg) and (b) jaws-open vs jaws-closed is a binary visual signal that survives photo compression and rendering differences. +30 deg matches the open-gripper outliers seen in the dark-shadow training data (ep08, ep24, ep32).

**Commanded target:**

```
  shoulder_pan    =    -2.00 deg
  shoulder_lift   =   -92.20 deg
  elbow_flex      =   +97.70 deg
  wrist_flex      =   +75.20 deg
  wrist_roll      =    +0.00 deg
  gripper         =   +30.00 deg
```

**Achieved (servo-reported after settle):**

```
  shoulder_pan    =    -1.80 deg  ( -0.0315 rad)
  shoulder_lift   =   -92.09 deg  ( -1.6072 rad)
  elbow_flex      =   +97.49 deg  ( +1.7016 rad)
  wrist_flex      =   +74.64 deg  ( +1.3027 rad)
  wrist_roll      =    +0.40 deg  ( +0.0069 rad)
  gripper         =   +29.64 deg  ( +0.5174 rad)
```

**Wrist cam frame:** `gripper_open.jpg`

**External photo taken by operator:** yes (during 10s window)

