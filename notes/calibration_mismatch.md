# Calibration mismatch — cross-machine SO-101 deploys

**Status (this Mac, 2026-05-17):** wrist_roll calibration shifted by +180°
(`homing_offset` -460 → 1588). Backup of the previous calibration at
`~/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower.json.bak-pre-wrist-shift`.

## The pitfall

LeRobot calibration JSONs live at
`~/.cache/huggingface/lerobot/calibration/robots/so_follower/`.
They are **per-machine**. They are **not** shipped with the dataset, and they
are **not** part of the policy. If you deploy a policy on machine B that was
trained from a dataset recorded on machine A, the same numeric joint command
(e.g. `wrist_roll = 0°`) can correspond to a different physical pose on B.

For SO-101 follower this typically manifests as:
- The arm appears 180° flipped during inference (wrist_roll continuous joint).
- A windowed offset on `shoulder_pan` (different operator zero choice).
- The wrist-mounted camera pointing in the wrong direction.

The symptom is silent: the policy loads, the camera opens, motors connect,
inference starts, but the arm goes to the wrong physical pose for every
commanded angle.

## Diagnosis flow (~5 min, no robot motion needed for steps 1-2)

1. **Read training-data home pose** with
   `tools/inspect_dataset_state.py`:

   ```bash
   python tools/inspect_dataset_state.py \
     --repo-id osammotg1/projet3-eval2-v1-tom-hugo \
     --episodes 10 --range-stride 50
   ```

   Records the mean first-frame state across N episodes (degrees, per joint).
   Tom's home was: `pan≈0, lift≈-92, elbow≈+96, wflex≈+76, wroll≈0, grip closed`.

2. **Read live follower position** without moving anything:

   ```python
   from lerobot.robots.so_follower.so_follower import SO101Follower
   from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
   r = SO101Follower(SO101FollowerConfig(
       port='/dev/tty.usbmodem<follower>', id='so101_follower'))
   r.connect()
   obs = r.get_observation()
   print({k.replace('.pos',''): round(v,2)
          for k,v in obs.items() if k.endswith('.pos')})
   r.disconnect()
   ```

3. **Compare per joint.** A diff > ~30° on any single joint is suspicious;
   a diff that is essentially ±180° on `wrist_roll` (a continuous-rotation joint)
   is the canonical cross-machine mounting-orientation bug.

## Fix — software shift, not recalibration

If the offset is a clean multiple of 90°/180° on a continuous joint,
edit `homing_offset` directly. The Feetech STS3215 has 4096 ticks per 360°,
so:

| Shift  | ticks |
|--------|-------|
| ±45°   | ±512  |
| ±90°   | ±1024 |
| ±180°  | ±2048 |

Procedure:
```bash
# 1. Back up
cp ~/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower.json \
   ~/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower.json.bak

# 2. Edit JSON, shift the relevant homing_offset by ±2048 (etc.)

# 3. Reconnect — lerobot will detect the JSON changed and prompt:
#    "Press ENTER to use provided calibration file, or type 'c' to recalibrate"
#    Press ENTER (or echo "" | python <script>) to push the new JSON
#    into the motors. This re-pushes the WHOLE calibration to all motors;
#    expect small (~3°) shifts on other joints too.

# 4. Verify with the read-position snippet above.
```

When NOT to use the JSON-shift shortcut:
- The offset isn't a clean rotational multiple (suggests genuine
  miscalibration, not mounting orientation).
- Multiple joints are wildly off (suggests the whole calibration is bad).
- The user-side `range_min`/`range_max` would now exclude poses the policy
  is known to produce (the JSON shift moves the window).

For those cases, re-run `lerobot-calibrate` with the arm placed in the
**same physical reference pose** the dataset's operator used. The repo
does not yet document Tom & Hugo's reference pose photographically;
when next at the robot, take a photo and add to `notes/`.

## Why this isn't caught by lerobot itself

LeRobot's `is_calibrated` check only verifies that the in-motor calibration
matches the JSON on disk — it cannot know whether that JSON corresponds to
the same physical reference frame the training data used. The dataset on disk
contains joint angles in *some* operator's frame; there is no metadata that
links those angles back to a calibration file. Two operators with
identically-oriented arms but differently-zeroed calibration JSONs produce
numerically-identical datasets that decode to physically-different poses.

## Diagnostic script

`tools/inspect_dataset_state.py` — re-runnable on any HF dataset, prints:
- First-frame state per joint across N episodes (the home pose distribution).
- Per-joint mean / std / min / max across home poses.
- Per-joint min / max / span across the whole strided dataset.

Use it first whenever a cross-machine deploy doesn't behave like the demos.

## History on this machine

- **2026-04-08:** initial calibration of SO-101 follower on this Mac
  (homing_offsets recorded for all 6 joints).
- **2026-04-28:** sanity check passed (5/5 ACT rollouts) — the eval1 ACT
  policy was trained on data this Mac recorded itself, so no cross-machine
  mismatch.
- **2026-05-17:** eval2 SmolVLA deploy attempt revealed 180° wrist flip.
  Root cause: dataset recorded by Tom & Hugo on a different follower with
  different `wrist_roll.homing_offset`. Fix: shifted this Mac's
  `wrist_roll.homing_offset` from -460 to 1588 (+2048 ticks = +180°).
  Backup retained as `*.bak-pre-wrist-shift`.
