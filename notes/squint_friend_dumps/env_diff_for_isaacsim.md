# Env diff: from the "no goal-color" checkpoint to `placecube_flattable_woodcube_run1`

Baseline: the previous checkpoint Isaac was loading (state_dim = 12). New
checkpoint: `runs/placecube_flattable_woodcube_run1/ckpt.pt` (state_dim = 18).

## 1. THE BREAKING CHANGE — state vector grew 12 → 18

The policy's state input was extended with a 6-dim **goal-color one-hot**.
If your Isaac side feeds a 12-d state, the model's first linear layer will
fail with a shape mismatch (`state_proj.0.weight` expects 18-in).

| dim range | content                              | unit / domain                 |
|-----------|--------------------------------------|-------------------------------|
| `[0:6]`   | `noisy_qpos` (joint positions)       | rad                           |
| `[6:12]`  | `controller target_qpos`             | rad                           |
| `[12:18]` | **goal_color one-hot** (NEW)         | 0/1, exactly one entry is 1   |

Joint order is the same as before:
`[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`.

Goal-color palette (`envs/place.py:COLOR_PALETTE`):

| idx | name   | RGB              |
|-----|--------|------------------|
| 0   | red    | (1.00, 0.00, 0.00) |
| 1   | blue   | (0.00, 0.00, 1.00) |
| 2   | green  | (0.00, 1.00, 0.00) |
| 3   | yellow | (1.00, 1.00, 0.00) |
| 4   | purple | (0.60, 0.00, 0.80) |
| 5   | orange | (1.00, 0.50, 0.00) |

On Isaac: pick the color the cube was rendered as, set that one-hot bit, and
concatenate `[qpos, target_qpos, one_hot]` (length 18) before feeding the
policy.

## 2. Cube physical params changed (matters for replay)

|                  | Old (b8ada9 run)      | New (flattable_woodcube_run1) |
|------------------|-----------------------|-------------------------------|
| Side length      | 22 – 28 mm (rand.)    | 18 – 22 mm (rand.)            |
| Density          | 200 kg/m³ (fixed)     | 700 kg/m³ (fixed, beech wood) |
| Mass @ mid size  | ~3.1 g                | ~5.6 g                        |
| Mass range       | 2.1 – 4.4 g           | 4.1 – 7.5 g                   |
| Friction range   | 0.1 – 0.5 (unchanged) | 0.1 – 0.5                     |

Source: `envs/place.py:PlaceRandomizationConfig`.

## 3. Distractor cube added

A second cube spawns face-to-face with the target, palette-colored to a
different color than the goal (never the same). Same physics, same size as
the target. Initial position relative to target:
- Direction: uniform random in `[0, 2π)`
- Center-to-center distance: `2 * half_size + uniform(0, 0.005)` m

On Isaac you can either:
- Mirror this (spawn a second cube with a different palette color and
  matched physics), or
- Train/evaluate the "single-cube" variant by passing
  `--no-include_distractor` on the Squint side (we added this flag in this
  iteration; the checkpoint was **trained with the distractor present**, so
  the policy expects it visually, but the goal-color conditioning is what
  tells it which one to grasp).

## 4. Table is now a flat matte box

- The decorative GLB visual was replaced with a uniform matte box visual
  at `#B8ADA9` (`SCENE_NEUTRAL_RGB`). Collision is unchanged (same
  `2.418 × 1.209 × 0.92 m` box).
- See `FlatTableSceneBuilder` in `envs/place.py`.

If your Isaac scene previously matched the decorative wood, swap to a
solid matte material (#B8ADA9) on a flat slab and the visual obs will line
up much better.

## 5. obs_mode default flipped (cosmetic, NOT a state change)

The default `obs_mode` in `train_squint.py` and `deploy.py` is now `"rgb"`
(was `"rgb+segmentation"`). Greenscreen / overlay is still off
(`apply_overlay=False`, unchanged). This only affects whether the
segmentation channel is rendered alongside RGB — it does **not** enter the
policy in either case. No Isaac change needed unless you were reading the
seg channel for something.

## 6. Sanity checks vs. this handoff

- `replay_trajs.json` contains 3 full episodes from `seed=0,1,2` with
  per-step `state` (18-d), `action` (6-d), and the initial cube/bin/robot
  poses. Use them to drive Isaac in open-loop replay and compare end-state.
- `physics_params.txt` shows the SAPIEN solver settings + cube
  density/friction/mass at runtime — match those on the PhysX side first.
- `action_sensitivity.txt` lets you confirm bit-identical policy output:
  same RGB + same 18-d state should produce the same 6-d action on Isaac.
- `settle_behavior.txt` and `action_replay.txt` are ground-truth
  qpos trajectories for the no-action and per-joint +0.5 cases.
