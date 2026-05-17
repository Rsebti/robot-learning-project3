# Upstream HIL-SERL reference configs — for diffing, NOT for use as-is

These two files are verbatim copies from
`huggingface.co/datasets/aractingi/lerobot-example-config-files`,
fetched 2026-05-17 via `hf download`:

- `env_config_so100.json` (3.2 KB)
- `train_config_hilserl_so100.json` (~12 KB)

## ⚠ They do not load against lerobot 0.5.2

The upstream files predate the `HILSerlProcessorConfig` rename and the
SO-100/101 robot-class consolidation. Concrete mismatches:

| Upstream shape | lerobot 0.5.2 shape |
|---|---|
| `robot.type: "so100_follower_end_effector"` | **class doesn't exist**; use `"so101_follower"` + `processor.inverse_kinematics` block |
| `robot.urdf_path` / `robot.target_frame_name` / `robot.end_effector_bounds` | moved under `processor.inverse_kinematics.*` |
| top-level `"wrapper": { ... }` block | renamed to `"processor": { observation, image_preprocessing, gripper, reset, inverse_kinematics }` |
| top-level `repo_id`, `dataset_root`, `num_episodes`, `episode` | nested under `dataset: { repo_id, num_episodes_to_record, ... }` |

## What they're useful for

- **Hyperparameter reference** — `batch_size: 256`, `temperature_init` defaults,
  `gripper_penalty: -0.02`, etc. carry over.
- **Field-name source-of-truth for image normalization stats** — the upstream
  `policy.dataset_stats` block format is still current.
- **Diffing** — when our `sibling-of-this-dir/env_config_so101.json` breaks,
  cross-reference the upstream to spot a missing nested key.

## What to actually use

See `../env_config_so101.json` and `../train_config_hilserl_so101.json` in
the parent directory. Those match the actual 0.5.2 dataclass shape we
probed via `HILSerlProcessorConfig` and `TrainRLServerPipelineConfig`.
