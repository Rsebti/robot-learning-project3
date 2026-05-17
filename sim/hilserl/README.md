# HIL-SERL — Eval 2 (single-color v1)

> Scaffold for the HIL-SERL pipeline on SO-101, Option C (single-color, yellow)
> per `notes/hilserl_eval2_plan.md`. Lerobot 0.5.2 editable install
> at `~/Documents/ETH/M4/Robot Learning /Project S101/lerobot/`. Reference:
> `https://huggingface.co/docs/lerobot/hilserl`.

## What lives here

```
sim/hilserl/
├── README.md                                    # ← you are here
├── assets/so101/
│   ├── README.md                                # how to fetch the meshes
│   └── urdf/
│       ├── so_arm101.urdf                       # 13 KB, committed
│       └── assets/*.stl                         # 15 MB, gitignored
└── configs/
    ├── env_config_so101.json                    # record-mode env config
    ├── train_config_hilserl_so101.json          # actor+learner training config
    └── _upstream/                                # stale HF reference (do not load)
        ├── README.md
        ├── env_config_so100.json
        └── train_config_hilserl_so100.json
```

## Run order (Day 1 at the robot → Day 3 eval)

```
        ┌──────────────────────────────────────────────────────────────┐
        │ 0. Sanity check the URDF loads + SmolVLA still imports       │
        │    (5 min, do this before plugging in the arm)               │
        └──────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────▼──────────────────────────────────┐
        │ 1. lerobot-find-joint-limits — record real EE workspace      │
        │    bounds; paste numbers into env_config_so101.json under    │
        │    processor.inverse_kinematics.end_effector_bounds          │
        └───────────────────────────┬──────────────────────────────────┘
                                    │
        ┌───────────────────────────▼──────────────────────────────────┐
        │ 2. Record 15 EE-space teleop demos (one yellow cube + one    │
        │    distractor cube of any other color + fixed bowl)          │
        │      python -m lerobot.rl.gym_manipulator \                   │
        │        --config_path sim/hilserl/configs/env_config_so101.json│
        │    Press 's' on success, 'esc' on fail, leader for actions.  │
        └───────────────────────────┬──────────────────────────────────┘
                                    │
        ┌───────────────────────────▼──────────────────────────────────┐
        │ 3. mode: "replay" sanity check (optional)                    │
        │    Edit env_config: mode → "replay", replay_episode → 0..14  │
        └───────────────────────────┬──────────────────────────────────┘
                                    │
        ┌───────────────────────────▼──────────────────────────────────┐
        │ 4. crop_dataset_roi — pick the 128×128 workspace ROI         │
        │    python -m lerobot.rl.crop_dataset_roi \                    │
        │        --repo-id osammotg1/projet3-hilserl-yellow-v1         │
        │    → writes cropped dataset to ...-cropped suffix.           │
        │    Paste output crop_params_dict into env_config_so101.json  │
        │    AND train_config_hilserl_so101.json.                      │
        └───────────────────────────┬──────────────────────────────────┘
                                    │
        ┌───────────────────────────▼──────────────────────────────────┐
        │ 5. Start the LEARNER on the 5090 (SSH'd in):                 │
        │      python -m lerobot.rl.learner \                           │
        │        --config_path sim/hilserl/configs/train_config_hilserl_so101.json
        │    Start the ACTOR on this Mac (separate terminal):          │
        │      python -m lerobot.rl.actor \                             │
        │        --config_path sim/hilserl/configs/train_config_hilserl_so101.json
        │    Watch wandb 'projet3-hilserl' for intervention-rate decay.│
        └───────────────────────────┬──────────────────────────────────┘
                                    │
        ┌───────────────────────────▼──────────────────────────────────┐
        │ 6. 5-rollout eval (yellow @ trained bowl) + 5-rollout        │
        │    bowl-OOD probe (+3 cm) — per Plan §7 Day-3 requirement.   │
        └──────────────────────────────────────────────────────────────┘
```

## Key config decisions — why each value is what it is

Every load-bearing decision in the JSONs is sourced; this table lets a
teammate (or future-you) audit the choice without re-deriving it.

### `env_config_so101.json` and the env-section of `train_config_hilserl_so101.json`

| Key | Value | Why |
|---|---|---|
| `mode` | `"record"` (env config) / `null` (train config) | HF doc — record collects demos; null is training-mode |
| `device` | `"mps"` (env) / `"cuda"` (train policy) | Mac actor uses MPS; learner on 5090 uses CUDA |
| `env.fps` | `10` | HF doc default — matches HIL-SERL paper's 10 Hz control loop |
| `env.robot.type` | `"so101_follower"` | Verified registered in `lerobot.robots.so_follower.config_so_follower` |
| `env.robot.port` | `/dev/tty.usbmodem5B141129871` | Mac default from `deploy/infer_smolvla.sh`. **Re-verify with `lerobot-find-port` if it moves.** |
| `env.robot.cameras.wrist` | OpenCV index 0, 640×480 @ 30 fps | Matches `deploy/infer_smolvla.sh` defaults |
| `env.teleop.type` | `"so101_leader"` | Locked in plan §5 D7 — we own the leader; no gamepad needed |
| `env.processor.control_mode` | `"leader"` | HF doc — required when teleop is leader-mode |
| `env.processor.observation.add_joint_velocity_to_observation` | `true` | Plan §5 D6 — gives policy motion awareness |
| `env.processor.image_preprocessing.crop_params_dict` | `{}` | Placeholder — `crop_dataset_roi` fills this after Day-1 demo collection |
| `env.processor.image_preprocessing.resize_size` | `[128, 128]` | HF doc default — validated input size for vision policies |
| `env.processor.gripper.gripper_penalty` | `-0.02` | Upstream so100 default — modest penalty to discourage gratuitous open/close |
| `env.processor.reset.control_time_s` | `20.0` | HF default; matches our existing demo episode length |
| `env.processor.inverse_kinematics.urdf_path` | `sim/hilserl/assets/so101/urdf/so_arm101.urdf` | Vendored from MuammerBay/isaac_so_arm101; loads via placo (see assets/README.md) |
| `env.processor.inverse_kinematics.target_frame_name` | `"gripper_frame_link"` | Verified link name in our URDF |
| `env.processor.inverse_kinematics.end_effector_bounds` | placeholder `[-0.30..0.50, -0.30..0.30, 0.02..0.40]` | **REPLACE after Day-1 `lerobot-find-joint-limits` run.** Conservative box for now. |
| `env.processor.inverse_kinematics.end_effector_step_sizes` | `0.02 m / axis` | HF doc default |
| `dataset.repo_id` | `osammotg1/projet3-hilserl-yellow-v1` | Option C — yellow only. Naming convention `projet3-hilserl-<color>-vN` |
| `dataset.num_episodes_to_record` | `15` | HF doc's basic-example default; expand if classifier or RL undertrains |

### Policy section of `train_config_hilserl_so101.json`

| Key | Value | Why |
|---|---|---|
| `policy.type` | `"sac"` | The only RL algorithm shipped in lerobot's HIL-SERL path |
| `policy.temperature_init` | `1e-2` | HF doc "Key hyperparameters" — too high makes interventions ineffective |
| `policy.storage_device` | `"cuda"` | HF doc — keeps weights on GPU; large throughput gain over CPU storage |
| `policy.discount` | `0.99` | Standard SAC default; matches RLPD paper |
| `policy.num_critics` / `num_subsample_critics` | `10 / 2` | REDQ pessimism (paper-level — confirm lerobot's defaults match) |
| `policy.policy_update_freq` | `1` | Updated every learner step |
| `policy.actor_learner_config.policy_parameters_push_frequency` | `1.5` (seconds) | HF doc — 1-2 s gives fresh weights without flooding gRPC |
| `policy.input_features` | wrist 128×128 + state `[12]` (6 joints + 6 joint velocities) | Will adjust at runtime if framework reports different state shape |
| `policy.output_features.action` | shape `[4]` (3 EE deltas + 1 gripper) | EE-space action — 4-dim because gripper is included with `use_gripper: true` |
| `batch_size` | `256` | Upstream default; RLPD paper canonical |
| `steps` | `100000` | Upstream default; tune down once we see when intervention rate hits the floor |
| `wandb.project` | `"projet3-hilserl"` | New project; sibling to `projet3-act` / `projet3-smolvla` |

## Open knobs (deferred to the lab)

1. **`end_effector_bounds`** — fill after `lerobot-find-joint-limits` (~10 min at the robot).
2. **`crop_params_dict`** — fill after `crop_dataset_roi` (~5 min after 1st demo batch).
3. **`fixed_reset_joint_positions`** — `null` for now; set if we want the arm to home to a specific pose between episodes.
4. **`storage_device`** — change to `"mps"` if running learner on the Mac instead of the 5090.
5. **`reward_classifier.pretrained_path`** — `null` for v1 (manual annotation via leader+keyboard per plan D3); fill at v1.5 once we train the classifier.

## Smoke tests to run before the arm moves

```bash
PY=/Users/admin/miniforge3/bin/python3.13

# 1. URDF still loads
$PY -c "from lerobot.model.kinematics import RobotKinematics
import numpy as np
rk = RobotKinematics(
    urdf_path='sim/hilserl/assets/so101/urdf/so_arm101.urdf',
    target_frame_name='gripper_frame_link')
print('joints:', rk.joint_names)
print('FK home:', rk.forward_kinematics(np.zeros(6))[:3, 3])"

# 2. SmolVLA still imports (we upgraded transformers 4.57.6 -> 5.3.0 with [hilserl])
$PY -c "from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
print('SmolVLAPolicy OK')"

# 3. SAC config + HIL-SERL learner imports
$PY -c "from lerobot.rl import learner, actor, gym_manipulator
from lerobot.policies.sac.configuration_sac import SACConfig
print('hilserl + sac OK')"

# 4. Our env config parses against HILSerlRobotEnvConfig
$PY -c "import json
from lerobot.envs.configs import HILSerlRobotEnvConfig
import draccus
cfg_dict = json.load(open('sim/hilserl/configs/env_config_so101.json'))['env']
# draccus will validate on actual load; this just JSON-checks
print('env JSON valid; keys:', list(cfg_dict.keys()))"
```

## Provenance pointers

- HF doc: `huggingface.co/docs/lerobot/hilserl` (fetched + audited 2026-05-17)
- HIL-SERL paper: Luo et al. 2024, arXiv:2410.21845
- Reward classifier reference: HF doc §"Training a Reward Classifier"
- Lerobot source: `~/Documents/ETH/M4/Robot Learning /Project S101/lerobot/` (commit `4eecbad3 chore(dependencies): Bump lerobot to 0.5.2 (#3307)`)
- Plan: `notes/hilserl_eval2_plan.md`
- Plan-eng-review verification log: same file §10
