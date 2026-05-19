# Checkpoint deploy protocol (one launcher)

## One command

```powershell
conda activate trim
cd C:\Users\hugod\project3

# Inspect any checkpoint first
python deploy/run_checkpoint.py --inspect --ckpt C:\Users\hugod\eval1_ckpt.pt
python deploy/run_checkpoint.py --inspect --ckpt hudela390/projet3-act-eval1-v1-no-cube

# Run (auto-routes SAC .pt vs ACT HF/folder)
python deploy/run_checkpoint.py --name eval1_ckpt --goal_color 3 --bowl_xyz 0.16 0.32 0
python deploy/run_checkpoint.py --ckpt deploy/eval1_v2/ckpt.pt
python deploy/run_checkpoint.py --ckpt hudela390/projet3-act-eval1-v1-no-cube --target_color yellow --bowl_x 0.16 --bowl_y 0.32
```

`run_checkpoint.py` replaces the old thin wrappers (`run_eval1_ckpt`, `run_hugod_ckpt`, `run_ckpt_best1`, `run_act_8d`) — those still work but all boil down to the same infer scripts.

## Where files live

| Location | Typical contents |
|----------|------------------|
| `C:\Users\hugod\*.pt` | Friend handoffs (`eval1_ckpt.pt`, `ckpt_best_1.pt`, …) |
| `deploy/eval1_v2/ckpt.pt` | Default SAC for `run_eval1_ckpt` |
| HuggingFace repo id | LeRobot ACT (`hudela390/...`) |
| Local folder | ACT export with `config.json` + weights |

`--name eval1_ckpt` searches: `~/eval1_ckpt.pt`, `~/checkpoints/`, `deploy/eval1_v2/`, `deploy/eval1/`, project root.

## Two backends

### A) SAC Squint `.pt` (encoder + actor)

**Auto-detected from weights:**

| Field | How |
|-------|-----|
| `n_state` | 12 = eval1 no colour; 18 = colour; 21 = colour + bowl xyz |
| `n_conv` / `image_size` | 2 / 16 (original) or 3 / 32 (`ckpt_best_*`) |
| Camera | Wrist `base_camera` 640×480 → center crop → 128 → CNN input |
| Actions | Delta joint targets (not absolute ACT positions) |

**Runs:** `deploy/eval1_v2/infer_sac_legacy.py` (handles both CNN variants).

**SAC flags** (pass after `--` or as trailing args):

`--checkpoint`, `--goal_color`, `--bowl_xyz`, `--action_scale`, `--episode_steps`, `--control_hz`, `--home` / `--no-home`, `--home_pose`, `--viz` / `--no-viz`, `--grip_force_steps`, `--park_pose`, …

### B) LeRobot ACT (folder or HF)

**Auto-detected from `config.json`:**

| Field | How |
|-------|-----|
| `env_state_dim` | 8 → nocube (colour + bowl xy); 10 → + cube xy |
| `image_keys` | Usually `wrist` |
| Actions | Absolute joint targets via LeRobot processors |

**Runs:**

- 8D → `deploy/infer_eval1_act_nocube.py`
- 10D → `deploy/infer_eval1_act.py`

**ACT flags:** `--policy_path`, `--target_color`, `--bowl_x`, `--bowl_y`, `--follower_port`, `--camera_index`, `--home` / `--no-home`, `--episode_time_s`, …

## Optional manifest (for friends)

When auto-detect is not enough, add a JSON sidecar:

**`eval1_ckpt.manifest.json`** next to **`eval1_ckpt.pt`**:

```json
{
  "backend": "sac",
  "goal_color": 3,
  "bowl_xyz": [0.16, 0.32, 0.0],
  "home_pose": "eval1_sac_legacy",
  "control_hz": 30,
  "camera_key": "base_camera",
  "notes": "3-conv CNN, n_state=18"
}
```

**ACT folder:** `deploy_manifest.json` inside the policy directory.

Run with manifest overrides:

```powershell
python deploy/run_checkpoint.py --ckpt C:\Users\hugod\eval1_ckpt.pt --trust-manifest
```

## Inspect-only tool

```powershell
python deploy/checkpoint_probe.py path\to\ckpt.pt
python deploy/checkpoint_probe.py path\to\ckpt.pt --json
```

## Scripted place (not a learned checkpoint)

`python -m toolset.kinematics.ik_relative` — separate from `run_checkpoint.py`.

## Quick routing table

| You have | Command |
|----------|---------|
| `~/eval1_ckpt.pt` | `python deploy/run_checkpoint.py --name eval1_ckpt` |
| `~/ckpt_best_1.pt` | `python deploy/run_checkpoint.py --name ckpt_best_1` |
| HF ACT 8D | `python deploy/run_checkpoint.py --ckpt hudela390/projet3-act-eval1-v1-no-cube --target_color yellow --bowl_x 0.16 --bowl_y 0.32` |
| Default v2 SAC | `python deploy/run_checkpoint.py` (uses `deploy/eval1_v2/ckpt.pt` if present) |
