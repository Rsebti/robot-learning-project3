# Checkpoint deploy protocol (one launcher)

## Friend's Squint ``*.pt`` (what you used to call ``run_best_ckpt``)

Use **`deploy/run_best_ckpt.py`** or **`deploy/run_checkpoint.py`** — same behavior.
You do **not** need to run `eval1_v2/infer_sac_legacy.py` by hand; that file is only
the engine those launchers start for local SAC checkpoints.

## One command

```powershell
conda activate trim
cd C:\Users\hugod\project3

# Inspect any checkpoint first
python deploy/run_best_ckpt.py --inspect --ckpt C:\Users\hugod\e1100lat.pt
python deploy/run_checkpoint.py --inspect --ckpt hudela390/projet3-act-eval1-v1-no-cube

# Run (auto-routes SAC .pt vs ACT HF/folder)
python deploy/run_best_ckpt.py --name ckpt_best_1 --goal_color 0 --bowl_xyz 0.2 0.1 0.0
python deploy/run_checkpoint.py --ckpt deploy/eval1_v2/ckpt.pt
python deploy/run_checkpoint.py --ckpt hudela390/projet3-act-eval1-v1-no-cube --target_color yellow --bowl_x 0.16 --bowl_y 0.32
```

`run_checkpoint.py` is the unified router; `run_best_ckpt.py` is an alias focused on
friend-style ``*.pt``. Older thin wrappers (`run_eval1_ckpt`, `run_hugod_ckpt`,
`run_ckpt_best1`) still work.

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
| `n_conv` / policy resolution | 2-conv → 16×16; 3-conv → 32×32 **or** 42×32 (1792-dim head), inferred from weights |
| Camera | Wrist `base_camera` 640×480 → crop + resize to policy H×W |
| Actions | Delta joint targets (not absolute ACT positions) |

**You run:** `deploy/run_best_ckpt.py` or `deploy/run_checkpoint.py` → **internally** `deploy/eval1_v2/infer_sac_legacy.py` (do not rely on calling the latter directly unless debugging).

**Homing (SAC, eval1_v2):** default `--home_pose auto` — picks **`eval1_sac_universal`** (newer universal physical rest in sim rad) when checkpoint weights look like the friend’s **1792-dim / rgb_emb=75** head; otherwise **`eval1_sac_legacy`**. Override with an explicit `--home_pose` if needed.

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
