# RLPD handoff — `projet3_demos_v1`

Everything your teammate needs for **real2sim QA in Isaac** and pointers to **ManiSkill RLPD** training. FK annotation is already done; data lives in `data/`.

Dataset: [Rsebti/projet3_demos_v1](https://huggingface.co/datasets/Rsebti/projet3_demos_v1) — 39 teleop episodes, 30 Hz.

---

## Folder layout

```
RLPD/
  README.md                 ← this file
  data/
    episodes.json           ← Isaac replay + verify (required)
    grasp_cube_annotations.csv
    annotation_report.md
    verify/                 ← created by verify script
  scripts/
    replay_lerobot_in_isaac.py
    verify_real2sim_transfer.py
    isaac_replay_common.py
```

**You do not need** `sim/demo_replay/annotate_grasp_fk.py` on the GPU machine unless re-annotating from scratch.

---

## What is in `episodes.json`

Per episode (39 total):

| Field | Use |
|-------|-----|
| `grasp_frame_local` | Index into trajectory at quasi-static closed grasp (last frame of best hold) |
| `cube_pose_urdf_world` | Spawn cube in Isaac `[x,y,z,qw,qx,qy,qz]` |
| `cube_xyz_user_m` | Human-readable cube center (user frame) |
| `trajectory_observation_state_deg` | Joint replay (motor deg, 6-DOF) |
| `trajectory_action_deg` | Optional; use `--use_action_trajectory` on replay |

Grasp selection (already applied): full rollout → static hold (≥8 frames, arm ≤8°/s, gripper ≤35°) → **most closed** hold → **last frame** of that hold. All 39 episodes: `static_hold_most_closed`.

---

## Isaac — visual replay (CUDA)

From repo root, in an **Isaac Lab** environment:

```powershell
cd C:\Users\hugod\project3
python RLPD/scripts/replay_lerobot_in_isaac.py --episode 0
python RLPD/scripts/replay_lerobot_in_isaac.py --episode 0 --headless
```

Defaults: `data/episodes.json`, task `Isaac-SquintNative-Place-Replay-v0`.

---

## Isaac — transfer QA (CUDA + optional CPU)

**Real wrist images only (CPU):**

```powershell
conda activate trim
python RLPD/scripts/verify_real2sim_transfer.py --skip_sim --episodes 0 1 2 3 4
```

**Full check (CUDA):**

```powershell
python RLPD/scripts/verify_real2sim_transfer.py --episodes 0 1 2 3 4 5
```

Open: `RLPD/data/verify/gallery.html`  
GOOD if `tcp_err_mm_at_grasp` &lt; 30 mm (tune `--tcp_good_mm` if needed).

---

## ManiSkill RLPD (friend’s main job)

Training stack: **`squint-rlpd`** (not this folder). Same dataset, separate ManiSkill annotations under:

`squint-rlpd/outputs/real2sim/projet3_demos_v1/`

```bash
cd squint-rlpd
python train_rlpd.py \
  --env_id SO101PlaceCube-v1 \
  --offline_path Rsebti/projet3_demos_v1 \
  --offline_ratio 0.5 \
  --offline_reward_mode sparse \
  --utd 16
```

See `squint-rlpd/scripts/README_real2sim.md`.

---

## RLPD implementation checklist (squint-rlpd)

- [x] LeRobot offline loader (`rlpd_utils._load_lerobot_dataset`)
- [ ] Symmetric 50/50 demo/online buffer when `--offline_path` set
- [ ] Critic LayerNorm + UTD 10–20
- [ ] Validate transfer gallery before changing reward / UTD together

---

## Legacy paths

Older copies may still exist under `outputs/datasets/projet3_demos_v1_grasp_fk/` and `sim/demo_replay/`. **Use `RLPD/` as the canonical handoff.**

Annotation pipeline (Hugo only, if re-running FK):

```powershell
python -m sim.demo_replay.annotate_grasp_fk
# then copy outputs into RLPD/data/
```
