# Friend handoff — cube masking + perception scripts

Everything needed to **mask any color cube** on wrist images (no robot required for masking).
Optional scripts for probing / maps / approach are included for reference.

**Repo:** https://github.com/Rsebti/robot-learning-project3  
**Conda env:** `trim` (or equivalent with `opencv-python`, `numpy`, `pyyaml`)

```powershell
cd <clone>\project3
conda activate trim
```

---

## Start here (blob-only masking)

| Step | Command |
|------|---------|
| 1 | Put images in a folder, e.g. `deploy/_snaps/my_frames/` |
| 2 | `python -m toolset.perception.cube_mask_batch --image_dir deploy/_snaps/my_frames --color yellow` |
| 3 | Open `deploy/_snaps/my_frames/cube_masks/gallery.html` |

Learn HSV from your frames: add `--learn_hsv`  
More cube sides: add `--mode full_blob`  
**Do not use** square postprocess if you only want full-blob masks.

Full doc: [docs/CUBE_MASKING.md](../docs/CUBE_MASKING.md)

---

## Example runners (this folder)

```powershell
# Windows
.\friend_handoff\examples\mask_folder_yellow.ps1 -ImageDir deploy\_snaps\my_frames
.\friend_handoff\examples\mask_folder_learn_hsv.ps1 -ImageDir deploy\_snaps\my_frames -Color blue
```

```bash
# Linux / Brev
bash friend_handoff/examples/mask_folder_yellow.sh deploy/_snaps/my_frames yellow
```

---

## All Python modules (`toolset/perception/`)

### Cube masking (friend main path)

| Module | Role |
|--------|------|
| `cube_mask_batch.py` | **Mask all images in a folder** + gallery |
| `blob_mask_utils.py` | Blob-only trim (no square cut) |
| `color_mask.py` | HSV bright-top / full blob |
| `calibrate_colors_from_images.py` | Learn HSV from multi-cube photo |
| `hsv_config.yaml` | Default HSV bands |

### Optional / legacy

| Module | Role |
|--------|------|
| `probe_remask_session.py` | Re-learn HSV on `probe_*` session folders |
| `probe_square_postprocess.py` | Square/wall cut (**not** blob-only) |
| `mask_square_fit.py` | Square fit internals |
| `refined_cube_detect.py` | Bright + square chain |
| `probe_analyze_test_images.py` | Test frames vs probe map |
| `estimate_cube_xy.py` | Single-image 3D (needs calib) |
| `cube_localization.py` | Full 3D localizer |
| `scout_cube.py` / `scout_cube_cli.py` | Scout helpers |
| `observe_cubes.py` | Observation utilities |
| `cv_probe_motion.py` | Motion probe |
| `probe_session_estimate.py` | Session estimate |

### Probing / map (needs robot + sessions)

| Module | Role |
|--------|------|
| `probe_camera_vs_fk.py` | Collect probe trials |
| `probe_space_map.py` | Build kNN map |
| `probe_map_data.py` | Map data helpers |
| `probe_map_guidance.py` | Coverage hints |
| `probe_map_feasibility.py` | Feasibility report |
| `probe_map_visual_report.py` | `probe_map_explainer.html` |
| `approach_plan.py` | Approach geometry |

### Deploy wrappers (repo root `deploy/`)

| Script | Role |
|--------|------|
| `run_probe_only.ps1` | Probe only |
| `run_probe_more_yellow.ps1` | More yellow trials |
| `run_live_map_validate.ps1` | Live map validation |
| `approach_from_probe_map.py` | Approach from map |

---

## RLPD / real2sim (GPU, separate)

| Path | Role |
|------|------|
| `RLPD/README.md` | Isaac replay + teleop annotations |
| `RLPD/data/episodes.json` | Pre-built grasp/cube poses |
| `RLPD/scripts/` | Isaac verify + replay |

ManiSkill training: clone `squint-rlpd` (separate repo).

---

## HTML galleries (examples on Hugo’s machine)

| Report | Path |
|--------|------|
| Probe mask + square | `deploy/_snaps/probe_1779205245/gallery.html` |
| Probe map | `deploy/_snaps/probe_1779205245/probe_map_explainer.html` |
| RLPD real2sim | `RLPD/data/verify/gallery.html` |
| Friend mask run | `<image_dir>/cube_masks/gallery.html` |

---

## Dependencies

```bash
pip install opencv-python numpy pyyaml pandas
# probe / robot scripts also need lerobot, etc.
```
