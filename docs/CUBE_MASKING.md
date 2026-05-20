# Cube masking (friend handoff) — any color, blob only

No probe map / robot / FK required. Masks the cube in still images or frame folders.

**Detection:** bright-top HSV (default — same as our latest good results).  
**Post-process:** keep **largest blob only** — removes pixels **outside** the cube.  
**Not used:** square fit, wall lines, color cut inside blob (`probe_square_postprocess`).

---

## Quick start (one command)

```powershell
cd C:\Users\hugod\project3
conda activate trim

# Put frames in a folder, e.g. deploy\_snaps\my_cube_frames\frame_00.png ...

python -m toolset.perception.cube_mask_batch `
    --image_dir deploy\_snaps\my_cube_frames `
    --color yellow
```

Open: `deploy\_snaps\my_cube_frames\cube_masks\gallery.html`

---

## All scripts (what to run)

| Script | When | Needs robot |
|--------|------|-------------|
| **`cube_mask_batch.py`** | **Main** — mask every image in a folder | No |
| `calibrate_colors_from_images.py` | One photo with **all 6** cube colors visible | No |
| `probe_remask_session.py` | Re-learn HSV from old `probe_*` session layout | No |
| `probe_square_postprocess.py` | Square/wall cut (**skip** for blob-only) | No |
| `probe_camera_vs_fk.py` | Build spatial map (probing) | Yes |
| `probe_analyze_test_images.py` | Test images vs map + square | No* |

\* map optional; uses square by default — friend should use **`cube_mask_batch`** instead.

---

## 1. Mask a folder (recommended)

```powershell
# Built-in HSV (toolset/perception/hsv_config.yaml)
python -m toolset.perception.cube_mask_batch `
    --image_dir PATH\to\frames `
    --color yellow
```

Colors: `red`, `orange`, `yellow`, `green`, `blue`, `violet`.

**Learn HSV from your own frames** (single cube color):

```powershell
python -m toolset.perception.cube_mask_batch `
    --image_dir PATH\to\frames `
    --color orange `
    --learn_hsv
```

Writes `PATH\to\frames\cube_masks\learned_hsv.yaml` and uses it for that run.

**Custom yaml** (from calibration or hand-edited):

```powershell
python -m toolset.perception.cube_mask_batch `
    --image_dir PATH\to\frames `
    --color red `
    --hsv_yaml PATH\to\my_hsv.yaml
```

**More of the cube sides** (looser HSV, still blob-only trim):

```powershell
python -m toolset.perception.cube_mask_batch `
    --image_dir PATH\to\frames `
    --color green `
    --mode full_blob
```

Outputs:

```
<image_dir>/cube_masks/
  gallery.html
  results.csv
  run_meta.json
  masks/*_mask.png
  overlays/*_overlay.png
  learned_hsv.yaml   (if --learn_hsv)
```

---

## 2. Calibrate HSV from a multi-cube photo

When you have **all cubes** in one top-down image:

```powershell
python -m toolset.perception.calibrate_colors_from_images `
    deploy\_snaps\snap_all_cubes.png `
    --out toolset\perception\hsv_config.yaml `
    --viz
```

Then use `--color <name>` in `cube_mask_batch` (or copy ranges into your own yaml).

---

## 3. Re-mask an old probe session (optional)

Same folder layout as `deploy/_snaps/probe_1779205245` (trial_XXX/home/frame_*.png):

```powershell
# Learn + rewrite masks (no square step)
python -m toolset.perception.probe_remask_session `
    --session_dir deploy\_snaps\probe_1779205245 `
    --color yellow

# Only refresh gallery from existing masks (no square column unless you run square postprocess)
```

For **blob-only**, prefer `cube_mask_batch` on exported frames instead of `probe_square_postprocess`.

---

## Do not run (unless you want square cutting)

```powershell
# Cuts inside blob using walls + color — NOT what we want for "mask full cube"
python -m toolset.perception.probe_square_postprocess --session_dir ...
```

---

## Example workflow (new color cube)

```powershell
cd C:\Users\hugod\project3
conda activate trim

mkdir deploy\_snaps\friend_test
# copy 5–10 wrist images of the cube at home into friend_test\

python -m toolset.perception.cube_mask_batch `
    --image_dir deploy\_snaps\friend_test `
    --color blue `
    --learn_hsv

start deploy\_snaps\friend_test\cube_masks\gallery.html
```

Check overlays: green should cover the cube top; no magenta “cut” regions (square pipeline only).

---

## Reference HTML galleries (Hugo’s session)

| Report | Path |
|--------|------|
| Probe masks + square pipeline | `deploy/_snaps/probe_1779205245/gallery.html` |
| Probe map explainer | `deploy/_snaps/probe_1779205245/probe_map_explainer.html` |
| RLPD real2sim grasp | `RLPD/data/verify/gallery.html` |

Friend blob-only runs create a new `cube_masks/gallery.html` under their image folder.
