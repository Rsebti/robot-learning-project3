# toolset/

Standalone analysis & calibration scripts for the SO-101 pick-and-place demos.
All outputs land in `toolset/configs/` (data) and `toolset/figs/` (plots).

| Script | Purpose |
| --- | --- |
| `probe_bowl_position.py` | Connect to the SO-101 follower, read joint angles, run FK -> save bowl xyz in robot base frame. Used at the robot. |
| `trim_still_frames.py` | Strip leading still frames from each demo episode (reset contamination), re-encode videos with ffmpeg, push trimmed dataset back to HF. |
| `map_cube_positions.py` | For each demo, find the grasp moment from gripper signal, FK the fingertip -> approximate cube xy in base frame. Color-coded scatter plot of workspace coverage. |
| `detect_cube_cv.py` | Classical-CV pixel-detect cube in one wrist-camera frame and back-project to base frame via `configs/camera_calibration.yaml`. |
| `calibrate_camera_from_demos.py` | Self-calibration: solve K + T_cam_in_wrist from grasp-time FK ground truth. Writes `configs/camera_calibration.yaml`. |
| `verify_eval2_demos.py` | Read the last frame of every Eval-2 demo, detect the cube color inside the bowl (white-surrounded), compare with task target. Saves annotated frames to `figs/eval_2_debug/`. |

## Outputs

```
toolset/
  configs/
    bowl_positions.yaml          # from probe (per real-robot session)
    cube_positions.csv           # grasp-time cube xy from FK, per episode
    camera_calibration.yaml      # K, T_cam_in_wrist, z_table
    eval2_verification.csv       # episode, target, detected, area, match
  figs/
    cube_position_map.png        # workspace coverage scatter
    calibration_residuals.png    # self-cal back-projection residuals
    eval_2_debug/ep_XXX.png      # 101 annotated last frames
    cube_detect_ep*_f*.png       # single-frame CV detection samples
```

## Dependencies

Created in the `trim` conda env (Python 3.12):

```
pip install datasets huggingface_hub pandas numpy pyarrow matplotlib pyyaml \
            opencv-contrib-python scipy "imageio[ffmpeg]" av
```

`ffmpeg` must be on PATH for `trim_still_frames.py` video re-encoding.
