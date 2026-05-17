# Eval-1 deploy (sim-trained policy handoff)

**Separate pipeline from `deploy/infer_eval2.py`** — this is a SAC-style
sim-trained CNN+MLP policy delivered as a self-contained handoff. Different
control rate (10 Hz vs 30), different image size (16×16 vs 480×640), different
network class, different LeRobot version (0.4.3 vs 0.5.1 used by Eval-2).

## Contents

| File | What |
|---|---|
| `infer_eval1.py`     | the handoff script with hugod-PC constants pre-filled (COM3, cam 1, calibration id `so101_follower`) |
| `eval1_ckpt.pt`      | policy weights (7.4 MB, md5 `0fbbcecfee4e07f3e78c3cfe36560ff2`) |
| `so101_follower.json`| copy of the on-PC SO-101 calibration so the script finds it locally |
| `DEPLOY_EVAL1.md`    | original handoff README from the teammate |

## Setup (fresh env, recommended)

```powershell
conda create -n squint python=3.10 -y
conda activate squint
pip install torch torchvision numpy opencv-python "lerobot[feetech]==0.4.3"
```

## Verify checkpoint

```powershell
(Get-FileHash deploy\eval1\eval1_ckpt.pt -Algorithm MD5).Hash.ToLower()
# Must equal: 0fbbcecfee4e07f3e78c3cfe36560ff2
```

## Run

```powershell
cd C:\Users\hugod\project3\deploy\eval1
python infer_eval1.py --checkpoint eval1_ckpt.pt --action_scale 0.1 --no-viz
```

- One cube (any color, no color conditioning) somewhere reasonable on the table
- Bowl in the workspace
- Press Enter to start an episode; Ctrl+C to quit (script auto-ramps back to rest pose)
- `--action_scale 0.1` for the first runs (safety multiplier). Bump toward 0.15-0.25 once motion looks sane
- `--n_episodes N` to run N back-to-back without waiting for Enter
