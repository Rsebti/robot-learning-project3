# Isaac Lab + isaac_so_arm101 — install on the fixed PC

> Goal: get a working Isaac Sim 5.1 + Isaac Lab 2.3 + `isaac_so_arm101` install
> on the user's RTX 5070 Windows machine, in ≤ 1 day. Pivot to Brev (H100 Linux)
> if Blackwell sm_120 issues block us.

---

## Prerequisites

| Item | Required |
|---|---|
| OS | Windows 11 (22H2 or 24H2) |
| GPU | NVIDIA RTX (RTX 5070 here, sm_120 Blackwell) |
| Driver | NVIDIA ≥ **576** |
| Disk | ~100 GB free on `C:` |
| Python | uv will install Python 3.11 in a venv (no manual conda env needed) |
| Git | Already installed (GitHub Desktop bundles git) |
| PowerShell | Built-in on Windows 11 |

Check these now:
```powershell
nvidia-smi                 # driver version in top-right corner of the table
Get-PSDrive C              # free space
git --version              # any version OK
```

---

## Install path: `uv sync` (recommended)

This is the path `isaac_so_arm101` README uses. It pulls Isaac Sim + Lab + all
dependencies as pip wheels into a managed venv. No need to build Isaac Sim
from source.

### 1. Install `uv`

`uv` is an ultra-fast Python package manager (drop-in for pip + venv). Install
in PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close & reopen PowerShell, verify:
```powershell
uv --version
```

### 2. Choose an install location OUTSIDE the project repo

`isaac_so_arm101` and Isaac Lab/Sim must NOT be committed to
`robot-learning-project3` (they are local-only per CLAUDE.md). Suggested:

```powershell
mkdir C:\Users\user\Desktop\MA2\isaac
cd C:\Users\user\Desktop\MA2\isaac
```

### 3. Clone `isaac_so_arm101`

```powershell
git clone https://github.com/MuammerBay/isaac_so_arm101.git
cd isaac_so_arm101
```

### 4. `uv sync` — the heavy step

```powershell
uv sync
```

What it does:
- Creates `.venv\` with Python 3.11
- Pulls Isaac Sim 5.1.0 wheels (~15-25 GB) from NVIDIA pip index
- Pulls Isaac Lab 2.3.0 (~500 MB)
- Pulls PyTorch + RL deps (rsl_rl, gymnasium, ...)
- Total disk cost: ~30-50 GB

Expect 30-90 min depending on the network. The download is resumable; if it
fails, just run `uv sync` again.

### 5. PyTorch sm_120 fix (probably needed)

Isaac Lab 2.3 ships PyTorch CU128 wheels by default, which DO support sm_120.
If the smoke test in step 6 errors with `no kernel image is available for
execution on the device`, force-reinstall the nightly:

```powershell
uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 6. Smoke test #1 — `zero_agent`

```powershell
uv run zero_agent --task SO-ARM100-Reach-Play-v0
```

Expected:
- An Omniverse 3D window opens
- The SO-101 sits on a table
- The arm stays still (zero actions)
- Console logs at 30-60 fps

Quit with `Ctrl+C` in the terminal, or close the window.

### 7. Smoke test #2 — short PPO training

```powershell
uv run train --task SO-ARM100-Reach-v0 --headless
```

Wait 5-10 min. You should see:
- `total_reward` increasing
- ~50-200 training steps/s
- No NaN

If reward goes up steadily → install validated, move on to Eval 2 env design.

---

## Common errors and fixes

### `CUDA error: no kernel image is available for execution on the device`
PyTorch doesn't have sm_120 kernels. Run:
```powershell
uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

### `RuntimeError: TiledCamera hangs` or window freezes
Known Blackwell bug with `omni.replicator` tiled rendering. Workaround:
use the regular `Camera` sensor in the env (not `TiledCamera`). For the
smoke tests this shouldn't matter; relevant when designing custom envs.

### Window opens then closes immediately, GPU not detected
Usually driver issue. Update to NVIDIA driver ≥ 576, reboot.

### `PhysX falls back to CPU` (silent, training is slow)
- Force CUDA: add `--device cuda:0` to the train command
- Check `nvidia-smi` during training: GPU utilization should be > 50%
- If still on CPU, this is a known Blackwell PhysX issue → pivot to Brev

### `uv sync` fails with `network timeout` or `connection reset`
Re-run; it's resumable. NVIDIA pip index is sometimes slow.

### `DLL load failed while importing _C` (PyTorch import)
Visual C++ Redistributable missing. Install from
https://aka.ms/vs/17/release/vc_redist.x64.exe

---

## Pivot to Brev (H100 Linux) — when

If by end of day local install can't complete the two smoke tests, pivot:

1. Redeem Brev coupon at https://brev.dev (the team captain has the coupon)
2. Create an instance:
   - GPU: 1× H100
   - Image: Ubuntu 22.04 + CUDA 12.4+
   - Disk: 200 GB
3. SSH in (Brev provides one-line `brev shell` from local CLI)
4. Repeat steps 1-7 above on the Linux box (same `uv` flow, just `bash` instead of `pwsh`)
5. Linux Isaac install is much smoother than Windows — expect smoke tests to pass

**Cost reminder:** ~$2-3/h for H100. Stop the instance when not training.

---

## After install — orient yourself in the repo

Once smoke tests pass, spend ~1h reading:

```powershell
cd C:\Users\user\Desktop\MA2\isaac\isaac_so_arm101
ls source\isaac_so_arm101\tasks    # task definitions
```

Read the SO-ARM100 Reach task end-to-end. Map:
- where the scene is built (USD assets, robot, table)
- where observations are defined
- where rewards are computed
- where the env is registered in gym

This is the template you'll extend for `sim/eval2/pick_in_clutter_env.py`
in our project repo.

---

## Time-tracking template

Date: ___________

- [ ] uv installed (5 min)
- [ ] isaac_so_arm101 cloned (2 min)
- [ ] `uv sync` complete (___ min)
- [ ] zero_agent smoke test passed
- [ ] Reach training smoke test passed
- [ ] Decision: stay local / pivot Brev

Notes / errors encountered:
