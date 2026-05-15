# Dual-boot Ubuntu Handoff — Continue Setup

> **For the Claude Code session that picks up after the user dual-boots to Ubuntu.**
> Read this whole file first, then read `CLAUDE.md` and `notes/isaac_lab_setup.md`
> (it's the Isaac Sim install procedure from the Windows side — most steps
> translate 1:1 to Ubuntu, with simpler venv handling).

---

## Where the user is

The user is mid-way through a Windows-to-Ubuntu dual-boot installation on
the fixed PC (RTX 5070, MSI B760 GAMING PLUS WIFI, i5-14600KF, 32 GB RAM,
1.86 TB SSD).

**Hardware state at handoff:**
- Windows 11 Pro still installed on `(C:)`, shrunk to ~1362 GB.
- **500 GB unallocated** at the end of Disk 0, between C: and the Recovery
  partition. Waiting for Ubuntu installer.
- 100 MB EFI System Partition (shared, will be reused by Ubuntu).
- 805 MB Recovery Partition (Windows).
- BitLocker is **OFF** on C:.
- Fast Startup is **disabled**.
- Secure Boot **ON** in BIOS — may need to be disabled temporarily if the
  Ubuntu USB shows "Security Violation" at boot (SBAT revocation on the
  Ubuntu 22.04.5 shim, predicted by Rufus).
- Bootable USB stick (16 GB, labeled `Ubuntu 22.04.5 LTS amd64`) was
  flashed with Rufus from `ubuntu-22.04.5-desktop-amd64.iso` in
  GPT / UEFI (non-CSM) / FAT32 mode.

**Project state at handoff** (this repo, branch `main`):
- Last commit: `aced716 Align Squint-native env with Squint canonical + env-local bowl_xyz`.
- All Squint canonical reward shaping is in (P1-P4 reverted).
- Spawn boxes aligned with Squint canonical: cube AND bowl in
  `(0.30, 0) ± 0.10`, non-overlapping.
- `bowl_xyz` observation fix: now in env-local frame (= robot frame),
  was previously polluted by Isaac Lab's world-frame env grid offset.
- Friction values: Squint fedecomi04 exact (cube/distractor 0.325, table 0.225,
  gripper 2.0+tpr=0.1, bowl 0.5).
- NoDR-Play env variant added for clean policy visualization.
- View script `view_deploy_multiseed.py` breaks episode on success.

**Next planned action** (after Ubuntu is set up):
- Launch SAC+C51 training from ckpt 8 (Squint native, 18-d) with
  zero-padding for the 3 new `bowl_xyz` state dimensions.
- 150k steps, 64 envs, warmstart auto-overrides active.

---

## Phase 4 — Install Ubuntu (user is here when this file is read)

### A. Boot from the live USB

If the user hasn't booted yet, the path is:
1. Shut down Windows properly.
2. Power on, mash `Delete` to enter MSI BIOS.
3. Boot menu / Boot Order → select `UEFI: <USB stick name>`.
4. F10 → Save & Exit.
5. GRUB menu → "Try or Install Ubuntu" → Enter.

If "Security Violation" appears at boot → BIOS → Settings → Advanced →
Windows OS Configuration → Secure Boot → Disabled → save → retry boot.

### B. Run the installer

From the Ubuntu live desktop, double-click **"Install Ubuntu 22.04.5 LTS"**.

1. **Language**: English (or French — user's choice).
2. **Keyboard**: French (AZERTY) — user has a French keyboard.
3. **Updates and other software**:
   - Choose **"Normal installation"**.
   - **CHECK** "Download updates while installing Ubuntu".
   - **CHECK** "Install third-party software for graphics and Wi-Fi hardware
     and additional media formats" — critical for the NVIDIA driver.
4. **Installation type — CRITICAL STEP**:
   - **DO NOT choose** "Erase disk and install Ubuntu". That destroys Windows.
   - **DO NOT choose** "Install Ubuntu alongside Windows Boot Manager"
     unless the user wants to gamble — we already pre-shrunk the partition
     and want manual control.
   - **CHOOSE "Something else"** for full manual partitioning.

### C. Manual partitioning ("Something else" screen)

The screen shows all partitions on `/dev/nvme0n1` (or `/dev/sda` depending
on detection). Identify them by size:
- `/dev/nvme0n1p1` — ~100 MB — EFI System Partition (existing, **leave alone**)
- `/dev/nvme0n1p2` (or similar) — ~16-128 MB — MSR partition (Windows, **leave alone**)
- `/dev/nvme0n1p3` — ~1362 GB NTFS — Windows C: drive (**leave alone**)
- **`free space` — ~500 GB — this is what we partition for Ubuntu**
- `/dev/nvme0n1pN` — ~805 MB — Recovery (**leave alone**)

Click the `free space` row, then click **"+"** to add partitions:

**Partition 1 — Swap (32 GB)**
- Size: `32768` MB
- Type for the new partition: **Primary**
- Location: **Beginning of this space**
- Use as: **swap area**
- Mount point: (leave empty — swap doesn't mount)
- Click OK.

**Partition 2 — Root `/` (the rest, ~468 GB)**
- Click the remaining `free space` → "+"
- Size: whatever is shown (the rest, ~468000 MB)
- Type: **Primary**
- Location: **Beginning of this space**
- Use as: **Ext4 journaling file system**
- Mount point: **`/`**
- Click OK.

**No separate `/home`** — for a single-user dev machine, one partition is
simpler. Keeps things flexible.

**Device for boot loader installation** (dropdown at the bottom):
- Select **`/dev/nvme0n1`** (the whole disk, NOT a partition).
- Ubuntu's GRUB will install in the existing EFI System Partition and
  chain-load both Ubuntu and the Windows Boot Manager.

Click **"Install Now"** → confirm the partition operations.

### D. Continue the wizard

5. **Where are you?**: Zurich (or wherever the user is).
6. **Who are you?**:
   - Your name: anything
   - Computer's name: `rayane-isaac` or similar (avoid spaces, avoid `_`)
   - Username: `rayane` (lowercase, no spaces)
   - Password: pick a strong one — typed many times for `sudo`
   - **Require my password to log in** (default, recommended)
7. **Install** → ~20 min.
8. When prompted "Installation Complete. Restart now" → click Restart.
9. The screen says **"Please remove the installation medium, then press ENTER"** → **unplug the USB stick now**, press Enter.

### E. First boot

PC reboots → GRUB menu appears:
```
*Ubuntu
 Advanced options for Ubuntu
 Memory test
 Windows Boot Manager (on /dev/nvme0n1p1)
```

Default = Ubuntu, 5 sec timeout, Enter to boot now.

Log in with the user/password. Welcome to Ubuntu 22.04.5 LTS.

---

## Phase 5 — Post-install setup (~1h)

### A. Initial system update

Open a terminal (`Ctrl+Alt+T`):

```bash
sudo apt update && sudo apt upgrade -y
sudo reboot
```

### B. NVIDIA driver

After reboot, check if the driver is already installed (the "third-party
software" checkbox during install should have done it):

```bash
nvidia-smi
```

If the RTX 5070 is listed with driver ≥ 555 → ✅ skip to section C.

If `command not found` or older driver:

```bash
sudo ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
sudo reboot
nvidia-smi
```

For the RTX 5070 (Blackwell, sm_120), the driver must be 555+. If
`ubuntu-drivers` only offers older versions, add the official PPA:

```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install nvidia-driver-555      # or whatever the latest stable is
sudo reboot
nvidia-smi
```

### C. Dev essentials

```bash
sudo apt install -y git curl wget build-essential ca-certificates \
    libgl1 libxext6 libxrender-dev libsm6 libxrandr2 libxinerama1 libxcursor1 \
    libxi6 libxxf86vm1 libegl1
```

### D. Node.js + Claude Code

Claude Code runs on Node 18+. Install Node via `nvm`:

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
source ~/.bashrc
nvm install --lts
node --version    # should print v20.x.x or similar
```

Install Claude Code:

```bash
npm install -g @anthropic-ai/claude-code
claude --version
```

First-time login:
```bash
claude
```
Follow the OAuth prompt to authenticate.

### E. Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3-installer.sh
bash ~/miniconda3-installer.sh
# Accept license, default install path (~/miniconda3), say "yes" to conda init
source ~/.bashrc
conda --version
```

### F. UV (used by isaac_so_arm101)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv --version
```

### G. Recreate the project workspace

```bash
mkdir -p ~/Desktop/MA2
cd ~/Desktop/MA2

# This repo (must be pushed before booting Ubuntu — branch main has commit
# aced716 as the latest at handoff time).
git clone https://github.com/Rsebti/robot-learning-project3.git
cd robot-learning-project3

# External Squint repo (reference — bowl mesh, place.py reference reward,
# etc.). Path used by sim/eval2/envs/squint_native/squint_scene.py.
git clone https://github.com/fedecomi04/squint.git
```

**Important**: the bowl USD asset path in `sim/eval2/envs/squint_native/squint_scene.py`
expects `squint/meshes/bowl.usd`. If the Squint repo doesn't ship the .usd
directly, regenerate it:

```bash
cd ~/Desktop/MA2/robot-learning-project3
# (After isaac_so_arm101 .venv is set up — see Phase 6)
.venv/bin/python sim/eval2/scripts/mesh_bowl_from_ply.py
.venv/bin/python sim/eval2/scripts/convert_bowl_to_usd.py
```

### H. HuggingFace token

```bash
mkdir -p ~/Desktop/MA2
# Copy the tokens.txt from the USB / cloud backup the user made before reboot.
# Example, if user has the file in Downloads:
cp ~/Downloads/tokens.txt ~/Desktop/MA2/tokens.txt
chmod 600 ~/Desktop/MA2/tokens.txt    # readable only by user
```

Then authenticate `huggingface-cli`:
```bash
pip install --upgrade huggingface_hub
HF_TOKEN=$(cat ~/Desktop/MA2/tokens.txt | tr -d '\n')
huggingface-cli login --token "$HF_TOKEN"
```

---

## Phase 6 — Isaac Sim 5.1 + Isaac Lab 2.3 + isaac_so_arm101 (~30 min)

### A. Clone isaac_so_arm101

```bash
cd ~/Desktop/MA2
mkdir -p isaac && cd isaac
git clone https://github.com/MuammerBay/isaac_so_arm101.git
cd isaac_so_arm101
```

### B. Set up the venv with uv

```bash
uv sync
```

This installs Isaac Sim 5.1, Isaac Lab 2.3, PyTorch, rsl_rl, and all
project dependencies into `./.venv/`. Takes 10-20 min depending on
bandwidth (~10 GB of wheels). uv is much faster than pip.

If there's a torch/CUDA mismatch warning at the end (Blackwell sm_120
not matching the default cu118 wheel):
```bash
uv pip install --force-reinstall \
    --index-url https://download.pytorch.org/whl/nightly/cu128 \
    torch torchvision
```

### C. Apply the rsl_rl LR floor patch

This is a project-specific patch (LR floor 1e-5 → 1e-4) that prevents
optimizer collapse cascade. Documented in `notes/rsl_rl_patches.md`.

```bash
cd ~/Desktop/MA2/robot-learning-project3
~/Desktop/MA2/isaac/isaac_so_arm101/.venv/bin/python sim/eval2/scripts/patch_rsl_rl.py
```

The script is idempotent.

### D. Smoke test

Launch the multi-seed viewer with ckpt 8 (which the user has at
`~/Downloads/ckpt (8).pt` — bring it over from the Windows side via the
USB key or download from HF):

```bash
cd ~/Desktop/MA2/isaac/isaac_so_arm101
PYTHONPATH=~/Desktop/MA2/robot-learning-project3 \
.venv/bin/python \
    ~/Desktop/MA2/robot-learning-project3/sim/eval2/scripts/view_deploy_multiseed.py \
    --ckpt "~/Downloads/ckpt (8).pt" \
    --task "Isaac-SquintNative-Place-NoDR-Play-v0" \
    --n_episodes 5 --n_steps 300
```

Should launch the Isaac Sim GUI window, you should see 5 episodes of the
SO-101 attempting to grasp and place. Same behavior as last seen on
Windows (grasps work, places fail because ckpt 8 doesn't know bowl_xyz).

---

## Phase 7 — Resume training

The plan we agreed on before the dual-boot:

**Warmstart from ckpt 8 (Squint native, 18-d state) with zero-padding for
the 3 new bowl_xyz state dimensions. 150k steps, 64 envs.**

The training script `train_squint_isaac.py` auto-detects `--warmstart_from`
and applies:
- `learning_starts=0` (no random rollout, use warm policy immediately)
- `policy_lr=1e-5`, `q_lr=3e-5`, `alpha_lr=3e-5` (slow fine-tune)
- `target_entropy_scale=2.0` (tolerate near-deterministic warm actor)
- `min_buffer_for_update=5000` (diverse buffer before any update)
- `tau=0.005` (slow target net update)
- `freeze_actor_steps=20000` (critic re-aligns first)
- `bc_loss_coef_start=1.0` (BC anchor to teacher actor, decays)

Launch command:

```bash
cd ~/Desktop/MA2/isaac/isaac_so_arm101
PYTHONPATH=~/Desktop/MA2/robot-learning-project3 \
PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
.venv/bin/python \
    ~/Desktop/MA2/robot-learning-project3/sim/eval2/scripts/train_squint_isaac.py \
    --num_envs 64 --total_steps 150000 --headless \
    --warmstart_from "~/Downloads/ckpt (8).pt" \
    --checkpoint_every 20000
```

ETA on RTX 5070 / Ubuntu: hopefully **2-4× faster than Windows** (cleaner
CUDA stack, no DWM compositor, no Windows Update reboots). Should land at
2-5h for 150k steps.

Expected trajectory:
| Steps | Phase | Expected behavior |
|---|---|---|
| 0 - 20k | Actor frozen | Critic re-aligns. Lift rate ≈ ckpt 8 baseline (decent, the user already saw this on Windows). Place rate near 0 (no bowl_xyz signal used). |
| 20k - 50k | Actor unfreeze, BC strong | Place rate starts climbing as bowl_xyz becomes useful. Lift may briefly dip then recover. |
| 50k - 100k | BC decays, full SAC | Place rate stabilizes. Target ≥ 40% success. |
| 100k - 150k | Fine-tune | Plateau or marginal improvement. |

Best checkpoints auto-saved: `_best_lift.pt`, `_best_success.pt`.

---

## Reference docs to read

In order of importance:
1. `CLAUDE.md` — project-wide context, warnings, working style preferences.
2. `notes/isaac_lab_setup.md` — Isaac Sim install procedure (Windows version,
   but the architecture and pitfalls transfer). Most important on Ubuntu:
   the Blackwell sm_120 PyTorch nightly note.
3. `notes/rsl_rl_patches.md` — what the LR floor patch does and why.
4. `notes/sanity_results.md` — the BC sanity check that's done, useful
   context for understanding the user's project arc.
5. `notes/project3_rl_final_details.md` — TA spec for the 3 evaluations.
   Eval 2 is the focus.

---

## Common gotchas on Ubuntu

- **`nvidia-smi` works but Isaac Sim fails to find GPU**: the kit Python
  needs `LD_LIBRARY_PATH` to include CUDA. uv-managed venvs handle this
  automatically; if you launch from outside the venv, set:
  ```bash
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12/lib64
  ```
- **Isaac Sim hangs on first launch**: it's downloading USD assets. Let it
  run. Subsequent launches are fast (cached in `~/.local/share/ov/`).
- **`uv sync` fails with "Lock file requires ..."`: pass `--upgrade` to
  re-resolve.
- **Audio / no sound**: irrelevant for headless ML, skip.
- **NumLock off at login**: GNOME setting, fix in Settings → Keyboard.

---

## What's been intentionally NOT done

- **Did not back up** the existing Windows-side `squint/runs/*.pt`
  checkpoints. The user accepted that the previous `1778753854` family
  of ckpts would not be reused (trained with wrong bowl_xyz frame and
  pre-canonical reward).
- **Did not back up** Anaconda env from Windows. Will recreate fresh on
  Ubuntu.
- **Did not push the Squint reference repo to our private GitHub**. The
  user re-clones `fedecomi04/squint` from upstream on Ubuntu.

---

## When in doubt, ask the user before

- Pushing to `origin/main` (auto-mode blocks this — user pushes from
  GitHub Desktop or a terminal).
- Deleting partitions / files in `/home`.
- Running anything that takes > 1h GPU time without confirming.
- Modifying `~/Desktop/MA2/isaac/isaac_so_arm101/` outside of the
  rsl_rl patch script (per `CLAUDE.md`, that repo is meant to stay clean).
