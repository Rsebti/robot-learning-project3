# Laptop setup — install everything from scratch

Goal: get the portable laptop ready to record demos with the SO-101 tomorrow.

## 1. Base tools (15 min)

### a) Git + GitHub Desktop
- Download **GitHub Desktop** : https://desktop.github.com/
- Install → log in with the same GitHub account (`Rsebti`)
- (GitHub Desktop bundles `git` so no separate install needed)

### b) Anaconda
- Download **Miniconda** (lighter than full Anaconda) : https://www.anaconda.com/download
- Install with default options
- **Important** : tick "Add to PATH" (or add it manually after) so `conda` works in any terminal

### c) (Optional) Visual Studio Code
- https://code.visualstudio.com/ — useful to edit files quickly

## 2. Clone the project (5 min)

In GitHub Desktop:
- `File → Clone repository → URL` :
  `https://github.com/Rsebti/robot-learning-project3`
- Local path: `C:\Users\<you>\Desktop\MA2\robot-learning-project3`

Or in PowerShell:
```powershell
mkdir C:\Users\<you>\Desktop\MA2
cd C:\Users\<you>\Desktop\MA2
git clone https://github.com/Rsebti/robot-learning-project3.git
```

## 3. Create the `lerobot` conda env (10 min)

Open **Anaconda Prompt** (start menu) :

```bash
conda create -n lerobot python=3.12 -y
conda activate lerobot
```

Install LeRobot + extras (non-editable, simpler than the fixed PC) :

```bash
pip install "lerobot[feetech,dataset]==0.5.2"
```

Quick check :
```bash
lerobot-record --help
```
→ should print the help text without error.

## 4. HuggingFace login (2 min)

Get the token from `C:\Users\user\Desktop\MA2\tokens.txt` on the **fixed PC**, copy it to a USB key or just retype it on the laptop (do NOT mail it / Slack it).

```bash
hf auth login
```
Paste the token when prompted. Verify :
```bash
hf auth whoami
```
→ must return `Rsebti`.

## 5. Pull the latest project files (1 min)

In GitHub Desktop : `Repository → Pull` (or `Ctrl+Shift+P`).
Equivalent CLI :
```bash
cd C:\Users\<you>\Desktop\MA2\robot-learning-project3
git pull
```

You now have:
- `teleop/record_demos.md` — recording command
- `train/train_bc.md` — training command (won't run on laptop without GPU, that's fine)
- `deploy/deploy_policy.md` — deploy command
- `notes/demo_protocol.md` — checklist

## 6. (Optional) Claude Code install

Only if you want the Claude assistant on the laptop too.

Prereq: **Node.js 20+** : https://nodejs.org/ (LTS installer).

Then in PowerShell :
```bash
npm install -g @anthropic-ai/claude-code
```

Launch :
```bash
claude
```
First run asks to log in via browser — use the same Anthropic account as on the fixed PC.

Open the project :
```bash
cd C:\Users\<you>\Desktop\MA2\robot-learning-project3
claude
```
Claude will read `CLAUDE.md` automatically (it's gitignored so make sure to copy it manually from the fixed PC's repo via USB key — it's not on GitHub).

## 7. Pre-recording sanity check (5 min)

With the SO-101 NOT yet plugged :
```bash
conda activate lerobot
hf auth whoami           # → Rsebti
nvidia-smi               # only if laptop has a GPU; ignore otherwise
python -c "import lerobot; print(lerobot.__version__)"   # → 0.5.2
```

Then plug the SO-101 and follow `teleop/record_demos.md`.

## Troubleshooting

- `conda: command not found` → reopen the terminal after installing, or add `C:\Users\<you>\miniconda3\Scripts` to PATH.
- `hf: command not found` → `pip install huggingface_hub` (should be pulled in by lerobot, but just in case).
- USB ports not seen → check Device Manager → COM ports. Sometimes need to install Feetech driver: https://www.feetechrc.com/ (download → drivers).
- `lerobot-record` errors on camera → `python -c "import cv2; print(cv2.VideoCapture(0).read())"` to test the webcam first.
