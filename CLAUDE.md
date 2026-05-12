# Robot Learning — Project 3

## ⚠️ READ FIRST — Current session context

> If you are Claude Code starting a session, read this whole file then `notes/project3_rl_final_details.md` (TA spec) and the `notes/sanity_results.md` recap.

**Where the user is now:** working from the **fixed PC (RTX 5070, Blackwell sm_120)**, after a successful laptop session at the robot.
- ✅ Sanity check **DONE** (5/5 successful pick-and-place rollouts on the real SO-101 with ACT, on 2026-04-28).
- 🆕 **Current focus: Eval 2** (RL-based color-conditioned pick in clutter). Must set up Isaac Lab, build the env, train PPO, transfer to real robot.
- ⏳ Eval 1 (BC + randomized positions) is **NOT** started yet but is straightforward (same pipeline as sanity, just more demos with variability).
- 🔑 HF token: stored in `C:\Users\user\Desktop\MA2\tokens.txt` on this fixed PC. **NEVER commit, paste, or echo this token.**

**Today's goal (and the days that follow):** start Eval 2.
- Install Isaac Lab + Isaac Sim. ⚠️ Known sm_120/Blackwell compatibility issues — be prepared to fall back to Brev (H100) if drivers misbehave on the 5070.
- Clone `MuammerBay/isaac_so_arm101` (TA-recommended repo with SO-101 already integrated for Isaac Lab).
- Draft a first env (`sim/eval2/pick_in_clutter_env.py`): SO-101 + 2 colored blocks + bowl + wrist camera + reward shaping + domain randomization.
- Run a small PPO training to validate the pipeline; full training likely on Brev.

**Behavior expected from Claude:**
- The user is intermediate now (just shipped a full BC pipeline) but **new to Isaac Lab + RL applied to robotics**. Be concrete, step-by-step, no skipped steps.
- Communicate in English. Code/technical terms stay in English (no change).
- The user codes on Windows (PowerShell), pushes via GitHub Desktop. Watch for the Windows-specific gotchas listed below.
- **Do NOT auto-run training/deploy commands without confirmation** — these involve the real robot or paid GPU time.

---

## Project context
Master's level Robot Learning course at ETH Zürich. Pick-and-place tasks on the SO-101 robot arm using BC and RL.

The user (Rayane) is working on it as part of a multi-student team.

## Team setup
- Team of multiple students. **One teammate has the SO-101 robot kit** (1 leader + 1 follower, bilateral teleop). The user does NOT have the robot at home.
- **Training is delegated to a teammate** (their machine has a working GPU stack). The user's RTX 5070 has Blackwell sm_120, which has known compatibility issues with Isaac Lab and sometimes with stock PyTorch wheels.
- Compute available: **$200 Brev credits (NVIDIA H100)** — used as fallback for heavy training and for Isaac Lab if the local install fails.
- Weekly Slack updates to the TAs (every Thursday).

---

## Tasks structure (TA spec — see `notes/project3_rl_final_details.md` for full details)
- **Sanity check** (✅ DONE): 19 quasi-identical demos → ACT → 5/5 deploy success.
- **Eval 1** (50 pts, ⏳ pending): single block + bowl with **randomized positions** (block always, bowl per rollout). 5 rollouts × 10 pts. **BC allowed.**
- **Eval 2** (50 pts, 🆕 current focus): **two adjacent blocks** of different colors, **target color specified as input**, target bowl xyz specified, 5 rollouts × 10 pts. **RL mandatory.**
- **Eval 3** (50 pts): four blocks, **sequence of 3 (color, bowl)** goals per rollout, 5 rollouts × (4+4+2 pts). **RL mandatory.** Bowl positions fixed within a rollout. Pushing/rearranging non-target blocks allowed.
- **Bonus** (50 pts): speed (in Eval 3) OR singulation of stacked blocks.

Important spec details:
- Target locations are **(x, y, z) in robot frame** and must be runtime-configurable (CLI/config).
- **Goal-conditioned policies are encouraged** (target color + target xyz as policy inputs).
- Wrist cam is **RGB only** — derive grayscale/depth/etc. in software if needed.
- Sim demos and real teleop demos can both be used. **Sanity-check demos can be reused as a replay buffer for offline/hybrid RL.**
- Bowls placed at **randomized positions** in robot base frame for Eval 1/2 (fixed within a rollout for Eval 3).

---

## What's been done so far

### Sanity check pipeline — DONE end-to-end (✅ 5/5)
1. ✅ Recorded **19 quasi-identical demos** (target was 20, but lerobot lost the last episode to a keyboard-buffer edge case — 19 was enough).
2. ✅ Pushed to HF: `Rsebti/projet3-demos-v1bis` (**now public** so the teammate could pull without setting up collaborator access).
3. ✅ Replay verification on episodes 0/5/10/15 — passed.
4. ✅ Trained ACT on the teammate's GPU (5090 / ETHRC box) → pushed to HF: `Rsebti/projet3-act-sanity` (~66 MB safetensors). Training script: `train/launch_act.sh` (see `train/launch_act.md`).
5. ✅ Deployed on real SO-101 from the laptop (CPU inference): **5/5 success**.

Detailed write-up in `notes/sanity_results.md`.

### Repo state
- GitHub repo `Rsebti/robot-learning-project3` (private) is **up-to-date** including:
  - `teleop/record_demos.md` — final record command (COM3 follower, COM5 leader, camera idx 1, dataset `v1bis`)
  - `train/launch_act.sh` + `launch_act.md` — teammate's training launcher
  - `deploy/inference.md` + `deploy/infer.sh` — teammate's deploy launcher
  - `deploy/deploy_policy.md` — physical pre-flight checklist
  - `notes/laptop_setup.md` — laptop install (lerobot 0.5.1 — **NOT** 0.5.2, that one isn't on PyPI)
  - `notes/project3_rl_final_details.md` — TA's official spec
  - `notes/sanity_results.md` — recap of the 5/5 deploy

### Cloud setup
- **GitHub:** `Rsebti/robot-learning-project3` (private). Teammate has access.
- **HF dataset:** `Rsebti/projet3-demos-v1bis` (**public**, 19 episodes, ~19 MB).
- **HF model:** `Rsebti/projet3-act-sanity` (private — ACT checkpoint from sanity training, used for the 5/5 deploy).
- **HF eval dataset:** `Rsebti/eval_projet3-sanity` (local-only, not pushed; required `eval_` prefix because lerobot enforces it when a policy is provided).
- **Brev:** $200 credits available, **not yet used** (the team training was done on a teammate's local 5090). Reserve for Isaac Lab + RL training.
- **HF token** location reminder: `C:\Users\user\Desktop\MA2\tokens.txt` on this fixed PC.

---

## Hardware (when next at the robot)

- **Robot:** SO-101 follower (Feetech STS3215 servos), bilateral teleop with the SO-101 leader.
- **USB ports** (last known on the laptop, may differ on the fixed PC — re-run `lerobot-find-port`):
  - Follower → COM3
  - Leader → COM5
- **Wrist camera:** OpenCV index **1** (laptop has built-in webcam at index 0). Re-check on a new machine with `lerobot-find-cameras opencv`.
- **Calibration files** are stored under `~/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower.json` and `.../teleoperators/so_leader/so101_leader.json`. They're machine-local — re-calibrate if you switch laptops.

---

## Local environments

### Fixed PC (this machine, RTX 5070)
- Anaconda installed, conda env `lerobot` exists (Python 3.12).
- LeRobot is installed in **editable mode** at `C:\Users\user\Desktop\MA2\lerobot`. **DO NOT modify the lerobot library code there.**
- HF authenticated.
- ⚠️ For Isaac Lab: **install carefully**. Blackwell sm_120 may need PyTorch nightly + matching CUDA, and Isaac Lab/Sim may have stability issues. If the local install eats more than ~1 day, fall back to Brev.

### Portable laptop (used for the sanity-check session at the robot)
- Miniconda installed at `C:\Users\sebti\miniconda3`, env `lerobot` (Python 3.12).
- LeRobot 0.5.1 installed non-editably via pip (`lerobot[feetech,dataset]==0.5.1`).
- HF authenticated.
- Don't try to edit lerobot source on the laptop.

---

## Pipeline (high level)

### Sanity check / Eval 1 (BC) — proven path
1. `lerobot-record` with bilateral teleop → push to HF
2. `lerobot-replay` (sanity check, optional)
3. Train ACT (`train/launch_act.sh`, on teammate's GPU or Brev) → push to HF
4. Deploy with `lerobot-record` + `--policy.path` → real robot eval

### Eval 2 / Eval 3 (RL — sim-to-real)
**Layered architecture (IMPORTANT — this is how we share work):**
- **Layer 1** (installed locally, never in repo): Isaac Sim + Isaac Lab.
- **Layer 2** (installed locally, never in repo): `MuammerBay/isaac_so_arm101` (SO-101 USD + base config).
- **Layer 3** (in this repo, what we collaborate on): `sim/eval2/`, `sim/eval3/` — env definitions, training scripts, configs.

**Eval 2 plan (to be detailed in `notes/eval2_plan.md`):**
1. Install Isaac Lab on this PC (or Brev if 5070 fights).
2. Clone `isaac_so_arm101`, run an example to confirm install works.
3. Create `sim/eval2/pick_in_clutter_env.py`: 2 colored blocks adjacent + bowl + wrist cam, target color as input, target bowl xyz as input.
4. Reward shaping (the hard part): dense distance-to-target-block, bonuses on grasp/lift/place, penalties on touching wrong block.
5. Domain randomization (light/textures/friction/initial poses) for sim-to-real.
6. Train PPO with `rsl_rl` or `stable-baselines3` (Isaac Lab integrates `rsl_rl`).
7. Eval in sim ≥80% success → deploy on real robot.
8. **Reuse sanity-check demos** as a replay buffer or BC warmstart if pure RL is too slow.

**Brev cost estimate:** ~$2-3/h on H100 → 60-100h budget. Reserve enough for Eval 2 + Eval 3 + dev/test (~60-90h total). **Stop instances when not in use.**

---

## Key technical quirks discovered (lessons from the sanity-check session)

These bit us during the laptop session — keep them in mind for any future record/deploy:

1. **PowerShell + lerobot CLI:** the camera JSON arg breaks under PS5.1 native arg parsing. Always use the `--%` stop-parsing token AND escape `"` as `\"` inside the JSON. Example pattern in `teleop/record_demos.md`.
2. **wgpu/Vulkan warnings spam** when `--display_data=true` is on. Mute them with `$env:RUST_LOG = "error"` before the lerobot command. Without this, the terminal floods at 30 fps and Ctrl+C becomes hard.
3. **Eval dataset name MUST start with `eval_`** when `--policy.path` is set, otherwise lerobot raises a `ValueError` (wants you to clearly separate eval data from training data).
4. **lerobot-record fails if the local cache dir already exists** (`FileExistsError` from `obj.root.mkdir(... exist_ok=False)`). Always `Remove-Item -Recurse -Force` the cache before each run, or use a fresh `repo_id`.
5. **Encoding phase between episodes is silent for ~10-15 s** ("Svt[info]:" dump). **Do NOT press any key during this phase** — keypresses are buffered and will fire on the next episode start, instantly skipping it with 0 frames (causes a `ValueError: You must add one or several frames before calling add_episode`).
6. **CPU inference** is fine for ACT thanks to the chunking (100 actions ahead). Expect "Record loop running slower than target FPS" warnings — they're cosmetic, the bras still moves smoothly.
7. **lerobot 0.5.2 is NOT on PyPI** as of 2026-04-28. The latest published is **0.5.1**. The fixed PC's `lerobot` install is from a git checkout (editable) and may report 0.5.2-dev — don't pin 0.5.2 in the laptop install.
8. **Conda init for PowerShell:** if `conda activate` doesn't work in PowerShell, run `conda init powershell` once and restart the shell.
9. **Between deploy episodes**, plug the leader and use it for the reset phase to bring the follower back to home pose. Without the leader, the follower's torque stays on and you have to power-cycle the alim to reposition it.

---

## Working style preferences

- Concrete numerical examples before abstract formulas.
- Step-by-step progressive reasoning.
- Socratic when learning new concepts: let the user attempt before correcting.
- No skipped steps in explanations.
- Visuals/widgets useful for spatial or statistical intuition.
- The user codes on Windows (PowerShell), GitHub Desktop for git, edits with VS Code.
- Communication in English. Technical terms and code in English.
- The user is OK with you running read-only commands and small clean-up commands (e.g. `Remove-Item` for stale lerobot caches), but **always ask before touching the real robot, before launching paid GPU runs (Brev), and before any push to HF that affects shared state**.

---

## Repo structure (as of 2026-04-30)

```
robot-learning-project3/
├── teleop/
│   ├── record_demos.md        # final record command (COM3/COM5, idx 1, v1bis)
│   └── demo_protocol.md       # physical-setup checklist
├── train/
│   ├── launch_act.sh          # teammate's ACT training wrapper
│   ├── launch_act.md          # ACT-on-small-data defaults explained
│   └── train_bc.md            # generic train command + Blackwell note
├── deploy/
│   ├── inference.md           # how to run the trained policy on the real robot
│   ├── infer.sh               # bash launcher
│   └── deploy_policy.md       # physical pre-flight + pass criteria
├── sim/                       # (to be created for Eval 2/3)
│   └── eval2/                 # (to be created)
├── notes/
│   ├── laptop_setup.md        # fresh-laptop install steps
│   ├── project3_rl_final_details.md   # TA spec — source of truth
│   ├── sanity_results.md      # 5/5 deploy recap
│   └── eval2_plan.md          # (to be created)
├── .gitignore                 # ignores .claude/, *.webp, document_pdf.pdf
├── README.md
└── CLAUDE.md                  # this file (gitignored — kept locally on each machine)
```

---

## What's NEXT (for the fixed PC, in order)

1. **Update teammates / TAs** that the sanity check passed (Slack/Thursday update).
2. **Decide Eval 2 dev location:** local Isaac Lab on 5070 vs. Brev. Try local first (1 day timebox); if drivers fight, switch to Brev.
3. **Install Isaac Lab + clone `isaac_so_arm101`** on whichever machine wins step 2.
4. **Run an existing `isaac_so_arm101` example** to validate the install (a basic sim of the SO-101 in Isaac Lab).
5. **Draft `sim/eval2/pick_in_clutter_env.py`** — start small (1 block, no clutter) to learn the API; extend to 2 colored blocks + color-conditioning afterward.
6. **First PPO training run** (small num_envs, short horizon) to verify the pipeline doesn't crash.
7. **Iterate on reward shaping + domain randomization** until ≥80% success in sim.
8. **Deploy on real robot** (back at the lab next to the SO-101, which means probably back to the laptop session pattern).
9. **In parallel:** record Eval 1 dataset (50-100 demos with randomized block positions). Eval 1 doesn't depend on Eval 2 progress.

---

## Open questions / open issues

- Will Isaac Lab run cleanly on RTX 5070 (sm_120)? If not → Brev. Set a 1-day timebox on local-install attempts.
- Do we want goal-conditioning at the **observation** level (target color as one-hot in obs) or as a **separate input head** to the policy? Default to the former (simpler) unless evidence says otherwise.
- For Eval 2, should we **warmstart with BC** on real teleop demos before PPO? TA spec encourages it ("Expert teleop data is encouraged for training efficiency"). Probably yes once basic PPO is wired up.

---

## Important warnings

- **DO NOT commit `tokens.txt` or any HF token.** Do not echo or paste the token in chat either.
- **DO NOT modify the `lerobot` library code** under `C:\Users\user\Desktop\MA2\lerobot` (fixed PC editable install) — that's the library, not the project.
- On the laptop, `lerobot` is installed non-editably via pip — same rule, do not edit its source.
- **All project code goes in this repo** (`robot-learning-project3`), regardless of machine. The exception is Isaac Lab itself + `isaac_so_arm101`, which are installed locally on each machine and never committed.
- **Before any real-robot deploy:** verify scene matches the recording (block on X1, bowl on X2, lighting unchanged), main proche de l'alim, ports correct, camera index correct.
- **Brev costs money.** Stop instances when not training. Don't leave overnight unless intentional.
