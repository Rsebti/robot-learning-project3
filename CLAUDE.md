# Robot Learning — Project 3

## ⚠️ READ FIRST — Current session context

> If you are Claude Code starting a session, read this whole file then
> `notes/project3_rl_final_details.md` (TA spec) and `notes/sanity_results.md`
> (last validated milestone).

**Where the user is now (2026-05-07):** working from the **fixed PC (RTX 5070,
Blackwell sm_120)**. The repo has just been **reset for Eval 2**. Everything
related to the previous Eval 2 attempts (PPO from-scratch state-based v0–v1.4,
CNN perception module, scripted IK controller, magic-attach grasp, BC/DAPG
scaffolding, 120-démo teleop plan) has been moved out of `main` into the
branch `archive/eval2-attempts-pre-reset`. The user explicitly chose a clean
slate.

**Current focus: Eval 2 only.** Strategy:
- **RL only**, no BC warmstart, no teleop demos, no DAPG.
- **Isaac Lab** as the simulator (proven install, see `notes/isaac_lab_setup.md`).
- **Lean on existing SO-101 resources** rather than rebuilding from zero —
  primary reference: `MuammerBay/isaac_so_arm101` (already cloned at
  `C:\Users\user\Desktop\MA2\isaac\isaac_so_arm101`), and Isaac Lab
  manipulation task templates.

**Eval 1 and Eval 3:** not started, not the focus right now.

**Sanity check** (BC pipeline) is **DONE** (5/5 successful pick-and-place
rollouts on the real SO-101 with ACT, on 2026-04-28). All sanity-related code
and docs are still in this repo (`teleop/`, `train/launch_act.*`, `deploy/`,
`notes/sanity_results.md`, `notes/full_pipeline_walkthrough.md`) — they
remain useful as reference and would also be the basis for Eval 1 if it
becomes relevant later.

**🔑 HF token:** stored in `C:\Users\user\Desktop\MA2\tokens.txt` on this
fixed PC. **NEVER commit, paste, or echo this token.**

---

## Lessons from the archived attempts (do NOT relearn the hard way)

The `archive/eval2-attempts-pre-reset` branch holds ~2 weeks of work that
hit dead ends. The user is aware of these — don't re-suggest them as if
they were new ideas.

- **PPO from-scratch state-based plateaued at 2–7 % success** (v1.0–v1.4).
  Modes of failure: (a) `std` exploded when there was no `action_l2`
  penalty; (b) once stabilized, the policy gamed the intermediate
  milestones (`reaching`, `lifting`, `above_bowl`) without ever committing
  to the final placement.
- **Scripted IK (closed-form + DLS) was math-correct (FK→IK→FK 0.0001 mm)
  but operationally failed.** SO-101 is 5-DoF with tight wrist_flex limits
  (±1.658 rad). For a 6-DoF goal (xyz + vertical-gripper orientation), the
  IK saturated or the PD controller couldn't track. Not worth retrying as
  a demo generator.
- **Modular CNN perception (image → block_xyz)** was trained on the
  trajectories of a chaotic policy → biased dataset, unusable.
- **`convex_decomposition`** on the SO-101 jaws self-blocks the moving jaw.
  `convex_hull` gives an imprecise contact. The PhysX grasp on this robot
  is intrinsically unstable for rigid cubes — the previous attempt worked
  around it with a "magic attach" (cube becomes kinematic + ghost during
  the grasp). For a pure-RL approach, this constraint matters: reward
  shaping must tolerate non-clean grasps, or the env must accept the
  constraint.

If pure RL on this robot turns out to plateau again at <10 %, the user
will likely revisit the strategy. Don't preemptively pivot — execute the
chosen plan first, measure, then reassess.

---

## Project context

Master's-level Robot Learning course at ETH Zürich. Pick-and-place tasks
on the SO-101 robot arm. The user (Rayane) works on this as part of a
multi-student team.

### Tasks (TA spec — see `notes/project3_rl_final_details.md` for full text)

Total: **150 pts** + optional **50 pts bonus**.

- **Eval 1** (50 pts) — single block + bowl, randomized positions. BC or RL.
- **Eval 2** (50 pts) — **two adjacent blocks** of distinct colors, target
  color provided as input, target bowl xyz provided as input. **RL mandatory.**
  5 rollouts × 10 pts.
- **Eval 3** (50 pts) — four blocks, sequence of 3 (color, bowl) goals per
  rollout. **RL mandatory.** Bowl positions fixed within a rollout. Pushing
  non-target blocks is allowed.
- **Bonus** (50 pts) — speed (in Eval 3) OR singulation of stacked blocks.

Important spec details:
- Target locations are **(x, y, z) in robot frame**, runtime-configurable
  (CLI/config).
- Wrist cam is **RGB only** (~640×480, 30 fps on the real robot).
- Table: light gray, ~`#B8ADA9`. Bowls and blocks: fixed sizes, known colors.
- Bowls placed at **randomized positions** for Eval 1/2 (fixed within a
  rollout for Eval 3). Block positions also randomized.
- Goal-conditioning is encouraged (target color + target xyz as policy
  inputs).

---

## Team / cloud setup

- Team of multiple students. **One teammate has the SO-101 robot kit**
  (1 leader + 1 follower for bilateral teleop). The user does NOT have
  the robot at home.
- Compute: **$200 Brev credits (NVIDIA H100)** still available, **not yet
  used**. Reserve for Eval 2 RL training. **Stop instances when not in use.**
- Weekly Slack updates to the TAs (every Thursday).
- **GitHub:** `Rsebti/robot-learning-project3` (private). Teammate has access.
- **HF dataset (sanity, public):** `Rsebti/projet3-demos-v1bis` (19 episodes,
  ~19 MB).
- **HF model (sanity, private):** `Rsebti/projet3-act-sanity` (ACT checkpoint
  used for the 5/5 deploy).

---

## Hardware (when next at the robot)

- **Robot:** SO-101 follower (Feetech STS3215 servos), bilateral teleop
  with the SO-101 leader.
- **USB ports** (last known on the laptop, will likely differ on a different
  machine — re-run `lerobot-find-port`):
  - Follower → COM3
  - Leader → COM5
- **Wrist camera:** OpenCV index **1** on the laptop (built-in webcam at
  index 0). Re-check with `lerobot-find-cameras opencv`.
- **Calibration files** stored under
  `~/.cache/huggingface/lerobot/calibration/` (machine-local, re-calibrate
  on a fresh machine).

---

## Local environments

### Fixed PC (this machine, RTX 5070, Blackwell sm_120)
- Anaconda installed, conda env `lerobot` (Python 3.12). Used for sanity
  pipeline pieces that need lerobot.
- LeRobot installed in **editable mode** at
  `C:\Users\user\Desktop\MA2\lerobot`. **DO NOT modify the lerobot library
  code there.**
- HF authenticated.
- **Isaac Sim 5.1 + Isaac Lab 2.3 + `isaac_so_arm101`** installed via `uv` at
  `C:\Users\user\Desktop\MA2\isaac\isaac_so_arm101`. The `.venv` there is
  the runtime for any sim work. Setup steps: `notes/isaac_lab_setup.md`.
- ⚠️ Blackwell sm_120 may need PyTorch nightly cu128 if a wheel mismatch
  shows up. `notes/isaac_lab_setup.md` documents the fix.

### Portable laptop (used for the sanity-check session at the robot)
- Miniconda at `C:\Users\sebti\miniconda3`, env `lerobot` (Python 3.12).
- LeRobot 0.5.1 installed non-editably via pip (no editable lerobot here).
- HF authenticated.

---

## Pipelines

### Sanity check / Eval 1 path (BC) — proven
1. `lerobot-record` with bilateral teleop → push to HF
2. `lerobot-replay` (sanity check, optional)
3. Train ACT (`train/launch_act.sh`) → push to HF
4. Deploy with `lerobot-record` + `--policy.path` → real robot eval

Reference docs in this repo: `teleop/`, `train/`, `deploy/`,
`notes/full_pipeline_walkthrough.md`.

### Eval 2 (RL only — sim-to-real) — to build

The user's chosen strategy:
- Isaac Lab + `isaac_so_arm101` as foundation.
- Build the env using existing SO-101 resources rather than reinventing
  the scene/managers from scratch.
- Pure RL (PPO via `rsl_rl`, the integration that ships with
  `isaac_so_arm101`). **No BC, no DAPG, no teleop demos.**
- Goal-conditioning: target color + target bowl xyz as policy inputs.
- Wrist-cam image observation from the start (so sim→real transfer is
  straightforward), with domain randomization baked in.
- Train on the local RTX 5070 first; move to Brev H100 if local can't
  hit acceptable success rate or is too slow.

⚠️ The plan above is the **starting direction**, not a finished design.
The first concrete step is to inspect what `isaac_so_arm101` already
provides (which task templates? what reward terms? what asset configs?)
and use that as the scaffold. Code under this repo will live in `sim/`
when re-created (currently empty after the reset).

---

## Key technical quirks (sanity session, 2026-04-28)

These bit during the laptop session — keep them in mind for any future
record/deploy:

1. **PowerShell + lerobot CLI:** the camera JSON arg breaks under PS5.1
   native arg parsing. Use the `--%` stop-parsing token AND escape `"` as
   `\"` inside the JSON. Pattern in `teleop/record_demos.md`.
2. **wgpu/Vulkan warnings spam** when `--display_data=true` is on. Mute
   with `$env:RUST_LOG = "error"` before the lerobot command.
3. **Eval dataset name MUST start with `eval_`** when `--policy.path` is
   set, or lerobot raises a `ValueError`.
4. **lerobot-record fails if the local cache dir already exists**
   (`FileExistsError`). `Remove-Item -Recurse -Force` the cache before
   each run, or use a fresh `repo_id`.
5. **Encoding phase between episodes is silent for ~10–15 s**
   ("Svt[info]:" dump). **Do NOT press any key during this phase** —
   keypresses are buffered and instantly skip the next episode.
6. **CPU inference** is fine for ACT thanks to chunking (100 actions
   ahead). "Record loop running slower than target FPS" warnings are
   cosmetic.
7. **lerobot 0.5.2 is NOT on PyPI** (as of 2026-04-28). Latest published
   is **0.5.1**. The fixed PC's editable lerobot may report 0.5.2-dev —
   don't pin 0.5.2 in the laptop install.
8. **Conda init for PowerShell:** if `conda activate` doesn't work, run
   `conda init powershell` once and restart the shell.
9. **Between deploy episodes:** plug the leader and use it to bring the
   follower back to home pose during reset. Without the leader the
   follower's torque stays on.

---

## Working style preferences

- Concrete numerical examples before abstract formulas.
- Step-by-step progressive reasoning, no skipped steps.
- Socratic when learning new concepts: let the user attempt before correcting.
- Visuals/widgets useful for spatial or statistical intuition.
- The user codes on Windows (PowerShell), GitHub Desktop for git, edits
  with VS Code.
- **Communication mostly in French**, but technical terms and code in English.
- The user is OK with you running read-only commands and small clean-up
  commands (e.g. `Remove-Item` for stale lerobot caches), but **always
  ask before** touching the real robot, before launching paid GPU runs
  on Brev, and before any push to HF that affects shared state.

---

## Repo structure (post-reset, 2026-05-07)

```
robot-learning-project3/
├── teleop/                              # sanity pipeline — proven
│   ├── record_demos.md
│   └── demo_protocol.md
├── train/                               # sanity pipeline — proven
│   ├── launch_act.sh
│   ├── launch_act.md
│   └── train_bc.md
├── deploy/                              # sanity pipeline — proven
│   ├── inference.md
│   ├── infer.sh
│   └── deploy_policy.md
├── notes/
│   ├── sanity_results.md                # 5/5 deploy recap
│   ├── full_pipeline_walkthrough.md     # detailed sanity walkthrough
│   ├── laptop_setup.md
│   ├── isaac_lab_setup.md               # Isaac Lab/Sim install on this PC
│   └── project3_rl_final_details.md     # TA spec — source of truth
├── (sim/ — to recreate when Eval 2 dev starts)
├── .gitignore
├── README.md
├── CLAUDE.md                            # this file (tracked, shared with team)
└── document_pdf.pdf                     # TA spec (gitignored locally)
```

The branch `archive/eval2-attempts-pre-reset` holds the full pre-reset
state (sim/eval2 with all envs/managers, perception CNN, scripted IK,
magic attach, BC/DAPG plans, 120-démo teleop plan, audit docs). Useful
if a specific piece needs to be referenced.

---

## What's NEXT (immediate)

1. **Inspect `isaac_so_arm101`** to understand what task templates and
   reward terms it ships with — that's the scaffold for the new Eval 2
   env. Don't write code yet.
2. **Decide the Eval 2 env shape** before any code: scene composition,
   action space (joint-pos vs cartesian delta vs IK), observation space
   (image yes/no, what state vector), reward terms (lean on isaac_so_arm101
   defaults). Validate the design with the user before implementing.
3. **Recreate `sim/eval2/`** with a minimal env — a single task version
   that runs PPO end-to-end (no shortcuts, no parallel "easier" version
   first). Get to "PPO converges on something" before adding the real
   Eval 2 complexity.
4. **Iterate on the real Eval 2 task** (2 colored blocks + bowl_xyz +
   target_color), with domain randomization and image observation from
   the start.
5. **Deploy on the real robot** (back at the lab).

---

## Important warnings

- **DO NOT commit `tokens.txt` or any HF token.** Do not echo or paste
  the token in chat either.
- **DO NOT modify the `lerobot` library** under
  `C:\Users\user\Desktop\MA2\lerobot` (editable install). Same rule for
  the laptop's pip install.
- **DO NOT modify `isaac_so_arm101`** under
  `C:\Users\user\Desktop\MA2\isaac\isaac_so_arm101`. That repo is a
  reference scaffold, kept clean for re-syncs. Project code goes in
  THIS repo's `sim/`.
- **One exception — rsl_rl is patched.** A single one-line patch is
  applied to
  `C:\Users\user\Desktop\MA2\isaac\isaac_so_arm101\.venv\Lib\site-packages\rsl_rl\algorithms\ppo.py`
  (LR floor 1e-5 → 1e-4) to prevent the optimizer-collapse cascade
  observed in V2.18 cold. The patch is NOT tracked by git; re-apply
  after any venv rebuild via:
      `python sim/eval2/scripts/patch_rsl_rl.py`
  The script is idempotent. See `notes/rsl_rl_patches.md` for context.
- **All project code lives in this repo.** Layer 1 (Isaac Sim/Lab) and
  Layer 2 (`isaac_so_arm101`) stay outside.
- **Before any real-robot deploy:** verify scene matches the recording
  conditions, ports correct, camera index correct.
- **Brev costs money.** Stop instances when not training. Don't leave
  overnight unless intentional.
- **Don't blindly resurrect ideas from the archived branch.** The lessons
  in section "Lessons from the archived attempts" above already capture
  what didn't work and why.
