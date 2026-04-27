# Robot Learning — Project 3

## Project context
Master's level Robot Learning course at ETH Zürich. The project is to do pick-and-place tasks on the SO-101 robot arm using Behavior Cloning (BC) and Reinforcement Learning (RL).

The user (Rayane) is a beginner on this stack but has done previous BC and RL coursework.

## Team setup
- Team of multiple students. One teammate has the SO-101 robot (full kit: 1 leader + 1 follower for bilateral teleop). User does NOT have the robot at home.
- User's PC: RTX 5070 (Blackwell sm_120) — known compatibility issues with Isaac Lab.
- User also has a portable laptop they will use tomorrow to work next to the robot.
- Compute: $200 Brev credits available (NVIDIA H100), but the team is leaning toward training locally for the sanity check.

## Tasks structure
- **Sanity check** (this week): 20 identical demos, BC training, deploy to verify pipeline works end-to-end.
- **Eval 1** (50 pts): single-block pick-and-place with randomized positions. Pure BC.
- **Eval 2** (50 pts): targeted pick in clutter, color-conditioned. **Must use RL.**
- **Eval 3** (50 pts): sequential 3-step pick-and-place. **Must use RL.**
- **Bonus** (50 pts): speed or singulation.

## Decisions taken
- Use **LeRobot** (HuggingFace) as the main framework.
- Format: **LeRobot dataset v3**.
- Initial sanity check uses BC (ACT or Diffusion Policy).
- For Evals 2 and 3, will use **Isaac Lab** for sim-to-real RL (NVIDIA's robotics framework). The repo `isaac_so_arm101` has the SO-101 already integrated.
- Reference repo (provided by TAs): https://github.com/MuammerBay/isaac_so_arm101

## Cloud accounts setup
- GitHub: `Rsebti/robot-learning-project3` (private)
- HuggingFace dataset: `Rsebti/projet3-demos-v1` (private)
- HuggingFace token: stored in `C:\Users\user\Desktop\MA2\tokens.txt` (NEVER commit this)
- Brev: pending — coupon not yet redeemed (waiting for team captain decision)

## Local environment
- Anaconda installed
- Conda env: `lerobot` (Python 3.12)
- LeRobot 0.5.2 installed in editable mode at `C:\Users\user\Desktop\MA2\lerobot`
- Installed extras: `feetech` + `dataset`
- HF authenticated locally

## Repo structure
robot-learning-project3/
├── teleop/         # Recording scripts
│   └── record_demos.md   # Pre-written commands for tomorrow
├── train/          # Training scripts (to be filled)
├── deploy/         # Deployment scripts (to be filled)
├── notes/          # Logs and protocols
│   └── demo_protocol.md  # Checklist for demo recording
├── .gitignore
├── README.md
└── CLAUDE.md       # This file


## Hardware/teleop config
- Robot: SO-101 follower (Feetech servos)
- Teleop: SO-101 leader (bilateral teleop)
- Need to identify USB ports tomorrow at the robot:
  - Linux: `ls /dev/tty*` (typically /dev/ttyACM0 and /dev/ttyACM1)
  - Windows: Device Manager → COM ports
- One wrist camera on the follower (RGB only per project specs)

## Pipeline (sanity check)
1. Record 20 identical pick-and-place demos with `lerobot-record`
2. Replay open-loop with `lerobot-replay` to verify (TA email step 2)
3. Push dataset to HF (`Rsebti/projet3-demos-v1`)
4. Train BC policy locally (or on Brev) with `lerobot-train`
5. Pull checkpoint, deploy on SO-101
6. If pipeline works → move to Eval 1

## Key constraints
- Demos must be quasi-identical: same motion, same start/end positions, same lighting.
- Block + bowl positions marked with tape and not moved between demos.
- Episode length: 10-15s.
- Dataset format: LeRobot v3 (mandatory per TA email).

## Working style preferences (from previous discussions with the user)
- Concrete numerical examples before abstract formulas.
- Step-by-step progressive reasoning.
- Socratic when learning new concepts: let the user attempt before correcting.
- No skipped steps in explanations.
- Visuals/widgets useful for spatial or statistical intuition.
- The user codes on Windows (PowerShell + Anaconda Prompt), GitHub Desktop for git.
- Communication mostly in French, but technical terms and code in English.

## What's been done so far
- All cloud accounts created and configured
- Local LeRobot environment ready
- `record_demos.md` (recording command template) drafted
- `demo_protocol.md` (recording checklist) drafted
- Files committed and pushed to GitHub

## What's NEXT (actions for tomorrow)
1. Set up LeRobot on the portable laptop (same install steps as on fixed PC)
2. At the robot: identify USB ports, fill in the placeholders in `record_demos.md`
3. Setup physical scene (tape positions, fixed lighting)
4. Record 20 demos with the bilateral teleop
5. Replay verification (4 demos)
6. Push dataset to HF
7. Train a BC policy (ACT or Diffusion Policy)
8. Deploy on real SO-101
9. Validate pipeline → move to Eval 1

## Open questions / open issues
- Should training happen locally on the user's GPU (5070) or on Brev? Currently leaning toward local for sanity check.
- Brev coupon not yet redeemed — pending team captain decision.
- TA-required weekly Slack updates (every Thursday).

## Important warnings
- DO NOT commit `tokens.txt` or any HF token.
- DO NOT modify the `lerobot` library code under `C:\Users\user\Desktop\MA2\lerobot` (it's the library install, not the project).
- All project code goes in `C:\Users\user\Desktop\MA2\robot-learning-project3`.