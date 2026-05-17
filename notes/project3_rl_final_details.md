# Project 3: Reinforcement Learning — Final Details

> Official final project specification from the TAs. This document is the source of truth for the evaluation protocol, rules, and scoring.

---

## General Guidelines

- **No restrictions on model architecture or training procedure.**
- **Task 2 and Task 3 MUST involve reinforcement learning components.**
- The policy must operate on **visual observation of blocks**.
- Target locations are specified as **(x, y, z) coordinates in the robot frame**.
- Target locations must be **easy to modify during evaluation** (e.g. via config file or CLI argument).
- The SO-101 wrist camera is **RGB only**. If you want grayscale / depth / other image styles, you need to convert them yourself. Pick whichever representation performs best.
- **Goal-conditioned policies are encouraged** — policy receives task-specific inputs (e.g. target color, target location) and adapts behavior.
- Both **end-to-end** and **modular** approaches (e.g. perception + control) are allowed, as long as the final behavior is governed by a learned policy.
- Training data may come from **simulation** or **teleoperation**.
- **Task 1 expert demos may be reused** for RL training (e.g. as a replay buffer for offline/hybrid RL).

---

## Evaluation Environment (standardized)

- **Table:** light gray, approximately `#B8ADA9`.
- **Objects:** fixed size, distinct but **known** colors.
- **Bowls:** placed at **randomized positions** in the robot base frame (x, y, z).
- **Object (block) positions:** randomized across rollouts.

> Students are expected to account for this variability during training.

---

## Evaluation Tasks

Total: **150 points** + optional **50 bonus**.

### Eval 1 — Single Object Pick-and-Place (50 pts)

- Single wooden block placed randomly on the table.
- Policy picks the block and places it into a bowl.
- **Block color:** not constrained.
- **Target bowl position:** specified.
- **5 rollouts.**
- **Scoring:** 10 pts per successful rollout.
- **Success:** block is placed inside the bowl AND released.
- **Methods allowed:** BC or RL.

---

### Eval 2 — Targeted Pick-and-Place in Clutter (50 pts) — RL REQUIRED

- Two blocks of different colors placed adjacent to each other (flat cluster).
- One target color is specified.
- Policy must identify the block of the specified color, grasp it, and place it into a specified bowl.
- **Target color:** provided as input.
- **Target bowl position:** specified.
- **5 rollouts.**
- **Scoring:** 10 pts per successful rollout.
- **Success:** correct block placed inside the bowl AND released.
- **Reinforcement learning is mandatory.**
- Expert teleop data is encouraged for training efficiency.

---

### Eval 3 — Sequential Multi-Step Pick-and-Place (50 pts) — RL REQUIRED

- Four blocks of distinct colors in the workspace.
- Policy receives a sequence of **three** pick-and-place goals, each specifying:
  - Target color
  - Target bowl position
- **Bowl positions are fixed within each rollout.**
- Policy must execute the sequence in order.
- **Policy switching** and/or **perception module switching** (e.g. color filters) is allowed.
- **Interacting with non-target blocks** (push, rearrange) is permitted and encouraged if it helps.
- **5 rollouts.**
- **Per-rollout scoring:** 4 pts + 4 pts + 2 pts for each step. Partial credit for partial completion.
- **Step success:** correct block placed inside the corresponding bowl AND released.
- **Reinforcement learning is mandatory.**
- Expert teleop data is encouraged.

---

### Bonus — Efficiency or Singulation (50 pts, optional)

Awarded under one of two criteria:

**Option A — Speed.** Among Eval 3 rollouts that complete all three steps, a rollout counts as "fast" if it finishes within a time limit. Bonus points scale with the number of fast rollouts.

**Option B — Singulation.** Given a stack or clustered arrangement of 3–4 blocks, the policy must separate them into individually graspable configurations.

> Further bonus evaluation details will be clarified before final evaluation.

---

## Additional Notes

- **No additional hardware, objects, or setup modifications** are allowed.

---

## References and Resources

- **SO-101 Tutorial:** SO-101 (course-provided)
- **SO-101 in Isaac Lab:** https://github.com/MuammerBay/isaac_so_arm101
- **SO-101 in MuJoCo:** https://github.com/RobotControlStack/robot-control-stack

---

## Implications for our planning (Rsebti / project repo)

- **Eval 1** is the first ship target — pure BC works. Aligns with the current sanity-check pipeline (20 demos → BC → deploy).
- **Eval 2 and Eval 3 require RL.** Plan to use Isaac Lab + `isaac_so_arm101` for sim-to-real, with teleop demos seeding the replay buffer.
- **Goal-conditioning is the right abstraction:** target color (Eval 2/3) and target bowl xyz (all evals) should be policy inputs from the start.
- **Bowl positions are randomized per rollout (Eval 1/2) but fixed within Eval 3 rollouts** — design the policy interface accordingly (xyz goal as input, not hardcoded).
- **Make target locations runtime-configurable** (config file or CLI arg) — required by the spec.
- **Wrist cam is RGB only** — any depth/grayscale must be derived in software.


