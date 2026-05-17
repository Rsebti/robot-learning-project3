# Handoff: evaluate `ggand0/hil-serl-so101` and port our HIL-SERL v1 work onto it

> Paste this into a fresh Claude Code session **on this Mac**, in the same
> repo (`/Users/admin/Documents/ETH/M4/Robot Learning /Project S101/robot-learning-project3`).
> Branch: `tom-main`. The current session built HIL-SERL scaffolding directly
> on top of lerobot 0.5.2 main and hit a wall — see "Current blocker" below.

**External repo to evaluate:** https://github.com/ggand0/hil-serl-so101

## TL;DR

We've spent today building HIL-SERL scaffolding on top of stock lerobot 0.5.2
for the Eval 2 (single-color yellow pick-and-place) task. Hit a hard blocker
that's not config-fixable: lerobot 0.5.2's HIL-SERL pipeline requires the
teleop device to provide both actions AND events (s/esc/space presses), but
the SO-101 leader arm has no buttons. The HF doc describes "leader + keyboard"
as a supported mode but the code wiring is missing.

`ggand0/hil-serl-so101` is the only public SO-101 HIL-SERL reproduction we
know of (mentioned in `notes/so101_robot_learning_playbook.md` — "reached
~70% on grasp-only after 3 weeks of fixes"). It likely solved the
leader+events problem one way or another.

**Your job:** evaluate ggand0's repo, decide whether to switch to it for v1
training, and (if yes) port over the decisions + assets we already built
without re-litigating them.

## Why we're considering switching (the current blocker, in detail)

When we run `python -m lerobot.rl.gym_manipulator --config_path .../env_config_so101.json`,
the pipeline crashes at:

```
File "lerobot/processor/hil_processor.py", line 89, in _check_teleop_with_events
    raise TypeError(
TypeError: Teleoperator SOLeader must implement get_teleop_events() method.
Compatible teleoperators: GamepadTeleop, KeyboardEndEffectorTeleop
```

Tracing the source code in `/Users/admin/Documents/ETH/M4/Robot Learning /Project S101/lerobot/src/lerobot`:

1. `rl/gym_manipulator.py:336` instantiates exactly ONE teleop:
   `teleop_device = make_teleoperator_from_config(cfg.teleop)`
2. Same `teleop_device` passes through both
   `AddTeleopActionAsComplimentaryDataStep(teleop_device=...)` (actions) and
   `AddTeleopEventsAsInfoStep(teleop_device=...)` (events) at `rl/gym_manipulator.py:474-475`.
3. `processor/hil_processor.py:89` runs `isinstance(teleop, HasTeleopEvents)` —
   crashes if the device lacks `get_teleop_events()`.

Only `GamepadTeleop` and `KeyboardEndEffectorTeleop` implement the events
protocol (confirmed via `grep -rln "def get_teleop_events" lerobot/teleoperators/`).

`SOLeader` is a passive arm — it physically has no buttons to map to s/esc/space.
The HF doc's "leader + use a keyboard alongside for events" pattern requires
either a second teleop slot OR a separate always-on keyboard listener thread.
Neither is in lerobot 0.5.2.

**Hardware reality:** the user has the SO-101 leader arm but does NOT have a
gamepad on hand. Keyboard-driven EE teleop technically works but is
unworkably imprecise for grasping small cubes.

## What ggand0's repo plausibly solves

Per the playbook excerpt (notes/so101_robot_learning_playbook.md, §T2):

> "ggando reproduction reached ~70% on grasp-only after three weeks of fixes:
> MuJoCo-FK replacement, state caching bug, lighting sensitivity."

We don't know yet if leader+events specifically was one of those fixes — that's
the first thing to check. If yes → switching saves us writing the patch.
If no → we may have to patch lerobot ourselves anyway, and the value
proposition shifts toward "stay on lerobot main, write a single targeted patch."

## Evaluation tasks for you (in priority order)

### Step 1 — clone + read (5 min)

```bash
mkdir -p ~/Desktop/eval-ggand0
cd ~/Desktop/eval-ggand0
git clone https://github.com/ggand0/hil-serl-so101
cd hil-serl-so101
cat README.md
ls -la
```

Note: this is INTENTIONALLY outside our project repo. We're evaluating, not
committing yet.

### Step 2 — answer the four critical questions

For each, find the file/line that supports your answer:

1. **Does ggand0 solve the leader+events problem?**
   - Does the repo have a custom teleop class (e.g. `SOLeaderWithEvents`)
     that combines leader actions with a keyboard listener?
   - Or does it use a different approach (gamepad-only, modified processor
     pipeline, monkey-patch)?
   - Or does it just document "use a gamepad" and not solve the leader case?

2. **What lerobot version does it pin to?**
   - Check `pyproject.toml` / `requirements.txt` / `setup.py`.
   - If it pins to a fork or specific commit of lerobot, that's important —
     we cannot easily mix it with the 0.5.2 editable install we have at
     `/Users/admin/Documents/ETH/M4/Robot Learning /Project S101/lerobot/`.
   - If it's a totally different RL stack (e.g. ports to the original
     `rail-berkeley/hil-serl` JAX codebase, not lerobot), that's also fine
     but a bigger pivot.

3. **What config schema does it use?**
   - Is it JSON/YAML configs like our `sim/hilserl/configs/env_config_so101.json`?
   - Or Python script + CLI flags?
   - Does it have an equivalent of our `processor.inverse_kinematics.end_effector_bounds`?

4. **What's its action/observation space?**
   - 6-D joint targets, or 3-D EE deltas + 1-D gripper, or something else?
   - 30 Hz or 10 Hz control loop?
   - This determines whether our converted demos (joint→EE @ 30 fps) can
     drop in directly or need re-conversion.

### Step 3 — produce a 1-page verdict

After Step 2, write `notes/ggand0_evaluation.md` with:

- **Verdict:** SWITCH / STAY / SWITCH-FOR-EVENTS-ONLY
- **Reasoning:** which of the 4 questions tipped it, with file:line citations
- **Migration cost estimate:** if SWITCH, how many hours of plumbing
- **Risk list:** the top 3 things that could surprise us mid-port

Then check with the user (Tom) before doing any actual porting work.

## What we already built today (DO NOT relitigate)

Everything below is committed in `tom-main` at commit `6a2d220`. If ggand0's
repo demands different formats, ADAPT the values — don't redo the thinking.

### Locked decisions (from `notes/hilserl_eval2_plan.md`, plan-eng-review v2)

- **D8: Option A** — SmolVLA stays a parallel BC baseline (already on HF as
  `osammotg1/projet3-smolvla-eval2-v1-dark-noise-100k`). HIL-SERL trains a
  fresh independent SAC policy. We submit whichever wins the 5-rollout
  matrix.
- **D1: single color (yellow) for v1.** ~15 demos. Multi-color is v1.5,
  unblocked by the dataset conversion running on the 5090 (see
  `notes/HANDOFF_5090_HILSERL_DATASET_CONVERSION.md`).
- **D2: train at fixed bowl pose (-15.5, 29.5) cm.** Goal-conditioning is v1.5.
- **D3: manual keyboard reward (s/esc) for v1.** Trained ResNet-10 classifier
  is v1.5.

### Assets you can lift directly

| Asset | Path | Notes |
|---|---|---|
| SO-101 URDF + meshes | `sim/hilserl/assets/so101/urdf/` | Vendored from `MuammerBay/isaac_so_arm101`. Meshes gitignored — refetch per `assets/so101/README.md`. URDF link names confirmed: 6 joints + `gripper_frame_link`. |
| Home pose (joint degrees) | `[-2.593, -95.429, 97.670, 57.670, -9.275, 0.0]` | Read from the user's chosen physical pose today. Use as `fixed_reset_joint_positions` in whatever format ggand0 expects. |
| EE workspace bounds (meters) | `min: [0.057, -0.244, -0.035]`, `max: [0.430, 0.286, 0.248]` | Measured via `lerobot-find-joint-limits` today, +1 cm margin. The user's actual physical workspace. |
| Yellow→EE conversion script | `sim/hilserl/scripts/convert_eval2_to_hilserl_yellow.py` | FK math validated: max EE deltas at 30 fps stay under 20 mm/frame; gripper signal non-trivial. See `outputs/datasets/projet3-hilserl-yellow-v1-converted/conversion_report.md`. |
| Lab checklist | `notes/hilserl_lab_checklist.md` | At-the-robot step-by-step. Adapt to ggand0's CLIs if SWITCH. |
| Design doc | `notes/hilserl_eval2_plan.md` | Plan-eng-review v2 outcomes. Source of truth for WHY we made each decision. |
| Implementation walkthrough (visual) | `notes/visualizations/hilserl_implementation_walkthrough.html` | Self-contained HTML, 8 figures explaining the architecture + state machine + plan. |

### Hardware state (Mac side)

- macOS, base conda Python 3.13 at `/Users/admin/miniforge3/bin/python3.13`
- SO-101 follower at `/dev/tty.usbmodem5B141129871`
- SO-101 leader at `/dev/tty.usbmodem5B141128171`
- Wrist camera: OpenCV index 0, 640×480 @ 30 fps
- Calibration files at `~/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower.json` and `teleoperators/so_leader/so101_leader.json` (includes the 2026-05-17 wrist_roll homing_offset fix at +1588)
- HF auth: logged in as `osammotg1` (has write to `osammotg1/*` repos)

### Parallel track that's STILL running

The 5090 workstation (`ethrc-rl-ws1`, SSH alias `robot_learning_teleop_eduroam_tommaso`)
is running the multi-color dataset conversion for v1.5. See
`notes/HANDOFF_5090_HILSERL_DATASET_CONVERSION.md`. It outputs
`osammotg1/projet3-hilserl-multicolor-v1` on HF.

That output is independent of whether we use ggand0's repo or stock lerobot
on the Mac — it's just a dataset, both stacks can consume it.

## What to NOT do

- ❌ **Don't blindly port everything onto ggand0's repo before evaluating.**
  Step 2 is a real evaluation; if the answer is STAY, we save hours.
- ❌ **Don't modify our `sim/hilserl/configs/*` files** to ggand0's format
  until the verdict is SWITCH. If we switch, write a NEW config file
  alongside; keep ours as the lerobot-main reference.
- ❌ **Don't push to HF as `osammotg1/projet3-hilserl-yellow-v1`** without
  user approval — that name is reserved for the eventual real recording.
  Use `*-smoke` or `*-ggand0test` for any test runs.
- ❌ **Don't modify the lerobot editable install** under
  `/Users/admin/Documents/ETH/M4/Robot Learning /Project S101/lerobot/`.
  If we need to patch it, that's a separate decision the user owns.
- ❌ **Don't re-run plan-eng-review on the same decisions.** D1, D2, D3, D8
  are locked. The CEO/eng review already challenged them.

## What's allowed

- ✅ Cloning ggand0's repo to `~/Desktop/eval-ggand0/` (outside our project)
- ✅ Reading any file in ggand0's repo, the lerobot editable install, our
  scaffold under `sim/hilserl/`
- ✅ Writing a fresh `notes/ggand0_evaluation.md` with the verdict
- ✅ Running ggand0's smoke tests / probe scripts to confirm it works
- ✅ Asking the user (Tom) before doing the actual port

## Acceptance criteria for the evaluation phase

You're done when:

1. `notes/ggand0_evaluation.md` exists with the verdict + reasoning + cost
   estimate + risks
2. You can answer all 4 critical questions with file:line citations
3. You have NOT modified any of our existing scaffolding
4. You have NOT pushed anything to HF
5. You've reported back to the user (Tom) with the verdict + your
   recommendation, and waited for them to choose SWITCH or STAY

## After the verdict (if SWITCH)

Once Tom approves the switch:

1. Create a new branch (e.g. `tom-main-ggand0`) so we can roll back
2. Set up ggand0's stack alongside (don't delete the lerobot editable install)
3. Port the locked decisions + assets above, ADAPTING to ggand0's formats
4. Smoke-test with 2 demos (use `osammotg1/projet3-hilserl-yellow-v1-ggand0smoke`)
5. If smoke passes, record the real 15 demos
6. Continue from `notes/hilserl_lab_checklist.md` Step 4 (crop_dataset_roi)
   onward, substituting ggand0's CLI for our `gym_manipulator` calls

## Provenance + pointers

- This handoff: written 2026-05-17 during the v1 kickoff session
- The current session's plan: `notes/hilserl_eval2_plan.md`
- The current session's lab checklist: `notes/hilserl_lab_checklist.md`
- The current session's git commit: `6a2d220` on `tom-main`
- The error trace that motivated this handoff: see "Current blocker" above
- ggand0's blog post (referenced in the playbook): `ggando.com/blog/so101-hil-serl/`
  (URL may have changed; cross-reference the GitHub repo's README)
- The lerobot HF doc that oversold leader-mode: `huggingface.co/docs/lerobot/hilserl`
- Our existing implementation walkthrough HTML:
  `notes/visualizations/hilserl_implementation_walkthrough.html`
