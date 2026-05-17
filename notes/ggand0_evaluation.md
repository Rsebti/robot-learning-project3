# `ggand0/hil-serl-so101` evaluation — verdict

**Date:** 2026-05-17
**Evaluator:** Claude Code session on Mac (tom-main, post-`6a2d220`)
**Sandbox:** `~/Desktop/eval-ggand0/{hil-serl-so101, lerobot}` (NOT in our project)
**Goal:** decide whether to switch HIL-SERL v1 work from stock lerobot 0.5.2 to ggand0's stack.

---

## TL;DR — Verdict: **SWITCH**

Recommendation: switch to `ggand0/hil-serl-so101` + the `ggand0/lerobot @ feat/hil-serl` fork for v1. This is the only known stack that actually runs HIL-SERL on an SO-101 leader arm end-to-end on real hardware, and it has 98 commits worth of real-world fixes that we would otherwise re-discover ourselves over weeks.

The catch: it pins to **lerobot 0.3.2** (not 0.5.2 like ours), so it cannot share the existing editable install. We'd run the two stacks side-by-side, not merged.

Migration cost estimate: **6–10 hours of focused plumbing** to adapt our locked decisions and assets to ggand0's formats and re-record/re-convert demos with their schema. No re-thinking of D1/D2/D3/D8.

---

## The 4 critical questions — answered with citations

All citations are paths inside `~/Desktop/eval-ggand0/`.

### Q1. Does ggand0 solve the leader+events problem? — **YES, by bypassing it.**

They do NOT implement `get_teleop_events()` on `SOLeader`. Instead, they treat the leader purely as an action source and add their own keyboard event channel inside a custom env wrapper.

- A `GearedLeaderControlWrapper` is selected when `wrapper.control_mode == "leader"` — see `lerobot/src/lerobot/scripts/rl/gym_manipulator.py:2972-2985`. (Also `keyboard_ee`, `gamepad`, and `leader_automatic` modes exist.)
- The wrapper runs **two** keyboard input paths in parallel:
  1. A pynput global background listener (`keyboard.Listener(on_press=on_press)` started at `gym_manipulator.py:2082-2084`) — works even when no window is focused.
  2. A cv2.waitKey poll inside the render loop (`gym_manipulator.py:2092-2123`) — works when the OpenCV camera window is focused.
- Keys mapped:
  - `ESC` → `episode_end`
  - `9` → `episode_success`
  - Left arrow → `rerecord_episode`
  - `7` → intervention toggle
  - `-` → pause-after-episode toggle
- The fork is based on lerobot **0.3.2** (`lerobot/pyproject.toml:version = "0.3.2"`). The `HasTeleopEvents` protocol and the `_check_teleop_with_events` check that block our 0.5.2 install **don't exist yet** in their codebase — `grep get_teleop_events|HasTeleopEvents|_check_teleop_with_events` on their src tree returns zero hits.

So they sidestep the entire events-protocol architecture we're hitting on 0.5.2. Their pattern is the standard Berkeley HIL-SERL pattern: leader = continuous action source, keyboard = discrete events, plumbed together inside a single env wrapper.

### Q2. What lerobot version does it pin to? — **A FORK based on 0.3.2.**

- `hil-serl-so101/pyproject.toml:52` → `lerobot = { path = "../lerobot", editable = true }` — sibling-directory editable install of their fork, not PyPI.
- `hil-serl-so101/README.md:19-28` → instructions explicitly tell you to `git clone https://github.com/ggand0/lerobot && git checkout feat/hil-serl && pip install -e ".[hilserl]"`.
- The fork has **98 commits** ahead of upstream (`git log main..HEAD` on `feat/hil-serl`) including:
  - Custom robot class `so101_follower_end_effector` (referenced as `robot.type` in every config)
  - MuJoCo-based FK/IK (`mujoco_model_path` field, `placo` dependency in pyproject)
  - `GearedLeaderControlWrapper`, `KeyboardControlWrapper`, etc.
  - Numerous SAC / reward-classifier / USB-stability fixes (epsilon-greedy, gripper bounds, sync_read retries, atexit torque-off, etc.)
- Fork uses **ROCm** wheels by default (`pyproject.toml` `[tool.uv.sources]` pulls from `pytorch-rocm` index) — they ran on AMD GPU. We'll override to CUDA on the 5090 and to MPS or CPU on the Mac.

**Implication:** their fork CANNOT coexist with our `/Users/admin/Documents/ETH/M4/Robot Learning /Project S101/lerobot` (0.5.2) install in the same Python environment. We'd install ggand0's stack into a separate `uv` venv (per their README), and keep our 0.5.2 install untouched for SmolVLA / sanity work.

### Q3. What config schema does it use? — **JSON, flat (lerobot 0.3.2 schema). Different from ours.**

- All configs are JSON in `hil-serl-so101/configs/` (also one `deploy.yaml` for hydra-driven deploy).
- Top-level keys are `type`, `task`, `fps`, `mode`, `repo_id`, `features`, `robot`, `teleop`, `wrapper` — sibling-flat, **NOT** nested under an `env.processor.*` like our `sim/hilserl/configs/env_config_so101.json` (which uses the 0.5.2 processor schema with nested `env.processor.{observation, gripper, reset, inverse_kinematics, …}`).
- Equivalent of our `processor.inverse_kinematics.end_effector_bounds`: **yes**, exists as `robot.end_effector_bounds` — see `configs/grasp_only_record_angled_10ep_config.json:55-58`. Same `{min: [x,y,z], max: [x,y,z]}` shape. Our measured workspace `min: [0.057, -0.244, -0.035] max: [0.430, 0.286, 0.248]` drops in directly into that field.
- Equivalent of our `inverse_kinematics.urdf_path`: **NO**. They use a **MuJoCo XML** (`mujoco_model_path: ".../so101/lift_cube.xml"`) instead of a URDF. The MuJoCo XML lives in the sibling repo `pick-101`, not in `hil-serl-so101`. So if we switch, we either (a) clone `pick-101` and use their XML, or (b) write a small URDF→MJCF conversion / point them at our URDF if their code supports it (need to check `so101_follower_end_effector` source).
- `fixed_reset_joint_positions`: present as `wrapper.fixed_reset_joint_positions` — same field, same units. Our value `[-2.593, -95.429, 97.670, 57.670, -9.275, 0.0]` drops in. They also have `wrapper.ik_reset_ee_pos: [0.25, 0.0, 0.07]` for IK-based resets, which is a nice alternative if a joint-pose reset misbehaves.
- They also have config-level reward-classifier and SAC hyperparams in `configs/*_hilserl_train_config.json` — see Q4.

### Q4. Action / observation space + control rate? — **EE-delta (4-D) @ 10 Hz, RGB wrist cam.**

From `configs/grasp_only_hilserl_train_config.json`:

| Field | Value |
|---|---|
| `env.fps` | **10** (line 125) |
| `observation.images.gripper_cam` | `[3, 128, 128]` after crop `[0, 80, 480, 480]` + resize `[128, 128]` (lines 134-138, 202-205) |
| `observation.state` | `[18]` — `add_full_proprioception: true` + leader/follower positions, see `dataset_stats.observation.state.min/max` for the 18-D layout (lines 38-41, 57-59) |
| `action` | `[4]` — 3D EE delta (x, y, z) + 1D gripper (lines 43-47) |
| `policy.type` | **SAC** with ResNet-10 vision encoder (`helper2424/resnet10`), `discount: 0.97`, `utd_ratio: 20`, `temperature_init: 0.01` (lines 67-87) |
| Action bounds | `[-1, 1]^4` (line 62-64), scaled by `robot.action_scale: 0.02` → 2 cm step in EE space per axis |
| `wrapper.control_mode` | `"leader"` — uses `GearedLeaderControlWrapper` (line 194) |

**Implications for our converted demos:**

- Our `convert_eval2_to_hilserl_yellow.py` script already produces an EE-delta-style action stream at 30 fps (per `outputs/datasets/.../conversion_report.md`). ggand0 expects **10 Hz**. We need to either (a) downsample to 10 fps during re-conversion, or (b) re-record at 10 fps. Probably (a) is cheapest — we keep our recording at 30 fps and just stride-3 during the conversion.
- Our action is currently 7-D-ish (depending on the conversion). ggand0 wants exactly **4-D**: `[dx, dy, dz, gripper]`. The conversion script needs to drop wrist-roll/wrist-flex/elbow info (the SAC policy learns those implicitly via IK) and quantize gripper to a single channel.
- The 18-D state vector in ggand0 includes the leader proprioception alongside the follower — this is unusual but consistent with the leader-as-intervention-source pattern. It's auto-computed by `wrapper.add_full_proprioception: true`, so we don't need to construct it manually.
- Image: 128×128 RGB after a center-y crop to 480×480 then resize. We feed 640×480 raw, and the wrapper handles crop/resize.

---

## Why SWITCH (the math)

| Cost / benefit | STAY (lerobot 0.5.2) | SWITCH (ggand0) |
|---|---|---|
| Leader+events plumbing | We write & maintain a patch for 0.5.2 (~1 day; risks breaking with each lerobot upgrade) | Already solved (`GearedLeaderControlWrapper` + dual keyboard listener) |
| MuJoCo-based IK + locked-joint handling | We re-engineer it (placo + manual joint-limit clamping) | Already shipped with 98 commits of fixes |
| SAC training loop | Stock 0.5.2 (newer, but never tested on SO-101 end-to-end by anyone we know of) | 70 % real-robot grasp success after 3 hrs of HIL training — only known reproduction |
| Reward classifier pipeline | We build from scratch | `scripts/train_reward_classifier.py` + live preview ship with the repo |
| Demo re-conversion to their format | n/a | Stride-3 our 30 fps output to 10 fps + 4-D action — small script change |
| Config schema diff | n/a | Write a NEW `configs/yellow_v1_*.json` in their flat schema; keep ours as the 0.5.2 reference |
| Lerobot version isolation | Keep our 0.5.2 editable install | Sibling `~/Desktop/eval-ggand0/lerobot` editable install in a separate `uv` venv — non-interfering |
| Risk of "yet another fork drift" | Low — we own the patch | Medium — fork is 1-person-maintained, may go stale; mitigated by treating it as frozen reference |

The dominating factor: **only ggand0 has demonstrated 70 % success on an SO-101 leader arm with real-world HIL-SERL.** Our v1 timeline is days, not weeks. Re-engineering their fixes from 0.5.2 main is the slowest path.

---

## Migration cost estimate (if SWITCH)

| Step | Time | Notes |
|---|---|---|
| Branch off `tom-main` → `tom-main-ggand0` | 5 min | Roll-back path preserved |
| Clone fork into `~/Desktop/eval-ggand0` (already done) + `uv sync` in a new venv | 30 min | Will need PyTorch CUDA/MPS override vs their ROCm pin |
| Get `pick-101` MuJoCo XML or test if their `so101_follower_end_effector` can accept our URDF | 30–60 min | Likely just clone `pick-101` and reuse `models/so101/lift_cube.xml` |
| Write `configs/yellow_v1_record.json` + `yellow_v1_train.json` in their schema | 1–2 hr | Translate locked decisions (D1: yellow only, D2: bowl @ (-15.5, 29.5) cm, fixed_reset_joint_positions, end_effector_bounds, action_scale, ...) |
| Adapt our `convert_eval2_to_hilserl_yellow.py` → output 4-D action @ 10 fps | 1–2 hr | Stride-3 + drop unused dims; re-run conversion |
| Smoke test: record 2 episodes via `gym_manipulator --mode=record`, push to `osammotg1/projet3-hilserl-yellow-v1-ggand0smoke` | 1 hr | Confirms cameras + ports + leader + keyboard work on the Mac |
| Update `notes/hilserl_lab_checklist.md` for ggand0 CLI | 30 min | Substitute their commands for our `gym_manipulator` invocations |
| Buffer for ROCm→MPS/CUDA wheel surprises | 1–2 hr | The fork is ROCm-only by default; Mac MPS support is untested |
| **Total** | **6–10 hr** | One focused session |

---

## Top 3 risks

1. **MuJoCo XML dependency drag.** Their robot class loads a MuJoCo XML for IK. If `so101_follower_end_effector` does NOT accept a URDF fallback, we must clone `pick-101` and may inherit their cube/table geometry as well — which is fine for IK but adds another sibling repo to maintain. **Mitigation:** read `so101_follower_end_effector` source in the fork early in the port (15 min) to confirm URDF support.

2. **MPS / Mac compatibility untested.** Their stack runs on an AMD ROCm Linux box. The pyproject pins `pytorch-rocm`. SAC training is fine on CPU/MPS for a smoke recording, but actual SAC online training on the Mac is uncertain. **Mitigation:** record demos on the Mac, push the offline buffer to HF, then run the actual SAC actor/learner on the 5090 workstation (where the multi-color conversion already runs). The Mac becomes a recording-only station.

3. **Lerobot 0.3.2 ↔ 0.5.2 dataset format drift.** The HF dataset format changed between 0.3.2 and 0.5.2 (stats files, episode index files, video backend). Our existing `osammotg1/projet3-hilserl-yellow-v1-converted` dataset was produced under 0.5.2. ggand0's stack may not load it. **Mitigation:** re-run the converter inside the ggand0 venv (which has lerobot 0.3.2 imported) — this regenerates stats/episodes in the format the 0.3.2 loader expects. Cost is already counted in the 6–10 hr estimate.

---

## What does NOT change if we SWITCH

The locked decisions and most assets carry over unchanged:

- **D1 single color (yellow)**, **D2 fixed bowl pose (-15.5, 29.5) cm**, **D3 manual keyboard reward**, **D8 Option A (SmolVLA parallel baseline)** — all still valid.
- Home pose `[-2.593, -95.429, 97.670, 57.670, -9.275, 0.0]` → drops into `wrapper.fixed_reset_joint_positions`.
- EE workspace bounds `min: [0.057, -0.244, -0.035] max: [0.430, 0.286, 0.248]` → drops into `robot.end_effector_bounds`.
- 5090 multi-color conversion (`notes/HANDOFF_5090_HILSERL_DATASET_CONVERSION.md`) — independent, outputs a dataset that we can re-convert to 10 fps / 4-D for ggand0.
- HF auth, SO-101 ports, camera index, calibration files — all the Mac-side hardware setup is identical.

---

## Recommended next move (waiting for Tom)

1. Tom confirms SWITCH (or pushes back).
2. Open `tom-main-ggand0` branch.
3. First 15 min of porting: read `so101_follower_end_effector` source in the fork to settle the URDF-vs-MJCF question.
4. Run the migration steps from the cost table above.
5. Smoke-test with 2 demos before committing to the full 15-demo recording.

If Tom prefers STAY, the alternative is writing a `SOLeaderWithEvents` shim in our scaffold that wraps the existing `SOLeader` with a pynput background listener and exposes `get_teleop_events()` to satisfy the 0.5.2 processor's check. Estimated cost: 4–6 hours plus the cost of re-discovering whatever else ggand0 patched in their 98 commits. Not recommended given v1 timeline.

---

## Provenance

- Files inspected:
  - `~/Desktop/eval-ggand0/hil-serl-so101/{README.md, pyproject.toml, configs/grasp_only_record_angled_10ep_config.json, configs/grasp_only_hilserl_train_config.json, docs/SIM2REAL.md, docs/DATASETS.md}`
  - `~/Desktop/eval-ggand0/lerobot/{pyproject.toml, src/lerobot/__init__.py, src/lerobot/scripts/rl/gym_manipulator.py (lines 2080-2300, 2950-3000)}`
  - `git log main..feat/hil-serl --oneline` (98 commits)
  - Our `sim/hilserl/configs/env_config_so101.json` for schema diff
- No file in our repo was modified during this evaluation. No HF push. No lerobot install touched.
