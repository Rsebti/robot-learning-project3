# HIL-SERL on SO-101 for Eval 2 — implementation plan

> Drafted 2026-05-15. Sources: HF HIL-SERL doc
> (`huggingface.co/docs/lerobot/hilserl`), local
> `notes/visualizations/generate_hilserl_walkthrough.py`, playbook §3 + §T2
> (`notes/so101_robot_learning_playbook.md`), TA spec
> (`notes/project3_rl_final_details.md`), and the SmolVLA work already on
> HF (`osammotg1/projet3-smolvla-eval2-v1*`).
>
> Goal of this doc: lock the architecture *before* the 2-h build session.
> Where it says "decide", that's a stop-gate for the plan-eng-review.

## 1. The goal in one line

Pass Eval 2 — 5 rollouts of *targeted-color* pick-and-place from a 2-cube
clutter, target color and target bowl xyz both given as inputs — with **RL
mandatory**, using HIL-SERL as the RL stage on top of the SmolVLA work we
already have.

## 2. What we already have (asset inventory)

| Asset | Where | Useful for HIL-SERL? |
|---|---|---|
| `osammotg1/projet3-eval2-v1-tom-hugo` (101 ep / 43,289 frames, 6 colors, **bowl fixed at (-15.5, 29.5) cm**) | HF dataset | **Partially.** Wrong action space (joint, not EE), wrong bowl variability (fixed, not randomized). Reusable as raw video for reward-classifier labeling. |
| `osammotg1/projet3-smolvla-eval2-v1-dark-noise-100k` (and 2 sibling SmolVLA finetunes) | HF model | As a **parallel BC baseline** for Eval 2, **not** as HIL-SERL's actor (arch mismatch). |
| `deploy/infer_smolvla.sh` (with interactive color picker, wrist→camera1 rename, leader-driven reset) | repo | Reference for the inference-side plumbing. The leader/follower port + camera-index pattern carries over. |
| `train/launch_smolvla.sh` (uv + accelerate + W&B + push-to-hub on the 5090) | repo | Reference for the training-side plumbing. |
| Bilateral teleop pipeline (SO-101 leader + follower, calibrated, ports known) | hardware + repo | **Yes.** Same hardware drives HIL-SERL teleop and interventions. |
| ACT eval2 sweep (4 policies, no-aug / dark-noise / dark-shadow / dark-noise-100k) | HF models | Optional cross-check baseline; not the HIL-SERL actor. |

## 3. What HIL-SERL needs (from the HF doc) — gap inventory

Each row: required artifact → status → action.

| Required for HIL-SERL | Source / spec | Status | Action |
|---|---|---|---|
| LeRobot `[hilserl]` extra installed on the robot PC | `pip install -e ".[hilserl]"` | **Missing** | Install in the lerobot venv we already use for record/replay. |
| End-effector robot config (`SO100FollowerEndEffector` or SO-101 equivalent) | HF doc; playbook flags LeRobot issue #1387 ("so101 can't work with HIL-SERL — only `so100_follower_end_effector` works out-of-box") | **Likely missing for SO-101** | Probe `lerobot/robots` for an SO-101-EE class. If absent, plan to (a) port from SO-100 or (b) use SO-101 joint config + custom EE wrapper. Decision point. |
| URDF for the SO-101 (for IK / kinematics) | HF doc references `lerobot/model/kinematics.py` | **Have it externally** in `isaac_so_arm101` clone (per CLAUDE.md). Need to confirm LeRobot bundles one. | Grep `lerobot/model/kinematics.py` for the SO-101 URDF path; otherwise vendor one in `sim/` (copy, never modify upstream). |
| End-effector workspace bounds | `lerobot-find-joint-limits` script | **Missing** | Run once at the robot, write bounds to a config file. 10 min task at the robot. |
| Teleop device for interventions (gamepad recommended, SO-101 leader supported) | HF doc | **Have leader; gamepad unknown** | We already have the SO-101 leader — use leader-mode interventions (HF doc explicitly supports `"control_mode": "leader"`). Keyboard fallback for success/fail labeling. |
| Demos recorded **in EE space** (~20–30 trajectories) | HF doc `mode: "record"` config | **Missing** | New recording session. Reuse `teleop/record_eval2.sh` pattern, but the `robot.type` switches to the EE variant. |
| Image crops (`crop_dataset_roi.py`) → 128×128 region of interest | HF doc | **Missing** | One-time interactive crop after first demo batch. ~10 min. |
| Reward classifier (ResNet-10 + MLP head, ~200 pos / ~1000 neg labeled frames, ≥95% acc) | HF doc + HIL-SERL paper §III | **Missing** | Two options: (a) train one from labels on our existing 101 episodes, (b) start with manual gamepad/keyboard reward annotation per HF doc's "optional" note. **Decision point.** |
| Training config JSON (`TrainRLServerPipelineConfig` w/ SAC + `processor` pipeline) | HF doc, examples on HF Hub (`aractingi/lerobot-example-config-files`) | **Missing** | Author one. ~30 min once decisions are locked. |
| Two processes: learner + actor on the same GPU machine | HF doc | **Need GPU at the robot** | Confirm laptop has a usable GPU/MPS *or* run learner on the 5090 over the network. Decision point. |
| W&B account + project for monitoring | `wandb.enable=true` | **Have it** (`tom-gazzini-ethrc/projet3-smolvla` exists; we'll add `projet3-hilserl`) | Reuse. |

## 4. The SmolVLA-as-base question — LOCKED: Option A

**Decision (2026-05-15, post plan-eng-review):** Option A — SmolVLA is the
shipped parallel BC baseline for Eval 2; HIL-SERL is an independent RL
track that trains a fresh SAC policy from new EE-space demos. Submit
whichever scores higher on the 5-rollout matrix.

**Why this is the right call:**
- LeRobot's HIL-SERL implementation hardcodes a SAC actor (ResNet-10 +
  small MLP). SmolVLA (450M, flow-matching action expert) cannot drop
  into that slot without a multi-day fork of `lerobot.policies.sac`.
- We already have `osammotg1/projet3-smolvla-eval2-v1-dark-noise-100k`
  on HF — it's a real, deployable artifact today.
- Two parallel tracks means the worst-case Eval 2 score is whichever
  policy works; we are never empty-handed.

**Options not chosen (deferred to TODOS):**
- **Option B** — SmolVLA generates the HIL-SERL demo buffer. Reconsider
  *only if* SmolVLA's on-arm success rate ≥40% on the bench probe.
- **Option C** — SmolVLA encoder transplanted into a custom SAC actor.
  Research path; logged in TODOS as a stretch goal, not on the critical
  path.

## 5. Decisions to lock in before any code is written

| # | Decision | Recommendation | Why |
|---|---|---|---|
| D1 | Color-conditioning shape | **One HIL-SERL policy per color, ≤6 policies** *or* **single goal-conditioned policy** | Per-color is simpler (no language input to SAC); single goal-conditioned needs feeding a color one-hot into the state vector (extra plumbing). **Recommend per-color for v1**, ~30 ep each at the bowl-fixed position. |
| D2 | Bowl-position randomization | **v1 trains at the fixed bowl pose; goal-conditioning is a v1.5 task once the bowl-OOD gap is measured** | **Revised after HF doc verification (2026-05-17):** `HILSerlProcessorConfig.ObservationConfig` only exposes `add_joint_velocity_to_observation` and `add_current_to_observation`. There is no documented hook to add an arbitrary goal vector (bowl xyz) to the SAC state. Adding it requires either (a) a custom processor step in `lerobot.rl.gym_manipulator` or (b) extending `policy.input_features` to consume a new dataset feature. Both are real plumbing, not the "30-min add" the original plan claimed. Concretely: train v1 at the fixed bowl pose; **Day-3 bowl-OOD probe (still required)** tells us whether goal-conditioning is actually needed. If yes, v1.5 = custom `GoalConditioningProcessorStep` + retrain. |
| D3 | Reward source | **Manual gamepad/keyboard annotation for v1; trained classifier for v2** | HF doc explicitly says the classifier is optional and manual annotation works for the first round. Drops ~30 min of frame labeling from today's kickoff and removes the classifier-overfitting risk on v1 data. Train the classifier between v1 and v2 once we have real failure-mode footage. |
| D4 | Where the learner runs | **5090 over SSH/gRPC, actor on the robot PC** | The robot PC's GPU is unknown (Mac laptop is MPS; 5090 fixed PC may or may not be at the lab). Decoupling learner→5090 / actor→robot via the HF doc's gRPC channel matches the design. Decision blocked on team confirming network latency. |
| D5 | Action space | **6D EE Cartesian twist + discrete gripper DQN** (per HIL-SERL paper) | Matches the HF doc default. Joint-space RL is what killed our archived PPO. |
| D6 | Image input | **Wrist 128×128 cropped + state vector (joint velocity only in v1)** | HF doc default. `add_joint_velocity_to_observation: true`. Bowl xyz dropped from v1 per revised D2 (no documented hook). **Action item:** ask the teammate-with-the-robot whether a USB webcam clamped above the workspace is a 30-min setup. Playbook claims +10–20 pp on SmolVLA finetunes from a top cam; HIL-SERL paper uses wrist+top on all 13 tasks. Decision lives at the lab, not here. |
| D7 | Intervention device | **SO-101 leader (we have it)** with keyboard for s/esc/space | Avoids a gamepad purchase. HF doc explicitly supports `"control_mode": "leader"`. |
| D8 | SmolVLA's role | **Option A** (parallel baseline) | See §4. |

## 6. The 2-hour kickoff scope (today, no robot) — POST-REVIEW

What we can finish without the robot in the room. Step 3 (frame labeling)
dropped per plan-eng-review scope reduction: D3 changed to manual
annotation for v1; classifier training moves to v1.5.

1. **(15 min)** Confirm SO-101 EE-robot support in LeRobot
   - Grep `lerobot.robots` for SO-101 + end-effector classes. If only
     `SO100FollowerEndEffector` exists (per LeRobot issue #1387), add a
     1-day "vendor + calibrate-swap" task to TODOS.md.
   - Same probe for `lerobot/model/kinematics.py` URDF availability for
     SO-101. If absent, plan to vendor one from `isaac_so_arm101`.
2. **(25 min)** Scaffold `sim/hilserl/` directory with:
   - `configs/env_config_so101.json` — record-mode config. Placeholders
     for `end_effector_bounds`, `crop_params_dict` (to be filled at the
     robot). **Set `add_joint_velocity_to_observation: true`. Drop the
     bowl-xyz stub per revised D2.**
   - `configs/train_config_hilserl_so101.json` — training-mode config.
     `policy.temperature_init: 1e-2`, `policy.storage_device: "cuda"`,
     `policy.actor_learner_config.policy_parameters_push_frequency: 1.5`
     (all explicit per HF "Key hyperparameters to tune" section).
   - `configs/_upstream/` — untouched copies of
     `aractingi/lerobot-example-config-files` env + train JSONs, pinned
     to a sha for future diffing (CQ-3).
   - `README.md` — exactly the run-order, copy-pasteable from the HF doc.
     Include `mode: "replay"` as a Day-1 sanity step before training.
3. **(15 min)** `teleop/_common.sh` — extract the shared port / leader /
   camera / cache-cleanup plumbing from `teleop/record_eval2.sh` per
   CQ-2. Then:
4. **(10 min)** `teleop/record_hilserl_demos.sh` — sources `_common.sh`,
   sets only `--robot.type` to the EE variant and the new dataset name.
   No robot call today.
5. **(15 min)** Sanity-check our env config against the HF reference JSON
   diff-by-diff.
6. **(15 min)** `notes/hilserl_lab_checklist.md` — at-the-robot checklist:
   network probe (5090 ↔ robot-PC gRPC latency via `ping`/`iperf3` per
   ARCH-3 / D4), `lerobot-find-joint-limits`, `crop_dataset_roi`, top-cam
   question for teammate (D6), bowl-OOD probe (test review Phase D).
7. **(25 min)** Buffer for the inevitable surprise from step 1.

**Out of scope for the 2-h block (unchanged):**
- Recording new demos (needs robot).
- Training the reward classifier (now v1.5 per D3).
- Running learner+actor (needs robot + lab GPU).
- Solving D4 (learner location) — that's a network/latency probe at the
  lab. The checklist in step 6 covers what to measure.

## 7. End-to-end Eval 2 timeline once we leave the 2-h kickoff — POST-REVIEW

```
Day 0 (today, 2 h, no robot):    plan + scaffold + decisions       ← this doc
Day 1 (at the robot, ~2 h):       network/latency probe (5090 ↔ robot-PC, ARCH-3)
                                  top-cam decision with teammate (D6)
                                  lerobot-find-joint-limits
                                  record 15 EE-space demos / color (HF default)
                                  mode: "replay" sanity-check the demos
                                  crop_dataset_roi
                                  (no reward-classifier labeling — manual @ v1 per D3)
Day 2 (at the robot, ~2.5 h):     HIL-SERL learner + actor run
                                  ~1 h human interventions via SO-101 leader
                                  monitor intervention rate decay in W&B
                                  monitor actor_lag if learner is remote
Day 3 (at the robot, ~45 min):    PRIMARY: 5-rollout eval at trained bowl pose
                                  REQUIRED: 5-rollout bowl-OOD probe (+3 cm) per
                                  test review Phase D — silent-failure mode
                                  detector for TA eval day
```

Per-color budget (if going per-color in D1): multiply by 6, or pick 2 colors
and chain the rest via FSM + a color-token swap in inference (cheaper).

## 8. Top risks and how we test for them early

| Risk | Probe | Mitigation |
|---|---|---|
| LeRobot HIL-SERL still doesn't work on SO-101 in current main (issue #1387 was open in May 2026) | Day 0: grep `lerobot.robots.so101*` for end-effector class; run the HF doc's sample env_config_so100.json with `--robot.type=so101_follower` to see how it dies | Be prepared to port from SO-100 (one-day patch); fall back to Option B (SmolVLA demos → HIL-SERL) if port stalls |
| Reward classifier underfits because cubes look similar to bowl | Day 1: hold out a per-color subset, measure per-class accuracy | Add color filter as a preprocessor; or scripted reward (HSV blob in goal region) |
| Human can't intervene fast enough at 10 Hz | Day 2: first 5 minutes of online run, watch wall-clock | Tune `control_time_s`, switch leader-mode to gamepad if needed |
| Wrist-only image isn't enough for 2-cube clutter | Day 1: visualize cropped wrist view of clutter scenes from the existing dataset | Decide before Day 2 whether to mount a fixed top cam (playbook says +10–20 pp on SmolVLA; same logic for HIL-SERL) |
| Bowl-position OOD at TA eval time (since v1 trains with fixed bowl) | Day 3: 5-rollout eval at the trained bowl pose + 5 at a 3 cm offset | v2 retrain on randomized bowls if v1 fails OOD |

## 9. Plan-eng-review outcomes (2026-05-15)

Resolved (changes folded into the relevant sections above):

1. **D8 SmolVLA base → Option A.** Locked in §4.
2. **D2 bowl xyz → added to v1 state vector.** Updated in §5.
3. **D3 reward source → manual annotation for v1, classifier for v1.5.**
   Updated in §5. Scope reduction: drops the frame-labeling script from
   the 2-h kickoff.
4. **D6 top camera → ask the teammate-with-the-robot at Day 1.** Updated
   in §5.
5. **2-h kickoff → simplified per §6.** Common shell extracted (CQ-2);
   upstream configs vendored (CQ-3); labeling script dropped (D3 change).
6. **Day 3 bowl-OOD probe → required.** Updated in §7.

Deferred to TODOS.md (not on the critical path):

- **Option B** (SmolVLA → HIL-SERL demos via FK conversion) — reconsider
  only if SmolVLA's on-arm success ≥40% on the Day-1 bench probe.
- **Option C** (SmolVLA encoder → SAC actor fork) — stretch / research.
- **Trained reward classifier (v1.5)** — once we have real failure-mode
  footage from the v1 run.
- **Bowl-distribution training data (v2)** — if v1 bowl-OOD probe fails.
- **Top camera mount** — teammate decision; not blocked on us.

Open risks (named for visibility, not blocking the 2-h kickoff):

- **ARCH-2:** SO-101 EE robot class may be missing in current lerobot.
  Today's Step 1 probe gates everything else.
- **ARCH-3 / D4:** Learner-location decision is deferred to Day 1
  network probe. If the lab PC has no CUDA GPU and 5090↔robot-PC gRPC
  latency is bad, we need a contingency.
- **Per-color teleop budget (D1):** 6 colors × 30 demos = 180 demos is
  a large bilateral-teleop commitment. Worth discussing with teammates
  before Day 1.

## 10. Verification against HF doc (2026-05-17)

Re-pulled `huggingface.co/docs/lerobot/hilserl` and audited every CLI / config
claim in this plan. Changes folded above:

- **Misattribution removed:** "SAC actor = ResNet-10 + small MLP" and "REDQ-10
  + UTD=10–20 + 50/50 minibatch" are HIL-SERL *paper* claims, not HF doc claims.
  The LeRobot SAC architecture is configurable via `configuration_sac.py` and
  may differ. Don't quote paper specs as if they were the lerobot defaults.
- **D2 revised:** bowl-xyz in state from v1 **dropped**. HF
  `HILSerlProcessorConfig.ObservationConfig` only exposes
  `add_joint_velocity_to_observation` / `add_current_to_observation` — no hook
  for an arbitrary goal vector. Goal-conditioning is real plumbing, not 30 min.
- **D6 revised** to match the new D2.
- **Demo count** trimmed from "20–30 / color" to "15 / color" (HF's documented
  starting point in the basic example).
- **`temperature_init: 1e-2` and `storage_device: "cuda"`** added to the §6
  scaffolding step — HF explicitly calls these out as high-impact knobs we
  were not setting.
- **`mode: "replay"`** added to Day-1 to sanity-check recorded demos before
  training.
- **`terminate_on_success: false`** logged for v1.5 reward-classifier data
  collection (HF flagged with "**Important**").
- **Reward classifier multi-camera caveat:** HF's example config uses two
  cameras. With wrist-only, we need a one-camera `input_features` block —
  log this for the v1.5 classifier work.

Items confirmed unchanged: install command, `lerobot-find-joint-limits` CLI,
`crop_dataset_roi` CLI, `learner` / `actor` module paths, `helper2424/resnet10`
classifier backbone, 128×128 default resolution, manual reward fallback for
v1, `policy_parameters_push_frequency: 1.5`, SO-101 leader `control_mode:
"leader"`, URDF path requirement, 10 Hz control loop, ARCH-2 SO-101 EE class
gap (HF doc still only names `SO100FollowerEndEffector`).

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | not run |
| Codex Review | `/codex review` | Independent 2nd opinion | 0 | — | not run |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 2 | issues_open | Run 1: 5 arch + 3 CQ + 6 test gaps + 2 perf; D8 resolved. Run 2 (HF doc verification): 3 misattributions removed, D2 revised (no goal-conditioning hook in HF), 4 explicit config additions, 3 new gaps logged for v1.5 |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | n/a (no UI) |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | not run |

- **UNRESOLVED:** 1 (D4 learner location — deferred to Day 1 network probe)
- **VERDICT:** ENG REVIEW COMPLETE; ready to begin 2-h kickoff under §6.
