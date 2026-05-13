# SO-101 Robot Learning Playbook for ETH Spring 2026

> Markdown transcription of *"SO-101 Robot Learning Playbook: ETH Spring 2026
> Strategy and Recommendations"* (source PDF at project root). Faithful to
> the original; citations preserved as inline mentions. Use as the strategy
> source-of-truth alongside `notes/project3_rl_final_details.md` (TA spec).

---

## Bottom line up front

For **Task 1** (single-block pick-and-place), train **ACT** on ~80–120 teleop
demos — community SO-101 reproductions hit **70–90% success in ~4 h on a 12 GB
GPU**, and the SmolVLA paper measures ACT at **48–70% on SO-100/SO-101** vs
SmolVLA-base at 78–90%.

For **Tasks 2 and 3** (RL-required, color-conditioned and sequential), the
highest-ROI path is **BC primitive (ACT or SmolVLA-finetune) wrapped in a
scripted FSM, then HIL-SERL fine-tune** using LeRobot's built-in `hilserl`
pipeline (RLPD core, 50/50 demo/online replay, 1–2.5 h wall-clock per task on
real Franka in the original paper).

**Avoid:** PPO-from-pixels-from-scratch, 7B VLAs with <200 demos, Diffusion
Policy with <50 demos in clutter, and sim-to-real on Isaac Lab unless a
teammate is fluent in it — the RTX 5070 has unresolved Blackwell sm_120 bugs
(Isaac Lab issues #2141, #2483, #2652, #4951) and you should burn Brev H100
credits instead.

The single biggest predictor of success in the LeRobot community is **data
quality and camera-pose consistency, not model size.**

---

## 1. TL;DR table

| Task | Recommended primary | Recommended fallback | Expected success | Demos | Train cost | Key references |
|------|---------------------|----------------------|------------------|-------|------------|----------------|
| **T1** Single block pick-and-place, randomized positions (BC allowed) | **ACT** (LeRobot built-in) | SmolVLA-base finetune | 70–90% in-dist; 40–75% OOD | 50–125 | ~3 h on H100 (~$6 of Brev); ~4 h RTX 3080 12 GB locally | Karkada blog (70%/10 ep); Sherry Chen blog (90% in-dist 125 ep); SmolVLA paper Table 3 |
| **T2** Color-targeted pick from 2-block clutter (RL required) | **ACT/SmolVLA primitive + HIL-SERL finetune** with language/color FiLM conditioning | RLPD on top of BC, no human-in-loop | 80–100% per primitive after HIL-SERL; ~50% with BC alone in clutter | ~30 BC demos + ~20 demos for RLPD buffer + 1–2 h online interventions | ~$10–20 Brev H100 + 1–2 h on real arm | HIL-SERL paper §III; LeRobot `hilserl` doc; ggando SO-101 reproduction (~70%) |
| **T3** Sequential 3-step / 4 blocks (RL required) | **Per-primitive ACT/SmolVLA finetune + Python FSM**, then HIL-SERL on the chained primitive | π0 / SmolVLA language-conditioned end-to-end | 60–85% end-to-end (assuming 90%/step) | 80–150 demos for one parameterized primitive | ~$30 Brev H100 + 2–3 h real-arm finetune | SmolVLA Table 3 multi-task; HIL-SERL chained tasks |
| **Bonus** Long-horizon | SmolVLA with explicit language sub-goals | π0-FAST | Untested at this scale | — | — | SmolVLA paper §5; OpenPI issue #692 |

---

## 2. Per-task deep dives

### Task 1 — single-object pick-and-place (BC allowed)

**Ranking, with primary numbers:**

1. **ACT** (Zhao et al. 2023, arXiv:2304.13705). The HuggingFace LeRobot doc
   (`huggingface.co/docs/lerobot/act`) states ACT *"often achieves high success
   rates with just 50 demonstrations"* and *"trains in a few hours on a single
   GPU"* with ~80M params (52M in the LeRobot default config per Sherry Chen).
   The strongest community SO-101 ACT reproduction is **Sherry Chen's
   `so101_bench` blog**: with 50 episodes ACT got **60% in-distribution / 10%
   OOD**; with 125 episodes (25 demos × 5 bins) **90% in-dist / 75% OOD**, ~4 h
   on an RTX 3080 12 GB at 100k steps. Karkada's reproduction reports **70%
   (7/10)** with 50 demos + 4 cameras at 30 FPS. The SmolVLA paper itself
   benchmarks ACT-from-scratch at **48.3% avg on SO-100 multitask** and
   **70% in-dist / 40% OOD on SO-101 pick-place**. Reference implementations
   to clone: `huggingface/lerobot` policy registry; `sherrychen1120/so101_bench`;
   Trelis substack ACT-on-SO-101 tutorial.

2. **SmolVLA-base finetune** (Shukor et al. 2026, arXiv:2506.01844). Beats ACT
   on the same data when (a) cameras are named consistently with the SmolVLA
   pretraining set (top + wrist), (b) the prompt language matches, (c) you
   have ~50 success-only episodes. Paper numbers: **SmolVLA 78.3% vs ACT 48.3%
   (SO-100 multitask); 90% in-dist / 50% OOD on SO-101 pick-place** vs ACT
   70/40. Recipe: `--policy.path=lerobot/smolvla_base --steps=20000
   --batch_size=64`, ~4 h on a single A100. **Caveat:** community
   reproductions vary wildly: Habuda's hackathon top-30 team got better
   results with SmolVLA than other models with **only 8 episodes**
   (camera-fixed); Sawane (Correll Lab) got **0% with 60 episodes** on SO-100;
   Kamath's Eindhoven team failed with 50 episodes. LeRobot issue #2915
   documents 0% with 120 SO-101 episodes. Sensitivity to data hygiene is high.
   References: `huggingface.co/blog/smolvla`, `huggingface.co/docs/lerobot/smolvla`,
   `lerobot/smolvla_base`.

3. **Diffusion Policy** (Chi et al. 2023). No published SO-100/SO-101
   success-rate number was found in either LeRobot docs or community blogs
   surveyed (LeRobot's only DP card is for `pusht`, 65.4% / ~5 h H100, 200k
   steps, batch 64). DP3 paper explicitly states DP *"necessitates 100 to 200
   human-collected demonstrations for each real-world task"* — too
   data-hungry for a 50-demo budget.

4. **VQ-BeT, TDMPC.** Supported in LeRobot (`--policy.type=vqbet`) but **zero
   documented SO-100/101 community results**. VQ-BeT shines on multi-modal
   demonstrations (multiple operators, multiple strategies); for a single-mode
   pick-and-place its advantage over ACT is marginal.

5. **Simple BC/MLP baseline.** HIL-SERL paper Table 1c: BC plateaus at **47%
   on object-flipping with 20 demos and 46% with 200 demos** — essentially
   flat. This is your floor.

**Recommended pick: ACT.** Lowest risk, fastest iteration, well-documented on
SO-101 specifically. Plan 80–120 demos per the SVRC heuristic (*"train a
baseline ACT every 25 episodes, stop when the success curve flattens —
typically 80–150"*). Reference implementations to clone: `sherrychen1120/so101_bench`,
the LeRobot `il_robots` notebook, Trelis's ACT-on-SO-101 substack. Run a
SmolVLA finetune in parallel on Brev (4 h ≈ $8) as a free-with-budget upside
bet.

---

### Task 2 — targeted-color pick from 2-block clutter (RL required)

**Ranking of RL-with-demos methods, with primary numbers:**

1. **HIL-SERL** (Luo et al. 2024, arXiv:2410.21845). Paper reports **100%
   success on all 13 tasks in 1–2.5 h wall-clock** on a Franka with single
   RTX 4090: 1 h IKEA Top, 1.25 h Cable Clip / Jenga / Object Flip, 1.5 h RAM,
   2 h SSD / Dashboard, 2.5 h USB / Handover. Gripper handled by a separate
   DQN with discrete action set. Vision: ResNet-10 pretrained (frozen) + small
   MLP. Reward: trained classifier from ~200 positives + ~1000 negatives
   (>95% acc). Demos: **20–30 + ongoing human interventions via gamepad**
   (interventions go to *both* demo and RL buffers). RLPD core: UTD up to 20,
   10-Q ensemble (subset of 2), critic LayerNorm, 256-wide 2-layer MLPs, lr
   3e-4, batch 256 (50/50 split). LeRobot has a first-class port at
   `huggingface.co/docs/lerobot/hilserl`. **SO-101 caveats:** issue #1387
   (*"so101 can't work with HIL-SERL"*) — only `so100_follower_end_effector`
   works out-of-box; ggando's reproduction (`ggando.com/blog/so101-hil-serl/`)
   reached **~70% on grasp-only after three weeks of fixes** (MuJoCo-FK
   replacement, state caching bug, lighting); Indraneel Patil's reproduction
   concludes *"performance is similar to IL with the same wall-clock"*. So
   expect to do real engineering, but the paper recipe and LeRobot port make
   this the most tractable RL pipeline for our setup.

2. **RLPD** (Ball et al. ICML 2023, arXiv:2302.02948). The mathematical
   backbone of SERL/HIL-SERL. **No BC pretrain, no BC regularizer.** Symmetric
   sampling: every minibatch is 50% from offline demos, 50% from online
   buffer. Critic LayerNorm + 10-Q ensemble + UTD=10 (vision) or 20 (state).
   On D4RL Adroit Door **2.5× over IQL+FT in ~10k online samples**. Code:
   `ikostrikov/rlpd` (JAX). Use this if you don't want a human in the loop —
   but expect more sample inefficiency than HIL-SERL.

3. **IBRL** (Hu et al. RSS 2024, arXiv:2311.02198). Standalone BC policy
   μ_ψ, then online TD3 with action selection a* = argmax over {a_BC, a_RL}
   under Q', and TD target uses the same argmax. **Beats RLPD by ~6.4× on
   PickPlaceCan**. Real Franka: Lift, Drawer, Cloth-Hang in ~2 h. Code:
   `hengyuan-hu/ibrl`. **No SO-101 reproduction known.** Strong if you already
   have a high-quality BC policy from teleop — exactly your situation.

4. **RFCL** (Tao et al. ICLR 2024, arXiv:2405.03379). Reverse curriculum:
   state-resets to demo states, shrinking horizon, then forward curriculum.
   **1–5 demos suffice on MS2 PegInsertSide** (<60 min on RTX 4090). Requires
   resettable env — sim only, or instrumented real. Best for *"train in Isaac
   Lab, deploy via sim-to-real"* path.

5. **ROT** (Haldar et al. 2022, arXiv:2206.15469). OT-based reward + adaptive
   BC regularization. Real xArm: **<1 h, 90.1% avg from 1 demo**. Underused
   but well-documented.

6. **DAPG, AWAC/IQL→online, Residual RL.** Historical baselines. DAPG (NPG +
   decaying demo gradient) and AWAC/IQL underperform RLPD/IBRL in modern
   comparisons (RLPD is **2.5× faster** than IQL+FT on Adroit Door). Use
   Residual RL only if you have a strong scripted base controller.

**Mathematical role of the BC checkpoint per method (precise):**

- **SERL/RLPD:** No BC actor. Demos seed an offline replay buffer; minibatches
  drawn 50/50 offline/online. The only "use" is data sampling.
- **HIL-SERL:** Same RLPD core + a *human-intervention buffer*. Intervention
  transitions go to *both* demo and RL buffers; autonomous transitions only to
  RL buffer. Gripper trained as separate DQN MDP.
- **IBRL:** Independent BC policy. (i) action selection: a* = argmax_{a∈{a_BC,a_RL}}
  Q'(s,a); (ii) TD target uses max over the same set. Demos pre-fill replay
  *without* oversampling.
- **RFCL:** Demos provide environment *states* for resets, not BC weights.
  Stage 1 reverse curriculum from demo end-states; Stage 2 forward curriculum
  on initial-state distribution.
- **ROT:** BC pretrains the actor; online loss = RL term + λ(t)·BC term where
  λ(t) decays whenever Q(π_RL) > Q(π_BC). Reward signal is OT distance between
  rollout features and demo features.
- **DAPG:** BC pretrain to init π. Augmented gradient g = g_NPG + λ_0·λ_t^k ·
  Σ ∇log(π(a|s))·A_w(s,a) over demos.
- **AWAC/IQL→online:** Offline RL warmup on demos (advantage-weighted
  regression / expectile-V), then continue same loss online. BC is implicit
  via AWR.
- **Residual RL:** No BC. a = π_hand(s) + π_θ(s); train π_θ end-to-end.

**Recommended pick for Task 2: ACT (or SmolVLA) primitive trained per-color →
HIL-SERL finetune via LeRobot.** Condition the primitive on color via either
(a) language token to SmolVLA, or (b) a one-hot color input fed via FiLM into
ACT's transformer encoder. Reference implementations:
`huggingface/lerobot/docs/source/hilserl.mdx`, `rail-berkeley/hil-serl`
(JAX original), `ggando.com/blog/so101-hil-serl/` (concrete fixes for SO-101
in LeRobot v0.4.1).

---

### Task 3 — sequential 3-step pick-and-place with 4 blocks (RL required)

**The candidate space and why most lose:**

- **Hierarchical RL / options (MAPLE, MoMaRT):** Sim-only published
  demonstrations; **no real low-cost-arm reproduction found.** Skip.
- **Language-conditioned long-horizon (BC-Z, RT-1, RT-2):** Operate at
  industrial scale — BC-Z used 25k robot demos + 18k human videos for **44%
  on 24 unseen tasks**; RT-1 used 130k episodes / 700+ tasks / 13 robots / 17
  months. Not reproducible at academic scale. Cite as motivation.
- **Goal-conditioned + HER:** Effective in sim with millions of steps;
  impractical from scratch on real hardware.
- **VLA end-to-end (π0, SmolVLA, OpenVLA-OFT):** π0's Trossen single-arm pick
  result (OpenPI issue #692) achieved *"near 100%"* with 300 trajectories on a
  6×12 cm area; π0.5 **failed** on the same task. SmolVLA paper benchmarks the
  SO-100 *multitask* suite (pick-place + stacking + sorting) at **78.3% avg
  with ~50 demos × 3 tasks**. OpenVLA-OFT (arXiv:2502.19645) reaches **97.1%
  on LIBERO** but needs **8× A100/H100 80 GB for 50–150k steps, 1–2 days** —
  that alone burns ~$200–400 of Brev. Realistic only if you keep the model
  frozen and add LoRA.
- **BC primitive + scripted FSM + HIL-SERL refinement:** This is what the
  LeRobot community actually does for sequential tasks. Train ONE
  parameterized primitive ("pick block at pose A, place at pose B"), call it
  3× from a Python state machine that tracks remaining blocks, refine
  end-to-end with HIL-SERL if needed.

**Recommended pick: BC primitive + scripted FSM, with optional HIL-SERL.**
Specifically:

1. Collect 80–150 demos of the primitive across all relevant block initial
   positions.
2. Train ACT (≈3 h on H100, ~$6) **and** SmolVLA finetune (≈4 h on H100, ~$8)
   in parallel; pick the higher-success per-primitive policy by per-step eval.
3. Wrap in a Python FSM that enumerates remaining blocks and chooses the next.
4. If end-to-end success < 70% (90%³ ≈ 73%), run HIL-SERL on the chained
   behavior with ~20 BC episodes + 1–2 h gamepad interventions. Reference
   implementations: SmolVLA's own SO-100 multitask recipe
   (`lerobot/svla_so100_pickplace`, `_stacking`, `_sorting` datasets are
   publicly sequenced this way); Pranav Saroha's #2-world LeRobot Hackathon
   project (bimanual t-shirt fold = 4-step sequential, used SmolVLA + ACT in
   parallel on Lightning AI H100, won with ~70–85% success).

---

## 3. The BC→RL pipeline: deep comparison and recommendation

Treat this as the operational heart of Tasks 2 and 3. The published comparison
space is:

| Method | BC role | Replay | Demos | Env steps | Wall-clock real | Final SR | Code |
|--------|---------|--------|-------|-----------|-----------------|----------|------|
| **SERL** | None — demos seed buffer | 50/50 symmetric, UTD≫1, LayerNorm, 10-Q | 20 | 10–25 k | 20 min PCB / 31 min cable / 105 min reloc | ≈100% | `rail-berkeley/serl` |
| **HIL-SERL** | Demos + human-intervention buffer | 50/50 symmetric + intervention buffer | 20–30 + ongoing | wall-clock-bounded | 1–2.5 h | **100% on 13 tasks** | `rail-berkeley/hil-serl`, LeRobot |
| **RLPD** | None (demos are data) | 50/50 symmetric, UTD 10–20, 10-Q LayerNorm | varies | ~10 k Adroit Door | n/a real | SoTA D4RL/Adroit | `ikostrikov/rlpd` |
| **IBRL** | Standalone BC; argmax-Q over {a_BC, a_RL} | Pre-fill, no oversample; TD3 backbone | 1–10 sim, ~20 real | ≤500 k Robomimic | ~2 h Franka | Beats RLPD 6.4× on PickPlaceCan | `hengyuan-hu/ibrl` |
| **RFCL** | Uses demo *states* for resets | RLPD core + reverse curriculum | 1–5 | <10 min MS2 PickCube | sim only | Solves PegInsertSide / PlugCharger | StoneT2000/RFCL |
| **ROT** | BC pretrain + decaying BC reg + OT reward | DrQ-v2 + replay | 1 | <1 h xArm | 90.1% avg | 7.8× faster than baselines | siddhanthaldar/ROT |
| **DAPG** | BC pretrain + decaying demo gradient | On-policy NPG | 25 | 1–10 M sim | sim mostly | Solves Adroit | aravindr93/DAPG |
| **AWAC/IQL→online** | Offline RL on demos, continue online | Single replay | varies | 25–100 k AntMaze | n/a | RLPD beats it 2.5× | `ikostrikov/rlpd` |

**Concrete hyperparameters for the top three** (quoted from primary sources):

- **RLPD:** 2-layer MLP 256 wide actor & critic; 10-critic ensemble, subset of
  2 per update (REDQ); UTD=20 state / 10 vision; batch 256 (128 offline + 128
  online); lr 3e-4 Adam; γ=0.99; τ=0.005; LayerNorm on every critic hidden
  layer; SAC auto-α; image aug = random shift (DrQ).
- **HIL-SERL:** ResNet-10 (frozen, ImageNet) + MLP; 128×128 images; 10 Hz
  control; 20–30 demo trajectories; reward classifier on ~200 pos / ~1000
  neg, ResNet-10 + MLP, >95% acc; 6D Cartesian twist + discrete gripper DQN;
  impedance ctrl at 1 kHz with Δ-clip; single RTX 4090; same RLPD core.
- **IBRL:** TD3 backbone, two Q-heads; BC is ResNet-18 + isotropic Gaussian;
  critic encoder is shallow ViT (1 transformer layer); actor dropout (key for
  stability); demos pre-fill replay without oversampling; 1–10 sim demos / ~20
  real; Oculus VR teleop, 10 Hz, 7-D action.

**Recommendations for the two scenarios you asked about:**

- **(i) Train in Isaac Lab + sim-to-real to SO-101** → use **RFCL**. Five
  demos suffice on MS2 PegInsertSide in <60 min on a 4090, exploits Isaac
  Lab's native state-reset, and is the cleanest fit for sim with
  demonstrations. RLPD or PPO+demos as backup if RFCL's reset assumption
  breaks.
- **(ii) Train directly on real SO-101 with bilateral teleop** → use
  **HIL-SERL via LeRobot**. It's the only method on this list with a public
  SO-100/SO-101-aware implementation in LeRobot, and the wall-clock budget
  (1–2.5 h) fits a semester. Plan to budget 1–3 weeks of debugging based on
  ggando and Patil reproductions.

**Warning data points:** HIL-SERL on SO-101 is rough as of LeRobot v0.4.1
(issues #1387, #2346); ggando required ~3 weeks of fixes (MuJoCo-FK
replacement, state caching bug, sensitivity to lighting). Patil's blunt
assessment: *"In terms of performance I don't see a whole lot of improvement
over IL — if you collect trajectories for 2 hours you will get similar
performance with IL."* So treat HIL-SERL as a robustness booster on top of a
working BC, not a magic substitute for demonstration data.

---

## 4. Sim-to-real and Isaac Lab notes

**State of MuammerBay/isaac_so_arm101 (your reference repo).** Stack: Isaac
Sim 5.1.0, Isaac Lab 2.3.0, Python 3.11, RSL-RL, `uv`. Implements **only**
`SO-ARM100-Reach-v0` — a single proprio reach task with PPO. README
explicitly says *"Sim2Real Transfer: Work in progress"* and ships no
manipulation policy or success-rate numbers. Companion repos:
`SO-ARM101_MoveIt_IsaacSim`, `SO-ARM2_ROS2_URDF`, `so-arm101-ros2-bridge`.
Tutorial: `lycheeai-hub.com/project-so-arm101-x-isaac-sim-x-isaac-lab-tutorial-series`.
The closest external SO-101 + Isaac Lab + PPO sim-to-real attempt is the
UT Dallas CS6341 Group 15 project
(`yuxng.github.io/Courses/CS6341Fall2025/project_group_15.pdf`): proprio-only
"raise EE" sim-to-real worked; vision task with ResNet18 + Spatial Softmax +
asymmetric AC + DR was **partial / inconsistent**, blamed on cheap webcam and
>4% servo error on some joints. Required RTX 5090 32 GB.

**Isaac Lab Blackwell sm_120 status (May 2026).** Isaac Sim 5.1.0 + Isaac Lab
2.3.0 *can* run on RTX 5070/5080/5090 but with multiple unresolved bugs:

- **#2141** *"The 5070TI is not functioning properly with IsaacLab"* — open,
  dependency-version conflict on `torch==2.5.1` pin.
- **#2483** RTX 5090 remote: IRAY does not support sm_120.
- **#2652** Docker container `nvcr.io/nvidia/isaac-lab:2.1.0` `RuntimeError:
  CUDA error: no kernel image is available for execution` despite torch 2.8
  upgrade (cached torch overrides).
- **#2869** RTX 5090 viewport noisy in RTX Real-Time mode; switch to
  Interactive Path Tracing.
- **#3612** PhysX GPU pipeline falls back to CPU on Blackwell 6000 Pro.
- **#4951** (Mar 2026) `TiledCamera` hangs on RTX 5090 sm_120. **Workaround:**
  replace `TiledCameraCfg` / `TiledCamera` with `CameraCfg` / `Camera` and
  slice `[..., :3]` for RGB. **This kills the parallel-vision-RL story on
  Blackwell.**

**Working baseline:** driver ≥570, CUDA 12.8, `pip install torch==2.9.1+cu128
torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`.
Ignore the dependency-resolver warning that Isaac Lab pins `torch==2.5.1`;
multiple confirmations (Discussion #1888) that training proceeds. Multi-GPU
training reported broken.

**Verdict on RTX 5070 locally:** runs proprio-only RL (CartPole, Reach, Lift);
**breaks or is painful for vision RL with TiledCamera as of May 2026.** $200
of Brev credit ≈ ~80–100 H100-hours or ~200+ L40S-hours, more than enough for
a PPO reach/lift convergence (1–3 GPU-hours typical) plus several ablations.
**Recommendation: train on Brev, deploy locally via LeRobot CPU path.**

**Concrete DR parameters validated on SO-101 (NVIDIA Sim-to-Real-SO-101
Workshop):** sky/dome light (`exposure_range`, `temperature_range`); robot
color randomized; external camera pose (`pos_range`, `rot_range`, small);
mat/object position & yaw (workshop tightened mat yaw to (-0.1, 0.1) rad);
registered as `EventTerm(mode="reset")` in `task_env_cfg.py`. Workshop
conclusion: *"75 demonstrations (~1 hour of teleop) is too few; DR is
essential; more diverse training data beats more identical training data."*

**Failure modes on SO-101 specifically:**

- Feetech STS3215 backlash (rated ~1%, observed >4% on some joints — CS6341).
- Half-duplex serial collisions when two processes share `/dev/ttyACMx` —
  single-process driver fix.
- Calibration drift between phosphobot and LeRobot (different zero
  conventions).
- USB port assignment plug-order dependent.
- Camera intrinsics/extrinsics mismatch — NVIDIA workshop strategy 2
  ("co-train with ~50 real demos") closes the gap.
- Gripper mimic-joint desync in Isaac Lab (NVIDIA forum #357780).
- Surface gripper D6 joint physics explosion on imported URDFs (#363946).
- End-effector frame on SO-101 misaligned with fixed jaw → self-collision
  with Lula IK (#368790).
- Action space: prefer **joint position targets** (or normalized [-1,1]
  mapped to joint targets, per CS6341), not velocities.

**MuJoCo / robot-control-stack as alternative.** Canonical repo:
`github.com/RobotControlStack/robot-control-stack` (NOT JadenVCX — that's a
fork). AGPL-3.0, ICRA 2026 paper accepted (Jülg et al.), v0.6.3 released
Feb 2026. Provides Gymnasium wrappers around MuJoCo + real hardware (FR3,
xArm7, UR5e, **SO101**), Pinocchio IK, no ROS dependency. **It is not an RL
trainer** — you bring SB3/CleanRL/TD-MPC. Single-env per process (no
4096-parallel rollouts). **Verdict:** viable lean alternative if you want to
avoid Isaac Lab pain; sufficient for prototyping reward shaping and DR;
expect slower training; AGPL is a copyleft concern only if you publish closed
downstream code. Other simpler MuJoCo paths: `gym-lowcostrobot`,
`ilonajulczuk/gym-so100-c` (HER GoalEnv cube-to-bin),
`box2ai/lerobot-kinematics`, `lachlanhurst/so100-mujoco-sim`, MuJoCo
Menagerie's SO-100 model, `michel-aractingi/Sim-LeRobotHackathon` (TD-MPC
baseline).

**Recommended path for this team:** Brev H100 + Isaac Lab for the official
**NVIDIA Sim-to-Real-SO-101 Workshop** (`isaac-sim/Sim-to-Real-SO-101-Workshop`)
as Week 1–2 onboarding, then either continue Isaac Lab on Brev, or pivot to
MuJoCo/RCS for Tasks 2 and 3 if a teammate is *not* fluent with Isaac Lab.
**Pin versions:** Ubuntu 22.04, driver ≥570, Isaac Sim 5.1.0, Isaac Lab 2.3.0,
IsaacLab-Arena release/0.1.1, Python 3.11 (Isaac) + 3.12 (LeRobot v0.5.0),
torch 2.9.1+cu128, numpy 1.26.0.

---

## 5. Prioritized reading and learning list

**Tier 1 — week 1 (must read).**
LeRobot framework — `github.com/huggingface/lerobot` README +
`huggingface.co/docs/lerobot/il_robots` walkthrough. ACT — arXiv:2304.13705
+ Karkada SO-101 walkthrough
(`medium.com/@deepkarkada/action-chunking-with-transformers-act-robot-policy-80519fc024bc`).
Diffusion Policy basics — arXiv:2303.04137 + Radek Osmulski's *"Diving into
Diffusion Policy with LeRobot"* + LeRobot model card `lerobot/diffusion_pusht`.
SAC with entropy regularization — Haarnoja et al. arXiv:1801.01290 + OpenAI
Spinning Up SAC + CleanRL SAC. Stable-Baselines3.

**Tier 2 — week 2 (core for Tasks 2 and 3).**
HIL-SERL — arXiv:2410.21845 + LeRobot guide `huggingface.co/docs/lerobot/hilserl`.
RLPD 50/50 replay — Ball et al. arXiv:2302.02948 (read App. B.2 hyperparameter
table). AWAC and IBRL — Nair arXiv:2006.09359 + BAIR blog, plus IBRL
`ibrl.hengyuanhu.com`. SmolVLA finetuning — `huggingface.co/blog/smolvla` +
paper arXiv:2506.01844 + phospho.ai end-to-end recipe
`docs.phospho.ai/learn/train-smolvla`. Reward shaping for manipulation —
HIL-SERL §III (reward classifier from human-labeled demos) and SERL §V
(per-task shaping).

**Tier 3 — week 3 (high-leverage).**
Residual policy learning — Silver et al. arXiv:1812.06298 (use only if you
have a scripted base controller). FiLM conditioning — Perez et al.
arXiv:1709.07871 + the Distill explainer; matters because color/language
conditioning of ACT is most cleanly done via FiLM. Goal conditioning + HER —
Andrychowicz et al. 2017 + SB3 HER docs. Vision encoders — R3M
arXiv:2203.12601 (*"+20% over from-scratch with 20 real demos"*), Theia
arXiv:2407.20179 (*"+15 pp avg over next-best on 4 multi-step tasks"*);
useful if ACT overfits.

**Tier 4 — week 4 (only if time permits).**
Sim-to-real DR — Tobin et al. arXiv:1703.06907 + OpenAI ADR Rubik's cube blog
+ Lilian Weng's DR primer. Skip if you stay real-only. VLA-finetune-RL
coupling — OpenVLA-OFT arXiv:2502.19645 (8× A100, 1–2 days) and Octo
arXiv:2405.12213.

**Cuts (deprioritized).** Pure HRL/options theory, heavy sim-to-real if
real-only, pixel PPO/DDPG from scratch.

---

## 6. Anti-recommendations (with citations)

**Do NOT train PPO from scratch in pixel space on a real robot in a
semester.** PPO is on-policy; the original paper measures performance over 1M
timesteps in MuJoCo (continuous control); pixel-based PPO often needs
10–100M. At 30 Hz on SO-101, 1M steps = ~9.3 h of pure rollout *if learning
works first try*. The successful real-robot pixel-RL papers all use off-policy
+ parallelism: QT-Opt used 580k grasps across 7 robots for 96%
(arXiv:1806.10293); HIL-SERL gets 100% in 1–2.5 h *only because of* RLPD +
offline demos + interventions. The Sim-to-Real Progressive Nets paper opens
with: *"Deep RL algorithms are too slow to achieve performance on a real
robot."* (arXiv:1610.04286).

**Do NOT use a 7B-param VLA when you have ~100 demos.** OpenVLA README:
*"OpenVLA typically requires fine-tuning on a small demonstration dataset
(~100 demos) from your target domain robot. Out-of-the-box, it only works
well on domains from the training dataset."* Haonan Yu blog: *"50% on
familiar tasks, <10% in novel scenes."* SVRC: *"Going from 300 to 600
well-curated demonstrations reliably outperforms going from a 1B to a 7B
model, at a fraction of the compute cost."* SmolVLA paper: 450M SmolVLA gets
**87.3% LIBERO vs 76.5% for 7B OpenVLA** — pretraining quality dominates raw
params. **Use SmolVLA-base (450M) or stay with ACT.** If you must, follow
OpenVLA-OFT (parallel decode + chunking + L1) — without it, finetuning
underperforms ACT.

**Do NOT expect Diffusion Policy with 20 demos to generalize to clutter.**
DP3 paper explicitly: *"Diffusion Policy necessitates 100 to 200
human-collected demonstrations for each real-world task."* Lan-o3dp paper:
*"Diffusion policy achieves bad results because of limited demonstrations
and poor generalization."* *"Demystifying Diffusion Policies"*
(arXiv:2505.05787): DP gets strong test performance via *memorization* of
training trajectories, not true generalization. **For 20 demos and clutter
use ACT**, or DP3 with depth (40 demos / 85% on 4 real tasks).

**Do NOT do naïve sim-to-real with no DR.** OpenAI Rubik's cube: *"Domain
randomization alone is not enough; we developed Automatic Domain
Randomization."* The Rubik's robot still solves only 60% (20% on max
scrambles). Tobin et al.: real-world transfer requires randomized rendering
*"with enough variability that the real world may appear to the model as
just another variation."* For a single-arm pick-and-place, **don't go through
sim at all** — train on real demos.

**Do NOT skip data-quality checks.** SmolVLA blog: *"We tried similar dataset
with 25 episodes, and it was not enough leading to a bad performance. So,
the data quality and quantity is definitely a key."* Failure modes
documented in the community: LeRobot issue #2915 (SmolVLA 0% with 120
SO-101 episodes), Sawane *"When Fine-Tuning Hurts"* (0% with 60 success-only
demos), Kamath's hackathon team failed because of inconsistent camera names.
**Camera-pose stability between train and eval is the #1 silent killer.**

**Do NOT over-record demos.** SVRC heuristic: *"Train a baseline ACT after
every 25 episodes, measure success, and stop when the success curve flattens.
You will often hit diminishing returns around 80–150 episodes."*

**Do NOT learn a dense reward from scratch on small data.** HIL-SERL learns a
*binary* reward classifier from demos but mitigates misspecification via human
interventions during exploration. Without interventions, classifier-driven RL
drifts.

**Do NOT use plain BC/MLP and expect more data to save you.** HIL-SERL Table
1c: BC plateaus at 47% with 20 demos and 46% with 200 — flat. ACT or DP is
the floor for BC.

---

## 7. Open questions for the team to decide before committing

1. **Is anyone on the team fluent in Isaac Lab / Isaac Sim?** If no, switch to
   MuJoCo (RobotControlStack or `gym-lowcostrobot`) for Tasks 2 and 3 — Isaac
   Lab Blackwell bugs and the steep learning curve will eat 3–4 weeks
   otherwise. If yes, use NVIDIA's Sim-to-Real-SO-101 Workshop as the spine.
2. **Will you train on Brev H100 or local RTX 5070?** RTX 5070 vision RL is
   fragile (TiledCamera #4951, PhysX CPU fallback #3612, multi-GPU broken).
   Brev for training, 5070 for deployment/visualization is the safer split.
3. **Are you willing to add a second camera (top or side)?** The SmolVLA
   pretraining set is dominated by top + wrist; community SmolVLA
   reproductions with wrist-only data underperform. If you can mount one
   fixed top camera, expect a **10–20 pp absolute gain on SmolVLA finetune.**
4. **Do you want a depth sensor?** Without depth, DP3 (85% on 4 real tasks
   with 40 demos) is off the table. Wrist RGB + standard ACT/SmolVLA is fine
   but caps your data efficiency.
5. **Color conditioning for Task 2 — language token (SmolVLA) or one-hot
   FiLM (ACT)?** Decide early because it affects demo-collection format.
   Recommend FiLM if you go ACT, language if you go SmolVLA. Keep the demo
   format consistent across tasks.
6. **Reward classifier or scripted success detector for HIL-SERL?** Scripted
   (e.g., color-blob-in-target-region check from wrist camera) is faster to
   bootstrap and more reliable than a learned classifier on small data. The
   HIL-SERL paper's classifier needed ~200 positives + ~1000 negatives for
   >95% accuracy.
7. **How many bilateral teleop hours can you commit?** A defensible budget
   for this project: 80 demos × ~30 s = 40 min for Task 1; 60 demos for Task
   2 with color labels; 100–150 demos for the Task 3 primitive. Plan ~3–4
   hours of teleop total. Add ~2 hours of HIL-SERL gamepad interventions per
   RL task.
8. **Do you intend to publish or demo this?** AGPL-3.0 of robot-control-stack
   matters only if you publish a closed system; in academic context, ignore.

---

## 8. VLA vs non-VLA, per task

| Task | Best VLA | Best non-VLA | Likely winner | Why |
|------|----------|--------------|---------------|-----|
| **T1 single pick-place** | SmolVLA-base finetune (50 ep, 20k steps, ~4 h A100) — 78–90% per paper; community reproductions 0–85% | ACT (50 ep, 100k steps, ~3 h H100) — 70% Karkada, 60–90% Sherry Chen | **ACT** for first run, SmolVLA in parallel as upside | ACT is robust to data hygiene; SmolVLA is high-variance; you have $200 to run both |
| **T2 color-conditioned in clutter** | SmolVLA with explicit color in language prompt + HIL-SERL | ACT with FiLM color conditioning + HIL-SERL | **SmolVLA if camera setup matches pretraining set (top + wrist), ACT otherwise** | SmolVLA was pretrained on color-grounded prompts; if your data hygiene is questionable, ACT+FiLM is safer |
| **T3 sequential 3-step** | SmolVLA per-primitive + FSM, optionally end-to-end with explicit task tokens | ACT per-primitive + FSM + HIL-SERL on chained primitive | **ACT primitive + scripted FSM** baseline; pursue SmolVLA end-to-end as a stretch goal | Per-primitive + FSM is what won the LeRobot Worldwide Hackathon multi-step tasks (Pranav Saroha t-shirt fold) |

**General VLA vs non-VLA pattern.** SmolVLA paper Table 3 shows VLA wins on
the SO-100 multitask suite (78.3% vs 48.3% ACT), but community reproductions
(Habuda, Saroha, Sawane, Kamath, LeRobot issue #2915) show VLA results spread
from 0% to 85%, almost entirely driven by data hygiene and camera-pose
consistency rather than method. ACT delivers more consistent 60–80% with
much lower variance. **For an academic semester, ACT is the safer baseline;
SmolVLA is the upside bet to run in parallel** since H100 cost for one
finetune is ~$8 of your $200. Treat π0/π0-FAST as too compute-heavy and
reproducibility-poor (LeRobot issue #952: LeRobot's port got 41.8% LIBERO-goal
vs OpenPI's 98.8% — implementation is buggy as of May 2026). Skip Octo (not
registered as a LeRobot policy; no SO-100/101 results). Skip OpenVLA /
OpenVLA-OFT unless you have 8× A100/H100 and 1–2 days.

---

## Conclusion: what changed by doing this research

Before this dive, the natural plan would be: try ACT for Task 1, try PPO/SAC
in Isaac Lab for Tasks 2 and 3, hope sim-to-real works. The community
evidence flips that plan upside-down. **Sim-to-real with vanilla RL is a
multi-week sinkhole on Blackwell GPUs in May 2026, and the most-reproduced
"RL" pipeline on a low-cost arm — HIL-SERL — is really BC + RLPD + human
gamepad, not RL-from-scratch.** The single highest-leverage move is to
**collect more, cleaner demos** (~100 per primitive, fixed camera, consistent
naming) and **chain a strong BC primitive via a scripted FSM**, treating RL
as a precision-finishing step, not the engine. SmolVLA is worth a parallel
run because it costs ~$8 and could win Tasks 1 and 2 outright; but ACT is the
safer floor. **Spend Brev credits on H100 ACT + SmolVLA training and HIL-SERL
reward-classifier training, not on sim-to-real PPO.** Read the HIL-SERL
paper, the LeRobot ACT and HIL-SERL docs, the SmolVLA paper Table 3, and
Sherry Chen's `so101_bench` blog this week. Decide by end of week 2 whether
anyone on the team will own Isaac Lab; if not, route Tasks 2 and 3 entirely
through real-hardware HIL-SERL on top of ACT primitives.
