# Handoff — SO-101 Eval1/Eval2 curriculum policies (2026-05-18)

Everything a teammate needs to **see the policies**, **run/visualize** them,
or **re-train** the curriculum. Read this file first.

> TL;DR: trained 64×64 raw-rgb SAC+C51 pick-place policies for **Eval1**
> (`SO101PlaceCube-v1`) and **Eval2** (`SO101PlaceCubeEval2-v1`, colour-cond)
> via a **chained warm-start curriculum** (base → graded domain randomization).
> The real↔sim mapping was audited & fixed (see `MAPPING.md`). The honest
> verdict is **visual** — look at the planches, the env success metric is
> NOT reliable.

---

## 1. SEE the policies (zero setup)
Open `planches/`. Each PNG is **50 evaluation rollouts** (final frame, clean
env) for one curriculum stage:
- `eval1_S1..S3` — Eval1 stages (S1 base no-DR → S3 +DR visu/exposure)
- `eval2_S1..S9` — Eval2 fine 9-stage graded curriculum

**Reading a planche:** the **green sphere is the goal marker — ignore it**.
The **colored cube is the object**. A genuine success = the colored cube
resting **inside the bowl** (NOT in the gripper / hovering / outside).
Count the genuine ones yourself — the env's reported `success_at_end`
over- and under-counts (documented; do not trust it).

## 2. RUN / VISUALIZE a policy on the real robot
1. The trained checkpoints are inside `eval1_curriculum_handoff.zip` /
   `eval2_curriculum_handoff.zip` (here on disk; gitignored — see
   `POLICIES.md`). `unzip eval1_curriculum_handoff.zip`.
2. Edit the top of `infer_eval1_64.py` (or `infer_eval2_64.py`):
   `ROBOT_PORT`, `CAMERA_INDEX`, `CALIBRATION_ID`, `CALIBRATION_DIR`.
3. ```bash
   python infer_eval1_64.py --checkpoint eval1_curriculum_handoff/ckpt_S2_warm_DRcontrol.pt --action_scale 0.1
   # Eval2 (colour-conditioned): add  --goal_color <color>
   ```
   Start at `--action_scale 0.1`, hand on the e-stop. These scripts carry the
   **validated real↔sim mapping** (`MAPPING.md`) and the correct 64px
   preprocessing/encoder (strict-load verified).

## 3. VISUALIZE the environment / a policy in sim
Needs the env/training repo **`squint-native-iso`** (`train_squint.py`,
`view_policy.py`, `envs/`). From that repo, conda env `squint`:
```bash
python view_policy.py --ckpt <path/to/ckpt_best.pt> --demos 50 \
    --render rgb_array --env-id SO101PlaceCube-v1 --obs-mode rgb \
    --image-size 64 --save        # writes a 50-demo planche to ~/Desktop
# drop --save and use --render human for an interactive GUI rollout
```
(Eval2: `--env-id SO101PlaceCubeEval2-v1`.)

## 4. FULL SOURCE — add DR / add cubes / change the MDP / re-train
**`squint-native-iso-src.zip`** = the complete env + training repo (code
only). `unzip` it, `conda env create -f squint-native-iso/environment.yaml`.
**`CODEBASE_GUIDE.md`** maps every file and gives how-to recipes:
- DR knobs → `envs/base_random_env.py` (`RandomizationConfig`) + `--dr-config-json`
- exposure DR severity → `EXPO_DR_SCALE` + `utils.ExposureDRWrapper`
- 6 cubes / multi-cube → `envs/place.py` (`SO101PlaceCubeEval3-v1` is the pattern)
- MDP (success/reward) → `envs/place.py` `evaluate()` (the genuine-ceiling fix)
- training/warm-start/curriculum → `train_squint.py` + `curriculum/`

`curriculum/` has the orchestrators (`_curriculum_run.sh`, `_eval2_fine.sh`)
+ DR JSONs + `CURRICULUM_STATE.md` (exact timestamped log of the run that
produced the shipped checkpoints). OOM-safe params: num-envs 512, buffer
200000, image 64, render 128. No secrets are shipped (codebase scanned).

---

## Status (honest)
- **Eval1** (3-stage): S1 base ≈52% genuine, **S2 (warm+DR-control) best
  ≈66%**, S3 collapsed (aggressive visu+exposure in one shot).
- **Eval2** (fine 9-stage, gentler graded exposure): no one-shot collapse;
  pick the best stage by eye from `planches/eval2_S*`.
- **Genuine ceiling = the exploitable success criterion** (hover / cube
  outside still flag success). This was deliberately NOT fixed — it touches
  `reward/evaluate` and the project rule is to keep the env NATIVE; that fix
  needs an explicit decision.
- The mapping is a *regional* compromise (faithful approach/grasp/place;
  not the curled parked home). **The real-robot run is the only true gate.**

See `WORKLOG_AND_TASKS.md` (what was done + open items + next steps),
`MAPPING.md` (deploy mapping — read before any real run),
`DEPLOY_EVAL{1,2}_CURRICULUM.md` (per-eval deploy details),
`POLICIES.md` (checkpoint inventory).
