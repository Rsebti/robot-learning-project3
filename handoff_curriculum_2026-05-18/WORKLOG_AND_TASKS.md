# Work log & task list — 2026-05-17/18

Chronological summary of everything done, with status. For teammates to see
exactly what was tackled and what remains.

## ✅ Completed

### A. Machine stability (blocking everything)
- RTX 5070 PC hard-froze every ~10–26 min under GPU load.
- **Fixed**: kernel param `pcie_aspm=off`. Validated by a soak that survived
  **2h50 under 96–98% GPU, Xid=0** (beat the pre-fix 2h44 record). No freeze
  since.

### B. Real↔sim mapping — audited, fixed, validated, shipped
- Eval1 failed on the **real robot** (false-grasp loop). Root-caused to the
  real↔sim joint/gripper mapping, NOT control freq (10vs30Hz was tested &
  ruled out on the physical robot).
- Used 4 ground-truth anchors (real photos + joint readouts) + a step-by-step
  HF-demo replay (contact frame f340) with **visual approval at each step**.
- **Findings:** gripper servo↔sim was inverted (real-closed → sim-open);
  arm `deg2rad`-only left the TCP ~3–4 cm too high at grasp; the
  home↔'rest' anchor is kinematically wrong on arm height (proved: no
  per-joint affine with that anchor descends correctly — needs external data
  or the constant-offset compromise).
- **Resolution (validated visually by the user):** constant per-joint offset
  `JOINT_OFFSET = [0, +0.080, +0.220, 0, 0, 0] rad` (lift/elbow homing-type
  recal) + gripper `servo[1,75]° ↔ sim[-18,120]°`. Applied bidirectionally,
  round-trip exact (~1e-15). Tuned so TCP z ≈ 0.010 m at grasp (tip below
  cube mid). Details in `MAPPING.md`.
- **Shipped:** patched `deploy_utils/manipulator.py` + the eval1/eval2
  deploy-handoff infer scripts + the dedicated 64px scripts
  (`infer_eval1_64.py`, `infer_eval2_64.py`). All strict-load & round-trip
  verified. DEPLOY docs corrected (old "gripper byte-identical" claim was
  false).

### C. Curriculum chain warm-start — Eval1 (3-stage)
- S1 base raw-rgb 64×64 **no DR** (cold, 4M) → S2 warm **+DR control**
  (3M) → S3 warm **+DR visu+exposure** (3M). Env stays NATIVE
  (reward/evaluate/terminations untouched; only DR knobs).
- **Visual /50 gate (counted by hand, not the metric):**
  S1 ≈ **26/50 (52%)**, S2 ≈ **33/50 (66%, best)**, S3 **collapsed**
  (aggressive visu+exposure in one shot → success 0.00).
- Delivered `~/Desktop/eval1_curriculum_handoff.zip` (also here).

### D. Curriculum chain warm-start — Eval2 (FINE 9-stage, redesigned)
- Per request: many more, finer stages + **gentler exposure**.
- `ExposureDRWrapper` envelope **reduced** (gain 0.5-2.0→0.7-1.5, wb
  .12→.07, gamma .7-1.5→.85-1.25, white_lift .10→.05) + new
  `EXPO_DR_SCALE` env var that linearly interpolates it from neutral
  (no-op) → reduced-max. (`import os` added to `utils.py`.) DR/obs only.
- 9 stages: S1 base → S2 ctrl-mild → S3 ctrl-full → S4 light-mild →
  S5 light-full → S6 +jitter → S7 expo×0.30 → S8 expo×0.60 → S9 expo×1.00,
  each warm-chained, 1.5M steps. **No one-shot collapse** (gentle ramp).
- Several fine `ckpt_best` stayed at warm-init (metric didn't improve under
  the added DR) → the best earlier policy is safely propagated (no
  degradation shipped). Pick the best stage by eye from the planches.
- Delivered `~/Desktop/eval2_curriculum_handoff.zip` (also here, 9 ckpts).

### E. Housekeeping
- 51 disposable diagnostic scripts removed (conclusions preserved in the
  state log / memory). Validation visual evidence kept.

## ⚠️ Known issues / open items
1. **Exploitable success criterion (the real ceiling).** `success_at_end`
   counts hover-with-cube-in-gripper and cube-outside-bowl as success. The
   reported metric is unreliable in BOTH directions (Eval1 S1: metric 0.88 vs
   visual 52%; S2: metric 0.44 vs visual 66%). Fixing it means changing
   `reward`/`evaluate` → conflicts the "keep env NATIVE" rule → **needs an
   explicit team decision** (NOT done autonomously).
2. **Mapping is a regional compromise** (faithful in the operating region,
   not the curled parked home). The real-robot run is the only true gate.
   The exact fix would need a 5th real-robot anchor (a non-home arm pose +
   joint readout + photo).
3. **Aggressive lighting/exposure DR caps raw-rgb16/64 policies** (Eval1 S3
   collapse; the Eval2 fine ramp mitigates but exposure×1.0 still suppresses
   the training metric).

## ▶️ Suggested next steps for the team
1. Eyeball `planches/` → pick the deployable ckpt per eval.
2. Real SO-101 run with the chosen ckpt via `infer_eval{1,2}_64.py`
   (mapping already validated; start `--action_scale 0.1`).
3. If the genuine ceiling blocks: decide on a success-criterion fix
   (release-required grasp / stricter in-bowl check) — this is the
   highest-leverage change and the one deliberately left for the team.
4. If a precise descent mapping is needed for production: capture the 5th
   real anchor (see `MAPPING.md`).

## Task checklist (all done this session)
- [x] Fix RTX5070 freeze (pcie_aspm=off, validated)
- [x] Audit infer mapping vs 4 ground-truth anchors
- [x] Derive + visually validate corrected real→sim mapping
- [x] Step-by-step descent diagnostic (S1→S4b, visual approval each step)
- [x] Patch manipulator.py + all infer scripts (16px handoff + 64px) + DEPLOY docs
- [x] Eval1 curriculum S1/S2/S3 + visual /50 gates + handoff zip
- [x] Eval2 redesigned fine 9-stage curriculum + gentler scaled exposure + handoff zip
- [x] Dedicated 64px infer scripts (eval1/eval2), strict-load verified
- [x] Cleanup disposable scripts
- [x] This teammate handoff folder
