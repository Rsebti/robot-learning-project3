# Checkpoint inventory

The trained policies are inside the two zips (here on disk, **gitignored** —
too large for git; share via Drive/LFS or copy the zip). `unzip` then point
the infer script's `--checkpoint` at the chosen `.pt`.

## `eval1_curriculum_handoff.zip` (24 MB) — Eval1, `SO101PlaceCube-v1`
3-stage chained warm-start, raw-rgb 64×64, n_state=12.
| file | stage | hand-counted genuine /50 |
|---|---|---|
| `ckpt_S1_base_noDR.pt` | base, no DR (cold 4M) | ≈26/50 (52%) |
| `ckpt_S2_warm_DRcontrol.pt` | warm + DR control (3M) | **≈33/50 (66%) — best** |
| `ckpt_S3_warm_DRvisu_exposure.pt` | warm + DR visu+exposure (3M) | collapsed |
+ `infer_eval1_64.py`, `planche_S{1,2,3}.png`, `DEPLOY_EVAL1_CURRICULUM.md`.
**Recommended: `ckpt_S2_warm_DRcontrol.pt`.**

## `eval2_curriculum_handoff.zip` (71 MB) — Eval2, `SO101PlaceCubeEval2-v1`
FINE 9-stage graded warm-start, raw-rgb 64×64, **n_state=18** (colour-cond:
qpos6+target6+goal_onehot6 — pass `--goal_color <color>` at deploy).
`ckpt_S1_base_noDR` → `S2_ctrl_mild` → `S3_ctrl_full` → `S4_light_mild` →
`S5_light_full` → `S6_jitter` → `S7_expo030` → `S8_expo060` → `S9_expo100`.
+ `infer_eval2_64.py`, `planche_S1..S9.png`, `DEPLOY_EVAL2_CURRICULUM.md`.
Not hand-counted (per instruction) — **pick the best stage by eye from the
planches**; the quality/robustness sweet spot is typically mid-late
(≈ S5–S7); S9 = max exposure-robust but lower training metric.

## Dependency
Running these (deploy) only needs LeRobot + the infer script + the ckpt.
Re-training / sim-visualizing needs the env/training repo
**`squint-native-iso`** (`train_squint.py`, `view_policy.py`, `envs/`,
conda env `squint`). The infer scripts are self-contained for deployment
(network defs inline; no env import needed).

## Reproduce / extend
`curriculum/_curriculum_run.sh` (Eval1), `curriculum/_eval2_fine.sh`
(Eval2 fine) + the `dr_*.json` configs drive the chain.
`curriculum/CURRICULUM_STATE.md` = exact timestamped log of the run that
produced these checkpoints.
