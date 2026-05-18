# Récap COMPLET des checkpoints — Eval1 & Eval2 (16×16 px ET 64×64 px)

> Vérifié **empiriquement** : architecture lue dans les poids (`conv1` kernel),
> config lue dans les logs faisant foi (`AUTONOMOUS_RUN2.md` pour la famille
> 16px, `CURRICULUM_STATE.md` pour la famille 64px). Les `.pt` ne stockent
> **aucune** config interne (seulement `encoder/actor/critic/log_alpha/global_step`)
> → la config vient des logs, pas d'une supposition.

## Comment distinguer 16px vs 64px (sûr, à 100 %)
| | 1er conv (`encoder`) | stack | venant de |
|---|---|---|---|
| **16×16** | `[32, 3, 4, 4]` (kernel **4×4**) | 2-conv | `train_squint.py image_size==16` |
| **64×64** | `[32, 3, 8, 8]` (kernel **8×8**) | 3-conv | `train_squint.py image_size==64` |

`n_state` : **Eval1 = 12** (qpos6 + target6) · **Eval2 = 18** (qpos6 + target6 +
goal_onehot6, color-conditional). Identique 16px et 64px.

## Règle commune à TOUS les checkpoints (les deux familles)
- **Env NATIVE** : `reward` + `evaluate()` + `terminations` **jamais modifiés**
  (HARD RULE, vérifié git-diff vide). Seule exception « criterion » =
  Eval2 = succès **conditionné couleur** (env séparé `SO101PlaceCubeEval2-v1`,
  c'est une *task def*, pas une altération du reward natif).
- La seule variable entre étapes = **DR knobs** (sim2real, pas le reward) +
  taille image + warm-start.
- ⚠️ **Le `success_at_end` rapporté n'est pas fiable dans les 2 sens**
  (loophole : hover-cube-en-pince et cube-hors-bol comptés True). Seul un
  **comptage visuel /50** vaut. Colonne « succès » ci-dessous = comptage
  visuel quand il existe, sinon métrique étiquetée *(non fiable)*.

---

# FAMILLE 16×16 px — premier livrable (campagne autonome 2026-05-17)

Source faisant foi : `runs_logs/AUTONOMOUS_RUN2.md`. Tous : raw-rgb 16×16,
conv 4×4, env natif, 10 Hz, SAC+C51.

## Eval1 16px — `SO101PlaceCube-v1`, n_state=12
Curriculum = échelle « iso » (chaque étape ajoute un cran de réalisme), puis
un warm-run final s1→env-S4 promu en livrable.

| Étape | Checkpoint sur disque | Config (ajout vs étape précédente) | Succès **visuel** validé |
|---|---|---|---|
| S1 base | `runs/iso_s1_actuator/ckpt.pt` | tâche native + actionneur/contrôle fidèle (10 Hz Feetech STS3215), **DR OFF** | non comptée /50 |
| S2 +DR ctrl | `runs/iso_s2_drctrl/ckpt.pt` | **+ DR dynamique** (stiffness/damping, delay, lag) | non comptée /50 |
| S3 +DR visu | `runs/iso_s3_drvis/ckpt.pt` | **+ DR visuel** (lighting/caméra) | non comptée /50 |
| S4 +6 couleurs | `runs/iso_s4_6color/ckpt.pt` | **+ 6 couleurs** (= env HEAD, DR complète) | **~62 % genuine (31/50)** |
| **LIVRÉ** | `runs/eval1_final_deliverable/ckpt.pt` = `eval1_deploy_handoff/eval1_ckpt.pt` (md5 `0fbbcecfee4e07f3e78c3cfe36560ff2`) | warm(iso_s1)→**env S4** (DR complète + 6 col), 3M | **~70-78 % genuine** (40/50 métrique, 13 frames full-res inspectées — **PAS** un comptage /50 complet) |

> ⚠️ **Réalité robot réel** : ce livrable 16px Eval1 **échoue sur le vrai
> robot** (boucle false-grasp). Le « ~75 % » était **sim + offline seulement**.
> C'est ce qui a déclenché l'audit du mapping (corrigé depuis — voir `MAPPING.md`).
> Les `.bak.20260518-0432` dans `eval1_deploy_handoff/` = l'`infer` AVANT
> patch mapping ; le `infer_eval1.py` courant porte le mapping validé.

## Eval2 16px — `SO101PlaceCubeEval2-v1`, n_state=18 (color-cond)
Curriculum = échelle « B » mildest-visual-first (chaque cran de DR visuel
ajouté séparément), tous comptés **visuellement /50** (mandat autonome).

| Étape | Checkpoint sur disque | Config (ajout) | Succès **visuel** validé /50 |
|---|---|---|---|
| base no-DR | `runs/eval2_nodr/ckpt.pt` | DR **OFF** | ~92 % — mais **non déployable** (zéro robustesse) |
| Stage A (échec) | `runs/eval2_warm_full/ckpt.pt` | warm nodr + **toute la DR d'un coup** | ~8 % (1/12) — gap trop grand, abandonné |
| B1 +DR ctrl | `runs/eval2_b1_ctrl/ckpt.pt` | **+ DR dynamique** (visuel OFF) | **74 % (37/50)** |
| B1.5 +caméra | `runs/eval2_b15_ctrlcam/ckpt.pt` | **+ bruit caméra** (lighting OFF) | **74 % (37/50)** — robuste caméra |
| B2 (full) | `runs/eval2_b2_full/ckpt.pt` | **+ lighting agressif** (full DR) | 64 % (32/50) — régression |
| B2′ (full agg.) | `runs/eval2_b2final_full/ckpt.pt` | + lighting agressif (depuis B1.5) | 62 % (31/50) |
| **LIVRÉ — B1.75** | `runs/eval2_final_deliverable/ckpt.pt` = `eval2_deploy_handoff/eval2_ckpt.pt` (md5 `a67cd585384c3b4cf03e038ae47d8461`) | warm B1.5 + **lighting réaliste étroit** (ambient 0.28-0.42, dir 0.85-1.15) | **68 % (34/50)** — meilleur compromis *déployable* |

> Plafond documenté : la DR lighting agressive cape les policies color-cond
> raw-rgb **16px** (~62 %). B1.75 (lighting réaliste) = compromis déployable
> ~68 %. Critère couleur vérifié robuste 3× (mauvaise couleur ⇒ False).

---

# FAMILLE 64×64 px — curriculum chain warm-start (cette session, 2026-05-18)

Source faisant foi : `runs_logs/CURRICULUM_STATE.md`. Tous : raw-rgb 64×64,
conv 8×8, env natif, SAC+C51. Params OOM-safe : num-envs 512, buffer 200000,
image 64, render 128. **Mapping validé** embarqué dans `infer_eval{1,2}_64.py`.

## Eval1 64px — `SO101PlaceCube-v1`, n_state=12 — chaîne 3 étapes
Livré : `eval1_curriculum_handoff.zip` (S1+S2+S3).

| Étape | Ckpt (run) → nom livré | Config | Succès **visuel** /50 |
|---|---|---|---|
| **S1** base | `runs/eval1_cur_s1/ckpt_best.pt` → `ckpt_S1_base_noDR.pt` | cold 4M, **DR OFF**, no-jitter, no-exposure (env pur natif) | **~26/50 (52 %)** (compté main) |
| **S2** warm +DR ctrl | `runs/eval1_cur_s2/ckpt_best.pt` → `ckpt_S2_warm_DRcontrol.pt` | warm S1, **+DR contrôle** (`dr_ctrl.json`: lighting off), 3M | **~33/50 (66 %) — MEILLEUR** (compté main) |
| **S3** warm +DR visu+expo | `runs/eval1_cur_s3/ckpt_best.pt` → `ckpt_S3_warm_DRvisu_exposure.pt` | warm S2, **+DR visuel + jitter + exposure** (agressif d'un coup), 3M | **COLLAPSE** (success 0.00, ckpt jamais amélioré) |

> Recommandé Eval1 64px = **S2** (66 % genuine, et réduit aussi le loophole
> hover). S3 = collapse training (visu+exposure trop fort en une fois).

## Eval2 64px — `SO101PlaceCubeEval2-v1`, n_state=18 — chaîne FINE 9 étapes
Exposure douce graduée sur enveloppe `ExposureDRWrapper` **réduite**
(gain 0.7-1.5 / wb .07 / gamma .85-1.25 / white_lift .05), `EXPO_DR_SCALE`
neutre→max. Livré : `eval2_curriculum_handoff.zip` (S1..S9).

| Étape | Ckpt (run) → nom livré | Config (ajout warm-chaîné) | Succès visuel /50 |
|---|---|---|---|
| **S1** | `runs/eval2_cur_s1/ckpt_best.pt` → `ckpt_S1_base_noDR.pt` | cold 4M, **DR OFF** (env pur natif) | *(à juger sur planche)* |
| **S2** | `runs/eval2f_s2/ckpt_best.pt` → `ckpt_S2_ctrl_mild.pt` | warm S1, **+DR ctrl MILD** (`dr_s2_ctrl_mild.json`), 1.5M | *(planche)* |
| **S3** | `runs/eval2f_s3/ckpt_best.pt` → `ckpt_S3_ctrl_full.pt` | **+DR ctrl FULL** (`dr_ctrl.json`) | *(planche)* |
| **S4** | `runs/eval2f_s4/ckpt_best.pt` → `ckpt_S4_light_mild.pt` | **+ lighting MILD** (`dr_s4_light_mild.json`) | *(planche)* |
| **S5** | `runs/eval2f_s5/ckpt_best.pt` → `ckpt_S5_light_full.pt` | **+ lighting FULL** | *(planche)* |
| **S6** | `runs/eval2f_s6/ckpt_best.pt` → `ckpt_S6_jitter.pt` | **+ color jitter** | *(planche)* |
| **S7** | `runs/eval2f_s7/ckpt_best.pt` → `ckpt_S7_expo030.pt` | **+ exposure ×0.30** | *(planche)* |
| **S8** | `runs/eval2f_s8/ckpt_best.pt` → `ckpt_S8_expo060.pt` | **+ exposure ×0.60** | *(planche)* |
| **S9** | `runs/eval2f_s9/ckpt_best.pt` → `ckpt_S9_expo100.pt` | **+ exposure ×1.00** (max enveloppe réduite) | *(planche)* |

> Fait factuel : plusieurs `ckpt_best` fins sont restés au **warm-init**
> (la métrique ne s'est pas améliorée sous la DR ajoutée) → la meilleure
> policy antérieure est propagée sans dégradation. Pas de collapse brutal
> (rampe douce, contrairement à S3-Eval1). **Tu choisis le meilleur stade à
> l'œil sur `planches/eval2_S*`** (sweet-spot typique ≈ S5–S7).

---

# Synthèse — « qu'est-ce que je livre / je garde »

| Cible | Famille | Checkpoint à déployer | Succès visuel honnête | État robot réel |
|---|---|---|---|---|
| Eval1 | 16px (1er livrable) | `eval1_deploy_handoff/eval1_ckpt.pt` | ~70-78 % sim (partiel) | **ÉCHEC** (false-grasp) — mapping corrigé depuis |
| Eval1 | **64px (recommandé)** | `eval1_curriculum_handoff/ckpt_S2_warm_DRcontrol.pt` | **66 % (33/50)** | à tester (mapping validé embarqué) |
| Eval2 | 16px (1er livrable) | `eval2_deploy_handoff/eval2_ckpt.pt` (B1.75) | **68 % (34/50)** | à tester |
| Eval2 | 64px | meilleure planche `eval2_S*` (≈ S5–S7) | à juger visuellement | à tester |

**Le run robot réel reste le seul vrai juge.** Tous les chiffres « sim » ci-dessus
souffrent du loophole de critère (documenté `WORKLOG_AND_TASKS.md` point #1).
