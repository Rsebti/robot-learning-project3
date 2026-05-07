# Eval 2 — IK + scripted controller pipeline (état au 2026-05-06 v2)

> **Contexte pour reprendre le travail dans une nouvelle conversation.**
> Tout ce qu'il faut savoir pour comprendre où on en est, pourquoi on fait IK,
> et la pipeline complète jusqu'au déploiement sur le SO-101 réel.
>
> **Version actuelle** : pipeline avec **magic attach** (cube kinematic+ghost
> pendant la prise) — la physique de grasp PhysX est intrinsèquement instable
> sur le SO-101 (jaw qui tourne au lieu de translater + meshes complexes), donc
> on bypass et on téléporte le cube avec le gripper. Voir section 3 pour les
> détails.

---

## 1. Pourquoi on fait de l'IK ?

### Objectif final
Eval 2 = pick-and-place sur le SO-101 réel. 2 cubes colorés (rouge + bleu) +
1 bowl. La policy doit identifier le cube de la couleur cible et le placer
dans le bowl. **RL obligatoire** d'après spec TA, mais démos teleop
explicitement encouragées comme warmstart.

### Pourquoi pas du PPO from-scratch ?
On a essayé v1.0–v1.8 (cf. [`sim/eval2.md`](../sim/eval2.md)). PPO from-scratch
plafonne à 2-7 % succès. Local optimum classique : la policy game les
milestones intermédiaires sans jamais commit au placement final.

### Pourquoi BC + DAPG ?
**BC pretrain (ACT) + DAPG finetune (PPO + aux BC loss)** est la voie standard
en RL manipulation (DexterousHands, Adroit, etc.). Mais ça demande des
**démos expertes**.

### Pourquoi IK pour générer les démos ?
Au lieu d'attendre des démos teleop réelles (besoin du robot + temps labo),
on peut **scripter un controller en sim** qui résout déterministe le pick-
and-place :

```
target xyz (cube ou bowl) → IK → joint targets → JointPositionAction
```

Ce controller, multiplié par 4096 envs en parallèle dans Isaac Lab, génère
~10k démos en quelques minutes. Plus rapide que teleop manuel.

### Pourquoi l'IK est plus complexe qu'on pensait
SO-101 est **5-DoF** mais on veut tracker un goal 6-DoF (xyz + orientation
gripper). Sous-déterminé. De plus, les limites articulaires sont serrées
(notamment `wrist_flex` = ±1.658 rad). Plusieurs approches abandonnées :

1. **Isaac Lab `DifferentialIKController`** : wrist drift, ostrich jump
2. **Closed-form analytical IK** : math correcte (FK→IK→FK = 0.0001 mm)
   mais wrist_flex sature en pratique pour les targets de DESCEND
3. **`pytorch_kinematics` PseudoInverseIK** : early-stopping foireux,
   sous-convergence (~30 % de la solution par appel)

→ **Solution actuelle** : DLS manuelle avec contraintes de wrist
(implémentée dans `sim/eval2/bc/ik_solver.py`).

---

## 2. Pipeline complète (IK → Real robot)

```
┌──────────────────────────────────────────────────────────────────────┐
│  Étape 1 — Scripted controller en sim (IK + state machine)            │
│  → Smoke test 50 envs × 100 episodes                                  │
│  → Critère : ≥ 50 % succès pour valider                               │
│  Files: sim/eval2/bc/{scripted_controller, ik_solver, run_scripted}.py│
└──────────────────────┬───────────────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Étape 2 — Génération de démos (4096 envs en parallèle)               │
│  → 10 000 démos sauvées en .npz (image, joint_pos, action, target_color│
│    bowl_xyz, last_action) pour seulement les épisodes succès          │
│  Files: sim/eval2/bc/generate_demos.py (à finaliser)                  │
└──────────────────────┬───────────────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Étape 3 — BC pretrain ACT end-to-end vision                          │
│  → Image wrist cam + joint_pos + target_color + bowl_xyz → action     │
│  → ~2 h sur RTX 5070 ou Brev H100                                     │
│  → Push HF: Rsebti/projet3-act-eval2-bc                               │
│  Files: train/launch_act.sh (adapter pour Eval 2 dataset)             │
│  Test: train/smoke_test_act.py (déjà PASS)                            │
└──────────────────────┬───────────────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Étape 4 — DAPG finetune (PPO + aux BC loss)                          │
│  → Charge le checkpoint BC, finetune avec PPO                         │
│  → λ_BC schedule : 1.0 → 0 sur ~5000 iter                              │
│  → Domain randomization activée (couleurs, lumière, friction)         │
│  → ~5-10 h sur Brev H100, ~$30-60                                     │
│  Files: sim/eval2/agents/rsl_rl_ppo_cfg.py (à modifier pour DAPG)     │
└──────────────────────┬───────────────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Étape 5 — Sim eval (50 envs × 100 episodes)                          │
│  → Cible : ≥ 80 % succès en sim                                       │
│  → Si gros gap avec étape 3 → DR plus aggressive                      │
└──────────────────────┬───────────────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Étape 6 — Refactor deploy/eval2_inference_e2e.py (déjà fait)         │
│  → Image directement dans la policy (pas de CNN perception)           │
│  → Concat (image, joint_pos, target_color, bowl_xyz) → action 6D      │
│  Files: deploy/eval2_inference_e2e.py (déjà écrit)                    │
└──────────────────────┬───────────────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Étape 7 — Deploy sur SO-101 réel                                     │
│  → 5 rollouts au labo avec configs TA (target_color, bowl_xyz)        │
│  → Score Eval 2 = 5 × 10 pts                                          │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Fichiers du scripted controller — détail

### `sim/eval2/bc/ik_solver.py` (le cœur de l'IK)

**Classe** : `SO101IKSolver`

**Rôle** : résout l'IK pour le SO-101. Charge l'URDF directement avec
`pytorch_kinematics`, fait un DLS manuel itéré jusqu'à convergence.

**Hyperparams clés** :
```python
n_iter=100              # max iterations DLS
lr=0.3                  # learning rate
damping=0.01            # DLS damping factor
early_stop_mm=0.5       # stop quand max err < 0.5 mm
converged_threshold_mm=5.0  # converged flag si err < 5 mm
lock_wrist_flex=True    # voir constraints ci-dessous
wrist_flex_value=π/2    # base value pour la contrainte
```

**Contraintes** (résolvent le problème SO-101 5-DoF) :
1. **`wrist_flex = π/2 − shoulder_lift − elbow_flex`**
   - Garantit que le gripper pointe **toujours vertical** peu importe la pose
   - Sans cette contrainte, le gripper se tilt dans la direction (s_l + e_f)
     et un jaw heurte le cube par le dessus pendant la descente
2. **`wrist_roll = 0`**
   - Garantit que les jaws sont alignées sur l'axe Y de la base
   - Sans, wrist_roll dérive (≈ 17°), les jaws ne sont plus parallèles aux
     faces du cube → forces non opposées → grasp foire

**Architecture du solveur** :
- Construction unique de la chain à partir de l'URDF
  (`build_serial_chain_from_urdf`)
- Chaque appel `solve(target_pos_b, current_joints)` :
  - `q = current_joints` (warm-start critique pour éviter ostrich jumps)
  - Enforce contraintes wrist_flex et wrist_roll au démarrage
  - Itère DLS sur `q[shoulder_pan, shoulder_lift, elbow_flex]` (3 DoF)
    avec Jacobian effective (incluant l'effet indirect de wrist_flex)
  - Re-enforce contraintes à chaque iter
  - Returns `(q, converged)` où `converged = err < 5 mm`

### `sim/eval2/bc/scripted_controller.py` (state machine + actions + magic attach)

**Classe** : `ScriptedPickController`

**Rôle** : orchestrer le pick-and-place via une machine à états, générer les
actions IK à chaque step, ET gérer le **magic attach** du cube au gripper
(bypass de la physique PhysX du grasp).

**Machine à états — 10 phases** :
```
0 APPROACH            gripper 3 cm au-dessus du cube, 4 cm DERRIÈRE en x
1 DESCEND             gripper au niveau cube, 4 cm DERRIÈRE en x
2 SLIDE_FORWARD       gripper avance vers cube_x ; advance sur fixed-jaw
                      contact (force > 0.05 N) ou POS_TOL ou timeout
3 CLOSE               jaws closent (cmd=+0.05 → ~1.5cm separation),
                      magic attach se déclenche ici sur fixed-jaw contact
4 LIFT                cube glué au gripper monte de 8 cm (rampe 12 steps)
5 ABOVE_BOWL          gripper au-dessus du bowl (z=bowl+8cm)
6 DESCEND_TO_RELEASE  gripper descend dans le bowl (z=bowl+4cm)
7 OPEN                jaws s'ouvrent, MAGIC DETACH (cube redevient
                      dynamic et tombe sous gravité)
8 RETREAT             1 step no-op (skipped)
9 DONE                idle, episode termine au timeout (12s)
```

**Transitions** :
- **Cartésiennes** (APPROACH/DESCEND/LIFT/ABOVE_BOWL/DESCEND_TO_RELEASE) :
  avancent quand `dist < POS_TOL=15mm` OU timeout
- **SLIDE_FORWARD** : advance EARLY sur **contact jaw fixe avec cube**
  (force > 0.05 N), sinon dist < POS_TOL, sinon timeout
- **Fixed-duration** (CLOSE, OPEN) : avancent uniquement sur step count

**GRASP_X_OFFSET = -0.040** (4 cm en arrière) :
- Appliqué SEULEMENT à APPROACH/DESCEND. Le gripper descend BIEN derrière
  le cube, la jaw mobile (qui dépasse en avant de gripper_frame_link) ne
  touche pas le cube. SLIDE_FORWARD ramène ensuite à cube_x sans offset.
- C'est la séquence demandée par le user : "descend avec offset, puis
  slide forward jusqu'à contact, puis close".

**MAGIC ATTACH** (le mécanisme qui rend tout ça possible) :
- Trigger : pendant SLIDE_FORWARD ou CLOSE, dès que la **jaw fixe** détecte
  un contact avec le cube cible (force > 0.05 N via ContactSensor)
- Action :
  1. Capture la pose du cube relative à `gripper_frame_link` AS-IS (le
     cube reste où il est, on snapshote juste le offset)
  2. Bascule le cube en **kinematic** (`PhysxRigidBodyAPI.kinematicEnabled
     = True`) → ne réagit plus aux forces de contact
  3. Désactive les collisions du cube (`UsdPhysicsCollisionAPI.collision
     Enabled = False`) → la jaw mobile peut fermer librement à travers
     le cube fantôme sans être repoussée
- Maintenance : à chaque step, on overwrite `cube_pose_world =
  gripper_pose_world * cube_attach_pos_local`. Plus la vélocité du cube
  pour matcher celle du gripper. Le cube est rigidement glué au gripper
  pendant LIFT/ABOVE_BOWL/DESCEND_TO_RELEASE.
- Hard floor : `cube_z` est clampé à >= 0.010 dans la maintenance, pour
  empêcher le cube de rentrer dans la table si le gripper a un PD overshoot
  bas.
- Detach : à l'entrée de OPEN, on reverse :
  1. Cube redevient dynamic (`kinematicEnabled = False`)
  2. Collisions réactivées
  Le cube tombe sous gravité dans le bowl.
- Au reset d'épisode, on s'assure aussi que le cube est dynamic (sinon il
  ne respecterait pas la randomisation de spawn).

**Pourquoi le magic attach** : la physique PhysX du SO-101 est intrinsèquement
instable pour le grasp d'un cube rigide :
- La jaw mobile rotate (au lieu de translater) → percute le cube avant
  de le pincer → bouncing massif (forces 50N+, joints qui flailent)
- `convex_decomposition` self-block le joint moving jaw à +0.36
  (sub-shapes qui collisionnent en interne)
- `convex_hull` enveloppe imprécise → le contact est mal défini
- Aucune combinaison de friction/stiffness/effort ne donne un grasp
  visuellement et physiquement propre

Au lieu de fight la physique, on génère des démos avec un grasp "magique".
La policy BC apprend la **séquence d'actions** (open/close gripper,
trajectoires bras), ce qui se transfert au robot RÉEL où la physique de la
pince fonctionne bien.

**Pipeline d'une action** :
```
target_pos_w = state machine retourne cible monde de la phase courante
target_pos_b = subtract_frame_transforms(robot_root_w, target_pos_w)
joint_pos_des, ik_converged = ik.solve(target_pos_b, current_arm)
fallback : si IK fail, garder current_arm sauf wrist (forcé)
ENFORCE wrist_flex = π/2 − shoulder_lift − elbow_flex ET wrist_roll = 0
arm_action = 2 * (joint_pos_des - default_arm_pos)
gripper_action = +1 (open) ou -1 (close) selon phase
action = cat([arm_action, gripper_action])

# Puis :
_advance_phases()  # transitions de state machine
_update_magic_attach()  # contact check + kinematic toggle + cube pose write
```

**Constantes importantes (état actuel)** :
```python
APPROACH_HEIGHT     = 0.030    # cube + 3cm
DESCEND_HEIGHT      = 0.000    # cube center exact
LIFT_HEIGHT         = 0.080    # 8cm — passe au-dessus du bowl rim (2.5cm)
LIFT_RAMP_STEPS     = 12       # rampe rapide vu que cube est kinematic
ABOVE_BOWL_HEIGHT   = 0.080
RELEASE_HEIGHT      = 0.040
RETREAT_HEIGHT      = 0.040    # = RELEASE (RETREAT skipped)
GRASP_X_OFFSET      = -0.040   # 4cm derrière, SLIDE_FORWARD compense
POS_TOL             = 0.015    # 15 mm
GRASP_FORCE_THRESHOLD_N = 0.05  # seuil contact pour magic attach
PHASE_MAX_STEPS     = (50, 150, 80, 120, 20, 60, 40, 50, 1, 500)
                      # APP DES SLI CLO LIF ABV DTR OPE RET DON
MAX_RETRIES         = 2        # retry CLOSE→APPROACH si pas d'attach
```

### `sim/eval2/bc/run_scripted.py` (le launcher de smoke test)

**Rôle** : lancer le scripted controller dans Isaac Lab pour valider qu'il
résout bien la tâche. Sortie = success rate + histogramme des phases finales.

**Args** :
- `--num_envs N --num_episodes M` : combien d'envs en parallèle, total épisodes
- `--headless` ou `--no_headless` : avec ou sans fenêtre Isaac Sim
- `--debug_env0` : print debug détaillé à chaque ~5 steps pour env 0
  (positions, joints, action, FK consistency)

**Vérifications de sanity intégrées** :
- `FK pytorch_kinematics` vs `FK Isaac (gripper_frame_link)` → diff < 1 mm
  garantit que l'URDF chargé par le solveur matche exactement Isaac Sim
- `[IK conv]` log toutes les 200 IK calls : err max + % converged

**Tâche par défaut** : `Eval2-PickInClutter-Play-v1` (sans cam, plus rapide
pour test).

### `sim/eval2/joint_pos_env_cfg.py` (config env qui spécifie le robot)

Modifications faites pour le scripted controller :
- `debug_vis=False` sur le `FrameTransformer` (sinon les flèches XYZ
  encombrent la vue)
- `close_command_expr={"gripper": -0.3}` (au lieu de `0.0` standard) :
  PD applique plus de torque pour serrer le cube
- `JointPositionActionCfg(scale=0.5, use_default_offset=True)` : intact

### `sim/eval2/bc/test_pytorch_kinematics.py` (test unitaire)

Smoke test `pytorch_kinematics` sur l'URDF SO-101 : FK home pose +
roundtrip FK→IK→FK sur 200 configs aléatoires.

### URDF utilisé

`C:/Users/user/Desktop/MA2/isaac/isaac_so_arm101/src/isaac_so_arm101/robots/trs_so101/urdf/so_arm101.urdf`

**Important** : c'est l'URDF embarqué dans `isaac_so_arm101` (le repo
Layer 2). C'est **exactement** ce qu'Isaac Sim charge → FK consistent à 0.00 mm.

Joints (ordre dans la chaîne) :
```
['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
```

Frame target IK : `gripper_frame_link` (= entre les jaws quand fermées)

---

## 4. Commandes de test

### Smoke test 1 env, visuel (debug)
```powershell
cd C:\Users\user\Desktop\MA2\isaac\isaac_so_arm101
$env:PYTHONUNBUFFERED = "1"
uv run python -m sim.eval2.bc.run_scripted --num_envs 1 --num_episodes 5 --no_headless --debug_env0
```

### Smoke test 50 envs, headless (validation)
```powershell
cd C:\Users\user\Desktop\MA2\isaac\isaac_so_arm101
$env:PYTHONUNBUFFERED = "1"
uv run python -m sim.eval2.bc.run_scripted --num_envs 50 --num_episodes 100 --headless
```

### Critères d'acceptation
- ≥ 50 % succès → on passe à la collecte de démos (4096 envs)
- 30-50 % → tuning fin (offsets, friction, PD gains)
- < 30 % → diagnostiquer en `--no_headless --debug_env0`

---

## 5. État actuel (2026-05-06 fin de soirée)

### Ce qui marche ✅
- IK math correcte (FK consistency 0.00 mm avec Isaac Sim)
- DLS manuelle converge à < 1 mm en ~12 itérations
- Toutes les 9 phases s'enchaînent (avec fallback timeout)
- Gripper vertical garanti par la contrainte
  `wrist_flex = π/2 - shoulder_lift - elbow_flex`
- wrist_roll forcé à 0 au niveau action
- GRASP_X_OFFSET=-0.015 fixe la collision moving jaw vs face avant cube
- Joints du robot stables : pas de flailing, pas d'elbow flip

### Ce qui reste à valider (le user était en train de tester) 🧪
- Le grasp tient-il le cube pendant le LIFT (avec close_command_expr=-0.3) ?
- Le cube atterrit-il dans le bowl à la fin ?
- Le success rate sur 100 episodes (50 envs)

### Problèmes connus / à surveiller ⚠️
- **PD overshoot** : 5-12 mm d'erreur même avec IK convergée à 0.5 mm.
  C'est pour ça qu'on a POS_TOL=15 mm et le timeout fallback.
- **Cube pushed pendant DESCEND** : visible si offset trop petit. Tuner
  GRASP_X_OFFSET (-0.010 à -0.025) selon comportement.
- **Cube éjecté pendant LIFT** : si grip pas assez fort. Tuner
  `close_command_expr` (-0.2 à -0.5).
- **Cube écrasé / déformé** : si grip trop fort. Réduire
  `close_command_expr` vers -0.1.

### Choses testées et abandonnées (ne pas refaire) 🚫
- **Clamp `wrist_flex.clamp(-1.55, 1.55)` dans `ik_solver.py`** : tenté
  pour empêcher l'IK de demander un wrist au-delà des limites URDF lors
  d'un LIFT en branche elbow-flipped. Le user a confirmé que ça rendait
  le robot instable ("le changement de stifness fait buguer le robot").
  **Revert systématique** : garder la contrainte non clampée.
- **Friction cubes (2.0/2.0 static/dynamic)** : ajoutée pour empêcher le
  cube de glisser hors de la prise. Combinée à un grip plus serré, elle
  participait au flailing post-CLOSE. Retirée — on revient aux defaults
  Isaac Sim. Si le grasp glisse à nouveau, plutôt tuner d'abord
  `close_command_expr` et `DESCEND_HEIGHT`.

---

## 6. Prochaines étapes (par ordre)

### Étape 1 — Finaliser le scripted controller (en cours)
Itérer sur les hyperparams jusqu'à ≥ 50 % succès sur 100 episodes :
- `GRASP_X_OFFSET` (-0.010 à -0.025)
- `close_command_expr` dans env_cfg (-0.1 à -0.5)
- `DESCEND_HEIGHT` (-0.005 à +0.005)
- `POS_TOL` (10 à 20 mm)

### Étape 2 — Generate demos
Modifier ou créer `sim/eval2/bc/generate_demos.py` :
```python
# Boucle 4096 envs × N episodes
# Pour chaque step, sauver (image, joint_pos, joint_vel, action, target_color, bowl_xyz)
# Filtrer les épisodes : garder uniquement ceux qui ont eu success_target_block_in_bowl
# Sauver en lerobot dataset format ou .npz par episode
```

### Étape 3 — BC pretrain ACT
Adapter `train/launch_act.sh` :
- Dataset = nouveau dataset Eval 2 (image + state)
- State features : joint_pos[6] + joint_vel[6] + target_color_oh[2] + bowl_xyz[3]
- Image features : wrist_cam RGB
- Output : action 6D
- ~20k steps, 1-2 h sur 5070

Test pipeline déjà fait : `train/smoke_test_act.py` PASS sur ce PC.

### Étape 4 — DAPG finetune
Modifier `sim/eval2/agents/rsl_rl_ppo_cfg.py` :
- Charger le checkpoint BC ACT comme actor initial
- Ajouter aux loss : `λ_BC × L1(action_policy, action_demo)`
- λ_BC schedule : 1.0 → 0 sur ~5000 iter
- Activer domain randomization (events.py)
- Run sur Brev H100 : ~5-10h

### Étape 5 — Refactor deploy
`deploy/eval2_inference_e2e.py` est déjà écrit pour image-in-policy.
Vérifier que le state vector matche le format des démos.

### Étape 6 — Real robot
Au labo avec le SO-101 :
- Setup : 2 cubes adjacents + bowl, target_color + bowl_xyz fournis par TA
- Run `lerobot-record --policy.path=Rsebti/projet3-eval2-final-...`
- 5 rollouts, score Eval 2

---

## 7. Architecture du repo (rappel)

```
robot-learning-project3/
├── sim/eval2/
│   ├── bc/                              <- scripted controller IK + tests
│   │   ├── ik_solver.py                 SO101IKSolver (DLS manuelle)
│   │   ├── scripted_controller.py       Phase machine + action gen
│   │   ├── run_scripted.py              Launcher du smoke test
│   │   ├── test_pytorch_kinematics.py   Test unitaire IK
│   │   ├── analytical_ik.py             ANCIEN — closed-form (archivé)
│   │   ├── measure_link_lengths.py      Calibration link lengths (utile)
│   │   └── so101_link_lengths.json      Calibration sortie
│   ├── perception/                      <- CNN ColorBlockCNN (obsolète,
│   │                                       gardé comme warmstart possible)
│   ├── agents/rsl_rl_ppo_cfg.py         Hyperparams PPO/DAPG
│   ├── mdp/                             reward, obs, terminations, events
│   ├── pick_in_clutter_env_cfg.py       Scene cfg (2 cubes + bowl)
│   ├── joint_pos_env_cfg.py             Wiring SO-101 + actions
│   └── eval2.md                         Doc journey complète Eval 2
├── deploy/
│   ├── eval2_inference.py               ANCIEN — modulaire avec CNN
│   └── eval2_inference_e2e.py           NOUVEAU — image-in-policy
├── train/
│   ├── launch_act.sh                    ACT training (à adapter Eval 2)
│   └── smoke_test_act.py                Test pipeline lerobot+ACT — PASS
└── notes/
    ├── eval2_journey_and_pipeline.md    (deprecated, voir eval2.md)
    ├── eval2_ik_retrospective.md        Retrospective des tentatives IK
    └── eval2_ik_pipeline_continuation.md   ← CE FICHIER
```

---

## 8. Liens / refs

- Sanity check précédent : [`notes/sanity_results.md`](sanity_results.md)
- Spec TA : [`notes/project3_rl_final_details.md`](project3_rl_final_details.md)
- Doc Eval 2 complète : [`sim/eval2.md`](../sim/eval2.md)
- Retrospective IK : [`notes/eval2_ik_retrospective.md`](eval2_ik_retrospective.md)
- Repo de référence SO-101 sim : `MuammerBay/isaac_so_arm101` (Layer 2)
- URDF source : `TheRobotStudio/SO-ARM100`
- pytorch_kinematics : `UM-ARM-Lab/pytorch_kinematics`

---

## 9. TL;DR pour reprendre

1. **L'IK marche math** (FK match 0 mm), **convergence rapide** (~12 iters,
   < 1 mm).
2. **Contraintes critiques sur les wrists** :
   - `wrist_flex = π/2 − shoulder_lift − elbow_flex` (gripper toujours
     vertical)
   - `wrist_roll = 0` (jaws toujours alignées Y)
   Forcées au niveau action layer (pas seulement IK).
3. **State machine 9 phases** avec timeout fallback et snapshot LIFT
   depuis position gripper.
4. **GRASP_X_OFFSET=-0.015** pour clearance moving jaw.
5. **Tuning final** en cours sur `close_command_expr` et `DESCEND_HEIGHT`.
6. Une fois le smoke test ≥ 50 % succès → générer 10k démos →
   BC ACT → DAPG → deploy SO-101 réel.

**Si tu reprends, la seule chose qui peut foirer maintenant c'est la
physique du grasp (force, friction, contact)** — pas l'IK ni les
trajectoires. Itérer sur `close_command_expr` (env_cfg) et
`DESCEND_HEIGHT` (scripted_controller) jusqu'à ce que les épisodes
réussissent visuellement.

⚠️ **Ne PAS refaire** :
- Clamper `wrist_flex` dans le solveur (rend les joints instables).
- Ajouter de la friction agressive sur les cubes (combiné à un grip
  serré ça déclenche du flailing post-CLOSE).
