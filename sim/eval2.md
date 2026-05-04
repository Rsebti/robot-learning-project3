# Eval 2 — Targeted Pick-and-Place in Clutter (état du chantier)

> Doc de référence pour l'équipe (et leurs IA d'assistance) — explique tout ce
> qui a été mis en place pour l'Eval 2 du projet 3.
>
> **Statut au 2026-04-30** : structure complète, tâches `v0` et `v1` qui
> entraînent (PPO converge) sur RTX 5070 local, success rate encore faible,
> training v1.1 en cours après une passe de compliance avec le PDF des TAs.

---

## 1. Contexte projet (rappel TA spec)

Tâche **Eval 2 — 50 pts** ([notes/project3_rl_final_details.md](../notes/project3_rl_final_details.md)) :

- **Deux blocs de couleurs différentes** placés **adjacents** (flat cluster) sur la table
- Une **target color** est donnée en input
- Le robot doit identifier le bloc de la bonne couleur, le saisir, et le placer dans un **bowl** dont la position xyz est aussi donnée en input
- 5 rollouts × 10 pts. Succès = bon bloc dans le bowl + relâché
- **RL obligatoire** (pas de pure BC)
- Démos teleop expert autorisées comme replay buffer / warmstart
- Observations doivent venir de la **caméra wrist RGB** (visual policy ou perception modulaire)

**Spec environnement** (générale, applicable à toutes les évals) :
- Table gris clair `#B8ADA9`
- Blocs taille fixe, couleurs distinctes mais connues
- **Bowls placés à des positions randomisées** dans le repère robot
- Position des blocs randomisée

---

## 2. Stack technique

```
┌─────────────────────────────────────────────────────────────┐
│ NOTRE CODE (ce repo)                                         │
│   sim/eval2/                                                 │
│     - env config + custom rewards/obs/terminations           │
│     - launchers train/play/view                              │
└──────────────────┬───────────────────────────────────────────┘
                   │ depends on
┌──────────────────┴───────────────────────────────────────────┐
│ isaac_so_arm101 (https://github.com/MuammerBay/isaac_so_arm101)│
│   - SO-101 USD + ArticulationCfg                             │
│   - rsl_rl PPO training/play scripts (réutilisés tels quels) │
│   - tâches `lift` et `reach` qu'on a utilisées comme template│
└──────────────────┬───────────────────────────────────────────┘
                   │ depends on
┌──────────────────┴───────────────────────────────────────────┐
│ Isaac Lab 2.3.0 + Isaac Sim 5.1.0 (NVIDIA)                  │
│   - ManagerBasedRLEnv, scene/managers/sensors APIs          │
│   - PhysX GPU-accelerated simulation                         │
└──────────────────────────────────────────────────────────────┘
```

**Architecture en couches** (cf. [CLAUDE.md](../CLAUDE.md)) :
- **Layer 1** (jamais dans ce repo) : Isaac Sim + Isaac Lab — installés via `uv sync` dans le venv `isaac_so_arm101/.venv`
- **Layer 2** (jamais dans ce repo) : `MuammerBay/isaac_so_arm101` cloné en local
- **Layer 3** (CE qu'on partage dans ce repo) : `sim/eval2/` (et plus tard `sim/eval3/`)

---

## 3. Pattern Isaac Lab "Manager-Based"

Une tâche RL Isaac Lab = **1 SceneCfg + N managers indépendants** :

```
Task                   = SceneCfg + Managers
   ┌── ObservationManager  "que voit la policy ?"
   ├── ActionManager       "que peut faire la policy ?"
   ├── RewardManager       "comment on note la policy ?"
   ├── TerminationManager  "quand fin d'épisode ?"
   ├── EventManager        "quoi faire au reset ?"
   ├── CommandManager      "quel objectif on échantillonne ?"
   └── CurriculumManager   "comment on adapte le training ?"
```

Chaque manager est composé de **termes** (functions Python) qu'on configure
via un dataclass `@configclass`. Au runtime, le `ManagerBasedRLEnv` exécute
tous les termes actifs à chaque step.

C'est cette architecture qui explique la structure de fichiers de notre
package : un fichier env_cfg pour les managers, un sous-module `mdp/` pour
les fonctions custom (rewards, observations, terminations, events).

---

## 4. Arborescence du package `sim/eval2/`

```
sim/
├── __init__.py                              ← marque sim/ comme package
└── eval2/
    ├── __init__.py                          ← gym.register() de toutes nos tâches
    ├── pick_env_cfg.py                      ← classe parent v0 (1 bloc, simple)
    ├── pick_in_clutter_env_cfg.py           ← classe parent v1 (2 blocs colorés)
    ├── joint_pos_env_cfg.py                 ← spécialisations SO-101 v0 et v1
    ├── agents/
    │   ├── __init__.py
    │   └── rsl_rl_ppo_cfg.py                ← hyperparams PPO (rsl_rl)
    ├── mdp/
    │   ├── __init__.py                      ← réexporte tout (Isaac Lab std + nos fns)
    │   ├── observations.py                  ← block_position, target_color_one_hot, ...
    │   ├── rewards.py                       ← target_block_*_distance_tanh, ...
    │   ├── terminations.py                  ← success_target_block_in_bowl, ...
    │   └── events.py                        ← reset_target_color, reset_cluster_uniform
    └── scripts/
        ├── __init__.py
        ├── train.py                         ← lance un PPO training
        ├── play.py                          ← rejoue un checkpoint
        └── view.py                          ← inspecte la scène 3D sans training
```

**Convention** : tout ce qui est `*_env_cfg.py` est une config Isaac Lab
(scène + managers). Tout ce qui est dans `mdp/` est une **fonction Python pure**
qui calcule un terme (reward, obs, etc.) sur des tenseurs PyTorch.

---

## 5. Versions et leurs différences

### v0 — `Eval2-PickInBowl-v0` (smoke test)

**Objectif** : valider le pipeline end-to-end avec la tâche la plus simple possible.

- 1 seul bloc (le `dex_cube` standard d'Isaac Sim, 2.5 cm scaled)
- 1 bowl plat (cylindre kinematic) à position fixe
- Observations : joint_pos (6) + joint_vel (6) + block_pos (3) + bowl_pos (3) + last_action (6) = **24D**
- Rewards : reach + lift + place + smoothness, sans goal-conditioning
- Pas de target color
- 6 reward terms

**Status** : ✅ entraîne, mean reward 70 à iter 499, success rate 0.25%
(le bowl plat fait que les blocs glissent et tombent)

### v1 — `Eval2-PickInClutter-v1` (la vraie tâche Eval 2)

**Objectif** : la tâche Eval 2 telle que demandée par les TAs.

Ajouts vs v0 :
- **2 blocs colorés** (rouge + bleu) côte à côte, primitives `CuboidCfg`
- **Target color** échantillonnée à chaque reset (0=rouge, 1=bleu), stockée dans `env.target_color`
- **One-hot 2D** ajouté aux observations (29D total)
- Rewards **target-aware** : `target_block_*_distance_tanh`, `target_block_is_lifted`, `target_block_in_bowl`
- Reward **distractor_disturbed** (poids -5.0) : pénalise le déplacement du bloc non-cible
- **Bowl en 5 primitives** (1 fond + 4 murs) qui forme un container ouvert (les blocs ne glissent plus)
- 8 reward terms, 4 termination terms, 4 event terms (dont `reset_target_color`)

**Status** : ✅ entraîne, success rate 1.30% à iter 499 (mieux que v0 malgré tâche plus dure)
**Constat** : `lifting_target` plafonne à 0.20 — la condition de "soulevé" est trop dure

### v1.1 — passe de compliance PDF + fixes physiques

**Objectif** : aligner v1 strictement avec la spec TA + résoudre les bugs
qui empêchaient la convergence.

Modifications majeures :
| Aspect | v1 | v1.1 |
|---|---|---|
| **Couleur table** | gris foncé (USD Isaac default) | **`#B8ADA9`** primitive (spec exacte) |
| **Taille table** | ~60 cm × 1 m (USD) | **80 cm × 1 m**, repositionnée pour que le robot soit ON the table |
| **Taille bowl** | 10 cm de côté | **12 cm** (≥ spec, marge pour ~12 cubes) |
| **Position bowl** | (0.30, -0.20), 36 cm du robot (limite reach) | **(0.20, -0.15), 25 cm** du robot |
| **Hauteur murs bowl** | 4 cm | 2.5 cm (= hauteur cube) |
| **Taille cubes** | 2.5 cm | **2 cm** (= vrais cubes en bois) |
| **Cubes adjacents** | 3 cm de gap | **collés** (PDF: "adjacent / flat cluster") |
| **Reset cluster** | 2 events indépendants | 1 event sync via `reset_cluster_uniform` |
| **Lifting reward** | seuil unique 5 cm | **2 stages** : low (2.5 cm, w=10) + high (5 cm, w=10) |
| **`target_to_bowl` gating** | h≥5 cm | h≥2.5 cm (déclenche plus tôt) |
| **`success_bonus` poids** | 50 | **100** |
| **PhysX `total_aggregate_pairs_capacity`** | 16K | **64K** (corrige erreurs `missing interactions`) |

**Status** : ✅ entraîne sainement. À iter 193/1000 sur RTX 5070 :
- mean reward 22.99 (vs 3.66 dans v1.0 à iter 499)
- `lifting_target_low` 1.71 (vs 0 dans v1.0/1.1 cassée)
- `block_*_dropped` 1-2% (vs 50% en v1.1 cassée à cause des cubes au bord de table)
- success rate 2.19% (vs 1.30% v1.0)

### v1.2 — randomization de la position du bowl

**Objectif** : rendre le bowl mobile à chaque reset, conforme au PDF :
> "Bowls placed at randomized positions in the robot base frame"

Implémentation :
- Nouvel event `randomize_bowl_position` qui réutilise `reset_cluster_uniform`
  pour appliquer le même décalage xy aux **5 primitives** du bowl
  (floor + 4 walls), préservant la forme.
- Range : ±4 cm en x, ±2 cm en y. Le y est volontairement modeste pour
  éviter que le bowl chevauche le cluster (cluster ±5 cm en y peut
  descendre à y=-0.05 ; bowl à y=-0.13 + wall_top 0.068 = -0.062, marge
  ~1 cm).

À élargir progressivement quand on aura validé que la policy gère le
goal-conditioning sur la position du bowl.

**Status** : ✅ entraîné. **success rate 7.46% à iter 999** sur 4096 envs
RTX 5070. C'est notre meilleure baseline. Avec plus d'iter (3000+) et/ou
boost du reward `target_to_bowl_fine`, on devrait dépasser 15-25 %.

### v2 — caméra wrist (RGB) ajoutée à la scène

**Objectif** : permettre la perception visuelle des blocs depuis le poignet
du SO-101, pour un déploiement réel.

Implémentation :
- `PickInClutterSceneCfgWithCam(PickInClutterSceneCfg)` ajoute un champ
  `wrist_cam: CameraCfg`.
- `Eval2PickInClutterEnvCfg_v2(Eval2PickInClutterEnvCfg_v1)` plug la cam :
  - prim_path : `{ENV_REGEX_NS}/Robot/gripper_link/wrist_cam`
    (le `gripper_link` = "wrist_roll_link" du doc TA dans cet URDF)
  - 240×320 RGB à 10 Hz (`update_period=0.1`)
  - Intrinsèques Isaac Lab manipulation tutorials :
    `focal_length=24.0`, `clipping_range=(0.1, 1e5)`
  - Offset calibré interactivement dans Property panel d'Isaac Sim :
    - `Translate (-0.02208, 0.05825, 0.03013)` m
    - `Orient XYZ (-15.544, -9.931, -90.069)` deg → quaternion
      `(0.70586, -0.03453, -0.15594, -0.69009)`
    - `convention="opengl"` (pas de transformation Isaac Lab appliquée,
      les valeurs vont directement dans `xformOp:orient`)
  - `ee_frame.debug_vis = False` en v2 pour ne pas polluer la vue cam

**Status** : ✅ caméra présente, calibrée pour une scan pose donnée. En
basculant le viewport sur `wrist_cam` (dropdown caméra en haut à gauche
d'Isaac Sim), on voit les 2 cubes + bowl + table.

⚠️ **La calibration de la caméra est valide pour UNE pose particulière du
robot** (la "scan pose" hardcodée dans `view.py` et `capture_dataset.py`).
Si on change la pose du robot, le `Translate`/`Orient` de la caméra doivent
être re-tunés en interactif dans le Property panel.

### v2 perception — module CNN (en cours)

**Objectif** : un CNN qui mappe l'image RGB de la caméra wrist aux
positions 3D `(x, y, z)` des blocs rouge et bleu en repère robot.
Approche **B (xyz directs)** au lieu de A (pixels + back-project) parce
qu'elle évite toute calibration de plan table à l'inférence.

Files (`sim/eval2/perception/`) :
| Fichier | Rôle |
|---|---|
| `model.py` | `ColorBlockCNN` — petit CNN ~250k params. 3 conv + AdaptiveAvgPool(4×4) + 2 FC. Sortie 6 floats (xyz_red + xyz_blue) en mètres. |
| `dataset.py` | `BlockPositionDataset` — wrappe le `.pt` produit par le capture. Brightness jitter, **pas de h-flip** (3D targets ne se flippent pas trivialement). |
| `capture_dataset.py` | **(déprécié)** Capture statique : robot figé en scan pose, snapshot à chaque reset. Donnait un dataset trop monotone — un seul viewpoint, peu représentatif du déploiement. |
| **`capture_with_policy.py`** | **(préféré)** Charge le checkpoint v1.2 et roule la policy dans l'env v2 ; capture image + xyz GT à chaque step. Le CNN voit la **vraie distribution de viewpoints** qu'aura le déploiement. |
| `train.py` | Training MSE, Adam, train/val 80/20, early stop. Reporte MAE par coordonnée en cm. |
| `inference.py` | `PerceptionPipeline` — charge un checkpoint, image → xyz. Pour le deploy. |
| `README.md` | Walkthrough du workflow. |

**Itérations sur l'architecture CNN** :
1. Premier essai avec `AdaptiveAvgPool2d(1)` (Global Avg Pool) → val_mae stuck à 2.5 cm = MAD d'une distribution uniforme. **Bug : Global Avg Pool détruit l'info spatiale**, le CNN apprenait juste à prédire la moyenne.
2. Deuxième essai sans BatchNorm + dataset partagé entre train/val (Subset bug) → val explosé à 900 m.
3. Version actuelle : `AdaptiveAvgPool2d((4, 4))` qui préserve un grid 4×4 de features → l'info spatiale "où est le rouge" / "où est le bleu" est exploitable par le head FC.

**Itérations sur la stratégie de capture** :
1. **`capture_dataset.py`** — robot figé en scan pose. Bug : la scan pose
   hardcodée plaçait le gripper hors d'alignement avec les positions de
   spawn des cubes ; les images du wrist cam montraient la table sans
   les cubes. Tentatives multiples de re-tuning du couple
   (pose joints + offset caméra) — abandonnées.
2. **`capture_with_policy.py`** ← stratégie actuelle. Au lieu de chercher
   une "bonne pose" arbitraire, on **roule la policy v1.2 entraînée**
   et on enregistre les paires (image, GT) pendant que le bras se
   déplace. Avantages :
   - Les viewpoints couvrent **toute la trajectoire de pick** (approche,
     descente, lift, transport, drop) → CNN robuste.
   - Distribution sim ≃ distribution déploiement (la vraie policy
     bougera le bras de la même façon).
   - Pas besoin de calibrer une scan pose.

### v1.3 — convergence run overnight (en cours, mise à jour 2026-05-04 23h)

**Diagnostic post-deploy** : en lançant le checkpoint v1.2 dans le pipeline complet
(`deploy/eval2_inference.py`), on a découvert que **la policy v1.2 n'est PAS
convergée**. Le checkpoint contient un `std` énorme (~3.7) — le PPO explorait
encore massivement à iter 999. Les "7.46% success" du training sont obtenus
PAR HASARD via l'exploration aléatoire (mean × bruit), pas par une policy
qui a appris la tâche. En mode déterministe (act_inference, sans bruit) ou
même en sampling, le success rate retombe à 0%.

**Cause racine** : sans pénalité explicite sur la magnitude des actions, le
gradient PPO a poussé `mean` vers ±15 (= joints saturés à ±7.5 rad après le
scale 0.5). La reward landscape ne pouvait plus distinguer les bonnes des
mauvaises actions → `std` n'a jamais convergé.

**Modifs config v1.3** :
| Aspect | v1.2 | v1.3 |
|---|---|---|
| `target_to_bowl_fine` weight | 5 | **25** (forcer placement précis) |
| `success_bonus` weight | 100 | **200** (signal de réussite plus fort) |
| `action_rate_l2` weight | -1e-4 | **-1e-3** (smoothness 10× plus forte) |
| **`action_l2_norm` reward (NEW)** | absent | **-1e-2** (empêche actions saturées) |
| `init_noise_std` | 1.0 | **0.5** (start moins explorateur) |
| `entropy_coef` | 0.006 | inchangé (défaut stable) |
| `max_iterations` | 1000 | **20 000** (8-10h overnight) |
| `save_interval` | 100 | 500 (40 checkpoints sur 20k) |
| Bowl rand | ±4 cm x, ±2 cm y | inchangé |

**Cible** : success rate sim ≥ 70 %, idéalement 90 %+, avec `std` < 0.5
en fin de training (= policy convergée, déployable en mode déterministe).

**Commande** :
```powershell
uv run python -m sim.eval2.scripts.train --task Eval2-PickInClutter-v1 \
    --headless --num_envs 4096 --max_iterations 20000
```

(On entraîne sur **v1**, PAS v2 : v2 ne diffère que par la cam dans la scène,
qui ne change rien à la policy state-based mais ralentit le training de 3-5×.)

**Note importante pour la suite** : si v1.3 converge, on devra re-capturer
le dataset de perception avec ce nouveau checkpoint (la distribution de
viewpoints sera très différente : moins chaotique, plus représentative du
vrai déploiement). Le CNN actuel a été entraîné sur des viewpoints issus
de la policy v1.2 chaotique → biais hérité.

---

**Status au 2026-05-04 — v2 perception ENTRAÎNÉE ✅** :

- ✅ `capture_with_policy.py` écrit. Charge le checkpoint v1.2 hardcodé :
  `C:/Users/user/Desktop/MA2/isaac/isaac_so_arm101/logs/rsl_rl/eval2_pick_in_bowl/2026-04-30_15-53-41/model_999.pt`
- ✅ Bug "5 images identiques" corrigé : on appelle
  `wrist_cam.update(dt=1.0)` après chaque `env.step` pour forcer le
  refresh du buffer caméra (sinon `update_period=0.1s` >
  `step_dt=0.02s` rend 4 frames sur 5 identiques).
- ✅ Dataset 5000 samples capturé pendant les rollouts de la policy
  v1.2 dans v2. Le dataset couvre des viewpoints variés : approche du
  cluster, descente, lift, transport vers bowl, drop. Quelques cubes
  visibles "en l'air" (lifted) dans les frames de transport.
- ✅ CNN entraîné 30 epochs (~2 min sur RTX 5070).

**Résultats du training** :

| Métrique | Valeur |
|---|---|
| Best val MSE | **0.00024** |
| Best val MAE | **0.87 cm** moyen |
| `x_red`  | ~1.0 cm |
| `y_red`  | ~1.3-1.7 cm |
| `z_red`  | ~0.6 cm |
| `x_blue` | ~0.85 cm |
| `y_blue` | ~0.8-1.2 cm |
| `z_blue` | ~0.6 cm |
| Train vs val gap | < 30 % | (pas d'overfitting) |
| Wall-clock | 132 s |

Cible (val_mae < 1.5 cm) **atteinte**. Pour des blocs de 2 cm, l'erreur
de 0.87 cm laisse à la pince une marge raisonnable au grasp. Le
checkpoint est à `sim/eval2/perception/checkpoint.pt` (gitignore).

### Reste à faire après v2 perception

1. **Validation visuelle** (~10 min, optionnel) : utiliser
   `inference.py` sur quelques images de test, comparer la prédiction
   au GT. Surtout sur les frames "lifted" (bloc en l'air pendant le
   transport) pour vérifier que le CNN gère la profondeur — `z` est
   facile sur les frames "table", il faut tester les autres.
2. **Deploy script réel** : `deploy/eval2_inference.py` à écrire.
   Structure :
   - Lit la wrist cam réelle via OpenCV/lerobot
   - Passe l'image dans `PerceptionPipeline.predict(rgb)` →
     `(red_xyz, blue_xyz)` en repère robot
   - Concatène l'observation 29-D :
     `joint_pos + joint_vel + perception_red_xyz + perception_blue_xyz
      + bowl_xyz_arg + target_color_arg + last_action`
   - Inférence policy v1.2 → action
   - Envoie aux servos Feetech à 30 Hz
   - CLI args `--bowl_x --bowl_y --bowl_z --target_color` fournis par
     les TAs au moment de l'éval.
3. **Domain randomization (v3)** pour le sim-to-real : varier couleurs,
   textures, lumière, frottements pendant le training perception.
   Sans ça, le CNN entraîné sur sim parfait pourrait sous-performer
   sur les vrais blocs en bois avec lumière différente.
4. **Améliorer la perception** (optionnel, après validation au robot
   réel) : capturer plus de samples (10-20k), data augmentation plus
   agressive (color jitter, gaussian noise), ou entraîner un modèle
   plus gros. Sans urgence tant que `val_mae` < 1 cm en sim.

**Alternative envisagée** (pas faite, n'a pas été nécessaire) : faire
varier légèrement la pose du robot à chaque sample pour augmenter la
diversité de viewpoints. Le rollout policy `capture_with_policy.py`
nous a déjà donné cette diversité naturellement.

---

## 6. Tâches gym enregistrées

```python
Eval2-PickInBowl-v0          # v0 training (4096 envs)
Eval2-PickInBowl-Play-v0     # v0 visu (50 envs, sans bruit)
Eval2-PickInClutter-v1       # v1 training (4096 envs) ← BASELINE PROUVÉE (7.46% succ)
Eval2-PickInClutter-Play-v1  # v1 visu
Eval2-PickInClutter-v2       # v1 + caméra wrist (training plus lent)
Eval2-PickInClutter-Play-v2  # v2 visu, à utiliser avec --enable_cameras
```

L'enregistrement se fait dans `sim/eval2/__init__.py` à l'import du module
(side-effect d'`import sim.eval2`).

---

## 7. Setup local (recap)

### Prérequis
- Isaac Sim 5.1 + Isaac Lab 2.3 installés (cf. [`notes/isaac_lab_setup.md`](../notes/isaac_lab_setup.md))
- Repo `MuammerBay/isaac_so_arm101` cloné quelque part hors de ce repo
- `uv sync` exécuté dans `isaac_so_arm101/`

### Brancher notre package au venv

```powershell
cd C:\path\to\isaac_so_arm101
uv pip install -e C:\path\to\robot-learning-project3
```

→ notre `sim.eval2` devient importable depuis le venv d'Isaac Sim.

---

## 8. Comment lancer

### Visualiser l'env sans entraîner

```powershell
uv run python -m sim.eval2.scripts.view --task Eval2-PickInClutter-v1 --num_envs 4
```

→ ouvre une fenêtre Omniverse, robot immobile (zero actions), tu inspectes
la scène (table, bowl, blocs, lumière).

### Entraîner

```powershell
# Smoke test rapide (3 iter, 32 envs)
uv run python -m sim.eval2.scripts.train \
    --task Eval2-PickInClutter-v1 \
    --headless --num_envs 32 --max_iterations 3

# Full training
uv run python -m sim.eval2.scripts.train \
    --task Eval2-PickInClutter-v1 \
    --headless --num_envs 4096 --max_iterations 1000
```

Logs dans `outputs/rsl_rl/eval2_pick_in_bowl/<timestamp>/`.

### Rejouer un checkpoint

```powershell
uv run python -m sim.eval2.scripts.play \
    --task Eval2-PickInClutter-Play-v1 \
    --checkpoint outputs/.../model_999.pt
```

---

## 9. Métriques à suivre pendant le training

PPO log toutes les itérations. Termes clés à surveiller :

| Terme | Signification | Bon signe |
|---|---|---|
| `Mean reward` | reward moyen agrégé | monte régulièrement |
| `Episode_Reward/reaching_target` | distance gripper-bloc cible (tanh) | 0 → 0.5+ très tôt (iter 100) |
| `Episode_Reward/lifting_target_low` | bloc cible soulevé > 2.5 cm | 0 → 5+ vers iter 100-200 |
| `Episode_Reward/lifting_target_high` | bloc cible soulevé > 5 cm | suit après low |
| `Episode_Reward/target_to_bowl_coarse` | bloc cible proche du bowl | monte vers iter 300+ |
| `Episode_Reward/success_bonus` | bloc cible **dans** bowl | rare, doit > 0 vers iter 500+ |
| `Episode_Reward/distractor_disturbed` | doit rester ≈ 0 | si grimpe → policy pousse le mauvais bloc |
| `Episode_Termination/success` | taux de succès | objectif > 5% à long terme |

---

## 10. Écarts connus vs spec PDF (à fixer)

### v1.2 — Bowl position randomisée ✅ IMPLÉMENTÉ
PDF : *"Bowls placed at randomized positions in the robot base frame"*.

**Implémentation** : nouvel event `randomize_bowl_position` dans la `EventCfg`
de `pick_in_clutter_env_cfg.py`. Réutilise `reset_cluster_uniform` avec les
5 noms d'assets du bowl. Range actuel : ±4 cm en x, ±2 cm en y.

**À faire** : entraîner et valider que la policy adapte sa trajectoire à la
position du bowl. Si le succès chute trop, élargir le range progressivement
(curriculum) ou augmenter le poids de `target_to_bowl_*`.

### v2 — Visual observation
PDF : *"The policy must operate on visual observation of blocks"*.
Notre v1.x : on triche avec les positions ground-truth (gratuites, données par
PhysX). C'est un raccourci pour valider la pipeline RL d'abord.

**Approche prévue (modulaire, autorisée par les TAs)** :
- Ajouter une `CameraCfg` montée sur `gripper_link` (84×84 RGB) dans la SceneCfg
- Entraîner ou pré-construire un module de **perception** (color filter HSV
  ou détecteur YOLO) qui prend l'image et sort `block_red_xy`, `block_blue_xy`
- La policy RL reste state-based mais sa source des positions devient la
  perception au lieu du ground-truth
- Avantage : sim-to-real plus simple, perception et policy testables séparément

### v3 — Domain randomization
Pour le sim-to-real :
- Couleurs des blocs randomisées dans une plage (au moins 2 paires de couleurs)
- Textures table légèrement perturbées
- Lumière (intensité, direction) randomisée
- Frottements et masses légèrement variables
- Bruit caméra
- Poses initiales du bras randomisées

---

## 11. Choix de design notables

### Pourquoi un bowl en primitives plutôt que USD ?
Le seul USD bowl trouvé dans Isaac Sim (`YCB/024_bowl.usd`) est un **mesh
décoratif sans `RigidBodyAPI`** → Isaac Lab refuse de le manipuler comme
`RigidObjectCfg`. Construire le bowl à partir de primitives `CuboidCfg`
kinematics donne une `RigidBodyAPI` propre à coût négligeable.

### Pourquoi une table en primitive plutôt que `SeattleLabTable.usd` ?
Pour avoir la **couleur exacte `#B8ADA9`** demandée par le PDF. L'USD a sa
propre material binding sur des sub-prims qui n'est pas overridable
fiable via `UsdFileCfg.visual_material`. Une primitive cuboid donne un
contrôle total. Conséquence : la table n'a pas de pieds, c'est un slab
flottant — ce qui n'a aucun impact sur la physique ou la policy.

### Pourquoi 4096 envs en parallèle ?
C'est la valeur par défaut d'Isaac Lab pour les tâches manipulation, et
notre RTX 5070 (12 GB VRAM) la tient sans OOM. Sur ce GPU on tourne à
~100-200k steps/s en physique pure (sans caméra). Pour v2 avec caméra,
il faudra probablement baisser à 1024 ou 2048 envs.

### Goal-conditioning : implémentation
La target color n'est pas un `CommandManager` standard (qui veut des
valeurs continues). On stocke l'index discret (0/1) dans
`env.target_color` (un buffer attribut ajouté à l'env), mis à jour par
l'event `reset_target_color` à chaque reset. Les obs/rewards/terminations
y accèdent directement.

---

## 12. Reproduire la baseline v1.2

Tout le code, la config et les hyperparams sont dans le repo : pour retomber
sur la baseline actuelle, **il suffit de re-entraîner**. ~25 min sur RTX 5070,
±10 % sur les métriques finales à cause de la stochasticité PPO.

Commande exacte qui a produit la baseline actuelle :
```powershell
uv run python -m sim.eval2.scripts.train \
    --task Eval2-PickInClutter-v1 \
    --headless --num_envs 4096 --max_iterations 1000 \
    --seed 42
```

Hyperparamètres dans `sim/eval2/agents/rsl_rl_ppo_cfg.py` (PPO standard, MLP 256-128-64, lr 1e-4 adaptive, 24 steps × 4096 envs par iter).

Métriques attendues à iter 999 (à ±10 % près à cause de la stochasticité PPO) :
- Mean reward ~ 119
- `lifting_target_low` ~ 6.8
- `target_to_bowl_coarse` ~ 5.9
- `success_bonus` ~ 0.014
- success rate ~ 7.5 %

Si tu vois des chiffres très différents (genre `lifting_target_low = 0`), c'est probablement un bug d'environnement (PhysX, table edge, etc.) — relire la section 5 sur les pièges trouvés en v1.1.

### Logs et artefacts

Pendant le training, rsl_rl écrit dans :
```
<isaac_so_arm101>/logs/rsl_rl/eval2_pick_in_bowl/<timestamp>/
├── model_100.pt            ← checkpoints tous les 100 iter
├── model_200.pt
├── ...
├── model_999.pt
├── params/
│   └── env.yaml            ← config env figée pour ce run
├── git/
│   └── isaac_so_arm101.diff
└── events.out.tfevents.*   ← TensorBoard
```

Pour visualiser les courbes :
```powershell
tensorboard --logdir <isaac_so_arm101>/logs/rsl_rl/eval2_pick_in_bowl
```

---

## 13. Que faire ensuite (par où continuer)

5 chantiers possibles, classés par priorité / ROI. Tu peux les attaquer dans
n'importe quel ordre, ils sont largement indépendants.

### Piste A — Pousser v1.2 plus loin (le plus simple, ~30 min - 1 h)

Le run baseline est arrêté à 1000 iter, encore en phase d'apprentissage actif
(`action_noise_std` à 3.7 = exploration en cours). Trois leviers à essayer :

**A.1 — Plus d'itérations** :
```powershell
uv run python -m sim.eval2.scripts.train \
    --task Eval2-PickInClutter-v1 --headless --num_envs 4096 \
    --max_iterations 5000
```
Si la courbe `success_bonus` continue de monter → on a juste manqué de temps.

**A.2 — Boost du reward `target_to_bowl_fine`** :
Dans `sim/eval2/pick_in_clutter_env_cfg.py`, augmenter le `weight` de
`target_to_bowl_fine` de **5 à 15**. Force la policy à se concentrer sur la
descente précise. Re-train.

**A.3 — Curriculum sur le bowl rand** :
Dans `EventCfg.randomize_bowl_position`, démarrer avec un range étroit
(`x: (-0.01, 0.01), y: (-0.005, 0.005)`), entraîner 500 iter, puis élargir
progressivement. Plus stable que d'attaquer la randomization complète d'un coup.

### Piste B — Implémenter v2 (caméra + perception modulaire) ← le gros morceau

C'est le **dernier vrai écart au PDF** ("policy must operate on visual observation
of blocks"). Approche modulaire (autorisée par les TAs) : la policy reste
state-based, mais les positions des blocs viennent d'un module de perception
au lieu du ground-truth.

**Étapes** :

1. **Ajouter une `CameraCfg` à la scène**
   Dans `sim/eval2/joint_pos_env_cfg.py`, à côté de `ee_frame`, ajouter :
   ```python
   from isaaclab.sensors import CameraCfg
   self.scene.wrist_cam = CameraCfg(
       prim_path="{ENV_REGEX_NS}/Robot/gripper_link/wrist_cam",
       update_period=0.1,  # 10 Hz
       width=84, height=84,
       data_types=["rgb"],
       spawn=sim_utils.PinholeCameraCfg(
           focal_length=24.0, focus_distance=400.0,
           horizontal_aperture=20.955,
       ),
       offset=CameraCfg.OffsetCfg(
           pos=(0.05, 0.0, 0.0),  # à ajuster avec la pose réelle de la cam
           rot=(0.5, -0.5, 0.5, -0.5),
       ),
   )
   ```
   ⚠️ Bug Blackwell connu : utiliser `Camera` standard, pas `TiledCamera`
   (cf. `notes/isaac_lab_setup.md`).

2. **Écrire un module de perception** dans `sim/eval2/perception/`
   - Le plus simple : **HSV color filter** (OpenCV). Convertit RGB en HSV,
     seuille sur les ranges rouge et bleu, calcule le centroid 2D pixel.
   - Pour passer du pixel xy à xyz monde : **calibration extrinsèque
     caméra-bras** (à mesurer une fois physiquement avec un ChAruco board).
   - Alternative plus robuste : YOLOv8-tiny entraîné sur des renders sim.

3. **Adapter les observations** dans `sim/eval2/mdp/observations.py`
   Pendant le training en sim, on peut continuer à donner le ground-truth
   (gratuit). On ajoute juste l'image en plus pour entraîner la perception
   séparément. Au déploiement, on remplacera le ground-truth par la sortie
   de la perception (mêmes shapes, mêmes ranges).

### Piste C — Domain randomization (v3, sim-to-real)

À faire en parallèle de la perception. Dans `EventCfg`, ajouter des events
qui s'exécutent à chaque reset :
- `randomize_block_colors` : tirer la teinte des deux blocs dans des plages
  rouge/bleu plus larges
- `randomize_table_friction` : varier le coefficient de frottement table-bloc
- `randomize_lighting` : varier intensité et direction du `DomeLight`
- `randomize_robot_initial_pose` : petite noise sur la pose home

→ La policy entraînée sur ces variations devient robuste au transfert réel.

### Piste D — Écrire le script de deploy au robot réel

Voir section 15 pour le scaffolding. Tâches concrètes :
1. Créer `deploy/eval2_inference.py` (s'inspirer de `deploy/inference.md` du
   sanity check)
2. Ajouter les flags CLI `--bowl_x --bowl_y --bowl_z --target_color --policy`
3. Brancher la perception (à la place du ground-truth)
4. Tester sur le SO-101 réel sur 5 rollouts

### Piste E — Faire Eval 1 en parallèle (gains rapides, 50 pts)

Eval 1 = single bloc + bowl, BC autorisé. Tu réutilises **exactement** le
pipeline du sanity check, juste avec :
- **50-100 démos teleop** au lieu de 20, en variant la position du bloc
- Re-train ACT
- Deploy

C'est plus simple que de tout finir Eval 2, et ça vaut autant de points.
**Si tu n'as qu'une journée au robot, fais Eval 1 d'abord.**

---

## 14. Liens utiles

- TA spec PDF : [`notes/project3_rl_final_details.md`](../notes/project3_rl_final_details.md)
- Plan de phase générale Eval 2 : [`notes/eval2_plan.md`](../notes/eval2_plan.md)
- Setup Isaac Lab : [`notes/isaac_lab_setup.md`](../notes/isaac_lab_setup.md)
- Repo SO-101 Isaac Lab : https://github.com/MuammerBay/isaac_so_arm101
- Doc Isaac Lab : https://isaac-sim.github.io/IsaacLab/
- Doc rsl_rl : https://github.com/leggedrobotics/rsl_rl

---

## 15. TL;DR pour quelqu'un qui débarque

État au **2026-05-04 fin de soirée** :

1. **v0, v1, v1.1, v1.2 RL state-based** : tous fonctionnels mais le
   **checkpoint v1.2 N'EST PAS DÉPLOYABLE** — diagnostiqué via le
   pipeline complet `deploy/eval2_inference.py`. Le success rate
   "7.46 %" reporté pendant le training vient de l'exploration
   stochastique (PPO sample mean+noise avec std≈3.7), pas d'une vraie
   policy convergée. En mode déploiement (déterministe ou même
   sampling), le success rate retombe à 0 % parce que la policy n'a
   jamais appris à exploiter — elle ne fait qu'explorer.
2. **v1.3 EN TRAINING (overnight, ~20k iter)** : nouvelle config qui
   ajoute une pénalité `action_l2` pour empêcher les actions saturées,
   booste les rewards de placement précis, et tourne 20× plus long.
   Cible : `std` < 0.5 et success rate ≥ 70 % en mode déterministe.
3. **v2 (caméra wrist)** : caméra présente dans la scène, calibrée. Tâches
   `Eval2-PickInClutter-{,Play-}v2` enregistrées. **Lance toujours avec
   `--enable_cameras`**.
4. **Module perception CNN ENTRAÎNÉ ✅** : `ColorBlockCNN` à val_mae 0.87 cm
   sur le dataset capturé. **MAIS** : ce dataset vient de la policy v1.2
   chaotique. Si v1.3 converge, **il faudra recapturer + retrain le CNN**
   avec la nouvelle distribution de viewpoints (plus propre, plus
   représentative).
5. **`deploy/eval2_inference.py` ÉCRIT ✅** : pipeline complet qui charge
   policy + CNN, lit la cam wrist, substitue la perception aux positions
   GT dans l'obs, fait tourner la policy. A permis le diagnostic v1.2.
6. **Pipeline complet à valider quand v1.3 sera prête** :
   - Récupérer le meilleur checkpoint v1.3
   - Recapture dataset perception avec `capture_with_policy.py`
   - Retrain CNN
   - Re-test `deploy/eval2_inference.py` avec et sans `--use_ground_truth`
7. **Reste à faire après ça pour points Eval 2** :
   - **Domain randomization v4** (couleurs / textures / lumière) pour
     sim-to-real
   - **Test sur le vrai SO-101** : 5 rollouts au laboratoire

## 16. Procédure deploy au robot réel (Eval 2 day)

Le PDF dit explicitement que la position du bowl et la couleur cible sont **fournies
en input** par les TAs — pas à détecter visuellement. Le seul truc à percevoir, ce
sont les positions des blocs depuis la wrist cam.

```
1. Les TAs placent le bowl + 2 blocs colorés sur la table
2. Les TAs mesurent / annoncent (bowl_x, bowl_y, bowl_z) en frame robot
3. Les TAs annoncent la couleur cible (e.g. "rouge")
4. On lance :
   $ python deploy_eval2.py \
       --bowl_x=0.20 --bowl_y=-0.18 --bowl_z=0.02 \
       --target_color=red \
       --policy=Rsebti/projet3-eval2-v1.x
5. Le script :
   - Lit joint_pos / joint_vel des servos Feetech à 30 Hz
   - Capture l'image wrist cam, détecte block_red_xy et block_blue_xy via le
     module de perception (HSV color filter ou détecteur entraîné)
   - Concatène l'observation : joint_pos + joint_vel + block_red + block_blue
     + bowl (constant, fourni en arg) + target_color (constant, fourni)
     + last_action
   - Inférence policy → action
   - Envoie l'action aux servos
6. Boucle jusqu'au succès ou time-out
```

Le `deploy_eval2.py` reste à écrire — il sera structurellement identique au
`deploy/inference.md` du sanity check, en remplaçant l'ACT par notre policy
PPO Eval 2 et en injectant les inputs goal-conditionnés.
