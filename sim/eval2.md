# Eval 2 — Targeted Pick-and-Place in Clutter (état du chantier)

> Doc de référence pour l'équipe (et leurs IA d'assistance) — explique tout ce
> qui a été mis en place pour l'Eval 2 du projet 3.
>
> **Statut au 2026-04-30** : structure complète, tâches `v0` et `v1` qui
> entraînent (PPO converge) sur RTX 5070 local, success rate encore faible,
> training v1.1 en cours après une passe de compliance avec le PDF des TAs.
>
> **Statut au 2026-05-05 (mise à jour décisive)** : après plusieurs jours de
> debug du scripted IK controller (jamais arrivé à grasp fiable à cause des
> contraintes 5-DoF du SO-101), bascule de stratégie. Voir
> [section 0 — Décision archi](#0-décision-archi-2026-05-05--end-to-end-vision--option-a-dapg).

---

## 0. Décision archi (2026-05-05) — end-to-end vision + Option A DAPG

### Ce qu'on abandonne

- **Approche modulaire** (image → CNN perception → block_xyz → policy
  state-based MLP) : le CNN entraîné (val_mae 0.87 cm) reste fonctionnel
  mais **n'est plus dans le chemin principal**. Il peut servir de
  pretrained warmstart pour le visual encoder de la policy end-to-end,
  rien de plus.
- **Scripted IK controller pour générer des démos en sim** : math correcte
  (FK→IK→FK 0.0001 mm) mais le SO-101 5-DoF + limites articulaires serrées
  font que le runtime sature ou se contorsionne. Voir
  [`notes/eval2_ik_retrospective.md`](../notes/eval2_ik_retrospective.md)
  pour l'audit complet. **Code conservé** dans `sim/eval2/bc/` mais pas
  utilisé pour la suite.
- **PPO from-scratch state-based** : v1.0–1.4 ont plafonné à 2–7 % succès,
  classique du local optimum sans démos. On arrête de patcher.

### Ce qu'on fait à la place — Option A : BC + DAPG end-to-end vision

```
[Étape 1 — au labo, fait par Federico]
   50–100 démos teleop Eval 2 (2 cubes colorés, bowl, target_color, bowl_xyz
   variés). Méthode grasping standardisée : top-down vertical, approche au-
   dessus du cube, descente, fermeture, lift, transport, drop, retreat.

[Étape 2 — au PC]
   BC pretrain ACT end-to-end (image wrist cam + joint_pos + target_color +
   bowl_xyz → action). Réutilise l'archi qui a fait 5/5 sur le sanity check.

[Étape 3 — sur Brev H100 (~$50-100 budget)]
   PPO finetune en sim avec image observation + auxiliary BC loss (DAPG).
   Domain randomization activée (couleurs, lumière, textures, frottements).
   Sim env : Eval2-PickInClutter-v2 (avec wrist cam).

[Étape 4 — au labo]
   Deploy au SO-101 réel via deploy/eval2_inference.py adapté pour image-
   in-policy (au lieu de perception module).
```

### Pourquoi Option A et pas autre chose

| Option considérée | Verdict |
|---|---|
| BC seul (ACT) sans RL | Spec TA dit "RL mandatory" — on perd des points |
| Pure PPO from-scratch image | PPO state-based plafonne déjà, image c'est pire |
| Diffusion policy + RL | Trop ambitieux pour la deadline (~mid-mai) |
| Pretrained encoder freeze + RL léger | Encoder peut être trop rigide |
| **BC + DAPG end-to-end image** ⭐ | **Standard littérature manipulation, conforme spec, faisable budget** |

### Ce qui change concrètement dans le repo

- `sim/eval2/perception/` : **mostly dead code**. Garder les fichiers en
  référence + warmstart potentiel mais ne pas continuer à itérer dessus.
- `sim/eval2/bc/` (scripted IK + analytical_ik) : **dead code** suite à
  l'abandon du scripted controller. À ne pas supprimer (utile comme outil
  de validation FK pour vérifier les calibrations futures).
- `deploy/eval2_inference.py` : **à refactor** pour passer l'image directement
  à la policy au lieu de la passer au CNN puis injecter les xyz.
- Training PPO : doit utiliser `Eval2-PickInClutter-v2` avec
  `--enable_cameras`, et **le rendering caméra ralentit le step de 5–10×**
  → quasiment obligatoire de passer sur Brev H100.

### Statut courant (2026-05-05 soir)

- ⏳ **En attente des démos teleop de Federico** (annoncé pour fin de
  semaine).
- En parallèle, possible de lancer le **Plan A du PPO state-based**
  (milestones conditionnels + boost success_bonus) en background sur le
  RTX 5070 — si ça pète 30%+ succès on a un fallback. Voir [section 13
  Piste A](#piste-a--améliorer-le-reward-shaping-1-h-de-code--4-h-training).

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

### v1.3 — première tentative de convergence (ABORTÉE à iter 3030)

**Diagnostic préalable post-deploy** : en lançant le checkpoint v1.2 dans
`deploy/eval2_inference.py`, on a découvert que **la policy v1.2 n'est PAS
convergée**. Le checkpoint contient un `std` énorme (~3.7). Les "7.46%
success" du training sont obtenus PAR HASARD via l'exploration aléatoire
(mean × bruit), pas par une policy qui a appris la tâche. En mode déploiement
(déterministe ou sampling), le success rate retombe à 0%.

**Cause racine** : sans pénalité explicite sur la magnitude des actions, le
gradient PPO a poussé `mean` vers ±15 (joints saturés à ±7.5 rad après le
scale 0.5). La reward landscape ne pouvait plus distinguer les bonnes des
mauvaises actions → `std` n'a jamais convergé.

**Modifs config v1.3** :
| Aspect | v1.2 | v1.3 |
|---|---|---|
| `target_to_bowl_fine` weight | 5 | **25** |
| `success_bonus` weight | 100 | **200** |
| `action_rate_l2` weight | -1e-4 | **-1e-3** |
| **`action_l2_norm` reward (NEW)** | absent | **-1e-2** |
| `init_noise_std` | 1.0 | **0.5** |
| `entropy_coef` | 0.006 | inchangé (0.006) |
| `max_iterations` | 1000 | **20 000** |

**Résultat (arrêté à iter 3030 / 20000)** :
| Iter | std | success |
|---|---|---|
| 5 | 0.49 | — |
| 441 | 0.74 | 2.85% |
| 1473 | **1.98** ⚠️ | 0.89% |
| 2912 | **4.41** ⚠️ | 1.40% |
| 3030 | **4.65** ⚠️ | 2.36% (et lifting commence à régresser) |

→ **Échec du même type que v1.2** : `std` explose, policy se met à explorer
chaotiquement au lieu de convergir. L'`action_l2 = -1e-2` était trop faible
pour contrebalancer l'`entropy_coef = 0.006`. Run arrêté avant qu'il aille
plus loin.

### v1.4 — convergence stable mais policy gamée (RUN COMPLÉTÉ, 6h17, 2.41% succ)

**Modifs vs v1.3** :
| Aspect | v1.3 | v1.4 |
|---|---|---|
| `action_l2_norm` weight | -1e-2 | **-1e-1** (×10 plus fort) |
| `entropy_coef` | 0.006 | **0.001** (÷6) |
| `init_noise_std` | 0.5 | **0.3** |
| Nouveaux milestone rewards | absents | **`grasp_success` +50, `above_bowl` +100** |

**Résultat (full 20 000 iter, 6h17)** :
| Métrique | Valeur finale |
|---|---|
| **`std`** | **0.40** ✅ (PROBLÈME RÉSOLU) |
| `Mean reward` | 948.65 (gros, mais trompeur — voir plus bas) |
| `lifting_target_low/high` | 8.6 / 8.5 (bonnes valeurs denses) |
| `target_to_bowl_coarse` | 7.7 |
| `target_to_bowl_fine` | 0.25 |
| `grasp_success` (max=15000/ép) | **47** (≈0.3% des frames) |
| `above_bowl` (max=30000/ép) | **82** (≈0.3% des frames) |
| `success_bonus` (max=60000/ép) | **0.014** (quasi nul) |
| `distractor_disturbed` | -0.01 ✅ |
| `block_dropped` | 0% ✅ |
| **`Episode_Termination/success`** | **2.41%** ❌ (même score que v1.3 chaotique) |

**Diagnostic** : la policy a "convergé" — au sens où `std` est sain (0.40)
et le comportement est stable, déterministe — mais elle a convergé sur **un
mauvais comportement**. Elle a découvert qu'elle peut accumuler ~130 points
de "consolation" par épisode via les milestones intermédiaires sans jamais
risquer le placement final :

- elle s'**approche** du bloc (reaching_target = 0.63)
- elle **touche** le grasp brièvement (47/15000 = 0.3% des frames)
- elle **survole** le bowl quelques frames (82/30000 = 0.3% des frames)
- elle **ne lâche jamais dans le bowl** (0.014/60000 = ~0%)

C'est le **classique problème du local optimum** quand les milestones sparse
sont **indépendants** : la policy peut gamer chacun sans faire la séquence
complète. Le success_bonus +200 (qui ne s'active que rarement) ne pèse pas
assez face aux 130 points consolation systématiques.

### Comparaison globale (toutes les versions)

| Run | iter | std final | succ rate | déployable ? |
|---|---|---|---|---|
| v1.2 | 1000 | **3.7** ❌ | 7.46% (= bruit) | non — chaotique en deploy |
| v1.3 | 3030 (arrêté) | **4.65** ❌ | 2.36% | non — chaotique en deploy |
| **v1.4** | **20 000** | **0.40** ✅ | **2.41%** | techniquement oui mais inutile à 2% |

→ On a appris à **stabiliser le training** (v1.4 résout v1.2/1.3) mais
**pas à atteindre un haut success rate**. C'est un problème différent qui
demande une approche différente.

### v1.5 — pistes prévues (à choisir lors de la prochaine session)

Trois directions possibles, par ordre de simplicité :

1. **Milestones conditionnels** (~30 min) : `above_bowl` ne s'active que SI
   `grasp_success` est actif au même step. `success_bonus` ne compte que
   suite à `above_bowl`. Empêche le gaming des milestones indépendants.
2. **Boost massif success_bonus** (~5 min) : weight 200 → 2000+. Force la
   policy à viser le succès au-dessus de tout.
3. **BC warmstart** (~quelques heures) : écrire un scripted controller avec
   IK, générer des demos en sim, BC pretrain, puis PPO finetune. Approche
   éprouvée en RL manipulation, recommandée par la TA spec
   ("Expert teleop data is encouraged").

Plan prévu pour la prochaine session :
- Tester rapidement les options 1+2 combinées (1h de code + 4h training)
- Si toujours <30 % success → bascule sur option 3 BC warmstart

**Note importante pour la suite (perception)** : le CNN actuel a été entraîné
sur la distribution de viewpoints de la policy v1.2 chaotique. Si on obtient
une policy correcte (option 1+2 ou BC warmstart), il faudra **re-capturer le
dataset perception** avec ce nouveau checkpoint et **retrain le CNN**.

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

**Mise à jour 2026-05-05 (post-pivot)** : voir [section 0](#0-décision-archi-2026-05-05--end-to-end-vision--option-a-dapg)
pour la stratégie active. Le chemin canonique est **Option A : BC ACT
end-to-end + DAPG finetune en sim avec image**. Les pistes A/B/C/D
ci-dessous sont des **alternatives ou compléments**, pas le plan
principal.

### Piste 0 (active) — Attendre les démos teleop, puis BC+DAPG end-to-end

Étapes détaillées dans la section 0. Statut : ⏳ en attente des démos
de Federico (annoncées fin de semaine).

Une fois les démos disponibles :
1. Push HF (`Rsebti/projet3-demos-eval2`).
2. BC pretrain ACT (image + joint_pos + target_color + bowl_xyz → action),
   ~2h sur RTX 5070.
3. PPO finetune en sim avec image obs + aux BC loss (DAPG), 5–10h sur
   Brev H100.
4. Domain randomization (couleurs, lumière, textures) avant ou pendant
   PPO.
5. Recapture rien — l'image ENTRE dans la policy, plus de module
   perception séparé.
6. Deploy `deploy/eval2_inference.py` refactor pour image-in.

### Piste A (FALLBACK) — Améliorer le reward shaping (~1 h de code + 4 h training)

⚠️ **Reléguée en fallback** depuis le pivot Option A end-to-end (section 0).
À lancer en background pendant qu'on attend les démos teleop : si ça pète
30%+ succès state-based, on aura un plan B robuste.

v1.4 a montré que la policy game les milestones sparse indépendants.
Deux corrections qui se cumulent :

**A.1 — Milestones conditionnels** :
Dans `sim/eval2/mdp/rewards.py`, modifier `target_block_above_bowl` pour
ne renvoyer 1 que SI `target_block_grasped` est aussi 1 au même step.
Et `target_block_in_bowl` (success_bonus) ne devrait compter que si la
policy était `above_bowl` au step précédent. Ça force la **séquence** :
impossible de toucher l'étape 2 sans avoir fait l'étape 1.

**A.2 — Boost massif success_bonus** :
Passer le weight de 200 à **2000**. Un seul succès vaut alors plus que
TOUTE l'accumulation des milestones intermédiaires. PPO sera contraint
de viser le succès.

Re-train 10k iter (~3-4h) avec ces deux modifs combinées. Si succès
rate >30 % → on continue à raffiner. Si toujours <10 % → bascule sur D.

### Piste B — ~~Implémenter v2 (caméra + perception)~~ ✅ FAIT
- `Eval2PickInClutterEnvCfg_v2` dans `sim/eval2/joint_pos_env_cfg.py`
- Module `sim/eval2/perception/` avec CNN entraîné (val_mae 0.87 cm)
- Pipeline `deploy/eval2_inference.py` qui branche tout
- ⚠️ Le CNN actuel est entraîné sur les viewpoints de v1.2 chaotique →
  à recapturer + retrain quand on aura une bonne policy.

### Piste B — ~~BC warmstart via scripted controller en sim~~ ❌ ABANDONNÉE

Tentative faite (cf. `sim/eval2/bc/`) :
- IK closed-form analytique implémentée + calibrée. Math validée
  (FK→IK→FK 0.0001 mm).
- Scripted state machine 9 phases.
- **Échec runtime** : SO-101 5-DoF + limites articulaires
  serrées (wrist_flex ±1.658) → demande IK ou sature ou contortionne.
  Adaptive phi search, phi-per-phase : aucune variante n'a passé un
  smoke test 50 envs ≥ 80 % succès.
- Audit complet : [`notes/eval2_ik_retrospective.md`](../notes/eval2_ik_retrospective.md).

**Remplacée par** : l'Option A (section 0) — démos teleop **réelles**
au lieu de démos sim scriptées. Plus solide pour le sim-to-real et
recommandé par la TA spec.

### Piste C — Domain randomization (v3, sim-to-real)

UNIQUEMENT pertinent quand on aura une bonne policy + perception. Dans
`EventCfg`, ajouter des events qui s'exécutent à chaque reset :
- `randomize_block_colors` : teintes rouge/bleu dans des plages plus larges
- `randomize_table_friction` : varier le frottement table-bloc
- `randomize_lighting` : varier intensité et direction du `DomeLight`
- `randomize_robot_initial_pose` : petite noise sur la pose home

→ La policy entraînée sur ces variations devient robuste au transfert réel.

### Piste D — Recapture + retrain perception

Une fois la policy convergée (par A ou B), **lancer**
`capture_with_policy.py` avec le nouveau checkpoint, puis `train.py`. Le
CNN actuel est entraîné sur les viewpoints de v1.2 chaotique → biais
hérité. Avec une policy qui fait des trajectoires propres, le dataset
sera beaucoup plus représentatif du déploiement réel.

### Piste E — Faire Eval 1 en parallèle (gains rapides, 50 pts)

Eval 1 = single bloc + bowl, BC autorisé. Tu réutilises **exactement** le
pipeline du sanity check, juste avec :
- **50-100 démos teleop** au lieu de 20, en variant la position du bloc
- Re-train ACT
- Deploy

C'est plus simple que de tout finir Eval 2, et ça vaut autant de points.
**Si tu n'as qu'une journée au robot, fais Eval 1 d'abord.**

### Piste F — Test au vrai SO-101 (à la fin)

Une fois la policy >50% en sim ET le CNN recapturé :
- Brancher `deploy/eval2_inference.py` sur le vrai robot (remplacer la
  source de joint_state et de cam par lerobot)
- 5 rollouts au laboratoire avec différentes positions de blocs/bowl
  fournies par les TAs
- Mesure du success rate réel

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

État au **2026-05-05 soir (post-pivot)** :

### Décision active

**Option A — BC ACT + DAPG end-to-end vision**. On abandonne (a) la
piste scripted IK pour générer des démos en sim et (b) l'approche
modulaire perception → policy state-based. La policy prend l'image
wrist cam directement en input. Voir [section 0](#0-décision-archi-2026-05-05--end-to-end-vision--option-a-dapg).

⏳ **Bloqué sur** : les démos teleop Eval 2 que Federico va enregistrer
fin de semaine.

### Historique condensé

1. **v0–v1.4 PPO from-scratch state-based** : plafonne 2–7 % succès.
   v1.4 a "convergé" sur un mauvais comportement (gaming milestones).
   AUCUNE n'est déployable (chaotique en deploy ou local optimum).

2. **v2 (caméra wrist)** : caméra présente dans `Eval2-PickInClutter-v2`,
   calibrée. **Toujours lancer avec `--enable_cameras`** maintenant qu'on
   est end-to-end.

3. **Module perception CNN** (`sim/eval2/perception/`) : entraîné à
   val_mae 0.87 cm, mais **mostly dead code** depuis le pivot end-to-end.
   À garder comme warmstart potentiel du visual encoder de la policy
   ACT, rien de plus.

4. **Scripted IK controller** (`sim/eval2/bc/`) : tentative abandonnée.
   Math correcte (test FK→IK→FK 0.0001 mm) mais SO-101 5-DoF + limites
   wrist_flex serrées → impossible de grasp fiable en sim. Audit
   complet : [`notes/eval2_ik_retrospective.md`](../notes/eval2_ik_retrospective.md).

5. **`deploy/eval2_inference.py`** : pipeline existe mais **à refactor**
   pour image-in-policy au lieu de CNN→state-policy.

### Reste à faire (par ordre)

1. ⏳ **Démos teleop Eval 2** (côté Federico, ~50–100 démos)
2. **Push HF** dataset
3. **BC pretrain ACT** end-to-end sur ces démos (~2h RTX 5070)
4. **DAPG finetune** en sim avec image obs sur Brev H100 (~5–10h, $50–100)
5. **Domain randomization v3** activée pendant DAPG (couleurs, lumière,
   textures, frottements)
6. **Refactor `deploy/eval2_inference.py`** pour image-in
7. **Deploy 5 rollouts** sur SO-101 réel au labo
8. **En parallèle** : Eval 1 (50 pts BC, pipeline sanity) au prochain
   passage labo
9. **Fallback** : Plan A reward shaping (milestones conditionnels +
   success_bonus boost) à lancer en background sur RTX 5070 pendant
   l'attente des démos. Si ça pète 30%+ succès state-based, on aura
   un plan B.

## 16. Procédure deploy au robot réel (Eval 2 day) — version end-to-end

Le PDF dit explicitement que la position du bowl et la couleur cible sont **fournies
en input** par les TAs. Avec le pivot end-to-end, l'image wrist cam entre
directement dans la policy — plus de module perception séparé qui extrait
les block_xyz.

```
1. Les TAs placent le bowl + 2 blocs colorés sur la table
2. Les TAs mesurent / annoncent (bowl_x, bowl_y, bowl_z) en frame robot
3. Les TAs annoncent la couleur cible (e.g. "rouge")
4. On lance :
   $ python deploy_eval2.py \
       --bowl_x=0.20 --bowl_y=-0.18 --bowl_z=0.02 \
       --target_color=red \
       --policy=Rsebti/projet3-eval2-vX
5. Le script :
   - Lit joint_pos / joint_vel des servos Feetech à 30 Hz
   - Capture l'image wrist cam (RGB)
   - Concatène l'observation pour la policy end-to-end :
       image (HxW x 3) + joint_pos + joint_vel + bowl_xyz (constant) +
       target_color_one_hot (constant) + last_action
   - Inférence policy → action 6-D
   - Envoie l'action aux servos
6. Boucle jusqu'au succès ou time-out
```

`deploy_eval2.py` reste à écrire en partant de l'existant `eval2_inference.py`,
en supprimant l'appel `PerceptionPipeline.predict()` et en passant l'image
directement à la policy. Architecture policy probablement ACT (réutilise le
checkpoint BC pretrain → DAPG finetune). Voir section 0 pour la stratégie
end-to-end complète.
