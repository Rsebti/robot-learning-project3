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

**Status** : implémenté, à entraîner.

---

## 6. Tâches gym enregistrées

```python
Eval2-PickInBowl-v0          # v0 training (4096 envs)
Eval2-PickInBowl-Play-v0     # v0 visu (50 envs, sans bruit)
Eval2-PickInClutter-v1       # v1 training (4096 envs) ← CIBLE PRINCIPALE
Eval2-PickInClutter-Play-v1  # v1 visu
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

## 13. Liens utiles

- TA spec PDF : [`notes/project3_rl_final_details.md`](../notes/project3_rl_final_details.md)
- Plan de phase générale Eval 2 : [`notes/eval2_plan.md`](../notes/eval2_plan.md)
- Setup Isaac Lab : [`notes/isaac_lab_setup.md`](../notes/isaac_lab_setup.md)
- Repo SO-101 Isaac Lab : https://github.com/MuammerBay/isaac_so_arm101
- Doc Isaac Lab : https://isaac-sim.github.io/IsaacLab/
- Doc rsl_rl : https://github.com/leggedrobotics/rsl_rl

---

## 14. TL;DR pour quelqu'un qui débarque

1. Repo Eval 2 fonctionnel : structure complète + 4 tâches gym enregistrées
2. v0, v1, v1.1, v1.2 entraînent en local sur RTX 5070 (~25 min pour 1000 iter à 4096 envs)
3. **Toute la spec PDF est implémentée** sauf l'observation visuelle (v2 prévu)
4. **Baseline v1.2** : success rate sim ~ 7.5 % à 1000 iter PPO (cf. section 12 pour la
   commande exacte de reproduction).
5. Goulot d'étranglement actuel : `target_to_bowl_fine` (descente précise au-dessus du
   bowl). Pistes : training plus long, boost reward fine, curriculum, BC warmstart.
6. Reste à faire pour points : v2 caméra + perception modulaire + deploy au robot réel.

## 15. Procédure deploy au robot réel (Eval 2 day)

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
