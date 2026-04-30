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

### v1.1 — passe de compliance PDF (en cours)

**Objectif** : aligner v1 strictement avec la spec TA.

Modifications majeures :
| Aspect | v1 | v1.1 |
|---|---|---|
| **Couleur table** | gris foncé (USD Isaac default) | **`#B8ADA9`** primitive (spec exacte) |
| **Taille bowl** | 10 cm de côté | **12 cm** (≥ spec, marge pour ~12 cubes) |
| **Hauteur murs bowl** | 4 cm | 2.5 cm (= hauteur cube) |
| **Taille cubes** | 2.5 cm | **2 cm** (= vrais cubes en bois) |
| **Cubes adjacents** | 3 cm de gap | **collés** (PDF: "adjacent / flat cluster") |
| **Reset cluster** | 2 events indépendants | 1 event sync via `reset_cluster_uniform` |
| **Lifting reward** | seuil unique 5 cm | **2 stages** : low (2.5 cm, w=10) + high (5 cm, w=10) |
| **`target_to_bowl` gating** | h≥5 cm | h≥2.5 cm (déclenche plus tôt) |
| **`success_bonus` poids** | 50 | **100** |
| **PhysX `total_aggregate_pairs_capacity`** | 16K | **64K** (avait causé erreurs PhysX) |

**Status** : training en cours.

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

### v1.2 — Bowl position randomisée
PDF : *"Bowls placed at randomized positions in the robot base frame"*.
Notre v1.1 : bowl à position fixe.

**Implementation prévue** : étendre `reset_cluster_uniform` (ou nouvelle fonction
`reset_bowl_position`) pour appliquer une translation xy synchronisée aux 5
primitives du bowl à chaque reset.

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

## 12. Liens utiles

- TA spec PDF : [`notes/project3_rl_final_details.md`](../notes/project3_rl_final_details.md)
- Plan de phase générale Eval 2 : [`notes/eval2_plan.md`](../notes/eval2_plan.md)
- Setup Isaac Lab : [`notes/isaac_lab_setup.md`](../notes/isaac_lab_setup.md)
- Repo SO-101 Isaac Lab : https://github.com/MuammerBay/isaac_so_arm101
- Doc Isaac Lab : https://isaac-sim.github.io/IsaacLab/
- Doc rsl_rl : https://github.com/leggedrobotics/rsl_rl

---

## 13. TL;DR pour quelqu'un qui débarque

1. Repo Eval 2 fonctionnel : structure complète + 4 tâches gym enregistrées
2. v0 et v1 entraînent en local sur RTX 5070 (~10-20 min pour 1000 iter)
3. Compliance PDF passée (couleur table, taille bowl, blocs adjacents) en v1.1
4. **Pas encore de caméra** — observations sont ground-truth pour l'instant.
   À fixer en v2 avec module de perception modulaire.
5. **Bowl pos pas encore randomisée** — à fixer en v1.2.
6. Success rate actuel : ~1% à 1000 iter PPO. Faible mais structure validée.
   Plan pour améliorer : reward shaping, training plus long (5000+ iter),
   éventuellement BC warmstart à partir des démos teleop.
