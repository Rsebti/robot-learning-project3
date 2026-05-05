# Eval 2 — Plan de travail

> **⚠️ HISTORIQUE (2026-04-30)** — Ce plan d'origine décrit comment on a démarré
> Eval 2 (install Isaac Lab, structure du package, env v0/v1, premières
> tentatives RL). Il a été suivi jusqu'à v1.4. Pour la stratégie courante
> (post-pivot end-to-end vision + BC ACT + DAPG), voir
> [`sim/eval2.md`](../sim/eval2.md) section 0.
>
> Conservé comme référence pour comprendre comment on est arrivé là.

## Tâche (rappel TA spec)
- Deux blocs **adjacents** de couleurs différentes sur la table
- **Couleur cible** donnée en input → policy doit ramasser **ce** bloc
- **Bowl cible** (xyz en robot frame) donné en input
- 5 rollouts × 10 pts
- Bowl positionné aléatoirement
- **RL obligatoire** (BC seul interdit pour cette éval)
- Démos teleop autorisées comme replay buffer / warmstart BC
- Caméra wrist RGB

## Architecture en couches (rappel CLAUDE.md)
- **Layer 1** (local, jamais dans le repo) : Isaac Sim 5.1.0 + Isaac Lab 2.3.0
- **Layer 2** (local, jamais dans le repo) : `MuammerBay/isaac_so_arm101`
- **Layer 3** (dans **ce** repo) : `sim/eval2/` — env custom, configs, training scripts

## Phases

### Phase A — Install + smoke tests (~1 jour, timebox)
Détails dans [`isaac_lab_setup.md`](isaac_lab_setup.md).
1. Vérifs prérequis (driver NVIDIA, espace disque)
2. Installer `uv`
3. Cloner `isaac_so_arm101` hors du repo projet
4. `uv sync` (télécharge Isaac Sim + Lab + dépendances)
5. `zero_agent` smoke test : SO-101 apparaît dans une fenêtre 3D
6. `train Reach` smoke test : PPO tourne, reward monte
7. **Décision local vs Brev**

Si timebox dépassé sans `train Reach` qui marche → bascule Brev (instance H100 Linux), réinstall sous Linux.

### Phase B — Étude de l'API isaac_so_arm101 (~0.5-1 jour)
Avant de coder, lire :
- `isaac_so_arm101/source/.../tasks/` — comment sont structurées les tâches existantes
- Une tâche exemple bout-en-bout : `SO-ARM100-Reach-v0` ou `SO-ARM100-PickAndPlace-v0`
- Comment Isaac Lab définit `Scene`, `Articulation`, `RewardCfg`, `ObservationCfg`
- Doc Isaac Lab sur `ManagerBasedRLEnv` ou `DirectRLEnv` (ici on partira sur Manager-Based pour rester proche du repo TA)

### Phase C — Env Eval 2 v0 (1 bloc, pas 2) (~1-2 jours)
Pour ne pas se planter sur la complexité, on fait d'abord un **pick-and-place 1 bloc + 1 bowl** en sim, juste pour valider la pipeline custom.

Fichier : `sim/eval2/pick_in_clutter_env.py`

Composants :
- **Scene** : table grise (`#B8ADA9`), SO-101, 1 bloc, 1 bowl, caméra wrist RGB
- **Observation space** :
  - Image wrist (84×84×3 ou 128×128×3 pour aller vite)
  - État du bras (7D : 6 joints + gripper)
  - Target color one-hot (input goal-conditioning, 2D pour 2 couleurs)
  - Target bowl xyz (3D, robot frame)
- **Action space** : 7D delta-joint ou cible cartésienne (à décider)
- **Reward (dense)** :
  - distance gripper → bloc cible (gradient)
  - bonus grasp réussi
  - bonus lift au-dessus du bowl
  - bonus place dans bowl
  - malus collision avec autre bloc
  - malus temps
- **Termination** : succès (bloc dans bowl) ou timeout

### Phase D — PPO smoke run (~0.5 jour)
- 1024 envs en parallèle (Isaac Lab massivement parallèle)
- ~5M-20M steps suffisent pour Reach ; pour pick-and-place, prévoir 50M-200M
- Si reward monte → bonne foi, on garde et on raffine
- Si reward stagne → diagnostiquer (reward shaping, action space, obs)

### Phase E — Extension à 2 blocs colorés (~1-2 jours)
- Ajouter le 2e bloc dans la scene
- Goal-conditioning : target color one-hot devient le moyen pour la policy de savoir quel bloc viser
- Adapter la reward : distance vers **le bon bloc**, malus pour toucher le mauvais
- Re-train PPO

### Phase F — Domain randomization (~1 jour)
Indispensable pour le sim-to-real :
- Couleurs des blocs randomisées (au moins 2 paires de couleurs vues à l'entraînement)
- Textures de la table légèrement perturbées
- Poses initiales du bras randomisées
- Positions des blocs et bowl randomisées dans une zone large
- Frottements et masses légèrement variables
- Lumière (intensité, direction) randomisée
- Bruit caméra

### Phase G — Eval en sim ≥ 80% (~1 jour, training long)
Lancer un training "définitif" :
- Long horizon (200M+ steps)
- Sur Brev H100 si pas déjà
- Snapshot checkpoint régulier
- Eval périodique sur des configs **non vues** au training

### Phase H — Deploy sur le vrai SO-101 (~1 jour, retour au robot)
- Wrap la policy entrainée pour la consommer via `lerobot-record --policy.path=...` ou un script custom (Isaac Lab donne un export ONNX/TorchScript)
- Test sur 5 rollouts dans la config TA :
  - Bloc cible + distracteur de couleur différente
  - Bowl à position randomisée
- Si ≥ 3/5 → on est en course pour les points
- Sinon → analyser modes d'échec, retour Phase F (plus de domain rand) ou Phase E (reward)

### Phase I — Optionnel : warmstart BC sur démos teleop (~1 jour)
Si pure RL est trop lent à converger :
- Reprendre les 19 démos sanity (déjà sur HF) + en enregistrer ~30 nouvelles avec target color variable
- Initialiser la policy par BC offline avant PPO
- Ou hybride : SAC + replay buffer rempli avec les démos

## Risques connus

| Risque | Probabilité | Impact | Mitigation |
|---|---|---|---|
| Isaac Lab crash sur sm_120 | moyenne | bloquant local | Pivot Brev |
| TiledCamera hang Blackwell | élevée si on l'utilise | bloquant | Utiliser `Camera` standard |
| PhysX fallback CPU silencieux | faible-moyenne | training 10x plus lent | `--device cuda` + check via timer |
| Sim-to-real gap trop grand | moyenne | deploy fail | Domain rand agressive, BC warmstart |
| Reward shaping pourri | élevée | RL ne converge pas | Itérer, lire la litterature pick-and-place RL |
| Brev coupon épuisé | faible si discipliné | pas de fallback | Stop instances quand pas en use |

## Fichiers à créer dans le repo (dans `sim/eval2/`)

À mesure qu'on avance :
- `sim/eval2/__init__.py`
- `sim/eval2/pick_in_clutter_env.py` — env Isaac Lab
- `sim/eval2/configs.py` — task configs (Reach v0, PickAndPlace v1, ClutterPick v2…)
- `sim/eval2/train_ppo.py` — script de training (ou utiliser le launcher de isaac_so_arm101)
- `sim/eval2/eval.py` — eval en sim
- `sim/eval2/deploy_real.py` — passerelle policy → vrai SO-101
- `sim/eval2/README.md` — comment lancer training et eval

Tout ce qui est sous `sim/` est dans le repo. Isaac Lab + isaac_so_arm101 ne le sont pas.
