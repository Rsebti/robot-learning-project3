# Eval 2 — Pipeline complète de A à Z

> Document de référence pour exécuter Eval 2 du début à la fin.
> Stratégie : **Option A — BC ACT end-to-end vision + DAPG finetune en sim**.
>
> Date de rédaction : 2026-05-06.
> À mettre à jour à mesure que la pipeline est validée phase par phase.

---

## TL;DR

1. **120 démos teleop physique** au robot, plan dans [`notes/eval2_demo_plan.csv`](eval2_demo_plan.csv)
2. **Push HF** → `Rsebti/projet3-eval2-demos`
3. **Data augmentation** dans le data loader (color jitter, brightness, gaussian noise — pas de flip)
4. **Train ACT goal-conditioned** end-to-end (image wrist + joint_state + target_color one-hot 6D + bowl_xyz)
5. **Étendre la sim Isaac Lab** à 6 couleurs (modifs `sim/eval2/joint_pos_env_cfg.py` + `mdp/observations.py` + `mdp/events.py`)
6. **RL finetune en sim** avec image obs + auxiliary BC loss (DAPG), domain randomization v3 activée → sur Brev H100 (~$50-100)
7. **Deploy** sur SO-101 réel via `deploy/eval2_inference.py` refactoré
8. **5 rollouts d'éval** au labo selon protocole TA

Estimation totale : ~10-15 jours (dont 1 journée labo démos, 1 journée labo eval, ~5-10h training Brev).

---

## Pourquoi cette stratégie

État au 2026-05-06 :

| Approche | Verdict | Raison |
|---|---|---|
| BC seul (ACT) sans RL | ❌ | Spec TA exige RL pour Eval 2 |
| Pure PPO from-scratch state-based | ❌ testé v1.0–v1.7 | plafonne 2-7% succès, gaming des milestones |
| Scripted IK pour générer démos sim | ❌ testé | SO-101 5-DoF + limites wrist_flex serrées → grasp pas fiable |
| Modulaire (CNN perception + MLP state-based) | ❌ | CNN entraîné sur policy chaotique, hérite biais ; chemin abandonné après pivot |
| **BC ACT end-to-end + DAPG** ⭐ | ✅ choisi | Standard littérature, encouragé TA spec, faisable budget |

Voir [`sim/eval2.md`](../sim/eval2.md) section 0 pour l'historique complet du pivot, et [`notes/eval2_ik_retrospective.md`](eval2_ik_retrospective.md) pour l'audit du scripted IK.

---

## Vue d'ensemble des phases

| Phase | Lieu | Durée | Préreq | Output |
|---|---|---|---|---|
| **A** Préparation session | PC fixe | 1h | csv déjà généré | matériel + table prête |
| **B** Session démos | labo (laptop) | 2-3h | hardware OK | 120 .npz dans cache lerobot |
| **C** Push HF | n'importe | 5-15 min | dataset valide | `Rsebti/projet3-eval2-demos` sur HF |
| **D** Préparer code training | PC fixe | 2-4h | dataset HF | `train/eval2_loader.py` + extension obs |
| **E** Data augmentation | PC fixe | 1-2h | code training | augmentations branchées au loader |
| **F** Train ACT | PC fixe ou Brev | 2-4h | code OK | `Rsebti/projet3-act-eval2` checkpoint |
| **G** Étendre sim 6 couleurs | PC fixe | 2-3h | sim qui marche | `Eval2-PickInClutter-v3` |
| **H** RL finetune (DAPG) | Brev H100 | 6-12h | tout au-dessus | policy finale converge |
| **I** Deploy + eval | labo | 1-2h | policy | 5/5 rollouts (idéalement) |

Total ~25-35h de travail réel + 6-12h GPU Brev.

---

## Phase A — Préparation session démos (PC fixe)

### Objectif
Avoir tout le matériel + repères + plan prêts pour ne pas improviser au labo.

### A.1 — Imprimer le plan de démos

Le csv [`notes/eval2_demo_plan.csv`](eval2_demo_plan.csv) contient les 120 lignes à shooter.

```powershell
# Ouvrir dans Excel, ajouter MFC sur colonne "fait" (1 → vert), enregistrer en .xlsx
# Imprimer ou afficher sur 2e écran pendant la session
```

Sur la feuille papier, prévois 2 colonnes vides à droite : "✅ done" / "❌ retake".

### A.2 — Marquer la table physique

À apporter au labo : **scotch noir, mètre ruban, marqueur, 2 cales fines (~2cm) pour les bowls surélevés**.

Sur la table du labo, marque les positions de bowls et clusters au scotch :

**Zone bowls (face robot, côté droit pour droitier)** :
| ID | x | y | z | Notes |
|---|---|---|---|---|
| B1 | 0.18 | -0.10 | 0.020 | sur table |
| B2 | 0.22 | -0.12 | 0.020 | sur table |
| B3 | 0.20 | -0.18 | 0.020 | sur table |
| B4 | 0.25 | -0.15 | 0.040 | sur cale 2cm |
| B5 | 0.18 | -0.20 | 0.020 | sur table |
| B6 | 0.23 | -0.20 | 0.040 | sur cale 2cm |

**Zone clusters (face robot, axe central et gauche)** :
| ID | x | y | z |
|---|---|---|---|
| C1 | 0.18 | 0.00 | 0.010 |
| C2 | 0.22 | 0.00 | 0.010 |
| C3 | 0.20 | 0.04 | 0.010 |
| C4 | 0.18 | 0.08 | 0.010 |
| C5 | 0.24 | 0.04 | 0.010 |
| C6 | 0.21 | 0.10 | 0.010 |

⚠️ Les positions xy sont **mesurées depuis la base du robot SO-101**, axe x face au robot, axe y vers la gauche (convention robot frame). Si Federico a un repère différent, valider avec lui avant la session.

⚠️ Vérifier que **toutes les positions sont dans la zone reachable** du SO-101. Distance max ≈ √(x² + y²) doit rester < 0.30 m.

### A.3 — Cubes physiques

6 couleurs disponibles : `blue, green, violet, yellow, red, orange`.

**Mapping canonique** (à figer maintenant, à utiliser partout : training, sim, deploy) :

```python
COLOR_TO_INDEX = {
    "blue":   0,
    "green":  1,
    "violet": 2,
    "yellow": 3,
    "red":    4,
    "orange": 5,
}
```

Ordre alphabétique pour réduire le risque d'erreur. **Toute mismatch entre training et deploy = points perdus.** À écrire dans une constante `sim/eval2/colors.py` ou similaire (Phase G).

### A.4 — Hardware checklist

Au labo :
- SO-101 follower + leader, en bon état
- Alimentation 5V × 2 (follower + leader)
- 3 câbles USB
- Caméra wrist montée + câblée
- Laptop avec env `lerobot` (Python 3.12, lerobot 0.5.1) + HF auth OK
- Pads scotch + mètre + cales (cf. A.2)
- 6 cubes 2×2×2 cm aux 6 couleurs
- 1 bowl ~12 cm de diamètre intérieur

### A.5 — Pré-flight au labo

Avant de lancer la première démo :

```powershell
conda activate lerobot
hf auth whoami        # = Rsebti
lerobot-find-port     # 2 fois (leader / follower) — note les COMs
lerobot-find-cameras opencv   # note l'index de la wrist cam (typiquement 1)
```

Calibration si machine différente du sanity check :
```powershell
lerobot-calibrate --robot.type=so101_follower --robot.port=COM3 --robot.id=so101_follower
lerobot-calibrate --teleop.type=so101_leader  --teleop.port=COM5 --teleop.id=so101_leader
```

Test téléop sans enregistrement (5 min de manipulation libre pour t'assurer que tout marche) :
```powershell
lerobot-teleoperate `
  --robot.type=so101_follower --robot.port=COM3 --robot.id=so101_follower `
  --teleop.type=so101_leader  --teleop.port=COM5 --teleop.id=so101_leader `
  --display_data=true
```

---

## Phase B — Session démos téléop physique (labo)

### Objectif
Enregistrer 120 démos selon le csv, avec discipline grasp constante.

### B.1 — Méthode de grasp à respecter

⚠️ **CRUCIAL pour BC** : si la méthode varie entre démos, ACT apprend une moyenne incohérente.

Pour chaque démo :

1. **Approche top-down vertical** : gripper pointe vers le bas (axe Z négatif)
2. **Hover ~8 cm au-dessus du cube cible** (la couleur annoncée dans le csv, pas le distractor)
3. **Descente verticale** jusqu'à ~5 mm au-dessus de la table
4. **Fermer gripper** sur le bloc cible
   - Approche perpendiculaire à l'axe du cluster : un finger devant, un derrière le cube cible
   - Évite de cogner le distractor adjacent
5. **Lift vertical ~12 cm** AVANT tout déplacement latéral
6. **Transport horizontal** au-dessus du bowl, gripper toujours vertical
7. **Descente** à ~4 cm au-dessus du fond du bowl
8. **Ouvrir gripper**, le cube tombe
9. **Retreat vertical** ~10 cm
10. → Bouton clavier "save" → reset auto

Si grasp foire en cours → bouton "discard" → refait la même ligne du csv. **Mieux 100 démos clean que 120 polluées.**

### B.2 — Workflow par démo

Pour chaque ligne du csv (en gardant le csv sous les yeux) :

1. Lire la ligne : `target_color, distractor_color, target_side, bowl_id, cluster_id`
2. Placer le **bowl** au repère `bowl_id` marqué au scotch (ajouter cale si surélevé)
3. Placer les **2 cubes collés** au repère `cluster_id` :
   - Si `target_side=left` : `target_color` à gauche du cluster, `distractor_color` à droite
   - Si `target_side=right` : `target_color` à droite, `distractor_color` à gauche
   - Cubes **collés** (face contre face), aucun gap
4. Bras du robot en pose home (le leader peut servir à la repositionner)
5. Lance lerobot-record (cf. B.3) — ou continue dans le run en cours
6. Pendant l'enregistrement (15-20s) : exécute la grasp method (B.1) sur le cube cible
7. À la fin : confirme save → coche la ligne csv
8. Pendant le reset (15-20s) : prépare la ligne suivante

### B.3 — Commande lerobot-record

⚠️ **Un seul run lerobot pour les 120 démos** : `single_task` est figé pour tout le run, et la correspondance avec target_color/bowl_xyz vient du csv au training. Le `single_task` reste générique.

```powershell
$env:RUST_LOG = "error"
lerobot-record --% `
  --robot.type=so101_follower `
  --robot.port=COM3 `
  --robot.id=so101_follower `
  --robot.cameras="{\"wrist\": {\"type\": \"opencv\", \"index_or_path\": 1, \"width\": 640, \"height\": 480, \"fps\": 30}}" `
  --teleop.type=so101_leader `
  --teleop.port=COM5 `
  --teleop.id=so101_leader `
  --display_data=true `
  --dataset.repo_id=Rsebti/projet3-eval2-demos `
  --dataset.num_episodes=120 `
  --dataset.fps=30 `
  --dataset.episode_time_s=20 `
  --dataset.reset_time_s=20 `
  --dataset.single_task="Eval2: pick target color, place in bowl" `
  --dataset.private=true `
  --dataset.push_to_hub=false
```

Notes :
- `episode_time_s=20` (sanity = 15) → trajectoire Eval 2 plus longue
- `reset_time_s=20` → temps de lire la ligne csv + redéplacer cubes + bowl
- `push_to_hub=false` → push manuel à la fin si tout est validé (Phase C)
- COM3/COM5 et idx 1 à valider avec `lerobot-find-port` / `lerobot-find-cameras` (Phase A.5)

### B.4 — Pendant la session

- **Pause de 5 min toutes les 30 démos** (ep 30, 60, 90). Fatigue → discipline grasp se dégrade.
- Si tu sens que tu es à côté de la plaque (mauvais grasp 3 fois d'affilée), pause 10 min, sinon arrête à 90-100 démos clean.
- **Ne touche pas la caméra wrist**. Si tu la déplaces accidentellement, recalibre la position avant de continuer (sinon le dataset post-déplacement est incohérent avec pré-).
- **Lumière constante** : pas d'allumer/éteindre la lampe entre démos.
- Si tu perds le csv physique, il y a la copie sur HF dans le dataset (`single_task` field) ou en local dans le repo Git.

### B.5 — Fin de session

- Vérifie que `Rsebti/projet3-eval2-demos` (cache local) contient bien 120 épisodes.
- Replay 4 épisodes de contrôle (eps 0, 30, 60, 90) avec :

```powershell
lerobot-replay `
  --robot.type=so101_follower --robot.port=COM3 --robot.id=so101_follower `
  --dataset.repo_id=Rsebti/projet3-eval2-demos `
  --dataset.episode=0
```

Le robot doit reproduire fidèlement chaque démo. Si drift important → calibration à refaire.

---

## Phase C — Push HF du dataset

### Objectif
Upload dataset sur HuggingFace pour pouvoir le récupérer depuis le PC fixe.

### C.1 — Push manuel

```powershell
hf upload Rsebti/projet3-eval2-demos `
  C:\Users\sebti\.cache\huggingface\lerobot\Rsebti\projet3-eval2-demos `
  --repo-type dataset
```

(adapter le path local selon la machine d'enregistrement — laptop ou PC fixe)

Durée : 10-30 min selon la connexion. Le dataset fait ~1-2 GB (120 vidéos × ~10 MB).

### C.2 — Push aussi le csv au repo Git

Le csv est la source de vérité pour les goals (target_color + bowl_xyz). Il **doit** être commité au repo :

```bash
cd C:\Users\user\Desktop\MA2\robot-learning-project3
git add notes/eval2_demo_plan.csv notes/eval2_pipeline.md
git commit -m "Eval 2 demos: 120-row plan + full pipeline doc"
git push
```

### C.3 — Vérifications

Sur https://huggingface.co/datasets/Rsebti/projet3-eval2-demos :
- 120 épisodes visibles
- Dataset Viewer fonctionne
- Vidéos streamables

Sur le PC fixe :
```powershell
python -c "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; d = LeRobotDataset('Rsebti/projet3-eval2-demos'); print(f'eps={d.num_episodes}, frames={d.num_frames}')"
```
→ doit afficher `eps=120, frames=~72000` (120 × 600 frames = 120 démos × 20s × 30 fps).

---

## Phase D — Préparer le code training (PC fixe)

### Objectif
Brancher le csv au dataset lerobot pour que la policy reçoive `target_color` + `bowl_xyz` comme inputs.

### D.1 — Le challenge du goal-conditioning dans lerobot

LeRobot ACT consomme :
- `observation.state` : vecteur des angles servos (6D pour SO-101)
- `observation.images.wrist` : image RGB
- `action` : vecteur de cible servos (6D)

Pour ajouter target_color (6D one-hot) + bowl_xyz (3D), deux options :

**Option D.A — Étendre `observation.state` à 15D** :
- Concaténer (joint_pos 6D, target_color_one_hot 6D, bowl_xyz 3D) = 15D
- Avantage : pas de modif d'architecture ACT, juste plus large
- Inconvénient : la stats normalization du `observation.state` doit être recomputed (joint range ~ ±π, one-hot ∈ {0,1}, xyz ~0.2 m → variances très différentes)

**Option D.B — Créer un nouvel observation field `observation.goal`** :
- Plus propre conceptuellement
- Demande de **patcher le dataloader lerobot** ou de wrapper ACT pour qu'il consomme le nouveau field
- Plus de plomberie

→ **Recommandation : Option D.A**. Plus simple, marche out-of-the-box avec lerobot.

### D.2 — Script de transformation du dataset

Créer `train/eval2_loader.py` qui :
1. Charge `Rsebti/projet3-eval2-demos` (raw, 6D state)
2. Charge `notes/eval2_demo_plan.csv`
3. Pour chaque épisode `i`, lit la ligne csv `i` → récupère `target_color`, `bowl_x`, `bowl_y`, `bowl_z`
4. Encode `target_color` en one-hot 6D selon `COLOR_TO_INDEX`
5. Étend `observation.state` à 15D : `[joint_pos_6D, color_onehot_6D, bowl_xyz_3D]`
6. Sauve un nouveau dataset local OU pousse en `Rsebti/projet3-eval2-demos-augmented`

Squelette :

```python
# train/eval2_loader.py
import pandas as pd
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

COLOR_TO_INDEX = {"blue": 0, "green": 1, "violet": 2,
                  "yellow": 3, "red": 4, "orange": 5}

def transform(src_repo, dst_repo, csv_path):
    src = LeRobotDataset(src_repo)
    plan = pd.read_csv(csv_path)
    # ... pour chaque épisode i:
    #   row = plan.iloc[i]
    #   color_oh = np.eye(6)[COLOR_TO_INDEX[row.target_color]]
    #   bowl_xyz = np.array([row.bowl_x, row.bowl_y, row.bowl_z])
    #   for each frame: state_15D = concat(state_6D, color_oh, bowl_xyz)
    # save to dst_repo
```

⚠️ La structure interne du `LeRobotDataset` (parquet + meta json) n'est **pas trivialement éditable**. Plusieurs options pour réécrire :
- (a) Charger frame par frame, créer un nouveau dataset via `LeRobotDataset.create(...)`, écrire frame par frame
- (b) Écrire un wrapper `Dataset` PyTorch qui modifie `state` à la volée à `__getitem__`, sans toucher au parquet

→ **Option (b) est plus simple** pour le BC training, **option (a) nécessaire** pour le push HF d'un dataset auto-suffisant.

### D.3 — Validation D

Avant de passer à E, vérifier que le loader marche :
```python
ds = transform(...)
sample = ds[0]
assert sample["observation.state"].shape == (15,)  # 6 joints + 6 color + 3 bowl
assert sample["observation.images.wrist"].shape[-3:] == (480, 640, 3)
print(f"target_color one-hot: {sample['observation.state'][6:12]}")
print(f"bowl_xyz: {sample['observation.state'][12:15]}")
```

---

## Phase E — Data augmentation

### Objectif
Augmenter la diversité visuelle effective du dataset sans nouvelles démos.

### E.1 — Ce qu'on peut augmenter

| Champ | Augmentation | Pourquoi |
|---|---|---|
| `observation.images.wrist` | color jitter | sim-to-real (conditions lumière variables) |
|  | brightness ±20% | idem |
|  | contrast ±15% | idem |
|  | gaussian noise σ=0.01 | bruit caméra réaliste |
|  | random crop léger (90-100% zoom) | tolérance position cam |
| `observation.state[0:6]` (joints) | gaussian noise σ=0.005 rad | bruit servos |
| `observation.state[6:12]` (target_color) | ❌ ne pas toucher | sémantique catégorielle |
| `observation.state[12:15]` (bowl_xyz) | ❌ ne pas toucher | la trajectoire de l'épisode dépend de bowl_xyz |
| `action` | ❌ ne pas toucher | imite les actions humaines exactement |

### E.2 — Ce qu'il NE faut PAS faire

- **Flip horizontal de l'image** : la position de bowl_xyz et la position des cubes dans le repère robot ne se mirroirent pas trivialement → incohérence training. Sauf si on se rajoute le coût de mirror aussi le state vector + action vector, ce qui est un bug en attente.
- **Random rotation de l'image** : même problème, casse la cohérence avec joint_state.
- **Mixup / cutmix** entre épisodes : casse la cohérence temporelle (ACT consomme des séquences).

### E.3 — Implémentation

LeRobot supporte les transforms via `AlbumentationsTransform` ou un torchvision transform passé au dataset. Squelette :

```python
import albumentations as A

augment = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.15, p=0.7),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
    A.GaussNoise(var_limit=(5, 30), p=0.4),
])

# Dans le dataloader, appliquer augment seulement sur `observation.images.wrist`
# pendant l'entraînement (pas en validation).
```

⚠️ Vérifie que ton ACT loader applique l'augmentation **en train mode uniquement**, pas en eval. Sinon les courbes de val_loss seront biaisées.

### E.4 — Validation E

Visualise 4-8 augmentations différentes d'une même image (ep 0 frame 0) avant de lancer le training. Si certaines sont irréalistes (image complètement noire, hue extrême violette qui rend rouge ≈ violet, etc.), réduire les paramètres.

---

## Phase F — Train ACT goal-conditioned

### Objectif
Entraîner ACT sur le dataset étendu (state 15D) jusqu'à convergence en BC pure.

### F.1 — Commande training

Sur PC fixe (RTX 5070) :

```powershell
$env:RUST_LOG = "error"
lerobot-train `
  --dataset.repo_id=Rsebti/projet3-eval2-demos-augmented `
  --policy.type=act `
  --policy.device=cuda `
  --policy.repo_id=Rsebti/projet3-act-eval2 `
  --output_dir=outputs/train/act_eval2 `
  --job_name=act_eval2 `
  --batch_size=8 `
  --steps=80000 `
  --save_freq=10000 `
  --eval_freq=0 `
  --log_freq=500 `
  --wandb.enable=true
```

Différences vs sanity (`steps=20000`) et Eval 1 (`steps=80000`) :
- 120 démos × variabilité (target_color × bowl_xyz × cluster_pos) → plus de modes que sanity (1 mode) ou Eval 1 (1 task, position randomisée)
- 80k steps suffisent en général, mais surveille la val_loss : si elle stagne à 30k, tu peux arrêter ; si elle descend encore à 80k, étends à 120k

⚠️ Si OOM Blackwell sm_120, soit (a) `--batch_size=4`, soit (b) PyTorch nightly cu128 (cf. `notes/full_pipeline_walkthrough.md` §10).

### F.2 — Métriques à surveiller

| Métrique | Bon signe | Mauvais signe |
|---|---|---|
| `loss` | descend de ~1.5 à ~0.15-0.30 | reste haute, oscille, NaN |
| `l1` | suit | NaN |
| `kl` | < 0.5 | grimpe à 100+ |
| `grad_norm` | < 5.0 | > 50 = instable |
| Mémoire GPU | 6-10 GB | OOM |
| Val loss (si activée) | gap < 50% vs train loss | si gap énorme → overfitting → réduire steps ou augmenter augmentation |

Loss finale attendue : `~0.15-0.25` (vs `~0.05` pour sanity, `~0.10-0.20` pour Eval 1). Plus haute parce que plus de variabilité dans les données.

### F.3 — Sanity test BC pure (avant RL finetune)

Avant de passer à G+H, **valider que la BC seule a un comportement sensé** :

1. Charge le checkpoint `outputs/train/act_eval2/checkpoints/last`
2. Roule l'env `Eval2-PickInClutter-Play-v2` (ou v3 quand prêt) avec cette policy en sim
3. Compte le success rate
4. **Cible BC pure : 30-50% succès** en sim (sans RL finetune)
   - Si > 50% : BC déjà costaud, RL ne servira qu'à raffiner
   - Si 20-50% : normal, le RL devrait bumper à >70%
   - Si < 20% : retour Phase B (démos pas assez bonnes / pas assez nombreuses) ou Phase E (augmentation trop agressive)

Pour ce sanity, il faut :
- Adapter `Eval2-PickInClutter-Play-v2` à l'observation 15D (pas 29D actuel) — c'est en partie la Phase G
- Wrapper ACT pour consommer cette observation et produire des actions 6D
- Petit script `deploy/eval2_bc_sim_eval.py` à écrire (~100 lignes)

### F.4 — Push checkpoint HF

```powershell
hf upload Rsebti/projet3-act-eval2 outputs/train/act_eval2/checkpoints/last/pretrained_model
```

Modèle ~200 MB, push 1-3 min.

---

## Phase G — Étendre la sim Isaac Lab à 6 couleurs

### Objectif
La sim doit pouvoir spawner 2 cubes parmi 6 couleurs et gérer un target_color one-hot 6D, pour matcher le format des démos physiques au training RL.

### G.1 — Fichiers à modifier

| Fichier | Modif |
|---|---|
| `sim/eval2/colors.py` | **CRÉER** : constante `COLOR_TO_INDEX` + `INDEX_TO_RGB` (6 entries) |
| `sim/eval2/joint_pos_env_cfg.py` | spawner 6 RigidObjectCfg `block_<color>` au lieu de 2 |
| `sim/eval2/pick_in_clutter_env_cfg.py` | étendre `target_color_one_hot` 2D → 6D dans observation ; randomization au reset choisit 2 couleurs parmi 6 et désactive les 4 autres |
| `sim/eval2/mdp/observations.py` | `target_color_one_hot` lit `env.target_color` ∈ [0..5] (au lieu de [0..1]) |
| `sim/eval2/mdp/events.py` | `reset_target_color(num_classes=6)` ; nouvelle fonction `reset_visible_pair` qui choisit 2 couleurs et place les 4 autres hors scène (z=-10) |
| `sim/eval2/mdp/rewards.py` | `_target_block_pos` doit dispatcher vers `block_<color>` selon `env.target_color` |
| `sim/eval2/__init__.py` | enregistrer `Eval2-PickInClutter-v3` et `-Play-v3` |

### G.2 — Constante centrale

```python
# sim/eval2/colors.py
COLOR_TO_INDEX = {
    "blue":   0,
    "green":  1,
    "violet": 2,
    "yellow": 3,
    "red":    4,
    "orange": 5,
}

INDEX_TO_COLOR = {v: k for k, v in COLOR_TO_INDEX.items()}

INDEX_TO_RGB = {
    0: (0.10, 0.20, 0.85),  # blue
    1: (0.10, 0.65, 0.10),  # green
    2: (0.55, 0.20, 0.75),  # violet
    3: (0.95, 0.85, 0.10),  # yellow
    4: (0.85, 0.10, 0.10),  # red
    5: (0.95, 0.50, 0.05),  # orange
}
```

⚠️ Importer cette constante **partout** (training loader, sim env, deploy script). Une seule source de vérité.

### G.3 — Sim cluster avec 2 cubes parmi 6

Stratégie au reset :
1. Sample 2 indices distincts parmi {0..5} → `pair = (a, b)`
2. Sample lequel est target → `target_color = a` ou `b`
3. Sample lequel est à gauche/droite dans le cluster → permutation aléatoire
4. Pour les 6 cubes physiques dans la sim :
   - Cube `a` placé au cluster_xy, à gauche du centre
   - Cube `b` placé au cluster_xy, à droite du centre
   - Les 4 autres cubes (`{0..5} \ {a, b}`) placés à `z=-10` (hors scène, invisibles à la cam)

Avantage de cette approche : **pas de spawning dynamique** (Isaac Lab n'aime pas), tous les 6 cubes existent toujours mais 4 sont cachés.

### G.4 — Validation G

Smoke test :
```powershell
cd C:\Users\user\Desktop\MA2\isaac\isaac_so_arm101
uv run python -m sim.eval2.scripts.view --task Eval2-PickInClutter-Play-v3 --num_envs 4 --enable_cameras
```

Tu dois voir 4 environnements, chacun avec **2 cubes visibles de couleurs distinctes** (différentes paires entre les 4 envs si tout va bien). Les 4 autres cubes invisibles sous le sol.

Vérif obs :
```python
obs = env.reset()
print(obs["observation.state"][6:12])  # one-hot 6D, exactement un 1
```

---

## Phase H — RL finetune en sim avec image (DAPG)

### Objectif
Affiner le checkpoint ACT pretrain via PPO + auxiliary BC loss, en sim, avec image observation et domain randomization.

### H.1 — Pourquoi DAPG et pas pure PPO

DAPG = Demo Augmented Policy Gradient (Rajeswaran et al. 2017). Combine :
- Loss PPO classique (gradient sur reward)
- Loss BC auxiliaire (gradient sur action des démos)
- Pondération qui décline avec le temps : démarre BC-heavy, devient PPO-heavy

Pourquoi ça marche où pure PPO échoue :
- Pure PPO from scratch : exploration aléatoire dans un espace 6D-action, reward sparse → plafonne 2-7% (cf. v1.4-v1.7 de notre repo)
- BC-only : converge mais peut avoir des écarts à des conditions OOD (compounding errors)
- **DAPG** : la BC tient la policy near-trajectoire-experte, le PPO la pousse à raffiner sur la reward → meilleur des deux mondes

Le checkpoint ACT (Phase F) sert de **policy initiale**. DAPG ne réinvente pas tout.

### H.2 — Code à écrire

LeRobot ne fournit pas de DAPG out-of-the-box. Trois options :

| Option | Effort | Qualité |
|---|---|---|
| **H.A** Réutiliser rsl_rl PPO + ajouter BC term | 1-2 jours | propre, maintenu |
| **H.B** Bibliothèque externe (D3RLPy, RLTools) | 0.5 jour install + intégration | dépend de la lib |
| **H.C** Code from-scratch sur torch | 3-4 jours | éducatif, plus de bugs |

→ **Recommandation : H.A**. Le repo actuel utilise déjà rsl_rl pour PPO ([sim/eval2/agents/rsl_rl_ppo_cfg.py](../sim/eval2/agents/rsl_rl_ppo_cfg.py)). On le branche au checkpoint ACT pretrain et on ajoute un terme de loss BC dans `OnPolicyRunner`.

Squelette des modifs (à itérer) :

```python
# sim/eval2/agents/rsl_rl_dapg_cfg.py
class Eval2DAPGRunnerCfg(Eval2PPORunnerCfg):
    bc_weight_initial = 1.0
    bc_weight_decay = 0.5  # exponential decay per iteration
    bc_dataset_path = "Rsebti/projet3-eval2-demos-augmented"
    pretrain_checkpoint = "Rsebti/projet3-act-eval2"
```

```python
# train/dapg.py
# Subclass OnPolicyRunner, override learn():
#   - Load BC dataset
#   - Each iter: PPO update + BC update on random demo batch
#   - bc_weight *= bc_weight_decay
```

⚠️ **Wrapping de ACT en policy rsl_rl** : ACT prend image+state, rsl_rl PPO prend juste state. Faut wrapper ACT pour qu'il expose une API `act_inference(obs)` compatible. C'est ~50 lignes de code, plus simple que de réécrire le cycle PPO.

Le wrapper doit :
- Recevoir `obs` (TensorDict avec policy_obs + image)
- Forward dans ACT
- Retourner `action` (la première du chunk de 100)

### H.3 — Domain randomization v3

Pendant DAPG, activer randomization à chaque reset :
- **Couleurs** : RGB jitter ±5% sur chaque cube (rouge un peu plus orangé, etc.)
- **Lumière** : intensité DomeLight 2000-4000 lux, direction ±20° rotation
- **Frottements** : friction table-bloc 0.4-0.8
- **Masses** : cube mass 0.04-0.06 kg
- **Caméra** : noise gaussien sur image (avant resize)
- **Pose initiale robot** : ±0.05 rad sur chaque joint

À implémenter dans `sim/eval2/mdp/events.py` comme nouvel `EventTerm("randomize_domain", mode="reset")`.

### H.4 — Lancer le training sur Brev H100

Une fois le code testé localement (1 iter), passer sur Brev :

```bash
brev shell
cd /workspace/isaac_so_arm101
git pull
cd /workspace/robot-learning-project3
git pull

uv run python -m train.dapg \
  --task Eval2-PickInClutter-v3 \
  --headless --enable_cameras \
  --num_envs 1024 \
  --max_iterations 5000 \
  --pretrain_checkpoint Rsebti/projet3-act-eval2 \
  --bc_weight_initial 1.0 --bc_weight_decay 0.5
```

⚠️ **`num_envs=1024`** au lieu de 4096 — le rendering caméra ralentit ~5-10×, OOM possible sinon.

⚠️ **Coût Brev** : ~$2-3/h × 6-12h = $15-40. **STOP l'instance dès que le training est fini.** (sur les $200 dispo, ça laisse de la marge pour itérer si la première run rate.)

### H.5 — Métriques à surveiller

| Métrique | Bon signe |
|---|---|
| `Episode_Termination/success` | démarre > 30% (BC pretrain), monte > 70% à conv |
| `Mean reward` | monte régulièrement |
| `bc_loss` | descend ou stable |
| `policy_loss` (PPO) | oscille dans [-0.5, +0.5] |
| `entropy` | descend de ~5 à ~2 |
| `std` | reste < 0.6 (sinon explosion comme v1.3-v1.4) |

Si la policy plafonne < 50% à 3000 iter : ne pas insister, stop l'instance et investiguer (probablement bug DR ou mismatch obs).

### H.6 — Push checkpoint final

```bash
hf upload Rsebti/projet3-eval2-final logs/eval2_dapg/<run>/model_4999.pt --repo-type model
```

---

## Phase I — Deploy + eval au labo

### Objectif
Évaluer la policy finale sur le SO-101 réel avec 5 rollouts dans les conditions des TAs.

### I.1 — Refactor `deploy/eval2_inference.py`

Le script actuel ([deploy/eval2_inference.py](../deploy/eval2_inference.py)) charge la perception CNN + policy state-based. À adapter pour :

- Plus de CNN perception (image entre directement dans la policy)
- Le `observation.state` à passer = 15D (joint_pos 6 + color_oh 6 + bowl_xyz 3) selon le format Phase D
- Pour le robot réel : remplacer le scene Isaac Sim par un wrapper LeRobot qui lit joints/cam à 30 Hz

Squelette du nouveau `deploy/deploy_eval2.py` (à écrire) :

```python
# CLI args
--target_color blue|green|violet|yellow|red|orange
--bowl_x 0.20 --bowl_y -0.15 --bowl_z 0.02
--policy Rsebti/projet3-eval2-final

# Loop
while True:
    joint_pos = robot.read_joints()       # via lerobot
    image = camera.read()                  # via opencv
    color_oh = one_hot(target_color)
    obs = concat(joint_pos, color_oh, bowl_xyz)
    action = policy.act(image, obs)
    robot.write(action)
    sleep(1/30)
```

### I.2 — Pré-flight au labo

Avant l'éval :
- Hardware OK (= sanity-check / Eval 1)
- Le bowl + 2 cubes de la couleur cible + distractor placés selon les conditions TAs
- Bowl_xyz mesuré (ruban + repère robot) et passé en CLI
- target_color annoncé par les TAs et passé en CLI

### I.3 — 5 rollouts d'éval

Pour chaque rollout :
1. TAs placent les 2 cubes (paire de couleurs) + bowl
2. TAs annoncent target_color + mesurent bowl_xyz
3. Tu lances `python deploy/deploy_eval2.py --target_color X --bowl_x X --bowl_y X --bowl_z X --policy Rsebti/projet3-eval2-final`
4. Robot exécute jusqu'à succès (cube cible dans bowl + relâché) ou time-out (60s)
5. Note le résultat (succès / échec + cause)

Critère TA : 5/5 max = 50 pts ; chaque succès = 10 pts.

### I.4 — Si ça foire

Modes d'échec typiques + actions :

| Symptôme | Cause | Action |
|---|---|---|
| Robot grasp le mauvais cube | Mismatch couleur sim/réel ou mismatch index | Vérifier `COLOR_TO_INDEX` partout |
| Robot va à mauvaise position bowl | Mismatch convention frame robot | Mesurer bowl_xyz en frame robot exacte |
| Robot frozen | Action hors range, OOD obs | Sanity check obs vs training distribution |
| Robot fait n'importe quoi | Sim-to-real gap | Plus de domain randomization, retour Phase H |
| Drop hors bowl | Drop trop tôt / trop loin | Plus de démos focus sur le drop précis (Phase B refait) |

---

## Annexes

### Annexe 1 — Mapping couleurs (à figer maintenant)

```python
COLOR_TO_INDEX = {
    "blue":   0,
    "green":  1,
    "violet": 2,
    "yellow": 3,
    "red":    4,
    "orange": 5,
}
```

À mettre dans `sim/eval2/colors.py` dès Phase D pour qu'il soit la **seule** source de vérité.

### Annexe 2 — Budget temps + Brev

| Phase | Temps réel | GPU local | Brev |
|---|---|---|---|
| A Préparation | 1h | — | — |
| B Démos | 2-3h | — | — |
| C Push HF | 15 min | — | — |
| D Code training | 2-4h | — | — |
| E Augmentation | 1-2h | — | — |
| F Train ACT | 2-4h | RTX 5070 ou Brev | $0 ou $5-15 |
| G Étendre sim | 2-3h | — | — |
| H DAPG | 0.5j code + 6-12h training | — | $15-40 |
| I Deploy + eval | 1-2h | — | — |
| **Total** | ~25-35h | — | **~$20-55 sur les $200 dispo** |

### Annexe 3 — Fichiers du repo touchés / à créer

Existants à modifier :
- [`sim/eval2/joint_pos_env_cfg.py`](../sim/eval2/joint_pos_env_cfg.py) — spawn 6 cubes (Phase G)
- [`sim/eval2/pick_in_clutter_env_cfg.py`](../sim/eval2/pick_in_clutter_env_cfg.py) — observation 6D one-hot (Phase G)
- [`sim/eval2/mdp/observations.py`](../sim/eval2/mdp/observations.py) — target_color_one_hot(num_classes=6) (Phase G)
- [`sim/eval2/mdp/events.py`](../sim/eval2/mdp/events.py) — reset_target_color(num_classes=6) + reset_visible_pair (Phase G)
- [`sim/eval2/mdp/rewards.py`](../sim/eval2/mdp/rewards.py) — `_target_block_pos` dispatch sur 6 cubes (Phase G)
- [`sim/eval2/__init__.py`](../sim/eval2/__init__.py) — register `Eval2-PickInClutter-v3` (Phase G)
- [`sim/eval2/agents/rsl_rl_ppo_cfg.py`](../sim/eval2/agents/rsl_rl_ppo_cfg.py) — subclass DAPG (Phase H)
- [`deploy/eval2_inference.py`](../deploy/eval2_inference.py) — refactor pour image-in (Phase I)

À créer :
- `sim/eval2/colors.py` — constante COLOR_TO_INDEX (Phase D)
- `train/eval2_loader.py` — joint csv + dataset, étend obs 15D (Phase D)
- `train/eval2_augment.py` — albumentations transforms (Phase E)
- `train/dapg.py` — script training DAPG (Phase H)
- `deploy/deploy_eval2.py` — deploy real avec image-in (Phase I)

### Annexe 4 — Pièges connus

1. **Mismatch index couleurs** sim ↔ démos ↔ deploy → vérifier `sim/eval2/colors.py` est importé partout
2. **Convention frame robot** : axe x face robot, axe y vers la **gauche** (ou droite ? à confirmer avec Federico). Toutes les coords du csv supposent une convention donnée
3. **Stats normalization** dans LeRobot : si on étend `observation.state` à 15D, la mean/std calculées par lerobot sur les 6 premières dims (joints) restent OK, mais sur les 9 nouvelles (one-hot + xyz) ça peut être pourri si le csv a peu de variabilité — verify
4. **Camera index** : OpenCV idx 0 sur PC fixe (webcam intégrée) ou 1 sur laptop. À re-checker avec `lerobot-find-cameras`
5. **Fenêtre calib lerobot** : si tu re-calibres entre Phase B et Phase I, sauvegarder l'ancienne calib avant. Une mismatch calib casse les actions
6. **Random seed** : fixe-le dans tous les training runs. Sans, comparaisons impossibles
7. **`hf auth login`** : la token expire parfois. Re-loginer si push HF refuse
8. **Brev cost** : oublier de stop = $2/h * 24h = $48/jour bloquant. Mettre une alarme téléphone

### Annexe 5 — Si tu pivotes en cours de route

Si Phase F BC pure < 20% succès sim → c'est que les démos ne suffisent pas. Options :
- Refaire 60+ démos supplémentaires sur les configs où ça foire le plus
- Réduire l'augmentation (peut-être trop agressive)
- Vérifier la cohérence csv ↔ episode_index (les goals correspondent bien aux bonnes démos)

Si Phase H DAPG plafonne < 30% en sim → fallback :
- Plan B : pure BC en sim sans RL, on perd les points "RL mandatory" mais on a quelque chose
- Plan C : RL+BC simplifié (offline RL au lieu de DAPG online)

Si Phase I deploy < 2/5 → sim-to-real gap trop grand. Options :
- Plus de domain randomization Phase H, re-train
- Ajouter quelques démos physiques de fine-tuning (BC sur les vraies conditions du labo)

---

## Liens utiles

- TA spec PDF : [`notes/project3_rl_final_details.md`](project3_rl_final_details.md)
- Sanity check pipeline (réutilisable) : [`notes/full_pipeline_walkthrough.md`](full_pipeline_walkthrough.md)
- Plan démos (csv) : [`notes/eval2_demo_plan.csv`](eval2_demo_plan.csv)
- État du chantier sim : [`sim/eval2.md`](../sim/eval2.md)
- Audit IK abandonné : [`notes/eval2_ik_retrospective.md`](eval2_ik_retrospective.md)
- Setup Isaac Lab : [`notes/isaac_lab_setup.md`](isaac_lab_setup.md)
- Doc DAPG (paper) : Rajeswaran et al. 2017, "Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations"
