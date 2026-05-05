# Sim teleop pour Eval 2 — guide d'utilisation

Script : [`teleop.py`](teleop.py)

## Pourquoi

Générer des démos expertes pour l'**Option A : BC ACT + DAPG finetune**.
Au lieu d'attendre l'accès au robot physique pour téléopérer, on
téléopère directement le SO-101 simulé dans Isaac Sim.

Avantages :
- Pas besoin du robot physique
- Image data parfaitement alignée avec le training
- Variations gratuites (chaque reset randomise positions + couleur cible)
- Pas d'annotation manuelle (target_color et bowl_xyz auto-loggés)

Limite :
- Sim-to-real gap reste, mitigé par domain randomization plus tard

## Deux modes : `joint` et `cartesian`

### Lancement — mode CARTESIAN (recommandé pour pick-and-place)

Tu commandes seulement le **tip** du gripper (xyz dans l'espace), l'IK
calcule les angles des joints automatiquement.

```powershell
cd C:\Users\user\Desktop\MA2\isaac\isaac_so_arm101
uv run python -m sim.eval2.scripts.teleop `
    --task Eval2-PickInClutter-Play-v2 `
    --mode cartesian `
    --output_dir C:\Users\user\Desktop\MA2\robot-learning-project3\data\teleop_demos_eval2
```

| Action | Touches |
|---|---|
| Tip **avance** (+x) | **→** |
| Tip **recule** (-x) | **←** |
| Tip **va à droite** (-y) | **↓** |
| Tip **va à gauche** (+y) | **↑** |
| Tip **monte** (+z) | **PageDown** |
| Tip **descend** (-z) | **PageUp** |
| Tilter gripper (plus vertical / plus tilté) | **P / ;** |
| Toggle gripper open/close | **Espace** |
| Reset env (discard épisode) | **N** |
| Save épisode + reset (mark success) | **M** |
| Quit | **Échap** |

Step size : 5 mm/frame en xyz (ajustable via `--cartesian_step_xyz`).

⚠️ **On évite Q/W/E/R/T/F** parce qu'Isaac Sim hijack ces touches pour
sa toolbar manipulateur (Sélection, Move, Rotate, Scale).

⚠️ **Si le robot ne bouge plus dans une direction**, c'est que l'IK a
saturé (cible hors de portée du SO-101 5-DoF). Recule (touche opposée)
et change de trajectoire. Ou utilise R/F pour ajuster le tilt du gripper.

### Lancement — mode JOINT (fallback robuste)

Tu commandes chaque joint individuellement. Pas d'IK donc pas de
saturation surprise, mais beaucoup moins intuitif.

```powershell
uv run python -m sim.eval2.scripts.teleop `
    --task Eval2-PickInClutter-Play-v2 `
    --mode joint `
    --output_dir C:\Users\user\Desktop\MA2\robot-learning-project3\data\teleop_demos_eval2
```

| Action | Touches |
|---|---|
| `shoulder_pan` +/- (rotation L/R du bras) | **← / →** |
| `shoulder_lift` +/- (lever / abaisser le bras) | **↑ / ↓** |
| `elbow_flex` +/- (plier / déplier coude) | **PageUp / PageDown** |
| `wrist_flex` +/- (incliner gripper) | **I / K** |
| `wrist_roll` +/- (rotation gripper sur axe) | **O / L** |
| Toggle gripper open/close | **Espace** |
| Reset env (discard épisode) | **N** |
| Save épisode + reset (mark success) | **M** |
| Quit | **Échap** |

Step size : 0.02 rad/frame (~1.15°). Ajustable via `--step_size`.

## Méthode de grasp (à respecter sur toutes les démos)

⚠️ **Critique pour BC** : si on ne suit pas la même méthode partout, la
policy apprend une moyenne de comportements incompatibles → useless.

1. **Approach top-down** : gripper vertical pointant vers le bas
2. Hover ~8 cm au-dessus du bloc cible (couleur annoncée dans le terminal au reset)
3. Descente verticale jusqu'à ~5 mm au-dessus de la table
4. **Espace** pour fermer le gripper sur le bloc
5. Lift vertical ~12 cm avant tout déplacement latéral
6. Transport au-dessus du bowl en gardant le gripper vertical
7. Descente à ~4 cm du fond du bowl
8. **Espace** pour ouvrir le gripper, le bloc tombe
9. Retreat vertical
10. **M** pour sauver l'épisode, l'env reset automatiquement

## Recovery

Si un grasp foire en cours :
- **N** pour discard l'épisode et redémarrer
- Mieux 80 démos clean que 100 avec recoveries pollués

## Format de sortie

Chaque épisode = un fichier `.npz` :

```
data/teleop_demos_eval2/
├── episode_000.npz
├── episode_001.npz
└── ...
```

Contenu d'un `.npz` :

| Clé | Shape | Type | Description |
|---|---|---|---|
| `joint_pos` | (T, 6) | float32 | Joints (5 arm + gripper) |
| `joint_vel` | (T, 6) | float32 | Vitesses |
| `action` | (T, 6) | float32 | Action commandée (5 arm + 1 gripper sign) |
| `image` | (T, H, W, 3) | uint8 | Wrist cam RGB |
| `target_color` | () | int32 | 0 = rouge, 1 = bleu |
| `bowl_xyz_b` | (3,) | float32 | Bowl en frame robot base |
| `block_red_xyz_b` | (T, 3) | float32 | GT (diagnostic, pas utilisé en deploy) |
| `block_blue_xyz_b` | (T, 3) | float32 | GT idem |

Les fichiers sont indexés à la suite : si le dossier contient déjà
`episode_007.npz`, le prochain run continue à `episode_008.npz`.

## Quantité visée

100 démos. Variez :
- Position cluster blocs (l'env randomise au reset, mais vérifie visuellement)
- Position bowl (idem)
- Couleur cible (auto, ~50/50 par l'env)

## Troubleshooting

**Le robot ne bouge pas quand j'appuie sur les touches** : assure-toi
que la fenêtre Isaac Sim a le focus (clique dessus). Les inputs Carb ne
sont captés que quand la fenêtre est focused.

**L'image dans `episode_X.npz` est trop sombre / trop claire** : c'est
juste le rendering Isaac Sim. La domain randomization en training
corrigera, mais si c'est trop extrême tu peux ajuster les lights dans
[`pick_in_clutter_env_cfg.py`](../pick_in_clutter_env_cfg.py).

**`scene["wrist_cam"]` n'existe pas** : tu lances sur la mauvaise
tâche. Utilise `Eval2-PickInClutter-Play-v2` (avec caméra), pas `-v1`.

**Le robot saturate les limits articulaires** : c'est normal, le step
clamp aux soft limits URDF. Si tu pousses contre la limite, le joint
ne bougera plus dans cette direction. Recule (touche opposée).

**Une démo prend > 30 secondes** : ralentit le step_size avec
`--step_size 0.01` pour plus de précision, mais essaie de rester
fluide. BC apprend la trajectoire entière, démos lentes = policy lente.
