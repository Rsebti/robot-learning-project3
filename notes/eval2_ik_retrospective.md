# Rétrospective IK — pourquoi on bug et faut-il tout refaire ?

> Audit honnête de ce qui marche, ce qui foire, et l'option de tout reset.
> Écrit le 2026-05-05 après ~2 jours de debug du scripted controller.

---

## 1. Distinction critique : math vs. opérationnel

### Ce qui MARCHE (math)
- IK closed-form mathématiquement correcte. **0.0001 mm** d'erreur sur 1000
  configs aléatoires (test_analytical_ik.py). Pas de bug dans la dérivation.
- Calibration des link lengths et offsets URDF↔IK : automatisée, mesurée
  empiriquement, vérifiée.
- State machine 9-phases : logique correcte.

### Ce qui FOIRE (opérationnel)
- En sim, le robot **n'atteint pas le tip target**. Stuck en APPROACH ou
  contorsionné.

**Donc le bug n'est PAS dans l'IK math.** Il est dans la couche entre
"l'IK donne X" et "le robot exécute X".

---

## 2. Liste exhaustive des sources de bug possibles

| # | Source possible | Diagnostic | Probabilité |
|---|---|---|---|
| A | IK math | Test roundtrip 0.0001 mm | **0 %** |
| B | Calibration L1/L2/L3 | Issue de measure_link_lengths.py automatisé | **0 %** |
| C | Convention IK↔URDF (offsets) | Test passe avec offsets, donc OK | **0 %** |
| D | Choix d'elbow branch (up vs down) | Home pose = elbow_down, on a pris elbow_down | Faible |
| E | Tip frame ≠ ce que l'IK calcule | `L3 = wrist_link → gripper_frame_link` ; `ee_frame` = FrameTransformer sur gripper_frame_link. Cohérent | Faible |
| F | Action scale mismatch | env `processed = action*0.5 + default`. On inverse `action = 2*(target - default)`. Cohérent | Faible |
| G | **Saturation joint au-delà des limites URDF** | Isaac Lab silently clamp les commandes hors limits. Si l'IK demande wrist_flex_urdf > 1.658, le joint est clampé et le tip n'atteint pas la cible | **HAUTE** |
| H | **PD tracking lag** | Effort limit du SO-101 = 1.9 N.m. Si on commande +2 rad de delta en 1 step, le PD met 30+ steps à converger | **HAUTE** |
| I | **Step-rate non limité** | On envoie l'IK target absolu chaque step, sans interpoler. Pour de gros déplacements, le PD lag derrière | **HAUTE** |
| J | Distance criterion sans timeout | Quand tracker fail, distance reste >POS_TOL forever → stuck. **Volontaire** mais fragile | Moyenne |
| K | Adaptive phi va trop bas (-2.20) → config articulaire que le PD ne peut pas tracker | Visible sur la photo : robot contorsionné figé | **HAUTE** |
| L | `elbow_up=False` mauvais pour certains targets | Pour des cubes loin/bas, peut-être faut-il flip de branche | Faible (limites URDF empêchent le flip) |

**Verdict** : les bugs sont concentrés dans G, H, I, K — tous liés à
**l'interaction entre l'IK et le contrôleur PD du sim**. Pas dans l'IK
elle-même.

---

## 3. Pourquoi c'est si compliqué — est-ce normal ?

### C'est normal pour deux raisons réelles :

1. **SO-101 est 5-DoF avec un goal 6-DoF (xyz + orientation)** — c'est
   structurellement sous-déterminé. Toute approche aura ce problème. Les
   bras industriels (UR5, KUKA) sont 6-DoF *exactement* pour éviter ça.

2. **Limites articulaires du wrist_flex (±1.658) sont serrées** — pour
   beaucoup de positions de cube réalistes, demander "gripper vertical
   strict" sort des limites. Pas un bug, une contrainte mécanique.

### MAIS on a ajouté de la complexité accidentelle :

- **Couche 1** : IK closed-form avec offsets URDF.
- **Couche 2** : Phase state machine 9 phases.
- **Couche 3** : Adaptive phi search (recherche itérative de phi qui passe).
- **Couche 4** : Distance-based transitions sans timeout.

Chaque couche a sa propre interaction avec le PD du sim. Le débogage
demande de comprendre les 4 simultanément.

---

## 4. Faut-il tout refaire ? Trois options

### Option A — Continuer à patcher l'approche actuelle

**Idée** : ajouter step-rate limit + simplifier l'adaptive phi.

**Effort** : 2-4 h de debug supplémentaires.

**Risque** : on ajoute une 5e couche. Si ça foire encore, on a perdu 4 h
et on ne sait toujours pas si on est près du but ou pas.

### Option B — Réécrire avec des **waypoints en joint-space**

**Idée** : abandonner complètement l'IK. Pour chaque phase, parameter le
joint config désiré directement comme une fonction (linéaire ou
piecewise) du cube xyz et bowl xyz. Calibrer 3-5 configs "ancres" en
sim manuellement, interpoler entre.

**Pourquoi ça marche** :
- Plus de IK → plus de saturation surprise.
- On ne commande que des configs **mesurées en sim et donc
  forcément dans les limites**.
- PD lag minimal car les transitions sont planifiées en joint-space
  (pas de gros saut Cartesian → gros saut joint).
- Robuste au domain randomization (les anchors restent valides tant que
  les cubes restent dans la zone calibrée).

**Effort** : 4-6 h pour calibrer + écrire le sequencer.

**Risque** : moins général que l'IK (workspace limité par les anchors).
Mais pour Eval 2 (cubes dans une fenêtre ~10 cm × 10 cm), c'est OK.

**Précédent** : c'est l'approche qu'utilisent BEAUCOUP de teams en
robotics research pour générer des démos expertes (ex : DAPG paper de
Rajeswaran utilise des trajectoires hand-scripted, pas de l'IK).

### Option C — Réécrire avec un **IK numérique via Pinocchio ou PyBullet**

**Idée** : remplacer notre IK closed-form par un solveur LM (Levenberg-
Marquardt) qui optimise *librement* tous les 5 DoF, avec une fonction
de coût qui pénalise (a) distance au target xyz (b) distance aux
limites articulaires (c) déviation d'un orientation préférée (gripper
vertical, mais soft).

**Pourquoi ça marche** :
- Pas de saturation : la pénalité aux limites empêche le solveur de les
  approcher.
- Orientation devient préférée, pas imposée → quand impossible, le
  solveur tilte le gripper sans planter.

**Effort** : 6-10 h (installer, intégrer, valider).

**Risque** : dépendance externe, comportement moins prédictible
(l'optimiseur peut converger vers des minima locaux différents selon
l'init).

---

## 5. Ma recommandation : **Option B**

### Justification

- L'IK math est correcte mais opérationnellement compliquée → switch à un
  truc plus simple.
- Pour générer des démos pour BC/DAPG, on n'a pas besoin de la solution
  optimale — juste d'une trajectoire qui pick le cube et le pose dans le
  bowl, fiable. Une lookup table de joint configs marche aussi bien.
- L'effort (4-6 h) est inférieur à continuer à patcher (effort cumulatif
  inconnu, déjà 2 jours).
- On peut **valider en sim immédiatement** : si on calibre 3 anchor configs
  et qu'on les rejoue, le robot se met dedans → succès garanti.

### Plan concret pour l'Option B

1. **Calibrer 5 joint configs ancres** (à la main, en bougeant le robot
   en sim) :
   - HOME : config par défaut.
   - ABOVE_CUBE : 3 cm au-dessus du cube, gripper aligné xy. Capture
     pour 5 positions de cube différentes (centre + 4 coins de la zone
     de randomization).
   - AT_CUBE : 5 mm au-dessus du sol, idem 5 positions.
   - ABOVE_BOWL : 8 cm au-dessus du bowl. Capture pour 3 positions de
     bowl.
   - AT_BOWL : 4 cm au-dessus du bowl floor, idem 3 positions.

2. **Fitter un mapping linéaire** `cube_xyz → joint_config` par
   régression linéaire sur les 5 anchors. Erreur d'extrapolation : ~5 mm
   tip position, OK pour grasp d'un cube de 2 cm.

3. **Replacement dans `scripted_controller.py`** :
   - Plus de `_solve_ik_adaptive_phi`.
   - À la place : `joint_config = anchor_table[phase] + dxy * gradient[phase]`.
   - Le state machine reste identique.

4. **Validation** : 50 envs × 100 episodes ≥ 80 % succès.

5. **Si OK** : on passe à la génération de démos.

### Pourquoi PAS l'Option C (Pinocchio)

- Trop de complexité externe.
- Pour un projet de cours avec deadline serrée, on veut l'approche la
  plus rapide à fiabiliser.
- Pinocchio est overkill — on n'a pas besoin de la généralité.

---

## 6. Ce qu'on garde / ce qu'on jette

### On garde
- ✅ La spec Eval 2 (cubes + bowl + couleur cible).
- ✅ Le pipeline BC + DAPG (toute l'archi RL).
- ✅ `analytical_ik.py` (utile comme outil de validation, à garder en
  référence — on pourrait l'utiliser dans Pinocchio plus tard).
- ✅ `measure_link_lengths.py` (les valeurs sont correctes et utiles
  même pour l'option B).
- ✅ Le state machine (phases et transitions).

### On jette / on réécrit
- ❌ `_solve_ik_adaptive_phi` (Option B remplace).
- ❌ La contrainte phi=-π/2 (Option B s'en fout).
- ❌ La transition distance-based (on revient à des transitions par
  step count, qui sont prédictibles avec des anchors joint-space).

---

## 7. Décision à prendre maintenant

**Question pour Rayane** :

1. On part en Option B (réécrire en joint-space anchors) ?
2. Ou on continue à patcher l'Option A pendant encore 2-4 h max ?

Recommandation : **Option B**. Time-box 6 h. Si à 6 h on n'a pas le
50 envs ≥ 80 %, on aura appris ce qui foire et on pourra basculer en
Option C. Mais je suis ~80 % confiant que B passe.

---

## 8. Note méta : qu'est-ce qu'on a appris ?

- **Ne pas utiliser un solveur IK générique pour 5-DoF + contrainte
  d'orientation**. DifferentialIK n'était pas adapté. On a passé du
  temps là-dessus avant d'accepter de coder du closed-form.
- **L'IK closed-form est correcte mais demande beaucoup de
  papier-crayon de validation**. Ne pas sous-estimer le temps de
  calibration des conventions URDF↔IK.
- **Pour de la génération de démos d'entrainement, scripted joint-space
  > scripted Cartesian-IK**. C'est moins général mais 10× plus fiable.
  On retrouve ça dans tous les papers d'imitation learning sérieux.
- **Time-boxer chaque approche**. On a passé ~2 jours à patcher l'IK,
  c'était trop. Limite à 1/2 journée par approche, puis pivot.
