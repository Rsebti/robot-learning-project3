# Eval 2 — Pipeline complète (LeIsaac + PPO + curriculum)

> **Document de référence** pour exécuter Eval 2 du début à la fin.
> Stratégie : **Pure RL**, pas de BC, pas d'humain dans la boucle. Scaffold
> LeIsaac (clean SO-101 USD + canonical wrist cam). PPO via rsl_rl —
> on a itéré 7 variantes d'hyperparams + reward shaping pour identifier
> les bugs. Curriculum learning au niveau de l'env, pas de la policy.
>
> **Date de rédaction initiale** : 2026-05-08, après pivot depuis isaac_so_arm101.
> **Dernière maj** : 2026-05-09 (V2.9 chaos diagnostiqué via play_diagnose_v2,
> V2.10 = smoothness fixes + USD edits cube 2cm + table #B8ADA9 + obs
> pre-allocation Phase C).

---

## TL;DR — les 4 phases

```
Phase A (~10h)        smoke test state-only lift              [VALIDÉ ✓]
Phase B (~22h/run)    wrist cam + ResNet, 9 variantes PPO     [EN COURS — V2.10]
Phase C (~1 semaine)  Eval 2 par curriculum (5-6 paliers, 6 couleurs)
Phase D (~1-2 jours)  deploy SO-101 réel + 5 rollouts TA
```

Estimation totale : ~10-15 jours de travail réel + ~5-10h GPU (~$30-80 Brev si on bascule).

---

## ⚠️ IMPORTANT — Eval 2 = 6 couleurs (pas 2)

Le projet Eval 2 utilise **6 couleurs distinctes** : `blue, green, violet, yellow, red, orange`. À chaque rollout TA, ils placent une **paire random** de cubes adjacents et donnent une **target_color** parmi les 6.

Mapping figé (à utiliser PARTOUT — sim, training, deploy) :

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

**Implications archi** :
- `target_color_one_hot` ∈ ℝ⁶ (pas ℝ²)
- Scène spawn TOUJOURS les 6 cubes ; au reset, 4 sont cachés à `z=-10` (hors scène, invisibles à la cam) et 2 sont visibles dans le cluster
- Reward dispatche sur l'index target_color via `_target_cube_pos(env)` / `_distractor_cube_pos(env)`

---

## Pourquoi cette stratégie

### Ce qu'on évite

| Approche | Verdict | Raison |
|---|---|---|
| BC seul | ❌ | Spec TA exige RL pour Eval 2 |
| PPO from-scratch sur isaac_so_arm101 | ❌ testé | Defaults instables (entropy explosion v1.x), ET bugs côté env (convex_decomposition gripper, curriculum) |
| Scripted IK pour générer démos sim | ❌ archivé | SO-101 5-DoF + wrist_flex serré → grasp pas fiable |
| HIL-SERL real-robot | ❌ | Demande humain au gamepad |
| ManiSkill (lerobot-sim2real) | ❌ | User a choisi de garder Isaac Lab |
| **LeIsaac LiftCube + rewards custom + PPO + curriculum** | ✅ choisi | Robot USD propre, wrist cam canonique, scaffold maintenu |

### Lessons archived attempts

Tout est dans la branche `archive/eval2-attempts-pre-reset` :
- v1.0–v1.4 PPO from-scratch state-based plafonné à 2-7%
- v2 module CNN perception → val_mae 0.87 cm mais entraîné sur policy chaotique
- Scripted IK + magic-attach abandonné (gripper bloqué à 0.36 mid-close avec convex_decomposition)

---

## État actuel (2026-05-08)

### Ce qui est en place ✅

- **Repo réorganisé** post reset 2026-05-07 : `sim/eval2/` recréé proprement
- **LeIsaac installé** dans le venv `isaac_so_arm101/.venv` :
  - Code Python via `uv pip install -e ../leisaac/source/leisaac`
  - USD assets téléchargés dans `C:\Users\user\Desktop\MA2\isaac\leisaac\assets\` :
    - `scenes/table_with_cube/` (depuis GitHub release v0.1.2, 5 MB)
    - `robots/so101_follower.usd` (depuis HF LightwheelAI/leisaac_env, 23 MB)
- **Notre repo installé** dans le même venv (éditable, `uv pip install -e <repo>`)
- **`LEISAAC_ASSETS_ROOT`** auto-set dans `sim/eval2/__init__.py` (sinon LeIsaac détecte le mauvais git root)

### Tasks Gym registered

```
sim/eval2/__init__.py
  Phase A / Phase B sur isaac_so_arm101 (legacy, archivé)
  ├── Isaac-SO-ARM101-Lift-Cube-Restricted-v0          (defaults, DIVERGE)
  ├── Isaac-SO-ARM101-Lift-Cube-Restricted-Stable-v0   (V2, local optimum)
  └── Isaac-SO-ARM101-Lift-Cube-Restricted-V2-v0       (V2, curriculum off)

  Phase A / Phase B sur LeIsaac (actif)
  ├── Isaac-LeIsaac-SO101-Lift-RL-v0          / -Visual-v0          (V2, [Phase A ✓ 695, Phase B 185 stalled])
  ├── Isaac-LeIsaac-SO101-Lift-RL-IsaacDefaults-v0  / -Visual-       (IsaacDefaults, DIVERGE confirmé)
  ├── Isaac-LeIsaac-SO101-Lift-RL-V25-v0      / -Visual-V25-v0      (V2.5 HW4-inspired)
  ├── Isaac-LeIsaac-SO101-Lift-RL-V26-v0      / -Visual-V26-v0      (V2.6 — adaptive KL, LR collapsed)
  ├── Isaac-LeIsaac-SO101-Lift-RL-V27-v0      / -Visual-V27-v0      (V2.7 — flick-exploit fixes)
  ├── Isaac-LeIsaac-SO101-Lift-RL-V28-v0      / -Visual-V28-v0      (V2.8 — Isaac Lab canonical PPO, "marché moyennement")
  └── Isaac-LeIsaac-SO101-Lift-RL-V285-v0     / -Visual-V285-v0     ⭐ V2.8.5 — 4 fixes empiriques + cube_dropped
```

### Bugs résolus

| Bug | Sévérité | Fix |
|---|---|---|
| `entropy_coef=0.006` (rsl_rl legged default) → entropy explosion | 🔴 critique | V2 : `entropy_coef=0.002` ; IsaacDefaults : test si LeIsaac env clean suffit à éviter le bug |
| `init_noise_std=1.0` trop bruité au démarrage | 🟡 | V2 : 0.6 ; IsaacDefaults : 1.0 (test) |
| `schedule="adaptive"` étrangle LR | 🟡 | V2 : "fixed" + LR 3e-4 ; IsaacDefaults : "adaptive" + LR 1e-4 (test) |
| `gamma=0.98` trop long horizon pour pick | 🟡 | V2 : 0.95 ; IsaacDefaults : 0.98 (test) |
| Curriculum isaac_so_arm101 ramps action_rate à -1e-1 | 🔴 critique | Désactivé via `num_steps=10**12` ; puis pivot LeIsaac (sans curriculum) |
| `convex_decomposition` gripper self-blocks à +0.36 | 🔴 critique | Pivot LeIsaac (USD propre, pas de convex_decomposition) |
| Cube spawn / goal hors task space SO-101 | 🔴 | Ranges restreintes : cube ±5cm en x, ±10cm en y ; goal x∈±5cm, y∈[-20,-10]cm, z∈[10,20]cm |
| `LEISAAC_ASSETS_ROOT` mal résolu | 🟡 | Auto-set dans `sim/eval2/__init__.py` |
| `lifting_object` fire à iter 1 (cube starts above world z 0.025 à cause de la table élevée) | 🔴 critique | Reward custom `cube_lifted_above_base` qui mesure cube z **relatif au robot base** (height_threshold=0.05) |
| Success metric `cube_height_above_base > 0.20m` mal aligné avec goal range pos_z=(0.10, 0.20) | 🟡 | Replaced by `cube_reached_goal` (3D distance < 5cm to commanded goal) |
| Phase B GPU OOM à 1024 envs avec wrist cam | 🔴 | num_envs descendu à 256 dans Visual env cfg |
| Stagnation Phase B V2 (lifting plafonne ~0.7, position_error stagnant) | 🟡 résolu via V2.7 | Stack des 4 fixes empiriques (voir section Audit) |
| **Flick exploit V2.6** (cube éjecté en l'air sans grasp, lifting=0.78 sans grasping=0.001) | 🔴 critique | V2.7 : `cube_lifted_and_grasped` (gate AND grasp ∧ lift), `cube_to_goal_distance_grasped_and_lifted` (idem), seuils grasp relâchés (diff 0.02→0.04, grasp 0.26→0.35), lifting weight 15→10 |
| **Adaptive KL collapse V2.6** (LR 3e-4 → 2e-5 en 100 iter, policy starvée) | 🔴 critique | V2.7 : `desired_kl=0.01 → 0.03` ; V2.8 : revenir à `desired_kl=0.01` mais avec `num_learning_epochs=5` (au lieu de 10) pour réduire le drift KL par rollout |
| **VF starvation V2.7** (Loss/value_function oscille 0.05→0.3, entropy plateau, success_bonus=200 ne se laisse pas prédire) | 🔴 critique | V2.8 : `value_loss_coef=0.01 → 1.0` (×100), Isaac Lab canonical PPO defaults (LR=1e-3, n_epochs=5, n_mini=4, init_noise=1.0, max_grad_norm=1.0) |
| **Lifting threshold "déjà lifted" au spawn** (cube.z - base.z = +0.0515 au reset, threshold était 0.05) | 🟡 → 🔴 V2.8.5 | V2.8.5 : `_V285_LIFT_HEIGHT_THRESHOLD=0.08` partout (lifting + tracking) — confirmé par `audit_scene.py` |
| **`reaching_object` std=0.05 trop tight** (EE→cube=0.24m au reset → reward 0.0001/step, no gradient) | 🔴 critique | V2.8.5 : `std=0.05 → 0.15` (×730 plus de signal à distance reset) |
| **`success_bonus=200` trop petit** (foregone hover discounted = +1900, donc PPO préfère hover à 5.1cm que finir) | 🔴 critique | V2.8.5 : `weight=200 → 1500` (couvre ~80% du foregone) |
| **Pas d'early failure cutoff** (cube tombé → 120 steps de zero-reward qui polluent l'advantage) | 🟡 | V2.8.5 : `cube_dropped` DoneTerm (world.z < 0.04 m, threshold validé empiriquement) |
| **Threshold `cube_dropped` initialement relatif à base, mais base.z ≈ floor** (la base du robot est à +0.01 world, donc relatif-base ne pourrait jamais déclencher) | 🟡 | V2.8.5 : passé en world-frame absolu (`world_z_threshold=0.04`) après mesure `audit_scene.py` |

---

## Phase A — Smoke test state-only lift [VALIDÉ ✓]

### Objectif
Valider que PPO converge sur lift simple SO-101 dans l'env LeIsaac, sans cam, sans Eval 2 specifics.

### Architecture
- Obs : 28D state-only (joint_pos 6 + joint_vel 6 + cube_pos 3 + goal_pose 7 + last_action 6)
- Action : 6D (5 arm joints scale=0.5 + 1 binary gripper)
- Reward : 6 termes (canonical Isaac Lab Lift adapté pour table élevée — voir section reward)
- Pas de curriculum
- Episode 5s, decimation=2, 4096 envs
- PPO V2 (gamma=0.95, entropy_coef=0.002, schedule=fixed, init_noise=0.6, LR=3e-4)

### Résultats
Run du 2026-05-08_01-26-42, arrêté à iter 695/1500 (~9h sur RTX 5070) :

| Métrique | Iter 695 |
|---|---|
| Mean action noise std | 0.18 (descendu de 0.6) |
| `Loss/entropy` | descend monotone (de ~5.5 à ~3) |
| `lifting_object` | 8.41 / 15 (~55%) |
| `object_goal_tracking` | 6.24 / 16 |
| `position_error` | 0.11m (descendu de 0.18m) |
| `success_rate` (nouveau metric) | 0.0001-0.001 (premiers succès) |
| Visuel : 4 robots play | ~50% des rollouts grasps + lifts vers goal, mouvements brusques |

### Conclusion Phase A
✅ **Le scaffold LeIsaac + V2 PPO converge.** La policy apprend pick-and-lift sur SO-101 en sim.
⚠️ Mouvements brusques (pas de pénalité forte sur smoothness — à fixer avant deploy réel).

---

## Phase B — Wrist cam + ResNet encoder [EN COURS]

### Objectif
Vérifier que la pipeline visuelle (wrist cam + frozen ResNet-18) converge aussi bien que Phase A.

### Architecture
- Obs : 540D = 28D state + 512D ResNet features
- ResNet-18 ImageNet, gelé en `eval()` mode (BN stats fixes), encoder pré-calcule features dans l'obs term `wrist_image_features` (pas dans la policy → buffer rollout petit)
- Wrist cam canonique LeIsaac : pos=(-0.001, 0.1, -0.04), focal=36.5, render 224×224 directement
- Image normalisée ImageNet `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]` dans l'encoder
- num_envs = 256 (≥1024 = OOM sur RTX 5070 12GB à cause des G-buffers / DLSS de chaque cam)

### Chronique des 9 variantes Phase B (V2 → V2.10)

Chaque variante teste une hypothèse précise. La progression est causale : chaque échec révèle un bug qu'on corrige dans la suivante.

| # | Variante | Changements vs préc. | Verdict | Lesson learned |
|---|---|---|---|---|
| 1 | **V2** | baseline lerobot-sim2real | 🟡 plafonne (`lifting=0.62, success≈0`) | Reward landscape OK pour lift mais pas pour transport |
| 2 | **V2 + `grasping_cube`** | + reward intermédiaire grasp (weight=5) | 🟡 grasping monte un peu, plafond persistant | Le gap reaching→lifting est comblé mais autre bug en aval |
| 3 | **IsaacDefaults** | replay defaults isaac_so_arm101 | 🔴 DIVERGE (entropy explosion) | Hypothèse "bugs côté env seulement" RÉFUTÉE — defaults structurellement mauvais |
| 4 | **V2.5** | V2 + 4 changements HW4 (`value_loss=0.01`, `n_epochs=10`, `gamma=0.99`, `hidden=[256,128,128]`) | 🟡 plafonne pareil que V2 (lifting 0.78, success≈0) | Headlines HW4 pas suffisants, manque la boucle d'adaptation KL |
| 5 | **V2.6** | V2.5 + 3 ajouts HW4 (`entropy=0.005`, `use_clipped_value=True`, `schedule="adaptive"` + `desired_kl=0.01`) | 🔴 LR collapse (3e-4 → 2e-5 en 107 iter) ET **flick exploit identifié** (lifting=0.78 sans grasping) | (a) KL adaptive avec desired_kl=0.01 trop tight pour 256 envs × 10 epochs ; (b) `lifting_object` n'est PAS gated sur grasp → flick paye |
| 6 | **V2.7** | V2.6 + flick fixes (`cube_lifted_and_grasped`, gate tracking sur grasp, seuils grasp relâchés) + `desired_kl=0.03` + `success_bonus=200` | 🔴 VF starvation (Loss/value_function oscille 0.05-0.3 sans descendre, entropy plateau) | `value_loss_coef=0.01` empêche la VF d'apprendre les spikes sparses du `success_bonus`. Le shaping causal est OK mais le critic ne suit pas. |
| 7 | **V2.8** | V2.7 + Isaac Lab canonical PPO (`value_loss_coef=1.0` ×100, `n_epochs=5`, `n_mini=4`, `LR=1e-3`, `desired_kl=0.01`, `init_noise=1.0`, `max_grad_norm=1.0`) | 🟡 "marché moyennement" — VF se stabilise, mais reach saturé + success rare | Le critic apprend, mais bugs reward shaping résiduels révélés par audit |
| 8 | **V2.8.5** | V2.8 + 4 fixes empiriques : `lift_height_threshold=0.05→0.08`, `reaching_object.std=0.05→0.15`, `success_bonus.weight=200→1500`, `cube_dropped` DoneTerm | 🟡 41% drop rate | Voir section "Audit empirique" |
| 9 | **V2.9** | V2.8.5 + `cube_dropped_penalty=-5` + reach fix (cube shifted -13cm en y) | 🟡 60% deterministic success **mais yeet exploit** : grasp dur 2.2 steps, cube éjecté à 5 m/s. Action histogram \|p95\|=6.4 (saturé). 50% des successes ont `grasp_count=0` (scoop sans grasp formel) | Diagnosed via `play_diagnose_v2` |
| 10 | **V2.10** | V2.9 + smoothness stack : `action_rate ×500` / `joint_vel ×100` / `joint_acc NEW` / `arm_action.scale 0.5→0.25` / `init_noise 1.0→0.4` / `entropy 0.005→0.002` + LeIsaac stock randomisation + USD edits (cube 2cm, table #B8ADA9) + obs pre-allocation Phase C | 🔴 **paralysie** (cold-start ET warm-start). `reaching` descend 1.5% → 0.42% en 95 iters | `joint_acc_l2 = -1e-3` produit -50/épisode vs -0.7 pour les deux autres smoothness → gradient "ne pas accélérer" écrase "approche le cube". Le warm-start V2.9 ne préserve pas non plus le yeet (programme moteur balistique non transférable au régime smooth). |
| 11 | **V2.10b** | V2.10 avec `joint_acc_l2 -1e-3 → -3e-5` (÷33), tout le reste identique. Cold-start uniquement. | 🟡 smoothness OK mais reaching plat à 1.7%, policy drift loin du cube | `joint_acc` n'écrase plus le gradient (3 termes en parité), MAIS `reaching tanh std=0.15` sature à d>60cm → la policy s'éloigne sans signal pour revenir |
| 12 | **V2.10c** | V2.10b + nouveau RewTerm `ee_to_cube_distance` linéaire `-||EE-cube||` weight=-1.0. Cold-start. | 🟡 marche mais lent (reaching 1.7% → 4.1% en 114 iters, projection convergence ~iter 1500+) | Linear distance débloque la dérive away from cube → policy s'approche enfin. Mais convergence trop lente parce que les poids smoothness V2.10 (héritages) sont trop forts. |
| 13 | **V2.11** ⭐ | Aligné Isaac Lab Lift défauts : smoothness `action_rate -1e-4, joint_vel -1e-4, joint_acc 0` ; gating `lifted` only (revert V2.7 flick fix) ; lifting=15 (Isaac Lab) ; entropy_coef=0.005 ; cube_dropped_penalty=-5 (gardé) ; success_bonus=1500 (gardé). Keeps V2.10's action.scale=0.25, USD edits, randomization, placeholders, et V2.10c's `ee_to_cube_distance`. Cold-start. | 🟡 EN COURS | Reset après audit research: pratique publiée met smoothness dans action space, pas dans reward. Un seul innovation gardée (linear distance), reste = Isaac Lab défauts. |

### Audit empirique de la scène (entre V2.8 et V2.8.5)

Suite à V2.8 "moyen", on a écrit deux scripts diagnostiques pour vérifier en sim toutes les hypothèses qui étaient hardcodées dans le code sans validation empirique :

- [`sim/eval2/scripts/measure_cube_height.py`](../sim/eval2/scripts/measure_cube_height.py) — distribution de `cube.z` et `base.z` au reset
- [`sim/eval2/scripts/audit_scene.py`](../sim/eval2/scripts/audit_scene.py) — audit complet : body names, joint ranges, goal distribution, FrameTransformer targets, distances EE↔cube↔goal au reset, episode timing, rewards à reset

**12 hypothèses validées** (body names corrects, joint ranges OK, gripper open/close dans range, FrameTransformer cohérent, episode = 5s × 30Hz = 150 steps, etc.).

**4 bugs majeurs révélés** :

| Bug | Mesure empirique | Impact |
|---|---|---|
| **"déjà lifted" au spawn** | `cube.z - base.z = +0.0515` au spawn, threshold `cube_lifted_above_base` était 0.05 | Le gate était presque toujours ouvert au reset → signal lift bruité, gate goal_tracking leak |
| **`reaching_object` std trop tight** | EE→cube = 0.24m au reset, `1 - tanh(0.24/0.05) = 0.0001/step` | Aucun gradient pour approcher le cube avant que l'EE soit déjà dans les 10cm — explique le "wandering" qu'on voyait |
| **`success_bonus=200` trop petit** | Reward dense à hover (cube near goal, grasped+lifted) ≈ +30/step. Foregone discounted reward = `Σ_{k=0..99} 0.99^k × 30 ≈ +1900`. Le bonus 200 ne couvre que 10% — PPO préfère hover à 5.1cm que finir | Aucun incentive concret à fermer la tâche |
| **Base du robot ≈ niveau du sol** | `base.z (world) = +0.0100`, table top à `cube.z - cube_half_extent = +0.0415` | Le `cube_dropped` initialement défini relatif à la base ne pouvait jamais déclencher — switché en world-frame absolu |

**4 fixes V2.8.5 (uniquement env-side, PPO V2.8 inchangé)** :

| # | Fichier | Avant | Après |
|---|---|---|---|
| 1 | `RewardsCfgV285.lifting_object` (et 2 tracking terms) | `height_threshold=0.05` | `height_threshold=0.08` (3 cm marge au-dessus spawn) |
| 2 | `RewardsCfgV285.reaching_object` | `std=0.05` | `std=0.15` (×730 plus de signal à distance 24cm) |
| 3 | `RewardsCfgV285.success_bonus` | `weight=200` | `weight=1500` (couvre 80% du foregone hover) |
| 4 | `LeIsaacLiftCubeRLEnvCfgV285.terminations.cube_dropped` | (n'existait pas) | NEW — `cube_dropped` DoneTerm, `world_z_threshold=0.04` |

### V2.9 — diagnostique complet via `play_diagnose_v2`

V2.9 a tourné jusqu'à iter 1500 et atteint **60% deterministic success rate**
au play. Au premier abord encourageant — mais le diagnostic via le script
`play_diagnose_v2.py` (avec event-based logging, action histograms, grasp
lifecycle, phase tracker) a révélé que **les "successes" sont des yeet
ballistiques chanceux**, pas des vraies trajectoires pick-and-place.

**Métriques critiques V2.9 deterministic** :

| Métrique | Valeur | Interprétation |
|---|---|---|
| Success rate | 60% | Atteint le goal sphere (5 cm) |
| `Time to first grasp` | 4.0 ± 0 steps | Quasiment fixe — bras yeet en 4 steps |
| `Time to success` | 5.5 ± 1.1 steps | Cube atteint le goal 1-2 steps après le grasp = projection ballistique |
| `Grasp duration mean` | 2.2 steps | Cube ejected en ~70 ms |
| `P(grasp ≥ 5 steps)` | 7.4% | Quasi jamais soutenu |
| `Mean cube speed at GRASP_FIRE` | 2.21 m/s | Bras frappe le cube |
| `Mean gripper qdot at GRASP_FIRE` | +1.6 rad/s | **Gripper OUVRE** au moment du grasp (predicate fire par coïncidence) |
| `Action histogram \|p95\|` | jusqu'à 6.4 | Policy outputs CLIPS à ±1, mais produit ±6+ |
| `Mean \|qdot\|max/step` | 8.66 / 10 | Joints saturés en permanence |
| `% successes WITHOUT grasp_count` | 50% (6/12) | **Scoop exploit** — cube poussé au goal sans grasp formel |
| `goal_z high` success rate | 0/5 | Yeet ne peut atteindre les goals hauts (gravité) |

**Conclusion V2.9** : la policy a appris une stratégie **chaotique mais
chanceuse** :
1. Joints saturés en permanence à ±10 rad/s (limite PhysX)
2. Bras whippe vers le cube en 4 steps
3. Coïncidence : cube proche du jaw + gripper partiellement fermé → predicate
   `cube_grasped` fire pour ~2 steps
4. Cube éjecté par l'inertie du bras dans la direction du goal
5. Si chance, cube traverse la sphère 5cm du goal → success
6. 50% des "successes" sont des scoops (cube poussé au goal sans grasp)

**V2.9 est inutilisable pour deploy réel** :
- Servos Feetech max ~360°/s, V2.9 commande jusqu'à 8500°/s (clip ramène à
  570°/s, toujours violent)
- Le yeet repose sur des contacts cinétiques que le sim approxime mal
- Le scoop ne marchera pas sur friction réelle

### V2.10 — design rationale et changements

V2.10 attaque chaque pathologie V2.9 avec un fix ciblé.

**Smoothness stack (côté reward)** :

| Fix | V2.9 | V2.10 | Pourquoi |
|---|---|---|---|
| `action_rate_l2` weight | -1e-4 | **-5e-2** (×500) | Action histogram \|p95\| = 6.4 → veut ramener à <1.5. Pénalise step-to-step changes |
| `joint_vel_l2` weight | -1e-4 | **-1e-2** (×100) | Mean \|qdot\|max = 8.66 → veut <3. Pénalise vitesse joint absolue |
| `joint_acc_l2` weight | n/a | **-1e-3 NEW** | Spikes impulsifs à 86 rad/s. Pénalise dérivée de la vitesse |
| `arm_action.scale` | 0.5 | **0.25** | Mécanique : delta joint /2 par step. Cap matériel sur le mouvement |

**PPO config (côté policy)** :

| Param | V2.9 | V2.10 | Justification |
|---|---|---|---|
| `init_noise_std` | 1.0 | **0.4** | V2.9 mean policy ≠ stochastic policy car bruit énorme. 0.4 → mean proche de la déployée. **Complémentaire à smoothness** : démarre directement dans le régime "petites samples" → moins de pénalité au step 0 → convergence plus rapide |
| `entropy_coef` | 0.005 | **0.002** | Réduit le bonus pour exploration aléatoire. Smoothness devient la principale forme d'exploration |

**Env spec match** :

| Modif | V2.9 | V2.10 | Source |
|---|---|---|---|
| Cube taille | 3 cm (LeIsaac default) | **2 cm** (USD `xformOp:scale=2/3`) | Spec TA |
| Table couleur | bois LeIsaac | **#B8ADA9** (USD PreviewSurface override) | Spec TA |
| Cube spawn xy range | x±3cm, y∈(-16,-10) (5×5cm) | **x±7.5cm, y±7.5cm, yaw±30°** | LeIsaac stock (15×15cm) |
| Cube init z | 0.0615 | **0.0565 (-5mm)** | Compense cube 2cm pour rester sur table |

**Phase C compatibility (NEW)** :

V2.10 ajoute 9D de placeholders zero-valued dans la policy obs :
- `target_color_placeholder` : 6D (one-hot 6 couleurs Eval 2)
- `bowl_xyz_placeholder` : 3D (position bowl en frame robot)

→ Obs dim V2.10 = obs dim Phase C → checkpoint warm-startable Phase C
sans rebuild network from scratch.

**USD modifications LeIsaac (avec backup automatique)** :

Scripts dédiés dans `sim/eval2/scripts/` :
- `resize_cube_usd.py` : ajoute `xformOp:scale=(2/3, 2/3, 2/3)` au prim cube
- `recolor_table_usd.py` : crée `UsdPreviewSurface` material #B8ADA9, bind sur `counter_right_main_group`

Backup en `scenes/table_with_cube/scene.usd.bak` (idempotent — preserve les modifs successives).

### Sanity check V2.9 sur new env (avant V2.10 launch)

V2.9 checkpoint testé sur la new env (cube 2cm + table grise + V29 pose_range)
avec `play_diagnose_v2` → **30% success** (vs 60% sur 3cm). Drop attendu (cube
plus petit = grasp plus dur), mais physique stable :

| Métrique | V2.9 sur 3cm | V2.9 sur 2cm (sanity) |
|---|---|---|
| Cube bbox | (0.030, 0.030, 0.030) | **(0.020, 0.020, 0.020)** ✓ |
| Success rate | 60% | 30% |
| Tip below table | 25% | **0%** ✓ (mieux !) |
| `Mean cube speed at GRASP_FIRE` | 2.21 m/s | 0.89 m/s (moins de yeet) |
| `Mean gripper qdot at GRASP_FIRE` | +1.6 (ouverture) | **-5.2 (fermeture)** ✓ |

→ Physique OK, V2.10 peut être lancée en confiance.

### V2.10 — résultats et diagnostic de l'échec (2026-05-09 → 2026-05-10)

**V2.10 cold-start** (Visual, init_noise=0.4, smoothness stack complet) →
paralysie totale. À iter 47 :
- `reaching_object` = 1.08% (descend depuis 1.49% à iter 18)
- `learning_rate` collapsé à 1e-5 (plancher adaptive KL)
- `joint_acc` pénalité = -17/épisode et trending **down** (la policy gèle)

Hypothèse formée à ce moment : peut-être que cold-start sans connaissance
préalable de la tâche est trop difficile sous ces poids → tenter un
warm-start depuis V2.9.

**V2.10 warm-start depuis V2.9** (padded checkpoint avec 9 dims zéros pour
les placeholders Phase C) → même paralysie, plus claire à diagnostiquer
parce que la baseline V2.9 (~30% success sur new env) était au point de
départ. À iter 95 :

| métrique | iter 4 | iter 95 |
|---|---|---|
| `reaching_object` | 0.73% | **0.42%** ↘ |
| `joint_acc` pénalité | -95 | -50 |
| `joint_vel` pénalité | -1.27 | -0.71 |
| `action_rate` pénalité | -1.15 | -0.71 |
| `mean_reward` | -497 | -260 |
| `value_function` | 1367 | 200 |

`reaching` descend de 1.5% à 0.42% en 95 iters — la policy **désapprend
activement** le reach V2.9. Le robot s'éloigne du cube (avg distance
EE↔cube = 45 cm au lieu de 24 cm au reset).

**Diagnostic causal — déséquilibre des 3 termes smoothness** :

À iter 95 du warm-start, contributions par épisode (sommé sur 150 steps) :

| terme | poids | contribution/ép |
|---|---|---|
| `action_rate_l2` | -5e-2 | -0.71 |
| `joint_vel_l2` | -1e-2 | -0.71 |
| `joint_acc_l2` | -1e-3 | **-50.27** |

`joint_acc` contribue **70× plus** que les deux autres réunis. La
gradient de "ne pas accélérer" écrase le gradient de "approche le cube".
Comme reach depuis le repos *exige* d'accélérer, PPO préfère l'inaction
à toute tentative — un piège local stable.

Le poids `-1e-3` sur `joint_acc_l2` avait été posé *à sec*, sans baseline
empirique de la magnitude. Avec |acc|² typique ~333 par step, il produit
des pénalités d'un ordre de grandeur supérieur aux deux autres termes.

**Hypothèse réfutée** : "le warm-start V2.9 → V2.10 préserverait le reach".
La V2.9 a appris un *programme moteur balistique* (yeet, joints saturés,
contacts impulsifs) qui n'est **pas une version rapide** d'un pick-and-place
lent — c'est un mode de contrôle structurellement différent. Quand PPO
réécrit le mapping `hidden → action` pour se conformer aux pénalités
smoothness, les features V2.9 se réajustent aussi. Après ~100 iters il ne
reste presque rien du checkpoint d'origine.

### V2.10b — design rationale (2026-05-10) — choix du poids `joint_acc_l2` audité

**Un seul changement** vs V2.10 : `joint_acc_l2 = -1e-3 → -3e-5` (÷33).

#### Pourquoi exactement ÷33 (calcul, pas heuristique)

Les 3 termes smoothness ne sont pas dans les mêmes unités physiques :

```
action_rate_l2 = Σ |Δa|²        (action units, post action.scale=0.25)
joint_vel_l2   = Σ |q_dot|²     ((rad/s)²)
joint_acc_l2   = Σ |q_acc|²     ((rad/s²)²)
```

Le ratio `|q_acc|² / |q_dot|² ≈ 712` (mesuré à iter 95 du warm-start).
C'est cohérent avec `q_acc ≈ Δq_dot/dt` et `dt ≈ 1/30s` → ²30 = 900×.

**Implication** : pour que `joint_vel` et `joint_acc` aient des contributions
commensurables, il faut `weight_acc ≈ weight_vel / 700`. V2.10 avait fixé
`-1e-3`, donc 70× au-dessus du bon ratio.

#### Cible chiffrée

Hiérarchie d'importance argumentée :
1. `joint_vel` : limite hardware directe (servos Feetech ~6 rad/s)
2. `action_rate` : smoothness des commandes (mesurable au deploy)
3. `joint_acc` : régularisation secondaire (redondant avec action_rate)

Budget smoothness total ciblé à régime "lent fonctionnel" : -3 à -5/ép.

| terme | observé iter 95 V2.10 | cible V2.10b |
|---|---|---|
| `action_rate` | -0.7/ép | -0.7/ép (gardé) |
| `joint_vel` | -0.7/ép | -0.7/ép (gardé) |
| `joint_acc` | -50.3/ép | **-1.5/ép** |

Pour une contribution -1.5/ép à raw_per_ep = 50270 (observé iter 95) :
`weight = -1.5 / 50270 ≈ -3e-5`.

#### Vérification cross-régime

Le même poids -3e-5 doit rester cohérent à différents régimes :

| régime | `|q_acc|_avg` | `Σ|q_acc|²/step` | raw/ép | contribution -3e-5 |
|---|---|---|---|---|
| V2.9 chaos (acc ~30 rad/s²) | 30 | ~5400 | ~810k | **-24/ép** (punit fort) |
| iter 95 V2.10 (acc ~7.5) | 7.5 | 335 | 50270 | -1.5/ép (cible) |
| smooth converged (acc ~3) | 3 | 54 | 8100 | -0.24/ép (négligeable) |

→ Le poids -3e-5 punit toujours le chaos balistique (-24/ép vs +1500
success_bonus = 1.6%) tout en s'effaçant à convergence smooth (-0.24/ép).

#### Vérification gradient task vs smoothness

Mouvement marginal de 1cm vers le cube depuis d=24cm, sur 100 steps :

- `reaching` gain (dérivée de `1 - tanh(d/0.15)`) : +0.85/ép
- `joint_acc` coût (acc² ~100/step typique) : weight × 100 × 100 = -0.3/ép

Net avec -3e-5 : **+0.55** → reach favorisé ✓
Net avec -1e-3 (V2.10) : -9.15 → reach défavorisé (paralysie observée)
Net avec -1e-4 (÷10) : -0.15 → quasi-indifférent (apprentissage trop lent)

→ -3e-5 est le seul des 3 candidats qui crée un gradient task net positif.

### V2.10c — design rationale (2026-05-10) — fix du `reaching` saturé

**Constat V2.10b après 22 iters cold-start** :
- Toutes les pénalités smoothness descendent monotone (action_rate -0.083 → -0.054, joint_vel -0.16 → -0.08, joint_acc -0.53 → -0.23) ✓
- noise_std descend (0.40 → 0.30), entropy descend (2.99 → 1.24) ✓ policy commit
- BUT `reaching_object` reste plat à ~0.017/ép et descend légèrement → la policy **s'éloigne du cube** (EE drift à d≈50cm)
- Per-step rate 0.000113 = `1 - tanh(d/0.15)` → tanh⁻¹(0.999) → d ≈ 0.50m (hors zone de signal)

**Diagnostic causal** : `reaching_object = 1 - tanh(d/0.15)` sature à d>60cm. À cette distance, gradient ≈ 0. Une fois la policy drift là-bas, aucun signal de reward ne la fait revenir. Le gradient smoothness ("ne bouge pas") gagne par défaut.

C'est un problème d'**absence de signal global**, pas de smoothness trop forte. V2.10b a réussi à ne pas paralyser, mais il faut maintenant un signal qui drive vers le cube *partout*, pas juste près.

**Fix V2.10c** : ajouter un nouveau RewTerm `ee_to_cube_distance` :

```python
ee_to_cube_distance = RewTerm(
    func=eval2_mdp.object_ee_distance_l2,  # raw ||EE - cube||
    weight=-1.0,
)
```

Effet : reward = `-d` linéaire à travers le workspace.

| d (cm) | tanh existant (std=0.15) | linéaire weight=-1.0 | gain Δd=-1cm tanh | gain Δd=-1cm linéaire |
|---|---|---|---|---|
| 60 | 0.001/step | -0.60/step | +0.0024 | **+0.010** |
| 45 | 0.005/step | -0.45/step | +0.006 | **+0.010** |
| 30 | 0.04/step | -0.30/step | +0.014 | +0.010 |
| 15 | 0.16/step | -0.15/step | +0.024 | +0.010 |
| 5 | 0.49/step | -0.05/step | +0.031 | +0.010 |

Le linéaire est plus fort à grande distance (>30cm), le tanh plus fort à courte distance (<15cm). Les deux sont **complémentaires** — le linéaire drive l'approche initiale, le tanh donne la précision finale.

**Mécanisme via PPO+GAE** :

À d=0.45m, mouvement de 1cm vers le cube :
- r_t instantané : -0.06 (smoothness) + 0.01 (linéaire) = -0.06
- V(s_t) ≈ -d/(1-γ) = -45 (rester à d=0.45 forever)
- V(s_{t+1}) ≈ -44 (même policy, mais à d=0.44 maintenant)
- **Advantage** = r_t + γ·V(s') - V(s) = -0.06 + 0.99·(-44) + 45 = **+1.38** ✓

L'advantage positif vient de la value function : être 1cm plus près est sustained pour les ~100 steps restants → V(s') > V(s) par +1, ce qui domine le coût instantané. Le critic encode cette différence en gradient continu vers le cube.

**Self-extinguishing en phase grasp/lift/transport** :

Une fois l'EE sur le cube (||EE - cube|| ≈ 0.02 m typique pendant grasp/transport), la pénalité linéaire ≈ -3/ép, négligeable face aux rewards positifs actifs (+5400/ép quand grasp+lift+tracking actifs). Le terme s'éteint naturellement et n'interfère pas avec les phases downstream.

**Pas d'exploit hover ou drop** :
- Hover à d=0.05 sans grasp : +124/ép (reaching tanh seul)
- Hover + grasp + lift + transport + success : +3874/ép (33× mieux)
- Drop le cube : pénalité immédiate -2282/ép (perd grasp+lift, gain linear pénalise massivement)

**Cible V2.10c à iter ~100** :
- `reaching_object` ≥ 30% (vs 1.7% V2.10b)
- `ee_to_cube_distance` reward ≈ -10/ép (passe de -67 à -10 = EE arrive proche du cube)
- `grasping_cube` > 0%
- noise_std en descente, entropy en descente

**Résultat V2.10c iter 27→114** :

| iter | reaching | ee_to_cube_distance | joint_acc | noise_std | entropy | mean_reward |
|---|---|---|---|---|---|---|
| 27 | 0.017 | -0.362 | -0.209 | 0.28 | +0.99 | -3.40 |
| 63 | 0.026 | -0.328 | -0.124 | 0.21 | -0.78 | -2.48 |
| 93 | 0.040 | -0.293 | -0.089 | 0.18 | -1.95 | -1.97 |
| 114 | 0.041 | -0.291 | -0.069 | 0.15 | -2.92 | -1.79 |

✓ Reaching ×2.4 vs V2.10b plat → linear distance fait son boulot
✓ Toutes les pénalités smoothness descendent
⚠ noise_std descend trop vite (0.40 → 0.15 en 114 iters → projection 0.07 à iter 300, quasi-deterministe)
⚠ Reaching plafonne à 4.1%, entropy à -2.92 = policy déjà très commitée
⚠ Aucun grasp/lift fired après 114 iters → la pince n'arrive pas à <5cm du cube

**Conclusion V2.10c** : le linear distance résout le pb de drive mais convergence trop lente parce que les poids smoothness V2.10 (héritages -5e-2, -1e-2) restent trop forts. La policy paie un coût important pour bouger même peu, et le critic learnt cette taxe en plus du gradient task — l'optimum local "approche modérément, reste vague" se forme.

### V2.11 — research-based reset (2026-05-10)

**Audit external research** sur les standards publiés (ManiSkill3, Isaac Lab Lift, robosuite, DextrAH-RGB, IndustReal) a révélé 4 divergences majeures de notre stack V2.5→V2.10c vs la pratique :

| divergence | published practice | notre V2.10c |
|---|---|---|
| smoothness via reward | -1e-4 noise floor (Isaac Lab Lift) ; smoothness réelle via action space (action.scale, vel clipping, geometric fabrics — DextrAH) | -5e-2 / -1e-2 / -3e-5 (×500-100-1) |
| `joint_acc_l2` | absent du Lift task ; -1e-7 dans locomotion templates | -3e-5 (×300 trop) |
| gating tracking | `lifted` only (Isaac Lab, robot-agnostic) ; `is_grasped` réservé aux Pandas avec contact propre | `grasp ∧ lift` (V2.7) — brittle sur SO-101 |
| stage envelope ordering | `max(reach) ≤ min(grasp) ≤ min(lift) ≤ min(track)` (robosuite invariant) | OK structurellement (1, 5, 10, 16) mais lifting weight 10 < Isaac Lab default 15 |

**V2.11 = retour aux défauts Isaac Lab Lift + nos 3 fixes valides** :
- USD edits cube 2cm (V2.10) — gardé (matches Eval 2 spec)
- Pose_range LeIsaac stock ±7.5cm + yaw ±30° (V2.10) — gardé
- Phase C placeholders 6+3 dims (V2.10) — gardé
- `action.scale = 0.5` (V2.9 default — l'attempt V2.11 v1 à 0.20 était basé sur une **erreur de compréhension** : `JointPositionActionCfg` avec `use_default_offset=True` est en mode **absolu** (`target = scale × action + default_pos`), donc scale=0.20 borne le **range articulaire** à ±0.20 rad (±11.5°) depuis home, **pas** la vitesse par step. Le diagnostic V2.11 v1 sur model_200 a confirmé : `action_sat_count = 150/150` à tous les épisodes, jaw atteint le niveau z du cube mais gripper reste à 12cm au-dessus → bras pas assez étendu latéralement). Avec scale=0.5 (±28.6°), V2.9 atteignait le cube. Protection yeet maintenant via `ee_to_cube_distance` linéaire, pas via cap mécanique inexistant.
- `ee_to_cube_distance` linéaire weight -1.0 (V2.10c) — gardé (notre seule innovation utile)
- Reward shaping V2.8.5 (height_threshold=0.08, reaching std=0.15, success_bonus=1500, cube_dropped_penalty=-5) — gardé

**Reset des poids vs V2.10/V2.10c** :
- `action_rate_l2` : -5e-2 → **-1e-4** (Isaac Lab default)
- `joint_vel_l2` : -1e-2 → **-1e-4** (Isaac Lab default)
- `joint_acc_l2` : -3e-5 → **0** (drop, absent du Lift défaut)
- gating `lifting_object` : `cube_lifted_and_grasped` → `cube_lifted_above_base` (lift-only)
- gating `object_goal_tracking{,_fine_grained}` : `cube_to_goal_distance_grasped_and_lifted` → `cube_to_goal_distance_above_base`
- weight `lifting_object` : 10 → **15** (Isaac Lab default)
- `entropy_coef` : 0.002 → **0.005** (Isaac Lab default, plus d'exploration headroom)

**Pourquoi le revert du V2.7 flick fix est safe en V2.11** :

V2.7 a ajouté le grasp gate parce que V2.6 avait des actions chaotiques qui produisaient des flicks (cube tossé en l'air par inertie sans grasp réel). En V2.11 :
1. `action.scale = 0.25` (×0.5 vs V2.6) — cap mécanique sur le delta joint par step
2. `action_rate_l2 = -1e-4` — léger pénalty smoothness (Isaac Lab défaut)
3. `ee_to_cube_distance = -1.0` — pénalise EE loin du cube → décourage le yeet (qui sort la pince)

Ces 3 contraintes ensemble suppriment le motor program chaotique V2.6 → flick rare/impossible → safe d'utiliser lift-only gating comme Isaac Lab.

**Cube_dropped_penalty gardé à -5** :

J'avais initialement envisagé de le réduire à -1 (pratique research). Calcul :
- Reaching à d=24cm = +0.16/step
- Si drop après 30 steps : reaching gain = +4.8
- Penalty -1 → drop net = -1 + 4.8 = **+3.8 (positive)** → drop incentivé !
- Penalty -5 → drop net = -5 + 4.8 = -0.2 (légèrement négatif) ✓

V2.9 calibration tient. Garde -5.

**Cible V2.11 à iter ~100** :
- `reaching_object` ≥ 10-20% (vs V2.10c 4%)
- `ee_to_cube_distance` reward < -20/ép (passe de -67 à <-20)
- noise_std > 0.20 à iter 100 (entropy_coef 0.005 → moins de commit prématuré)
- Premiers `grasping_cube > 0%` à iter 100-200
- Premiers success à iter 500-1000

Si V2.11 plafonne à reaching <10% ET noise_std descend < 0.15 sans grasp, c'est qu'il y a un autre bug (pas smoothness, pas gating). À ce moment-là on regardera : workspace constraints, observation noise, action space upper-bound (vel clipping).

#### V2.11 v1 (scale=0.20) — diagnostic d'échec et fix

**Run V2.11 v1** (state-only fix appliqué partout, scale=0.20) → reaching plateau à ~42% à iter 179 sans aucun grasp. Diagnostic via `play_diagnose_v2.py` sur model_200 (6 épisodes, 2 envs) :

```
ep env  outcome   deepest    grasps  min_jaw_z  min_grip_z  max_qdot  mean_qd    sat
 0   0  TIMEOUT  PRE_REACH      0      6.7cm     13.5cm      6.46     0.525   150/150
 1   1  TIMEOUT  PRE_REACH      0      7.0cm     13.4cm      6.67     0.664   150/150
 3   0  TIMEOUT  PRE_REACH      0      6.3cm     13.9cm     10.13     1.543   150/150
 4   1  TIMEOUT  PRE_REACH      0      5.9cm     13.0cm     10.11     0.796   150/150
 5   0  TIMEOUT  PRE_REACH      0      5.6cm     12.0cm     10.12     0.517   150/150
 6   1  TIMEOUT  PRE_REACH      0      5.1cm     12.6cm     10.11     0.923   150/150
```

**6/6 épisodes : `action_sat_count = 150/150` (saturée à chaque step).** La policy commande à fond pour étendre le bras, mais le joint target plafonne.

**Cause racine** : `JointPositionActionCfg` est en mode **ABSOLU** avec `use_default_offset=True` :
```
joint_target = scale × action + default_joint_pos
```
Avec `action ∈ [-1, 1]` et `scale = 0.20` :
```
joint_target ∈ [default - 0.20, default + 0.20] rad = ±11.5° de home
```

Plage articulaire trop restrictive pour atteindre un cube à 24 cm de distance horizontale. La jaw descend bien au niveau du cube en z (5-7cm vs cube z=5.6cm) parce que le wrist_flex bend down ne demande pas une grande plage articulaire, mais le **gripper body reste à 12-14cm** parce que shoulder_pan/lift/elbow ne peuvent pas suffisamment étendre le bras latéralement.

**Mon erreur conceptuelle** : j'avais pensé que `scale=0.20` cappait la **vélocité par step** à `0.20/(1/30s) = 6 rad/s` (matching Feetech). C'est le calcul valide pour un control DELTA (target = previous_target + scale × action), mais pas pour ABSOLU. En ABSOLU, scale borne la plage atteignable, pas la vitesse. La vitesse réelle est gérée par les PD gains du controller.

**Fix V2.11 v2** : revert `action.scale = 0.20 → 0.5` (V2.9 default, ±28.6° par joint). Protection contre yeet via `ee_to_cube_distance` linéaire (-1.0) qui pénalise EE loin du cube.

#### V2.11 v2 (scale=0.5, smoothness=-1e-4) — yeet revient

User observation au play : robot va à fond, comportement chaotique non-smooth. Configuration V2.11 v2 = V2.9 essentiellement (scale=0.5, smoothness=-1e-4) → mêmes pathologies.

`ee_to_cube_distance` seul ne suffit pas à empêcher le yeet pendant la phase chaotique du début : à iter 0, la policy n'a aucune incentive à être smooth, et avant que `ee_to_cube_distance` ne crée le gradient global vers le cube, la policy explore en saturant les joints.

#### V2.11 v3 — medium smoothness reward + scale=0.5

Smoothness rewards remontés à un niveau qui **kill yeet** sans paralyser :

| terme | V2.10 (paralysie) | V2.11 v2 (yeet) | V2.11 v3 |
|---|---|---|---|
| `action_rate_l2` | -5e-2 | -1e-4 | **-5e-3** |
| `joint_vel_l2` | -1e-2 | -1e-4 | **-5e-3** |
| `joint_acc_l2` | -1e-3 | 0 | **0** (gardé drop) |

Calibration : à yeet dynamics (|q_dot|² total ~450/step), `joint_vel = -5e-3 × 450 = -2.25/step → -340/ép`. Vs gain task d'un yeet réussi (~+100/ép briefly), yeet est net -240/ép → supprimé. À smooth motion (|q_dot|² ~1/step), penalty = -0.75/ép, négligeable.

C'est ÷2 vs V2.10 (qui paralysait à cause de joint_acc, pas de ces deux termes), et ×50 vs Isaac Lab default. Le default Isaac Lab marche pour Franka (impedance control) mais pas pour SO-101 + JointPositionAction qui permet structurellement le chaos.

| terme | V2.10 contribution | V2.10b contribution attendue |
|---|---|---|
| `action_rate_l2` (-5e-2) | -0.71/ép | -0.71/ép |
| `joint_vel_l2` (-1e-2) | -0.71/ép | -0.71/ép |
| `joint_acc_l2` (-3e-5 vs -1e-3) | **-50/ép** | **~-1.5/ép** |

Les trois termes smoothness contribueraient ~ -0.7 à -1.5/épisode chacun,
total budget smoothness ~-3/ép au lieu de -52/ép. Le gradient de la tâche
(+0.6/ép sur reaching à V2.9 baseline) redevient compétitif.

**Cold-start uniquement** — pas de warm-start V2.9. La leçon de V2.10
warm-start : le yeet motor program n'est pas transférable au régime
smooth. Mieux vaut apprendre les deux skills (reach et smoothness) en
parallèle dès le départ.

**Tout le reste de V2.10 est conservé** :
- USD edits (cube 2cm, table #B8ADA9)
- LeIsaac stock pose_range (±7.5cm xy, yaw ±30°)
- `arm_action.scale = 0.25`
- Phase C placeholders (target_color 6D + bowl_xyz 3D)
- PPO config V2.10 inchangée (`init_noise=0.4`, `entropy=0.002`,
  `value_loss_coef=1.0`, schedule adaptive)

**Référence cold-start fonctionnel** : V2 sur Phase A avait convergé à
55% lifting en 695 iters cold-start. V2.10b a un setup proche (PPO
canonique Isaac Lab + reward shaping V2.8.5/V2.9) avec smoothness en plus.
On vise une convergence comparable, en mode lent.

**Cible V2.10b à iter ~150** :
- `reaching_object` ≥ 30%
- `grasping_cube` > 0%
- pas d'effondrement de la noise vers 0 (signe de paralysie)

### Résultats préliminaires V2.8.5 (à iter 80)

- `Episode_Reward/reaching_object` = **0.16** (vs ~0 sur V2.8 au même point) — fix #2 confirmé : ×30 plus de signal
- `grasping_cube` = 0.023 (×2 depuis iter 12) — apprentissage du grasp progresse
- `success_bonus` peaks à **0.15** — quelques succès rares fire (×7.5 amplitude vs V2.8)
- `Loss/value_function` calmé à 0.5 mean (vs 1.5-3.8 sur V2.7) — VF apprend les spikes sparses
- `cube_dropped` rate = 0.4% — pas de give-up exploit
- `Loss/entropy` 8.51 → 8.60 légère hausse (à surveiller iter 200, signe potentiel de "reach abandonment")

### Note sur la métrique `Metrics/position_error`

Cette métrique est auto-loggée par `UniformPoseCommandCfg` et compare le **body `gripper`** au goal (pas le cube). C'est cohérent avec le fait que `body_name="gripper"` dans la commande, mais ça ne reflète **pas** la métrique de tâche réelle (cube → goal). Pour l'instant on lit `position_error` comme un proxy "le bras va dans la bonne direction", et on regarde `Episode_Termination/success` + `Episode_Reward/object_goal_tracking_fine_grained` pour la vraie progression du cube. À ajouter en V2.9 si besoin : un `RewTerm` weight=0 qui logue `‖cube - goal‖` directement.

---

## Reward shaping (état actuel — 7 termes)

Fichier : `sim/eval2/leisaac_lift_env_cfg.py::RewardsCfg`. Les fonctions custom sont dans `sim/eval2/mdp/rewards.py`.

### Lineage du code

Approche **compose-reuse**, pas de duplication :

1. **Isaac Lab canonical Lift** (`isaaclab_tasks...lift.mdp`) → `object_ee_distance` réutilisé tel quel
2. **LeIsaac existing functions** :
   - `terminations.cube_height_above_base` (height check relatif au robot base) → wrappé en reward `cube_lifted_above_base`
   - `observations.object_grasped` (cube proche jaw + gripper fermé) → wrappé en reward `cube_grasped`
3. **Notre code custom** : seulement 1 fonction (`cube_to_goal_distance_above_base`) qui combine la check LeIsaac avec la formule canonical Isaac Lab `object_goal_distance`

### Les 7 termes

| # | Nom | Fonction | Poids | Origine | Rôle |
|---|---|---|---|---|---|
| 1 | `reaching_object` | `lift_mdp.object_ee_distance` (std=0.05) | +1.0 | Isaac Lab canonical | Dense — guide ee → cube |
| 2 | `grasping_cube` ⭐ | `eval2_mdp.cube_grasped` (wrap LeIsaac obs) | **+5.0** | LeIsaac obs réutilisée | **Binary intermediate** — gripper fermé + cube proche |
| 3 | `lifting_object` | `eval2_mdp.cube_lifted_above_base` (wrap LeIsaac termination) | +15.0 | LeIsaac termination réutilisée | Binary — cube > 5cm above base |
| 4 | `object_goal_tracking` | `eval2_mdp.cube_to_goal_distance_above_base` (std=0.3) | +16.0 | LeIsaac height + Isaac Lab tanh | Dense gated — vise goal large |
| 5 | `object_goal_tracking_fine_grained` | même fn (std=0.05) | +5.0 | idem | Précision finale |
| 6 | `action_rate` | `base_mdp.action_rate_l2` | -1e-4 | Isaac Lab core | Smoothness régularisation |
| 7 | `joint_vel` | `base_mdp.joint_vel_l2` | -1e-4 | Isaac Lab core | Vitesses régularisation |

### Pourquoi `grasping_cube` (term 2)

Avant son ajout, le reward landscape avait un **gap entre reaching saturé (≈1.0) et lifting binary (0 → 1.0)**. La policy doit apprendre par essai-erreur que fermer le gripper est utile.

`grasping_cube` (binary, basé sur `object_grasped` de LeIsaac) fire quand :
- distance(jaw_frame, cube) < 0.02m (2cm)
- ET gripper joint angle < 0.26 rad (suffisamment fermé)

Avec poids 5.0, c'est un **stepping stone** : reaching=1 → grasping=5 → lifting=15. La policy a un signal continu pour la séquence d'actions.

### Termination success

`sim/eval2/mdp/terminations.py::cube_reached_goal` :
- 3D distance(cube_world, goal_world) < 5cm
- ⚠️ remplace `cube_height_above_base` (LeIsaac) qui était mal aligné avec notre goal range pos_z=(0.10, 0.20)

---

## PPO configs disponibles

9 variantes dans `sim/eval2/agents/`. Récap des hyperparams clés (uniquement les colonnes qui changent) :

| Param | V2 | V2.5 | V2.6 | V2.7 | V2.8 / V2.8.5 / V2.9 | **V2.10** |
|---|---|---|---|---|---|---|
| `init_noise_std` | 0.6 | 0.6 | 0.6 | 0.6 | 1.0 | **0.4** ⭐ |
| `hidden_dims` | [256,128,64] | [256,128,128] | [256,128,128] | [256,128,128] | [256,128,128] | [256,128,128] |
| `value_loss_coef` | 0.5 | 0.01 | 0.01 | 0.01 | **1.0** | 1.0 |
| `entropy_coef` | 0.002 | 0.002 | 0.005 | 0.005 | 0.005 | **0.002** ⭐ |
| `num_learning_epochs` | 4 | 10 | 10 | 10 | **5** | 5 |
| `num_mini_batches` | 32 | 32 | 32 | 32 | **4** | 4 |
| `learning_rate` | 3e-4 | 3e-4 | 3e-4 | 3e-4 | **1e-3** | 1e-3 |
| `schedule` | fixed | fixed | adaptive | adaptive | adaptive | adaptive |
| `desired_kl` | 0.2 | 0.2 | 0.01 | 0.03 | 0.01 | 0.01 |
| `gamma` | 0.95 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 |
| `max_grad_norm` | 0.5 | 0.5 | 0.5 | 0.5 | 1.0 | 1.0 |

**Reward shaping** (côté env, pas PPO) :

| Reward fix | V2.6 | V2.7 | V2.8 | V2.8.5 | V2.9 | V2.10 | V2.10b | V2.10c | **V2.11** |
|---|---|---|---|---|---|---|---|---|---|
| Lift gate | `lifted` only | `grasped ∧ lifted` ✓ | idem | idem | idem | idem | idem | idem | **`lifted` only** ⭐ revert |
| Tracking gate | `lifted` only | `grasped ∧ lifted` ✓ | idem | idem | idem | idem | idem | idem | **`lifted` only** ⭐ revert |
| `lifting_object` weight | 15 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | **15** ⭐ Isaac Lab default |
| `lift_height_threshold` | 0.05 (bug) | 0.05 (bug) | 0.05 (bug) | **0.08** ✓ | 0.08 | 0.08 | 0.08 | 0.08 | 0.08 |
| `reaching_object.std` | 0.05 (bug) | 0.05 (bug) | 0.05 (bug) | **0.15** ✓ | 0.15 | 0.15 | 0.15 | 0.15 | 0.15 |
| `ee_to_cube_distance` weight (linéaire) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | -1.0 NEW | **-1.0** gardé |
| `success_bonus` weight | n/a | 200 | 200 | **1500** ✓ | 1500 | 1500 | 1500 | 1500 | 1500 |
| `cube_dropped` DoneTerm | n/a | n/a | n/a | **+ ajouté** | idem | idem | idem | idem | idem |
| `cube_dropped_penalty` weight | n/a | n/a | n/a | n/a | **-5** ✓ | -5 | -5 | -5 | -5 |
| `action_rate_l2` weight | -1e-4 | -1e-4 | -1e-4 | -1e-4 | -1e-4 | -5e-2 (×500) | -5e-2 | -5e-2 | **-5e-3** ⭐ medium (v3) |
| `joint_vel_l2` weight | -1e-4 | -1e-4 | -1e-4 | -1e-4 | -1e-4 | -1e-2 (×100) | -1e-2 | -1e-2 | **-5e-3** ⭐ medium (v3) |
| `joint_acc_l2` weight | n/a | n/a | n/a | n/a | n/a | -1e-3 NEW | -3e-5 (÷33) | -3e-5 | **0** ⭐ drop |
| `arm_action.scale` | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.25 | 0.25 | 0.25 | **0.5** ⭐ revert (scale est ABSOLU pas delta — voir ci-dessous) |
| `entropy_coef` (PPO) | 0.005 | 0.005 | 0.005 | 0.005 | 0.005 | **0.002** | 0.002 | 0.002 | **0.005** ⭐ revert |
| Cube spawn pose_range | x±5cm, y∈(-16,-10) | idem | idem | idem | idem | x±7.5cm, y±7.5cm, yaw±30° | idem | idem | idem |
| Cube taille (USD) | 3 cm | 3 cm | 3 cm | 3 cm | 3 cm | 2 cm | 2 cm | 2 cm | 2 cm |
| Table couleur (USD) | bois LeIsaac | bois | bois | bois | bois | #B8ADA9 | #B8ADA9 | #B8ADA9 | #B8ADA9 |

✓ = fix vérifié empiriquement (audit_scene.py / play_diagnose_v2.py)
⭐ = changement V2.10

---

### V2 — `LiftCubePPORunnerCfgV2` (lerobot-sim2real-derived)

```python
init_noise_std       = 0.6           # vs 1.0 default
entropy_coef         = 0.002         # vs 0.006 default
gamma                = 0.95          # vs 0.98 default
lam                  = 0.95
schedule             = "fixed"       # vs "adaptive" default
learning_rate        = 3e-4          # vs 1e-4 default
desired_kl           = 0.2           # vs 0.01 default (irrelevant under fixed)
value_loss_coef      = 0.5           # vs 1.0
use_clipped_value_loss = False       # vs True
num_steps_per_env    = 50            # vs 24
num_mini_batches     = 32            # vs 4
num_learning_epochs  = 4             # vs 5
max_grad_norm        = 0.5           # vs 1.0
experiment_name      = "lift_v2"
```

Validé sur Phase A (state-only) : converge à ~50% rollouts succès à iter 695. Mais Phase B (visual) stagne avec ce config (Run 1).

### IsaacDefaults — `LiftCubePPORunnerCfgIsaacDefaults` (replay original isaac_so_arm101)

```python
init_noise_std       = 1.0
entropy_coef         = 0.006
gamma                = 0.98
lam                  = 0.95
schedule             = "adaptive"
learning_rate        = 1e-4
desired_kl           = 0.01
value_loss_coef      = 1.0
use_clipped_value_loss = True
num_steps_per_env    = 24
num_mini_batches     = 4
num_learning_epochs  = 5
max_grad_norm        = 1.0
experiment_name      = "lift_isaac_defaults"
```

Replay des defaults qui DIVERGEAIENT sur isaac_so_arm101 (entropy explosion). Hypothèse : la divergence venait des bugs côté env (convex_decomposition, curriculum), pas du PPO.

🧪 **Testé sur LeIsaac clean** (run 2026-05-08_14-02-51) → **DIVERGE** :
- `Policy/mean_noise_std` 1.00 → 1.01 (montait au lieu de descendre)
- `Loss/entropy` 8.51 → 8.58 (entropy explosion confirmée)

→ **Hypothèse réfutée**. Les defaults sont structurellement mauvais pour SO-101 manipulation, indépendamment du scaffold. Conservé pour reproductibilité du test.

### V2.5 — `LiftCubePPORunnerCfgV25` (V2 + HW4-inspired) ⭐

```python
init_noise_std       = 0.6           # = V2
entropy_coef         = 0.002         # = V2
gamma                = 0.99          # ⭐ vs V2's 0.95 (HW4)
lam                  = 0.95          # = V2
schedule             = "fixed"       # = V2
learning_rate        = 3e-4          # = V2
desired_kl           = 0.2           # = V2 (irrelevant under fixed)
value_loss_coef      = 0.01          # ⭐⭐ vs V2's 0.5 (HW4, biggest impact)
use_clipped_value_loss = False       # = V2
num_steps_per_env    = 50            # = V2
num_mini_batches     = 32            # = V2
num_learning_epochs  = 10            # ⭐ vs V2's 4 (HW4)
max_grad_norm        = 0.5           # = V2
hidden_dims          = [256, 128, 128]  # ⭐ vs V2's [256, 128, 64]
experiment_name      = "lift_v2_5"
```

Construit sur V2 (toujours valide, anti-entropy-explosion) en intégrant 4 changements de **ETH Robot Learning HW4 PPO config** (validée sur SO-100 EE tracking, convergence à 500 iter, voir section "HW4 inspiration" plus bas).

**Pourquoi `value_loss_coef=0.01`** est le changement le plus important : quand le value head est mal entraîné en début de PPO (ce qui est NORMAL), un value_loss élevé pollue le gradient total. La policy n'arrive pas à s'améliorer parce que l'optimiseur passe son temps à fix the value function. À 0.01 (50× plus bas), la policy s'améliore via le surrogate, le value head suit. HW4 confirme que cette config converge sur SO-100.

🧪 À tester (run à venir).

---

## HW4 inspiration — config PPO validée sur SO-100

Trouvée 2026-05-08 dans `C:\Users\user\Desktop\MA2\Robot-Learning-ETHz\hw4_reinforcement_learning\` (homework universitaire ETH Robot Learning) — un PPO **qui converge** sur SO-100 EE tracking task (mean return 54.91 à iter 500, error 0.017m).

### Config (extrait de `exercises/ex3_ppo_config.py`)

```python
PPO_PARAMETERS = {
    "hidden_sizes":         [256, 128, 128],
    "total_iterations":     500,
    "n_steps":              2048,
    "mini_batch_size":      1024,
    "n_epochs":             10,
    "gamma":                0.99,
    "gae_lambda":           0.95,
    "value_loss_coeff":     0.01,
    "entropy_coeff":        0.005,
    "clip_ratio":           0.2,
    "learning_rate":        3e-4,
    "target_kl":            0.01,
    "max_grad_norm":        0.5,
}
```

### Reward (extrait de `envs/so100_mdp_utils.py::compute_reward`)

Pattern **dense exponentiel + 4 thresholds en escalier** :

```python
def compute_reward(ee_tracking_error, q_vel):
    reward = exp(-10.0 * error)               # base dense (max 1.0)
    if error < 0.10: reward += 0.2             # +0.2 à 10cm
    if error < 0.05: reward += 0.2             # +0.2 à 5cm
    if error < 0.02: reward += 0.5             # +0.5 à 2cm
    if error < 0.005: reward += 0.5            # +0.5 à 5mm
    reward -= 0.01 * max(q_vel ** 2)           # malus vitesse
    return reward
```

Total reward max ~2.4 (très proche). Les thresholds discrets donnent à la policy des **paliers concrets** à viser, plus stable que tanh lisse.

### Episode timing

```python
ctrl_decimation = 50          # 1 ctrl step par 50 sim steps
ctrl_timestep = 0.1s          # = 10 Hz contrôle
max_episode_length_s = 3      # = 30 ctrl steps/épisode
```

8× moins de control steps que notre setup (250). Chaque action a plus d'impact, gradient PPO plus net.

### Action mapping (extrait de `envs/so100_mdp_utils.py::process_action`)

```python
def process_action(action, jnt_range):
    # Mappe action [-1, +1] LINEAIREMENT à la plage complète du joint
    target_qpos = (action + 1.0) * 0.5 * (high - low) + low
```

Différent de notre `JointPositionActionCfg(scale=0.5, use_default_offset=True)`. Leur "full range" marche parce qu'ils ont 10 Hz contrôle (PD a le temps de converger). À 50 Hz, scale=0.5 reste mieux pour nous.

### Observation explicite du `pos_error`

Ils incluent `pos_error_base = target_pos_base - ee_pos_base` dans l'obs. Donne à la policy le **vecteur d'erreur direct**, pas seulement les deux positions séparées. **Idée à reprendre pour Eval 2** : ajouter `goal - cube_pos` comme obs term explicite.

### Ce qu'on a porté dans V2.5 (vs V2)

| Param | V2 | V2.5 (HW4-inspired) |
|---|---|---|
| `value_loss_coef` | 0.5 | **0.01** |
| `num_learning_epochs` | 4 | **10** |
| `gamma` | 0.95 | **0.99** |
| `hidden_dims` | [256, 128, 64] | **[256, 128, 128]** |

Pas porté (mais à considérer plus tard) :
- Reward staging (à appliquer en Phase C, plus risqué pour Phase B où tanh marchait déjà partiellement)
- `decimation=50` (trop violent comme changement, risque de perdre la précision grasp)
- Action full-range (pas adapté à notre fréquence de contrôle 50 Hz)

---

## Phase C — Eval 2 par curriculum (~1 semaine)

### Architecture finale (fixée DÈS LE DÉPART du palier 0)

```
[Wrist cam 224×224×3]
   ↓ ResNet-18 gelé (Phase B reuse)
   ↓ features 512D
   ↓ concat
[State Eval 2]:
   joint_pos (6) + joint_vel (6)
   + target_color_one_hot (6)         ← 6 couleurs
   + bowl_xyz (3)
   + last_action (6)
   = 27D state
   ↓ total = 539D obs
   ↓
[MLP 256-128-64]
   ↓
action 6D
```

**KEY** : au palier 0, `target_color_one_hot=[0,0,0,0,1,0]` (rouge) constant et `bowl_xyz=fixed` constant. La policy "voit" ces inputs mais ils ne varient pas → elle apprend à les ignorer. Aux paliers suivants ils commencent à varier → la policy apprend à les utiliser. **MÊME architecture, MÊME checkpoint loadable à travers tous les paliers.**

### Scène — 6 cubes, 4 cachés à chaque rollout

Pour gérer les couleurs random sans spawning dynamique :

```
Au reset :
  1. Sample 2 indices distincts parmi {0..5}  → pair = (a, b)
  2. Sample lequel est target_color           → soit a, soit b
  3. Sample qui est gauche/droite dans le cluster
  4. Pour les 6 cubes physiques :
     - Cube a → placé à cluster_xy_left,  visible
     - Cube b → placé à cluster_xy_right, visible
     - Cubes {0..5}\{a,b} → placés à z=-10 (hors scène, invisibles)
```

### Les 5-6 paliers (étalement des couleurs)

| Palier | Cubes visibles | target_color | Distractor | Bowl | Iter cumulé |
|---|---|---|---|---|---|
| 0 | 1 (rouge) | rouge fixe | aucun | fixe | 1000 |
| 1 | 1 (rouge) | rouge fixe | aucun | randomisé | 2000 |
| 2 | 2 (rouge + bleu) | rouge fixe | bleu fixe | randomisé | 3500 |
| **3** | 2 (rouge + 1 random parmi 5) | rouge fixe | random parmi {bleu, vert, violet, jaune, orange} | randomisé | 5000 |
| **4** | 2 (paire random parmi 6) | random parmi 6 | l'autre de la paire | randomisé | 6500 |
| (4.5) | idem | idem | idem | + Domain Randomization | 7500 |

L'étalement (palier 3 = distractor random, palier 4 = target_color random) sépare deux compétences :
1. "Ignorer un distractor de n'importe quelle couleur" (palier 3)
2. "Lire target_color one-hot pour savoir lequel viser" (palier 4)

Si on saute directement de "1 distractor fixe" à "tout aléatoire", la policy doit apprendre les deux en même temps → risque de plafonnement.

### Reward Eval 2 (10 termes)

```
Phase B (7 termes)              Phase C Eval 2 (10 termes)
─────────────────────────       ──────────────────────────────
reaching_object       (1.0)     reaching_target          (1.0)   ← target dispatch
grasping_cube         (5.0)     grasping_target          (5.0)   ← target dispatch
lifting_object       (15.0)     lifting_target          (15.0)   ← target dispatch
goal_tracking        (16.0)     target_to_bowl_coarse   (16.0)
goal_tracking_fine    (5.0)     target_to_bowl_fine      (5.0)
                                success_bonus           (200)    ← NEW (rare event)
                                distractor_disturbed     (-5)    ← NEW (Eval 2 specific)
                                wrong_cube_lifted       (-50)    ← NEW (Eval 2 specific)
action_rate         (-1e-4)     action_rate           (-1e-4)
joint_vel           (-1e-4)     joint_vel             (-1e-4)
                                action_l2_norm        (-1e-2)    ← NEW (anti std-explosion fallback)
```

### Goal-conditioning : dispatch sur target_color

```python
def _target_cube_pos(env):
    target_idx = env.target_color_idx  # int dans {0..5}
    cube_name = INDEX_TO_NAME[target_idx]  # "cube_red", "cube_blue", etc.
    return env.scene[cube_name].data.root_pos_w

def _distractor_cube_pos(env):
    distractor_idx = env.distractor_color_idx
    cube_name = INDEX_TO_NAME[distractor_idx]
    return env.scene[cube_name].data.root_pos_w
```

Toutes les rewards `_target` utilisent `_target_cube_pos`, toutes les `_distractor` utilisent `_distractor_cube_pos`.

### Code à écrire (estimation)

| Fichier | Contenu | Lignes |
|---|---|---|
| `sim/eval2/colors.py` (NOUVEAU) | constante `COLOR_TO_INDEX`, `INDEX_TO_RGB` | ~30 |
| `sim/eval2/pick_in_clutter_env_cfg.py` | scène avec 6 cubes (4 hidden) + bowl + cam | ~200 |
| `sim/eval2/mdp/observations.py` (étendre) | + `target_color_one_hot`, `bowl_position_world` | +40 |
| `sim/eval2/mdp/rewards.py` (étendre) | + 6 nouveaux termes Eval 2 (target dispatch, distractor, etc.) | +150 |
| `sim/eval2/mdp/events.py` (NOUVEAU) | `reset_visible_pair`, `reset_target_color`, `reset_bowl_position` | ~120 |
| `sim/eval2/mdp/terminations.py` (étendre) | `success_target_in_bowl` | +40 |
| `sim/eval2/eval2_paliers_env_cfg.py` (NOUVEAU) | 5-6 cfg classes Palier{0..4}Cfg | ~150 |
| `sim/eval2/__init__.py` (étendre) | register `Eval2-Palier{0..4}-v0` + Play | +50 |

**Total : ~830 lignes de nouveau code.** ~2.5-3 jours de codage propre.

---

## Phase D — Deploy SO-101 réel (~1-2 jours au labo)

### D.1 — Camera alignment réel
- Brancher cam wrist sur le vrai SO-101
- Script qui montre **sim cam ↔ real cam superposées en live**
- Ajuster offset/focal jusqu'à match visuel
- Sauvegarder la calibration

### D.2 — Bridge sim → real

Nouveau `deploy/deploy_eval2.py` :
- Charge checkpoint Phase C palier 4
- Tourne à 30 Hz :
  - lit cam + joints via lerobot
  - construit obs au format identique à la sim
  - inférence policy → action 6D
  - envoie aux servos Feetech via lerobot

CLI :
```bash
python deploy/deploy_eval2.py \
  --target_color blue \
  --distractor_color orange \
  --bowl_x 0.20 --bowl_y -0.15 --bowl_z 0.02 \
  --policy <hf-checkpoint-id>
```

### D.3 — 5 rollouts d'éval avec les TAs

Configs annoncées par les TAs (target_color + bowl_xyz + paire de couleurs présente). Score = succès × 10 pts. Cible : ≥ 3/5 (30 pts), idéalement 5/5 (50 pts).

### D.4 — Plan B si ça foire au réel
**Sim-to-real fine-tuning** :
- 5-10 démos teleop physiques sur le vrai robot
- Fine-tune le checkpoint sur ces démos (BC court)
- Re-deploy

C'est l'option **HIL-light** — pas du HIL pur mais on injecte un peu de réel pour combler le gap si pure sim ne transfère pas.

---

## Questions ouvertes

### Q1 — V2 vs IsaacDefaults : laquelle gagne sur Phase B ? [RÉPONDU]

**Aucune des deux**. IsaacDefaults a divergé (entropy explosion confirmée sur env clean). V2 a plafonné sur le **flick exploit** (`lifting_object` non gated sur grasp). Solution finale = stack V2.7→V2.8.5 (gating causal grasp ∧ lift, Isaac Lab canonical PPO, fixes empiriques sur thresholds).

### Q2 — Brev H100 ou local 5070 pour Phase C ?

Phase C avec image obs = ~5-10× plus lent que state-only. Estimation :
- Phase A wallclock = ~9h sur 5070 (à iter 695)
- Phase B wallclock ≈ 16-20h sur 5070 (256 envs)
- Phase C palier 0–4 cumulé = ~80-100h sur 5070

Sur Brev H100 (~2× plus rapide) : ~40-50h. Cost : ~$80-100 (sur les $200 dispo).

**Recommandation** : Phase A et B en local, Phase C sur Brev quand on est sûr du pipeline. Économise du temps. Important : **stop l'instance dès que le training est fini**.

### Q3 — Eval 1 en parallèle ?

Eval 1 = 50 pts, BC autorisée, scaffolds connus (sanity check pipeline). Un teammate pourrait la faire pendant qu'on fait Eval 2. Score combiné meilleur (Eval 1 + Eval 2 = 100 pts).

**Recommandation** : oui si un teammate est dispo, sinon on s'en occupera APRÈS Eval 2.

### Q4 — Quand tester sur le vrai robot ?

Le user n'a pas le SO-101 chez lui. Il faut une session au labo. Idéalement après Phase C palier 4 validé en sim. Coordination équipe nécessaire.

**Recommandation** : prévoir 2 sessions labo. (1) **Camera alignment** dès que Phase B est validé (le checkpoint n'est pas encore final mais on cale la calibration cam). (2) **5 rollouts d'éval** après Phase C complète.

### Q5 — Smoothness pour deploy réel ?

Phase A montre que la policy fait des mouvements brusques (`action_rate` weight=-1e-4 trop faible). Pour deploy réel, faut adoucir :
- Bumper `action_rate` weight à -1e-3 ou -1e-2
- Ou ajouter `joint_acceleration_l2` reward
- À tester en Phase B/C ou en Phase D

---

## Liens & ressources

### Code (ce repo)
- [`sim/eval2/__init__.py`](../sim/eval2/__init__.py) — gym tasks registry (V2 → V2.8.5)
- [`sim/eval2/leisaac_lift_env_cfg.py`](../sim/eval2/leisaac_lift_env_cfg.py) — Phase A/B env (state-only + visual + RewardsCfgV27 + RewardsCfgV285)
- [`sim/eval2/mdp/rewards.py`](../sim/eval2/mdp/rewards.py) — custom rewards (compose LeIsaac + Isaac Lab + custom gating + V2.7 grasp gates + V2.7 success bonus)
- [`sim/eval2/mdp/observations.py`](../sim/eval2/mdp/observations.py) — wrist_image_features (ResNet pre-encoded)
- [`sim/eval2/mdp/terminations.py`](../sim/eval2/mdp/terminations.py) — `cube_reached_goal` + `cube_dropped` (V2.8.5)
- [`sim/eval2/policy/visual_encoder.py`](../sim/eval2/policy/visual_encoder.py) — frozen ResNet-18 ImageNet
- [`sim/eval2/agents/rsl_rl_ppo_cfg_v2.py`](../sim/eval2/agents/rsl_rl_ppo_cfg_v2.py) — V2 PPO config
- [`sim/eval2/agents/rsl_rl_ppo_cfg_isaac_defaults.py`](../sim/eval2/agents/rsl_rl_ppo_cfg_isaac_defaults.py) — IsaacDefaults (DIVERGE)
- [`sim/eval2/agents/rsl_rl_ppo_cfg_v2_5.py`](../sim/eval2/agents/rsl_rl_ppo_cfg_v2_5.py) — V2.5 (HW4-inspired headlines)
- [`sim/eval2/agents/rsl_rl_ppo_cfg_v2_6.py`](../sim/eval2/agents/rsl_rl_ppo_cfg_v2_6.py) — V2.6 (V2.5 + 3 ajouts HW4)
- [`sim/eval2/agents/rsl_rl_ppo_cfg_v2_7.py`](../sim/eval2/agents/rsl_rl_ppo_cfg_v2_7.py) — V2.7 (desired_kl=0.03)
- [`sim/eval2/agents/rsl_rl_ppo_cfg_v2_8.py`](../sim/eval2/agents/rsl_rl_ppo_cfg_v2_8.py) — V2.8 (Isaac Lab canonical)
- [`sim/eval2/agents/rsl_rl_ppo_cfg_v2_8_5.py`](../sim/eval2/agents/rsl_rl_ppo_cfg_v2_8_5.py) — V2.8.5 (alias V2.8 avec experiment_name séparé)
- [`sim/eval2/scripts/{view,train,play}.py`](../sim/eval2/scripts/) — wrappers
- [`sim/eval2/scripts/measure_cube_height.py`](../sim/eval2/scripts/measure_cube_height.py) ⭐ — diagnostic cube z / base z au reset
- [`sim/eval2/scripts/audit_scene.py`](../sim/eval2/scripts/audit_scene.py) ⭐ — audit complet : body names, joint ranges, goal distribution, FrameTransformer targets, rewards à reset

### Code (externes)
- LeIsaac code : `C:\Users\user\Desktop\MA2\isaac\leisaac\source\leisaac\`
- LeIsaac assets : `C:\Users\user\Desktop\MA2\isaac\leisaac\assets\`
- isaac_so_arm101 : `C:\Users\user\Desktop\MA2\isaac\isaac_so_arm101\` (utilisé seulement pour ses scripts via runpy)

### Docs
- [`notes/sanity_results.md`](sanity_results.md) — sanity check précédent (5/5)
- [`notes/full_pipeline_walkthrough.md`](full_pipeline_walkthrough.md) — walkthrough sanity (utile pour Eval 1)
- [`notes/isaac_lab_setup.md`](isaac_lab_setup.md) — install Isaac Lab
- [`notes/project3_rl_final_details.md`](project3_rl_final_details.md) — TA spec
- [LeIsaac doc](https://lightwheelai.github.io/leisaac/)
- [LeIsaac GitHub](https://github.com/LightwheelAI/leisaac)
- [Isaac Lab Lift task source](https://github.com/isaac-sim/IsaacLab/tree/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift)
- [lerobot-sim2real (référence hyperparams)](https://github.com/StoneT2000/lerobot-sim2real)

### HF
- Sanity dataset : `Rsebti/projet3-demos-v1bis`
- Sanity model : `Rsebti/projet3-act-sanity`
- (à venir) Eval 2 model : `Rsebti/projet3-eval2-final`

### Cluster
- Brev credits : $200, non utilisés à ce jour

---

## Annexes

### Annexe A — Bugs ne pas refaire

1. **Ne pas utiliser `entropy_coef=0.006`** sans avoir un env clean (bug observé en run 1)
2. **Ne pas utiliser `schedule="adaptive"` avec `desired_kl=0.01`** → LR throttle, paralysie
3. **Ne pas utiliser `gamma=0.98`** pour manipulation court horizon
4. **Ne pas utiliser `convex_decomposition`** sur le gripper SO-101 → self-blocking
5. **Ne pas mesurer la hauteur du cube en world z** dans les rewards LeIsaac → table à élévation, fire trivial. **Toujours utiliser `cube_height_above_base` (LeIsaac) ou son wrapper `cube_lifted_above_base`.**
6. **Ne pas faire confiance au curriculum d'isaac_so_arm101** → pour la locomotion, pas la manipulation
7. **Ne pas mélanger des reward fns Isaac Lab Lift avec des body_name LeIsaac** → `gripper` vs `gripper_link`, `cube` vs `object`
8. **Ne pas dupliquer la logique de `cube_height_above_base`** → toujours réutiliser la fonction LeIsaac (single source of truth)
9. **Ne pas spawn dynamiquement des cubes pour les 6 couleurs** → spawn les 6 toujours, cache 4 sous le sol au reset
10. **Ne pas charger un checkpoint Phase B au palier 0** → dim d'obs change (540 → 539)
11. **Ne pas oublier `--enable_cameras`** quand on lance un task Visual (sinon Isaac Lab refuse)
12. **Ne pas dépasser num_envs=256 avec wrist cam** sur RTX 5070 12GB → BAR1 saturé, OOM
13. **Ne pas verrouiller le PC** pendant un training overnight → suspend = training stoppé

### Annexe B — Mapping de noms entre les 2 stacks

| isaac_so_arm101 | LeIsaac |
|---|---|
| `gripper_link` | `gripper` |
| `base_link` | `base` |
| `Object` (l'asset cube) | `cube` |
| URDF | USD |

### Annexe C — Hyperparams V2 (figés pour Phase A→C par défaut)

```python
# rsl_rl PPO config V2 (lerobot-sim2real-derived)
num_steps_per_env = 50
num_envs = 4096                       # 256 si Visual env (cam OOM)
max_iterations = 1500                 # par phase

policy:
  init_noise_std = 0.6
  hidden_dims = [256, 128, 64]
  activation = "elu"

algorithm:
  value_loss_coef = 0.5
  use_clipped_value_loss = False
  clip_param = 0.2
  entropy_coef = 0.002
  num_learning_epochs = 4
  num_mini_batches = 32
  learning_rate = 3e-4
  schedule = "fixed"
  gamma = 0.95
  lam = 0.95
  desired_kl = 0.2                    # irrelevant under fixed schedule
  max_grad_norm = 0.5
```

### Annexe D — Hyperparams IsaacDefaults (TESTÉ, DIVERGE)

```python
# rsl_rl PPO config IsaacDefaults (replay original isaac_so_arm101)
num_steps_per_env = 24
max_iterations = 1500

policy:
  init_noise_std = 1.0
  hidden_dims = [256, 128, 64]
  activation = "elu"

algorithm:
  value_loss_coef = 1.0
  use_clipped_value_loss = True
  clip_param = 0.2
  entropy_coef = 0.006
  num_learning_epochs = 5
  num_mini_batches = 4
  learning_rate = 1e-4
  schedule = "adaptive"
  gamma = 0.98
  lam = 0.95
  desired_kl = 0.01
  max_grad_norm = 1.0
```

Run 2026-05-08_14-02-51 sur LeIsaac Visual + grasping_cube reward → entropy explosion (mean_noise_std monte 1.00→1.01, Loss/entropy 8.51→8.58 en 33 iter). **Hypothèse "bugs étaient seulement côté env" RÉFUTÉE**.

### Annexe E — Hyperparams V2.5 (HW4-inspired, à tester)

```python
# rsl_rl PPO config V2.5 (V2 + 4 changements ETH HW4)
num_steps_per_env = 50
max_iterations = 1500

policy:
  init_noise_std = 0.6
  hidden_dims = [256, 128, 128]        # ⭐ +last layer
  activation = "elu"

algorithm:
  value_loss_coef = 0.01               # ⭐⭐ -50x (HW4)
  use_clipped_value_loss = False
  clip_param = 0.2
  entropy_coef = 0.002
  num_learning_epochs = 10             # ⭐ +6 (HW4)
  num_mini_batches = 32
  learning_rate = 3e-4
  schedule = "fixed"
  gamma = 0.99                         # ⭐ +0.04 (HW4)
  lam = 0.95
  desired_kl = 0.2                     # irrelevant under fixed
  max_grad_norm = 0.5
```

⭐ = changement HW4-inspired vs V2.

Source : ETH Robot Learning HW4, `exercises/ex3_ppo_config.py`. Leur config a convergé sur SO-100 EE tracking en 500 iter (mean return 54.91, error 0.017m).
