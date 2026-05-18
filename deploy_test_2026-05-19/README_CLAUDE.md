# README pour Claude Code (laptop) — déploiement robot réel 2026-05-19

> **Tu es Claude Code sur le laptop de Rayane, au robot SO-101.** Lis ce
> fichier EN ENTIER avant d'agir. But du jour : déployer **tous** les
> checkpoints de `./checkpoints/` un par un sur le vrai robot (Eval1 et
> Eval2). **RÈGLE CLÉ : chaque checkpoint est testé DEUX fois — une fois
> avec `--mapping correct` (le mapping validé de Rayane) et une fois avec
> `--mapping native` (mapping squint brut) — pour comparer les deux sur le
> vrai robot.** Noter ce qu'on voit à l'œil. Ce dossier est **autonome** :
> tout est ici (ckpts + `infer.py` + docs).

---

## 0. CONTEXTE EN UNE PHRASE

Des policies RL sim-to-real (SAC+C51, wrist-cam) ont été entraînées pour
Eval1 (1 cube → bol) et Eval2 (2 cubes couleur, ranger la couleur cible).
Deux familles : **16×16 px** (1er livrable) et **64×64 px** (curriculum).
Sur le vrai robot avant correction, Eval1 bouclait en *false-grasp* : le
mapping réel↔sim a été audité et corrigé (mapping **correct**). Aujourd'hui
on teste tout pour de vrai — le run réel est le seul vrai juge.

---

## 1. ENVIRONNEMENT (à faire une fois)

⚠️ **`infer.py` utilise l'API lerobot 0.4.3** (`lerobot.robots.utils`,
`lerobot.robots.so_follower`, `lerobot.motors.motors_bus.MotorNormMode`).
L'env `lerobot` 0.5.1 du laptop **ne marchera PAS** (chemins d'import
différents). **Ne modifie PAS** la lerobot éditable / le repo lerobot
(règle `robot-learning-project3/CLAUDE.md`). Crée un env neuf dédié :

```powershell
conda create -n deploy043 python=3.10 -y
conda activate deploy043
pip install torch torchvision numpy opencv-python "lerobot[feetech]==0.4.3"
pip install rerun-sdk        # viewer live (optionnel ; sinon --no-viz)
```

Vérif rapide (sans robot) que les réseaux chargent bien tous les ckpts :
```powershell
python -c "import torch,glob; [print(p) for p in glob.glob('checkpoints/**/*.pt',recursive=True)]"
```
(Le strict-load + forward des 4 combinaisons a déjà été validé offline ici
avant livraison — l'archi est auto-détectée, aucune édition par ckpt.)

---

## 2. COMMENT `infer.py` MARCHE (lis ça avant de lancer)

- **Auto-détection depuis les poids** : kernel conv.0 = 4 → **16px** (2-conv),
  = 8 → **64px** (3-conv). `state_proj` largeur 12 → **Eval1** (pas de
  couleur), 18 → **Eval2** (couleur, `--goal_color` utilisé). Tu ne touches
  jamais le fichier — **même commande, on change juste `--checkpoint`**.
- **`--mapping correct`** (DÉFAUT, recommandé) = mapping réel↔sim **validé**
  (4 ancres ground-truth + replay démo HF, validé visuellement le 18/05) :
  bras `deg2rad + offset constant lift +0.080 / elbow +0.220 rad`
  (recalage type homing — pure deg2rad laissait le TCP ~3-4 cm trop haut),
  gripper servo **[1°,75°] ↔ sim [-18°,120°]** (la pince se ferme vraiment ;
  c'est le fix du false-grasp). Round-trip exact.
- **`--mapping native`** = mapping squint **brut d'origine** : bras pure
  `deg2rad` (aucun offset), gripper servo **[-60.13°,66.73°] ↔ sim
  [-10°,120°]**. Conservé exprès pour **comparer A/B sur le vrai robot**.
- `REST_QPOS` (pose repos, wrist_roll = −π/2) identique aux deux mappings ;
  le bras y retourne en douceur à chaque fin d'épisode et au Ctrl+C.
- Contrôle **10 Hz** (tous les ckpts ont été entraînés à control_freq=10 —
  **ne pas changer**). `--action_scale` = multiplicateur de sécurité sur
  l'action ; **commence à 0.1**.

---

## 3. PRÉ-VOL (checklist matérielle — `CLAUDE.md` du repo)

```
[ ] Robot SO-101 follower branché. Port série :
      dernier connu = COM3 (follower), COM5 (leader).
      Re-vérifier : lerobot-find-port      -> passe --port COMx
[ ] Caméra wrist : index OpenCV. Laptop = webcam interne index 0,
      wrist souvent index 1. Re-vérifier : lerobot-find-cameras opencv
      -> passe --camera-index N
[ ] Calibration .json du bras : réutiliser celle des runs infer.py
      précédents (même robot). Stockée sous
      ~/.cache/huggingface/lerobot/calibration/ (machine-locale ;
      RE-CALIBRER si machine fraîche). Mets le .json à côté de infer.py
      OU passe --calibration-dir <dossier> --calibration-id <nom_sans_ext>.
[ ] Scène = celle de la sim :
      Eval1 -> 1 cube ~2 cm + 1 bol, table gris clair (~#B8ADA9).
      Eval2 -> 2 cubes ADJACENTS de couleurs distinctes + 1 bol ;
               --goal_color = couleur du cube à ranger.
[ ] Arrêt d'urgence à portée de main. Espace de travail dégagé.
[ ] Leader branché pour ramener le follower au repos entre épisodes
      (sinon le couple reste actif sur le follower).
```

---

## 4. LANCER (Windows PowerShell — `^` = continuation de ligne)

**Eval1** (1 cube, pas de couleur) :
```powershell
python infer.py --checkpoint checkpoints/eval1/64px/eval1_64px_S2_DRctrl_BEST.pt ^
  --mapping correct --action_scale 0.1 --port COM3 --camera-index 1
```

**Eval2** (2 cubes — `--goal_color` = couleur cible :
0 red 1 blue 2 green 3 yellow 4 purple 5 orange) :
```powershell
python infer.py --checkpoint checkpoints/eval2/16px/eval2_16px_DELIVERED_B175.pt ^
  --mapping correct --goal_color 0 --action_scale 0.1 --port COM3 --camera-index 1
```

**Comparer le mapping** (même ckpt, juste `--mapping native`) :
```powershell
python infer.py --checkpoint checkpoints/eval1/16px/eval1_16px_DELIVERED_warmS4.pt ^
  --mapping native --action_scale 0.1 --port COM3 --camera-index 1
```

Options utiles : `--no-viz` (pas de fenêtre Rerun) · `--n_episodes 5`
(5 épisodes d'affilée, sinon Entrée entre chaque) · `--episode_steps 150`
(15 s @10Hz) · `--log_dir logs/<nom>` (dump npz par épisode pour debug).
`Entrée` lance un épisode, `Ctrl+C` quitte (retour repos automatique).

---

## 5. PROTOCOLE DE TEST

**RÈGLE : tout checkpoint testé se teste avec LES DEUX mappings** —
`--mapping correct` d'abord, puis **`--mapping native`** sur le même
checkpoint, dos à dos, pour comparer en vrai. (= 2 runs par checkpoint.
46 runs si tu fais les 23 ; sinon priorise les candidats ci-dessous mais
toujours les 2 mappings chacun.)

L'ordre conseillé + la config/le succès-visuel de **chaque** checkpoint
sont dans **`CHECKPOINTS.md`** (lis-le). Résumé candidats prioritaires
(chacun ×2 mappings) :

```
EVAL1 top : eval1_64px_S2_DRctrl_BEST.pt   (66% sim, recommandé)
            eval1_16px_DELIVERED_warmS4.pt (1er livrable)
EVAL2 top : eval2_16px_DELIVERED_B175.pt   (68% sim, déployable)
            eval2_64px_S6_jitter / S5 / S7 (sweet-spot ~S5-S7)
```

**Commence toujours `--action_scale 0.1`**, monte vers 0.15–0.25 seulement
si le mouvement est sain. Une ligne `RESULTS.md` par (checkpoint × mapping).

**Juge VISUELLEMENT** (consigne ferme de Rayane) : succès *genuine* = le
cube est **vraiment dans le bol**, pince **ouverte et retirée**, **pas** de
hover-avec-cube-en-pince, **pas** de cube hors table. Ne te fie à **aucune**
métrique. Note tout dans **`RESULTS.md`** (gabarit fourni).

---

## 6. PIÈGES CONNUS (vécus en session sanity — `CLAUDE.md`)

- **PowerShell + JSON d'arg lerobot** : si tu passes un arg JSON caméra à un
  CLI lerobot, utilise le token `--%` (stop-parsing) et échappe `"` en `\"`.
  (Ici `infer.py` prend des flags simples, pas de JSON — surtout pertinent
  si tu lances `lerobot-find-cameras` / un record lerobot.)
- **Spam wgpu/Vulkan** si affichage live : `$env:RUST_LOG = "error"` avant.
- **lerobot-record** échoue si le cache local existe (`FileExistsError`) →
  `Remove-Item -Recurse -Force` le cache, ou repo_id frais. (Non requis
  pour `infer.py`, mais utile si tu enregistres un dataset d'éval.)
- **Phase d'encodage silencieuse ~10-15 s** entre épisodes lerobot-record :
  ne touche AUCUNE touche (bufferisé → saute l'épisode suivant).
- **conda activate** ne marche pas en PS → `conda init powershell` puis
  rouvrir le shell.
- **Entre épisodes de deploy** : utilise le leader pour ramener le follower
  au home pendant le reset.

---

## 7. À NE PAS FAIRE

- ❌ Ne commit/colle/echo **aucun token HF** (`tokens.txt`). 
- ❌ Ne modifie **pas** la lib `lerobot` (éditable / pip) ni `isaac_so_arm101`.
- ❌ Ne change pas `CONTROL_HZ`, `DELTA_CAP`, l'archi réseau, le mapping
  *correct* dans `infer.py` (validés ; le seul réglage = `--action_scale`
  et `--mapping`).
- ❌ Ne conclus jamais sur une métrique sim. Le **run réel** est le juge.
- ⚠️ Le mapping *correct* est un **compromis régional** : fidèle dans la
  zone approche/grasp/place (là où la policy opère), pas au home replié
  (hors zone d'opération, sans impact déploiement). C'est attendu.

---

## 8. SI ÇA NE MARCHE PAS

- Import lerobot KO → tu n'es pas dans l'env `deploy043` (lerobot 0.4.3).
- `robot.connect()` KO → mauvais `--port` (refais `lerobot-find-port`).
- Image noire / policy folle → mauvais `--camera-index` ; vérifie que le
  cadrage wrist ressemble à un cube+bol vu de la pince.
- Le bras part trop fort → baisse `--action_scale` (0.05), main sur l'arrêt.
- false-grasp (la pince n'attrape jamais) en `--mapping native` = ATTENDU
  (c'est le bug d'origine) → repasse `--mapping correct`.
- Strict-load error → ckpt corrompu ; re-copie depuis ce dossier (les md5
  des 2 livrés sont dans `CHECKPOINTS.md`).

---

## 9. CONTEXTE TECHNIQUE (pour comprendre, pas à refaire)

Récap complet des configs/succès par étape :
`../handoff_curriculum_2026-05-18/CHECKPOINTS_RECAP.md` (et `MAPPING.md`,
`WORKLOG_AND_TASKS.md`). Le mapping correct est dérivé de 4 ancres réelles
+ replay pas-à-pas de la démo téléop HF, validé visuellement par Rayane le
2026-05-18 (TCP z ≈ 0.010 m au grasp). Plafond connu : critère de succès
sim exploitable (hover / cube hors bol comptés True) → les % sim sont
indicatifs, pas la vérité. C'est pour ça qu'on teste en vrai aujourd'hui.
