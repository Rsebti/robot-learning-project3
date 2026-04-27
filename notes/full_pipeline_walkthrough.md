# Pipeline complet — Sanity check + Eval 1

Document de référence ultra-détaillé pour les deux premières phases du projet.
Lecture cible : un débutant qui n'a jamais touché au stack LeRobot/SO-101.

---

## 0. Comprendre la téléopération bilatérale

### Le concept
Tu manipules **2 bras SO-101 identiques** :

- **Leader** : c'est le bras que tu tiens avec ta main. Ses servos sont en
  mode "lecture". Ils mesurent en continu où ta main place chaque articulation,
  à environ 30 Hz. Tu ne tapes rien au clavier ; tu **bouges physiquement** le
  leader comme on bouge une marionnette.
- **Follower** : le bras qui exécute la vraie tâche (avec le gripper qui
  saisit le bloc). Ses servos sont en mode "commande". À chaque instant,
  ils reçoivent les angles lus sur le leader et s'y positionnent.

Tout ce circuit (lecture leader → écriture follower) tourne dans LeRobot,
qui orchestre la communication USB avec les deux bras.

### Pourquoi bilatéral et pas un joystick ?
- Le mapping est **direct** : 1 articulation leader = 1 articulation follower.
  Pas besoin d'apprendre une commande abstraite.
- Tu ressens en quelque sorte la pose finale du robot (puisque le leader
  est dans la même pose).
- Les démos sont **fluides** : ce sont les vrais mouvements humains, pas des
  trajectoires interpolées entre deux clics.

### Les seuls moments où tu touches le clavier
Entre deux épisodes, pour dire à LeRobot :
- "Cette démo est terminée, passe au reset" → Espace ou flèche droite
- "Cette démo est ratée, refais-la" → flèche gauche
- "Stop tout" → Esc

C'est tout.

---

## 1. Setup matériel à la session

### Sur la table
1. Pose le **follower** côté bloc + bowl, solidement scotché à la table avec
   du gaffer ou un serre-joint. Si le bras se déplace de 5 mm entre deux
   démos, le dataset est foutu.
2. Pose le **leader** à côté ou en face. Espace dégagé autour pour bouger ta
   main librement (pas de chaise, pas de mur).
3. Vérifie que la **caméra wrist** est bien fixée sur l'avant-bras du
   follower (vis serrées). Câble pas tendu, pas dans l'axe du gripper.
4. Place le **bloc** au repère X1 (scotch noir, marquage net) et le **bowl**
   au repère X2.

### Branchement, ordre conseillé

L'ordre suivant aide à identifier les ports plus tard :

1. **Alimentation 5V follower** → secteur. Les LEDs des servos s'allument
   en rouge / vert selon le statut. Si rien ne s'allume → alim morte ou
   mauvais voltage.
2. **Alimentation 5V leader** → secteur.
3. **USB follower** → port USB du portable. **Note mentalement** "le
   follower je viens de le brancher".
4. **USB leader** → autre port USB du portable.
5. **USB caméra wrist** → 3ème port USB.

### Astuce : ports physiques
Sur Windows, le numéro de COM est lié au port USB physique (pour la plupart
des cartes mère). Si tu débranches puis rebranches au même endroit, tu
récupères le même `COMx`. Donc utilise toujours les mêmes ports — sinon,
quand tu remplaceras `<FOLLOWER_PORT>` dans tes commandes, ça va décaler.

### Pourquoi 5V ?
Les servos Feetech STS3215 (ceux du SO-101) tournent en 5-7.4V. L'alim
fournie avec le kit est calibrée. Ne pas brancher du 12V "par sécurité" :
tu grilles tous les servos en quelques secondes.

---

## 2. Identifier les ports COM Windows

Les servos USB apparaissent comme des "ports COM" virtuels (héritage des
ports série RS-232). Sur Windows, tu vois `COM3`, `COM4`, etc. dans le
gestionnaire de périphériques.

### Méthode 1 : Device Manager
1. **Windows + X** → **Gestionnaire de périphériques**
2. Ouvre la rubrique **"Ports (COM et LPT)"**
3. Tu verras 2 entrées du genre `USB Serial Device (COM3)`
4. Pour distinguer leader/follower :
   - Débranche le follower (USB)
   - Une entrée disparaît → c'est lui
   - Rebranche → l'entrée revient
5. Note sur papier : `LEADER = COM_?`, `FOLLOWER = COM_?`

### Méthode 2 : outil LeRobot (plus propre)
Dans **Anaconda Prompt** avec env activé :
```bash
lerobot-find-port
```
Le programme te demande :
- Débranche le câble que tu veux identifier
- Appuie Entrée
- Rebranche
- Il te dit "le port qui a disparu/réapparu était `COM4`"

À faire 2 fois (une pour leader, une pour follower).

### Pourquoi pas /dev/ttyACMx comme sur Linux ?
Sur Windows, l'API série utilise des "noms de ports COM" hérités. Sur Linux,
on a des fichiers de périphérique (`/dev/ttyACM0`, `/dev/ttyUSB0`). LeRobot
gère les deux de manière transparente — tu donnes juste le bon nom dans
`--robot.port`.

---

## 3. Activer l'environnement Anaconda

### Quel terminal ?
Sur Windows, tu as plusieurs choix :
- **PowerShell** : terminal Windows par défaut. Conda peut y marcher mais
  parfois capricieux (politique d'exécution, scripts d'init).
- **Anaconda Prompt** : terminal préconfigué pour conda. **À utiliser.**
- **CMD** : ancien terminal, fonctionne aussi.

→ Cherche **"Anaconda Prompt"** dans le menu démarrer et lance-le.

### Activer l'env
```bash
conda activate lerobot
```
Le prompt change pour indiquer l'env actif :
```
(lerobot) C:\Users\user>
```

Si conda dit "command not found" → conda n'est pas dans ton PATH. Ouvre à
la place "Anaconda Prompt (lerobot)" si l'option existe, ou réinstalle
Miniconda en cochant "Add to PATH".

### Vérifier que tout est OK
```bash
hf auth whoami
```
→ doit afficher `Rsebti`. Si pas loggé : `hf auth login`, puis colle ton
token HF (récupéré depuis ton mail).

```bash
python -c "import lerobot; print(lerobot.__version__)"
```
→ doit afficher `0.5.2`.

```bash
cd C:\Users\<toi>\Desktop\MA2\robot-learning-project3
```

Tu es prêt.

---

## 4. Calibration des bras (à faire une fois)

### Pourquoi
Chaque servo a un encodeur qui mesure l'angle. Mais chaque servo a un offset
mécanique : "0 degré" pour l'encodeur n'est pas forcément la pose neutre du
bras. La calibration enregistre des poses de référence (typiquement bras
tendu, bras replié, gripper ouvert, gripper fermé) pour mapper les valeurs
brutes des servos en angles "humains" en degrés.

Sans calibration, le follower ne saura pas suivre correctement le leader :
les angles bruts ne correspondront pas.

### Calibrer le follower
```bash
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=<FOLLOWER_PORT> \
  --robot.id=so101_follower
```

Suis les instructions à l'écran :
1. Mets le bras dans la pose "neutre" (bras tendu vers l'avant, gripper
   ouvert) → Entrée
2. Ferme le gripper jusqu'à butée → Entrée
3. Ouvre le gripper jusqu'à butée → Entrée
4. (selon la version) Plie le coude à 90° → Entrée

Le programme enregistre les valeurs et les sauve dans :
```
C:\Users\<toi>\.cache\huggingface\lerobot\calibration\so101_follower.json
```

### Calibrer le leader
```bash
lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port=<LEADER_PORT> \
  --teleop.id=so101_leader
```

Mêmes étapes. Sauve dans `so101_leader.json`.

### Bon à savoir
- Tant que tu ne changes pas le câblage et que tu ne démontes pas un servo,
  la calibration reste valable. Pas besoin de refaire à chaque session.
- Si tu changes de PC (portable au lieu du fixe), il faut recalibrer car
  les fichiers de calibration sont locaux à la machine.
- Si tu vois des mouvements bizarres après calibration → recalibre.

---

## 5. Test de téléopération sans enregistrement

Avant d'enregistrer 20 démos, valide que tout marche :

```bash
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=<FOLLOWER_PORT> \
  --robot.id=so101_follower \
  --teleop.type=so101_leader \
  --teleop.port=<LEADER_PORT> \
  --teleop.id=so101_leader \
  --display_data=true
```

### Ce qui se passe
1. Une fenêtre s'ouvre avec le flux de la caméra wrist
2. Le terminal affiche `Teleoperation running at 30 Hz...`
3. **Bouge le leader avec ta main** : le follower doit suivre en temps réel,
   avec un délai imperceptible (~30-50 ms)

### Tests à faire
- Ouvre/ferme le gripper du leader → le gripper du follower suit
- Plie chaque articulation indépendamment
- Fais le mouvement complet "approche bloc, ferme, lève, va au bowl, ouvre"
  sans le bloc, juste pour t'entraîner

### Quitter
**Ctrl+C** dans le terminal. La fenêtre se ferme.

### Si ça ne marche pas

| Symptôme | Cause probable | Action |
|---|---|---|
| Aucun mouvement | Mauvais port COM | Re-vérifier avec `lerobot-find-port` |
| Mouvement saccadé | USB hub/cable de mauvaise qualité | Changer de port USB |
| Erreur "servo not responding" | Câble interne mal branché ou servo HS | Inspecter le câblage des servos |
| Caméra noire | Mauvais index OpenCV | Tester `index_or_path: 1` au lieu de `0` |
| Lag énorme (>500 ms) | CPU saturé | Fermer Chrome/autres apps |

---

## 6. Enregistrement des 20 démos

### Commande complète (Windows / Anaconda Prompt)

⚠️ Attention au quoting Windows : les guillemets internes doivent être
échappés avec `\"` au lieu de l'apostrophe simple bash.

```bash
lerobot-record ^
  --robot.type=so101_follower ^
  --robot.port=<FOLLOWER_PORT> ^
  --robot.id=so101_follower ^
  --robot.cameras="{\"wrist\": {\"type\": \"opencv\", \"index_or_path\": 0, \"width\": 640, \"height\": 480, \"fps\": 30}}" ^
  --teleop.type=so101_leader ^
  --teleop.port=<LEADER_PORT> ^
  --teleop.id=so101_leader ^
  --display_data=true ^
  --dataset.repo_id=Rsebti/projet3-demos-v1 ^
  --dataset.num_episodes=20 ^
  --dataset.fps=30 ^
  --dataset.episode_time_s=15 ^
  --dataset.reset_time_s=10 ^
  --dataset.single_task="Pick block and place in bowl" ^
  --dataset.private=true ^
  --dataset.push_to_hub=true
```

(Le `^` est le caractère de continuation de ligne sous Windows. En bash,
c'est `\`.)

### Décortiquer chaque flag

- `--robot.type=so101_follower` : on dit à LeRobot quel driver charger pour
  le bras qui exécute (gestion des servos Feetech).
- `--robot.port` / `--robot.id` : port COM physique et identifiant logique
  (juste un nom, peu importe la valeur tant qu'elle est cohérente avec la
  calibration).
- `--robot.cameras` : description des caméras au format JSON. Une seule
  ici, nommée `wrist`, qui sera l'observation visuelle du modèle. `width`,
  `height`, `fps` : caractéristiques de capture. `index_or_path: 0` =
  première webcam détectée par OpenCV.
- `--teleop.type=so101_leader` : driver pour le bras de téléop.
- `--display_data=true` : ouvre la fenêtre avec flux caméra + indicateurs.
  À mettre `false` si ton portable rame.
- `--dataset.repo_id=Rsebti/projet3-demos-v1` : nom du dataset HF à créer.
  Format `<user>/<dataset>`. Si tu mets un nom déjà existant, LeRobot
  reprend le dataset là où il s'était arrêté (utile en cas d'interruption).
- `--dataset.num_episodes=20` : nombre total d'épisodes à enregistrer.
- `--dataset.fps=30` : fréquence d'échantillonnage. Doit matcher la `fps`
  de la caméra.
- `--dataset.episode_time_s=15` : durée maximale d'un épisode. Tu peux finir
  plus tôt avec flèche droite.
- `--dataset.reset_time_s=10` : durée du reset entre deux épisodes (pour
  remettre le bloc au repère et le robot en pose home).
- `--dataset.single_task="..."` : description textuelle de la tâche.
  Utilisée par certains modèles conditionnés sur le langage. Pour ACT, ça
  sert juste de label.
- `--dataset.private=true` : le dataset HF sera privé.
- `--dataset.push_to_hub=true` : à la fin, upload automatique sur HF.

### Déroulement à l'écran

**Phase 1 : init** (~5-10s)
```
INFO:lerobot:Loading robot so101_follower on COM4...
INFO:lerobot:Loading teleop so101_leader on COM3...
INFO:lerobot:Loading camera wrist (640x480 @ 30fps)...
INFO:lerobot:Calibration loaded.
INFO:lerobot:Recording dataset Rsebti/projet3-demos-v1
INFO:lerobot:Episode 0/20 starting in 3 seconds...
```

**Phase 2 : enregistrement épisode**
La fenêtre affiche :
- Flux caméra wrist en haut
- Compteur "Episode 1/20 — 0.0s / 15.0s"
- Indicateur état (recording/resetting)
- Hotkeys rappels en bas

Toi pendant ces 15s :
- Bouge le leader avec ta main
- Fais le pick-and-place lentement (3-5s pour aller au bloc, 1s pour saisir,
  3-5s pour aller au bowl, 1s pour lâcher)
- Si la démo est ratée (collision, raté du bloc, raté du bowl) → flèche
  gauche → l'épisode est jeté et tu recommences au même numéro
- Si tu finis tôt et que tout est OK → flèche droite

**Phase 3 : reset**
```
INFO:lerobot:Episode 0/20 saved.
INFO:lerobot:Resetting (10s)...
```

Pendant ces 10s :
- Replace le bloc au repère X1 avec ta main (sans bouger le robot)
- Bouge le leader pour ramener le follower en pose home (le scotch peut
  marquer une pose home aussi)
- Quand prêt → la phase reset finit toute seule, ou flèche droite pour
  passer plus vite

**Boucle** : 20 fois.

**Phase 4 : finalisation**
```
INFO:lerobot:All 20 episodes recorded.
INFO:lerobot:Encoding videos to mp4...
INFO:lerobot:Pushing dataset to HuggingFace Hub...
INFO:lerobot:Done. Dataset available at https://huggingface.co/datasets/Rsebti/projet3-demos-v1
```

L'encoding vidéo prend ~2-5 min, le push ~3-10 min selon ta connexion.

### Discipline pendant les démos

- **Mouvement identique** : même trajectoire à chaque fois. Vise une
  trajectoire en arc, pas un mouvement chaotique. Si tu varies trop, ACT
  apprendra une "moyenne" de tes mouvements qui peut être incohérente.
- **Vitesse identique** : compte mentalement "1-2-3" pour chaque phase.
- **Grasp identique** : même point de saisie sur le bloc, mêmes doigts du
  gripper, même hauteur d'approche.
- **Rejeter sans pitié** : si une démo est moche, flèche gauche. Mieux vaut
  20 démos quasi-parfaites que 25 démos dont 5 sont du bruit.
- **Rien dans le champ caméra** : pas tes mains sur le bloc pendant l'épisode
  (seulement pendant le reset).

### Pendant les 30-45 minutes d'enregistrement

C'est répétitif, le but est précisément que ça le soit. Si tu fatigues vers
l'épisode 10, fais une pause de 2 min — la qualité chute sinon.

---

## 7. Où vont les démos ?

### Sur disque local (pendant l'enregistrement)
```
C:\Users\<toi>\.cache\huggingface\lerobot\Rsebti\projet3-demos-v1\
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ... (20 fichiers)
├── videos/
│   └── chunk-000/
│       └── observation.images.wrist/
│           ├── episode_000000.mp4
│           └── ... (20 fichiers)
└── meta/
    ├── info.json          ← schéma général du dataset
    ├── episodes.jsonl     ← une ligne par épisode (durée, task, etc.)
    ├── stats.json         ← stats normalisation (mean, std des actions)
    └── tasks.jsonl        ← liste des tâches
```

### Que contient un parquet (data) ?
Un fichier parquet par épisode, avec une ligne par frame (donc ~450 lignes
pour 15s à 30 fps). Colonnes :
- `observation.state` : vecteur 7D des angles servos du follower
- `action` : vecteur 7D de la commande envoyée aux servos (= angles du leader)
- `episode_index` : 0 à 19
- `frame_index` : 0 à ~450
- `timestamp` : temps en secondes depuis le début de l'épisode
- `task_index` : indice dans tasks.jsonl

### Que contient un mp4 (videos) ?
La vidéo de la caméra wrist pour cet épisode, encodée en H.264, ~5 MB par
fichier de 15s à 640x480.

Les vidéos sont **séparées** des parquets pour des raisons de performance
(streaming, lecture sélective).

### Sur HuggingFace Hub
Avec `push_to_hub=true`, tout le dossier ci-dessus est uploadé. Tu vas sur
https://huggingface.co/datasets/Rsebti/projet3-demos-v1 :
- Le **Dataset Viewer** te permet de naviguer épisode par épisode
- Tu peux **streamer** les vidéos directement dans le navigateur
- Onglet "Files" = tous les fichiers de l'arbre

Le dataset est privé → seul toi (et qui tu invites) peut y accéder.

---

## 8. Replay verification (étape 2 du mail TAs)

### Pourquoi
On rejoue 4 démos sur le **vrai robot** sans modèle, juste pour vérifier
que :
1. Les actions enregistrées sont cohérentes avec ce qu'on a vu en live
2. Le robot, en repassant la séquence d'actions enregistrée, refait
   bien la trajectoire (donc pas de drift de calibration)

C'est un test purement mécanique — aucun apprentissage.

### Setup
- Bloc au repère X1 (exactement comme pendant l'enregistrement)
- Bowl au repère X2
- Robot en pose home

### Commande
```bash
lerobot-replay \
  --robot.type=so101_follower \
  --robot.port=<FOLLOWER_PORT> \
  --robot.id=so101_follower \
  --dataset.repo_id=Rsebti/projet3-demos-v1 \
  --dataset.episode=0
```

### Ce qui se passe
- LeRobot télécharge l'épisode 0 si pas déjà en cache
- Lit la séquence d'actions (= 450 vecteurs 7D)
- Les envoie au follower à 30 Hz
- Le follower exécute → le robot reproduit la démo

### Tests à faire
Refais avec `--dataset.episode=5`, puis `10`, puis `15`. Sur chaque :
- Le robot doit aller au bloc et le saisir
- Le robot doit le poser dans le bowl
- Pas de mouvement saccadé, pas d'erreur servo

Si tout marche → **étapes 1+2 du mail TAs validées**.

### Si ça ne marche pas
- Robot atteint pas le bloc → le bloc n'est pas exactement à X1, ou la
  calibration a changé. Recalibre et redéploie le bloc au scotch.
- Erreur servo "overcurrent" → trajectoire trop rapide pour les servos.
  Étonnant à 30 fps mais possible. Refaire les démos plus lentement.
- Robot finit en singularité → la démo elle-même passait par une singularité.
  Refaire la démo avec une trajectoire plus arquée.

---

## 9. Rentrer au PC fixe — préparer le training

### Pourquoi changer de machine ?
Le portable n'a pas (ou peu de) GPU. Le training d'ACT sur 20 000 steps
demande typiquement 6-10 GB de VRAM et tourne en 30-60 min sur une RTX 5070.
Sur CPU ou petit GPU laptop, ça serait des heures.

### Pas de transfert manuel
Comme `push_to_hub=true` a déjà uploadé tout sur HF, le PC fixe va juste
retélécharger depuis HF. Pas besoin de clé USB pour le dataset.

### Sur le PC fixe
1. **GitHub Desktop → Pull** : récupère les dernières notes/scripts
2. **Anaconda Prompt** :
   ```bash
   conda activate lerobot
   cd C:\Users\user\Desktop\MA2\robot-learning-project3
   hf auth whoami       # Rsebti
   nvidia-smi           # vérifie que la RTX 5070 est visible
   ```
3. **Télécharger le dataset localement** (en cache) :
   ```bash
   python -c "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; d = LeRobotDataset('Rsebti/projet3-demos-v1'); print(d.num_episodes, d.num_frames)"
   ```
   Doit afficher quelque chose comme `20 9000`. Le téléchargement met
   ~500 MB - 1 GB en cache (`~/.cache/huggingface/lerobot/`).

---

## 10. Lancer le training ACT

### La commande
(Dans `train/train_bc.md` aussi.)

```bash
lerobot-train \
  --dataset.repo_id=Rsebti/projet3-demos-v1 \
  --policy.type=act \
  --policy.device=cuda \
  --policy.repo_id=Rsebti/projet3-act-sanity \
  --output_dir=outputs/train/act_sanity \
  --job_name=act_sanity \
  --batch_size=8 \
  --steps=20000 \
  --save_freq=5000 \
  --eval_freq=0 \
  --log_freq=200 \
  --wandb.enable=false
```

### Décortiquer chaque flag
- `--dataset.repo_id` : source des données
- `--policy.type=act` : on choisit ACT (Action Chunking Transformer, Stanford 2023).
  Alternatives : `diffusion`, `vqbet`, `tdmpc`. ACT est le meilleur compromis pour
  overfitter sur peu de démos.
- `--policy.device=cuda` : entraînement sur GPU. Mettre `cpu` en cas de problème.
- `--policy.repo_id` : où push le checkpoint final sur HF (optionnel).
- `--output_dir` : dossier local de sortie (logs + checkpoints).
- `--job_name` : nom symbolique du run.
- `--batch_size=8` : nb d'échantillons par step. À 8, le training utilise ~6-8 GB
  de VRAM. Si OOM → baisser à 4 ou 2.
- `--steps=20000` : nombre total de mises à jour des poids. Pour 20 démos
  identiques, 20k suffit largement.
- `--save_freq=5000` : checkpoint toutes les 5k étapes.
- `--eval_freq=0` : pas d'éval en simulation pendant le training. Pas pertinent
  pour BC sur démos réelles.
- `--log_freq=200` : affichage console toutes les 200 steps.
- `--wandb.enable=false` : pas de Weights & Biases. Si tu veux suivre joliment
  les courbes, mets `true` et `wandb login` au préalable.

### Le modèle ACT en deux mots
- **Encoder visuel** : ResNet-18 qui prend l'image wrist 640x480 et sort un
  embedding 512D
- **Encoder transformer** : prend l'embedding image + l'état du bras (7D) et
  produit une représentation latente
- **Decoder transformer** : à partir de cette latente, produit en parallèle
  les **prochaines 100 actions** (chunking horizon = 100 frames ≈ 3.3s à 30 Hz)
- **Loss** : L1 entre actions prédites et actions des démos, plus une loss
  de régularisation latente (KL contre prior gaussien)

Au déploiement, on prend les K premières actions du chunk (typiquement
K=50 = 1.6s), on les exécute, puis on re-prédit. Ça donne de la stabilité.

### Logs pendant le training
```
INFO Step 0/20000 | loss: 1.234 | l1: 0.987 | kl: 0.123 | lr: 1e-4 | grad_norm: 5.67 | dt: 0.45s
INFO Step 200/20000 | loss: 0.847 | ...
INFO Step 400/20000 | loss: 0.612 | ...
...
INFO Step 19800/20000 | loss: 0.038 | ...
INFO Step 20000/20000 | loss: 0.035 | l1: 0.029 | kl: 0.008 | grad_norm: 0.42 | dt: 0.18s
INFO Saving final checkpoint to outputs/train/act_sanity/checkpoints/020000/pretrained_model
INFO Training complete.
```

### Indicateurs à surveiller

| Métrique | Bon signe | Mauvais signe |
|---|---|---|
| `loss` | descend monotone de ~1.0 à ~0.05 | reste haute, oscille, ou explose en NaN |
| `l1` | descend en parallèle | NaN |
| `kl` | reste faible (< 0.5) | grimpe à 100+ |
| `grad_norm` | < 5.0 | > 50 = instable |
| `dt` (temps par step) | stable autour de 0.15-0.30s | grimpe → swap mémoire |
| Mémoire GPU (`nvidia-smi`) | 6-10 GB/12 | OOM |

Sur RTX 5070 : 20 000 steps ≈ 35-50 minutes.

### Erreurs possibles

#### Erreur CUDA Blackwell
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```
Cause : la RTX 5070 utilise l'architecture Blackwell (sm_120) qui n'est pas
encore supportée par les wheels PyTorch officielles pré-livrées dans LeRobot
0.5.2.

Solutions par ordre de préférence :
1. **PyTorch nightly cu124+** :
   ```bash
   pip uninstall torch torchvision torchaudio -y
   pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124
   ```
2. **Fallback CPU** : `--policy.device=cpu`. Très lent (10-20x), donc baisser
   `--steps=5000` pour rester raisonnable. Sanity check fonctionne quand même.
3. **Brev (H100)** : si on a vraiment du temps perdu. Coupon $200 pas encore utilisé.

#### OutOfMemoryError
```
torch.cuda.OutOfMemoryError: CUDA out of memory.
```
Réduire `--batch_size=4` puis `--batch_size=2` si nécessaire.

#### Loss à NaN dès le step 0
Souvent un bug dataset. Vérifier qu'il n'y a pas d'épisode vide :
```bash
python -c "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; d = LeRobotDataset('Rsebti/projet3-demos-v1'); [print(i, d.episode_data_index['from'][i], d.episode_data_index['to'][i]) for i in range(d.num_episodes)]"
```

### Output final
```
outputs/train/act_sanity/
├── checkpoints/
│   ├── 005000/pretrained_model/
│   │   ├── config.json
│   │   └── model.safetensors
│   ├── 010000/pretrained_model/...
│   ├── 015000/pretrained_model/...
│   ├── 020000/pretrained_model/...
│   └── last → 020000/
├── train_config.json
└── (wandb/...)
```

`pretrained_model/` contient :
- `config.json` : la config ACT (dim hidden, nb couches, chunk size)
- `model.safetensors` : les poids appris (~200 MB)

C'est le `--policy.path` que tu utiliseras au déploiement.

---

## 11. (Optionnel) Push du modèle sur HF

Pour ne pas avoir à transférer le checkpoint via clé USB :

```bash
hf upload Rsebti/projet3-act-sanity outputs/train/act_sanity/checkpoints/last/pretrained_model
```

Avantage : depuis le portable, tu fais `--policy.path=Rsebti/projet3-act-sanity`
et LeRobot téléchargera tout automatiquement.

---

## 12. Retour au robot pour le deploy

### Setup scène — IDENTIQUE à l'enregistrement
- Bloc à X1 (exact)
- Bowl à X2 (exact)
- Caméra wrist non touchée
- Lumière identique
- Robot en pose home

Si tu déplaces le bloc de 2 cm → le modèle peut échouer. Le sanity check
teste l'overfit, pas la généralisation.

### Récupérer le checkpoint
- Si push HF (étape 11) → rien à faire, LeRobot téléchargera
- Sinon → copier `outputs/train/act_sanity/checkpoints/last/pretrained_model/`
  via clé USB du PC fixe au portable

### Lancer le deploy
(Dans `deploy/deploy_policy.md` aussi.)

```bash
lerobot-record ^
  --robot.type=so101_follower ^
  --robot.port=<FOLLOWER_PORT> ^
  --robot.id=so101_follower ^
  --robot.cameras="{\"wrist\": {\"type\": \"opencv\", \"index_or_path\": 0, \"width\": 640, \"height\": 480, \"fps\": 30}}" ^
  --display_data=true ^
  --policy.path=outputs/train/act_sanity/checkpoints/last/pretrained_model ^
  --policy.device=cuda ^
  --dataset.repo_id=Rsebti/projet3-eval-sanity ^
  --dataset.num_episodes=5 ^
  --dataset.fps=30 ^
  --dataset.episode_time_s=15 ^
  --dataset.reset_time_s=10 ^
  --dataset.single_task="Pick block and place in bowl" ^
  --dataset.private=true ^
  --dataset.push_to_hub=false
```

⚠️ **Plus de `--teleop.*`** : c'est la policy qui pilote.

### Ce qui se passe
1. La fenêtre caméra s'ouvre
2. Console : `Episode 1/5 — running policy at 30 Hz...`
3. **Le follower bouge tout seul** :
   - À chaque frame (30 fois par seconde) :
     - LeRobot lit l'image caméra wrist (640x480)
     - LeRobot lit l'état des servos (vecteur 7D)
     - Passe les deux dans ACT (inférence ~10-20 ms sur 5070)
     - ACT prédit les 100 prochaines actions
     - LeRobot prend la 1ère et l'envoie au follower
     - On loop
4. Au bout de 15s → fin épisode → reset 10s → épisode suivant
5. 5 épisodes au total

### Critère de succès du sanity check
Sur les **5 épisodes** :
- 3+ succès (bloc dans bowl) → pipeline validé, on passe à Eval 1
- 1-2 succès → marginal, peut-être plus de steps ou meilleures démos
- 0 succès → debug

### Ce qui peut foirer au deploy
- **Robot drift dès le départ** : caméra index différent, ou block/bowl pas
  exactement aux repères, ou calibration différente entre training et deploy
- **Robot freeze** : la policy prédit toujours la même action. Sous-entraînée,
  augmenter `--steps`. Ou bug du dataset (toutes les démos identiques au bit
  près).
- **Servo overcurrent** : la policy prédit des actions trop rapides. Souvent
  parce que les démos étaient trop rapides → les refaire plus lentement.
- **Robot fait n'importe quoi** : démos pas assez identiques, ACT a appris
  une moyenne incohérente. Refaire des démos plus disciplinées.

**Garde la main près du bouton power** au cas où.

---

## 13. Slack update aux TAs (jeudi)

Format suggéré pour le sanity check :
```
Update Group X — Sanity check
- 20 demos recorded (Rsebti/projet3-demos-v1)
- Replay verification on 4 episodes: OK
- ACT trained for 20k steps, final loss 0.04
- Deployed on real SO-101: 4/5 success on identical scene
→ Pipeline validated, moving to Eval 1.
```

Si ça a foiré, dire honnêtement où ça bloque — les TAs aident plus si tu
décris les symptômes précisément.

---

## 14. Ce qui clôt le sanity check

À la fin de cette phase, tu dois avoir :
- [x] Dataset HF `Rsebti/projet3-demos-v1` avec 20 épisodes
- [x] Replay 4/4 OK sur le robot réel
- [x] Checkpoint `Rsebti/projet3-act-sanity` (ou local)
- [x] 3+/5 succès au déploiement
- [x] Update Slack envoyé

Tu connais maintenant le **pipeline end-to-end**. Eval 1 réutilise tout ça,
juste avec une tâche un poil plus dure.

---

## 15. Eval 1 — pick-and-place avec position randomisée

### La tâche
Même pick-and-place qu'au sanity check, **mais le bloc peut être à
n'importe quelle position dans une zone définie** (typiquement un carré
de 20×20 cm sur la table). À l'évaluation, les TAs choisiront des positions
qu'ils veulent — le modèle doit généraliser.

Le bowl reste à une position fixe (sauf si l'énoncé dit le contraire).

### Pourquoi BC peut suffire
ACT est un modèle **conditionné sur l'image**. À chaque frame, il regarde
où est le bloc et ajuste sa trajectoire. Si tu lui donnes assez de démos
couvrant la zone de positions, il apprend à interpoler.

C'est différent du sanity check où tu pouvais "overfitter" la trajectoire
unique. Ici, le modèle doit vraiment apprendre la **politique** "regarde
le bloc, va vers lui".

### Les changements concrets vs sanity check

#### Plus de démos
- Sanity : 20 démos
- Eval 1 : **50 à 100 démos** typiquement, peut-être plus

Pourquoi : pour couvrir la variabilité spatiale, il faut un maillage
suffisant. Si tu fais 10 démos avec le bloc dans le coin haut-gauche et
zéro ailleurs, le modèle ne saura pas faire le coin bas-droit.

#### Sampling des positions
Définis une zone (carré 20×20 cm marqué au scotch). Avant chaque démo, place
le bloc à une **position aléatoire** dans la zone. Pour avoir un sampling
uniforme :
- Méthode papier : dessine une grille 5×5 sur la zone, fais 4 démos par
  case = 100 démos
- Méthode aléatoire : génère 100 paires (x, y) au hasard sur ton phone,
  pose le bloc à chaque pair

L'orientation du bloc peut aussi être randomisée (rotation autour de l'axe Z) :
0°, 30°, 60°, 90°. Ou laissé fixe pour simplifier.

#### Plus de variabilité dans le mouvement
Tes trajectoires de démos doivent **adapter le geste à la position du bloc** :
- Bloc à gauche → bras part vers la gauche
- Bloc à droite → bras part vers la droite
- Bloc proche → trajectoire courte
- Bloc loin → trajectoire longue

C'est l'exact opposé du sanity check où on cherchait l'identité.

#### Training plus long
- Sanity : 20 000 steps
- Eval 1 : **50 000 à 100 000 steps**, parce qu'il y a plus de modes à
  apprendre dans les données

Sur RTX 5070 : 50k steps ≈ 1h30, 100k ≈ 3h.

#### Augmentation de données (optionnel)
On peut activer dans LeRobot des transforms image (color jitter, blur,
crop) pour rendre la policy plus robuste. À expérimenter si la première
version foire au deploy.

### Pipeline Eval 1, étape par étape

#### a) Définir la zone
- Marque au scotch un carré 20×20 cm sur la table (4 coins)
- Photographie pour archive
- Note les coordonnées dans `notes/eval1_protocol.md` (à créer)

#### b) Préparer le sampling
- 5×5 = 25 cases × 4 démos par case = 100 démos
- Ou aléatoire pur : 100 positions tirées au hasard
- Trace la grille sur la table (au crayon léger, ou marque mentale)

#### c) Enregistrer les démos
- Même commande `lerobot-record` qu'au sanity check
- `--dataset.repo_id=Rsebti/projet3-eval1-demos`
- `--dataset.num_episodes=100`
- Pour chaque épisode : place le bloc à la nouvelle position, fais le pick
  qui colle à cette position, place dans le bowl

Temps estimé : ~2-3h pour 100 démos avec pauses.

#### d) Replay verification
- Comme au sanity check, rejoue 5-10 épisodes répartis : épisodes 0, 20,
  40, 60, 80
- Le robot doit reproduire chaque démo correctement, à des positions de
  bloc différentes

#### e) Push HF
- Push automatique via `push_to_hub=true`

#### f) Train
```bash
lerobot-train \
  --dataset.repo_id=Rsebti/projet3-eval1-demos \
  --policy.type=act \
  --policy.device=cuda \
  --policy.repo_id=Rsebti/projet3-act-eval1 \
  --output_dir=outputs/train/act_eval1 \
  --job_name=act_eval1 \
  --batch_size=8 \
  --steps=80000 \
  --save_freq=10000 \
  --eval_freq=0 \
  --log_freq=500 \
  --wandb.enable=true
```

Différences :
- `--steps=80000` (plus long)
- `--save_freq=10000` (8 checkpoints)
- `--wandb.enable=true` recommandé pour suivre les courbes

Suivre la loss : doit descendre **moins bas** qu'au sanity check (il y a
plus de variabilité, donc impossible d'overfit complètement). Loss finale
attendue : ~0.10-0.20 (vs 0.03-0.05 au sanity).

#### g) Deploy + auto-éval
Tu vas au robot avec le checkpoint. Tu fais le test sur **20 positions** :
- 10 positions vues dans le dataset (pour vérifier qu'il les fait bien)
- 10 positions inédites (pour tester la généralisation)

Compte le taux de réussite :
```
20 / 20 = 100% — excellent
15-19 / 20 = bon
10-14 / 20 = moyen, à itérer
< 10 / 20 = mauvais, retourner aux démos
```

#### h) Iteration
Si moyen ou mauvais, identifie le mode d'échec :
- Échoue sur les positions vues → sous-entraîné, ajoute des steps
- Échoue sur les positions nouvelles uniquement → manque de données,
  ajoute 50 démos sur les zones où ça foire
- Échoue partout → bug ailleurs (calibration, scène, mauvaise
  caméra index)

### Critères de validation Eval 1
À soumettre aux TAs :
- Le dataset HF
- Le checkpoint HF
- Une vidéo de 10 épisodes consécutifs au deploy
- Un bref rapport (1 page) : approche, hyperparamètres, taux de succès,
  modes d'échec observés

### Pièges classiques

#### Démos pas assez variées
20 démos très différentes < 100 démos avec petites variations contrôlées.
Le sampling doit être uniforme dans la zone.

#### Pose home variable
Si le robot ne part pas exactement de la même pose home, le modèle apprend
à compenser ces variations → bruit. Discipline : home fixe, marque les
articulations.

#### Caméra qui bouge entre démos et deploy
Si tu démontes/remontes la caméra wrist, l'angle change → distribution
d'images différente du training → modèle confus. Ne touche pas la caméra.

#### Lighting variable
Si tu enregistres en plusieurs sessions avec lumière différente, ACT peut
apprendre des dépendances inutiles à l'éclairage. Une seule session si
possible, ou domain augmentation au training.

---

## Annexe — fichiers de référence du repo

- `teleop/record_demos.md` : commande `lerobot-record` template
- `teleop/demo_protocol.md` : checklist de la session de recording
- `train/train_bc.md` : commande `lerobot-train` template
- `deploy/deploy_policy.md` : commande de déploiement
- `notes/laptop_setup.md` : install à zéro sur le portable
- `notes/full_pipeline_walkthrough.md` : ce fichier
