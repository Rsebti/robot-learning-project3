# Eval 2 — RL pick-and-place on SO-101 (Isaac Lab + LeIsaac)

Pure-RL pipeline for the Eval 2 task (single colored cube + bowl xyz, no
BC, no DAPG, no teleop demos). Built on the LeIsaac scaffold and trained
with PPO via `rsl_rl`.

**Quick start (where to look first)**
- [`notes/eval2_env_overview.md`](../../notes/eval2_env_overview.md) — scene, dimensions, axes, action / observation / reward spaces (one-page reference).
- [`notes/eval2_pipeline.md`](../../notes/eval2_pipeline.md) — full pipeline doc with variant chronicle, dead-ends, cheat-sheet commands.
- [`leisaac_lift_env_cfg.py`](leisaac_lift_env_cfg.py) — all env configs (V2 → V2.15).
- [`mdp/rewards.py`](mdp/rewards.py) — custom reward functions.
- [`scripts/`](scripts/) — train / play / diagnose / dump utilities.
- [`__init__.py`](__init__.py) — gym task registrations.

## Current variant

**V2.15** (cold-start, ~14–19 h on RTX 5070). Tasks :
- `Isaac-LeIsaac-SO101-Lift-Visual-V215-v0` (training, wrist cam + ResNet)
- `Isaac-LeIsaac-SO101-Lift-Visual-V215-Play-v0` (replay, smaller scene)
- `Isaac-LeIsaac-SO101-Lift-RL-V215-v0` (state-only training, no cam)
- `Isaac-LeIsaac-SO101-Lift-RL-V215-Play-v0` (state-only replay)

See `notes/eval2_pipeline.md` § "Stack V2.15" for the exact reward
weights, action class, episode length, and PPO config.

## Visualiser l'environnement (sans entraîner)

Pour générer des captures d'écran de la scène (table, cube, robot SO-101,
wrist cam) à plusieurs angles :

```powershell
cd C:\Users\user\Desktop\MA2\robot-learning-project3
$env:RUST_LOG = "error"
C:/Users/user/Desktop/MA2/isaac/isaac_so_arm101/.venv/Scripts/python.exe `
  sim/eval2/scripts/showcase_env.py `
  --task Isaac-LeIsaac-SO101-Lift-Visual-V215-Play-v0 `
  --headless --enable_cameras
```

Sortie : `sim/eval2/showcase/` (PNG screenshots + wrist cam frames +
scene info dump).

## Cheat-sheet commands

```powershell
# Train V2.15 cold-start
... sim/eval2/scripts/train.py --task Isaac-LeIsaac-SO101-Lift-Visual-V215-v0 --headless --enable_cameras --num_envs 256

# Monitor live (other terminal)
... -m sim.eval2.scripts.monitor_training --experiment lift_v2_13 --interval 30

# TensorBoard
... -m tensorboard.main --logdir logs/rsl_rl/lift_v2_13 --port 6006

# Replay a checkpoint (visual)
... sim/eval2/scripts/play.py --task Isaac-LeIsaac-SO101-Lift-Visual-V215-Play-v0 --num_envs 4 --enable_cameras --checkpoint logs/rsl_rl/lift_v2_13/<TS>/model_<N>.pt

# Diagnostic complet (action histogram, grasp lifecycle, posture)
... sim/eval2/scripts/play_diagnose_v2.py --task Isaac-LeIsaac-SO101-Lift-Visual-V215-Play-v0 --num_envs 4 --num_episodes 16 --headless --enable_cameras --checkpoint <path>

# Verify env wiring without launching training
... sim/eval2/scripts/dump_scene_frames.py --task Isaac-LeIsaac-SO101-Lift-Visual-V215-Play-v0 --headless --enable_cameras
```

Voir `notes/eval2_pipeline.md` § "Commandes cheat-sheet" pour la
documentation complète des flags et l'historique des variants.

## Dépendances externes (non-tracked)

- **Isaac Sim 5.1** + **Isaac Lab 2.3** + **`isaac_so_arm101`** : installés via `uv` à `C:\Users\user\Desktop\MA2\isaac\isaac_so_arm101\`. Le venv `isaac_so_arm101/.venv` est le runtime utilisé pour tous les scripts.
- **Setup détaillé** : `notes/isaac_lab_setup.md`.

Le repo `isaac_so_arm101` est un scaffold de référence et NE DOIT PAS être
modifié. Tout le code projet vit dans ce repo (`sim/eval2/`).
