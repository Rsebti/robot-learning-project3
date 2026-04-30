"""Eval 2 — Targeted pick-and-place in clutter (RL).

Task progression (v0 -> v3):
- v0: single block + single bowl (no clutter yet) — validates the env structure.
- v1: two adjacent colored blocks + bowl, target color as a goal input.
- v2: + wrist camera as observation.
- v3: + aggressive domain randomization (sim-to-real).

Importing this module registers the gym environments below.
"""

import gymnasium as gym

from . import agents

# ---------------------------------------------------------------------------
# v0 — single block, single bowl. Closest to the upstream Lift task, with the
# virtual goal pose replaced by a real bowl as the placement target.
# ---------------------------------------------------------------------------

gym.register(
    id="Eval2-PickInBowl-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:Eval2PickInBowlEnvCfg_v0",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Eval2PPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Eval2-PickInBowl-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:Eval2PickInBowlEnvCfg_v0_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Eval2PPORunnerCfg",
    },
    disable_env_checker=True,
)
