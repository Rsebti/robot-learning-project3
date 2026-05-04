"""Eval 2 — Targeted pick-and-place in clutter (RL).

Task progression (v0 -> v3):
- v0: single block + single flat bowl (smoke-test pipeline).
- v1: two adjacent colored blocks + open-top bowl, target color one-hot input.
- v2: + wrist camera as observation (visual policy or modular perception).
- v3: + aggressive domain randomization for sim-to-real.

Importing this module registers the gym environments below.
"""

import gymnasium as gym

from . import agents

# ---------------------------------------------------------------------------
# v0 — single block, single flat bowl (smoke test pipeline).
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

# ---------------------------------------------------------------------------
# v1 — two adjacent colored blocks + concave bowl + target color goal.
# ---------------------------------------------------------------------------
gym.register(
    id="Eval2-PickInClutter-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:Eval2PickInClutterEnvCfg_v1",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Eval2PPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Eval2-PickInClutter-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:Eval2PickInClutterEnvCfg_v1_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Eval2PPORunnerCfg",
    },
    disable_env_checker=True,
)

# ---------------------------------------------------------------------------
# v2 — v1 + SO-101 wrist camera in the scene (RGB).
# Used for visual development, perception module training/validation, and
# real-robot deploy (where the camera obs is consumed live).
# ---------------------------------------------------------------------------
gym.register(
    id="Eval2-PickInClutter-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:Eval2PickInClutterEnvCfg_v2",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Eval2PPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Eval2-PickInClutter-Play-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:Eval2PickInClutterEnvCfg_v2_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Eval2PPORunnerCfg",
    },
    disable_env_checker=True,
)
