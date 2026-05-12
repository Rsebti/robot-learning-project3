"""Eval 2 — RL pick-and-place on SO-101.

Importing this module registers our custom Gym tasks. The actual env_cfg and
agent classes are imported lazily by Isaac Lab's `gym.make()` call (string
entry points), which means this module is safe to import BEFORE
`AppLauncher` is initialized — the heavy `isaac_so_arm101.tasks` import only
happens when the env is actually constructed.

Side effect: sets ``LEISAAC_ASSETS_ROOT`` to the LeIsaac local clone if the
variable is not already defined. LeIsaac otherwise auto-detects the git root
of the current working directory, which resolves to ``isaac_so_arm101`` when
training is launched from there — leading to USD load failures.
"""
import os
from pathlib import Path

# Default LeIsaac assets path used by the team. Override by setting
# LEISAAC_ASSETS_ROOT before importing this module.
_DEFAULT_LEISAAC_ASSETS = Path(r"C:\Users\user\Desktop\MA2\isaac\leisaac\assets")
if "LEISAAC_ASSETS_ROOT" not in os.environ and _DEFAULT_LEISAAC_ASSETS.is_dir():
    os.environ["LEISAAC_ASSETS_ROOT"] = str(_DEFAULT_LEISAAC_ASSETS)

import gymnasium as gym


# v1 — upstream isaac_so_arm101 PPO config (the one that diverged in our run)
gym.register(
    id="Isaac-SO-ARM101-Lift-Cube-Restricted-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.lift_smoketest_env_cfg:SoArm101LiftCubeRestrictedEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "isaac_so_arm101.tasks.lift.agents.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-Lift-Cube-Restricted-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.lift_smoketest_env_cfg:SoArm101LiftCubeRestrictedEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "isaac_so_arm101.tasks.lift.agents.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg"
        ),
    },
    disable_env_checker=True,
)


# v2 — Stable PPO config ported from lerobot-sim2real
gym.register(
    id="Isaac-SO-ARM101-Lift-Cube-Restricted-Stable-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.lift_smoketest_env_cfg:SoArm101LiftCubeRestrictedEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_stable:LiftCubePPORunnerCfgStable"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-Lift-Cube-Restricted-Stable-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.lift_smoketest_env_cfg:SoArm101LiftCubeRestrictedEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_stable:LiftCubePPORunnerCfgStable"
        ),
    },
    disable_env_checker=True,
)


# v3 — Stable PPO + curriculum disabled + gamma 0.95 + small entropy
gym.register(
    id="Isaac-SO-ARM101-Lift-Cube-Restricted-V2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.lift_smoketest_env_cfg:SoArm101LiftCubeRestrictedV2EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2:LiftCubePPORunnerCfgV2"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-Lift-Cube-Restricted-V2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.lift_smoketest_env_cfg:SoArm101LiftCubeRestrictedV2EnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2:LiftCubePPORunnerCfgV2"
        ),
    },
    disable_env_checker=True,
)


# v4 — pivot to LeIsaac scaffold. Cleaner robot config (no convex_decomposition
# magic-attach baggage), canonical SO-101 wrist cam already calibrated, more
# maintained codebase. Same V2 PPO hyperparameters (gamma=0.95, entropy=0.002,
# fixed LR schedule). State-only observations for Phase A — Phase B will add
# the wrist cam through a frozen ResNet encoder.
gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2:LiftCubePPORunnerCfgV2"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2:LiftCubePPORunnerCfgV2"
        ),
    },
    disable_env_checker=True,
)


# v5 — Phase B visual variant: wrist cam re-enabled, ResNet-18 frozen features
# concatenated with state in policy obs (28 + 512 = 540 D).
# Launch with --enable_cameras (Isaac Lab requires the flag for TiledCamera).
gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2:LiftCubePPORunnerCfgV2"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2:LiftCubePPORunnerCfgV2"
        ),
    },
    disable_env_checker=True,
)


# v6 — replay the original isaac_so_arm101 PPO defaults on the clean LeIsaac
# scaffold. Earlier runs failed (entropy explosion) but had env-side bugs
# (convex_decomposition gripper, action_rate curriculum ramp). On the clean
# LeIsaac env, those defaults might actually converge — this is the test.
# Same env as `Isaac-LeIsaac-SO101-Lift-RL-v0`/`-Visual-v0`, only the agent
# config changes.
gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-IsaacDefaults-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_isaac_defaults:"
            "LiftCubePPORunnerCfgIsaacDefaults"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-IsaacDefaults-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_isaac_defaults:"
            "LiftCubePPORunnerCfgIsaacDefaults"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-IsaacDefaults-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_isaac_defaults:"
            "LiftCubePPORunnerCfgIsaacDefaults"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-IsaacDefaults-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_isaac_defaults:"
            "LiftCubePPORunnerCfgIsaacDefaults"
        ),
    },
    disable_env_checker=True,
)


# v7 — V2.5: V2 baseline + HW4-inspired changes (value_loss_coef=0.01,
# num_learning_epochs=10, gamma=0.99, hidden_dims=[256,128,128]). Same env
# as the V2 tasks, only the agent config changes.
gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V25-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_5:LiftCubePPORunnerCfgV25"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V25-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_5:LiftCubePPORunnerCfgV25"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V25-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_5:LiftCubePPORunnerCfgV25"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V25-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_5:LiftCubePPORunnerCfgV25"
        ),
    },
    disable_env_checker=True,
)


# v8 — V2.6: V2.5 + 3 HW4 alignments missed in V2.5 (entropy_coef=0.005,
# use_clipped_value_loss=True, schedule="adaptive" with desired_kl=0.01).
# Same env as V2/V2.5, only the agent config changes.
gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V26-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_6:LiftCubePPORunnerCfgV26"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V26-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_6:LiftCubePPORunnerCfgV26"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V26-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_6:LiftCubePPORunnerCfgV26"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V26-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_6:LiftCubePPORunnerCfgV26"
        ),
    },
    disable_env_checker=True,
)


# v9 — V2.7: V2.6 + flick-exploit fixes (lifting + goal_tracking gated on
# grasp via cube_lifted_and_grasped / cube_to_goal_distance_grasped_and_lifted),
# grasp thresholds relaxed (diff 0.02→0.04, grasp 0.26→0.35), lifting weight
# 15→10, desired_kl 0.01→0.03. New env classes (V27 RewardsCfg).
gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V27-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfgV27"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_7:LiftCubePPORunnerCfgV27"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V27-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfgV27_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_7:LiftCubePPORunnerCfgV27"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V27-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfgV27"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_7:LiftCubePPORunnerCfgV27"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V27-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfgV27_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_7:LiftCubePPORunnerCfgV27"
        ),
    },
    disable_env_checker=True,
)


# v10 — V2.8: V27 envs (no cube_dropped) + Isaac Lab canonical PPO config
# (value_loss_coef=1.0 ×100, num_learning_epochs=5, num_mini_batches=4,
# learning_rate=1e-3, desired_kl=0.01, max_grad_norm=1.0,
# init_noise_std=1.0). This is the baseline the user actually ran —
# preserved here for reproducibility / comparison with V2.8.5.
gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V28-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfgV27"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_8:LiftCubePPORunnerCfgV28"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V28-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfgV27_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_8:LiftCubePPORunnerCfgV28"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V28-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfgV27"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_8:LiftCubePPORunnerCfgV28"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V28-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfgV27_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_8:LiftCubePPORunnerCfgV28"
        ),
    },
    disable_env_checker=True,
)


# v11 — V2.8.5: V2.8 PPO unchanged + cube_dropped failure termination
# (V285 envs). Logs go to `logs/rsl_rl/lift_v2_8_5/` for clean separation
# from the V2.8 baseline. The only difference vs V2.8 is one new DoneTerm.
gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V285-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfgV285"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_8_5:LiftCubePPORunnerCfgV285"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V285-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfgV285_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_8_5:LiftCubePPORunnerCfgV285"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V285-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfgV285"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_8_5:LiftCubePPORunnerCfgV285"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V285-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfgV285_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_8_5:LiftCubePPORunnerCfgV285"
        ),
    },
    disable_env_checker=True,
)


# v12 — V2.9: V2.8.5 envs + cube_dropped_penalty RewTerm (weight=-5.0).
# V2.8.5 had ~41% cube_dropped rate at iter 80 because dropping was
# net-positive EV (no explicit cost). V2.9 pairs the existing
# cube_dropped DoneTerm with a -5.0 RewTerm penalty that fires once at
# the drop step, making "bump-and-drop" net negative. PPO config V2.8
# unchanged; only experiment_name differs (lift_v2_9).
gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V29-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfgV29"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_9:LiftCubePPORunnerCfgV29"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V29-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfgV29_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_9:LiftCubePPORunnerCfgV29"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V29-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfgV29"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_9:LiftCubePPORunnerCfgV29"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V29-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfgV29_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_9:LiftCubePPORunnerCfgV29"
        ),
    },
    disable_env_checker=True,
)


# v13 — V2.10: V2.9 + smoothness fixes (action_rate ×500, joint_vel ×100,
# joint_acc NEW, action.scale 0.5→0.25, init_noise_std 1.0→0.4, entropy
# 0.005→0.002) + table color override (#B8ADA9). Targets the chaotic
# joint-velocity saturation observed at deploy time on V2.9.
gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V210-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfgV210"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_10:LiftCubePPORunnerCfgV210"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V210-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfgV210_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_10:LiftCubePPORunnerCfgV210"
        ),
    },
    disable_env_checker=True,
)

# View task: 1 env, 2s episodes — for visual inspection of cube spawns
# without any policy (use with `view.py` / zero_agent).
gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V210-View-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfgV210_VIEW"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_10:LiftCubePPORunnerCfgV210"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V210-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfgV210"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_10:LiftCubePPORunnerCfgV210"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V210-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfgV210_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_10:LiftCubePPORunnerCfgV210"
        ),
    },
    disable_env_checker=True,
)


# v14 — V2.10b: V2.10 with joint_acc_l2 weight rebalanced (-1e-3 → -3e-5,
# ÷33). All other V2.10 changes (USD cube 2cm, table #B8ADA9, LeIsaac
# randomization, action.scale=0.25, obs pre-allocation, init_noise=0.4,
# entropy=0.002) preserved. Cold-start only — V2.10 warm-start from V2.9
# failed because the V2.9 yeet motor program is structurally incompatible
# with the V2.10 smoothness regime.
gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V210b-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfgV210b"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_10b:LiftCubePPORunnerCfgV210b"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V210b-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfgV210b_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_10b:LiftCubePPORunnerCfgV210b"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V210b-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfgV210b"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_10b:LiftCubePPORunnerCfgV210b"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V210b-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfgV210b_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_10b:LiftCubePPORunnerCfgV210b"
        ),
    },
    disable_env_checker=True,
)


# v15 — V2.10c: V2.10b + dense linear distance reward EE↔cube (weight=-1.0).
# Fixes the "policy drifts away from cube" failure mode of V2.10b: the
# tanh `reaching_object` saturates past d ≈ 60 cm, so the policy gets no
# signal to come back when it drifts. The linear term gives a constant
# -1/m gradient throughout the workspace, creating a global drive in the
# value function. Term self-extinguishes during grasp/lift/transport
# (||EE - cube|| → 0 when held).
gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V210c-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfgV210c"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_10c:LiftCubePPORunnerCfgV210c"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V210c-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfgV210c_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_10c:LiftCubePPORunnerCfgV210c"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V210c-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfgV210c"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_10c:LiftCubePPORunnerCfgV210c"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V210c-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfgV210c_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_10c:LiftCubePPORunnerCfgV210c"
        ),
    },
    disable_env_checker=True,
)


# v16 — V2.11: aligned with Isaac Lab Lift defaults + retain ee_to_cube_distance.
# Reset smoothness weights to noise floor (-1e-4 each, drop joint_acc), revert
# tracking gating to lift-only (vs V2.7's grasp+lift), restore Isaac Lab Lift
# weights (lifting=15). Keep V2.10's action.scale=0.25, USD edits, randomization,
# placeholders. Keep V2.10c's ee_to_cube_distance linear penalty.
gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V211-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfgV211"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_11:LiftCubePPORunnerCfgV211"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V211-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfgV211_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_11:LiftCubePPORunnerCfgV211"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V211-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfgV211"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_11:LiftCubePPORunnerCfgV211"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V211-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfgV211_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_11:LiftCubePPORunnerCfgV211"
        ),
    },
    disable_env_checker=True,
)


# v17 — V2.12: V2.9 reward + V2.9 PPO + DELTA action control.
# `RelativeJointPositionActionCfg(scale=0.20, use_zero_offset=True)` gives
# a hard mechanical vmax cap = 6 rad/s (Feetech-aligned). With velocity
# bounded structurally, all the reward shaping that V2.10/V2.11 added to
# fight the velocity issue is reverted. V2.9 reward + PPO config restored
# verbatim. Single addition kept from V2.10c: `ee_to_cube_distance` linear
# distance penalty for global value-function gradient.
gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V212-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfgV212"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_12:LiftCubePPORunnerCfgV212"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V212-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfgV212_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_12:LiftCubePPORunnerCfgV212"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V212-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfgV212"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_12:LiftCubePPORunnerCfgV212"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V212-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfgV212_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_12:LiftCubePPORunnerCfgV212"
        ),
    },
    disable_env_checker=True,
)


# v18 — V2.13: V2.12 + action clip + posture penalties to force top-down grasp.
# Diagnostic V2.12 showed action saturation (raw actions ±6 vs expected ±1)
# and snake/scoop motor program (gripper pointing UP at grasp, jaw scraping
# table). V2.13 fixes both with: clip={".*": (-1.0, 1.0)} on action class
# (real vmax cap 6 rad/s), gripper_orientation_penalty (-1.0 weight, no
# free hover bonus), scoop_grasp_penalty (-10.0 weight, geometric
# constraint wrist_z >= ee_z). All other V2.12 reward shaping preserved.
gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V213-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfgV213"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_13:LiftCubePPORunnerCfgV213"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V213-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfgV213_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_13:LiftCubePPORunnerCfgV213"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V213-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfgV213"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_13:LiftCubePPORunnerCfgV213"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V213-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfgV213_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_13:LiftCubePPORunnerCfgV213"
        ),
    },
    disable_env_checker=True,
)


# V2.13 v3 — sign-bug fix on gripper_orientation_penalty +
# reach landscape overhaul. Same PPO config as V213 (env-side changes only).
# See `leisaac_lift_env_cfg.py::LeIsaacLiftCubeRLEnvCfgV213v3` for rationale.
gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V213v3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfgV213v3"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_13:LiftCubePPORunnerCfgV213"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V213v3-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfgV213v3_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_13:LiftCubePPORunnerCfgV213"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V213v3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfgV213v3"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_13:LiftCubePPORunnerCfgV213"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V213v3-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfgV213v3_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_13:LiftCubePPORunnerCfgV213"
        ),
    },
    disable_env_checker=True,
)


# V2.14 — V2.13 v3 + 10s episodes + arm_action.scale=0.10 (vmax 3 rad/s) +
# joint_vel & action_rate ×10 to kill V213v3's Bang-Bang vertical smash.
# Same PPO config (env-side changes only).
gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V214-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfgV214"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_13:LiftCubePPORunnerCfgV213"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V214-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfgV214_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_13:LiftCubePPORunnerCfgV213"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V214-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfgV214"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_13:LiftCubePPORunnerCfgV213"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V214-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfgV214_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_13:LiftCubePPORunnerCfgV213"
        ),
    },
    disable_env_checker=True,
)


# V2.15 — V2.14 + strict-top-down via palm→jaw direction + jaw_below_cube_penalty
# Same PPO config (env-side reward changes only).
gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V215-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfgV215"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_13:LiftCubePPORunnerCfgV213"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V215-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfgV215_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_13:LiftCubePPORunnerCfgV213"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V215-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfgV215"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_13:LiftCubePPORunnerCfgV213"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V215-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfgV215_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_13:LiftCubePPORunnerCfgV213"
        ),
    },
    disable_env_checker=True,
)


# V2.16 — V2.15 + cube_height_above_spawn dense lift reward (+30) to break
# the "grasp-only local optimum" observed in V2.15 model_300. Same PPO
# config (env-side change only).
gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V216-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfgV216"
        ),
        # V216 uses the resume-enabled PPO config (resume=true hardcoded
        # because Hydra CLI override of bool fields doesn't apply on
        # Windows PowerShell — only strings make it through).
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_16_resume:LiftCubePPORunnerCfgV216Resume"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V216-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfgV216_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_16_resume:LiftCubePPORunnerCfgV216Resume"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V216-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfgV216"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_16_resume:LiftCubePPORunnerCfgV216Resume"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V216-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfgV216_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_16_resume:LiftCubePPORunnerCfgV216Resume"
        ),
    },
    disable_env_checker=True,
)


# V2.17 — V2.16 + 5x boost on cube_height_above_spawn (+30→+150) + half
# binary lift threshold (0.08→0.04). Resume from V2.16 model_350. User
# MUST pass --resume on the CLI for resume to fire (Isaac Lab CLI handler
# overrides agent_cfg.resume from --resume argparse flag).
gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V217-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfgV217"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_17_resume:LiftCubePPORunnerCfgV217Resume"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V217-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfgV217_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_17_resume:LiftCubePPORunnerCfgV217Resume"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V217-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfgV217"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_17_resume:LiftCubePPORunnerCfgV217Resume"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V217-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfgV217_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_17_resume:LiftCubePPORunnerCfgV217Resume"
        ),
    },
    disable_env_checker=True,
)


# V2.18 — "Precision Landing" structural rewrite (Claude search design).
# Bounded-magnitude (|r|≤5 budget), multiplicatively-gated reward stack.
# Resumes from V2.15 model_300 (clean baseline; V2.16/V2.17 abandoned).
# User MUST pass --resume on the CLI for resume to fire.
gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V218-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfgV218"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_18_resume:LiftCubePPORunnerCfgV218Resume"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V218-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfgV218_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_18_resume:LiftCubePPORunnerCfgV218Resume"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V218-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfgV218"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_18_resume:LiftCubePPORunnerCfgV218Resume"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V218-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfgV218_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_18_resume:LiftCubePPORunnerCfgV218Resume"
        ),
    },
    disable_env_checker=True,
)


# V2.18 COLD — same env as V2.18, but a cold-start PPO config (no resume).
# Used after the V2.18 resume from V2.15 model_300 failed because V2.15's
# "false grasp" policy cannot fire V2.18's strict predicate, leaving the
# resumed run without any positive grasp signal.
gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V218-cold-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfgV218"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_18_cold:LiftCubePPORunnerCfgV218Cold"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V218-cold-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfgV218_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_18_cold:LiftCubePPORunnerCfgV218Cold"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V218-cold-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfgV218"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_18_cold:LiftCubePPORunnerCfgV218Cold"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V218-cold-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfgV218_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_18_cold:LiftCubePPORunnerCfgV218Cold"
        ),
    },
    disable_env_checker=True,
)


# V2.18 b — anti-hover-stall + anti-VF-blowup hot-fix.
# Env-side : hover_height weight 0.5 → 0.2 (RewardsCfgV218B).
# PPO-side : 9 changes from Claude search (notes/v218b_ppo_search_claude.md):
#   empirical_norm=True, noise_std=log, desired_kl=0.01, lr_init=3e-4,
#   entropy_coef=0.01, value_clip=False, RND, wider critic, more SGD.
gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V218B-cold-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfgV218B"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_18b_cold:LiftCubePPORunnerCfgV218BCold"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-RL-V218B-cold-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLEnvCfgV218B_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_18b_cold:LiftCubePPORunnerCfgV218BCold"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V218B-cold-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfgV218B"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_18b_cold:LiftCubePPORunnerCfgV218BCold"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V218B-cold-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfgV218B_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_18b_cold:LiftCubePPORunnerCfgV218BCold"
        ),
    },
    disable_env_checker=True,
)


# V2.18 b PICKLIFT — symmetric actor-critic, no cube ground-truth anywhere.
# Both actor and critic input: joint_pos + joint_vel + wrist_features = 524 dims.
# Same rewards / geometry / PPO hyperparams as V218B; only obs structure changes.
# Sim2real-clean: critic stays in the same "no perception" regime as the actor.
gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V218B-picklift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:LeIsaacLiftCubeRLVisualEnvCfgV218BPickLift"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_18b_picklift_cold:"
            "LiftCubePPORunnerCfgV218BPickLiftCold"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LeIsaac-SO101-Lift-Visual-V218B-picklift-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "sim.eval2.leisaac_lift_env_cfg:"
            "LeIsaacLiftCubeRLVisualEnvCfgV218BPickLift_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "sim.eval2.agents.rsl_rl_ppo_cfg_v2_18b_picklift_cold:"
            "LiftCubePPORunnerCfgV218BPickLiftCold"
        ),
    },
    disable_env_checker=True,
)
