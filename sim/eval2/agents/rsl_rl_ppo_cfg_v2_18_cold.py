"""V2.18 PPO config — COLD-START variant (no resume).

Same hyperparameters as V2.18 resume, but no `load_run` / `load_checkpoint`
settings. Used for V218-cold-* gym task IDs.

Why a cold-start V2.18:
    Resuming V2.18 from V2.15 model_300 failed because V2.15's "false grasp"
    policy (jaws closing beside the cube) cannot trigger V2.18's strict_grasp
    predicate (which requires actual geometric containment between jaws).
    With no positive grasp signal, the resume run degrades reach skills
    without finding the strict-grasp manifold.
    Cold-start gives V2.18 a clean slate to learn the precision-landing
    behavior the design was calibrated for. ETA ~14-19 h on RTX 5070.
"""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class LiftCubePPORunnerCfgV218Cold(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 50
    max_iterations = 1500
    save_interval = 50
    experiment_name = "lift_v2_13"
    empirical_normalization = False

    # No resume / load_run / load_checkpoint — cold-start from random init.

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.02,
        max_grad_norm=1.0,
    )
