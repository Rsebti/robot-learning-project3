"""V2.17 PPO config with hardcoded resume from V2.16 model_350.

V2.17 = V2.16 + 5x boost on cube_height_above_spawn weight + half binary
lift threshold (env-side changes only). PPO unchanged.

Hardcodes resume from V2.16 run 2026-05-11_20-11-35 (which itself was
resumed from V2.15 model_300, then trained 50 iters of V2.16 dense-lift
exploration before plateauing).

NB: ``resume`` is set in __post_init__ because Isaac Lab's CLI handler
(`update_rsl_rl_cfg`) overwrites ``agent_cfg.resume`` with the argparse
``--resume`` flag value (default False) regardless of class hardcoding.
The user MUST also pass ``--resume`` on the CLI for resume to actually
fire — the hardcoded values here just provide ``load_run`` and
``load_checkpoint`` defaults.
"""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class LiftCubePPORunnerCfgV217Resume(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 50
    max_iterations = 1500
    save_interval = 50
    experiment_name = "lift_v2_13"
    empirical_normalization = False

    # Resume from V2.15 model_300 (the same checkpoint V2.16 used as start).
    # We skip V2.16's 50 iters of dense-lift exploration because they
    # degraded the gripper_orientation posture (-0.12 → -0.50) without
    # gaining any real lift (h_above plateau at noise zone). Restarting
    # from V2.15 model_300 with V2.17's stronger lift signal (5x boost +
    # halved binary threshold) gives the policy a fresh attempt with the
    # full top-down + grasp skills intact.
    load_run: str = "2026-05-11_17-25-46"
    load_checkpoint: str = "model_300.pt"

    def __post_init__(self):
        parent = super()
        if hasattr(parent, "__post_init__"):
            parent.__post_init__()
        self.resume = True

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,                                    # V2.9 default
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
        desired_kl=0.02,                                       # V2.12 inherited
        max_grad_norm=1.0,
    )
