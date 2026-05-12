"""V2.18 PPO config with hardcoded resume from V2.15 model_300.

V2.18 = "Precision Landing" structural rewrite of the reward function
(Claude search design). Env-side changes only; PPO unchanged from V2.9
baseline.

Resume target = V2.15 model_300 (the clean checkpoint with reach+grasp
+top-down posture skills established). V2.16, V2.17, V2.17 runs are
abandoned (regression / collapse).

User MUST pass `--resume` on the CLI for resume to actually fire (Isaac
Lab's `cli_args.update_rsl_rl_cfg` overrides agent_cfg.resume from the
argparse `--resume` flag value, default False).
"""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class LiftCubePPORunnerCfgV218Resume(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 50
    max_iterations = 1500
    save_interval = 50
    experiment_name = "lift_v2_13"
    empirical_normalization = False

    # Resume from V2.15 model_300 (clean baseline with reach+grasp+top-down
    # already learned). V2.18's structural reward rewrite gives the policy
    # the precision-landing + lift signals it was missing.
    load_run: str = "2026-05-11_17-25-46"
    load_checkpoint: str = "model_300.pt"

    def __post_init__(self):
        parent = super()
        if hasattr(parent, "__post_init__"):
            parent.__post_init__()
        self.resume = True

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
