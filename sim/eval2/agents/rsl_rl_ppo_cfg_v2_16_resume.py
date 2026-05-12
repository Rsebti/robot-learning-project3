"""V2.16 PPO config with hardcoded resume from V2.15 model_300.

Workaround: Hydra CLI override of boolean fields (``agent.resume=true``)
does not apply on Windows PowerShell in our setup — verified via
agent.yaml after launch (string fields like load_run/load_checkpoint
ARE applied, but the boolean resume stays at default `false`). So we
hardcode the three resume fields here, and the V2.16 gym registration
points to THIS config (not the V2.13 one).

Hyperparameters are IDENTICAL to V2.13 (= V2.9 baseline). Only the
resume metadata differs.

NB: this config has ``resume = True`` permanently. Re-launching V2.16
will always try to resume from V2.15 ``model_300.pt``. If you ever
want a cold-start V2.16 (env-side only), use a different task ID or
override via CLI.
"""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class LiftCubePPORunnerCfgV216Resume(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 50
    max_iterations = 1500
    save_interval = 50
    experiment_name = "lift_v2_13"
    empirical_normalization = False

    # Hardcoded resume from V2.15 model_300 (the run that converged on
    # "grasp-only" before V2.16's cube_height_above_spawn was added).
    # We set load_run/load_checkpoint at class level (string fields,
    # Hydra & @configclass.to_dict() respect these). resume is set in
    # __post_init__ because BOTH class-level `resume = True` AND
    # `resume: bool = True` were silently lost during the @configclass /
    # dataclass / Hydra round-trip in our setup (verified empirically:
    # agent.yaml dumped `resume: false` in either case). The instance-
    # level assignment in __post_init__ is the only path that survives.
    load_run: str = "2026-05-11_17-25-46"
    load_checkpoint: str = "model_300.pt"

    def __post_init__(self):
        # Call parent __post_init__ first if it exists (RslRlOnPolicyRunnerCfg
        # may have one for default resolution). Safe to skip with try/except
        # if the parent doesn't define it.
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
