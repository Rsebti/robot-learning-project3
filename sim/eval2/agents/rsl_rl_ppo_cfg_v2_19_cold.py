"""V2.19 PPO config — cold-start over the full doc-driven reward rewrite.

Inherits all PPO hyperparameters from V218B Cold (empirical_normalization,
log noise_std, RND, wider critic [512,256,128], desired_kl=0.01,
learning_rate=3e-4, entropy_coef=0.01, clipped_value_loss=False, etc.).
The PPO machinery did NOT cause V2.18b's stall — the reward stack did.
We reuse the exact same hyperparameters here so that any difference in
the V2.19 run is attributable to the reward rewrite, not to PPO tuning.

Symmetric AC in the V2.19 default env (both actor and critic see the
same `policy` group). The doc recommends asymmetric AC + an auxiliary
cube-xyz prediction head; that requires modifying rsl_rl source and is
deferred to a future iteration. For now V2.19 uses symmetric AC like
V2.18b PickLift, which is also sim2real-clean.

Network input dim (auto-derived):
    actor first-layer  : <obs_dim>  (depends on whether we use the full
                                     V218B-Visual obs (~550 dims) or the
                                     PickLift-style minimal obs (~524))
    critic first-layer : same as actor (symmetric)

For the default `Isaac-LeIsaac-SO101-Lift-Visual-V219-cold-v0` task we
keep V218B Visual's full obs so the network has all signals while we
debug the new reward stack. Sim2real-deployable PickLift variant can
come later as `V219-picklift-v0`.
"""
from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg_v2_18b_cold import LiftCubePPORunnerCfgV218BCold


@configclass
class LiftCubePPORunnerCfgV219Cold(LiftCubePPORunnerCfgV218BCold):
    """V2.19 cold-start PPO. Same hyperparams as V218B Cold, ditto symmetric
    AC. Reward stack is fully replaced (see RewardsCfgV219)."""

    # No obs_groups override → defaults to whatever the env exposes (single
    # `policy` group used for both actor and critic by rsl_rl convention).
    # If we later add a `privileged` group for asymmetric AC, set:
    #   obs_groups = {"policy": ["policy"], "critic": ["policy", "privileged"]}
