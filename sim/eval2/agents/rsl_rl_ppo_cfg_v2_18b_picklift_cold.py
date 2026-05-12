"""V2.18 b PickLift PPO config — symmetric actor-critic over the
minimal sim2real-clean env.

Identical to `LiftCubePPORunnerCfgV218BCold` except for the
``obs_groups`` mapping, which tells rsl_rl that the actor and the critic
both consume the same single `policy` observation group:

    obs_groups = {
        "policy": ["policy"],   # actor input
        "critic": ["policy"],   # critic input (same group → symmetric AC)
    }

Network input dims (auto-derived by rsl_rl from the obs space at
runtime — we do NOT need to set them in the config):

    actor first-layer  : 524  (= joint_pos 6 + joint_vel 6 + wrist_features 512)
    critic first-layer : 524  (same as actor — symmetric AC)

Why symmetric (not asymmetric):
    Cube ground-truth is unavailable at deploy on the real SO-101. We
    keep the critic in the same "no perception" regime as the actor so
    the value estimates stay robust to the perception noise we'll
    introduce later, and the value network can never learn to rely on
    info the deployed policy can't see.

Hidden dims unchanged from V218B Cold:
    actor_hidden_dims  = [256, 128, 128]
    critic_hidden_dims = [512, 256, 128]   (wider critic, Andrychowicz)

All other PPO hyperparams (learning rate, KL, RND, log noise, etc.)
inherited verbatim from V218B Cold — see that file for rationale.
"""
from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg_v2_18b_cold import LiftCubePPORunnerCfgV218BCold


@configclass
class LiftCubePPORunnerCfgV218BPickLiftCold(LiftCubePPORunnerCfgV218BCold):
    """V2.18b PickLift cold-start PPO with symmetric AC obs_groups."""

    obs_groups = {
        "policy": ["policy"],   # actor input: 524-D
        "critic": ["policy"],   # critic input: 524-D (same as actor)
    }
