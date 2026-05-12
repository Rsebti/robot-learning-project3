"""V2.8 PPO config — fix the value-function bottleneck identified on V2.7.

V2.7 unblocked Phase B reward shaping (anti-flick gates + success bonus
+200) and ran with a stable adaptive LR (~2-3e-4 across iter 100). But two
pathologies in the loss curves told us PPO itself was bottlenecked:

1. ``Loss/value_function`` oscillated 0.05-0.3+ across iter 0-100 with no
   downward trend and discrete spikes co-located with each
   ``success_bonus`` event (+200 unmodeled reward). With
   ``value_loss_coef=0.01`` (HW4 + V2.5/V2.6 inheritance), the value head
   couldn't allocate enough gradient to learn the sparse success
   discontinuity — every success was a fresh surprise.

2. ``Loss/entropy`` plateaued at ~5.60 (actually rose +0.20 from iter 0)
   while V2.6 had decayed cleanly 5.40→5.00 on the same window. The
   policy couldn't commit to a mode because the critic gave it noisy
   advantages.

These two symptoms are mechanically linked — bad VF → noisy advantages →
no policy convergence — and the root cause is the ×0.01 value coefficient.

V2.8 follows the Isaac Lab canonical PPO config for `Lift-Cube-Franka`
exactly, except for the rsl_rl-specific knobs we already validated on V2.7
(``schedule="adaptive"``, ``entropy_coef=0.005``, ``hidden_dims=[256,128,
128]``, ``use_clipped_value_loss=True``).

Changes vs V2.7 (7 changes):

+----------------------------+--------+--------+-----------------------+
| Parameter                  | V2.7   | V2.8   | Why                   |
+----------------------------+--------+--------+-----------------------+
| ``value_loss_coef``        | 0.01   | 1.0    | ×100. The critical    |
|                            |        |        | fix. Lets the VF      |
|                            |        |        | learn the +200 sparse |
|                            |        |        | bonus instead of      |
|                            |        |        | being surprised every |
|                            |        |        | time.                 |
+----------------------------+--------+--------+-----------------------+
| ``num_learning_epochs``    | 10     | 5      | Isaac Lab canonical.  |
|                            |        |        | Halves the per-       |
|                            |        |        | rollout overfit, so   |
|                            |        |        | desired_kl can stay   |
|                            |        |        | tight.                |
+----------------------------+--------+--------+-----------------------+
| ``num_mini_batches``       | 32     | 4      | Isaac Lab canonical.  |
|                            |        |        | Bigger batches per    |
|                            |        |        | grad step → less      |
|                            |        |        | noisy gradient on the |
|                            |        |        | critic.               |
+----------------------------+--------+--------+-----------------------+
| ``learning_rate``          | 3e-4   | 1e-3   | Isaac Lab canonical.  |
|                            |        |        | Faster early; the     |
|                            |        |        | adaptive schedule     |
|                            |        |        | will regulate.        |
+----------------------------+--------+--------+-----------------------+
| ``desired_kl``             | 0.03   | 0.01   | With 5 epochs not 10, |
|                            |        |        | tightening is safe;   |
|                            |        |        | matches Isaac Lab     |
|                            |        |        | canonical.            |
+----------------------------+--------+--------+-----------------------+
| ``max_grad_norm``          | 0.5    | 1.0    | Isaac Lab canonical.  |
|                            |        |        | Lets the success      |
|                            |        |        | bonus gradient        |
|                            |        |        | propagate without     |
|                            |        |        | clipping.             |
+----------------------------+--------+--------+-----------------------+
| ``init_noise_std``         | 0.6    | 1.0    | Isaac Lab canonical.  |
|                            |        |        | Wider initial         |
|                            |        |        | exploration, decays   |
|                            |        |        | with the policy.      |
+----------------------------+--------+--------+-----------------------+

Unchanged (validated on V2.7): ``hidden_dims=[256, 128, 128]``,
``use_clipped_value_loss=True``, ``entropy_coef=0.005``,
``schedule="adaptive"``, ``gamma=0.99``, ``lam=0.95``,
``num_steps_per_env=50``, ``clip_param=0.2``.

Reward shaping unchanged: V2.8 reuses ``LeIsaacLiftCubeRLEnvCfgV27`` /
``LeIsaacLiftCubeRLVisualEnvCfgV27`` (anti-flick gates + success_bonus
weight 200), validated empirically on V2.7.

Expected signature on a healthy V2.8 run (first 100-200 iters):
- ``Loss/value_function`` descends below 0.05 and stays (vs V2.7's
  chronic 0.05-0.3 oscillation)
- ``Loss/entropy`` decreases monotonically (vs V2.7 plateau)
- ``Episode_Reward/grasping_cube`` reaches V2.7 iter-88 level (~0.08)
  in ~30-40 iters
- Non-spurious successes start showing on
  ``Episode_Termination/success`` smoothed > 0 around iter 80-150
"""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class LiftCubePPORunnerCfgV28(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 50
    max_iterations = 1500
    save_interval = 50
    experiment_name = "lift_v2_8"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,                  # V2.8: was 0.6 (Isaac Lab canonical)
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,                 # V2.8: was 0.01 — CRITICAL FIX
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,               # V2.8: was 10 (Isaac Lab canonical)
        num_mini_batches=4,                  # V2.8: was 32 (Isaac Lab canonical)
        learning_rate=1.0e-3,                # V2.8: was 3e-4 (Isaac Lab canonical)
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,                     # V2.8: was 0.03 (safe with 5 epochs)
        max_grad_norm=1.0,                   # V2.8: was 0.5 (canonical)
    )
