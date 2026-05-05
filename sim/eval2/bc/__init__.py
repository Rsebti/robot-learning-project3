"""Eval 2 — BC warmstart pipeline.

Workflow:

    1. scripted_controller.py   IK + state-machine controller that solves
                                pick-in-clutter deterministically.
    2. run_scripted.py          Smoke test on Eval2-PickInClutter-Play-v1
                                (50 envs). Reports success rate. Goal: >=80%.
    3. collect_demos.py         Roll the scripted controller in 4096 parallel
                                envs, keep success episodes only, dump
                                (obs, action) pairs to demos.pt.
    4. train_bc.py              Train an MLP actor [256, 128, 64] (matching
                                the rsl_rl PPO actor) on the demos via MSE.
    5. (modify ../scripts/train.py) Load the BC actor weights into the rsl_rl
                                runner before runner.learn() to PPO-finetune.

Why BC warmstart
    PPO from-scratch on Eval 2 plateaus at ~2% success because the policy
    games milestone rewards without ever committing to the full grasp -> lift
    -> drop sequence. A scripted controller can solve the task by IK at >=90%
    success; BC distills that into the actor; PPO then refines.
"""
