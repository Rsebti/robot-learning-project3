"""Eval 2 — Scripted BC pipeline (ARCHIVED, abandoned 2026-05-05).

This module attempted to generate expert demos in sim via a closed-form
analytical IK + state machine, to warmstart BC pretraining + PPO finetune.

**Status: abandoned.** The IK math is correct (test_analytical_ik.py passes
at 0.0001 mm roundtrip) but the SO-101 5-DoF arm + tight wrist_flex limits
(±1.658 rad) make the runtime IK saturate or contort for many normal
targets, regardless of phi strategy (fixed, adaptive, per-phase).

**See** ``notes/eval2_ik_retrospective.md`` for the full audit.

**Replaced by** real teleop demos (sim or robot) — see ``sim/eval2.md``
section 0 for the new strategy.

What's still useful in this folder:
    - ``measure_link_lengths.py``     standalone calibration tool, may be
                                      reused for future kinematics work.
    - ``so101_link_lengths.json``     calibration data (L1, L2, L3, etc).
    - ``analytical_ik.py``            closed-form IK + FK, validated.
                                      Could be reused as ground-truth
                                      reference for verifying any future
                                      kinematics implementation.
    - ``test_analytical_ik.py``       FK<->IK roundtrip test (0.0001 mm).

Files that became dead code:
    - ``scripted_controller.py``      runtime controller (saturation issue).
    - ``run_scripted.py``             smoke test launcher.
    - ``generate_demos.py``           batch demo collection.
"""
