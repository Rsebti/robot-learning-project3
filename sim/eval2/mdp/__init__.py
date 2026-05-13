"""Custom MDP terms for our LeIsaac-based RL tasks.

We add reward functions that measure cube height **relative to the robot
base** rather than in absolute world frame. Required because LeIsaac's
table-with-cube scene puts the cube on a real table whose top sits several
centimeters above world z = 0; the canonical Isaac Lab Lift rewards
(``object_is_lifted`` / ``object_goal_distance``) compare the cube's world
z against a fixed `minimal_height` threshold, so they fire trivially as
soon as the cube spawns on the table.
"""
from .observations import (
    bowl_xyz_zero,                                   # V2.10 Phase C placeholder
    cube_to_goal_vector,                             # V2.12 relative vector obs
    ee_to_cube_vector,                               # V2.12 relative vector obs
    target_color_zero,                               # V2.10 Phase C placeholder
    wrist_image_features,
)
from .rewards import (
    cube_at_goal,                                    # V2.7 success bonus
    cube_at_goal_with_lift,                          # V2.18 success bonus + lift gate
    cube_dropped_float,                              # V2.9 drop penalty (RewTerm pair)
    cube_grasped,
    cube_grasped_strict,                             # V2.18 strict 6-condition grasp predicate
    cube_grasped_strict_float,                       # V2.18 float-cast for RewTerm
    cube_height_above_spawn,                         # V2.16 dense lift reward
    cube_lifted_above_base,
    cube_lifted_and_grasped,
    cube_to_goal_distance_above_base,
    cube_to_goal_distance_grasped_and_lifted,
    ee_to_cube_distance_clipped,                     # V2.18 linear distance penalty (clipped)
    goal_tracking_gated,                             # V2.18 goal tracking, grasp + lifted gated
    gripper_orientation_penalty,                     # V2.13 posture penalty (jaw-vs-palm Z)
    gripper_pointing_direction_penalty,              # V2.15 posture penalty (palm→jaw direction)
    hover_height_gaussian,                           # V2.18 Gaussian hover bonus
    jaw_below_cube_penalty,                          # V2.15 physics-breach prevention
    jaw_table_impact_penalty,                        # V2.18 ramped jaw-near-table penalty
    lift_height_gated,                               # V2.18 bounded lift × strict_grasp
    object_ee_distance_l2,                           # V2.10c linear distance penalty
    palm_to_jaw_orient_v218,                         # V2.18 palm→jaw direction (sign-fixed)
    palm_xy_above_cube,                              # V2.18 precision lateral alignment
    scoop_grasp_penalty,                             # V2.13 geometric anti-scoop
    # V2.19 — full rewrite per notes/v219_reward_search_claude.md
    close_no_contact_penalty_v219,
    cube_grasped_contact_v219,
    cube_grasped_contact_v219_float,
    finger_straddle_v219,
    goal_tracking_lift_gated_dual_scale,
    grasp_milestone_v219,
    joint_limit_penalty_v219,
    lift_clipped_gated,
    lift_milestone_v219,
    orientation_quat_tracking_v219,
    reach_coarse_v219,
    reach_fine_v219,
    success_terminal_v219,
    torque_penalty_v219,
    work_penalty_v219,
)
from .squint_rewards import (
    squint_is_above_bin,
    squint_not_lifted_penalty,
    squint_place_back_bonus,
    squint_place_final_dense,
    squint_place_success,
    squint_place_z_staged,
    squint_reach_dense,
)
from .terminations import cube_dropped, cube_reached_goal, ee_far_from_cube

__all__ = [
    "bowl_xyz_zero",                                 # V2.10 Phase C placeholder
    "cube_at_goal",                                  # V2.7 success bonus
    "cube_dropped",                                  # V2.8.5 failure cutoff (DoneTerm)
    "cube_dropped_float",                            # V2.9 drop penalty (RewTerm)
    "cube_at_goal_with_lift",                        # V2.18
    "cube_grasped",
    "cube_grasped_strict",                           # V2.18
    "cube_grasped_strict_float",                     # V2.18
    "cube_height_above_spawn",                       # V2.16 dense lift reward
    "cube_lifted_above_base",
    "cube_lifted_and_grasped",
    "ee_to_cube_distance_clipped",                   # V2.18
    "goal_tracking_gated",                           # V2.18
    "hover_height_gaussian",                         # V2.18
    "jaw_table_impact_penalty",                      # V2.18
    "lift_height_gated",                             # V2.18
    "palm_to_jaw_orient_v218",                       # V2.18
    "palm_xy_above_cube",                            # V2.18
    "cube_reached_goal",
    "cube_to_goal_distance_above_base",
    "cube_to_goal_distance_grasped_and_lifted",
    "cube_to_goal_vector",                           # V2.12 relative vector obs
    "ee_far_from_cube",                              # V2.12 wandering cutoff DoneTerm
    "ee_to_cube_vector",                             # V2.12 relative vector obs
    "gripper_orientation_penalty",                   # V2.13 posture penalty (jaw-vs-palm Z)
    "gripper_pointing_direction_penalty",            # V2.15 posture penalty (palm→jaw direction)
    "jaw_below_cube_penalty",                        # V2.15 physics-breach prevention
    "object_ee_distance_l2",                         # V2.10c linear distance penalty
    "scoop_grasp_penalty",                           # V2.13 geometric anti-scoop
    "squint_is_above_bin",                           # Squint Place — bin xy gate
    "squint_not_lifted_penalty",                     # Squint Lift/Place — fast-lift incentive
    "squint_place_back_bonus",                       # Squint Lift — return-to-rest while grasped
    "squint_place_final_dense",                      # Squint Place — cube→goal dense
    "squint_place_success",                          # Squint Place — terminal bonus
    "squint_place_z_staged",                         # Squint Place — far(hover) vs close(descent)
    "squint_reach_dense",                            # Squint Lift/Place — tanh reach
    "target_color_zero",                             # V2.10 Phase C placeholder
    "wrist_image_features",
]
