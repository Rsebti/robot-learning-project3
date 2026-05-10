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
    cube_dropped_float,                              # V2.9 drop penalty (RewTerm pair)
    cube_grasped,
    cube_lifted_above_base,
    cube_lifted_and_grasped,
    cube_to_goal_distance_above_base,
    cube_to_goal_distance_grasped_and_lifted,
    object_ee_distance_l2,                           # V2.10c linear distance penalty
)
from .terminations import cube_dropped, cube_reached_goal, ee_far_from_cube

__all__ = [
    "bowl_xyz_zero",                                 # V2.10 Phase C placeholder
    "cube_at_goal",                                  # V2.7 success bonus
    "cube_dropped",                                  # V2.8.5 failure cutoff (DoneTerm)
    "cube_dropped_float",                            # V2.9 drop penalty (RewTerm)
    "cube_grasped",
    "cube_lifted_above_base",
    "cube_lifted_and_grasped",
    "cube_reached_goal",
    "cube_to_goal_distance_above_base",
    "cube_to_goal_distance_grasped_and_lifted",
    "cube_to_goal_vector",                           # V2.12 relative vector obs
    "ee_far_from_cube",                              # V2.12 wandering cutoff DoneTerm
    "ee_to_cube_vector",                             # V2.12 relative vector obs
    "object_ee_distance_l2",                         # V2.10c linear distance penalty
    "target_color_zero",                             # V2.10 Phase C placeholder
    "wrist_image_features",
]
