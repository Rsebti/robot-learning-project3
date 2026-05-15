record teleop for eval 3 

this is eval 3
Eval 3 (50pts): Sequential Multi-Step Pick-and-Place Four blocks of distinct colors are placed in the workspace. The policy is given a sequence of three pick-and-place goals, each specifying: ●  A target color ●  A target bowl position The bowl positions remain fixed within each rollout. The policy must execute the sequence in order. Additionally: ●  Policy switching and/or switching perception modules (e.g., color filters) is allowed ●  Interaction with non-target blocks (e.g., pushing or rearranging) is permitted and encouraged if it facilitates task completion ●  5 rollouts will be performed Scoring per rollout: ●  4, 4, and 2 pts for completing each step ●  Partial credit is awarded for partial completion A step is considered successful if the correct block is placed inside the corresponding bowl and released. This task must involve reinforcement learning.  Expert data collected from teleoperation is encouraged to use for training efficiency. 


so we need to record eval 3 episodes with as input :
-the goal position
-3 colors in the order to take them
-maybe the 4th color

we need 100 teleop