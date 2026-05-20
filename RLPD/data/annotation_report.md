# Grasp FK annotations — `Rsebti/projet3_demos_v1`

- Episodes: **39**
- FPS: **30.0**
- FK target: **gripper_tip**
- Cube placement: **grasp_xy_table_z**

## Summary

- In workspace: **39/39**
- cube_z err vs table (mm): median **0.0**, max **0.0**
- Grasp pick reasons: `{'static_hold_most_closed': 39}`

## Columns (CSV)

`cube_xyz_user_m` / `cube_pose_urdf_world` → spawn cube in Isaac replay.
`grasp_xyz_user_m` → FK between jaws; cube XY matches, Z = table + offset.
