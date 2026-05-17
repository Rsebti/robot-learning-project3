# SO-101 URDF + meshes for HIL-SERL kinematics

`lerobot.model.kinematics.RobotKinematics` (placo-backed) needs a URDF + the
STL mesh files it references. Upstream lerobot does not ship a URDF for the
SO-ARM family — it must be vendored.

## What's here

- `urdf/so_arm101.urdf` — 13 KB, committed. Source of truth for kinematics.
- `urdf/assets/*.stl` — 13 STL files, ~15 MB total, **gitignored**.
  Regenerable from the upstream repo (see "Fetching meshes" below).

## Provenance

Both URDF and meshes are pulled verbatim from `MuammerBay/isaac_so_arm101`
(commit ID at the time of vendor: see git log on this directory).

URL: `https://github.com/MuammerBay/isaac_so_arm101/tree/main/src/isaac_so_arm101/robots/trs_so101/urdf`

License: TheRobotStudio open-source SO-ARM project (Apache-2.0 / CC-BY-SA;
check the upstream repo for the canonical license at the time of use).

## Fetching meshes (after a fresh clone)

```bash
TMP=/tmp/isaac_so_arm101_for_urdf
DEST=sim/hilserl/assets/so101/urdf/assets
rm -rf "$TMP"
git clone --depth=1 https://github.com/MuammerBay/isaac_so_arm101.git "$TMP"
mkdir -p "$DEST"
cp "$TMP/src/isaac_so_arm101/robots/trs_so101/urdf/assets/"*.stl "$DEST/"
rm -rf "$TMP"
```

## Sanity check the URDF loads

```python
from lerobot.model.kinematics import RobotKinematics
import numpy as np

rk = RobotKinematics(
    urdf_path="sim/hilserl/assets/so101/urdf/so_arm101.urdf",
    target_frame_name="gripper_frame_link",
)
print(rk.joint_names)
# ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
print(rk.forward_kinematics(np.zeros(6)))
# 4x4 homogeneous transform; home pose EE is ~(0.39, 0.0, 0.23) m
```

placo will print 3 self-collision warnings at load time
(`lower_arm_link_2 collides with wrist_link_1`, etc.). These are noise from
the upstream URDF's collision geometry overlap at rest pose. They do not
affect kinematics; ignore.
