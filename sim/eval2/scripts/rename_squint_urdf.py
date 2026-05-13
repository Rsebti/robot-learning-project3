"""Pre-process Squint URDF to match LeIsaac SO-101 link naming.

LeIsaac SO-101 USD uses bare link names (base, shoulder, upper_arm,
lower_arm, wrist, gripper, jaw). Squint URDF uses *_link suffix
(base_link, shoulder_link, etc.) and a separate moving_jaw_so101_v1_link.

To make a Squint-URDF-derived USD usable with our LeIsaac scaffold
(FrameTransformer, reward functions referencing body names), we copy the
URDF and rename the links to match LeIsaac.

Mapping:
    base_link                       → base
    shoulder_link                   → shoulder
    upper_arm_link                  → upper_arm
    lower_arm_link                  → lower_arm
    wrist_link                      → wrist
    gripper_link                    → gripper
    moving_jaw_so101_v1_link        → jaw
    (gripper_frame_link, finger1_tip, finger2_tip kept as-is — they're
     virtual frames LeIsaac doesn't reference)

Run AFTER this:
    python -m sim.eval2.scripts.convert_squint_urdf --urdf <path-to-renamed>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


RENAME_MAP = {
    "base_link": "base",
    "shoulder_link": "shoulder",
    "upper_arm_link": "upper_arm",
    "lower_arm_link": "lower_arm",
    "wrist_link": "wrist",
    "gripper_link": "gripper",
    "moving_jaw_so101_v1_link": "jaw",
}

SRC = Path(r"C:/Users/user/Desktop/MA2/robot-learning-project3/squint/envs/robot/so101.urdf")
DST = Path(r"C:/Users/user/Desktop/MA2/robot-learning-project3/squint/envs/robot/so101_renamed.urdf")


def main() -> None:
    if not SRC.exists():
        print(f"ERROR: source URDF not found: {SRC}", file=sys.stderr)
        sys.exit(1)
    text = SRC.read_text(encoding="utf-8")

    # Replace whole-word link references (avoid partial matches).
    # The renames apply in:
    #   <link name="X">
    #   <parent link="X"/>
    #   <child link="X"/>
    # We apply substitutions in REVERSE LENGTH order so longer names match
    # first (avoids "base_link" partially matching after "base" was renamed).
    pairs = sorted(RENAME_MAP.items(), key=lambda kv: -len(kv[0]))
    for old, new in pairs:
        text = re.sub(rf'\b{re.escape(old)}\b', new, text)

    DST.write_text(text, encoding="utf-8")
    print(f"[rename] Wrote renamed URDF: {DST}")
    print(f"[rename] Renames applied:")
    for old, new in pairs:
        print(f"    {old:32s} → {new}")


if __name__ == "__main__":
    main()
