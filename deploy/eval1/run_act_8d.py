"""
run_act_8d.py - thin wrapper around deploy/infer_eval1_act_nocube.py with
Eval-1-friendly defaults pre-injected. Zero hardcoded fixes; every default
is overridable from the command line.

Defaults set here:
    --policy_path        hudela390/projet3-act-eval1-v1-no-cube  (current 8-D weights)
    --target_color       red
    --bowl_x / --bowl_y  0.16 / 0.32                              (Eval-1 bowl, user frame)
    --follower_port      COM3
    --camera_index       1
    --episode_time_s     15
    --num_episodes       1
    --no-home                                                     (start from current pose)

Pass any flag to override; for example:
    python deploy/eval1/run_act_8d.py --target_color yellow
    python deploy/eval1/run_act_8d.py --bowl_x 0.10 --bowl_y 0.30
    python deploy/eval1/run_act_8d.py --home                    # re-enable homing
    python deploy/eval1/run_act_8d.py --policy_path some/other-8d-repo
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEPLOY = HERE.parent

_argv = sys.argv[1:]


def _has(*flags: str) -> bool:
    return any(flag in _argv for flag in flags)


injected: list[str] = []
if not _has("--target_color"):
    injected += ["--target_color", "red"]
if not _has("--bowl_x"):
    injected += ["--bowl_x", "0.16"]
if not _has("--bowl_y"):
    injected += ["--bowl_y", "0.32"]
if not _has("--follower_port"):
    injected += ["--follower_port", "COM3"]
if not _has("--camera_index"):
    injected += ["--camera_index", "1"]
if not _has("--episode_time_s"):
    injected += ["--episode_time_s", "15"]
if not _has("--num_episodes"):
    injected += ["--num_episodes", "1"]
if not _has("--home", "--no-home"):
    injected += ["--no-home"]

sys.argv = [sys.argv[0]] + injected + _argv

# Hand off to the real 8-D ACT script. No monkey-patching here.
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))
from infer_eval1_act_nocube import main  # noqa: E402

if __name__ == "__main__":
    main()
