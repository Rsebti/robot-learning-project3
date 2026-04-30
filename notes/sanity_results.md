# Sanity check — results

**Date** : 2026-04-28
**Status** : ✅ **PASSED — 5/5 episodes**

## Pipeline validated end-to-end

| Step | Result |
|---|---|
| 1. Record 19 quasi-identical demos via bilateral teleop | ✅ |
| 2. Replay verification (episodes 0/5/10/15) | ✅ |
| 3. Push dataset to HF Hub (`Rsebti/projet3-demos-v1bis`, public) | ✅ |
| 4. Train ACT policy on teammate's GPU → push to `Rsebti/projet3-act-sanity` | ✅ |
| 5. Deploy on real SO-101 follower (CPU inference on the laptop) | ✅ **5/5** |

## Setup notes

- Robot: SO-101 (Feetech servos), bilateral teleop with leader
- Recording machine: laptop (no GPU)
- Training machine: teammate's machine (GPU)
- Deploy machine: laptop (CPU inference, ACT chunking @ 100 actions)
- Dataset: 19 episodes (target was 20 — last episode lost to a lerobot keyboard-buffer
  edge case, but 19 was enough)
- Camera: wrist camera, OpenCV index 1, 640×480 @ 30 fps
- Ports on laptop: follower COM3, leader COM5

## Key quirks encountered (for the team)

- `lerobot 0.5.2` not on PyPI yet → pinned to `0.5.1` on the laptop
- PowerShell + lerobot CLI: use `--%` stop-parsing token + `\"` escaped JSON for the
  camera config; use `$env:RUST_LOG = "error"` to mute wgpu/Vulkan warnings
- Eval dataset name **must start with `eval_`** (lerobot enforces this when a policy
  is provided), e.g. `Rsebti/eval_projet3-sanity` (not `projet3-eval-sanity`)
- Between deploy episodes: leader stays plugged so it can be used to bring the
  follower back to home pose during the reset phase. During the silent encoding
  phase (~10-15 s, "Svt[info]:" dump), **do not press any key** — keypresses are
  buffered and will fire on the next episode start, instantly skipping it.
- LeRobot record auto-fails on a pre-existing dataset cache directory: each new
  deploy run is prefixed with a `Remove-Item` of the cache to start clean.

## Next

- Eval 1 (50 pts): pure BC, single block + bowl with **randomized positions**.
  Same pipeline as sanity, but with 50–100 demos covering more position
  variability.
- Eval 2 / 3 (RL with Isaac Lab) — separate workstream.
