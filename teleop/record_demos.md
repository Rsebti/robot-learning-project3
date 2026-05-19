# Record 20 demos for sanity check

## Prerequisites
- LeRobot installed
- Follower calibrated → `deploy/calibration/so101_follower.json` (see `deploy/calibration/README.md`)
- HF token logged in (`hf auth login`)
- SO-101 leader + follower both connected via USB
- Ports identified on the laptop (via `lerobot-find-port`):
  - **Follower → COM3**
  - **Leader → COM5**
  - (If ports change after replug, re-run `lerobot-find-port` and update.)

## Command

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=COM3 \
  --robot.id=so101_follower \
  --robot.cameras='{"wrist": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}}' \
  --teleop.type=so101_leader \
  --teleop.port=COM5 \
  --teleop.id=so101_leader \
  --display_data=true \
  --dataset.repo_id=Rsebti/projet3-demos-v1bis \
  --dataset.num_episodes=20 \
  --dataset.fps=30 \
  --dataset.episode_time_s=15 \
  --dataset.reset_time_s=10 \
  --dataset.single_task="Pick block and place in bowl" \
  --dataset.private=true \
  --dataset.push_to_hub=true
```

PowerShell-friendly one-liner (escape JSON quotes with `\"` after `--%` stop-parsing token; suppress wgpu warnings via `RUST_LOG`):

```powershell
$env:RUST_LOG = "error"
lerobot-record --% --robot.type=so101_follower --robot.port=COM3 --robot.id=so101_follower --robot.cameras={\"wrist\":{\"type\":\"opencv\",\"index_or_path\":1,\"width\":640,\"height\":480,\"fps\":30}} --teleop.type=so101_leader --teleop.port=COM5 --teleop.id=so101_leader --display_data=true --dataset.repo_id=Rsebti/projet3-demos-v1bis --dataset.num_episodes=20 --dataset.fps=30 --dataset.episode_time_s=15 --dataset.reset_time_s=10 --dataset.single_task="Pick block and place in bowl" --dataset.private=true --dataset.push_to_hub=true
```

## Replay verification (TA email step 2)

```bash
lerobot-replay \
  --robot.type=so101_follower \
  --robot.port=COM3 \
  --robot.id=so101_follower \
  --dataset.repo_id=Rsebti/projet3-demos-v1bis \
  --dataset.episode=0
```

Run on episodes 0, 5, 10, 15 to spot-check the recording quality.