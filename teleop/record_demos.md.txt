# Record 20 demos for sanity check

## Prerequisites
- LeRobot installed
- HF token logged in (`hf auth login`)
- SO-101 leader + follower both connected via USB
- Identify ports:
  - Linux: `ls /dev/tty*` (probably /dev/ttyACM0, /dev/ttyACM1)
  - Windows: Device Manager → COM ports

## Command

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=<FOLLOWER_PORT> \
  --robot.id=so101_follower \
  --robot.cameras='{"wrist": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}' \
  --teleop.type=so101_leader \
  --teleop.port=<LEADER_PORT> \
  --teleop.id=so101_leader \
  --display_data=true \
  --dataset.repo_id=Rsebti/projet3-demos-v1 \
  --dataset.num_episodes=20 \
  --dataset.fps=30 \
  --dataset.episode_time_s=15 \
  --dataset.reset_time_s=10 \
  --dataset.single_task="Pick block and place in bowl" \
  --dataset.private=true \
  --dataset.push_to_hub=true
```

## Replay verification (TA email step 2)

```bash
lerobot-replay \
  --robot.type=so101_follower \
  --robot.port=<FOLLOWER_PORT> \
  --robot.id=so101_follower \
  --dataset.repo_id=Rsebti/projet3-demos-v1 \
  --dataset.episode=0
```

Run on episodes 0, 5, 10, 15 to spot-check the recording quality.