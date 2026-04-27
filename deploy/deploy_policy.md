# Deploy trained BC policy on the real SO-101

## Goal
Run the ACT checkpoint on the real follower arm and check it reproduces the
pick-and-place motion from the demos (TA email step 4).

## Prerequisites
- Training finished, checkpoint at `outputs/train/act_sanity/checkpoints/last/pretrained_model/`
  (or pulled from HF: `Rsebti/projet3-act-sanity`)
- SO-101 follower connected, USB port known (`<FOLLOWER_PORT>`)
- Wrist camera connected (same index as during recording)
- Block + bowl placed at the **exact same tape positions** as the demos
- Same lighting as during recording

## Command — eval on real robot

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=<FOLLOWER_PORT> \
  --robot.id=so101_follower \
  --robot.cameras='{"wrist": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}' \
  --display_data=true \
  --policy.path=outputs/train/act_sanity/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --dataset.repo_id=Rsebti/projet3-eval-sanity \
  --dataset.num_episodes=5 \
  --dataset.fps=30 \
  --dataset.episode_time_s=15 \
  --dataset.reset_time_s=10 \
  --dataset.single_task="Pick block and place in bowl" \
  --dataset.private=true \
  --dataset.push_to_hub=false
```

Notes:
- No `--teleop.*` flags → the policy controls the follower (autonomous).
- We still use `lerobot-record` because it logs eval episodes; set
  `push_to_hub=false` for sanity (just want to watch the robot move).
- Replace `<FOLLOWER_PORT>` with the same port as in recording (e.g. `/dev/ttyACM0` on Linux, `COM5` on Windows).
- If the checkpoint is on HF, use `--policy.path=Rsebti/projet3-act-sanity` instead of the local path.

## Pre-flight checklist
- [ ] Block at tape mark X1
- [ ] Bowl at tape mark X2
- [ ] Robot at home pose
- [ ] Camera index matches recording
- [ ] Emergency stop reachable (hand on the e-stop / power)

## Pass criteria (sanity)
- Robot moves smoothly to the block (no jitter, no servo error).
- Closes gripper on the block.
- Lifts and places into the bowl.
- 3 / 5 episodes succeed → pipeline validated → move to Eval 1.

## If it fails
- Robot drifts immediately → camera index wrong, or block/bowl moved.
- Robot freezes → policy outputting same action → undertrained, increase `--steps`.
- Servo overcurrent → motion too fast, the demos might have been too jerky.
