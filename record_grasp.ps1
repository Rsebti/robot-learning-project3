Remove-Item -Recurse -Force "C:\Users\hugod\.cache\huggingface\lerobot\hudela390\projet3-demos-grasp-2026-05-20" -ErrorAction SilentlyContinue

cd C:\Users\hugod\project3
$env:PYTHONIOENCODING              = "utf-8"
$env:OPENCV_VIDEOIO_PRIORITY_MSMF  = "0"
$env:OPENCV_VIDEOIO_PRIORITY_DSHOW = "1000"

lerobot-record --% --robot.type=so101_follower --robot.port=COM3 --robot.id=so101_follower --robot.cameras={\"wrist\":{\"type\":\"opencv\",\"index_or_path\":0,\"width\":640,\"height\":480,\"fps\":30}} --teleop.type=so101_leader --teleop.port=COM4 --teleop.id=so101_leader --display_data=false --dataset.repo_id=hudela390/projet3-demos-grasp-2026-05-20 --dataset.num_episodes=10 --dataset.fps=30 --dataset.episode_time_s=15 --dataset.reset_time_s=8 --dataset.single_task="grasp cube" --dataset.private=true --dataset.push_to_hub=true
