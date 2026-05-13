0. Record more demos, appending to the existing training dataset

  # Target dataset: osammotg1/projet3-eval1-bowl1-v1 (currently 40 episodes,
  # 8 per color × yellow/blue/green/violet/red). This run adds 20 NEW demos:
  #   - 2 each for yellow / blue / green / violet / red  (10 episodes)
  #   - 10 for orange (a new color, not in the original dataset)
  # Bowl stays at position 1 (16 cm right, 32 cm forward).

  hf auth whoami   # should print osammotg1; if not: hf auth login

  RESUME=true \
  COLORS=yellow,blue,green,violet,red,orange \
  EPISODES_PER_COLOR=2 \
  EPISODES_ORANGE=10 \
  FOLLOWER_PORT=/dev/tty.usbmodem5B141129871 \
  LEADER_PORT=/dev/tty.usbmodem5B141128171 \
  CAM_INDEX=0 \
    bash teleop/record_eval1_bowl1.sh

  # After this run the dataset will have 40 + 5*2 + 10 = 60 episodes.
  # Watch the banner before pressing ENTER: it must say
  # "Mode: RESUME (appending to existing dataset, no wipe)".
  # Banner should also show:
  #   Per color: yellow=2 blue=2 green=2 violet=2 red=2 orange=10
  #   TOTAL:    20 demos

  ----

1. Auth + pre-download both checkpoints

  hf auth whoami   # should print osammotg1; if not: hf auth login
  hf download osammotg1/projet3-act-eval1-v1-dark-noise
  hf download osammotg1/projet3-act-eval1-v1-dark-shadow

  2. Run inference — dark_noise

  FOLLOWER_PORT=/dev/tty.usbmodem5B141129871 \
  POLICY_PATH=osammotg1/projet3-act-eval1-v1-dark-noise \
  POLICY_DEVICE=mps \
  EVAL_REPO_ID=osammotg1/eval_projet3-dark-noise \
  NUM_EPISODES=10 \
  EPISODE_TIME_S=20 \
  RESET_TIME_S=8 \
  SINGLE_TASK="Pick yellow block and place in bowl at (16,32) cm [bowl_pos=1]" \
  PUSH_TO_HUB=true \
    bash deploy/infer.sh



  3. Run inference — dark_shadow

  FOLLOWER_PORT=/dev/tty.usbmodem5B141129871 \
  POLICY_PATH=osammotg1/projet3-act-eval1-v1-dark-shadow \
  POLICY_DEVICE=mps \
  EVAL_REPO_ID=osammotg1/eval_projet3-dark-shadow \
  NUM_EPISODES=10 \
  EPISODE_TIME_S=20 \
  RESET_TIME_S=8 \
  SINGLE_TASK="Pick yellow block and place in bowl at (16,32) cm [bowl_pos=1]" \
  PUSH_TO_HUB=true \
    bash deploy/infer.sh


RESUME=true \
  COLORS=yellow,blue,green,violet,red,orange \
  EPISODES_PER_COLOR=2 \
  EPISODES_ORANGE=10 \
  FOLLOWER_PORT=/dev/tty.usbmodem5B141129871 \
  LEADER_PORT=/dev/tty.usbmodem5B141128171 \
  CAM_INDEX=0 \
    bash teleop/record_eval1_bowl1.sh



### Eval 2 inference
  FOLLOWER_PORT=/dev/tty.usbmodem5B141129871 \
  POLICY_PATH=osammotg1/projet3-act-eval2-dark-noise-100k \
  POLICY_DEVICE=mps \
  EVAL_REPO_ID=osammotg1/eval_projet3-eval2-dark-noise-100k \
  NUM_EPISODES=8 \
  EPISODE_TIME_S=30 \ 
  RESET_TIME_S=10 \
  SINGLE_TASK="Pick <TARGET_COLOR> block and place in bowl at (-15.5,29.5) cm" \
  PUSH_TO_HUB=true \
    bash deploy/infer.sh