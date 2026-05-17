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


cd /home/tommaso/Desktop/robot-learning-project3 &&
  /home/tommaso/isaac/isaac_so_arm101/.venv/bin/python -m sim.tommaso_eval2.scripts.view_scene
  --num_envs 1

CAMERA_INDEX=0 \ FOLLOWER_PORT=/dev/tty.usbmodem<correct_one> \
    LEADER_PORT=/dev/tty.usbmodem<the_other> \
    NUM_EPISODES=1 EPISODE_TIME_S=5 \
    POLICY_PATH=osammotg1/projet3-smolvla-eval2-v1-step38k \
    bash deploy/infer_smolvla.sh



### Eval 2 SmolVLA inference (final 50k checkpoint)

  # On the robot PC:
  cd "/Users/admin/Documents/ETH/M4/Robot Learning /Project S101/robot-learning-project3"
  git pull --ff-only origin tom-main

  # One-time fresh-Mac dep — SmolVLM's processor uses num2words to spell numerics
  # and lerobot 0.5.2 doesn't pull it in transitively. Skip if already installed.
  pip install num2words

  # Optional spot-test before plugging the arm in (catches a broken push).
  # NOTE: the naive `policy.select_action(raw_batch)` fails with
  # `KeyError: observation.language.tokens` because the policy expects the
  # preprocessor to have already tokenized `task` and renamed the camera key.
  # The version below goes through the saved preprocessor with the device
  # override that lerobot-record applies automatically at run time.
  python3 - <<'PY'
  import torch
  from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
  from lerobot.policies.factory import make_pre_post_processors
  REPO = "osammotg1/projet3-smolvla-eval2-v1"
  policy = SmolVLAPolicy.from_pretrained(REPO).eval()
  preprocessor, _ = make_pre_post_processors(
      policy_cfg=policy.config,
      pretrained_path=REPO,
      preprocessor_overrides={"device_processor": {"device": "mps"}},
      postprocessor_overrides={"device_processor": {"device": "mps"}},
  )
  batch = {
      "observation.state": torch.zeros(1, 6),
      "observation.images.wrist": torch.zeros(1, 3, 480, 640),
      "task": "Pick yellow block and place in bowl at (-15.5,29.5) cm",
  }
  with torch.no_grad():
      action = policy.select_action(preprocessor(batch))
  print("action.shape:", action.shape, "  finite:", action.isfinite().all().item())
  PY

  # Then, with arms + camera plugged in (CAMERA_INDEX=0 is correct on this Mac —
  # verified across teleop and ACT inference; not 1 like CLAUDE.md's Windows note):
  bash deploy/infer_smolvla.sh
  # → interactive picker asks for target cube color
  # → POLICY_PATH defaults to osammotg1/projet3-smolvla-eval2-v1 (final 50k checkpoint)

  # First short rollout — SmolVLA on MPS is ~5s/forward, so a 5s episode only
  # produces one inference; bump to 30s so chunked motion actually plays out:
  NUM_EPISODES=1 EPISODE_TIME_S=30 \
    POLICY_PATH=osammotg1/projet3-smolvla-eval2-v1 \
    bash deploy/infer_smolvla.sh

  # Full TA-spec 5-rollout eval (push the recorded eval dataset to HF for review):
  NUM_EPISODES=5 EPISODE_TIME_S=30 PUSH_TO_HUB=true \
    POLICY_PATH=osammotg1/projet3-smolvla-eval2-v1 \
    bash deploy/infer_smolvla.sh

  # Run provenance (for the laptop Claude):
  #   Repo:   osammotg1/projet3-smolvla-eval2-v1
  #   Steps:  50000  (final loss 0.007, grad_norm 0.13, lr 2.5e-6 — end of cosine decay)
  #   W&B:    https://wandb.ai/tom-gazzini-ethrc/projet3-smolvla/runs/1sxlvpgv
  #   Train rename: --rename_map='{"observation.images.wrist":"observation.images.camera1"}'
  #     infer_smolvla.sh routes the wrist view into camera1 directly via robot config,
  #     so no rename is needed at deploy time (kept in the script as a no-op safety net).
  #   Sanity-comparison checkpoint also on HF: osammotg1/projet3-smolvla-eval2-v1-step38k