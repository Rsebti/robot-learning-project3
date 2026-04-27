# robot-learning-project3





Pick-and-place tasks on SO-101 using BC + RL.



\## Structure

\- `teleop/` — scripts for recording demonstrations

\- `train/` — training configs and scripts (BC, RL)

\- `deploy/` — deployment scripts on real SO-101

\- `notes/` — observations, debug logs, weekly updates



\## Pipeline

1\. Record demos with SO-101 → LeRobot v3 format

2\. Push dataset to HuggingFace Hub

3\. Train policy on Brev (H100 instance)

4\. Deploy checkpoint on real robot



\## Status

\- \[ ] Sanity check (20 demos + BC + deploy)

\- \[ ] Eval 1 (BC, randomized positions)

\- \[ ] Eval 2 (RL, color-conditioned)

\- \[ ] Eval 3 (sequential pick-and-place)

