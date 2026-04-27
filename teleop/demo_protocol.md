# Sanity check — demo recording protocol

## Goal
20 quasi-identical demonstrations of pick-and-place, to overfit a BC policy and validate the full pipeline.

## Setup before recording
- [ ] Robot base fixed (taped to table)
- [ ] Block position marked with tape (X1)
- [ ] Bowl position marked with tape (X2)
- [ ] Lighting: artificial only, curtains closed
- [ ] No clutter in camera FOV
- [ ] Wrist camera not touched after first demo
- [ ] Robot home pose defined

## During recording
- [ ] Robot starts at home pose every time
- [ ] Slow, smooth movements
- [ ] No corrections — if a demo is messy, discard and redo
- [ ] Target episode duration: 10-15s
- [ ] Same grasping point every time

## After recording
- [ ] Visual replay of all 20 demos (must look near-identical)
- [ ] Open-loop replay on robot for 4 demos (TA email step 2)
- [ ] Discard outliers if needed
- [ ] Verify dataset is on HF Hub