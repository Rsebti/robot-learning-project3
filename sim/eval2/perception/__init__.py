"""Perception module for Eval 2.

Maps a wrist-camera RGB image (84x84x3) to 2D pixel coordinates of the
red and blue blocks. The xyz position in the robot frame is recovered
afterwards in inference.py via camera intrinsics + the known table plane.

Pipeline at deploy time:
    image (camera) -> CNN -> (x_red_px, y_red_px, x_blue_px, y_blue_px)
                          -> ray-plane intersection -> (xyz_red, xyz_blue) in robot frame
                          -> goal-conditioned policy

Files:
    model.py            ColorBlockCNN (PyTorch)
    dataset.py          BlockPositionDataset (loads sim-rendered training data)
    capture_dataset.py  one-off Isaac Sim script that generates training data
    train.py            training loop (MSE on 4 targets)
    inference.py        loads a checkpoint, runs prediction, post-processes to xyz
"""
