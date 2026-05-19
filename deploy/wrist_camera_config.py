"""Shared wrist USB camera settings for deploy (native 1920x1080 + MJPG)."""
from __future__ import annotations

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from toolset.perception.cube_localization import WRIST_CAM_HEIGHT, WRIST_CAM_WIDTH


def make_wrist_opencv_camera_config(
    index_or_path: int | str = 0,
    *,
    width: int | None = None,
    height: int | None = None,
    fps: int = 30,
    fourcc: str | None = "MJPG",
    warmup_s: int = 3,
) -> OpenCVCameraConfig:
    """LeRobot OpenCV camera config for the wrist cam."""
    return OpenCVCameraConfig(
        index_or_path=index_or_path,
        fps=fps,
        width=width if width is not None else WRIST_CAM_WIDTH,
        height=height if height is not None else WRIST_CAM_HEIGHT,
        fourcc=fourcc,
        warmup_s=warmup_s,
    )
