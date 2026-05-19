"""Direct OpenCV wrist camera (same pattern as manual index probe scripts)."""
from __future__ import annotations

import cv2
import numpy as np

_OriginalVideoCapture = cv2.VideoCapture


class _DShowVideoCapture(_OriginalVideoCapture):
    def __init__(self, *args, **kwargs):
        if len(args) >= 2 and isinstance(args[0], int) and args[1] in (
            cv2.CAP_ANY, cv2.CAP_MSMF
        ):
            args = (args[0], cv2.CAP_DSHOW)
        elif len(args) == 1 and isinstance(args[0], int):
            args = (args[0], cv2.CAP_DSHOW)
        super().__init__(*args, **kwargs)


cv2.VideoCapture = _DShowVideoCapture


class DirectWristCamera:
    """640x480 DSHOW capture — avoids LeRobot camera thread on the same index."""

    def __init__(
        self,
        index: int = 0,
        width: int = 640,
        height: int = 480,
        warmup_reads: int = 8,
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self._cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        for _ in range(warmup_reads):
            self._cap.read()

    def read_bgr(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return np.asarray(frame, dtype=np.uint8)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> DirectWristCamera:
        return self

    def __exit__(self, *args) -> None:
        self.release()
