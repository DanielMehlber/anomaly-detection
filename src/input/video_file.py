"""Video file input provider for local .mp4 and .avi files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2

from src.input.abstract import AbstractInputProvider
from src.models.frame import Frame


class VideoFileInputProvider(AbstractInputProvider):
    """Reads local video files; video runtime is the environment factor."""

    SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

    def __init__(self, video_path: str) -> None:
        self.video_path = Path(video_path)
        self._capture: cv2.VideoCapture | None = None
        self._fps: float = 0.0

    def open(self) -> None:
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        suffix = self.video_path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported video format '{suffix}'. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
            )

        self._capture = cv2.VideoCapture(str(self.video_path))
        if not self._capture.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        self._fps = self._capture.get(cv2.CAP_PROP_FPS) or 30.0

    def frames(self) -> Iterator[Frame]:
        if self._capture is None:
            raise RuntimeError("Input provider is not open. Call open() first.")

        frame_index = 0
        while True:
            success, image = self._capture.read()
            if not success:
                break

            timestamp_seconds = frame_index / self._fps
            yield Frame(
                image=image,
                timestamp_seconds=timestamp_seconds,
                frame_index=frame_index,
                metadata={"source": str(self.video_path), "fps": self._fps},
            )
            frame_index += 1

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
