"""
Frame data model passed through the analysis pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Frame:
    """
    Read-only frame container with environment metadata.

    ``timestamp_seconds`` is the environment factor for file-based inputs
    (video runtime). Live camera providers can populate ``metadata`` with
    temperature, humidity, or other chamber values.
    """

    image: np.ndarray
    timestamp_seconds: float
    frame_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def grayscale(self) -> np.ndarray:
        """Return a single-channel version of ``image`` for analysis."""
        import cv2

        if len(self.image.shape) == 2:
            return self.image
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
