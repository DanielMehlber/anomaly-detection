"""Optional preprocessing filter for global brightness normalization."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.calibration.calibrator import BaselineStats
from src.filters.base import AbstractFilter
from src.models.events import AnomalyClass, AnomalyType, FilterResult
from src.models.frame import Frame


class PreprocessingFilter(AbstractFilter):
    """Normalizes brightness to reduce slow environmental drift."""

    name = "preprocessing"

    def __init__(self) -> None:
        self._enabled = False
        self._normalize = True
        self._target_brightness = 128.0

    def configure(self, baseline: BaselineStats, config: dict[str, Any]) -> None:
        self._enabled = bool(config.get("enabled", False))
        self._normalize = bool(config.get("brightness_normalization", True))
        self._target_brightness = baseline.mean_brightness

    def process(self, frame: Frame, processed_image: np.ndarray | None = None) -> FilterResult:
        return FilterResult(
            filter_name=self.name,
            anomaly_class=AnomalyClass.TEMPORAL,
            anomaly_type=AnomalyType.FLICKER,
            metric_value=0.0,
            threshold=0.0,
            is_anomaly=False,
        )

    def apply(self, frame: Frame) -> np.ndarray:
        """Return a preprocessed copy of the frame image."""
        if not self._enabled or not self._normalize:
            return frame.image.copy()

        gray = frame.grayscale
        current = float(np.mean(gray))
        if current < 1e-6:
            return frame.image.copy()

        scale = self._target_brightness / current
        adjusted = np.clip(frame.image.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        return adjusted
