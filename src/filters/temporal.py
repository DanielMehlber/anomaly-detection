"""Temporal anomaly filters: flicker and black screen detection."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.calibration.calibrator import BaselineStats
from src.filters.base import AbstractFilter
from src.models.events import AnomalyClass, AnomalyType, FilterResult
from src.models.frame import Frame


class TemporalFilter(AbstractFilter):
    """Detects time-based anomalies by comparing consecutive frames."""

    name = "temporal"

    def __init__(self) -> None:
        self._baseline: BaselineStats | None = None
        self._flicker_multiplier = 3.5
        self._black_screen_brightness = 10.0
        self._prev_brightness: float | None = None
        self._enabled = True

    def configure(self, baseline: BaselineStats, config: dict[str, Any]) -> None:
        self._baseline = baseline
        self._enabled = bool(config.get("enabled", True))
        self._flicker_multiplier = float(config.get("flicker_threshold_multiplier", 3.5))
        self._black_screen_brightness = float(config.get("black_screen_brightness", 10))

    def process(self, frame: Frame, processed_image: np.ndarray | None = None) -> list[FilterResult]:
        image = processed_image if processed_image is not None else frame.image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        brightness = float(np.mean(gray))

        if not self._enabled or self._baseline is None:
            return [
                self._neutral_result(AnomalyType.BLACK_SCREEN, brightness),
                self._neutral_result(AnomalyType.FLICKER, brightness),
            ]

        flicker_threshold = self._baseline.brightness_std * self._flicker_multiplier
        black_threshold = min(self._black_screen_brightness, self._baseline.black_screen_threshold)

        flicker_intensity = 0.0
        if self._prev_brightness is not None:
            flicker_intensity = abs(brightness - self._prev_brightness)
        self._prev_brightness = brightness

        is_black = brightness <= black_threshold
        black_result = FilterResult(
            filter_name=self.name,
            anomaly_class=AnomalyClass.TEMPORAL,
            anomaly_type=AnomalyType.BLACK_SCREEN,
            metric_value=brightness,
            threshold=black_threshold,
            is_anomaly=is_black,
            highlight_image=self._highlight_full_frame(image) if is_black else None,
            metadata={"brightness": brightness, "mode": "black_screen"},
        )

        is_flicker = (not is_black) and flicker_intensity > flicker_threshold
        flicker_result = FilterResult(
            filter_name=self.name,
            anomaly_class=AnomalyClass.TEMPORAL,
            anomaly_type=AnomalyType.FLICKER,
            metric_value=flicker_intensity,
            threshold=flicker_threshold,
            is_anomaly=is_flicker,
            highlight_image=self._highlight_full_frame(image, color=(0, 0, 255)) if is_flicker else None,
            metadata={"flicker_intensity": flicker_intensity, "brightness": brightness},
        )

        return [black_result, flicker_result]

    def _neutral_result(self, anomaly_type: AnomalyType, brightness: float) -> FilterResult:
        return FilterResult(
            filter_name=self.name,
            anomaly_class=AnomalyClass.TEMPORAL,
            anomaly_type=anomaly_type,
            metric_value=0.0,
            threshold=0.0,
            is_anomaly=False,
            metadata={"brightness": brightness},
        )

    @staticmethod
    def _highlight_full_frame(image: np.ndarray, color: tuple[int, int, int] = (0, 0, 200)) -> np.ndarray:
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1] - 1, overlay.shape[0] - 1), color, 8)
        return overlay
