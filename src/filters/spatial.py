"""Spatial anomaly filters: keypoint loss and geometric distortion."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.calibration.calibrator import BaselineStats
from src.filters.base import AbstractFilter
from src.calibration.spatial_metrics import compute_spatial_metrics
from src.models.events import AnomalyClass, AnomalyType, FilterResult
from src.models.frame import Frame


class SpatialFilter(AbstractFilter):
    """Detects structural anomalies within individual frames."""

    name = "spatial"

    def __init__(self) -> None:
        self._baseline: BaselineStats | None = None
        self._enabled = True
        self._orb = cv2.ORB_create(nfeatures=500)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def configure(self, baseline: BaselineStats, config: dict[str, Any]) -> None:
        self._baseline = baseline
        self._enabled = bool(config.get("enabled", True))

    def process(self, frame: Frame, processed_image: np.ndarray | None = None) -> list[FilterResult]:
        image = processed_image if processed_image is not None else frame.image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        neutral = [
            self._neutral_result(AnomalyType.KEYPOINT_LOSS),
            self._neutral_result(AnomalyType.GEOMETRIC_DISTORTION),
        ]

        if not self._enabled or self._baseline is None:
            return neutral

        ref_kp = self._baseline.reference_keypoints
        ref_desc = self._baseline.reference_descriptors
        if not ref_kp or ref_desc is None:
            return neutral

        metrics = compute_spatial_metrics(gray, ref_kp, ref_desc, self._orb, self._matcher)
        keypoints, descriptors = self._orb.detectAndCompute(gray, None)
        keypoints = list(keypoints or [])
        matches = self._matcher.match(ref_desc, descriptors) if descriptors is not None else []

        loss_threshold = self._baseline.keypoint_loss_threshold
        disp_threshold = self._baseline.displacement_threshold
        is_keypoint_loss = metrics.loss_ratio > loss_threshold
        is_distortion = metrics.displacement_percent > disp_threshold

        base_metadata = {
            "matched_keypoints": metrics.matched_keypoints,
            "reference_keypoints": metrics.reference_keypoints,
            "loss_ratio": metrics.loss_ratio,
            "displacement_percent": metrics.displacement_percent,
        }

        highlight = None
        if is_keypoint_loss or is_distortion:
            highlight = self._draw_keypoint_highlight(image, keypoints, matches, ref_kp)

        return [
            FilterResult(
                filter_name=self.name,
                anomaly_class=AnomalyClass.SPATIAL,
                anomaly_type=AnomalyType.KEYPOINT_LOSS,
                metric_value=metrics.loss_ratio,
                threshold=loss_threshold,
                is_anomaly=is_keypoint_loss,
                highlight_image=highlight if is_keypoint_loss else None,
                metadata=base_metadata,
            ),
            FilterResult(
                filter_name=self.name,
                anomaly_class=AnomalyClass.SPATIAL,
                anomaly_type=AnomalyType.GEOMETRIC_DISTORTION,
                metric_value=metrics.displacement_percent,
                threshold=disp_threshold,
                is_anomaly=is_distortion,
                highlight_image=highlight if is_distortion else None,
                metadata=base_metadata,
            ),
        ]

    def _draw_keypoint_highlight(
        self,
        image: np.ndarray,
        keypoints: list[cv2.KeyPoint],
        matches: list[cv2.DMatch],
        ref_kp: list[cv2.KeyPoint],
    ) -> np.ndarray:
        highlight = image.copy()
        matched_ref = {m.queryIdx for m in matches}
        for idx, kp in enumerate(ref_kp):
            x, y = int(kp.pt[0]), int(kp.pt[1])
            color = (0, 255, 0) if idx in matched_ref else (0, 0, 255)
            cv2.circle(highlight, (x, y), 6, color, 2)

        for kp in keypoints[:100]:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            cv2.circle(highlight, (x, y), 3, (255, 165, 0), 1)

        return highlight

    def _neutral_result(self, anomaly_type: AnomalyType) -> FilterResult:
        return FilterResult(
            filter_name=self.name,
            anomaly_class=AnomalyClass.SPATIAL,
            anomaly_type=anomaly_type,
            metric_value=0.0,
            threshold=0.0,
            is_anomaly=False,
        )
