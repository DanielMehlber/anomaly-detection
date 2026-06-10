"""
Shared ORB matching utilities for spatial filters.
"""

from __future__ import annotations

from typing import Any

import cv2

from src.calibration.calibrator import BaselineStats
from src.calibration.spatial_metrics import compute_spatial_metrics
from src.models.frame import Frame


class SpatialMatcher:
    """Reusable ORB matcher against the calibration reference frame."""

    def __init__(self) -> None:
        self._baseline: BaselineStats | None = None
        self._enabled = True
        self._orb = cv2.ORB_create(nfeatures=500)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def configure(self, baseline: BaselineStats, config: dict[str, Any]) -> None:
        """Store calibrated reference features and the enabled flag."""
        self._baseline = baseline
        self._enabled = bool(config.get("enabled", True))

    @property
    def enabled(self) -> bool:
        return self._enabled and self._baseline is not None

    def analyze(self, frame: Frame, processed_image):
        """Return grayscale frame, metrics, keypoints, and descriptor matches."""
        image = processed_image if processed_image is not None else frame.image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        if not self.enabled or self._baseline is None:
            return None

        ref_kp = self._baseline.reference_keypoints
        ref_desc = self._baseline.reference_descriptors
        if not ref_kp or ref_desc is None:
            return None

        metrics = compute_spatial_metrics(gray, ref_kp, ref_desc, self._orb, self._matcher)
        keypoints, descriptors = self._orb.detectAndCompute(gray, None)
        keypoints = list(keypoints or [])
        matches = self._matcher.match(ref_desc, descriptors) if descriptors is not None else []

        return {
            "image": image,
            "gray": gray,
            "metrics": metrics,
            "ref_kp": ref_kp,
            "keypoints": keypoints,
            "matches": matches,
            "loss_threshold": self._baseline.keypoint_loss_threshold,
            "disp_threshold": self._baseline.displacement_threshold,
        }
