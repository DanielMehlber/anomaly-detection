"""Self-calibration from the first stable segment of the data stream."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.calibration.spatial_metrics import compute_spatial_metrics
from src.models.frame import Frame


@dataclass
class BaselineStats:
    """Statistical baseline derived from the calibration window."""

    mean_brightness: float
    brightness_std: float
    spatial_noise_std: float
    reference_keypoints: list[cv2.KeyPoint] = field(default_factory=list)
    reference_descriptors: np.ndarray | None = None
    reference_shape: tuple[int, int] | None = None
    threshold_multiplier: float = 3.5
    mean_keypoint_loss: float = 0.0
    keypoint_loss_std: float = 0.01
    mean_displacement_percent: float = 0.0
    displacement_std: float = 0.01

    @property
    def flicker_threshold(self) -> float:
        return self.brightness_std * self.threshold_multiplier

    @property
    def black_screen_threshold(self) -> float:
        return max(0.0, self.mean_brightness - self.brightness_std * self.threshold_multiplier)

    @property
    def keypoint_loss_threshold(self) -> float:
        return min(
            0.95,
            self.mean_keypoint_loss + self.keypoint_loss_std * self.threshold_multiplier,
        )

    @property
    def displacement_threshold(self) -> float:
        return self.mean_displacement_percent + self.displacement_std * self.threshold_multiplier


class Calibrator:
    """Collects calibration frames and computes dynamic tolerance limits."""

    def __init__(self, duration_seconds: float, threshold_multiplier: float) -> None:
        self.duration_seconds = duration_seconds
        self.threshold_multiplier = threshold_multiplier
        self._brightness_values: list[float] = []
        self._spatial_noise_values: list[float] = []
        self._keypoint_loss_samples: list[float] = []
        self._displacement_samples: list[float] = []
        self._reference_frame: np.ndarray | None = None
        self._reference_keypoints: list[cv2.KeyPoint] = []
        self._reference_descriptors: np.ndarray | None = None
        self._orb = cv2.ORB_create(nfeatures=500)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def is_calibration_frame(self, frame: Frame) -> bool:
        return frame.timestamp_seconds < self.duration_seconds

    def add_frame(self, frame: Frame) -> None:
        gray = frame.grayscale
        self._brightness_values.append(float(np.mean(gray)))
        self._spatial_noise_values.append(float(np.std(gray)))

        if self._reference_frame is None:
            self._reference_frame = gray.copy()
            keypoints, descriptors = self._orb.detectAndCompute(gray, None)
            self._reference_keypoints = list(keypoints or [])
            self._reference_descriptors = descriptors
            return

        if self._reference_descriptors is not None and self._reference_keypoints:
            metrics = compute_spatial_metrics(
                gray,
                self._reference_keypoints,
                self._reference_descriptors,
                self._orb,
                self._matcher,
            )
            self._keypoint_loss_samples.append(metrics.loss_ratio)
            self._displacement_samples.append(metrics.displacement_percent)

    def finalize(self) -> BaselineStats:
        if not self._brightness_values:
            raise RuntimeError("No calibration frames collected.")

        mean_brightness = float(np.mean(self._brightness_values))
        brightness_std = float(np.std(self._brightness_values))
        spatial_noise_std = float(np.mean(self._spatial_noise_values))

        if self._keypoint_loss_samples:
            mean_keypoint_loss = float(np.mean(self._keypoint_loss_samples))
            keypoint_loss_std = float(np.std(self._keypoint_loss_samples))
            mean_displacement = float(np.mean(self._displacement_samples))
            displacement_std = float(np.std(self._displacement_samples))
        else:
            mean_keypoint_loss = 0.0
            keypoint_loss_std = 0.05
            mean_displacement = 0.0
            displacement_std = 0.05

        return BaselineStats(
            mean_brightness=mean_brightness,
            brightness_std=max(brightness_std, 1e-6),
            spatial_noise_std=max(spatial_noise_std, 1e-6),
            reference_keypoints=self._reference_keypoints,
            reference_descriptors=self._reference_descriptors,
            reference_shape=self._reference_frame.shape if self._reference_frame is not None else None,
            threshold_multiplier=self.threshold_multiplier,
            mean_keypoint_loss=mean_keypoint_loss,
            keypoint_loss_std=max(keypoint_loss_std, 0.01),
            mean_displacement_percent=mean_displacement,
            displacement_std=max(displacement_std, 0.01),
        )
