"""Self-calibration from the first stable segment of the data stream.

The calibrator learns what "normal" looks like for the current test setup:
brightness statistics for temporal filters and ORB keypoint stability for
spatial filters. No hardcoded thresholds are used downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.calibration.control_elements import (
    _indices_for_positions,
    control_element_loss_samples,
    select_control_elements,
)
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
    mean_flicker_intensity: float = 0.0
    flicker_intensity_std: float = 0.5
    control_element_positions: list[tuple[float, float]] = field(default_factory=list)
    control_element_indices: list[int] = field(default_factory=list)
    mean_control_missing_ratio: float = 0.0
    control_missing_std: float = 0.01

    @property
    def flicker_threshold(self) -> float:
        """Maximum frame-to-frame brightness jump still considered normal."""
        return self.mean_flicker_intensity + self.flicker_intensity_std * self.threshold_multiplier

    @property
    def black_screen_threshold(self) -> float:
        """Mean brightness below which the frame is treated as black."""
        return max(0.0, self.mean_brightness - self.brightness_std * self.threshold_multiplier)

    @property
    def keypoint_loss_threshold(self) -> float:
        """Maximum fraction of missing reference keypoints still considered normal."""
        return min(
            0.95,
            self.mean_keypoint_loss + self.keypoint_loss_std * self.threshold_multiplier,
        )

    @property
    def displacement_threshold(self) -> float:
        """Maximum average keypoint displacement (percent of frame diagonal)."""
        return self.mean_displacement_percent + self.displacement_std * self.threshold_multiplier

    @property
    def control_missing_threshold(self) -> float:
        """Maximum fraction of registered control elements that may be absent."""
        return min(
            0.95,
            self.mean_control_missing_ratio + self.control_missing_std * self.threshold_multiplier,
        )


class Calibrator:
    """Collect calibration frames and compute dynamic tolerance limits."""

    def __init__(self, duration_seconds: float, threshold_multiplier: float) -> None:
        """
        Args:
            duration_seconds: Length of the initial stable window (usually 60s).
            threshold_multiplier: Number of standard deviations used for limits.
        """
        self.duration_seconds = duration_seconds
        self.threshold_multiplier = threshold_multiplier
        self._brightness_values: list[float] = []
        self._spatial_noise_values: list[float] = []
        self._keypoint_loss_samples: list[float] = []
        self._displacement_samples: list[float] = []
        self._flicker_intensity_samples: list[float] = []
        self._previous_brightness: float | None = None
        self._reference_frame: np.ndarray | None = None
        self._reference_keypoints: list[cv2.KeyPoint] = []
        self._reference_descriptors: np.ndarray | None = None
        self._per_frame_matched_indices: list[set[int]] = []
        self._per_frame_keypoints: list[list[cv2.KeyPoint]] = []
        self._orb = cv2.ORB_create(nfeatures=500)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def is_calibration_frame(self, frame: Frame) -> bool:
        """Return True while ``frame`` is still inside the calibration window."""
        return frame.timestamp_seconds < self.duration_seconds

    def add_frame(self, frame: Frame) -> None:
        """Record statistics from one calibration frame."""
        gray = frame.grayscale
        brightness = float(np.mean(gray))
        self._brightness_values.append(brightness)
        self._spatial_noise_values.append(float(np.std(gray)))

        if self._previous_brightness is not None:
            self._flicker_intensity_samples.append(abs(brightness - self._previous_brightness))
        self._previous_brightness = brightness

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

            keypoints, descriptors = self._orb.detectAndCompute(gray, None)
            keypoints = list(keypoints or [])
            if descriptors is not None:
                matches = self._matcher.match(self._reference_descriptors, descriptors)
                self._per_frame_matched_indices.append({match.queryIdx for match in matches})
                self._per_frame_keypoints.append(keypoints)
            else:
                self._per_frame_matched_indices.append(set())
                self._per_frame_keypoints.append(keypoints)

    def finalize(self) -> BaselineStats:
        """Compute baseline statistics after the calibration window ends."""
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

        if self._flicker_intensity_samples:
            mean_flicker = float(np.mean(self._flicker_intensity_samples))
            flicker_std = float(np.std(self._flicker_intensity_samples))
        else:
            mean_flicker = 0.0
            flicker_std = 0.5

        control_positions = select_control_elements(
            self._reference_keypoints,
            self._per_frame_matched_indices,
        )
        control_indices = _indices_for_positions(
            self._reference_keypoints,
            control_positions,
        )
        control_loss_samples = control_element_loss_samples(
            control_positions,
            self._per_frame_keypoints,
        )
        if control_loss_samples:
            mean_control_missing = float(np.mean(control_loss_samples))
            control_missing_std = float(np.std(control_loss_samples))
        else:
            mean_control_missing = 0.0
            control_missing_std = 0.05

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
            mean_flicker_intensity=mean_flicker,
            flicker_intensity_std=max(flicker_std, 0.5),
            control_element_positions=control_positions,
            control_element_indices=control_indices,
            mean_control_missing_ratio=mean_control_missing,
            control_missing_std=max(control_missing_std, 0.01),
        )
