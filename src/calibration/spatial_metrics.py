"""Shared spatial metric computation for calibration and filtering."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SpatialMetrics:
    loss_ratio: float
    displacement_percent: float
    matched_keypoints: int
    reference_keypoints: int


def compute_spatial_metrics(
    gray: np.ndarray,
    reference_keypoints: list[cv2.KeyPoint],
    reference_descriptors: np.ndarray,
    orb: cv2.ORB,
    matcher: cv2.BFMatcher,
) -> SpatialMetrics:
    """Measure keypoint loss and mean displacement relative to the reference frame."""
    ref_count = max(len(reference_keypoints), 1)
    frame_diagonal = float(np.hypot(gray.shape[1], gray.shape[0]))

    keypoints, descriptors = orb.detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) == 0:
        return SpatialMetrics(
            loss_ratio=1.0,
            displacement_percent=100.0,
            matched_keypoints=0,
            reference_keypoints=ref_count,
        )

    matches = matcher.match(reference_descriptors, descriptors)
    matched = len(matches)
    loss_ratio = 1.0 - (matched / ref_count)

    if matched == 0:
        displacement_percent = 100.0
    else:
        displacements = [
            np.hypot(
                reference_keypoints[m.queryIdx].pt[0] - keypoints[m.trainIdx].pt[0],
                reference_keypoints[m.queryIdx].pt[1] - keypoints[m.trainIdx].pt[1],
            )
            for m in matches
        ]
        displacement_percent = (float(np.mean(displacements)) / frame_diagonal) * 100.0

    return SpatialMetrics(
        loss_ratio=float(loss_ratio),
        displacement_percent=float(displacement_percent),
        matched_keypoints=matched,
        reference_keypoints=ref_count,
    )
