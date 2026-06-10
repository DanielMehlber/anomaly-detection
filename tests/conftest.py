"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.calibration.calibrator import BaselineStats, Calibrator
from src.models.frame import Frame


@pytest.fixture
def project_root() -> Path:
    return ROOT


@pytest.fixture
def textured_frame() -> np.ndarray:
    """Stable scene with five distinct control markers."""
    height, width = 480, 640
    image = np.full((height, width, 3), 180, dtype=np.uint8)

    for x in range(40, width - 40, 40):
        cv2.line(image, (x, 40), (x, height - 40), (140, 140, 140), 1)
    for y in range(40, height - 40, 40):
        cv2.line(image, (40, y), (width - 40, y), (140, 140, 140), 1)

    points = [(80, 80), (560, 80), (80, 400), (560, 400), (320, 240)]
    for x, y in points:
        cv2.circle(image, (x, y), 14, (40, 40, 190), -1)
        cv2.circle(image, (x, y), 16, (255, 255, 255), 2)

    return image


@pytest.fixture
def partial_frame(textured_frame) -> np.ndarray:
    """Same scene but only two control markers are drawn."""
    image = textured_frame.copy()
    overlay = np.full_like(image, 180)
    for x in range(40, image.shape[1] - 40, 40):
        cv2.line(overlay, (x, 40), (x, image.shape[0] - 40), (140, 140, 140), 1)
    for y in range(40, image.shape[0] - 40, 40):
        cv2.line(overlay, (40, y), (image.shape[1] - 40, y), (140, 140, 140), 1)
    for x, y in [(80, 80), (560, 80)]:
        cv2.circle(overlay, (x, y), 14, (40, 40, 190), -1)
    return overlay


@pytest.fixture
def calibrated_baseline(textured_frame) -> BaselineStats:
    """Run a short synthetic calibration on the full control scene."""
    calibrator = Calibrator(duration_seconds=2.0, threshold_multiplier=3.5)
    for index in range(60):
        frame = Frame(
            image=textured_frame.copy(),
            timestamp_seconds=index / 30.0,
            frame_index=index,
        )
        if calibrator.is_calibration_frame(frame):
            calibrator.add_frame(frame)
    return calibrator.finalize()
