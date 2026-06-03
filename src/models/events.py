"""Anomaly detection result and event models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class AnomalyClass(str, Enum):
    TEMPORAL = "temporal"
    SPATIAL = "spatial"


class AnomalyType(str, Enum):
    FLICKER = "flicker"
    BLACK_SCREEN = "black_screen"
    KEYPOINT_LOSS = "keypoint_loss"
    GEOMETRIC_DISTORTION = "geometric_distortion"


@dataclass
class FilterResult:
    """Output of a single filter for one frame."""

    filter_name: str
    anomaly_class: AnomalyClass
    anomaly_type: AnomalyType
    metric_value: float
    threshold: float
    is_anomaly: bool
    highlight_image: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSample:
    """Time-series metric sample for plotting."""

    timestamp_seconds: float
    filter_name: str
    metric_value: float


@dataclass
class AnomalyEvent:
    """Aggregated anomaly event spanning multiple frames."""

    id: int | None
    test_run_id: int
    anomaly_class: AnomalyClass
    anomaly_type: AnomalyType
    filter_name: str
    start_timestamp: float
    end_timestamp: float | None
    peak_metric: float
    peak_timestamp: float
    keyframe_path: str | None
    clip_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
