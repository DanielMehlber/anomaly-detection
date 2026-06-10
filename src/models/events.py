"""
Data models for frames, filter outputs, and persisted anomaly events.

These types are shared across the input layer, analysis filters, pipeline,
and persistence components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class AnomalyClass(str, Enum):
    """High-level grouping of anomaly detectors."""

    TEMPORAL = "temporal"
    SPATIAL = "spatial"


class AnomalyType(str, Enum):
    """Concrete anomaly kinds produced by individual filters."""

    FLICKER = "flicker"
    BLACK_SCREEN = "black_screen"
    KEYPOINT_LOSS = "keypoint_loss"
    MISSING_ELEMENT = "missing_element"
    GEOMETRIC_DISTORTION = "geometric_distortion"


@dataclass(frozen=True)
class HighlightRegion:
    """
    Drawable annotation passed from a filter to the clip exporter.

    Spatial filters use these regions to mark missing or displaced structures
    directly on GIF frames. The clip writer renders them with OpenCV.

    Supported shapes:
    - ``marker``: semi-transparent highlighter stain; requires ``center`` and ``radius``
    - ``circle``: outline circle; requires ``center`` and ``radius``
    - ``rect``: requires ``bbox`` as (x1, y1, x2, y2)
    - ``polygon``: requires ``polygon`` as a list of (x, y) points
    """

    shape: str
    color_bgr: tuple[int, int, int] = (0, 220, 255)
    thickness: int = 2
    alpha: float = 0.42
    center: tuple[int, int] | None = None
    radius: int | None = None
    bbox: tuple[int, int, int, int] | None = None
    polygon: tuple[tuple[int, int], ...] | None = None
    label: str | None = None


@dataclass
class FilterResult:
    """
    Output of a single filter for one video frame.

    ``is_anomaly`` drives event aggregation. ``highlight_regions`` are optional
    spatial markers burned into exported GIF clips.
    """

    filter_name: str
    anomaly_class: AnomalyClass
    anomaly_type: AnomalyType
    metric_value: float
    threshold: float
    is_anomaly: bool
    highlight_image: np.ndarray | None = None
    highlight_regions: list[HighlightRegion] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSample:
    """Single metric value recorded for dashboard time-series plots."""

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
