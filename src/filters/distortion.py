"""
Geometric distortion detection filter.

Measures how far matched reference markers have moved relative to the
calibrated scene layout.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.calibration.calibrator import BaselineStats
from src.filters.base import AbstractFilter
from src.filters.spatial_common import SpatialMatcher
from src.models.events import (
    AnomalyClass,
    AnomalyType,
    FilterResult,
    HighlightRegion,
)
from src.models.frame import Frame

_DISPLACEMENT_HIGHLIGHT_PX = 8
_MAX_MARKER_HIGHLIGHTS = 6


class DistortionFilter(AbstractFilter):
    """Detects geometric displacement of the control scene."""

    name = "spatial"

    def __init__(self) -> None:
        self._matcher = SpatialMatcher()

    def configure(self, baseline: BaselineStats, config: dict[str, Any]) -> None:
        """Apply calibration baseline and enable/disable flag."""
        self._matcher.configure(baseline, config)

    def process(self, frame: Frame, processed_image=None) -> FilterResult:
        """Return a result describing average marker displacement."""
        analysis = self._matcher.analyze(frame, processed_image)
        if analysis is None:
            return self._neutral_result()

        metrics = analysis["metrics"]
        is_distortion = metrics.displacement_percent > analysis["disp_threshold"]
        regions = (
            self._displacement_regions(analysis["ref_kp"], analysis["keypoints"], analysis["matches"])
            if is_distortion
            else []
        )
        highlight = self._draw_highlight(analysis["image"], regions) if regions else None

        return FilterResult(
            filter_name=self.name,
            anomaly_class=AnomalyClass.SPATIAL,
            anomaly_type=AnomalyType.GEOMETRIC_DISTORTION,
            metric_value=metrics.displacement_percent,
            threshold=analysis["disp_threshold"],
            is_anomaly=is_distortion,
            highlight_image=highlight,
            highlight_regions=regions,
            metadata={
                "matched_keypoints": metrics.matched_keypoints,
                "reference_keypoints": metrics.reference_keypoints,
                "displacement_percent": metrics.displacement_percent,
            },
        )

    def _displacement_regions(self, ref_kp, cur_kp, matches) -> list[HighlightRegion]:
        """Mark the strongest displacements with orange highlighter stains."""
        candidates: list[tuple[float, tuple[int, int]]] = []

        for match in matches:
            ref_point = ref_kp[match.queryIdx].pt
            cur_point = cur_kp[match.trainIdx].pt
            displacement = float(np.hypot(ref_point[0] - cur_point[0], ref_point[1] - cur_point[1]))
            if displacement < _DISPLACEMENT_HIGHLIGHT_PX:
                continue
            candidates.append((displacement, (int(cur_point[0]), int(cur_point[1]))))

        candidates.sort(key=lambda item: item[0], reverse=True)
        return [
            HighlightRegion(
                shape="marker",
                center=center,
                radius=24,
                color_bgr=(0, 180, 255),
                alpha=0.42,
                label="shift",
            )
            for _, center in candidates[:_MAX_MARKER_HIGHLIGHTS]
        ]

    def _draw_highlight(self, image, regions):
        from src.persistence.highlight_renderer import draw_highlight_regions

        return draw_highlight_regions(image, regions)

    def _neutral_result(self) -> FilterResult:
        return FilterResult(
            filter_name=self.name,
            anomaly_class=AnomalyClass.SPATIAL,
            anomaly_type=AnomalyType.GEOMETRIC_DISTORTION,
            metric_value=0.0,
            threshold=0.0,
            is_anomaly=False,
        )
