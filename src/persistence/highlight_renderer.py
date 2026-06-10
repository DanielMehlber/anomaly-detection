"""
Render spatial highlight regions onto video frames for GIF export.

Spatial anomalies are annotated with semi-transparent marker stains rather than
outlines so operators can see both the highlight and the underlying image.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.models.events import HighlightRegion

# Maximum gap (seconds) when matching a GIF frame to stored highlight data.
_HIGHLIGHT_MATCH_TOLERANCE_SECONDS = 0.2


def draw_highlight_regions(
    frame_bgr: np.ndarray,
    regions: list[HighlightRegion],
) -> np.ndarray:
    """
    Burn highlight regions onto a copy of ``frame_bgr``.

    Marker regions use alpha blending to mimic a physical text highlighter.
    """
    if not regions:
        return frame_bgr

    annotated = frame_bgr.copy()
    for region in regions:
        _draw_region(annotated, region)
    return annotated


def lookup_highlight_regions(
    frame_timestamp: float,
    event_start: float,
    event_end: float,
    frame_highlights: list[tuple[float, list[HighlightRegion]]],
) -> list[HighlightRegion]:
    """
    Find highlight regions for a GIF frame timestamp.

    Highlights are only applied inside the event window. The nearest stored
    per-frame annotation within ``_HIGHLIGHT_MATCH_TOLERANCE_SECONDS`` is used.
    """
    if frame_timestamp < event_start or frame_timestamp > event_end or not frame_highlights:
        return []

    nearest: list[HighlightRegion] = []
    nearest_delta = _HIGHLIGHT_MATCH_TOLERANCE_SECONDS + 1.0

    for timestamp, regions in frame_highlights:
        delta = abs(timestamp - frame_timestamp)
        if delta <= _HIGHLIGHT_MATCH_TOLERANCE_SECONDS and delta < nearest_delta:
            nearest = regions
            nearest_delta = delta

    return nearest


def _draw_region(frame_bgr: np.ndarray, region: HighlightRegion) -> None:
    """Draw a single highlight primitive onto ``frame_bgr`` in place."""
    if region.shape == "marker" and region.center and region.radius:
        _draw_marker(frame_bgr, region)
        return

    if region.shape == "circle" and region.center and region.radius:
        cv2.circle(frame_bgr, region.center, region.radius, region.color_bgr, region.thickness)
        return

    if region.shape == "rect" and region.bbox:
        x1, y1, x2, y2 = region.bbox
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), region.color_bgr, region.thickness)
        return

    if region.shape == "polygon" and region.polygon:
        points = np.array(region.polygon, dtype=np.int32)
        cv2.polylines(frame_bgr, [points], isClosed=True, color=region.color_bgr, thickness=region.thickness)


def _draw_marker(frame_bgr: np.ndarray, region: HighlightRegion) -> None:
    """Render a soft, semi-transparent highlighter stain."""
    assert region.center is not None and region.radius is not None

    overlay = frame_bgr.copy()
    cv2.circle(overlay, region.center, region.radius, region.color_bgr, thickness=-1)
    alpha = min(max(region.alpha, 0.05), 0.85)
    cv2.addWeighted(overlay, alpha, frame_bgr, 1.0 - alpha, 0, frame_bgr)

    if region.label:
        cv2.putText(
            frame_bgr,
            region.label,
            (region.center[0] - region.radius, region.center[1] - region.radius - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            region.color_bgr,
            1,
            cv2.LINE_AA,
        )
