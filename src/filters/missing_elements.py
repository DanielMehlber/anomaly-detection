"""

Missing control-element detection filter.



Uses registered control-element positions from calibration (stable markers that

were consistently visible during the baseline window). Highlights are placed

only at those known positions when they disappear.

"""



from __future__ import annotations



from typing import Any



from src.calibration.calibrator import BaselineStats

from src.calibration.control_elements import control_missing_ratio, detect_missing_controls

from src.filters.base import AbstractFilter

from src.filters.spatial_common import SpatialMatcher

from src.models.events import (

    AnomalyClass,

    AnomalyType,

    FilterResult,

    HighlightRegion,

)

from src.models.frame import Frame





class MissingElementsFilter(AbstractFilter):

    """Detects when calibrated control markers are no longer visible."""



    name = "missing_elements"



    def __init__(self) -> None:

        self._matcher = SpatialMatcher()

        self._control_positions: list[tuple[float, float]] = []

        self._missing_threshold = 0.5



    def configure(self, baseline: BaselineStats, config: dict[str, Any]) -> None:

        """Apply calibration baseline and enable/disable flag."""

        self._matcher.configure(baseline, config)

        self._control_positions = list(baseline.control_element_positions)

        self._missing_threshold = baseline.control_missing_threshold



    def process(self, frame: Frame, processed_image=None) -> FilterResult:

        """Return a result describing how many registered controls are missing."""

        analysis = self._matcher.analyze(frame, processed_image)

        if analysis is None or not self._control_positions:

            return self._neutral_result()



        keypoints = analysis["keypoints"]

        loss_ratio = control_missing_ratio(self._control_positions, keypoints)

        is_missing = loss_ratio > self._missing_threshold

        missing_positions = (

            detect_missing_controls(self._control_positions, keypoints) if is_missing else []

        )

        regions = self._missing_regions(missing_positions)

        highlight = self._draw_highlight(analysis["image"], regions) if regions else None



        return FilterResult(

            filter_name=self.name,

            anomaly_class=AnomalyClass.SPATIAL,

            anomaly_type=AnomalyType.MISSING_ELEMENT,

            metric_value=loss_ratio,

            threshold=self._missing_threshold,

            is_anomaly=is_missing,

            highlight_image=highlight,

            highlight_regions=regions,

            metadata={

                "missing_controls": len(missing_positions),

                "registered_controls": len(self._control_positions),

                "loss_ratio": loss_ratio,

            },

        )



    def _missing_regions(

        self,

        missing_positions: list[tuple[int, int]],

    ) -> list[HighlightRegion]:

        """Build highlighter stains only at registered control positions."""

        return [

            HighlightRegion(

                shape="marker",

                center=center,

                radius=26,

                color_bgr=(0, 220, 255),

                alpha=0.45,

                label="missing",

            )

            for center in missing_positions

        ]



    def _draw_highlight(self, image, regions):

        from src.persistence.highlight_renderer import draw_highlight_regions



        return draw_highlight_regions(image, regions)



    def _neutral_result(self) -> FilterResult:

        return FilterResult(

            filter_name=self.name,

            anomaly_class=AnomalyClass.SPATIAL,

            anomaly_type=AnomalyType.MISSING_ELEMENT,

            metric_value=0.0,

            threshold=0.0,

            is_anomaly=False,

        )


