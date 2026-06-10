"""
Abstract base class for analysis filters in the pipe-and-filter pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.calibration.calibrator import BaselineStats
from src.models.events import FilterResult
from src.models.frame import Frame


class AbstractFilter(ABC):
    """
    Isolated analysis unit.

    Each filter receives a read-only ``Frame`` plus an optional preprocessed
    image and returns one or more ``FilterResult`` objects.
    """

    name: str = "abstract"

    @abstractmethod
    def configure(self, baseline: BaselineStats, config: dict[str, Any]) -> None:
        """
        Apply calibration statistics and YAML settings.

        Called once after the calibration window completes.
        """

    @abstractmethod
    def process(
        self,
        frame: Frame,
        processed_image: np.ndarray | None = None,
    ) -> FilterResult | list[FilterResult]:
        """
        Analyze a frame and return one or more metric results.

        Implementations should not raise uncaught exceptions; the pipeline wraps
        calls in a protective try/except block, but local handling is preferred.
        """
