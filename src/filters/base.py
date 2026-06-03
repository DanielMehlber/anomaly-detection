"""Base class for pipeline filters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.calibration.calibrator import BaselineStats
from src.models.events import FilterResult
from src.models.frame import Frame


class AbstractFilter(ABC):
    """Isolated filter unit in the pipe-and-filter architecture."""

    name: str = "abstract"

    @abstractmethod
    def configure(self, baseline: BaselineStats, config: dict[str, Any]) -> None:
        """Apply baseline statistics and filter-specific configuration."""

    @abstractmethod
    def process(self, frame: Frame, processed_image: np.ndarray | None = None) -> FilterResult:
        """Analyze a frame and return a metric result."""
