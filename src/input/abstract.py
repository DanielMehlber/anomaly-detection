"""Abstract input provider interface for extensible data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from src.models.frame import Frame


class AbstractInputProvider(ABC):
    """Base class for all input sources (video files, cameras, combined APIs)."""

    @abstractmethod
    def open(self) -> None:
        """Prepare the input source for reading."""

    @abstractmethod
    def frames(self) -> Iterator[Frame]:
        """Yield frames with environment metadata (timestamp as runtime factor)."""

    @abstractmethod
    def close(self) -> None:
        """Release input resources."""

    def __enter__(self) -> "AbstractInputProvider":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
