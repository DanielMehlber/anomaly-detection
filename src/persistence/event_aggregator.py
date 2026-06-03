"""Event aggregation with debounce logic, keyframe and GIF clip persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.models.events import AnomalyEvent, FilterResult
from src.persistence.clip_writer import export_event_gif
from src.persistence.database import Database


@dataclass
class _ActiveEvent:
    test_run_id: int
    result: FilterResult
    start_timestamp: float
    peak_metric: float
    peak_timestamp: float
    peak_highlight: np.ndarray | None = None
    consecutive_anomaly_frames: int = 1
    consecutive_normal_frames: int = 0
    db_event_id: int | None = None
    event_key: str = ""

    def update_peak(self, timestamp: float, result: FilterResult) -> None:
        if result.metric_value >= self.peak_metric:
            self.peak_metric = result.metric_value
            self.peak_timestamp = timestamp
            if result.highlight_image is not None:
                self.peak_highlight = result.highlight_image.copy()


class EventAggregator:
    """Manages event start/update/end lifecycle with debouncing."""

    def __init__(
        self,
        database: Database,
        keyframe_dir: str,
        debounce_frames: int,
        end_debounce_frames: int,
        *,
        video_path: str | None = None,
        clip_pre_seconds: float = 2.0,
        clip_post_seconds: float = 2.0,
        clip_max_seconds: float = 10.0,
        clip_fps: float = 8.0,
    ) -> None:
        self.database = database
        self.keyframe_dir = Path(keyframe_dir)
        self.keyframe_dir.mkdir(parents=True, exist_ok=True)
        self.debounce_frames = debounce_frames
        self.end_debounce_frames = end_debounce_frames
        self.video_path = video_path
        self.clip_pre_seconds = clip_pre_seconds
        self.clip_post_seconds = clip_post_seconds
        self.clip_max_seconds = clip_max_seconds
        self.clip_fps = clip_fps
        self._active: dict[str, _ActiveEvent] = {}
        self._total_events = 0

    @property
    def total_events(self) -> int:
        return self._total_events

    @staticmethod
    def _event_key(result: FilterResult) -> str:
        return f"{result.filter_name}:{result.anomaly_type.value}"

    def process(self, test_run_id: int, timestamp: float, result: FilterResult) -> None:
        key = self._event_key(result)

        if result.is_anomaly:
            self._handle_anomaly_frame(test_run_id, timestamp, result, key)
        else:
            self._handle_normal_frame(timestamp, key)

    def _handle_anomaly_frame(
        self,
        test_run_id: int,
        timestamp: float,
        result: FilterResult,
        key: str,
    ) -> None:
        active = self._active.get(key)

        if active is None:
            self._active[key] = _ActiveEvent(
                test_run_id=test_run_id,
                result=result,
                start_timestamp=timestamp,
                peak_metric=result.metric_value,
                peak_timestamp=timestamp,
                peak_highlight=result.highlight_image.copy() if result.highlight_image is not None else None,
                event_key=key,
            )
            return

        active.consecutive_anomaly_frames += 1
        active.consecutive_normal_frames = 0
        active.update_peak(timestamp, result)

        if active.db_event_id is None and active.consecutive_anomaly_frames >= self.debounce_frames:
            event = self._build_event(active, result, end_timestamp=None)
            event_id = self.database.insert_event(event)
            active.db_event_id = event_id
            self._total_events += 1
        elif active.db_event_id is not None:
            self.database.update_event(self._build_event(active, result, end_timestamp=None))

    def _handle_normal_frame(self, timestamp: float, key: str) -> None:
        active = self._active.get(key)
        if active is None:
            return

        if active.db_event_id is None:
            self._active.pop(key, None)
            return

        active.consecutive_normal_frames += 1
        if active.consecutive_normal_frames >= self.end_debounce_frames:
            self._close_event(active, timestamp)
            self._active.pop(key, None)

    def finalize(self, timestamp: float) -> None:
        for key, active in list(self._active.items()):
            if active.db_event_id is not None:
                self._close_event(active, timestamp)
            self._active.pop(key, None)

    def _close_event(self, active: _ActiveEvent, end_timestamp: float) -> None:
        event = self._build_event(active, active.result, end_timestamp=end_timestamp)
        event.clip_path = self._export_clip(active, end_timestamp)
        self.database.update_event(event)

    def _media_suffix(self, active: _ActiveEvent) -> str:
        ts_ms = int(active.start_timestamp * 1000)
        if active.db_event_id is not None:
            return f"{active.db_event_id}_{ts_ms}"
        return str(ts_ms)

    def _build_event(
        self,
        active: _ActiveEvent,
        result: FilterResult,
        *,
        end_timestamp: float | None,
    ) -> AnomalyEvent:
        return AnomalyEvent(
            id=active.db_event_id,
            test_run_id=active.test_run_id,
            anomaly_class=result.anomaly_class,
            anomaly_type=result.anomaly_type,
            filter_name=result.filter_name,
            start_timestamp=active.start_timestamp,
            end_timestamp=end_timestamp,
            peak_metric=active.peak_metric,
            peak_timestamp=active.peak_timestamp,
            keyframe_path=self._save_keyframe(
                active.test_run_id, active.event_key, active.peak_highlight, self._media_suffix(active)
            ),
            metadata=dict(result.metadata),
        )

    def _export_clip(self, active: _ActiveEvent, end_timestamp: float) -> str | None:
        if not self.video_path:
            return None

        safe_key = active.event_key.replace(":", "_")
        gif_path = self.keyframe_dir / f"run{active.test_run_id}_{safe_key}_{self._media_suffix(active)}.gif"
        return export_event_gif(
            self.video_path,
            active.start_timestamp,
            end_timestamp,
            str(gif_path),
            pre_seconds=self.clip_pre_seconds,
            post_seconds=self.clip_post_seconds,
            max_duration=self.clip_max_seconds,
            clip_fps=self.clip_fps,
        )

    def _save_keyframe(
        self, test_run_id: int, event_key: str, image: np.ndarray | None, suffix: str
    ) -> str | None:
        if image is None:
            return None

        safe_key = event_key.replace(":", "_")
        filename = f"run{test_run_id}_{safe_key}_{suffix}.jpg"
        path = self.keyframe_dir / filename
        cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return str(path)
