"""
Main pipe-and-filter processing orchestrator.

The processor wires together calibration, optional preprocessing, temporal and
spatial filters, metric logging, and event aggregation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from src.calibration.calibrator import BaselineStats, Calibrator
from src.config import AppConfig
from src.filters.distortion import DistortionFilter
from src.filters.missing_elements import MissingElementsFilter
from src.filters.preprocessing import PreprocessingFilter
from src.filters.temporal import TemporalFilter
from src.input.abstract import AbstractInputProvider
from src.input.video_file import VideoFileInputProvider
from src.models.events import FilterResult, MetricSample
from src.persistence.database import Database
from src.persistence.event_aggregator import EventAggregator

logger = logging.getLogger(__name__)


def create_input_provider(config: AppConfig) -> AbstractInputProvider:
    """
    Instantiate the configured input provider.

    Additional providers (camera SDKs, CSV temperature feeds, etc.) can be
    registered here without changing the analysis pipeline.
    """
    provider_name = config.input.provider
    if provider_name == "VideoFileInputProvider":
        if not config.input.video_path:
            raise ValueError("input.video_path must be set in config.yaml")
        return VideoFileInputProvider(config.input.video_path)
    raise ValueError(f"Unknown input provider: {provider_name}")


class PipelineProcessor:
    """
    Orchestrate calibration, filtering, and event aggregation for one test run.

    Frames flow through the pipeline in four stages:
    1. calibration window
    2. optional preprocessing
    3. temporal + spatial filters
    4. priority-aware event aggregation
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.database = Database(config.persistence.database_path)
        self.aggregator = EventAggregator(
            database=self.database,
            keyframe_dir=config.persistence.keyframe_dir,
            debounce_frames=config.events.debounce_frames,
            end_debounce_frames=config.events.end_debounce_frames,
            video_path=config.input.video_path or None,
            clip_pre_seconds=config.events.clip_pre_seconds,
            clip_post_seconds=config.events.clip_post_seconds,
            clip_max_seconds=config.events.clip_max_seconds,
            clip_fps=config.events.clip_fps,
            anomaly_priority=config.events.anomaly_priority,
            suppression_rules=config.events.suppression_rules,
        )
        self.calibrator = Calibrator(
            duration_seconds=config.calibration.duration_seconds,
            threshold_multiplier=config.calibration.threshold_multiplier,
        )
        self.preprocessing = PreprocessingFilter()
        self.temporal = TemporalFilter()
        self.missing_elements = MissingElementsFilter()
        self.distortion = DistortionFilter()
        self._baseline: BaselineStats | None = None
        self._test_run_id: int | None = None

    def run(self) -> int:
        """
        Process the configured input stream end-to-end.

        Returns:
            The database ID of the created test run.
        """
        provider = create_input_provider(self.config)
        started_at = datetime.now(timezone.utc).isoformat()

        with provider:
            source = getattr(provider, "video_path", "unknown")
            self._test_run_id = self.database.create_test_run(str(source), started_at)
            test_run_id = self._test_run_id

            last_timestamp = 0.0
            for frame in provider.frames():
                last_timestamp = frame.timestamp_seconds
                try:
                    if self.calibrator.is_calibration_frame(frame):
                        self.calibrator.add_frame(frame)
                        self._update_status(
                            test_run_id,
                            frame,
                            calibration_complete=False,
                            status_message="Calibrating",
                        )
                        continue

                    if self._baseline is None:
                        self._baseline = self.calibrator.finalize()
                        self._configure_filters()
                        logger.info("Calibration complete. Baseline established.")

                    processed_image = self.preprocessing.apply(frame)
                    frame_results = self._collect_filter_results(frame, processed_image)
                    self._record_metric_samples(test_run_id, frame.timestamp_seconds, frame_results)
                    self.aggregator.process_frame(test_run_id, frame.timestamp_seconds, frame_results)

                    self._update_status(
                        test_run_id,
                        frame,
                        calibration_complete=True,
                        status_message="Analyzing",
                    )

                except Exception:
                    logger.exception("Failed processing frame %s", frame.frame_index)

            self.aggregator.finalize(last_timestamp)

        finished_at = datetime.now(timezone.utc).isoformat()
        self.database.finish_test_run(
            test_run_id, finished_at, self.aggregator.total_events
        )
        return test_run_id

    def _collect_filter_results(
        self,
        frame,
        processed_image,
    ) -> list[FilterResult]:
        """
        Run all analysis filters for one frame.

        Filter failures are isolated so one broken detector cannot stop the
        entire run.
        """
        results: list[FilterResult] = []

        for analysis_filter in (self.temporal, self.missing_elements, self.distortion):
            try:
                output = analysis_filter.process(frame, processed_image)
                if isinstance(output, list):
                    results.extend(output)
                else:
                    results.append(output)
            except Exception:
                logger.exception(
                    "Filter %s failed on frame %s",
                    analysis_filter.name,
                    frame.frame_index,
                )

        return results

    def _record_metric_samples(
        self,
        test_run_id: int,
        timestamp_seconds: float,
        results: list[FilterResult],
    ) -> None:
        """Persist raw metrics for dashboard plots, independent of event priority."""
        for result in results:
            metric_key = f"{result.filter_name}/{result.anomaly_type.value}"
            self.database.add_metric_sample(
                test_run_id,
                MetricSample(
                    timestamp_seconds=timestamp_seconds,
                    filter_name=metric_key,
                    metric_value=result.metric_value,
                ),
            )

    def _configure_filters(self) -> None:
        """Push calibrated baselines and YAML settings into all filters."""
        assert self._baseline is not None
        cfg = self.config
        pre_cfg: dict[str, Any] = {
            "enabled": cfg.filters.preprocessing.enabled,
            "brightness_normalization": cfg.filters.preprocessing.brightness_normalization,
        }
        temp_cfg: dict[str, Any] = {
            "enabled": cfg.filters.temporal.enabled,
            "flicker_threshold_multiplier": cfg.filters.temporal.flicker_threshold_multiplier,
            "black_screen_brightness": cfg.filters.temporal.black_screen_brightness,
            "flicker_hold_frames": cfg.filters.temporal.flicker_hold_frames,
        }
        missing_cfg: dict[str, Any] = {
            "enabled": cfg.filters.missing_elements.enabled,
        }
        spatial_cfg: dict[str, Any] = {
            "enabled": cfg.filters.spatial.enabled,
        }
        self.preprocessing.configure(self._baseline, pre_cfg)
        self.temporal.configure(self._baseline, temp_cfg)
        self.missing_elements.configure(self._baseline, missing_cfg)
        self.distortion.configure(self._baseline, spatial_cfg)

    def _update_status(
        self,
        test_run_id: int,
        frame,
        *,
        calibration_complete: bool,
        status_message: str,
    ) -> None:
        """Write progress information consumed by the live dashboard."""
        self.database.update_run_progress(
            test_run_id,
            last_timestamp=frame.timestamp_seconds,
            frame_index=frame.frame_index,
            event_count=self.aggregator.total_events,
            calibration_complete=calibration_complete,
            status_message=status_message,
        )
