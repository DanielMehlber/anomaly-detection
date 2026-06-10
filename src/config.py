"""Load and expose application configuration from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.anomaly_priority import DEFAULT_ANOMALY_PRIORITY, DEFAULT_SUPPRESSION_RULES


@dataclass
class InputConfig:
    provider: str
    video_path: str


@dataclass
class CalibrationConfig:
    duration_seconds: float
    threshold_multiplier: float


@dataclass
class PreprocessingConfig:
    enabled: bool
    brightness_normalization: bool


@dataclass
class TemporalFilterConfig:
    enabled: bool
    flicker_threshold_multiplier: float
    black_screen_brightness: float
    flicker_hold_frames: int


@dataclass
class MissingElementsFilterConfig:
    enabled: bool


@dataclass
class SpatialFilterConfig:
    enabled: bool


@dataclass
class FiltersConfig:
    preprocessing: PreprocessingConfig
    temporal: TemporalFilterConfig
    missing_elements: MissingElementsFilterConfig
    spatial: SpatialFilterConfig


@dataclass
class EventsConfig:
    debounce_frames: int
    end_debounce_frames: int
    clip_pre_seconds: float
    clip_post_seconds: float
    clip_max_seconds: float
    clip_fps: float
    anomaly_priority: list[str]
    suppression_rules: dict[str, list[str]]


@dataclass
class PersistenceConfig:
    database_path: str
    keyframe_dir: str


@dataclass
class UIConfig:
    refresh_interval_seconds: int
    host: str
    port: int


@dataclass
class AppConfig:
    input: InputConfig
    calibration: CalibrationConfig
    filters: FiltersConfig
    events: EventsConfig
    persistence: PersistenceConfig
    ui: UIConfig


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    return dict(data.get(key, {}))


def load_config(path: str | Path) -> AppConfig:
    """
    Load and validate application settings from a YAML file.

    Args:
        path: Path to ``config.yaml`` or an override file.

    Returns:
        A fully populated ``AppConfig`` dataclass tree.
    """
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    input_cfg = _section(raw, "input")
    cal_cfg = _section(raw, "calibration")
    filters_raw = _section(raw, "filters")
    pre_cfg = _section(filters_raw, "preprocessing")
    temp_cfg = _section(filters_raw, "temporal")
    missing_cfg = _section(filters_raw, "missing_elements")
    spatial_cfg = _section(filters_raw, "spatial")
    events_cfg = _section(raw, "events")
    persist_cfg = _section(raw, "persistence")
    ui_cfg = _section(raw, "ui")

    return AppConfig(
        input=InputConfig(
            provider=input_cfg.get("provider", "VideoFileInputProvider"),
            video_path=input_cfg.get("video_path", ""),
        ),
        calibration=CalibrationConfig(
            duration_seconds=float(cal_cfg.get("duration_seconds", 60)),
            threshold_multiplier=float(cal_cfg.get("threshold_multiplier", 3.5)),
        ),
        filters=FiltersConfig(
            preprocessing=PreprocessingConfig(
                enabled=bool(pre_cfg.get("enabled", False)),
                brightness_normalization=bool(pre_cfg.get("brightness_normalization", True)),
            ),
            temporal=TemporalFilterConfig(
                enabled=bool(temp_cfg.get("enabled", True)),
                flicker_threshold_multiplier=float(
                    temp_cfg.get("flicker_threshold_multiplier", 3.5)
                ),
                black_screen_brightness=float(temp_cfg.get("black_screen_brightness", 10)),
                flicker_hold_frames=int(temp_cfg.get("flicker_hold_frames", 4)),
            ),
            missing_elements=MissingElementsFilterConfig(
                enabled=bool(missing_cfg.get("enabled", True)),
            ),
            spatial=SpatialFilterConfig(
                enabled=bool(spatial_cfg.get("enabled", True)),
            ),
        ),
        events=EventsConfig(
            debounce_frames=int(events_cfg.get("debounce_frames", 5)),
            end_debounce_frames=int(events_cfg.get("end_debounce_frames", 3)),
            clip_pre_seconds=float(events_cfg.get("clip_pre_seconds", 2.0)),
            clip_post_seconds=float(events_cfg.get("clip_post_seconds", 2.0)),
            clip_max_seconds=float(events_cfg.get("clip_max_seconds", 10.0)),
            clip_fps=float(events_cfg.get("clip_fps", 8.0)),
            anomaly_priority=list(
                events_cfg.get("anomaly_priority", list(DEFAULT_ANOMALY_PRIORITY))
            ),
            suppression_rules=dict(
                events_cfg.get("suppression_rules", DEFAULT_SUPPRESSION_RULES)
            ),
        ),
        persistence=PersistenceConfig(
            database_path=persist_cfg.get("database_path", "data/anomalies.db"),
            keyframe_dir=persist_cfg.get("keyframe_dir", "data/keyframes"),
        ),
        ui=UIConfig(
            refresh_interval_seconds=int(ui_cfg.get("refresh_interval_seconds", 5)),
            host=ui_cfg.get("host", "localhost"),
            port=int(ui_cfg.get("port", 8501)),
        ),
    )
