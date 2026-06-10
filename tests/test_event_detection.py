"""Integration tests for end-to-end anomaly event detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.generate_test_video import generate_test_video
from src.config import load_config
from src.persistence.database import Database
from src.pipeline.processor import PipelineProcessor

EXPECTED_EVENTS = {
    "flicker": (74.0, 79.0),
    "missing_element": (87.0, 93.0),
    "black_screen": (94.0, 98.0),
    "geometric_distortion": (104.0, 113.0),
}


@pytest.fixture
def analysis_run(project_root: Path, tmp_path: Path):
    video_path = tmp_path / "sample_test.mp4"
    db_path = tmp_path / "anomalies.db"
    keyframe_dir = tmp_path / "keyframes"

    generate_test_video(str(video_path))

    config = load_config(project_root / "config.yaml")
    config.input.video_path = str(video_path)
    config.persistence.database_path = str(db_path)
    config.persistence.keyframe_dir = str(keyframe_dir)

    run_id = PipelineProcessor(config).run()
    db = Database(str(db_path))
    events = db.list_events(run_id)
    return run_id, events, db


def _find_event(events: list[dict], anomaly_type: str) -> dict:
    matches = [event for event in events if event["anomaly_type"] == anomaly_type]
    assert matches, f"No event for {anomaly_type}"
    return matches[0]


def test_all_expected_anomaly_types_detected(analysis_run):
    _, events, _ = analysis_run
    detected = {event["anomaly_type"] for event in events}
    assert detected == set(EXPECTED_EVENTS.keys())


def test_event_windows_match_injected_anomalies(analysis_run):
    _, events, _ = analysis_run

    for anomaly_type, (start_min, start_max) in {
        "flicker": (74.5, 76.0),
        "missing_element": (87.5, 89.0),
        "black_screen": (94.5, 96.0),
        "geometric_distortion": (105.0, 106.5),
    }.items():
        event = _find_event(events, anomaly_type)
        assert start_min <= event["start_timestamp"] <= start_max, anomaly_type

    for anomaly_type, (end_min, end_max) in {
        "flicker": (77.5, 79.5),
        "missing_element": (91.5, 93.5),
        "black_screen": (96.5, 98.0),
        "geometric_distortion": (111.0, 113.5),
    }.items():
        event = _find_event(events, anomaly_type)
        end = event.get("end_timestamp")
        assert end is not None, anomaly_type
        assert end_min <= end <= end_max, anomaly_type


def test_missing_element_event_has_valid_highlight_metadata(analysis_run):
    _, events, _ = analysis_run
    event = _find_event(events, "missing_element")

    assert event["filter_name"] == "missing_elements"
    metadata = __import__("json").loads(event.get("metadata_json") or "{}")
    assert metadata.get("registered_controls", 0) > 0
    assert metadata.get("missing_controls", 0) > 0
    assert event.get("clip_path") or event.get("keyframe_path")


def test_events_do_not_overlap_in_invalid_ways(analysis_run):
    """Black screen must not produce a parallel missing-element event."""
    _, events, _ = analysis_run
    black = _find_event(events, "black_screen")
    missing = _find_event(events, "missing_element")

    overlap_start = max(black["start_timestamp"], missing["start_timestamp"])
    overlap_end = min(black.get("end_timestamp") or 0, missing.get("end_timestamp") or 0)
    assert overlap_end <= overlap_start
