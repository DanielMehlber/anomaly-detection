"""Verify that all four demo anomaly types are detected in the sample video."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_test_video import generate_test_video
from src.config import load_config
from src.persistence.database import Database
from src.pipeline.processor import PipelineProcessor

EXPECTED = {
    "flicker",
    "missing_element",
    "black_screen",
    "geometric_distortion",
}


def main() -> int:
    video_path = ROOT / "data" / "sample_test.mp4"
    db_path = ROOT / "data" / "anomalies.db"

    if db_path.exists():
        db_path.unlink()

    generate_test_video(str(video_path))

    config = load_config(ROOT / "config.yaml")
    config.input.video_path = str(video_path)
    config.persistence.database_path = str(db_path)
    config.persistence.keyframe_dir = str(ROOT / "data" / "keyframes")

    run_id = PipelineProcessor(config).run()
    events = Database(str(db_path)).list_events(run_id)
    detected = {event["anomaly_type"] for event in events}

    print("Detected events:")
    for event in events:
        end = event.get("end_timestamp") or 0.0
        print(
            f"  - {event['anomaly_type']}: "
            f"{event['start_timestamp']:.1f}s – {end:.1f}s"
        )

    missing = EXPECTED - detected
    if missing:
        print(f"FAILED: missing detections: {sorted(missing)}")
        return 1

    print("PASSED: all expected anomaly types detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
