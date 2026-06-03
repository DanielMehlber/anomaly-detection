"""Generate a synthetic test video for demonstration."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _draw_control_screen(
    frame: np.ndarray,
    points: list[tuple[int, int]],
    *,
    brightness: int = 180,
) -> None:
    """Draw a textured control screen with distinctive ORB-friendly markers."""
    height, width = frame.shape[:2]
    frame[:] = (brightness, brightness, brightness)

    # Grid texture gives stable keypoints during calibration.
    for x in range(40, width - 40, 40):
        cv2.line(frame, (x, 40), (x, height - 40), (140, 140, 140), 1)
    for y in range(40, height - 40, 40):
        cv2.line(frame, (40, y), (width - 40, y), (140, 140, 140), 1)

    cv2.rectangle(frame, (60, 60), (width - 60, height - 60), (90, 90, 90), 2)

    for x, y in points:
        cv2.circle(frame, (x, y), 14, (40, 40, 190), -1)
        cv2.circle(frame, (x, y), 16, (255, 255, 255), 2)
        cv2.putText(
            frame,
            "+",
            (x - 6, y + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )


def generate_test_video(output_path: str, duration_seconds: float = 120, fps: float = 30) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    total_frames = int(duration_seconds * fps)
    rng = np.random.default_rng(42)
    base_points = [(80, 80), (560, 80), (80, 400), (560, 400), (320, 240)]

    for i in range(total_frames):
        t = i / fps
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Slow brightness drift during stable periods (within calibration tolerance).
        drift = int(4 * np.sin(t / 18.0))
        _draw_control_screen(frame, base_points, brightness=178 + drift)

        # Flicker anomaly at t=75-78
        if 75 <= t <= 78 and int(t * 10) % 2 == 0:
            frame = np.clip(frame.astype(np.int16) + 70, 0, 255).astype(np.uint8)

        # Black screen at t=95-97
        if 95 <= t <= 97:
            frame[:] = 5

        # Spatial distortion at t=105-112 only, then scene returns to normal
        if 105 <= t <= 112:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            offset = int((t - 105) * 12)
            shifted = [(min(x + offset, width - 80), min(y + offset // 2, height - 80)) for x, y in base_points]
            _draw_control_screen(frame, shifted, brightness=178)

        noise = rng.integers(-2, 3, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)

    writer.release()
    print(f"Generated test video: {path} ({total_frames} frames, {duration_seconds}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic test video")
    parser.add_argument(
        "-o",
        "--output",
        default="data/sample_test.mp4",
        help="Output video path",
    )
    parser.add_argument("--duration", type=float, default=120, help="Duration in seconds")
    parser.add_argument("--fps", type=float, default=30, help="Frames per second")
    args = parser.parse_args()
    generate_test_video(args.output, args.duration, args.fps)


if __name__ == "__main__":
    main()
