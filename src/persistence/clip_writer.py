"""Export short GIF clips for anomaly events from the source video."""

from __future__ import annotations

from pathlib import Path

import cv2
from PIL import Image


def export_event_gif(
    video_path: str,
    start_timestamp: float,
    end_timestamp: float,
    output_path: str,
    *,
    pre_seconds: float = 2.0,
    post_seconds: float = 2.0,
    max_duration: float = 10.0,
    clip_fps: float = 8.0,
) -> str | None:
    """Extract a GIF spanning the event window, padded before/after, capped at max_duration."""
    path = Path(video_path)
    if not path.exists():
        return None

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    event_duration = max(end_timestamp - start_timestamp, 0.1)
    total_duration = min(pre_seconds + event_duration + post_seconds, max_duration)

    clip_start = max(0.0, start_timestamp - pre_seconds)
    clip_end = clip_start + total_duration

    frame_step = max(1, int(round(source_fps / clip_fps)))
    start_frame = int(clip_start * source_fps)
    end_frame = int(clip_end * source_fps)

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    pil_frames: list[Image.Image] = []
    frame_idx = start_frame

    while frame_idx <= end_frame:
        success, frame = capture.read()
        if not success:
            break
        if (frame_idx - start_frame) % frame_step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frames.append(Image.fromarray(rgb))
        frame_idx += 1

    capture.release()

    if not pil_frames:
        return None

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, int(1000 / clip_fps))
    pil_frames[0].save(
        out,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return str(out)
