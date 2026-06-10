# Image Change Detect

Minimal, extensible Python framework for detecting temporal and spatial anomalies in long-term camera sensor tests (climate chamber monitoring).

## Features

- **Pipe-and-filter architecture** with isolated, configurable analysis filters
- **Self-calibration** from the first 60 seconds of stable footage
- **Temporal filters**: flicker intensity, black screen detection
- **Spatial filters**: keypoint loss, geometric distortion
- **Event-based persistence** with SQLite and one JPEG keyframe per event
- **Offline Streamlit dashboard** for live monitoring, history, and event reports

## Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Documentation

- [Operator guide](for-users.md) — setup, configuration, dashboard usage
- [Developer guide](for-developers.md) — architecture, algorithms, extension points

## Quick Start

1. Generate a sample test video (optional):

```bash
python scripts/generate_test_video.py
```

2. Run analysis:

```bash
python main.py --video data/sample_test.mp4
```

3. Verify all four demo anomalies are detected:

```bash
python scripts/verify_detection.py
```

4. Launch the offline dashboard (from the project root):

```bash
python run_dashboard.py
```

Or: `streamlit run src/dashboard/app.py` (must be run from the project root directory).

## Configuration

All behavior is driven by `config.yaml` — no hardcoded thresholds:

| Section | Purpose |
|---------|---------|
| `input` | Video path and input provider class |
| `calibration` | Calibration window duration and tolerance multiplier |
| `filters` | Enable/disable and tune temporal/spatial filters |
| `events` | Debounce frames, GIF clip settings, anomaly priority ranking |
| `persistence` | SQLite database and keyframe directory |
| `ui` | Dashboard refresh interval |

## Architecture

```
VideoFileInputProvider → Calibrator (60s) → [Preprocessing] → TemporalFilter
                                                           → SpatialFilter
                                                                    ↓
                                                          EventAggregator → SQLite
```

Extend the system by implementing new subclasses of `AbstractInputProvider` or `AbstractFilter` without changing the pipeline.

### Anomaly priority

When several filters fire on the same frame, only the highest-priority root cause becomes an event. The default order is configured in `config.yaml`:

1. Black Screen
2. Spatial Distortion
3. Missing Control Points
4. Image Flicker

Spatial filters can also pass `HighlightRegion` annotations that are burned into exported GIF clips.

## Project Layout

```
├── config.yaml
├── main.py
├── requirements.txt
├── scripts/generate_test_video.py
└── src/
    ├── calibration/
    ├── dashboard/
    ├── filters/
    ├── input/
    ├── models/
    ├── persistence/
    └── pipeline/
```
"# anomaly-detection" 
