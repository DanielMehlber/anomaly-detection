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

## Quick Start

1. Generate a sample test video (optional):

```bash
python scripts/generate_test_video.py
```

2. Set the video path in `config.yaml` or pass it on the command line:

```bash
python main.py --video data/sample_test.mp4
```

3. Launch the offline dashboard (from the project root):

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
| `events` | Debounce frames for event start/end |
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
