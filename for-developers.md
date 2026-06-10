# Developer Guide — Image Change Detect

Technical overview for engineers extending or maintaining the anomaly-detection framework.

## Purpose

The system automates long-term camera monitoring in climate-chamber tests. It ingests a video stream (or future live sources), learns a per-test baseline during an initial stable window, then flags temporal and spatial anomalies. Results are stored as **events** (not per-frame images) in SQLite, with one annotated GIF clip per event.

## Architecture

The design follows a **pipe-and-filter** pattern:

```
Input Provider → Calibration → [Preprocessing] → Temporal Filter ─┐
                                              → Spatial Filter  ─┤
                                                                 ↓
                                    Priority / Event Aggregator → SQLite + GIF clips
                                                                 ↓
                                                          Streamlit Dashboard
```

### Data flow per frame

1. **Input** yields a `Frame` (image + timestamp + metadata).
2. During the calibration window, frames only update the `Calibrator`.
3. After calibration, optional **preprocessing** adjusts the image (e.g. brightness normalization).
4. **Temporal**, **missing-elements**, and **distortion** filters each return `FilterResult` objects with metrics, anomaly flags, and optional `HighlightRegion` annotations.
5. The **event aggregator** applies symptom-suppression rules, debounces anomalies, and writes events.
6. Raw metrics are always logged for dashboard plots, even when an anomaly is suppressed.

### Symptom suppression

Multiple filters may fire on the same frame. Not all detections represent independent root causes. Suppression rules in `config.yaml` define which dominant anomalies hide dependent symptoms—for example, a black screen suppresses spatial keypoint loss because the image content is gone, not because the scene geometry changed.

Unrelated anomalies (flicker vs. spatial distortion at different times) are **not** forced into a single winner.

## Repository layout

| Path | Role |
|------|------|
| `main.py` | CLI entry point for offline analysis |
| `run_dashboard.py` | Launches the Streamlit operator UI |
| `config.yaml` | All tunable behaviour (no hardcoded thresholds in code) |
| `src/input/` | Pluggable input providers (`VideoFileInputProvider` baseline) |
| `src/calibration/` | Self-calibration and spatial metric helpers |
| `src/filters/` | Analysis filters (temporal, missing elements, distortion, preprocessing) |
| `src/pipeline/` | Orchestration (`PipelineProcessor`) |
| `src/persistence/` | SQLite access, event aggregation, GIF export |
| `src/anomaly_priority.py` | Symptom-suppression rule resolution |
| `src/dashboard/` | Offline Streamlit application |
| `src/models/` | Shared dataclasses (`Frame`, `FilterResult`, `AnomalyEvent`) |

## Algorithms

### Calibration (first N seconds)

The calibrator treats the opening segment as a known-good reference:

- **Temporal baseline:** mean brightness, brightness variation over time, and frame-to-frame brightness deltas (used for flicker thresholding).
- **Spatial baseline:** ORB keypoints/descriptors from the first calibration frame; during calibration, keypoint loss ratios and mean displacement percentages are sampled to build dynamic tolerances.

Thresholds are computed as `mean + std × threshold_multiplier` from `config.yaml`.

### Temporal filter

Uses global mean grayscale brightness per frame:

- **Black screen:** brightness falls below a calibrated lower bound.
- **Flicker:** absolute brightness jump between consecutive frames exceeds the calibrated flicker threshold. A short **hold counter** (`flicker_hold_frames`) keeps flicker active across alternating bright/dark frames so debouncing can open a single event.

Both checks return separate results so events can open and close independently.

### Missing-elements filter

Compares each frame to the calibration reference using **ORB** feature detection and **Hamming-distance** brute-force matching:

- **Missing element:** fraction of reference control markers without a match exceeds the calibrated loss threshold.

### Distortion filter

Uses the same ORB reference matching:

- **Geometric distortion:** mean pixel displacement of matched keypoints, expressed as a percentage of the frame diagonal.

When both missing-element and distortion fire on the same frame, the filter with the stronger `metric / threshold` ratio wins; the other is suppressed as a secondary symptom.

ORB is used for speed and offline operation; it works best on textured scenes (the sample video includes a grid for this reason).

### Event aggregation

An anomaly must persist for `debounce_frames` before an event is created, and remain absent for `end_debounce_frames` before it closes. While open, the peak metric frame and per-frame `HighlightRegion` lists are tracked. On close, a GIF is exported from the source video with marker overlays.

### GIF highlights

Spatial filters attach `HighlightRegion` objects (`shape="marker"`)—semi-transparent highlighter stains rendered via alpha blending in `highlight_renderer.py`. This replaces outline boxes for operator clarity.

## Extending the system

### New input source

Subclass `AbstractInputProvider` in `src/input/`, implement `open()`, `frames()`, and `close()`. Register the class name in `create_input_provider()` inside `processor.py`. The pipeline only requires `Frame` objects with a monotonic environment timestamp.

### New filter

Subclass `AbstractFilter`, implement `configure()` and `process()`. Return a `FilterResult` or list of results. Add the filter to `_collect_filter_results()` in the processor. Update suppression rules if the new anomaly type has symptom relationships.

### Configuration

All behaviour-critical values belong in `config.yaml`. The `load_config()` function in `src/config.py` maps YAML to typed dataclasses.

## Persistence

- **SQLite** (`data/anomalies.db` by default) stores test runs, events, and metric time series.
- **JPEG keyframes** and **GIF clips** live under `data/keyframes/`.
- WAL journal mode is enabled for safer concurrent dashboard reads during analysis.

## Running locally

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
python scripts/generate_test_video.py
python main.py --video data/sample_test.mp4
python scripts/verify_detection.py   # expects flicker, missing_element, black_screen, geometric_distortion
python run_dashboard.py
```

## Dependencies

- **OpenCV** — video I/O, ORB, image drawing
- **NumPy** — numerical operations
- **PyYAML** — configuration
- **Streamlit** — offline dashboard
- **Pillow** — GIF encoding
- **matplotlib** — event metric plots in the dashboard
