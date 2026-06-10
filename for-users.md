# Operator Guide — Image Change Detect

How to set up, run, and review camera anomaly tests.

## What this tool does

The software watches a test video (or future live camera feed) and automatically detects problems such as:

- **Image Flicker** — rapid brightness changes between frames
- **Black Screen** — sudden loss of image content
- **Spatial Distortion** — control points shifting or the scene geometry changing
- **Missing Element** — expected control markers disappearing from the image

The first minute of each run is treated as a **calibration period** where the scene is assumed normal. The tool learns acceptable brightness and structure from that window, then monitors the remainder of the test.

## Requirements

- Windows, macOS, or Linux
- Python 3.10+ recommended
- Enough disk space for the SQLite database and GIF clips (typically a few megabytes per test run)

## Installation

1. Open a terminal in the project folder.

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   **Windows**
   ```powershell
   .venv\Scripts\activate
   ```

   **macOS / Linux**
   ```bash
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Edit `config.yaml` before a test run. The most important settings:

| Setting | Meaning |
|---------|---------|
| `input.video_path` | Path to your `.mp4` / `.avi` test video |
| `calibration.duration_seconds` | Length of the initial stable period (default 60 s) |
| `calibration.threshold_multiplier` | How strict detection is (higher = fewer false alarms) |
| `events.debounce_frames` | Frames an anomaly must persist before an event is recorded |
| `persistence.database_path` | Where results are stored |
| `ui.refresh_interval_seconds` | Live dashboard refresh rate |

You can also pass a video path on the command line without editing the file (see below).

## Running an analysis

1. Place your test video somewhere accessible, e.g. `data/my_test.mp4`.

2. Start the analysis:

   ```bash
   python main.py --video data/my_test.mp4
   ```

   Or set `input.video_path` in `config.yaml` and run:

   ```bash
   python main.py
   ```

3. Wait for the message `Analysis complete. Test run ID: …`. Long videos may take several minutes.

### Demo video and self-test

To try the included demo anomalies (flicker, missing elements, black screen, distortion):

```bash
python scripts/generate_test_video.py
python main.py --video data/sample_test.mp4
python scripts/verify_detection.py
```

`verify_detection.py` passes when all four anomaly types are detected.

### Trying the built-in demo

```bash
python scripts/generate_test_video.py
python main.py --video data/sample_test.mp4
```

The demo video contains injected flicker, a black screen, and spatial distortion for training purposes.

## Opening the dashboard

From the project folder:

```bash
python run_dashboard.py
```

Your browser opens an offline dashboard. No internet connection is required.

### Dashboard pages

**Live Monitor** — shows the active or most recent test run, frame progress, and event count. Refreshes automatically.

**History Archive** — lists past test runs. Click **View events** to jump to the report for that run.

**Event Report** — chronological list of detected anomalies for the selected run. Each entry includes:

- A readable event name (e.g. *Spatial Distortion*)
- Start time, end time, and duration
- An animated **GIF clip** showing the anomaly in context, with highlighter marks on affected areas
- A metric chart over time

## Understanding events

Each event represents one continuous anomaly episode, not a single frame. Short glitches below the debounce setting are ignored to reduce noise.

When several detectors react to the same underlying problem (for example, spatial markers disappearing during a black screen), only the **root cause** is reported. Secondary symptoms are suppressed automatically.

## Output files

| Location | Contents |
|----------|----------|
| `data/anomalies.db` | SQLite database with all runs and events |
| `data/keyframes/` | Peak JPEG stills and GIF clips per event |

Keep these files if you need audit trails. Delete `data/anomalies.db` to start with a clean history.

## Tips for reliable results

- Ensure the calibration period shows a **stable, representative** scene.
- Avoid large intentional changes in the first 60 seconds.
- Use textured scenes (labels, grids, markers) so spatial detection has features to track.
- If you see too many false alarms, increase `calibration.threshold_multiplier` or `events.debounce_frames`.
- If real anomalies are missed, decrease `threshold_multiplier` slightly.

## Troubleshooting

| Problem | Suggestion |
|---------|------------|
| `Config file not found` | Run commands from the project root folder |
| `No video path configured` | Pass `--video path\to\file.mp4` or set `input.video_path` |
| Dashboard shows no events | Confirm analysis finished and `data/anomalies.db` exists |
| Import errors when starting dashboard | Activate the virtual environment and reinstall requirements |

## Getting help from engineering

Share the test run ID, the source video path, and the relevant section of `config.yaml` when reporting issues. Event GIFs and the technical details expander in the dashboard are especially useful for diagnosis.
