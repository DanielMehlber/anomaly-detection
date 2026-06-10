"""Offline Streamlit dashboard for live monitoring and historical analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

from src.config import load_config
from src.persistence.database import Database

PAGES = ["Live Monitor", "History Archive", "Event Report"]

ANOMALY_LABELS = {
    "flicker": "Image Flicker",
    "black_screen": "Black Screen",
    "keypoint_loss": "Missing Control Points",
    "missing_element": "Missing Element",
    "geometric_distortion": "Spatial Distortion",
}

CLASS_LABELS = {
    "temporal": "Temporal",
    "spatial": "Spatial",
}


def _inject_auto_refresh(seconds: int) -> None:
    if seconds > 0:
        components.html(
            f'<meta http-equiv="refresh" content="{seconds}">',
            height=0,
        )


def _inject_light_theme_css() -> None:
    st.markdown(
        """
        <style>
            .stApp { background-color: #FFFFFF; }
            [data-testid="stSidebar"] { background-color: #F7F7F7; }
            [data-testid="stMetric"] {
                background-color: #F7F7F7;
                padding: 0.75rem 1rem;
                border-radius: 0.5rem;
                border: 1px solid #E5E5E5;
            }
            div[data-testid="stExpander"] details {
                background-color: #FAFAFA;
                border: 1px solid #E5E5E5;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _format_duration(start: float, end: float | None) -> str:
    if end is None:
        return f"{start:.1f}s → ongoing"
    return f"{start:.1f}s – {end:.1f}s ({end - start:.1f}s)"


def _resolve_media_path(path: str | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    rooted = _PROJECT_ROOT / candidate
    if rooted.exists():
        return rooted
    if candidate.exists():
        return candidate
    return None


def _anomaly_label(anomaly_type: str) -> str:
    return ANOMALY_LABELS.get(anomaly_type, anomaly_type.replace("_", " ").title())


def _navigate_to(page: str, *, run_id: int | None = None) -> None:
    st.session_state.page = page
    if run_id is not None:
        st.session_state.selected_run_id = run_id
    st.rerun()


def render_live_monitor(db: Database, refresh_seconds: int) -> None:
    st.header("Live Monitor")
    _inject_auto_refresh(refresh_seconds)

    run = db.get_latest_running_run()
    if run is None:
        runs = db.list_test_runs()
        if runs and runs[0]["status"] == "completed":
            st.info("No active test run. Showing most recent completed run.")
            run = runs[0]
        else:
            st.warning("No test runs found. Start analysis with: python main.py --config config.yaml")
            return

    status = db.get_run_status(run["id"]) or {}
    st.metric("Test Run ID", run["id"])
    col1, col2, col3 = st.columns(3)
    col1.metric("Status", run.get("status", "unknown"))
    col2.metric("Events", run.get("total_events", 0))
    col3.metric("Current Time (s)", f"{status.get('current_timestamp', 0):.1f}")

    st.write(f"**Source:** {run.get('source_path', '—')}")
    st.write(f"**Message:** {status.get('status_message', '—')}")
    st.write(
        f"**Calibration complete:** {'Yes' if status.get('calibration_complete') else 'No'}"
    )


def render_history_archive(db: Database) -> None:
    st.header("History Archive")
    runs = db.list_test_runs()
    if not runs:
        st.info("No test runs stored yet.")
        return

    for run in runs:
        with st.expander(
            f"Run #{run['id']} — {run.get('status')} — {run.get('source_path', '')}"
        ):
            st.write(f"Started: {run.get('started_at')}")
            st.write(f"Finished: {run.get('finished_at') or '—'}")
            st.write(f"Frames processed: {run.get('total_frames', 0)}")
            st.write(f"Events: {run.get('total_events', 0)}")
            if st.button("View events", key=f"view_run_{run['id']}"):
                _navigate_to("Event Report", run_id=run["id"])


def render_event_report(db: Database) -> None:
    st.header("Event Report")

    runs = db.list_test_runs()
    if not runs:
        st.info("No events to display.")
        return

    run_ids = [r["id"] for r in runs]
    default_idx = 0
    if "selected_run_id" in st.session_state and st.session_state["selected_run_id"] in run_ids:
        default_idx = run_ids.index(st.session_state["selected_run_id"])

    selected_run_id = st.selectbox(
        "Select test run",
        run_ids,
        index=default_idx,
        key="event_report_run_select",
    )
    st.session_state.selected_run_id = selected_run_id

    events = db.list_events(selected_run_id)
    if not events:
        st.success("No anomalies detected for this run.")
        return

    st.subheader(f"{len(events)} event(s) detected")

    for event in events:
        metadata = json.loads(event.get("metadata_json") or "{}")
        label = _anomaly_label(event["anomaly_type"])
        class_label = CLASS_LABELS.get(event["anomaly_class"], event["anomaly_class"])

        with st.container(border=True):
            st.markdown(f"### {label}")
            st.caption(f"{class_label} anomaly")

            col1, col2 = st.columns(2)
            col1.write(f"**Duration:** {_format_duration(event['start_timestamp'], event.get('end_timestamp'))}")
            col2.write(f"**Peak:** {event['peak_metric']:.3f} at {event['peak_timestamp']:.1f}s")

            clip_path = _resolve_media_path(event.get("clip_path"))
            keyframe_path = _resolve_media_path(event.get("keyframe_path"))

            if clip_path:
                st.image(str(clip_path), caption="Event clip (before → during → after)", use_container_width=True)
            elif keyframe_path:
                st.image(str(keyframe_path), caption="Peak frame", use_container_width=True)

            metric_key = f"{event['filter_name']}/{event['anomaly_type']}"
            samples = db.get_metric_samples(selected_run_id, metric_key)
            if samples:
                times = [s["timestamp_seconds"] for s in samples]
                values = [s["metric_value"] for s in samples]
                fig, ax = plt.subplots(figsize=(8, 2.5))
                fig.patch.set_facecolor("white")
                ax.set_facecolor("white")
                ax.plot(times, values, linewidth=1, color="#404040")
                end_ts = event.get("end_timestamp") or event["peak_timestamp"]
                ax.axvspan(
                    event["start_timestamp"],
                    end_ts,
                    color="#FF6B6B",
                    alpha=0.15,
                    label="Event window",
                )
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Metric")
                ax.set_title(f"{label} over time")
                ax.legend(loc="upper right")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            if metadata:
                with st.expander("Technical details"):
                    st.json(metadata)


def main() -> None:
    st.set_page_config(
        page_title="Anomaly Detection Dashboard",
        page_icon="📷",
        layout="wide",
    )
    _inject_light_theme_css()

    if "page" not in st.session_state:
        st.session_state.page = PAGES[0]

    config_path = _PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        st.error("config.yaml not found.")
        return

    config = load_config(config_path)
    db_path = Path(config.persistence.database_path)
    if not db_path.is_absolute():
        db_path = _PROJECT_ROOT / db_path
    db = Database(str(db_path))

    st.title("Sensor & Image Anomaly Detection")
    st.caption("Offline monitoring dashboard for long-term camera tests")

    page_index = PAGES.index(st.session_state.page) if st.session_state.page in PAGES else 0
    page = st.sidebar.radio("Navigation", PAGES, index=page_index)
    st.session_state.page = page

    if page == "Live Monitor":
        render_live_monitor(db, config.ui.refresh_interval_seconds)
    elif page == "History Archive":
        render_history_archive(db)
    else:
        render_event_report(db)


if __name__ == "__main__":
    main()
