"""SQLite persistence layer for test runs, events, and metrics."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from src.models.events import AnomalyClass, AnomalyEvent, AnomalyType, MetricSample


class Database:
    """Thread-safe SQLite access for anomaly events and test runs."""

    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _init_schema(self) -> None:
        with self._connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS test_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_path TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    status TEXT NOT NULL DEFAULT 'running',
                    total_frames INTEGER DEFAULT 0,
                    total_events INTEGER DEFAULT 0,
                    last_timestamp REAL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_run_id INTEGER NOT NULL,
                    anomaly_class TEXT NOT NULL,
                    anomaly_type TEXT NOT NULL,
                    filter_name TEXT NOT NULL,
                    start_timestamp REAL NOT NULL,
                    end_timestamp REAL,
                    peak_metric REAL NOT NULL,
                    peak_timestamp REAL NOT NULL,
                    keyframe_path TEXT,
                    metadata_json TEXT,
                    FOREIGN KEY (test_run_id) REFERENCES test_runs(id)
                );

                CREATE TABLE IF NOT EXISTS metric_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_run_id INTEGER NOT NULL,
                    filter_name TEXT NOT NULL,
                    timestamp_seconds REAL NOT NULL,
                    metric_value REAL NOT NULL,
                    FOREIGN KEY (test_run_id) REFERENCES test_runs(id)
                );

                CREATE TABLE IF NOT EXISTS run_status (
                    test_run_id INTEGER PRIMARY KEY,
                    current_timestamp REAL DEFAULT 0,
                    frame_index INTEGER DEFAULT 0,
                    event_count INTEGER DEFAULT 0,
                    calibration_complete INTEGER DEFAULT 0,
                    status_message TEXT,
                    FOREIGN KEY (test_run_id) REFERENCES test_runs(id)
                );
                """
            )
            self._ensure_column(conn, "events", "clip_path", "TEXT")

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def create_test_run(self, source_path: str, started_at: str) -> int:
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO test_runs (source_path, started_at, status)
                VALUES (?, ?, 'running')
                """,
                (source_path, started_at),
            )
            run_id = int(cursor.lastrowid)
            conn.execute(
                "INSERT INTO run_status (test_run_id, status_message) VALUES (?, ?)",
                (run_id, "Initializing"),
            )
            return run_id

    def update_run_progress(
        self,
        test_run_id: int,
        *,
        last_timestamp: float,
        frame_index: int,
        event_count: int,
        calibration_complete: bool,
        status_message: str,
    ) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE test_runs
                SET last_timestamp = ?, total_frames = ?, total_events = ?
                WHERE id = ?
                """,
                (last_timestamp, frame_index, event_count, test_run_id),
            )
            conn.execute(
                """
                INSERT INTO run_status (
                    test_run_id, current_timestamp, frame_index, event_count,
                    calibration_complete, status_message
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(test_run_id) DO UPDATE SET
                    current_timestamp = excluded.current_timestamp,
                    frame_index = excluded.frame_index,
                    event_count = excluded.event_count,
                    calibration_complete = excluded.calibration_complete,
                    status_message = excluded.status_message
                """,
                (
                    test_run_id,
                    last_timestamp,
                    frame_index,
                    event_count,
                    int(calibration_complete),
                    status_message,
                ),
            )

    def finish_test_run(self, test_run_id: int, finished_at: str, total_events: int) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE test_runs
                SET finished_at = ?, status = 'completed', total_events = ?
                WHERE id = ?
                """,
                (finished_at, total_events, test_run_id),
            )

    def insert_event(self, event: AnomalyEvent) -> int:
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO events (
                    test_run_id, anomaly_class, anomaly_type, filter_name,
                    start_timestamp, end_timestamp, peak_metric, peak_timestamp,
                    keyframe_path, clip_path, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.test_run_id,
                    event.anomaly_class.value,
                    event.anomaly_type.value,
                    event.filter_name,
                    event.start_timestamp,
                    event.end_timestamp,
                    event.peak_metric,
                    event.peak_timestamp,
                    event.keyframe_path,
                    event.clip_path,
                    json.dumps(event.metadata),
                ),
            )
            return int(cursor.lastrowid)

    def update_event(self, event: AnomalyEvent) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE events
                SET end_timestamp = ?, peak_metric = ?, peak_timestamp = ?,
                    keyframe_path = ?, clip_path = ?, metadata_json = ?
                WHERE id = ?
                """,
                (
                    event.end_timestamp,
                    event.peak_metric,
                    event.peak_timestamp,
                    event.keyframe_path,
                    event.clip_path,
                    json.dumps(event.metadata),
                    event.id,
                ),
            )

    def add_metric_sample(self, test_run_id: int, sample: MetricSample) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO metric_samples (test_run_id, filter_name, timestamp_seconds, metric_value)
                VALUES (?, ?, ?, ?)
                """,
                (test_run_id, sample.filter_name, sample.timestamp_seconds, sample.metric_value),
            )

    def list_test_runs(self) -> list[dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM test_runs ORDER BY id DESC"
            ).fetchall()
            return [dict(row) for row in rows]

    def get_test_run(self, test_run_id: int) -> dict[str, Any] | None:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM test_runs WHERE id = ?", (test_run_id,)).fetchone()
            return dict(row) if row else None

    def get_run_status(self, test_run_id: int) -> dict[str, Any] | None:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM run_status WHERE test_run_id = ?", (test_run_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_latest_running_run(self) -> dict[str, Any] | None:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM test_runs WHERE status = 'running' ORDER BY id DESC LIMIT 1"
            ).fetchone()
            return dict(row) if row else None

    def list_events(self, test_run_id: int) -> list[dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE test_run_id = ? ORDER BY start_timestamp",
                (test_run_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_metric_samples(
        self, test_run_id: int, filter_name: str | None = None
    ) -> list[dict[str, Any]]:
        with self._connection() as conn:
            if filter_name:
                rows = conn.execute(
                    """
                    SELECT * FROM metric_samples
                    WHERE test_run_id = ? AND filter_name = ?
                    ORDER BY timestamp_seconds
                    """,
                    (test_run_id, filter_name),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM metric_samples
                    WHERE test_run_id = ?
                    ORDER BY timestamp_seconds
                    """,
                    (test_run_id,),
                ).fetchall()
            return [dict(row) for row in rows]

    @staticmethod
    def row_to_event(row: dict[str, Any]) -> AnomalyEvent:
        metadata = json.loads(row.get("metadata_json") or "{}")
        return AnomalyEvent(
            id=row["id"],
            test_run_id=row["test_run_id"],
            anomaly_class=AnomalyClass(row["anomaly_class"]),
            anomaly_type=AnomalyType(row["anomaly_type"]),
            filter_name=row["filter_name"],
            start_timestamp=row["start_timestamp"],
            end_timestamp=row.get("end_timestamp"),
            peak_metric=row["peak_metric"],
            peak_timestamp=row["peak_timestamp"],
            keyframe_path=row.get("keyframe_path"),
            clip_path=row.get("clip_path"),
            metadata=metadata,
        )
