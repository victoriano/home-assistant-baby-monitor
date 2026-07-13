from __future__ import annotations

import math
import sqlite3
import struct
from datetime import timedelta
from pathlib import Path

import pytest
from baby_monitor.database import Database, StorageError
from baby_monitor.media import (
    SAMPLE_RATE,
    MediaError,
    _ffconcat_source,
    _input_args,
    _subprocess_env,
    analyze_pcm,
)
from baby_monitor.models import SecretChanges, SleepEventCreate, SleepEventPatch, utc_now


def test_sleep_crud_and_retention_keep_metadata(tmp_path: Path) -> None:
    database = Database(tmp_path)
    assert tmp_path.stat().st_mode & 0o077 == 0
    assert database.frames_dir.stat().st_mode & 0o077 == 0
    assert database.db_path.stat().st_mode & 0o077 == 0
    started = utc_now() - timedelta(hours=2)
    event = database.add_sleep_event(SleepEventCreate(started_at=started, kind="nap", source="manual"))
    assert database.open_sleep_event().id == event.id
    ended = started + timedelta(hours=1)
    updated = database.update_sleep_event(event.id, SleepEventPatch(ended_at=ended, notes="Rested"))
    assert updated is not None and updated.notes == "Rested"
    assert database.open_sleep_event() is None

    frame = database.add_frame(b"not-a-real-jpeg", "image/jpeg", utc_now() - timedelta(days=40))
    estimate = database.retention_estimate(utc_now() - timedelta(days=30))
    assert estimate == {"frames": 1, "bytes": len(b"not-a-real-jpeg")}
    database.purge_frames_before(utc_now() - timedelta(days=30))
    retained = database.get_frame(frame.id)
    assert retained is not None and retained.image_available is False
    assert retained.sha256 == frame.sha256
    assert database.get_frame_path(frame.id) is None
    assert database.delete_sleep_event(event.id) is True


def test_database_context_releases_connection(tmp_path: Path) -> None:
    database = Database(tmp_path)
    connection = database._connect()

    with connection as active:
        assert active.execute("SELECT 1").fetchone()[0] == 1

    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")


def test_nearest_frames_ignores_images_from_an_unrelated_day(tmp_path: Path) -> None:
    database = Database(tmp_path)
    requested = utc_now()
    far = database.add_frame(b"far", "image/jpeg", requested - timedelta(hours=7))
    near = database.add_frame(b"near", "image/jpeg", requested - timedelta(minutes=8))

    nearest = database.nearest_frames(requested)

    assert [frame.id for frame in nearest] == [near.id]
    assert far.id not in {frame.id for frame in nearest}


def test_schema_v3_adds_locations_and_sleep_details_without_touching_frame_files(tmp_path: Path) -> None:
    frame_path = tmp_path / "frames" / "legacy.jpg"
    frame_path.parent.mkdir(parents=True)
    frame_path.write_bytes(b"preserve-this-image")
    with sqlite3.connect(tmp_path / "baby_monitor.sqlite3") as connection:
        connection.executescript(
            """
            CREATE TABLE frames (
                id TEXT PRIMARY KEY, captured_at TEXT NOT NULL, camera_entity_id TEXT,
                relative_path TEXT, mime_type TEXT NOT NULL, size_bytes INTEGER NOT NULL,
                sha256 TEXT NOT NULL, image_available INTEGER NOT NULL DEFAULT 1,
                label_json TEXT, provider TEXT, model TEXT, purged_at TEXT
            );
            CREATE TABLE sleep_events (
                id TEXT PRIMARY KEY, started_at TEXT NOT NULL, ended_at TEXT, kind TEXT NOT NULL,
                source TEXT NOT NULL, notes TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE cry_events (
                id TEXT PRIMARY KEY, detected_at TEXT NOT NULL, ended_at TEXT, source TEXT NOT NULL,
                confidence REAL, metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
            );
            PRAGMA user_version = 1;
            """
        )
    database = Database(tmp_path)
    assert database.SCHEMA_VERSION == 3
    assert frame_path.read_bytes() == b"preserve-this-image"
    with sqlite3.connect(database.db_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
        for table in ("frames", "sleep_events", "cry_events"):
            columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
            assert "location_id" in columns
        sleep_columns = {row[1] for row in connection.execute("PRAGMA table_info(sleep_events)")}
        assert "details_json" in sleep_columns


def test_sleep_events_cannot_overlap(tmp_path: Path) -> None:
    database = Database(tmp_path)
    start = utc_now() - timedelta(hours=2)
    first = database.add_sleep_event(
        SleepEventCreate(
            started_at=start,
            ended_at=start + timedelta(hours=1),
            kind="nap",
            source="manual",
        )
    )
    with pytest.raises(StorageError, match="overlaps"):
        database.add_sleep_event(
            SleepEventCreate(
                started_at=start + timedelta(minutes=30),
                ended_at=start + timedelta(hours=2),
                kind="nap",
                source="manual",
            )
        )

    other = database.add_sleep_event(
        SleepEventCreate(
            started_at=start + timedelta(hours=2),
            ended_at=start + timedelta(hours=3),
            kind="nap",
            source="manual",
        )
    )
    with pytest.raises(StorageError, match="overlaps"):
        database.update_sleep_event(other.id, SleepEventPatch(started_at=start + timedelta(minutes=30)))
    assert database.get_sleep_event(first.id) is not None


def test_fft_cry_analysis_and_ffmpeg_args_do_not_expose_stream_url() -> None:
    samples = [int(0.12 * 32767 * math.sin(2 * math.pi * 800 * index / SAMPLE_RATE)) for index in range(8000)]
    raw = b"".join(struct.pack("<h", sample) for sample in samples)
    metrics = analyze_pcm(raw)
    assert metrics.positive is True
    assert metrics.cry_core_db > metrics.low_db
    secret_url = "rtsp://user:very-secret@example.test/stream"
    assert secret_url not in " ".join(_input_args())
    whitelist = _input_args()[_input_args().index("-protocol_whitelist") + 1].split(",")
    assert "file" not in whitelist
    assert "crypto" not in whitelist
    assert b"option rtsp_transport tcp" in _ffconcat_source(secret_url)
    assert b"option rtsp_transport" not in _ffconcat_source("https://camera.example.test/live.mjpeg")


def test_ffmpeg_environment_does_not_inherit_application_secrets(monkeypatch) -> None:
    monkeypatch.setenv("SUPERVISOR_TOKEN", "supervisor-secret")
    monkeypatch.setenv("BABY_MONITOR_ADMIN_TOKEN", "standalone-secret")
    child = _subprocess_env()
    assert "SUPERVISOR_TOKEN" not in child
    assert "BABY_MONITOR_ADMIN_TOKEN" not in child


@pytest.mark.parametrize(
    "value",
    [
        "file:///etc/passwd",
        "ftp://camera.example.test/live",
        "rtsp://camera.example.test/live\nfile '/etc/passwd'",
        "rtsp:///missing-host",
    ],
)
def test_stream_urls_reject_local_protocols_and_ffconcat_injection(value: str) -> None:
    with pytest.raises(ValueError):
        SecretChanges(camera_stream_url=value)
    with pytest.raises(MediaError):
        _ffconcat_source(value)
