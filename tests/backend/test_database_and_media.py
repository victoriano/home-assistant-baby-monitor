from __future__ import annotations

import math
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
