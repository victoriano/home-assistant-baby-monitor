from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from baby_monitor.database import Database

from tools.migrate_legacy_sqlite import _prepare_private_target, migrate

LEGACY_SCHEMA = """
CREATE TABLE crib_frames (
    id INTEGER PRIMARY KEY,
    captured_at TEXT NOT NULL,
    image BLOB NOT NULL,
    image_mime TEXT NOT NULL,
    model TEXT,
    baby_visible INTEGER NOT NULL,
    in_crib INTEGER NOT NULL,
    asleep INTEGER NOT NULL,
    crying INTEGER NOT NULL,
    stirring INTEGER NOT NULL,
    state TEXT NOT NULL,
    confidence REAL NOT NULL,
    source TEXT NOT NULL,
    note TEXT NOT NULL,
    raw_observation_json TEXT NOT NULL
);
CREATE TABLE cry_events (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,
    source TEXT NOT NULL,
    threshold_db REAL,
    note TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE manual_sleep_events (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT NOT NULL,
    event_type TEXT NOT NULL,
    note TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _legacy_database(path: Path) -> Path:
    with sqlite3.connect(path) as connection:
        connection.executescript(LEGACY_SCHEMA)
        connection.executemany(
            "INSERT INTO crib_frames VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    1,
                    "2026-01-01T01:00:00Z",
                    b"first-image",
                    "image/jpeg",
                    "vision-model",
                    1,
                    1,
                    1,
                    0,
                    0,
                    "sleep",
                    0.91,
                    "provider:model",
                    "Sleeping",
                    "{}",
                ),
                (
                    2,
                    "2026-01-01T01:05:00Z",
                    b"second-image",
                    "image/jpeg",
                    "vision-model",
                    0,
                    0,
                    0,
                    0,
                    0,
                    "out",
                    0.8,
                    "provider:model",
                    "",
                    "{}",
                ),
            ],
        )
        connection.execute(
            "INSERT INTO cry_events VALUES (?, ?, ?, ?, ?, ?)",
            (1, "2026-01-01T02:00:00Z", "legacy:cry", -38.0, "", "2026-01-01T02:00:01Z"),
        )
        connection.executemany(
            "INSERT INTO manual_sleep_events VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    1,
                    "2026-01-01T01:00:00Z",
                    "2026-01-01T01:45:00Z",
                    "nap",
                    "Imported nap",
                    "2026-01-01T01:46:00Z",
                ),
                (
                    2,
                    "2026-01-01T01:30:00Z",
                    "2026-01-01T02:00:00Z",
                    "nap",
                    "Later correction",
                    "2026-01-01T02:01:00Z",
                ),
            ],
        )
    return path


def test_migration_dry_run_does_not_create_target(tmp_path: Path) -> None:
    source = _legacy_database(tmp_path / "legacy.sqlite3")
    target = tmp_path / "private" / "data"

    report = migrate(source, target, apply=False)

    assert report.source_frames == 2
    assert report.source_frame_bytes == len(b"first-image") + len(b"second-image")
    assert report.source_cry_events == 1
    assert report.source_sleep_events == 2
    assert not target.exists()


def test_migration_is_idempotent_and_preserves_private_images(tmp_path: Path) -> None:
    source = _legacy_database(tmp_path / "legacy.sqlite3")
    target = tmp_path / "private" / "data"

    first = migrate(source, target, apply=True, location_id="madrid")
    second = migrate(source, target, apply=True, location_id="madrid")
    database = Database(target)
    frames, total = database.list_frames()
    sleep, sleep_total = database.list_sleep_events()
    cry, cry_total = database.list_cry_events()

    assert first.imported_frames == 2
    assert first.imported_sleep_events == 1
    assert first.imported_cry_events == 1
    assert second.imported_frames == 0
    assert second.imported_sleep_events == 0
    assert second.imported_cry_events == 0
    assert second.repaired_cry_events == 0
    assert second.already_imported == 4
    assert total == 2
    assert sleep_total == 1
    assert cry_total == 1
    assert len(list((target / "frames").rglob("*.jpg"))) == 2
    assert {item.label.state for item in frames if item.label} == {"asleep", "uncertain"}
    assert {item.location_id for item in frames} == {"madrid"}
    assert sleep[0].location_id == "madrid"
    assert cry[0].location_id == "madrid"
    assert sleep[0].source == "import"
    assert sleep[0].started_at.isoformat() == "2026-01-01T01:00:00+00:00"
    assert sleep[0].ended_at is not None
    assert sleep[0].ended_at.isoformat() == "2026-01-01T02:00:00+00:00"
    assert cry[0].source == "import"
    assert cry[0].ended_at is not None


def test_migration_refuses_root_home_and_unrecognized_existing_directories(tmp_path: Path) -> None:
    for unsafe in (Path("/"), Path.home()):
        with pytest.raises(ValueError, match="dedicated app data directory"):
            _prepare_private_target(unsafe)

    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "household.txt").write_text("do not touch")
    original_mode = unrelated.stat().st_mode
    with pytest.raises(ValueError, match="not a recognized"):
        _prepare_private_target(unrelated)
    assert unrelated.stat().st_mode == original_mode
