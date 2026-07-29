from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from baby_monitor_edge_ml.dataset import (
    CaptureWindow,
    apply_capture_windows,
    chronological_group_split,
    load_examples,
    prepare_dataset,
    read_manifest,
    write_temporal_label_flips,
)


def _database(tmp_path: Path) -> tuple[Path, Path]:
    frames = tmp_path / "frames"
    frames.mkdir()
    database = tmp_path / "baby_monitor.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """CREATE TABLE frames (
                id TEXT PRIMARY KEY,
                captured_at TEXT NOT NULL,
                location_id TEXT NOT NULL,
                relative_path TEXT,
                sha256 TEXT NOT NULL,
                provider TEXT,
                model TEXT,
                image_available INTEGER NOT NULL,
                label_json TEXT
            )"""
        )
    return database, frames


def _add_frame(
    database: Path,
    frames: Path,
    *,
    frame_id: str,
    day: str,
    location: str = "home",
    baby_present: bool = True,
    state: str = "asleep",
    pacifier: str = "no",
    face_visible: str = "yes",
    confidence: float = 0.95,
    tags: list[str] | None = None,
    content: bytes | None = None,
    time: str = "12:00:00",
) -> None:
    image = content or frame_id.encode()
    relative = f"{day}/{frame_id}.jpg"
    target = frames / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(image)
    label = {
        "baby_present": baby_present,
        "state": state,
        "pacifier": pacifier,
        "face_visible": face_visible,
        "confidence": confidence,
        "tags": tags or [],
    }
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO frames VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)",
            (
                frame_id,
                f"{day}T{time}Z",
                location,
                relative,
                hashlib.sha256(image).hexdigest(),
                "teacher",
                "vision",
                json.dumps(label),
            ),
        )


def test_load_examples_masks_conditional_labels_and_deduplicates(tmp_path: Path) -> None:
    database, frames = _database(tmp_path)
    _add_frame(database, frames, frame_id="awake", day="2026-01-01", state="awake", pacifier="yes")
    _add_frame(
        database,
        frames,
        frame_id="absent",
        day="2026-01-02",
        baby_present=False,
        state="uncertain",
        pacifier="no",
    )
    _add_frame(
        database,
        frames,
        frame_id="hidden",
        day="2026-01-03",
        face_visible="no",
        pacifier="no",
    )
    _add_frame(
        database,
        frames,
        frame_id="duplicate",
        day="2026-01-04",
        content=b"awake",
    )
    _add_frame(
        database,
        frames,
        frame_id="unusable",
        day="2026-01-05",
        tags=["image_unusable"],
    )

    result = load_examples(database, frames)

    assert len(result.examples) == 3
    by_id = {example.frame_id: example for example in result.examples}
    assert by_id["awake"].awake_mask == 1
    assert by_id["awake"].pacifier_mask == 1
    assert by_id["absent"].awake_mask == 0
    assert by_id["absent"].pacifier_mask == 0
    assert by_id["hidden"].pacifier_mask == 0
    assert result.skipped == {"duplicate_sha256": 1, "image_unusable": 1}


def test_chronological_split_keeps_location_days_together(tmp_path: Path) -> None:
    database, frames = _database(tmp_path)
    for location in ("madrid", "granada"):
        for index in range(10):
            day = f"2026-01-{index + 1:02d}"
            _add_frame(database, frames, frame_id=f"{location}-{index}", day=day, location=location)
    loaded = load_examples(database, frames)

    split = chronological_group_split(loaded.examples, validation_fraction=0.2, test_fraction=0.2)

    for location in ("madrid", "granada"):
        by_day: dict[str, set[str]] = {}
        for example in split:
            if example.location_id == location:
                by_day.setdefault(example.day, set()).add(example.split)
        assert all(len(values) == 1 for values in by_day.values())
        assert [next(iter(by_day[day])) for day in sorted(by_day)] == [
            *(["train"] * 6),
            *(["validation"] * 2),
            *(["test"] * 2),
        ]


def test_prepare_writes_manifest_summary_and_review_queue(tmp_path: Path) -> None:
    database, frames = _database(tmp_path)
    for index in range(6):
        _add_frame(
            database,
            frames,
            frame_id=f"frame-{index}",
            day=f"2026-01-{index + 1:02d}",
            state="awake" if index % 2 else "asleep",
            pacifier="yes" if index % 2 else "no",
        )
    output = tmp_path / "dataset"

    summary = prepare_dataset(database, frames, output)

    assert summary["examples"] == 6
    assert len(read_manifest(output / "manifest.csv")) == 6
    assert json.loads((output / "summary.json").read_text())["examples"] == 6
    with (output / "review_queue.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["frame_id"] for row in rows} == {f"frame-{index}" for index in range(6)}


def test_split_requires_three_days_per_location(tmp_path: Path) -> None:
    database, frames = _database(tmp_path)
    _add_frame(database, frames, frame_id="one", day="2026-01-01")
    _add_frame(database, frames, frame_id="two", day="2026-01-02")

    with pytest.raises(ValueError, match="at least three"):
        chronological_group_split(load_examples(database, frames).examples)


def test_capture_window_excludes_obsolete_camera_geometry(tmp_path: Path) -> None:
    database, frames = _database(tmp_path)
    _add_frame(database, frames, frame_id="old", day="2026-01-01")
    _add_frame(database, frames, frame_id="current", day="2026-01-02")
    loaded = load_examples(database, frames)

    examples, skipped = apply_capture_windows(
        loaded.examples,
        {"home": CaptureWindow(start="2026-01-02T00:00:00Z")},
    )

    assert [example.frame_id for example in examples] == ["current"]
    assert skipped == {"outside_capture_window": 1}


def test_temporal_label_flips_are_written_for_manual_review(tmp_path: Path) -> None:
    database, frames = _database(tmp_path)
    _add_frame(
        database,
        frames,
        frame_id="before",
        day="2026-01-01",
        time="12:00:00",
        state="asleep",
        pacifier="no",
    )
    _add_frame(
        database,
        frames,
        frame_id="after",
        day="2026-01-01",
        time="12:05:00",
        state="awake",
        pacifier="yes",
    )
    output = tmp_path / "flips.csv"

    counts = write_temporal_label_flips(load_examples(database, frames).examples, output)

    assert counts == {"awake": 1, "pacifier": 1}
    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["task"] for row in rows] == ["awake", "pacifier"]
