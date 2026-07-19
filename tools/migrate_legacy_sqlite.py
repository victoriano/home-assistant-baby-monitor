#!/usr/bin/env python3
"""Import the legacy crib-monitor SQLite schema into Baby Monitor.

The source database is always opened read-only. By default this command only
prints an import plan; pass ``--apply`` to write to a private target data
directory. The target must not live inside a Git working tree because it can
contain camera frames and household history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5
from zoneinfo import ZoneInfo

from baby_monitor.database import Database
from baby_monitor.models import VisionLabel, validate_location_id

IMPORT_NAMESPACE = UUID("130a5548-4e4e-4c00-955f-cb881a99a4fe")
REQUIRED_COLUMNS = {
    "crib_frames": {
        "id",
        "captured_at",
        "image",
        "image_mime",
        "model",
        "baby_visible",
        "in_crib",
        "asleep",
        "crying",
        "stirring",
        "state",
        "confidence",
        "source",
        "note",
        "raw_observation_json",
    },
    "cry_events": {
        "id",
        "started_at",
        "source",
        "threshold_db",
        "note",
        "created_at",
    },
    "manual_sleep_events": {
        "id",
        "started_at",
        "ended_at",
        "event_type",
        "note",
        "created_at",
    },
}


@dataclass
class ImportReport:
    source_frames: int = 0
    source_frame_bytes: int = 0
    source_cry_events: int = 0
    source_sleep_events: int = 0
    imported_frames: int = 0
    repaired_frame_files: int = 0
    imported_cry_events: int = 0
    imported_sleep_events: int = 0
    derived_sleep_events: int = 0
    replaced_sleep_events: int = 0
    repaired_cry_events: int = 0
    already_imported: int = 0


def _utc(value: str | None) -> datetime:
    if not value:
        return datetime.now(UTC)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _iso(value: str | None) -> str:
    return _utc(value).isoformat().replace("+00:00", "Z")


def _legacy_id(table: str, row_id: int, *parts: str) -> str:
    material = ":".join((table, str(row_id), *parts))
    return str(uuid5(IMPORT_NAMESPACE, material))


def _read_only(source: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{source.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _validate_source(connection: sqlite3.Connection) -> None:
    quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
    if quick_check != "ok":
        raise ValueError(f"source SQLite quick_check failed: {quick_check}")
    for table, required in REQUIRED_COLUMNS.items():
        columns = {str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")').fetchall()}
        missing = required - columns
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"legacy table {table!r} is missing columns: {names}")


def _inside_git_worktree(path: Path) -> bool:
    candidate = path.resolve()
    return any((parent / ".git").exists() for parent in (candidate, *candidate.parents))


def _prepare_private_target(path: Path) -> Path:
    if path.is_symlink():
        raise ValueError("target data directory must not be a symbolic link")
    target = path.resolve()
    if target in {Path(target.anchor), Path.home().resolve()}:
        raise ValueError("target must be a dedicated app data directory, not root or the home directory")
    if _inside_git_worktree(target):
        raise ValueError("target data directory must be outside every Git working tree")
    if target.exists():
        if not target.is_dir():
            raise ValueError("target data path must be a directory")
        allowed = {
            ".history-transfer-state.json",
            ".secret.key",
            ".transfers",
            "baby_monitor.sqlite3",
            "baby_monitor.sqlite3-shm",
            "baby_monitor.sqlite3-wal",
            "frames",
            "secrets.enc.json",
            "settings.json",
        }
        entries = {item.name for item in target.iterdir()}
        if entries and (not entries.issubset(allowed) or not entries.intersection({"baby_monitor.sqlite3", "frames"})):
            raise ValueError("existing target is not a recognized Baby Monitor data directory")
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(target, 0o700)
    return target


def _source_counts(connection: sqlite3.Connection) -> ImportReport:
    frames = connection.execute("SELECT COUNT(*), COALESCE(SUM(length(image)), 0) FROM crib_frames").fetchone()
    return ImportReport(
        source_frames=int(frames[0]),
        source_frame_bytes=int(frames[1]),
        source_cry_events=int(connection.execute("SELECT COUNT(*) FROM cry_events").fetchone()[0]),
        source_sleep_events=int(connection.execute("SELECT COUNT(*) FROM manual_sleep_events").fetchone()[0]),
    )


def _description(row: sqlite3.Row) -> str:
    note = str(row["note"] or "").strip()
    if note:
        return note[:500]
    state = str(row["state"] or "unknown").replace("_", " ")
    return f"Imported legacy observation: {state}"[:500]


def _tags(row: sqlite3.Row) -> list[str]:
    tags = ["legacy-import"]
    if bool(row["in_crib"]):
        tags.append("in-crib")
    if bool(row["crying"]):
        tags.append("crying")
    if bool(row["stirring"]):
        tags.append("stirring")
    legacy_state = str(row["state"] or "").strip().lower()
    if legacy_state and legacy_state not in tags:
        tags.append(legacy_state[:80])
    return tags


def _label(row: sqlite3.Row) -> VisionLabel:
    baby_present = bool(row["baby_visible"])
    legacy_state = str(row["state"] or "").strip().lower()
    if not baby_present or legacy_state == "out":
        state = "uncertain"
    elif bool(row["asleep"]):
        state = "asleep"
    else:
        state = "awake"
    confidence = min(1.0, max(0.0, float(row["confidence"] or 0.0)))
    return VisionLabel(
        baby_present=baby_present,
        state=state,
        confidence=confidence,
        description=_description(row),
        tags=_tags(row),
        in_crib=bool(row["in_crib"]),
        face_visible=_legacy_choice(_row_value(row, "face_visible")),
        head_side=_legacy_head_side(_row_value(row, "head_side")),
        body_position=str(_row_value(row, "body_position") or "unknown")[:80],
        clothing_items=_legacy_clothing_items(row),
        pacifier=_legacy_choice(_row_value(row, "pacifier")),
        mouth_open=_legacy_choice(_row_value(row, "mouth_open")),
    )


def _row_value(row: sqlite3.Row, name: str) -> Any:
    return row[name] if name in row.keys() else None  # noqa: SIM118 -- sqlite3.Row membership checks values


def _legacy_choice(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    return text if text in {"yes", "no", "unknown"} else "unknown"


def _legacy_head_side(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    aliases = {"up": "back", "back_up": "back", "down": "face_down", "front": "face_down"}
    text = aliases.get(text, text)
    return text if text in {"left", "right", "back", "face_down", "unknown"} else "unknown"


def _legacy_clothing_items(row: sqlite3.Row) -> list[str]:
    raw = _row_value(row, "visual_attributes_json")
    try:
        attributes = json.loads(str(raw or "{}"))
    except json.JSONDecodeError:
        attributes = {}
    values = attributes.get("clothing_items") if isinstance(attributes, dict) else None
    if not isinstance(values, list):
        values = []
    allowed = {
        "diaper_only",
        "short_sleeve_onesie",
        "long_sleeve_onesie",
        "sleep_sack",
        "blanket",
        "unknown",
    }
    normalized = [str(value or "").strip().lower() for value in values]
    normalized = [value for value in normalized if value in allowed]
    if normalized:
        return normalized[:5]
    legacy_type = str(_row_value(row, "clothing_type") or "").strip().lower()
    sleeves = str(_row_value(row, "clothing_sleeves") or "").strip().lower()
    if legacy_type in {"diaper", "diaper_only", "naked_diaper"}:
        return ["diaper_only"]
    if legacy_type in {"sleep_sack", "sack", "sleeping_bag"}:
        return ["sleep_sack"]
    if legacy_type in {"onesie", "pajama", "shirt"}:
        return ["long_sleeve_onesie" if sleeves == "long" else "short_sleeve_onesie"]
    return ["unknown"]


def _provider(row: sqlite3.Row) -> str | None:
    source = str(row["source"] or "").strip()
    if not source:
        return "legacy-import"
    return source.split(":", 1)[0][:120]


def _write_frame_file(path: Path, image: bytes, expected_sha: str) -> bool:
    if path.is_file():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
            raise ValueError(f"existing target frame has a different digest: {path.name}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(image)
        handle.flush()
        os.fsync(handle.fileno())
    return True


def _frame_rows(connection: sqlite3.Connection) -> Iterator[sqlite3.Row]:
    available = {str(row[1]) for row in connection.execute("PRAGMA table_info(crib_frames)")}
    optional = [
        "face_visible",
        "head_side",
        "body_position",
        "clothing_type",
        "clothing_sleeves",
        "pacifier",
        "mouth_open",
        "visual_attributes_json",
    ]
    optional_sql = ", ".join(name if name in available else f"NULL AS {name}" for name in optional)
    cursor = connection.execute(
        f"""
        SELECT id, captured_at, image, image_mime, model, baby_visible, in_crib,
               asleep, crying, stirring, state, confidence, source, note,
               raw_observation_json, {optional_sql}
        FROM crib_frames
        ORDER BY id
        """
    )
    while rows := cursor.fetchmany(100):
        yield from rows


def _import_frames(
    source: sqlite3.Connection,
    target: Database,
    report: ImportReport,
    location_id: str,
) -> None:
    extensions = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
    with sqlite3.connect(target.db_path, timeout=30) as destination:
        destination.execute("PRAGMA busy_timeout = 30000")
        for index, row in enumerate(_frame_rows(source), start=1):
            image = bytes(row["image"])
            mime_type = str(row["image_mime"] or "image/jpeg").lower()
            if mime_type not in extensions:
                raise ValueError(f"legacy frame {row['id']} has unsupported MIME type {mime_type!r}")
            digest = hashlib.sha256(image).hexdigest()
            captured_at = _iso(str(row["captured_at"]))
            frame_id = _legacy_id("crib_frames", int(row["id"]), captured_at, digest)
            captured = _utc(captured_at)
            relative = Path(f"{captured:%Y/%m/%d}") / f"{frame_id}{extensions[mime_type]}"
            target_file = target.frames_dir / relative
            known = destination.execute(
                "SELECT relative_path, sha256, image_available FROM frames WHERE id = ?",
                (frame_id,),
            ).fetchone()
            if known is not None:
                if known[1] != digest:
                    raise ValueError(f"target frame ID collision for legacy frame {row['id']}")
                if bool(known[2]) and not target_file.is_file() and _write_frame_file(target_file, image, digest):
                    report.repaired_frame_files += 1
                label = _label(row)
                destination.execute(
                    """UPDATE frames
                       SET label_json = ?, provider = ?, model = ?, location_id = ?
                       WHERE id = ?""",
                    (
                        label.model_dump_json(),
                        _provider(row),
                        str(row["model"] or "")[:120] or None,
                        location_id,
                        frame_id,
                    ),
                )
                report.already_imported += 1
                continue
            _write_frame_file(target_file, image, digest)
            label = _label(row)
            destination.execute(
                """
                INSERT INTO frames
                (id, captured_at, camera_entity_id, location_id, relative_path, mime_type,
                 size_bytes, sha256, image_available, label_json, provider, model)
                VALUES (?, ?, NULL, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                """,
                (
                    frame_id,
                    captured_at,
                    location_id,
                    str(relative),
                    mime_type,
                    len(image),
                    digest,
                    label.model_dump_json(),
                    _provider(row),
                    str(row["model"] or "")[:120] or None,
                ),
            )
            report.imported_frames += 1
            if index % 100 == 0:
                destination.commit()


@dataclass(frozen=True)
class _LegacySleepEvent:
    started_at: datetime
    ended_at: datetime
    kind: str
    source: str
    note: str = ""
    manual: bool = False

    @property
    def minutes(self) -> int:
        return max(0, int((self.ended_at - self.started_at).total_seconds() // 60))


def _legacy_sleep_kind(started_at: datetime, ended_at: datetime, timezone_name: str) -> str:
    tz = ZoneInfo(timezone_name)
    local_start = started_at.astimezone(tz)
    local_end = ended_at.astimezone(tz)
    minutes = int((ended_at - started_at).total_seconds() // 60)
    if minutes >= 300:
        return "night"
    if local_start.hour >= 19 or (local_end.hour, local_end.minute) <= (8, 30):
        return "night"
    if local_start.hour < 8 and (local_end.hour, local_end.minute) <= (9, 45) and minutes >= 45:
        return "night"
    if minutes >= 210 and (
        (local_start.hour, local_start.minute) >= (17, 30) or (local_end.hour, local_end.minute) <= (9, 30)
    ):
        return "night"
    return "nap"


def _automatic_sleep_events(source: sqlite3.Connection, timezone_name: str) -> list[_LegacySleepEvent]:
    tz = ZoneInfo(timezone_name)
    cry_columns = {str(row[1]) for row in source.execute("PRAGMA table_info(cry_events)")}
    query = "SELECT raw_observation_json, captured_at AS observed_at FROM crib_frames"
    if "raw_observation_json" in cry_columns:
        query += " UNION ALL SELECT raw_observation_json, started_at AS observed_at FROM cry_events"
    query += " ORDER BY observed_at"
    rows = source.execute(query).fetchall()
    slots: dict[datetime, tuple[tuple[int, float, datetime], dict[str, Any]]] = {}
    for row in rows:
        try:
            observation = json.loads(str(row["raw_observation_json"] or "{}"))
        except json.JSONDecodeError:
            continue
        if not isinstance(observation, dict):
            continue
        if not {"timestamp", "in_crib", "asleep"}.issubset(observation):
            continue
        captured = _utc(str(observation.get("timestamp") or row["observed_at"])).astimezone(tz)
        slot = captured.replace(minute=(captured.minute // 5) * 5, second=0, microsecond=0)
        asleep = bool(observation.get("asleep"))
        crying = bool(observation.get("crying"))
        stirring = bool(observation.get("stirring"))
        baby_visible = bool(observation.get("baby_visible", True))
        in_crib = bool(observation.get("in_crib"))
        observation["_legacy_state"] = (
            "crying"
            if crying
            else "sleep"
            if asleep
            else "out"
            if not in_crib or not baby_visible
            else "stirring"
            if stirring
            else "awake"
        )
        if crying:
            priority = 4
        elif asleep:
            priority = 3
        elif stirring:
            priority = 2
        elif baby_visible and in_crib:
            priority = 1
        else:
            priority = 0
        rank = (priority, float(observation.get("confidence") or 0), captured)
        if slot not in slots or rank > slots[slot][0]:
            slots[slot] = (rank, observation)

    source_segments: list[_LegacySleepEvent] = []
    current: tuple[datetime, datetime, str, str, str] | None = None

    def append_current(run: tuple[datetime, datetime, str, str, str]) -> None:
        started_at, ended_at, state, run_source, _ = run
        minutes = int((ended_at - started_at).total_seconds() // 60)
        if state == "sleep" and minutes >= 10:
            source_segments.append(
                _LegacySleepEvent(
                    started_at, ended_at, _legacy_sleep_kind(started_at, ended_at, timezone_name), run_source
                )
            )
        elif state == "out" and minutes >= 5:
            source_segments.append(_LegacySleepEvent(started_at, ended_at, "awake", run_source))

    for slot, (_, observation) in sorted(slots.items()):
        state = str(observation.get("_legacy_state") or "awake")
        source_name = str(observation.get("source") or "legacy-vision")
        sleep_day = (slot - timedelta(days=1) if slot.hour < 8 else slot).date().isoformat()
        started_at = slot.astimezone(UTC)
        ended_at = (slot + timedelta(minutes=5)).astimezone(UTC)
        if (
            current is not None
            and started_at == current[1]
            and state == current[2]
            and source_name == current[3]
            and sleep_day == current[4]
        ):
            current = (current[0], ended_at, state, source_name, sleep_day)
        else:
            if current is not None:
                append_current(current)
            current = (started_at, ended_at, state, source_name, sleep_day)
    if current is not None:
        append_current(current)

    merged: list[_LegacySleepEvent] = []
    for event in source_segments:
        if merged and merged[-1].kind == event.kind and event.started_at <= merged[-1].ended_at + timedelta(minutes=10):
            previous = merged[-1]
            merged[-1] = _LegacySleepEvent(
                previous.started_at,
                max(previous.ended_at, event.ended_at),
                previous.kind,
                previous.source if previous.source == event.source else "mixed",
            )
        else:
            merged.append(event)
    return merged


def _manual_sleep_events(source: sqlite3.Connection) -> list[_LegacySleepEvent]:
    available = {str(row[1]) for row in source.execute("PRAGMA table_info(manual_sleep_events)")}
    source_sql = "source" if "source" in available else "'manual' AS source"
    confidence_sql = "confidence" if "confidence" in available else "1.0 AS confidence"
    rows = source.execute(
        f"""SELECT id, started_at, ended_at, event_type, note, {source_sql}, {confidence_sql}
            FROM manual_sleep_events ORDER BY started_at, id"""
    ).fetchall()
    keep = [True] * len(rows)
    for left_index, left in enumerate(rows):
        if not keep[left_index] or str(left["event_type"]) not in {"nap", "night"}:
            continue
        for right_index in range(left_index + 1, len(rows)):
            right = rows[right_index]
            if not keep[right_index] or str(right["event_type"]) not in {"nap", "night"}:
                continue
            if str(left["event_type"]) == str(right["event_type"]):
                continue
            left_start = _utc(str(left["started_at"]))
            right_start = _utc(str(right["started_at"]))
            left_end = _utc(str(left["ended_at"]))
            right_end = _utc(str(right["ended_at"]))
            if abs(left_start - right_start) > timedelta(minutes=1) or not (
                left_start < right_end and right_start < left_end
            ):
                continue
            loser = left_index if int(left["id"]) < int(right["id"]) else right_index
            keep[loser] = False
            if loser == left_index:
                break
    rows = [row for row, should_keep in zip(rows, keep, strict=True) if should_keep]
    events = [
        _LegacySleepEvent(
            _utc(str(row["started_at"])),
            _utc(str(row["ended_at"])),
            str(row["event_type"] or "unknown").strip().lower(),
            str(row["source"] or "manual"),
            str(row["note"] or "")[:1000],
            True,
        )
        for row in rows
        if _utc(str(row["ended_at"])) > _utc(str(row["started_at"]))
    ]
    merged: list[_LegacySleepEvent] = []
    for event in sorted(events, key=lambda item: (item.kind, item.started_at, item.ended_at)):
        if merged and merged[-1].kind == event.kind and event.started_at <= merged[-1].ended_at:
            previous = merged[-1]
            merged[-1] = _LegacySleepEvent(
                min(previous.started_at, event.started_at),
                max(previous.ended_at, event.ended_at),
                event.kind,
                "manual",
                previous.note or event.note,
                True,
            )
        else:
            merged.append(event)
    return sorted(merged, key=lambda item: item.started_at)


def _subtract_sleep_boundary(event: _LegacySleepEvent, boundary: _LegacySleepEvent) -> list[_LegacySleepEvent]:
    if event.ended_at <= boundary.started_at or event.started_at >= boundary.ended_at:
        return [event]
    parts: list[_LegacySleepEvent] = []
    if event.started_at < boundary.started_at:
        parts.append(
            _LegacySleepEvent(event.started_at, boundary.started_at, event.kind, event.source, event.note, event.manual)
        )
    if boundary.ended_at < event.ended_at:
        parts.append(
            _LegacySleepEvent(boundary.ended_at, event.ended_at, event.kind, event.source, event.note, event.manual)
        )
    return [part for part in parts if part.minutes >= 10]


def _clip_sleep_to_manual_boundary(event: _LegacySleepEvent, boundary: _LegacySleepEvent) -> _LegacySleepEvent:
    started_at = max(event.started_at, boundary.started_at)
    ended_at = min(event.ended_at, boundary.ended_at)
    manual_fully_inside = boundary.started_at >= event.started_at and boundary.ended_at <= event.ended_at
    if manual_fully_inside:
        source = boundary.source
        manual = True
    elif started_at == boundary.started_at or ended_at == boundary.ended_at:
        source = f"manual-boundary:{event.source}"
        manual = True
    else:
        source = event.source
        manual = event.manual
    return _LegacySleepEvent(started_at, ended_at, boundary.kind, source, event.note, manual)


def _reconciled_sleep_events(source: sqlite3.Connection, timezone_name: str) -> list[_LegacySleepEvent]:
    automatic = _automatic_sleep_events(source, timezone_name)
    manual = _manual_sleep_events(source)
    boundaries = [event for event in manual if event.kind in {"nap", "night"}]
    awake = [event for event in manual if event.kind == "awake"]
    if not boundaries:
        result = automatic
    else:
        result: list[_LegacySleepEvent] = []
        for event in automatic:
            fragments = [event]
            for boundary in boundaries:
                fragments = [part for fragment in fragments for part in _subtract_sleep_boundary(fragment, boundary)]
            result.extend(part for part in fragments if part.minutes >= 10)
        for boundary in boundaries:
            clipped = [
                _clip_sleep_to_manual_boundary(event, boundary)
                for event in automatic
                if event.started_at < boundary.ended_at and boundary.started_at < event.ended_at
            ]
            clipped = [event for event in clipped if event.minutes >= 10]
            if not clipped:
                result.append(boundary)
                continue
            clipped.sort(key=lambda item: item.started_at)
            if int((clipped[0].started_at - boundary.started_at).total_seconds() // 60) >= 10:
                result.append(
                    _LegacySleepEvent(
                        boundary.started_at, clipped[0].started_at, boundary.kind, "manual", boundary.note, True
                    )
                )
            result.extend(clipped)
            if int((boundary.ended_at - clipped[-1].ended_at).total_seconds() // 60) >= 10:
                result.append(
                    _LegacySleepEvent(
                        clipped[-1].ended_at, boundary.ended_at, boundary.kind, "manual", boundary.note, True
                    )
                )

    for awake_event in awake:
        result = [part for event in result for part in _subtract_sleep_boundary(event, awake_event)]

    merged: list[_LegacySleepEvent] = []
    for event in sorted(result, key=lambda item: item.started_at):
        if (
            merged
            and merged[-1].kind == event.kind
            and not merged[-1].manual
            and not event.manual
            and event.started_at <= merged[-1].ended_at + timedelta(minutes=10)
        ):
            previous = merged[-1]
            merged[-1] = _LegacySleepEvent(
                previous.started_at,
                max(previous.ended_at, event.ended_at),
                previous.kind,
                previous.source if previous.source == event.source else "mixed",
                previous.note or event.note,
                False,
            )
        else:
            merged.append(event)
    return [event for event in merged if event.kind in {"nap", "night"} and event.minutes >= 10]


def _old_manual_group_ids(source: sqlite3.Connection) -> list[str]:
    rows = source.execute(
        """SELECT id, started_at, ended_at, event_type, note, created_at
           FROM manual_sleep_events ORDER BY id"""
    ).fetchall()
    groups: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: _utc(str(item["started_at"]))):
        started = _utc(str(row["started_at"]))
        ended = _utc(str(row["ended_at"]))
        if groups and started < groups[-1]["ended"]:
            groups[-1]["rows"].append(row)
            groups[-1]["started"] = min(groups[-1]["started"], started)
            groups[-1]["ended"] = max(groups[-1]["ended"], ended)
        else:
            groups.append({"rows": [row], "started": started, "ended": ended})
    return [
        _legacy_id(
            "manual_sleep_event_group",
            0,
            ",".join(str(row["id"]) for row in group["rows"]),
            group["started"].isoformat().replace("+00:00", "Z"),
            group["ended"].isoformat().replace("+00:00", "Z"),
        )
        for group in groups
    ]


def _import_sleep_events(
    source: sqlite3.Connection,
    target: Database,
    report: ImportReport,
    location_id: str,
    timezone_name: str,
) -> None:
    events = _reconciled_sleep_events(source, timezone_name)
    report.derived_sleep_events = len(events)
    prepared = []
    for event in events:
        started_at = event.started_at.astimezone(UTC).isoformat().replace("+00:00", "Z")
        ended_at = event.ended_at.astimezone(UTC).isoformat().replace("+00:00", "Z")
        event_id = _legacy_id("derived_sleep_event", 0, started_at, ended_at, event.kind, event.source)
        prepared.append((event, event_id, started_at, ended_at))
    with sqlite3.connect(target.db_path, timeout=30) as destination:
        destination.execute("PRAGMA busy_timeout = 30000")
        old_ids = _old_manual_group_ids(source)
        if old_ids:
            placeholders = ",".join("?" for _ in old_ids)
            removed = destination.execute(
                f"DELETE FROM sleep_events WHERE id IN ({placeholders})",
                old_ids,
            ).rowcount
            report.replaced_sleep_events += removed
        current_ids = [item[1] for item in prepared]
        stale_query = "DELETE FROM sleep_events WHERE location_id = ? AND notes LIKE 'legacy-derived:%'"
        stale_parameters: list[Any] = [location_id]
        if current_ids:
            stale_query += f" AND id NOT IN ({','.join('?' for _ in current_ids)})"
            stale_parameters.extend(current_ids)
        report.replaced_sleep_events += destination.execute(stale_query, stale_parameters).rowcount
        for event, event_id, started_at, ended_at in prepared:
            note = f"legacy-derived:{event.source}"
            if event.note:
                note = f"{note}; {event.note}"[:1000]
            cursor = destination.execute(
                """INSERT OR IGNORE INTO sleep_events
                   (id, started_at, ended_at, kind, source, notes, location_id, created_at)
                   VALUES (?, ?, ?, ?, 'import', ?, ?, ?)""",
                (event_id, started_at, ended_at, event.kind, note, location_id, started_at),
            )
            if cursor.rowcount:
                report.imported_sleep_events += 1
            else:
                report.already_imported += 1


def _import_cry_events(
    source: sqlite3.Connection,
    target: Database,
    report: ImportReport,
    location_id: str,
) -> None:
    rows = source.execute(
        """
        SELECT id, started_at, source, threshold_db, note, created_at
        FROM cry_events ORDER BY id
        """
    ).fetchall()
    with sqlite3.connect(target.db_path, timeout=30) as destination:
        for row in rows:
            detected = _utc(str(row["started_at"]))
            created = _utc(str(row["created_at"] or row["started_at"]))
            ended = max(created, detected + timedelta(microseconds=1))
            detected_at = detected.isoformat().replace("+00:00", "Z")
            ended_at = ended.isoformat().replace("+00:00", "Z")
            event_id = _legacy_id("cry_events", int(row["id"]), detected_at)
            metadata: dict[str, Any] = {
                "legacy_source": str(row["source"] or "")[:200],
            }
            if row["threshold_db"] is not None:
                metadata["threshold_db"] = float(row["threshold_db"])
            if row["note"]:
                metadata["note"] = str(row["note"])[:1000]
            cursor = destination.execute(
                """
                INSERT OR IGNORE INTO cry_events
                (id, detected_at, ended_at, source, confidence, metadata_json, location_id, created_at)
                VALUES (?, ?, ?, 'import', NULL, ?, ?, ?)
                """,
                (
                    event_id,
                    detected_at,
                    ended_at,
                    json.dumps(metadata, separators=(",", ":")),
                    location_id,
                    created.isoformat().replace("+00:00", "Z"),
                ),
            )
            if cursor.rowcount:
                report.imported_cry_events += 1
            else:
                repaired = destination.execute(
                    "UPDATE cry_events SET ended_at = ? WHERE id = ? AND ended_at IS NULL",
                    (ended_at, event_id),
                )
                report.repaired_cry_events += repaired.rowcount
                report.already_imported += 1


def migrate(
    source_path: Path,
    target_path: Path | None,
    *,
    apply: bool,
    location_id: str = "home",
    timezone_name: str = "Europe/Madrid",
) -> ImportReport:
    validate_location_id(location_id)
    ZoneInfo(timezone_name)
    if not source_path.is_file():
        raise ValueError(f"source database does not exist: {source_path}")
    with _read_only(source_path) as source:
        _validate_source(source)
        report = _source_counts(source)
        if not apply:
            return report
        if target_path is None:
            raise ValueError("--target is required with --apply")
        target = Database(_prepare_private_target(target_path))
        _import_frames(source, target, report, location_id)
        _import_sleep_events(source, target, report, location_id, timezone_name)
        _import_cry_events(source, target, report, location_id)
        if not target.ready():
            raise ValueError("target SQLite quick_check failed after import")
        return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Legacy SQLite database")
    parser.add_argument(
        "--target",
        type=Path,
        help="Private Baby Monitor data directory, required with --apply",
    )
    parser.add_argument(
        "--location-id",
        default="home",
        help="Lowercase home/location identifier attached to imported history (default: home)",
    )
    parser.add_argument(
        "--timezone",
        default="Europe/Madrid",
        help="IANA timezone used to classify historical naps and night sleep",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the import (without this flag only a plan is printed)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = migrate(
            args.source,
            args.target,
            apply=args.apply,
            location_id=args.location_id,
            timezone_name=args.timezone,
        )
    except (OSError, sqlite3.Error, ValueError) as exc:
        print(f"Migration failed: {exc}", file=sys.stderr)
        return 1
    payload = asdict(report)
    payload["mode"] = "applied" if args.apply else "dry-run"
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
