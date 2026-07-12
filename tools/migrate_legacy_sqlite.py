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
            ".secret.key",
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
    )


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
    cursor = connection.execute(
        """
        SELECT id, captured_at, image, image_mime, model, baby_visible, in_crib,
               asleep, crying, stirring, state, confidence, source, note,
               raw_observation_json
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


def _import_sleep_events(
    source: sqlite3.Connection,
    target: Database,
    report: ImportReport,
    location_id: str,
) -> None:
    rows = source.execute(
        """
        SELECT id, started_at, ended_at, event_type, note, created_at
        FROM manual_sleep_events ORDER BY id
        """
    ).fetchall()
    merged_rows: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: _utc(str(item["started_at"]))):
        started = _utc(str(row["started_at"]))
        ended = _utc(str(row["ended_at"]))
        if merged_rows and started < merged_rows[-1]["ended"]:
            group = merged_rows[-1]
            group["rows"].append(row)
            group["started"] = min(group["started"], started)
            group["ended"] = max(group["ended"], ended)
            if int(row["id"]) > int(group["canonical"]["id"]):
                group["canonical"] = row
            continue
        merged_rows.append({"rows": [row], "canonical": row, "started": started, "ended": ended})

    with sqlite3.connect(target.db_path, timeout=30) as destination:
        for group in merged_rows:
            canonical = group["canonical"]
            started_at = group["started"].isoformat().replace("+00:00", "Z")
            ended_at = group["ended"].isoformat().replace("+00:00", "Z")
            source_ids = ",".join(str(row["id"]) for row in group["rows"])
            event_id = _legacy_id("manual_sleep_event_group", 0, source_ids, started_at, ended_at)
            kind = str(canonical["event_type"] or "unknown").strip().lower()
            kind = kind if kind in {"nap", "night"} else "unknown"
            cursor = destination.execute(
                """
                INSERT OR IGNORE INTO sleep_events
                (id, started_at, ended_at, kind, source, notes, location_id, created_at)
                VALUES (?, ?, ?, ?, 'import', ?, ?, ?)
                """,
                (
                    event_id,
                    started_at,
                    ended_at,
                    kind,
                    str(canonical["note"] or "")[:1000] or None,
                    location_id,
                    _iso(str(canonical["created_at"] or canonical["started_at"])),
                ),
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


def migrate(source_path: Path, target_path: Path | None, *, apply: bool, location_id: str = "home") -> ImportReport:
    validate_location_id(location_id)
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
        _import_sleep_events(source, target, report, location_id)
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
        "--apply",
        action="store_true",
        help="Perform the import (without this flag only a plan is printed)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = migrate(args.source, args.target, apply=args.apply, location_id=args.location_id)
    except (OSError, sqlite3.Error, ValueError) as exc:
        print(f"Migration failed: {exc}", file=sys.stderr)
        return 1
    payload = asdict(report)
    payload["mode"] = "applied" if args.apply else "dry-run"
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
