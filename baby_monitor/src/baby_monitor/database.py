from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import threading
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import (
    CryEvent,
    CryEventCreate,
    FrameRecord,
    SleepDetails,
    SleepEvent,
    SleepEventCreate,
    SleepEventPatch,
    VisionLabel,
)


class StorageError(RuntimeError):
    pass


class _ClosingConnection(sqlite3.Connection):
    """Commit or roll back a context block, then release its file descriptors."""

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        try:
            return bool(super().__exit__(exc_type, exc_value, traceback))
        finally:
            self.close()


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _iso(value: datetime) -> str:
    return _as_utc(value).isoformat().replace("+00:00", "Z")


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class Database:
    SCHEMA_VERSION = 3

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.frames_dir = data_dir / "frames"
        self.db_path = data_dir / "baby_monitor.sqlite3"
        self._lock = threading.RLock()
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.frames_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        with suppress(PermissionError):
            os.chmod(self.data_dir, 0o700)
            os.chmod(self.frames_dir, 0o700)
        self._migrate()
        self._secure_database_files()

    def _secure_database_files(self) -> None:
        for path in (
            self.db_path,
            self.db_path.with_name(f"{self.db_path.name}-wal"),
            self.db_path.with_name(f"{self.db_path.name}-shm"),
        ):
            with suppress(FileNotFoundError, PermissionError):
                os.chmod(path, 0o600)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.db_path,
            timeout=10,
            factory=_ClosingConnection,
        )
        self._secure_database_files()
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _migrate(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS frames (
                    id TEXT PRIMARY KEY,
                    captured_at TEXT NOT NULL,
                    camera_entity_id TEXT,
                    location_id TEXT NOT NULL DEFAULT 'home',
                    relative_path TEXT,
                    mime_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
                    sha256 TEXT NOT NULL,
                    image_available INTEGER NOT NULL DEFAULT 1,
                    label_json TEXT,
                    provider TEXT,
                    model TEXT,
                    purged_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_frames_captured_at ON frames(captured_at DESC);

                CREATE TABLE IF NOT EXISTS sleep_events (
                    id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    kind TEXT NOT NULL,
                    source TEXT NOT NULL,
                    notes TEXT,
                    details_json TEXT NOT NULL DEFAULT '{}',
                    location_id TEXT NOT NULL DEFAULT 'home',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_sleep_started_at ON sleep_events(started_at DESC);

                CREATE TABLE IF NOT EXISTS cry_events (
                    id TEXT PRIMARY KEY,
                    detected_at TEXT NOT NULL,
                    ended_at TEXT,
                    source TEXT NOT NULL,
                    confidence REAL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    location_id TEXT NOT NULL DEFAULT 'home',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_cry_detected_at ON cry_events(detected_at DESC);
                """
            )
            for table in ("frames", "sleep_events", "cry_events"):
                columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
                if "location_id" not in columns:
                    connection.execute(f"ALTER TABLE {table} ADD COLUMN location_id TEXT NOT NULL DEFAULT 'home'")
                if table == "sleep_events" and "details_json" not in columns:
                    connection.execute(
                        "ALTER TABLE sleep_events ADD COLUMN details_json TEXT NOT NULL DEFAULT '{}'"
                    )
            connection.execute(f"PRAGMA user_version = {self.SCHEMA_VERSION}")

    def ready(self) -> bool:
        try:
            with self._connect() as connection:
                return connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        except sqlite3.Error:
            return False

    def add_frame(
        self,
        image: bytes,
        mime_type: str,
        captured_at: datetime,
        camera_entity_id: str | None = None,
        location_id: str = "home",
        label: VisionLabel | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> FrameRecord:
        extensions = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
        if mime_type not in extensions:
            raise ValueError("mime_type must be image/jpeg, image/png, or image/webp")
        if not image:
            raise ValueError("image must not be empty")
        if len(image) > 25 * 1024 * 1024:
            raise ValueError("image exceeds the 25 MB local limit")
        captured = _as_utc(captured_at)
        frame_id = str(uuid4())
        relative = Path(f"{captured:%Y/%m/%d}") / f"{frame_id}{extensions[mime_type]}"
        target = self.frames_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temp = target.with_name(f".{target.name}.tmp")
        digest = hashlib.sha256(image).hexdigest()
        with self._lock:
            descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(image)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp, target)
                with self._connect() as connection:
                    connection.execute(
                        """INSERT INTO frames
                        (id, captured_at, camera_entity_id, location_id, relative_path, mime_type, size_bytes,
                         sha256, image_available, label_json, provider, model)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)""",
                        (
                            frame_id,
                            _iso(captured),
                            camera_entity_id,
                            location_id,
                            str(relative),
                            mime_type,
                            len(image),
                            digest,
                            label.model_dump_json() if label else None,
                            provider,
                            model,
                        ),
                    )
            except Exception:
                target.unlink(missing_ok=True)
                raise
            finally:
                temp.unlink(missing_ok=True)
        return FrameRecord(
            id=frame_id,
            captured_at=captured,
            camera_entity_id=camera_entity_id,
            location_id=location_id,
            mime_type=mime_type,
            size_bytes=len(image),
            sha256=digest,
            label=label,
            provider=provider,
            model=model,
        )

    @staticmethod
    def _frame(row: sqlite3.Row) -> FrameRecord:
        return FrameRecord(
            id=row["id"],
            captured_at=_dt(row["captured_at"]),
            camera_entity_id=row["camera_entity_id"],
            location_id=row["location_id"],
            mime_type=row["mime_type"],
            size_bytes=row["size_bytes"],
            sha256=row["sha256"],
            image_available=bool(row["image_available"]),
            label=VisionLabel.model_validate_json(row["label_json"]) if row["label_json"] else None,
            provider=row["provider"],
            model=row["model"],
        )

    def list_frames(self, limit: int = 50, offset: int = 0) -> tuple[list[FrameRecord], int]:
        with self._connect() as connection:
            total = connection.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
            rows = connection.execute(
                "SELECT * FROM frames ORDER BY captured_at DESC LIMIT ? OFFSET ?", (limit, offset)
            ).fetchall()
        return [self._frame(row) for row in rows], total

    def nearest_frames(
        self,
        captured_at: datetime,
        limit: int = 5,
        within_minutes: int = 360,
    ) -> list[FrameRecord]:
        """Return the closest private frames without exposing filesystem paths."""

        lower = captured_at - timedelta(minutes=within_minutes)
        upper = captured_at + timedelta(minutes=within_minutes)
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM frames
                   WHERE captured_at BETWEEN ? AND ?
                   ORDER BY ABS((julianday(captured_at) - julianday(?)) * 86400), captured_at
                   LIMIT ?""",
                (_iso(lower), _iso(upper), _iso(captured_at), limit),
            ).fetchall()
        return [self._frame(row) for row in rows]

    def vision_labels_between(
        self,
        start: datetime,
        end: datetime,
        *,
        limit: int = 250_000,
    ) -> list[tuple[datetime, VisionLabel]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT captured_at, label_json FROM frames
                   WHERE captured_at >= ? AND captured_at <= ? AND label_json IS NOT NULL
                   ORDER BY captured_at ASC LIMIT ?""",
                (_iso(start), _iso(end), limit),
            ).fetchall()
        return [
            (_dt(row["captured_at"]), VisionLabel.model_validate_json(row["label_json"]))
            for row in rows
        ]

    def latest_frame(self) -> FrameRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM frames ORDER BY captured_at DESC LIMIT 1").fetchone()
        return self._frame(row) if row else None

    def get_frame(self, frame_id: str) -> FrameRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM frames WHERE id = ?", (frame_id,)).fetchone()
        return self._frame(row) if row else None

    def get_frame_path(self, frame_id: str) -> Path | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT relative_path, image_available FROM frames WHERE id = ?", (frame_id,)
            ).fetchone()
        if not row or not row["image_available"] or not row["relative_path"]:
            return None
        candidate = (self.frames_dir / row["relative_path"]).resolve()
        frames_root = self.frames_dir.resolve()
        if not candidate.is_relative_to(frames_root):
            raise StorageError("frame path escaped the private frame directory")
        return candidate if candidate.is_file() else None

    def set_frame_label(self, frame_id: str, label: VisionLabel, provider: str, model: str) -> FrameRecord | None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE frames SET label_json = ?, provider = ?, model = ? WHERE id = ?",
                (label.model_dump_json(), provider, model, frame_id),
            )
        return self.get_frame(frame_id)

    def purge_frames_before(self, before: datetime) -> dict[str, int]:
        purged = 0
        bytes_removed = 0
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """SELECT id, relative_path, size_bytes FROM frames
                   WHERE image_available = 1 AND captured_at < ?""",
                (_iso(before),),
            ).fetchall()
            for row in rows:
                if row["relative_path"]:
                    path = (self.frames_dir / row["relative_path"]).resolve()
                    if path.is_relative_to(self.frames_dir.resolve()):
                        path.unlink(missing_ok=True)
                purged += 1
                bytes_removed += row["size_bytes"]
            connection.execute(
                """UPDATE frames SET image_available = 0, relative_path = NULL, purged_at = ?
                   WHERE image_available = 1 AND captured_at < ?""",
                (_iso(datetime.now(UTC)), _iso(before)),
            )
        return {"frames": purged, "bytes": bytes_removed}

    def add_sleep_event(self, event: SleepEventCreate) -> SleepEvent:
        values = event.model_dump()
        values["started_at"] = _as_utc(event.started_at)
        values["ended_at"] = _as_utc(event.ended_at) if event.ended_at else None
        item = SleepEvent(id=str(uuid4()), **values)
        with self._lock, self._connect() as connection:
            if self._sleep_overlaps(connection, item.started_at, item.ended_at):
                raise StorageError("sleep event overlaps an existing event")
            connection.execute(
                """INSERT INTO sleep_events
                (id, started_at, ended_at, kind, source, notes, details_json, location_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    item.id,
                    _iso(item.started_at),
                    _iso(item.ended_at) if item.ended_at else None,
                    item.kind,
                    item.source,
                    item.notes,
                    item.details.model_dump_json(),
                    item.location_id,
                    _iso(item.created_at),
                ),
            )
        return item

    @staticmethod
    def _sleep(row: sqlite3.Row) -> SleepEvent:
        try:
            details = SleepDetails.model_validate_json(row["details_json"] or "{}")
        except (ValueError, KeyError):
            details = SleepDetails()
        return SleepEvent(
            id=row["id"],
            started_at=_dt(row["started_at"]),
            ended_at=_dt(row["ended_at"]) if row["ended_at"] else None,
            kind=row["kind"],
            source=row["source"],
            notes=row["notes"],
            details=details,
            location_id=row["location_id"],
            created_at=_dt(row["created_at"]),
        )

    def list_sleep_events(self, limit: int = 50, offset: int = 0) -> tuple[list[SleepEvent], int]:
        with self._connect() as connection:
            total = connection.execute("SELECT COUNT(*) FROM sleep_events").fetchone()[0]
            rows = connection.execute(
                "SELECT * FROM sleep_events ORDER BY started_at DESC LIMIT ? OFFSET ?", (limit, offset)
            ).fetchall()
        return [self._sleep(row) for row in rows], total

    def get_sleep_event(self, event_id: str) -> SleepEvent | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM sleep_events WHERE id = ?", (event_id,)).fetchone()
        return self._sleep(row) if row else None

    def latest_sleep_event(self) -> SleepEvent | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM sleep_events WHERE kind != 'awake' ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
        return self._sleep(row) if row else None

    def open_sleep_event(self) -> SleepEvent | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT * FROM sleep_events
                   WHERE ended_at IS NULL AND kind != 'awake'
                   ORDER BY started_at DESC LIMIT 1"""
            ).fetchone()
        return self._sleep(row) if row else None

    def update_sleep_event(self, event_id: str, patch: SleepEventPatch) -> SleepEvent | None:
        current = self.get_sleep_event(event_id)
        if current is None:
            return None
        values = patch.model_dump(exclude_unset=True)
        merged = SleepEventCreate.model_validate(
            {
                "started_at": values.get("started_at", current.started_at),
                "ended_at": values.get("ended_at", current.ended_at),
                "kind": values.get("kind", current.kind),
                "source": current.source,
                "notes": values.get("notes", current.notes),
                "details": values.get("details", current.details),
                "location_id": current.location_id,
            }
        )
        with self._lock, self._connect() as connection:
            if self._sleep_overlaps(
                connection,
                merged.started_at,
                merged.ended_at,
                exclude_id=event_id,
            ):
                raise StorageError("sleep event overlaps an existing event")
            connection.execute(
                """UPDATE sleep_events
                   SET started_at = ?, ended_at = ?, kind = ?, notes = ?, details_json = ? WHERE id = ?""",
                (
                    _iso(merged.started_at),
                    _iso(merged.ended_at) if merged.ended_at else None,
                    merged.kind,
                    merged.notes,
                    merged.details.model_dump_json(),
                    event_id,
                ),
            )
        return self.get_sleep_event(event_id)

    @staticmethod
    def _sleep_overlaps(
        connection: sqlite3.Connection,
        started_at: datetime,
        ended_at: datetime | None,
        *,
        exclude_id: str | None = None,
    ) -> bool:
        end_limit = _iso(ended_at) if ended_at else "9999-12-31T23:59:59.999999Z"
        query = """
            SELECT 1 FROM sleep_events
            WHERE started_at < ?
              AND COALESCE(ended_at, '9999-12-31T23:59:59.999999Z') > ?
        """
        parameters: list[str] = [end_limit, _iso(started_at)]
        if exclude_id is not None:
            query += " AND id != ?"
            parameters.append(exclude_id)
        return connection.execute(query, parameters).fetchone() is not None

    def delete_sleep_event(self, event_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM sleep_events WHERE id = ?", (event_id,))
        return cursor.rowcount > 0

    def sleep_events_overlapping(self, start: datetime, end: datetime) -> list[SleepEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM sleep_events
                   WHERE started_at < ? AND COALESCE(ended_at, ?) > ?
                   ORDER BY started_at""",
                (_iso(end), _iso(end), _iso(start)),
            ).fetchall()
        return [self._sleep(row) for row in rows]

    def add_cry_event(self, event: CryEventCreate) -> CryEvent:
        values = event.model_dump()
        values["detected_at"] = _as_utc(event.detected_at)
        values["ended_at"] = _as_utc(event.ended_at) if event.ended_at else None
        item = CryEvent(id=str(uuid4()), **values)
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO cry_events
                (id, detected_at, ended_at, source, confidence, metadata_json, location_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    item.id,
                    _iso(item.detected_at),
                    _iso(item.ended_at) if item.ended_at else None,
                    item.source,
                    item.confidence,
                    json.dumps(item.metadata, separators=(",", ":")),
                    item.location_id,
                    _iso(item.created_at),
                ),
            )
        return item

    @staticmethod
    def _cry(row: sqlite3.Row) -> CryEvent:
        return CryEvent(
            id=row["id"],
            detected_at=_dt(row["detected_at"]),
            ended_at=_dt(row["ended_at"]) if row["ended_at"] else None,
            source=row["source"],
            confidence=row["confidence"],
            metadata=json.loads(row["metadata_json"]),
            location_id=row["location_id"],
            created_at=_dt(row["created_at"]),
        )

    def list_cry_events(self, limit: int = 50, offset: int = 0) -> tuple[list[CryEvent], int]:
        with self._connect() as connection:
            total = connection.execute("SELECT COUNT(*) FROM cry_events").fetchone()[0]
            rows = connection.execute(
                "SELECT * FROM cry_events ORDER BY detected_at DESC LIMIT ? OFFSET ?", (limit, offset)
            ).fetchall()
        return [self._cry(row) for row in rows], total

    def latest_cry_event(self) -> CryEvent | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM cry_events ORDER BY detected_at DESC LIMIT 1").fetchone()
        return self._cry(row) if row else None

    def close_cry_event(self, event_id: str, ended_at: datetime) -> CryEvent | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM cry_events WHERE id = ?", (event_id,)).fetchone()
            if row is None:
                return None
            event = self._cry(row)
            if ended_at <= event.detected_at:
                ended_at = event.detected_at + timedelta(microseconds=1)
            connection.execute(
                "UPDATE cry_events SET ended_at = ? WHERE id = ?",
                (_iso(ended_at), event_id),
            )
            updated = connection.execute("SELECT * FROM cry_events WHERE id = ?", (event_id,)).fetchone()
        return self._cry(updated)

    def open_cry_event(self) -> CryEvent | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM cry_events WHERE ended_at IS NULL ORDER BY detected_at DESC LIMIT 1"
            ).fetchone()
        return self._cry(row) if row else None

    def retention_estimate(self, before: datetime) -> dict[str, int]:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM frames
                   WHERE image_available = 1 AND captured_at < ?""",
                (_iso(before),),
            ).fetchone()
        return {"frames": int(row[0]), "bytes": int(row[1])}

    def summary(self) -> dict[str, Any]:
        with self._connect() as connection:
            frames = connection.execute("SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM frames").fetchone()
            images = connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM frames WHERE image_available = 1"
            ).fetchone()
            sleep_count = connection.execute("SELECT COUNT(*) FROM sleep_events").fetchone()[0]
            cry_count = connection.execute("SELECT COUNT(*) FROM cry_events").fetchone()[0]
            last_frame = connection.execute("SELECT MAX(captured_at) FROM frames").fetchone()[0]
        return {
            "frames": frames[0],
            "stored_images": images[0],
            "stored_image_bytes": images[1],
            "historical_frame_bytes": frames[1],
            "sleep_events": sleep_count,
            "cry_events": cry_count,
            "last_frame_at": last_frame,
        }

    def prepare_history_snapshot(self, target_dir: Path) -> tuple[Path, Path]:
        """Create a point-in-time database and stable frame tree for export.

        Frame files are hard-linked when the filesystem supports it, so a
        large export does not temporarily duplicate every image before the
        archive itself is written. The database lock prevents retention from
        unlinking a frame between the SQLite backup and the hard-link step.
        """

        snapshot_db = target_dir / "history.sqlite3"
        snapshot_frames = target_dir / "frames"
        target_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
        snapshot_frames.mkdir(mode=0o700)
        with self._lock:
            with self._connect() as source, sqlite3.connect(snapshot_db) as destination:
                source.backup(destination)
            os.chmod(snapshot_db, 0o600)
            with sqlite3.connect(snapshot_db) as snapshot:
                snapshot.row_factory = sqlite3.Row
                rows = snapshot.execute(
                    """SELECT relative_path FROM frames
                       WHERE image_available = 1 AND relative_path IS NOT NULL"""
                ).fetchall()
            frames_root = self.frames_dir.resolve()
            for row in rows:
                relative = Path(row["relative_path"])
                source_path = (self.frames_dir / relative).resolve()
                if not source_path.is_relative_to(frames_root) or not source_path.is_file():
                    raise StorageError(f"stored frame is missing or unsafe: {relative}")
                target_path = snapshot_frames / relative
                target_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                try:
                    os.link(source_path, target_path)
                except OSError:
                    shutil.copyfile(source_path, target_path)
                os.chmod(target_path, 0o600)
        return snapshot_db, snapshot_frames

    def replace_history(self, snapshot_db: Path, snapshot_frames: Path, work_dir: Path) -> None:
        """Atomically replace history while preserving settings and secrets.

        The caller must fully validate the staged snapshot first. A private
        rollback directory is kept until the replacement passes SQLite's
        integrity check, then removed.
        """

        rollback = work_dir / "rollback"
        incoming = work_dir / "incoming"
        rollback.mkdir(parents=True, exist_ok=False, mode=0o700)
        incoming.mkdir(parents=True, exist_ok=False, mode=0o700)
        incoming_db = incoming / self.db_path.name
        incoming_frames = incoming / "frames"
        os.replace(snapshot_db, incoming_db)
        os.replace(snapshot_frames, incoming_frames)
        previous_db = rollback / self.db_path.name
        previous_frames = rollback / "frames"
        moved_database = False
        moved_frames = False
        installed_database = False
        installed_frames = False
        with self._lock:
            try:
                with self._connect() as connection:
                    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                for suffix in ("-wal", "-shm"):
                    self.db_path.with_name(f"{self.db_path.name}{suffix}").unlink(missing_ok=True)
                if self.db_path.exists():
                    os.replace(self.db_path, previous_db)
                    moved_database = True
                if self.frames_dir.exists():
                    os.replace(self.frames_dir, previous_frames)
                    moved_frames = True
                os.replace(incoming_db, self.db_path)
                installed_database = True
                os.replace(incoming_frames, self.frames_dir)
                installed_frames = True
                with self._connect() as connection:
                    if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                        raise StorageError("imported history failed SQLite integrity_check")
                # Archives from older public releases remain importable. Run
                # additive migrations only after the validated database swap.
                self._migrate()
                self._secure_database_files()
            except Exception:
                if installed_database:
                    self.db_path.unlink(missing_ok=True)
                if installed_frames:
                    shutil.rmtree(self.frames_dir, ignore_errors=True)
                if moved_database and previous_db.exists():
                    os.replace(previous_db, self.db_path)
                if moved_frames and previous_frames.exists():
                    os.replace(previous_frames, self.frames_dir)
                self._secure_database_files()
                raise
        shutil.rmtree(rollback, ignore_errors=True)
        shutil.rmtree(incoming, ignore_errors=True)

    def clear_history(self, work_dir: Path) -> None:
        """Replace all history with an empty, current-schema database.

        Settings and encrypted secrets live outside this database and are not
        affected. ``replace_history`` keeps the destructive step atomic and
        rolls the previous history back if the empty replacement cannot be
        installed cleanly.
        """

        empty_source = work_dir / "empty-source"
        replacement_work = work_dir / "replacement"
        empty = Database(empty_source)
        snapshot_db, snapshot_frames = empty.prepare_history_snapshot(work_dir / "empty-snapshot")
        replacement_work.mkdir(mode=0o700)
        self.replace_history(snapshot_db, snapshot_frames, replacement_work)
