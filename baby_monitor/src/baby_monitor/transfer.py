from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import sqlite3
import tempfile
import threading
import zipfile
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import UUID, uuid4

from .database import Database

ARCHIVE_CONTENT_TYPE = "application/vnd.baby-monitor.history+zip"
ARCHIVE_FORMAT = "baby-monitor-history"
ARCHIVE_VERSION = 1
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024 * 1024
STATE_FILE = ".history-transfer-state.json"

FRAME_COLUMNS = (
    "id",
    "captured_at",
    "camera_entity_id",
    "location_id",
    "relative_path",
    "mime_type",
    "size_bytes",
    "sha256",
    "image_available",
    "label_json",
    "provider",
    "model",
    "purged_at",
)
SLEEP_COLUMNS = (
    "id",
    "started_at",
    "ended_at",
    "kind",
    "source",
    "notes",
    "location_id",
    "created_at",
)
CRY_COLUMNS = (
    "id",
    "detected_at",
    "ended_at",
    "source",
    "confidence",
    "metadata_json",
    "location_id",
    "created_at",
)


class TransferError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        os.chmod(path, 0o600)
    finally:
        temp.unlink(missing_ok=True)


def _safe_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        not name
        or name.startswith("/")
        or "\\" in name
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise TransferError(f"archive contains an unsafe path: {name!r}")
    return path


def _csv_bytes(connection: sqlite3.Connection, table: str, columns: tuple[str, ...], *, frames: bool = False) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    header = [*columns, "archive_image_path"] if frames else list(columns)
    writer.writerow(header)
    query = f"SELECT {', '.join(columns)} FROM {table} ORDER BY {columns[1]}, id"  # noqa: S608
    for row in connection.execute(query):
        values = [row[column] for column in columns]
        if frames:
            image_path = ""
            if row["image_available"] and row["relative_path"]:
                image_path = f"images/{row['location_id']}/{row['relative_path']}"
            values.append(image_path)
        writer.writerow(["" if value is None else value for value in values])
    return output.getvalue().encode("utf-8")


class HistoryTransferManager:
    def __init__(self, data_dir: Path, database: Database, *, app_version: str) -> None:
        self.data_dir = data_dir
        self.database = database
        self.app_version = app_version
        self.state_path = data_dir / STATE_FILE
        self.transfer_dir = data_dir / ".transfers"
        self.outgoing_dir = self.transfer_dir / "outgoing"
        self.incoming_dir = self.transfer_dir / "incoming"
        self._lock = threading.RLock()
        self.transfer_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.outgoing_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.incoming_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._state = self._load_state()

    def _default_state(self) -> dict[str, Any]:
        return {
            "version": 1,
            "installation_id": str(uuid4()),
            "dataset_id": str(uuid4()),
            "generation": 0,
            "status": "active",
            "operation": None,
            "outgoing": None,
            "last_import": None,
        }

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            state = self._default_state()
            _atomic_json(self.state_path, state)
            return state
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            UUID(str(state["installation_id"]))
            UUID(str(state["dataset_id"]))
            if state.get("version") != 1 or state.get("status") not in {
                "active",
                "preparing",
                "pending",
                "retired",
            }:
                raise ValueError
            if not isinstance(state.get("generation"), int) or state["generation"] < 0:
                raise ValueError
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise TransferError("history transfer state is invalid") from exc
        operation = state.get("operation")
        if operation not in {None, "export", "import", "retire"}:
            raise TransferError("history transfer operation state is invalid")
        if state["status"] == "preparing":
            if operation == "export":
                state["status"] = "active"
                state["outgoing"] = None
            elif operation == "retire" and isinstance(state.get("outgoing"), dict):
                state["status"] = "pending"
            else:
                # An interrupted import may already have swapped the database.
                # Keep it read-only until the archive is explicitly retried.
                state["status"] = "retired"
                state["outgoing"] = None
            state["operation"] = None
            _atomic_json(self.state_path, state)
        return state

    def _save_state(self) -> None:
        _atomic_json(self.state_path, self._state)

    @property
    def writable(self) -> bool:
        with self._lock:
            return self._state["status"] == "active"

    def ensure_writable(self) -> None:
        with self._lock:
            if self._state["status"] == "retired":
                raise TransferError(
                    "this history copy has been safely retired; import a newer transfer to make it active again"
                )
            if self._state["status"] != "active":
                raise TransferError(
                    "history is read-only because a transfer is pending; import it at the destination "
                    "or cancel the transfer"
                )

    def public_status(self) -> dict[str, Any]:
        with self._lock:
            outgoing = self._state.get("outgoing")
            return {
                "status": self._state["status"],
                "writable": self._state["status"] == "active",
                "datasetId": self._state["dataset_id"],
                "generation": self._state["generation"],
                "outgoing": {
                    "archiveId": outgoing["archive_id"],
                    "filename": outgoing["filename"],
                    "createdAt": outgoing["created_at"],
                    "manifestSha256": outgoing["manifest_sha256"],
                    "bytes": outgoing["bytes"],
                    "counts": outgoing["counts"],
                    "downloadUrl": f"api/v1/history-transfer/exports/{outgoing['archive_id']}",
                }
                if isinstance(outgoing, dict)
                else None,
                "lastImport": self._state.get("last_import"),
            }

    def _archive_details(self, outgoing: dict[str, Any]) -> dict[str, Any]:
        path = self.outgoing_dir / outgoing["stored_name"]
        if not path.is_file():
            raise TransferError("prepared transfer archive is missing")
        return {
            **self.public_status()["outgoing"],
            "path": path,
            "contentType": ARCHIVE_CONTENT_TYPE,
        }

    def prepare_export(self) -> dict[str, Any]:
        with self._lock:
            if self._state["status"] == "pending" and isinstance(self._state.get("outgoing"), dict):
                return self._archive_details(self._state["outgoing"])
            if self._state["status"] == "retired":
                raise TransferError("this history copy is retired; import a newer transfer before exporting again")
            if self._state["status"] != "active":
                raise TransferError("another history transfer operation is already running")
            generation = self._state["generation"] + 1
            self._state["status"] = "preparing"
            self._state["operation"] = "export"
            self._state["generation"] = generation
            self._save_state()
            workspace = Path(tempfile.mkdtemp(prefix="export-", dir=self.transfer_dir))
            try:
                snapshot_dir = workspace / "snapshot"
                snapshot_db, snapshot_frames = self.database.prepare_history_snapshot(snapshot_dir)
                archive_id = str(uuid4())
                timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
                filename = f"baby-monitor-history-{timestamp}-g{generation}.zip"
                stored_name = f"{archive_id}.zip"
                temporary_archive = workspace / stored_name
                manifest = self._write_archive(
                    temporary_archive,
                    snapshot_db,
                    snapshot_frames,
                    dataset_id=self._state["dataset_id"],
                    generation=generation,
                )
                final_archive = self.outgoing_dir / stored_name
                os.replace(temporary_archive, final_archive)
                os.chmod(final_archive, 0o600)
                outgoing = {
                    "archive_id": archive_id,
                    "filename": filename,
                    "stored_name": stored_name,
                    "created_at": manifest["created_at"],
                    "manifest_sha256": manifest["manifest_sha256"],
                    "bytes": final_archive.stat().st_size,
                    "counts": manifest["counts"],
                }
                self._state["status"] = "pending"
                self._state["operation"] = None
                self._state["outgoing"] = outgoing
                self._save_state()
                return self._archive_details(outgoing)
            except Exception:
                self._state["status"] = "active"
                self._state["operation"] = None
                self._state["outgoing"] = None
                self._save_state()
                raise
            finally:
                shutil.rmtree(workspace, ignore_errors=True)

    def _write_archive(
        self,
        archive_path: Path,
        snapshot_db: Path,
        snapshot_frames: Path,
        *,
        dataset_id: str,
        generation: int,
    ) -> dict[str, Any]:
        with sqlite3.connect(snapshot_db) as connection:
            connection.row_factory = sqlite3.Row
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise TransferError("history database failed SQLite integrity_check before export")
            schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
            counts = {
                "frames": connection.execute("SELECT COUNT(*) FROM frames").fetchone()[0],
                "storedImages": connection.execute("SELECT COUNT(*) FROM frames WHERE image_available = 1").fetchone()[
                    0
                ],
                "sleepEvents": connection.execute("SELECT COUNT(*) FROM sleep_events").fetchone()[0],
                "cryEvents": connection.execute("SELECT COUNT(*) FROM cry_events").fetchone()[0],
            }
            locations = sorted(
                row[0]
                for row in connection.execute(
                    """SELECT location_id FROM frames UNION SELECT location_id FROM sleep_events
                       UNION SELECT location_id FROM cry_events"""
                )
                if row[0]
            )
            images: list[dict[str, Any]] = []
            for row in connection.execute(
                """SELECT id, location_id, relative_path, size_bytes, sha256 FROM frames
                   WHERE image_available = 1 ORDER BY captured_at, id"""
            ):
                if not row["relative_path"]:
                    raise TransferError(f"frame {row['id']} is available but has no path")
                relative = _safe_member(row["relative_path"])
                image = (snapshot_frames / Path(*relative.parts)).resolve()
                if not image.is_relative_to(snapshot_frames.resolve()) or not image.is_file():
                    raise TransferError(f"frame {row['id']} image is missing")
                size = image.stat().st_size
                digest = _sha256_path(image)
                if size != row["size_bytes"] or digest != row["sha256"]:
                    raise TransferError(f"frame {row['id']} failed size or SHA-256 verification")
                archive_image = f"images/{row['location_id']}/{relative.as_posix()}"
                _safe_member(archive_image)
                images.append(
                    {
                        "id": row["id"],
                        "relative_path": relative.as_posix(),
                        "archive_path": archive_image,
                        "size_bytes": size,
                        "sha256": digest,
                    }
                )
            csv_files = {
                "data/frames.csv": _csv_bytes(connection, "frames", FRAME_COLUMNS, frames=True),
                "data/sleep_events.csv": _csv_bytes(connection, "sleep_events", SLEEP_COLUMNS),
                "data/cry_events.csv": _csv_bytes(connection, "cry_events", CRY_COLUMNS),
            }
        manifest = {
            "format": ARCHIVE_FORMAT,
            "format_version": ARCHIVE_VERSION,
            "created_at": _utc_now(),
            "app_version": self.app_version,
            "database_schema_version": schema_version,
            "dataset_id": dataset_id,
            "generation": generation,
            "database_sha256": _sha256_path(snapshot_db),
            "counts": counts,
            "locations": locations,
            "images": images,
        }
        manifest_bytes = json.dumps(manifest, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        readme = (
            b"Baby Monitor history export\n\n"
            b"The CSV files under data/ are intended for analysis in other tools.\n"
            b"Images are grouped under images/<location>/<year>/<month>/<day>/.\n"
            b"internal/history.sqlite3 and manifest.json allow lossless import by Baby Monitor.\n"
            b"This archive contains private family history and camera images. "
            b"It contains no API keys or Home Assistant tokens.\n"
        )
        with zipfile.ZipFile(archive_path, "w", allowZip64=True) as archive:
            archive.writestr("README.txt", readme, compress_type=zipfile.ZIP_DEFLATED)
            archive.writestr("manifest.json", manifest_bytes, compress_type=zipfile.ZIP_DEFLATED)
            archive.write(snapshot_db, "internal/history.sqlite3", compress_type=zipfile.ZIP_DEFLATED)
            for name, content in csv_files.items():
                archive.writestr(name, content, compress_type=zipfile.ZIP_DEFLATED)
            for image in images:
                source = snapshot_frames / image["relative_path"]
                archive.write(source, image["archive_path"], compress_type=zipfile.ZIP_STORED)
        return {**manifest, "manifest_sha256": manifest_sha256}

    def export_path(self, archive_id: str) -> tuple[Path, str]:
        with self._lock:
            outgoing = self._state.get("outgoing")
            if (
                self._state["status"] != "pending"
                or not isinstance(outgoing, dict)
                or outgoing.get("archive_id") != archive_id
            ):
                raise TransferError("transfer archive not found")
            details = self._archive_details(outgoing)
            return details["path"], details["filename"]

    def cancel_export(self) -> dict[str, Any]:
        with self._lock:
            if self._state["status"] != "pending" or not isinstance(self._state.get("outgoing"), dict):
                raise TransferError("there is no pending export to cancel")
            archive = self.outgoing_dir / self._state["outgoing"]["stored_name"]
            archive.unlink(missing_ok=True)
            self._state["status"] = "active"
            self._state["operation"] = None
            self._state["outgoing"] = None
            self._save_state()
            return self.public_status()

    def retire_exported_history(self, receipt: Any, *, delete_history: bool) -> dict[str, Any]:
        """Delete the source history only after a matching import receipt.

        The receipt is an operational guard against deleting the only copy by
        accident. It is intentionally checked against the exact pending
        archive (dataset, generation, manifest hash, and record counts).
        """

        with self._lock:
            outgoing = self._state.get("outgoing")
            if self._state["status"] != "pending" or not isinstance(outgoing, dict):
                raise TransferError("there is no pending export to finalize")
            if not delete_history:
                raise TransferError("explicit confirmation is required before deleting source history")
            self._validate_import_receipt(receipt, outgoing)
            workspace = Path(tempfile.mkdtemp(prefix="retire-", dir=self.transfer_dir))
            previous_state = json.loads(json.dumps(self._state))
            archive = self.outgoing_dir / outgoing["stored_name"]
            try:
                self._state["status"] = "preparing"
                self._state["operation"] = "retire"
                self._save_state()
                self.database.clear_history(workspace)
                self._state["status"] = "retired"
                self._state["operation"] = None
                self._state["outgoing"] = None
                self._state["last_import"] = None
                self._save_state()
                with suppress(OSError):
                    archive.unlink(missing_ok=True)
                return {"ok": True, "deleted": True, "status": self.public_status()}
            except Exception:
                self._state = previous_state
                self._save_state()
                raise
            finally:
                shutil.rmtree(workspace, ignore_errors=True)

    def _validate_import_receipt(self, receipt: Any, outgoing: dict[str, Any]) -> None:
        if not isinstance(receipt, dict):
            raise TransferError("import receipt must be a JSON object")
        if receipt.get("format") != "baby-monitor-import-receipt" or receipt.get("formatVersion") != 1:
            raise TransferError("import receipt format or version is invalid")
        try:
            dataset_id = str(UUID(str(receipt["datasetId"])))
            destination_id = str(UUID(str(receipt["destinationInstallationId"])))
        except (KeyError, TypeError, ValueError) as exc:
            raise TransferError("import receipt installation or dataset ID is invalid") from exc
        if destination_id == self._state["installation_id"]:
            raise TransferError("import receipt must come from a different installation")
        if dataset_id != self._state["dataset_id"]:
            raise TransferError("import receipt belongs to a different history dataset")
        if receipt.get("generation") != self._state["generation"]:
            raise TransferError("import receipt belongs to a different history generation")
        if receipt.get("manifestSha256") != outgoing["manifest_sha256"]:
            raise TransferError("import receipt does not match the pending archive")
        if receipt.get("counts") != outgoing["counts"]:
            raise TransferError("import receipt counts do not match the pending archive")
        imported_at = receipt.get("importedAt")
        if not isinstance(imported_at, str):
            raise TransferError("import receipt timestamp is invalid")
        try:
            datetime.fromisoformat(imported_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise TransferError("import receipt timestamp is invalid") from exc

    def import_archive(self, archive_path: Path, *, replace_existing: bool) -> dict[str, Any]:
        with self._lock:
            if self._state["status"] == "pending":
                raise TransferError("cancel the pending export on this installation before importing another history")
            if self._state["status"] == "preparing":
                raise TransferError("another history transfer operation is already running")
            workspace = Path(tempfile.mkdtemp(prefix="validated-", dir=self.incoming_dir))
            previous_state = json.loads(json.dumps(self._state))
            try:
                self._state["status"] = "preparing"
                self._state["operation"] = "import"
                self._save_state()
                manifest, manifest_sha256, snapshot_db, snapshot_frames = self._validate_and_extract(
                    archive_path, workspace
                )
                summary = self.database.summary()
                has_history = any(summary[key] for key in ("frames", "sleep_events", "cry_events"))
                same_dataset = manifest["dataset_id"] == self._state["dataset_id"]
                if same_dataset and manifest["generation"] < self._state["generation"]:
                    raise TransferError("the selected archive is older than this history copy")
                last_import = self._state.get("last_import")
                if (
                    same_dataset
                    and manifest["generation"] == self._state["generation"]
                    and isinstance(last_import, dict)
                    and last_import.get("manifestSha256") == manifest_sha256
                ):
                    return {"ok": True, "idempotent": True, "receipt": last_import, "counts": manifest["counts"]}
                if has_history and not replace_existing:
                    raise TransferError(
                        "this installation already has history; confirm replacement after verifying "
                        "the selected archive"
                    )
                replacement_work = workspace / "replacement"
                replacement_work.mkdir(mode=0o700)
                self.database.replace_history(snapshot_db, snapshot_frames, replacement_work)
                receipt = {
                    "format": "baby-monitor-import-receipt",
                    "formatVersion": 1,
                    "datasetId": manifest["dataset_id"],
                    "generation": manifest["generation"],
                    "manifestSha256": manifest_sha256,
                    "destinationInstallationId": self._state["installation_id"],
                    "importedAt": _utc_now(),
                    "counts": manifest["counts"],
                }
                self._state["dataset_id"] = manifest["dataset_id"]
                self._state["generation"] = manifest["generation"]
                self._state["status"] = "active"
                self._state["operation"] = None
                self._state["outgoing"] = None
                self._state["last_import"] = receipt
                self._save_state()
                return {"ok": True, "idempotent": False, "receipt": receipt, "counts": manifest["counts"]}
            except Exception:
                self._state = previous_state
                self._save_state()
                raise
            finally:
                archive_path.unlink(missing_ok=True)
                shutil.rmtree(workspace, ignore_errors=True)

    def _validate_and_extract(self, archive_path: Path, workspace: Path) -> tuple[dict[str, Any], str, Path, Path]:
        try:
            archive = zipfile.ZipFile(archive_path)
        except (OSError, zipfile.BadZipFile) as exc:
            raise TransferError("selected file is not a valid Baby Monitor ZIP archive") from exc
        with archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise TransferError("archive contains duplicate file names")
            for info in infos:
                _safe_member(info.filename)
                if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                    raise TransferError("archive uses an unsupported compression method")
                unix_mode = (info.external_attr >> 16) & 0o170000
                if unix_mode == 0o120000:
                    raise TransferError("archive must not contain symbolic links")
            try:
                manifest_info = archive.getinfo("manifest.json")
            except KeyError as exc:
                raise TransferError("archive does not contain manifest.json") from exc
            if manifest_info.file_size > MAX_MANIFEST_BYTES:
                raise TransferError("archive manifest is too large")
            try:
                manifest_bytes = archive.read(manifest_info)
                manifest = json.loads(manifest_bytes)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise TransferError("archive manifest is invalid") from exc
            manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            self._validate_manifest(manifest)
            expected = {
                "README.txt",
                "manifest.json",
                "internal/history.sqlite3",
                "data/frames.csv",
                "data/sleep_events.csv",
                "data/cry_events.csv",
                *(image["archive_path"] for image in manifest["images"]),
            }
            if set(names) != expected:
                missing = expected.difference(names)
                extra = set(names).difference(expected)
                raise TransferError(
                    f"archive contents do not match its manifest (missing={len(missing)}, extra={len(extra)})"
                )
            total_uncompressed = sum(info.file_size for info in infos)
            free = shutil.disk_usage(self.data_dir).free
            if total_uncompressed + 256 * 1024 * 1024 > free:
                raise TransferError("not enough free disk space to validate and import this archive")
            snapshot_db = workspace / "history.sqlite3"
            self._extract_member(archive, "internal/history.sqlite3", snapshot_db)
            if _sha256_path(snapshot_db) != manifest["database_sha256"]:
                raise TransferError("history database SHA-256 does not match the manifest")
            self._validate_snapshot_database(snapshot_db, manifest)
            snapshot_frames = workspace / "frames"
            snapshot_frames.mkdir(mode=0o700)
            for image in manifest["images"]:
                relative = _safe_member(image["relative_path"])
                target = snapshot_frames / Path(*relative.parts)
                self._extract_member(archive, image["archive_path"], target)
                if target.stat().st_size != image["size_bytes"] or _sha256_path(target) != image["sha256"]:
                    raise TransferError(f"image {image['id']} failed size or SHA-256 verification")
            return manifest, manifest_sha256, snapshot_db, snapshot_frames

    @staticmethod
    def _extract_member(archive: zipfile.ZipFile, name: str, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with archive.open(name) as source, os.fdopen(descriptor, "wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
                destination.flush()
                os.fsync(destination.fileno())
        except Exception:
            target.unlink(missing_ok=True)
            raise

    @staticmethod
    def _validate_manifest(manifest: Any) -> None:
        if not isinstance(manifest, dict):
            raise TransferError("archive manifest must be an object")
        if manifest.get("format") != ARCHIVE_FORMAT or manifest.get("format_version") != ARCHIVE_VERSION:
            raise TransferError("archive format or version is not supported")
        try:
            UUID(str(manifest["dataset_id"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise TransferError("archive dataset ID is invalid") from exc
        if not isinstance(manifest.get("generation"), int) or manifest["generation"] < 1:
            raise TransferError("archive generation is invalid")
        if not isinstance(manifest.get("database_schema_version"), int):
            raise TransferError("archive database schema version is invalid")
        digest = manifest.get("database_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise TransferError("archive database SHA-256 is invalid")
        counts = manifest.get("counts")
        if not isinstance(counts, dict) or any(
            not isinstance(counts.get(key), int) or counts[key] < 0
            for key in ("frames", "storedImages", "sleepEvents", "cryEvents")
        ):
            raise TransferError("archive counts are invalid")
        images = manifest.get("images")
        if not isinstance(images, list) or len(images) != counts["storedImages"] or len(images) > 1_000_000:
            raise TransferError("archive image manifest is invalid")
        seen_ids: set[str] = set()
        seen_paths: set[str] = set()
        for image in images:
            if not isinstance(image, dict):
                raise TransferError("archive image entry is invalid")
            image_id = image.get("id")
            relative = image.get("relative_path")
            archive_path = image.get("archive_path")
            sha256 = image.get("sha256")
            size = image.get("size_bytes")
            if (
                not isinstance(image_id, str)
                or not image_id
                or not isinstance(relative, str)
                or not isinstance(archive_path, str)
                or not archive_path.startswith("images/")
                or not isinstance(size, int)
                or size < 0
                or not isinstance(sha256, str)
                or len(sha256) != 64
            ):
                raise TransferError("archive image entry is invalid")
            _safe_member(relative)
            _safe_member(archive_path)
            if image_id in seen_ids or relative in seen_paths or archive_path in seen_paths:
                raise TransferError("archive image entries are duplicated")
            seen_ids.add(image_id)
            seen_paths.update({relative, archive_path})

    def _validate_snapshot_database(self, snapshot_db: Path, manifest: dict[str, Any]) -> None:
        try:
            with sqlite3.connect(f"file:{snapshot_db}?mode=ro", uri=True) as connection:
                connection.row_factory = sqlite3.Row
                if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise TransferError("archive history failed SQLite integrity_check")
                version = connection.execute("PRAGMA user_version").fetchone()[0]
                if version != manifest["database_schema_version"] or version > Database.SCHEMA_VERSION:
                    raise TransferError("archive history uses an unsupported database schema")
                required = {
                    "frames": set(FRAME_COLUMNS),
                    "sleep_events": set(SLEEP_COLUMNS),
                    "cry_events": set(CRY_COLUMNS),
                }
                for table, columns in required.items():
                    available = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
                    if not columns.issubset(available):
                        raise TransferError(f"archive history table {table} is incompatible")
                counts = {
                    "frames": connection.execute("SELECT COUNT(*) FROM frames").fetchone()[0],
                    "storedImages": connection.execute(
                        "SELECT COUNT(*) FROM frames WHERE image_available = 1"
                    ).fetchone()[0],
                    "sleepEvents": connection.execute("SELECT COUNT(*) FROM sleep_events").fetchone()[0],
                    "cryEvents": connection.execute("SELECT COUNT(*) FROM cry_events").fetchone()[0],
                }
                if counts != manifest["counts"]:
                    raise TransferError("archive database counts do not match the manifest")
                images = {image["id"]: image for image in manifest["images"]}
                for row in connection.execute(
                    """SELECT id, relative_path, size_bytes, sha256 FROM frames
                       WHERE image_available = 1"""
                ):
                    image = images.get(row["id"])
                    if (
                        image is None
                        or image["relative_path"] != row["relative_path"]
                        or image["size_bytes"] != row["size_bytes"]
                        or image["sha256"] != row["sha256"]
                    ):
                        raise TransferError(f"archive image metadata for frame {row['id']} is inconsistent")
        except sqlite3.Error as exc:
            raise TransferError("archive history database is unreadable") from exc

    def new_incoming_path(self) -> Path:
        descriptor, raw_path = tempfile.mkstemp(prefix="upload-", suffix=".zip", dir=self.incoming_dir)
        os.close(descriptor)
        path = Path(raw_path)
        os.chmod(path, 0o600)
        return path

    def cleanup(self) -> None:
        with suppress(OSError):
            for path in self.incoming_dir.glob("upload-*.zip"):
                path.unlink(missing_ok=True)
        for pattern, root in (
            ("export-*", self.transfer_dir),
            ("retire-*", self.transfer_dir),
            ("validated-*", self.incoming_dir),
        ):
            for path in root.glob(pattern):
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
