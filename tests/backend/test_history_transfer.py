from __future__ import annotations

import csv
import io
import json
import zipfile
from datetime import timedelta
from pathlib import Path

from baby_monitor.main import create_app
from baby_monitor.models import CryEventCreate, SleepEventCreate, utc_now
from fastapi.testclient import TestClient


def _configured_payload(location_id: str, location_name: str) -> dict:
    return {
        "schema_version": 1,
        "baby": {
            "name": "Alex",
            "birth_date": None,
            "timezone": "Europe/Madrid",
            "location_id": location_id,
            "location_name": location_name,
        },
        "home_assistant": {"mode": "auto", "base_url": None},
        "camera": {"enabled": False, "entity_id": None, "capture_interval_seconds": 300},
        "cry": {
            "mode": "disabled",
            "entity_id": None,
            "positive_windows": 2,
            "window_seconds": 0.5,
            "clear_after_seconds": 8,
            "sensitivity": "balanced",
        },
        "lights": {
            "entity_ids": [],
            "duration_seconds": 45,
            "brightness_percent": 35,
            "color_rgb": [255, 125, 72],
            "restore_previous_state": True,
        },
        "ai": {
            "provider": "disabled",
            "model": None,
            "base_url": None,
            "cloud_image_consent": False,
            "detail": "low",
        },
        "notifications": {"recipients": [], "lead_minutes": 10},
        "retention": {"mode": "forever", "days": None},
        "secrets": {"camera_stream_url": "rtsp://private.example.test/camera", "clear": []},
    }


def _source_archive(source_dir: Path) -> tuple[bytes, str, str]:
    app = create_app(data_dir=source_dir, runtime="test", start_workers=False)
    with TestClient(app) as client:
        assert client.put("/api/v1/settings", json=_configured_payload("madrid", "Madrid")).status_code == 200
        captured = utc_now() - timedelta(minutes=15)
        frame = app.state.database.add_frame(
            b"private-jpeg-payload",
            "image/jpeg",
            captured,
            camera_entity_id="camera.crib",
            location_id="madrid",
        )
        app.state.database.add_sleep_event(
            SleepEventCreate(
                started_at=captured - timedelta(hours=1),
                ended_at=captured,
                kind="nap",
                source="manual",
                location_id="madrid",
            )
        )
        app.state.database.add_cry_event(
            CryEventCreate(
                detected_at=captured - timedelta(minutes=5),
                ended_at=captured - timedelta(minutes=4),
                source="manual",
                location_id="madrid",
            )
        )
        prepared = client.post("/api/v1/history-transfer/exports")
        assert prepared.status_code == 200, prepared.text
        export = prepared.json()
        assert export["counts"] == {"frames": 1, "storedImages": 1, "sleepEvents": 1, "cryEvents": 1}
        assert client.get("/api/v1/history-transfer").json()["status"] == "pending"
        blocked = client.post(
            "/api/v1/sleep/start",
            json={"started_at": utc_now().isoformat(), "kind": "nap", "source": "manual"},
        )
        assert blocked.status_code == 409
        downloaded = client.get("/" + export["downloadUrl"])
        assert downloaded.status_code == 200
        return downloaded.content, export["manifestSha256"], frame.id


def _import_receipt(destination_dir: Path, archive_bytes: bytes) -> dict:
    destination = create_app(data_dir=destination_dir, runtime="test", start_workers=False)
    with TestClient(destination) as client:
        response = client.post("/api/v1/history-transfer/imports", content=archive_bytes)
        assert response.status_code == 200, response.text
        return response.json()["receipt"]


def test_public_export_is_analyzable_and_import_is_lossless(tmp_path: Path) -> None:
    archive_bytes, manifest_sha256, frame_id = _source_archive(tmp_path / "source")
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        names = set(archive.namelist())
        assert {
            "README.txt",
            "manifest.json",
            "internal/history.sqlite3",
            "data/frames.csv",
            "data/sleep_events.csv",
            "data/cry_events.csv",
        }.issubset(names)
        assert f"images/madrid/{archive.read('data/frames.csv').decode().splitlines()[1].split(',')[4]}" in names
        assert "settings.json" not in names
        assert "secrets.enc.json" not in names
        assert ".secret.key" not in names
        assert b"private.example.test" not in archive_bytes
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["counts"]["storedImages"] == 1
        rows = list(csv.DictReader(io.StringIO(archive.read("data/frames.csv").decode())))
        assert rows[0]["id"] == frame_id
        assert rows[0]["location_id"] == "madrid"
        assert rows[0]["archive_image_path"].startswith("images/madrid/")

    destination = create_app(data_dir=tmp_path / "destination", runtime="test", start_workers=False)
    with TestClient(destination) as client:
        assert client.put("/api/v1/settings", json=_configured_payload("granada", "Granada")).status_code == 200
        response = client.post(
            "/api/v1/history-transfer/imports",
            content=archive_bytes,
            headers={"Content-Type": "application/zip"},
        )
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["idempotent"] is False
        assert result["receipt"]["manifestSha256"] == manifest_sha256
        assert result["status"]["status"] == "active"
        settings = client.get("/api/v1/settings").json()
        assert settings["baby"]["location_id"] == "granada"
        assert settings["camera"]["stream_url_configured"] is True
        assert client.get("/api/v1/frames").json()["total"] == 1
        assert client.get(f"/api/v1/frames/{frame_id}/image").content == b"private-jpeg-payload"
        assert client.get("/api/v1/sleep").json()["total"] == 1
        assert client.get("/api/v1/cry-events").json()["total"] == 1

        repeated = client.post(
            "/api/v1/history-transfer/imports",
            content=archive_bytes,
            headers={"Content-Type": "application/zip"},
        )
        assert repeated.status_code == 200, repeated.text
        assert repeated.json()["idempotent"] is True


def test_import_requires_confirmation_before_replacing_existing_history(tmp_path: Path) -> None:
    archive_bytes, _, _ = _source_archive(tmp_path / "source")
    destination = create_app(data_dir=tmp_path / "destination", runtime="test", start_workers=False)
    existing = destination.state.database.add_frame(b"granada", "image/jpeg", utc_now(), location_id="granada")
    with TestClient(destination) as client:
        refused = client.post("/api/v1/history-transfer/imports", content=archive_bytes)
        assert refused.status_code == 409
        assert destination.state.database.get_frame(existing.id) is not None

        accepted = client.post("/api/v1/history-transfer/imports?replace=true", content=archive_bytes)
        assert accepted.status_code == 200, accepted.text
        assert destination.state.database.get_frame(existing.id) is None


def test_corrupt_image_is_rejected_without_touching_destination(tmp_path: Path) -> None:
    archive_bytes, _, _ = _source_archive(tmp_path / "source")
    original = zipfile.ZipFile(io.BytesIO(archive_bytes))
    tampered_buffer = io.BytesIO()
    with original, zipfile.ZipFile(tampered_buffer, "w") as tampered:
        for info in original.infolist():
            content = original.read(info.filename)
            if info.filename.startswith("images/"):
                content = b"tampered-image"
            tampered.writestr(info.filename, content, compress_type=info.compress_type)

    destination = create_app(data_dir=tmp_path / "destination", runtime="test", start_workers=False)
    existing = destination.state.database.add_frame(b"keep-me", "image/jpeg", utc_now(), location_id="granada")
    with TestClient(destination) as client:
        response = client.post(
            "/api/v1/history-transfer/imports?replace=true",
            content=tampered_buffer.getvalue(),
        )
        assert response.status_code == 409
        assert destination.state.database.get_frame(existing.id) is not None
        assert destination.state.database.get_frame_path(existing.id).read_bytes() == b"keep-me"


def test_pending_export_can_be_cancelled_and_writes_resume(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path, runtime="test", start_workers=False)
    with TestClient(app) as client:
        assert client.post("/api/v1/history-transfer/exports").status_code == 200
        assert client.get("/api/v1/history-transfer").json()["writable"] is False
        cancelled = client.post("/api/v1/history-transfer/cancel")
        assert cancelled.status_code == 200
        assert cancelled.json()["writable"] is True
        created = client.post(
            "/api/v1/sleep/start",
            json={"started_at": utc_now().isoformat(), "kind": "nap", "source": "manual"},
        )
        assert created.status_code == 201, created.text


def test_matching_import_receipt_retires_only_source_history(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    archive_bytes, _, _ = _source_archive(source_dir)
    receipt = _import_receipt(tmp_path / "destination", archive_bytes)
    source = create_app(data_dir=source_dir, runtime="test", start_workers=False)
    with TestClient(source) as client:
        settings_before = client.get("/api/v1/settings").json()
        response = client.post(
            "/api/v1/history-transfer/finalize?delete=true",
            json=receipt,
        )
        assert response.status_code == 200, response.text
        assert response.json()["status"]["status"] == "retired"
        assert source.state.database.summary()["frames"] == 0
        assert source.state.database.summary()["sleep_events"] == 0
        assert source.state.database.summary()["cry_events"] == 0
        assert list((source_dir / "frames").rglob("*.jpg")) == []
        settings_after = client.get("/api/v1/settings").json()
        assert settings_after == settings_before
        blocked = client.post(
            "/api/v1/sleep/start",
            json={"started_at": utc_now().isoformat(), "kind": "nap", "source": "manual"},
        )
        assert blocked.status_code == 409


def test_wrong_receipt_never_deletes_source_history(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    archive_bytes, _, frame_id = _source_archive(source_dir)
    receipt = _import_receipt(tmp_path / "destination", archive_bytes)
    receipt["manifestSha256"] = "0" * 64
    source = create_app(data_dir=source_dir, runtime="test", start_workers=False)
    with TestClient(source) as client:
        response = client.post(
            "/api/v1/history-transfer/finalize?delete=true",
            json=receipt,
        )
        assert response.status_code == 409
        assert source.state.database.get_frame(frame_id) is not None
        status = client.get("/api/v1/history-transfer").json()
        assert status["status"] == "pending"
        assert status["outgoing"] is not None


def test_interrupted_retirement_recovers_as_pending_not_writable(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    _source_archive(source_dir)
    state_path = source_dir / ".history-transfer-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["status"] = "preparing"
    state["operation"] = "retire"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    recovered = create_app(data_dir=source_dir, runtime="test", start_workers=False)
    with TestClient(recovered) as client:
        status = client.get("/api/v1/history-transfer").json()
        assert status["status"] == "pending"
        assert status["writable"] is False
        assert status["outgoing"] is not None


def test_interrupted_import_recovers_retired_until_archive_is_retried(tmp_path: Path) -> None:
    destination_dir = tmp_path / "destination"
    create_app(data_dir=destination_dir, runtime="test", start_workers=False)
    state_path = destination_dir / ".history-transfer-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["status"] = "preparing"
    state["operation"] = "import"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    recovered = create_app(data_dir=destination_dir, runtime="test", start_workers=False)
    with TestClient(recovered) as client:
        status = client.get("/api/v1/history-transfer").json()
        assert status["status"] == "retired"
        assert status["writable"] is False
