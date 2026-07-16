from __future__ import annotations

import json
from pathlib import Path

from baby_monitor.main import create_app
from fastapi.testclient import TestClient


def test_settings_are_typed_encrypted_and_write_only(tmp_path: Path, ui_settings_payload: dict) -> None:
    app = create_app(data_dir=tmp_path, runtime="test", start_workers=False)
    payload = ui_settings_payload
    payload["ai"].update(
        {
            "provider": "openai",
            "model": "gpt-5.6-luna",
            "cloud_image_consent": True,
        }
    )
    payload["secrets"]["ai_api_key"] = "fake-ai-key-never-return-me"
    with TestClient(app) as client:
        response = client.put("/api/v1/settings", json=payload)
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["configured"] is True
        assert data["ai"]["api_key_configured"] is True
        assert "api_key" not in data["ai"]
        assert "fake-ai-key-never-return-me" not in response.text

    assert "fake-ai-key-never-return-me" not in (tmp_path / "settings.json").read_text()
    assert "fake-ai-key-never-return-me" not in (tmp_path / "secrets.enc.json").read_text()
    assert (tmp_path / ".secret.key").stat().st_mode & 0o077 == 0
    assert (tmp_path / "secrets.enc.json").stat().st_mode & 0o077 == 0

    restarted = create_app(data_dir=tmp_path, runtime="test", start_workers=False)
    with TestClient(restarted) as client:
        data = client.get("/api/v1/settings").json()
        assert data["baby"]["name"] == "Alex"
        assert data["ai"]["api_key_configured"] is True


def test_invalid_retention_and_entity_ids_are_rejected(tmp_path: Path, ui_settings_payload: dict) -> None:
    app = create_app(data_dir=tmp_path, runtime="test", start_workers=False)
    with TestClient(app) as client:
        ui_settings_payload["retention"] = {"mode": "days", "days": 0}
        assert client.put("/api/v1/settings", json=ui_settings_payload).status_code == 422
        ui_settings_payload["retention"] = {"mode": "days", "days": 30}
        ui_settings_payload["lights"]["entity_ids"] = ["switch.not_a_light"]
        assert client.put("/api/v1/settings", json=ui_settings_payload).status_code == 422
        ui_settings_payload["lights"]["entity_ids"] = ["light.safe/../../config"]
        assert client.put("/api/v1/settings", json=ui_settings_payload).status_code == 422


def test_canonical_round_trip_preserves_private_connections_and_notifications(
    tmp_path: Path, ui_settings_payload: dict
) -> None:
    ui_settings_payload["home_assistant"] = {
        "mode": "standalone",
        "base_url": "http://homeassistant.local:8123",
    }
    ui_settings_payload["camera"] = {
        "enabled": True,
        "entity_id": None,
        "capture_interval_seconds": 120,
    }
    ui_settings_payload["cry"].update({"mode": "audio", "entity_id": None})
    ui_settings_payload["ai"].update(
        {
            "provider": "gemini",
            "model": "gemini-3.1-flash-lite",
            "cloud_image_consent": True,
        }
    )
    ui_settings_payload["notifications"] = {
        "recipients": [
            {
                "person_entity_id": "person.parent",
                "name": "Parent",
                "notify_service": "notify.mobile_app_parent",
                "targets": ["parent_phone"],
                "enabled": True,
                "language": "en",
                "events": ["cry_started", "sleep_predicted_soon"],
            }
        ],
        "lead_minutes": 10,
    }
    secrets = {
        "home_assistant_access_token": "ha-private-token",
        "camera_stream_url": "rtsp://camera-user:camera-pass@camera.local/live",
        "cry_audio_stream_url": "rtsp://camera-user:camera-pass@camera.local/audio",
        "ai_api_key": "gemini-private-key",
        "clear": [],
    }
    ui_settings_payload["secrets"] = secrets

    app = create_app(data_dir=tmp_path, runtime="test", start_workers=False)
    with TestClient(app) as client:
        response = client.put("/api/v1/settings", json=ui_settings_payload)
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["home_assistant"]["access_token_configured"] is True
        assert data["camera"]["stream_url_configured"] is True
        assert data["cry"]["mode"] == "rtsp_audio"
        assert data["cry"]["audio_stream_url_configured"] is True
        assert data["ai"]["provider"] == "gemini"
        assert data["ai"]["api_key_configured"] is True
        assert data["notifications"] == {
            "recipients": [
                {
                    "person_entity_id": "person.parent",
                    "name": "Parent",
                    "notify_service": "notify.mobile_app_parent",
                    "targets": ["parent_phone"],
                    "enabled": True,
                    "language": "en",
                    "events": ["cry_started", "sleep_predicted_soon"],
                }
            ],
            "lead_minutes": 10,
        }
        response_text = response.text
        for secret in secrets.values():
            if isinstance(secret, str):
                assert secret not in response_text
        assert client.get("/api/v1/settings").json() == data

    persisted = "".join(
        path.read_text(errors="ignore") for path in (tmp_path / "settings.json", tmp_path / "secrets.enc.json")
    )
    assert "camera-pass" not in persisted
    assert "gemini-private-key" not in persisted


def test_endpoint_changes_require_new_bound_credentials(tmp_path: Path, ui_settings_payload: dict) -> None:
    ui_settings_payload["home_assistant"] = {
        "mode": "standalone",
        "base_url": "http://homeassistant-one.local:8123",
    }
    ui_settings_payload["ai"].update(
        {
            "provider": "openai",
            "model": "gpt-5.6-luna",
            "cloud_image_consent": True,
        }
    )
    ui_settings_payload["secrets"].update(
        {
            "home_assistant_access_token": "first-ha-token",
            "ai_api_key": "first-ai-key",
        }
    )
    app = create_app(data_dir=tmp_path, runtime="test", start_workers=False)
    with TestClient(app) as client:
        assert client.put("/api/v1/settings", json=ui_settings_payload).status_code == 200

        changed_ha = json.loads(json.dumps(ui_settings_payload))
        changed_ha["home_assistant"]["base_url"] = "http://homeassistant-two.local:8123"
        changed_ha["secrets"].pop("home_assistant_access_token")
        changed_ha["secrets"].pop("ai_api_key")
        response = client.put("/api/v1/settings", json=changed_ha)
        assert response.status_code == 422
        assert "re-entering its access token" in response.text
        response = client.post("/api/v1/settings/test/home_assistant", json=changed_ha)
        assert response.status_code == 422
        assert "re-entering its access token" in response.text

        changed_ai = json.loads(json.dumps(ui_settings_payload))
        changed_ai["ai"].update({"provider": "gemini", "model": "gemini-3.1-flash-lite"})
        changed_ai["secrets"].pop("home_assistant_access_token")
        changed_ai["secrets"].pop("ai_api_key")
        response = client.put("/api/v1/settings", json=changed_ai)
        assert response.status_code == 422
        assert "re-entering its API key" in response.text
        response = client.post("/api/v1/settings/test/vision", json=changed_ai)
        assert response.status_code == 422
        assert "re-entering its API key" in response.text


def test_cloud_ai_rejects_custom_base_url(tmp_path: Path, ui_settings_payload: dict) -> None:
    ui_settings_payload["ai"].update(
        {
            "provider": "openai",
            "model": "gpt-5.6-luna",
            "base_url": "https://attacker.example/v1",
            "cloud_image_consent": True,
        }
    )
    ui_settings_payload["secrets"]["ai_api_key"] = "private-ai-key"
    app = create_app(data_dir=tmp_path, runtime="test", start_workers=False)
    with TestClient(app) as client:
        response = client.put("/api/v1/settings", json=ui_settings_payload)
        assert response.status_code == 422
        assert "base_url is only valid" in response.text


def test_local_ai_endpoint_also_requires_image_sharing_consent(
    tmp_path: Path,
    ui_settings_payload: dict,
) -> None:
    ui_settings_payload["ai"].update(
        {
            "provider": "ollama",
            "model": "qwen2.5vl:3b",
            "base_url": "http://ollama.local:11434/v1",
            "cloud_image_consent": False,
        }
    )
    app = create_app(data_dir=tmp_path, runtime="test", start_workers=False)
    with TestClient(app) as client:
        response = client.put("/api/v1/settings", json=ui_settings_payload)
        assert response.status_code == 422
        assert "sending images to any AI endpoint" in response.text


def test_validation_errors_never_echo_secret_input(tmp_path: Path, ui_settings_payload: dict) -> None:
    secret = "private-camera-password-never-echo"
    invalid_url = f"rtsp://user:{secret}@camera.local/live\nfile '/etc/passwd'"
    ui_settings_payload["camera"]["enabled"] = True
    ui_settings_payload["secrets"]["camera_stream_url"] = invalid_url
    app = create_app(data_dir=tmp_path, runtime="test", start_workers=False)
    with TestClient(app) as client:
        for method, path in (
            ("put", "/api/v1/settings"),
            ("post", "/api/v1/settings/test/camera"),
        ):
            response = getattr(client, method)(path, json=ui_settings_payload)
            assert response.status_code == 422
            assert secret not in response.text
            assert invalid_url not in response.text

        response = client.patch(
            "/api/v1/settings",
            json={"secrets": {"camera_stream_url": invalid_url, "clear": []}},
        )
        assert response.status_code == 422
        assert secret not in response.text


def test_openai_compatible_alias_round_trip(tmp_path: Path, ui_settings_payload: dict) -> None:
    ui_settings_payload["ai"].update(
        {
            "provider": "local",
            "model": "local-vision-model",
            "base_url": "http://127.0.0.1:11434/v1",
            "cloud_image_consent": True,
        }
    )
    app = create_app(data_dir=tmp_path, runtime="test", start_workers=False)
    with TestClient(app) as client:
        response = client.put("/api/v1/settings", json=ui_settings_payload)
        assert response.status_code == 200, response.text
        assert response.json()["ai"]["provider"] == "ollama"
        test = client.post("/api/v1/settings/test/vision", json={**ui_settings_payload, "ai": {"provider": "disabled"}})
        assert test.status_code == 200
        assert test.json()["ok"] is True
