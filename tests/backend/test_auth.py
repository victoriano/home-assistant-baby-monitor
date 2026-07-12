from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from baby_monitor.main import create_app
from fastapi.testclient import TestClient


def test_standalone_rejects_short_admin_token(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BABY_MONITOR_ADMIN_TOKEN", "too-short")
    with pytest.raises(RuntimeError, match="at least 32 bytes"):
        create_app(data_dir=tmp_path, runtime="standalone", start_workers=False)


def test_standalone_requires_token_and_uses_browser_session(tmp_path: Path, monkeypatch) -> None:
    admin_token = "correct-horse-battery-staple-123456789"
    monkeypatch.setenv("BABY_MONITOR_ADMIN_TOKEN", admin_token)
    app = create_app(data_dir=tmp_path, runtime="standalone", start_workers=False)
    with TestClient(app) as client:
        response = client.get("/", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "./login"
        assert client.get("/api/v1/settings").status_code == 401
        assert client.post("/login", data={"token": "wrong"}).status_code == 401
        login = client.post("/login", data={"token": admin_token}, follow_redirects=False)
        assert login.status_code == 303
        assert admin_token not in login.headers.get("set-cookie", "")
        settings = client.get("/api/v1/settings")
        assert settings.status_code == 200
        assert settings.headers["cache-control"] == "no-store"
        assert "default-src 'self'" in settings.headers["content-security-policy"]
        assert settings.headers["referrer-policy"] == "no-referrer"
        assert settings.headers["x-content-type-options"] == "nosniff"


def test_standalone_runtime_requires_ui_connection_settings(
    tmp_path: Path,
    monkeypatch,
    ui_settings_payload: dict,
) -> None:
    admin_token = "standalone-admin-token-at-least-32-bytes"
    monkeypatch.setenv("BABY_MONITOR_ADMIN_TOKEN", admin_token)
    app = create_app(data_dir=tmp_path, runtime="standalone", start_workers=False)
    headers = {"Authorization": f"Bearer {admin_token}"}
    with TestClient(app) as client:
        initial = client.get("/api/v1/settings", headers=headers)
        assert initial.status_code == 200
        assert initial.json()["home_assistant"]["mode"] == "standalone"

        response = client.put("/api/v1/settings", json=ui_settings_payload, headers=headers)
        assert response.status_code == 422
        assert "standalone Home Assistant connection mode" in response.text

        ui_settings_payload["home_assistant"] = {
            "mode": "standalone",
            "base_url": "http://homeassistant.local:8123",
        }
        response = client.put("/api/v1/settings", json=ui_settings_payload, headers=headers)
        assert response.status_code == 422
        assert "requires base_url and access token" in response.text

        ui_settings_payload["secrets"]["home_assistant_access_token"] = "ha-access-token"
        response = client.put("/api/v1/settings", json=ui_settings_payload, headers=headers)
        assert response.status_code == 200
        assert response.json()["home_assistant"]["access_token_configured"] is True


async def test_ingress_rejects_synthetic_or_forwarded_peer(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path, runtime="home_assistant_app", start_workers=False)
    untrusted = httpx.ASGITransport(app=app, client=("198.51.100.4", 1234))
    async with httpx.AsyncClient(transport=untrusted, base_url="http://test") as client:
        response = await client.get(
            "/api/v1/settings",
            headers={"X-Forwarded-For": "172.30.32.2", "X-Hass-Is-Admin": "true"},
        )
        assert response.status_code == 403

    trusted = httpx.ASGITransport(app=app, client=("172.30.32.2", 1234))
    async with httpx.AsyncClient(transport=trusted, base_url="http://test") as client:
        assert (await client.get("/api/v1/settings", headers={"X-Hass-Is-Admin": "true"})).status_code == 200
        assert (await client.get("/api/v1/settings", headers={"X-Hass-Is-Admin": "false"})).status_code == 403
