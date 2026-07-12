from __future__ import annotations

from pathlib import Path

import httpx
from baby_monitor.home_assistant import HomeAssistantClient
from baby_monitor.models import SecretChanges, SettingsWrite
from baby_monitor.security import EncryptedSecretStore
from baby_monitor.settings import SettingsRepository, SettingsService


def settings_service(tmp_path: Path) -> SettingsService:
    return SettingsService(SettingsRepository(tmp_path), EncryptedSecretStore(tmp_path))


async def test_supervisor_entity_discovery_camera_and_service_calls(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SUPERVISOR_TOKEN", "supervisor-secret")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.headers["authorization"] == "Bearer supervisor-secret"
        if request.url.path.endswith("/states"):
            return httpx.Response(
                200,
                json=[
                    {
                        "entity_id": "camera.nursery",
                        "state": "idle",
                        "attributes": {"friendly_name": "Nursery"},
                    },
                    {"entity_id": "light.hall", "state": "off", "attributes": {}},
                ],
            )
        if request.url.path.endswith("/services"):
            return httpx.Response(
                200,
                json=[
                    {
                        "domain": "notify",
                        "services": {"mobile_app_parent": {"name": "Parent phone"}},
                    }
                ],
            )
        if "/camera_proxy/" in request.url.path:
            return httpx.Response(200, content=b"jpeg", headers={"content-type": "image/jpeg"})
        if "/services/light/turn_on" in request.url.path:
            return httpx.Response(200, json=[])
        return httpx.Response(404)

    client = HomeAssistantClient(settings_service(tmp_path), transport=httpx.MockTransport(handler))
    cameras = await client.list_entities("camera")
    assert [(item.entity_id, item.name) for item in cameras] == [("camera.nursery", "Nursery")]
    notifications = await client.list_entities("notify")
    assert notifications[0].entity_id == "notify.mobile_app_parent"
    assert await client.camera_snapshot("camera.nursery") == (b"jpeg", "image/jpeg")
    await client.call_service("light", "turn_on", {"entity_id": "light.hall"})
    assert any(request.url.path.endswith("/core/api/services/light/turn_on") for request in seen)


async def test_standalone_uses_configured_url_and_encrypted_token(tmp_path: Path) -> None:
    service = settings_service(tmp_path)
    service.replace(
        SettingsWrite(
            home_assistant={
                "mode": "standalone",
                "base_url": "http://ha.internal:8123",
            },
            secrets=SecretChanges(home_assistant_access_token="ha-token"),
        )
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "http://ha.internal:8123/api/"
        assert request.headers["authorization"] == "Bearer ha-token"
        return httpx.Response(200, json={"message": "API running."})

    client = HomeAssistantClient(service, transport=httpx.MockTransport(handler))
    assert await client.ping() == {"message": "API running."}


async def test_dynamic_path_segments_are_percent_encoded(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SUPERVISOR_TOKEN", "supervisor-secret")
    seen: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.raw_path)
        return httpx.Response(200, json={"state": "off", "attributes": {}})

    client = HomeAssistantClient(settings_service(tmp_path), transport=httpx.MockTransport(handler))
    await client.get_state("camera.foo/../../config")
    await client.call_service("notify", "foo/../../light/turn_on", {})

    assert seen[0].startswith(b"/core/api/states/camera.foo%2F..%2F..%2Fconfig")
    assert seen[1].startswith(b"/core/api/services/notify/foo%2F..%2F..%2Flight%2Fturn_on")
    assert all(b"/api/config" not in path for path in seen)
