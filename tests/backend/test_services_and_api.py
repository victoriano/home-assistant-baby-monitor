from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

from baby_monitor.main import create_app
from baby_monitor.models import (
    HAEntity,
    LightAlertConfig,
    NotificationConfig,
    SettingsPatch,
    SleepEventCreate,
    VisionLabel,
    utc_now,
)
from baby_monitor.services import CryAlertService, DashboardService, FrameService
from fastapi.testclient import TestClient


class FakeHomeAssistant:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def get_state(self, entity_id: str) -> dict[str, Any]:
        return {"entity_id": entity_id, "state": "off", "attributes": {}}

    async def call_service(self, domain: str, service: str, data: dict[str, Any]) -> list:
        self.calls.append((domain, service, data))
        return []


async def test_cry_alert_preserves_and_restores_selected_lights(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path, runtime="test", start_workers=False)
    app.state.settings.patch(
        SettingsPatch(
            lights=LightAlertConfig(
                entity_ids=["light.nursery", "light.hall"],
                duration_seconds=300,
                brightness_percent=40,
            ),
            notifications=NotificationConfig(service="notify.mobile_app_parent", targets=["parent_phone"]),
        )
    )
    fake = FakeHomeAssistant()
    service = CryAlertService(app.state.database, app.state.settings, fake)  # type: ignore[arg-type]
    event = await service.set_state("on", observed_at=utc_now(), source="manual", metadata={"test": True})
    assert event is not None
    assert fake.calls[0][0:2] == ("scene", "create")
    assert fake.calls[1][0:2] == ("light", "turn_on")
    assert fake.calls[1][2]["entity_id"] == ["light.nursery", "light.hall"]
    notification = next(call for call in fake.calls if call[0:2] == ("notify", "mobile_app_parent"))
    assert notification[2]["target"] == ["parent_phone"]
    closed = await service.set_state("off", observed_at=utc_now() + timedelta(seconds=1), source="manual")
    assert closed is not None and closed.ended_at is not None
    assert any(call[0:2] == ("scene", "turn_on") for call in fake.calls)
    await service.close()


def test_manual_sleep_history_and_cry_webhook(tmp_path: Path, ui_settings_payload: dict) -> None:
    app = create_app(data_dir=tmp_path, runtime="test", start_workers=False)
    with TestClient(app) as client:
        ui_settings_payload["baby"].update({"location_id": "granada", "location_name": "Granada"})
        assert client.put("/api/v1/settings", json=ui_settings_payload).status_code == 200
        start = utc_now() - timedelta(hours=1)
        created = client.post(
            "/api/v1/sleep",
            json={
                "started_at": start.isoformat(),
                "ended_at": (start + timedelta(minutes=30)).isoformat(),
                "kind": "nap",
                "source": "manual",
                "notes": "",
            },
        )
        assert created.status_code == 201, created.text
        assert created.json()["locationId"] == "granada"
        event_id = created.json()["id"]
        assert client.get("/api/v1/sleep").json()["total"] == 1
        assert client.patch(f"/api/v1/sleep/{event_id}", json={"notes": "Corrected"}).json()["notes"] == "Corrected"

        cry = client.post(
            "/api/v1/cry/webhook",
            json={"state": "on", "source": "manual", "metadata": {"detector": "test"}},
        )
        assert cry.status_code == 200
        assert cry.json()["active"] is True
        off = client.post("/api/v1/cry/webhook", json={"state": "off", "source": "manual"})
        assert off.status_code == 200 and off.json()["active"] is False
        summary = client.get("/api/v1/summary").json()
        assert summary["sleepTodayMinutes"] >= 0
        assert summary["lastCryAt"] is not None
        assert client.delete(f"/api/v1/sleep/{event_id}").status_code == 204
        started = client.post(
            "/api/v1/sleep/start",
            json={"started_at": utc_now().isoformat(), "kind": "nap", "source": "manual"},
        )
        assert started.status_code == 201, started.text
        stopped = client.post("/api/v1/sleep/stop", json={"ended_at": (utc_now() + timedelta(minutes=1)).isoformat()})
        assert stopped.status_code == 200, stopped.text


def test_camera_snapshot_is_private_app_data(tmp_path: Path, ui_settings_payload: dict) -> None:
    app = create_app(data_dir=tmp_path, runtime="test", start_workers=False)
    ui_settings_payload["camera"].update({"enabled": True, "entity_id": "camera.nursery"})

    async def snapshot(_: str) -> tuple[bytes, str]:
        return b"fake-private-jpeg", "image/jpeg"

    app.state.home_assistant.camera_snapshot = snapshot
    with TestClient(app) as client:
        assert client.put("/api/v1/settings", json=ui_settings_payload).status_code == 200
        response = client.post("/api/v1/camera/snapshot")
        assert response.status_code == 200, response.text
        record = response.json()
        assert record["imageUrl"].startswith("api/v1/frames/")
        image = client.get("/" + record["imageUrl"])
        assert image.content == b"fake-private-jpeg"
        assert image.headers["cache-control"] == "no-store"
        assert not (tmp_path / "www").exists()


def test_entity_discovery_does_not_expose_home_assistant_attributes(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path, runtime="test", start_workers=False)

    async def list_entities(_: str) -> list[HAEntity]:
        return [
            HAEntity(
                entity_id="camera.nursery",
                state="idle",
                name="Nursery",
                attributes={"access_token": "private-camera-token", "entity_picture": "/signed/path"},
            )
        ]

    app.state.home_assistant.list_entities = list_entities
    with TestClient(app) as client:
        response = client.get("/api/v1/home-assistant/entities?domain=camera")
        assert response.status_code == 200
        assert response.json()["items"][0]["attributes"] == {}
        assert "private-camera-token" not in response.text


def test_notification_connection_can_be_tested(tmp_path: Path, ui_settings_payload: dict) -> None:
    app = create_app(data_dir=tmp_path, runtime="test", start_workers=False)
    calls: list[tuple[str, str, dict[str, Any]]] = []

    async def call_service(domain: str, service: str, data: dict[str, Any]) -> list:
        calls.append((domain, service, data))
        return []

    app.state.home_assistant.call_service = call_service
    ui_settings_payload["notifications"] = {
        "service": "notify.mobile_app_parent",
        "targets": ["parent_phone"],
    }
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/settings/test/notifications",
            json=ui_settings_payload,
        )
        assert response.status_code == 200
        assert response.json()["ok"] is True
    assert calls == [
        (
            "notify",
            "mobile_app_parent",
            {
                "title": "Baby Monitor test",
                "message": "The Baby Monitor notification connection works.",
                "target": ["parent_phone"],
            },
        )
    ]


async def test_two_consecutive_vision_labels_start_and_stop_sleep(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path, runtime="test", start_workers=False)
    service = FrameService(app.state.database, app.state.settings, FakeHomeAssistant())  # type: ignore[arg-type]
    base = utc_now() - timedelta(minutes=10)
    asleep = VisionLabel(
        baby_present=True,
        state="asleep",
        confidence=0.92,
        description="Asleep",
        tags=[],
    )
    first = app.state.database.add_frame(b"one", "image/jpeg", base, label=asleep)
    second = app.state.database.add_frame(b"two", "image/jpeg", base + timedelta(minutes=5), label=asleep)
    await service._reconcile_vision_sleep(second.id)
    event = app.state.database.open_sleep_event()
    assert event is not None
    assert event.source == "vision"
    assert event.started_at == first.captured_at

    awake = VisionLabel(
        baby_present=True,
        state="awake",
        confidence=0.9,
        description="Awake",
        tags=[],
    )
    third = app.state.database.add_frame(b"three", "image/jpeg", base + timedelta(minutes=10), label=awake)
    fourth = app.state.database.add_frame(b"four", "image/jpeg", base + timedelta(minutes=15), label=awake)
    await service._reconcile_vision_sleep(fourth.id)
    closed = app.state.database.get_sleep_event(event.id)
    assert closed is not None and closed.ended_at == third.captured_at


def test_prediction_blends_recent_wake_intervals_with_age_baseline(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path, runtime="test", start_workers=False)
    base = utc_now() - timedelta(hours=20)
    latest = None
    for index in range(5):
        start = base + timedelta(hours=index * 3)
        latest = app.state.database.add_sleep_event(
            SleepEventCreate(
                started_at=start,
                ended_at=start + timedelta(hours=1),
                kind="nap",
                source="manual",
            )
        )
    assert latest is not None and latest.ended_at is not None

    result = DashboardService(app.state.database, app.state.settings).summary()
    learned_minutes = (result["next_sleep_at"] - latest.ended_at).total_seconds() / 60
    assert 120 < learned_minutes < 180
    assert result["prediction_confidence"] > 0.6
    assert "recent wake intervals" in result["prediction_reason"]
