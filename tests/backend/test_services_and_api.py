from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

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


def test_frontend_missing_assets_do_not_fall_back_to_html(tmp_path: Path) -> None:
    frontend_dir = tmp_path / "frontend"
    assets_dir = frontend_dir / "assets"
    assets_dir.mkdir(parents=True)
    (frontend_dir / "index.html").write_text("<html><body>Baby Monitor</body></html>")
    (assets_dir / "current.js").write_text("export const ready = true;")
    app = create_app(
        data_dir=tmp_path / "data",
        frontend_dir=frontend_dir,
        runtime="test",
        start_workers=False,
    )

    with TestClient(app) as client:
        current = client.get("/assets/current.js")
        assert current.status_code == 200
        assert "javascript" in current.headers["content-type"]
        assert current.headers["cache-control"] == "no-cache, max-age=0, must-revalidate"

        missing = client.get("/assets/previous-build.js")
        assert missing.status_code == 404
        assert "html" not in missing.headers["content-type"]

        client_route = client.get("/settings")
        assert client_route.status_code == 200
        assert client_route.headers["content-type"].startswith("text/html")
        assert "Baby Monitor" in client_route.text


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
    assert any(call[0:2] == ("scene", "create") for call in fake.calls)
    light_calls = [call for call in fake.calls if call[0:2] == ("light", "turn_on")]
    assert [call[2]["entity_id"] for call in light_calls] == ["light.nursery", "light.hall"]
    notification = next(call for call in fake.calls if call[0:2] == ("notify", "mobile_app_parent"))
    assert notification[2]["target"] == ["parent_phone"]
    closed = await service.set_state("off", observed_at=utc_now() + timedelta(seconds=1), source="manual")
    assert closed is not None and closed.ended_at is not None
    assert any(call[0:2] == ("scene", "turn_on") for call in fake.calls)
    await service.close()


async def test_cry_notification_is_not_blocked_by_slow_lights(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path, runtime="test", start_workers=False)
    app.state.settings.patch(
        SettingsPatch(
            lights=LightAlertConfig(entity_ids=["light.nursery"]),
            notifications=NotificationConfig(service="notify.mobile_app_parent"),
        )
    )
    fake = FakeHomeAssistant()
    light_started = asyncio.Event()
    release_light = asyncio.Event()
    notification_sent = asyncio.Event()

    async def call_service(domain: str, service_name: str, data: dict[str, Any]) -> list:
        fake.calls.append((domain, service_name, data))
        if (domain, service_name) == ("scene", "create"):
            light_started.set()
            await release_light.wait()
        if domain == "notify":
            notification_sent.set()
        return []

    fake.call_service = call_service  # type: ignore[method-assign]
    service = CryAlertService(app.state.database, app.state.settings, fake)  # type: ignore[arg-type]
    task = asyncio.create_task(service.set_state("on", observed_at=utc_now(), source="audio"))
    await asyncio.wait_for(light_started.wait(), 0.5)
    await asyncio.wait_for(notification_sent.wait(), 0.5)
    release_light.set()
    await task
    await service.close()


async def test_cry_alert_uses_xy_for_hue_style_lights(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path, runtime="test", start_workers=False)
    app.state.settings.patch(
        SettingsPatch(
            lights=LightAlertConfig(
                entity_ids=["light.hue_strip"],
                duration_seconds=300,
                brightness_percent=35,
                color_rgb=(255, 125, 72),
            )
        )
    )
    fake = FakeHomeAssistant()

    async def get_state(entity_id: str) -> dict[str, Any]:
        return {
            "entity_id": entity_id,
            "state": "on",
            "attributes": {
                "brightness": 120,
                "color_mode": "xy",
                "supported_color_modes": ["xy"],
                "xy_color": [0.3, 0.32],
            },
        }

    fake.get_state = get_state  # type: ignore[method-assign]
    service = CryAlertService(app.state.database, app.state.settings, fake)  # type: ignore[arg-type]
    await service.set_state("on", observed_at=utc_now(), source="audio")

    alert = next(call for call in fake.calls if call[0:2] == ("light", "turn_on"))
    assert alert[2]["entity_id"] == "light.hue_strip"
    assert alert[2]["xy_color"] == [0.5842, 0.3506]
    assert "rgb_color" not in alert[2]
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
                "details": {
                    "tags": ["in_bed", "woke_happy"],
                    "pauses": [
                        {
                            "started_at": (start + timedelta(minutes=10)).isoformat(),
                            "ended_at": (start + timedelta(minutes=15)).isoformat(),
                        }
                    ],
                },
            },
        )
        assert created.status_code == 201, created.text
        assert created.json()["locationId"] == "granada"
        assert created.json()["details"]["tags"] == ["in_bed", "woke_happy"]
        assert len(created.json()["details"]["pauses"]) == 1
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
        predictions = client.get("/api/v1/predictions")
        assert predictions.status_code == 200
        assert len(predictions.json()["plans"]) == 2
        assert client.delete(f"/api/v1/sleep/{event_id}").status_code == 204
        started = client.post(
            "/api/v1/sleep/start",
            json={"started_at": utc_now().isoformat(), "kind": "nap", "source": "manual"},
        )
        assert started.status_code == 201, started.text
        stopped = client.post("/api/v1/sleep/stop", json={"ended_at": (utc_now() + timedelta(minutes=1)).isoformat()})
        assert stopped.status_code == 200, stopped.text


def test_sleep_editor_lists_every_interval_frame_and_deletion_preserves_images(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path, runtime="test", start_workers=False)
    start = utc_now() - timedelta(hours=2)
    end = start + timedelta(hours=1)
    frames = [
        app.state.database.add_frame(
            f"frame-{index}".encode(),
            "image/jpeg",
            start + timedelta(minutes=index * 20),
            location_id="granada",
        )
        for index in range(4)
    ]
    app.state.database.add_frame(b"other-home", "image/jpeg", start + timedelta(minutes=10), location_id="madrid")
    event = app.state.database.add_sleep_event(
        SleepEventCreate(
            started_at=start,
            ended_at=end,
            kind="nap",
            source="vision",
            location_id="granada",
        )
    )

    with TestClient(app) as client:
        first_page = client.get(
            "/api/v1/frames/range",
            params={
                "start": start.isoformat(),
                "end": end.isoformat(),
                "location_id": "granada",
                "limit": 2,
            },
        )
        assert first_page.status_code == 200, first_page.text
        assert first_page.json()["total"] == 4
        assert [item["id"] for item in first_page.json()["items"]] == [frames[0].id, frames[1].id]

        second_page = client.get(
            "/api/v1/frames/range",
            params={
                "start": start.isoformat(),
                "end": end.isoformat(),
                "location_id": "granada",
                "limit": 2,
                "offset": 2,
            },
        )
        assert [item["id"] for item in second_page.json()["items"]] == [frames[2].id, frames[3].id]
        assert client.get(
            "/api/v1/frames/range",
            params={"start": end.isoformat(), "end": start.isoformat()},
        ).status_code == 400

        assert client.delete(f"/api/v1/sleep/{event.id}").status_code == 204
        assert client.get(f"/api/v1/frames/{frames[0].id}/image").content == b"frame-0"


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


def test_camera_webrtc_negotiates_through_the_fixed_local_relay(tmp_path: Path, ui_settings_payload: dict) -> None:
    app = create_app(data_dir=tmp_path, runtime="test", start_workers=False)
    ui_settings_payload["camera"].update({"enabled": True, "entity_id": "camera.nursery"})
    answer = "v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\n"
    app.state.go2rtc.negotiate = AsyncMock(return_value=answer)
    offer = "v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\n"

    with TestClient(app) as client:
        assert client.put("/api/v1/settings", json=ui_settings_payload).status_code == 200
        response = client.post(
            "/api/v1/camera/webrtc",
            content=offer,
            headers={"Content-Type": "application/sdp"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/sdp")
    assert response.text == answer
    app.state.go2rtc.negotiate.assert_awaited_once_with(offer)


def test_camera_webrtc_rejects_invalid_sdp_without_contacting_relay(tmp_path: Path, ui_settings_payload: dict) -> None:
    app = create_app(data_dir=tmp_path, runtime="test", start_workers=False)
    ui_settings_payload["camera"].update({"enabled": True, "entity_id": "camera.nursery"})
    app.state.go2rtc.negotiate = AsyncMock(side_effect=ValueError("invalid WebRTC SDP offer"))

    with TestClient(app) as client:
        assert client.put("/api/v1/settings", json=ui_settings_payload).status_code == 200
        response = client.post(
            "/api/v1/camera/webrtc",
            content="not-sdp",
            headers={"Content-Type": "application/sdp"},
        )

    assert response.status_code == 400
    assert response.json() == {"detail": "invalid WebRTC SDP offer"}


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


async def test_empty_crib_closes_sleep_across_an_unusable_frame(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path, runtime="test", start_workers=False)
    service = FrameService(app.state.database, app.state.settings, FakeHomeAssistant())  # type: ignore[arg-type]
    base = utc_now() - timedelta(minutes=30)
    asleep = VisionLabel(
        baby_present=True,
        state="asleep",
        confidence=0.95,
        description="Asleep in crib",
        tags=["crib"],
        in_crib=True,
    )
    first = app.state.database.add_frame(b"sleep-one", "image/jpeg", base, label=asleep)
    second = app.state.database.add_frame(b"sleep-two", "image/jpeg", base + timedelta(minutes=5), label=asleep)
    await service._reconcile_vision_sleep(second.id)
    event = app.state.database.open_sleep_event()
    assert event is not None and event.started_at == first.captured_at

    empty = VisionLabel(
        baby_present=False,
        state="uncertain",
        confidence=1,
        description="The crib is clearly empty",
        tags=["empty_crib"],
        in_crib=False,
    )
    unusable = VisionLabel(
        baby_present=False,
        state="uncertain",
        confidence=1,
        description="Camera static",
        tags=["corrupted", "no_visibility"],
        in_crib=False,
    )
    first_empty = app.state.database.add_frame(b"empty-one", "image/jpeg", base + timedelta(minutes=10), label=empty)
    app.state.database.add_frame(b"static", "image/jpeg", base + timedelta(minutes=15), label=unusable)
    second_empty = app.state.database.add_frame(b"empty-two", "image/jpeg", base + timedelta(minutes=20), label=empty)
    await service._reconcile_vision_sleep(second_empty.id)

    closed = app.state.database.get_sleep_event(event.id)
    assert closed is not None and closed.ended_at == first_empty.captured_at


async def test_baby_outside_crib_closes_sleep_even_when_asleep(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path, runtime="test", start_workers=False)
    service = FrameService(app.state.database, app.state.settings, FakeHomeAssistant())  # type: ignore[arg-type]
    base = utc_now() - timedelta(minutes=25)
    asleep_in_crib = VisionLabel(
        baby_present=True,
        state="asleep",
        confidence=0.95,
        description="Asleep in crib",
        tags=["crib"],
        in_crib=True,
    )
    first = app.state.database.add_frame(b"sleep-one", "image/jpeg", base, label=asleep_in_crib)
    second = app.state.database.add_frame(b"sleep-two", "image/jpeg", base + timedelta(minutes=5), label=asleep_in_crib)
    await service._reconcile_vision_sleep(second.id)
    event = app.state.database.open_sleep_event()
    assert event is not None and event.started_at == first.captured_at

    asleep_outside = VisionLabel(
        baby_present=True,
        state="asleep",
        confidence=0.95,
        description="Asleep outside the crib",
        tags=["outside_crib"],
        in_crib=False,
    )
    first_outside = app.state.database.add_frame(
        b"outside-one", "image/jpeg", base + timedelta(minutes=10), label=asleep_outside
    )
    second_outside = app.state.database.add_frame(
        b"outside-two", "image/jpeg", base + timedelta(minutes=15), label=asleep_outside
    )
    await service._reconcile_vision_sleep(second_outside.id)

    closed = app.state.database.get_sleep_event(event.id)
    assert closed is not None and closed.ended_at == first_outside.captured_at


async def test_uncertain_in_crib_frame_does_not_hide_sleep_start(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path, runtime="test", start_workers=False)
    service = FrameService(app.state.database, app.state.settings, FakeHomeAssistant())  # type: ignore[arg-type]
    base = utc_now() - timedelta(minutes=15)
    asleep = VisionLabel(
        baby_present=True,
        state="asleep",
        confidence=0.95,
        description="Asleep",
        tags=[],
        in_crib=True,
    )
    uncertain = VisionLabel(
        baby_present=True,
        state="uncertain",
        confidence=0.9,
        description="Face hidden",
        tags=[],
        in_crib=True,
    )
    first = app.state.database.add_frame(b"first", "image/jpeg", base, label=asleep)
    app.state.database.add_frame(b"uncertain", "image/jpeg", base + timedelta(minutes=5), label=uncertain)
    last = app.state.database.add_frame(b"last", "image/jpeg", base + timedelta(minutes=10), label=asleep)
    await service._reconcile_vision_sleep(last.id)

    event = app.state.database.open_sleep_event()
    assert event is not None and event.started_at == first.captured_at


async def test_stale_vision_sleep_is_split_when_sleep_resumes(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path, runtime="test", start_workers=False)
    service = FrameService(app.state.database, app.state.settings, FakeHomeAssistant())  # type: ignore[arg-type]
    base = utc_now() - timedelta(hours=6)
    asleep = VisionLabel(
        baby_present=True,
        state="asleep",
        confidence=0.95,
        description="Asleep",
        tags=[],
        in_crib=True,
    )
    awake = VisionLabel(
        baby_present=True,
        state="awake",
        confidence=0.95,
        description="Awake",
        tags=[],
        in_crib=True,
    )
    empty = VisionLabel(
        baby_present=False,
        state="uncertain",
        confidence=1,
        description="Empty crib",
        tags=["empty_crib"],
        in_crib=False,
    )
    uncertain = VisionLabel(
        baby_present=True,
        state="uncertain",
        confidence=0.9,
        description="Face hidden",
        tags=[],
        in_crib=True,
    )

    first = app.state.database.add_frame(b"first", "image/jpeg", base, label=asleep)
    second = app.state.database.add_frame(b"second", "image/jpeg", base + timedelta(minutes=5), label=asleep)
    await service._reconcile_vision_sleep(second.id)
    stale = app.state.database.open_sleep_event()
    assert stale is not None

    first_awake = app.state.database.add_frame(b"awake", "image/jpeg", base + timedelta(minutes=20), label=awake)
    app.state.database.add_frame(b"empty", "image/jpeg", base + timedelta(minutes=25), label=empty)
    resumed = app.state.database.add_frame(b"resumed", "image/jpeg", base + timedelta(hours=3), label=asleep)
    app.state.database.add_frame(b"hidden", "image/jpeg", base + timedelta(hours=3, minutes=5), label=uncertain)
    latest = app.state.database.add_frame(b"latest", "image/jpeg", base + timedelta(hours=3, minutes=10), label=asleep)

    # This single reconciliation simulates upgrading an installation whose old
    # detector left the morning event open across an awake gap.
    await service._reconcile_vision_sleep(latest.id)
    events, total = app.state.database.list_sleep_events(limit=10)
    ordered = sorted(events, key=lambda item: item.started_at)
    assert total == 2
    assert ordered[0].id == stale.id
    assert ordered[0].started_at == first.captured_at
    assert ordered[0].ended_at == first_awake.captured_at
    assert ordered[1].started_at == resumed.captured_at
    assert ordered[1].ended_at is None


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
