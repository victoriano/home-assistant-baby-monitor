from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

from baby_monitor.main import create_app
from baby_monitor.models import (
    BabyProfile,
    NotificationConfig,
    NotificationEvent,
    NotificationRecipient,
    SettingsPatch,
    SleepEventCreate,
    utc_now,
)
from baby_monitor.notifications import NotificationDispatcher, NotificationScheduler


class FakeHomeAssistant:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def call_service(self, domain: str, service: str, data: dict[str, Any]) -> list[Any]:
        self.calls.append((domain, service, data))
        return []


def recipient(
    name: str,
    service: str,
    *events: NotificationEvent,
    enabled: bool = True,
    language: str = "en",
) -> NotificationRecipient:
    return NotificationRecipient(
        person_entity_id=f"person.{name.lower()}",
        name=name,
        notify_service=f"notify.{service}",
        enabled=enabled,
        language=language,  # type: ignore[arg-type]
        events=list(events),
    )


async def test_dispatcher_respects_each_caregivers_subscriptions(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path, runtime="test", start_workers=False)
    app.state.settings.patch(
        SettingsPatch(
            baby=BabyProfile(name="Alex"),
            notifications=NotificationConfig(
                recipients=[
                    recipient("Ana", "mobile_app_ana", NotificationEvent.CRY_STARTED, language="es"),
                    recipient("Alex", "mobile_app_alex", NotificationEvent.SLEEP_STARTED),
                    recipient("Muted", "mobile_app_muted", NotificationEvent.CRY_STARTED, enabled=False),
                ]
            ),
        )
    )
    fake = FakeHomeAssistant()
    dispatcher = NotificationDispatcher(app.state.settings, fake)  # type: ignore[arg-type]

    result = await dispatcher.send(NotificationEvent.CRY_STARTED, {"event_id": "cry-1"})

    assert result.sent == ("person.ana",)
    assert [(domain, service) for domain, service, _ in fake.calls] == [("notify", "mobile_app_ana")]
    assert fake.calls[0][2]["title"] == "Alex está llorando"
    assert fake.calls[0][2]["data"]["event"] == "cry_started"


async def test_expected_sleep_end_alert_is_durable_and_not_duplicated(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path, runtime="test", start_workers=False)
    app.state.settings.patch(
        SettingsPatch(
            notifications=NotificationConfig(
                recipients=[
                    recipient("Parent", "mobile_app_parent", NotificationEvent.SLEEP_ENDING_SOON),
                ],
                lead_minutes=10,
            )
        )
    )
    now = utc_now().replace(microsecond=0)
    event = app.state.database.add_sleep_event(
        SleepEventCreate(
            started_at=now - timedelta(minutes=36),
            kind="nap",
            source="manual",
            location_id="home",
        )
    )
    fake = FakeHomeAssistant()
    dispatcher = NotificationDispatcher(app.state.settings, fake)  # type: ignore[arg-type]
    scheduler = NotificationScheduler(tmp_path, app.state.database, app.state.settings, dispatcher)

    await scheduler.poll(now)
    await scheduler.poll(now + timedelta(seconds=30))
    restarted = NotificationScheduler(tmp_path, app.state.database, app.state.settings, dispatcher)
    await restarted.poll(now + timedelta(minutes=1))

    calls = [call for call in fake.calls if call[2]["data"]["event"] == "sleep_ending_soon"]
    assert len(calls) == 1
    assert calls[0][2]["data"]["event_id"] == event.id
    assert "usual duration" in calls[0][2]["message"]
    assert (tmp_path / "notification-ledger.json").stat().st_mode & 0o077 == 0


async def test_sleep_start_and_end_are_independent_subscriptions(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path, runtime="test", start_workers=False)
    app.state.settings.patch(
        SettingsPatch(
            notifications=NotificationConfig(
                recipients=[
                    recipient("Start", "mobile_app_start", NotificationEvent.SLEEP_STARTED),
                    recipient("End", "mobile_app_end", NotificationEvent.SLEEP_ENDED),
                ]
            )
        )
    )
    now = utc_now().replace(microsecond=0)
    event = app.state.database.add_sleep_event(
        SleepEventCreate(
            started_at=now - timedelta(minutes=2),
            ended_at=now - timedelta(minutes=1),
            kind="nap",
            source="vision",
            location_id="home",
        )
    )
    fake = FakeHomeAssistant()
    scheduler = NotificationScheduler(
        tmp_path,
        app.state.database,
        app.state.settings,
        NotificationDispatcher(app.state.settings, fake),  # type: ignore[arg-type]
    )

    await scheduler.poll(now)

    assert [(call[1], call[2]["data"]["event"]) for call in fake.calls] == [
        ("mobile_app_start", "sleep_started"),
        ("mobile_app_end", "sleep_ended"),
    ]
    assert all(call[2]["data"]["event_id"] == event.id for call in fake.calls)


async def test_awake_intervals_do_not_emit_inverted_sleep_notifications(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path, runtime="test", start_workers=False)
    app.state.settings.patch(
        SettingsPatch(
            notifications=NotificationConfig(
                recipients=[
                    recipient(
                        "Parent",
                        "mobile_app_parent",
                        NotificationEvent.SLEEP_STARTED,
                        NotificationEvent.SLEEP_ENDED,
                    ),
                ]
            )
        )
    )
    now = utc_now().replace(microsecond=0)
    app.state.database.add_sleep_event(
        SleepEventCreate(
            started_at=now - timedelta(minutes=2),
            ended_at=now - timedelta(minutes=1),
            kind="awake",
            source="manual",
            location_id="home",
        )
    )
    fake = FakeHomeAssistant()
    scheduler = NotificationScheduler(
        tmp_path,
        app.state.database,
        app.state.settings,
        NotificationDispatcher(app.state.settings, fake),  # type: ignore[arg-type]
    )

    await scheduler.poll(now)

    assert fake.calls == []


async def test_predicted_sleep_alert_fires_inside_configured_lead_window(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    app = create_app(data_dir=tmp_path, runtime="test", start_workers=False)
    app.state.settings.patch(
        SettingsPatch(
            notifications=NotificationConfig(
                recipients=[
                    recipient("Parent", "mobile_app_parent", NotificationEvent.SLEEP_PREDICTED_SOON),
                ],
                lead_minutes=10,
            )
        )
    )
    now = utc_now().replace(microsecond=0)
    target = now + timedelta(minutes=9)
    monkeypatch.setattr(
        "baby_monitor.notifications.build_sleep_plan",
        lambda *args, **kwargs: {
            "averageNapMinutes": 50,
            "averageNightMinutes": 600,
            "plans": [
                {
                    "dayNapPredictions": [
                        {
                            "kind": "nap",
                            "recommendedStart": target.isoformat(),
                            "durationMinutes": 50,
                        }
                    ],
                    "nightPrediction": {
                        "kind": "night",
                        "recommendedStart": (now + timedelta(hours=8)).isoformat(),
                        "durationMinutes": 600,
                    },
                }
            ],
        },
    )
    fake = FakeHomeAssistant()
    scheduler = NotificationScheduler(
        tmp_path,
        app.state.database,
        app.state.settings,
        NotificationDispatcher(app.state.settings, fake),  # type: ignore[arg-type]
    )

    await scheduler.poll(now)

    assert len(fake.calls) == 1
    assert fake.calls[0][2]["data"]["event"] == "sleep_predicted_soon"
    assert "in 10 min" in fake.calls[0][2]["title"]


def test_legacy_notification_service_migrates_to_cry_only_recipient(tmp_path: Path) -> None:
    data = tmp_path / "settings.json"
    data.write_text(
        '{"notifications":{"service":"notify.mobile_app_parent","targets":["phone"]}}',
        encoding="utf-8",
    )
    app = create_app(data_dir=tmp_path, runtime="test", start_workers=False)

    settings = app.state.settings.get()

    assert len(settings.notifications.recipients) == 1
    migrated = settings.notifications.recipients[0]
    assert migrated.notify_service == "notify.mobile_app_parent"
    assert migrated.targets == ["phone"]
    assert migrated.events == [NotificationEvent.CRY_STARTED]
    assert "service" not in settings.notifications.model_dump(mode="json")
