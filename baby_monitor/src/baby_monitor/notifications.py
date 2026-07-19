from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .database import Database
from .home_assistant import HomeAssistantClient, HomeAssistantError
from .models import NotificationEvent, NotificationRecipient, SleepEvent, utc_now
from .prediction import build_sleep_plan
from .settings import SettingsService


@dataclass(frozen=True)
class DeliveryResult:
    attempted: tuple[str, ...]
    sent: tuple[str, ...]


def _recipient_key(recipient: NotificationRecipient) -> str:
    return recipient.person_entity_id or recipient.notify_service


def _clock(value: datetime, timezone_name: str) -> str:
    return value.astimezone(ZoneInfo(timezone_name)).strftime("%H:%M")


def _duration(minutes: int, language: str) -> str:
    hours, remainder = divmod(max(0, round(minutes)), 60)
    if language == "es":
        return f"{hours} h {remainder} min" if hours else f"{remainder} min"
    return f"{hours} h {remainder} min" if hours else f"{remainder} min"


class NotificationDispatcher:
    """Send one event only to caregivers who subscribed to that event."""

    def __init__(self, settings: SettingsService, home_assistant: HomeAssistantClient) -> None:
        self.settings = settings
        self.home_assistant = home_assistant

    def recipients(self, event: NotificationEvent) -> list[NotificationRecipient]:
        return [
            recipient
            for recipient in self.settings.get().notifications.recipients
            if recipient.enabled and event in recipient.events
        ]

    def _copy(
        self,
        recipient: NotificationRecipient,
        event: NotificationEvent,
        context: dict[str, Any],
    ) -> tuple[str, str]:
        settings = self.settings.get()
        baby = settings.baby.name
        language = recipient.language
        at = context.get("at")
        at_label = _clock(at, settings.baby.timezone) if isinstance(at, datetime) else ""
        lead = int(context.get("lead_minutes") or settings.notifications.lead_minutes)
        duration = _duration(int(context.get("duration_minutes") or 0), language)

        if language == "es":
            copies = {
                NotificationEvent.CRY_STARTED: (
                    f"{baby} está llorando",
                    "Se ha detectado llanto ahora.",
                ),
                NotificationEvent.SLEEP_STARTED: (
                    f"{baby} se ha dormido",
                    f"Sueño detectado a las {at_label}.",
                ),
                NotificationEvent.SLEEP_PREDICTED_SOON: (
                    f"Posible sueño en {lead} min",
                    f"El ritmo de {baby} apunta a las {at_label}.",
                ),
                NotificationEvent.SLEEP_ENDING_SOON: (
                    f"El sueño podría terminar en {lead} min",
                    f"La duración habitual usada para estimarlo es {duration}.",
                ),
                NotificationEvent.SLEEP_ENDED: (
                    f"{baby} se ha despertado",
                    f"El sueño terminó a las {at_label} y duró {duration}.",
                ),
                NotificationEvent.CAMERA_OFFLINE: (
                    "Revisa la cámara del bebé",
                    f"No hay capturas nuevas desde hace {duration}.",
                ),
            }
        else:
            copies = {
                NotificationEvent.CRY_STARTED: (
                    f"{baby} is crying",
                    "Crying was detected just now.",
                ),
                NotificationEvent.SLEEP_STARTED: (
                    f"{baby} fell asleep",
                    f"Sleep was detected at {at_label}.",
                ),
                NotificationEvent.SLEEP_PREDICTED_SOON: (
                    f"Possible sleep in {lead} min",
                    f"{baby}'s rhythm points to {at_label}.",
                ),
                NotificationEvent.SLEEP_ENDING_SOON: (
                    f"Sleep may end in {lead} min",
                    f"The usual duration used for this estimate is {duration}.",
                ),
                NotificationEvent.SLEEP_ENDED: (
                    f"{baby} woke up",
                    f"Sleep ended at {at_label} after {duration}.",
                ),
                NotificationEvent.CAMERA_OFFLINE: (
                    "Check the baby camera",
                    f"There have been no new captures for {duration}.",
                ),
            }
        return copies[event]

    async def send(
        self,
        event: NotificationEvent,
        context: dict[str, Any] | None = None,
        *,
        recipients: Iterable[NotificationRecipient] | None = None,
    ) -> DeliveryResult:
        context = context or {}
        selected = list(recipients) if recipients is not None else self.recipients(event)
        attempted: list[str] = []
        sent: list[str] = []
        for recipient in selected:
            key = _recipient_key(recipient)
            attempted.append(key)
            title, message = self._copy(recipient, event, context)
            payload: dict[str, Any] = {
                "title": title,
                "message": message,
                "data": {
                    "tag": f"baby-monitor-{event.value}",
                    "group": "baby-monitor",
                    "url": "/baby-monitor",
                    "event": event.value,
                },
            }
            event_id = context.get("event_id")
            if event_id:
                payload["data"]["event_id"] = str(event_id)
            if recipient.targets:
                payload["target"] = recipient.targets
            try:
                await self.home_assistant.call_service(
                    "notify",
                    recipient.notify_service.split(".", 1)[1],
                    payload,
                )
            except HomeAssistantError:
                continue
            sent.append(key)
        return DeliveryResult(tuple(attempted), tuple(sent))

    async def send_test(self, recipient: NotificationRecipient) -> None:
        copy = (
            (
                "Prueba de Baby Monitor",
                f"{recipient.name} recibirá aquí los avisos que tenga activados.",
            )
            if recipient.language == "es"
            else (
                "Baby Monitor test",
                f"{recipient.name} will receive enabled alerts here.",
            )
        )
        payload: dict[str, Any] = {
            "title": copy[0],
            "message": copy[1],
            "data": {"tag": "baby-monitor-test", "group": "baby-monitor", "url": "/baby-monitor"},
        }
        if recipient.targets:
            payload["target"] = recipient.targets
        await self.home_assistant.call_service(
            "notify",
            recipient.notify_service.split(".", 1)[1],
            payload,
        )


class NotificationLedger:
    """Tiny durable idempotency ledger kept outside the historical SQLite."""

    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir / "notification-ledger.json"
        self._lock = threading.RLock()

    def load(self) -> dict[str, str]:
        with self._lock:
            if not self.path.exists():
                return {}
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {}
            sent = raw.get("sent") if isinstance(raw, dict) else None
            return {str(key): str(value) for key, value in sent.items()} if isinstance(sent, dict) else {}

    def save(self, sent: dict[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = self.path.with_name(f".{self.path.name}.tmp-{os.getpid()}")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump({"sent": sent}, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            os.chmod(self.path, 0o600)
        finally:
            temporary.unlink(missing_ok=True)


class NotificationScheduler:
    """Translate predictions and sleep/camera state into deduplicated alerts."""

    RECENT_EVENT_WINDOW = timedelta(minutes=3)

    def __init__(
        self,
        data_dir: Path,
        database: Database,
        settings: SettingsService,
        dispatcher: NotificationDispatcher,
    ) -> None:
        self.database = database
        self.settings = settings
        self.dispatcher = dispatcher
        self.ledger = NotificationLedger(data_dir)
        self.started_at = utc_now()

    async def _deliver_once(
        self,
        sent: dict[str, str],
        key: str,
        event: NotificationEvent,
        context: dict[str, Any],
        now: datetime,
    ) -> None:
        eligible = self.dispatcher.recipients(event)
        pending = [recipient for recipient in eligible if f"{key}:{_recipient_key(recipient)}" not in sent]
        if not pending:
            return
        result = await self.dispatcher.send(event, context, recipients=pending)
        for recipient_key in result.sent:
            sent[f"{key}:{recipient_key}"] = now.astimezone(UTC).isoformat()

    @staticmethod
    def _sleep_duration(event: SleepEvent, ended_at: datetime) -> int:
        return max(0, round((ended_at - event.started_at).total_seconds() / 60))

    async def poll(self, now: datetime | None = None) -> None:
        now = now or utc_now()
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        now = now.astimezone(UTC)
        sent = self.ledger.load()
        settings = self.settings.get()
        history, _ = self.database.list_sleep_events(limit=2_000)
        plan = build_sleep_plan(
            history,
            birth_date=settings.baby.birth_date,
            timezone_name=settings.baby.timezone,
            now=now,
        )

        recent_cutoff = now - self.RECENT_EVENT_WINDOW
        for sleep in history[:20]:
            # Awake intervals describe gaps inside sleep history. The sleep
            # event that surrounds them is responsible for start/end alerts;
            # treating the gap itself as sleep would send inverted duplicates.
            if sleep.kind == "awake":
                continue
            started = sleep.started_at.astimezone(UTC)
            if recent_cutoff <= started <= now:
                await self._deliver_once(
                    sent,
                    f"sleep-started:{sleep.id}",
                    NotificationEvent.SLEEP_STARTED,
                    {"event_id": sleep.id, "at": sleep.started_at},
                    now,
                )
            if sleep.ended_at is not None:
                ended = sleep.ended_at.astimezone(UTC)
                if recent_cutoff <= ended <= now:
                    await self._deliver_once(
                        sent,
                        f"sleep-ended:{sleep.id}",
                        NotificationEvent.SLEEP_ENDED,
                        {
                            "event_id": sleep.id,
                            "at": sleep.ended_at,
                            "duration_minutes": self._sleep_duration(sleep, sleep.ended_at),
                        },
                        now,
                    )

        lead = settings.notifications.lead_minutes
        lead_delta = timedelta(minutes=lead)
        for day in plan["plans"]:
            for target in [*day["dayNapPredictions"], day["nightPrediction"]]:
                target_at = datetime.fromisoformat(target["recommendedStart"]).astimezone(UTC)
                if now < target_at <= now + lead_delta:
                    await self._deliver_once(
                        sent,
                        f"sleep-predicted:{target['kind']}:{target['recommendedStart']}",
                        NotificationEvent.SLEEP_PREDICTED_SOON,
                        {
                            "at": target_at,
                            "lead_minutes": lead,
                            "duration_minutes": target["durationMinutes"],
                        },
                        now,
                    )

        active_sleep = self.database.open_sleep_event()
        if active_sleep is not None:
            expected_minutes = (
                int(plan["averageNightMinutes"]) if active_sleep.kind == "night" else int(plan["averageNapMinutes"])
            )
            expected_end = active_sleep.started_at + timedelta(minutes=expected_minutes)
            if now < expected_end.astimezone(UTC) <= now + lead_delta:
                await self._deliver_once(
                    sent,
                    f"sleep-ending:{active_sleep.id}",
                    NotificationEvent.SLEEP_ENDING_SOON,
                    {
                        "event_id": active_sleep.id,
                        "at": expected_end,
                        "lead_minutes": lead,
                        "duration_minutes": expected_minutes,
                    },
                    now,
                )

        if settings.camera.enabled:
            latest = self.database.latest_frame()
            threshold_minutes = max(15, round(settings.camera.capture_interval_seconds * 3 / 60))
            stale_minutes = (
                round((now - latest.captured_at.astimezone(UTC)).total_seconds() / 60)
                if latest is not None
                else round((now - self.started_at.astimezone(UTC)).total_seconds() / 60)
            )
            if stale_minutes >= threshold_minutes:
                await self._deliver_once(
                    sent,
                    f"camera-offline:{latest.id if latest else settings.baby.location_id}",
                    NotificationEvent.CAMERA_OFFLINE,
                    {"duration_minutes": stale_minutes},
                    now,
                )

        prune_before = now - timedelta(days=30)
        recent_sent: dict[str, str] = {}
        for key, value in sent.items():
            try:
                delivered_at = datetime.fromisoformat(value).astimezone(UTC)
            except (TypeError, ValueError):
                continue
            if delivered_at >= prune_before:
                recent_sent[key] = value
        sent = recent_sent
        self.ledger.save(sent)
