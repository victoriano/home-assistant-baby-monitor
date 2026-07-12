from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import UTC, datetime, time, timedelta
from statistics import median
from typing import Any
from zoneinfo import ZoneInfo

from .database import Database, StorageError
from .home_assistant import HomeAssistantClient, HomeAssistantError
from .media import MediaError, snapshot_from_stream
from .models import (
    AIProviderName,
    CryEvent,
    CryEventCreate,
    FrameRecord,
    SecretName,
    SleepEvent,
    SleepEventCreate,
    SleepEventPatch,
    utc_now,
)
from .providers import ProviderError, build_provider
from .settings import SettingsService


class ServiceError(RuntimeError):
    pass


class FrameService:
    def __init__(
        self,
        database: Database,
        settings: SettingsService,
        home_assistant: HomeAssistantClient,
    ) -> None:
        self.database = database
        self.settings = settings
        self.home_assistant = home_assistant
        self._capture_lock = asyncio.Lock()
        self._sleep_lock = asyncio.Lock()

    async def capture(self, *, label: bool = True) -> FrameRecord:
        async with self._capture_lock:
            config = self.settings.get()
            if not config.camera.enabled:
                raise ServiceError("camera is disabled")
            try:
                if config.camera.entity_id:
                    image, mime_type = await self.home_assistant.camera_snapshot(config.camera.entity_id)
                else:
                    stream_url = self.settings.get_secret(SecretName.CAMERA_STREAM_URL)
                    if not stream_url:
                        raise ServiceError("camera stream URL is not configured")
                    image, mime_type = await snapshot_from_stream(stream_url)
            except (HomeAssistantError, MediaError) as exc:
                raise ServiceError(str(exc)) from exc

            frame = self.database.add_frame(
                image=image,
                mime_type=mime_type,
                captured_at=utc_now(),
                camera_entity_id=config.camera.entity_id,
            )
            if label and config.ai.provider != AIProviderName.DISABLED:
                # A private frame remains useful even when an optional AI
                # service is temporarily unavailable.
                with suppress(ProviderError, ServiceError):
                    frame = await self.label(frame.id)
            return frame

    async def label(self, frame_id: str) -> FrameRecord:
        frame = self.database.get_frame(frame_id)
        path = self.database.get_frame_path(frame_id)
        if frame is None or path is None:
            raise ServiceError("frame image is unavailable")
        config = self.settings.get().ai
        if config.provider == AIProviderName.DISABLED:
            raise ServiceError("image labeling is disabled")
        if not config.cloud_image_consent:
            raise ServiceError("image-sharing consent is required for the configured AI endpoint")
        provider = build_provider(
            config,
            self.settings.get_secret(SecretName.AI_API_KEY),
        )
        image = await asyncio.to_thread(path.read_bytes)
        label = await provider.label(image, frame.mime_type, config.detail)
        updated = self.database.set_frame_label(frame_id, label, provider.name, provider.model)
        if updated is None:
            raise ServiceError("frame no longer exists")
        await self._reconcile_vision_sleep(frame_id)
        return updated

    async def _reconcile_vision_sleep(self, frame_id: str) -> None:
        """Debounce two recent image labels into an automatic sleep event."""

        async with self._sleep_lock:
            recent, _ = self.database.list_frames(limit=2)
            if len(recent) < 2 or recent[0].id != frame_id:
                return
            newest, previous = recent
            labels = (newest.label, previous.label)
            if any(
                label is None or not label.baby_present or label.state == "uncertain" or label.confidence < 0.65
                for label in labels
            ):
                return
            assert newest.label is not None and previous.label is not None
            if newest.label.state != previous.label.state:
                return
            settings = self.settings.get()
            max_gap = min(1800, max(300, settings.camera.capture_interval_seconds * 3))
            if (newest.captured_at - previous.captured_at).total_seconds() > max_gap:
                return

            open_event = self.database.open_sleep_event()
            if newest.label.state == "asleep":
                if open_event is not None:
                    return
                local_start = previous.captured_at.astimezone(ZoneInfo(settings.baby.timezone))
                kind = "night" if local_start.hour >= 19 or local_start.hour < 7 else "nap"
                try:
                    self.database.add_sleep_event(
                        SleepEventCreate(
                            started_at=previous.captured_at,
                            kind=kind,
                            source="vision",
                            notes="Started after two consecutive image labels.",
                        )
                    )
                except StorageError:
                    return
                return

            if open_event is not None and open_event.source == "vision":
                ended_at = max(previous.captured_at, open_event.started_at + timedelta(microseconds=1))
                with suppress(StorageError):
                    self.database.update_sleep_event(open_event.id, SleepEventPatch(ended_at=ended_at))


class CryAlertService:
    """Persists cry events and coordinates reversible Home Assistant alerts."""

    SCENE_ENTITY_ID = "scene.baby_monitor_cry_restore"

    def __init__(
        self,
        database: Database,
        settings: SettingsService,
        home_assistant: HomeAssistantClient,
    ) -> None:
        self.database = database
        self.settings = settings
        self.home_assistant = home_assistant
        self._lock = asyncio.Lock()
        self._restore_task: asyncio.Task[None] | None = None
        self._fallback_states: dict[str, dict[str, Any]] = {}
        self._scene_created = False

    @property
    def active(self) -> bool:
        return self.database.open_cry_event() is not None

    async def set_state(
        self,
        state: str,
        *,
        observed_at: datetime,
        source: str,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CryEvent | None:
        async with self._lock:
            if state == "off":
                event = self.database.open_cry_event()
                if event is None:
                    return None
                closed = self.database.close_cry_event(event.id, observed_at)
                await self._restore_lights_locked()
                return closed

            current = self.database.open_cry_event()
            if current is not None:
                self._schedule_restore(self.settings.get().lights.duration_seconds)
                return current

            event = self.database.add_cry_event(
                CryEventCreate(
                    detected_at=observed_at,
                    source=source,
                    confidence=confidence,
                    metadata=metadata or {},
                )
            )
            await self._activate_lights_locked()
            await self._send_notifications_locked(event)
            return event

    async def _activate_lights_locked(self) -> None:
        config = self.settings.get().lights
        if not config.entity_ids:
            return
        self._fallback_states = {}
        for entity_id in config.entity_ids:
            try:
                self._fallback_states[entity_id] = await self.home_assistant.get_state(entity_id)
            except HomeAssistantError:
                continue
        try:
            await self.home_assistant.call_service(
                "scene",
                "create",
                {
                    "scene_id": self.SCENE_ENTITY_ID.split(".", 1)[1],
                    "snapshot_entities": config.entity_ids,
                },
            )
            self._scene_created = True
        except HomeAssistantError:
            self._scene_created = False
        # The event remains valid if a selected light disappeared.
        with suppress(HomeAssistantError):
            await self.home_assistant.call_service(
                "light",
                "turn_on",
                {
                    "entity_id": config.entity_ids,
                    "brightness_pct": config.brightness_percent,
                    "rgb_color": list(config.color_rgb),
                },
            )
        self._schedule_restore(config.duration_seconds)

    def _schedule_restore(self, delay: int) -> None:
        if self._restore_task is not None:
            self._restore_task.cancel()
        self._restore_task = asyncio.create_task(self._restore_after(delay))

    async def _restore_after(self, delay: int) -> None:
        try:
            await asyncio.sleep(delay)
            async with self._lock:
                await self._restore_lights_locked(cancel_timer=False)
        except asyncio.CancelledError:
            raise

    async def _restore_lights_locked(self, *, cancel_timer: bool = True) -> None:
        current_task = asyncio.current_task()
        if cancel_timer and self._restore_task is not None and self._restore_task is not current_task:
            self._restore_task.cancel()
        self._restore_task = None
        if self._scene_created:
            try:
                await self.home_assistant.call_service("scene", "turn_on", {"entity_id": self.SCENE_ENTITY_ID})
                with suppress(HomeAssistantError):
                    await self.home_assistant.call_service("scene", "delete", {"entity_id": self.SCENE_ENTITY_ID})
                self._scene_created = False
                self._fallback_states = {}
                return
            except HomeAssistantError:
                pass
        for entity_id, snapshot in self._fallback_states.items():
            try:
                if snapshot.get("state") != "on":
                    await self.home_assistant.call_service("light", "turn_off", {"entity_id": entity_id})
                    continue
                attributes = snapshot.get("attributes") if isinstance(snapshot.get("attributes"), dict) else {}
                data: dict[str, Any] = {"entity_id": entity_id}
                if isinstance(attributes.get("brightness"), int):
                    data["brightness"] = attributes["brightness"]
                # Pick one supported colour representation, avoiding conflicting
                # HA service fields in the same restore call.
                for key in ("rgb_color", "hs_color", "xy_color", "color_temp_kelvin", "color_temp"):
                    if attributes.get(key) is not None:
                        data[key] = attributes[key]
                        break
                await self.home_assistant.call_service("light", "turn_on", data)
            except HomeAssistantError:
                continue
        self._fallback_states = {}
        if self._scene_created:
            with suppress(HomeAssistantError):
                await self.home_assistant.call_service("scene", "delete", {"entity_id": self.SCENE_ENTITY_ID})
            self._scene_created = False

    async def _send_notifications_locked(self, event: CryEvent) -> None:
        settings = self.settings.get()
        if not settings.notifications.service:
            return
        service = settings.notifications.service.split(".", 1)[1]
        payload: dict[str, Any] = {
            "title": f"{settings.baby.name}: cry detected",
            "message": "The baby monitor detected crying.",
            "data": {"tag": "baby-monitor-cry", "event_id": event.id},
        }
        if settings.notifications.targets:
            payload["target"] = settings.notifications.targets
        with suppress(HomeAssistantError):
            await self.home_assistant.call_service("notify", service, payload)

    async def close(self) -> None:
        async with self._lock:
            event = self.database.open_cry_event()
            if event is not None:
                self.database.close_cry_event(event.id, utc_now())
            await self._restore_lights_locked()


class DashboardService:
    def __init__(self, database: Database, settings: SettingsService) -> None:
        self.database = database
        self.settings = settings

    def summary(self) -> dict[str, Any]:
        settings = self.settings.get()
        latest_sleep = self.database.latest_sleep_event()
        open_sleep = self.database.open_sleep_event()
        if open_sleep:
            state = "sleeping"
            state_since = open_sleep.started_at
        elif latest_sleep:
            state = "awake"
            state_since = latest_sleep.ended_at or latest_sleep.started_at
        else:
            state = "unknown"
            state_since = None

        timezone_info = ZoneInfo(settings.baby.timezone)
        now_local = datetime.now(timezone_info)
        day_start_local = datetime.combine(now_local.date(), time.min, tzinfo=timezone_info)
        day_end_local = day_start_local + timedelta(days=1)
        day_start = day_start_local.astimezone(UTC)
        day_end = day_end_local.astimezone(UTC)
        sleep_minutes = 0.0
        for event in self.database.sleep_events_overlapping(day_start, day_end):
            start = max(event.started_at.astimezone(UTC), day_start)
            end = min((event.ended_at or utc_now()).astimezone(UTC), day_end)
            if end > start:
                sleep_minutes += (end - start).total_seconds() / 60

        prediction_history, _ = self.database.list_sleep_events(limit=30)
        next_sleep_at, prediction_confidence, prediction_reason = self._prediction(
            settings.baby.birth_date,
            latest_sleep,
            open_sleep,
            prediction_history,
        )
        latest_cry = self.database.latest_cry_event()
        frames, _ = self.database.list_frames(limit=1)
        recent_sleep, _ = self.database.list_sleep_events(limit=5)
        recent_cry, _ = self.database.list_cry_events(limit=5)
        return {
            "state": state,
            "state_since": state_since,
            "current_sleep": open_sleep,
            "next_sleep_at": next_sleep_at,
            "prediction_confidence": prediction_confidence,
            "prediction_reason": prediction_reason,
            "sleep_today_minutes": round(sleep_minutes),
            "last_cry_at": latest_cry.detected_at if latest_cry else None,
            "cry_active": latest_cry is not None and latest_cry.ended_at is None,
            "latest_frame": frames[0] if frames else None,
            "recent_sleep": recent_sleep,
            "recent_cry": recent_cry,
        }

    @staticmethod
    def _prediction(
        birth_date: Any,
        latest: SleepEvent | None,
        open_sleep: SleepEvent | None,
        history: list[SleepEvent],
    ) -> tuple[datetime | None, float | None, str | None]:
        if open_sleep or latest is None or latest.ended_at is None:
            return None, None, None
        if birth_date is None:
            baseline_minutes = 180
        else:
            age_days = max(0, (datetime.now(UTC).date() - birth_date).days)
            if age_days < 90:
                baseline_minutes = 75
            elif age_days < 180:
                baseline_minutes = 120
            elif age_days < 270:
                baseline_minutes = 165
            elif age_days < 365:
                baseline_minutes = 210
            elif age_days < 548:
                baseline_minutes = 270
            elif age_days < 730:
                baseline_minutes = 300
            else:
                baseline_minutes = 360

        closed = sorted(
            (event for event in history if event.ended_at is not None),
            key=lambda event: event.started_at,
        )
        wake_intervals: list[float] = []
        for previous, following in zip(closed, closed[1:], strict=False):
            assert previous.ended_at is not None
            gap_minutes = (following.started_at - previous.ended_at).total_seconds() / 60
            if 15 <= gap_minutes <= 12 * 60:
                wake_intervals.append(gap_minutes)

        if len(wake_intervals) >= 3:
            sample = wake_intervals[-12:]
            learned_minutes = float(median(sample))
            history_weight = min(0.65, 0.25 + len(sample) * 0.04)
            wake_minutes = round(baseline_minutes * (1 - history_weight) + learned_minutes * history_weight)
            confidence = min(0.9, 0.55 + len(sample) * 0.025)
            reason = "Blended age guidance with recent wake intervals"
        else:
            wake_minutes = baseline_minutes
            confidence = 0.5 if birth_date is None else 0.6
            reason = "Age-based wake window while more history is collected"
        return latest.ended_at + timedelta(minutes=wake_minutes), confidence, reason
