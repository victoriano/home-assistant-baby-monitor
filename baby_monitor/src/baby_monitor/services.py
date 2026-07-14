from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import UTC, datetime, time, timedelta
from statistics import median
from typing import Any, Literal
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
    VisionLabel,
    utc_now,
)
from .prediction import build_sleep_plan
from .providers import ProviderError, build_provider
from .settings import SettingsService


class ServiceError(RuntimeError):
    pass


VisionSleepState = Literal["asleep", "awake"]

# These labels mean that the camera could not provide evidence either way. A
# clearly empty crib is wake evidence, but a broken/covered frame must never end
# a sleep event just because no baby could be seen.
_UNUSABLE_VISION_TAGS = {
    "black_frame",
    "blocked",
    "camera_blocked",
    "corrupted",
    "image_unusable",
    "no_visibility",
    "obscured",
    "static",
}


def _vision_sleep_state(label: VisionLabel | None) -> VisionSleepState | None:
    """Normalize a model label into the evidence used by the sleep tracker."""

    if label is None or label.confidence < 0.65:
        return None
    normalized_tags = {tag.strip().lower().replace("-", "_").replace(" ", "_") for tag in label.tags}
    if normalized_tags & _UNUSABLE_VISION_TAGS:
        return None
    # The public providers correctly use `uncertain` when there is no baby to
    # classify. A clearly empty crib or a baby outside it still confirms that a
    # crib sleep has ended, matching the original Esteban tracker.
    if label.in_crib is False:
        return "awake"
    if label.baby_present and label.state in {"asleep", "awake"}:
        return label.state
    return None


def _confirmed_vision_transitions(
    labels: list[tuple[datetime, VisionLabel]],
    max_gap: timedelta,
) -> list[tuple[VisionSleepState, datetime]]:
    """Return state changes backed by two nearby decisive observations.

    Inconclusive frames neither confirm nor contradict a transition. This is
    important for overhead crib cameras where a face can be temporarily hidden
    while the preceding and following observations still agree.
    """

    previous: tuple[datetime, VisionSleepState] | None = None
    confirmed: VisionSleepState | None = None
    transitions: list[tuple[VisionSleepState, datetime]] = []
    for captured_at, label in labels:
        state = _vision_sleep_state(label)
        if state is None:
            if previous is not None and captured_at - previous[0] > max_gap:
                previous = None
            continue
        if (
            previous is not None
            and previous[1] == state
            and captured_at - previous[0] <= max_gap
            and confirmed != state
        ):
            transitions.append((state, previous[0]))
            confirmed = state
        previous = (captured_at, state)
    return transitions


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
                location_id=config.baby.location_id,
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
        """Debounce vision evidence and reconcile the current automatic event."""

        async with self._sleep_lock:
            newest = self.database.latest_frame()
            if newest is None or newest.id != frame_id or newest.label is None:
                return
            settings = self.settings.get()
            max_gap = timedelta(seconds=min(1800, max(300, settings.camera.capture_interval_seconds * 3)))
            open_event = self.database.open_sleep_event()
            if open_event is not None and open_event.source != "vision":
                return

            # If an older build left a vision event open, replay only that open
            # interval. This repairs its missing wake transition without
            # rewriting closed or manually corrected history.
            lookback = open_event.started_at if open_event is not None else newest.captured_at - max_gap
            labels = self.database.vision_labels_between(lookback, newest.captured_at)
            transitions = _confirmed_vision_transitions(labels, max_gap)
            if not transitions:
                return

            if open_event is None:
                state, started_at = transitions[-1]
                if state != "asleep":
                    return
                try:
                    self.database.add_sleep_event(
                        SleepEventCreate(
                            started_at=started_at,
                            kind=self._sleep_kind(started_at),
                            source="vision",
                            notes="Started after two consecutive image labels.",
                            location_id=settings.baby.location_id,
                        )
                    )
                except StorageError:
                    return
                return

            current: SleepEvent | None = open_event
            for state, observed_at in transitions:
                if state == "awake" and current is not None:
                    ended_at = max(observed_at, current.started_at + timedelta(microseconds=1))
                    try:
                        self.database.update_sleep_event(current.id, SleepEventPatch(ended_at=ended_at))
                    except StorageError:
                        return
                    current = None
                elif state == "asleep" and current is None:
                    try:
                        current = self.database.add_sleep_event(
                            SleepEventCreate(
                                started_at=observed_at,
                                kind=self._sleep_kind(observed_at),
                                source="vision",
                                notes="Started after two consecutive image labels.",
                                location_id=settings.baby.location_id,
                            )
                        )
                    except StorageError:
                        return

    def _sleep_kind(self, started_at: datetime) -> Literal["nap", "night"]:
        local_start = started_at.astimezone(ZoneInfo(self.settings.get().baby.timezone))
        return "night" if local_start.hour >= 19 or local_start.hour < 7 else "nap"


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

    @staticmethod
    def _rgb_to_xy(color: tuple[int, int, int]) -> list[float]:
        def linear(value: int) -> float:
            channel = max(0, min(255, value)) / 255
            return ((channel + 0.055) / 1.055) ** 2.4 if channel > 0.04045 else channel / 12.92

        red, green, blue = (linear(value) for value in color)
        x_value = red * 0.664511 + green * 0.154324 + blue * 0.162028
        y_value = red * 0.283881 + green * 0.668433 + blue * 0.047685
        z_value = red * 0.000088 + green * 0.072310 + blue * 0.986039
        total = x_value + y_value + z_value
        if total == 0:
            return [0.3127, 0.329]
        return [round(x_value / total, 4), round(y_value / total, 4)]

    @staticmethod
    def _rgb_to_hs(color: tuple[int, int, int]) -> list[float]:
        import colorsys

        hue, saturation, _ = colorsys.rgb_to_hsv(*(max(0, min(255, value)) / 255 for value in color))
        return [round(hue * 360, 2), round(saturation * 100, 2)]

    def _alert_light_data(
        self,
        entity_id: str,
        snapshot: dict[str, Any],
        brightness_percent: int,
        color_rgb: tuple[int, int, int],
    ) -> dict[str, Any]:
        attributes = snapshot.get("attributes") if isinstance(snapshot.get("attributes"), dict) else {}
        modes = set(attributes.get("supported_color_modes") or [])
        data: dict[str, Any] = {"entity_id": entity_id, "brightness_pct": brightness_percent}
        if "xy" in modes:
            data["xy_color"] = self._rgb_to_xy(color_rgb)
        elif modes & {"rgb", "rgbw", "rgbww"} or not modes:
            data["rgb_color"] = list(color_rgb)
        elif "hs" in modes:
            data["hs_color"] = self._rgb_to_hs(color_rgb)
        return data

    @staticmethod
    def _restore_color_data(attributes: dict[str, Any]) -> dict[str, Any]:
        color_mode = attributes.get("color_mode")
        preferred = {
            "xy": "xy_color",
            "hs": "hs_color",
            "rgb": "rgb_color",
            "rgbw": "rgbw_color",
            "rgbww": "rgbww_color",
            "color_temp": "color_temp_kelvin",
        }.get(color_mode)
        keys = [preferred] if preferred else []
        keys.extend(
            key
            for key in (
                "xy_color",
                "hs_color",
                "rgb_color",
                "rgbw_color",
                "rgbww_color",
                "color_temp_kelvin",
                "color_temp",
            )
            if key != preferred
        )
        for key in keys:
            if key and attributes.get(key) is not None:
                return {key: attributes[key]}
        return {}

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
                    location_id=self.settings.get().baby.location_id,
                )
            )
            # A slow light integration must not delay the caregiver alert (and
            # vice versa). Both side effects start as soon as the event exists.
            await asyncio.gather(
                self._activate_lights_locked(),
                self._send_notifications_locked(event),
            )
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
        # Send per-light colour data because Hue and other integrations expose
        # different service colour modes (notably XY-only strips).
        for entity_id in config.entity_ids:
            with suppress(HomeAssistantError):
                await self.home_assistant.call_service(
                    "light",
                    "turn_on",
                    self._alert_light_data(
                        entity_id,
                        self._fallback_states.get(entity_id, {}),
                        config.brightness_percent,
                        config.color_rgb,
                    ),
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
                data.update(self._restore_color_data(attributes))
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
            if event.kind == "awake":
                continue
            start = max(event.started_at.astimezone(UTC), day_start)
            end = min((event.ended_at or utc_now()).astimezone(UTC), day_end)
            if end > start:
                sleep_minutes += (end - start).total_seconds() / 60
                for pause in event.details.pauses:
                    pause_start = max(pause.started_at.astimezone(UTC), start)
                    pause_end = min(pause.ended_at.astimezone(UTC), end)
                    if pause_end > pause_start:
                        sleep_minutes -= (pause_end - pause_start).total_seconds() / 60

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

    def predictions(self) -> dict[str, Any]:
        settings = self.settings.get()
        history, _ = self.database.list_sleep_events(limit=2_000)
        return build_sleep_plan(
            history,
            birth_date=settings.baby.birth_date,
            timezone_name=settings.baby.timezone,
        )

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
            (event for event in history if event.kind != "awake" and event.ended_at is not None),
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
