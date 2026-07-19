from __future__ import annotations

import asyncio
import time
from datetime import timedelta
from typing import Any

from .database import Database
from .home_assistant import HomeAssistantClient, HomeAssistantError
from .media import AudioWindowReader, MediaError, analyze_pcm
from .models import CryMode, RetentionMode, SecretName, utc_now
from .notifications import NotificationScheduler
from .services import CryAlertService, FrameService
from .settings import SettingsService


class RuntimeWorkers:
    def __init__(
        self,
        database: Database,
        settings: SettingsService,
        home_assistant: HomeAssistantClient,
        frames: FrameService,
        cry_alerts: CryAlertService,
        notification_scheduler: NotificationScheduler,
    ) -> None:
        self.database = database
        self.settings = settings
        self.home_assistant = home_assistant
        self.frames = frames
        self.cry_alerts = cry_alerts
        self.notification_scheduler = notification_scheduler
        self._tasks: list[asyncio.Task[None]] = []
        self.errors: dict[str, str] = {}

    def start(self) -> None:
        if self._tasks:
            return
        self._tasks = [
            asyncio.create_task(self._retention_loop(), name="baby-monitor-retention"),
            asyncio.create_task(self._capture_loop(), name="baby-monitor-capture"),
            asyncio.create_task(self._cry_loop(), name="baby-monitor-cry"),
            asyncio.create_task(self._notification_loop(), name="baby-monitor-notifications"),
        ]

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        await self.cry_alerts.close()

    def status(self) -> dict[str, Any]:
        return {
            "running": bool(self._tasks) and all(not task.done() for task in self._tasks),
            "workers": {task.get_name(): not task.done() for task in self._tasks},
            "errors": dict(self.errors),
        }

    async def _retention_loop(self) -> None:
        while True:
            try:
                policy = self.settings.get().retention
                if policy.mode == RetentionMode.DAYS and policy.days is not None:
                    self.database.purge_frames_before(utc_now() - timedelta(days=policy.days))
                self.errors.pop("retention", None)
            except Exception as exc:  # worker must remain resilient
                self.errors["retention"] = type(exc).__name__
            await asyncio.sleep(3600)

    async def _capture_loop(self) -> None:
        while True:
            try:
                config = self.settings.get()
                if config.camera.enabled:
                    latest = self.database.latest_frame()
                    due = (
                        latest is None
                        or (utc_now() - latest.captured_at).total_seconds() >= config.camera.capture_interval_seconds
                    )
                    if due:
                        await self.frames.capture(label=True)
                self.errors.pop("capture", None)
            except Exception as exc:
                self.errors["capture"] = type(exc).__name__
            await asyncio.sleep(10)

    async def _notification_loop(self) -> None:
        while True:
            try:
                await self.notification_scheduler.poll()
                self.errors.pop("notifications", None)
            except Exception as exc:  # notifications must never stop monitoring
                self.errors["notifications"] = type(exc).__name__
            await asyncio.sleep(30)

    async def _cry_loop(self) -> None:
        previous_sensor_state: bool | None = None
        reader: AudioWindowReader | None = None
        reader_key: tuple[str, float] | None = None
        positive_streak = 0
        active = False
        last_positive = time.monotonic()
        try:
            if self.cry_alerts.active:
                await self.cry_alerts.set_state("off", observed_at=utc_now(), source="worker_restart")
            while True:
                config = self.settings.get().cry
                if config.mode == CryMode.DISABLED:
                    if self.cry_alerts.active:
                        await self.cry_alerts.set_state("off", observed_at=utc_now(), source="disabled")
                    active = False
                    previous_sensor_state = None
                    if reader:
                        await reader.close()
                        reader = None
                        reader_key = None
                    await asyncio.sleep(2)
                    continue

                if config.mode == CryMode.BINARY_SENSOR:
                    if reader:
                        await reader.close()
                        reader = None
                        reader_key = None
                        if self.cry_alerts.active:
                            await self.cry_alerts.set_state("off", observed_at=utc_now(), source="cry_source_changed")
                    try:
                        state = await self.home_assistant.get_state(config.entity_id or "")
                        is_on = str(state.get("state", "off")).lower() in {"on", "true", "detected"}
                        if previous_sensor_state is None or is_on != previous_sensor_state:
                            await self.cry_alerts.set_state(
                                "on" if is_on else "off",
                                observed_at=utc_now(),
                                source="binary_sensor",
                                metadata={"entity_id": config.entity_id},
                            )
                        previous_sensor_state = is_on
                        self.errors.pop("cry", None)
                    except HomeAssistantError as exc:
                        self.errors["cry"] = type(exc).__name__
                        if self.cry_alerts.active:
                            await self.cry_alerts.set_state(
                                "off", observed_at=utc_now(), source="binary_sensor_unavailable"
                            )
                        previous_sensor_state = None
                    await asyncio.sleep(2)
                    continue

                stream_url = self.settings.get_secret(SecretName.CRY_AUDIO_STREAM_URL)
                if not stream_url:
                    self.errors["cry"] = "MissingAudioStream"
                    await asyncio.sleep(3)
                    continue
                key = (stream_url, config.window_seconds)
                if reader is None or key != reader_key:
                    if reader:
                        await reader.close()
                    if self.cry_alerts.active:
                        await self.cry_alerts.set_state("off", observed_at=utc_now(), source="audio_source_changed")
                    reader = AudioWindowReader(stream_url, config.window_seconds)
                    reader_key = key
                    positive_streak = 0
                    active = False
                try:
                    raw = await reader.read()
                    metrics = await asyncio.to_thread(analyze_pcm, raw, config.sensitivity)
                    now = time.monotonic()
                    if metrics.positive:
                        positive_streak += 1
                        last_positive = now
                    else:
                        positive_streak = 0
                    if not active and positive_streak >= config.positive_windows:
                        active = True
                        await self.cry_alerts.set_state(
                            "on",
                            observed_at=utc_now(),
                            source="audio",
                            metadata=metrics.as_dict(),
                        )
                    elif active and not metrics.positive and now - last_positive >= config.clear_after_seconds:
                        active = False
                        await self.cry_alerts.set_state(
                            "off",
                            observed_at=utc_now(),
                            source="audio",
                            metadata=metrics.as_dict(),
                        )
                    self.errors.pop("cry", None)
                except (MediaError, OSError) as exc:
                    self.errors["cry"] = type(exc).__name__
                    if self.cry_alerts.active:
                        await self.cry_alerts.set_state("off", observed_at=utc_now(), source="audio_unavailable")
                    active = False
                    await reader.close()
                    reader = None
                    reader_key = None
                    await asyncio.sleep(3)
        finally:
            if reader:
                await reader.close()
