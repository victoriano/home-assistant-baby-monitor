from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .models import (
    AIProviderName,
    AppSettings,
    CryMode,
    HomeAssistantMode,
    SecretChanges,
    SecretName,
    SettingsPatch,
    SettingsWrite,
)
from .security import EncryptedSecretStore


class SettingsError(ValueError):
    pass


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        os.chmod(path, 0o600)
    finally:
        temp.unlink(missing_ok=True)


class SettingsRepository:
    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / "settings.json"
        self._lock = threading.RLock()

    def load(self) -> AppSettings:
        with self._lock:
            if not self._path.exists():
                return AppSettings()
            try:
                payload = json.loads(self._path.read_text(encoding="utf-8"))
                # 0.1 development builds used shorter enum names. Keep local
                # installs readable while exposing one stable public contract.
                if isinstance(payload, dict):
                    cry = payload.get("cry")
                    if isinstance(cry, dict) and cry.get("mode") == "audio":
                        cry["mode"] = "rtsp_audio"
                    ai = payload.get("ai")
                    if isinstance(ai, dict) and ai.get("provider") == "local":
                        ai["provider"] = "ollama"
                return AppSettings.model_validate(payload)
            except (OSError, json.JSONDecodeError, ValidationError) as exc:
                raise SettingsError("settings file is invalid") from exc

    def save(self, settings: AppSettings) -> None:
        # Secret-presence booleans are derived from the encrypted store. Keeping
        # them out of persisted config avoids stale or misleading state.
        payload = settings.model_dump(mode="json")
        payload["home_assistant"].pop("access_token_configured", None)
        payload["camera"].pop("stream_url_configured", None)
        payload["cry"].pop("audio_stream_url_configured", None)
        payload["ai"].pop("api_key_configured", None)
        with self._lock:
            _atomic_json_write(self._path, payload)

    def configured(self) -> bool:
        return self._path.is_file()


class SettingsService:
    def __init__(
        self,
        repository: SettingsRepository,
        secrets: EncryptedSecretStore,
        runtime: str = "development",
    ) -> None:
        self._repository = repository
        self._secrets = secrets
        self._runtime = runtime
        self._lock = threading.RLock()

    def get(self) -> AppSettings:
        with self._lock:
            loaded = self._repository.load()
            if self._runtime == "standalone" and not self._repository.configured():
                data = self._stored_shape(loaded)
                data["home_assistant"]["mode"] = HomeAssistantMode.STANDALONE
                loaded = AppSettings.model_validate(data)
            return self._hydrate(loaded, self._secrets.configured())

    def get_secret(self, name: SecretName) -> str | None:
        return self._secrets.get(name.value)

    def configured(self) -> bool:
        return self._repository.configured()

    def validate_secret_bindings(self, candidate: AppSettings, changes: SecretChanges) -> None:
        """Validate an unsaved connection test against write-only secret bindings."""

        with self._lock:
            self._validate_secret_bindings(
                self.get(),
                candidate,
                changes.values(),
                {item.value for item in changes.clear},
            )

    def replace(self, update: SettingsWrite) -> AppSettings:
        data = update.model_dump(exclude={"secrets"}, mode="python")
        return self._commit(data, update.secrets)

    def patch(self, update: SettingsPatch) -> AppSettings:
        with self._lock:
            current = self.get()
            data = self._stored_shape(current)
            for field in (
                "baby",
                "home_assistant",
                "camera",
                "cry",
                "lights",
                "ai",
                "notifications",
                "retention",
            ):
                value = getattr(update, field)
                if value is not None:
                    data[field] = value.model_dump(mode="python")
            return self._commit(data, update.secrets)

    def _commit(self, data: dict[str, Any], changes: SecretChanges) -> AppSettings:
        with self._lock:
            current = self.get()
            updates = changes.values()
            clear = {item.value for item in changes.clear}
            configured = self._secrets.configured()
            configured.difference_update(clear)
            configured.update(updates)
            try:
                candidate = self._hydrate(AppSettings.model_validate(data), configured)
            except ValidationError as exc:
                raise SettingsError(str(exc)) from exc
            self._validate_secret_bindings(current, candidate, updates, clear)
            self._validate_ready(candidate)
            self._secrets.apply(updates, clear)
            self._repository.save(candidate)
            return candidate

    @staticmethod
    def _validate_secret_bindings(
        current: AppSettings,
        candidate: AppSettings,
        updates: dict[str, str],
        clear: set[str],
    ) -> None:
        ha_secret = SecretName.HOME_ASSISTANT_ACCESS_TOKEN.value
        if (
            current.home_assistant.access_token_configured
            and current.home_assistant.base_url != candidate.home_assistant.base_url
            and ha_secret not in updates
            and ha_secret not in clear
        ):
            raise SettingsError("changing the Home Assistant URL requires re-entering its access token")

        ai_secret = SecretName.AI_API_KEY.value
        current_endpoint = (
            None if current.ai.provider == AIProviderName.DISABLED else (current.ai.provider, current.ai.base_url)
        )
        candidate_endpoint = (
            None if candidate.ai.provider == AIProviderName.DISABLED else (candidate.ai.provider, candidate.ai.base_url)
        )
        if (
            current.ai.api_key_configured
            and candidate_endpoint is not None
            and current_endpoint != candidate_endpoint
            and ai_secret not in updates
            and ai_secret not in clear
        ):
            raise SettingsError("changing the AI provider endpoint requires re-entering its API key")

    @staticmethod
    def _stored_shape(settings: AppSettings) -> dict[str, Any]:
        data = settings.model_dump(mode="python")
        data["home_assistant"].pop("access_token_configured", None)
        data["camera"].pop("stream_url_configured", None)
        data["cry"].pop("audio_stream_url_configured", None)
        data["ai"].pop("api_key_configured", None)
        return data

    @staticmethod
    def _hydrate(settings: AppSettings, configured: set[str]) -> AppSettings:
        data = SettingsService._stored_shape(settings)
        data["home_assistant"]["access_token_configured"] = SecretName.HOME_ASSISTANT_ACCESS_TOKEN.value in configured
        data["camera"]["stream_url_configured"] = SecretName.CAMERA_STREAM_URL.value in configured
        data["cry"]["audio_stream_url_configured"] = SecretName.CRY_AUDIO_STREAM_URL.value in configured
        data["ai"]["api_key_configured"] = SecretName.AI_API_KEY.value in configured
        return AppSettings.model_validate(data)

    def _validate_ready(self, settings: AppSettings) -> None:
        if self._runtime == "standalone" and settings.home_assistant.mode != HomeAssistantMode.STANDALONE:
            raise SettingsError("standalone Docker mode requires the standalone Home Assistant connection mode")
        if self._runtime == "home_assistant_app" and settings.home_assistant.mode == HomeAssistantMode.STANDALONE:
            raise SettingsError("Home Assistant App mode must use the Supervisor connection")
        if settings.home_assistant.mode == HomeAssistantMode.STANDALONE and (
            not settings.home_assistant.base_url or not settings.home_assistant.access_token_configured
        ):
            raise SettingsError("standalone Home Assistant mode requires base_url and access token")
        if settings.camera.enabled and not (settings.camera.entity_id or settings.camera.stream_url_configured):
            raise SettingsError("enabled camera requires a camera entity or private stream URL")
        if settings.cry.mode == CryMode.AUDIO and not settings.cry.audio_stream_url_configured:
            raise SettingsError("audio cry detection requires a private audio stream URL")
        if (
            settings.ai.provider in {AIProviderName.GEMINI, AIProviderName.OPENAI}
            and not settings.ai.api_key_configured
        ):
            raise SettingsError("cloud AI provider requires an API key")
