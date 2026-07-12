from __future__ import annotations

import re
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any, Literal
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


ENTITY_ID_PATTERN = re.compile(r"^[a-z0-9_]+\.[a-z0-9_]+$")


def validate_entity_id(value: str, domain: str | None = None) -> str:
    if not ENTITY_ID_PATTERN.fullmatch(value):
        raise ValueError("entity ID must contain only a lowercase domain and object ID")
    if domain is not None and not value.startswith(f"{domain}."):
        raise ValueError(f"entity ID must start with {domain}.")
    return value


STREAM_URL_SCHEMES = frozenset({"http", "https", "rtsp", "rtsps"})


def validate_stream_url(value: str) -> str:
    """Validate a camera/audio URL before it reaches ffmpeg.

    Local network hosts and embedded credentials are intentionally supported,
    but file-like protocols and ffconcat control characters are not.
    """

    if not value or len(value) > 4096 or any(character.isspace() for character in value):
        raise ValueError("stream URL is invalid")
    try:
        parsed = urlparse(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError("stream URL is invalid") from exc
    if (
        parsed.scheme.lower() not in STREAM_URL_SCHEMES
        or not parsed.netloc
        or not hostname
        or (port is not None and not 1 <= port <= 65535)
    ):
        schemes = ", ".join(sorted(STREAM_URL_SCHEMES))
        raise ValueError(f"stream URL must be an absolute {schemes} URL")
    return value


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class HomeAssistantMode(StrEnum):
    AUTO = "auto"
    SUPERVISOR = "supervisor"
    STANDALONE = "standalone"


class CryMode(StrEnum):
    DISABLED = "disabled"
    BINARY_SENSOR = "binary_sensor"
    RTSP_AUDIO = "rtsp_audio"
    # Kept as a code-level alias for callers created before the public API was
    # named. Serialised settings always use ``rtsp_audio``.
    AUDIO = "rtsp_audio"


class AIProviderName(StrEnum):
    DISABLED = "disabled"
    GEMINI = "gemini"
    OPENAI = "openai"
    OLLAMA = "ollama"
    # Ollama exposes an OpenAI-compatible endpoint. The adapter is intentionally
    # generic enough for LM Studio and other compatible local servers too.
    LOCAL = "ollama"


class RetentionMode(StrEnum):
    FOREVER = "forever"
    DAYS = "days"


class BabyProfile(StrictModel):
    name: str = Field(default="Baby", min_length=1, max_length=80)
    birth_date: date | None = None
    timezone: str = "UTC"

    @field_validator("timezone")
    @classmethod
    def valid_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("timezone must be a valid IANA timezone") from exc
        return value


class HomeAssistantConfigBase(StrictModel):
    mode: HomeAssistantMode = HomeAssistantMode.AUTO
    base_url: str | None = None

    @field_validator("base_url")
    @classmethod
    def valid_base_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password:
            raise ValueError("base_url must not contain embedded credentials")
        return value.rstrip("/")


class HomeAssistantConfig(HomeAssistantConfigBase):
    access_token_configured: bool = False


class CameraConfigBase(StrictModel):
    enabled: bool = False
    entity_id: str | None = None
    capture_interval_seconds: int = Field(default=300, ge=30, le=86_400)

    @field_validator("entity_id")
    @classmethod
    def camera_entity(cls, value: str | None) -> str | None:
        if value is not None:
            validate_entity_id(value, "camera")
        return value


class CameraConfig(CameraConfigBase):
    stream_url_configured: bool = False


class CrySourceConfigBase(StrictModel):
    mode: CryMode = CryMode.DISABLED
    entity_id: str | None = None
    positive_windows: int = Field(default=2, ge=1, le=10)
    window_seconds: float = Field(default=0.5, ge=0.1, le=10)
    clear_after_seconds: int = Field(default=8, ge=1, le=300)
    sensitivity: Literal["low", "balanced", "high"] = "balanced"

    @field_validator("mode", mode="before")
    @classmethod
    def legacy_audio_name(cls, value: object) -> object:
        return "rtsp_audio" if value == "audio" else value

    @model_validator(mode="after")
    def valid_source(self) -> CrySourceConfigBase:
        if self.mode == CryMode.BINARY_SENSOR:
            if not self.entity_id:
                raise ValueError("binary_sensor mode requires a binary_sensor entity_id")
            validate_entity_id(self.entity_id, "binary_sensor")
        elif self.entity_id is not None:
            raise ValueError("entity_id is only valid for binary_sensor mode")
        return self


class CrySourceConfig(CrySourceConfigBase):
    audio_stream_url_configured: bool = False


class LightAlertConfig(StrictModel):
    entity_ids: list[str] = Field(default_factory=list, max_length=32)
    duration_seconds: int = Field(default=30, ge=1, le=3600)
    brightness_percent: int = Field(default=100, ge=1, le=100)
    color_rgb: tuple[int, int, int] = (255, 40, 20)
    restore_previous_state: Literal[True] = True

    @field_validator("entity_ids")
    @classmethod
    def valid_lights(cls, value: list[str]) -> list[str]:
        for entity_id in value:
            validate_entity_id(entity_id, "light")
        if len(value) != len(set(value)):
            raise ValueError("light entity_ids must be unique")
        return value

    @field_validator("color_rgb")
    @classmethod
    def valid_color(cls, value: tuple[int, int, int]) -> tuple[int, int, int]:
        if any(channel < 0 or channel > 255 for channel in value):
            raise ValueError("RGB channels must be between 0 and 255")
        return value


class AIConfigBase(StrictModel):
    provider: AIProviderName = AIProviderName.DISABLED
    model: str | None = Field(default=None, max_length=120)
    base_url: str | None = None
    cloud_image_consent: bool = False
    detail: Literal["low", "high", "auto"] = "low"

    @field_validator("provider", mode="before")
    @classmethod
    def compatible_provider_name(cls, value: object) -> object:
        return "ollama" if value == "local" else value

    @field_validator("base_url")
    @classmethod
    def valid_base_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password:
            raise ValueError("base_url must not contain embedded credentials")
        return value.rstrip("/")

    @model_validator(mode="after")
    def cloud_requires_consent(self) -> AIConfigBase:
        if self.provider != AIProviderName.OLLAMA and self.base_url is not None:
            raise ValueError("base_url is only valid for a local OpenAI-compatible provider")
        if self.provider != AIProviderName.DISABLED and not self.cloud_image_consent:
            raise ValueError("cloud_image_consent is required before sending images to any AI endpoint")
        return self


class AIConfig(AIConfigBase):
    api_key_configured: bool = False


class RetentionPolicy(StrictModel):
    mode: RetentionMode = RetentionMode.FOREVER
    days: int | None = Field(default=None, ge=1, le=3650)

    @model_validator(mode="after")
    def valid_policy(self) -> RetentionPolicy:
        if self.mode == RetentionMode.DAYS and self.days is None:
            raise ValueError("days is required when retention mode is days")
        if self.mode == RetentionMode.FOREVER and self.days is not None:
            raise ValueError("days must be omitted when retention mode is forever")
        return self


class NotificationConfig(StrictModel):
    service: str | None = None
    targets: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("service")
    @classmethod
    def notification_service(cls, value: str | None) -> str | None:
        if value is not None:
            validate_entity_id(value, "notify")
        return value

    @field_validator("targets")
    @classmethod
    def unique_targets(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("notification targets must be unique")
        return value


class AppSettings(StrictModel):
    schema_version: Literal[1] = 1
    baby: BabyProfile = Field(default_factory=BabyProfile)
    home_assistant: HomeAssistantConfig = Field(default_factory=HomeAssistantConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    cry: CrySourceConfig = Field(default_factory=CrySourceConfig)
    lights: LightAlertConfig = Field(default_factory=LightAlertConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    notifications: NotificationConfig = Field(default_factory=NotificationConfig)
    retention: RetentionPolicy = Field(default_factory=RetentionPolicy)


class SecretName(StrEnum):
    HOME_ASSISTANT_ACCESS_TOKEN = "home_assistant_access_token"
    CAMERA_STREAM_URL = "camera_stream_url"
    CRY_AUDIO_STREAM_URL = "cry_audio_stream_url"
    AI_API_KEY = "ai_api_key"


class SecretChanges(StrictModel):
    home_assistant_access_token: SecretStr | None = None
    camera_stream_url: SecretStr | None = None
    cry_audio_stream_url: SecretStr | None = None
    ai_api_key: SecretStr | None = None
    clear: list[SecretName] = Field(default_factory=list)

    @field_validator("camera_stream_url", "cry_audio_stream_url")
    @classmethod
    def valid_stream_secret(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None:
            validate_stream_url(value.get_secret_value())
        return value

    @field_validator("clear")
    @classmethod
    def unique_clear(cls, value: list[SecretName]) -> list[SecretName]:
        if len(value) != len(set(value)):
            raise ValueError("secret clear entries must be unique")
        return value

    def values(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for name in SecretName:
            value = getattr(self, name.value)
            if value is not None:
                raw = value.get_secret_value()
                if not raw:
                    raise ValueError(f"{name.value} must not be empty")
                result[name.value] = raw
        return result


class SettingsWrite(StrictModel):
    schema_version: Literal[1] = 1
    baby: BabyProfile = Field(default_factory=BabyProfile)
    home_assistant: HomeAssistantConfigBase = Field(default_factory=HomeAssistantConfigBase)
    camera: CameraConfigBase = Field(default_factory=CameraConfigBase)
    cry: CrySourceConfigBase = Field(default_factory=CrySourceConfigBase)
    lights: LightAlertConfig = Field(default_factory=LightAlertConfig)
    ai: AIConfigBase = Field(default_factory=AIConfigBase)
    notifications: NotificationConfig = Field(default_factory=NotificationConfig)
    retention: RetentionPolicy = Field(default_factory=RetentionPolicy)
    secrets: SecretChanges = Field(default_factory=SecretChanges)


class SettingsPatch(StrictModel):
    baby: BabyProfile | None = None
    home_assistant: HomeAssistantConfigBase | None = None
    camera: CameraConfigBase | None = None
    cry: CrySourceConfigBase | None = None
    lights: LightAlertConfig | None = None
    ai: AIConfigBase | None = None
    notifications: NotificationConfig | None = None
    retention: RetentionPolicy | None = None
    secrets: SecretChanges = Field(default_factory=SecretChanges)


class HAEntity(StrictModel):
    entity_id: str
    state: str
    name: str
    attributes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("entity_id")
    @classmethod
    def valid_entity_id(cls, value: str) -> str:
        return validate_entity_id(value)


class VisionLabel(StrictModel):
    baby_present: bool
    state: Literal["awake", "asleep", "uncertain"]
    confidence: float = Field(ge=0, le=1)
    description: str = Field(max_length=500)
    tags: list[str] = Field(default_factory=list, max_length=20)


class FrameRecord(StrictModel):
    id: str
    captured_at: datetime
    camera_entity_id: str | None = None
    mime_type: str
    size_bytes: int = Field(ge=0)
    sha256: str
    image_available: bool = True
    label: VisionLabel | None = None
    provider: str | None = None
    model: str | None = None


class SleepEventCreate(StrictModel):
    started_at: datetime
    ended_at: datetime | None = None
    kind: Literal["nap", "night", "unknown"] = "unknown"
    source: Literal["manual", "vision", "import"] = "manual"
    notes: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def valid_dates(self) -> SleepEventCreate:
        if self.ended_at is not None and self.ended_at <= self.started_at:
            raise ValueError("ended_at must be after started_at")
        return self


class SleepEventPatch(StrictModel):
    started_at: datetime | None = None
    ended_at: datetime | None = None
    kind: Literal["nap", "night", "unknown"] | None = None
    notes: str | None = Field(default=None, max_length=1000)


class SleepStartRequest(StrictModel):
    started_at: datetime = Field(default_factory=utc_now)
    kind: Literal["nap", "night", "unknown"] = "unknown"
    notes: str | None = Field(default=None, max_length=1000)
    source: Literal["manual"] = "manual"


class SleepStopRequest(StrictModel):
    ended_at: datetime = Field(default_factory=utc_now)


class CryWebhook(StrictModel):
    state: Literal["on", "off"] = "on"
    observed_at: datetime = Field(default_factory=utc_now)
    source: Literal["binary_sensor", "audio", "manual"] = "manual"
    confidence: float | None = Field(default=None, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SleepEvent(SleepEventCreate):
    id: str
    created_at: datetime = Field(default_factory=utc_now)


class CryEventCreate(StrictModel):
    detected_at: datetime
    ended_at: datetime | None = None
    source: Literal["binary_sensor", "audio", "manual", "import"]
    confidence: float | None = Field(default=None, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_dates(self) -> CryEventCreate:
        if self.ended_at is not None and self.ended_at <= self.detected_at:
            raise ValueError("ended_at must be after detected_at")
        return self


class CryEvent(CryEventCreate):
    id: str
    created_at: datetime = Field(default_factory=utc_now)


class Page(StrictModel):
    items: list[Any]
    limit: int
    offset: int
    total: int


class TestResult(StrictModel):
    ok: bool
    component: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
