from __future__ import annotations

import base64
import json
from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic import ValidationError

from .models import AIConfig, AIProviderName, VisionLabel

OPENAI_DEFAULT_MODEL = "gpt-5.6-luna"
GEMINI_DEFAULT_MODEL = "gemini-3.1-flash-lite"
OLLAMA_DEFAULT_MODEL = "qwen2.5vl:3b"
OPENAI_BASE_URL = "https://api.openai.com/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

VISION_PROMPT = (
    "Label this baby-monitor frame. Decide whether a baby is visible and, only from visible evidence, "
    "whether the baby appears awake, asleep, or uncertain. Be conservative: never infer sleep when the "
    "baby is absent or occluded. Return only the requested structured object."
)

VISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "baby_present": {"type": "boolean"},
        "state": {"type": "string", "enum": ["awake", "asleep", "uncertain"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "description": {"type": "string", "maxLength": 500},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
    },
    "required": ["baby_present", "state", "confidence", "description", "tags"],
    "additionalProperties": False,
}


class ProviderError(RuntimeError):
    pass


def _data_url(image: bytes, mime_type: str) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(image).decode('ascii')}"


def _validate_base_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ProviderError("provider base URL must be absolute HTTP(S)")
    if parsed.username or parsed.password:
        raise ProviderError("provider base URL must not contain embedded credentials")
    return value.rstrip("/")


def _parse_label(text: str) -> VisionLabel:
    try:
        payload = json.loads(text)
        return VisionLabel.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ProviderError("AI provider returned an invalid vision label") from exc


def _openai_output_text(payload: dict[str, Any]) -> str:
    if payload.get("status") != "completed":
        raise ProviderError("OpenAI response did not complete")
    for item in payload.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if not isinstance(part, dict):
                continue
            if part.get("type") == "refusal":
                raise ProviderError("OpenAI refused to label this image")
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                return part["text"]
    raise ProviderError("OpenAI response did not contain structured output")


class VisionProvider(ABC):
    name: str
    model: str

    @abstractmethod
    async def label(self, image: bytes, mime_type: str, detail: str) -> VisionLabel:
        raise NotImplementedError

    @abstractmethod
    async def probe(self) -> None:
        raise NotImplementedError


class _HTTPProvider(VisionProvider):
    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._injected_client = client

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        try:
            if self._injected_client is not None:
                response = await self._injected_client.request(method, url, **kwargs)
            else:
                async with httpx.AsyncClient(timeout=httpx.Timeout(45, connect=8), follow_redirects=False) as client:
                    response = await client.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPError as exc:
            status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
            suffix = f" (HTTP {status})" if status else ""
            raise ProviderError(f"AI provider request failed{suffix}") from exc


class OpenAIResponsesProvider(_HTTPProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str | None, base_url: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if base_url is not None and _validate_base_url(base_url) != OPENAI_BASE_URL:
            raise ProviderError("OpenAI uses its fixed official API endpoint")
        self.api_key = api_key
        self.model = model or OPENAI_DEFAULT_MODEL
        self.base_url = OPENAI_BASE_URL

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    async def label(self, image: bytes, mime_type: str, detail: str) -> VisionLabel:
        payload = {
            "model": self.model,
            "store": False,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": VISION_PROMPT},
                        {
                            "type": "input_image",
                            "image_url": _data_url(image, mime_type),
                            "detail": detail,
                        },
                    ],
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "vision_label",
                    "strict": True,
                    "schema": VISION_SCHEMA,
                }
            },
        }
        response = await self._request("POST", f"{self.base_url}/responses", headers=self._headers, json=payload)
        data = response.json()
        if not isinstance(data, dict):
            raise ProviderError("OpenAI returned an invalid response")
        return _parse_label(_openai_output_text(data))

    async def probe(self) -> None:
        await self._request("GET", f"{self.base_url}/models/{self.model}", headers=self._headers)


class GeminiInteractionsProvider(_HTTPProvider):
    name = "gemini"

    def __init__(self, api_key: str, model: str | None, base_url: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if base_url is not None and _validate_base_url(base_url) != GEMINI_BASE_URL:
            raise ProviderError("Gemini uses its fixed official API endpoint")
        self.api_key = api_key
        self.model = model or GEMINI_DEFAULT_MODEL
        self.base_url = GEMINI_BASE_URL

    @property
    def _headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}

    @staticmethod
    def _output_text(payload: dict[str, Any]) -> str:
        # Interactions responses have evolved while in beta. Only accept known
        # text containers; never stringify the whole response (which could leak
        # metadata or accidentally accept an unstructured answer).
        if isinstance(payload.get("output_text"), str):
            return payload["output_text"]
        for key in ("outputs", "output"):
            raw = payload.get(key)
            items = raw if isinstance(raw, list) else [raw] if isinstance(raw, dict) else []
            for item in items:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                parts = content if isinstance(content, list) else [content] if isinstance(content, dict) else []
                for part in parts:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        return part["text"]
                if isinstance(item.get("text"), str):
                    return item["text"]
        if isinstance(payload.get("text"), str):
            return payload["text"]
        raise ProviderError("Gemini response did not contain structured output")

    async def label(self, image: bytes, mime_type: str, detail: str) -> VisionLabel:
        del detail  # Gemini chooses image resolution for inline image input.
        payload = {
            "model": self.model,
            "input": [
                {"type": "text", "text": VISION_PROMPT},
                {
                    "type": "image",
                    "data": base64.b64encode(image).decode("ascii"),
                    "mime_type": mime_type,
                },
            ],
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": VISION_SCHEMA,
            },
        }
        response = await self._request("POST", f"{self.base_url}/interactions", headers=self._headers, json=payload)
        data = response.json()
        if not isinstance(data, dict):
            raise ProviderError("Gemini returned an invalid response")
        return _parse_label(self._output_text(data))

    async def probe(self) -> None:
        await self._request("GET", f"{self.base_url}/models/{self.model}", headers=self._headers)


class OpenAICompatibleProvider(_HTTPProvider):
    name = "ollama"

    def __init__(self, api_key: str | None, model: str | None, base_url: str | None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.api_key = api_key
        self.model = model or OLLAMA_DEFAULT_MODEL
        self.base_url = _validate_base_url(base_url or "http://127.0.0.1:11434/v1")

    @property
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def label(self, image: bytes, mime_type: str, detail: str) -> VisionLabel:
        del detail
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_PROMPT},
                        {"type": "image_url", "image_url": {"url": _data_url(image, mime_type)}},
                    ],
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "vision_label", "strict": True, "schema": VISION_SCHEMA},
            },
            "stream": False,
        }
        response = await self._request("POST", f"{self.base_url}/chat/completions", headers=self._headers, json=payload)
        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("OpenAI-compatible provider returned an invalid response") from exc
        if not isinstance(content, str):
            raise ProviderError("OpenAI-compatible provider returned non-text output")
        return _parse_label(content)

    async def probe(self) -> None:
        await self._request("GET", f"{self.base_url}/models", headers=self._headers)


def build_provider(
    config: AIConfig,
    api_key: str | None,
    *,
    client: httpx.AsyncClient | None = None,
) -> VisionProvider:
    kwargs = {"client": client} if client is not None else {}
    if config.provider == AIProviderName.OPENAI:
        if not api_key:
            raise ProviderError("OpenAI API key is not configured")
        return OpenAIResponsesProvider(api_key, config.model, config.base_url, **kwargs)
    if config.provider == AIProviderName.GEMINI:
        if not api_key:
            raise ProviderError("Gemini API key is not configured")
        return GeminiInteractionsProvider(api_key, config.model, config.base_url, **kwargs)
    if config.provider == AIProviderName.OLLAMA:
        return OpenAICompatibleProvider(api_key, config.model, config.base_url, **kwargs)
    raise ProviderError("image labeling is disabled")
