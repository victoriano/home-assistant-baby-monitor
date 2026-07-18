from __future__ import annotations

import json

import httpx
import pytest
from baby_monitor.providers import (
    VISION_PROMPT,
    VISION_SCHEMA,
    GeminiInteractionsProvider,
    OpenAICompatibleProvider,
    OpenAIResponsesProvider,
    ProviderError,
)

LABEL = {
    "baby_present": True,
    "state": "asleep",
    "confidence": 0.93,
    "description": "Baby appears asleep.",
    "tags": ["crib"],
    "in_crib": True,
    "sleep_surface": "crib",
}


def test_vision_contract_supports_crib_and_family_bed_without_using_adult_state() -> None:
    assert VISION_SCHEMA["properties"]["sleep_surface"]["enum"] == [
        "crib",
        "family_bed",
        "other",
        "unknown",
    ]
    assert "sleep_surface" in VISION_SCHEMA["required"]
    assert "Never use an adult's" in VISION_PROMPT
    assert "Both crib and family_bed are valid monitored sleep surfaces" in VISION_PROMPT


async def test_openai_responses_payload_is_private_and_structured() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": json.dumps(LABEL)}],
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIResponsesProvider("secret", None, client=client)
        result = await provider.label(b"jpeg", "image/jpeg", "low")
    assert result.state == "asleep"
    assert result.sleep_surface == "crib"
    assert captured["store"] is False
    assert captured["text"]["format"] == {
        "type": "json_schema",
        "name": "vision_label",
        "strict": True,
        "schema": VISION_SCHEMA,
    }
    assert captured["input"][0]["content"][1]["type"] == "input_image"
    assert captured["input"][0]["content"][1]["image_url"].startswith("data:image/jpeg;base64,")


async def test_openai_refusal_and_incomplete_status_are_errors() -> None:
    for response_json in (
        {"status": "failed", "output": []},
        {"status": "completed", "output": [{"type": "message", "content": [{"type": "refusal"}]}]},
    ):
        transport = httpx.MockTransport(lambda _, body=response_json: httpx.Response(200, json=body))
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(ProviderError):
                await OpenAIResponsesProvider("secret", None, client=client).label(b"jpeg", "image/jpeg", "low")


async def test_gemini_interactions_payload_matches_current_rest_contract() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-goog-api-key"] == "secret"
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "steps": [
                    {
                        "type": "model_output",
                        "content": [{"type": "text", "text": json.dumps(LABEL)}],
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await GeminiInteractionsProvider("secret", None, client=client).label(b"jpeg", "image/jpeg", "low")
    assert result.baby_present is True
    assert captured["input"][0]["type"] == "text"
    assert captured["input"][1] == {
        "type": "image",
        "data": "anBlZw==",
        "mime_type": "image/jpeg",
    }
    assert captured["response_format"] == {
        "type": "text",
        "mime_type": "application/json",
        "schema": VISION_SCHEMA,
    }


async def test_gemini_interactions_rejects_incomplete_response() -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={"status": "failed", "steps": []}))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(ProviderError, match="did not complete"):
            await GeminiInteractionsProvider("secret", None, client=client).label(b"jpeg", "image/jpeg", "low")


async def test_openai_compatible_uses_chat_completions_without_redirects() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(LABEL)}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await OpenAICompatibleProvider(None, "vision-model", "http://127.0.0.1:11434/v1", client=client).label(
            b"jpeg", "image/jpeg", "low"
        )
    assert result.confidence == 0.93
    assert captured["stream"] is False
    assert captured["response_format"]["type"] == "json_schema"


def test_provider_url_rejects_embedded_credentials() -> None:
    with pytest.raises(ProviderError):
        OpenAICompatibleProvider(None, "model", "http://user:pass@localhost:11434/v1")


def test_cloud_providers_reject_custom_endpoints() -> None:
    with pytest.raises(ProviderError):
        OpenAIResponsesProvider("secret", None, "https://attacker.example/v1")
    with pytest.raises(ProviderError):
        GeminiInteractionsProvider("secret", None, "https://attacker.example/v1beta")
