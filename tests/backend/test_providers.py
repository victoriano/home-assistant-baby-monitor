from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
from baby_monitor.providers import (
    VISION_PROMPT,
    VISION_SCHEMA,
    GeminiInteractionsProvider,
    OpenAICompatibleProvider,
    OpenAIResponsesProvider,
    ProviderError,
    YoloLocalProvider,
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


def _yolo_artifacts(tmp_path) -> YoloLocalProvider:
    root = tmp_path / "yolo-model"
    models = root / "models"
    models.mkdir(parents=True)
    for task in ("presence", "awake", "pacifier"):
        (models / f"{task}.pt").write_bytes(b"test model")
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "format": "Ultralytics YOLO26 classification",
                "version": "baby-monitor-yolo-test",
                "image_size": 320,
                "tasks": {
                    "presence": {
                        "path": "models/presence.pt",
                        "positive_class": "present",
                        "threshold": 0.5,
                    },
                    "awake": {
                        "path": "models/awake.pt",
                        "positive_class": "awake",
                        "threshold": 0.5,
                    },
                    "pacifier": {
                        "path": "models/pacifier.pt",
                        "positive_class": "yes",
                        "threshold": 0.5,
                    },
                },
                "roi_profiles": {
                    "granada": [
                        {
                            "name": "family_bed",
                            "rect": [0.0, 0.05, 0.65, 0.7],
                            "surface": "family_bed",
                        },
                        {
                            "name": "crib",
                            "rect": [0.5, 0.45, 1.0, 0.95],
                            "surface": "crib",
                        },
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    return YoloLocalProvider(str(root))


def test_yolo_local_provider_maps_calibrated_scores_to_vision_contract(tmp_path) -> None:
    provider = _yolo_artifacts(tmp_path)
    profiles = provider.metadata["roi_profiles"]["granada"]

    label = provider._build_label(profiles, [0.92, 0.08], 0.88, 0.81)

    assert label.baby_present is True
    assert label.state == "awake"
    assert label.sleep_surface == "family_bed"
    assert label.in_crib is False
    assert label.pacifier == "yes"
    assert {"local_yolo", "baby_present", "awake", "pacifier"} <= set(label.tags)


def test_yolo_local_provider_is_conservative_for_absent_or_ambiguous_frames(tmp_path) -> None:
    provider = _yolo_artifacts(tmp_path)
    profiles = provider.metadata["roi_profiles"]["granada"]

    absent = provider._build_label(profiles, [0.04, 0.08], None, None)
    ambiguous = provider._build_label(profiles, [0.53, 0.12], None, None)

    assert absent.baby_present is False
    assert absent.state == "uncertain"
    assert "baby_absent" in absent.tags
    assert ambiguous.state == "uncertain"
    assert ambiguous.sleep_surface == "unknown"
    assert "image_uncertain" in ambiguous.tags


def test_yolo_local_provider_marks_unavailable_detail_without_guessing(tmp_path) -> None:
    provider = _yolo_artifacts(tmp_path)
    profiles = provider.metadata["roi_profiles"]["granada"]

    label = provider._build_label(profiles, [0.92, 0.08], None, None)

    assert label.baby_present is True
    assert label.state == "uncertain"
    assert label.pacifier == "unknown"
    assert "detail_unavailable" in label.tags


def test_yolo_local_provider_rejects_model_paths_outside_artifact(tmp_path) -> None:
    provider = _yolo_artifacts(tmp_path)
    root = provider.root
    outside = root.parent / "outside.pt"
    outside.write_bytes(b"outside")
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    metadata["tasks"]["presence"]["path"] = "../outside.pt"
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ProviderError, match="model file is unavailable"):
        YoloLocalProvider(str(root))


def test_yolo_local_provider_rejects_model_hash_mismatch(tmp_path) -> None:
    provider = _yolo_artifacts(tmp_path)
    root = provider.root
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    metadata["tasks"]["presence"]["sha256"] = "0" * 64
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ProviderError, match="integrity check failed"):
        YoloLocalProvider(str(root))


def test_yolo_local_provider_rejects_pose_model_paths_outside_artifact(tmp_path) -> None:
    provider = _yolo_artifacts(tmp_path)
    root = provider.root
    outside = root.parent / "pose.pt"
    outside.write_bytes(b"pose")
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    metadata["detail_crop"] = {
        "strategy": "yolo26_pose_head",
        "image_size": 640,
        "model": {"path": "../pose.pt"},
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ProviderError, match="pose model file is unavailable"):
        YoloLocalProvider(str(root))


def test_yolo_local_provider_aggregates_ensemble_scores_with_max(tmp_path) -> None:
    np = pytest.importorskip("numpy")
    provider = _yolo_artifacts(tmp_path)
    root = provider.root
    auxiliary = root / "models" / "awake-aux.pt"
    auxiliary.write_bytes(b"auxiliary model")
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    metadata["tasks"]["awake"]["ensemble"] = [{"path": "models/awake-aux.pt"}]
    metadata["tasks"]["awake"]["aggregation"] = "max"
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    provider = YoloLocalProvider(str(root))

    class FakeTensor:
        def __init__(self, values: list[float]) -> None:
            self.values = values

        def __getitem__(self, index: int) -> FakeTensor:
            return FakeTensor([self.values[index]])

        def detach(self) -> FakeTensor:
            return self

        def cpu(self) -> float:
            return self.values[0]

    class FakeModel:
        def __init__(self, scores: list[float]) -> None:
            self.scores = scores
            self.sources: list[object] = []

        def predict(self, **kwargs: object) -> list[SimpleNamespace]:
            self.sources = list(kwargs["source"])  # type: ignore[arg-type]
            return [
                SimpleNamespace(
                    probs=SimpleNamespace(data=FakeTensor([1 - score, score])),
                    names={0: "asleep", 1: "awake"},
                )
                for score in self.scores
            ]

    first_model = FakeModel([0.2, 0.7])
    second_model = FakeModel([0.8, 0.4])
    provider._models = {"awake": [first_model, second_model]}

    scores = provider._scores(
        "awake",
        [
            np.full((8, 8, 3), [10, 20, 30], dtype=np.uint8),
            np.zeros((8, 8, 3), dtype=np.uint8),
        ],
    )

    assert scores == [0.8, 0.7]
    assert first_model.sources[0][0, 0].tolist() == [30, 20, 10]


def test_yolo_local_provider_rejects_ensemble_paths_outside_artifact(tmp_path) -> None:
    provider = _yolo_artifacts(tmp_path)
    root = provider.root
    outside = root.parent / "awake-aux.pt"
    outside.write_bytes(b"outside")
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    metadata["tasks"]["awake"]["ensemble"] = [{"path": "../awake-aux.pt"}]
    metadata["tasks"]["awake"]["aggregation"] = "max"
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ProviderError, match="ensemble model 1 file is unavailable"):
        YoloLocalProvider(str(root))
