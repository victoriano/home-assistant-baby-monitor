from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import math
import os
import threading
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
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
    "Label this baby-monitor frame. The image may contain a crib or sidecar cot, a nearby adult/family "
    "bed, the baby, and one or more adults. Decide whether the BABY is visible and, only from evidence "
    "visible on the baby, whether the baby appears awake, asleep, or uncertain. Never use an adult's "
    "posture, closed eyes, or stillness to classify the baby's state. Set sleep_surface=crib when the "
    "baby rests on the crib/cot mattress, family_bed when the baby rests on the adult/shared-bed mattress "
    "whether alone or beside an adult, other for another visible surface or being held, and unknown when "
    "the baby's surface is unclear. Both crib and family_bed are valid monitored sleep surfaces. Set "
    "in_crib=true only for sleep_surface=crib and false for family_bed or other. Also describe only clearly "
    "visible details: face visibility, head side, body position, clothing, pacifier use, and whether the "
    "mouth is open. Be conservative: use unknown when the baby is occluded or unclear, never infer sleep "
    "when the baby is absent, and never infer a pacifier from a closed mouth or shadow. When the monitored "
    "area is clearly visible without the baby, set baby_present=false, state=uncertain, in_crib=false and "
    "include the tag baby_absent; include empty_crib too when the crib is clearly empty. When the image "
    "itself is blocked, corrupted or unusable, use state=uncertain and include the tag image_unusable "
    "instead; an unusable image is not evidence that either sleep surface is empty. "
    "Confidence describes the sleep/occupancy classification, not merely confidence that the pixels are "
    "corrupted. "
    "Return only the requested structured object."
)

VISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "baby_present": {"type": "boolean"},
        "state": {"type": "string", "enum": ["awake", "asleep", "uncertain"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "description": {"type": "string", "maxLength": 500},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
        "in_crib": {"type": "boolean"},
        "sleep_surface": {"type": "string", "enum": ["crib", "family_bed", "other", "unknown"]},
        "face_visible": {"type": "string", "enum": ["yes", "no", "unknown"]},
        "head_side": {"type": "string", "enum": ["left", "right", "back", "face_down", "unknown"]},
        "body_position": {"type": "string", "maxLength": 80},
        "clothing_items": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "diaper_only",
                    "short_sleeve_onesie",
                    "long_sleeve_onesie",
                    "sleep_sack",
                    "blanket",
                    "unknown",
                ],
            },
            "maxItems": 5,
        },
        "pacifier": {"type": "string", "enum": ["yes", "no", "unknown"]},
        "mouth_open": {"type": "string", "enum": ["yes", "no", "unknown"]},
    },
    "required": [
        "baby_present",
        "state",
        "confidence",
        "description",
        "tags",
        "in_crib",
        "sleep_surface",
        "face_visible",
        "head_side",
        "body_position",
        "clothing_items",
        "pacifier",
        "mouth_open",
    ],
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
    async def label(
        self,
        image: bytes,
        mime_type: str,
        detail: str,
        *,
        location_id: str | None = None,
    ) -> VisionLabel:
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

    async def label(
        self,
        image: bytes,
        mime_type: str,
        detail: str,
        *,
        location_id: str | None = None,
    ) -> VisionLabel:
        del location_id
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
        status = payload.get("status")
        if isinstance(status, str) and status != "completed":
            raise ProviderError("Gemini response did not complete")
        if isinstance(payload.get("output_text"), str):
            return payload["output_text"]
        # Current REST responses expose model output under steps. Keep the
        # earlier output containers for compatibility with beta revisions and
        # SDK-normalized responses.
        for key in ("steps", "outputs", "output"):
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

    async def label(
        self,
        image: bytes,
        mime_type: str,
        detail: str,
        *,
        location_id: str | None = None,
    ) -> VisionLabel:
        del detail, location_id  # Gemini chooses image resolution for inline image input.
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

    async def label(
        self,
        image: bytes,
        mime_type: str,
        detail: str,
        *,
        location_id: str | None = None,
    ) -> VisionLabel:
        del detail, location_id
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


class YoloLocalProvider(VisionProvider):
    """Run private fine-tuned YOLO classifiers without uploading the frame."""

    name = "yolo"
    _UNCERTAIN_MARGIN = 0.08
    _PACIFIER_UNCERTAIN_MARGIN = 0.1

    def __init__(self, model_dir: str | None) -> None:
        configured = model_dir or os.environ.get("BABY_MONITOR_YOLO_MODEL_DIR")
        if not configured:
            raise ProviderError("YOLO model directory is not configured")
        self.root = Path(configured).expanduser().resolve()
        metadata_path = self.root / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProviderError("YOLO model metadata is unavailable or invalid") from exc
        self.metadata = self._validate_metadata(metadata)
        self.model = str(self.metadata["version"])
        self._models: dict[str, Any] | None = None
        self._lock = threading.RLock()

    def _validate_metadata(self, metadata: object) -> dict[str, Any]:
        if not isinstance(metadata, dict):
            raise ProviderError("YOLO model metadata must be an object")
        image_size = metadata.get("image_size")
        tasks = metadata.get("tasks")
        profiles = metadata.get("roi_profiles")
        if not isinstance(image_size, int) or image_size < 96:
            raise ProviderError("YOLO model metadata has an invalid image size")
        if not isinstance(tasks, dict) or not isinstance(profiles, dict):
            raise ProviderError("YOLO model metadata is incomplete")

        def validate_model_reference(
            task: str,
            reference: object,
            *,
            label: str,
        ) -> Path:
            if not isinstance(reference, dict):
                raise ProviderError(f"YOLO model metadata for {task} is invalid")
            path = reference.get("path")
            if not isinstance(path, str) or not path:
                raise ProviderError(f"YOLO model metadata for {task} is invalid")
            model_path = (self.root / path).resolve()
            if not model_path.is_relative_to(self.root) or not model_path.is_file():
                raise ProviderError(f"YOLO {task} {label} file is unavailable")
            expected_sha256 = reference.get("sha256")
            if expected_sha256 is not None and (
                not isinstance(expected_sha256, str)
                or len(expected_sha256) != 64
                or hashlib.sha256(model_path.read_bytes()).hexdigest() != expected_sha256
            ):
                raise ProviderError(f"YOLO {task} {label} integrity check failed")
            return model_path

        for task in ("presence", "awake", "pacifier"):
            task_metadata = tasks.get(task)
            if not isinstance(task_metadata, dict):
                raise ProviderError(f"YOLO model metadata is missing {task}")
            threshold = task_metadata.get("threshold")
            positive_class = task_metadata.get("positive_class")
            if (
                not isinstance(threshold, int | float)
                or not 0 <= float(threshold) <= 1
                or not isinstance(positive_class, str)
                or not positive_class
            ):
                raise ProviderError(f"YOLO model metadata for {task} is invalid")
            validate_model_reference(task, task_metadata, label="model")
            ensemble = task_metadata.get("ensemble")
            aggregation = task_metadata.get("aggregation")
            if aggregation is not None and aggregation != "max":
                raise ProviderError(f"YOLO ensemble aggregation for {task} is invalid")
            if ensemble is not None:
                if not isinstance(ensemble, list) or not ensemble:
                    raise ProviderError(f"YOLO ensemble metadata for {task} is invalid")
                paths = {task_metadata["path"]}
                for index, reference in enumerate(ensemble, start=1):
                    model_path = validate_model_reference(
                        task,
                        reference,
                        label=f"ensemble model {index}",
                    )
                    relative_path = str(model_path.relative_to(self.root))
                    if relative_path in paths:
                        raise ProviderError(f"YOLO ensemble metadata for {task} has duplicate models")
                    paths.add(relative_path)
            selective = task_metadata.get("thresholds")
            if selective is not None:
                if not isinstance(selective, dict) or "overall" not in selective:
                    raise ProviderError(f"YOLO selective thresholds for {task} are invalid")
                for location, bounds in selective.items():
                    if (
                        not isinstance(location, str)
                        or not isinstance(bounds, dict)
                        or not isinstance(bounds.get("negative"), int | float)
                        or not isinstance(bounds.get("positive"), int | float)
                        or not 0 <= float(bounds["negative"]) < float(bounds["positive"]) <= 1
                    ):
                        raise ProviderError(f"YOLO selective thresholds for {task} are invalid")
        detail_crop = metadata.get("detail_crop")
        if detail_crop is not None:
            if not isinstance(detail_crop, dict) or detail_crop.get("strategy") != "yolo26_pose_head":
                raise ProviderError("YOLO detail crop metadata is invalid")
            pose_image_size = detail_crop.get("image_size")
            pose_model = detail_crop.get("model")
            if (
                not isinstance(pose_image_size, int)
                or pose_image_size < 320
                or not isinstance(pose_model, dict)
                or not isinstance(pose_model.get("path"), str)
                or not pose_model["path"]
            ):
                raise ProviderError("YOLO detail crop metadata is invalid")
            pose_path = (self.root / pose_model["path"]).resolve()
            if not pose_path.is_relative_to(self.root) or not pose_path.is_file():
                raise ProviderError("YOLO pose model file is unavailable")
            pose_sha256 = pose_model.get("sha256")
            if pose_sha256 is not None and (
                not isinstance(pose_sha256, str)
                or len(pose_sha256) != 64
                or hashlib.sha256(pose_path.read_bytes()).hexdigest() != pose_sha256
            ):
                raise ProviderError("YOLO pose model integrity check failed")
            for key, default in (
                ("detection_confidence", 0.2),
                ("nose_confidence", 0.45),
                ("head_keypoint_confidence", 0.3),
            ):
                value = detail_crop.get(key, default)
                if not isinstance(value, int | float) or not 0 <= float(value) <= 1:
                    raise ProviderError("YOLO detail crop metadata is invalid")
        for location, raw_profiles in profiles.items():
            if not isinstance(location, str) or not isinstance(raw_profiles, list) or not raw_profiles:
                raise ProviderError("YOLO ROI profiles are invalid")
            for profile in raw_profiles:
                if not isinstance(profile, dict):
                    raise ProviderError("YOLO ROI profile must be an object")
                rect = profile.get("rect")
                surface = profile.get("surface")
                if (
                    not isinstance(rect, list)
                    or len(rect) != 4
                    or not all(isinstance(value, int | float) for value in rect)
                    or surface not in {"crib", "family_bed", "other", "unknown"}
                ):
                    raise ProviderError("YOLO ROI profile is invalid")
                x0, y0, x1, y1 = (float(value) for value in rect)
                if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                    raise ProviderError("YOLO ROI coordinates are invalid")
        version = metadata.get("version")
        if not isinstance(version, str) or not version:
            raise ProviderError("YOLO model metadata has no version")
        return metadata

    def _ensure_models(self) -> dict[str, Any]:
        with self._lock:
            if self._models is not None:
                return self._models
            try:
                from ultralytics import YOLO
                from ultralytics.utils import SETTINGS
            except ImportError as exc:
                raise ProviderError("YOLO runtime is not installed") from exc
            # Keep this provider local even if a machine-wide Ultralytics
            # configuration has optional telemetry or integrations enabled.
            dict.update(
                SETTINGS,
                {
                    "sync": False,
                    "clearml": False,
                    "comet": False,
                    "dvc": False,
                    "hub": False,
                    "mlflow": False,
                    "neptune": False,
                    "raytune": False,
                    "tensorboard": False,
                    "wandb": False,
                },
            )
            models: dict[str, Any] = {}
            try:
                for task in ("presence", "awake", "pacifier"):
                    task_metadata = self.metadata["tasks"][task]
                    references = [task_metadata, *task_metadata.get("ensemble", [])]
                    models[task] = [
                        YOLO(str((self.root / reference["path"]).resolve()))
                        for reference in references
                    ]
                detail_crop = self.metadata.get("detail_crop")
                if isinstance(detail_crop, dict):
                    pose_path = (self.root / detail_crop["model"]["path"]).resolve()
                    models["detail_pose"] = YOLO(str(pose_path))
            except Exception as exc:
                raise ProviderError("YOLO model files could not be loaded") from exc
            self._models = models
            return models

    def _profiles(self, location_id: str | None) -> list[dict[str, Any]]:
        profiles = self.metadata["roi_profiles"]
        selected = profiles.get(location_id or "")
        if not isinstance(selected, list):
            selected = profiles.get("home")
        if not isinstance(selected, list):
            if len(profiles) == 1:
                selected = next(iter(profiles.values()))
            else:
                raise ProviderError(f"YOLO has no ROI profile for location {location_id or 'unknown'}")
        return selected

    def _scores(self, task: str, images: list[Any]) -> list[float]:
        models = self._ensure_models()
        try:
            import numpy as np

            positive_class = self.metadata["tasks"][task]["positive_class"]
            member_scores: list[list[float]] = []
            # Ultralytics treats in-memory ndarrays as OpenCV/BGR sources,
            # whereas Pillow images are RGB. Match the file-path preprocessing
            # used during training and evaluation, including for daytime color
            # frames where a channel swap materially changes predictions.
            sources = [np.asarray(image)[:, :, ::-1].copy() for image in images]
            for model in models[task]:
                results = model.predict(
                    source=sources,
                    imgsz=int(self.metadata["image_size"]),
                    device=os.environ.get("BABY_MONITOR_YOLO_DEVICE", "cpu"),
                    batch=max(1, len(images)),
                    verbose=False,
                )
                scores: list[float] = []
                for result in results:
                    if result.probs is None:
                        raise ProviderError(f"YOLO {task} model returned no probabilities")
                    positive_index = next(
                        index
                        for index, name in result.names.items()
                        if name == positive_class
                    )
                    scores.append(float(result.probs.data[positive_index].detach().cpu()))
                if len(scores) != len(images):
                    raise ProviderError(f"YOLO {task} model returned an invalid result count")
                member_scores.append(scores)
            # The only supported ensemble is a conservative union: a feature
            # is positive when either independently trained classifier sees it.
            # Abstention thresholds still guard the final decision.
            return [max(scores) for scores in zip(*member_scores, strict=True)]
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"YOLO {task} inference failed") from exc

    @staticmethod
    def _crop(image: Any, profile: dict[str, Any], image_size: int) -> Any:
        from PIL import Image, ImageOps

        width, height = image.size
        x0, y0, x1, y1 = (float(value) for value in profile["rect"])
        cropped = image.crop(
            (
                round(x0 * width),
                round(y0 * height),
                round(x1 * width),
                round(y1 * height),
            )
        )
        return ImageOps.pad(
            cropped,
            (image_size, image_size),
            method=Image.Resampling.LANCZOS,
            color=(114, 114, 114),
            centering=(0.5, 0.5),
        )

    def _detail_crop(self, image: Any, profile: dict[str, Any], image_size: int) -> Any | None:
        """Return a baby-head crop, or None when pose localization is not decisive."""

        detail = self.metadata.get("detail_crop")
        if not isinstance(detail, dict):
            return self._crop(image, profile, image_size)
        models = self._ensure_models()
        try:
            import numpy as np
            from PIL import Image, ImageOps

            result = models["detail_pose"].predict(
                # In-memory Ultralytics sources use OpenCV/BGR channel order.
                source=np.asarray(image)[:, :, ::-1].copy(),
                imgsz=int(detail["image_size"]),
                device=os.environ.get("BABY_MONITOR_YOLO_DEVICE", "cpu"),
                conf=min(0.15, float(detail.get("detection_confidence", 0.2))),
                verbose=False,
            )[0]
            if result.boxes is None or result.keypoints is None or result.keypoints.conf is None:
                return None
            boxes = result.boxes.xyxyn.detach().cpu().tolist()
            box_confidences = result.boxes.conf.detach().cpu().tolist()
            keypoints = result.keypoints.xyn.detach().cpu().tolist()
            keypoint_confidences = result.keypoints.conf.detach().cpu().tolist()
            x0, y0, x1, y1 = (float(value) for value in profile["rect"])
            minimum_detection = float(detail.get("detection_confidence", 0.2))
            minimum_nose = float(detail.get("nose_confidence", 0.45))
            candidates: list[tuple[float, list[float], list[list[float]], list[float]]] = []
            for box, confidence, points, point_confidences in zip(
                boxes,
                box_confidences,
                keypoints,
                keypoint_confidences,
                strict=True,
            ):
                nose_x, nose_y = points[0]
                nose_confidence = point_confidences[0]
                if (
                    confidence < minimum_detection
                    or nose_confidence < minimum_nose
                    or not (x0 <= nose_x <= x1 and y0 <= nose_y <= y1)
                ):
                    continue
                area = max((box[2] - box[0]) * (box[3] - box[1]), 1e-5)
                baby_preference = confidence * nose_confidence / math.sqrt(area)
                candidates.append((baby_preference, box, points, point_confidences))
            if not candidates:
                return None
            _, box, points, point_confidences = max(candidates, key=lambda item: item[0])
            width, height = image.size
            minimum_head = float(detail.get("head_keypoint_confidence", 0.3))
            visible_head = [
                point
                for point, confidence in zip(
                    points[:5],
                    point_confidences[:5],
                    strict=True,
                )
                if confidence >= minimum_head
            ]
            if not visible_head:
                return None
            head_x = [point[0] * width for point in visible_head]
            head_y = [point[1] * height for point in visible_head]
            span = max(max(head_x) - min(head_x), max(head_y) - min(head_y))
            box_width = (box[2] - box[0]) * width
            box_height = (box[3] - box[1]) * height
            side = max(2.2 * span, 0.22 * box_width, 0.16 * box_height, 96)
            side = min(side, 0.5 * min(width, height))
            nose_x = points[0][0] * width
            nose_y = points[0][1] * height
            left = max(0.0, min(width - side, nose_x - side / 2))
            top = max(0.0, min(height - side, nose_y + 0.12 * side - side / 2))
            cropped = image.crop(
                (
                    round(left),
                    round(top),
                    round(left + side),
                    round(top + side),
                )
            )
            return ImageOps.fit(
                cropped,
                (image_size, image_size),
                method=Image.Resampling.LANCZOS,
            )
        except Exception as exc:
            raise ProviderError("YOLO pose localization failed") from exc

    def _build_label(
        self,
        profiles: list[dict[str, Any]],
        presence_scores: list[float],
        awake_score: float | None,
        pacifier_score: float | None,
        location_id: str | None = None,
    ) -> VisionLabel:
        winner = max(range(len(profiles)), key=presence_scores.__getitem__)
        profile = profiles[winner]
        presence_score = presence_scores[winner]
        presence_negative, presence_positive = self._threshold_bounds("presence", location_id)
        baby_present = presence_score >= presence_positive
        tags = ["local_yolo"]

        if not baby_present and presence_score > presence_negative:
            return VisionLabel(
                baby_present=False,
                state="uncertain",
                confidence=0.6,
                description="Local YOLO could not make a decisive occupancy classification.",
                tags=[*tags, "image_uncertain"],
                in_crib=None,
                sleep_surface="unknown",
            )
        if not baby_present:
            tags.append("baby_absent")
            if any(item["surface"] == "crib" for item in profiles):
                tags.append("empty_crib")
            return VisionLabel(
                baby_present=False,
                state="uncertain",
                confidence=max(0.65, min(0.99, 1 - presence_score)),
                description="Local YOLO found no baby in the monitored sleep areas.",
                tags=tags,
                in_crib=False,
                sleep_surface="unknown",
            )

        surface = str(profile["surface"])
        tags.extend(("baby_present", surface))
        awake_negative, awake_positive = self._threshold_bounds("awake", location_id)
        state = "uncertain"
        state_confidence = 0.6
        if awake_score is not None and (awake_score <= awake_negative or awake_score >= awake_positive):
            state = "awake" if awake_score >= awake_positive else "asleep"
            state_confidence = max(awake_score, 1 - awake_score)
            tags.append(state)

        pacifier = "unknown"
        if pacifier_score is not None:
            pacifier_negative, pacifier_positive = self._threshold_bounds(
                "pacifier",
                location_id,
            )
            if pacifier_score <= pacifier_negative or pacifier_score >= pacifier_positive:
                pacifier = "yes" if pacifier_score >= pacifier_positive else "no"
                if pacifier == "yes":
                    tags.append("pacifier")
        if awake_score is None and pacifier_score is None:
            tags.append("detail_unavailable")

        in_crib = True if surface == "crib" else False if surface in {"family_bed", "other"} else None
        confidence = min(max(0.65, min(0.99, presence_score)), state_confidence)
        description = f"Local YOLO detected the baby on {surface}; state={state}; pacifier={pacifier}."
        return VisionLabel(
            baby_present=True,
            state=state,
            confidence=confidence,
            description=description,
            tags=tags,
            in_crib=in_crib,
            sleep_surface=surface,
            pacifier=pacifier,
        )

    def _threshold_bounds(
        self,
        task: str,
        location_id: str | None,
    ) -> tuple[float, float]:
        task_metadata = self.metadata["tasks"][task]
        selective = task_metadata.get("thresholds")
        if isinstance(selective, dict):
            bounds = selective.get(location_id or "", selective.get("overall"))
            if isinstance(bounds, dict):
                return float(bounds["negative"]), float(bounds["positive"])
        threshold = float(task_metadata["threshold"])
        margin = self._PACIFIER_UNCERTAIN_MARGIN if task == "pacifier" else self._UNCERTAIN_MARGIN
        return max(0.0, threshold - margin), min(1.0, threshold + margin)

    def _label_sync(self, image: bytes, location_id: str | None) -> VisionLabel:
        try:
            from PIL import Image, ImageOps, UnidentifiedImageError

            with Image.open(io.BytesIO(image)) as raw:
                decoded = ImageOps.exif_transpose(raw).convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise ProviderError("YOLO could not decode the camera frame") from exc
        profiles = self._profiles(location_id)
        image_size = int(self.metadata["image_size"])
        crops = [self._crop(decoded, profile, image_size) for profile in profiles]
        with self._lock:
            presence_scores = self._scores("presence", crops)
            winner = max(range(len(crops)), key=presence_scores.__getitem__)
            _, presence_positive = self._threshold_bounds("presence", location_id)
            decisive_present = presence_scores[winner] >= presence_positive
            detail_crop = self._detail_crop(decoded, profiles[winner], image_size) if decisive_present else None
            awake_score = self._scores("awake", [detail_crop])[0] if detail_crop is not None else None
            pacifier_score = self._scores("pacifier", [detail_crop])[0] if detail_crop is not None else None
        return self._build_label(
            profiles,
            presence_scores,
            awake_score,
            pacifier_score,
            location_id,
        )

    async def label(
        self,
        image: bytes,
        mime_type: str,
        detail: str,
        *,
        location_id: str | None = None,
    ) -> VisionLabel:
        del mime_type, detail
        return await asyncio.to_thread(self._label_sync, image, location_id)

    def _probe_sync(self) -> None:
        from PIL import Image

        image_size = int(self.metadata["image_size"])
        blank = Image.new("RGB", (image_size, image_size), color=(114, 114, 114))
        for task in ("presence", "awake", "pacifier"):
            self._scores(task, [blank])

    async def probe(self) -> None:
        await asyncio.to_thread(self._probe_sync)


@lru_cache(maxsize=4)
def _cached_yolo_provider(model_dir: str | None) -> YoloLocalProvider:
    return YoloLocalProvider(model_dir)


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
    if config.provider == AIProviderName.YOLO:
        if client is not None:
            raise ProviderError("YOLO does not use an HTTP client")
        return _cached_yolo_provider(config.model)
    raise ProviderError("image labeling is disabled")
