from __future__ import annotations

import base64
import csv
import hashlib
import json
import os
import random
import shutil
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

GEMINI_PILOT_MARKER = ".baby-monitor-gemini-label-pilot"
GEMINI_ANALYSIS_MARKER = ".baby-monitor-gemini-label-analysis"
GEMINI_PROMPT_VERSION = "baby-visible-features-v2"
GEMINI_THINKING_LEVEL = "high"
GEMINI_DEFAULT_MODELS = ("gemini-3.1-pro-preview", "gemini-3.6-flash")
GEMINI_PRICING_SNAPSHOT_DATE = "2026-07-29"
GEMINI_STANDARD_PRICES_PER_MILLION = {
    "gemini-3.1-pro-preview": {"input": 2.0, "output": 12.0},
    "gemini-3.6-flash": {"input": 1.5, "output": 7.5},
}
FEATURE_CONFIDENCE = ("high", "medium", "low", "unknown")
GEMINI_ANALYSIS_FEATURES = {
    "head_orientation": "head_confidence",
    "body_position": "body_confidence",
    "mouth_state": "mouth_confidence",
    "pacifier": "pacifier_confidence",
    "adult_presence": "adult_confidence",
    "adult_count": "adult_confidence",
}
GEMINI_DETAIL_FEATURES = {
    "head_orientation",
    "body_position",
    "mouth_state",
    "pacifier",
}
GEMINI_TO_DETAIL_TASK = {
    "head_side": (
        "head_orientation",
        {
            "toward_camera": "back",
            "image_left": "left",
            "image_right": "right",
        },
    ),
    "body_position": (
        "body_position",
        {
            "supine": "back",
            "prone": "belly",
            "side": "side",
        },
    ),
    "mouth_open": (
        "mouth_state",
        {
            "closed": "no",
            "open": "yes",
        },
    ),
}
GEMINI_LABEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "baby_visible": {
            "type": "string",
            "enum": ["yes", "no", "unknown"],
            "description": "Whether an infant is visibly present in the evidence board.",
        },
        "detail_panels_match_infant": {
            "type": "string",
            "enum": ["yes", "no", "unknown"],
            "description": "Whether BODY, HEAD, and MOUTH depict the same infant as SCENE.",
        },
        "detail_panels_confidence": {
            "type": "string",
            "enum": list(FEATURE_CONFIDENCE),
        },
        "head_orientation": {
            "type": "string",
            "enum": [
                "image_left",
                "image_right",
                "toward_camera",
                "away_from_camera",
                "face_down",
                "unknown",
            ],
        },
        "head_confidence": {"type": "string", "enum": list(FEATURE_CONFIDENCE)},
        "body_position": {
            "type": "string",
            "enum": ["supine", "prone", "side", "upright", "held", "unknown"],
        },
        "body_confidence": {"type": "string", "enum": list(FEATURE_CONFIDENCE)},
        "mouth_state": {
            "type": "string",
            "enum": ["open", "closed", "unknown"],
        },
        "mouth_confidence": {"type": "string", "enum": list(FEATURE_CONFIDENCE)},
        "pacifier": {
            "type": "string",
            "enum": ["present", "absent", "unknown"],
        },
        "pacifier_confidence": {"type": "string", "enum": list(FEATURE_CONFIDENCE)},
        "adult_presence": {
            "type": "string",
            "enum": ["yes", "no", "unknown"],
        },
        "adult_count": {
            "anyOf": [
                {"type": "integer", "minimum": 0, "maximum": 4},
                {"type": "null"},
            ],
        },
        "adult_confidence": {"type": "string", "enum": list(FEATURE_CONFIDENCE)},
        "limitations": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "blur",
                    "dark",
                    "occluded_face",
                    "occluded_body",
                    "out_of_frame",
                    "ambiguous_pose",
                    "ambiguous_person_count",
                    "none",
                ],
            },
            "maxItems": 5,
        },
    },
    "required": [
        "baby_visible",
        "detail_panels_match_infant",
        "detail_panels_confidence",
        "head_orientation",
        "head_confidence",
        "body_position",
        "body_confidence",
        "mouth_state",
        "mouth_confidence",
        "pacifier",
        "pacifier_confidence",
        "adult_presence",
        "adult_count",
        "adult_confidence",
        "limitations",
    ],
    "additionalProperties": False,
}
GEMINI_LABEL_PROMPT = """You are creating conservative ground-truth candidates for a fixed
infrared baby-monitor camera. The single evidence-board image contains four panels from the
same instant: SCENE (complete camera frame), BODY (localized infant), HEAD (enlarged infant
head), and MOUTH (enlarged lower face).

Report only directly visible facts. Use unknown rather than guessing through blankets,
motion blur, darkness, crop boundaries, or an unresolvable viewing angle. First verify
that BODY, HEAD, and MOUTH depict the infant seen in SCENE rather than a nearby adult.
If that correspondence is no or unknown, set head_orientation, body_position,
mouth_state, pacifier, and their four confidence fields to unknown. Adult fields
must always be judged from SCENE.

Definitions:
- head_orientation is the direction the infant's face/nose points in image coordinates.
  image_left and image_right never mean the infant's anatomical left/right.
  Use image_left only when the nose/face axis visibly points toward the image's left
  edge, and image_right only when it points toward the image's right edge.
  toward_camera means a substantially frontal face; away_from_camera means the back of
  the head; face_down means the face points into the mattress and cannot be treated as
  left/right.
- body_position describes the torso, never the head: supine = back on the mattress,
  prone = belly/chest on the mattress, side = torso on either side, upright = torso
  substantially vertical, held = supported in a person's arms. These cameras look down
  from above: a visible front of the shirt/chest/belly normally means supine even if
  the head is turned; a visible back with chest/belly against the mattress means prone.
- mouth_state is known only when the lips/opening are visibly resolvable. A dark line
  between closed lips is closed; open requires a visible gap between the lips.
- pacifier=present only for a pacifier visibly in the infant's mouth. Ignore loose objects.
- adult_presence concerns any visible adult person or unmistakable adult body part in
  SCENE, including a partially covered adult. Count distinct visible adults only. Use a
  null count when presence is clear but exact count is not. Do not infer sex, gender,
  identity, age beyond infant-versus-adult, ethnicity, or any other demographic trait.
- high confidence means unmistakable visual evidence; medium means likely but still
  reviewable; low means weak evidence and should normally accompany an unknown label.

Return only the requested JSON object."""


@dataclass(frozen=True, slots=True)
class PilotFrame:
    frame_id: str
    captured_at: str
    location_id: str
    split: str
    relative_path: str
    source_sha256: str
    board_path: str
    board_sha256: str
    historical_head_orientation: str
    historical_body_position: str
    historical_mouth_state: str


class GeminiLabelingError(RuntimeError):
    """Raised for an invalid teacher response or unsafe pilot workspace."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_generated_directory(output_dir: Path, *, overwrite: bool) -> Path:
    output = output_dir.resolve()
    if output == Path(output.anchor) or len(output.parts) < 4:
        raise ValueError(f"refusing broad output directory: {output}")
    marker = output / GEMINI_PILOT_MARKER
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output} is not empty")
        if not marker.is_file():
            raise ValueError(f"refusing to replace unmarked pilot directory: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    marker.write_text("generated private Gemini labeling pilot\n", encoding="utf-8")
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_source_frames(source_manifest: Path) -> dict[str, dict[str, str]]:
    frames: dict[str, dict[str, str]] = {}
    with source_manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frame_id = row["frame_id"]
            current = frames.get(frame_id)
            if current is None:
                frames[frame_id] = row
                continue
            for key in ("captured_at", "location_id", "relative_path", "sha256", "split"):
                if current[key] != row[key]:
                    raise ValueError(f"inconsistent source metadata for frame {frame_id}")
    return frames


def _read_detail_frames(detail_index: Path) -> dict[str, dict[str, dict[str, str]]]:
    frames: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    with detail_index.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            task = row["task"]
            if task not in {"head_side", "body_position", "mouth_open"}:
                continue
            current = frames[row["frame_id"]].get(task)
            if current is not None and (
                current["crop_path"] != row["crop_path"]
                or current["class_name"] != row["class_name"]
            ):
                raise ValueError(f"duplicate inconsistent detail row for {row['frame_id']}:{task}")
            frames[row["frame_id"]][task] = row
    return dict(frames)


def _candidate_score(tasks: dict[str, dict[str, str]]) -> int:
    score = 0
    if tasks["body_position"]["class_name"] in {"belly", "side"}:
        score += 8
    if tasks["mouth_open"]["class_name"] == "yes":
        score += 10
    if tasks["head_side"]["class_name"] == "back":
        score += 2
    return score


def _select_pilot_ids(
    source_frames: dict[str, dict[str, str]],
    detail_frames: dict[str, dict[str, dict[str, str]]],
    *,
    samples_per_location: int | None,
    seed: int,
) -> list[str]:
    if samples_per_location is not None and samples_per_location < 1:
        raise ValueError("samples_per_location must be positive")
    complete = {
        frame_id: tasks
        for frame_id, tasks in detail_frames.items()
        if set(tasks) == {"head_side", "body_position", "mouth_open"}
        and frame_id in source_frames
    }
    locations = sorted({source_frames[frame_id]["location_id"] for frame_id in complete})
    if not locations:
        raise ValueError("no complete head/body/mouth frame groups found")
    selected: list[str] = []
    for location in locations:
        candidates = [
            frame_id
            for frame_id in complete
            if source_frames[frame_id]["location_id"] == location
        ]
        target = len(candidates) if samples_per_location is None else samples_per_location
        if len(candidates) < target:
            raise ValueError(
                f"location {location} has only {len(candidates)} complete frames; "
                f"requested {target}"
            )
        rng = random.Random(f"{seed}:{location}")
        jitter = {frame_id: rng.random() for frame_id in candidates}
        candidates.sort(
            key=lambda frame_id: (
                source_frames[frame_id]["split"] == "train",
                -_candidate_score(complete[frame_id]),
                jitter[frame_id],
                frame_id,
            )
        )
        chosen: list[str] = []

        def add_matching(
            predicate: Any,
            count: int,
            pool: list[str] = candidates,
            selected_for_location: list[str] = chosen,
            selection_target: int = target,
        ) -> None:
            for candidate in pool:
                if len(selected_for_location) >= selection_target or count <= 0:
                    return
                if candidate not in selected_for_location and predicate(
                    candidate, complete[candidate]
                ):
                    selected_for_location.append(candidate)
                    count -= 1

        # Mine every scarce historical positive that fits, then deliberately
        # cover each directional class. These are sampling hints only: the
        # historical teacher remains weak supervision.
        add_matching(
            lambda _frame_id, tasks: tasks["body_position"]["class_name"] in {"belly", "side"}
            or tasks["mouth_open"]["class_name"] == "yes",
            target,
        )
        directional_target = max(1, target // 4)
        for class_name in ("left", "right", "back"):
            already = sum(
                complete[frame_id]["head_side"]["class_name"] == class_name
                for frame_id in chosen
            )
            add_matching(
                lambda _frame_id, tasks, expected=class_name: (
                    tasks["head_side"]["class_name"] == expected
                ),
                max(0, directional_target - already),
            )
        represented_splits = {source_frames[frame_id]["split"] for frame_id in chosen}
        for split in ("test", "validation", "train"):
            if split not in represented_splits:
                add_matching(
                    lambda frame_id, _tasks, expected=split: (
                        source_frames[frame_id]["split"] == expected
                    ),
                    1,
                )
        for frame_id in candidates:
            if frame_id not in chosen:
                chosen.append(frame_id)
            if len(chosen) == target:
                break
        selected.extend(chosen)
    return sorted(
        selected,
        key=lambda frame_id: (
            source_frames[frame_id]["location_id"],
            source_frames[frame_id]["captured_at"],
            frame_id,
        ),
    )


def _safe_child(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root.resolve()) or not candidate.is_file():
        raise ValueError(f"unsafe or missing source image: {relative_path}")
    return candidate


def _fit_panel(image: Image.Image, size: tuple[int, int], label: str) -> Image.Image:
    width, height = size
    label_height = 34
    panel = Image.new("RGB", size, (10, 12, 18))
    fitted = ImageOps.contain(
        ImageOps.exif_transpose(image).convert("RGB"),
        (width, height - label_height),
        method=Image.Resampling.LANCZOS,
    )
    x = (width - fitted.width) // 2
    y = label_height + (height - label_height - fitted.height) // 2
    panel.paste(fitted, (x, y))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, width, label_height), fill=(15, 20, 31))
    draw.text((12, 9), label, fill=(235, 240, 251), font=ImageFont.load_default())
    return panel


def build_evidence_board(
    scene_path: Path,
    body_path: Path,
    head_path: Path,
    mouth_path: Path,
    output_path: Path,
    *,
    size: int = 1024,
    horizontal_flip: bool = False,
) -> None:
    if size < 512 or size % 2:
        raise ValueError("evidence board size must be an even integer of at least 512")
    half = size // 2
    board = Image.new("RGB", (size, size), (8, 10, 15))
    for source, position, label in (
        (scene_path, (0, 0), "SCENE - complete frame"),
        (body_path, (half, 0), "BODY - localized infant"),
        (head_path, (0, half), "HEAD - enlarged infant head"),
        (mouth_path, (half, half), "MOUTH - enlarged lower face"),
    ):
        with Image.open(source) as raw:
            prepared = ImageOps.exif_transpose(raw).convert("RGB")
            if horizontal_flip:
                prepared = ImageOps.mirror(prepared)
            board.paste(_fit_panel(prepared, (half, half), label), position)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    board.save(output_path, format="JPEG", quality=94, optimize=True)


def prepare_gemini_label_pilot(
    source_manifest: Path,
    frames_dir: Path,
    detail_dataset_dir: Path,
    output_dir: Path,
    *,
    samples_per_location: int | None = 30,
    seed: int = 20260730,
    board_size: int = 1024,
    horizontal_flip: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_manifest = source_manifest.resolve()
    frames_dir = frames_dir.resolve()
    detail_dataset = detail_dataset_dir.resolve()
    detail_index = detail_dataset / "index.csv"
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)
    if not detail_index.is_file():
        raise FileNotFoundError(detail_index)
    output = _safe_generated_directory(output_dir, overwrite=overwrite)
    source_frames = _read_source_frames(source_manifest)
    detail_frames = _read_detail_frames(detail_index)
    selected = _select_pilot_ids(
        source_frames,
        detail_frames,
        samples_per_location=samples_per_location,
        seed=seed,
    )
    records: list[PilotFrame] = []
    for frame_id in selected:
        source = source_frames[frame_id]
        tasks = detail_frames[frame_id]
        scene = _safe_child(frames_dir, source["relative_path"])
        crops = {
            task: _safe_child(detail_dataset, row["crop_path"])
            for task, row in tasks.items()
        }
        board = output / "boards" / f"{frame_id}.jpg"
        build_evidence_board(
            scene,
            crops["body_position"],
            crops["head_side"],
            crops["mouth_open"],
            board,
            size=board_size,
            horizontal_flip=horizontal_flip,
        )
        records.append(
            PilotFrame(
                frame_id=frame_id,
                captured_at=source["captured_at"],
                location_id=source["location_id"],
                split=source["split"],
                relative_path=source["relative_path"],
                source_sha256=source["sha256"],
                board_path=board.relative_to(output).as_posix(),
                board_sha256=_sha256(board),
                historical_head_orientation=tasks["head_side"]["class_name"],
                historical_body_position=tasks["body_position"]["class_name"],
                historical_mouth_state=tasks["mouth_open"]["class_name"],
            )
        )
    fields = list(PilotFrame.__dataclass_fields__)
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: getattr(record, field) for field in fields})
    counts = {
        "frames": len(records),
        "locations": dict(sorted(Counter(record.location_id for record in records).items())),
        "splits": dict(sorted(Counter(record.split for record in records).items())),
        "historical": {
            "head_orientation": dict(
                sorted(Counter(record.historical_head_orientation for record in records).items())
            ),
            "body_position": dict(
                sorted(Counter(record.historical_body_position for record in records).items())
            ),
            "mouth_state": dict(
                sorted(Counter(record.historical_mouth_state for record in records).items())
            ),
        },
    }
    summary = {
        "created_at": _utc_now(),
        "prompt_version": GEMINI_PROMPT_VERSION,
        "seed": seed,
        "board_size": board_size,
        "horizontal_flip": horizontal_flip,
        "source_manifest_sha256": _sha256(source_manifest),
        "detail_index_sha256": _sha256(detail_index),
        **counts,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_gemini_request(image_bytes: bytes) -> dict[str, Any]:
    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": GEMINI_LABEL_PROMPT},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": base64.b64encode(image_bytes).decode("ascii"),
                        },
                        "media_resolution": {"level": "MEDIA_RESOLUTION_HIGH"},
                    },
                ],
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseJsonSchema": GEMINI_LABEL_SCHEMA,
            "thinkingConfig": {"thinkingLevel": GEMINI_THINKING_LEVEL},
            "temperature": 0,
        },
    }


def validate_gemini_label(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GeminiLabelingError("teacher label is not an object")
    value = dict(value)
    expected = set(GEMINI_LABEL_SCHEMA["properties"])
    if set(value) != expected:
        raise GeminiLabelingError(
            f"teacher label fields differ: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )
    enum_fields = {
        field: set(schema["enum"])
        for field, schema in GEMINI_LABEL_SCHEMA["properties"].items()
        if "enum" in schema
    }
    for field, allowed in enum_fields.items():
        if value[field] not in allowed:
            raise GeminiLabelingError(f"invalid {field}: {value[field]!r}")
    adult_count = value["adult_count"]
    if adult_count is not None and (
        type(adult_count) is not int or not 0 <= adult_count <= 4
    ):
        raise GeminiLabelingError(f"invalid adult_count: {adult_count!r}")
    # `no` already determines an exact count. Some otherwise valid structured
    # responses use null here, so canonicalize the redundant field instead of
    # paying for repeated identical retries.
    if value["adult_presence"] == "no" and adult_count is None:
        value["adult_count"] = 0
        adult_count = 0
    if value["adult_presence"] == "no" and adult_count != 0:
        raise GeminiLabelingError("adult_presence=no requires adult_count=0")
    if value["adult_presence"] == "yes" and adult_count == 0:
        raise GeminiLabelingError("adult_presence=yes cannot have adult_count=0")
    if value["adult_presence"] == "unknown" and adult_count is not None:
        raise GeminiLabelingError("adult_presence=unknown requires adult_count=null")
    if value["detail_panels_match_infant"] != "yes":
        for field, confidence_field in (
            ("head_orientation", "head_confidence"),
            ("body_position", "body_confidence"),
            ("mouth_state", "mouth_confidence"),
            ("pacifier", "pacifier_confidence"),
        ):
            if value[field] != "unknown":
                raise GeminiLabelingError(
                    f"{field} must be unknown when detail panels do not clearly match infant"
                )
            if value[confidence_field] != "unknown":
                raise GeminiLabelingError(
                    f"{confidence_field} must be unknown when detail panels do not clearly match infant"
                )
    limitations = value["limitations"]
    allowed_limitations = set(
        GEMINI_LABEL_SCHEMA["properties"]["limitations"]["items"]["enum"]
    )
    if (
        not isinstance(limitations, list)
        or len(limitations) > 5
        or any(item not in allowed_limitations for item in limitations)
    ):
        raise GeminiLabelingError("invalid limitations")
    return value


def _extract_response_label(response: dict[str, Any]) -> dict[str, Any]:
    try:
        candidates = response["candidates"]
        parts = candidates[0]["content"]["parts"]
        text = next(part["text"] for part in parts if part.get("text"))
        parsed = json.loads(text)
    except (KeyError, IndexError, StopIteration, TypeError, json.JSONDecodeError) as exc:
        raise GeminiLabelingError("teacher response has no valid structured label") from exc
    return validate_gemini_label(parsed)


def _estimated_standard_cost(model: str, usage: dict[str, Any]) -> float | None:
    prices = GEMINI_STANDARD_PRICES_PER_MILLION.get(model)
    if prices is None:
        return None
    input_tokens = int(usage.get("promptTokenCount") or 0)
    output_tokens = int(usage.get("candidatesTokenCount") or 0) + int(
        usage.get("thoughtsTokenCount") or 0
    )
    return (
        input_tokens * prices["input"] + output_tokens * prices["output"]
    ) / 1_000_000


def _request_one(
    client: Any,
    *,
    api_key: str,
    model: str,
    image_bytes: bytes,
    timeout_seconds: float,
    retries: int,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        started = time.perf_counter()
        try:
            response = client.post(
                url,
                headers={"x-goog-api-key": api_key},
                json=build_gemini_request(image_bytes),
                timeout=timeout_seconds,
            )
            latency_ms = (time.perf_counter() - started) * 1000
            if response.status_code in {429, 500, 502, 503, 504} and attempt < retries:
                time.sleep(min(2**attempt, 8))
                continue
            response.raise_for_status()
            body = response.json()
            return _extract_response_label(body), body.get("usageMetadata", {}), latency_ms
        except Exception as exc:  # HTTP and schema failures use the same bounded retry ledger.
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
                continue
            break
    raise GeminiLabelingError(f"Gemini request failed after {retries + 1} attempts") from last_error


def _latest_results(results_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    if not results_path.is_file():
        return latest
    with results_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                key = (str(row["frame_id"]), str(row["model"]))
                latest[key] = row
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise GeminiLabelingError(
                    f"invalid results JSONL at line {line_number}"
                ) from exc
    return latest


def _completed_pairs(results_path: Path) -> set[tuple[str, str]]:
    """Return only successful pairs so an error ledger entry remains retryable."""

    return {
        key
        for key, row in _latest_results(results_path).items()
        if row.get("label") is not None and row.get("error") is None
    }


def _safe_analysis_directory(output_dir: Path, *, overwrite: bool) -> Path:
    output = output_dir.resolve()
    if output == Path(output.anchor) or len(output.parts) < 4:
        raise ValueError(f"refusing broad output directory: {output}")
    marker = output / GEMINI_ANALYSIS_MARKER
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output} is not empty")
        if not marker.is_file():
            raise ValueError(f"refusing to replace unmarked analysis directory: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    marker.write_text("generated private Gemini teacher analysis\n", encoding="utf-8")
    return output


def _read_pilot_manifest(pilot: Path) -> dict[str, dict[str, str]]:
    manifest_path = pilot / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    manifest = {row["frame_id"]: row for row in rows}
    if len(manifest) != len(rows):
        raise GeminiLabelingError(f"duplicate frame IDs in {manifest_path}")
    return manifest


def _read_pilot_summary(pilot: Path) -> dict[str, Any]:
    summary_path = pilot / "summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GeminiLabelingError(f"invalid pilot summary: {summary_path}") from exc
    if not isinstance(summary, dict):
        raise GeminiLabelingError(f"invalid pilot summary: {summary_path}")
    return summary


def _mirror_normalized(field: str, value: Any) -> Any:
    if field != "head_orientation":
        return value
    return {
        "image_left": "image_right",
        "image_right": "image_left",
    }.get(value, value)


def _strict_candidate_reasons(
    *,
    field: str,
    values: list[Any],
    confidences: list[str],
    baby_visible: list[str],
    panel_matches: list[str],
    panel_confidences: list[str],
    adult_presence: list[str],
) -> list[str]:
    reasons: list[str] = []
    if len(set(values)) != 1:
        reasons.append("teacher_or_mirror_disagreement")
    elif values[0] in {None, "unknown"}:
        reasons.append("unknown_value")
    if any(confidence != "high" for confidence in confidences):
        reasons.append("confidence_below_high")
    if field in GEMINI_DETAIL_FEATURES:
        if any(value != "yes" for value in baby_visible):
            reasons.append("baby_not_visible_in_all_views")
        if any(value != "yes" for value in panel_matches) or any(
            confidence != "high" for confidence in panel_confidences
        ):
            reasons.append("detail_panel_correspondence_not_high")
    if field == "adult_count" and len(set(adult_presence)) != 1:
        reasons.append("adult_presence_disagreement")
    return reasons


def _manual_review_priority(field: str, value: Any) -> bool:
    return (
        field == "head_orientation"
        or (field == "body_position" and value != "supine")
        or (field == "mouth_state" and value == "open")
        or (field == "pacifier" and value == "present")
        or (field == "adult_presence" and value == "yes")
        or (field == "adult_count" and value not in {0, 1})
    )


def _latency_summary(records: list[dict[str, Any]]) -> dict[str, float | None]:
    values = sorted(
        float(record["latency_ms"])
        for record in records
        if record.get("latency_ms") is not None
    )
    if not values:
        return {"median_ms": None, "p95_ms": None, "maximum_ms": None}
    return {
        "median_ms": round(values[len(values) // 2], 3),
        "p95_ms": round(values[round((len(values) - 1) * 0.95)], 3),
        "maximum_ms": round(values[-1], 3),
    }


def analyze_gemini_teacher_pilots(
    original_pilot_dir: Path,
    flipped_pilot_dir: Path,
    output_dir: Path,
    *,
    models: tuple[str, ...] = GEMINI_DEFAULT_MODELS,
    allow_incomplete: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create conservative pseudo-label candidates from two teachers and a mirror test.

    A candidate is emitted only when both teachers agree on the original and
    horizontally mirrored evidence boards, every confidence is high, and
    task-specific visibility gates pass. These are training candidates, never
    independent held-out ground truth.
    """

    if len(models) < 2 or len(set(models)) != len(models):
        raise ValueError("analysis requires at least two unique teacher models")
    original = original_pilot_dir.resolve()
    flipped = flipped_pilot_dir.resolve()
    for pilot in (original, flipped):
        if not (pilot / GEMINI_PILOT_MARKER).is_file():
            raise ValueError(f"not a marked Gemini pilot directory: {pilot}")
    original_summary = _read_pilot_summary(original)
    flipped_summary = _read_pilot_summary(flipped)
    if original_summary.get("horizontal_flip") is not False:
        raise GeminiLabelingError("original pilot is not marked as unflipped")
    if flipped_summary.get("horizontal_flip") is not True:
        raise GeminiLabelingError("flipped pilot is not marked as horizontally flipped")
    for key in ("prompt_version", "source_manifest_sha256", "detail_index_sha256"):
        if original_summary.get(key) != flipped_summary.get(key):
            raise GeminiLabelingError(f"pilot summaries disagree on {key}")

    original_manifest = _read_pilot_manifest(original)
    flipped_manifest = _read_pilot_manifest(flipped)
    if set(original_manifest) != set(flipped_manifest):
        raise GeminiLabelingError("original and flipped pilots contain different frames")
    for frame_id, row in original_manifest.items():
        mirror = flipped_manifest[frame_id]
        for key in (
            "captured_at",
            "location_id",
            "split",
            "relative_path",
            "source_sha256",
        ):
            if row.get(key) != mirror.get(key):
                raise GeminiLabelingError(
                    f"pilot manifests disagree for {frame_id}:{key}"
                )

    original_results = _latest_results(original / "results.jsonl")
    flipped_results = _latest_results(flipped / "results.jsonl")
    all_expected = {
        (frame_id, model)
        for frame_id in original_manifest
        for model in models
    }
    missing_by_view: dict[str, list[tuple[str, str]]] = {}
    for name, results in (("original", original_results), ("flipped", flipped_results)):
        missing = sorted(
            key
            for key in all_expected
            if key not in results
            or results[key].get("label") is None
            or results[key].get("error") is not None
        )
        missing_by_view[name] = missing
        if missing and not allow_incomplete:
            preview = ", ".join(f"{frame}:{model}" for frame, model in missing[:3])
            raise GeminiLabelingError(
                f"{name} pilot has {len(missing)} missing successful results: {preview}"
            )
    excluded_frame_ids = {
        frame_id
        for missing in missing_by_view.values()
        for frame_id, _model in missing
    }
    analyzed_frame_ids = set(original_manifest) - excluded_frame_ids
    if not analyzed_frame_ids:
        raise GeminiLabelingError("no complete teacher/mirror frame matrix remains")

    output = _safe_analysis_directory(output_dir, overwrite=overwrite)
    teacher_agreement: dict[str, int] = Counter()
    mirror_consistency: dict[str, Counter[str]] = {
        field: Counter() for field in GEMINI_ANALYSIS_FEATURES
    }
    candidates: dict[str, Counter[Any]] = {
        field: Counter() for field in GEMINI_ANALYSIS_FEATURES
    }
    candidate_locations: dict[str, Counter[str]] = {
        field: Counter() for field in GEMINI_ANALYSIS_FEATURES
    }
    candidate_splits: dict[str, Counter[str]] = {
        field: Counter() for field in GEMINI_ANALYSIS_FEATURES
    }
    panel_matches: dict[str, Counter[str]] = {
        model: Counter() for model in models
    }
    comparison_rows: list[dict[str, Any]] = []
    for frame_id in sorted(
        analyzed_frame_ids,
        key=lambda item: (
            original_manifest[item]["location_id"],
            original_manifest[item]["captured_at"],
            item,
        ),
    ):
        manifest_row = original_manifest[frame_id]
        labels: dict[str, dict[str, dict[str, Any]]] = {
            "original": {},
            "flipped": {},
        }
        for model in models:
            labels["original"][model] = original_results[(frame_id, model)]["label"]
            labels["flipped"][model] = flipped_results[(frame_id, model)]["label"]
            for view in ("original", "flipped"):
                label = labels[view][model]
                panel_matches[model][
                    f"{view}:{label['detail_panels_match_infant']}:"
                    f"{label['detail_panels_confidence']}"
                ] += 1

        field_rows: dict[str, Any] = {}
        for field, confidence_field in GEMINI_ANALYSIS_FEATURES.items():
            original_values = [labels["original"][model][field] for model in models]
            if len(set(original_values)) == 1:
                teacher_agreement[field] += 1
            for model in models:
                if labels["original"][model][field] == _mirror_normalized(
                    field, labels["flipped"][model][field]
                ):
                    mirror_consistency[field][model] += 1

            observations: dict[str, Any] = {}
            confidences: dict[str, str] = {}
            values: list[Any] = []
            confidence_values: list[str] = []
            baby_visible: list[str] = []
            matches: list[str] = []
            match_confidences: list[str] = []
            adult_presence: list[str] = []
            for view in ("original", "flipped"):
                for model in models:
                    label = labels[view][model]
                    key = f"{view}:{model}"
                    value = label[field]
                    if view == "flipped":
                        value = _mirror_normalized(field, value)
                    observations[key] = value
                    confidences[key] = label[confidence_field]
                    values.append(value)
                    confidence_values.append(label[confidence_field])
                    baby_visible.append(label["baby_visible"])
                    matches.append(label["detail_panels_match_infant"])
                    match_confidences.append(label["detail_panels_confidence"])
                    adult_presence.append(label["adult_presence"])
            reasons = _strict_candidate_reasons(
                field=field,
                values=values,
                confidences=confidence_values,
                baby_visible=baby_visible,
                panel_matches=matches,
                panel_confidences=match_confidences,
                adult_presence=adult_presence,
            )
            value = values[0] if not reasons else None
            if value is not None:
                candidates[field][value] += 1
                candidate_locations[field][manifest_row["location_id"]] += 1
                candidate_splits[field][manifest_row["split"]] += 1
            field_rows[field] = {
                "candidate": value,
                "observations": observations,
                "confidences": confidences,
                "rejection_reasons": reasons,
                "manual_review_priority": (
                    _manual_review_priority(field, value) if value is not None else False
                ),
            }
        comparison_rows.append(
            {
                "frame_id": frame_id,
                "captured_at": manifest_row["captured_at"],
                "location_id": manifest_row["location_id"],
                "split": manifest_row["split"],
                "source_sha256": manifest_row["source_sha256"],
                "fields": field_rows,
            }
        )

    with (output / "consensus-candidates.jsonl").open("w", encoding="utf-8") as handle:
        for row in comparison_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    total = len(analyzed_frame_ids)
    view_results = {
        "original": [
            original_results[key]
            for key in sorted(all_expected)
            if key in original_results
            and original_results[key].get("label") is not None
            and original_results[key].get("error") is None
        ],
        "flipped": [
            flipped_results[key]
            for key in sorted(all_expected)
            if key in flipped_results
            and flipped_results[key].get("label") is not None
            and flipped_results[key].get("error") is None
        ],
    }
    usage_by_view: dict[str, Any] = {}
    total_cost = 0.0
    for view, records in view_results.items():
        model_usage: dict[str, Any] = {}
        for model in models:
            model_records = [record for record in records if record["model"] == model]
            model_cost = sum(
                float(record["estimated_standard_cost_usd"])
                for record in model_records
                if record.get("estimated_standard_cost_usd") is not None
            )
            total_cost += model_cost
            model_usage[model] = {
                "requests": len(model_records),
                "estimated_standard_cost_usd": round(model_cost, 6),
                "latency": _latency_summary(model_records),
            }
        usage_by_view[view] = model_usage
    summary = {
        "created_at": _utc_now(),
        "prompt_version": original_summary["prompt_version"],
        "models": list(models),
        "frames": total,
        "frames_input": len(original_manifest),
        "excluded_incomplete": [
            {
                "frame_id": frame_id,
                "missing": [
                    f"{view}:{model}"
                    for view, missing in missing_by_view.items()
                    for missing_frame, model in missing
                    if missing_frame == frame_id
                ],
            }
            for frame_id in sorted(excluded_frame_ids)
        ],
        "label_status": "training_candidates_not_ground_truth",
        "api_usage": {
            "estimated_standard_cost_usd": round(total_cost, 6),
            "pricing_snapshot_date": GEMINI_PRICING_SNAPSHOT_DATE,
            "views": usage_by_view,
        },
        "rule": (
            "Both teachers and both mirror-normalized views must agree with high "
            "confidence; detail tasks also require confirmed infant crop correspondence."
        ),
        "teacher_agreement_original": {
            field: {
                "count": teacher_agreement[field],
                "rate": round(teacher_agreement[field] / total, 6),
            }
            for field in GEMINI_ANALYSIS_FEATURES
        },
        "mirror_consistency": {
            field: {
                model: {
                    "count": mirror_consistency[field][model],
                    "rate": round(mirror_consistency[field][model] / total, 6),
                }
                for model in models
            }
            for field in GEMINI_ANALYSIS_FEATURES
        },
        "strict_candidates": {
            field: {
                "total": sum(candidates[field].values()),
                "values": dict(sorted(candidates[field].items(), key=lambda item: str(item[0]))),
                "locations": dict(sorted(candidate_locations[field].items())),
                "splits": dict(sorted(candidate_splits[field].items())),
            }
            for field in GEMINI_ANALYSIS_FEATURES
        },
        "detail_panel_checks": {
            model: dict(sorted(counts.items()))
            for model, counts in panel_matches.items()
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _balanced_candidate_rows(
    rows: list[dict[str, str]],
    classes: tuple[str, ...],
    *,
    seed: int,
    max_minority_repeats: int,
) -> list[tuple[dict[str, str], int]]:
    by_class: dict[str, list[dict[str, str]]] = {}
    for class_name in classes:
        by_class[class_name] = sorted(
            (row for row in rows if row["class_name"] == class_name),
            key=lambda row: hashlib.sha256(
                f"{seed}:{row['task']}:{row['frame_id']}".encode()
            ).hexdigest(),
        )
    missing = [class_name for class_name, values in by_class.items() if not values]
    if missing:
        raise GeminiLabelingError(
            "Gemini candidate split has no examples for: " + ", ".join(missing)
        )
    desired = min(
        max(len(values) for values in by_class.values()),
        min(len(values) for values in by_class.values()) * max_minority_repeats,
    )
    balanced: list[tuple[dict[str, str], int]] = []
    for values in by_class.values():
        selected = values[:desired]
        for index in range(desired):
            balanced.append((selected[index % len(selected)], index // len(selected)))
    return sorted(
        balanced,
        key=lambda item: hashlib.sha256(
            f"{seed}:{item[0]['task']}:{item[0]['frame_id']}:{item[1]}".encode()
        ).hexdigest(),
    )


def prepare_gemini_detail_dataset(
    analysis_dir: Path,
    source_detail_dataset_dir: Path,
    output_dir: Path,
    *,
    tasks: tuple[str, ...],
    seed: int = 20260730,
    max_minority_repeats: int = 8,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Materialize selected strict candidates for diagnostic YOLO training.

    The resulting validation and test labels remain teacher consensus, not
    manually adjudicated ground truth. The summary records that limitation so
    this dataset cannot be mistaken for deployment evidence.
    """

    from .detail_training import (
        DETAIL_DATASET_MARKER,
        DETAIL_INDEX_FIELDS,
        DETAIL_TASK_CLASSES,
        DETAIL_TASK_CROPS,
    )
    from .yolo_training import _safe_generated_directory

    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("tasks must be a non-empty unique tuple")
    unsupported = set(tasks) - set(GEMINI_TO_DETAIL_TASK)
    if unsupported:
        raise ValueError(f"unsupported Gemini detail tasks: {sorted(unsupported)}")
    analysis = analysis_dir.resolve()
    source = source_detail_dataset_dir.resolve()
    if not (analysis / GEMINI_ANALYSIS_MARKER).is_file():
        raise ValueError(f"not a marked Gemini analysis directory: {analysis}")
    if not (source / "index.csv").is_file():
        raise FileNotFoundError(source / "index.csv")
    if max_minority_repeats < 1:
        raise ValueError("max_minority_repeats must be positive")
    output = _safe_generated_directory(
        output_dir,
        DETAIL_DATASET_MARKER,
        overwrite=overwrite,
    )

    analysis_summary = json.loads(
        (analysis / "summary.json").read_text(encoding="utf-8")
    )
    analysis_rows = [
        json.loads(line)
        for line in (analysis / "consensus-candidates.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    source_rows: dict[tuple[str, str], dict[str, str]] = {}
    with (source / "index.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["frame_id"], row["task"])
            current = source_rows.get(key)
            if current is not None and current["crop_path"] != row["crop_path"]:
                raise GeminiLabelingError(
                    f"inconsistent source crop for {row['frame_id']}:{row['task']}"
                )
            source_rows[key] = row

    candidates: list[dict[str, str]] = []
    for analysis_row in analysis_rows:
        frame_id = analysis_row["frame_id"]
        for task in tasks:
            feature, mapping = GEMINI_TO_DETAIL_TASK[task]
            teacher_value = analysis_row["fields"][feature]["candidate"]
            class_name = mapping.get(teacher_value)
            if class_name is None:
                continue
            source_row = source_rows.get((frame_id, task))
            if source_row is None:
                raise GeminiLabelingError(f"missing source crop for {frame_id}:{task}")
            candidates.append(
                {
                    "task": task,
                    "sample_id": source_row["sample_id"],
                    "frame_id": frame_id,
                    "captured_at": analysis_row["captured_at"],
                    "location_id": analysis_row["location_id"],
                    "split": analysis_row["split"],
                    "class_name": class_name,
                    "crop_name": DETAIL_TASK_CROPS[task],
                    "source_crop_path": source_row["crop_path"],
                }
            )

    index_rows: list[dict[str, str]] = []
    summary: dict[str, Any] = {
        "created_at": _utc_now(),
        "label_status": "gemini_consensus_candidates_not_ground_truth",
        "source_analysis_sha256": _sha256(analysis / "summary.json"),
        "source_detail_index_sha256": _sha256(source / "index.csv"),
        "source_analysis_frames": int(analysis_summary["frames"]),
        "image_size": int(
            json.loads((source / "summary.json").read_text(encoding="utf-8"))[
                "image_size"
            ]
        ),
        "seed": seed,
        "tasks": {},
    }
    crops_dir = output / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        classes = DETAIL_TASK_CLASSES[task]
        task_rows = [row for row in candidates if row["task"] == task]
        split_rows = {
            split: [row for row in task_rows if row["split"] == split]
            for split in ("train", "validation", "test")
        }
        for split, rows in split_rows.items():
            missing = set(classes) - {row["class_name"] for row in rows}
            if missing:
                raise GeminiLabelingError(
                    f"{task}/{split} lacks strict candidates for: {', '.join(sorted(missing))}"
                )
        balanced_train = _balanced_candidate_rows(
            split_rows["train"],
            classes,
            seed=seed,
            max_minority_repeats=max_minority_repeats,
        )
        balanced_validation = _balanced_candidate_rows(
            split_rows["validation"],
            classes,
            seed=seed
            + int(hashlib.sha256(f"{task}:validation".encode()).hexdigest()[:8], 16),
            max_minority_repeats=max_minority_repeats,
        )
        yolo_rows = {
            "train": balanced_train,
            "val": balanced_validation,
            "test": [(row, 0) for row in split_rows["test"]],
        }
        output_crop_by_frame: dict[str, Path] = {}
        for yolo_split, rows in yolo_rows.items():
            for row, repeat in rows:
                output_crop = output_crop_by_frame.get(row["frame_id"])
                if output_crop is None:
                    source_crop = (source / row["source_crop_path"]).resolve()
                    if (
                        not source_crop.is_relative_to(source)
                        or not source_crop.is_file()
                    ):
                        raise GeminiLabelingError(
                            f"unsafe source crop: {row['source_crop_path']}"
                        )
                    output_crop = crops_dir / f"{task}-{row['frame_id']}.jpg"
                    if not output_crop.exists():
                        output_crop.symlink_to(
                            os.path.relpath(source_crop, output_crop.parent)
                        )
                    output_crop_by_frame[row["frame_id"]] = output_crop
                alias = f"{task}-{row['frame_id']}__{repeat}.jpg"
                link = output / task / yolo_split / row["class_name"] / alias
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(os.path.relpath(output_crop, link.parent))
        for row in task_rows:
            output_crop = output_crop_by_frame[row["frame_id"]]
            index_rows.append(
                {
                    **{field: row[field] for field in DETAIL_INDEX_FIELDS if field != "crop_path"},
                    "crop_path": output_crop.relative_to(output).as_posix(),
                }
            )

        task_summary: dict[str, Any] = {
            "classes": list(classes),
            "crop": DETAIL_TASK_CROPS[task],
            "candidate_coverage": len(task_rows) / int(analysis_summary["frames"]),
            "model_selection_validation_classes": dict(
                sorted(
                    Counter(row["class_name"] for row, _ in balanced_validation).items()
                )
            ),
            "splits": {},
        }
        for split, rows in split_rows.items():
            locations: dict[str, Any] = {}
            for location in sorted({row["location_id"] for row in rows}):
                location_rows = [
                    row for row in rows if row["location_id"] == location
                ]
                counts = dict(
                    sorted(Counter(row["class_name"] for row in location_rows).items())
                )
                locations[location] = {
                    "eligible": len(location_rows),
                    "detected": len(location_rows),
                    "coverage": 1.0,
                    "eligible_classes": counts,
                    "detected_classes": counts,
                }
            counts = dict(sorted(Counter(row["class_name"] for row in rows).items()))
            task_summary["splits"][split] = {
                "eligible": len(rows),
                "detected": len(rows),
                "coverage": 1.0,
                "eligible_classes": counts,
                "detected_classes": counts,
                "locations": locations,
            }
        summary["tasks"][task] = task_summary

    with (output / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(index_rows)
    summary["unique_crops"] = len({row["crop_path"] for row in index_rows})
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def prepare_gemini_adult_dataset(
    analysis_dir: Path,
    source_manifest: Path,
    frames_dir: Path,
    output_dir: Path,
    *,
    image_size: int = 320,
    seed: int = 20260730,
    max_minority_repeats: int = 1,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build a diagnostic full-scene adult dataset from strict teacher consensus."""

    from .adult_training import (
        ADULT_CLASSES,
        ADULT_DATASET_MARKER,
        ADULT_INDEX_FIELDS,
        ADULT_TASK,
        _adult_crop_path,
        _balanced_rows_by_location,
        _materialize_scene,
    )
    from .detail_training import DetailExample
    from .yolo_training import _safe_generated_directory

    if image_size < 96:
        raise ValueError("adult image size must be at least 96 pixels")
    if max_minority_repeats < 1:
        raise ValueError("max_minority_repeats must be positive")
    analysis = analysis_dir.resolve()
    source_manifest = source_manifest.resolve()
    frames = frames_dir.resolve()
    if not (analysis / GEMINI_ANALYSIS_MARKER).is_file():
        raise ValueError(f"not a marked Gemini analysis directory: {analysis}")
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)
    output = _safe_generated_directory(
        output_dir,
        ADULT_DATASET_MARKER,
        overwrite=overwrite,
    )
    source_frames = _read_source_frames(source_manifest)
    analysis_summary = json.loads(
        (analysis / "summary.json").read_text(encoding="utf-8")
    )
    analysis_rows = [
        json.loads(line)
        for line in (analysis / "consensus-candidates.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    examples: list[DetailExample] = []
    for row in analysis_rows:
        class_name = row["fields"]["adult_presence"]["candidate"]
        if class_name not in ADULT_CLASSES:
            continue
        source = source_frames.get(row["frame_id"])
        if source is None:
            raise GeminiLabelingError(f"missing source frame: {row['frame_id']}")
        examples.append(
            DetailExample(
                task=ADULT_TASK,
                sample_id=row["frame_id"],
                frame_id=row["frame_id"],
                captured_at=row["captured_at"],
                location_id=row["location_id"],
                relative_path=source["relative_path"],
                sha256=source["sha256"],
                crop_name="scene",
                crop_x0=0.0,
                crop_y0=0.0,
                crop_x1=1.0,
                crop_y1=1.0,
                class_name=class_name,
                split=row["split"],
            )
        )
    for split in ("train", "validation", "test"):
        values = {
            example.class_name for example in examples if example.split == split
        }
        if values != set(ADULT_CLASSES):
            raise GeminiLabelingError(
                f"adult/{split} lacks strict candidates for: "
                f"{', '.join(sorted(set(ADULT_CLASSES) - values))}"
            )

    training = [example for example in examples if example.split == "train"]
    validation = [example for example in examples if example.split == "validation"]
    test = [example for example in examples if example.split == "test"]
    balanced_train = _balanced_rows_by_location(
        training,
        seed=seed,
        max_minority_repeats=max_minority_repeats,
    )
    balanced_validation = _balanced_rows_by_location(
        validation,
        seed=seed + 1,
        max_minority_repeats=max_minority_repeats,
    )
    folder_rows = {
        "train": balanced_train,
        "val": balanced_validation,
        "test": [(example, 0) for example in test],
    }
    crop_paths: dict[str, Path] = {}
    for yolo_split, rows in folder_rows.items():
        for example, repeat in rows:
            crop_path = crop_paths.setdefault(
                example.sample_id,
                _adult_crop_path(output, example),
            )
            _materialize_scene(
                example,
                frames,
                crop_path,
                image_size=image_size,
            )
            link = (
                output
                / ADULT_TASK
                / yolo_split
                / example.class_name
                / f"{crop_path.stem}__{repeat}.jpg"
            )
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(os.path.relpath(crop_path, link.parent))

    indexed = {
        example.sample_id: example
        for example, repeat in balanced_train
        if repeat == 0
    }
    indexed.update(
        {
            example.sample_id: example
            for example in examples
            if example.split in {"validation", "test"}
        }
    )
    index_rows: list[dict[str, Any]] = []
    for example in sorted(
        indexed.values(),
        key=lambda item: (item.captured_at, item.sample_id),
    ):
        crop_path = crop_paths.setdefault(
            example.sample_id,
            _adult_crop_path(output, example),
        )
        _materialize_scene(
            example,
            frames,
            crop_path,
            image_size=image_size,
        )
        index_rows.append(
            {
                "sample_id": example.sample_id,
                "frame_id": example.frame_id,
                "captured_at": example.captured_at,
                "location_id": example.location_id,
                "relative_path": example.relative_path,
                "split": example.split,
                "target": int(example.class_name == "yes"),
                "class_name": example.class_name,
                "crop_path": crop_path.relative_to(output).as_posix(),
            }
        )
    with (output / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ADULT_INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(index_rows)

    def counts(items: list[DetailExample]) -> dict[str, int]:
        return dict(sorted(Counter(item.class_name for item in items).items()))

    def balanced_counts(
        items: list[tuple[DetailExample, int]],
    ) -> dict[str, int]:
        return dict(sorted(Counter(item.class_name for item, _ in items).items()))

    summary = {
        "created_at": _utc_now(),
        "image_size": image_size,
        "seed": seed,
        "label_status": "gemini_consensus_candidates_not_ground_truth",
        "label_contract": (
            "Any visible adult or unmistakable adult body part in the full scene; "
            "teacher consensus is diagnostic weak supervision, not held-out truth."
        ),
        "candidate_coverage": len(examples) / int(analysis_summary["frames"]),
        "natural_counts": {
            split: counts(
                [example for example in examples if example.split == split]
            )
            for split in ("train", "validation", "test")
        },
        "model_selection_counts": {
            "train": balanced_counts(balanced_train),
            "validation": balanced_counts(balanced_validation),
        },
        "training_after_consensus": counts(training),
        "source_analysis_sha256": _sha256(analysis / "summary.json"),
        "source_manifest_sha256": _sha256(source_manifest),
        "split": "inherited location/day groups",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def run_gemini_label_pilot(
    pilot_dir: Path,
    *,
    api_key: str,
    models: tuple[str, ...] = GEMINI_DEFAULT_MODELS,
    max_workers: int = 2,
    timeout_seconds: float = 120,
    retries: int = 3,
    limit: int | None = None,
) -> dict[str, Any]:
    if not api_key:
        raise ValueError("Gemini API key is required")
    if not models or len(set(models)) != len(models):
        raise ValueError("models must be a non-empty unique tuple")
    if max_workers < 1 or max_workers > 8:
        raise ValueError("max_workers must be between 1 and 8")
    pilot = pilot_dir.resolve()
    if not (pilot / GEMINI_PILOT_MARKER).is_file():
        raise ValueError(f"not a marked Gemini pilot directory: {pilot}")
    manifest_path = pilot / "manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    results_path = pilot / "results.jsonl"
    completed = _completed_pairs(results_path)
    work = [
        (row, model)
        for row in rows
        for model in models
        if (row["frame_id"], model) not in completed
    ]
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError(
            "Gemini labeling requires the optional dependency: uv sync --extra gemini"
        ) from exc
    lock = threading.Lock()
    records: list[dict[str, Any]] = []
    errors = 0

    def execute(row: dict[str, str], model: str) -> dict[str, Any]:
        board = _safe_child(pilot, row["board_path"])
        started_at = _utc_now()
        try:
            with httpx.Client() as client:
                label, usage, latency_ms = _request_one(
                    client,
                    api_key=api_key,
                    model=model,
                    image_bytes=board.read_bytes(),
                    timeout_seconds=timeout_seconds,
                    retries=retries,
                )
            return {
                "frame_id": row["frame_id"],
                "model": model,
                "prompt_version": GEMINI_PROMPT_VERSION,
                "thinking_level": GEMINI_THINKING_LEVEL,
                "requested_at": started_at,
                "board_sha256": row["board_sha256"],
                "latency_ms": round(latency_ms, 3),
                "usage": usage,
                "estimated_standard_cost_usd": _estimated_standard_cost(model, usage),
                "pricing_snapshot_date": GEMINI_PRICING_SNAPSHOT_DATE,
                "label": label,
                "error": None,
            }
        except Exception as exc:
            return {
                "frame_id": row["frame_id"],
                "model": model,
                "prompt_version": GEMINI_PROMPT_VERSION,
                "thinking_level": GEMINI_THINKING_LEVEL,
                "requested_at": started_at,
                "board_sha256": row["board_sha256"],
                "latency_ms": None,
                "usage": {},
                "estimated_standard_cost_usd": None,
                "pricing_snapshot_date": GEMINI_PRICING_SNAPSHOT_DATE,
                "label": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(execute, row, model): (row["frame_id"], model)
            for row, model in work
        }
        for future in as_completed(futures):
            record = future.result()
            if record["error"]:
                errors += 1
            serialized = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            with lock:
                with results_path.open("a", encoding="utf-8") as handle:
                    handle.write(serialized + "\n")
                records.append(record)
    latest_records = list(_latest_results(results_path).values())
    successful = [
        record
        for record in latest_records
        if record.get("label") is not None and record.get("error") is None
    ]
    current_errors = [record for record in latest_records if record.get("error")]
    attempts_total = 0
    if results_path.is_file():
        with results_path.open(encoding="utf-8") as handle:
            attempts_total = sum(1 for line in handle if line.strip())
    cost = sum(
        float(record["estimated_standard_cost_usd"])
        for record in successful
        if record.get("estimated_standard_cost_usd") is not None
    )
    latency_values = sorted(
        float(record["latency_ms"])
        for record in successful
        if record.get("latency_ms") is not None
    )
    summary = {
        "updated_at": _utc_now(),
        "models": list(models),
        "frames_selected": len(rows),
        "requested_now": len(work),
        "attempts_total": attempts_total,
        "completed_total": len(latest_records),
        "successful_total": len(successful),
        "errors_now": errors,
        "errors_current": len(current_errors),
        "estimated_standard_cost_usd": round(cost, 6),
        "pricing_snapshot_date": GEMINI_PRICING_SNAPSHOT_DATE,
        "latency_ms": {
            "median": (
                round(latency_values[len(latency_values) // 2], 3)
                if latency_values
                else None
            ),
            "maximum": round(max(latency_values), 3) if latency_values else None,
        },
    }
    (pilot / "run-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
