from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sqlite3
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .dataset import FrameExample, read_manifest
from .yolo_training import (
    PoseHeadConfig,
    YoloTrainingConfig,
    _disable_ultralytics_integrations,
    _mouth_rect,
    _pose_body_rect,
    _pose_head_rect,
    _safe_frame_path,
    _safe_generated_directory,
    _select_pose_candidate,
)

DETAIL_DATASET_MARKER = ".baby-monitor-yolo-detail-dataset"
DETAIL_ARTIFACT_MARKER = ".baby-monitor-yolo-detail-artifacts"
DETAIL_TASK_CLASSES = {
    "head_side": ("back", "left", "right"),
    "body_position": ("back", "belly", "side"),
    "mouth_open": ("no", "yes"),
}
DETAIL_TASK_CROPS = {
    "head_side": "head",
    "body_position": "body",
    "mouth_open": "mouth",
}
DETAIL_PRECISION_TARGETS = {
    "head_side": 0.90,
    "body_position": 0.90,
    "mouth_open": 0.95,
}
DETAIL_DEPLOYMENT_REQUIREMENTS = {
    "head_side": {
        "selective_accuracy": 0.90,
        "coverage": 0.40,
        "class_precision": 0.85,
        "minimum_class_decisions": 10,
        "minimum_location_decisions": 10,
        "location_selective_accuracy": 0.85,
    },
    "body_position": {
        "selective_accuracy": 0.90,
        "coverage": 0.25,
        "class_precision": 0.85,
        "minimum_class_decisions": 3,
        "minimum_location_decisions": 10,
        "location_selective_accuracy": 0.85,
    },
    "mouth_open": {
        "selective_accuracy": 0.95,
        "coverage": 0.20,
        "class_precision": 0.90,
        "minimum_class_decisions": 3,
        "minimum_location_decisions": 10,
        "location_selective_accuracy": 0.90,
    },
}
DETAIL_INDEX_FIELDS = (
    "task",
    "sample_id",
    "frame_id",
    "captured_at",
    "location_id",
    "split",
    "class_name",
    "crop_name",
    "crop_path",
)


@dataclass(frozen=True, slots=True)
class DetailExample:
    task: str
    sample_id: str
    frame_id: str
    captured_at: str
    location_id: str
    relative_path: str
    sha256: str
    crop_name: str
    crop_x0: float
    crop_y0: float
    crop_x1: float
    crop_y1: float
    class_name: str
    split: str = ""

    @property
    def day(self) -> str:
        return self.captured_at[:10]

    @property
    def roi(self) -> tuple[float, float, float, float]:
        return self.crop_x0, self.crop_y0, self.crop_x1, self.crop_y1


@dataclass(frozen=True, slots=True)
class PoseRects:
    head: tuple[float, float, float, float] | None
    body: tuple[float, float, float, float] | None
    mouth: tuple[float, float, float, float] | None


def normalize_body_position(value: object) -> str | None:
    """Map free-form teacher descriptions to a small observable posture contract."""

    normalized = " ".join(
        str(value or "").strip().lower().replace("_", " ").replace("-", " ").split()
    )
    if not normalized or normalized in {"unknown", "none", "n/a", "not applicable"}:
        return None
    if any(token in normalized for token in ("held", "adult", "cradled", "lap", "upright")):
        return None
    if (
        normalized == "back"
        or "supine" in normalized
        or "lying on back" in normalized
        or normalized == "on back"
    ):
        return "back"
    if normalized == "belly" or "prone" in normalized or "stomach" in normalized:
        return "belly"
    explicit_side = {
        "side left",
        "side right",
        "lying on side",
        "side lying",
        "lying on left side",
        "lying on right side",
    }
    if normalized in explicit_side:
        return "side"
    if normalized.startswith("lying on side") and not any(
        token in normalized for token in ("back", "stomach", "adult")
    ):
        return "side"
    return None


def _mouth_is_observable(label: dict[str, Any]) -> bool:
    if (
        label.get("baby_present") is not True
        or label.get("face_visible") != "yes"
        or label.get("pacifier") != "no"
        or label.get("mouth_open") not in {"yes", "no"}
    ):
        return False
    evidence = " ".join(
        (
            str(label.get("description", "")),
            *(str(tag) for tag in label.get("tags", [])),
        )
    ).lower()
    return not any(token in evidence for token in ("bottle", "feeding", "nursing"))


def _label_classes(label: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    head_side = str(label.get("head_side", "unknown")).strip().lower()
    if (
        label.get("baby_present") is True
        and label.get("face_visible") == "yes"
        and head_side in DETAIL_TASK_CLASSES["head_side"]
    ):
        result["head_side"] = head_side
    body_position = normalize_body_position(label.get("body_position"))
    if label.get("baby_present") is True and body_position is not None:
        result["body_position"] = body_position
    if _mouth_is_observable(label):
        result["mouth_open"] = str(label["mouth_open"])
    return result


def _read_labels(database: Path) -> dict[str, dict[str, Any]]:
    if not database.is_file():
        raise FileNotFoundError(database)
    uri = f"{database.resolve().as_uri()}?mode=ro"
    labels: dict[str, dict[str, Any]] = {}
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """SELECT id, provider, label_json
               FROM frames
               WHERE image_available = 1
                 AND relative_path IS NOT NULL
                 AND label_json IS NOT NULL"""
        ).fetchall()
    for row in rows:
        if str(row["provider"] or "").strip().lower() == "yolo":
            continue
        try:
            label = json.loads(row["label_json"])
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(label, dict):
            labels[str(row["id"])] = label
    return labels


def load_detail_examples(
    source_manifest: Path,
    database: Path,
) -> tuple[tuple[DetailExample, ...], dict[str, int]]:
    """Join private structured labels to the already audited positive ROI rows."""

    manifest = read_manifest(source_manifest)
    labels = _read_labels(database)
    skipped: Counter[str] = Counter()
    positive_rows: dict[str, FrameExample] = {}
    for row in manifest:
        if not row.presence_target:
            continue
        existing = positive_rows.get(row.frame_id)
        if existing is not None and existing.sample_id != row.sample_id:
            raise ValueError(f"frame {row.frame_id} has multiple positive ROI rows")
        positive_rows[row.frame_id] = row

    examples: list[DetailExample] = []
    for frame_id, row in positive_rows.items():
        label = labels.get(frame_id)
        if label is None:
            skipped["missing_or_local_label"] += 1
            continue
        classes = _label_classes(label)
        if not classes:
            skipped["no_observable_detail_label"] += 1
            continue
        for task, class_name in classes.items():
            examples.append(
                DetailExample(
                    task=task,
                    sample_id=row.sample_id,
                    frame_id=row.frame_id,
                    captured_at=row.captured_at,
                    location_id=row.location_id,
                    relative_path=row.relative_path,
                    sha256=row.sha256,
                    crop_name=row.crop_name,
                    crop_x0=row.crop_x0,
                    crop_y0=row.crop_y0,
                    crop_x1=row.crop_x1,
                    crop_y1=row.crop_y1,
                    class_name=class_name,
                )
            )
    return tuple(examples), dict(sorted(skipped.items()))


def _day_selection_score(
    counts: Counter[str],
    desired: dict[str, float],
    *,
    selected_days: int,
    target_days: int,
) -> float:
    progress = selected_days / target_days
    score = 0.0
    for class_name, target in desired.items():
        expected = target * progress
        score += abs(counts[class_name] - expected) / max(target, 1.0)
        if counts[class_name] > target:
            score += (counts[class_name] - target) / max(target, 1.0)
    return score


def _choose_group_days(
    day_counts: dict[str, Counter[str]],
    count: int,
    *,
    seed: int,
) -> set[str]:
    if count < 1 or count >= len(day_counts):
        raise ValueError("evaluation day count must leave at least one training day")
    total: Counter[str] = Counter()
    for values in day_counts.values():
        total.update(values)
    desired = {
        class_name: value * count / len(day_counts)
        for class_name, value in total.items()
    }
    selected: set[str] = set()
    selected_counts: Counter[str] = Counter()
    while len(selected) < count:
        candidates: list[tuple[float, str, Counter[str]]] = []
        for day, values in day_counts.items():
            if day in selected:
                continue
            proposed = selected_counts + values
            score = _day_selection_score(
                proposed,
                desired,
                selected_days=len(selected) + 1,
                target_days=count,
            )
            jitter = int(hashlib.sha256(f"{seed}:{day}".encode()).hexdigest()[:8], 16) / 2**32
            candidates.append((score + jitter * 1e-6, day, proposed))
        _, chosen, proposed = min(candidates, key=lambda item: (item[0], item[1]))
        selected.add(chosen)
        selected_counts = proposed
    return selected


def stratified_group_split(
    examples: tuple[DetailExample, ...],
    *,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 20260730,
) -> tuple[DetailExample, ...]:
    """Stratify rare labels while keeping every task/location/day in one split."""

    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must leave a training split")
    split_by_group: dict[tuple[str, str, str], str] = {}
    grouped: dict[tuple[str, str], dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    for example in examples:
        grouped[(example.task, example.location_id)][example.day][example.class_name] += 1

    for (task, location), day_counts in sorted(grouped.items()):
        if len(day_counts) < 3:
            raise ValueError(f"{task}/{location} needs at least three capture days")
        test_days_count = max(1, round(len(day_counts) * test_fraction))
        validation_days_count = max(1, round(len(day_counts) * validation_fraction))
        while test_days_count + validation_days_count >= len(day_counts):
            if test_days_count >= validation_days_count and test_days_count > 1:
                test_days_count -= 1
            elif validation_days_count > 1:
                validation_days_count -= 1
            else:
                raise ValueError(f"{task}/{location} has too few grouped days")
        test_days = _choose_group_days(
            day_counts,
            test_days_count,
            seed=seed + int(hashlib.sha256(f"{task}:{location}:test".encode()).hexdigest()[:8], 16),
        )
        remaining = {day: values for day, values in day_counts.items() if day not in test_days}
        validation_days = _choose_group_days(
            remaining,
            validation_days_count,
            seed=seed + int(hashlib.sha256(f"{task}:{location}:validation".encode()).hexdigest()[:8], 16),
        )
        for day in day_counts:
            split = "test" if day in test_days else "validation" if day in validation_days else "train"
            split_by_group[(task, location, day)] = split
    return tuple(
        replace(
            example,
            split=split_by_group[(example.task, example.location_id, example.day)],
        )
        for example in examples
    )


def _temporal_consensus_ids(
    examples: Iterable[DetailExample],
    *,
    maximum_gap_seconds: float = 20 * 60,
) -> set[tuple[str, str]]:
    grouped: dict[tuple[str, str, str], list[DetailExample]] = defaultdict(list)
    for example in examples:
        if example.split == "train":
            grouped[(example.task, example.location_id, example.crop_name)].append(example)
    accepted: set[tuple[str, str]] = set()
    for group in grouped.values():
        ordered = sorted(group, key=lambda item: (item.captured_at, item.frame_id))
        timestamps = [
            datetime.fromisoformat(item.captured_at.replace("Z", "+00:00"))
            for item in ordered
        ]
        for index, example in enumerate(ordered):
            for neighbor_index in (index - 1, index + 1):
                if not 0 <= neighbor_index < len(ordered):
                    continue
                neighbor = ordered[neighbor_index]
                gap = abs((timestamps[index] - timestamps[neighbor_index]).total_seconds())
                if gap <= maximum_gap_seconds and example.class_name == neighbor.class_name:
                    accepted.add((example.task, example.sample_id))
                    break
    return accepted


def _locate_pose_rects(
    examples: tuple[DetailExample, ...],
    frames_dir: Path,
    pose_model: Path,
    *,
    config: PoseHeadConfig,
) -> dict[str, PoseRects]:
    from ultralytics import YOLO

    _disable_ultralytics_integrations()
    if not pose_model.is_file():
        raise FileNotFoundError(pose_model)
    unique: dict[str, DetailExample] = {}
    for example in examples:
        existing = unique.get(example.sample_id)
        if existing is not None and existing.roi != example.roi:
            raise ValueError(f"sample {example.sample_id} has inconsistent ROI geometry")
        unique[example.sample_id] = example
    model = YOLO(str(pose_model.resolve()))
    items = sorted(unique.items())
    located: dict[str, PoseRects] = {}
    for start in range(0, len(items), config.batch_size):
        batch = items[start : start + config.batch_size]
        paths = [str(_safe_frame_path(frames_dir, example.relative_path)) for _, example in batch]
        results = model.predict(
            paths,
            imgsz=config.image_size,
            device=config.device,
            batch=config.batch_size,
            conf=min(0.15, config.detection_confidence),
            verbose=False,
        )
        for (sample_id, example), result in zip(batch, results, strict=True):
            if result.boxes is None or result.keypoints is None or result.keypoints.conf is None:
                located[sample_id] = PoseRects(None, None, None)
                continue
            candidate = _select_pose_candidate(
                boxes=result.boxes.xyxyn.detach().cpu().tolist(),
                box_confidences=result.boxes.conf.detach().cpu().tolist(),
                keypoints=result.keypoints.xyn.detach().cpu().tolist(),
                keypoint_confidences=result.keypoints.conf.detach().cpu().tolist(),
                roi=example.roi,
                config=config,
            )
            if candidate is None:
                located[sample_id] = PoseRects(None, None, None)
                continue
            box, points, point_confidences = candidate
            height, width = result.orig_shape
            head = _pose_head_rect(
                box=box,
                points=points,
                point_confidences=point_confidences,
                width=width,
                height=height,
                config=config,
            )
            located[sample_id] = PoseRects(
                head=head,
                body=_pose_body_rect(box=box, roi=example.roi),
                mouth=_mouth_rect(head) if head is not None else None,
            )
        completed = min(start + config.batch_size, len(items))
        if completed == len(items) or completed % 256 == 0:
            print(
                f"Localized {completed}/{len(items)} unique detail frames",
                file=sys.stderr,
                flush=True,
            )
    return located


def _crop_path(output: Path, example: DetailExample, crop_name: str) -> Path:
    digest = hashlib.sha256(f"{example.sample_id}:{crop_name}".encode()).hexdigest()[:24]
    return output / "crops" / f"{example.frame_id}-{digest}-{crop_name}.jpg"


def _materialize_crop(
    example: DetailExample,
    frames_dir: Path,
    path: Path,
    rect: tuple[float, float, float, float],
    *,
    image_size: int,
    fit: bool,
) -> None:
    if path.is_file():
        return
    source = _safe_frame_path(frames_dir, example.relative_path)
    with Image.open(source) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
        width, height = image.size
        x0, y0, x1, y1 = rect
        cropped = image.crop(
            (
                round(x0 * width),
                round(y0 * height),
                round(x1 * width),
                round(y1 * height),
            )
        )
        if fit:
            prepared = ImageOps.fit(
                cropped,
                (image_size, image_size),
                method=Image.Resampling.LANCZOS,
            )
        else:
            prepared = ImageOps.pad(
                cropped,
                (image_size, image_size),
                method=Image.Resampling.LANCZOS,
                color=(114, 114, 114),
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        prepared.save(path, format="JPEG", quality=92, optimize=True)


def _balanced_rows(
    examples: list[DetailExample],
    classes: tuple[str, ...],
    *,
    seed: int,
    max_minority_repeats: int,
) -> list[tuple[DetailExample, int]]:
    by_class: dict[str, list[DetailExample]] = {}
    for class_name in classes:
        candidates = [example for example in examples if example.class_name == class_name]
        by_class[class_name] = sorted(
            candidates,
            key=lambda item: hashlib.sha256(
                f"{seed}:{item.task}:{class_name}:{item.sample_id}".encode()
            ).hexdigest(),
        )
    if any(not rows for rows in by_class.values()):
        missing = [class_name for class_name, rows in by_class.items() if not rows]
        raise ValueError(f"training split has no examples for: {', '.join(missing)}")
    desired = min(
        max(len(rows) for rows in by_class.values()),
        min(len(rows) for rows in by_class.values()) * max_minority_repeats,
    )
    balanced: list[tuple[DetailExample, int]] = []
    for rows in by_class.values():
        selected = rows[:desired]
        for index in range(desired):
            balanced.append((selected[index % len(selected)], index // len(selected)))
    return sorted(
        balanced,
        key=lambda item: hashlib.sha256(
            f"{seed}:{item[0].task}:{item[0].sample_id}:{item[1]}".encode()
        ).hexdigest(),
    )


def rebalance_detail_validation(
    dataset_dir: Path,
    *,
    seed: int = 20260730,
    max_minority_repeats: int = 8,
) -> dict[str, dict[str, int]]:
    """Balance model-selection folders without changing natural holdout rows.

    Ultralytics chooses ``best.pt`` from aggregate validation accuracy. A
    naturally imbalanced validation folder can therefore prefer a majority-only
    classifier even though the later deployment gate checks every class. The
    CSV index remains untouched so calibration and final metrics still use the
    complete natural validation distribution.
    """

    dataset = dataset_dir.resolve()
    if not (dataset / DETAIL_DATASET_MARKER).is_file():
        raise ValueError(f"{dataset} is not a Baby Monitor detail dataset")
    if max_minority_repeats < 1:
        raise ValueError("max minority repeats must be positive")
    with (dataset / "index.csv").open(newline="", encoding="utf-8") as handle:
        index = list(csv.DictReader(handle))

    result: dict[str, dict[str, int]] = {}
    for task, classes in DETAIL_TASK_CLASSES.items():
        examples: list[DetailExample] = []
        crop_paths: dict[str, Path] = {}
        for row in index:
            if row["task"] != task or row["split"] != "validation":
                continue
            sample_id = row["sample_id"]
            crop_path = (dataset / row["crop_path"]).resolve()
            if not crop_path.is_relative_to(dataset) or not crop_path.is_file():
                raise ValueError(f"unsafe or missing detail crop: {row['crop_path']}")
            crop_paths[sample_id] = crop_path
            examples.append(
                DetailExample(
                    task=task,
                    sample_id=sample_id,
                    frame_id=row["frame_id"],
                    captured_at=row["captured_at"],
                    location_id=row["location_id"],
                    relative_path="",
                    sha256="",
                    crop_name=row["crop_name"],
                    crop_x0=0,
                    crop_y0=0,
                    crop_x1=1,
                    crop_y1=1,
                    class_name=row["class_name"],
                    split="validation",
                )
            )
        balanced = _balanced_rows(
            examples,
            classes,
            seed=seed + int(hashlib.sha256(f"{task}:validation".encode()).hexdigest()[:8], 16),
            max_minority_repeats=max_minority_repeats,
        )
        validation_dir = (dataset / task / "val").resolve()
        if (
            not validation_dir.is_relative_to(dataset)
            or validation_dir == dataset
            or validation_dir.name != "val"
        ):
            raise ValueError(f"unsafe validation directory: {validation_dir}")
        if validation_dir.exists():
            shutil.rmtree(validation_dir)
        for example, repeat in balanced:
            crop_path = crop_paths[example.sample_id]
            alias = f"{crop_path.stem}__{repeat}.jpg"
            link = validation_dir / example.class_name / alias
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(os.path.relpath(crop_path, link.parent))
        result[task] = dict(
            sorted(Counter(example.class_name for example, _ in balanced).items())
        )

    summary_path = dataset / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["model_selection_validation"] = {
        "strategy": "class-balanced deterministic sampling",
        "max_minority_repeats": max_minority_repeats,
        "counts": result,
        "natural_distribution_preserved_in": "index.csv",
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _counts_by_split(
    examples: Iterable[DetailExample],
) -> dict[str, dict[str, dict[str, int]]]:
    counts: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for example in examples:
        counts[example.task][example.split][example.class_name] += 1
    return {
        task: {
            split: dict(sorted(values.items()))
            for split, values in sorted(split_counts.items())
        }
        for task, split_counts in sorted(counts.items())
    }


def prepare_detail_dataset(
    source_manifest: Path,
    database: Path,
    frames_dir: Path,
    pose_model: Path,
    output_dir: Path,
    *,
    image_size: int = 320,
    pose_config: PoseHeadConfig | None = None,
    seed: int = 20260730,
    max_minority_repeats: int = 8,
    temporal_consensus: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build private head, body, and mouth datasets with grouped holdouts."""

    if image_size < 96:
        raise ValueError("detail image size must be at least 96 pixels")
    output = _safe_generated_directory(
        output_dir,
        DETAIL_DATASET_MARKER,
        overwrite=overwrite,
    )
    loaded, skipped = load_detail_examples(source_manifest, database)
    examples = stratified_group_split(loaded, seed=seed)
    consensus = _temporal_consensus_ids(examples)
    pose_config = pose_config or PoseHeadConfig()
    rects = _locate_pose_rects(
        examples,
        frames_dir.resolve(),
        pose_model.resolve(),
        config=pose_config,
    )
    index_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "database_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
        "image_size": image_size,
        "seed": seed,
        "skipped": skipped,
        "source_counts": _counts_by_split(examples),
        "tasks": {},
        "pose": {
            "image_size": pose_config.image_size,
            "detection_confidence": pose_config.detection_confidence,
            "nose_confidence": pose_config.nose_confidence,
            "head_keypoint_confidence": pose_config.head_keypoint_confidence,
            "model_sha256": hashlib.sha256(pose_model.read_bytes()).hexdigest(),
        },
    }
    crop_cache: dict[tuple[str, str], Path] = {}

    for task, classes in DETAIL_TASK_CLASSES.items():
        task_examples = [example for example in examples if example.task == task]
        crop_name = DETAIL_TASK_CROPS[task]
        detected = [
            example
            for example in task_examples
            if getattr(rects[example.sample_id], crop_name) is not None
        ]
        training = [example for example in detected if example.split == "train"]
        excluded_inconsistent = 0
        # Mouth positives are genuinely rare, often isolated events. They have
        # already passed the stricter visibility, pacifier, and feeding filters;
        # requiring a nearby duplicate would collapse the class to a few scenes.
        if temporal_consensus and task != "mouth_open":
            excluded_inconsistent = sum(
                (example.task, example.sample_id) not in consensus for example in training
            )
            training = [
                example
                for example in training
                if (example.task, example.sample_id) in consensus
            ]
        balanced = _balanced_rows(
            training,
            classes,
            seed=seed,
            max_minority_repeats=max_minority_repeats,
        )
        validation = [
            example
            for example in detected
            if example.split == "validation"
        ]
        balanced_validation = _balanced_rows(
            validation,
            classes,
            seed=seed
            + int(hashlib.sha256(f"{task}:validation".encode()).hexdigest()[:8], 16),
            max_minority_repeats=max_minority_repeats,
        )
        task_summary: dict[str, Any] = {
            "classes": list(classes),
            "crop": crop_name,
            "excluded_inconsistent_train_labels": excluded_inconsistent,
            "model_selection_validation_classes": dict(
                sorted(
                    Counter(
                        example.class_name
                        for example, _ in balanced_validation
                    ).items()
                )
            ),
            "splits": {},
        }
        for split in ("train", "validation", "test"):
            source_split = [example for example in task_examples if example.split == split]
            detected_split = [example for example in detected if example.split == split]
            by_location: dict[str, Any] = {}
            for location in sorted({example.location_id for example in source_split}):
                eligible_location = [
                    example for example in source_split if example.location_id == location
                ]
                detected_location = [
                    example for example in detected_split if example.location_id == location
                ]
                by_location[location] = {
                    "eligible": len(eligible_location),
                    "detected": len(detected_location),
                    "coverage": len(detected_location) / len(eligible_location),
                    "eligible_classes": dict(
                        sorted(Counter(item.class_name for item in eligible_location).items())
                    ),
                    "detected_classes": dict(
                        sorted(Counter(item.class_name for item in detected_location).items())
                    ),
                }
            task_summary["splits"][split] = {
                "eligible": len(source_split),
                "detected": len(detected_split),
                "coverage": len(detected_split) / len(source_split),
                "eligible_classes": dict(
                    sorted(Counter(item.class_name for item in source_split).items())
                ),
                "detected_classes": dict(
                    sorted(Counter(item.class_name for item in detected_split).items())
                ),
                "locations": by_location,
            }

        for source_split, yolo_split in (
            ("train", "train"),
            ("validation", "val"),
            ("test", "test"),
        ):
            rows = {
                "train": balanced,
                "validation": balanced_validation,
                "test": [
                    (example, 0)
                    for example in detected
                    if example.split == "test"
                ],
            }[source_split]
            for example, repeat in rows:
                key = (example.sample_id, crop_name)
                crop_path = crop_cache.get(key)
                if crop_path is None:
                    crop_path = _crop_path(output, example, crop_name)
                    rect = getattr(rects[example.sample_id], crop_name)
                    if rect is None:
                        raise RuntimeError("detected example unexpectedly has no crop")
                    _materialize_crop(
                        example,
                        frames_dir,
                        crop_path,
                        rect,
                        image_size=image_size,
                        fit=crop_name in {"head", "mouth"},
                    )
                    crop_cache[key] = crop_path
                alias = f"{crop_path.stem}__{repeat}.jpg"
                link = output / task / yolo_split / example.class_name / alias
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(os.path.relpath(crop_path, link.parent))
        indexed_examples = {
            (example.sample_id, example.split): example
            for example, repeat in balanced
            if repeat == 0
        }
        indexed_examples.update(
            {
                (example.sample_id, example.split): example
                for example in detected
                if example.split in {"validation", "test"}
            }
        )
        for example in indexed_examples.values():
            key = (example.sample_id, crop_name)
            crop_path = crop_cache.get(key)
            if crop_path is None:
                crop_path = _crop_path(output, example, crop_name)
                rect = getattr(rects[example.sample_id], crop_name)
                if rect is None:
                    raise RuntimeError("detected example unexpectedly has no crop")
                _materialize_crop(
                    example,
                    frames_dir,
                    crop_path,
                    rect,
                    image_size=image_size,
                    fit=crop_name in {"head", "mouth"},
                )
                crop_cache[key] = crop_path
            index_rows.append(
                {
                    "task": task,
                    "sample_id": example.sample_id,
                    "frame_id": example.frame_id,
                    "captured_at": example.captured_at,
                    "location_id": example.location_id,
                    "split": example.split,
                    "class_name": example.class_name,
                    "crop_name": crop_name,
                    "crop_path": str(crop_path.relative_to(output)),
                }
            )
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


def train_detail_classifiers(
    dataset_dir: Path,
    output_dir: Path,
    *,
    config: YoloTrainingConfig | None = None,
    overwrite: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Fine-tune one compact classifier per secondary visible attribute."""

    from ultralytics import YOLO
    from ultralytics import __version__ as ultralytics_version

    _disable_ultralytics_integrations()
    config = config or YoloTrainingConfig(seed=20260730)
    dataset = dataset_dir.resolve()
    if not (dataset / DETAIL_DATASET_MARKER).is_file():
        raise ValueError(f"{dataset} is not a Baby Monitor detail dataset")
    if overwrite and resume:
        raise ValueError("overwrite and resume cannot be used together")
    if resume:
        output = output_dir.resolve()
        if not (output / DETAIL_ARTIFACT_MARKER).is_file():
            raise ValueError(f"{output} is not a resumable detail artifact")
    else:
        output = _safe_generated_directory(
            output_dir,
            DETAIL_ARTIFACT_MARKER,
            overwrite=overwrite,
        )
    plan = {
        "config": asdict(config),
        "dataset_summary": json.loads((dataset / "summary.json").read_text(encoding="utf-8")),
    }
    plan_path = output / "training-plan.json"
    if resume:
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("resume configuration or detail dataset changed")
    else:
        plan_path.write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    models_dir = output / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    trained: dict[str, Any] = {}
    for task in DETAIL_TASK_CLASSES:
        target = models_dir / f"{task}.pt"
        interrupted_best = output / "runs" / task / "weights" / "best.pt"
        if resume and target.is_file():
            trained[task] = {
                "path": str(target.relative_to(output)),
                "bytes": target.stat().st_size,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
            continue
        if resume and interrupted_best.is_file():
            # Ultralytics writes best.pt after every improving validation
            # epoch. Reuse that durable checkpoint when a long MPS run was
            # deliberately stopped after throughput degraded.
            shutil.copy2(interrupted_best, target)
            trained[task] = {
                "path": str(target.relative_to(output)),
                "bytes": target.stat().st_size,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
            continue
        model = YOLO(config.base_model)
        model.train(
            data=str(dataset / task),
            epochs=config.epochs,
            imgsz=config.image_size,
            batch=config.batch_size,
            device=config.device,
            workers=config.workers,
            patience=config.patience,
            lr0=config.learning_rate,
            optimizer="AdamW",
            cos_lr=True,
            dropout=0.1,
            auto_augment="randaugment",
            erasing=0.1,
            # Mirroring a directional label without also swapping left/right
            # teaches the head classifier contradictory supervision.
            fliplr=0.0 if task == "head_side" else 0.5,
            scale=0.1,
            amp=False,
            deterministic=True,
            seed=config.seed,
            project=str(output / "runs"),
            name=task,
            exist_ok=True,
            plots=False,
            verbose=True,
        )
        best = output / "runs" / task / "weights" / "best.pt"
        if not best.is_file():
            raise RuntimeError(f"Ultralytics did not produce {best}")
        shutil.copy2(best, target)
        trained[task] = {
            "path": str(target.relative_to(output)),
            "bytes": target.stat().st_size,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "ultralytics_version": ultralytics_version,
        **plan,
        "models": trained,
    }
    (output / "training.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _predict_probabilities(
    model_path: Path,
    rows: list[dict[str, str]],
    dataset: Path,
    *,
    image_size: int,
    device: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    from ultralytics import YOLO

    _disable_ultralytics_integrations()
    model = YOLO(str(model_path.resolve()))
    scored: list[dict[str, Any]] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        paths = [str((dataset / row["crop_path"]).resolve()) for row in batch]
        results = model.predict(
            paths,
            imgsz=image_size,
            device=device,
            batch=batch_size,
            verbose=False,
        )
        for row, result in zip(batch, results, strict=True):
            if result.probs is None:
                raise RuntimeError("YOLO detail classifier returned no probabilities")
            probabilities = {
                str(name): float(result.probs.data[index].detach().cpu())
                for index, name in result.names.items()
            }
            ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
            scored.append(
                {
                    **row,
                    "probabilities": probabilities,
                    "top_class": ordered[0][0],
                    "top_probability": ordered[0][1],
                    "margin": ordered[0][1] - ordered[1][1],
                }
            )
    return scored


def _candidate_values(values: Iterable[float], *, floor: float, steps: int = 30) -> list[float]:
    observed = sorted(set(round(float(value), 6) for value in values))
    if len(observed) <= steps:
        return sorted(set((floor, *observed)))
    quantiles = [
        observed[round(index * (len(observed) - 1) / (steps - 1))]
        for index in range(steps)
    ]
    return sorted(set((floor, *quantiles)))


def _calibrate_class_rules(
    rows: list[dict[str, Any]],
    classes: tuple[str, ...],
    *,
    precision_target: float,
    minimum_decisions: int = 3,
) -> dict[str, dict[str, Any]]:
    rules: dict[str, dict[str, Any]] = {}
    for class_name in classes:
        candidates = [row for row in rows if row["top_class"] == class_name]
        probability_values = _candidate_values(
            (row["top_probability"] for row in candidates),
            floor=1 / len(classes),
        )
        margin_values = _candidate_values(
            (row["margin"] for row in candidates),
            floor=0.0,
            steps=20,
        )
        choices: list[tuple[int, float, float, float]] = []
        for probability in probability_values:
            for margin in margin_values:
                selected = [
                    row
                    for row in candidates
                    if row["top_probability"] >= probability and row["margin"] >= margin
                ]
                if len(selected) < minimum_decisions:
                    continue
                correct = sum(row["class_name"] == class_name for row in selected)
                precision = correct / len(selected)
                if precision >= precision_target:
                    choices.append((len(selected), precision, probability, margin))
        if not choices:
            rules[class_name] = {
                "enabled": False,
                "probability": 1.0,
                "margin": 1.0,
                "validation_decisions": 0,
                "validation_precision": None,
            }
            continue
        decisions, precision, probability, margin = max(
            choices,
            key=lambda item: (item[0], item[1], -item[2], -item[3]),
        )
        rules[class_name] = {
            "enabled": True,
            "probability": probability,
            "margin": margin,
            "validation_decisions": decisions,
            "validation_precision": precision,
        }
    return rules


def _decision(row: dict[str, Any], rules: dict[str, dict[str, Any]]) -> str:
    rule = rules[row["top_class"]]
    if (
        rule["enabled"]
        and row["top_probability"] >= rule["probability"]
        and row["margin"] >= rule["margin"]
    ):
        return str(row["top_class"])
    return "unknown"


def _selective_metrics(
    rows: list[dict[str, Any]],
    localization: dict[str, Any],
    classes: tuple[str, ...],
) -> dict[str, Any]:
    eligible = int(localization["eligible"])
    eligible_classes = {
        class_name: int(localization["eligible_classes"].get(class_name, 0))
        for class_name in classes
    }
    decisions = [row for row in rows if row["decision"] != "unknown"]
    correct = sum(row["decision"] == row["class_name"] for row in decisions)
    result: dict[str, Any] = {
        "eligible": eligible,
        "localized": len(rows),
        "decisions": len(decisions),
        "abstained": eligible - len(decisions),
        "coverage": len(decisions) / eligible if eligible else 0.0,
        "selective_accuracy": correct / len(decisions) if decisions else None,
        "classes": {},
    }
    for class_name in classes:
        predicted = [row for row in decisions if row["decision"] == class_name]
        true_positive = sum(row["class_name"] == class_name for row in predicted)
        result["classes"][class_name] = {
            "eligible": eligible_classes[class_name],
            "predicted": len(predicted),
            "true_positive": true_positive,
            "precision": true_positive / len(predicted) if predicted else None,
            "recall": (
                true_positive / eligible_classes[class_name]
                if eligible_classes[class_name]
                else None
            ),
        }
    return result


def _task_gate(task: str, test_report: dict[str, Any]) -> dict[str, Any]:
    requirements = DETAIL_DEPLOYMENT_REQUIREMENTS[task]
    overall = test_report["overall"]
    overall_checks = {
        "selective_accuracy": (
            overall["selective_accuracy"] is not None
            and overall["selective_accuracy"] >= requirements["selective_accuracy"]
        ),
        "coverage": overall["coverage"] >= requirements["coverage"],
    }
    class_checks = {
        class_name: {
            "minimum_decisions": (
                values["predicted"] >= requirements["minimum_class_decisions"]
            ),
            "precision": (
                values["precision"] is not None
                and values["precision"] >= requirements["class_precision"]
            ),
        }
        for class_name, values in overall["classes"].items()
    }
    location_checks: dict[str, dict[str, Any]] = {}
    for location, metrics in test_report.items():
        if location == "overall":
            continue
        checks = {
            "minimum_decisions": (
                metrics["decisions"] >= requirements["minimum_location_decisions"]
            ),
            "selective_accuracy": (
                metrics["selective_accuracy"] is not None
                and metrics["selective_accuracy"]
                >= requirements["location_selective_accuracy"]
            ),
            "coverage": metrics["coverage"] >= requirements["coverage"],
        }
        location_checks[location] = {
            "passed": all(checks.values()),
            "checks": checks,
        }
    passed = (
        all(overall_checks.values())
        and all(all(checks.values()) for checks in class_checks.values())
        and all(item["passed"] for item in location_checks.values())
    )
    return {
        "passed": passed,
        "requirements": requirements,
        "overall_checks": overall_checks,
        "class_checks": class_checks,
        "locations": location_checks,
    }


def _detail_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Secondary YOLO camera features",
        "",
        "All thresholds were selected on grouped validation days. Test days were not "
        "used for training or calibration.",
        "",
        "| Task | Test accuracy | Runtime coverage | Gate |",
        "| --- | ---: | ---: | --- |",
    ]
    for task in DETAIL_TASK_CLASSES:
        metrics = report["tasks"][task]["test"]["overall"]
        accuracy = metrics["selective_accuracy"]
        lines.append(
            f"| {task} | "
            f"{'n/a' if accuracy is None else f'{accuracy * 100:.1f}%'} | "
            f"{metrics['coverage'] * 100:.1f}% | "
            f"{'pass' if report['tasks'][task]['gate']['passed'] else 'fail'} |"
        )
    lines.extend(
        (
            "",
            "Historical model labels remain weak supervision. A passing automated gate "
            "still requires visual review of decisive held-out predictions before deployment.",
            "",
        )
    )
    return "\n".join(lines)


def evaluate_detail_classifiers(
    dataset_dir: Path,
    artifact_dir: Path,
    *,
    device: str = "cpu",
    batch_size: int = 8,
) -> dict[str, Any]:
    """Calibrate abstention rules and score untouched grouped detail holdouts."""

    dataset = dataset_dir.resolve()
    artifacts = artifact_dir.resolve()
    training = json.loads((artifacts / "training.json").read_text(encoding="utf-8"))
    summary = json.loads((dataset / "summary.json").read_text(encoding="utf-8"))
    with (dataset / "index.csv").open(newline="", encoding="utf-8") as handle:
        index = list(csv.DictReader(handle))
    image_size = int(training["config"]["image_size"])
    prediction_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "split": "task/location/day grouped and rare-label stratified",
        "tasks": {},
    }
    metadata_tasks: dict[str, Any] = {}

    for task, classes in DETAIL_TASK_CLASSES.items():
        task_rows = [
            row
            for row in index
            if row["task"] == task and row["split"] in {"validation", "test"}
        ]
        scored = _predict_probabilities(
            artifacts / training["models"][task]["path"],
            task_rows,
            dataset,
            image_size=image_size,
            device=device,
            batch_size=batch_size,
        )
        validation_rows = [row for row in scored if row["split"] == "validation"]
        rules = _calibrate_class_rules(
            validation_rows,
            classes,
            precision_target=DETAIL_PRECISION_TARGETS[task],
        )
        for row in scored:
            row["decision"] = _decision(row, rules)
            prediction_rows.append(row)

        task_report: dict[str, Any] = {
            "classes": list(classes),
            "crop": DETAIL_TASK_CROPS[task],
            "thresholds": {"overall": rules},
        }
        for split in ("validation", "test"):
            split_rows = [row for row in scored if row["split"] == split]
            split_report: dict[str, Any] = {}
            localization = summary["tasks"][task]["splits"][split]
            split_report["overall"] = _selective_metrics(
                split_rows,
                localization,
                classes,
            )
            for location, location_summary in localization["locations"].items():
                split_report[location] = _selective_metrics(
                    [row for row in split_rows if row["location_id"] == location],
                    location_summary,
                    classes,
                )
            task_report[split] = split_report
        task_report["gate"] = _task_gate(task, task_report["test"])
        report["tasks"][task] = task_report

        model = training["models"][task]
        metadata_tasks[task] = {
            "path": model["path"],
            "bytes": model["bytes"],
            "sha256": model["sha256"],
            "classes": list(classes),
            "crop": DETAIL_TASK_CROPS[task],
            "thresholds": {"overall": rules},
        }

    report["automated_gate_passed"] = all(
        task["gate"]["passed"] for task in report["tasks"].values()
    )
    report["decision"] = (
        "awaiting_manual_review"
        if report["automated_gate_passed"]
        else "not_accepted"
    )
    serializable_rows: list[dict[str, Any]] = []
    for row in prediction_rows:
        serializable_rows.append(
            {
                **{key: value for key, value in row.items() if key != "probabilities"},
                **{
                    f"probability_{class_name}": probability
                    for class_name, probability in row["probabilities"].items()
                },
            }
        )
    fields = sorted({key for row in serializable_rows for key in row})
    with (artifacts / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(serializable_rows)
    (artifacts / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (artifacts / "report.md").write_text(_detail_markdown(report), encoding="utf-8")
    (artifacts / "metadata-fragment.json").write_text(
        json.dumps({"tasks": metadata_tasks}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def assemble_reviewed_detail_artifact(
    base_artifact: Path,
    detail_artifact: Path,
    output_dir: Path,
    *,
    reviewed_tasks: tuple[str, ...],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Add explicitly reviewed, gate-passing detail models to an accepted artifact."""

    base = base_artifact.resolve()
    details = detail_artifact.resolve()
    if not reviewed_tasks:
        raise ValueError("at least one manually reviewed detail task is required")
    if not (base / ".baby-monitor-yolo-artifacts").is_file():
        raise ValueError(f"{base} is not a Baby Monitor YOLO artifact")
    if not (details / DETAIL_ARTIFACT_MARKER).is_file():
        raise ValueError(f"{details} is not a Baby Monitor detail artifact")
    report = json.loads((details / "report.json").read_text(encoding="utf-8"))
    fragment = json.loads((details / "metadata-fragment.json").read_text(encoding="utf-8"))
    requested = set(reviewed_tasks)
    unknown = requested - set(DETAIL_TASK_CLASSES)
    if unknown:
        raise ValueError(f"unknown reviewed detail tasks: {', '.join(sorted(unknown))}")
    failed = {
        task
        for task in requested
        if not report["tasks"][task]["gate"]["passed"]
    }
    if failed:
        raise ValueError(
            "refusing to assemble tasks that failed the automated gate: "
            + ", ".join(sorted(failed))
        )

    output = _safe_generated_directory(
        output_dir,
        ".baby-monitor-yolo-artifacts",
        overwrite=overwrite,
    )
    for source in base.iterdir():
        if source.name == ".baby-monitor-yolo-artifacts":
            continue
        target = output / source.name
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)

    metadata = json.loads((base / "metadata.json").read_text(encoding="utf-8"))
    for task in sorted(requested):
        task_metadata = dict(fragment["tasks"][task])
        source_model = (details / task_metadata["path"]).resolve()
        if not source_model.is_relative_to(details) or not source_model.is_file():
            raise ValueError(f"detail model for {task} is unavailable")
        target_model = output / "models" / f"{task}.pt"
        shutil.copy2(source_model, target_model)
        task_metadata["path"] = str(target_model.relative_to(output))
        task_metadata["bytes"] = target_model.stat().st_size
        task_metadata["sha256"] = hashlib.sha256(target_model.read_bytes()).hexdigest()
        metadata["tasks"][task] = task_metadata
    metadata["format"] = (
        "Ultralytics YOLO26 classification ensemble with pose-localized "
        "head, body, and mouth features"
    )
    fingerprint_payload = {
        "base_version": metadata["version"],
        "tasks": {
            task: metadata["tasks"][task]
            for task in sorted(metadata["tasks"])
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    metadata["version"] = f"baby-monitor-yolo-private-{fingerprint}"
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    base_report_path = output / "report.json"
    base_report = json.loads(base_report_path.read_text(encoding="utf-8"))
    labels = {
        "head_side": "head orientation: back, left, or right",
        "body_position": "body position: back, belly, or side",
        "mouth_open": "visibly open versus closed mouth",
    }
    base_report["artifact_version"] = metadata["version"]
    base_report["evaluation"]["secondary_features"] = {
        task: report["tasks"][task] for task in sorted(requested)
    }
    outputs = base_report["scope"]["outputs"]
    for task in sorted(requested):
        if labels[task] not in outputs:
            outputs.append(labels[task])
    base_report["scope"]["not_validated"] = [
        value
        for value in base_report["scope"]["not_validated"]
        if not any(task.replace("_", " ") in value for task in requested)
    ]
    base_report["scope"]["unknown_policy"] = (
        "Ambiguous detail, failed localization, or a class below its reviewed "
        "probability and margin thresholds returns unknown."
    )
    base_report["limitations"].append(
        "Secondary-feature labels began as Gemini weak supervision; deployment "
        "acceptance combines grouped holdouts, selective thresholds, and explicit "
        "visual review."
    )
    (output / "report.json").write_text(
        json.dumps(base_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(details / "report.json", output / "secondary-report.json")
    shutil.copy2(details / "report.md", output / "secondary-report.md")
    return {
        "version": metadata["version"],
        "reviewed_tasks": sorted(requested),
        "output_dir": str(output),
    }
