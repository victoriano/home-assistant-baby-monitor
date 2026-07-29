from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
from bisect import bisect_left
from collections import Counter
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .dataset import read_manifest
from .detail_training import (
    DetailExample,
    _balanced_rows,
    _read_labels,
    _temporal_consensus_ids,
    stratified_group_split,
)
from .metrics import selective_binary_metrics
from .yolo_training import (
    YoloTrainingConfig,
    _calibrate_by_location,
    _disable_ultralytics_integrations,
    _safe_frame_path,
    _safe_generated_directory,
)

ADULT_DATASET_MARKER = ".baby-monitor-yolo-adult-dataset"
ADULT_ARTIFACT_MARKER = ".baby-monitor-yolo-adult-artifacts"
ADULT_TASK = "adult_presence"
ADULT_CLASSES = ("no", "yes")
ADULT_INDEX_FIELDS = (
    "sample_id",
    "frame_id",
    "captured_at",
    "location_id",
    "relative_path",
    "split",
    "target",
    "class_name",
    "crop_path",
)
ADULT_POSITIVE_TAGS = {
    "adult",
    "adult_hand",
    "adult_holding_baby",
    "adult_in_bed",
    "adult_interaction",
    "adult_nearby",
    "adult_present",
    "adult_visible",
    "baby_in_adult_arms",
    "baby_with_adult",
    "held_by_adult",
}
ADULT_NEGATIVE_PHRASES = (
    "no adult is visible",
    "no adult visible",
    "neither a baby nor an adult",
    "neither baby nor adult",
    "without an adult",
)
ADULT_EVIDENCE = re.compile(
    r"\b(?:an|the)\s+adult\b|"
    r"\badult(?:'s)?\s+(?:arm|body|hand|head|leg|person)\b|"
    r"\badult\s+(?:appears|holds|is|lies|rests|sits|sleeps|stands|touches|who)\b|"
    r"\b(?:against|beside|by|held by|next to|with)\s+(?:an?|the)\s+adult\b",
)
ADULT_BED_ONLY = re.compile(r"\badult(?:-sized| size)?\s+bed\b")
ADULT_REQUIREMENTS = {
    "selective_accuracy": 0.95,
    "coverage": 0.40,
    "positive_precision": 0.90,
    "negative_precision": 0.95,
    "positive_recall": 0.35,
    "negative_recall": 0.50,
}


def adult_presence_target(label: dict[str, Any]) -> int | None:
    """Extract visible-adult weak supervision without treating a bed as a person."""

    explicit_presence = label.get("adult_present")
    if explicit_presence in {"yes", "no"}:
        return int(explicit_presence == "yes")
    if explicit_presence == "unknown":
        return None
    explicit_count = label.get("adult_count")
    if type(explicit_count) is int and explicit_count >= 0:
        return int(explicit_count > 0)
    normalized_tags = {
        "_".join(
            str(tag).strip().lower().replace("-", " ").replace("_", " ").split()
        )
        for tag in label.get("tags", [])
    }
    if normalized_tags & {"image_unusable", "image_uncertain"}:
        return None
    description = " ".join(str(label.get("description", "")).lower().split())
    if any(phrase in description for phrase in ADULT_NEGATIVE_PHRASES):
        return 0
    if normalized_tags & ADULT_POSITIVE_TAGS:
        return 1
    without_bed = ADULT_BED_ONLY.sub("bed", description)
    return int(bool(ADULT_EVIDENCE.search(without_bed)))


def load_adult_examples(
    source_manifest: Path,
    database: Path,
) -> tuple[DetailExample, ...]:
    """Join every audited full frame to adult-presence weak labels."""

    labels = _read_labels(database)
    unique_frames = {}
    for row in read_manifest(source_manifest):
        existing = unique_frames.get(row.frame_id)
        if existing is None:
            unique_frames[row.frame_id] = row
        elif (
            existing.location_id != row.location_id
            or existing.relative_path != row.relative_path
            or existing.sha256 != row.sha256
        ):
            raise ValueError(f"frame {row.frame_id} has inconsistent source metadata")

    examples: list[DetailExample] = []
    for frame_id, row in unique_frames.items():
        label = labels.get(frame_id)
        target = adult_presence_target(label) if label is not None else None
        if target is None:
            continue
        examples.append(
            DetailExample(
                task=ADULT_TASK,
                sample_id=f"{frame_id}:scene",
                frame_id=frame_id,
                captured_at=row.captured_at,
                location_id=row.location_id,
                relative_path=row.relative_path,
                sha256=row.sha256,
                crop_name="scene",
                crop_x0=0,
                crop_y0=0,
                crop_x1=1,
                crop_y1=1,
                class_name=ADULT_CLASSES[target],
            )
        )
    return tuple(examples)


def repair_bracketed_adult_positives(
    examples: tuple[DetailExample, ...],
    *,
    maximum_gap_seconds: float = 6 * 60,
) -> tuple[tuple[DetailExample, ...], int]:
    """Repair only omissions bracketed by nearby positive frames."""

    if maximum_gap_seconds <= 0:
        raise ValueError("maximum adult propagation gap must be positive")
    positives: dict[str, list[datetime]] = {}
    for example in examples:
        if example.class_name != "yes":
            continue
        positives.setdefault(example.location_id, []).append(
            datetime.fromisoformat(example.captured_at.replace("Z", "+00:00"))
        )
    for timestamps in positives.values():
        timestamps.sort()

    repaired: list[DetailExample] = []
    changed = 0
    for example in examples:
        if example.class_name == "yes":
            repaired.append(example)
            continue
        timestamp = datetime.fromisoformat(
            example.captured_at.replace("Z", "+00:00")
        )
        candidates = positives.get(example.location_id, [])
        index = bisect_left(candidates, timestamp)
        previous = candidates[index - 1] if index > 0 else None
        following = candidates[index] if index < len(candidates) else None
        bracketed = (
            previous is not None
            and following is not None
            and 0 < (timestamp - previous).total_seconds() <= maximum_gap_seconds
            and 0 < (following - timestamp).total_seconds() <= maximum_gap_seconds
        )
        if bracketed:
            repaired.append(replace(example, class_name="yes"))
            changed += 1
        else:
            repaired.append(example)
    return tuple(repaired), changed


def _adult_crop_path(output: Path, example: DetailExample) -> Path:
    digest = hashlib.sha256(example.sample_id.encode()).hexdigest()[:24]
    return output / "crops" / f"{example.frame_id}-{digest}-scene.jpg"


def _balanced_rows_by_location(
    examples: list[DetailExample],
    *,
    seed: int,
    max_minority_repeats: int,
) -> list[tuple[DetailExample, int]]:
    """Balance yes/no independently so camera identity is not a label shortcut."""

    balanced: list[tuple[DetailExample, int]] = []
    for location in sorted({example.location_id for example in examples}):
        location_seed = seed + int(
            hashlib.sha256(location.encode()).hexdigest()[:8],
            16,
        )
        balanced.extend(
            _balanced_rows(
                [
                    example
                    for example in examples
                    if example.location_id == location
                ],
                ADULT_CLASSES,
                seed=location_seed,
                max_minority_repeats=max_minority_repeats,
            )
        )
    return sorted(
        balanced,
        key=lambda item: hashlib.sha256(
            f"{seed}:{item[0].location_id}:{item[0].sample_id}:{item[1]}".encode()
        ).hexdigest(),
    )


def _materialize_scene(
    example: DetailExample,
    frames_dir: Path,
    target: Path,
    *,
    image_size: int,
) -> None:
    if target.is_file():
        return
    with Image.open(_safe_frame_path(frames_dir, example.relative_path)) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
        prepared = ImageOps.pad(
            image,
            (image_size, image_size),
            method=Image.Resampling.LANCZOS,
            color=(114, 114, 114),
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        prepared.save(target, format="JPEG", quality=92, optimize=True)


def prepare_adult_dataset(
    source_manifest: Path,
    database: Path,
    frames_dir: Path,
    output_dir: Path,
    *,
    image_size: int = 320,
    seed: int = 20260730,
    max_minority_repeats: int = 1,
    temporal_consensus: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build full-scene adult-presence data with location/day holdouts."""

    output = _safe_generated_directory(
        output_dir,
        ADULT_DATASET_MARKER,
        overwrite=overwrite,
    )
    loaded = load_adult_examples(source_manifest, database)
    enriched, bracketed_positive_repairs = repair_bracketed_adult_positives(loaded)
    natural = stratified_group_split(enriched, seed=seed)
    consensus = _temporal_consensus_ids(natural)
    training = [item for item in natural if item.split == "train"]
    if temporal_consensus:
        training = [
            item
            for item in training
            if (item.task, item.sample_id) in consensus
        ]
    validation = [item for item in natural if item.split == "validation"]
    test = [item for item in natural if item.split == "test"]
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
        "test": [(item, 0) for item in test],
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
                frames_dir.resolve(),
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
        item.sample_id: item
        for item, repeat in balanced_train
        if repeat == 0
    }
    indexed.update(
        {
            item.sample_id: item
            for item in natural
            if item.split in {"validation", "test"}
        }
    )
    index_rows: list[dict[str, Any]] = []
    for example in sorted(indexed.values(), key=lambda item: (item.captured_at, item.sample_id)):
        crop_path = crop_paths.setdefault(
            example.sample_id,
            _adult_crop_path(output, example),
        )
        _materialize_scene(
            example,
            frames_dir.resolve(),
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
                "crop_path": str(crop_path.relative_to(output)),
            }
        )
    with (output / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ADULT_INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(index_rows)

    def counts(items: list[DetailExample]) -> dict[str, int]:
        return dict(sorted(Counter(item.class_name for item in items).items()))

    def location_counts(
        rows: list[tuple[DetailExample, int]],
    ) -> dict[str, dict[str, int]]:
        return {
            location: counts(
                [
                    item
                    for item, _ in rows
                    if item.location_id == location
                ]
            )
            for location in sorted({item.location_id for item, _ in rows})
        }

    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "image_size": image_size,
        "seed": seed,
        "label_contract": (
            "Visible adult evidence after removing the non-person phrase 'adult bed'; "
            "only a negative bracketed by positives within six minutes on both sides "
            "repairs an intermittent teacher omission; "
            "historical descriptions remain weak supervision."
        ),
        "bracketed_positive_repairs": bracketed_positive_repairs,
        "split": "location/day grouped",
        "natural_counts": {
            split: counts([item for item in natural if item.split == split])
            for split in ("train", "validation", "test")
        },
        "model_selection_counts": {
            "train": dict(
                sorted(Counter(item.class_name for item, _ in balanced_train).items())
            ),
            "validation": dict(
                sorted(
                    Counter(item.class_name for item, _ in balanced_validation).items()
                )
            ),
        },
        "model_selection_location_counts": {
            "train": location_counts(balanced_train),
            "validation": location_counts(balanced_validation),
        },
        "temporal_consensus": temporal_consensus,
        "training_after_consensus": counts(training),
        "source_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
        "database_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def train_adult_classifier(
    dataset_dir: Path,
    output_dir: Path,
    *,
    config: YoloTrainingConfig | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Fine-tune the full-scene adult-presence classifier."""

    from ultralytics import YOLO
    from ultralytics import __version__ as ultralytics_version

    _disable_ultralytics_integrations()
    config = config or YoloTrainingConfig(seed=20260730)
    dataset = dataset_dir.resolve()
    if not (dataset / ADULT_DATASET_MARKER).is_file():
        raise ValueError(f"{dataset} is not a Baby Monitor adult dataset")
    output = _safe_generated_directory(
        output_dir,
        ADULT_ARTIFACT_MARKER,
        overwrite=overwrite,
    )
    model = YOLO(config.base_model)
    model.train(
        data=str(dataset / ADULT_TASK),
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
        fliplr=0.5,
        scale=0.1,
        amp=False,
        deterministic=True,
        seed=config.seed,
        project=str(output / "runs"),
        name=ADULT_TASK,
        exist_ok=True,
        plots=False,
        verbose=True,
    )
    best = output / "runs" / ADULT_TASK / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"Ultralytics did not produce {best}")
    target = output / "models" / f"{ADULT_TASK}.pt"
    target.parent.mkdir()
    shutil.copy2(best, target)
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "ultralytics_version": ultralytics_version,
        "config": asdict(config),
        "dataset_summary": json.loads(
            (dataset / "summary.json").read_text(encoding="utf-8")
        ),
        "model": {
            "path": str(target.relative_to(output)),
            "bytes": target.stat().st_size,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        },
    }
    (output / "training.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _adult_scores(
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
        results = model.predict(
            [str((dataset / row["crop_path"]).resolve()) for row in batch],
            imgsz=image_size,
            device=device,
            batch=batch_size,
            verbose=False,
        )
        for row, result in zip(batch, results, strict=True):
            if result.probs is None:
                raise RuntimeError("adult classifier returned no probabilities")
            positive_index = next(
                index for index, name in result.names.items() if name == "yes"
            )
            scored.append(
                {
                    **row,
                    "target": int(row["target"]),
                    "score": float(
                        result.probs.data[positive_index].detach().cpu()
                    ),
                }
            )
    return scored


def _metrics_by_location(
    rows: list[dict[str, Any]],
    thresholds: dict[str, dict[str, float]],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for location in ("overall", *sorted({row["location_id"] for row in rows})):
        subset = (
            rows
            if location == "overall"
            else [row for row in rows if row["location_id"] == location]
        )
        selected = thresholds.get(location, thresholds["overall"])
        report[location] = selective_binary_metrics(
            np.asarray([row["target"] for row in subset], dtype=np.int8),
            np.asarray([row["score"] for row in subset], dtype=np.float32),
            selected["negative"],
            selected["positive"],
        )
    return report


def _adult_gate(test: dict[str, Any]) -> dict[str, Any]:
    locations: dict[str, Any] = {}
    for location, metrics in test.items():
        checks = {
            metric: metrics.get(metric) is not None
            and float(metrics[metric]) >= minimum
            for metric, minimum in ADULT_REQUIREMENTS.items()
        }
        locations[location] = {"passed": all(checks.values()), "checks": checks}
    return {
        "passed": all(item["passed"] for item in locations.values()),
        "requirements": ADULT_REQUIREMENTS,
        "locations": locations,
    }


def _adult_markdown(report: dict[str, Any]) -> str:
    def percentage(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.1%}"

    lines = [
        "# Visible-adult YOLO report",
        "",
        "Thresholds were calibrated on location/day-grouped validation frames. "
        "The table below uses untouched natural test days.",
        "",
        "| Domain | Samples | Decisions | Coverage | Selective accuracy | "
        "Positive precision | Negative precision | Gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for location, metrics in report["test"].items():
        gate = report["gate"]["locations"][location]
        lines.append(
            f"| {location} | {metrics['samples']} | {metrics['decisions']} | "
            f"{percentage(metrics['coverage'])} | "
            f"{percentage(metrics['selective_accuracy'])} | "
            f"{percentage(metrics['positive_precision'])} | "
            f"{percentage(metrics['negative_precision'])} | "
            f"{'pass' if gate['passed'] else 'fail'} |"
        )
    lines.extend(
        (
            "",
            "Historical AI descriptions are weak supervision. A passing gate still "
            "requires visual review of decisive disagreements and positive predictions.",
            "",
        )
    )
    return "\n".join(lines)


def evaluate_adult_classifier(
    dataset_dir: Path,
    artifact_dir: Path,
    *,
    device: str = "cpu",
    batch_size: int = 16,
) -> dict[str, Any]:
    """Calibrate on validation days and score untouched adult-presence days."""

    dataset = dataset_dir.resolve()
    artifacts = artifact_dir.resolve()
    training = json.loads((artifacts / "training.json").read_text(encoding="utf-8"))
    with (dataset / "index.csv").open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["split"] in {"validation", "test"}
        ]
    scored = _adult_scores(
        artifacts / training["model"]["path"],
        rows,
        dataset,
        image_size=int(training["config"]["image_size"]),
        device=device,
        batch_size=batch_size,
    )
    validation = [row for row in scored if row["split"] == "validation"]
    test_rows = [row for row in scored if row["split"] == "test"]
    thresholds, validation_metrics = _calibrate_by_location(validation)
    test_metrics = _metrics_by_location(test_rows, thresholds)
    gate = _adult_gate(test_metrics)
    for row in scored:
        selected = thresholds.get(row["location_id"], thresholds["overall"])
        row["decision"] = (
            "yes"
            if row["score"] >= selected["positive"]
            else "no"
            if row["score"] <= selected["negative"]
            else "unknown"
        )
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "validation": validation_metrics,
        "test": test_metrics,
        "thresholds": thresholds,
        "gate": gate,
        "decision": "awaiting_manual_review" if gate["passed"] else "not_accepted",
    }
    with (artifacts / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [*ADULT_INDEX_FIELDS, "score", "decision"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in scored:
            writer.writerow(
                {
                    **{field: row[field] for field in ADULT_INDEX_FIELDS},
                    "score": row["score"],
                    "decision": row["decision"],
                }
            )
    with (artifacts / "high_confidence_errors.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        fields = [*ADULT_INDEX_FIELDS, "score", "decision"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                **{field: row[field] for field in ADULT_INDEX_FIELDS},
                "score": row["score"],
                "decision": row["decision"],
            }
            for row in scored
            if row["split"] == "test"
            and row["decision"] != "unknown"
            and row["decision"] != row["class_name"]
        )
    (artifacts / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (artifacts / "report.md").write_text(
        _adult_markdown(report),
        encoding="utf-8",
    )
    metadata = {
        "tasks": {
            ADULT_TASK: {
                **training["model"],
                "positive_class": "yes",
                "threshold": 0.5,
                "thresholds": thresholds,
                "crop": "scene",
            }
        }
    }
    (artifacts / "metadata-fragment.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def assemble_reviewed_adult_artifact(
    base_artifact: Path,
    adult_artifact: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Add a manually reviewed, gate-passing adult classifier to an artifact."""

    base = base_artifact.resolve()
    adult = adult_artifact.resolve()
    if not (base / ".baby-monitor-yolo-artifacts").is_file():
        raise ValueError(f"{base} is not a Baby Monitor YOLO artifact")
    if not (adult / ADULT_ARTIFACT_MARKER).is_file():
        raise ValueError(f"{adult} is not a Baby Monitor adult artifact")
    report = json.loads((adult / "report.json").read_text(encoding="utf-8"))
    if not report["gate"]["passed"]:
        raise ValueError("refusing to assemble an adult classifier that failed its gate")
    fragment = json.loads(
        (adult / "metadata-fragment.json").read_text(encoding="utf-8")
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
    task = dict(fragment["tasks"][ADULT_TASK])
    source_model = (adult / task["path"]).resolve()
    target_model = output / "models" / f"{ADULT_TASK}.pt"
    shutil.copy2(source_model, target_model)
    task["path"] = str(target_model.relative_to(output))
    task["bytes"] = target_model.stat().st_size
    task["sha256"] = hashlib.sha256(target_model.read_bytes()).hexdigest()
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["tasks"][ADULT_TASK] = task
    metadata["format"] = (
        "Ultralytics YOLO26 classification ensemble with pose-localized "
        "baby details and full-scene adult presence"
    )
    fingerprint = hashlib.sha256(
        json.dumps(metadata["tasks"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    metadata["version"] = f"baby-monitor-yolo-private-{fingerprint}"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    base_report_path = output / "report.json"
    base_report = json.loads(base_report_path.read_text(encoding="utf-8"))
    base_report["artifact_version"] = metadata["version"]
    base_report["evaluation"]["adult_presence"] = report
    outputs = base_report["scope"]["outputs"]
    if "visible adult presence" not in outputs:
        outputs.append("visible adult presence")
    not_validated = base_report["scope"]["not_validated"]
    if "exact adult counts above one" not in not_validated:
        not_validated.append("exact adult counts above one")
    base_report["scope"]["unknown_policy"] = (
        "Ambiguous detail, failed localization, or an adult-presence score "
        "between its reviewed thresholds returns unknown; there is no cloud fallback."
    )
    base_report["limitations"].append(
        "Adult-presence reference labels began as historical AI descriptions; "
        "grouped holdouts and visual review reduce but do not eliminate teacher error."
    )
    base_report_path.write_text(
        json.dumps(base_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(adult / "report.json", output / "adult-presence-report.json")
    shutil.copy2(adult / "report.md", output / "adult-presence-report.md")
    shutil.copy2(
        adult / "high_confidence_errors.csv",
        output / "adult-presence-high-confidence-errors.csv",
    )
    return {
        "version": metadata["version"],
        "output_dir": str(output),
        "task": ADULT_TASK,
    }
