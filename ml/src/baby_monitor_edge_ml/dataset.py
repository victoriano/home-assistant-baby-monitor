from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from . import TASKS

MANIFEST_FIELDS = (
    "sample_id",
    "frame_id",
    "captured_at",
    "location_id",
    "relative_path",
    "sha256",
    "provider",
    "model",
    "crop_name",
    "crop_x0",
    "crop_y0",
    "crop_x1",
    "crop_y1",
    "split",
    "presence_target",
    "presence_mask",
    "awake_target",
    "awake_mask",
    "pacifier_target",
    "pacifier_mask",
    "confidence",
    "teacher_state",
    "teacher_pacifier",
    "face_visible",
    "sleep_surface",
    "in_crib",
)


@dataclass(frozen=True, slots=True)
class FrameExample:
    sample_id: str
    frame_id: str
    captured_at: str
    location_id: str
    relative_path: str
    sha256: str
    provider: str
    model: str
    crop_name: str
    crop_x0: float
    crop_y0: float
    crop_x1: float
    crop_y1: float
    split: str
    presence_target: int
    presence_mask: int
    awake_target: int
    awake_mask: int
    pacifier_target: int
    pacifier_mask: int
    confidence: float
    teacher_state: str
    teacher_pacifier: str
    face_visible: str
    sleep_surface: str
    in_crib: bool | None

    @property
    def day(self) -> str:
        return self.captured_at[:10]

    def target(self, task: str) -> int:
        return int(getattr(self, f"{task}_target"))

    def mask(self, task: str) -> int:
        return int(getattr(self, f"{task}_mask"))


@dataclass(frozen=True, slots=True)
class BuildResult:
    examples: tuple[FrameExample, ...]
    skipped: dict[str, int]


@dataclass(frozen=True, slots=True)
class CaptureWindow:
    start: str | None = None
    end: str | None = None

    def contains(self, captured_at: str) -> bool:
        return (self.start is None or captured_at >= self.start) and (self.end is None or captured_at < self.end)


def _parse_optional_bool(value: str | None) -> bool | None:
    normalized = (value or "").strip().lower()
    if not normalized:
        return None
    if normalized in {"1", "true", "yes", "y", "si", "sí"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"invalid boolean override: {value!r}")


def read_overrides(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        if "frame_id" not in (rows.fieldnames or ()):
            raise ValueError("override CSV must contain frame_id")
        return {
            row["frame_id"].strip(): {key: (value or "").strip() for key, value in row.items()}
            for row in rows
            if row.get("frame_id", "").strip()
        }


def _safe_image_path(frames_dir: Path, relative_path: str) -> Path:
    root = frames_dir.resolve()
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"frame path escapes the frames directory: {relative_path}")
    return candidate


def _manual_target(override: dict[str, str], task: str) -> int | None:
    raw = override.get(task, "").strip().lower()
    if not raw:
        return None
    values = {
        "presence": {"present": 1, "absent": 0, "yes": 1, "no": 0, "1": 1, "0": 0},
        "awake": {"awake": 1, "asleep": 0, "yes": 1, "no": 0, "1": 1, "0": 0},
        "pacifier": {"yes": 1, "no": 0, "1": 1, "0": 0},
    }
    try:
        return values[task][raw]
    except KeyError as exc:
        raise ValueError(f"invalid {task} override: {raw!r}") from exc


def load_examples(
    database: Path,
    frames_dir: Path,
    *,
    min_confidence: float = 0.8,
    overrides_path: Path | None = None,
    require_visible_face_for_pacifier: bool = True,
) -> BuildResult:
    """Read weak labels without changing the live Baby Monitor database."""

    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between 0 and 1")
    if not database.is_file():
        raise FileNotFoundError(database)
    if not frames_dir.is_dir():
        raise NotADirectoryError(frames_dir)

    overrides = read_overrides(overrides_path)
    skipped: Counter[str] = Counter()
    uri = f"{database.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """SELECT id, captured_at, location_id, relative_path, sha256,
                      provider, model, label_json
               FROM frames
               WHERE image_available = 1
                 AND relative_path IS NOT NULL
                 AND label_json IS NOT NULL
               ORDER BY captured_at, id"""
        ).fetchall()

    examples: list[FrameExample] = []
    seen_hashes: set[str] = set()
    for row in rows:
        frame_id = str(row["id"])
        override = overrides.get(frame_id, {})
        if _parse_optional_bool(override.get("exclude")) is True:
            skipped["manual_exclude"] += 1
            continue
        try:
            label = json.loads(row["label_json"])
        except (TypeError, json.JSONDecodeError):
            skipped["invalid_label_json"] += 1
            continue
        if not isinstance(label, dict):
            skipped["invalid_label_json"] += 1
            continue
        tags = {str(tag).strip().lower() for tag in label.get("tags", [])}
        if "image_unusable" in tags:
            skipped["image_unusable"] += 1
            continue
        try:
            confidence = float(label["confidence"])
        except (KeyError, TypeError, ValueError):
            skipped["invalid_confidence"] += 1
            continue
        if confidence < min_confidence:
            skipped["low_confidence"] += 1
            continue
        relative_path = str(row["relative_path"])
        try:
            image_path = _safe_image_path(frames_dir, relative_path)
        except ValueError:
            skipped["unsafe_path"] += 1
            continue
        if not image_path.is_file():
            skipped["missing_image"] += 1
            continue
        digest = str(row["sha256"])
        if digest in seen_hashes:
            skipped["duplicate_sha256"] += 1
            continue
        seen_hashes.add(digest)

        presence_override = _manual_target(override, "presence")
        baby_present = int(bool(label.get("baby_present"))) if presence_override is None else presence_override
        state = str(label.get("state", "uncertain"))
        pacifier = str(label.get("pacifier", "unknown"))
        face_visible = str(label.get("face_visible", "unknown"))

        awake_override = _manual_target(override, "awake")
        if awake_override is not None:
            awake_target, awake_mask = awake_override, int(baby_present == 1)
        elif baby_present and state in {"awake", "asleep"}:
            awake_target, awake_mask = int(state == "awake"), 1
        else:
            awake_target, awake_mask = 0, 0

        pacifier_override = _manual_target(override, "pacifier")
        if pacifier_override is not None:
            pacifier_target, pacifier_mask = pacifier_override, int(baby_present == 1)
        elif (
            baby_present
            and pacifier in {"yes", "no"}
            and (face_visible == "yes" or not require_visible_face_for_pacifier)
        ):
            pacifier_target, pacifier_mask = int(pacifier == "yes"), 1
        else:
            pacifier_target, pacifier_mask = 0, 0

        examples.append(
            FrameExample(
                sample_id=frame_id,
                frame_id=frame_id,
                captured_at=str(row["captured_at"]),
                location_id=str(row["location_id"]),
                relative_path=relative_path,
                sha256=digest,
                provider=str(row["provider"] or ""),
                model=str(row["model"] or ""),
                crop_name="full_frame",
                crop_x0=0.0,
                crop_y0=0.0,
                crop_x1=1.0,
                crop_y1=1.0,
                split="",
                presence_target=baby_present,
                presence_mask=1,
                awake_target=awake_target,
                awake_mask=awake_mask,
                pacifier_target=pacifier_target,
                pacifier_mask=pacifier_mask,
                confidence=confidence,
                teacher_state=state,
                teacher_pacifier=pacifier,
                face_visible=face_visible,
                sleep_surface=str(label.get("sleep_surface", "unknown") or "unknown"),
                in_crib=label.get("in_crib") if isinstance(label.get("in_crib"), bool) else None,
            )
        )
    return BuildResult(tuple(examples), dict(sorted(skipped.items())))


@dataclass(frozen=True, slots=True)
class RegionOfInterest:
    name: str
    rect: tuple[float, float, float, float]
    in_crib: bool | None = None
    surfaces: tuple[str, ...] = ()

    def matches(self, example: FrameExample) -> bool:
        if self.in_crib is not None and example.in_crib is not None:
            return self.in_crib == example.in_crib
        return bool(self.surfaces and example.sleep_surface in self.surfaces)


def read_roi_config(path: Path | None) -> dict[str, tuple[RegionOfInterest, ...]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ROI config must be an object keyed by location ID")
    result: dict[str, tuple[RegionOfInterest, ...]] = {}
    for location, raw_regions in payload.items():
        if not isinstance(raw_regions, list) or not raw_regions:
            raise ValueError(f"ROI config for {location!r} must be a non-empty array")
        regions: list[RegionOfInterest] = []
        for raw in raw_regions:
            if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
                raise ValueError(f"invalid ROI entry for {location!r}")
            rect = raw.get("rect")
            if (
                not isinstance(rect, list)
                or len(rect) != 4
                or not all(isinstance(value, int | float) for value in rect)
            ):
                raise ValueError(f"ROI {raw.get('name')!r} must have four numeric coordinates")
            x0, y0, x1, y1 = (float(value) for value in rect)
            if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                raise ValueError(f"ROI {raw['name']!r} coordinates must be ordered within [0, 1]")
            in_crib = raw.get("in_crib")
            if in_crib is not None and not isinstance(in_crib, bool):
                raise ValueError(f"ROI {raw['name']!r} in_crib must be boolean or null")
            surfaces = raw.get("surfaces", [])
            if not isinstance(surfaces, list) or not all(isinstance(item, str) for item in surfaces):
                raise ValueError(f"ROI {raw['name']!r} surfaces must be strings")
            regions.append(
                RegionOfInterest(
                    name=raw["name"],
                    rect=(x0, y0, x1, y1),
                    in_crib=in_crib,
                    surfaces=tuple(surfaces),
                )
            )
        result[str(location)] = tuple(regions)
    return result


def read_capture_windows(path: Path | None) -> dict[str, CaptureWindow]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("capture-window config must be an object keyed by location ID")
    result: dict[str, CaptureWindow] = {}
    for location, raw_window in payload.items():
        if not isinstance(raw_window, dict):
            raise ValueError(f"capture window for {location!r} must be an object")
        unknown = set(raw_window) - {"start", "end"}
        if unknown:
            raise ValueError(f"capture window for {location!r} has unknown keys: {sorted(unknown)}")
        start = raw_window.get("start")
        end = raw_window.get("end")
        if start is not None and not isinstance(start, str):
            raise ValueError(f"capture window start for {location!r} must be a timestamp string")
        if end is not None and not isinstance(end, str):
            raise ValueError(f"capture window end for {location!r} must be a timestamp string")
        if start is not None and end is not None and start >= end:
            raise ValueError(f"capture window start for {location!r} must precede its end")
        result[str(location)] = CaptureWindow(start=start, end=end)
    return result


def apply_capture_windows(
    examples: tuple[FrameExample, ...],
    capture_windows: dict[str, CaptureWindow],
) -> tuple[tuple[FrameExample, ...], dict[str, int]]:
    """Remove frames captured outside the camera geometry represented by a profile."""

    kept: list[FrameExample] = []
    skipped: Counter[str] = Counter()
    for example in examples:
        window = capture_windows.get(example.location_id)
        if window is not None and not window.contains(example.captured_at):
            skipped["outside_capture_window"] += 1
            continue
        kept.append(example)
    return tuple(kept), dict(sorted(skipped.items()))


def apply_roi_config(
    examples: tuple[FrameExample, ...],
    roi_config: dict[str, tuple[RegionOfInterest, ...]],
) -> tuple[tuple[FrameExample, ...], dict[str, int]]:
    """Use one known-positive ROI, while retaining every ROI for empty frames."""

    expanded: list[FrameExample] = []
    skipped: Counter[str] = Counter()
    for example in examples:
        regions = roi_config.get(example.location_id)
        if not regions:
            expanded.append(example)
            continue
        if example.presence_target:
            matches = [region for region in regions if region.matches(example)]
            if len(matches) != 1:
                skipped["positive_without_unique_roi"] += 1
                continue
            positive_region = matches[0]
        else:
            positive_region = None
        for region in regions:
            x0, y0, x1, y1 = region.rect
            region_has_baby = positive_region is not None and region.name == positive_region.name
            expanded.append(
                replace(
                    example,
                    sample_id=f"{example.frame_id}:{region.name}",
                    crop_name=region.name,
                    crop_x0=x0,
                    crop_y0=y0,
                    crop_x1=x1,
                    crop_y1=y1,
                    presence_target=int(region_has_baby),
                    awake_mask=example.awake_mask if region_has_baby else 0,
                    pacifier_mask=example.pacifier_mask if region_has_baby else 0,
                )
            )
    return tuple(expanded), dict(sorted(skipped.items()))


def chronological_group_split(
    examples: tuple[FrameExample, ...],
    *,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> tuple[FrameExample, ...]:
    """Keep every location/day together and reserve its newest days for evaluation."""

    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must be positive and leave room for training")
    days_by_location: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        days_by_location[example.location_id].add(example.day)

    split_by_group: dict[tuple[str, str], str] = {}
    for location, day_set in days_by_location.items():
        days = sorted(day_set)
        if len(days) < 3:
            raise ValueError(f"location {location!r} needs at least three capture days")
        test_days = max(1, round(len(days) * test_fraction))
        validation_days = max(1, round(len(days) * validation_fraction))
        while test_days + validation_days >= len(days):
            if test_days >= validation_days and test_days > 1:
                test_days -= 1
            elif validation_days > 1:
                validation_days -= 1
            else:
                raise ValueError(f"location {location!r} does not have enough days for all splits")
        validation_start = len(days) - test_days - validation_days
        test_start = len(days) - test_days
        for index, day in enumerate(days):
            split = "train" if index < validation_start else "validation" if index < test_start else "test"
            split_by_group[(location, day)] = split

    return tuple(replace(example, split=split_by_group[(example.location_id, example.day)]) for example in examples)


def summarize(examples: tuple[FrameExample, ...], skipped: dict[str, int]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "examples": len(examples),
        "unique_frames": len({example.frame_id for example in examples}),
        "skipped": skipped,
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        subset = [example for example in examples if example.split == split]
        split_summary: dict[str, Any] = {
            "examples": len(subset),
            "unique_frames": len({example.frame_id for example in subset}),
            "locations": dict(sorted(Counter(example.location_id for example in subset).items())),
            "capture_days": {
                location: sorted({example.day for example in subset if example.location_id == location})
                for location in sorted({example.location_id for example in subset})
            },
            "tasks": {},
        }
        for task in TASKS:
            eligible = [example for example in subset if example.mask(task)]
            positives = sum(example.target(task) for example in eligible)
            split_summary["tasks"][task] = {
                "eligible": len(eligible),
                "positive": positives,
                "negative": len(eligible) - positives,
            }
        result["splits"][split] = split_summary
    return result


def write_manifest(examples: tuple[FrameExample, ...], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for example in examples:
            writer.writerow(asdict(example))


def read_manifest(path: Path) -> tuple[FrameExample, ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        if tuple(rows.fieldnames or ()) != MANIFEST_FIELDS:
            raise ValueError("manifest columns do not match the expected schema")
        return tuple(
            FrameExample(
                sample_id=row["sample_id"],
                frame_id=row["frame_id"],
                captured_at=row["captured_at"],
                location_id=row["location_id"],
                relative_path=row["relative_path"],
                sha256=row["sha256"],
                provider=row["provider"],
                model=row["model"],
                crop_name=row["crop_name"],
                crop_x0=float(row["crop_x0"]),
                crop_y0=float(row["crop_y0"]),
                crop_x1=float(row["crop_x1"]),
                crop_y1=float(row["crop_y1"]),
                split=row["split"],
                presence_target=int(row["presence_target"]),
                presence_mask=int(row["presence_mask"]),
                awake_target=int(row["awake_target"]),
                awake_mask=int(row["awake_mask"]),
                pacifier_target=int(row["pacifier_target"]),
                pacifier_mask=int(row["pacifier_mask"]),
                confidence=float(row["confidence"]),
                teacher_state=row["teacher_state"],
                teacher_pacifier=row["teacher_pacifier"],
                face_visible=row["face_visible"],
                sleep_surface=row["sleep_surface"],
                in_crib=_parse_optional_bool(row["in_crib"]),
            )
            for row in rows
        )


def write_review_queue(examples: tuple[FrameExample, ...], path: Path, per_bucket: int = 12) -> None:
    """Create a deterministic, stratified manual-review queue without copying images."""

    selected: dict[str, FrameExample] = {}
    for split in ("train", "validation", "test"):
        for task in TASKS:
            for target in (0, 1):
                candidates = [
                    example
                    for example in examples
                    if example.split == split and example.mask(task) and example.target(task) == target
                ]
                candidates.sort(
                    key=lambda example: hashlib.sha256(
                        f"{split}:{task}:{target}:{example.frame_id}".encode()
                    ).hexdigest()
                )
                for example in candidates[:per_bucket]:
                    selected[example.sample_id] = example

    fields = (
        "sample_id",
        "frame_id",
        "captured_at",
        "location_id",
        "relative_path",
        "crop_name",
        "crop_x0",
        "crop_y0",
        "crop_x1",
        "crop_y1",
        "split",
        "teacher_presence",
        "teacher_awake",
        "teacher_pacifier",
        "exclude",
        "presence",
        "awake",
        "pacifier",
        "notes",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for example in sorted(selected.values(), key=lambda item: (item.captured_at, item.frame_id)):
            writer.writerow(
                {
                    "sample_id": example.sample_id,
                    "frame_id": example.frame_id,
                    "captured_at": example.captured_at,
                    "location_id": example.location_id,
                    "relative_path": example.relative_path,
                    "crop_name": example.crop_name,
                    "crop_x0": example.crop_x0,
                    "crop_y0": example.crop_y0,
                    "crop_x1": example.crop_x1,
                    "crop_y1": example.crop_y1,
                    "split": example.split,
                    "teacher_presence": "present" if example.presence_target else "absent",
                    "teacher_awake": example.teacher_state,
                    "teacher_pacifier": example.teacher_pacifier,
                    "exclude": "",
                    "presence": "",
                    "awake": "",
                    "pacifier": "",
                    "notes": "",
                }
            )


def write_temporal_label_flips(
    examples: tuple[FrameExample, ...],
    path: Path,
    *,
    maximum_gap_seconds: float = 10 * 60,
) -> dict[str, int]:
    """Write adjacent weak-label changes that deserve paired manual review."""

    by_location: dict[str, list[FrameExample]] = defaultdict(list)
    for example in examples:
        by_location[example.location_id].append(example)
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for location, location_examples in by_location.items():
        ordered = sorted(location_examples, key=lambda example: (example.captured_at, example.frame_id))
        for previous, current in zip(ordered, ordered[1:], strict=False):
            previous_at = datetime.fromisoformat(previous.captured_at.replace("Z", "+00:00"))
            current_at = datetime.fromisoformat(current.captured_at.replace("Z", "+00:00"))
            gap_seconds = (current_at - previous_at).total_seconds()
            if gap_seconds < 0 or gap_seconds > maximum_gap_seconds:
                continue
            for task in ("awake", "pacifier"):
                if previous.mask(task) and current.mask(task) and previous.target(task) != current.target(task):
                    counts[task] += 1
                    rows.append(
                        {
                            "location_id": location,
                            "task": task,
                            "gap_seconds": round(gap_seconds, 3),
                            "previous_frame_id": previous.frame_id,
                            "previous_captured_at": previous.captured_at,
                            "previous_relative_path": previous.relative_path,
                            "previous_label": previous.target(task),
                            "current_frame_id": current.frame_id,
                            "current_captured_at": current.captured_at,
                            "current_relative_path": current.relative_path,
                            "current_label": current.target(task),
                            "review_notes": "",
                        }
                    )
    fields = (
        "location_id",
        "task",
        "gap_seconds",
        "previous_frame_id",
        "previous_captured_at",
        "previous_relative_path",
        "previous_label",
        "current_frame_id",
        "current_captured_at",
        "current_relative_path",
        "current_label",
        "review_notes",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return dict(sorted(counts.items()))


def prepare_dataset(
    database: Path,
    frames_dir: Path,
    output_dir: Path,
    *,
    min_confidence: float = 0.8,
    overrides_path: Path | None = None,
    require_visible_face_for_pacifier: bool = True,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    roi_config_path: Path | None = None,
    capture_windows_path: Path | None = None,
) -> dict[str, Any]:
    loaded = load_examples(
        database,
        frames_dir,
        min_confidence=min_confidence,
        overrides_path=overrides_path,
        require_visible_face_for_pacifier=require_visible_face_for_pacifier,
    )
    windowed_examples, window_skipped = apply_capture_windows(
        loaded.examples,
        read_capture_windows(capture_windows_path),
    )
    examples, roi_skipped = apply_roi_config(
        windowed_examples,
        read_roi_config(roi_config_path),
    )
    skipped = Counter(loaded.skipped)
    skipped.update(window_skipped)
    skipped.update(roi_skipped)
    examples = chronological_group_split(
        examples,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(examples, output_dir / "manifest.csv")
    write_review_queue(examples, output_dir / "review_queue.csv")
    summary = summarize(examples, dict(sorted(skipped.items())))
    summary["temporal_label_flips"] = write_temporal_label_flips(
        windowed_examples,
        output_dir / "temporal_label_flips.csv",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
