from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from baby_monitor.providers import YoloLocalProvider

if TYPE_CHECKING:
    from PIL import Image

MARKER = ".baby-monitor-yolo-review-gallery"
TASKS = ("presence", "awake", "pacifier")
CSV_FIELDS = (
    "frame_id",
    "captured_at",
    "location_id",
    "overall_outcome",
    "winner_roi",
    "head_side",
    "body_position",
    "mouth_open",
    "adult_present",
    "adult_count",
    "presence_score",
    "presence_decision",
    "presence_reference",
    "presence_outcome",
    "awake_score",
    "awake_decision",
    "awake_reference",
    "awake_outcome",
    "pacifier_score",
    "pacifier_decision",
    "pacifier_reference",
    "pacifier_outcome",
    "detail_available",
    "frame_image",
    "roi_image",
    "head_image",
)


def _parse_int(value: str | None) -> int | None:
    if value in {"0", "1"}:
        return int(value)
    return None


def _masked_target(rows: list[dict[str, str]], task: str) -> int | None:
    values = {
        parsed
        for row in rows
        if row.get(f"{task}_mask") == "1" and (parsed := _parse_int(row.get(f"{task}_target"))) is not None
    }
    if not values:
        return None
    if len(values) != 1:
        raise ValueError(f"frame {rows[0]['frame_id']} has conflicting {task} targets")
    return values.pop()


def _reference(rows: list[dict[str, str]]) -> dict[str, Any]:
    presence_values = [
        parsed
        for row in rows
        if row.get("presence_mask") == "1" and (parsed := _parse_int(row.get("presence_target"))) is not None
    ]
    presence = max(presence_values) if presence_values else None
    positive_row = next(
        (row for row in rows if row.get("presence_mask") == "1" and row.get("presence_target") == "1"),
        rows[0],
    )
    return {
        "presence": None if presence is None else ("present" if presence else "absent"),
        "awake": _target_name("awake", _masked_target(rows, "awake")),
        "pacifier": _target_name("pacifier", _masked_target(rows, "pacifier")),
        "surface": positive_row.get("sleep_surface") or "unknown",
        "target_roi": positive_row.get("crop_name") if presence else None,
        "provider": rows[0].get("provider") or "unknown",
        "model": rows[0].get("model") or "unknown",
        "confidence": _optional_float(rows[0].get("confidence")),
        "face_visible": rows[0].get("face_visible") or "unknown",
    }


def _target_name(task: str, target: int | None) -> str | None:
    if target is None:
        return None
    labels = {
        "presence": ("absent", "present"),
        "awake": ("asleep", "awake"),
        "pacifier": ("no", "yes"),
    }
    return labels[task][target]


def _optional_float(value: str | None) -> float | None:
    try:
        return float(value) if value not in {None, ""} else None
    except ValueError:
        return None


def _decision(task: str, score: float | None, bounds: tuple[float, float]) -> str:
    if score is None:
        return "not_run"
    negative, positive = bounds
    if score <= negative:
        return _target_name(task, 0) or "unknown"
    if score >= positive:
        return _target_name(task, 1) or "unknown"
    return "unknown"


def _outcome(reference: str | None, decision: str) -> str:
    if reference is None:
        return "not_labeled"
    if decision in {"unknown", "not_run"}:
        return "abstain"
    return "match" if reference == decision else "mismatch"


def _overall_outcome(tasks: dict[str, dict[str, Any]]) -> str:
    outcomes = [item["outcome"] for item in tasks.values()]
    if "mismatch" in outcomes:
        return "mismatch"
    if "abstain" in outcomes:
        return "abstain"
    if "match" in outcomes:
        return "match"
    return "not_labeled"


def _review_priority(outcome: str, tasks: dict[str, dict[str, Any]]) -> float:
    bucket = {"mismatch": 0.0, "abstain": 1.0, "match": 2.0, "not_labeled": 3.0}[outcome]
    margins: list[float] = []
    for item in tasks.values():
        score = item["score"]
        if score is None:
            continue
        negative, positive = item["thresholds"]
        margins.append(min(abs(score - negative), abs(score - positive)))
    return bucket + min(margins, default=1.0)


def _group_manifest(
    manifest_path: Path,
    split: str,
    *,
    limit: int | None,
) -> list[list[dict[str, str]]]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == split]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["frame_id"]].append(row)
    ordered = sorted(
        grouped.values(),
        key=lambda frame_rows: (frame_rows[0]["captured_at"], frame_rows[0]["frame_id"]),
    )
    return ordered if limit is None else ordered[:limit]


def _prepare_output(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise ValueError(f"{output_dir} is not empty; pass --overwrite to rebuild it")
        if not (output_dir / MARKER).is_file():
            raise ValueError(f"refusing to overwrite unmarked directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / MARKER).write_text("Baby Monitor private YOLO review gallery\n", encoding="utf-8")
    for name in ("frames", "rois", "heads"):
        (output_dir / name).mkdir()


def _verify_source(frame_path: Path, expected_sha256: str) -> None:
    digest = hashlib.sha256(frame_path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"source frame integrity check failed: {frame_path}")


def _annotated_frame(
    image: Image.Image,
    profiles: list[dict[str, Any]],
    scores: list[float],
    winner: int,
    *,
    max_width: int = 960,
) -> Image.Image:
    from PIL import Image, ImageDraw, ImageFont

    preview = image.copy()
    preview.thumbnail((max_width, max_width), Image.Resampling.LANCZOS)
    scale_x = preview.width / image.width
    scale_y = preview.height / image.height
    draw = ImageDraw.Draw(preview, "RGBA")
    try:
        font = ImageFont.load_default(size=max(13, round(preview.width / 70)))
    except TypeError:
        font = ImageFont.load_default()
    for index, (profile, score) in enumerate(zip(profiles, scores, strict=True)):
        rect = profile["rect"]
        box = (
            round(float(rect[0]) * image.width * scale_x),
            round(float(rect[1]) * image.height * scale_y),
            round(float(rect[2]) * image.width * scale_x),
            round(float(rect[3]) * image.height * scale_y),
        )
        selected = index == winner
        color = (174, 255, 55, 255) if selected else (255, 176, 32, 235)
        width = 5 if selected else 3
        draw.rectangle(box, outline=color, width=width)
        label = f"{profile['name']}  {score:.1%}"
        text_box = draw.textbbox((box[0], box[1]), label, font=font, stroke_width=0)
        padding = 5
        background = (
            text_box[0] - padding,
            text_box[1] - padding,
            text_box[2] + padding,
            text_box[3] + padding,
        )
        draw.rectangle(background, fill=(10, 13, 18, 210))
        draw.text((box[0], box[1]), label, fill=color, font=font)
    return preview


def _save_webp(image: Image.Image, path: Path, *, quality: int) -> None:
    image.save(path, "WEBP", quality=quality, method=5)


def _profile_payload(
    profiles: list[dict[str, Any]],
    scores: list[float],
    winner: int,
) -> list[dict[str, Any]]:
    return [
        {
            "name": str(profile["name"]),
            "surface": str(profile["surface"]),
            "rect": [float(value) for value in profile["rect"]],
            "score": score,
            "selected": index == winner,
        }
        for index, (profile, score) in enumerate(zip(profiles, scores, strict=True))
    ]


def _summary(frames: list[dict[str, Any]]) -> dict[str, Any]:
    overall = Counter(frame["overall_outcome"] for frame in frames)
    locations = Counter(frame["location_id"] for frame in frames)
    task_summary: dict[str, Any] = {}
    for task in TASKS:
        outcomes = Counter(frame["tasks"][task]["outcome"] for frame in frames)
        labeled = sum(outcomes[name] for name in ("match", "mismatch", "abstain"))
        decisions = outcomes["match"] + outcomes["mismatch"]
        task_summary[task] = {
            "labeled": labeled,
            "decisions": decisions,
            "matches": outcomes["match"],
            "mismatches": outcomes["mismatch"],
            "abstentions": outcomes["abstain"],
            "coverage": decisions / labeled if labeled else None,
            "agreement": outcomes["match"] / decisions if decisions else None,
        }
    return {
        "frames": len(frames),
        "locations": dict(sorted(locations.items())),
        "overall": {name: overall[name] for name in ("match", "mismatch", "abstain", "not_labeled")},
        "detail_available": sum(frame["detail_available"] for frame in frames),
        "tasks": task_summary,
    }


def _copy_static_assets(output_dir: Path) -> None:
    source = Path(__file__).with_name("yolo_review_static")
    for name in ("index.html", "styles.css", "app.js"):
        shutil.copy2(source / name, output_dir / name)


def _write_csv(frames: Iterable[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for frame in frames:
            tasks = frame["tasks"]
            writer.writerow(
                {
                    "frame_id": frame["frame_id"],
                    "captured_at": frame["captured_at"],
                    "location_id": frame["location_id"],
                    "overall_outcome": frame["overall_outcome"],
                    "winner_roi": frame["winner_roi"],
                    "head_side": frame["prediction"]["head_side"],
                    "body_position": frame["prediction"]["body_position"],
                    "mouth_open": frame["prediction"]["mouth_open"],
                    "adult_present": frame["prediction"]["adult_present"],
                    "adult_count": frame["prediction"]["adult_count"],
                    **{
                        f"{task}_{field}": tasks[task][field]
                        for task in TASKS
                        for field in ("score", "decision", "reference", "outcome")
                    },
                    "detail_available": frame["detail_available"],
                    "frame_image": frame["images"]["frame"],
                    "roi_image": frame["images"]["roi"],
                    "head_image": frame["images"]["head"],
                }
            )


def _write_readme(
    output_dir: Path,
    *,
    manifest_path: Path,
    frames_dir: Path,
    model_dir: Path,
    split: str,
) -> None:
    text = f"""# Local YOLO review gallery

Open `index.html` directly in a browser. No server or network connection is
required.

The gallery was generated with the production `YoloLocalProvider`, including
the exact ROI selection, RGB/BGR conversion, pose-head localizer, awake
ensemble, and per-location abstention thresholds.

- Artifact: `{model_dir}`
- Manifest: `{manifest_path}`
- Source frames: `{frames_dir}`
- Split: `{split}`
- Flat export: `predictions.csv`

The reference column contains historical weak labels from the configured
teacher model. Agreement is useful for regression testing, but it is not
equivalent to manually verified ground truth.

All copied frames and crops are private and this whole output directory is
ignored by Git.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def build_gallery(
    manifest_path: Path,
    frames_dir: Path,
    model_dir: Path,
    output_dir: Path,
    *,
    split: str = "test",
    device: str = "cpu",
    overwrite: bool = False,
    limit: int | None = None,
    verify_sha256: bool = True,
) -> dict[str, Any]:
    from PIL import Image, ImageOps

    manifest_path = manifest_path.expanduser().resolve()
    frames_dir = frames_dir.expanduser().resolve()
    model_dir = model_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    os.environ["BABY_MONITOR_YOLO_DEVICE"] = device
    groups = _group_manifest(manifest_path, split, limit=limit)
    if not groups:
        raise ValueError(f"manifest contains no frames for split {split!r}")
    _prepare_output(output_dir, overwrite=overwrite)
    _copy_static_assets(output_dir)
    provider = YoloLocalProvider(str(model_dir))
    image_size = int(provider.metadata["image_size"])
    frames: list[dict[str, Any]] = []

    for index, rows in enumerate(groups, start=1):
        row = rows[0]
        frame_path = frames_dir / row["relative_path"]
        if verify_sha256:
            _verify_source(frame_path, row["sha256"])
        with Image.open(frame_path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        profiles = provider._profiles(row["location_id"])
        rois = [provider._crop(image, profile, image_size) for profile in profiles]
        with provider._lock:
            presence_scores = provider._scores("presence", rois)
            winner = max(range(len(rois)), key=presence_scores.__getitem__)
            presence_bounds = provider._threshold_bounds("presence", row["location_id"])
            decisive_present = presence_scores[winner] >= presence_bounds[1]
            detail_crops = (
                provider._detail_crops(image, profiles[winner], image_size)
                if decisive_present
                else {
                    "head": None,
                    "body": None,
                    "mouth": None,
                    "adult_pose_present": False,
                    "adult_count": None,
                }
            )
            head = detail_crops["head"]
            awake_score = provider._scores("awake", [head])[0] if head is not None else None
            pacifier_crop = detail_crops[
                provider.metadata["tasks"]["pacifier"].get("crop", "head")
            ]
            pacifier_score = (
                provider._scores("pacifier", [pacifier_crop])[0]
                if pacifier_crop is not None
                else None
            )
            details = {
                "face_visible": "yes" if head is not None else "unknown",
                "adult_count": detail_crops["adult_count"],
                "adult_present": (
                    "yes"
                    if detail_crops["adult_pose_present"]
                    else provider._adult_presence_decision(
                        image,
                        detail_crops["adult_count"],
                        row["location_id"],
                    )
                ),
                **{
                    task: provider._multiclass_decision(
                        task,
                        detail_crops[crop_name],
                        row["location_id"],
                    )
                    for task, crop_name in provider._OPTIONAL_TASK_CROPS.items()
                },
            }
        label = provider._build_label(
            profiles,
            presence_scores,
            awake_score,
            pacifier_score,
            row["location_id"],
            details,
        )
        reference = _reference(rows)
        task_scores = {
            "presence": presence_scores[winner],
            "awake": awake_score,
            "pacifier": pacifier_score,
        }
        tasks: dict[str, dict[str, Any]] = {}
        for task in TASKS:
            bounds = provider._threshold_bounds(task, row["location_id"])
            decision = _decision(task, task_scores[task], bounds)
            task_reference = reference[task]
            tasks[task] = {
                "score": task_scores[task],
                "decision": decision,
                "reference": task_reference,
                "outcome": _outcome(task_reference, decision),
                "thresholds": list(bounds),
            }
        overall = _overall_outcome(tasks)
        stem = row["frame_id"]
        frame_ref = f"frames/{stem}.webp"
        roi_ref = f"rois/{stem}.webp"
        head_ref = f"heads/{stem}.webp" if head is not None else None
        _save_webp(
            _annotated_frame(image, profiles, presence_scores, winner),
            output_dir / frame_ref,
            quality=82,
        )
        _save_webp(rois[winner], output_dir / roi_ref, quality=88)
        if head is not None and head_ref is not None:
            _save_webp(head, output_dir / head_ref, quality=92)
        frame = {
            "frame_id": stem,
            "captured_at": row["captured_at"],
            "location_id": row["location_id"],
            "relative_path": row["relative_path"],
            "winner_roi": str(profiles[winner]["name"]),
            "winner_surface": str(profiles[winner]["surface"]),
            "roi_profiles": _profile_payload(profiles, presence_scores, winner),
            "detail_available": head is not None,
            "prediction": {
                "baby_present": label.baby_present,
                "state": label.state,
                "pacifier": label.pacifier,
                "head_side": label.head_side,
                "body_position": label.body_position,
                "mouth_open": label.mouth_open,
                "adult_present": label.adult_present,
                "adult_count": label.adult_count,
                "confidence": label.confidence,
                "sleep_surface": label.sleep_surface,
                "tags": label.tags,
            },
            "reference": reference,
            "tasks": tasks,
            "overall_outcome": overall,
            "review_priority": _review_priority(overall, tasks),
            "images": {
                "frame": frame_ref,
                "roi": roi_ref,
                "head": head_ref,
            },
        }
        frames.append(frame)
        if index == 1 or index % 100 == 0 or index == len(groups):
            print(f"[{index}/{len(groups)}] {row['captured_at']} {row['location_id']}", flush=True)

    payload = {
        "meta": {
            "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "artifact_version": provider.model,
            "artifact_format": provider.metadata["format"],
            "split": split,
            "device": device,
            "manifest": str(manifest_path),
            "frames_dir": str(frames_dir),
            "model_dir": str(model_dir),
            "reference_note": (
                "Historical teacher labels are weak supervision. A match is not a substitute "
                "for manual visual verification."
            ),
        },
        "summary": _summary(frames),
        "frames": frames,
    }
    compact_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    (output_dir / "data.js").write_text(
        f"window.YOLO_REVIEW_DATA={compact_json};\n",
        encoding="utf-8",
    )
    (output_dir / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(frames, output_dir / "predictions.csv")
    _write_readme(
        output_dir,
        manifest_path=manifest_path,
        frames_dir=frames_dir,
        model_dir=model_dir,
        split=split,
    )
    return {
        "output_dir": str(output_dir),
        "index": str(output_dir / "index.html"),
        "frames": len(frames),
        "summary": payload["summary"],
        "artifact_version": provider.model,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a private, offline HTML gallery from the exact production YOLO runtime.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-sha256", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = build_gallery(
        args.manifest,
        args.frames_dir,
        args.model_dir,
        args.output_dir,
        split=args.split,
        device=args.device,
        overwrite=args.overwrite,
        limit=args.limit,
        verify_sha256=not args.skip_sha256,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
