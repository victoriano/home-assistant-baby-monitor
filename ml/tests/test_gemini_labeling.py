from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from baby_monitor_edge_ml.gemini_labeling import (
    GEMINI_ANALYSIS_MARKER,
    GEMINI_LABEL_SCHEMA,
    GEMINI_PILOT_MARKER,
    GEMINI_PRICING_SNAPSHOT_DATE,
    _completed_pairs,
    analyze_gemini_teacher_pilots,
    build_evidence_board,
    build_gemini_request,
    prepare_gemini_adult_dataset,
    prepare_gemini_detail_dataset,
    prepare_gemini_label_pilot,
    validate_gemini_label,
)


def _image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (80, 60), color).save(path)


def _valid_label() -> dict[str, object]:
    return {
        "baby_visible": "yes",
        "detail_panels_match_infant": "yes",
        "detail_panels_confidence": "high",
        "head_orientation": "image_left",
        "head_confidence": "high",
        "body_position": "supine",
        "body_confidence": "high",
        "mouth_state": "closed",
        "mouth_confidence": "medium",
        "pacifier": "absent",
        "pacifier_confidence": "high",
        "adult_presence": "no",
        "adult_count": 0,
        "adult_confidence": "high",
        "limitations": ["none"],
    }


def test_build_evidence_board_has_four_panels(tmp_path: Path) -> None:
    paths = []
    for index, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0))):
        path = tmp_path / f"{index}.jpg"
        _image(path, color)
        paths.append(path)
    output = tmp_path / "board.jpg"
    build_evidence_board(*paths, output, size=512)
    with Image.open(output) as board:
        assert board.size == (512, 512)


def test_request_uses_high_resolution_and_closed_schema() -> None:
    request = build_gemini_request(b"jpeg")
    image_part = request["contents"][0]["parts"][1]
    assert image_part["media_resolution"]["level"] == "MEDIA_RESOLUTION_HIGH"
    assert request["generationConfig"]["responseJsonSchema"] == GEMINI_LABEL_SCHEMA
    assert request["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "high"}
    assert request["generationConfig"]["temperature"] == 0


def test_validate_gemini_label_enforces_adult_consistency() -> None:
    assert validate_gemini_label(_valid_label())["adult_count"] == 0
    canonicalized = _valid_label()
    canonicalized["adult_count"] = None
    assert validate_gemini_label(canonicalized)["adult_count"] == 0
    invalid = _valid_label()
    invalid["adult_presence"] = "yes"
    with pytest.raises(RuntimeError, match="cannot have adult_count=0"):
        validate_gemini_label(invalid)


def test_prepare_pilot_preserves_location_and_split(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    details = tmp_path / "details"
    frames.mkdir()
    (details / "crops").mkdir(parents=True)
    source_rows = []
    detail_rows = []
    for location in ("granada", "madrid"):
        for index, split in enumerate(("train", "validation", "test")):
            frame_id = f"{location}-{index}"
            relative = f"{frame_id}.jpg"
            _image(frames / relative, (40 + index, 50, 60))
            source_rows.append(
                {
                    "frame_id": frame_id,
                    "captured_at": f"2026-07-0{index + 1}T00:00:00Z",
                    "location_id": location,
                    "relative_path": relative,
                    "sha256": "a" * 64,
                    "split": split,
                }
            )
            for task, class_name, suffix in (
                ("head_side", "left", "head"),
                ("body_position", "back", "body"),
                ("mouth_open", "no", "mouth"),
            ):
                crop = f"crops/{frame_id}-{suffix}.jpg"
                _image(details / crop, (70, 80 + index, 90))
                detail_rows.append(
                    {
                        "frame_id": frame_id,
                        "task": task,
                        "class_name": class_name,
                        "crop_path": crop,
                    }
                )
    source_manifest = tmp_path / "source.csv"
    with source_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)
    with (details / "index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)
    output = tmp_path / "private" / "gemini-pilot"
    summary = prepare_gemini_label_pilot(
        source_manifest,
        frames,
        details,
        output,
        samples_per_location=3,
        board_size=512,
    )
    assert summary["frames"] == 6
    assert summary["locations"] == {"granada": 3, "madrid": 3}
    assert summary["splits"] == {"test": 2, "train": 2, "validation": 2}
    assert (output / GEMINI_PILOT_MARKER).is_file()
    assert len(list((output / "boards").glob("*.jpg"))) == 6
    persisted = json.loads((output / "summary.json").read_text())
    assert persisted["prompt_version"] == "baby-visible-features-v2"


def test_completed_pairs_retries_errors_and_accepts_later_success(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    failed = {
        "frame_id": "frame-1",
        "model": "teacher",
        "label": None,
        "error": "temporary failure",
    }
    results.write_text(json.dumps(failed) + "\n", encoding="utf-8")
    assert _completed_pairs(results) == set()

    succeeded = {
        "frame_id": "frame-1",
        "model": "teacher",
        "label": _valid_label(),
        "error": None,
    }
    with results.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(succeeded) + "\n")
    assert _completed_pairs(results) == {("frame-1", "teacher")}


def test_analyze_pilots_requires_teacher_and_mirror_consensus(tmp_path: Path) -> None:
    models = ("teacher-a", "teacher-b")
    original = tmp_path / "original"
    flipped = tmp_path / "flipped"
    manifest_fields = [
        "frame_id",
        "captured_at",
        "location_id",
        "split",
        "relative_path",
        "source_sha256",
    ]
    manifest_row = {
        "frame_id": "frame-1",
        "captured_at": "2026-07-01T00:00:00Z",
        "location_id": "granada",
        "split": "test",
        "relative_path": "frame.jpg",
        "source_sha256": "a" * 64,
    }
    for pilot, horizontal_flip in ((original, False), (flipped, True)):
        pilot.mkdir()
        (pilot / GEMINI_PILOT_MARKER).write_text("private\n", encoding="utf-8")
        with (pilot / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=manifest_fields)
            writer.writeheader()
            writer.writerow(manifest_row)
        (pilot / "summary.json").write_text(
            json.dumps(
                {
                    "horizontal_flip": horizontal_flip,
                    "prompt_version": "test-v1",
                    "source_manifest_sha256": "b" * 64,
                    "detail_index_sha256": "c" * 64,
                }
            ),
            encoding="utf-8",
        )
        with (pilot / "results.jsonl").open("w", encoding="utf-8") as handle:
            for model in models:
                label = _valid_label()
                if horizontal_flip:
                    label["head_orientation"] = "image_right"
                handle.write(
                    json.dumps(
                        {
                            "frame_id": "frame-1",
                            "model": model,
                            "label": label,
                            "error": None,
                        }
                    )
                    + "\n"
                )

    output = tmp_path / "analysis"
    summary = analyze_gemini_teacher_pilots(
        original,
        flipped,
        output,
        models=models,
    )
    assert summary["strict_candidates"]["head_orientation"]["values"] == {
        "image_left": 1
    }
    assert summary["mirror_consistency"]["head_orientation"]["teacher-a"]["rate"] == 1
    assert summary["label_status"] == "training_candidates_not_ground_truth"
    assert (
        summary["api_usage"]["pricing_snapshot_date"]
        == GEMINI_PRICING_SNAPSHOT_DATE
    )
    assert (output / GEMINI_ANALYSIS_MARKER).is_file()


def test_prepare_gemini_details_materializes_selected_tasks(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis"
    source = tmp_path / "source-details"
    analysis.mkdir()
    (source / "crops").mkdir(parents=True)
    (analysis / GEMINI_ANALYSIS_MARKER).write_text("private\n", encoding="utf-8")
    (analysis / "summary.json").write_text(
        json.dumps({"frames": 9}),
        encoding="utf-8",
    )
    source_index: list[dict[str, str]] = []
    comparison_rows: list[dict[str, object]] = []
    head_values = ("image_left", "image_right", "toward_camera")
    for split_index, split in enumerate(("train", "validation", "test")):
        for class_index, head_value in enumerate(head_values):
            frame_id = f"{split}-{class_index}"
            comparison_rows.append(
                {
                    "frame_id": frame_id,
                    "captured_at": f"2026-07-0{split_index + 1}T00:00:00Z",
                    "location_id": "granada",
                    "split": split,
                    "source_sha256": "a" * 64,
                    "fields": {
                        "head_orientation": {"candidate": head_value},
                        "mouth_state": {
                            "candidate": "open" if class_index != 1 else "closed"
                        },
                    },
                }
            )
            for task in ("head_side", "mouth_open"):
                crop_path = f"crops/{frame_id}-{task}.jpg"
                _image(source / crop_path, (30 + class_index, 40, 50))
                source_index.append(
                    {
                        "frame_id": frame_id,
                        "task": task,
                        "sample_id": f"{frame_id}:crib",
                        "crop_path": crop_path,
                    }
                )
    with (analysis / "consensus-candidates.jsonl").open("w", encoding="utf-8") as handle:
        for row in comparison_rows:
            handle.write(json.dumps(row) + "\n")
    with (source / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("frame_id", "task", "sample_id", "crop_path"),
        )
        writer.writeheader()
        writer.writerows(source_index)
    (source / "summary.json").write_text(
        json.dumps({"image_size": 320}),
        encoding="utf-8",
    )

    output = tmp_path / "gemini-details"
    summary = prepare_gemini_detail_dataset(
        analysis,
        source,
        output,
        tasks=("head_side", "mouth_open"),
    )
    assert set(summary["tasks"]) == {"head_side", "mouth_open"}
    assert summary["tasks"]["head_side"]["splits"]["test"]["eligible_classes"] == {
        "back": 1,
        "left": 1,
        "right": 1,
    }
    assert summary["tasks"]["mouth_open"]["splits"]["train"]["eligible_classes"] == {
        "no": 1,
        "yes": 2,
    }
    assert next((output / "crops").iterdir()).is_symlink()


def test_prepare_gemini_adults_balances_each_camera(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis"
    frames = tmp_path / "frames"
    analysis.mkdir()
    frames.mkdir()
    (analysis / GEMINI_ANALYSIS_MARKER).write_text("private\n", encoding="utf-8")
    source_rows: list[dict[str, str]] = []
    comparison_rows: list[dict[str, object]] = []
    for split_index, split in enumerate(("train", "validation", "test")):
        for location in ("granada", "madrid"):
            for class_name in ("no", "yes"):
                frame_id = f"{split}-{location}-{class_name}"
                relative_path = f"{frame_id}.jpg"
                _image(frames / relative_path, (20 + split_index, 30, 40))
                source_rows.append(
                    {
                        "frame_id": frame_id,
                        "captured_at": f"2026-07-0{split_index + 1}T00:00:00Z",
                        "location_id": location,
                        "relative_path": relative_path,
                        "sha256": "a" * 64,
                        "split": split,
                    }
                )
                comparison_rows.append(
                    {
                        "frame_id": frame_id,
                        "captured_at": f"2026-07-0{split_index + 1}T00:00:00Z",
                        "location_id": location,
                        "split": split,
                        "source_sha256": "a" * 64,
                        "fields": {
                            "adult_presence": {"candidate": class_name},
                        },
                    }
                )
    (analysis / "summary.json").write_text(
        json.dumps({"frames": len(comparison_rows)}),
        encoding="utf-8",
    )
    with (analysis / "consensus-candidates.jsonl").open("w", encoding="utf-8") as handle:
        for row in comparison_rows:
            handle.write(json.dumps(row) + "\n")
    source_manifest = tmp_path / "source.csv"
    with source_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)

    output = tmp_path / "gemini-adults"
    summary = prepare_gemini_adult_dataset(
        analysis,
        source_manifest,
        frames,
        output,
    )
    assert summary["model_selection_counts"]["train"] == {"no": 2, "yes": 2}
    assert summary["natural_counts"]["test"] == {"no": 2, "yes": 2}
    assert summary["label_status"] == "gemini_consensus_candidates_not_ground_truth"
    assert (output / ".baby-monitor-yolo-adult-dataset").is_file()
