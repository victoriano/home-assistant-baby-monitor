from __future__ import annotations

import csv
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from PIL import Image

from baby_monitor_edge_ml.dataset import FrameExample, write_manifest
from baby_monitor_edge_ml.yolo_training import (
    YOLO_DATASET_MARKER,
    PoseHeadConfig,
    _deployment_gate,
    _include_pose_abstentions,
    _select_pose_head_rect,
    prepare_yolo_dataset,
)


def _example(
    frames_dir: Path,
    *,
    index: int,
    captured_at: datetime,
    split: str,
    presence: int,
    awake: int = 0,
    awake_mask: int = 0,
    pacifier: int = 0,
    pacifier_mask: int = 0,
) -> FrameExample:
    relative_path = f"{index}.jpg"
    Image.new("RGB", (160, 90), color=(index * 13 % 255, 40, 80)).save(frames_dir / relative_path)
    return FrameExample(
        sample_id=f"frame-{index}:sleep_area",
        frame_id=f"frame-{index}",
        captured_at=captured_at.isoformat().replace("+00:00", "Z"),
        location_id="home",
        relative_path=relative_path,
        sha256=f"{index:064x}",
        provider="teacher",
        model="vision",
        crop_name="sleep_area",
        crop_x0=0.0,
        crop_y0=0.1,
        crop_x1=0.85,
        crop_y1=0.95,
        split=split,
        presence_target=presence,
        presence_mask=1,
        awake_target=awake,
        awake_mask=awake_mask,
        pacifier_target=pacifier,
        pacifier_mask=pacifier_mask,
        confidence=0.95,
        teacher_state="awake" if awake else "asleep",
        teacher_pacifier="yes" if pacifier else "no",
        face_visible="yes" if presence else "unknown",
        sleep_surface="crib" if presence else "unknown",
        in_crib=bool(presence),
    )


def _manifest(tmp_path: Path) -> tuple[Path, Path]:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    started = datetime(2026, 1, 1, 12, tzinfo=UTC)
    examples = [
        _example(
            frames_dir,
            index=0,
            captured_at=started,
            split="train",
            presence=1,
            awake=0,
            awake_mask=1,
            pacifier=0,
            pacifier_mask=1,
        ),
        _example(
            frames_dir,
            index=1,
            captured_at=started + timedelta(minutes=5),
            split="train",
            presence=1,
            awake=0,
            awake_mask=1,
            pacifier=0,
            pacifier_mask=1,
        ),
        _example(
            frames_dir,
            index=2,
            captured_at=started + timedelta(minutes=10),
            split="train",
            presence=1,
            awake=1,
            awake_mask=1,
            pacifier=1,
            pacifier_mask=1,
        ),
        _example(
            frames_dir,
            index=3,
            captured_at=started + timedelta(minutes=15),
            split="train",
            presence=1,
            awake=1,
            awake_mask=1,
            pacifier=1,
            pacifier_mask=1,
        ),
        _example(
            frames_dir,
            index=4,
            captured_at=started + timedelta(minutes=20),
            split="train",
            presence=0,
        ),
        _example(
            frames_dir,
            index=5,
            captured_at=started + timedelta(minutes=25),
            split="train",
            presence=0,
        ),
        # This isolated detail label is deliberately excluded from YOLO training.
        _example(
            frames_dir,
            index=6,
            captured_at=started + timedelta(days=1),
            split="train",
            presence=1,
            awake=1,
            awake_mask=1,
            pacifier=1,
            pacifier_mask=1,
        ),
    ]
    for split_index, split in enumerate(("validation", "test"), start=1):
        base = started + timedelta(days=split_index + 1)
        examples.extend(
            (
                _example(
                    frames_dir,
                    index=10 + split_index * 10,
                    captured_at=base,
                    split=split,
                    presence=1,
                    awake=0,
                    awake_mask=1,
                    pacifier=0,
                    pacifier_mask=1,
                ),
                _example(
                    frames_dir,
                    index=11 + split_index * 10,
                    captured_at=base + timedelta(minutes=5),
                    split=split,
                    presence=1,
                    awake=1,
                    awake_mask=1,
                    pacifier=1,
                    pacifier_mask=1,
                ),
                _example(
                    frames_dir,
                    index=12 + split_index * 10,
                    captured_at=base + timedelta(minutes=10),
                    split=split,
                    presence=0,
                ),
            )
        )
    manifest = tmp_path / "manifest.csv"
    write_manifest(tuple(examples), manifest)
    return manifest, frames_dir


def test_prepare_yolo_dataset_balances_training_and_preserves_test_split(
    tmp_path: Path,
) -> None:
    manifest, frames_dir = _manifest(tmp_path)
    output = tmp_path / "yolo"

    summary = prepare_yolo_dataset(
        manifest,
        frames_dir,
        output,
        image_size=96,
        max_minority_repeats=2,
    )

    assert (output / YOLO_DATASET_MARKER).is_file()
    assert summary["tasks"]["awake"]["excluded_inconsistent_train_labels"] == 1
    assert summary["tasks"]["pacifier"]["excluded_inconsistent_train_labels"] == 1
    for task in ("presence", "awake", "pacifier"):
        counts = summary["tasks"][task]["splits"]["train"]["classes"]
        assert len(set(counts.values())) == 1
        assert (output / task / "train").is_dir()
        assert (output / task / "val").is_dir()
        assert (output / task / "test").is_dir()
    with (output / "index.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["split"] for row in rows} == {"train", "validation", "test"}
    assert all((output / row["crop_path"]).is_file() for row in rows)
    with Image.open(output / rows[0]["crop_path"]) as crop:
        assert crop.size == (96, 96)


def test_prepare_yolo_dataset_only_overwrites_its_marked_directory(tmp_path: Path) -> None:
    manifest, frames_dir = _manifest(tmp_path)
    output = tmp_path / "yolo"
    prepare_yolo_dataset(manifest, frames_dir, output, image_size=96)

    rebuilt = prepare_yolo_dataset(
        manifest,
        frames_dir,
        output,
        image_size=96,
        overwrite=True,
    )

    assert rebuilt["unique_crops"] > 0
    unmarked = tmp_path / "unmarked"
    unmarked.mkdir()
    (unmarked / "keep.txt").write_text("user data", encoding="utf-8")
    with pytest.raises(ValueError, match="unmarked"):
        prepare_yolo_dataset(
            manifest,
            frames_dir,
            unmarked,
            image_size=96,
            overwrite=True,
        )


def test_deployment_gate_requires_high_precision_in_every_location() -> None:
    passing = {
        task: {
            "overall": {
                "selective_accuracy": 0.99,
                "coverage": 0.9,
                "positive_precision": 0.99,
                "negative_precision": 0.99,
                "positive_recall": 0.9,
                "negative_recall": 0.9,
            },
            "home": {
                "selective_accuracy": 0.99,
                "coverage": 0.9,
                "positive_precision": 0.99,
                "negative_precision": 0.99,
                "positive_recall": 0.9,
                "negative_recall": 0.9,
            },
        }
        for task in ("presence", "awake", "pacifier")
    }

    assert _deployment_gate(passing)["automated_metrics_passed"] is True
    passing["pacifier"]["home"]["coverage"] = 0.1
    gate = _deployment_gate(passing)
    assert gate["automated_metrics_passed"] is False
    assert gate["tasks"]["pacifier"]["locations"]["home"]["passed"] is False


def test_pose_head_selector_prefers_smaller_person_inside_monitored_roi() -> None:
    adult_points = [[0.2, 0.2], [0.18, 0.2], [0.22, 0.2], [0.17, 0.21], [0.23, 0.21]]
    baby_points = [[0.7, 0.6], [0.68, 0.6], [0.72, 0.6], [0.67, 0.61], [0.73, 0.61]]

    rect = _select_pose_head_rect(
        boxes=[[0.05, 0.05, 0.5, 0.95], [0.6, 0.5, 0.82, 0.82]],
        box_confidences=[0.95, 0.8],
        keypoints=[adult_points, baby_points],
        keypoint_confidences=[[0.9] * 5, [0.9] * 5],
        roi=(0.55, 0.45, 0.9, 0.9),
        width=1280,
        height=720,
        config=PoseHeadConfig(),
    )

    assert rect is not None
    left, top, right, bottom = rect
    assert left < 0.7 < right
    assert top < 0.6 < bottom
    assert right - left == pytest.approx((bottom - top) * 720 / 1280)


def test_pose_misses_count_as_abstentions_and_reduce_effective_recall() -> None:
    detected_metrics = {
        "home": {
            "samples": 10,
            "positive": 3,
            "negative": 7,
            "decisions": 8,
            "abstained": 2,
            "coverage": 0.8,
            "true_positive": 2,
            "true_negative": 5,
            "false_positive": 1,
            "false_negative": 0,
            "selective_accuracy": 0.875,
            "positive_precision": 2 / 3,
            "negative_precision": 1.0,
            "positive_recall": 2 / 3,
            "negative_recall": 5 / 7,
        }
    }
    task_summary = {
        "pose_detection": {
            "test": {
                "home": {
                    "eligible": 20,
                    "detected": 10,
                    "eligible_negative": 15,
                    "eligible_positive": 5,
                    "detected_negative": 7,
                    "detected_positive": 3,
                }
            }
        }
    }

    adjusted = _include_pose_abstentions(detected_metrics, task_summary, "test")["home"]

    assert adjusted["samples"] == 20
    assert adjusted["abstained"] == 12
    assert adjusted["coverage"] == 0.4
    assert adjusted["localizer_coverage"] == 0.5
    assert adjusted["classifier_coverage"] == 0.8
    assert adjusted["positive_recall"] == 0.4
    assert adjusted["negative_recall"] == pytest.approx(1 / 3)
