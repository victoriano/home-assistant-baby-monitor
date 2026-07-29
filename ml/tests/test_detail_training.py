from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import UTC, datetime, timedelta

from baby_monitor_edge_ml.detail_training import (
    DETAIL_DATASET_MARKER,
    DetailExample,
    _label_classes,
    _task_gate,
    normalize_body_position,
    rebalance_detail_validation,
    stratified_group_split,
)
from baby_monitor_edge_ml.yolo_training import _mouth_rect, _pose_body_rect


def _example(
    *,
    task: str,
    location: str,
    day: int,
    index: int,
    class_name: str,
) -> DetailExample:
    captured_at = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=day, minutes=index)
    return DetailExample(
        task=task,
        sample_id=f"{task}-{location}-{day}-{index}:crib",
        frame_id=f"{task}-{location}-{day}-{index}",
        captured_at=captured_at.isoformat().replace("+00:00", "Z"),
        location_id=location,
        relative_path=f"{index}.jpg",
        sha256=f"{index:064x}",
        crop_name="crib",
        crop_x0=0.1,
        crop_y0=0.1,
        crop_x1=0.9,
        crop_y1=0.9,
        class_name=class_name,
    )


def test_normalize_body_position_rejects_free_text_shortcuts() -> None:
    assert normalize_body_position("supine, arms raised") == "back"
    assert normalize_body_position("prone (on stomach)") == "belly"
    assert normalize_body_position("side_left") == "side"
    assert normalize_body_position("lying on side, knees bent") == "side"
    assert normalize_body_position("supine, arms out to the sides") == "back"
    assert normalize_body_position("being held by an adult") is None
    assert normalize_body_position("lying down") is None


def test_mouth_labels_require_an_unoccluded_mouth() -> None:
    base = {
        "baby_present": True,
        "face_visible": "yes",
        "pacifier": "no",
        "mouth_open": "yes",
        "head_side": "left",
        "body_position": "supine",
        "description": "The mouth is visibly open.",
        "tags": [],
    }
    assert _label_classes(base) == {
        "head_side": "left",
        "body_position": "back",
        "mouth_open": "yes",
    }
    assert "mouth_open" not in _label_classes({**base, "pacifier": "yes"})
    assert "mouth_open" not in _label_classes(
        {**base, "description": "An adult is bottle-feeding the baby."}
    )


def test_detail_split_keeps_days_together_and_preserves_rare_labels() -> None:
    examples: list[DetailExample] = []
    for location in ("one", "two"):
        for day in range(10):
            for index in range(6):
                class_name = "yes" if index == 0 and day % 2 == 0 else "no"
                examples.append(
                    _example(
                        task="mouth_open",
                        location=location,
                        day=day,
                        index=index,
                        class_name=class_name,
                    )
                )
    split = stratified_group_split(tuple(examples), seed=17)
    groups: dict[tuple[str, str, str], set[str]] = {}
    for example in split:
        groups.setdefault((example.task, example.location_id, example.day), set()).add(
            example.split
        )
    assert all(len(values) == 1 for values in groups.values())
    counts = Counter((example.split, example.class_name) for example in split)
    assert counts[("validation", "yes")] > 0
    assert counts[("test", "yes")] > 0
    assert counts[("train", "yes")] > 0


def test_pose_subcrops_stay_inside_the_selected_regions() -> None:
    body = _pose_body_rect(
        box=[0.2, 0.2, 0.8, 0.8],
        roi=(0.25, 0.1, 0.9, 0.75),
    )
    assert body == (0.25, 0.128, 0.8720000000000001, 0.75)
    mouth = _mouth_rect((0.1, 0.2, 0.5, 0.6))
    assert mouth == (0.14800000000000002, 0.312, 0.452, 0.576)


def test_detail_gate_requires_every_camera_and_class_to_pass() -> None:
    classes = {
        class_name: {
            "predicted": 12,
            "precision": 1.0,
        }
        for class_name in ("back", "left", "right")
    }
    overall = {
        "decisions": 60,
        "coverage": 0.6,
        "selective_accuracy": 0.95,
        "classes": classes,
    }
    location = {
        "decisions": 30,
        "coverage": 0.6,
        "selective_accuracy": 0.9,
        "classes": classes,
    }
    report = {
        "overall": overall,
        "one": location,
        "two": location,
    }
    assert _task_gate("head_side", report)["passed"] is True

    failed_camera = {
        **report,
        "two": {
            **location,
            "selective_accuracy": 0.7,
        },
    }
    assert _task_gate("head_side", failed_camera)["passed"] is False

    failed_class = {
        **report,
        "overall": {
            **overall,
            "classes": {
                **classes,
                "left": {"predicted": 2, "precision": 1.0},
            },
        },
    }
    assert _task_gate("head_side", failed_class)["passed"] is False


def test_rebalance_validation_keeps_natural_index_and_balances_model_selection(
    tmp_path,
) -> None:
    dataset = tmp_path / "details"
    dataset.mkdir()
    (dataset / DETAIL_DATASET_MARKER).write_text("private\n", encoding="utf-8")
    rows = []
    task_classes = {
        "head_side": ("back", "left", "right"),
        "body_position": ("back", "belly", "side"),
        "mouth_open": ("no", "yes"),
    }
    for task, classes in task_classes.items():
        for class_name in classes:
            count = 10 if class_name in {"back", "no"} else 2
            for index in range(count):
                crop = dataset / "crops" / f"{task}-{class_name}-{index}.jpg"
                crop.parent.mkdir(exist_ok=True)
                crop.write_bytes(b"crop")
                rows.append(
                    {
                        "task": task,
                        "sample_id": f"{task}-{class_name}-{index}",
                        "frame_id": f"frame-{task}-{class_name}-{index}",
                        "captured_at": "2026-01-01T00:00:00Z",
                        "location_id": "one",
                        "split": "validation",
                        "class_name": class_name,
                        "crop_name": "head",
                        "crop_path": str(crop.relative_to(dataset)),
                    }
                )
    index_path = dataset / "index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    (dataset / "summary.json").write_text("{}\n", encoding="utf-8")
    original_index = index_path.read_bytes()

    counts = rebalance_detail_validation(
        dataset,
        seed=7,
        max_minority_repeats=3,
    )

    assert index_path.read_bytes() == original_index
    assert counts["head_side"] == {"back": 6, "left": 6, "right": 6}
    assert counts["body_position"] == {"back": 6, "belly": 6, "side": 6}
    assert counts["mouth_open"] == {"no": 6, "yes": 6}
    assert len(list((dataset / "mouth_open" / "val" / "yes").iterdir())) == 6
    summary = json.loads((dataset / "summary.json").read_text(encoding="utf-8"))
    assert (
        summary["model_selection_validation"]["natural_distribution_preserved_in"]
        == "index.csv"
    )
