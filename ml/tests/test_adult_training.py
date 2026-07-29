from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from baby_monitor_edge_ml.adult_training import (
    ADULT_ARTIFACT_MARKER,
    _adult_gate,
    _balanced_rows_by_location,
    adult_presence_target,
    assemble_reviewed_adult_artifact,
    repair_bracketed_adult_positives,
)
from baby_monitor_edge_ml.detail_training import DetailExample


def test_adult_presence_ignores_bed_phrases_but_keeps_visible_people() -> None:
    base = {"baby_present": True, "tags": []}
    assert (
        adult_presence_target(
            {
                **base,
                "description": "The baby is sleeping alone on an adult bed.",
            }
        )
        == 0
    )
    assert (
        adult_presence_target(
            {
                **base,
                "description": "An adult is sleeping in the adjacent bed.",
            }
        )
        == 1
    )
    assert (
        adult_presence_target(
            {
                **base,
                "description": "The baby holds an adult's hand.",
            }
        )
        == 1
    )
    assert (
        adult_presence_target(
            {
                **base,
                "description": "The room contains an adult bed but no adult is visible.",
            }
        )
        == 0
    )
    assert (
        adult_presence_target(
            {
                **base,
                "description": "The baby is partly occluded.",
                "tags": ["adult_nearby"],
            }
        )
        == 1
    )
    assert (
        adult_presence_target(
            {
                **base,
                "baby_present": False,
                "description": "No baby is visible; an adult is sleeping nearby.",
            }
        )
        == 1
    )
    assert (
        adult_presence_target(
            {
                **base,
                "adult_present": "unknown",
                "description": "An ambiguous shape may be an adult.",
            }
        )
        is None
    )
    assert adult_presence_target({**base, "adult_count": 2}) == 1


def test_adult_gate_requires_every_location() -> None:
    passing = {
        key: {
            "selective_accuracy": 0.98,
            "coverage": 0.7,
            "positive_precision": 0.95,
            "negative_precision": 0.98,
            "positive_recall": 0.6,
            "negative_recall": 0.8,
        }
        for key in ("overall", "granada", "madrid")
    }
    assert _adult_gate(passing)["passed"] is True
    failing = {
        **passing,
        "madrid": {
            **passing["madrid"],
            "positive_precision": 0.7,
        },
    }
    assert _adult_gate(failing)["passed"] is False


def test_only_bracketed_nearby_positive_repairs_an_omission() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)

    def example(minutes: int, class_name: str) -> DetailExample:
        return DetailExample(
            task="adult_presence",
            sample_id=f"sample-{minutes}",
            frame_id=f"frame-{minutes}",
            captured_at=(start + timedelta(minutes=minutes))
            .isoformat()
            .replace("+00:00", "Z"),
            location_id="one",
            relative_path=f"{minutes}.jpg",
            sha256=f"{minutes:064x}",
            crop_name="scene",
            crop_x0=0,
            crop_y0=0,
            crop_x1=1,
            crop_y1=1,
            class_name=class_name,
        )

    repaired, changes = repair_bracketed_adult_positives(
        (
            example(0, "yes"),
            example(5, "no"),
            example(10, "yes"),
            example(15, "no"),
        )
    )
    assert [item.class_name for item in repaired] == ["yes", "yes", "yes", "no"]
    assert changes == 1


def test_adult_training_is_balanced_inside_each_location() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)

    def example(location: str, index: int, class_name: str) -> DetailExample:
        return DetailExample(
            task="adult_presence",
            sample_id=f"{location}-{class_name}-{index}",
            frame_id=f"{location}-{class_name}-{index}",
            captured_at=(start + timedelta(minutes=index)).isoformat(),
            location_id=location,
            relative_path=f"{location}-{index}.jpg",
            sha256=f"{index:064x}",
            crop_name="scene",
            crop_x0=0,
            crop_y0=0,
            crop_x1=1,
            crop_y1=1,
            class_name=class_name,
        )

    rows = [
        *[example("one", index, "no") for index in range(6)],
        *[example("one", index, "yes") for index in range(2)],
        *[example("two", index, "no") for index in range(3)],
        *[example("two", index, "yes") for index in range(5)],
    ]
    balanced = _balanced_rows_by_location(
        rows,
        seed=7,
        max_minority_repeats=1,
    )
    counts = {
        (location, class_name): sum(
            item.location_id == location and item.class_name == class_name
            for item, _ in balanced
        )
        for location in ("one", "two")
        for class_name in ("no", "yes")
    }
    assert counts == {
        ("one", "no"): 2,
        ("one", "yes"): 2,
        ("two", "no"): 3,
        ("two", "yes"): 3,
    }


def test_reviewed_adult_assembly_updates_the_full_artifact(tmp_path) -> None:
    base = tmp_path / "base"
    (base / "models").mkdir(parents=True)
    (base / ".baby-monitor-yolo-artifacts").write_text("test\n")
    (base / "metadata.json").write_text(
        json.dumps(
            {
                "version": "base-v1",
                "format": "test",
                "tasks": {"presence": {"path": "models/presence.pt"}},
            }
        )
    )
    (base / "models" / "presence.pt").write_bytes(b"presence")
    (base / "report.json").write_text(
        json.dumps(
            {
                "artifact_version": "base-v1",
                "evaluation": {},
                "scope": {
                    "outputs": ["baby presence"],
                    "not_validated": [],
                    "unknown_policy": "unknown",
                },
                "limitations": [],
            }
        )
    )

    adult = tmp_path / "adult"
    (adult / "models").mkdir(parents=True)
    (adult / ADULT_ARTIFACT_MARKER).write_text("test\n")
    model = adult / "models" / "adult_presence.pt"
    model.write_bytes(b"adult")
    report = {"gate": {"passed": True}}
    (adult / "report.json").write_text(json.dumps(report))
    (adult / "report.md").write_text("# report\n")
    (adult / "high_confidence_errors.csv").write_text("frame_id\n")
    (adult / "metadata-fragment.json").write_text(
        json.dumps(
            {
                "tasks": {
                    "adult_presence": {
                        "path": "models/adult_presence.pt",
                        "bytes": model.stat().st_size,
                        "sha256": "replaced during assembly",
                        "positive_class": "yes",
                        "threshold": 0.5,
                        "thresholds": {
                            "overall": {"negative": 0.2, "positive": 0.8}
                        },
                        "crop": "scene",
                    }
                }
            }
        )
    )

    output = tmp_path / "assembled"
    result = assemble_reviewed_adult_artifact(base, adult, output)
    metadata = json.loads((output / "metadata.json").read_text())
    assembled_report = json.loads((output / "report.json").read_text())

    assert result["task"] == "adult_presence"
    assert "adult_presence" in metadata["tasks"]
    assert "visible adult presence" in assembled_report["scope"]["outputs"]
    assert assembled_report["evaluation"]["adult_presence"] == report
    assert (output / "adult-presence-report.md").is_file()
