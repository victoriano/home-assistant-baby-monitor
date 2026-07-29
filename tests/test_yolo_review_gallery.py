from __future__ import annotations

from pathlib import Path

import pytest

from tools.yolo_review_gallery import (
    MARKER,
    _decision,
    _overall_outcome,
    _prepare_output,
    _reference,
)


def test_reference_uses_any_positive_roi_for_frame_presence() -> None:
    rows = [
        {
            "frame_id": "frame",
            "crop_name": "family_bed",
            "presence_mask": "1",
            "presence_target": "0",
            "awake_mask": "0",
            "awake_target": "0",
            "pacifier_mask": "0",
            "pacifier_target": "0",
            "sleep_surface": "crib",
            "provider": "gemini",
            "model": "flash-lite",
            "confidence": "0.97",
            "face_visible": "yes",
        },
        {
            "frame_id": "frame",
            "crop_name": "crib",
            "presence_mask": "1",
            "presence_target": "1",
            "awake_mask": "1",
            "awake_target": "0",
            "pacifier_mask": "1",
            "pacifier_target": "1",
            "sleep_surface": "crib",
        },
    ]

    reference = _reference(rows)

    assert reference["presence"] == "present"
    assert reference["awake"] == "asleep"
    assert reference["pacifier"] == "yes"
    assert reference["target_roi"] == "crib"


@pytest.mark.parametrize(
    ("task", "score", "expected"),
    [
        ("presence", 0.02, "absent"),
        ("presence", 0.5, "unknown"),
        ("presence", 0.99, "present"),
        ("awake", 0.01, "asleep"),
        ("awake", 0.5, "unknown"),
        ("awake", 0.95, "awake"),
        ("pacifier", None, "not_run"),
    ],
)
def test_decision_applies_dual_abstention_thresholds(
    task: str,
    score: float | None,
    expected: str,
) -> None:
    assert _decision(task, score, (0.02 if task == "presence" else 0.01, 0.95)) == expected


def test_overall_outcome_prioritizes_mismatch_then_abstention() -> None:
    assert (
        _overall_outcome(
            {
                "presence": {"outcome": "match"},
                "awake": {"outcome": "abstain"},
                "pacifier": {"outcome": "mismatch"},
            }
        )
        == "mismatch"
    )


def test_output_overwrite_requires_gallery_marker(tmp_path: Path) -> None:
    output = tmp_path / "review"
    output.mkdir()
    (output / "keep.txt").write_text("private", encoding="utf-8")

    with pytest.raises(ValueError, match="unmarked"):
        _prepare_output(output, overwrite=True)

    (output / MARKER).write_text("marked", encoding="utf-8")
    _prepare_output(output, overwrite=True)

    assert (output / MARKER).is_file()
    assert not (output / "keep.txt").exists()
