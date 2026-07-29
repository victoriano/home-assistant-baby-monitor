from __future__ import annotations

import numpy as np

from baby_monitor_edge_ml.metrics import (
    binary_metrics,
    select_abstention_thresholds,
    select_threshold,
    selective_binary_metrics,
)


def test_binary_metrics_reports_confusion_matrix() -> None:
    metrics = binary_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.8, 0.9, 0.2]),
        0.5,
    )

    assert metrics["true_positive"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["balanced_accuracy"] == 0.5


def test_select_threshold_finds_separating_value() -> None:
    threshold, metrics = select_threshold(
        np.array([0, 0, 1, 1]),
        np.array([0.05, 0.2, 0.7, 0.95]),
    )

    assert 0.2 < threshold <= 0.7
    assert metrics["f1"] == 1.0


def test_selective_metrics_report_accuracy_and_abstention_coverage() -> None:
    metrics = selective_binary_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0.05, 0.45, 0.55, 0.95]),
        0.2,
        0.8,
    )

    assert metrics["decisions"] == 2
    assert metrics["abstained"] == 2
    assert metrics["coverage"] == 0.5
    assert metrics["selective_accuracy"] == 1.0


def test_select_abstention_thresholds_targets_precision_on_both_classes() -> None:
    negative, positive, metrics = select_abstention_thresholds(
        np.array([0] * 10 + [1] * 10),
        np.array(
            [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.4, 0.55, 0.6]
            + [0.4, 0.45, 0.7, 0.8, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99]
        ),
        target_precision=0.95,
        minimum_decisions=3,
    )

    assert negative < positive
    assert metrics["positive_precision"] == 1.0
    assert metrics["negative_precision"] == 1.0
    assert metrics["coverage"] >= 0.5


def test_select_abstention_thresholds_keeps_imbalanced_class_bounds_ordered() -> None:
    negative, positive, metrics = select_abstention_thresholds(
        np.array([0] * 57 + [1] * 10),
        np.array([0.01] * 56 + [0.91] + [0.6, 0.7] + [0.99] * 8),
        target_precision=0.95,
        minimum_decisions=3,
    )

    assert negative < positive
    assert positive > 0.91
    assert metrics["positive_precision"] == 1.0
    assert metrics["negative_precision"] >= 0.95
