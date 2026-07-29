from __future__ import annotations

from typing import Any

import numpy as np


def binary_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    truth = np.asarray(y_true, dtype=np.int8).reshape(-1)
    scores = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    if truth.size != scores.size or not truth.size:
        raise ValueError("truth and probabilities must be non-empty and have equal size")
    predicted = (scores >= threshold).astype(np.int8)
    true_positive = int(np.sum((truth == 1) & (predicted == 1)))
    true_negative = int(np.sum((truth == 0) & (predicted == 0)))
    false_positive = int(np.sum((truth == 0) & (predicted == 1)))
    false_negative = int(np.sum((truth == 1) & (predicted == 0)))

    def ratio(numerator: int | float, denominator: int | float) -> float | None:
        return float(numerator / denominator) if denominator else None

    precision = ratio(true_positive, true_positive + false_positive)
    recall = ratio(true_positive, true_positive + false_negative)
    specificity = ratio(true_negative, true_negative + false_positive)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    balanced_accuracy = (recall + specificity) / 2 if recall is not None and specificity is not None else None
    return {
        "samples": int(truth.size),
        "positive": int(np.sum(truth == 1)),
        "negative": int(np.sum(truth == 0)),
        "threshold": float(threshold),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "accuracy": float(np.mean(truth == predicted)),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
    }


def select_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict[str, Any]]:
    """Choose a validation threshold using both F1 and balanced accuracy."""

    best: tuple[float, float, dict[str, Any]] | None = None
    for threshold in np.linspace(0.05, 0.95, 91):
        metrics = binary_metrics(y_true, probabilities, float(threshold))
        f1 = metrics["f1"]
        balanced = metrics["balanced_accuracy"]
        if f1 is None or balanced is None:
            continue
        score = (f1 + balanced) / 2
        candidate = (score, -abs(float(threshold) - 0.5), metrics)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        raise ValueError("threshold selection requires positive and negative validation examples")
    return float(best[2]["threshold"]), best[2]


def selective_binary_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    negative_threshold: float,
    positive_threshold: float,
) -> dict[str, Any]:
    """Measure a binary classifier that abstains between two thresholds."""

    truth = np.asarray(y_true, dtype=np.int8).reshape(-1)
    scores = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    if truth.size != scores.size or not truth.size:
        raise ValueError("truth and probabilities must be non-empty and have equal size")
    if not 0 <= negative_threshold < positive_threshold <= 1:
        raise ValueError("selective thresholds must be ordered within [0, 1]")
    negative = scores <= negative_threshold
    positive = scores >= positive_threshold
    decided = negative | positive
    true_positive = int(np.sum((truth == 1) & positive))
    true_negative = int(np.sum((truth == 0) & negative))
    false_positive = int(np.sum((truth == 0) & positive))
    false_negative = int(np.sum((truth == 1) & negative))
    decisions = int(np.sum(decided))

    def ratio(numerator: int | float, denominator: int | float) -> float | None:
        return float(numerator / denominator) if denominator else None

    return {
        "samples": int(truth.size),
        "positive": int(np.sum(truth == 1)),
        "negative": int(np.sum(truth == 0)),
        "negative_threshold": float(negative_threshold),
        "positive_threshold": float(positive_threshold),
        "decisions": decisions,
        "abstained": int(truth.size - decisions),
        "coverage": float(decisions / truth.size),
        "selective_accuracy": ratio(true_positive + true_negative, decisions),
        "positive_precision": ratio(true_positive, true_positive + false_positive),
        "negative_precision": ratio(true_negative, true_negative + false_negative),
        "positive_recall": ratio(true_positive, int(np.sum(truth == 1))),
        "negative_recall": ratio(true_negative, int(np.sum(truth == 0))),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def select_abstention_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    target_precision: float = 0.95,
    minimum_decisions: int = 10,
) -> tuple[float, float, dict[str, Any]]:
    """Calibrate high-precision positive/negative decisions on validation data."""

    truth = np.asarray(y_true, dtype=np.int8).reshape(-1)
    scores = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    if truth.size != scores.size or not truth.size:
        raise ValueError("truth and probabilities must be non-empty and have equal size")
    if not 0.5 <= target_precision <= 1:
        raise ValueError("target precision must be between 0.5 and 1")
    if minimum_decisions < 1:
        raise ValueError("minimum decisions must be positive")

    required = min(minimum_decisions, max(1, truth.size // 10))
    # Keep candidate and inference dtypes identical so a score lying exactly
    # on a selected boundary is treated the same during calibration and use.
    candidates = np.unique(np.concatenate((np.linspace(0.01, 0.99, 99), scores)).astype(np.float32))

    def positive_candidates() -> list[tuple[float, int, float]]:
        choices: list[tuple[float, int, float]] = []
        for threshold in candidates:
            selected = scores >= threshold
            count = int(np.sum(selected))
            if count < required:
                continue
            precision = float(np.mean(truth[selected] == 1))
            choices.append((float(threshold), count, precision))
        return choices

    def negative_candidates() -> list[tuple[float, int, float]]:
        choices: list[tuple[float, int, float]] = []
        for threshold in candidates:
            selected = scores <= threshold
            count = int(np.sum(selected))
            if count < required:
                continue
            precision = float(np.mean(truth[selected] == 0))
            choices.append((float(threshold), count, precision))
        return choices

    ordered_pairs = [
        (negative, positive)
        for negative in negative_candidates()
        for positive in positive_candidates()
        if negative[0] < positive[0]
    ]
    eligible_pairs = [
        pair for pair in ordered_pairs if pair[0][2] >= target_precision and pair[1][2] >= target_precision
    ]
    if eligible_pairs:
        negative, positive = max(
            eligible_pairs,
            key=lambda pair: (
                pair[0][1] + pair[1][1],
                min(pair[0][2], pair[1][2]),
                pair[0][2] + pair[1][2],
                pair[1][0] - pair[0][0],
            ),
        )
        negative_threshold = negative[0]
        positive_threshold = positive[0]
    elif ordered_pairs:
        negative, positive = max(
            ordered_pairs,
            key=lambda pair: (
                min(pair[0][2], pair[1][2]),
                pair[0][2] + pair[1][2],
                pair[0][1] + pair[1][1],
                pair[1][0] - pair[0][0],
            ),
        )
        negative_threshold = negative[0]
        positive_threshold = positive[0]
    else:
        center, _ = select_threshold(truth, scores)
        negative_threshold = max(0.0, center - 0.01)
        positive_threshold = min(1.0, center + 0.01)
    metrics = selective_binary_metrics(
        truth,
        scores,
        negative_threshold,
        positive_threshold,
    )
    metrics["target_precision"] = target_precision
    metrics["minimum_calibration_decisions"] = required
    return negative_threshold, positive_threshold, metrics
