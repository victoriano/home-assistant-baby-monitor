from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from .models import VisionLabel

DEFAULT_FRAME_BUCKET_MINUTES = 5
MAX_FRAME_BUCKET_MINUTES = 10


def _bucket_minutes(rows: list[tuple[datetime, VisionLabel]], index: int) -> int:
    current = rows[index][0]
    following = rows[index + 1][0] if index + 1 < len(rows) else None
    if following is not None and following > current:
        minutes = round((following - current).total_seconds() / 60)
    else:
        minutes = DEFAULT_FRAME_BUCKET_MINUTES
    return max(1, min(MAX_FRAME_BUCKET_MINUTES, int(minutes)))


def _segments(
    values: dict[str, int],
    label_for_key: Callable[[str], str],
    color_for_key: Callable[[str], str],
) -> list[dict[str, Any]]:
    total = sum(int(value or 0) for value in values.values())
    if total <= 0:
        return []
    return [
        {
            "key": key,
            "label": label_for_key(key),
            "minutes": int(minutes),
            "percent": round((int(minutes) / total) * 100),
            "color": color_for_key(key),
        }
        for key, minutes in sorted(values.items(), key=lambda item: int(item[1] or 0), reverse=True)
        if int(minutes or 0) > 0
    ]


def _binary_metric(
    positive: int,
    negative: int,
    positive_label: str,
    negative_label: str,
    positive_color: str,
    negative_color: str,
) -> dict[str, Any]:
    values = {"positive": positive, "negative": negative}
    return {
        "positive_minutes": positive,
        "negative_minutes": negative,
        "total_minutes": positive + negative,
        "segments": _segments(
            values,
            lambda key: positive_label if key == "positive" else negative_label,
            lambda key: positive_color if key == "positive" else negative_color,
        ),
    }


def _category_metric(values: dict[str, int], colors: dict[str, str]) -> dict[str, Any]:
    return {
        "total_minutes": sum(values.values()),
        "segments": _segments(
            values,
            lambda key: key.replace("_", " "),
            lambda key: colors.get(key, _stable_color(key)),
        ),
    }


def _stable_color(key: str) -> str:
    palette = ["#7770f7", "#f7f5ff", "#79e2ad", "#ffbe55", "#ee8c59", "#9d94ff", "#46409a"]
    stable = sum((index + 1) * ord(char) for index, char in enumerate(key))
    return palette[stable % len(palette)]


def vision_summary(
    rows: list[tuple[datetime, VisionLabel]],
    start: datetime,
    end: datetime,
    timezone_name: str,
) -> dict[str, Any]:
    timezone = ZoneInfo(timezone_name)
    observed_minutes = 0
    visible_minutes = 0
    visible_samples = 0
    pacifier_yes = 0
    pacifier_no = 0
    mouth_yes = 0
    mouth_no = 0
    head_minutes: dict[str, int] = defaultdict(int)
    clothing_minutes: dict[str, int] = defaultdict(int)
    daily: dict[str, dict[str, Any]] = {}

    for index, (captured_at, label) in enumerate(rows):
        minutes = _bucket_minutes(rows, index)
        observed_minutes += minutes
        date_key = captured_at.astimezone(timezone).date().isoformat()
        day = daily.setdefault(
            date_key,
            {
                "date": date_key,
                "sample_count": 0,
                "visible_sample_count": 0,
                "observed_minutes": 0,
                "visible_minutes": 0,
                "pacifier_minutes": 0,
                "mouth_open_minutes": 0,
            },
        )
        day["sample_count"] += 1
        day["observed_minutes"] += minutes
        # Family-bed observations are first-class visual evidence too. Keep
        # legacy unknown/in_crib-null labels visible exactly as before.
        visible = label.baby_present and label.resolved_sleep_surface() != "other"
        if not visible:
            continue
        visible_samples += 1
        visible_minutes += minutes
        day["visible_sample_count"] += 1
        day["visible_minutes"] += minutes
        if label.pacifier == "yes":
            pacifier_yes += minutes
            day["pacifier_minutes"] += minutes
        elif label.pacifier == "no":
            pacifier_no += minutes
        if label.mouth_open == "yes":
            mouth_yes += minutes
            day["mouth_open_minutes"] += minutes
        elif label.mouth_open == "no":
            mouth_no += minutes
        if label.head_side != "unknown":
            head_minutes[label.head_side] += minutes
        for clothing in label.clothing_items:
            if clothing != "unknown":
                clothing_minutes[clothing] += minutes

    return {
        "range": {"start": start.isoformat(), "end": end.isoformat()},
        "sample_count": len(rows),
        "visible_sample_count": visible_samples,
        "observed_minutes": observed_minutes,
        "visible_minutes": visible_minutes,
        "metrics": {
            "pacifier": _binary_metric(pacifier_yes, pacifier_no, "pacifier", "no pacifier", "#79e2ad", "#46409a"),
            "mouth_open": _binary_metric(mouth_yes, mouth_no, "mouth open", "mouth closed", "#ffbe55", "#46409a"),
            "head_side": _category_metric(
                head_minutes,
                {"left": "#7770f7", "right": "#f7f5ff", "back": "#79e2ad", "face_down": "#ffbe55"},
            ),
            "clothing": _category_metric(clothing_minutes, {}),
        },
        "daily": [daily[key] for key in sorted(daily)],
    }
