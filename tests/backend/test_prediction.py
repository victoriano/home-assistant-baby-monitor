from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from baby_monitor.models import SleepEvent
from baby_monitor.prediction import build_sleep_plan


def _event(index: int, start: datetime, minutes: int, kind: str = "nap") -> SleepEvent:
    return SleepEvent(
        id=f"event-{index}",
        started_at=start,
        ended_at=start + timedelta(minutes=minutes),
        kind=kind,
        source="manual",
        location_id="granada",
    )


def test_prediction_restores_today_and_tomorrow_plans() -> None:
    now = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)
    history = [
        _event(1, datetime(2026, 7, 11, 8, 30, tzinfo=UTC), 45),
        _event(2, datetime(2026, 7, 11, 12, 15, tzinfo=UTC), 50),
        _event(3, datetime(2026, 7, 11, 19, 30, tzinfo=UTC), 600, "night"),
        _event(4, datetime(2026, 7, 12, 8, 45, tzinfo=UTC), 45),
        _event(5, datetime(2026, 7, 12, 12, 25, tzinfo=UTC), 50),
        _event(6, datetime(2026, 7, 12, 19, 35, tzinfo=UTC), 600, "night"),
    ]

    result = build_sleep_plan(
        history,
        birth_date=date(2025, 7, 1),
        timezone_name="Europe/Madrid",
        now=now,
    )

    assert [plan["date"] for plan in result["plans"]] == ["2026-07-13", "2026-07-14"]
    assert all(plan["nightPrediction"]["kind"] == "night" for plan in result["plans"])
    assert result["plans"][1]["dayNapPredictions"]
    assert result["nextSleepAt"] is not None
    assert 0 < result["confidence"] <= 1

    details = result["modelDetails"]
    assert details["baseline"] == {
        "ageBand": "12-17m",
        "birthDateKnown": True,
        "wakeWindowMinutes": 240,
        "expectedNaps": 2,
    }
    assert details["wakeWindows"]["valuesMinutes"] == [180.0, 385.0, 195.0, 175.0, 380.0]
    assert details["wakeWindows"]["medianMinutes"] == 195.0
    assert details["wakeWindows"]["historyWeight"] == 0.475
    assert details["wakeWindows"]["finalMinutes"] == result["wakeWindowMinutes"] == 219
    assert details["napDurations"]["medianMinutes"] == 47.5
    assert details["bedtimes"]["count"] == 2
    assert details["morningWakes"]["count"] == 2

    first_nap = result["plans"][0]["dayNapPredictions"][0]
    calculation = first_nap["calculation"]
    assert calculation["method"] == "wake_window"
    assert calculation["anchorType"] == "last_observed_wake"
    assert calculation["startSampleCount"] == 5
    anchor = datetime.fromisoformat(calculation["anchorAt"])
    base = datetime.fromisoformat(calculation["baseRecommendedStart"])
    assert base == anchor + timedelta(minutes=result["wakeWindowMinutes"])
    assert calculation["adjustmentReason"] == "past_window"

    night = result["plans"][0]["nightPrediction"]
    assert night["calculation"]["method"] == "bedtime_pattern"
    assert night["calculation"]["anchorType"] == "recent_bedtime_median"
    assert night["calculation"]["startSampleCount"] == details["bedtimes"]["count"]
    assert night["calculation"]["morningWakeSampleCount"] == details["morningWakes"]["count"]
