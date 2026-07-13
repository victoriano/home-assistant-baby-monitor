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

