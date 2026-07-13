from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, date, datetime, time, timedelta
from statistics import median
from typing import Any
from zoneinfo import ZoneInfo

from .models import SleepEvent


def _age_baseline(birth_date: date | None, today: date) -> tuple[int, int, str]:
    """Return wake-window minutes, expected naps and the age band."""

    if birth_date is None:
        return 180, 2, "unknown"
    age_days = max(0, (today - birth_date).days)
    if age_days < 90:
        return 105, 5, "0-3m"
    if age_days < 180:
        return 135, 4, "4-5m"
    if age_days < 270:
        return 165, 3, "6-8m"
    if age_days < 365:
        return 195, 2, "9-11m"
    if age_days < 548:
        return 240, 2, "12-17m"
    if age_days < 730:
        return 300, 1, "18-23m"
    return 360, 1, "24m+"


def _duration_minutes(event: SleepEvent) -> float | None:
    if event.ended_at is None:
        return None
    return max(0.0, (event.ended_at - event.started_at).total_seconds() / 60)


def _kind(event: SleepEvent) -> str:
    if event.kind in {"nap", "night"}:
        return event.kind
    duration = _duration_minutes(event) or 0
    local_start = event.started_at
    if duration >= 180 or local_start.hour >= 18 or local_start.hour < 5:
        return "night"
    return "nap"


def _minute_of_day(value: datetime) -> int:
    return value.hour * 60 + value.minute


def _local_at(day: date, minute: int, timezone: ZoneInfo) -> datetime:
    normalized = minute % (24 * 60)
    return datetime.combine(
        day,
        time(hour=normalized // 60, minute=normalized % 60),
        tzinfo=timezone,
    )


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _recent_median(values: Iterable[float], fallback: int, *, minimum: int, maximum: int) -> int:
    sample = list(values)
    if not sample:
        return fallback
    return round(max(minimum, min(maximum, float(median(sample)))))


def _target(
    *,
    kind: str,
    label: str,
    recommended: datetime,
    duration_minutes: int,
    margin_minutes: int,
    confidence: float,
    explanation: str,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "label": label,
        "recommendedStart": _iso(recommended),
        "windowStart": _iso(recommended - timedelta(minutes=margin_minutes)),
        "windowEnd": _iso(recommended + timedelta(minutes=margin_minutes)),
        "durationMinutes": duration_minutes,
        "confidence": confidence,
        "explanation": explanation,
    }


def build_sleep_plan(
    events: Iterable[SleepEvent],
    *,
    birth_date: date | None,
    timezone_name: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build the two-day plan used by the original Esteban rhythm view.

    The model intentionally remains local and deterministic. It blends age
    guidance with the family's recent wake windows, nap duration, bedtime and
    morning wake history. The output contains complete plans for today and
    tomorrow instead of only one next-sleep timestamp.
    """

    timezone = ZoneInfo(timezone_name)
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    current_local = current.astimezone(timezone)
    today = current_local.date()
    baseline_wake, expected_naps, age_band = _age_baseline(birth_date, today)

    closed = sorted(
        (
            event
            for event in events
            if event.kind != "awake" and event.ended_at is not None and event.ended_at > event.started_at
        ),
        key=lambda event: event.started_at,
    )
    recent = closed[-80:]
    wake_intervals: list[float] = []
    for previous, following in zip(recent, recent[1:], strict=False):
        assert previous.ended_at is not None
        minutes = (following.started_at - previous.ended_at).total_seconds() / 60
        if 15 <= minutes <= 12 * 60:
            wake_intervals.append(minutes)

    interval_sample = wake_intervals[-12:]
    if len(interval_sample) >= 3:
        learned_wake = float(median(interval_sample))
        history_weight = min(0.70, 0.30 + len(interval_sample) * 0.035)
        wake_minutes = round(baseline_wake * (1 - history_weight) + learned_wake * history_weight)
        confidence = min(0.92, 0.57 + len(interval_sample) * 0.025)
        reason = "Blended age guidance with recent wake intervals"
    else:
        wake_minutes = baseline_wake
        confidence = 0.58 if birth_date is not None else 0.46
        reason = "Using age guidance while recent wake intervals are still being learned"

    if len(interval_sample) >= 4:
        center = float(median(interval_sample))
        deviation = float(median(abs(value - center) for value in interval_sample))
        margin_minutes = round(max(25, min(75, deviation * 1.7)))
    else:
        margin_minutes = 35

    nap_durations = [
        duration
        for event in recent[-30:]
        if _kind(event) == "nap" and (duration := _duration_minutes(event)) is not None and 15 <= duration <= 180
    ]
    average_nap = _recent_median(nap_durations[-20:], 45, minimum=20, maximum=150)

    night_events = [event for event in recent[-30:] if _kind(event) == "night"]
    bedtime_by_evening: dict[date, int] = {}
    wake_by_morning: dict[date, int] = {}
    night_bounds: dict[date, tuple[datetime, datetime]] = {}
    for event in night_events:
        local_start = event.started_at.astimezone(timezone)
        start_minute = _minute_of_day(local_start)
        if start_minute >= 17 * 60:
            existing = bedtime_by_evening.get(local_start.date())
            bedtime_by_evening[local_start.date()] = (
                min(existing, start_minute) if existing is not None else start_minute
            )
        if event.ended_at is not None:
            local_end = event.ended_at.astimezone(timezone)
            if 4 <= local_end.hour <= 11:
                end_minute = _minute_of_day(local_end)
                existing_wake = wake_by_morning.get(local_end.date())
                wake_by_morning[local_end.date()] = (
                    max(existing_wake, end_minute) if existing_wake is not None else end_minute
                )
            night_date = local_end.date() if local_start.hour < 12 else local_start.date() + timedelta(days=1)
            existing_bounds = night_bounds.get(night_date)
            if existing_bounds is None:
                night_bounds[night_date] = (local_start, local_end)
            else:
                night_bounds[night_date] = (min(existing_bounds[0], local_start), max(existing_bounds[1], local_end))

    bedtime_minutes = list(bedtime_by_evening.values())
    wake_minutes_of_day = list(wake_by_morning.values())
    night_durations = [
        (end - start).total_seconds() / 60
        for start, end in night_bounds.values()
        if 4 * 60 <= (end - start).total_seconds() / 60 <= 14 * 60
    ]

    age_night_anchor = 20 * 60 + (30 if age_band in {"0-3m", "4-5m", "unknown"} else 0)
    typical_bedtime = _recent_median(bedtime_minutes[-10:], age_night_anchor, minimum=17 * 60, maximum=27 * 60)
    typical_bedtime %= 24 * 60
    typical_wake = _recent_median(wake_minutes_of_day[-14:], 7 * 60 + 30, minimum=4 * 60, maximum=11 * 60)
    average_night = _recent_median(night_durations[-14:], 10 * 60, minimum=6 * 60, maximum=13 * 60)

    plans: list[dict[str, Any]] = []
    for offset in (0, 1):
        plan_day = today + timedelta(days=offset)
        morning_wake = _local_at(plan_day, typical_wake, timezone)
        bedtime = _local_at(plan_day, typical_bedtime, timezone)
        next_morning = _local_at(plan_day + timedelta(days=1), typical_wake, timezone)
        night_duration = round((next_morning - bedtime).total_seconds() / 60)
        if not 6 * 60 <= night_duration <= 13 * 60:
            night_duration = average_night
            next_morning = bedtime + timedelta(minutes=night_duration)

        actual_naps = [
            event
            for event in closed
            if _kind(event) == "nap" and event.started_at.astimezone(timezone).date() == plan_day
        ]
        remaining = max(0, expected_naps - len(actual_naps))
        latest_wake = max(
            (
                event.ended_at.astimezone(timezone)
                for event in closed
                if event.ended_at is not None
                and event.ended_at.astimezone(timezone).date() == plan_day
                and event.ended_at.astimezone(timezone) <= current_local + timedelta(minutes=5)
            ),
            default=None,
        )
        anchor = latest_wake if offset == 0 and latest_wake is not None else morning_wake
        candidate = anchor + timedelta(minutes=wake_minutes)
        if offset == 0 and candidate < current_local - timedelta(minutes=margin_minutes):
            candidate = current_local + timedelta(minutes=15)

        nap_targets: list[dict[str, Any]] = []
        latest_nap_start = bedtime - timedelta(minutes=90)
        for index in range(remaining):
            if candidate >= latest_nap_start:
                break
            nap_targets.append(
                _target(
                    kind="nap",
                    label=f"Nap {len(actual_naps) + index + 1}",
                    recommended=candidate,
                    duration_minutes=average_nap,
                    margin_minutes=margin_minutes,
                    confidence=confidence,
                    explanation=(
                        "Projected from the last wake and recent wake-window rhythm"
                        if offset == 0 and index == 0
                        else "Chained from the preceding predicted nap and the recent daily rhythm"
                    ),
                )
            )
            candidate += timedelta(minutes=average_nap + wake_minutes)

        night_target = _target(
            kind="night",
            label="Night sleep",
            recommended=bedtime,
            duration_minutes=night_duration,
            margin_minutes=max(25, min(45, margin_minutes)),
            confidence=confidence,
            explanation="Learned from recent bedtimes and morning wake times",
        )
        plans.append(
            {
                "date": plan_day.isoformat(),
                "morningWakeAt": _iso(morning_wake),
                "nightStartAt": _iso(bedtime),
                "nightEndAt": _iso(next_morning),
                "dayNapPredictions": nap_targets,
                "nightPrediction": night_target,
                "explanation": reason,
            }
        )

    future_targets = [
        target
        for plan in plans
        for target in [*plan["dayNapPredictions"], plan["nightPrediction"]]
        if datetime.fromisoformat(target["recommendedStart"]).astimezone(timezone) > current_local
    ]
    future_targets.sort(key=lambda target: target["recommendedStart"])
    next_target = future_targets[0] if future_targets else None
    return {
        "generatedAt": _iso(current_local),
        "ageBand": age_band,
        "confidence": confidence,
        "reason": reason,
        "recentSampleCount": len(interval_sample),
        "wakeWindowMinutes": wake_minutes,
        "wakeWindowMarginMinutes": margin_minutes,
        "averageNapMinutes": average_nap,
        "averageNightMinutes": average_night,
        "nextSleepAt": next_target["recommendedStart"] if next_target else None,
        "windowStart": next_target["windowStart"] if next_target else None,
        "windowEnd": next_target["windowEnd"] if next_target else None,
        "nextKind": next_target["kind"] if next_target else None,
        "plans": plans,
    }
