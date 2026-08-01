from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import baby_monitor.runtime as runtime
from baby_monitor.media import MediaError
from baby_monitor.runtime import RuntimeWorkers


async def test_runtime_restarts_a_worker_after_an_unexpected_failure(monkeypatch) -> None:
    cry_alerts = SimpleNamespace(close=AsyncMock())
    workers = RuntimeWorkers(None, None, None, None, cry_alerts, None)
    parked = asyncio.Event()
    restarted = asyncio.Event()
    cry_runs = 0

    async def park() -> None:
        await parked.wait()

    async def flaky_cry() -> None:
        nonlocal cry_runs
        cry_runs += 1
        if cry_runs == 1:
            raise MediaError("temporary RTSP interruption")
        restarted.set()
        await parked.wait()

    monkeypatch.setattr(runtime, "WORKER_RESTART_DELAY_SECONDS", 0)
    monkeypatch.setattr(workers, "_retention_loop", park)
    monkeypatch.setattr(workers, "_capture_loop", park)
    monkeypatch.setattr(workers, "_cry_loop", flaky_cry)
    monkeypatch.setattr(workers, "_notification_loop", park)

    workers.start()
    try:
        await asyncio.wait_for(restarted.wait(), 0.5)

        assert cry_runs == 2
        assert workers.status()["workers"]["baby-monitor-cry"] is True
    finally:
        await workers.stop()

    cry_alerts.close.assert_awaited_once()
