from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture
def ui_settings_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "baby": {"name": "Alex", "birth_date": None, "timezone": "UTC"},
        "home_assistant": {"mode": "auto", "base_url": None},
        "camera": {"enabled": False, "entity_id": None, "capture_interval_seconds": 300},
        "cry": {
            "mode": "disabled",
            "entity_id": None,
            "positive_windows": 2,
            "window_seconds": 0.5,
            "clear_after_seconds": 8,
        },
        "lights": {
            "entity_ids": [],
            "duration_seconds": 45,
            "brightness_percent": 35,
            "color_rgb": [255, 125, 72],
            "restore_previous_state": True,
        },
        "ai": {
            "provider": "disabled",
            "model": None,
            "base_url": None,
            "cloud_image_consent": False,
            "detail": "low",
        },
        "retention": {"mode": "forever", "days": None},
        "notifications": {"recipients": [], "lead_minutes": 10},
        "secrets": {"clear": []},
    }
