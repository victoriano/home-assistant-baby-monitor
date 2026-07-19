from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from baby_monitor.main import create_app
from baby_monitor.models import VisionLabel, utc_now
from fastapi.testclient import TestClient


def test_visual_statistics_preserve_legacy_attributes(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path, runtime="test", start_workers=False)
    start = utc_now() - timedelta(minutes=10)
    labels = [
        VisionLabel(
            baby_present=True,
            in_crib=False,
            sleep_surface="family_bed",
            state="asleep",
            confidence=0.9,
            description="sleeping",
            head_side="left",
            clothing_items=["sleep_sack"],
            pacifier="yes",
            mouth_open="no",
        ),
        VisionLabel(
            baby_present=True,
            in_crib=True,
            state="asleep",
            confidence=0.9,
            description="sleeping",
            head_side="right",
            clothing_items=["sleep_sack"],
            pacifier="no",
            mouth_open="yes",
        ),
    ]
    for index, label in enumerate(labels):
        app.state.database.add_frame(
            image=f"frame-{index}".encode(),
            mime_type="image/jpeg",
            camera_entity_id="camera.nursery",
            location_id="madrid",
            captured_at=start + timedelta(minutes=index * 5),
            label=label,
            provider="legacy",
            model="legacy",
        )

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/statistics/vision",
            params={"start": start.isoformat(), "end": utc_now().isoformat()},
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["sample_count"] == 2
    assert payload["visible_sample_count"] == 2
    assert {row["key"] for row in payload["metrics"]["head_side"]["segments"]} == {"left", "right"}
    assert payload["metrics"]["clothing"]["segments"][0]["key"] == "sleep_sack"
    assert payload["metrics"]["pacifier"]["positive_minutes"] == 5
    assert payload["metrics"]["mouth_open"]["positive_minutes"] == 5
