from __future__ import annotations

import os
import re

import httpx


class Go2RTCError(RuntimeError):
    """Raised when the local go2rtc WebRTC bridge cannot negotiate a session."""


class Go2RTCClient:
    _STREAM_NAME = re.compile(r"^[A-Za-z0-9_.:-]+$")

    def __init__(self, base_url: str | None = None, stream_name: str | None = None) -> None:
        self.base_url = (base_url or os.environ.get("BABY_MONITOR_GO2RTC_URL", "http://127.0.0.1:1984")).rstrip("/")
        self.stream_name = stream_name or os.environ.get("BABY_MONITOR_GO2RTC_STREAM", "baby_monitor_live")
        if not self._STREAM_NAME.fullmatch(self.stream_name):
            raise RuntimeError("BABY_MONITOR_GO2RTC_STREAM has an unsupported value")

    async def negotiate(self, offer: str) -> str:
        encoded = offer.encode("utf-8")
        if not offer.startswith("v=0") or b"m=video" not in encoded or len(encoded) > 128_000:
            raise ValueError("invalid WebRTC SDP offer")
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(10.0, connect=2.0),
                trust_env=False,
            ) as client:
                response = await client.post(
                    f"{self.base_url}/api/webrtc",
                    params={"src": self.stream_name},
                    content=encoded,
                    headers={"Accept": "application/sdp", "Content-Type": "application/sdp"},
                )
        except httpx.HTTPError as exc:
            raise Go2RTCError("the local WebRTC relay is unavailable") from exc
        if response.status_code >= 400:
            raise Go2RTCError("the local WebRTC relay rejected the camera session")
        answer = response.text
        if not answer.startswith("v=0"):
            raise Go2RTCError("the local WebRTC relay returned an invalid answer")
        return answer
