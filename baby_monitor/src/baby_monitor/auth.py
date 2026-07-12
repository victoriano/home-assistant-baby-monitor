from __future__ import annotations

import hashlib
import hmac
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

SESSION_COOKIE = "baby_monitor_session"
INGRESS_PEER = "172.30.32.2"


def validate_admin_token(runtime: str, admin_token: str | None) -> None:
    if runtime == "standalone" and (not admin_token or len(admin_token.encode("utf-8")) < 32):
        raise RuntimeError("BABY_MONITOR_ADMIN_TOKEN must contain at least 32 bytes in standalone mode")


def session_value(admin_token: str, expires_at: int | None = None) -> str:
    expires_at = expires_at or int(time.time()) + 12 * 60 * 60
    message = f"baby-monitor-browser-session-v1:{expires_at}".encode()
    signature = hmac.new(admin_token.encode("utf-8"), message, hashlib.sha256).hexdigest()
    return f"{expires_at}.{signature}"


def session_valid(value: str, admin_token: str) -> bool:
    expires, separator, signature = value.partition(".")
    if not separator or not expires.isdigit() or int(expires) < int(time.time()):
        return False
    message = f"baby-monitor-browser-session-v1:{expires}".encode()
    expected = hmac.new(admin_token.encode("utf-8"), message, hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, expected)


class AccessControlMiddleware:
    """Protect ingress by peer IP and standalone with an explicit admin token.

    `X-Forwarded-For` is intentionally ignored. In Home Assistant App mode the
    TCP peer itself must be the Supervisor ingress gateway.
    """

    def __init__(self, app: Callable[..., Awaitable[None]], runtime: str, admin_token: str | None) -> None:
        self.app = app
        self.runtime = runtime
        self.admin_token = admin_token
        validate_admin_token(runtime, admin_token)

    @staticmethod
    async def _deny(send: Callable[..., Awaitable[None]], status: int, message: str) -> None:
        body = json.dumps({"detail": message}, separators=(",", ":")).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"cache-control", b"no-store"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    @staticmethod
    async def _redirect_login(send: Callable[..., Awaitable[None]]) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 303,
                "headers": [
                    (b"location", b"./login"),
                    (b"content-length", b"0"),
                    (b"cache-control", b"no-store"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": b""})

    @staticmethod
    def _headers(scope: dict[str, Any]) -> dict[bytes, bytes]:
        return {key.lower(): value for key, value in scope.get("headers", [])}

    @staticmethod
    def _cookie(headers: dict[bytes, bytes], name: str) -> str | None:
        raw = headers.get(b"cookie", b"").decode("latin-1")
        for part in raw.split(";"):
            key, separator, value = part.strip().partition("=")
            if separator and key == name:
                return value
        return None

    def _standalone_authorized(self, headers: dict[bytes, bytes]) -> bool:
        assert self.admin_token is not None
        candidates: list[str] = []
        authorization = headers.get(b"authorization", b"").decode("latin-1")
        if authorization.lower().startswith("bearer "):
            candidates.append(authorization[7:].strip())
        custom = headers.get(b"x-baby-monitor-token")
        if custom:
            candidates.append(custom.decode("latin-1"))
        cookie = self._cookie(headers, SESSION_COOKIE)
        if cookie and session_valid(cookie, self.admin_token):
            return True
        return any(hmac.compare_digest(candidate, self.admin_token) for candidate in candidates)

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        path = str(scope.get("path", "/"))
        client = scope.get("client")
        peer = client[0] if isinstance(client, (tuple, list)) and client else ""
        headers = self._headers(scope)

        if self.runtime == "home_assistant_app":
            if path == "/healthz" and peer in {INGRESS_PEER, "127.0.0.1", "::1"}:
                await self.app(scope, receive, send)
                return
            if peer != INGRESS_PEER:
                await self._deny(send, 403, "request did not originate from Home Assistant ingress")
                return
            admin = headers.get(b"x-hass-is-admin") or headers.get(b"x-home-assistant-is-admin")
            if admin is not None and admin.decode("ascii", "ignore").lower() not in {"1", "true", "yes"}:
                await self._deny(send, 403, "Home Assistant administrator access is required")
                return
            await self.app(scope, receive, send)
            return

        public = path in {"/healthz", "/login"}
        if self.runtime == "standalone":
            if public or self._standalone_authorized(headers):
                await self.app(scope, receive, send)
            elif path == "/":
                await self._redirect_login(send)
            else:
                await self._deny(send, 401, "standalone administrator authentication is required")
            return

        # Development is deliberately loopback-only. TestClient's synthetic
        # peer is accepted only in the non-production development/test runtime.
        if peer not in {"127.0.0.1", "::1", "testclient"}:
            await self._deny(send, 403, "development server is loopback-only")
            return
        await self.app(scope, receive, send)
