from __future__ import annotations

import asyncio
import hmac
import json
import os
import shutil
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from pydantic import ValidationError

from . import __version__
from .auth import SESSION_COOKIE, AccessControlMiddleware, session_value, validate_admin_token
from .database import Database, StorageError
from .home_assistant import HomeAssistantClient, HomeAssistantError
from .media import AudioWindowReader, MediaError, analyze_pcm, snapshot_from_stream, stream_mjpeg_from_stream
from .models import (
    AIProviderName,
    AppSettings,
    CryEventCreate,
    CryMode,
    CryWebhook,
    SecretName,
    SettingsPatch,
    SettingsWrite,
    SleepEventCreate,
    SleepEventPatch,
    SleepStartRequest,
    SleepStopRequest,
    utc_now,
)
from .notifications import NotificationDispatcher, NotificationScheduler
from .providers import ProviderError, build_provider
from .runtime import RuntimeWorkers
from .security import EncryptedSecretStore
from .services import CryAlertService, DashboardService, FrameService, ServiceError
from .settings import SettingsError, SettingsRepository, SettingsService
from .statistics import vision_summary
from .transfer import MAX_ARCHIVE_BYTES, HistoryTransferManager, TransferError
from .webrtc import Go2RTCClient, Go2RTCError

API_PREFIX = "/api/v1"


def _safe_validation_detail(error: ValidationError | RequestValidationError) -> list[dict[str, Any]]:
    return [
        {
            "type": item.get("type", "value_error"),
            "loc": list(item.get("loc", ())),
            "msg": item.get("msg", "Invalid value"),
        }
        for item in error.errors()
    ]


def _default_data_dir(runtime: str) -> Path:
    configured = os.environ.get("BABY_MONITOR_DATA_DIR")
    if configured:
        return Path(configured)
    if runtime in {"home_assistant_app", "standalone"}:
        return Path("/data")
    return Path.home() / ".local" / "share" / "home-assistant-baby-monitor"


def _public_settings(settings: AppSettings, configured: bool) -> dict[str, Any]:
    payload = settings.model_dump(mode="json")
    payload["configured"] = configured
    return payload


def _frame(frame: Any) -> dict[str, Any]:
    label = frame.label
    return {
        "id": frame.id,
        "capturedAt": frame.captured_at,
        "cameraEntityId": frame.camera_entity_id,
        "locationId": frame.location_id,
        "imageUrl": f"api/v1/frames/{frame.id}/image" if frame.image_available else "",
        "imageAvailable": frame.image_available,
        "mimeType": frame.mime_type,
        "sizeBytes": frame.size_bytes,
        "sha256": frame.sha256,
        "label": label.model_dump(mode="json") if label else None,
        "provider": frame.provider,
        "model": frame.model,
    }


def _sleep(event: Any) -> dict[str, Any]:
    return {
        "id": event.id,
        "startedAt": event.started_at,
        "endedAt": event.ended_at,
        "kind": event.kind,
        "source": event.source,
        "notes": event.notes,
        "details": {
            "tags": event.details.tags,
            "pauses": [{"startedAt": pause.started_at, "endedAt": pause.ended_at} for pause in event.details.pauses],
        },
        "locationId": event.location_id,
        "createdAt": event.created_at,
    }


def _cry(event: Any) -> dict[str, Any]:
    return {
        "id": event.id,
        "detectedAt": event.detected_at,
        "endedAt": event.ended_at,
        "source": event.source,
        "confidence": event.confidence,
        "metadata": event.metadata,
        "locationId": event.location_id,
        "createdAt": event.created_at,
    }


def create_app(
    *,
    data_dir: Path | None = None,
    frontend_dir: Path | None = None,
    runtime: str | None = None,
    start_workers: bool | None = None,
) -> FastAPI:
    runtime = runtime or os.environ.get("BABY_MONITOR_RUNTIME", "development")
    if runtime not in {"development", "test", "standalone", "home_assistant_app"}:
        raise RuntimeError("BABY_MONITOR_RUNTIME has an unsupported value")
    admin_token = os.environ.get("BABY_MONITOR_ADMIN_TOKEN")
    validate_admin_token(runtime, admin_token)
    data_dir = data_dir or _default_data_dir(runtime)
    frontend_dir = frontend_dir or Path(os.environ.get("BABY_MONITOR_FRONTEND_DIR", "/app/frontend-dist"))
    database = Database(data_dir)
    history_transfer = HistoryTransferManager(data_dir, database, app_version=__version__)
    secret_store = EncryptedSecretStore(data_dir)
    settings_repository = SettingsRepository(data_dir)
    settings = SettingsService(settings_repository, secret_store, runtime=runtime)
    home_assistant = HomeAssistantClient(settings)
    frames = FrameService(database, settings, home_assistant)
    notifications = NotificationDispatcher(settings, home_assistant)
    notification_scheduler = NotificationScheduler(data_dir, database, settings, notifications)
    cry_alerts = CryAlertService(database, settings, home_assistant, notifications)
    dashboard = DashboardService(database, settings)
    workers = RuntimeWorkers(
        database,
        settings,
        home_assistant,
        frames,
        cry_alerts,
        notification_scheduler,
    )
    go2rtc = Go2RTCClient()
    if start_workers is None:
        start_workers = (
            runtime in {"standalone", "home_assistant_app"} and os.environ.get("BABY_MONITOR_DISABLE_WORKERS") != "1"
        )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        history_transfer.cleanup()
        if start_workers and history_transfer.writable:
            workers.start()
        try:
            yield
        finally:
            if start_workers:
                await workers.stop()

    app = FastAPI(
        title="Baby Monitor for Home Assistant",
        version=__version__,
        docs_url=f"{API_PREFIX}/docs" if runtime != "home_assistant_app" else None,
        redoc_url=None,
        openapi_url=f"{API_PREFIX}/openapi.json" if runtime != "home_assistant_app" else None,
        lifespan=lifespan,
    )
    app.add_middleware(
        AccessControlMiddleware,
        runtime=runtime,
        admin_token=admin_token,
    )

    @app.middleware("http")
    async def privacy_headers(request: Request, call_next: Any) -> Response:
        response = await call_next(request)
        if request.url.path.startswith(API_PREFIX) or request.url.path in {"/", "/login", "/logout"}:
            response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; base-uri 'self'; connect-src 'self'; "
            "font-src 'self'; form-action 'self'; frame-ancestors 'self'; "
            "img-src 'self' data: blob:; media-src 'self' blob:; object-src 'none'; "
            "script-src 'self'; style-src 'self' 'unsafe-inline'"
        )
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        return response

    app.state.database = database
    app.state.history_transfer = history_transfer
    app.state.settings = settings
    app.state.home_assistant = home_assistant
    app.state.frames = frames
    app.state.cry_alerts = cry_alerts
    app.state.notifications = notifications
    app.state.notification_scheduler = notification_scheduler
    app.state.dashboard = dashboard
    app.state.workers = workers
    app.state.go2rtc = go2rtc
    app.state.runtime = runtime
    login_failures: dict[str, list[float]] = {}

    @app.exception_handler(SettingsError)
    async def settings_error(_: Request, error: SettingsError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(error)})

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(_: Request, error: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"detail": _safe_validation_detail(error)},
            headers={"Cache-Control": "no-store"},
        )

    @app.exception_handler(HomeAssistantError)
    async def ha_error(_: Request, error: HomeAssistantError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(error)})

    @app.exception_handler(ProviderError)
    async def provider_error(_: Request, error: ProviderError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(error)})

    @app.exception_handler(ServiceError)
    async def service_error(_: Request, error: ServiceError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(error)})

    @app.exception_handler(StorageError)
    async def storage_error(_: Request, error: StorageError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(error)})

    @app.exception_handler(TransferError)
    async def transfer_error(_: Request, error: TransferError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(error)})

    @app.exception_handler(MediaError)
    async def media_error(_: Request, error: MediaError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(error)})

    @app.get("/healthz")
    async def healthz() -> Response:
        return JSONResponse(
            status_code=200 if database.ready() else 503,
            content={"status": "ok" if database.ready() else "unavailable"},
            headers={"Cache-Control": "no-store"},
        )

    @app.get(f"{API_PREFIX}/health")
    async def api_health() -> dict[str, Any]:
        return {
            "ok": database.ready(),
            "database": database.ready(),
            "runtime": runtime,
            "background": workers.status(),
        }

    @app.get("/login", response_class=HTMLResponse)
    async def login_form() -> HTMLResponse:
        if runtime != "standalone":
            raise HTTPException(404)
        return HTMLResponse(
            """<!doctype html><html><head><meta name="viewport" content="width=device-width">
            <title>Baby Monitor login</title></head><body><main><h1>Baby Monitor</h1>
            <form method="post"><label>Admin token <input name="token" type="password" required
            autocomplete="current-password"></label><button type="submit">Sign in</button></form>
            </main></body></html>""",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/login", response_class=HTMLResponse)
    async def login(request: Request) -> Response:
        if runtime != "standalone":
            raise HTTPException(404)
        raw = await request.body()
        if len(raw) > 4096:
            raise HTTPException(413, "login request is too large")
        peer = request.client.host if request.client else "unknown"
        now = time.monotonic()
        recent_failures = [timestamp for timestamp in login_failures.get(peer, []) if now - timestamp < 300]
        login_failures[peer] = recent_failures
        if len(recent_failures) >= 5:
            return HTMLResponse(
                "Too many failed sign-in attempts. Try again later.",
                status_code=429,
                headers={"Cache-Control": "no-store", "Retry-After": "300"},
            )
        token = parse_qs(raw.decode("utf-8", "replace")).get("token", [""])[0]
        expected = os.environ.get("BABY_MONITOR_ADMIN_TOKEN", "")
        if not expected or not hmac.compare_digest(token, expected):
            recent_failures.append(now)
            return HTMLResponse(
                "Invalid administrator token.",
                status_code=401,
                headers={"Cache-Control": "no-store"},
            )
        login_failures.pop(peer, None)
        response = RedirectResponse("./", status_code=303)
        response.set_cookie(
            SESSION_COOKIE,
            session_value(expected),
            httponly=True,
            secure=request.url.scheme == "https" or os.environ.get("BABY_MONITOR_COOKIE_SECURE") == "1",
            samesite="strict",
            max_age=12 * 60 * 60,
            path="/",
        )
        return response

    @app.post("/logout")
    async def logout() -> Response:
        response = RedirectResponse("./login", status_code=303)
        response.delete_cookie(SESSION_COOKIE, path="/")
        return response

    @app.get(f"{API_PREFIX}/settings")
    async def get_settings() -> dict[str, Any]:
        return _public_settings(settings.get(), settings.configured())

    @app.get(f"{API_PREFIX}/history-transfer")
    async def history_transfer_status() -> dict[str, Any]:
        return history_transfer.public_status()

    @app.post(f"{API_PREFIX}/history-transfer/exports")
    async def prepare_history_export() -> dict[str, Any]:
        if history_transfer.public_status()["status"] == "pending":
            return history_transfer.public_status()["outgoing"]
        workers_were_running = workers.status()["running"]
        if workers_were_running:
            await workers.stop()
        try:
            result = await asyncio.to_thread(history_transfer.prepare_export)
            return {key: value for key, value in result.items() if key not in {"path", "contentType"}}
        except Exception:
            if workers_were_running and history_transfer.writable:
                workers.start()
            raise

    @app.get(f"{API_PREFIX}/history-transfer/exports/{{archive_id}}")
    async def download_history_export(archive_id: str) -> Response:
        path, filename = history_transfer.export_path(archive_id)
        return FileResponse(
            path,
            filename=filename,
            media_type="application/zip",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    @app.post(f"{API_PREFIX}/history-transfer/cancel")
    async def cancel_history_export() -> dict[str, Any]:
        result = history_transfer.cancel_export()
        if start_workers and history_transfer.writable:
            workers.start()
        return result

    @app.post(f"{API_PREFIX}/history-transfer/finalize")
    async def finalize_history_export(request: Request, delete: bool = Query(False)) -> dict[str, Any]:
        raw = await request.body()
        if not raw:
            raise HTTPException(400, "import receipt is empty")
        if len(raw) > 64 * 1024:
            raise HTTPException(413, "import receipt is too large")
        try:
            receipt = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(400, "import receipt is not valid JSON") from exc
        return await asyncio.to_thread(
            history_transfer.retire_exported_history,
            receipt,
            delete_history=delete,
        )

    @app.post(f"{API_PREFIX}/history-transfer/imports")
    async def import_history(request: Request, replace: bool = Query(False)) -> dict[str, Any]:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                expected_bytes = int(content_length)
            except ValueError as exc:
                raise HTTPException(400, "Content-Length is invalid") from exc
            if expected_bytes <= 0 or expected_bytes > MAX_ARCHIVE_BYTES:
                raise HTTPException(413, "history archive is too large")
            if expected_bytes + 256 * 1024 * 1024 > shutil.disk_usage(data_dir).free:
                raise HTTPException(507, "not enough free disk space to upload this history archive")
        upload = history_transfer.new_incoming_path()
        received = 0
        try:
            with upload.open("wb") as destination:
                async for chunk in request.stream():
                    received += len(chunk)
                    if received > MAX_ARCHIVE_BYTES:
                        raise HTTPException(413, "history archive is too large")
                    destination.write(chunk)
                destination.flush()
                os.fsync(destination.fileno())
            if received == 0:
                raise HTTPException(400, "history archive is empty")
            workers_were_running = workers.status()["running"]
            if workers_were_running:
                await workers.stop()
            try:
                result = await asyncio.to_thread(
                    history_transfer.import_archive,
                    upload,
                    replace_existing=replace,
                )
            finally:
                if workers_were_running and history_transfer.writable:
                    workers.start()
            return {**result, "status": history_transfer.public_status()}
        finally:
            upload.unlink(missing_ok=True)

    @app.put(f"{API_PREFIX}/settings")
    async def put_settings(request: Request) -> dict[str, Any]:
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("settings payload must be an object")
            update = SettingsWrite.model_validate(payload)
            saved = settings.replace(update)
            return _public_settings(saved, True)
        except ValidationError as exc:
            return JSONResponse(status_code=422, content={"detail": _safe_validation_detail(exc)})
        except (ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.patch(f"{API_PREFIX}/settings")
    async def patch_settings(update: SettingsPatch) -> dict[str, Any]:
        saved = settings.patch(update)
        return _public_settings(saved, True)

    @app.get(f"{API_PREFIX}/home-assistant/entities")
    async def entities(
        domain: Literal["camera", "binary_sensor", "light", "notify", "person"] = Query(...),
    ) -> dict[str, Any]:
        items = await home_assistant.list_entities(domain)
        return {
            "items": [
                {
                    "entityId": item.entity_id,
                    "name": item.name,
                    "state": item.state,
                    "available": item.state != "unavailable",
                    "attributes": (
                        {"userId": item.attributes.get("user_id")}
                        if domain == "person" and item.attributes.get("user_id")
                        else {}
                    ),
                }
                for item in items
            ]
        }

    @app.post(f"{API_PREFIX}/settings/test/{{kind}}")
    async def test_settings(
        kind: Literal["home_assistant", "camera", "cry", "lights", "notifications", "vision"],
        request: Request,
    ) -> dict[str, Any]:
        try:
            raw = await request.json()
            if not isinstance(raw, dict):
                raise ValueError("settings payload must be an object")
            update = SettingsWrite.model_validate(raw)
            candidate = AppSettings.model_validate(update.model_dump(exclude={"secrets"}, mode="python"))
            settings.validate_secret_bindings(candidate, update.secrets)
            incoming = update.secrets.values()
            if kind == "home_assistant":
                access_token = incoming.get("home_assistant_access_token") or settings.get_secret(
                    SecretName.HOME_ASSISTANT_ACCESS_TOKEN
                )
                await home_assistant.ping(candidate.home_assistant, access_token)
            elif kind == "camera":
                if candidate.camera.entity_id:
                    await home_assistant.camera_snapshot(candidate.camera.entity_id)
                else:
                    stream_url = incoming.get("camera_stream_url") or settings.get_secret(SecretName.CAMERA_STREAM_URL)
                    if not stream_url:
                        return {"ok": False, "message": "No camera entity or private stream URL is configured."}
                    await snapshot_from_stream(stream_url)
            elif kind == "cry":
                if candidate.cry.mode == CryMode.DISABLED:
                    return {"ok": True, "message": "Cry detection is disabled."}
                if candidate.cry.mode == CryMode.BINARY_SENSOR:
                    await home_assistant.get_state(candidate.cry.entity_id or "")
                else:
                    stream_url = incoming.get("cry_audio_stream_url") or settings.get_secret(
                        SecretName.CRY_AUDIO_STREAM_URL
                    )
                    if not stream_url:
                        return {"ok": False, "message": "An RTSP audio stream URL is required."}
                    reader = AudioWindowReader(stream_url, candidate.cry.window_seconds)
                    try:
                        metrics = await reader.read()
                        await __import__("asyncio").to_thread(analyze_pcm, metrics, candidate.cry.sensitivity)
                    finally:
                        await reader.close()
            elif kind == "lights":
                for entity_id in candidate.lights.entity_ids:
                    await home_assistant.get_state(entity_id)
            elif kind == "notifications":
                recipients = [item for item in candidate.notifications.recipients if item.enabled]
                if not recipients:
                    return {"ok": True, "message": "Notifications are disabled."}
                for recipient in recipients:
                    await notifications.send_test(recipient)
            else:
                if candidate.ai.provider == AIProviderName.DISABLED:
                    return {"ok": True, "message": "Image labeling is disabled."}
                provider = build_provider(
                    candidate.ai,
                    incoming.get("ai_api_key") or settings.get_secret(SecretName.AI_API_KEY),
                )
                await provider.probe()
            return {"ok": True, "message": f"{kind.title()} connection succeeded."}
        except ValidationError as exc:
            return JSONResponse(status_code=422, content={"detail": _safe_validation_detail(exc)})
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get(f"{API_PREFIX}/summary")
    async def summary() -> dict[str, Any]:
        result = dashboard.summary()
        next_sleep = result["next_sleep_at"]
        return {
            "state": result["state"],
            "stateSince": result["state_since"],
            "currentSleep": _sleep(result["current_sleep"]) if result["current_sleep"] else None,
            "prediction": {
                "nextSleepAt": next_sleep,
                "windowStart": result["prediction_window_start"],
                "windowEnd": result["prediction_window_end"],
                "confidence": result["prediction_confidence"],
                "reason": result["prediction_reason"],
            },
            "sleepTodayMinutes": result["sleep_today_minutes"],
            "lastCryAt": result["last_cry_at"],
            "cryActive": result["cry_active"],
            "latestFrame": _frame(result["latest_frame"]) if result["latest_frame"] else None,
            "recentSleep": [_sleep(item) for item in result["recent_sleep"]],
            "recentCry": [_cry(item) for item in result["recent_cry"]],
            "updatedAt": utc_now(),
        }

    @app.get(f"{API_PREFIX}/predictions")
    async def predictions() -> dict[str, Any]:
        return dashboard.predictions()

    @app.get(f"{API_PREFIX}/frames")
    async def list_frames(limit: int = Query(24, ge=1, le=200), offset: int = Query(0, ge=0)) -> dict[str, Any]:
        items, total = database.list_frames(limit, offset)
        return {"items": [_frame(item) for item in items], "limit": limit, "offset": offset, "total": total}

    @app.get(f"{API_PREFIX}/frames/range")
    async def frames_in_range(
        start: datetime,
        end: datetime,
        location_id: str | None = Query(None, min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9_-]*$"),
        limit: int = Query(200, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> dict[str, Any]:
        if end <= start:
            raise HTTPException(400, "end must be after start")
        items, total = database.list_frames_between(
            start,
            end,
            location_id=location_id,
            limit=limit,
            offset=offset,
        )
        return {
            "items": [_frame(item) for item in items],
            "limit": limit,
            "offset": offset,
            "total": total,
            "start": start,
            "end": end,
        }

    @app.get(f"{API_PREFIX}/frames/nearest")
    async def nearest_frames(
        at: datetime,
        limit: int = Query(5, ge=1, le=12),
        within_minutes: int = Query(360, ge=1, le=1_440),
    ) -> dict[str, Any]:
        items = database.nearest_frames(at, limit, within_minutes)
        return {"items": [_frame(item) for item in items], "requestedAt": at}

    @app.get(f"{API_PREFIX}/statistics/vision")
    async def visual_statistics(start: datetime, end: datetime) -> dict[str, Any]:
        if end <= start:
            raise HTTPException(400, "end must be after start")
        rows = database.vision_labels_between(start, end)
        return vision_summary(rows, start, end, settings.get().baby.timezone)

    @app.get(f"{API_PREFIX}/frames/{{frame_id}}/image")
    async def frame_image(frame_id: str) -> Response:
        frame = database.get_frame(frame_id)
        path = database.get_frame_path(frame_id)
        if frame is None:
            raise HTTPException(404, "frame not found")
        if path is None:
            raise HTTPException(410, "frame image was removed by the retention policy")
        return FileResponse(
            path,
            media_type=frame.mime_type,
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    @app.post(f"{API_PREFIX}/frames/{{frame_id}}/label")
    async def label_frame(frame_id: str) -> dict[str, Any]:
        history_transfer.ensure_writable()
        return _frame(await frames.label(frame_id))

    @app.post(f"{API_PREFIX}/camera/snapshot")
    async def camera_snapshot() -> dict[str, Any]:
        history_transfer.ensure_writable()
        return _frame(await frames.capture(label=True))

    @app.get(f"{API_PREFIX}/camera/live")
    async def camera_live() -> Response:
        camera = settings.get().camera
        if not camera.enabled:
            raise HTTPException(409, "camera is disabled")

        async def stream():
            if camera.entity_id:
                async with home_assistant.camera_stream(camera.entity_id) as upstream:
                    async for chunk in upstream.aiter_bytes():
                        yield chunk
                return
            stream_url = settings.get_secret(SecretName.CAMERA_STREAM_URL)
            if not stream_url:
                raise ServiceError("camera stream URL is not configured")
            async for chunk in stream_mjpeg_from_stream(stream_url):
                yield chunk

        return StreamingResponse(
            stream(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    @app.post(f"{API_PREFIX}/camera/webrtc")
    async def camera_webrtc(request: Request) -> Response:
        if not settings.get().camera.enabled:
            raise HTTPException(409, "camera is disabled")
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > 128_000:
                    raise HTTPException(413, "WebRTC SDP offer is too large")
            except ValueError as exc:
                raise HTTPException(400, "Content-Length is invalid") from exc
        try:
            raw_offer = await request.body()
            if len(raw_offer) > 128_000:
                raise HTTPException(413, "WebRTC SDP offer is too large")
            offer = raw_offer.decode("utf-8")
            answer = await app.state.go2rtc.negotiate(offer)
        except (UnicodeDecodeError, ValueError) as exc:
            raise HTTPException(400, "invalid WebRTC SDP offer") from exc
        except Go2RTCError as exc:
            raise HTTPException(502, str(exc)) from exc
        return Response(
            answer,
            media_type="application/sdp",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    @app.get(f"{API_PREFIX}/sleep")
    async def list_sleep(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)) -> dict[str, Any]:
        items, total = database.list_sleep_events(limit, offset)
        return {"items": [_sleep(item) for item in items], "limit": limit, "offset": offset, "total": total}

    @app.post(f"{API_PREFIX}/sleep", status_code=201)
    async def add_sleep(event: SleepEventCreate) -> dict[str, Any]:
        history_transfer.ensure_writable()
        located = event.model_copy(update={"location_id": settings.get().baby.location_id})
        return _sleep(database.add_sleep_event(located))

    @app.patch(f"{API_PREFIX}/sleep/{{event_id}}")
    async def patch_sleep(event_id: str, update: SleepEventPatch) -> dict[str, Any]:
        history_transfer.ensure_writable()
        event = database.update_sleep_event(event_id, update)
        if event is None:
            raise HTTPException(404, "sleep event not found")
        return _sleep(event)

    @app.delete(f"{API_PREFIX}/sleep/{{event_id}}", status_code=204)
    async def delete_sleep(event_id: str) -> Response:
        history_transfer.ensure_writable()
        if not database.delete_sleep_event(event_id):
            raise HTTPException(404, "sleep event not found")
        return Response(status_code=204)

    @app.post(f"{API_PREFIX}/sleep/start", status_code=201)
    async def start_sleep(payload: SleepStartRequest) -> dict[str, Any]:
        history_transfer.ensure_writable()
        if database.open_sleep_event() is not None:
            raise HTTPException(409, "a sleep session is already active")
        return _sleep(
            database.add_sleep_event(
                SleepEventCreate(
                    started_at=payload.started_at,
                    kind=payload.kind,
                    source="manual",
                    notes=payload.notes,
                    location_id=settings.get().baby.location_id,
                )
            )
        )

    @app.post(f"{API_PREFIX}/sleep/stop")
    async def stop_sleep(payload: SleepStopRequest) -> dict[str, Any]:
        history_transfer.ensure_writable()
        current = database.open_sleep_event()
        if current is None:
            raise HTTPException(409, "there is no active sleep session")
        event = database.update_sleep_event(current.id, SleepEventPatch(ended_at=payload.ended_at))
        assert event is not None
        return _sleep(event)

    @app.get(f"{API_PREFIX}/cry-events")
    async def list_cries(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)) -> dict[str, Any]:
        items, total = database.list_cry_events(limit, offset)
        return {"items": [_cry(item) for item in items], "limit": limit, "offset": offset, "total": total}

    @app.post(f"{API_PREFIX}/cry-events", status_code=201)
    async def add_cry(event: CryEventCreate) -> dict[str, Any]:
        history_transfer.ensure_writable()
        result = await cry_alerts.set_state(
            "on",
            observed_at=event.detected_at,
            source=event.source,
            confidence=event.confidence,
            metadata=event.metadata,
        )
        assert result is not None
        if event.ended_at is not None:
            result = await cry_alerts.set_state("off", observed_at=event.ended_at, source=event.source)
            assert result is not None
        return _cry(result)

    @app.post(f"{API_PREFIX}/cry/webhook")
    async def cry_webhook(payload: CryWebhook) -> dict[str, Any]:
        history_transfer.ensure_writable()
        if len(json.dumps(payload.metadata, separators=(",", ":"))) > 32_768:
            raise HTTPException(413, "cry metadata is too large")
        event = await cry_alerts.set_state(
            payload.state,
            observed_at=payload.observed_at,
            source=payload.source,
            confidence=payload.confidence,
            metadata=payload.metadata,
        )
        return {"ok": True, "active": cry_alerts.active, "event": _cry(event) if event else None}

    @app.get(f"{API_PREFIX}/retention/estimate")
    async def retention_estimate(days: int = Query(..., ge=1, le=3650)) -> dict[str, int]:
        return database.retention_estimate(utc_now() - timedelta(days=days))

    @app.get("/", response_class=HTMLResponse)
    async def index() -> Response:
        index_path = frontend_dir / "index.html"
        if index_path.is_file():
            return FileResponse(index_path, headers={"Cache-Control": "no-store"})
        return HTMLResponse("<h1>Baby Monitor for Home Assistant</h1><p>The frontend has not been built yet.</p>")

    @app.get("/{asset_path:path}")
    async def frontend_asset(asset_path: str) -> Response:
        if asset_path.startswith("api/"):
            raise HTTPException(404)
        root = frontend_dir.resolve()
        candidate = (root / asset_path).resolve()
        if candidate.is_relative_to(root) and candidate.is_file():
            # The Home Assistant panel lives inside an iframe whose lifecycle is
            # controlled by the HA frontend/service worker. Always revalidate
            # assets so reopening the panel cannot retain an older UI build.
            return FileResponse(candidate, headers={"Cache-Control": "no-cache, max-age=0, must-revalidate"})
        if asset_path.startswith("assets/"):
            # Never disguise a missing content-hashed bundle as the SPA shell.
            # Browsers reject that HTML as JavaScript/CSS and the misleading 200
            # makes deployment races much harder for an embedding panel to spot.
            raise HTTPException(404)
        index_path = root / "index.html"
        if index_path.is_file():
            return FileResponse(index_path, headers={"Cache-Control": "no-store"})
        raise HTTPException(404)

    return app


app = create_app()
