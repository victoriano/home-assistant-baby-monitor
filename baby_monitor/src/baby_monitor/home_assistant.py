from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import quote

import httpx

from .models import HAEntity, HomeAssistantConfig, HomeAssistantMode, SecretName
from .settings import SettingsService


class HomeAssistantError(RuntimeError):
    pass


class HomeAssistantClient:
    """Small async client for both Home Assistant App and standalone modes."""

    def __init__(
        self,
        settings: SettingsService,
        timeout: float = 15.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings_service = settings
        self._timeout = httpx.Timeout(timeout, connect=min(timeout, 5.0))
        self._transport = transport

    def _connection(
        self,
        config: HomeAssistantConfig | None = None,
        access_token: str | None = None,
    ) -> tuple[str, str]:
        settings = config or self._settings_service.get().home_assistant
        supervisor_token = os.environ.get("SUPERVISOR_TOKEN")
        mode = settings.mode
        if mode == HomeAssistantMode.AUTO:
            mode = HomeAssistantMode.SUPERVISOR if supervisor_token else HomeAssistantMode.STANDALONE

        if mode == HomeAssistantMode.SUPERVISOR:
            if not supervisor_token:
                raise HomeAssistantError("SUPERVISOR_TOKEN is unavailable")
            root = os.environ.get("BABY_MONITOR_HA_URL", "http://supervisor/core").rstrip("/")
            return f"{root}/api", supervisor_token

        token = (
            access_token
            or self._settings_service.get_secret(SecretName.HOME_ASSISTANT_ACCESS_TOKEN)
            or os.environ.get("BABY_MONITOR_HA_TOKEN")
        )
        base_url = settings.base_url or os.environ.get("BABY_MONITOR_HA_URL")
        if not base_url or not token:
            raise HomeAssistantError("standalone Home Assistant URL and token are not configured")
        return f"{base_url.rstrip('/')}/api", token

    def _client(
        self,
        config: HomeAssistantConfig | None = None,
        access_token: str | None = None,
    ) -> httpx.AsyncClient:
        base_url, token = self._connection(config, access_token)
        return httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            timeout=self._timeout,
            follow_redirects=False,
            transport=self._transport,
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        config: HomeAssistantConfig | None = None,
        access_token: str | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        try:
            async with self._client(config, access_token) as client:
                response = await client.request(method, path.lstrip("/"), **kwargs)
                response.raise_for_status()
                return response
        except (httpx.HTTPError, HomeAssistantError) as exc:
            if isinstance(exc, HomeAssistantError):
                raise
            status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
            suffix = f" (HTTP {status})" if status else ""
            raise HomeAssistantError(f"Home Assistant request failed{suffix}") from exc

    async def ping(
        self,
        config: HomeAssistantConfig | None = None,
        access_token: str | None = None,
    ) -> dict[str, Any]:
        response = await self._request("GET", "/", config=config, access_token=access_token)
        try:
            data = response.json()
        except ValueError as exc:
            raise HomeAssistantError("Home Assistant returned an invalid response") from exc
        return data if isinstance(data, dict) else {"message": str(data)}

    async def get_state(self, entity_id: str) -> dict[str, Any]:
        response = await self._request("GET", f"/states/{quote(entity_id, safe='')}")
        data = response.json()
        if not isinstance(data, dict):
            raise HomeAssistantError("Home Assistant returned an invalid entity state")
        return data

    async def list_entities(self, domain: str) -> list[HAEntity]:
        if domain == "notify":
            return await self._list_notify_services()
        response = await self._request("GET", "/states")
        raw = response.json()
        if not isinstance(raw, list):
            raise HomeAssistantError("Home Assistant returned an invalid state list")
        prefix = f"{domain}."
        entities: list[HAEntity] = []
        for item in raw:
            if not isinstance(item, dict) or not str(item.get("entity_id", "")).startswith(prefix):
                continue
            entity_id = str(item["entity_id"])
            attributes = item.get("attributes") if isinstance(item.get("attributes"), dict) else {}
            entities.append(
                HAEntity(
                    entity_id=entity_id,
                    state=str(item.get("state", "unknown")),
                    name=str(attributes.get("friendly_name") or entity_id),
                    attributes=attributes,
                )
            )
        return sorted(entities, key=lambda item: item.name.casefold())

    async def _list_notify_services(self) -> list[HAEntity]:
        response = await self._request("GET", "/services")
        raw = response.json()
        if not isinstance(raw, list):
            raise HomeAssistantError("Home Assistant returned an invalid services list")
        results: list[HAEntity] = []
        for domain in raw:
            if not isinstance(domain, dict) or domain.get("domain") != "notify":
                continue
            services = domain.get("services")
            if not isinstance(services, dict):
                continue
            for name, description in services.items():
                info = description if isinstance(description, dict) else {}
                results.append(
                    HAEntity(
                        entity_id=f"notify.{name}",
                        state="available",
                        name=str(info.get("name") or name.replace("_", " ").title()),
                        attributes={},
                    )
                )
        return sorted(results, key=lambda item: item.name.casefold())

    async def call_service(self, domain: str, service: str, data: dict[str, Any]) -> Any:
        response = await self._request(
            "POST",
            f"/services/{quote(domain, safe='')}/{quote(service, safe='')}",
            json=data,
        )
        try:
            return response.json()
        except ValueError:
            return None

    async def camera_snapshot(self, entity_id: str) -> tuple[bytes, str]:
        try:
            async with (
                self._client() as client,
                client.stream("GET", f"camera_proxy/{quote(entity_id, safe='')}") as response,
            ):
                response.raise_for_status()
                mime_type = response.headers.get("content-type", "image/jpeg").split(";", 1)[0].lower()
                if mime_type not in {"image/jpeg", "image/png", "image/webp"}:
                    raise HomeAssistantError("camera returned an unsupported image type")
                image = bytearray()
                async for chunk in response.aiter_bytes():
                    image.extend(chunk)
                    if len(image) > 25 * 1024 * 1024:
                        raise HomeAssistantError("camera snapshot exceeds 25 MB")
        except httpx.HTTPError as exc:
            raise HomeAssistantError("Home Assistant camera snapshot failed") from exc
        if not image:
            raise HomeAssistantError("camera snapshot is empty")
        return bytes(image), mime_type

    @asynccontextmanager
    async def camera_stream(self, entity_id: str) -> AsyncIterator[httpx.Response]:
        client = self._client()
        try:
            async with client.stream("GET", f"camera_proxy_stream/{quote(entity_id, safe='')}") as response:
                response.raise_for_status()
                yield response
        except httpx.HTTPError as exc:
            raise HomeAssistantError("Home Assistant camera stream failed") from exc
        finally:
            await client.aclose()
