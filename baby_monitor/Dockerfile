FROM node:24.18.0-alpine@sha256:a0b9bf06e4e6193cf7a0f58816cc935ff8c2a908f81e6f1a95432d679c54fbfd AS frontend-builder

WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --ignore-scripts --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM ghcr.io/astral-sh/uv:0.11.28@sha256:0f36cb9361a3346885ca3677e3767016687b5a170c1a6b88465ec14aefec90aa AS uv

FROM python:3.14.6-slim-bookworm@sha256:4ff4b92a68355dbdb52584ab3391dff8d371a61d4e063468bfd0130e3189c6d9 AS python-dependencies

ENV UV_NO_CACHE=1 \
    UV_NO_PROGRESS=1 \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /build
COPY --from=uv /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock README.md ./
RUN python -m venv /opt/venv \
    && uv export \
        --locked \
        --no-dev \
        --no-emit-project \
        --format requirements-txt \
        --output-file /tmp/requirements.lock \
    && uv pip install \
        --python /opt/venv/bin/python \
        --require-hashes \
        --no-deps \
        --requirement /tmp/requirements.lock \
    && rm /tmp/requirements.lock

FROM python:3.14.6-slim-bookworm@sha256:4ff4b92a68355dbdb52584ab3391dff8d371a61d4e063468bfd0130e3189c6d9 AS runtime

ARG TARGETARCH
ARG BUILD_ARCH=${TARGETARCH}
ARG BUILD_REVISION=unknown
ARG BUILD_VERSION=dev

ENV BABY_MONITOR_DATA_DIR=/data \
    BABY_MONITOR_FRONTEND_DIR=/app/frontend-dist \
    HOME=/data \
    PATH=/opt/venv/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/baby_monitor/src:/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Package revisions intentionally follow the security repository paired with
# the digest-pinned Debian base image.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install --no-install-recommends --yes ca-certificates ffmpeg \
    && mkdir -p /usr/share/baby-monitor \
    && dpkg-query -W -f='${binary:Package}\t${Version}\t${source:Package}\t${source:Version}\n' \
        > /tmp/debian-packages.tsv \
    && LC_ALL=C sort /tmp/debian-packages.tsv \
        -o /usr/share/baby-monitor/debian-packages.tsv \
    && ffmpeg -hide_banner -version > /usr/share/baby-monitor/ffmpeg-build.txt \
    && rm /tmp/debian-packages.tsv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=python-dependencies /opt/venv /opt/venv
COPY baby_monitor/src ./baby_monitor/src
COPY tools ./tools
COPY LICENSE THIRD_PARTY_NOTICES.md /usr/share/doc/baby-monitor/

COPY --from=frontend-builder /build/frontend/dist /app/frontend-dist
COPY baby_monitor/run.sh /usr/local/bin/baby-monitor-run
RUN chmod 0755 /usr/local/bin/baby-monitor-run \
    && mkdir -p /data \
    && chmod 0700 /data

LABEL io.hass.arch="${BUILD_ARCH}" \
      io.hass.description="Local-first baby monitor for Home Assistant" \
      io.hass.name="Baby Monitor" \
      io.hass.type="app" \
      io.hass.version="${BUILD_VERSION}" \
      org.opencontainers.image.description="Local-first baby monitor for Home Assistant" \
      org.opencontainers.image.documentation="https://github.com/victoriano/home-assistant-baby-monitor/blob/main/baby_monitor/DOCS.md" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.revision="${BUILD_REVISION}" \
      org.opencontainers.image.source="https://github.com/victoriano/home-assistant-baby-monitor" \
      org.opencontainers.image.title="Baby Monitor for Home Assistant" \
      org.opencontainers.image.version="${BUILD_VERSION}"

EXPOSE 8099
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8099/healthz', timeout=3)"]

ENTRYPOINT ["/usr/local/bin/baby-monitor-run"]
