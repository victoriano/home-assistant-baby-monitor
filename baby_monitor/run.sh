#!/bin/sh
set -eu

umask 077

: "${BABY_MONITOR_DATA_DIR:=/data}"
: "${BABY_MONITOR_FRONTEND_DIR:=/app/frontend-dist}"
: "${BABY_MONITOR_HOST:=0.0.0.0}"
: "${BABY_MONITOR_PORT:=8099}"

export BABY_MONITOR_DATA_DIR
export BABY_MONITOR_FRONTEND_DIR

mkdir -p "${BABY_MONITOR_DATA_DIR}"

exec python -m uvicorn baby_monitor.main:app \
  --host "${BABY_MONITOR_HOST}" \
  --port "${BABY_MONITOR_PORT}" \
  --workers 1 \
  --no-proxy-headers \
  --no-server-header
