#!/usr/bin/env bash
set -euo pipefail

TAG="qr-gcs-backup-core"

log() {
  local msg="$1"
  echo "[$TAG] $msg"
  logger -t "$TAG" "$msg" 2>/dev/null || true
}

is_true() {
  local value="${1:-}"
  local default="${2:-0}"
  if [[ -z "$value" ]]; then
    value="$default"
  fi
  case "${value,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

load_env_file() {
  local path="$1"
  if [[ -f "$path" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$path"
    set +a
  fi
}

ROOT_DIR="${QR_ROOT_DIR:-/home/tossaki/QuantRabbit}"
LOG_DIR="${QR_LOG_DIR:-${ROOT_DIR}/logs}"
RUNTIME_ENV_FILE="${QR_RUNTIME_ENV_FILE:-${ROOT_DIR}/ops/env/quant-v2-runtime.env}"
CORE_ENV_FILE="${QR_CORE_BACKUP_ENV_FILE:-${ROOT_DIR}/ops/env/quant-core-backup.env}"
HOST_ENV_FILE="${QR_HOST_ENV_FILE:-/etc/quantrabbit.env}"

load_env_file "$HOST_ENV_FILE"
load_env_file "$RUNTIME_ENV_FILE"
load_env_file "$CORE_ENV_FILE"

if ! is_true "${QR_CORE_BACKUP_ENABLED:-1}" 1; then
  log "disabled (QR_CORE_BACKUP_ENABLED=0)"
  exit 0
fi

BUCKET="${GCS_BACKUP_BUCKET:-}"
if [[ -z "$BUCKET" ]]; then
  log "GCS_BACKUP_BUCKET is empty, skip"
  exit 0
fi
BUCKET="${BUCKET#gs://}"

if ! command -v sqlite3 >/dev/null 2>&1; then
  log "sqlite3 not found, skip"
  exit 0
fi
if ! command -v curl >/dev/null 2>&1; then
  log "curl not found, skip"
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  log "python3 not found, skip"
  exit 1
fi

LOCK_FILE="${QR_CORE_BACKUP_LOCK_FILE:-/var/lock/qr-gcs-backup-core.lock}"
mkdir -p "$(dirname "$LOCK_FILE")"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  log "another backup process is running, skip"
  exit 0
fi

LOAD_1M="$(awk '{print $1}' /proc/loadavg)"
MAX_LOAD_1M="${QR_CORE_BACKUP_MAX_LOAD1:-8.0}"
if awk -v cur="$LOAD_1M" -v max="$MAX_LOAD_1M" 'BEGIN { exit !(cur > max) }'; then
  log "skip by load guard (load1=${LOAD_1M} > ${MAX_LOAD_1M})"
  exit 0
fi

DSTATE_COUNT="$(ps -eo state= | awk '$1=="D"{n++} END{print n+0}')"
MAX_DSTATE="${QR_CORE_BACKUP_MAX_DSTATE:-12}"
if (( DSTATE_COUNT > MAX_DSTATE )); then
  log "skip by dstate guard (dstate=${DSTATE_COUNT} > ${MAX_DSTATE})"
  exit 0
fi

MEM_AVAILABLE_KB="$(awk '/MemAvailable:/{print $2}' /proc/meminfo)"
MIN_MEM_AVAILABLE_MB="${QR_CORE_BACKUP_MIN_MEM_AVAILABLE_MB:-1024}"
if (( MEM_AVAILABLE_KB < MIN_MEM_AVAILABLE_MB * 1024 )); then
  log "skip by memory guard (avail_mb=$((MEM_AVAILABLE_KB / 1024)) < ${MIN_MEM_AVAILABLE_MB})"
  exit 0
fi

SWAP_TOTAL_KB="$(awk '/SwapTotal:/{print $2}' /proc/meminfo)"
SWAP_FREE_KB="$(awk '/SwapFree:/{print $2}' /proc/meminfo)"
SWAP_USED_MB="$(( (SWAP_TOTAL_KB - SWAP_FREE_KB) / 1024 ))"
MAX_SWAP_USED_MB="${QR_CORE_BACKUP_MAX_SWAP_USED_MB:-1536}"
if (( SWAP_TOTAL_KB > 0 && SWAP_USED_MB > MAX_SWAP_USED_MB )); then
  log "skip by swap guard (swap_used_mb=${SWAP_USED_MB} > ${MAX_SWAP_USED_MB})"
  exit 0
fi

TMP_DIR="$(mktemp -d /tmp/qr_core_backup.XXXXXX)"
ARCHIVE_TMP="/tmp/qr_core_backup_$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
trap 'rm -rf "$TMP_DIR" "$ARCHIVE_TMP"' EXIT

BUSY_TIMEOUT_MS="${QR_CORE_BACKUP_BUSY_TIMEOUT_MS:-250}"
SNAPSHOT_TIMEOUT_SEC="${QR_CORE_BACKUP_SNAPSHOT_TIMEOUT_SEC:-20}"
UPLOAD_TIMEOUT_SEC="${QR_CORE_BACKUP_UPLOAD_TIMEOUT_SEC:-90}"
MAX_DB_MB="${QR_CORE_BACKUP_MAX_DB_MB:-1024}"
MAX_FILE_MB="${QR_CORE_BACKUP_MAX_FILE_MB:-256}"

file_size_mb() {
  local path="$1"
  local size
  size="$(stat -c '%s' "$path" 2>/dev/null || echo 0)"
  echo $(( size / 1024 / 1024 ))
}

snapshot_sqlite_db() {
  local src="$1"
  local out_name="$2"
  if [[ ! -f "$src" ]]; then
    return 0
  fi
  local size_mb
  size_mb="$(file_size_mb "$src")"
  if (( size_mb > MAX_DB_MB )); then
    log "skip sqlite snapshot by size (${out_name}=${size_mb}MB > ${MAX_DB_MB}MB)"
    return 0
  fi
  if timeout "$SNAPSHOT_TIMEOUT_SEC" sqlite3 "$src" ".timeout ${BUSY_TIMEOUT_MS}" ".backup '${TMP_DIR}/${out_name}'"; then
    log "snapshotted ${out_name} (${size_mb}MB)"
    return 0
  fi
  log "snapshot failed: ${src} (skip)"
  return 0
}

copy_regular_file() {
  local src="$1"
  local out_name="$2"
  if [[ ! -f "$src" ]]; then
    return 0
  fi
  local size_mb
  size_mb="$(file_size_mb "$src")"
  if (( size_mb > MAX_FILE_MB )); then
    log "skip file by size (${out_name}=${size_mb}MB > ${MAX_FILE_MB}MB)"
    return 0
  fi
  cp -f "$src" "${TMP_DIR}/${out_name}"
  log "copied ${out_name} (${size_mb}MB)"
}

if is_true "${QR_CORE_BACKUP_INCLUDE_TRADES_DB:-1}" 1; then
  snapshot_sqlite_db "${LOG_DIR}/trades.db" "trades.db"
fi
if is_true "${QR_CORE_BACKUP_INCLUDE_SIGNALS_DB:-1}" 1; then
  snapshot_sqlite_db "${LOG_DIR}/signals.db" "signals.db"
fi
if is_true "${QR_CORE_BACKUP_INCLUDE_ORDERS_DB:-0}" 0; then
  snapshot_sqlite_db "${LOG_DIR}/orders.db" "orders.db"
fi
if is_true "${QR_CORE_BACKUP_INCLUDE_METRICS_DB:-0}" 0; then
  snapshot_sqlite_db "${LOG_DIR}/metrics.db" "metrics.db"
fi
if is_true "${QR_CORE_BACKUP_INCLUDE_ORDERS_SNAPSHOT_DB:-1}" 1; then
  copy_regular_file "${LOG_DIR}/orders_snapshot_48h.db" "orders_snapshot_48h.db"
fi
if is_true "${QR_CORE_BACKUP_INCLUDE_TRADES_SNAPSHOT_DB:-1}" 1; then
  copy_regular_file "${LOG_DIR}/trades_snapshot.db" "trades_snapshot.db"
fi
if is_true "${QR_CORE_BACKUP_INCLUDE_PIPELINE_LOG:-1}" 1; then
  copy_regular_file "${ROOT_DIR}/pipeline.log" "pipeline.log"
fi
if is_true "${QR_CORE_BACKUP_INCLUDE_HEALTH_JSON:-1}" 1; then
  copy_regular_file "${LOG_DIR}/health_snapshot.json" "health_snapshot.json"
fi

if [[ -z "$(find "$TMP_DIR" -maxdepth 1 -type f -print -quit)" ]]; then
  log "no backup candidates after guards, skip"
  exit 0
fi

tar -czf "$ARCHIVE_TMP" -C "$TMP_DIR" .
if [[ ! -s "$ARCHIVE_TMP" ]]; then
  log "archive is empty, skip"
  exit 1
fi

TOKEN="$(curl -s -H 'Metadata-Flavor: Google' \
  'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin).get("access_token",""))')"
if [[ -z "$TOKEN" ]]; then
  log "failed to fetch metadata access token"
  exit 1
fi

HOST_NAME="$(hostname -s)"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OBJECT_PATH="qr-logs/${HOST_NAME}/core_${TS}.tar.gz"
ENCODED_OBJECT="$(python3 -c 'import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1], safe=""))' "$OBJECT_PATH")"

HTTP_CODE="$(curl -s -o /dev/null -w '%{http_code}' \
  --max-time "$UPLOAD_TIMEOUT_SEC" \
  -X POST \
  -H "Authorization: Bearer ${TOKEN}" \
  -H 'Content-Type: application/octet-stream' \
  --data-binary "@${ARCHIVE_TMP}" \
  "https://storage.googleapis.com/upload/storage/v1/b/${BUCKET}/o?uploadType=media&name=${ENCODED_OBJECT}")"

if [[ "$HTTP_CODE" != "200" && "$HTTP_CODE" != "201" ]]; then
  log "upload failed (http=${HTTP_CODE}, object=${OBJECT_PATH})"
  exit 1
fi

log "uploaded gs://${BUCKET}/${OBJECT_PATH}"
