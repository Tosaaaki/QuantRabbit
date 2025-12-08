#!/usr/bin/env bash
# Lightweight cleanup for QuantRabbit VM.
# - Optionally sync replay logs to GCS
# - Prune old replay files
# - Vacuum journal logs

set -euo pipefail

BASE="${BASE:-/home/tossaki/QuantRabbit}"
REPLAY_DIR="${REPLAY_DIR:-${BASE}/logs/replay}"
OANDA_DIR="${OANDA_DIR:-${BASE}/logs/oanda}"
REPLAY_KEEP_DAYS="${REPLAY_KEEP_DAYS:-3}"
OANDA_KEEP_DAYS="${OANDA_KEEP_DAYS:-0}"  # 0 => do not delete OANDA logs by default
JOURNAL_VACUUM_DAYS="${JOURNAL_VACUUM_DAYS:-7}"
GCS_BUCKET="${GCS_BACKUP_BUCKET:-}"

log() { echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] $*"; }

if [[ -n "$GCS_BUCKET" && -d "$REPLAY_DIR" ]]; then
  if command -v gsutil >/dev/null 2>&1; then
    log "Sync replay logs to gs://$GCS_BUCKET/replay ..."
    gsutil -m rsync -r "$REPLAY_DIR" "gs://${GCS_BUCKET}/replay" || log "warn: gsutil rsync failed"
  else
    log "warn: gsutil not found; skip GCS sync"
  fi
fi

if [[ -d "$REPLAY_DIR" ]]; then
  log "Prune replay older than ${REPLAY_KEEP_DAYS}d"
  find "$REPLAY_DIR" -type f -mtime "+${REPLAY_KEEP_DAYS}" -print -delete
fi

if [[ "$OANDA_KEEP_DAYS" != "0" && -d "$OANDA_DIR" ]]; then
  log "Prune oanda logs older than ${OANDA_KEEP_DAYS}d"
  find "$OANDA_DIR" -type f -mtime "+${OANDA_KEEP_DAYS}" -print -delete
fi

log "Vacuum journal to ${JOURNAL_VACUUM_DAYS}d"
if command -v journalctl >/dev/null 2>&1; then
  journalctl --rotate || true
  journalctl --vacuum-time="${JOURNAL_VACUUM_DAYS}d" || true
else
  log "warn: journalctl not found"
fi

log "Cleanup complete"
