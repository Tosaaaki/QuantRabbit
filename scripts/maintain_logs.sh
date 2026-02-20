#!/usr/bin/env bash
# Log maintenance utility for QuantRabbit.
# - Rotates pipeline.log by gzipping a timestamped copy and truncating the live file.
# - Archives logs/replay into a tar.gz and removes the original directory.
# - WAL checkpoints (orders/trades/metrics) to shrink WAL files.
# - Prunes archived artifacts older than LOG_ROTATE_MAX_AGE_DAYS (default: 7).

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
ARCHIVE_DIR="${ARCHIVE_DIR:-$LOG_DIR/archive}"
MAX_AGE_DAYS="${LOG_ROTATE_MAX_AGE_DAYS:-7}"
ARCHIVE_BUCKET="${LOG_ARCHIVE_BUCKET:-}"
ARCHIVE_PREFIX="${LOG_ARCHIVE_PREFIX:-$(hostname -s)/logs}"
TS="$(date +%Y%m%d-%H%M%S)"

mkdir -p "$ARCHIVE_DIR"

rotate_pipeline() {
  local src="$LOG_DIR/pipeline.log"
  if [[ -f "$src" ]]; then
    local dst="$ARCHIVE_DIR/pipeline.log.${TS}.gz"
    echo "[rotate] pipeline.log -> $dst"
    gzip -c "$src" > "$dst"
    : > "$src"
  else
    echo "[rotate] pipeline.log not found (skip)"
  fi
}

archive_replay() {
  local src="$LOG_DIR/replay"
  if [[ -d "$src" ]]; then
    local stage="$ARCHIVE_DIR/replay.${TS}.dir"
    local dst="$ARCHIVE_DIR/replay.${TS}.tgz"
    echo "[archive] replay stage -> $stage"
    if ! mv "$src" "$stage"; then
      echo "[archive] replay move failed (skip)"
      return
    fi
    # Keep the live path available for writers while we archive the staged snapshot.
    mkdir -p "$src"
    echo "[archive] replay -> $dst"
    if tar -czf "$dst" -C "$ARCHIVE_DIR" "$(basename "$stage")"; then
      if ! rm -rf "$stage"; then
        echo "[archive] warning: staged replay cleanup failed: $stage"
      fi
    else
      echo "[archive] warning: replay archive failed, preserving staged dir: $stage"
    fi
  else
    echo "[archive] replay not found (skip)"
  fi
}

checkpoint_db() {
  local db="$1"
  if [[ -f "$db" ]]; then
    echo "[checkpoint] $db"
    sqlite3 "$db" "pragma wal_checkpoint(full);"
  fi
}

prune_old() {
  echo "[prune] deleting archives older than ${MAX_AGE_DAYS} days in $ARCHIVE_DIR"
  find "$ARCHIVE_DIR" -type f -mtime +"$MAX_AGE_DAYS" -print -delete
}

upload_to_gcs() {
  if [[ -z "$ARCHIVE_BUCKET" ]]; then
    echo "[gcs] LOG_ARCHIVE_BUCKET not set; skip upload"
    return
  fi
  shopt -s nullglob
  local files=("$ARCHIVE_DIR"/*.gz "$ARCHIVE_DIR"/*.tgz)
  if [[ ${#files[@]} -eq 0 ]]; then
    echo "[gcs] no archives to upload"
    return
  fi
  local dest="${ARCHIVE_BUCKET%/}/${ARCHIVE_PREFIX}"
  echo "[gcs] uploading ${#files[@]} archive(s) to ${dest}"
  gsutil -m cp "${files[@]}" "${dest}/"
}

echo "[start] log maintenance at $TS (LOG_DIR=$LOG_DIR)"
rotate_pipeline
archive_replay
checkpoint_db "$LOG_DIR/orders.db"
checkpoint_db "$LOG_DIR/trades.db"
checkpoint_db "$LOG_DIR/metrics.db"
upload_to_gcs
prune_old
echo "[done] log maintenance complete."
