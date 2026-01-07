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
    local dst="$ARCHIVE_DIR/replay.${TS}.tgz"
    echo "[archive] replay -> $dst"
    tar -czf "$dst" -C "$LOG_DIR" replay
    rm -rf "$src"
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

echo "[start] log maintenance at $TS (LOG_DIR=$LOG_DIR)"
rotate_pipeline
archive_replay
checkpoint_db "$LOG_DIR/orders.db"
checkpoint_db "$LOG_DIR/trades.db"
checkpoint_db "$LOG_DIR/metrics.db"
prune_old
echo "[done] log maintenance complete."

