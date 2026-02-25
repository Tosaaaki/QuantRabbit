#!/usr/bin/env bash
# Lightweight cleanup for QuantRabbit VM.
# - Optionally sync replay logs to GCS
# - Prune old replay files
# - Prune stale temporary core archives
# - Vacuum journal logs
# - Auto-tighten retention when disk is near limit

set -euo pipefail

BASE="${BASE:-/home/tossaki/QuantRabbit}"
REPLAY_DIR="${REPLAY_DIR:-${BASE}/logs/replay}"
OANDA_DIR="${OANDA_DIR:-${BASE}/logs/oanda}"
LOG_DIR="${LOG_DIR:-${BASE}/logs}"
LOG_ARCHIVE_DIR="${LOG_ARCHIVE_DIR:-${LOG_DIR}/archive}"
TMP_DIR="${TMP_DIR:-/tmp}"
TMP_CORE_PATTERN="${TMP_CORE_PATTERN:-qr_logs_core_*.tar*}"
TMP_CORE_KEEP_DAYS="${TMP_CORE_KEEP_DAYS:-3}"
TMP_CORE_KEEP_FILES="${TMP_CORE_KEEP_FILES:-8}"
TMP_CORE_REMOVE_ZERO_BYTES="${TMP_CORE_REMOVE_ZERO_BYTES:-1}"
TMP_CORE_ZERO_GRACE_MIN="${TMP_CORE_ZERO_GRACE_MIN:-0}"
REPLAY_KEEP_DAYS="${REPLAY_KEEP_DAYS:-3}"
OANDA_KEEP_DAYS="${OANDA_KEEP_DAYS:-0}"  # 0 => do not delete OANDA logs by default
LOG_ARCHIVE_KEEP_DAYS="${LOG_ARCHIVE_KEEP_DAYS:-7}"
JOURNAL_VACUUM_DAYS="${JOURNAL_VACUUM_DAYS:-7}"
DB_MAINTENANCE_ENABLED="${DB_MAINTENANCE_ENABLED:-1}"
DB_VACUUM_TRIGGER_MB="${DB_VACUUM_TRIGGER_MB:-300}"
DB_VACUUM_MIN_AVAIL_MB="${DB_VACUUM_MIN_AVAIL_MB:-2048}"
DB_VACUUM_BUSY_TIMEOUT_MS="${DB_VACUUM_BUSY_TIMEOUT_MS:-3000}"
DB_TARGET_FILES="${DB_TARGET_FILES:-orders.db trades.db metrics.db trades_snapshot.db orders_snapshot_48h.db}"
DB_VACUUM_SKIP_FILES="${DB_VACUUM_SKIP_FILES:-orders.db trades.db metrics.db}"
DB_VACUUM_ALLOW_HOT_DBS="${DB_VACUUM_ALLOW_HOT_DBS:-0}"
DB_FORCE_VACUUM="${DB_FORCE_VACUUM:-0}"
DISK_BASED_LIGHTEN="${DISK_BASED_LIGHTEN:-1}" # 1 => tighten cleanup when disk is high
DISK_ROOT_PATH="${DISK_ROOT_PATH:-/}"
DISK_WARNING_PERCENT="${DISK_WARNING_PERCENT:-85}"
DISK_CRITICAL_PERCENT="${DISK_CRITICAL_PERCENT:-92}"
DISK_AGGR_REPLAY_KEEP_DAYS="${DISK_AGGR_REPLAY_KEEP_DAYS:-1}"
DISK_AGGR_OANDA_KEEP_DAYS="${DISK_AGGR_OANDA_KEEP_DAYS:-3}"
DISK_AGGR_TMP_CORE_KEEP_DAYS="${DISK_AGGR_TMP_CORE_KEEP_DAYS:-1}"
DISK_AGGR_TMP_CORE_KEEP_FILES="${DISK_AGGR_TMP_CORE_KEEP_FILES:-2}"
DISK_AGGR_JOURNAL_VACUUM_DAYS="${DISK_AGGR_JOURNAL_VACUUM_DAYS:-1}"
DISK_AGGR_LOG_ARCHIVE_KEEP_DAYS="${DISK_AGGR_LOG_ARCHIVE_KEEP_DAYS:-2}"
CLEANUP_LOCK_FILE="${CLEANUP_LOCK_FILE:-${TMP_DIR}/cleanup-qr-logs.${USER:-unknown}.lock}"
GCS_BUCKET="${GCS_BACKUP_BUCKET:-}"
CLEANUP_LOCK_PATH=""

log() {
  echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] $*"
}

cleanup_lock() {
  local lock_path="$CLEANUP_LOCK_FILE"
  if [[ -e "$lock_path" && ! -w "$lock_path" ]]; then
    lock_path="${TMP_DIR}/cleanup-qr-logs.$$.$(date +%s).lock"
    log "warn: cannot write default lock file, fallback ${lock_path}"
  fi
  if ! exec 9>"$lock_path" 2>/dev/null; then
    lock_path="${TMP_DIR}/cleanup-qr-logs.$$.$(date +%s).lock"
    log "warn: cannot open lock file ${lock_path}, fallback once"
    exec 9>"$lock_path"
  fi
  if command -v flock >/dev/null 2>&1; then
    if ! flock -n 9; then
      log "cleanup already running; skip."
      exit 0
    fi
  else
    log "warn: flock not available; skip lock (may run concurrently)"
  fi
  CLEANUP_LOCK_PATH="$lock_path"
}

cleanup_release_lock() {
  local lock_path="$CLEANUP_LOCK_PATH"
  if [[ -n "${lock_path}" && -f "$lock_path" ]]; then
    rm -f "$lock_path"
  fi
}

cleanup_lock_cleanup() {
  cleanup_release_lock
}

trap cleanup_lock_cleanup EXIT

bytes_to_mb() {
  local bytes="$1"
  echo $(( bytes / 1024 / 1024 ))
}

get_avail_mb() {
  df -P "$DISK_ROOT_PATH" | awk 'NR==2 {gsub(/G|M|K/, "", $4); print int($4)}'
}

get_disk_used_pct() {
  df -P "$1" | awk 'NR==2 {gsub(/%/, "", $5); print $5}'
}

apply_disk_guard() {
  if [[ "$DISK_BASED_LIGHTEN" != "1" ]]; then
    return
  fi

  local root_pct
  root_pct="$(get_disk_used_pct "$DISK_ROOT_PATH")"
  log "Disk usage on ${DISK_ROOT_PATH}: ${root_pct}%"

  if [[ "$root_pct" -ge "$DISK_CRITICAL_PERCENT" ]]; then
    log "Critical disk usage (${root_pct}% >= ${DISK_CRITICAL_PERCENT}%), tighten retention aggressively"
    REPLAY_KEEP_DAYS="$DISK_AGGR_REPLAY_KEEP_DAYS"
    OANDA_KEEP_DAYS="$DISK_AGGR_OANDA_KEEP_DAYS"
    TMP_CORE_KEEP_DAYS="$DISK_AGGR_TMP_CORE_KEEP_DAYS"
    TMP_CORE_KEEP_FILES="$DISK_AGGR_TMP_CORE_KEEP_FILES"
    JOURNAL_VACUUM_DAYS="$DISK_AGGR_JOURNAL_VACUUM_DAYS"
    LOG_ARCHIVE_KEEP_DAYS="$DISK_AGGR_LOG_ARCHIVE_KEEP_DAYS"
  elif [[ "$root_pct" -ge "$DISK_WARNING_PERCENT" ]]; then
    log "High disk usage (${root_pct}% >= ${DISK_WARNING_PERCENT}%). Keep logs tighter."
    if (( REPLAY_KEEP_DAYS > 1 )); then
      REPLAY_KEEP_DAYS=1
    fi
    TMP_CORE_KEEP_DAYS="${DISK_AGGR_TMP_CORE_KEEP_DAYS}"
    TMP_CORE_KEEP_FILES="${DISK_AGGR_TMP_CORE_KEEP_FILES}"
  elif [[ "$root_pct" -ge "$((DISK_WARNING_PERCENT + 5))" ]]; then
    DB_FORCE_VACUUM=1
  fi
}

db_maintenance() {
  if [[ "$DB_MAINTENANCE_ENABLED" != "1" ]]; then
    return
  fi
  if ! command -v sqlite3 >/dev/null 2>&1; then
    log "warn: sqlite3 not found; skip DB maintenance"
    return
  fi

  local -a db_files=($DB_TARGET_FILES)
  if [[ ${#db_files[@]} -eq 0 ]]; then
    return
  fi

  local available_mb
  available_mb="$(get_avail_mb)"
  local db
  local skip_db
  local db_path
  local size_bytes
  local size_mb
  local should_vacuum
  local skip_vacuum

  for db in "${db_files[@]}"; do
    db_path="${LOG_DIR}/${db}"
    if [[ ! -f "$db_path" ]]; then
      continue
    fi
    size_bytes="$(stat -c '%s' "$db_path")"
    size_mb="$(bytes_to_mb "$size_bytes")"

    log "DB maintenance: checking ${db_path} (${size_mb}MB)"
    sqlite3 "$db_path" "PRAGMA busy_timeout=${DB_VACUUM_BUSY_TIMEOUT_MS}; PRAGMA wal_checkpoint(TRUNCATE);" || {
      log "warn: DB checkpoint failed: ${db_path}"
      continue
    }
    should_vacuum=0
    if [[ "$DB_FORCE_VACUUM" == "1" || "$size_mb" -ge "$DB_VACUUM_TRIGGER_MB" ]]; then
      should_vacuum=1
    fi
    if [[ "$should_vacuum" != "1" ]]; then
      continue
    fi

    skip_vacuum=0
    if [[ "$DB_VACUUM_ALLOW_HOT_DBS" != "1" ]]; then
      for skip_db in $DB_VACUUM_SKIP_FILES; do
        if [[ "$db" == "$skip_db" ]]; then
          skip_vacuum=1
          break
        fi
      done
    fi
    if [[ "$skip_vacuum" == "1" ]]; then
      log "DB maintenance: skip VACUUM for hot DB ${db_path} (DB_VACUUM_ALLOW_HOT_DBS=0)"
      continue
    fi

    if (( available_mb < DB_VACUUM_MIN_AVAIL_MB )); then
      log "warn: skip vacuum for ${db_path} (available MB=${available_mb} < threshold=${DB_VACUUM_MIN_AVAIL_MB})"
      continue
    fi
    log "DB maintenance: VACUUM ${db_path}"
    if ! sqlite3 "$db_path" "VACUUM;"; then
      log "warn: DB vacuum failed: ${db_path}"
    fi
  done
}

cleanup_archives() {
  if [[ ! -d "$LOG_ARCHIVE_DIR" ]]; then
    return
  fi
  if [[ "$LOG_ARCHIVE_KEEP_DAYS" == "0" ]]; then
    return
  fi
  log "Prune log archives older than ${LOG_ARCHIVE_KEEP_DAYS}d in ${LOG_ARCHIVE_DIR}"
  find "$LOG_ARCHIVE_DIR" -maxdepth 1 -type f \
    \( -name '*.tar' -o -name '*.tgz' -o -name '*.gz' -o -name '*.log' \) \
    -mtime "+${LOG_ARCHIVE_KEEP_DAYS}" -print -delete
}

cleanup_tmp_core_archives() {
  if [[ ! -d "$TMP_DIR" ]]; then
    log "warn: TMP_DIR not found: $TMP_DIR"
    return
  fi

  log "Prune tmp core archives older than ${TMP_CORE_KEEP_DAYS}d (pattern=${TMP_CORE_PATTERN})"
  if [[ "$TMP_CORE_REMOVE_ZERO_BYTES" == "1" ]]; then
    find "$TMP_DIR" -maxdepth 1 -type f -name "$TMP_CORE_PATTERN" -size 0 -mmin "+${TMP_CORE_ZERO_GRACE_MIN}" -print -delete
  fi
  if [[ "$TMP_CORE_KEEP_DAYS" != "0" ]]; then
    find "$TMP_DIR" -maxdepth 1 -type f -name "$TMP_CORE_PATTERN" -mtime "+${TMP_CORE_KEEP_DAYS}" -print -delete
  fi

  if [[ "$TMP_CORE_KEEP_FILES" == "0" ]]; then
    return
  fi

  local -a ordered=()

  mapfile -t ordered < <(find "$TMP_DIR" -maxdepth 1 -type f -name "$TMP_CORE_PATTERN" -printf '%T@ %p\n' | sort -n)
  local total=${#ordered[@]}
  if [[ $total -le "$TMP_CORE_KEEP_FILES" ]]; then
    log "tmp core archive retention by file count: keep all ($total <= ${TMP_CORE_KEEP_FILES})"
    return
  fi

  local to_delete=$(( total - TMP_CORE_KEEP_FILES ))
  local idx=0
  local line=""
  local file=""
  log "tmp core archive retention by file count: keep ${TMP_CORE_KEEP_FILES}, delete ${to_delete} older file(s)"
  for ((idx = 0; idx < to_delete; idx++)); do
    line="${ordered[$idx]}"
    file=${line#* }
    [[ -f "$file" ]] && rm -f "$file"
  done
}

cleanup_lock

if [[ -n "$GCS_BUCKET" && -d "$REPLAY_DIR" ]]; then
  if command -v gsutil >/dev/null 2>&1; then
    log "Sync replay logs to gs://$GCS_BUCKET/replay ..."
    gsutil -m rsync -r "$REPLAY_DIR" "gs://${GCS_BUCKET}/replay" || log "warn: gsutil rsync failed"
  else
    log "warn: gsutil not found; skip GCS sync"
  fi
fi

apply_disk_guard

if [[ -d "$REPLAY_DIR" ]]; then
  log "Prune replay older than ${REPLAY_KEEP_DAYS}d"
  find "$REPLAY_DIR" -type f -mtime "+${REPLAY_KEEP_DAYS}" -print -delete
fi

cleanup_tmp_core_archives
cleanup_archives

if [[ "$OANDA_KEEP_DAYS" != "0" && -d "$OANDA_DIR" ]]; then
  log "Prune oanda logs older than ${OANDA_KEEP_DAYS}d"
  find "$OANDA_DIR" -type f -mtime "+${OANDA_KEEP_DAYS}" -print -delete
fi

db_maintenance

log "Vacuum journal to ${JOURNAL_VACUUM_DAYS}d"
if command -v journalctl >/dev/null 2>&1; then
  if [[ "$JOURNAL_VACUUM_DAYS" != "0" ]]; then
    journalctl --rotate || true
    journalctl --vacuum-time="${JOURNAL_VACUUM_DAYS}d" || true
  else
    log "skip journal vacuum (JOURNAL_VACUUM_DAYS=0)"
  fi
else
  log "warn: journalctl not found"
fi

log "Cleanup complete"
