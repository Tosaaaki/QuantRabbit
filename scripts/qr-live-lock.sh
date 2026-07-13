#!/usr/bin/env bash
# Shared live-runtime lock helpers for QuantRabbit shell wrappers.

QR_LIVE_LOCK_DIR=""
QR_LIVE_LOCK_TOKEN=""
QR_LIVE_LOCK_LABEL=""
QR_LIVE_LOCK_GUARD_HELD=0
QR_LIVE_LOCK_GUARD_PATH=""

qr_live_lock_pid() {
  local lock_dir="$1"
  if [[ -f "${lock_dir}/pid" ]]; then
    cat "${lock_dir}/pid" 2>/dev/null || true
  fi
}

qr_live_lock_pid_state() {
  local pid="$1"
  if ! [[ "$pid" =~ ^[0-9]+$ ]]; then
    return 1
  fi
  ps -p "$pid" -o stat= 2>/dev/null || true
}

qr_live_lock_pid_is_defunct() {
  local pid="$1"
  local state
  state="$(qr_live_lock_pid_state "$pid")"
  state="${state#"${state%%[![:space:]]*}"}"
  [[ "$state" == Z* ]]
}

qr_live_lock_pid_is_running() {
  local pid="$1"
  [[ "$pid" =~ ^[0-9]+$ ]] \
    && kill -0 "$pid" 2>/dev/null \
    && ! qr_live_lock_pid_is_defunct "$pid"
}

qr_live_lock_process_started_at() {
  local pid="$1"
  local value=""
  if ! [[ "$pid" =~ ^[0-9]+$ ]]; then
    return 1
  fi
  value="$(LC_ALL=C ps -p "$pid" -o lstart= 2>/dev/null || true)"
  qr_live_lock_normalize_process_started_at "$value"
}

qr_live_lock_normalize_process_started_at() {
  local value="$1"
  printf '%s\n' "$value" | LC_ALL=C awk '{$1=$1; print}'
}

qr_live_lock_process_started_at_is_valid() {
  local value="$1"
  if [[ "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
    return 1
  fi
  printf '%s\n' "$value" | LC_ALL=C grep -Eq \
    '^[[:space:]]*(Mon|Tue|Wed|Thu|Fri|Sat|Sun)[[:space:]]+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[[:space:]]+[0-9]{1,2}[[:space:]]+[0-9]{2}:[0-9]{2}:[0-9]{2}[[:space:]]+[0-9]{4}[[:space:]]*$'
}

qr_live_lock_pid_matches_owner() {
  local lock_dir="$1"
  local pid="$2"
  local recorded_started_at=""
  local current_started_at=""
  local recorded_line_count=""

  if [[ ! -f "${lock_dir}/process_started_at" ]]; then
    return 0
  fi
  recorded_line_count="$(LC_ALL=C wc -l < "${lock_dir}/process_started_at" | tr -d '[:space:]')"
  if [[ "$recorded_line_count" != "1" ]] \
    || LC_ALL=C od -An -v -tx1 "${lock_dir}/process_started_at" 2>/dev/null \
      | grep -Eq '(^|[[:space:]])00([[:space:]]|$)'; then
    return 0
  fi
  recorded_started_at="$(cat "${lock_dir}/process_started_at" 2>/dev/null || true)"
  current_started_at="$(qr_live_lock_process_started_at "$pid")"
  # Legacy locks and a temporarily unavailable `ps` remain fail-closed. Only
  # an explicit birth-time mismatch proves that the PID has been recycled.
  if ! qr_live_lock_process_started_at_is_valid "$recorded_started_at" \
    || ! qr_live_lock_process_started_at_is_valid "$current_started_at"; then
    return 0
  fi
  recorded_started_at="$(qr_live_lock_normalize_process_started_at "$recorded_started_at")"
  current_started_at="$(qr_live_lock_normalize_process_started_at "$current_started_at")"
  [[ "$recorded_started_at" == "$current_started_at" ]]
}

qr_live_lock_guard_acquire() {
  local lock_dir="$1"
  local timeout_seconds="${QR_LIVE_LOCK_GUARD_TIMEOUT_SECONDS:-10}"
  local guard_path="${lock_dir}.acquire.guard"

  if [[ "$QR_LIVE_LOCK_GUARD_HELD" == "1" ]]; then
    [[ "$QR_LIVE_LOCK_GUARD_PATH" == "$guard_path" ]]
    return
  fi
  if ! [[ "$timeout_seconds" =~ ^[0-9]+$ ]]; then
    echo "[qr-live-lock] invalid generation-guard timeout: ${timeout_seconds}" >&2
    return 2
  fi
  mkdir -p "$(dirname "$guard_path")"
  exec 9>>"$guard_path"
  if command -v lockf >/dev/null 2>&1; then
    if ! lockf -s -t "$timeout_seconds" 9; then
      exec 9>&-
      return 75
    fi
  elif command -v flock >/dev/null 2>&1; then
    if ! flock -x -w "$timeout_seconds" 9; then
      exec 9>&-
      return 75
    fi
  else
    echo "[qr-live-lock] neither lockf nor flock is available; refusing unsafe lock mutation." >&2
    exec 9>&-
    return 2
  fi
  QR_LIVE_LOCK_GUARD_HELD=1
  QR_LIVE_LOCK_GUARD_PATH="$guard_path"
}

qr_live_lock_guard_release() {
  if [[ "$QR_LIVE_LOCK_GUARD_HELD" != "1" ]]; then
    return 0
  fi
  exec 9>&-
  QR_LIVE_LOCK_GUARD_HELD=0
  QR_LIVE_LOCK_GUARD_PATH=""
}

qr_live_lock_release() {
  if [[ "${QR_AUTOTRADE_LOCK_HELD:-0}" != "1" || -z "$QR_LIVE_LOCK_DIR" ]]; then
    return 0
  fi

  if ! qr_live_lock_guard_acquire "$QR_LIVE_LOCK_DIR"; then
    echo "[${QR_LIVE_LOCK_LABEL:-qr-live-lock}] could not acquire generation guard for owner release; leaving lock fail-closed." >&2
    return 0
  fi

  local current_token=""
  local current_pid=""
  if [[ -f "${QR_LIVE_LOCK_DIR}/token" ]]; then
    current_token="$(cat "${QR_LIVE_LOCK_DIR}/token" 2>/dev/null || true)"
  fi
  if [[ -f "${QR_LIVE_LOCK_DIR}/pid" ]]; then
    current_pid="$(cat "${QR_LIVE_LOCK_DIR}/pid" 2>/dev/null || true)"
  fi

  if [[ -n "$QR_LIVE_LOCK_TOKEN" && "$current_token" == "$QR_LIVE_LOCK_TOKEN" ]]; then
    rm -rf "$QR_LIVE_LOCK_DIR"
  elif [[ -z "$current_token" && "$current_pid" == "$$" ]]; then
    rm -rf "$QR_LIVE_LOCK_DIR"
  fi
  qr_live_lock_guard_release
}

qr_live_lock_install_traps() {
  trap qr_live_lock_release EXIT
  trap 'qr_live_lock_release; exit 130' INT
  trap 'qr_live_lock_release; exit 143' TERM
}

qr_live_lock_write_owner() {
  local lock_dir="$1"
  local label="$2"
  QR_LIVE_LOCK_DIR="$lock_dir"
  QR_LIVE_LOCK_LABEL="$label"
  QR_LIVE_LOCK_TOKEN="${$}:$(date +%s):${RANDOM:-0}"
  printf '%s\n' "$$" > "${lock_dir}/pid"
  printf '%s\n' "$QR_LIVE_LOCK_TOKEN" > "${lock_dir}/token"
  printf '%s\n' "$label" > "${lock_dir}/command"
  local process_started_at
  process_started_at="$(qr_live_lock_process_started_at "$$")"
  if [[ -n "$process_started_at" ]]; then
    printf '%s\n' "$process_started_at" > "${lock_dir}/process_started_at"
  fi
  date -u '+%Y-%m-%dT%H:%M:%SZ' > "${lock_dir}/started_at_utc"
  export QR_AUTOTRADE_LOCK_HELD=1
  export QR_AUTOTRADE_LOCK_OWNER_TOKEN="$QR_LIVE_LOCK_TOKEN"
  qr_live_lock_install_traps
}

qr_live_lock_acquire() {
  local lock_dir="$1"
  local label="$2"
  local wait_seconds="${3:-0}"
  local wait_command_pattern="${4:-}"
  local poll_seconds="${5:-2}"
  local init_grace_seconds="${QR_LIVE_LOCK_INIT_GRACE_SECONDS:-1}"

  if ! [[ "$wait_seconds" =~ ^[0-9]+$ ]]; then
    echo "[${label}] invalid lock wait seconds: ${wait_seconds}" >&2
    exit 2
  fi
  if ! [[ "$init_grace_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "[${label}] invalid lock initialization grace seconds: ${init_grace_seconds}" >&2
    exit 2
  fi
  mkdir -p "$(dirname "$lock_dir")"

  local started_at
  local announced_wait=0
  local existing_pid
  local existing_command
  local existing_label
  local missing_pid_observed=0
  local now
  local elapsed
  started_at="$(date +%s)"

  while true; do
    if ! qr_live_lock_guard_acquire "$lock_dir"; then
      echo "[${label}] generation guard is busy; refusing unsafe lock acquisition." >&2
      exit 75
    fi
    if mkdir "$lock_dir" 2>/dev/null; then
      qr_live_lock_write_owner "$lock_dir" "$label"
      qr_live_lock_guard_release
      return 0
    fi

    existing_pid="$(qr_live_lock_pid "$lock_dir")"
    if ! [[ "$existing_pid" =~ ^[0-9]+$ ]]; then
      if [[ "$missing_pid_observed" -eq 0 ]]; then
        # `mkdir` wins the lock before its owner can persist metadata. Give
        # that tiny initialization window one bounded grace interval instead
        # of deleting a newly acquired lock and admitting overlapping cycles.
        echo "[${label}] lock owner metadata is initializing: ${lock_dir}; waiting ${init_grace_seconds}s." >&2
        missing_pid_observed=1
        qr_live_lock_guard_release
        sleep "$init_grace_seconds"
        continue
      fi
      echo "[${label}] removing stale lock with missing owner metadata: ${lock_dir}" >&2
      rm -rf "$lock_dir"
      missing_pid_observed=0
      qr_live_lock_guard_release
      continue
    fi
    missing_pid_observed=0
    if qr_live_lock_pid_is_defunct "$existing_pid"; then
      echo "[${label}] removing defunct lock holder pid=${existing_pid}: ${lock_dir}" >&2
      rm -rf "$lock_dir"
      qr_live_lock_guard_release
      continue
    fi
    if qr_live_lock_pid_is_running "$existing_pid" \
      && ! qr_live_lock_pid_matches_owner "$lock_dir" "$existing_pid"; then
      echo "[${label}] removing recycled-pid lock holder pid=${existing_pid}: ${lock_dir}" >&2
      rm -rf "$lock_dir"
      qr_live_lock_guard_release
      continue
    fi
    if qr_live_lock_pid_is_running "$existing_pid"; then
      existing_command="$(ps -p "$existing_pid" -o command= 2>/dev/null || true)"
      existing_label=""
      if [[ -f "${lock_dir}/command" ]]; then
        existing_label="$(cat "${lock_dir}/command" 2>/dev/null || true)"
      fi
      # The holder may exit between the state probe above and this command
      # probe. Darwin exposes that transition as `<defunct>` in `command`
      # before every observer necessarily sees `Z` in `stat`.
      if [[ "$existing_command" == *"<defunct>"* ]] \
        || qr_live_lock_pid_is_defunct "$existing_pid"; then
        echo "[${label}] removing defunct lock holder pid=${existing_pid}: ${lock_dir}" >&2
        rm -rf "$lock_dir"
        qr_live_lock_guard_release
        continue
      fi
      if ! kill -0 "$existing_pid" 2>/dev/null; then
        echo "[${label}] removing stale lock: ${lock_dir}" >&2
        rm -rf "$lock_dir"
        qr_live_lock_guard_release
        continue
      fi
      if [[ -n "$wait_command_pattern" \
        && ( "$existing_label" == *"$wait_command_pattern"* || "$existing_command" == *"$wait_command_pattern"* ) \
        && "$wait_seconds" -gt 0 ]]; then
        now="$(date +%s)"
        elapsed=$((now - started_at))
        if [[ "$elapsed" -lt "$wait_seconds" ]]; then
          if [[ "$announced_wait" -eq 0 ]]; then
            echo "[${label}] lock held by ${existing_pid} (label=${existing_label:-unknown}; command=${existing_command:-unknown}); waiting up to ${wait_seconds}s." >&2
            announced_wait=1
          fi
          qr_live_lock_guard_release
          sleep "$poll_seconds"
          continue
        fi
      fi

      if [[ -n "$existing_command" ]]; then
        echo "[${label}] another autotrade cycle is already running pid=${existing_pid}; command=${existing_command}; refusing overlap." >&2
      else
        echo "[${label}] another autotrade cycle is already running pid=${existing_pid}; refusing overlap." >&2
      fi
      qr_live_lock_guard_release
      exit 75
    fi

    echo "[${label}] removing stale lock: ${lock_dir}" >&2
    rm -rf "$lock_dir"
    qr_live_lock_guard_release
  done
}
