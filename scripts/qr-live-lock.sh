#!/usr/bin/env bash
# Shared live-runtime lock helpers for QuantRabbit shell wrappers.

QR_LIVE_LOCK_DIR=""
QR_LIVE_LOCK_TOKEN=""
QR_LIVE_LOCK_LABEL=""

qr_live_lock_pid() {
  local lock_dir="$1"
  if [[ -f "${lock_dir}/pid" ]]; then
    cat "${lock_dir}/pid" 2>/dev/null || true
  fi
}

qr_live_lock_pid_is_running() {
  local pid="$1"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

qr_live_lock_release() {
  if [[ "${QR_AUTOTRADE_LOCK_HELD:-0}" != "1" || -z "$QR_LIVE_LOCK_DIR" ]]; then
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
  date -u '+%Y-%m-%dT%H:%M:%SZ' > "${lock_dir}/started_at_utc"
  export QR_AUTOTRADE_LOCK_HELD=1
  qr_live_lock_install_traps
}

qr_live_lock_acquire() {
  local lock_dir="$1"
  local label="$2"
  local wait_seconds="${3:-0}"
  local wait_command_pattern="${4:-}"
  local poll_seconds="${5:-2}"

  if ! [[ "$wait_seconds" =~ ^[0-9]+$ ]]; then
    echo "[${label}] invalid lock wait seconds: ${wait_seconds}" >&2
    exit 2
  fi
  mkdir -p "$(dirname "$lock_dir")"

  local started_at
  local announced_wait=0
  local existing_pid
  local existing_command
  local now
  local elapsed
  started_at="$(date +%s)"

  while true; do
    if mkdir "$lock_dir" 2>/dev/null; then
      qr_live_lock_write_owner "$lock_dir" "$label"
      return 0
    fi

    existing_pid="$(qr_live_lock_pid "$lock_dir")"
    if qr_live_lock_pid_is_running "$existing_pid"; then
      existing_command="$(ps -p "$existing_pid" -o command= 2>/dev/null || true)"
      if [[ "$existing_command" == *"<defunct>"* ]]; then
        echo "[${label}] removing defunct lock holder pid=${existing_pid}: ${lock_dir}" >&2
        rm -rf "$lock_dir"
        continue
      fi
      if [[ -n "$wait_command_pattern" && "$existing_command" == *"$wait_command_pattern"* && "$wait_seconds" -gt 0 ]]; then
        now="$(date +%s)"
        elapsed=$((now - started_at))
        if [[ "$elapsed" -lt "$wait_seconds" ]]; then
          if [[ "$announced_wait" -eq 0 ]]; then
            echo "[${label}] lock held by ${existing_pid} (${existing_command}); waiting up to ${wait_seconds}s." >&2
            announced_wait=1
          fi
          sleep "$poll_seconds"
          continue
        fi
      fi

      if [[ -n "$existing_command" ]]; then
        echo "[${label}] another autotrade cycle is already running pid=${existing_pid}; command=${existing_command}; refusing overlap." >&2
      else
        echo "[${label}] another autotrade cycle is already running pid=${existing_pid}; refusing overlap." >&2
      fi
      exit 75
    fi

    echo "[${label}] removing stale lock: ${lock_dir}" >&2
    rm -rf "$lock_dir"
  done
}
