#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STACK_SCRIPT="${ROOT_DIR}/scripts/local_v2_stack.sh"
LOG_DIR="${ROOT_DIR}/logs"
LOG_FILE="${LOG_DIR}/local_v2_autorecover.log"
LOCK_DIR="${LOG_DIR}/local_v2_autorecover.lock"
LOCK_PID_FILE="${LOCK_DIR}/pid"
STATE_FILE="${LOG_DIR}/local_v2_autorecover.state"

PROFILE="${QR_LOCAL_V2_PROFILE:-trade_min}"
ENV_FILE="${QR_LOCAL_V2_ENV_FILE:-${ROOT_DIR}/ops/env/local-v2-stack.env}"
SERVICES="${QR_LOCAL_V2_SERVICES:-}"
NETWORK_HOST="${QR_LOCAL_V2_NET_CHECK_HOST:-api-fxtrade.oanda.com}"
NETWORK_PORT="${QR_LOCAL_V2_NET_CHECK_PORT:-443}"
NETWORK_TIMEOUT_SEC="${QR_LOCAL_V2_NET_TIMEOUT_SEC:-2.0}"
VERBOSE="${QR_LOCAL_V2_AUTORECOVER_VERBOSE:-0}"
RESUME_GAP_SEC="${QR_LOCAL_V2_RESUME_GAP_SEC:-90}"
NET_RECOVERY_RESTART_MARKET_DATA="${QR_LOCAL_V2_NET_RECOVERY_RESTART_MARKET_DATA:-1}"
NET_RECOVERY_RESTART_COOLDOWN_SEC="${QR_LOCAL_V2_NET_RECOVERY_RESTART_COOLDOWN_SEC:-60}"
STALE_RECOVERY_ENABLED="${QR_LOCAL_V2_STALE_RECOVERY_ENABLED:-0}"

usage() {
  cat <<'USAGE'
Usage:
  scripts/local_v2_autorecover_once.sh [options]

Options:
  --profile <name>    local_v2_stack profile (default: trade_min)
  --env <file>        override env file (default: ops/env/local-v2-stack.env)
  --services <csv>    optional explicit services list
  -h, --help          show help
USAGE
}

log() {
  local msg="$1"
  mkdir -p "${LOG_DIR}"
  printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "${msg}" >>"${LOG_FILE}"
}

as_positive_int() {
  local value="${1:-}"
  local default="$2"
  if [[ "${value}" =~ ^[0-9]+$ ]] && (( value > 0 )); then
    printf '%s\n' "${value}"
    return 0
  fi
  printf '%s\n' "${default}"
}

is_truthy() {
  local raw="${1:-}"
  case "$(printf '%s' "${raw}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

load_state() {
  PREV_EPOCH=""
  PREV_NETWORK=""
  PREV_NET_RECOVERY_RESTART_EPOCH=""
  if [[ ! -f "${STATE_FILE}" ]]; then
    return 0
  fi

  while IFS='=' read -r key value; do
    case "${key}" in
      last_epoch) PREV_EPOCH="${value}" ;;
      last_network) PREV_NETWORK="${value}" ;;
      last_net_recovery_restart_epoch) PREV_NET_RECOVERY_RESTART_EPOCH="${value}" ;;
    esac
  done <"${STATE_FILE}"
}

persist_state() {
  local now_epoch="$1"
  local network_state="$2"
  local last_restart_epoch="$3"
  local tmp

  tmp="${STATE_FILE}.tmp.$$"
  printf 'last_epoch=%s\n' "${now_epoch}" >"${tmp}"
  printf 'last_network=%s\n' "${network_state}" >>"${tmp}"
  printf 'last_net_recovery_restart_epoch=%s\n' "${last_restart_epoch}" >>"${tmp}"
  mv "${tmp}" "${STATE_FILE}"
}

acquire_lock() {
  local owner_pid=""

  if mkdir "${LOCK_DIR}" 2>/dev/null; then
    printf '%s\n' "$$" >"${LOCK_PID_FILE}"
    return 0
  fi

  if [[ -f "${LOCK_PID_FILE}" ]]; then
    owner_pid="$(cat "${LOCK_PID_FILE}" 2>/dev/null || true)"
  fi

  if [[ -n "${owner_pid}" ]] && kill -0 "${owner_pid}" 2>/dev/null; then
    return 1
  fi

  rm -rf "${LOCK_DIR}" >/dev/null 2>&1 || true
  if mkdir "${LOCK_DIR}" 2>/dev/null; then
    printf '%s\n' "$$" >"${LOCK_PID_FILE}"
    log "[warn] removed stale lock owner_pid=${owner_pid:-unknown}"
    return 0
  fi

  return 1
}

release_lock() {
  local owner_pid=""
  if [[ -f "${LOCK_PID_FILE}" ]]; then
    owner_pid="$(cat "${LOCK_PID_FILE}" 2>/dev/null || true)"
  fi
  if [[ "${owner_pid}" == "$$" ]]; then
    rm -rf "${LOCK_DIR}" >/dev/null 2>&1 || true
  fi
}

run_stack() {
  local cmd="$1"
  shift || true
  local args=("${STACK_SCRIPT}" "${cmd}" "--profile" "${PROFILE}")
  if [[ -n "${ENV_FILE}" ]]; then
    args+=("--env" "${ENV_FILE}")
  fi
  if [[ -n "${SERVICES}" ]]; then
    args+=("--services" "${SERVICES}")
  fi
  if [[ $# -gt 0 ]]; then
    args+=("$@")
  fi
  "${args[@]}"
}

network_ready() {
  local py_exec=""
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    py_exec="${ROOT_DIR}/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    py_exec="$(command -v python3)"
  else
    return 1
  fi

  "${py_exec}" - "${NETWORK_HOST}" "${NETWORK_PORT}" "${NETWORK_TIMEOUT_SEC}" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
timeout = float(sys.argv[3])

try:
    with socket.create_connection((host, port), timeout=timeout):
        pass
except Exception:
    raise SystemExit(1)

raise SystemExit(0)
PY
}

if [[ ! -x "${STACK_SCRIPT}" ]]; then
  echo "[error] local_v2_stack script not executable: ${STACK_SCRIPT}" >&2
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --env)
      ENV_FILE="$2"
      shift 2
      ;;
    --services)
      SERVICES="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[error] unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

case "${ENV_FILE}" in
  /*) ;;
  *) ENV_FILE="${ROOT_DIR}/${ENV_FILE}" ;;
esac

if [[ -n "${ENV_FILE}" && ! -f "${ENV_FILE}" ]]; then
  echo "[error] env file not found: ${ENV_FILE}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
if ! acquire_lock; then
  # Already running; skip overlapping recovery jobs.
  exit 0
fi
trap release_lock EXIT INT TERM

now_epoch="$(date +%s)"
resume_gap_sec="$(as_positive_int "${RESUME_GAP_SEC}" "90")"
restart_cooldown_sec="$(as_positive_int "${NET_RECOVERY_RESTART_COOLDOWN_SEC}" "60")"
load_state
if [[ -n "${PREV_EPOCH}" ]] && [[ "${PREV_EPOCH}" =~ ^[0-9]+$ ]] && (( PREV_EPOCH >= 946684800 )) && (( now_epoch > PREV_EPOCH )); then
  gap_sec=$((now_epoch - PREV_EPOCH))
  if (( gap_sec >= resume_gap_sec )); then
    log "[resume] detected polling gap=${gap_sec}s threshold=${resume_gap_sec}s"
  fi
fi

network_state="down"
if network_ready; then
  network_state="up"
fi

if [[ "${PREV_NETWORK}" == "down" && "${network_state}" == "up" ]]; then
  log "[resume] network recovered host=${NETWORK_HOST}:${NETWORK_PORT}"
elif [[ "${PREV_NETWORK}" == "up" && "${network_state}" == "down" ]]; then
  log "[wait] network became unavailable host=${NETWORK_HOST}:${NETWORK_PORT}"
fi

last_restart_epoch="0"
if [[ -n "${PREV_NET_RECOVERY_RESTART_EPOCH}" ]] && [[ "${PREV_NET_RECOVERY_RESTART_EPOCH}" =~ ^[0-9]+$ ]]; then
  last_restart_epoch="${PREV_NET_RECOVERY_RESTART_EPOCH}"
fi

if [[ "${PREV_NETWORK}" == "down" && "${network_state}" == "up" ]] && is_truthy "${NET_RECOVERY_RESTART_MARKET_DATA}"; then
  if (( now_epoch - last_restart_epoch >= restart_cooldown_sec )); then
    set +e
    restart_out="$(run_stack restart --services quant-market-data-feed 2>&1)"
    restart_rc=$?
    set -e
    if [[ ${restart_rc} -eq 0 ]]; then
      last_restart_epoch="${now_epoch}"
      log "[recover] network recovered -> restarted quant-market-data-feed"
    elif [[ ${restart_rc} -eq 3 ]]; then
      log "[skip] parity conflict; skip market-data-feed restart on network recovery"
    else
      log "[warn] market-data-feed restart on network recovery failed rc=${restart_rc}: ${restart_out//$'\n'/ ; }"
    fi
  else
    log "[skip] market-data-feed restart cooldown active (${restart_cooldown_sec}s)"
  fi
fi

persist_state "${now_epoch}" "${network_state}" "${last_restart_epoch}"

set +e
status_out="$(run_stack status 2>&1)"
status_rc=$?
set -e
if [[ ${status_rc} -ne 0 ]]; then
  log "[warn] status failed rc=${status_rc}: ${status_out//$'\n'/ ; }"
  exit 0
fi

stopped_lines="$(printf '%s\n' "${status_out}" | grep -E '^\[stopped\]' || true)"
stale_lines="$(printf '%s\n' "${status_out}" | grep -E '^\[stale\]' || true)"

if [[ -z "${stopped_lines}" && -z "${stale_lines}" ]]; then
  if [[ "${VERBOSE}" == "1" ]]; then
    log "[ok] stack healthy profile=${PROFILE}"
  fi
  exit 0
fi

if [[ "${network_state}" != "up" ]]; then
  log "[wait] network unavailable host=${NETWORK_HOST}:${NETWORK_PORT}; defer recovery"
  exit 0
fi

if [[ -z "${stopped_lines}" && -n "${stale_lines}" ]] && ! is_truthy "${STALE_RECOVERY_ENABLED}"; then
  if [[ "${VERBOSE}" == "1" ]]; then
    log "[skip] stale services detected but stale recovery disabled: ${stale_lines//$'\n'/ ; }"
  fi
  exit 0
fi

set +e
up_out="$(run_stack up 2>&1)"
up_rc=$?
set -e

if [[ ${up_rc} -eq 0 ]]; then
  log "[recover] stack up succeeded profile=${PROFILE} services=${SERVICES:-<profile>}"
  exit 0
fi

if [[ ${up_rc} -eq 3 ]]; then
  log "[skip] parity conflict detected; local_v2 recovery deferred"
  exit 0
fi

log "[error] stack up failed rc=${up_rc}: ${up_out//$'\n'/ ; }"
exit 0
