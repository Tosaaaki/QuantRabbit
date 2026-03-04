#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ONCE_SCRIPT="${ROOT_DIR}/scripts/local_v2_autorecover_once.sh"
LOG_DIR="${ROOT_DIR}/logs/local_v2_stack"
PID_FILE="${LOG_DIR}/watchdog.pid"
DAEMON_LOG="${LOG_DIR}/watchdog.log"

DEFAULT_PROFILE="trade_min"
DEFAULT_INTERVAL_SEC=10
DEFAULT_RESUME_GAP_SEC=90
DEFAULT_ENV_FILE="${ROOT_DIR}/ops/env/local-v2-stack.env"

PROFILE="${DEFAULT_PROFILE}"
ENV_FILE="${DEFAULT_ENV_FILE}"
SERVICES=""
INTERVAL_SEC="${DEFAULT_INTERVAL_SEC}"
RESUME_GAP_SEC="${DEFAULT_RESUME_GAP_SEC}"
AUTORECOVER_VERBOSE="0"

usage() {
  cat <<'USAGE'
Usage:
  scripts/local_v2_watchdog.sh <start|run|once|stop|status> [options]

Actions:
  start                Start watchdog daemon (idempotent)
  run                  Run watchdog loop in foreground
  once                 Run a single health/recovery cycle
  stop                 Stop watchdog daemon
  status               Show watchdog daemon status

Options:
  --profile <name>       local_v2_stack profile (default: trade_min)
  --env <file>           override env file (default: ops/env/local-v2-stack.env)
  --services <csv>       optional explicit services list
  --interval-sec <sec>   polling interval seconds (default: 10)
  --resume-gap-sec <sec> sleep/wake gap threshold for log marker (default: 90)
  --verbose              verbose autorecover logs
  -h, --help             show help
USAGE
}

is_running_pid() {
  local pid="$1"
  [[ -n "${pid}" ]] || return 1
  kill -0 "${pid}" 2>/dev/null
}

read_pid() {
  if [[ -f "${PID_FILE}" ]]; then
    tr -d '[:space:]' < "${PID_FILE}"
  fi
}

cleanup_stale_pid_file() {
  local pid=""
  pid="$(read_pid || true)"
  if [[ -n "${pid}" ]] && ! is_running_pid "${pid}"; then
    rm -f "${PID_FILE}"
  fi
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

run_once() {
  local args=("${ONCE_SCRIPT}" "--profile" "${PROFILE}" "--env" "${ENV_FILE}")
  if [[ -n "${SERVICES}" ]]; then
    args+=("--services" "${SERVICES}")
  fi
  if [[ "${AUTORECOVER_VERBOSE}" == "1" ]]; then
    QR_LOCAL_V2_AUTORECOVER_VERBOSE=1 "${args[@]}"
    return $?
  fi
  "${args[@]}"
}

run_loop() {
  local interval_sec resume_gap_sec
  local now_epoch=0
  local last_epoch=0
  local gap_sec=0

  interval_sec="$(as_positive_int "${INTERVAL_SEC}" "${DEFAULT_INTERVAL_SEC}")"
  resume_gap_sec="$(as_positive_int "${RESUME_GAP_SEC}" "${DEFAULT_RESUME_GAP_SEC}")"

  mkdir -p "${LOG_DIR}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] watchdog started profile=${PROFILE} interval=${interval_sec}s"

  while true; do
    now_epoch="$(date +%s)"
    if (( last_epoch > 0 )); then
      gap_sec=$((now_epoch - last_epoch))
      if (( gap_sec >= resume_gap_sec )); then
        echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] [resume] polling gap ${gap_sec}s (threshold=${resume_gap_sec}s)"
      fi
    fi
    last_epoch="${now_epoch}"

    set +e
    run_once
    rc=$?
    set -e
    if [[ ${rc} -ne 0 ]]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] [warn] autorecover_once failed rc=${rc}"
    fi

    sleep "${interval_sec}"
  done
}

start_daemon() {
  local pid=""
  local args=()

  mkdir -p "${LOG_DIR}"
  cleanup_stale_pid_file
  pid="$(read_pid || true)"
  if [[ -n "${pid}" ]] && is_running_pid "${pid}"; then
    echo "[skip] watchdog already running pid=${pid}"
    return 0
  fi

  args=("$0" "run" "--profile" "${PROFILE}" "--env" "${ENV_FILE}" "--interval-sec" "${INTERVAL_SEC}" "--resume-gap-sec" "${RESUME_GAP_SEC}")
  if [[ -n "${SERVICES}" ]]; then
    args+=("--services" "${SERVICES}")
  fi
  if [[ "${AUTORECOVER_VERBOSE}" == "1" ]]; then
    args+=("--verbose")
  fi

  nohup "${args[@]}" >> "${DAEMON_LOG}" 2>&1 < /dev/null &

  pid=$!
  echo "${pid}" > "${PID_FILE}"
  sleep 0.2

  if ! is_running_pid "${pid}"; then
    echo "[error] watchdog exited immediately pid=${pid}" >&2
    rm -f "${PID_FILE}"
    tail -n 60 "${DAEMON_LOG}" 2>/dev/null || true
    return 1
  fi

  echo "[up] watchdog daemon pid=${pid} log=${DAEMON_LOG}"
}

stop_daemon() {
  local pid=""
  pid="$(read_pid || true)"

  if [[ -z "${pid}" ]]; then
    echo "[stopped] watchdog"
    rm -f "${PID_FILE}"
    return 0
  fi

  if ! is_running_pid "${pid}"; then
    echo "[stale] watchdog pid file found but process not running pid=${pid}"
    rm -f "${PID_FILE}"
    return 0
  fi

  kill "${pid}" 2>/dev/null || true
  for _ in $(seq 1 20); do
    if ! is_running_pid "${pid}"; then
      rm -f "${PID_FILE}"
      echo "[down] watchdog stopped pid=${pid}"
      return 0
    fi
    sleep 0.25
  done

  if is_running_pid "${pid}"; then
    kill -9 "${pid}" 2>/dev/null || true
  fi
  rm -f "${PID_FILE}"
  echo "[down] watchdog stopped pid=${pid}"
}

status_daemon() {
  local pid=""
  cleanup_stale_pid_file
  pid="$(read_pid || true)"
  if [[ -n "${pid}" ]] && is_running_pid "${pid}"; then
    echo "[running] watchdog pid=${pid} log=${DAEMON_LOG}"
  else
    echo "[stopped] watchdog"
  fi
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

ACTION="$1"
shift

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
    --interval-sec)
      INTERVAL_SEC="$2"
      shift 2
      ;;
    --resume-gap-sec)
      RESUME_GAP_SEC="$2"
      shift 2
      ;;
    --verbose)
      AUTORECOVER_VERBOSE="1"
      shift
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

INTERVAL_SEC="$(as_positive_int "${INTERVAL_SEC}" "${DEFAULT_INTERVAL_SEC}")"
RESUME_GAP_SEC="$(as_positive_int "${RESUME_GAP_SEC}" "${DEFAULT_RESUME_GAP_SEC}")"

case "${ACTION}" in
  start|run|once)
    if [[ ! -x "${ONCE_SCRIPT}" ]]; then
      echo "[error] autorecover script not executable: ${ONCE_SCRIPT}" >&2
      exit 1
    fi
    if [[ ! -f "${ENV_FILE}" ]]; then
      echo "[error] env file not found: ${ENV_FILE}" >&2
      exit 1
    fi
    ;;
esac

case "${ACTION}" in
  start)
    start_daemon
    ;;
  run)
    run_loop
    ;;
  once)
    run_once
    ;;
  stop)
    stop_daemon
    ;;
  status)
    status_daemon
    ;;
  *)
    echo "[error] unknown action: ${ACTION}" >&2
    usage
    exit 2
    ;;
esac
