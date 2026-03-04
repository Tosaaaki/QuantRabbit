#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${ROOT_DIR}/logs/local_v2_stack"
PID_DIR="${STATE_DIR}/pids"
LOG_DIR="${STATE_DIR}"

DEFAULT_BASE_ENV="${ROOT_DIR}/ops/env/quant-v2-runtime.env"
DEFAULT_PROFILE="core"
DEFAULT_HEALTH_WAIT_SEC=25
DEFAULT_HEALTH_POLL_SEC=1
DEFAULT_PORT_RELEASE_WAIT_SEC=8
LOCAL_PARITY_SESSION_NAME="qr-local-parity"
LOCAL_PARITY_SUPERVISOR_REL="scripts/local_vm_parity_supervisor.py"

KNOWN_SERVICES=(
  "quant-market-data-feed"
  "quant-strategy-control"
  "quant-order-manager"
  "quant-position-manager"
  "quant-scalp-ping-5s-b"
  "quant-scalp-ping-5s-b-exit"
  "quant-micro-rangebreak"
  "quant-micro-rangebreak-exit"
)

PROFILE_core=(
  "quant-market-data-feed"
  "quant-strategy-control"
  "quant-order-manager"
  "quant-position-manager"
)

PROFILE_trade_min=(
  "quant-market-data-feed"
  "quant-strategy-control"
  "quant-order-manager"
  "quant-position-manager"
  "quant-scalp-ping-5s-b"
  "quant-scalp-ping-5s-b-exit"
  "quant-micro-rangebreak"
  "quant-micro-rangebreak-exit"
)

usage() {
  cat <<'USAGE'
Usage:
  scripts/local_v2_stack.sh <up|down|restart|status|logs> [options]

Options:
  --profile <core|trade_min>   Service profile (default: core)
  --services <csv/list>        Explicit service list (e.g. "quant-order-manager,quant-position-manager")
  --base-env <file>            Base runtime env (default: ops/env/quant-v2-runtime.env)
  --env <file>                 Optional local override env (loaded after base env)
  --tail <n>                   Tail lines for logs command (default: 120)
  --follow                     Follow logs (for logs command)
  --service <name>             Single service target for logs command
  --force-conflict             Bypass local parity conflict guard (up/down/restart only)
  -h, --help                   Show help

Examples:
  scripts/local_v2_stack.sh up --profile core --env ops/env/local-v2-stack.env
  scripts/local_v2_stack.sh status --profile trade_min
  scripts/local_v2_stack.sh logs --service quant-order-manager --tail 200 --follow
  scripts/local_v2_stack.sh down --services "quant-scalp-ping-5s-b,quant-scalp-ping-5s-b-exit"
  scripts/local_v2_stack.sh up --services quant-position-manager --force-conflict
  scripts/local_v2_stack.sh up --services "quant-order-manager,quant-position-manager" --env ops/env/local-v2-sidecar-ports.env --force-conflict
USAGE
}

normalize_service_name() {
  local name="$1"
  name="${name%.service}"
  printf '%s\n' "${name}"
}

module_for_service() {
  local svc="$1"
  case "${svc}" in
    quant-market-data-feed) printf '%s\n' "workers.market_data_feed.worker" ;;
    quant-strategy-control) printf '%s\n' "workers.strategy_control.worker" ;;
    quant-order-manager) printf '%s\n' "workers.order_manager.worker" ;;
    quant-position-manager) printf '%s\n' "workers.position_manager.worker" ;;
    quant-scalp-ping-5s-b) printf '%s\n' "workers.scalp_ping_5s_b.worker" ;;
    quant-scalp-ping-5s-b-exit) printf '%s\n' "workers.scalp_ping_5s_b.exit_worker" ;;
    quant-micro-rangebreak) printf '%s\n' "workers.micro_rangebreak.worker" ;;
    quant-micro-rangebreak-exit) printf '%s\n' "workers.micro_rangebreak.exit_worker" ;;
    *) return 1 ;;
  esac
}

fail_unknown_service() {
  local svc="$1"
  if ! module_for_service "${svc}" >/dev/null 2>&1; then
    echo "[error] unknown service: ${svc}" >&2
    echo "[hint] known services:" >&2
    printf '  - %s\n' "${KNOWN_SERVICES[@]}" >&2
    exit 2
  fi
}

is_running_pid() {
  local pid="$1"
  [[ -n "${pid}" ]] || return 1
  kill -0 "${pid}" 2>/dev/null
}

pid_file_for() {
  local svc="$1"
  printf '%s/%s.pid\n' "${PID_DIR}" "${svc}"
}

log_file_for() {
  local svc="$1"
  printf '%s/%s.log\n' "${LOG_DIR}" "${svc}"
}

service_port_for() {
  local svc="$1"
  case "${svc}" in
    quant-order-manager)
      (
        load_env_for_service "${svc}"
        printf '%s\n' "${ORDER_MANAGER_SERVICE_PORT:-8300}"
      )
      ;;
    quant-position-manager)
      (
        load_env_for_service "${svc}"
        printf '%s\n' "${POSITION_MANAGER_SERVICE_PORT:-8301}"
      )
      ;;
    *)
      ;;
  esac
}

service_health_url_for() {
  local svc="$1"
  local port
  port="$(service_port_for "${svc}" || true)"
  case "${svc}" in
    quant-order-manager|quant-position-manager)
      if [[ -n "${port}" ]]; then
        printf 'http://127.0.0.1:%s/health\n' "${port}"
      fi
      ;;
    *)
      ;;
  esac
}

read_pid() {
  local svc="$1"
  local pf
  pf="$(pid_file_for "${svc}")"
  if [[ -f "${pf}" ]]; then
    tr -d '[:space:]' < "${pf}"
  fi
}

pid_command() {
  local pid="$1"
  ps -o command= -p "${pid}" 2>/dev/null || true
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

load_env_file_if_exists() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    # shellcheck disable=SC1090
    . "${path}"
  fi
}

load_env_for_service() {
  local svc="$1"

  if [[ ! -f "${BASE_ENV}" ]]; then
    echo "[error] base env not found: ${BASE_ENV}" >&2
    exit 1
  fi

  set -a
  # shellcheck disable=SC1090
  . "${BASE_ENV}"

  case "${svc}" in
    quant-order-manager)
      load_env_file_if_exists "${ROOT_DIR}/ops/env/quant-order-manager.env"
      export ORDER_MANAGER_SERVICE_PORT="${ORDER_MANAGER_SERVICE_PORT:-8300}"
      ;;
    quant-position-manager)
      load_env_file_if_exists "${ROOT_DIR}/ops/env/quant-position-manager.env"
      export POSITION_MANAGER_SERVICE_PORT="${POSITION_MANAGER_SERVICE_PORT:-8301}"
      ;;
    quant-scalp-ping-5s-b)
      load_env_file_if_exists "${ROOT_DIR}/ops/env/quant-scalp-ping-5s-b.env"
      load_env_file_if_exists "${ROOT_DIR}/ops/env/scalp_ping_5s_b.env"
      export SCALP_PING_5S_B_PERF_GUARD_MODE="${SCALP_PING_5S_B_PERF_GUARD_MODE:-reduce}"
      export SCALP_PING_5S_B_SL_STREAK_DIRECTION_FLIP_ENABLED="${SCALP_PING_5S_B_SL_STREAK_DIRECTION_FLIP_ENABLED:-1}"
      export SCALP_PING_5S_B_SIDE_METRICS_DIRECTION_FLIP_ENABLED="${SCALP_PING_5S_B_SIDE_METRICS_DIRECTION_FLIP_ENABLED:-1}"
      export SCALP_PING_5S_B_SIDE_ADVERSE_STACK_UNITS_ENABLED="${SCALP_PING_5S_B_SIDE_ADVERSE_STACK_UNITS_ENABLED:-1}"
      ;;
    quant-scalp-ping-5s-b-exit)
      load_env_file_if_exists "${ROOT_DIR}/ops/env/quant-scalp-ping-5s-b-exit.env"
      ;;
    quant-micro-rangebreak)
      load_env_file_if_exists "${ROOT_DIR}/ops/env/quant-micro-rangebreak.env"
      ;;
    quant-micro-rangebreak-exit)
      load_env_file_if_exists "${ROOT_DIR}/ops/env/quant-micro-rangebreak-exit.env"
      ;;
    quant-market-data-feed)
      export MARKET_DATA_FEED_ENABLED="${MARKET_DATA_FEED_ENABLED:-1}"
      export MARKET_DATA_FEED_INSTRUMENT="${MARKET_DATA_FEED_INSTRUMENT:-USD_JPY}"
      export MARKET_DATA_FEED_TIMEFRAMES="${MARKET_DATA_FEED_TIMEFRAMES:-M1,M5,H1,H4,D1}"
      ;;
    quant-strategy-control)
      export STRATEGY_CONTROL_POLL_SEC="${STRATEGY_CONTROL_POLL_SEC:-5}"
      export STRATEGY_CONTROL_HEARTBEAT_SEC="${STRATEGY_CONTROL_HEARTBEAT_SEC:-60}"
      ;;
  esac

  if [[ -n "${OVERRIDE_ENV}" ]]; then
    if [[ ! -f "${OVERRIDE_ENV}" ]]; then
      echo "[error] override env not found: ${OVERRIDE_ENV}" >&2
      exit 1
    fi
    # shellcheck disable=SC1090
    . "${OVERRIDE_ENV}"
  fi

  set +a
  export PYTHONUNBUFFERED=1
}

resolve_profile_services() {
  local profile="$1"
  case "${profile}" in
    core)
      printf '%s\n' "${PROFILE_core[@]}"
      ;;
    trade_min)
      printf '%s\n' "${PROFILE_trade_min[@]}"
      ;;
    *)
      echo "[error] unknown profile: ${profile}" >&2
      exit 2
      ;;
  esac
}

contains_service() {
  local target="$1"
  shift
  local item
  for item in "$@"; do
    if [[ "${item}" == "${target}" ]]; then
      return 0
    fi
  done
  return 1
}

local_parity_screen_exists() {
  local out
  if ! command -v screen >/dev/null 2>&1; then
    return 1
  fi
  out="$(screen -ls 2>/dev/null || true)"
  printf '%s\n' "${out}" | grep -Fq "${LOCAL_PARITY_SESSION_NAME}"
}

local_parity_supervisor_running() {
  local line lines
  if command -v pgrep >/dev/null 2>&1; then
    lines="$(pgrep -af "local_vm_parity_supervisor.py" 2>/dev/null || true)"
  else
    lines="$(ps ax -o command= 2>/dev/null | grep "local_vm_parity_supervisor.py" | grep -v grep || true)"
  fi
  [[ -n "${lines}" ]] || return 1

  while IFS= read -r line; do
    [[ -n "${line}" ]] || continue
    if [[ "${line}" == *"${ROOT_DIR}/${LOCAL_PARITY_SUPERVISOR_REL}"* ]]; then
      return 0
    fi
  done <<EOF_PARITY
${lines}
EOF_PARITY
  return 1
}

guard_parity_conflict() {
  local reasons=()
  local reason_csv

  if local_parity_screen_exists; then
    reasons+=("screen:${LOCAL_PARITY_SESSION_NAME}")
  fi
  if local_parity_supervisor_running; then
    reasons+=("process:${LOCAL_PARITY_SUPERVISOR_REL}")
  fi
  if [[ ${#reasons[@]} -eq 0 ]]; then
    return 0
  fi

  reason_csv="$(IFS=, ; printf '%s' "${reasons[*]}")"
  if [[ "${FORCE_CONFLICT}" == "1" ]]; then
    CONFLICT_SAFE_MODE=1
    echo "[warn] parity conflict detected (${reason_csv}); continuing due to --force-conflict"
    echo "[warn] conflict-safe mode enabled; module-wide cleanup/kill is disabled for this run"
    return 0
  fi

  echo "[error] parity conflict detected (${reason_csv}); refusing '${CMD}'." >&2
  echo "[hint] stop parity stack first: scripts/local_vm_parity_stack.sh stop" >&2
  echo "[hint] force override only when intentional: --force-conflict" >&2
  exit 3
}

RESOLVED_SERVICES=()

resolve_services() {
  RESOLVED_SERVICES=()

  if [[ -n "${SERVICES_RAW}" ]]; then
    local normalized token
    normalized="$(printf '%s' "${SERVICES_RAW}" | tr ',' ' ')"
    for token in ${normalized}; do
      token="$(normalize_service_name "${token}")"
      [[ -n "${token}" ]] || continue
      fail_unknown_service "${token}"
      if ! contains_service "${token}" "${RESOLVED_SERVICES[@]-}"; then
        RESOLVED_SERVICES+=("${token}")
      fi
    done
  else
    while IFS= read -r svc; do
      [[ -n "${svc}" ]] || continue
      fail_unknown_service "${svc}"
      if ! contains_service "${svc}" "${RESOLVED_SERVICES[@]-}"; then
        RESOLVED_SERVICES+=("${svc}")
      fi
    done <<EOF_PROFILE
$(resolve_profile_services "${PROFILE}")
EOF_PROFILE
  fi

  if [[ ${#RESOLVED_SERVICES[@]} -eq 0 ]]; then
    echo "[error] no services resolved" >&2
    exit 2
  fi
}

python_bin() {
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    printf '%s\n' "${ROOT_DIR}/.venv/bin/python"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  echo "[error] python not found (.venv/bin/python or python3)" >&2
  exit 1
}

module_patterns_for_service() {
  local svc="$1"
  case "${svc}" in
    quant-scalp-ping-5s-b)
      # B worker can appear as wrapper module or resolved generic module.
      printf '%s\n' "workers.scalp_ping_5s_b.worker"
      printf '%s\n' "workers.scalp_ping_5s.worker"
      ;;
    *)
      module_for_service "${svc}" || true
      ;;
  esac
}

pid_matches_service() {
  local svc="$1"
  local pid="$2"
  local cmd patterns pattern

  if ! is_running_pid "${pid}"; then
    return 1
  fi

  cmd="$(pid_command "${pid}")"
  [[ -n "${cmd}" ]] || return 1

  patterns="$(module_patterns_for_service "${svc}" || true)"
  while IFS= read -r pattern; do
    [[ -n "${pattern}" ]] || continue
    if [[ "${cmd}" == *"${pattern}"* ]]; then
      return 0
    fi
  done <<EOF_PATTERNS
${patterns}
EOF_PATTERNS

  case "${svc}" in
    quant-order-manager)
      [[ "${cmd}" == *"workers.order_manager.worker:app"* ]]
      ;;
    quant-position-manager)
      [[ "${cmd}" == *"workers.position_manager.worker:app"* ]]
      ;;
    *)
      return 1
      ;;
  esac
}

terminate_pid() {
  local pid="$1"
  local attempts="${2:-20}"
  local i

  [[ -n "${pid}" ]] || return 0
  if ! is_running_pid "${pid}"; then
    return 0
  fi

  kill "${pid}" 2>/dev/null || true
  for i in $(seq 1 "${attempts}"); do
    if ! is_running_pid "${pid}"; then
      return 0
    fi
    sleep 0.25
  done

  if is_running_pid "${pid}"; then
    kill -9 "${pid}" 2>/dev/null || true
  fi
}

port_listener_pids() {
  local port="$1"

  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -t -iTCP:"${port}" -sTCP:LISTEN 2>/dev/null | sort -u
    return 0
  fi

  if command -v ss >/dev/null 2>&1; then
    ss -ltnp "( sport = :${port} )" 2>/dev/null \
      | grep -Eo 'pid=[0-9]+' \
      | cut -d= -f2 \
      | sort -u || true
    return 0
  fi

  return 0
}

wait_for_port_release() {
  local port="$1"
  local wait_sec="${2:-${DEFAULT_PORT_RELEASE_WAIT_SEC}}"
  local i

  for i in $(seq 1 "${wait_sec}"); do
    if [[ -z "$(port_listener_pids "${port}")" ]]; then
      return 0
    fi
    sleep 1
  done
  [[ -z "$(port_listener_pids "${port}")" ]]
}

cleanup_service_port_binding() {
  local svc="$1"
  local keep_pid="${2:-}"
  local strict="${3:-0}"
  local port pids pid remaining

  port="$(service_port_for "${svc}" || true)"
  [[ -n "${port}" ]] || return 0

  pids="$(port_listener_pids "${port}")"
  [[ -n "${pids}" ]] || return 0

  while IFS= read -r pid; do
    [[ -n "${pid}" ]] || continue
    if [[ -n "${keep_pid}" ]] && [[ "${pid}" == "${keep_pid}" ]]; then
      continue
    fi

    if pid_matches_service "${svc}" "${pid}"; then
      echo "[clean] ${svc} terminating port holder pid=${pid} port=${port}"
      terminate_pid "${pid}" 20
      continue
    fi

    echo "[warn] ${svc} port=${port} occupied by non-target pid=${pid}: $(pid_command "${pid}")" >&2
  done <<EOF_PIDS
${pids}
EOF_PIDS

  wait_for_port_release "${port}" "${DEFAULT_PORT_RELEASE_WAIT_SEC}" || true
  remaining="$(port_listener_pids "${port}")"
  if [[ -n "${remaining}" ]]; then
    if [[ "${strict}" == "1" ]]; then
      echo "[error] ${svc} port=${port} remains occupied (start aborted for safety): ${remaining}" >&2
      return 1
    fi
    echo "[warn] ${svc} port=${port} remains occupied after stop cleanup: ${remaining}" >&2
  fi
}

assert_service_port_available() {
  local svc="$1"
  local keep_pid="${2:-}"
  local port pids pid

  port="$(service_port_for "${svc}" || true)"
  [[ -n "${port}" ]] || return 0

  pids="$(port_listener_pids "${port}")"
  [[ -n "${pids}" ]] || return 0

  while IFS= read -r pid; do
    [[ -n "${pid}" ]] || continue
    if [[ -n "${keep_pid}" ]] && [[ "${pid}" == "${keep_pid}" ]]; then
      continue
    fi
    echo "[error] ${svc} port=${port} already occupied by pid=${pid}: $(pid_command "${pid}")" >&2
    return 1
  done <<EOF_PIDS
${pids}
EOF_PIDS
  return 0
}

http_health_ok() {
  local url="$1"
  local py_exec
  if command -v curl >/dev/null 2>&1; then
    curl -fsS --max-time 1 "${url}" >/dev/null 2>&1
    return $?
  fi

  py_exec="$(python_bin)"
  "${py_exec}" - "${url}" <<'PY' >/dev/null 2>&1
import sys
from urllib.request import Request, urlopen

url = sys.argv[1]
req = Request(url, headers={"User-Agent": "local-v2-stack-healthcheck"})
with urlopen(req, timeout=1.0):
    pass
PY
}

wait_for_service_health() {
  local svc="$1"
  local pid="${2:-}"
  local url wait_sec poll_sec i

  url="$(service_health_url_for "${svc}" || true)"
  [[ -n "${url}" ]] || return 0

  wait_sec="$(as_positive_int "${LOCAL_V2_HEALTH_WAIT_SEC:-}" "${DEFAULT_HEALTH_WAIT_SEC}")"
  poll_sec="$(as_positive_int "${LOCAL_V2_HEALTH_POLL_SEC:-}" "${DEFAULT_HEALTH_POLL_SEC}")"

  for i in $(seq 1 "${wait_sec}"); do
    if [[ -n "${pid}" ]] && ! is_running_pid "${pid}"; then
      echo "[error] ${svc} exited before health became ready" >&2
      return 1
    fi
    if http_health_ok "${url}"; then
      echo "[ready] ${svc} health ok (${url})"
      return 0
    fi
    sleep "${poll_sec}"
  done

  echo "[error] ${svc} health wait timed out (${url})" >&2
  return 1
}

service_process_pids() {
  local svc="$1"
  local patterns pattern hits
  if ! command -v pgrep >/dev/null 2>&1; then
    return 0
  fi
  patterns="$(module_patterns_for_service "${svc}" || true)"
  [[ -n "${patterns}" ]] || return 0
  while IFS= read -r pattern; do
    [[ -n "${pattern}" ]] || continue
    hits="$(pgrep -f "${pattern}" 2>/dev/null || true)"
    [[ -n "${hits}" ]] || continue
    printf '%s\n' "${hits}"
  done <<EOF_PATTERNS
${patterns}
EOF_PATTERNS
}

cleanup_service_processes() {
  local svc="$1"
  local keep_pid="${2:-}"
  local pids pid
  pids="$(service_process_pids "${svc}" | sort -u)"
  [[ -n "${pids}" ]] || return 0

  while IFS= read -r pid; do
    [[ -n "${pid}" ]] || continue
    if [[ -n "${keep_pid}" ]] && [[ "${pid}" == "${keep_pid}" ]]; then
      continue
    fi
    terminate_pid "${pid}" 20
  done <<EOF_PIDS
${pids}
EOF_PIDS
}

start_service() {
  local svc="$1"
  local module pf lf pid py extra_pids
  module="$(module_for_service "${svc}")"
  pf="$(pid_file_for "${svc}")"
  lf="$(log_file_for "${svc}")"

  pid="$(read_pid "${svc}" || true)"
  if [[ -n "${pid}" ]] && is_running_pid "${pid}" && ! pid_matches_service "${svc}" "${pid}"; then
    echo "[warn] ${svc} pid file points to non-target pid=${pid}; ignoring stale pid file"
    pid=""
  fi
  extra_pids="$(service_process_pids "${svc}" | sort -u | grep -vx "${pid}" || true)"
  if [[ -n "${pid}" ]] && is_running_pid "${pid}" && [[ -z "${extra_pids}" ]]; then
    if wait_for_service_health "${svc}" "${pid}"; then
      echo "[skip] ${svc} already running (pid=${pid})"
      return 0
    fi
    echo "[warn] ${svc} existing pid=${pid} is not healthy; recycling"
    stop_service "${svc}"
    pid=""
  fi

  if [[ -n "${extra_pids}" ]]; then
    if [[ "${CONFLICT_SAFE_MODE}" == "1" ]]; then
      echo "[warn] ${svc} conflict-safe mode: skip module-wide stale pid cleanup"
    else
      echo "[clean] ${svc} removing stale worker pids"
      cleanup_service_processes "${svc}" "${pid}"
    fi
  fi

  if [[ "${CONFLICT_SAFE_MODE}" == "1" ]]; then
    assert_service_port_available "${svc}" "${pid}" || return 1
  else
    cleanup_service_port_binding "${svc}" "${pid}" "1"
  fi

  rm -f "${pf}"
  py="$(python_bin)"

  (
    cd "${ROOT_DIR}"
    load_env_for_service "${svc}"
    exec nohup "${py}" -m "${module}"
  ) >> "${lf}" 2>&1 < /dev/null &

  pid=$!
  echo "${pid}" > "${pf}"
  echo "[up] ${svc} pid=${pid} log=${lf}"

  sleep 0.2
  if ! is_running_pid "${pid}"; then
    echo "[error] ${svc} exited immediately after launch (pid=${pid})" >&2
    tail -n 40 "${lf}" >&2 || true
    rm -f "${pf}"
    return 1
  fi

  if ! wait_for_service_health "${svc}" "${pid}"; then
    echo "[error] ${svc} failed health check; stopping service" >&2
    stop_service "${svc}"
    return 1
  fi
}

stop_service() {
  local svc="$1"
  local pf pid
  pf="$(pid_file_for "${svc}")"
  pid="$(read_pid "${svc}" || true)"

  if [[ -z "${pid}" ]]; then
    rm -f "${pf}"
    echo "[skip] ${svc} no pid file (cleaning by module pattern)"
  else
    if is_running_pid "${pid}"; then
      if pid_matches_service "${svc}" "${pid}"; then
        terminate_pid "${pid}" 20
      else
        echo "[warn] ${svc} pid=${pid} does not match target module; not terminating by pid-file" >&2
      fi
    fi
    rm -f "${pf}"
  fi

  if [[ "${CONFLICT_SAFE_MODE}" == "1" ]]; then
    echo "[warn] ${svc} conflict-safe mode: skip module-wide cleanup"
  else
    cleanup_service_processes "${svc}"
    cleanup_service_port_binding "${svc}" "" "0" || true
  fi
  echo "[down] ${svc} stopped"
}

status_service() {
  local svc="$1"
  local pid
  pid="$(read_pid "${svc}" || true)"
  if [[ -n "${pid}" ]] && is_running_pid "${pid}" && pid_matches_service "${svc}" "${pid}"; then
    echo "[running] ${svc} pid=${pid} log=$(log_file_for "${svc}")"
  elif [[ -n "${pid}" ]] && is_running_pid "${pid}"; then
    echo "[stale] ${svc} pid-file pid=${pid} is running but not target module"
  else
    echo "[stopped] ${svc}"
  fi
}

logs_service() {
  local svc="$1"
  local lf
  lf="$(log_file_for "${svc}")"
  if [[ ! -f "${lf}" ]]; then
    echo "[error] log not found: ${lf}" >&2
    exit 1
  fi
  if [[ "${FOLLOW}" == "1" ]]; then
    tail -n "${TAIL_LINES}" -F "${lf}"
  else
    tail -n "${TAIL_LINES}" "${lf}"
  fi
}

mkdir -p "${PID_DIR}" "${LOG_DIR}"

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

if [[ "${1}" == "-h" || "${1}" == "--help" ]]; then
  usage
  exit 0
fi

CMD="$1"
shift

PROFILE="${DEFAULT_PROFILE}"
SERVICES_RAW=""
BASE_ENV="${DEFAULT_BASE_ENV}"
OVERRIDE_ENV=""
TAIL_LINES=120
FOLLOW=0
LOG_SERVICE=""
FORCE_CONFLICT=0
CONFLICT_SAFE_MODE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --services)
      SERVICES_RAW="$2"
      shift 2
      ;;
    --base-env)
      BASE_ENV="$2"
      shift 2
      ;;
    --env)
      OVERRIDE_ENV="$2"
      shift 2
      ;;
    --tail)
      TAIL_LINES="$2"
      shift 2
      ;;
    --follow)
      FOLLOW=1
      shift
      ;;
    --service)
      LOG_SERVICE="$(normalize_service_name "$2")"
      shift 2
      ;;
    --force-conflict)
      FORCE_CONFLICT=1
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

resolve_services

case "${CMD}" in
  up)
    guard_parity_conflict
    for svc in "${RESOLVED_SERVICES[@]}"; do
      start_service "${svc}"
    done
    ;;
  down)
    guard_parity_conflict
    for ((i=${#RESOLVED_SERVICES[@]}-1; i>=0; i--)); do
      stop_service "${RESOLVED_SERVICES[$i]}"
    done
    ;;
  restart)
    guard_parity_conflict
    for ((i=${#RESOLVED_SERVICES[@]}-1; i>=0; i--)); do
      stop_service "${RESOLVED_SERVICES[$i]}"
    done
    for svc in "${RESOLVED_SERVICES[@]}"; do
      start_service "${svc}"
    done
    ;;
  status)
    for svc in "${RESOLVED_SERVICES[@]}"; do
      status_service "${svc}"
    done
    ;;
  logs)
    if [[ -n "${LOG_SERVICE}" ]]; then
      fail_unknown_service "${LOG_SERVICE}"
      logs_service "${LOG_SERVICE}"
    else
      if [[ "${FOLLOW}" == "1" ]]; then
        LOG_FILES=()
        for svc in "${RESOLVED_SERVICES[@]}"; do
          LOG_FILES+=("$(log_file_for "${svc}")")
        done
        tail -n "${TAIL_LINES}" -F "${LOG_FILES[@]}"
      else
        for svc in "${RESOLVED_SERVICES[@]}"; do
          lf="$(log_file_for "${svc}")"
          if [[ -f "${lf}" ]]; then
            echo "===== ${svc} (${lf}) ====="
            tail -n "${TAIL_LINES}" "${lf}"
          else
            echo "===== ${svc} ====="
            echo "[missing] ${lf}"
          fi
        done
      fi
    fi
    ;;
  *)
    echo "[error] unknown command: ${CMD}" >&2
    usage
    exit 2
    ;;
esac
