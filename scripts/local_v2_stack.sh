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
DEFAULT_START_RETRY_COUNT=3
DEFAULT_START_RETRY_DELAY_SEC=1
DEFAULT_POST_UP_RECONCILE_ROUNDS=2
DEFAULT_POST_UP_SETTLE_SEC=2
DEFAULT_WATCHDOG_INTERVAL_SEC=10
DEFAULT_WATCHDOG_RESUME_GAP_SEC=90
DEFAULT_STACK_OP_LOCK_WAIT_SEC=120
LOCAL_PARITY_SESSION_NAME="qr-local-parity"
LOCAL_PARITY_SUPERVISOR_REL="scripts/local_vm_parity_supervisor.py"
STACK_OP_LOCK_FILE="${STATE_DIR}/stack_ops.lock"
STACK_OP_LOCK_FALLBACK_DIR="${STATE_DIR}/stack_ops.lockdir"

KNOWN_SERVICES=(
  "quant-market-data-feed"
  "quant-strategy-control"
  "quant-order-manager"
  "quant-position-manager"
  "quant-strategy-feedback"
  "quant-scalp-ping-5s-b"
  "quant-scalp-ping-5s-b-exit"
  "quant-micro-rangebreak"
  "quant-micro-rangebreak-exit"
  "quant-m1scalper"
  "quant-m1scalper-exit"
  "quant-micro-compressionrevert"
  "quant-micro-compressionrevert-exit"
  "quant-micro-levelreactor"
  "quant-micro-levelreactor-exit"
  "quant-micro-momentumburst"
  "quant-micro-momentumburst-exit"
  "quant-micro-momentumpulse"
  "quant-micro-momentumpulse-exit"
  "quant-micro-momentumstack"
  "quant-micro-momentumstack-exit"
  "quant-micro-pullbackema"
  "quant-micro-pullbackema-exit"
  "quant-micro-trendmomentum"
  "quant-micro-trendmomentum-exit"
  "quant-micro-trendretest"
  "quant-micro-trendretest-exit"
  "quant-micro-vwapbound"
  "quant-micro-vwapbound-exit"
  "quant-micro-vwaprevert"
  "quant-micro-vwaprevert-exit"
  "quant-scalp-extrema-reversal"
  "quant-scalp-extrema-reversal-exit"
  "quant-scalp-failed-break-reverse"
  "quant-scalp-failed-break-reverse-exit"
  "quant-scalp-false-break-fade"
  "quant-scalp-false-break-fade-exit"
  "quant-scalp-level-reject"
  "quant-scalp-level-reject-exit"
  "quant-scalp-macd-rsi-div"
  "quant-scalp-macd-rsi-div-exit"
  "quant-scalp-macd-rsi-div-b"
  "quant-scalp-macd-rsi-div-b-exit"
  "quant-scalp-ping-5s-c"
  "quant-scalp-ping-5s-c-exit"
  "quant-scalp-ping-5s-d"
  "quant-scalp-ping-5s-d-exit"
  "quant-scalp-ping-5s-flow"
  "quant-scalp-ping-5s-flow-exit"
  "quant-scalp-pullback-continuation"
  "quant-scalp-pullback-continuation-exit"
  "quant-scalp-rangefader"
  "quant-scalp-rangefader-exit"
  "quant-scalp-squeeze-pulse-break"
  "quant-scalp-squeeze-pulse-break-exit"
  "quant-scalp-tick-imbalance"
  "quant-scalp-tick-imbalance-exit"
  "quant-scalp-trend-breakout"
  "quant-scalp-trend-breakout-exit"
  "quant-scalp-wick-reversal-blend"
  "quant-scalp-wick-reversal-blend-exit"
  "quant-scalp-wick-reversal-pro"
  "quant-scalp-wick-reversal-pro-exit"
  "quant-session-open"
  "quant-session-open-exit"
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
  "quant-strategy-feedback"
  "quant-scalp-ping-5s-b"
  "quant-scalp-ping-5s-b-exit"
  "quant-scalp-trend-breakout"
  "quant-scalp-trend-breakout-exit"
  "quant-micro-momentumburst"
  "quant-micro-momentumburst-exit"
  "quant-micro-levelreactor"
  "quant-micro-levelreactor-exit"
  "quant-micro-trendretest"
  "quant-micro-trendretest-exit"
  "quant-m1scalper"
  "quant-m1scalper-exit"
)

PROFILE_trade_all=(
  "quant-market-data-feed"
  "quant-strategy-control"
  "quant-order-manager"
  "quant-position-manager"
  "quant-strategy-feedback"
  "quant-scalp-ping-5s-b"
  "quant-scalp-ping-5s-b-exit"
  "quant-micro-rangebreak"
  "quant-micro-rangebreak-exit"
  "quant-m1scalper"
  "quant-m1scalper-exit"
  "quant-micro-compressionrevert"
  "quant-micro-compressionrevert-exit"
  "quant-micro-levelreactor"
  "quant-micro-levelreactor-exit"
  "quant-micro-momentumburst"
  "quant-micro-momentumburst-exit"
  "quant-micro-momentumpulse"
  "quant-micro-momentumpulse-exit"
  "quant-micro-momentumstack"
  "quant-micro-momentumstack-exit"
  "quant-micro-pullbackema"
  "quant-micro-pullbackema-exit"
  "quant-micro-trendmomentum"
  "quant-micro-trendmomentum-exit"
  "quant-micro-trendretest"
  "quant-micro-trendretest-exit"
  "quant-micro-vwapbound"
  "quant-micro-vwapbound-exit"
  "quant-micro-vwaprevert"
  "quant-micro-vwaprevert-exit"
  "quant-scalp-extrema-reversal"
  "quant-scalp-extrema-reversal-exit"
  "quant-scalp-failed-break-reverse"
  "quant-scalp-failed-break-reverse-exit"
  "quant-scalp-false-break-fade"
  "quant-scalp-false-break-fade-exit"
  "quant-scalp-level-reject"
  "quant-scalp-level-reject-exit"
  "quant-scalp-macd-rsi-div"
  "quant-scalp-macd-rsi-div-exit"
  "quant-scalp-macd-rsi-div-b"
  "quant-scalp-macd-rsi-div-b-exit"
  "quant-scalp-ping-5s-c"
  "quant-scalp-ping-5s-c-exit"
  "quant-scalp-ping-5s-d"
  "quant-scalp-ping-5s-d-exit"
  "quant-scalp-ping-5s-flow"
  "quant-scalp-ping-5s-flow-exit"
  "quant-scalp-pullback-continuation"
  "quant-scalp-pullback-continuation-exit"
  "quant-scalp-rangefader"
  "quant-scalp-rangefader-exit"
  "quant-scalp-squeeze-pulse-break"
  "quant-scalp-squeeze-pulse-break-exit"
  "quant-scalp-tick-imbalance"
  "quant-scalp-tick-imbalance-exit"
  "quant-scalp-trend-breakout"
  "quant-scalp-trend-breakout-exit"
  "quant-scalp-wick-reversal-blend"
  "quant-scalp-wick-reversal-blend-exit"
  "quant-scalp-wick-reversal-pro"
  "quant-scalp-wick-reversal-pro-exit"
  "quant-session-open"
  "quant-session-open-exit"
)

usage() {
  cat <<'USAGE'
Usage:
  scripts/local_v2_stack.sh <up|down|restart|status|logs|watchdog|watchdog-stop|watchdog-status> [options]

Options:
  --profile <core|trade_min|trade_all>  Service profile (default: core)
  --services <csv/list>        Explicit service list (e.g. "quant-order-manager,quant-position-manager")
  --base-env <file>            Base runtime env (default: ops/env/quant-v2-runtime.env)
  --env <file[,file2,...]>     Optional local override env list (loaded after base/service env)
  --tail <n>                   Tail lines for logs command (default: 120)
  --follow                     Follow logs (for logs command)
  --service <name>             Single service target for logs command
  --daemon                     Run watchdog command as daemon (watchdog command only)
  --once                       Run single watchdog recovery cycle (watchdog command only)
  --interval-sec <sec>         Watchdog polling interval (default: 10)
  --resume-gap-sec <sec>       Sleep/wake gap threshold for watchdog logs (default: 90)
  --verbose                    Verbose watchdog logs (watchdog command only)
  --force-conflict             Bypass local parity conflict guard (up/down/restart only)
  -h, --help                   Show help

Examples:
  scripts/local_v2_stack.sh up --profile core --env ops/env/local-v2-stack.env
  scripts/local_v2_stack.sh up --profile core --env ops/env/local-v2-stack.env,ops/env/profiles/brain-ollama.env
  scripts/local_v2_stack.sh status --profile trade_min
  scripts/local_v2_stack.sh up --profile trade_all --env ops/env/local-v2-stack.env
  scripts/local_v2_stack.sh logs --service quant-order-manager --tail 200 --follow
  scripts/local_v2_stack.sh down --services "quant-scalp-ping-5s-b,quant-scalp-ping-5s-b-exit"
  scripts/local_v2_stack.sh up --services quant-position-manager --force-conflict
  scripts/local_v2_stack.sh up --services "quant-order-manager,quant-position-manager" --env ops/env/local-v2-sidecar-ports.env --force-conflict
  scripts/local_v2_stack.sh watchdog --daemon --profile trade_min --env ops/env/local-v2-stack.env --interval-sec 10
  scripts/local_v2_stack.sh watchdog-status
  scripts/local_v2_stack.sh watchdog-stop
USAGE
}

stack_op_lock_wait_sec() {
  as_positive_int "${LOCAL_V2_STACK_OP_LOCK_WAIT_SEC:-}" "${DEFAULT_STACK_OP_LOCK_WAIT_SEC}"
}

release_stack_operation_lock() {
  if [[ -n "${STACK_OP_LOCK_FALLBACK_HELD:-}" ]]; then
    rm -rf "${STACK_OP_LOCK_FALLBACK_DIR}" >/dev/null 2>&1 || true
    STACK_OP_LOCK_FALLBACK_HELD=""
  fi
}

acquire_stack_operation_lock() {
  local wait_sec start_ts
  wait_sec="$(stack_op_lock_wait_sec)"

  if command -v flock >/dev/null 2>&1; then
    # Serialize whole-stack up/down/restart so manual runs and autorecover cannot overlap.
    exec {STACK_OP_LOCK_FD}>"${STACK_OP_LOCK_FILE}"
    if ! flock -w "${wait_sec}" "${STACK_OP_LOCK_FD}"; then
      echo "[error] could not acquire stack operation lock within ${wait_sec}s: ${STACK_OP_LOCK_FILE}" >&2
      exit 1
    fi
    return 0
  fi

  start_ts="$(date +%s)"
  while ! mkdir "${STACK_OP_LOCK_FALLBACK_DIR}" 2>/dev/null; do
    if (( $(date +%s) - start_ts >= wait_sec )); then
      echo "[error] could not acquire stack operation lock within ${wait_sec}s: ${STACK_OP_LOCK_FALLBACK_DIR}" >&2
      exit 1
    fi
    sleep 1
  done
  STACK_OP_LOCK_FALLBACK_HELD=1
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
    quant-strategy-feedback) printf '%s\n' "analysis.strategy_feedback_worker" ;;
    quant-m1scalper) printf '%s\n' "workers.scalp_m1scalper.worker" ;;
    quant-m1scalper-exit) printf '%s\n' "workers.scalp_m1scalper.exit_worker" ;;
    quant-micro-compressionrevert) printf '%s\n' "workers.micro_compressionrevert.worker" ;;
    quant-micro-compressionrevert-exit) printf '%s\n' "workers.micro_compressionrevert.exit_worker" ;;
    quant-micro-levelreactor) printf '%s\n' "workers.micro_levelreactor.worker" ;;
    quant-micro-levelreactor-exit) printf '%s\n' "workers.micro_levelreactor.exit_worker" ;;
    quant-micro-momentumburst) printf '%s\n' "workers.micro_momentumburst.worker" ;;
    quant-micro-momentumburst-exit) printf '%s\n' "workers.micro_momentumburst.exit_worker" ;;
    quant-micro-momentumpulse) printf '%s\n' "workers.micro_momentumpulse.worker" ;;
    quant-micro-momentumpulse-exit) printf '%s\n' "workers.micro_momentumpulse.exit_worker" ;;
    quant-micro-momentumstack) printf '%s\n' "workers.micro_momentumstack.worker" ;;
    quant-micro-momentumstack-exit) printf '%s\n' "workers.micro_momentumstack.exit_worker" ;;
    quant-micro-pullbackema) printf '%s\n' "workers.micro_pullbackema.worker" ;;
    quant-micro-pullbackema-exit) printf '%s\n' "workers.micro_pullbackema.exit_worker" ;;
    quant-scalp-ping-5s-b) printf '%s\n' "workers.scalp_ping_5s_b.worker" ;;
    quant-scalp-ping-5s-b-exit) printf '%s\n' "workers.scalp_ping_5s_b.exit_worker" ;;
    quant-scalp-ping-5s-c) printf '%s\n' "workers.scalp_ping_5s_c.worker" ;;
    quant-scalp-ping-5s-c-exit) printf '%s\n' "workers.scalp_ping_5s_c.exit_worker" ;;
    quant-scalp-ping-5s-d) printf '%s\n' "workers.scalp_ping_5s_d.worker" ;;
    quant-scalp-ping-5s-d-exit) printf '%s\n' "workers.scalp_ping_5s_d.exit_worker" ;;
    quant-scalp-ping-5s-flow) printf '%s\n' "workers.scalp_ping_5s_flow.worker" ;;
    quant-scalp-ping-5s-flow-exit) printf '%s\n' "workers.scalp_ping_5s_flow.exit_worker" ;;
    quant-scalp-extrema-reversal) printf '%s\n' "workers.scalp_extrema_reversal.worker" ;;
    quant-scalp-extrema-reversal-exit) printf '%s\n' "workers.scalp_extrema_reversal.exit_worker" ;;
    quant-scalp-failed-break-reverse) printf '%s\n' "workers.scalp_failed_break_reverse.worker" ;;
    quant-scalp-failed-break-reverse-exit) printf '%s\n' "workers.scalp_failed_break_reverse.exit_worker" ;;
    quant-scalp-false-break-fade) printf '%s\n' "workers.scalp_false_break_fade.worker" ;;
    quant-scalp-false-break-fade-exit) printf '%s\n' "workers.scalp_false_break_fade.exit_worker" ;;
    quant-scalp-level-reject) printf '%s\n' "workers.scalp_level_reject.worker" ;;
    quant-scalp-level-reject-exit) printf '%s\n' "workers.scalp_level_reject.exit_worker" ;;
    quant-scalp-macd-rsi-div) printf '%s\n' "workers.scalp_macd_rsi_div.worker" ;;
    quant-scalp-macd-rsi-div-exit) printf '%s\n' "workers.scalp_macd_rsi_div.exit_worker" ;;
    quant-scalp-macd-rsi-div-b) printf '%s\n' "workers.scalp_macd_rsi_div_b.worker" ;;
    quant-scalp-macd-rsi-div-b-exit) printf '%s\n' "workers.scalp_macd_rsi_div_b.exit_worker" ;;
    quant-scalp-pullback-continuation) printf '%s\n' "workers.scalp_pullback_continuation.worker" ;;
    quant-scalp-pullback-continuation-exit) printf '%s\n' "workers.scalp_pullback_continuation.exit_worker" ;;
    quant-scalp-rangefader) printf '%s\n' "workers.scalp_rangefader.worker" ;;
    quant-scalp-rangefader-exit) printf '%s\n' "workers.scalp_rangefader.exit_worker" ;;
    quant-scalp-squeeze-pulse-break) printf '%s\n' "workers.scalp_squeeze_pulse_break.worker" ;;
    quant-scalp-squeeze-pulse-break-exit) printf '%s\n' "workers.scalp_squeeze_pulse_break.exit_worker" ;;
    quant-scalp-tick-imbalance) printf '%s\n' "workers.scalp_tick_imbalance.worker" ;;
    quant-scalp-tick-imbalance-exit) printf '%s\n' "workers.scalp_tick_imbalance.exit_worker" ;;
    quant-scalp-trend-breakout) printf '%s\n' "workers.scalp_trend_breakout.worker" ;;
    quant-scalp-trend-breakout-exit) printf '%s\n' "workers.scalp_trend_breakout.exit_worker" ;;
    quant-scalp-wick-reversal-blend) printf '%s\n' "workers.scalp_wick_reversal_blend.worker" ;;
    quant-scalp-wick-reversal-blend-exit) printf '%s\n' "workers.scalp_wick_reversal_blend.exit_worker" ;;
    quant-scalp-wick-reversal-pro) printf '%s\n' "workers.scalp_wick_reversal_pro.worker" ;;
    quant-scalp-wick-reversal-pro-exit) printf '%s\n' "workers.scalp_wick_reversal_pro.exit_worker" ;;
    quant-micro-rangebreak) printf '%s\n' "workers.micro_rangebreak.worker" ;;
    quant-micro-rangebreak-exit) printf '%s\n' "workers.micro_rangebreak.exit_worker" ;;
    quant-micro-trendmomentum) printf '%s\n' "workers.micro_trendmomentum.worker" ;;
    quant-micro-trendmomentum-exit) printf '%s\n' "workers.micro_trendmomentum.exit_worker" ;;
    quant-micro-trendretest) printf '%s\n' "workers.micro_trendretest.worker" ;;
    quant-micro-trendretest-exit) printf '%s\n' "workers.micro_trendretest.exit_worker" ;;
    quant-micro-vwapbound) printf '%s\n' "workers.micro_vwapbound.worker" ;;
    quant-micro-vwapbound-exit) printf '%s\n' "workers.micro_vwapbound.exit_worker" ;;
    quant-micro-vwaprevert) printf '%s\n' "workers.micro_vwaprevert.worker" ;;
    quant-micro-vwaprevert-exit) printf '%s\n' "workers.micro_vwaprevert.exit_worker" ;;
    quant-session-open) printf '%s\n' "workers.session_open.worker" ;;
    quant-session-open-exit) printf '%s\n' "workers.session_open.exit_worker" ;;
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
  local label="${2:-service}"
  local resolved
  resolved="$(resolve_env_path "${path}")"
  if [[ ! -e "${resolved}" ]]; then
    return 0
  fi
  source_env_file_once "${resolved}" "${label}"
}

resolve_env_path() {
  local path="$1"
  local abs dir base
  if [[ "${path}" = /* ]]; then
    abs="${path}"
  else
    abs="${ROOT_DIR}/${path}"
  fi
  dir="$(cd -P -- "$(dirname -- "${abs}")" 2>/dev/null && pwd || true)"
  if [[ -n "${dir}" ]]; then
    base="$(basename -- "${abs}")"
    printf '%s/%s\n' "${dir}" "${base}"
  else
    printf '%s\n' "${abs}"
  fi
}

contains_value() {
  local target="$1"
  shift || true
  local item
  for item in "$@"; do
    if [[ "${item}" == "${target}" ]]; then
      return 0
    fi
  done
  return 1
}

is_world_writable_path() {
  local path="$1"
  local perms
  perms="$(LC_ALL=C ls -ld "${path}" 2>/dev/null | awk '{print $1}' || true)"
  case "${perms}" in
    ????????w*) return 0 ;;
    *) return 1 ;;
  esac
}

guard_env_file() {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    echo "[error] ${label} env not found: ${path}" >&2
    exit 1
  fi
  if [[ ! -f "${path}" ]]; then
    echo "[error] ${label} env is not a regular file: ${path}" >&2
    exit 1
  fi
  if [[ ! -r "${path}" ]]; then
    echo "[error] ${label} env is not readable: ${path}" >&2
    exit 1
  fi
  if is_world_writable_path "${path}"; then
    echo "[warn] ${label} env is world-writable: ${path}" >&2
  fi
}

ENV_CHAIN_ENTRIES=()
ENV_CHAIN_SEEN_PATHS=()

reset_env_chain_trace() {
  ENV_CHAIN_ENTRIES=()
  ENV_CHAIN_SEEN_PATHS=()
}

record_env_chain_entry() {
  local label="$1"
  local path="$2"
  ENV_CHAIN_ENTRIES+=("${label}=${path}")
}

log_env_chain_order() {
  local svc="$1"
  local entry chain
  chain=""
  for entry in "${ENV_CHAIN_ENTRIES[@]}"; do
    if [[ -n "${chain}" ]]; then
      chain="${chain} -> "
    fi
    chain="${chain}${entry}"
  done
  if [[ -n "${chain}" ]]; then
    echo "[env-chain] ${svc} ${chain}"
  fi
}

source_env_file_once() {
  local path="$1"
  local label="$2"
  local resolved
  resolved="$(resolve_env_path "${path}")"
  guard_env_file "${resolved}" "${label}"
  if contains_value "${resolved}" "${ENV_CHAIN_SEEN_PATHS[@]-}"; then
    return 0
  fi
  # shellcheck disable=SC1090
  . "${resolved}"
  ENV_CHAIN_SEEN_PATHS+=("${resolved}")
  record_env_chain_entry "${label}" "${resolved}"
}

load_env_chain() {
  local raw="$1"
  local label="$2"
  local normalized token resolved
  local dedup_paths=()
  normalized="$(printf '%s' "${raw}" | tr ',' ' ')"
  for token in ${normalized}; do
    [[ -n "${token}" ]] || continue
    resolved="$(resolve_env_path "${token}")"
    if ! contains_value "${resolved}" "${dedup_paths[@]-}"; then
      dedup_paths+=("${resolved}")
    fi
  done
  for resolved in "${dedup_paths[@]}"; do
    source_env_file_once "${resolved}" "${label}"
  done
}

log_effective_env_snapshot() {
  local svc="$1"
  local key
  case "${svc}" in
    quant-order-manager)
      echo "[env] ${svc} effective Brain settings:"
      for key in \
        BRAIN_ENABLED \
        ORDER_MANAGER_BRAIN_GATE_ENABLED \
        ORDER_MANAGER_BRAIN_GATE_APPLY_WITH_PRESERVE_INTENT \
        BRAIN_BACKEND \
        BRAIN_OLLAMA_URL \
        BRAIN_OLLAMA_MODEL \
        BRAIN_SAMPLE_RATE \
        BRAIN_TTL_SEC \
        BRAIN_FAIL_POLICY \
        BRAIN_PROMPT_AUTO_TUNE_ENABLED \
        BRAIN_PROMPT_AUTO_TUNE_INTERVAL_SEC \
        BRAIN_PROMPT_AUTO_TUNE_MIN_DECISIONS \
        BRAIN_PROMPT_AUTO_TUNE_LOOKBACK_HOURS \
        BRAIN_PROMPT_PROFILE_PATH \
        BRAIN_PROMPT_REPORT_LATEST_PATH \
        BRAIN_PROMPT_REPORT_HISTORY_PATH; do
        printf '[env] %s=%s\n' "${key}" "${!key:-}"
      done
      ;;
  esac
}

load_env_for_service() {
  local svc="$1"
  local emit_chain_log="${2:-0}"

  reset_env_chain_trace
  set -a
  source_env_file_once "${BASE_ENV}" "base"

  case "${svc}" in
    quant-order-manager)
      load_env_file_if_exists "${ROOT_DIR}/ops/env/quant-order-manager.env" "service"
      export ORDER_MANAGER_SERVICE_PORT="${ORDER_MANAGER_SERVICE_PORT:-8300}"
      ;;
    quant-position-manager)
      load_env_file_if_exists "${ROOT_DIR}/ops/env/quant-position-manager.env" "service"
      export POSITION_MANAGER_SERVICE_PORT="${POSITION_MANAGER_SERVICE_PORT:-8301}"
      ;;
    quant-scalp-ping-5s-b)
      load_env_file_if_exists "${ROOT_DIR}/ops/env/quant-scalp-ping-5s-b.env" "service"
      load_env_file_if_exists "${ROOT_DIR}/ops/env/scalp_ping_5s_b.env" "service"
      export SCALP_PING_5S_B_PERF_GUARD_MODE="${SCALP_PING_5S_B_PERF_GUARD_MODE:-reduce}"
      export SCALP_PING_5S_B_SL_STREAK_DIRECTION_FLIP_ENABLED="${SCALP_PING_5S_B_SL_STREAK_DIRECTION_FLIP_ENABLED:-1}"
      export SCALP_PING_5S_B_SIDE_METRICS_DIRECTION_FLIP_ENABLED="${SCALP_PING_5S_B_SIDE_METRICS_DIRECTION_FLIP_ENABLED:-1}"
      export SCALP_PING_5S_B_SIDE_ADVERSE_STACK_UNITS_ENABLED="${SCALP_PING_5S_B_SIDE_ADVERSE_STACK_UNITS_ENABLED:-1}"
      ;;
    quant-scalp-ping-5s-c)
      load_env_file_if_exists "${ROOT_DIR}/ops/env/quant-scalp-ping-5s-c.env" "service"
      load_env_file_if_exists "${ROOT_DIR}/ops/env/scalp_ping_5s_c.env" "service"
      ;;
    quant-scalp-ping-5s-d)
      load_env_file_if_exists "${ROOT_DIR}/ops/env/quant-scalp-ping-5s-d.env" "service"
      load_env_file_if_exists "${ROOT_DIR}/ops/env/scalp_ping_5s_d.env" "service"
      ;;
    quant-scalp-ping-5s-flow)
      load_env_file_if_exists "${ROOT_DIR}/ops/env/quant-scalp-ping-5s-flow.env" "service"
      load_env_file_if_exists "${ROOT_DIR}/ops/env/scalp_ping_5s_flow.env" "service"
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
    *)
      load_env_file_if_exists "${ROOT_DIR}/ops/env/${svc}.env" "service"
      ;;
  esac

  if [[ -n "${OVERRIDE_ENV}" ]]; then
    load_env_chain "${OVERRIDE_ENV}" "override"
  fi
  if [[ -n "${LOCAL_V2_EXTRA_ENV_FILES:-}" ]]; then
    load_env_chain "${LOCAL_V2_EXTRA_ENV_FILES}" "extra"
  fi

  if [[ "${emit_chain_log}" == "1" ]]; then
    log_env_chain_order "${svc}"
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
    trade_all)
      printf '%s\n' "${PROFILE_trade_all[@]}"
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

watchdog_script_path() {
  printf '%s\n' "${ROOT_DIR}/scripts/local_v2_watchdog.sh"
}

run_watchdog_action() {
  local action="$1"
  local script args

  script="$(watchdog_script_path)"
  if [[ ! -x "${script}" ]]; then
    echo "[error] watchdog script not executable: ${script}" >&2
    exit 1
  fi

  args=("${script}" "${action}" "--profile" "${PROFILE}" "--interval-sec" "${WATCHDOG_INTERVAL_SEC}" "--resume-gap-sec" "${WATCHDOG_RESUME_GAP_SEC}")
  if [[ -n "${OVERRIDE_ENV}" ]]; then
    args+=("--env" "${OVERRIDE_ENV}")
  fi
  if [[ -n "${SERVICES_RAW}" ]]; then
    args+=("--services" "${SERVICES_RAW}")
  fi
  if [[ "${WATCHDOG_VERBOSE}" == "1" ]]; then
    args+=("--verbose")
  fi
  "${args[@]}"
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
  local patterns pattern line pid cmd
  local -a pattern_list=()
  patterns="$(module_patterns_for_service "${svc}" || true)"
  [[ -n "${patterns}" ]] || return 0
  while IFS= read -r pattern; do
    [[ -n "${pattern}" ]] || continue
    pattern_list+=("${pattern}")
  done <<EOF_PATTERNS
${patterns}
EOF_PATTERNS
  [[ ${#pattern_list[@]} -gt 0 ]] || return 0

  while IFS= read -r line; do
    [[ -n "${line}" ]] || continue
    # `ps -axo pid=,command=` includes leading spaces in the pid column on macOS.
    # Normalize by extracting pid/command via regex instead of naive splitting.
    if [[ "${line}" =~ ^[[:space:]]*([0-9]+)[[:space:]]+(.*)$ ]]; then
      pid="${BASH_REMATCH[1]}"
      cmd="${BASH_REMATCH[2]}"
    else
      continue
    fi
    for pattern in "${pattern_list[@]}"; do
      if [[ "${cmd}" == *"${pattern}"* ]]; then
        printf '%s\n' "${pid}"
        break
      fi
    done
  done < <(ps -axo pid=,command= 2>/dev/null || true)
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

service_pid_is_healthy() {
  local svc="$1"
  local pid
  pid="$(read_pid "${svc}" || true)"
  [[ -n "${pid}" ]] || return 1
  is_running_pid "${pid}" || return 1
  pid_matches_service "${svc}" "${pid}" || return 1
  return 0
}

launch_detached_python_module() {
  local py="$1"
  local module="$2"
  local log_path="$3"

  "${py}" - "${py}" "${module}" "${log_path}" <<'PY'
import subprocess
import sys

py, module, log_path = sys.argv[1], sys.argv[2], sys.argv[3]

with open(log_path, "ab", buffering=0) as log_file:
    proc = subprocess.Popen(
        [py, "-m", module],
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )

print(proc.pid, end="")
PY
}

start_service() {
  local svc="$1"
  local module pf lf pid py extra_pids
  local max_attempts retry_delay attempt
  module="$(module_for_service "${svc}")"
  pf="$(pid_file_for "${svc}")"
  lf="$(log_file_for "${svc}")"
  max_attempts="$(as_positive_int "${LOCAL_V2_START_RETRY_COUNT:-}" "${DEFAULT_START_RETRY_COUNT}")"
  retry_delay="$(as_positive_int "${LOCAL_V2_START_RETRY_DELAY_SEC:-}" "${DEFAULT_START_RETRY_DELAY_SEC}")"

  for attempt in $(seq 1 "${max_attempts}"); do
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

    pid="$(
      (
        cd "${ROOT_DIR}"
        load_env_for_service "${svc}" "1" >> "${lf}" 2>&1
        log_effective_env_snapshot "${svc}" >> "${lf}" 2>&1
        launch_detached_python_module "${py}" "${module}" "${lf}"
      ) 2>> "${lf}" < /dev/null
    )"
    pid="$(printf '%s' "${pid}" | tr -d '[:space:]')"
    if [[ -z "${pid}" ]]; then
      echo "[warn] ${svc} launcher returned empty pid" >&2
      if [[ "${attempt}" -lt "${max_attempts}" ]]; then
        echo "[retry] ${svc} launch retry ${attempt}/${max_attempts}" >&2
        sleep "${retry_delay}"
        continue
      fi
      return 1
    fi
    echo "${pid}" > "${pf}"
    echo "[up] ${svc} pid=${pid} log=${lf}"

    sleep 0.2
    if ! is_running_pid "${pid}"; then
      echo "[warn] ${svc} exited immediately after launch (pid=${pid})" >&2
      tail -n 40 "${lf}" >&2 || true
      rm -f "${pf}"
      if [[ "${attempt}" -lt "${max_attempts}" ]]; then
        echo "[retry] ${svc} launch retry ${attempt}/${max_attempts}" >&2
        sleep "${retry_delay}"
        continue
      fi
      return 1
    fi

    if ! wait_for_service_health "${svc}" "${pid}"; then
      echo "[warn] ${svc} failed health check; stopping service" >&2
      stop_service "${svc}"
      if [[ "${attempt}" -lt "${max_attempts}" ]]; then
        echo "[retry] ${svc} health retry ${attempt}/${max_attempts}" >&2
        sleep "${retry_delay}"
        continue
      fi
      return 1
    fi

    return 0
  done

  return 1
}

reconcile_started_services() {
  local rounds settle round svc
  local -a services=("$@")
  local -a unresolved=()
  rounds="$(as_positive_int "${LOCAL_V2_POST_UP_RECONCILE_ROUNDS:-}" "${DEFAULT_POST_UP_RECONCILE_ROUNDS}")"
  settle="$(as_positive_int "${LOCAL_V2_POST_UP_SETTLE_SEC:-}" "${DEFAULT_POST_UP_SETTLE_SEC}")"
  if [[ ${#services[@]} -eq 0 ]] || [[ "${rounds}" -le 0 ]]; then
    return 0
  fi

  for round in $(seq 1 "${rounds}"); do
    if [[ "${settle}" -gt 0 ]]; then
      sleep "${settle}"
    fi

    unresolved=()
    for svc in "${services[@]}"; do
      if ! service_pid_is_healthy "${svc}"; then
        unresolved+=("${svc}")
      fi
    done

    if [[ ${#unresolved[@]} -eq 0 ]]; then
      return 0
    fi

    if [[ "${round}" -lt "${rounds}" ]]; then
      echo "[reconcile] round=${round}/${rounds} restarting services=$(IFS=, ; printf '%s' "${unresolved[*]}")" >&2
      for svc in "${unresolved[@]}"; do
        start_service "${svc}" || true
      done
    fi
  done

  echo "[error] unresolved services after reconcile: $(IFS=, ; printf '%s' "${unresolved[*]}")" >&2
  return 1
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
  local pid pf port
  local has_pid_file=0 pid_running=0 pid_match=0 stale_pid_file=0
  local pattern_raw pattern_pid
  local pattern_pids=()
  local listener_raw listener_pid
  local listener_pids=()
  local listener_target=0 listener_non_target=0

  pf="$(pid_file_for "${svc}")"
  pid="$(read_pid "${svc}" || true)"

  if [[ -f "${pf}" ]]; then
    has_pid_file=1
  fi

  if [[ -n "${pid}" ]] && is_running_pid "${pid}"; then
    pid_running=1
    if pid_matches_service "${svc}" "${pid}"; then
      pid_match=1
    fi
  fi

  if [[ "${has_pid_file}" == "1" ]] && ([[ -z "${pid}" ]] || [[ "${pid_running}" != "1" ]] || [[ "${pid_match}" != "1" ]]); then
    stale_pid_file=1
  fi

  pattern_raw="$(service_process_pids "${svc}" | sort -u)"
  while IFS= read -r pattern_pid; do
    [[ -n "${pattern_pid}" ]] || continue
    if ! is_running_pid "${pattern_pid}"; then
      continue
    fi
    if ! pid_matches_service "${svc}" "${pattern_pid}"; then
      continue
    fi
    if ! contains_value "${pattern_pid}" "${pattern_pids[@]-}"; then
      pattern_pids+=("${pattern_pid}")
    fi
  done <<EOF_PATTERN_PIDS
${pattern_raw}
EOF_PATTERN_PIDS

  port="$(service_port_for "${svc}" || true)"
  if [[ -n "${port}" ]]; then
    listener_raw="$(port_listener_pids "${port}")"
    while IFS= read -r listener_pid; do
      [[ -n "${listener_pid}" ]] || continue
      if ! is_running_pid "${listener_pid}"; then
        continue
      fi
      if ! contains_value "${listener_pid}" "${listener_pids[@]-}"; then
        listener_pids+=("${listener_pid}")
      fi
      if pid_matches_service "${svc}" "${listener_pid}"; then
        listener_target=1
      else
        listener_non_target=1
      fi
    done <<EOF_LISTENER_PIDS
${listener_raw}
EOF_LISTENER_PIDS
  fi

  if [[ "${pid_match}" == "1" ]]; then
    if [[ -n "${port}" ]] && [[ ${#listener_pids[@]} -gt 0 ]] && [[ "${listener_target}" != "1" ]]; then
      echo "[port_conflict] ${svc} pid=${pid} port=${port} listeners=$(IFS=, ; printf '%s' "${listener_pids[*]}")"
      return 0
    fi
    echo "[running] ${svc} pid=${pid} log=$(log_file_for "${svc}")"
    return 0
  fi

  if [[ "${stale_pid_file}" == "1" ]] && [[ ${#pattern_pids[@]} -eq 1 ]]; then
    printf '%s\n' "${pattern_pids[0]}" > "${pf}"
    echo "[running_by_pattern] ${svc} pid=${pattern_pids[0]} healed_pid_file=1 log=$(log_file_for "${svc}")"
    return 0
  fi

  if [[ "${has_pid_file}" != "1" ]] && [[ ${#pattern_pids[@]} -eq 1 ]]; then
    printf '%s\n' "${pattern_pids[0]}" > "${pf}"
    echo "[running_by_pattern] ${svc} pid=${pattern_pids[0]} healed_pid_file=1 log=$(log_file_for "${svc}")"
    return 0
  fi

  if [[ -n "${port}" ]] && [[ ${#listener_pids[@]} -gt 0 ]] && [[ "${listener_target}" != "1" ]]; then
    echo "[port_conflict] ${svc} port=${port} listeners=$(IFS=, ; printf '%s' "${listener_pids[*]}")"
    return 0
  fi

  if [[ "${stale_pid_file}" == "1" ]] && [[ ${#pattern_pids[@]} -eq 0 ]] && [[ ${#listener_pids[@]} -eq 0 ]]; then
    echo "[stopped] ${svc} stale_pid_file=${pid:-<empty>}"
    return 0
  fi

  if [[ "${stale_pid_file}" == "1" ]]; then
    if [[ ${#pattern_pids[@]} -gt 0 ]]; then
      echo "[stale_pid_file] ${svc} pid_file=${pid:-<empty>} pattern_pids=$(IFS=, ; printf '%s' "${pattern_pids[*]}")"
    else
      echo "[stale_pid_file] ${svc} pid_file=${pid:-<empty>}"
    fi
    return 0
  fi

  if [[ ${#pattern_pids[@]} -gt 0 ]]; then
    echo "[running_by_pattern] ${svc} pids=$(IFS=, ; printf '%s' "${pattern_pids[*]}") log=$(log_file_for "${svc}")"
    return 0
  fi

  echo "[stopped] ${svc}"
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
WATCHDOG_DAEMON=0
WATCHDOG_ONCE=0
WATCHDOG_VERBOSE=0
WATCHDOG_INTERVAL_SEC="${DEFAULT_WATCHDOG_INTERVAL_SEC}"
WATCHDOG_RESUME_GAP_SEC="${DEFAULT_WATCHDOG_RESUME_GAP_SEC}"

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
    --daemon)
      WATCHDOG_DAEMON=1
      shift
      ;;
    --once)
      WATCHDOG_ONCE=1
      shift
      ;;
    --interval-sec)
      WATCHDOG_INTERVAL_SEC="$2"
      shift 2
      ;;
    --resume-gap-sec)
      WATCHDOG_RESUME_GAP_SEC="$2"
      shift 2
      ;;
    --verbose)
      WATCHDOG_VERBOSE=1
      shift
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
  up|down|restart)
    acquire_stack_operation_lock
    trap release_stack_operation_lock EXIT
    ;;
esac

case "${CMD}" in
  up)
    guard_parity_conflict
    for svc in "${RESOLVED_SERVICES[@]}"; do
      start_service "${svc}"
    done
    reconcile_started_services "${RESOLVED_SERVICES[@]}"
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
    reconcile_started_services "${RESOLVED_SERVICES[@]}"
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
  watchdog)
    if [[ "${WATCHDOG_DAEMON}" == "1" && "${WATCHDOG_ONCE}" == "1" ]]; then
      echo "[error] watchdog cannot use --daemon and --once together" >&2
      exit 2
    fi
    if [[ "${WATCHDOG_ONCE}" == "1" ]]; then
      run_watchdog_action "once"
    elif [[ "${WATCHDOG_DAEMON}" == "1" ]]; then
      run_watchdog_action "start"
    else
      run_watchdog_action "run"
    fi
    ;;
  watchdog-stop)
    run_watchdog_action "stop"
    ;;
  watchdog-status)
    run_watchdog_action "status"
    ;;
  *)
    echo "[error] unknown command: ${CMD}" >&2
    usage
    exit 2
    ;;
esac
