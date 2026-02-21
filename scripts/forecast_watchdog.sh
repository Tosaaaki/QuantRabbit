#!/usr/bin/env bash
set -euo pipefail

TAG="qr-forecast-watchdog"

SERVICE="${FORECAST_WATCHDOG_SERVICE:-quant-forecast.service}"
BQ_SERVICE="${FORECAST_WATCHDOG_BQ_SERVICE:-quant-bq-sync.service}"
URL="${FORECAST_WATCHDOG_URL:-http://127.0.0.1:8302/health}"
TIMEOUT_SEC="${FORECAST_WATCHDOG_TIMEOUT_SEC:-4}"
MAX_FAILS="${FORECAST_WATCHDOG_MAX_FAILS:-3}"
RESTART_COOLDOWN_SEC="${FORECAST_WATCHDOG_RESTART_COOLDOWN_SEC:-120}"
RECOVER_WAIT_SEC="${FORECAST_WATCHDOG_RECOVER_WAIT_SEC:-8}"
DISABLE_BQ_ON_ESCALATE="${FORECAST_WATCHDOG_DISABLE_BQ_ON_ESCALATE:-1}"
STATE_FILE="${FORECAST_WATCHDOG_STATE_FILE:-/var/tmp/quant-forecast-watchdog.state}"
ENABLED="${FORECAST_WATCHDOG_ENABLED:-1}"

if [[ "${ENABLED}" != "1" ]]; then
  exit 0
fi

if ! systemctl list-unit-files --type=service | grep -q "^${SERVICE}"; then
  logger -t "${TAG}" "target service not found: ${SERVICE}"
  exit 0
fi

read_state() {
  fail_count=0
  restart_count=0
  last_restart_epoch=0
  if [[ -f "${STATE_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${STATE_FILE}" || true
  fi
  fail_count="${fail_count:-0}"
  restart_count="${restart_count:-0}"
  last_restart_epoch="${last_restart_epoch:-0}"
}

write_state() {
  local dir
  dir="$(dirname "${STATE_FILE}")"
  mkdir -p "${dir}"
  cat >"${STATE_FILE}" <<EOF
fail_count=${fail_count}
restart_count=${restart_count}
last_restart_epoch=${last_restart_epoch}
EOF
}

health_ok() {
  curl -fsS -m "${TIMEOUT_SEC}" "${URL}" >/dev/null
}

now_epoch="$(date +%s)"
read_state

if health_ok; then
  fail_count=0
  restart_count=0
  write_state
  exit 0
fi

fail_count=$((fail_count + 1))
write_state

if (( fail_count < MAX_FAILS )); then
  logger -t "${TAG}" "health failed (${fail_count}/${MAX_FAILS}), waiting before restart"
  exit 0
fi

if (( now_epoch - last_restart_epoch < RESTART_COOLDOWN_SEC )); then
  logger -t "${TAG}" "health failed but restart cooldown active (${RESTART_COOLDOWN_SEC}s)"
  exit 0
fi

logger -t "${TAG}" "health failed ${fail_count} times; restarting ${SERVICE}"
systemctl restart "${SERVICE}" || true
sleep "${RECOVER_WAIT_SEC}"
now_epoch="$(date +%s)"
last_restart_epoch="${now_epoch}"
restart_count=$((restart_count + 1))

if health_ok; then
  fail_count=0
  restart_count=0
  write_state
  logger -t "${TAG}" "service recovered after restart"
  exit 0
fi

if [[ "${DISABLE_BQ_ON_ESCALATE}" == "1" ]] && systemctl list-unit-files --type=service | grep -q "^${BQ_SERVICE}"; then
  if systemctl is-active --quiet "${BQ_SERVICE}"; then
    logger -t "${TAG}" "escalation: stopping ${BQ_SERVICE} to protect forecast stability"
    systemctl stop "${BQ_SERVICE}" || true
  fi
fi

systemctl restart "${SERVICE}" || true
sleep "${RECOVER_WAIT_SEC}"
if health_ok; then
  fail_count=0
  restart_count=0
  logger -t "${TAG}" "service recovered after escalation restart"
else
  fail_count="${MAX_FAILS}"
  logger -t "${TAG}" "service still unhealthy after escalation"
fi
write_state
exit 0
