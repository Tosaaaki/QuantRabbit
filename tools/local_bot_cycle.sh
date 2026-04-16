#!/bin/bash
set -euo pipefail

ROOT="$(cd -P -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
LOG_DIR="${ROOT}/logs"
LOCK_NAME="local_bot_cycle"
TIMEOUT_MIN=3
PYTHON_BIN="${ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

mkdir -p "${LOG_DIR}"

if ! "${PYTHON_BIN}" "${ROOT}/tools/task_lock.py" acquire "${LOCK_NAME}" "${TIMEOUT_MIN}" --pid $$ --caller local-bot-cycle >/dev/null 2>&1; then
  exit 0
fi

cleanup() {
  "${PYTHON_BIN}" "${ROOT}/tools/task_lock.py" release "${LOCK_NAME}" --caller local-bot-cycle >/dev/null 2>&1 || true
}
trap cleanup EXIT

{
  printf '\n[%s] LOCAL_BOT_CYCLE start\n' "$(date -u '+%Y-%m-%d %H:%M:%SZ')"
  "${PYTHON_BIN}" "${ROOT}/tools/refresh_factor_cache.py" --all --tf=M1,M5,M15,H1 --quiet || true
  # Brake layer (must run before any entry decisions consume brake_state.json)
  "${PYTHON_BIN}" "${ROOT}/tools/inventory_brake.py" || true
  "${PYTHON_BIN}" "${ROOT}/tools/regime_switch.py" || true
  # Existing policy + lifecycle layer
  "${PYTHON_BIN}" "${ROOT}/tools/bot_policy_guard.py" || true
  "${PYTHON_BIN}" "${ROOT}/tools/bot_trade_manager.py" || true
  # Entry bots (gated by brake_state + regime_state via brake_gate)
  "${PYTHON_BIN}" "${ROOT}/tools/trend_bot.py" || true
  "${PYTHON_BIN}" "${ROOT}/tools/range_bot.py" || true
  # Stranded inventory drainer — sets BE-ish TP on accumulated counter-side bags
  "${PYTHON_BIN}" "${ROOT}/tools/stranded_drain.py" || true
  printf '[%s] LOCAL_BOT_CYCLE end\n' "$(date -u '+%Y-%m-%d %H:%M:%SZ')"
} >> "${LOG_DIR}/local_bot_cycle.log" 2>&1
