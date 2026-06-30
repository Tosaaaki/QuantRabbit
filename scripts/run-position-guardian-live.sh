#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ROOT_DIR="${QR_TRADER_ROOT_DIR:-/Users/tossaki/App/QuantRabbit-live}"
cd "$ROOT_DIR"

export PYTHONPATH="src"
if [[ -z "${QR_PYTHON:-}" ]]; then
  if [[ -x /opt/homebrew/bin/python3 ]]; then
    QR_PYTHON="/opt/homebrew/bin/python3"
  else
    QR_PYTHON="/usr/bin/python3"
  fi
fi
readonly QR_PYTHON

export QR_OANDA_ENV_FILE="${QR_OANDA_ENV_FILE:-.env.local}"
export QR_LIVE_ENABLED="${QR_LIVE_ENABLED:-0}"

load_live_enabled_from_env_file() {
  if [[ -n "${QR_LIVE_ENABLED:-}" && "${QR_LIVE_ENABLED}" != "0" ]]; then
    return 0
  fi
  if [[ ! -f "$QR_OANDA_ENV_FILE" ]]; then
    return 0
  fi
  local line value
  line="$(grep -E "^[[:space:]]*(export[[:space:]]+)?QR_LIVE_ENABLED[[:space:]]*=" "$QR_OANDA_ENV_FILE" | tail -n 1 || true)"
  if [[ -z "$line" ]]; then
    return 0
  fi
  value="${line#*=}"
  value="${value%%#*}"
  value="$(printf '%s' "$value" | tr -d "[:space:]\"'")"
  case "$value" in
    0|1) export QR_LIVE_ENABLED="$value" ;;
    *)
      echo "[run-position-guardian-live] invalid QR_LIVE_ENABLED in ${QR_OANDA_ENV_FILE}; expected 0 or 1." >&2
      exit 2
      ;;
  esac
}

readonly QR_AUTOTRADE_LOCK_DIR="${QR_AUTOTRADE_LOCK_DIR:-${ROOT_DIR}/.quant_rabbit_live.lock}"
readonly QR_AUTOTRADE_LOCK_WAIT_SECONDS="${QR_AUTOTRADE_LOCK_WAIT_SECONDS:-0}"
readonly QR_AUTOTRADE_LOCK_WAIT_COMMAND_PATTERN="${QR_AUTOTRADE_LOCK_WAIT_COMMAND_PATTERN:-}"
readonly QR_AUTOTRADE_LOCK_POLL_SECONDS="${QR_AUTOTRADE_LOCK_POLL_SECONDS:-2}"
# launchd can fire this guardian every 30s. When the full trader cycle is
# already holding the live lock, the guardian must yield instead of turning
# profit protection into a cycle-blocking error stream.
readonly QR_POSITION_GUARDIAN_LOCK_BUSY_MODE="${QR_POSITION_GUARDIAN_LOCK_BUSY_MODE:-skip}"

source "${SCRIPT_DIR}/qr-live-lock.sh"

acquire_lock() {
  qr_live_lock_acquire \
    "$QR_AUTOTRADE_LOCK_DIR" \
    "run-position-guardian-live" \
    "$QR_AUTOTRADE_LOCK_WAIT_SECONDS" \
    "$QR_AUTOTRADE_LOCK_WAIT_COMMAND_PATTERN" \
    "$QR_AUTOTRADE_LOCK_POLL_SECONDS"
}

skip_if_live_lock_busy() {
  local existing_pid existing_command existing_label
  existing_pid="$(qr_live_lock_pid "$QR_AUTOTRADE_LOCK_DIR")"
  if ! qr_live_lock_pid_is_running "$existing_pid"; then
    return 0
  fi
  existing_command="$(ps -p "$existing_pid" -o command= 2>/dev/null || true)"
  existing_label=""
  if [[ -f "${QR_AUTOTRADE_LOCK_DIR}/command" ]]; then
    existing_label="$(cat "${QR_AUTOTRADE_LOCK_DIR}/command" 2>/dev/null || true)"
  fi
  if [[ "$QR_POSITION_GUARDIAN_LOCK_BUSY_MODE" == "skip" ]]; then
    echo "[run-position-guardian-live] live runtime lock busy pid=${existing_pid} label=${existing_label:-unknown}; skipped guardian cycle." >&2
    if [[ -n "$existing_command" ]]; then
      echo "[run-position-guardian-live] lock owner command=${existing_command}" >&2
    fi
    exit 0
  fi
}

open_trader_pairs() {
  "$QR_PYTHON" - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text())
except Exception:
    print("")
    raise SystemExit(0)
pairs = sorted({
    str(item.get("pair") or item.get("instrument") or "").upper()
    for item in payload.get("positions", []) or []
    if str(item.get("owner") or "") == "trader" and str(item.get("pair") or item.get("instrument") or "")
})
print(",".join(pairs))
PY
}

write_no_position_artifact() {
  "$QR_PYTHON" - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path

payload = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "NO_POSITION",
    "sent": False,
    "reason": "no trader-owned open position; guardian skipped position execution",
}
path = Path("data/position_guardian.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
PY
}

load_live_enabled_from_env_file
skip_if_live_lock_busy
acquire_lock

# Mirror the live trader risk/protection defaults so direct CLI calls and the
# full autotrade wrapper read protected SL-free positions the same way.
export QR_GEOMETRY_ATR_MULT="${QR_GEOMETRY_ATR_MULT:-5.0}"
export QR_GEOMETRY_SPREAD_FLOOR_MULT="${QR_GEOMETRY_SPREAD_FLOOR_MULT:-12.0}"
export QR_TRADER_DISABLE_SL_REPAIR="${QR_TRADER_DISABLE_SL_REPAIR:-1}"
export QR_MAX_PORTFOLIO_POSITIONS="${QR_MAX_PORTFOLIO_POSITIONS:-10}"
export QR_TRADER_POSITION_NAV_PCT="${QR_TRADER_POSITION_NAV_PCT:-30}"
export QR_TRADER_BASE_UNITS="${QR_TRADER_BASE_UNITS:-3000}"
export QR_DISABLE_AUTO_CLOSE="${QR_DISABLE_AUTO_CLOSE:-1}"
export QR_REQUIRE_FORECAST_FOR_LIVE="${QR_REQUIRE_FORECAST_FOR_LIVE:-1}"
export QR_REQUIRE_TELEMETRY_FOR_LIVE="${QR_REQUIRE_TELEMETRY_FOR_LIVE:-1}"

guardian_snapshot="${QR_POSITION_GUARDIAN_SNAPSHOT:-data/position_guardian_broker_snapshot.json}"
guardian_management="${QR_POSITION_GUARDIAN_MANAGEMENT:-data/position_guardian_management.json}"
guardian_management_report="${QR_POSITION_GUARDIAN_MANAGEMENT_REPORT:-docs/position_guardian_management_report.md}"
guardian_execution="${QR_POSITION_GUARDIAN_EXECUTION:-data/position_guardian_execution.json}"
guardian_execution_report="${QR_POSITION_GUARDIAN_EXECUTION_REPORT:-docs/position_guardian_execution_report.md}"
guardian_charts="${QR_POSITION_GUARDIAN_PAIR_CHARTS:-data/position_guardian_pair_charts.json}"
guardian_report="${QR_POSITION_GUARDIAN_PAIR_CHARTS_REPORT:-docs/position_guardian_pair_charts_report.md}"
guardian_count="${QR_POSITION_GUARDIAN_CANDLE_COUNT:-120}"

run_guardian_event_router() {
  "$QR_PYTHON" -m quant_rabbit.cli guardian-event-router \
    --snapshot "$guardian_snapshot" \
    --pair-charts "$guardian_charts" \
    --order-intents "${QR_POSITION_GUARDIAN_ORDER_INTENTS:-data/order_intents.json}" \
    --position-management "$guardian_management" \
    --thesis-evolution "${QR_POSITION_GUARDIAN_THESIS_EVOLUTION:-data/thesis_evolution_report.json}" \
    --forecast-persistence "${QR_POSITION_GUARDIAN_FORECAST_PERSISTENCE:-data/forecast_persistence_report.json}" \
    --market-context-matrix "${QR_POSITION_GUARDIAN_MARKET_CONTEXT_MATRIX:-data/market_context_matrix.json}" \
    --output "${QR_POSITION_GUARDIAN_EVENTS:-data/guardian_events.json}" \
    --escalation-output "${QR_POSITION_GUARDIAN_ESCALATION:-data/guardian_escalation.json}" \
    --report "${QR_POSITION_GUARDIAN_EVENT_REPORT:-docs/guardian_event_report.md}" \
    --action-review-report "${QR_POSITION_GUARDIAN_ACTION_REVIEW:-docs/guardian_action_review.md}"
}

"$QR_PYTHON" -m quant_rabbit.cli broker-snapshot --output "$guardian_snapshot"
pairs="$(open_trader_pairs "$guardian_snapshot")"
if [[ -z "$pairs" ]]; then
  run_guardian_event_router
  write_no_position_artifact
  echo "[run-position-guardian-live] no trader-owned open positions; skipped." >&2
  exit 0
fi

"$QR_PYTHON" -m quant_rabbit.cli pair-charts \
  --pairs "$pairs" \
  --timeframes M1,M5,M15,M30,H1,H4,D \
  --count "$guardian_count" \
  --output "$guardian_charts" \
  --report "$guardian_report"

"$QR_PYTHON" -m quant_rabbit.cli position-management \
  --snapshot "$guardian_snapshot" \
  --pair-charts "$guardian_charts" \
  --output "$guardian_management" \
  --report "$guardian_management_report"

pexec_args=(
  position-execution
  --snapshot "$guardian_snapshot"
  --position-management "$guardian_management"
  --output "$guardian_execution"
  --report "$guardian_execution_report"
)
if [[ "$QR_LIVE_ENABLED" == "1" ]]; then
  pexec_args+=(--send --confirm-live)
fi

"$QR_PYTHON" -m quant_rabbit.cli "${pexec_args[@]}"
run_guardian_event_router
