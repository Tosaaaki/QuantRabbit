#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR_DEFAULT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="${REPO_DIR:-$REPO_DIR_DEFAULT}"
PY_BIN="${PY_BIN:-$REPO_DIR/.venv/bin/python}"

TRADES_DB="${PING5S_D_CANARY_TRADES_DB:-logs/trades.db}"
ORDERS_DB="${PING5S_D_CANARY_ORDERS_DB:-logs/orders.db}"
ENV_FILE="${PING5S_D_CANARY_ENV_FILE:-ops/env/scalp_ping_5s_d.env}"
STRATEGY_TAG="${PING5S_D_CANARY_STRATEGY_TAG:-scalp_ping_5s_d_live}"
WINDOW_MINUTES="${PING5S_D_CANARY_WINDOW_MINUTES:-120}"

MIN_JPY_PER_HOUR="${PING5S_D_CANARY_MIN_JPY_PER_HOUR:-0}"
MIN_TRADES_PER_HOUR="${PING5S_D_CANARY_MIN_TRADES_PER_HOUR:-6}"
MIN_OBSERVED_TRADES="${PING5S_D_CANARY_MIN_OBSERVED_TRADES:-6}"
MAX_MARGIN_REJECT="${PING5S_D_CANARY_MAX_MARGIN_REJECT:-0}"
ROLLBACK_JPY_PER_HOUR="${PING5S_D_CANARY_ROLLBACK_JPY_PER_HOUR:--200}"
PROMOTE_UNITS="${PING5S_D_CANARY_PROMOTE_UNITS:-22000}"
ROLLBACK_UNITS="${PING5S_D_CANARY_ROLLBACK_UNITS:-15000}"
TRADES_LOOKBACK_ROWS="${PING5S_D_CANARY_TRADES_LOOKBACK_ROWS:-20000}"
ORDERS_LOOKBACK_ROWS="${PING5S_D_CANARY_ORDERS_LOOKBACK_ROWS:-50000}"

OUT_JSON="${PING5S_D_CANARY_OUT_JSON:-logs/ping5s_d_canary_guard_latest.json}"
HISTORY_JSONL="${PING5S_D_CANARY_HISTORY_JSONL:-logs/ping5s_d_canary_guard_history.jsonl}"
APPLY="${PING5S_D_CANARY_APPLY:-0}"

cd "$REPO_DIR"
mkdir -p logs
mkdir -p "$(dirname "$OUT_JSON")"
mkdir -p "$(dirname "$HISTORY_JSONL")"
mkdir -p logs/.lock

lock_file="logs/.lock/ping5s_d_canary_guard.lock"

if command -v flock >/dev/null 2>&1; then
  exec 9>"$lock_file"
  if ! flock -n 9; then
    echo "[ping5s_d_canary_guard] another run is active; skip"
    exit 0
  fi
else
  echo "[ping5s_d_canary_guard] flock not found; run without lock"
fi

args=(
  "scripts/ping5s_d_canary_guard.py"
  "--trades-db" "$TRADES_DB"
  "--orders-db" "$ORDERS_DB"
  "--env-file" "$ENV_FILE"
  "--strategy-tag" "$STRATEGY_TAG"
  "--window-minutes" "$WINDOW_MINUTES"
  "--trades-lookback-rows" "$TRADES_LOOKBACK_ROWS"
  "--orders-lookback-rows" "$ORDERS_LOOKBACK_ROWS"
  "--min-jpy-per-hour" "$MIN_JPY_PER_HOUR"
  "--min-trades-per-hour" "$MIN_TRADES_PER_HOUR"
  "--min-observed-trades" "$MIN_OBSERVED_TRADES"
  "--max-margin-reject" "$MAX_MARGIN_REJECT"
  "--rollback-jpy-per-hour" "$ROLLBACK_JPY_PER_HOUR"
  "--promote-units" "$PROMOTE_UNITS"
  "--rollback-units" "$ROLLBACK_UNITS"
  "--out" "$OUT_JSON"
)

case "$APPLY" in
  1|[Tt][Rr][Uu][Ee]|[Yy][Ee][Ss]|[Oo][Nn])
    args+=("--apply")
    ;;
esac

"$PY_BIN" "${args[@]}"

if [[ -s "$OUT_JSON" ]]; then
  tr -d '\n' < "$OUT_JSON" >> "$HISTORY_JSONL"
  printf '\n' >> "$HISTORY_JSONL"
fi
