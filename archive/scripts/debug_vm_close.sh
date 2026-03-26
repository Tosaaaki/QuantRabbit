#!/usr/bin/env bash
set -euo pipefail

# QuantRabbit: VM close-diagnosis helper
#
# Pulls recent service logs and DB rows from the trading VM to help explain
# why an exit/close just occurred.
#
# Usage:
#   scripts/debug_vm_close.sh [-p PROJECT] [-z ZONE] [-m INSTANCE] [-w MIN]
#                             [-t] [-k SSH_KEYFILE]
# Defaults:
#   PROJECT=quantrabbit, ZONE=asia-northeast1-a, INSTANCE=fx-trader-vm, MIN=15
#   -t enables IAP tunnel; -k sets OS Login SSH key file if needed.

PROJ="quantrabbit"
ZONE="asia-northeast1-a"
INST="fx-trader-vm"
WINDOW_MIN=15
USE_IAP=0
KEYFILE=""

# Ensure gcloud binary is reachable when PATH is minimal
if ! command -v gcloud >/dev/null 2>&1; then
  SDK_BIN="$HOME/google-cloud-sdk/bin"
  if [[ -d "$SDK_BIN" ]]; then
    export PATH="$SDK_BIN:$PATH"
  fi
fi

while getopts ":p:z:m:w:k:t" opt; do
  case $opt in
    p) PROJ="$OPTARG" ;;
    z) ZONE="$OPTARG" ;;
    m) INST="$OPTARG" ;;
    w) WINDOW_MIN="$OPTARG" ;;
    k) KEYFILE="$OPTARG" ;;
    t) USE_IAP=1 ;;
    :) echo "Option -$OPTARG requires an argument" >&2; exit 2 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; exit 2 ;;
  esac
done

if ! command -v gcloud >/dev/null 2>&1; then
  echo "[error] gcloud not found. Install and authenticate first." >&2
  echo "        See docs/GCP_DEPLOY_SETUP.md or run scripts/install_gcloud.sh" >&2
  exit 1
fi

GCLOUD_SSH=(gcloud compute ssh "$INST" --project "$PROJ" --zone "$ZONE")
if [[ $USE_IAP -eq 1 ]]; then
  GCLOUD_SSH+=(--tunnel-through-iap)
fi
if [[ -n "$KEYFILE" ]]; then
  GCLOUD_SSH+=(--ssh-key-file "$KEYFILE")
fi

run_remote() {
  local cmd="$1"
  "${GCLOUD_SSH[@]}" --command "$cmd"
}

echo "[info] Project: $PROJ  Zone: $ZONE  Instance: $INST  Window: ${WINDOW_MIN}m"
echo "[info] Checking VM time and service status..."
run_remote "echo '== VM time (UTC) =='; date -u; echo; systemctl status quantrabbit.service -n 20 --no-pager || true"

echo
echo "[info] Recent service logs (filtered)"
run_remote "journalctl -u quantrabbit.service --since '-${WINDOW_MIN} min' --no-pager -o short-iso | egrep -i '\\[EXIT\\]|CLOSE|Risk|range_mode|event|reverse|ADX|RSI|takeProfit|stopLoss|stage|cooldown|allow_reentry|halt_reason|stream_disconnected|SCALP-EXIT|SCALP-TICK' || true"

echo
echo "[info] trades.db: last 5 exits"
run_remote "sudo -u tossaki sqlite3 /home/tossaki/QuantRabbit/logs/trades.db \"SELECT close_time, pocket, ROUND(pl_pips,2), close_reason, ticket_id FROM trades WHERE close_time IS NOT NULL ORDER BY close_time DESC LIMIT 5;\"" || true

echo
echo "[info] trades.db: exits within window"
run_remote "sudo -u tossaki sqlite3 /home/tossaki/QuantRabbit/logs/trades.db \"SELECT close_time, pocket, ROUND(pl_pips,2), close_reason, ticket_id FROM trades WHERE close_time >= datetime('now','-${WINDOW_MIN} minutes') ORDER BY close_time DESC;\"" || true

echo
echo "[info] orders.db: last 8 rows (if present)"
run_remote "sudo -u tossaki bash -lc 'test -f /home/tossaki/QuantRabbit/logs/orders.db && sqlite3 /home/tossaki/QuantRabbit/logs/orders.db \"SELECT ts, status, pocket, side, units, executed_price, client_order_id, ticket_id FROM orders ORDER BY ts DESC LIMIT 12;\" || echo orders.db\ not\ found'" || true

echo
echo "[info] metrics: last ${WINDOW_MIN} min (key metrics)"
run_remote "sudo -u tossaki sqlite3 /home/tossaki/QuantRabbit/logs/metrics.db \"SELECT ts, metric, value, substr(tags,1,120) FROM metrics WHERE ts >= datetime('now','-${WINDOW_MIN} minutes') AND metric IN ('decision_latency_ms','data_lag_ms','range_mode','state','order_success_rate','reject_rate','fast_scalp_exit','exit_reason') ORDER BY ts DESC LIMIT 100;\"" || true

echo
echo "[hint] Common close reasons to look for:"
echo "  - ExitManager reverse-signal threshold hit (e.g., confidence >= 70)"
echo "  - Range mode active: early TP/fast cut (e.g., +1.6p or -1.0p)"
echo "  - Event lock / state transitions (EVENT_LOCK, MICRO_STOP, RECOVERY)"
echo "  - RiskGuard clamp/reject or stage cooldown block"
echo "  - SL/TP touched; partial reductions from order_manager"
