#!/usr/bin/env bash
set -u

SERIAL_OUT=""
if [[ -w /dev/ttyS0 ]]; then
  SERIAL_OUT="/dev/ttyS0"
fi
LOGGER_TAG="qr-health"
HAS_LOGGER=0
if command -v logger >/dev/null 2>&1; then
  HAS_LOGGER=1
fi

emit() {
  echo "$*"
  if [[ -n "$SERIAL_OUT" ]]; then
    echo "$*" >"$SERIAL_OUT" || true
  fi
  if [[ "$HAS_LOGGER" -eq 1 ]]; then
    logger -t "$LOGGER_TAG" -- "$*" || true
  fi
}

emit_block() {
  while IFS= read -r line; do
    emit "$line"
  done
}

ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
emit "[health] ts=$ts"

if [[ -f /var/lib/quantrabbit/deploy_id ]]; then
  emit "[health] deploy_id=$(cat /var/lib/quantrabbit/deploy_id)"
fi

if command -v uptime >/dev/null 2>&1; then
  emit "[health] uptime=$(uptime -p)"
fi

if command -v systemctl >/dev/null 2>&1; then
  state="$(systemctl is-active quantrabbit.service 2>/dev/null || true)"
  emit "[health] quantrabbit.service=$state"
  systemctl list-units --no-pager --plain 'quantrabbit.service' 'quant-*' 'qr-*' 2>/dev/null | sed -n '1,50p' | emit_block
fi

if command -v journalctl >/dev/null 2>&1; then
  emit "[health] journalctl quantrabbit.service (tail 40)"
  journalctl -u quantrabbit.service -n 40 --no-pager 2>/dev/null | emit_block || true
fi

if command -v sqlite3 >/dev/null 2>&1; then
  DB_BASE="/home/tossaki/QuantRabbit/logs"
  TRADES_DB="$DB_BASE/trades.db"
  SIGNALS_DB="$DB_BASE/signals.db"
  ORDERS_DB="$DB_BASE/orders.db"
  if [[ -f "$TRADES_DB" ]]; then
    emit "[health] trades last 5"
    sqlite3 "$TRADES_DB" "select ticket_id,pocket,units,entry_time,close_time,pl_pips from trades order by entry_time desc limit 5;" | emit_block || true
    emit "[health] trades count last 24h"
    sqlite3 "$TRADES_DB" "select count(*) from trades where entry_time >= datetime('now','-1 day');" | emit_block || true
  else
    emit "[health] trades.db missing: $TRADES_DB"
  fi
  if [[ -f "$SIGNALS_DB" ]]; then
    emit "[health] signals last 5"
    sqlite3 "$SIGNALS_DB" "select datetime(ts_ms/1000,'unixepoch') as ts, json_extract(payload,'$.pocket'), json_extract(payload,'$.action'), json_extract(payload,'$.confidence'), json_extract(payload,'$.tag') from signals order by ts_ms desc limit 5;" | emit_block || true
  else
    emit "[health] signals.db missing: $SIGNALS_DB"
  fi
  if [[ -f "$ORDERS_DB" ]]; then
    emit "[health] orders last 5"
    sqlite3 "$ORDERS_DB" "select ts,pocket,side,units,client_order_id,status from orders order by ts desc limit 5;" | emit_block || true
  else
    emit "[health] orders.db missing: $ORDERS_DB"
  fi
fi
