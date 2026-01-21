#!/usr/bin/env bash
set -u

ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[health] ts=$ts"

if [[ -f /var/lib/quantrabbit/deploy_id ]]; then
  echo "[health] deploy_id=$(cat /var/lib/quantrabbit/deploy_id)"
fi

if command -v uptime >/dev/null 2>&1; then
  echo "[health] uptime=$(uptime -p)"
fi

if command -v systemctl >/dev/null 2>&1; then
  state="$(systemctl is-active quantrabbit.service 2>/dev/null || true)"
  echo "[health] quantrabbit.service=$state"
  systemctl list-units --no-pager --plain 'quantrabbit.service' 'quant-*' 'qr-*' 2>/dev/null | sed -n '1,50p'
fi

if command -v journalctl >/dev/null 2>&1; then
  echo "[health] journalctl quantrabbit.service (tail 40)"
  journalctl -u quantrabbit.service -n 40 --no-pager 2>/dev/null || true
fi

if command -v sqlite3 >/dev/null 2>&1; then
  DB_BASE="/home/tossaki/QuantRabbit/logs"
  TRADES_DB="$DB_BASE/trades.db"
  SIGNALS_DB="$DB_BASE/signals.db"
  ORDERS_DB="$DB_BASE/orders.db"
  if [[ -f "$TRADES_DB" ]]; then
    echo "[health] trades last 5"
    sqlite3 "$TRADES_DB" "select ticket_id,pocket,units,entry_time,close_time,pl_pips from trades order by entry_time desc limit 5;" || true
    echo "[health] trades count last 24h"
    sqlite3 "$TRADES_DB" "select count(*) from trades where entry_time >= datetime('now','-1 day');" || true
  else
    echo "[health] trades.db missing: $TRADES_DB"
  fi
  if [[ -f "$SIGNALS_DB" ]]; then
    echo "[health] signals last 5"
    sqlite3 "$SIGNALS_DB" "select ts,pocket,action,confidence,tag from signals order by ts desc limit 5;" || true
  else
    echo "[health] signals.db missing: $SIGNALS_DB"
  fi
  if [[ -f "$ORDERS_DB" ]]; then
    echo "[health] orders last 5"
    sqlite3 "$ORDERS_DB" "select ts,pocket,side,units,client_order_id,status from orders order by ts desc limit 5;" || true
  else
    echo "[health] orders.db missing: $ORDERS_DB"
  fi
fi
