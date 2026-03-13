#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUERY="${1:-}"
LIMIT="${2:-5}"

if [[ -z "${QUERY}" ]]; then
  echo "usage: scripts/change_preflight.sh \"<strategy_tag or hypothesis_key or close_reason>\" [limit]" >&2
  exit 2
fi

echo "CHANGE preflight"
echo "repo=${ROOT_DIR}"
echo "query=${QUERY}"
echo "limit=${LIMIT}"
echo

echo "== Required Rules =="
echo "- local-only: VM/GCP/Cloud Run は使わない"
echo "- USD/JPY 市況確認を先にやる"
echo "- TRADE_FINDINGS を先に読む"
echo "- 変更後は docs/TRADE_FINDINGS.md へ Why/Hypothesis/Observed/Verdict/Next Action を残す"
echo

echo "== Local Health Refresh =="
PYTHONWARNINGS="ignore::FutureWarning" "${ROOT_DIR}/scripts/collect_local_health.sh"
echo

echo "== USD/JPY Market =="
python3 - "${ROOT_DIR}" <<'PY'
from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def fmt(value, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


root = Path(sys.argv[1])
health = load_json(root / "logs" / "health_snapshot.json") or {}
tick_cache = load_json(root / "logs" / "tick_cache.json") or []
factor_cache = load_json(root / "logs" / "factor_cache.json") or {}
account = load_json(root / "logs" / "oanda_account_snapshot_live.json") or {}

latest_tick = tick_cache[-1] if isinstance(tick_cache, list) and tick_cache else {}
latest_epoch = latest_tick.get("epoch")
latest_bid = latest_tick.get("bid")
latest_ask = latest_tick.get("ask")
latest_mid = latest_tick.get("mid")
if latest_mid is None and latest_bid is not None and latest_ask is not None:
    latest_mid = (latest_bid + latest_ask) / 2.0
spread_pips = None
if latest_bid is not None and latest_ask is not None:
    spread_pips = (latest_ask - latest_bid) * 100.0

tick_age_sec = None
range_6m_pips = None
range_30m_pips = None
if latest_epoch is not None:
    tick_age_sec = max(0.0, time.time() - float(latest_epoch))
    mids_6m = [
        float(row["mid"])
        for row in tick_cache
        if isinstance(row, dict)
        and row.get("mid") is not None
        and latest_epoch - float(row.get("epoch", 0.0)) <= 360.0
    ]
    mids_30m = [
        float(row["mid"])
        for row in tick_cache
        if isinstance(row, dict)
        and row.get("mid") is not None
        and latest_epoch - float(row.get("epoch", 0.0)) <= 1800.0
    ]
    if len(mids_6m) >= 2:
        range_6m_pips = (max(mids_6m) - min(mids_6m)) * 100.0
    if len(mids_30m) >= 2:
        range_30m_pips = (max(mids_30m) - min(mids_30m)) * 100.0

m1 = factor_cache.get("M1") if isinstance(factor_cache.get("M1"), dict) else {}
m1_atr_pips = m1.get("atr_pips")
if m1_atr_pips is None and m1.get("atr") is not None:
    try:
        m1_atr_pips = float(m1["atr"]) * 100.0
    except Exception:
        m1_atr_pips = None

orders_db = root / "logs" / "orders.db"
fills_15m = 0
fills_30m = 0
rejects_30m = 0
if orders_db.exists():
    conn = sqlite3.connect(orders_db)
    cur = conn.cursor()
    time_expr = "julianday(replace(substr(ts,1,19),'T',' '))"
    cur.execute(
        f"select count(*) from orders where {time_expr} >= julianday('now','-15 minutes') and lower(status)='filled'"
    )
    fills_15m = int(cur.fetchone()[0] or 0)
    cur.execute(
        f"select count(*) from orders where {time_expr} >= julianday('now','-30 minutes') and lower(status)='filled'"
    )
    fills_30m = int(cur.fetchone()[0] or 0)
    cur.execute(
        f"""
        select count(*)
        from orders
        where {time_expr} >= julianday('now','-30 minutes')
          and (
            lower(status) like '%reject%'
            or lower(status) in ('rejected','failed','error','cancelled')
          )
        """
    )
    rejects_30m = int(cur.fetchone()[0] or 0)
    conn.close()

account_data = account.get("data") if isinstance(account.get("data"), dict) else {}
margin_used = account_data.get("margin_used")
free_margin_ratio = account_data.get("free_margin_ratio")
health_buffer = account_data.get("health_buffer")

mechanism_ok = None
if isinstance(health.get("mechanism_integrity"), dict):
    mechanism_ok = health["mechanism_integrity"].get("ok")

order_status_rows = health.get("orders_status_1h") or []
top_status = ", ".join(
    f"{row.get('status')}={row.get('count')}"
    for row in order_status_rows[:5]
    if isinstance(row, dict)
)

warnings: list[str] = []
if mechanism_ok is False:
    warnings.append("mechanism_integrity_fail")
if spread_pips is not None and spread_pips > 1.2:
    warnings.append(f"spread_wide:{spread_pips:.2f}p")
if tick_age_sec is not None and tick_age_sec > 30.0:
    warnings.append(f"tick_stale:{tick_age_sec:.1f}s")
if health.get("data_lag_ms") is not None and float(health["data_lag_ms"]) > 1500.0:
    warnings.append(f"data_lag_high:{float(health['data_lag_ms']):.1f}ms")
if fills_15m == 0:
    warnings.append(f"low_activity_15m:fills_15m={fills_15m}")
if fills_30m <= 1:
    warnings.append(f"low_activity_30m:fills_30m={fills_30m}")
if rejects_30m >= max(12, fills_30m * 3):
    warnings.append(f"reject_pressure_high:rejects_30m={rejects_30m}")

print(
    f"generated_at={health.get('generated_at', 'n/a')} "
    f"mechanism_integrity={fmt(mechanism_ok)}"
)
print(
    f"bid={fmt(latest_bid, 3)} ask={fmt(latest_ask, 3)} mid={fmt(latest_mid, 3)} "
    f"spread_pips={fmt(spread_pips, 2)} tick_age_sec={fmt(tick_age_sec, 1)}"
)
print(
    f"m1_atr14_pips={fmt(m1_atr_pips, 2)} "
    f"range_6m_pips={fmt(range_6m_pips, 2)} "
    f"range_30m_pips={fmt(range_30m_pips, 2)}"
)
print(
    f"data_lag_ms={fmt(health.get('data_lag_ms'), 1)} "
    f"decision_latency_ms={fmt(health.get('decision_latency_ms'), 1)}"
)
print(
    f"fills_15m={fills_15m} fills_30m={fills_30m} rejects_30m={rejects_30m}"
)
print(
    f"margin_used={fmt(margin_used, 1)} "
    f"free_margin_ratio={fmt(free_margin_ratio, 3)} "
    f"health_buffer={fmt(health_buffer, 3)}"
)
print(f"orders_status_1h={top_status or 'n/a'}")
print(f"preflight_status={'warn' if warnings else 'ok'}")
if warnings:
    print("warnings=" + ", ".join(warnings))
PY
echo

echo "== TRADE_FINDINGS Review =="
python3 "${ROOT_DIR}/scripts/trade_findings_review.py" --query "${QUERY}" --limit "${LIMIT}"
