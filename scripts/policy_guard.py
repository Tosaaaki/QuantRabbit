#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from analytics.policy_apply import load_policy_snapshot, save_policy_snapshot
from analytics.policy_ledger import PolicyLedger
from analytics.policy_diff import utc_now_iso
from utils.market_hours import is_market_open


def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    if q <= 0:
        return float(min(values))
    if q >= 1:
        return float(max(values))
    data = sorted(values)
    n = len(data)
    if n == 1:
        return float(data[0])
    k = (n - 1) * q
    f = int(k)
    c = min(f + 1, n - 1)
    if f == c:
        return float(data[int(k)])
    d0 = data[f] * (c - k)
    d1 = data[c] * (k - f)
    return float(d0 + d1)


def _load_metric_values(db_path: Path, metric: str, since_iso: str) -> List[float]:
    if not db_path.exists():
        return []
    try:
        con = sqlite3.connect(db_path)
        cur = con.execute(
            "SELECT value FROM metrics WHERE metric = ? AND ts >= ?",
            (metric, since_iso),
        )
        values = []
        for (val,) in cur.fetchall():
            try:
                values.append(float(val))
            except Exception:
                continue
        con.close()
        return values
    except Exception:
        return []


def _load_metric_latest_ts(db_path: Path, metric: str) -> Optional[datetime]:
    if not db_path.exists():
        return None
    try:
        con = sqlite3.connect(db_path)
        cur = con.execute(
            "SELECT ts FROM metrics WHERE metric = ? ORDER BY ts DESC LIMIT 1",
            (metric,),
        )
        row = cur.fetchone()
        con.close()
        if not row or not row[0]:
            return None
        raw = str(row[0]).strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _reject_streak(orders_db: Path, limit: int = 5) -> int:
    if not orders_db.exists():
        return 0
    try:
        con = sqlite3.connect(orders_db)
        cur = con.execute(
            "SELECT status, error_code FROM orders ORDER BY ts DESC LIMIT ?",
            (int(limit),),
        )
        streak = 0
        for status, error_code in cur.fetchall():
            status = str(status or "").upper()
            failed = status in {"REJECTED", "FAILED", "ERROR", "CANCELLED"} or bool(error_code)
            if not failed:
                break
            streak += 1
        con.close()
        return streak
    except Exception:
        return 0


def _load_applied_at(snapshot: Dict[str, object]) -> Optional[datetime]:
    notes = snapshot.get("notes")
    if not isinstance(notes, dict):
        return None
    raw = notes.get("policy_applied_at")
    if not raw:
        return None
    text = str(raw).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def evaluate_slo(
    *,
    metrics_db: Path,
    orders_db: Path,
    lookback_min: int,
) -> Dict[str, object]:
    since = (datetime.now(timezone.utc) - timedelta(minutes=lookback_min)).isoformat()
    decision_latency_values = _load_metric_values(metrics_db, "decision_latency_ms", since)
    data_lag_values = _load_metric_values(metrics_db, "data_lag_ms", since)
    decision_latency = _percentile(decision_latency_values, 0.95)
    data_lag = _percentile(data_lag_values, 0.95)
    decision_latest_ts = _load_metric_latest_ts(metrics_db, "decision_latency_ms")
    data_lag_latest_ts = _load_metric_latest_ts(metrics_db, "data_lag_ms")
    drawdown = _load_metric_values(metrics_db, "drawdown_pct", since)
    drawdown_max = max(drawdown) if drawdown else None
    order_success = _load_metric_values(metrics_db, "order_success_rate", since)
    order_success_min = min(order_success) if order_success else None
    reject_rate = _load_metric_values(metrics_db, "reject_rate", since)
    reject_rate_max = max(reject_rate) if reject_rate else None
    gpt_timeout = _load_metric_values(metrics_db, "gpt_timeout_rate", since)
    gpt_timeout_max = max(gpt_timeout) if gpt_timeout else None
    reject_streak = _reject_streak(orders_db)
    return {
        "decision_latency_p95": decision_latency,
        "data_lag_p95": data_lag,
        "decision_latency_count": len(decision_latency_values),
        "data_lag_count": len(data_lag_values),
        "decision_latency_latest_ts": decision_latest_ts.isoformat() if decision_latest_ts else None,
        "data_lag_latest_ts": data_lag_latest_ts.isoformat() if data_lag_latest_ts else None,
        "drawdown_pct_max": drawdown_max,
        "order_success_min": order_success_min,
        "reject_rate_max": reject_rate_max,
        "gpt_timeout_rate_max": gpt_timeout_max,
        "reject_streak": reject_streak,
    }


def _parse_metric_ts(raw: object) -> Optional[datetime]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def collect_violations(
    metrics: Dict[str, object],
    *,
    max_decision_ms: float,
    max_data_lag_ms: float,
    max_drawdown_pct: float,
    min_order_success: float,
    max_reject_rate: float,
    max_gpt_timeout: float,
    reject_streak_threshold: int,
    require_slo_metrics_when_open: bool,
    slo_metrics_max_stale_sec: float,
    market_open: bool,
    now_utc: Optional[datetime] = None,
) -> List[str]:
    violations: List[str] = []
    if metrics.get("decision_latency_p95") and metrics["decision_latency_p95"] > max_decision_ms:
        violations.append("decision_latency_p95")
    if metrics.get("data_lag_p95") and metrics["data_lag_p95"] > max_data_lag_ms:
        violations.append("data_lag_p95")
    if metrics.get("drawdown_pct_max") and metrics["drawdown_pct_max"] > max_drawdown_pct:
        violations.append("drawdown_pct_max")
    if metrics.get("order_success_min") is not None and metrics["order_success_min"] < min_order_success:
        violations.append("order_success_rate")
    if metrics.get("reject_rate_max") and metrics["reject_rate_max"] > max_reject_rate:
        violations.append("reject_rate_max")
    if metrics.get("gpt_timeout_rate_max") and metrics["gpt_timeout_rate_max"] > max_gpt_timeout:
        violations.append("gpt_timeout_rate_max")
    if metrics.get("reject_streak", 0) >= reject_streak_threshold:
        violations.append("reject_streak")

    if require_slo_metrics_when_open and market_open:
        if now_utc is None:
            now_utc = datetime.now(timezone.utc)
        for prefix in ("decision_latency", "data_lag"):
            count = int(metrics.get(f"{prefix}_count") or 0)
            latest_dt = _parse_metric_ts(metrics.get(f"{prefix}_latest_ts"))
            if count <= 0:
                violations.append(f"{prefix}_missing")
                continue
            if latest_dt is None:
                violations.append(f"{prefix}_missing")
                continue
            age_sec = max(0.0, (now_utc - latest_dt).total_seconds())
            if age_sec > slo_metrics_max_stale_sec:
                violations.append(f"{prefix}_stale")
    return violations


def main() -> None:
    ap = argparse.ArgumentParser(description="Policy guard and auto rollback.")
    ap.add_argument("--metrics-db", default=os.getenv("POLICY_GUARD_METRICS_DB", "logs/metrics.db"))
    ap.add_argument("--orders-db", default=os.getenv("POLICY_GUARD_ORDERS_DB", "logs/orders.db"))
    ap.add_argument("--overlay-path", default=os.getenv("POLICY_OVERLAY_PATH", "logs/policy_overlay.json"))
    ap.add_argument("--stable-path", default=os.getenv("POLICY_GUARD_STABLE_PATH", "logs/policy_stable.json"))
    ap.add_argument("--latest-path", default=os.getenv("POLICY_LATEST_PATH", "logs/policy_latest.json"))
    ap.add_argument("--lookback-min", type=int, default=int(os.getenv("POLICY_GUARD_LOOKBACK_MIN", "120")))
    ap.add_argument("--stable-min-sec", type=int, default=int(os.getenv("POLICY_GUARD_STABLE_MIN_SEC", "1800")))
    ap.add_argument("--max-decision-ms", type=float, default=float(os.getenv("POLICY_GUARD_MAX_DECISION_MS", "2000")))
    # Data lag p95 default: align with the "stale" guardrail (3000ms) unless overridden explicitly.
    ap.add_argument("--max-data-lag-ms", type=float, default=float(os.getenv("POLICY_GUARD_MAX_DATA_LAG_MS", "3000")))
    ap.add_argument("--max-drawdown-pct", type=float, default=float(os.getenv("POLICY_GUARD_MAX_DRAWDOWN_PCT", "0.18")))
    ap.add_argument("--min-order-success", type=float, default=float(os.getenv("POLICY_GUARD_MIN_ORDER_SUCCESS", "0.995")))
    ap.add_argument("--max-reject-rate", type=float, default=float(os.getenv("POLICY_GUARD_MAX_REJECT_RATE", "0.01")))
    ap.add_argument("--max-gpt-timeout", type=float, default=float(os.getenv("POLICY_GUARD_MAX_GPT_TIMEOUT", "0.05")))
    ap.add_argument("--reject-streak", type=int, default=int(os.getenv("POLICY_GUARD_REJECT_STREAK", "3")))
    ap.add_argument(
        "--require-slo-metrics-when-open",
        type=int,
        default=int(os.getenv("POLICY_GUARD_REQUIRE_SLO_METRICS_WHEN_OPEN", "1")),
    )
    ap.add_argument(
        "--slo-metrics-max-stale-sec",
        type=float,
        default=float(os.getenv("POLICY_GUARD_SLO_METRICS_MAX_STALE_SEC", "900")),
    )
    ap.add_argument("--no-ledger", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    metrics = evaluate_slo(
        metrics_db=Path(args.metrics_db),
        orders_db=Path(args.orders_db),
        lookback_min=args.lookback_min,
    )
    violations = collect_violations(
        metrics,
        max_decision_ms=args.max_decision_ms,
        max_data_lag_ms=args.max_data_lag_ms,
        max_drawdown_pct=args.max_drawdown_pct,
        min_order_success=args.min_order_success,
        max_reject_rate=args.max_reject_rate,
        max_gpt_timeout=args.max_gpt_timeout,
        reject_streak_threshold=args.reject_streak,
        require_slo_metrics_when_open=bool(args.require_slo_metrics_when_open),
        slo_metrics_max_stale_sec=max(1.0, float(args.slo_metrics_max_stale_sec)),
        market_open=is_market_open(),
    )

    overlay_path = Path(args.overlay_path)
    stable_path = Path(args.stable_path)
    latest_path = Path(args.latest_path)

    ledger = None if args.no_ledger else PolicyLedger()

    if violations:
        if not stable_path.exists():
            logging.warning("[POLICY_GUARD] violations but no stable policy: %s", violations)
            return
        stable_snapshot = load_policy_snapshot(stable_path)
        save_policy_snapshot(overlay_path, stable_snapshot)
        save_policy_snapshot(latest_path, stable_snapshot)
        policy_id = (
            stable_snapshot.get("policy_id")
            or (stable_snapshot.get("notes", {}).get("policy_id") if isinstance(stable_snapshot.get("notes"), dict) else None)
            or "stable"
        )
        payload = {
            "policy_id": policy_id,
            "generated_at": utc_now_iso(),
            "source": "policy_guard",
            "no_change": True,
            "reason": "rollback",
            "notes": {"violations": violations, "metrics": metrics},
        }
        logging.warning("[POLICY_GUARD] rollback applied: %s", ", ".join(violations))
        if ledger:
            ledger.record(payload, status="rollback", summary={"violations": violations, "metrics": metrics})
        return

    if overlay_path.exists():
        current = load_policy_snapshot(overlay_path)
        applied_at = _load_applied_at(current)
        if applied_at is None:
            applied_at = datetime.now(timezone.utc) - timedelta(seconds=args.stable_min_sec)
        age = (datetime.now(timezone.utc) - applied_at).total_seconds()
        if age >= args.stable_min_sec:
            save_policy_snapshot(stable_path, current)
            logging.info("[POLICY_GUARD] stable snapshot updated.")
            if ledger:
                policy_id = (
                    current.get("policy_id")
                    or (current.get("notes", {}).get("policy_id") if isinstance(current.get("notes"), dict) else None)
                    or "stable"
                )
                ledger.record(
                    {
                        "policy_id": policy_id,
                        "generated_at": utc_now_iso(),
                        "source": "policy_guard",
                        "no_change": True,
                        "reason": "mark_stable",
                        "notes": {"metrics": metrics},
                    },
                    status="stable",
                    summary={"metrics": metrics},
                )
    logging.info("[POLICY_GUARD] ok")


if __name__ == "__main__":
    main()
