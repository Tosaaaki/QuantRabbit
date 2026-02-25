#!/usr/bin/env python3
"""Canary guard for scalp_ping_5s_d live tuning.

Evaluates recent live results from trades/orders DB and returns a decision:
  - promote: move units up (default -> 22000)
  - hold: keep current units
  - rollback: move units down (default -> 15000)

This script does not require market-open state. It is a pure windowed check.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class TradeMetrics:
    trade_count: int
    realized_pl_jpy: float
    realized_pl_pips: float
    trades_per_hour: float
    jpy_per_hour: float
    first_trade_time: str | None
    last_trade_time: str | None


@dataclass
class OrderMetrics:
    total_orders: int
    filled_count: int
    submit_attempt_count: int
    entry_probability_reject_count: int
    perf_block_count: int
    hard_reject_count: int
    margin_reject_count: int


@dataclass
class GuardDecision:
    decision: str
    reasons: list[str]
    target_units: int
    applied: bool
    env_updated: bool


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if math.isnan(out) or math.isinf(out):
        return float(default)
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _hours(window_minutes: int) -> float:
    return max(1.0 / 60.0, float(window_minutes) / 60.0)


def _read_trade_metrics(
    db_path: Path,
    *,
    strategy_tag: str,
    window_minutes: int,
) -> TradeMetrics:
    if not db_path.exists():
        return TradeMetrics(0, 0.0, 0.0, 0.0, 0.0, None, None)
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        row = con.execute(
            """
            SELECT
              COUNT(*) AS n,
              COALESCE(SUM(realized_pl), 0.0) AS sum_jpy,
              COALESCE(SUM(pl_pips), 0.0) AS sum_pips,
              MIN(COALESCE(close_time, entry_time)) AS first_ts,
              MAX(COALESCE(close_time, entry_time)) AS last_ts
            FROM trades
            WHERE strategy_tag = ?
              AND datetime(substr(COALESCE(close_time, entry_time), 1, 19)) >= datetime('now', ?)
            """,
            (strategy_tag, f"-{int(window_minutes)} minutes"),
        ).fetchone()
    finally:
        con.close()

    n = _safe_int(row["n"] if row else 0, 0)
    sum_jpy = _safe_float(row["sum_jpy"] if row else 0.0, 0.0)
    sum_pips = _safe_float(row["sum_pips"] if row else 0.0, 0.0)
    h = _hours(window_minutes)
    return TradeMetrics(
        trade_count=n,
        realized_pl_jpy=sum_jpy,
        realized_pl_pips=sum_pips,
        trades_per_hour=n / h,
        jpy_per_hour=sum_jpy / h,
        first_trade_time=str(row["first_ts"]) if row and row["first_ts"] else None,
        last_trade_time=str(row["last_ts"]) if row and row["last_ts"] else None,
    )


def _read_order_metrics(
    db_path: Path,
    *,
    strategy_tag: str,
    window_minutes: int,
) -> OrderMetrics:
    if not db_path.exists():
        return OrderMetrics(0, 0, 0, 0, 0, 0, 0)
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        row = con.execute(
            """
            WITH recent AS (
              SELECT
                status,
                COALESCE(error_code, '') AS ec,
                COALESCE(error_message, '') AS em,
                COALESCE(
                  json_extract(request_json, '$.entry_thesis.strategy_tag'),
                  json_extract(request_json, '$.strategy_tag'),
                  ''
                ) AS tag
              FROM orders
              WHERE datetime(substr(ts, 1, 19)) >= datetime('now', ?)
            )
            SELECT
              COUNT(*) AS total_n,
              SUM(CASE WHEN lower(status) = 'filled' THEN 1 ELSE 0 END) AS filled_n,
              SUM(CASE WHEN lower(status) = 'submit_attempt' THEN 1 ELSE 0 END) AS submit_n,
              SUM(CASE WHEN lower(status) = 'entry_probability_reject' THEN 1 ELSE 0 END) AS prob_reject_n,
              SUM(CASE WHEN lower(status) = 'perf_block' THEN 1 ELSE 0 END) AS perf_block_n,
              SUM(
                CASE
                  WHEN lower(status) IN ('rejected', 'failed', 'error', 'cancelled') THEN 1
                  ELSE 0
                END
              ) AS hard_reject_n,
              SUM(
                CASE
                  WHEN (
                    lower(status || ' ' || ec || ' ' || em) LIKE '%margin%'
                    OR lower(status || ' ' || ec || ' ' || em) LIKE '%closeout%'
                    OR lower(status || ' ' || ec || ' ' || em) LIKE '%insufficient%'
                  ) THEN 1
                  ELSE 0
                END
              ) AS margin_reject_n
            FROM recent
            WHERE tag = ?
            """,
            (f"-{int(window_minutes)} minutes", strategy_tag),
        ).fetchone()
    finally:
        con.close()

    return OrderMetrics(
        total_orders=_safe_int(row["total_n"] if row else 0, 0),
        filled_count=_safe_int(row["filled_n"] if row else 0, 0),
        submit_attempt_count=_safe_int(row["submit_n"] if row else 0, 0),
        entry_probability_reject_count=_safe_int(row["prob_reject_n"] if row else 0, 0),
        perf_block_count=_safe_int(row["perf_block_n"] if row else 0, 0),
        hard_reject_count=_safe_int(row["hard_reject_n"] if row else 0, 0),
        margin_reject_count=_safe_int(row["margin_reject_n"] if row else 0, 0),
    )


def _read_current_units(env_file: Path) -> int:
    if not env_file.exists():
        return 0
    text = env_file.read_text(encoding="utf-8")
    m = re.search(r"(?m)^SCALP_PING_5S_D_BASE_ENTRY_UNITS=(\d+)\s*$", text)
    if not m:
        return 0
    return _safe_int(m.group(1), 0)


def _update_units(env_file: Path, *, units: int) -> bool:
    text = env_file.read_text(encoding="utf-8")
    original = text
    repl_base = f"SCALP_PING_5S_D_BASE_ENTRY_UNITS={int(units)}"
    repl_max = f"SCALP_PING_5S_D_MAX_UNITS={int(units)}"

    if re.search(r"(?m)^SCALP_PING_5S_D_BASE_ENTRY_UNITS=\d+\s*$", text):
        text = re.sub(
            r"(?m)^SCALP_PING_5S_D_BASE_ENTRY_UNITS=\d+\s*$",
            repl_base,
            text,
        )
    else:
        text = text.rstrip("\n") + "\n" + repl_base + "\n"

    if re.search(r"(?m)^SCALP_PING_5S_D_MAX_UNITS=\d+\s*$", text):
        text = re.sub(
            r"(?m)^SCALP_PING_5S_D_MAX_UNITS=\d+\s*$",
            repl_max,
            text,
        )
    else:
        text = text.rstrip("\n") + "\n" + repl_max + "\n"

    if text == original:
        return False
    env_file.write_text(text, encoding="utf-8")
    return True


def _decide(
    *,
    trade_metrics: TradeMetrics,
    order_metrics: OrderMetrics,
    current_units: int,
    min_jpy_per_hour: float,
    min_trades_per_hour: float,
    min_observed_trades: int,
    max_margin_reject: int,
    rollback_jpy_per_hour: float,
    promote_units: int,
    rollback_units: int,
) -> GuardDecision:
    reasons: list[str] = []
    decision = "hold"
    target_units = current_units if current_units > 0 else rollback_units

    if trade_metrics.trade_count < max(0, min_observed_trades):
        reasons.append(
            f"insufficient_trade_samples:{trade_metrics.trade_count}<{min_observed_trades}"
        )

    if order_metrics.margin_reject_count > max_margin_reject:
        reasons.append(
            f"margin_reject_count_exceeded:{order_metrics.margin_reject_count}>{max_margin_reject}"
        )
        decision = "rollback"
        target_units = int(rollback_units)
    elif trade_metrics.jpy_per_hour <= rollback_jpy_per_hour:
        reasons.append(
            f"jpy_per_hour_below_rollback:{trade_metrics.jpy_per_hour:.2f}<={rollback_jpy_per_hour:.2f}"
        )
        decision = "rollback"
        target_units = int(rollback_units)
    else:
        pass_jpy = trade_metrics.jpy_per_hour > min_jpy_per_hour
        pass_freq = trade_metrics.trades_per_hour >= min_trades_per_hour
        pass_margin = order_metrics.margin_reject_count <= max_margin_reject
        if pass_jpy and pass_freq and pass_margin and trade_metrics.trade_count >= min_observed_trades:
            reasons.append("all_pass")
            decision = "promote"
            target_units = int(promote_units)
        else:
            if not pass_jpy:
                reasons.append(
                    f"jpy_per_hour_below_target:{trade_metrics.jpy_per_hour:.2f}<={min_jpy_per_hour:.2f}"
                )
            if not pass_freq:
                reasons.append(
                    f"trades_per_hour_below_target:{trade_metrics.trades_per_hour:.2f}<{min_trades_per_hour:.2f}"
                )
            if not pass_margin:
                reasons.append(
                    f"margin_reject_count_exceeded:{order_metrics.margin_reject_count}>{max_margin_reject}"
                )

    if not reasons:
        reasons.append("no_rule_matched")

    return GuardDecision(
        decision=decision,
        reasons=reasons,
        target_units=target_units,
        applied=False,
        env_updated=False,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Canary guard for scalp_ping_5s_d.")
    ap.add_argument("--trades-db", type=Path, default=Path("logs/trades.db"))
    ap.add_argument("--orders-db", type=Path, default=Path("logs/orders.db"))
    ap.add_argument("--env-file", type=Path, default=Path("ops/env/scalp_ping_5s_d.env"))
    ap.add_argument("--strategy-tag", default="scalp_ping_5s_d_live")
    ap.add_argument("--window-minutes", type=int, default=120)
    ap.add_argument("--min-jpy-per-hour", type=float, default=0.0)
    ap.add_argument("--min-trades-per-hour", type=float, default=6.0)
    ap.add_argument("--min-observed-trades", type=int, default=6)
    ap.add_argument("--max-margin-reject", type=int, default=0)
    ap.add_argument("--rollback-jpy-per-hour", type=float, default=-200.0)
    ap.add_argument("--promote-units", type=int, default=22000)
    ap.add_argument("--rollback-units", type=int, default=15000)
    ap.add_argument("--apply", action="store_true", help="Apply units decision to env file.")
    ap.add_argument(
        "--decision-exit-codes",
        action="store_true",
        help="Return 10 on promote, 20 on rollback, 0 on hold.",
    )
    ap.add_argument("--out", type=Path, default=Path("logs/ping5s_d_canary_guard_latest.json"))
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    now_utc = datetime.now(timezone.utc).isoformat()
    current_units = _read_current_units(args.env_file)

    trade_metrics = _read_trade_metrics(
        args.trades_db,
        strategy_tag=str(args.strategy_tag),
        window_minutes=max(1, int(args.window_minutes)),
    )
    order_metrics = _read_order_metrics(
        args.orders_db,
        strategy_tag=str(args.strategy_tag),
        window_minutes=max(1, int(args.window_minutes)),
    )

    decision = _decide(
        trade_metrics=trade_metrics,
        order_metrics=order_metrics,
        current_units=current_units,
        min_jpy_per_hour=float(args.min_jpy_per_hour),
        min_trades_per_hour=float(args.min_trades_per_hour),
        min_observed_trades=int(args.min_observed_trades),
        max_margin_reject=int(args.max_margin_reject),
        rollback_jpy_per_hour=float(args.rollback_jpy_per_hour),
        promote_units=int(args.promote_units),
        rollback_units=int(args.rollback_units),
    )

    if args.apply and decision.decision in {"promote", "rollback"}:
        if args.env_file.exists():
            decision.env_updated = _update_units(
                args.env_file,
                units=int(decision.target_units),
            )
        decision.applied = True

    payload = {
        "meta": {
            "generated_at_utc": now_utc,
            "strategy_tag": str(args.strategy_tag),
            "window_minutes": int(args.window_minutes),
            "trades_db": str(args.trades_db),
            "orders_db": str(args.orders_db),
            "env_file": str(args.env_file),
            "thresholds": {
                "min_jpy_per_hour": float(args.min_jpy_per_hour),
                "min_trades_per_hour": float(args.min_trades_per_hour),
                "min_observed_trades": int(args.min_observed_trades),
                "max_margin_reject": int(args.max_margin_reject),
                "rollback_jpy_per_hour": float(args.rollback_jpy_per_hour),
                "promote_units": int(args.promote_units),
                "rollback_units": int(args.rollback_units),
            },
        },
        "metrics": {
            "trade": asdict(trade_metrics),
            "order": asdict(order_metrics),
            "current_units": int(current_units),
        },
        "decision": asdict(decision),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.decision_exit_codes:
        if decision.decision == "promote":
            return 10
        if decision.decision == "rollback":
            return 20
    return 0


if __name__ == "__main__":
    sys.exit(main())
