"""Compare target cadence policies against 2025 QuantRabbit precedent.

Read-only analysis tool. It writes local JSON/Markdown artifacts and never
calls broker write paths, order gateways, cancellation paths, or position
management code.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANUAL_HISTORY = ROOT / "data" / "manual_history_2025_mining.json"
DEFAULT_OPERATOR_PRECEDENT = ROOT / "data" / "operator_precedent_audit.json"
DEFAULT_MANUAL_CONTEXT = ROOT / "data" / "manual_market_context_audit.json"
DEFAULT_DAILY_TARGET_STATE = ROOT / "data" / "daily_target_state.json"
DEFAULT_EXECUTION_LEDGER = ROOT / "data" / "execution_ledger.db"
DEFAULT_TARGET_JSON = ROOT / "data" / "target_cadence_analysis.json"
DEFAULT_TARGET_REPORT = ROOT / "docs" / "target_cadence_analysis_report.md"
DEFAULT_SHAPE_JSON = ROOT / "data" / "trade_shape_precedent.json"
DEFAULT_SHAPE_REPORT = ROOT / "docs" / "trade_shape_precedent_report.md"

CALENDAR_DAYS = 30
TRADING_DAYS_PER_30D = 22
DAILY_PACE_RETURN = 0.05
GOOD_MARKET_RETURN = 0.10
ROLLING_30D_MULTIPLIER = 4.0


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_json(path: Path, *, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _parse_time(value: object) -> datetime | None:
    if not value:
        return None
    text = str(value)
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        if "." in text:
            head, tail = text.split(".", 1)
            zone = "+00:00" if "+" not in tail else "+" + tail.split("+", 1)[1]
            frac = tail.split("+", 1)[0][:6]
            text = f"{head}.{frac}{zone}"
        return datetime.fromisoformat(text).astimezone(timezone.utc)
    except ValueError:
        return None


def _round(value: float | int | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return round(float(value), digits)


def _pct(value: float | int | None, digits: int = 2) -> float | None:
    if value is None:
        return None
    return _round(float(value) * 100.0, digits)


def _session_jst_from_open_time(value: object) -> str:
    dt = _parse_time(value)
    if not dt:
        return "UNKNOWN"
    hour = dt.hour
    if 0 <= hour < 6:
        return "TOKYO"
    if 6 <= hour < 12:
        return "LONDON_AM"
    if 12 <= hour < 18:
        return "NY_OVERLAP"
    return "OFF_HOURS"


def _side_from_units(units: object) -> str:
    try:
        return "LONG" if float(units) > 0 else "SHORT"
    except (TypeError, ValueError):
        return "UNKNOWN"


def _profit_factor(rows: list[dict[str, Any]]) -> float | None:
    wins = sum(float(row.get("realized_pl") or 0.0) for row in rows if float(row.get("realized_pl") or 0.0) > 0)
    losses = sum(float(row.get("realized_pl") or 0.0) for row in rows if float(row.get("realized_pl") or 0.0) < 0)
    if losses == 0:
        return None
    return round(wins / abs(losses), 4)


def _stats_from_breakdown(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(name): value for name, value in sorted(rows.items())} if isinstance(rows, dict) else {}


def _dominant(values: list[str]) -> str | None:
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def _daily_return_table(manual_history: dict[str, Any]) -> list[dict[str, Any]]:
    analysis = manual_history.get("analysis") or {}
    daily_pl = dict(analysis.get("daily_pl") or {})
    trades = list(manual_history.get("trades") or [])
    cash = analysis.get("cash_flows") or {}
    initial_balance = float(cash.get("initial_balance") or (analysis.get("balance_start") or {}).get("balance") or 0.0)

    initial_time = _parse_time((analysis.get("balance_start") or {}).get("time"))
    transfers_by_day: dict[str, float] = defaultdict(float)
    for transfer in cash.get("transfers") or []:
        amount = float(transfer.get("amount") or 0.0)
        ts = _parse_time(transfer.get("time"))
        if not ts:
            continue
        if initial_time and ts == initial_time:
            continue
        transfers_by_day[ts.date().isoformat()] += amount

    trades_by_close_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        day = str(trade.get("close_time") or "")[:10]
        if day:
            trades_by_close_day[day].append(trade)

    equity = initial_balance
    cumulative_pl = 0.0
    rows: list[dict[str, Any]] = []
    for day in sorted(daily_pl):
        day_start = equity
        realized = float(daily_pl[day] or 0.0)
        rows_for_day = trades_by_close_day.get(day, [])
        wins = [row for row in rows_for_day if float(row.get("realized_pl") or 0.0) > 0]
        losses = [row for row in rows_for_day if float(row.get("realized_pl") or 0.0) < 0]
        largest_win = max(wins, key=lambda row: float(row.get("realized_pl") or 0.0), default=None)
        largest_loss = min(losses, key=lambda row: float(row.get("realized_pl") or 0.0), default=None)
        daily_return = realized / day_start if day_start else None
        cumulative_pl += realized
        cumulative_return = cumulative_pl / initial_balance if initial_balance else None
        rows.append(
            {
                "utc_day": day,
                "realized_pl_jpy": _round(realized, 4),
                "estimated_day_start_equity_jpy": _round(day_start, 4),
                "intraday_cash_flow_jpy": _round(transfers_by_day.get(day, 0.0), 4),
                "daily_return_pct": _pct(daily_return),
                "cumulative_funding_adjusted_return_pct": _pct(cumulative_return),
                "hit_5pct": bool(daily_return is not None and daily_return >= DAILY_PACE_RETURN),
                "hit_10pct": bool(daily_return is not None and daily_return >= GOOD_MARKET_RETURN),
                "red": realized < 0,
                "largest_win_trade": _trade_brief(largest_win),
                "largest_loss_trade": _trade_brief(largest_loss),
                "dominant_pair": _dominant([str(row.get("pair") or "UNKNOWN") for row in rows_for_day]),
                "dominant_side": _dominant([_side_from_units(row.get("units")) for row in rows_for_day]),
                "dominant_session_jst": _dominant([_session_jst_from_open_time(row.get("open_time")) for row in rows_for_day]),
                "exit_count": len(rows_for_day),
            }
        )
        equity += realized + transfers_by_day.get(day, 0.0)
    return rows


def _trade_brief(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    return {
        "trade_id": row.get("trade_id"),
        "pair": row.get("pair"),
        "side": _side_from_units(row.get("units")),
        "realized_pl_jpy": _round(float(row.get("realized_pl") or 0.0), 4),
        "close_reason": row.get("close_reason"),
        "open_time": row.get("open_time"),
        "close_time": row.get("close_time"),
    }


def _red_day_stats(daily_rows: list[dict[str, Any]]) -> dict[str, Any]:
    returns = [float(row["daily_return_pct"]) / 100.0 for row in daily_rows if row.get("daily_return_pct") is not None]
    red_returns = [value for value in returns if value < 0]
    if not red_returns:
        return {
            "active_days": len(returns),
            "red_days": 0,
            "median_red_return_pct": None,
            "average_red_return_pct": None,
        }
    sorted_red = sorted(red_returns)
    mid = len(sorted_red) // 2
    median = sorted_red[mid] if len(sorted_red) % 2 else (sorted_red[mid - 1] + sorted_red[mid]) / 2
    return {
        "active_days": len(returns),
        "red_days": len(red_returns),
        "green_days": len([value for value in returns if value > 0]),
        "hit_5pct_days": len([row for row in daily_rows if row.get("hit_5pct")]),
        "hit_10pct_days": len([row for row in daily_rows if row.get("hit_10pct")]),
        "median_red_return_pct": _pct(median),
        "average_red_return_pct": _pct(sum(red_returns) / len(red_returns)),
        "worst_daily_return_pct": _pct(min(returns)),
        "best_daily_return_pct": _pct(max(returns)),
    }


def _max_red_days(target_multiplier: float, *, red_return: float, days: int = CALENDAR_DAYS) -> dict[str, Any]:
    max_allowed = -1
    best_multiplier = None
    for red_days in range(days + 1):
        good_days = days - red_days
        multiplier = ((1.0 + GOOD_MARKET_RETURN) ** good_days) * ((1.0 + red_return) ** red_days)
        if multiplier >= target_multiplier:
            max_allowed = red_days
            best_multiplier = multiplier
    return {
        "assumed_good_day_return_pct": _pct(GOOD_MARKET_RETURN),
        "assumed_red_day_return_pct": _pct(red_return),
        "max_red_days": max_allowed if max_allowed >= 0 else 0,
        "good_days_required": days - max_allowed if max_allowed >= 0 else days + 1,
        "ending_multiplier_at_boundary": _round(best_multiplier, 4) if best_multiplier is not None else None,
    }


def _drawdown_sensitivity(target_multiplier: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for elapsed_days in (1, 10, 15):
        remaining = CALENDAR_DAYS - elapsed_days
        for drawdown in (0.05, 0.10, 0.20):
            required = (target_multiplier / (1.0 - drawdown)) ** (1.0 / remaining) - 1.0
            rows.append(
                {
                    "elapsed_days": elapsed_days,
                    "remaining_days": remaining,
                    "drawdown_pct": _pct(drawdown),
                    "required_remaining_daily_return_pct": _pct(required),
                }
            )
    return rows


def _target_models(daily_rows: list[dict[str, Any]]) -> dict[str, Any]:
    model_a_multiplier = (1.0 + DAILY_PACE_RETURN) ** CALENDAR_DAYS
    model_b_multiplier = ROLLING_30D_MULTIPLIER
    red_stats = _red_day_stats(daily_rows)
    empirical_red = (red_stats.get("median_red_return_pct") or -1.0) / 100.0
    if empirical_red >= 0:
        empirical_red = -0.01

    return {
        "assumptions": {
            "calendar_days": CALENDAR_DAYS,
            "trading_days_per_30d": TRADING_DAYS_PER_30D,
            "good_market_day_return_pct": _pct(GOOD_MARKET_RETURN),
            "daily_pace_marker_pct": _pct(DAILY_PACE_RETURN),
            "empirical_red_day_source": "median red day from reconstructed 2025 UTC daily table",
            "empirical_red_day_return_pct": _round(red_stats.get("median_red_return_pct"), 2),
        },
        "model_a_fixed_daily_5pct": {
            "description": "Fixed daily +5%, +10% on good markets, daily underperformance pressure.",
            "required_average_daily_return_pct": _pct(DAILY_PACE_RETURN),
            "required_average_trading_day_return_pct": _pct(model_a_multiplier ** (1.0 / TRADING_DAYS_PER_30D) - 1.0),
            "equivalent_30d_multiplier": _round(model_a_multiplier, 4),
            "drawdown_sensitivity": _drawdown_sensitivity(model_a_multiplier),
            "red_days_allowed_if_good_days_hit_10pct": {
                "policy_allowed": 0,
                "mathematical_flat_red_days_for_30d_multiplier": _max_red_days(model_a_multiplier, red_return=0.0),
                "mathematical_empirical_red_days_for_30d_multiplier": _max_red_days(model_a_multiplier, red_return=empirical_red),
            },
            "risk_of_forced_overtrading": "HIGH: a hard same-day +5% obligation turns normal red/no-edge days into incident pressure and can force churn after the edge is gone.",
        },
        "model_b_rolling_30d_4x": {
            "description": "Rolling 30-day 4x account growth, +5% as daily pace marker, +10% only on favorable days, no forced bad-day trading.",
            "required_average_daily_return_pct": _pct(model_b_multiplier ** (1.0 / CALENDAR_DAYS) - 1.0),
            "required_average_trading_day_return_pct": _pct(model_b_multiplier ** (1.0 / TRADING_DAYS_PER_30D) - 1.0),
            "equivalent_30d_multiplier": _round(model_b_multiplier, 4),
            "drawdown_sensitivity": _drawdown_sensitivity(model_b_multiplier),
            "red_days_allowed_if_good_days_hit_10pct": {
                "policy_allowed": "Allowed when no current edge; the rolling window must recover with later high-quality days.",
                "mathematical_flat_red_days_for_30d_multiplier": _max_red_days(model_b_multiplier, red_return=0.0),
                "mathematical_empirical_red_days_for_30d_multiplier": _max_red_days(model_b_multiplier, red_return=empirical_red),
            },
            "risk_of_forced_overtrading": "LOWER: the model keeps the 4x arithmetic target but lets the trader skip bad days and treat +5% as pace/diagnostic evidence instead of a forced execution trigger.",
        },
    }


def _best_30_window_daily(daily_rows: list[dict[str, Any]], manual_history: dict[str, Any]) -> dict[str, Any]:
    best = (((manual_history.get("analysis") or {}).get("cash_flows") or {}).get("best_30d_funding_adjusted") or {})
    start = _parse_time(best.get("start_time"))
    end = _parse_time(best.get("end_time"))
    if not start or not end:
        return {}
    rows = []
    for row in daily_rows:
        day_start = _parse_time(f"{row['utc_day']}T00:00:00Z")
        if day_start and start.date() <= day_start.date() <= end.date():
            rows.append(row)
    return {
        "start_time": best.get("start_time"),
        "end_time": best.get("end_time"),
        "funding_adjusted_profit_jpy": _round(best.get("profit"), 4),
        "funding_adjusted_return_pct": _round(best.get("return_pct"), 2),
        "equivalent_multiplier": _round(1.0 + float(best.get("return_pct") or 0.0) / 100.0, 4),
        "active_utc_days": len(rows),
        "red_utc_days": len([row for row in rows if row.get("red")]),
        "hit_5pct_utc_days": len([row for row in rows if row.get("hit_5pct")]),
        "hit_10pct_utc_days": len([row for row in rows if row.get("hit_10pct")]),
    }


def _origin_from_execution_event(row: sqlite3.Row) -> tuple[str, str]:
    raw_text = row["raw_json"] or "{}"
    lane_id = (row["lane_id"] or "").lower()
    client_order_id = (row["client_order_id"] or "").lower()
    exit_reason = (row["exit_reason"] or "").upper()
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError:
        raw = {}
    client_ids = []
    for key in ("clientOrderID", "clientTradeID"):
        if raw.get(key):
            client_ids.append(str(raw.get(key)))
    for item in raw.get("tradesClosed") or []:
        if isinstance(item, dict) and item.get("clientTradeID"):
            client_ids.append(str(item.get("clientTradeID")))
    client_blob = " ".join(client_ids).lower()
    evidence_blob = " ".join([lane_id, client_order_id, client_blob, raw_text.lower()])
    if "user_alpha" in evidence_blob or "operator_alpha" in evidence_blob or "reload" in evidence_blob:
        return "USER_ALPHA_RELOAD", "alpha/reload marker found in lane/client/raw evidence"
    if "qrv1-" in evidence_blob or lane_id:
        if exit_reason in {"MARKET_ORDER_TRADE_CLOSE", "MARKET_ORDER_POSITION_CLOSEOUT"}:
            return "SYSTEM_MANAGEMENT", "qrv1/system trade closed by management-style market close"
        return "SYSTEM_FRESH_ENTRY", "qrv1/system trade resolved through entry-attached or broker-native exit"
    if client_blob:
        return "UNKNOWN", "non-qrv1 client id present"
    return "OPERATOR_MANUAL", "no system client id or lane id on OANDA realized event"


def _execution_ledger_origin_breakdown(path: Path) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = {
        key: {"events": 0, "realized_pl_jpy": 0.0, "evidence": Counter()}
        for key in ("OPERATOR_MANUAL", "SYSTEM_FRESH_ENTRY", "SYSTEM_MANAGEMENT", "USER_ALPHA_RELOAD", "UNKNOWN")
    }
    if not path.exists():
        return {"available": False, "path": str(path), "buckets": buckets}
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            select ts_utc, source, event_type, lane_id, client_order_id, pair, side,
                   realized_pl_jpy, exit_reason, raw_json
              from execution_events
             where realized_pl_jpy is not null
               and abs(realized_pl_jpy) > 0
             order by ts_utc
            """
        ).fetchall()
    finally:
        conn.close()
    for row in rows:
        origin, reason = _origin_from_execution_event(row)
        bucket = buckets[origin]
        bucket["events"] += 1
        bucket["realized_pl_jpy"] += float(row["realized_pl_jpy"] or 0.0)
        bucket["evidence"][reason] += 1
    for bucket in buckets.values():
        bucket["realized_pl_jpy"] = _round(bucket["realized_pl_jpy"], 4)
        bucket["evidence"] = dict(bucket["evidence"].most_common())
    return {
        "available": True,
        "path": str(path),
        "classification_contract": {
            "operator_manual_not_system_profitability": True,
            "qrv1_or_lane_id_required_for_system_attribution": True,
            "system_management_is_reported_separately_from_fresh_entry_edge": True,
        },
        "buckets": buckets,
    }


def _origin_breakdown(manual_history: dict[str, Any], execution_ledger: Path) -> dict[str, Any]:
    manual_trades = list(manual_history.get("trades") or [])
    manual_pl = sum(float(row.get("realized_pl") or 0.0) for row in manual_trades)
    ledger = _execution_ledger_origin_breakdown(execution_ledger)
    return {
        "manual_2025": {
            "origin": "OPERATOR_MANUAL",
            "events": len(manual_trades),
            "realized_pl_jpy": _round(manual_pl, 4),
            "system_profitability_counted": False,
            "evidence": "data/manual_history_2025_mining.json is operator manual precedent / teacher data.",
        },
        "execution_ledger": ledger,
    }


def _precedent_2025(
    manual_history: dict[str, Any],
    manual_context: dict[str, Any],
    operator_precedent: dict[str, Any],
    daily_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    analysis = manual_history.get("analysis") or {}
    trades = list(manual_history.get("trades") or [])
    cash = analysis.get("cash_flows") or {}
    close_reason = analysis.get("by_close_reason") or {}
    position_building = manual_context.get("position_building_profile") or {}
    bounded_by_build = {
        row.get("bucket"): row for row in position_building.get("bounded_by_build_type") or []
    }
    raw_by_build = {row.get("bucket"): row for row in position_building.get("by_build_type") or []}
    excluded_tail = manual_context.get("excluded_tail_profile") or {}
    return {
        "period": {
            "configured_window": manual_history.get("window"),
            "active_window": ((operator_precedent.get("precedent") or {}).get("sample") or {}).get("active_window"),
        },
        "balances": {
            "start": analysis.get("balance_start"),
            "end": analysis.get("balance_end"),
            "peak": analysis.get("balance_peak"),
        },
        "deposits_withdrawals": cash.get("transfers") or [],
        "funding_adjusted": {
            "end_pl_jpy": _round(cash.get("transfer_adjusted_end_profit"), 4),
            "end_return_pct": _round(cash.get("transfer_adjusted_end_return_pct"), 2),
            "peak_pl_jpy": _round(cash.get("transfer_adjusted_peak_profit"), 4),
            "peak_return_pct": _round(cash.get("transfer_adjusted_peak_return_pct"), 2),
            "best_30d": cash.get("best_30d_funding_adjusted"),
            "best_30d_daily_evidence": _best_30_window_daily(daily_rows, manual_history),
        },
        "exit_count": manual_history.get("exit_events"),
        "total_realized_pl_jpy": _round((analysis.get("overall") or {}).get("net"), 4),
        "win_rate": (analysis.get("overall") or {}).get("win_rate"),
        "profit_factor": _profit_factor(trades),
        "best_day": analysis.get("best_day"),
        "worst_day": analysis.get("worst_day"),
        "pair_breakdown": _stats_from_breakdown(analysis.get("by_pair") or {}),
        "side_breakdown": _stats_from_breakdown(analysis.get("by_side") or {}),
        "session_breakdown": _stats_from_breakdown(analysis.get("by_session_jst") or {}),
        "exit_reason_breakdown": _stats_from_breakdown(close_reason),
        "margin_closeout_losses": close_reason.get("MARKET_ORDER_MARGIN_CLOSEOUT"),
        "long_hold_tail_risk": {
            "excluded_tail_overall": excluded_tail.get("overall"),
            "excluded_tail_by_hold_bucket": excluded_tail.get("by_hold_bucket"),
            "reason": ">=12h holds and margin closeouts are excluded from bounded replay because they represent unattended carry/margin-closeout tail risk.",
        },
        "with_move_pyramiding_pl": {
            "bounded_lt_12h_ex_margin_closeout": bounded_by_build.get("PYRAMID_WITH_MOVE"),
            "raw": raw_by_build.get("PYRAMID_WITH_MOVE"),
        },
        "bounded_adverse_add_pl": {
            "adverse_adds_summary": position_building.get("adverse_adds"),
            "bounded_average_into_adverse": bounded_by_build.get("AVERAGE_INTO_ADVERSE"),
            "contract": (position_building.get("contract") or {}),
        },
    }


def _source_provenance(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "local": [
            str(args.manual_history),
            str(args.operator_precedent),
            str(args.manual_context),
            str(args.daily_target_state),
            str(args.execution_ledger),
            "docs/trade_case_studies/",
            "collab_trade/strategy_memory.md",
            "docs/CHANGELOG.md",
        ],
        "notion_reviewed": [
            {
                "title": "quantrabbit trade reference",
                "url": "https://app.notion.com/p/38ef1c8e53a781eb8243ee2342891f2c",
                "used_for": "confirmed QuantRabbit local/project reference.",
            },
            {
                "title": "NAV percent for asset-relative numbers",
                "url": "https://app.notion.com/p/38ef1c8e53a7818499b4ef739905e1a5",
                "used_for": "kept target, drawdown, and return comparisons in NAV percent rather than stale JPY thresholds.",
            },
            {
                "title": "basket execution + pace cap",
                "url": "https://app.notion.com/p/38ef1c8e53a7816eb2fff7e443a76524",
                "used_for": "forced-overtrading risk: extreme pace divisors and narrow basket selection can shrink trade size into the noise floor.",
            },
            {
                "title": "routine owns trading",
                "url": "https://app.notion.com/p/38ef1c8e53a7815ab946d33a299600b5",
                "used_for": "kept this task read-only and analysis/report-only.",
            },
        ],
    }


def _recommendation(target_models: dict[str, Any], precedent: dict[str, Any], daily_rows: list[dict[str, Any]]) -> dict[str, Any]:
    red_stats = _red_day_stats(daily_rows)
    model_b_red = target_models["model_b_rolling_30d_4x"]["red_days_allowed_if_good_days_hit_10pct"]
    return {
        "target_policy": "Optimize for Model B: rolling 30-day 4x account growth.",
        "why": [
            "The 2025 manual precedent verifies a best funding-adjusted 30-day return above 4x while still containing red days; that shape is rolling-window performance, not a clean fixed +5% every UTC day.",
            "Model A compounds to a slightly higher 30-day multiplier than 4x and treats ordinary bad/no-edge days as failures, increasing forced overtrading risk.",
            "Model B preserves the arithmetic ambition while letting +5% act as a pace marker and incident diagnostic instead of an order-forcing rule.",
        ],
        "use_of_5pct": "PACE_MARKER: use +5% to measure whether the rolling plan is on schedule, to trigger review when missed, and to unlock protection-first behavior when reached; do not force bad-day trades solely to print +5%.",
        "when_10pct_allowed": "Only when the existing 10% Extension Gate is YES: strong progress or protected S/A carry, hero thesis still paying, broad theme/trend confirmation, stable spread, no near whipsaw event, and real reload/second-shot levels.",
        "bad_red_day_tolerance": {
            "observed_2025_active_days": red_stats,
            "model_b_red_day_math": model_b_red,
            "interpretation": "With +10% good days, the 30-day 4x plan can mathematically absorb several red/flat days; the exact allowance must be recomputed from realized red-day severity, not assumed as permission to take weak trades.",
        },
        "behaviors_to_block": [
            "Counting OPERATOR_MANUAL / USER_ALPHA results as system profitability.",
            "Forcing entries on no-edge days to satisfy a same-day hard target.",
            "Margin-closeout tolerance or any unbounded long unattended hold.",
            "With-move pyramiding without current basket/portfolio risk validation.",
            "Adverse adds that are not bounded by thesis invalidation, current location, and margin budget.",
            "USD_JPY-only rules; convert precedent into pair-agnostic theme/location/session/shape checks.",
            "Broker SL or close logic that does not represent thesis invalidation.",
        ],
        "missing_data": [
            "Intraday account-equity curve around deposits/withdrawals, so day-start equity remains estimated.",
            "MFE/MAE path per 2025 manual exit for exact drawdown and red-day severity modeling.",
            "Pair-agnostic H4/D context for every 2025 manual trade; current manual-context audit is H1/M5-centered.",
            "Explicit origin tags in 2025 history distinguishing operator discovery, system management, and user-alpha reloads.",
            "Current execution ledger close rows do not always preserve lane/client id on the closing event; attribution is conservative.",
            "No Notion page was found that directly defines the rolling 30-day 4x policy; this report derives it from the user objective plus local precedent.",
        ],
    }


def _lesson(
    pattern_id: str,
    lesson: str,
    evidence: dict[str, Any],
    reusable_rule: str,
    blocked_behavior: str,
    *,
    allowed_use: str = "Advisory precedent only; current forecast, risk, spread, event, and gateway checks remain authoritative.",
) -> dict[str, Any]:
    return {
        "pattern_id": pattern_id,
        "lesson": lesson,
        "evidence": evidence,
        "reusable_rule": reusable_rule,
        "allowed_use": allowed_use,
        "blocked_behavior": blocked_behavior,
    }


def _trade_shape_precedent(manual_history: dict[str, Any], manual_context: dict[str, Any]) -> dict[str, Any]:
    bounded = manual_context.get("bounded_replay_profile") or {}
    excluded = manual_context.get("excluded_tail_profile") or {}
    position_building = manual_context.get("position_building_profile") or {}
    case_study = {
        "file": "docs/trade_case_studies/2026-06-29-usdjpy-eurjpy-162-fade.md",
        "lesson": "Major-figure/intervention fade needs thesis invalidation, not a tight noise-zone broker SL.",
    }
    by_build = {row.get("bucket"): row for row in position_building.get("bounded_by_build_type") or []}
    lessons = [
        _lesson(
            "THEME_READ",
            "A trade shape starts from the current theme, not from a single pair memory.",
            {"case_study": case_study},
            "Map the cleanest expression of the bought/sold currency theme, then select pair/vehicle only after current evidence confirms the theme.",
            "Do not replay the 2025 USD_JPY precedent as a USD_JPY-only rule or as permission to ignore other pairs.",
        ),
        _lesson(
            "LOCATION_24H",
            "24h location is a first-class edge filter.",
            {"bounded_side_entry_location_24h": bounded.get("by_side_entry_location_24h")},
            "Prefer pair-agnostic broad-discount LONG and broad-premium SHORT structures when they match current thesis and risk geometry.",
            "Block sell-the-low / buy-the-high churn unless a separate breakout/trend proof exists.",
        ),
        _lesson(
            "H1_H4_ALIGNMENT",
            "H1/H4 alignment must be evidence, not a slogan.",
            {"bounded_h1_alignment": bounded.get("by_h1_alignment"), "h4_gap": "H4 was not fully reconstructed for 2025 manual rows."},
            "Use H1/H4 as current context buckets and require extra current reason when the current alignment conflicts with the positive bounded precedent.",
            "Do not infer H4 support from H1-only data; missing H4 is a data gap, not permission.",
        ),
        _lesson(
            "SESSION",
            "Session changes payoff and should shape aggressiveness.",
            {"bounded_session_jst": bounded.get("by_session_jst")},
            "Rank session-conditioned shapes separately and prefer the current liquid/session-compatible expression.",
            "Do not force the same vehicle in Tokyo, London, NY overlap, and off-hours without session-specific evidence.",
        ),
        _lesson(
            "BOUNDED_ADVERSE_ADD",
            "Adverse adds were only useful when bounded and thesis-specific.",
            {"bounded_adverse_adds": position_building.get("adverse_adds"), "bounded_average_into_adverse": by_build.get("AVERAGE_INTO_ADVERSE")},
            "An adverse add can only be considered when invalidation, max entries, margin, and harvest path are explicit before the add.",
            "Block martingale-style averaging, margin rescue, or adds whose only reason is a red open P/L.",
        ),
        _lesson(
            "WITH_MOVE_PYRAMID",
            "With-move pyramiding was not automatically positive in bounded replay.",
            {"bounded_pyramid_with_move": by_build.get("PYRAMID_WITH_MOVE")},
            "Pyramids need independent fresh edge and portfolio room; a move already in profit is not enough.",
            "Block stack-on-green behavior when the new layer lacks its own entry location, TP, invalidation, and margin proof.",
        ),
        _lesson(
            "SL_FREE_THESIS_INVALIDATION",
            "SL-free does not mean loss-free; it means exits must be thesis invalidation based.",
            {"case_study": case_study},
            "When using SL-free or wide-catastrophe-stop logic, store the thesis, invalidation level/timeframe, and review trigger before entry.",
            "Block tight broker SLs inside normal noise/major-figure battle zones and block loss-side closes from red P/L alone.",
        ),
        _lesson(
            "HOUSEKEEPING_HARVEST",
            "Harvest exits are the reusable profit engine; housekeeping must not become panic closing.",
            {"bounded_close_reason": bounded.get("by_close_reason")},
            "Prefer attached TP / harvest / profit capture when current forecast weakens or range rail is reached.",
            "Block market-close leakage where one large give-up close erases multiple average winners.",
        ),
        _lesson(
            "MARGIN_CLOSEOUT_FAILURE",
            "Margin closeout is a hard failure mode, not a strategy exit.",
            {"manual_close_reason": (manual_history.get("analysis") or {}).get("by_close_reason", {}).get("MARKET_ORDER_MARGIN_CLOSEOUT")},
            "Treat margin capacity as pre-entry and add-layer risk control; once margin pressure appears, do not add risk to rescue the basket.",
            "Block any target policy that requires trading through margin pressure to maintain a daily cadence.",
        ),
        _lesson(
            "LONG_UNATTENDED_HOLD_FAILURE",
            "Long unattended holds contaminate precedent because they mix thesis edge with carry tail risk.",
            {"excluded_tail_profile": excluded.get("overall"), "excluded_tail_by_hold_bucket": excluded.get("by_hold_bucket")},
            "Separate bounded intraday precedent from >=12h carry/tail rows before citing edge.",
            "Block using raw long-hold winners to justify unattended exposure without current thesis-evolution and margin controls.",
        ),
    ]
    return {
        "generated_at_utc": _now_utc(),
        "status": "TRADE_SHAPE_PRECEDENT_READY",
        "source_sample": {
            "manual_history_window": manual_history.get("window"),
            "manual_exit_events": manual_history.get("exit_events"),
            "sample_pairs": sorted({str(row.get("pair")) for row in manual_history.get("trades", []) if row.get("pair")}),
            "pair_agnostic_constraint": "The observed 2025 sample is USD_JPY-heavy, so lessons are expressed as theme/location/session/shape contracts, not as pair-specific permission.",
        },
        "lessons": lessons,
    }


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join("" if value is None else str(value) for value in row) + " |")
    return "\n".join(out)


def _model_rows(models: dict[str, Any]) -> list[list[Any]]:
    a = models["model_a_fixed_daily_5pct"]
    b = models["model_b_rolling_30d_4x"]
    return [
        [
            "A fixed daily +5%",
            a["required_average_daily_return_pct"],
            a["required_average_trading_day_return_pct"],
            a["equivalent_30d_multiplier"],
            a["red_days_allowed_if_good_days_hit_10pct"]["policy_allowed"],
            a["risk_of_forced_overtrading"],
        ],
        [
            "B rolling 30d 4x",
            b["required_average_daily_return_pct"],
            b["required_average_trading_day_return_pct"],
            b["equivalent_30d_multiplier"],
            b["red_days_allowed_if_good_days_hit_10pct"]["mathematical_empirical_red_days_for_30d_multiplier"]["max_red_days"],
            b["risk_of_forced_overtrading"],
        ],
    ]


def _breakdown_rows(breakdown: dict[str, Any], limit: int | None = None) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for name, data in breakdown.items():
        if not isinstance(data, dict):
            continue
        rows.append(
            [
                name,
                data.get("trades"),
                data.get("net"),
                data.get("win_rate"),
                data.get("payoff"),
                data.get("median_hold_hours"),
            ]
        )
    rows.sort(key=lambda row: float(row[2] or 0.0), reverse=True)
    return rows[:limit] if limit is not None else rows


def _drawdown_rows(model: dict[str, Any], elapsed_days: int) -> list[list[Any]]:
    return [
        [
            row["drawdown_pct"],
            row["remaining_days"],
            row["required_remaining_daily_return_pct"],
        ]
        for row in model.get("drawdown_sensitivity") or []
        if row.get("elapsed_days") == elapsed_days
    ]


def _summary_fragment(row: dict[str, Any] | None, keys: list[str]) -> str:
    if not isinstance(row, dict):
        return "missing"
    parts = []
    for key in keys:
        if key in row:
            parts.append(f"{key}={row.get(key)}")
    return ", ".join(parts) if parts else "missing"


def _render_target_report(payload: dict[str, Any]) -> str:
    models = payload["target_models"]
    precedent = payload["precedent_2025"]
    rec = payload["recommendation"]
    daily = payload["daily_return_summary"]
    origin = payload["origin_breakdown"]
    best_30 = precedent["funding_adjusted"]["best_30d_daily_evidence"]
    lines = [
        "# Target Cadence Analysis",
        "",
        f"- Generated at UTC: `{payload['generated_at_utc']}`",
        "- Scope: analysis/report generation only; no live orders, cancels, or position changes.",
        f"- Recommendation: **{rec['target_policy']}**",
        "",
        "## Target Model Comparison",
        "",
        _table(
            ["model", "avg daily %", "avg trading-day %", "30d multiplier", "red days allowed", "forced overtrading risk"],
            _model_rows(models),
        ),
        "",
        "Model A is arithmetically stricter than the requested rolling 4x plan: +5% every calendar day compounds to "
        f"{models['model_a_fixed_daily_5pct']['equivalent_30d_multiplier']}x over 30 days. "
        "Model B requires "
        f"{models['model_b_rolling_30d_4x']['required_average_daily_return_pct']}% average calendar-day return.",
        "",
        "## Drawdown Sensitivity",
        "",
        "Required remaining daily return after a drawdown on day 10:",
        "",
        "Model A:",
        "",
        _table(["drawdown %", "remaining days", "required daily %"], _drawdown_rows(models["model_a_fixed_daily_5pct"], 10)),
        "",
        "Model B:",
        "",
        _table(["drawdown %", "remaining days", "required daily %"], _drawdown_rows(models["model_b_rolling_30d_4x"], 10)),
        "",
        "## 2025 Precedent",
        "",
        f"- Active/best 30d evidence: `{best_30.get('start_time')}` to `{best_30.get('end_time')}`.",
        f"- Best funding-adjusted 30d return: `{best_30.get('funding_adjusted_return_pct')}`% "
        f"= `{best_30.get('equivalent_multiplier')}`x.",
        f"- End funding-adjusted P/L: `{precedent['funding_adjusted']['end_pl_jpy']}` JPY "
        f"(`{precedent['funding_adjusted']['end_return_pct']}`%).",
        f"- Exits: `{precedent['exit_count']}`, realized P/L: `{precedent['total_realized_pl_jpy']}` JPY, "
        f"win rate: `{precedent['win_rate']}`, profit factor: `{precedent['profit_factor']}`.",
        f"- Best/worst day: `{precedent['best_day']}` / `{precedent['worst_day']}`.",
        f"- Best 30d active UTC days: `{best_30.get('active_utc_days')}`, red UTC days: `{best_30.get('red_utc_days')}`, "
        f"+5% days: `{best_30.get('hit_5pct_utc_days')}`, +10% days: `{best_30.get('hit_10pct_utc_days')}`.",
        "",
        "## Precedent Breakdowns",
        "",
        "Pair:",
        "",
        _table(["bucket", "exits", "net JPY", "win rate", "payoff", "median hold h"], _breakdown_rows(precedent["pair_breakdown"])),
        "",
        "Side:",
        "",
        _table(["bucket", "exits", "net JPY", "win rate", "payoff", "median hold h"], _breakdown_rows(precedent["side_breakdown"])),
        "",
        "Session:",
        "",
        _table(["bucket", "exits", "net JPY", "win rate", "payoff", "median hold h"], _breakdown_rows(precedent["session_breakdown"])),
        "",
        "Exit reason:",
        "",
        _table(["bucket", "exits", "net JPY", "win rate", "payoff", "median hold h"], _breakdown_rows(precedent["exit_reason_breakdown"])),
        "",
        "## Tail And Build Risks",
        "",
        f"- Margin closeout: `{_summary_fragment(precedent['margin_closeout_losses'], ['trades', 'net', 'win_rate', 'median_hold_hours', 'expectancy'])}`.",
        f"- Long-hold excluded tail: `{_summary_fragment(precedent['long_hold_tail_risk']['excluded_tail_overall'], ['trades', 'net_jpy', 'win_rate', 'median_hold_hours', 'expectancy_jpy'])}`.",
        f"- With-move pyramid bounded replay: `{_summary_fragment((precedent['with_move_pyramiding_pl'] or {}).get('bounded_lt_12h_ex_margin_closeout'), ['clusters', 'entries', 'net_jpy', 'win_rate', 'expectancy_jpy'])}`.",
        f"- Bounded adverse-add summary: `{_summary_fragment((precedent['bounded_adverse_add_pl'] or {}).get('adverse_adds_summary'), ['clusters', 'entries', 'net_jpy', 'win_rate', 'expectancy_jpy', 'max_entries'])}`.",
        "",
        "## Daily Return Evidence",
        "",
        f"- Active UTC days reconstructed: `{daily['active_days']}`.",
        f"- Red days: `{daily['red_days']}`; green days: `{daily.get('green_days')}`.",
        f"- Days hitting +5%: `{daily.get('hit_5pct_days')}`; days hitting +10%: `{daily.get('hit_10pct_days')}`.",
        f"- Median red day return: `{daily.get('median_red_return_pct')}`%; worst day return: `{daily.get('worst_daily_return_pct')}`%.",
        "",
        "## Origin Separation",
        "",
        f"- 2025 manual history is classified as `{origin['manual_2025']['origin']}` and **not** counted as system profitability.",
        f"- 2025 manual realized P/L: `{origin['manual_2025']['realized_pl_jpy']}` JPY.",
        "- Current execution-ledger origin buckets are included in the JSON artifact with conservative qrv1/lane-id attribution.",
        "",
        "## Policy Answer",
        "",
        f"- Optimize for: **{rec['target_policy']}**",
        f"- Use +5% as: **{rec['use_of_5pct']}**",
        f"- Allow +10% when: {rec['when_10pct_allowed']}",
        f"- Bad/red days: {rec['bad_red_day_tolerance']['interpretation']}",
        "",
        "## Behavior To Block",
        "",
    ]
    lines.extend(f"- {item}" for item in rec["behaviors_to_block"])
    lines.extend(["", "## Missing Data", ""])
    lines.extend(f"- {item}" for item in rec["missing_data"])
    lines.extend(["", "## Sources", ""])
    lines.extend(f"- `{src}`" for src in payload["source_provenance"]["local"])
    lines.extend(f"- Notion: [{src['title']}]({src['url']})" for src in payload["source_provenance"]["notion_reviewed"])
    return "\n".join(lines) + "\n"


def _render_shape_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Trade Shape Precedent",
        "",
        f"- Generated at UTC: `{payload['generated_at_utc']}`",
        f"- Status: `{payload['status']}`",
        f"- Sample pairs: `{', '.join(payload['source_sample']['sample_pairs'])}`",
        f"- Constraint: {payload['source_sample']['pair_agnostic_constraint']}",
        "",
        "## Lessons",
        "",
    ]
    for lesson in payload["lessons"]:
        lines.extend(
            [
                f"### {lesson['pattern_id']}",
                "",
                f"- Lesson: {lesson['lesson']}",
                f"- Reusable rule: {lesson['reusable_rule']}",
                f"- Blocked behavior: {lesson['blocked_behavior']}",
                f"- Allowed use: {lesson['allowed_use']}",
                "",
            ]
        )
    return "\n".join(lines)


def build_analysis(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    manual_history = _load_json(args.manual_history, default={})
    operator_precedent = _load_json(args.operator_precedent, default={})
    manual_context = _load_json(args.manual_context, default={})
    daily_target_state = _load_json(args.daily_target_state, default={})
    daily_rows = _daily_return_table(manual_history)
    target_models = _target_models(daily_rows)
    precedent = _precedent_2025(manual_history, manual_context, operator_precedent, daily_rows)
    target_payload = {
        "generated_at_utc": _now_utc(),
        "status": "TARGET_CADENCE_ANALYSIS_READY",
        "source_provenance": _source_provenance(args),
        "daily_target_state_snapshot": {
            "available": bool(daily_target_state),
            "mode": daily_target_state.get("mode"),
            "target_trades_per_day": daily_target_state.get("target_trades_per_day"),
            "progress_pct": daily_target_state.get("progress_pct"),
        },
        "target_models": target_models,
        "precedent_2025": precedent,
        "daily_return_summary": _red_day_stats(daily_rows),
        "daily_return_table": daily_rows,
        "origin_breakdown": _origin_breakdown(manual_history, args.execution_ledger),
    }
    target_payload["recommendation"] = _recommendation(target_models, precedent, daily_rows)
    shape_payload = _trade_shape_precedent(manual_history, manual_context)
    return target_payload, shape_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-history", type=Path, default=DEFAULT_MANUAL_HISTORY)
    parser.add_argument("--operator-precedent", type=Path, default=DEFAULT_OPERATOR_PRECEDENT)
    parser.add_argument("--manual-context", type=Path, default=DEFAULT_MANUAL_CONTEXT)
    parser.add_argument("--daily-target-state", type=Path, default=DEFAULT_DAILY_TARGET_STATE)
    parser.add_argument("--execution-ledger", type=Path, default=DEFAULT_EXECUTION_LEDGER)
    parser.add_argument("--target-json", type=Path, default=DEFAULT_TARGET_JSON)
    parser.add_argument("--target-report", type=Path, default=DEFAULT_TARGET_REPORT)
    parser.add_argument("--shape-json", type=Path, default=DEFAULT_SHAPE_JSON)
    parser.add_argument("--shape-report", type=Path, default=DEFAULT_SHAPE_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_payload, shape_payload = build_analysis(args)
    _write_json(args.target_json, target_payload)
    _write_text(args.target_report, _render_target_report(target_payload))
    _write_json(args.shape_json, shape_payload)
    _write_text(args.shape_report, _render_shape_report(shape_payload))
    print(f"wrote {args.target_json}")
    print(f"wrote {args.target_report}")
    print(f"wrote {args.shape_json}")
    print(f"wrote {args.shape_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
