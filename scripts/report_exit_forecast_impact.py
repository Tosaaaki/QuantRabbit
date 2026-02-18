#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable


TP_REASON_TOKENS = (
    "take_profit",
    "trail_take",
    "trail_lock",
    "profit_lock",
    "lock_floor",
    "lock_trail",
    "rsi_take",
    "partial_take",
    "take_profit_order",
)

SL_REASON_TOKENS = (
    "hard_stop",
    "tech_hard_stop",
    "max_adverse",
    "time_stop",
    "timeout",
    "drawdown",
    "max_drawdown",
    "free_margin_low",
    "margin_health",
    "margin_usage_high",
    "stop_loss_order",
    "loss_cut",
)


@dataclass
class TradeRow:
    strategy: str
    close_reason: str
    pl_pips: float
    hold_sec: float | None
    state: str
    exit_class: str



def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()



def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None



def _to_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
    return None



def _parse_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}



def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value)
    try:
        if text.endswith("Z"):
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        return datetime.fromisoformat(text)
    except Exception:
        return None



def _exit_class(close_reason: str) -> str:
    reason = (close_reason or "").strip().lower()
    if any(token in reason for token in TP_REASON_TOKENS):
        return "tp"
    if any(token in reason for token in SL_REASON_TOKENS):
        return "sl"
    return "other"



def _trade_side(units: int, thesis: dict[str, Any]) -> str | None:
    side = thesis.get("side")
    if isinstance(side, str):
        side_l = side.strip().lower()
        if side_l in {"long", "buy"}:
            return "long"
        if side_l in {"short", "sell"}:
            return "short"
    if units > 0:
        return "long"
    if units < 0:
        return "short"
    return None



def _forecast_state(side: str | None, forecast: dict[str, Any]) -> str:
    if not forecast:
        return "no_forecast"

    allowed = _to_bool(forecast.get("allowed"))
    p_up = _to_float(forecast.get("p_up"))
    expected_pips = _to_float(forecast.get("expected_pips"))

    direction_prob = None
    if p_up is not None and side in {"long", "short"}:
        direction_prob = p_up if side == "long" else 1.0 - p_up

    if allowed is False:
        return "forecast_blocked"

    if direction_prob is not None:
        if direction_prob >= 0.55:
            return "forecast_favorable"
        if direction_prob <= 0.45:
            return "forecast_contra"

    if expected_pips is not None:
        if expected_pips >= 0.20:
            return "forecast_favorable"
        if expected_pips <= -0.20:
            return "forecast_contra"

    return "forecast_neutral"



def _safe_avg(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values]
    if not vals:
        return None
    return sum(vals) / float(len(vals))



def _safe_median(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values]
    if not vals:
        return None
    return float(median(vals))



def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"



def _summarize(rows: list[TradeRow], early_tp_sec: float, early_sl_sec: float) -> dict[str, Any]:
    trades = len(rows)
    if trades == 0:
        return {
            "trades": 0,
            "win_rate": None,
            "avg_pips": None,
            "median_pips": None,
            "avg_hold_sec": None,
            "median_hold_sec": None,
            "tp_rate": None,
            "sl_rate": None,
            "avg_win_pips": None,
            "avg_loss_pips": None,
            "avg_loss_hold_sec": None,
            "early_tp_rate": None,
            "early_sl_rate": None,
        }

    wins = [r for r in rows if r.pl_pips > 0.0]
    losses = [r for r in rows if r.pl_pips < 0.0]
    pips = [r.pl_pips for r in rows]
    holds = [r.hold_sec for r in rows if r.hold_sec is not None]

    tp_rows = [r for r in rows if r.exit_class == "tp"]
    sl_rows = [r for r in rows if r.exit_class == "sl"]

    tp_early = [r for r in tp_rows if r.hold_sec is not None and r.hold_sec <= early_tp_sec]
    sl_early = [r for r in sl_rows if r.hold_sec is not None and r.hold_sec <= early_sl_sec]

    loss_holds = [r.hold_sec for r in losses if r.hold_sec is not None]

    return {
        "trades": trades,
        "win_rate": len(wins) / trades,
        "avg_pips": _safe_avg(pips),
        "median_pips": _safe_median(pips),
        "avg_hold_sec": _safe_avg(holds),
        "median_hold_sec": _safe_median(holds),
        "tp_rate": len(tp_rows) / trades,
        "sl_rate": len(sl_rows) / trades,
        "avg_win_pips": _safe_avg([r.pl_pips for r in wins]),
        "avg_loss_pips": _safe_avg([r.pl_pips for r in losses]),
        "avg_loss_hold_sec": _safe_avg(loss_holds),
        "early_tp_rate": (len(tp_early) / len(tp_rows)) if tp_rows else None,
        "early_sl_rate": (len(sl_early) / len(sl_rows)) if sl_rows else None,
    }



def _impact_deltas(state_summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    favorable = state_summary.get("forecast_favorable")
    adverse = state_summary.get("forecast_contra") or state_summary.get("forecast_blocked")
    if not favorable or not adverse:
        return {}

    def _delta(key: str) -> float | None:
        a = adverse.get(key)
        b = favorable.get(key)
        if a is None or b is None:
            return None
        try:
            return float(a) - float(b)
        except Exception:
            return None

    return {
        "avg_pips_delta_adverse_minus_favorable": _delta("avg_pips"),
        "avg_loss_pips_delta_adverse_minus_favorable": _delta("avg_loss_pips"),
        "avg_hold_sec_delta_adverse_minus_favorable": _delta("avg_hold_sec"),
        "avg_loss_hold_sec_delta_adverse_minus_favorable": _delta("avg_loss_hold_sec"),
        "early_tp_rate_delta_adverse_minus_favorable": _delta("early_tp_rate"),
        "early_sl_rate_delta_adverse_minus_favorable": _delta("early_sl_rate"),
    }



def _load_rows(db: Path, lookback_hours: int, instrument: str | None) -> list[TradeRow]:
    if not db.exists():
        raise FileNotFoundError(f"db not found: {db}")

    uri = f"file:{db}?mode=ro"
    con = sqlite3.connect(uri, uri=True, timeout=8.0, isolation_level=None)
    try:
        instrument_sql = " AND instrument = :instrument " if instrument else ""
        sql = f"""
            SELECT close_time, open_time, strategy_tag, strategy, close_reason, pl_pips, units, entry_thesis
            FROM trades
            WHERE close_time IS NOT NULL
              AND close_time >= datetime('now', :lookback)
              {instrument_sql}
            ORDER BY close_time DESC
        """
        params: dict[str, Any] = {"lookback": f"-{int(lookback_hours)} hour"}
        if instrument:
            params["instrument"] = instrument

        out: list[TradeRow] = []
        for close_time, open_time, strategy_tag, strategy, close_reason, pl_pips, units, entry_thesis in con.execute(sql, params):
            thesis = _parse_json(entry_thesis)
            forecast = thesis.get("forecast") if isinstance(thesis.get("forecast"), dict) else {}
            side = _trade_side(int(units or 0), thesis)
            state = _forecast_state(side=side, forecast=forecast)

            close_dt = _parse_time(close_time)
            open_dt = _parse_time(open_time)
            hold_sec = None
            if close_dt and open_dt:
                hold_sec = max(0.0, (close_dt - open_dt).total_seconds())

            out.append(
                TradeRow(
                    strategy=(strategy_tag or strategy or "unknown"),
                    close_reason=(close_reason or "unknown"),
                    pl_pips=float(pl_pips or 0.0),
                    hold_sec=hold_sec,
                    state=state,
                    exit_class=_exit_class(close_reason or ""),
                )
            )
        return out
    finally:
        con.close()



def main() -> None:
    ap = argparse.ArgumentParser(description="Report 1h exit forecast impact from trades.db")
    ap.add_argument("--db", default="logs/trades.db", help="Path to trades.db")
    ap.add_argument("--lookback-hours", type=int, default=1, help="Lookback window in hours")
    ap.add_argument("--instrument", default=None, help="Optional instrument filter (e.g. USD_JPY)")
    ap.add_argument("--early-tp-sec", type=float, default=120.0, help="Threshold for early TP")
    ap.add_argument("--early-sl-sec", type=float, default=180.0, help="Threshold for early SL")
    ap.add_argument("--strategy-top", type=int, default=12, help="Max strategies in text summary")
    ap.add_argument("--json-out", default=None, help="Optional JSON output path")
    args = ap.parse_args()

    rows = _load_rows(Path(args.db), lookback_hours=args.lookback_hours, instrument=args.instrument)
    total = len(rows)
    if total == 0:
        print("[exit-forecast-impact] no trades in window")
        return

    by_state: dict[str, list[TradeRow]] = defaultdict(list)
    by_strategy: dict[str, list[TradeRow]] = defaultdict(list)
    for row in rows:
        by_state[row.state].append(row)
        by_strategy[row.strategy].append(row)

    state_summary = {
        state: _summarize(state_rows, early_tp_sec=args.early_tp_sec, early_sl_sec=args.early_sl_sec)
        for state, state_rows in sorted(by_state.items())
    }

    strategy_rows = []
    for strategy, strategy_trades in by_strategy.items():
        summary = _summarize(strategy_trades, early_tp_sec=args.early_tp_sec, early_sl_sec=args.early_sl_sec)
        with_forecast = sum(1 for r in strategy_trades if r.state != "no_forecast")
        summary["forecast_coverage"] = with_forecast / max(len(strategy_trades), 1)
        summary["strategy"] = strategy
        strategy_rows.append(summary)
    strategy_rows.sort(key=lambda item: int(item.get("trades") or 0), reverse=True)

    forecast_trades = sum(1 for r in rows if r.state != "no_forecast")
    payload = {
        "generated_at_utc": _now_iso(),
        "db": str(Path(args.db)),
        "lookback_hours": int(args.lookback_hours),
        "instrument": args.instrument,
        "total_trades": total,
        "forecast_trade_count": forecast_trades,
        "forecast_coverage": forecast_trades / max(total, 1),
        "state_summary": state_summary,
        "impact_deltas": _impact_deltas(state_summary),
        "strategy_summary": strategy_rows,
    }

    print(
        "[exit-forecast-impact] "
        f"trades={total} forecast_coverage={payload['forecast_coverage']:.3f} "
        f"window={args.lookback_hours}h"
    )

    print("\n[state summary]")
    for state, summary in state_summary.items():
        print(
            f"- {state}: trades={summary['trades']}"
            f" win_rate={_fmt(summary['win_rate'])}"
            f" avg_pips={_fmt(summary['avg_pips'])}"
            f" avg_hold_sec={_fmt(summary['avg_hold_sec'], 1)}"
            f" sl_rate={_fmt(summary['sl_rate'])}"
            f" avg_loss_pips={_fmt(summary['avg_loss_pips'])}"
            f" avg_loss_hold_sec={_fmt(summary['avg_loss_hold_sec'], 1)}"
            f" early_tp_rate={_fmt(summary['early_tp_rate'])}"
            f" early_sl_rate={_fmt(summary['early_sl_rate'])}"
        )

    print("\n[strategy top]")
    for row in strategy_rows[: max(int(args.strategy_top), 0)]:
        print(
            f"- {row['strategy']}: trades={row['trades']}"
            f" forecast_cov={_fmt(row.get('forecast_coverage'))}"
            f" win_rate={_fmt(row.get('win_rate'))}"
            f" avg_pips={_fmt(row.get('avg_pips'))}"
            f" avg_hold_sec={_fmt(row.get('avg_hold_sec'), 1)}"
            f" avg_loss_pips={_fmt(row.get('avg_loss_pips'))}"
            f" early_tp_rate={_fmt(row.get('early_tp_rate'))}"
            f" early_sl_rate={_fmt(row.get('early_sl_rate'))}"
        )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[exit-forecast-impact] json={out_path}")


if __name__ == "__main__":
    main()
