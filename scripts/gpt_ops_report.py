#!/usr/bin/env python3
"""Deterministic market playbook report (LLM disabled).

This script keeps the existing `gpt_ops_report.py` entrypoint but replaces the
stub payload with a structured, rule-based market playbook. It does not call
any LLM provider.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sqlite3
import sys
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from analytics.policy_diff import normalize_policy_diff

UTC = timezone.utc
JST = timezone(timedelta(hours=9))

REJECT_STATUSES = {
    "rejected",
    "failed",
    "error",
    "cancelled",
    "timeout",
    "quote_retry_failed",
}


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _to_iso(dt: datetime) -> str:
    return dt.astimezone(UTC).replace(microsecond=0).isoformat()


def _to_jst_label(dt: datetime) -> str:
    return dt.astimezone(JST).strftime("%Y-%m-%d %H:%M JST")


def _parse_iso_datetime(raw: object, *, default_tz: timezone = UTC) -> Optional[datetime]:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        dt = raw
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=default_tz)
        return dt.astimezone(UTC)
    if isinstance(raw, (int, float)):
        value = float(raw)
        if value > 10_000_000_000:
            value /= 1000.0
        if value <= 0:
            return None
        return datetime.fromtimestamp(value, tz=UTC)

    text = str(raw).strip()
    if not text:
        return None

    candidate = text.replace("/", "-")
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(candidate)
    except ValueError:
        dt = None

    if dt is None:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y%m%d %H:%M"):
            try:
                dt = datetime.strptime(candidate, fmt)
                break
            except ValueError:
                continue
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=default_tz)
    return dt.astimezone(UTC)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _load_factors() -> dict[str, dict[str, Any]]:
    try:
        from indicators.factor_cache import all_factors
    except Exception as exc:
        logging.warning("[OPS_REPORT] factor_cache import failed: %s", exc)
        return {}
    try:
        payload = all_factors()
    except Exception as exc:
        logging.warning("[OPS_REPORT] all_factors() failed: %s", exc)
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for tf, row in payload.items():
        if isinstance(row, dict):
            out[str(tf)] = dict(row)
    return out


def _forecast_decision_to_dict(decision: Any) -> Optional[dict[str, Any]]:
    if decision is None:
        return None
    return {
        "allowed": bool(getattr(decision, "allowed", False)),
        "scale": _safe_float(getattr(decision, "scale", 1.0), 1.0),
        "reason": str(getattr(decision, "reason", "") or ""),
        "horizon": str(getattr(decision, "horizon", "") or ""),
        "edge": _safe_float(getattr(decision, "edge", 0.0), 0.0),
        "p_up": _safe_float(getattr(decision, "p_up", 0.5), 0.5),
        "expected_pips": _safe_float(getattr(decision, "expected_pips", 0.0), 0.0),
        "target_reach_prob": _safe_float(getattr(decision, "target_reach_prob", 0.0), 0.0),
        "future_flow": str(getattr(decision, "future_flow", "") or ""),
        "style": str(getattr(decision, "style", "") or ""),
        "trend_state": str(getattr(decision, "trend_state", "") or ""),
        "range_state": str(getattr(decision, "range_state", "") or ""),
        "volatility_state": str(getattr(decision, "volatility_state", "") or ""),
        "feature_ts": str(getattr(decision, "feature_ts", "") or ""),
    }


def _load_forecast_snapshot(
    *,
    strategy_tag: str,
    pocket: str,
    units: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "enabled": False,
        "buy": None,
        "sell": None,
        "reference": None,
    }
    try:
        from workers.common import forecast_gate
    except Exception as exc:
        logging.info("[OPS_REPORT] forecast gate unavailable: %s", exc)
        return out

    try:
        buy_decision = forecast_gate.decide(
            strategy_tag=strategy_tag,
            pocket=pocket,
            side="buy",
            units=units,
            meta={"instrument": "USD_JPY"},
        )
        sell_decision = forecast_gate.decide(
            strategy_tag=strategy_tag,
            pocket=pocket,
            side="sell",
            units=units,
            meta={"instrument": "USD_JPY"},
        )
    except Exception as exc:
        logging.warning("[OPS_REPORT] forecast decide failed: %s", exc)
        return out

    buy = _forecast_decision_to_dict(buy_decision)
    sell = _forecast_decision_to_dict(sell_decision)
    out["enabled"] = bool(buy or sell)
    out["buy"] = buy
    out["sell"] = sell
    out["reference"] = buy or sell
    return out


def _trend_sign(label: str) -> float:
    text = str(label or "").strip().lower()
    if text in {"up", "bull", "bullish", "trend_up"}:
        return 1.0
    if text in {"down", "bear", "bearish", "trend_down"}:
        return -1.0
    return 0.0


def _extract_levels(story: dict[str, Any], current_price: float) -> tuple[float, float]:
    levels = story.get("major_levels") if isinstance(story.get("major_levels"), dict) else {}
    candidates: list[float] = []
    for tf in ("h4", "d1"):
        row = levels.get(tf) if isinstance(levels, dict) else None
        if not isinstance(row, dict):
            continue
        for key in ("pivot", "r1", "s1", "fib50", "fib61", "recent_high", "recent_low"):
            value = _safe_float(row.get(key), 0.0)
            if value > 0:
                candidates.append(value)
    above = [v for v in candidates if v > current_price]
    below = [v for v in candidates if v < current_price]
    support = max(below) if below else current_price - 0.15
    resistance = min(above) if above else current_price + 0.15
    if support >= current_price:
        support = current_price - 0.15
    if resistance <= current_price:
        resistance = current_price + 0.15
    return round(support, 3), round(resistance, 3)


def _load_chart_story(fac_m1: dict[str, Any], fac_h4: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not fac_m1 or not fac_h4:
        return out
    try:
        from analysis.chart_story import ChartStory
    except Exception as exc:
        logging.info("[OPS_REPORT] chart_story unavailable: %s", exc)
        return out
    try:
        snapshot = ChartStory().update(fac_m1, fac_h4)
    except Exception as exc:
        logging.warning("[OPS_REPORT] chart_story update failed: %s", exc)
        return out
    if snapshot is None:
        return out
    out = {
        "macro_trend": snapshot.macro_trend,
        "micro_trend": snapshot.micro_trend,
        "higher_trend": snapshot.higher_trend,
        "structure_bias": _safe_float(snapshot.structure_bias),
        "volatility_state": snapshot.volatility_state,
        "summary": snapshot.summary,
        "major_levels": snapshot.major_levels,
        "pattern_summary": snapshot.pattern_summary,
    }
    return out


def _extract_candle_ranges(candles: list[dict[str, Any]], bars: int) -> tuple[float, float]:
    if not candles:
        return 0.0, 0.0
    window = candles[-min(len(candles), max(1, bars)) :]
    highs: list[float] = []
    lows: list[float] = []
    for row in window:
        if not isinstance(row, dict):
            continue
        highs.append(_safe_float(row.get("high"), 0.0))
        lows.append(_safe_float(row.get("low"), 0.0))
    if not highs or not lows:
        return 0.0, 0.0
    return max(highs), min(lows)


def _extract_price_snapshot(
    *,
    factors: dict[str, dict[str, Any]],
    story: dict[str, Any],
    forecast: dict[str, Any],
    policy: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    fac_m1 = factors.get("M1") or {}
    fac_h4 = factors.get("H4") or {}

    current_price = _safe_float(fac_m1.get("close"), 0.0)
    atr_pips = max(0.0, _safe_float(fac_m1.get("atr_pips"), 0.0))
    m1_gap_pips = (_safe_float(fac_m1.get("ma10")) - _safe_float(fac_m1.get("ma20"))) / 0.01
    h4_gap_pips = (_safe_float(fac_h4.get("ma10")) - _safe_float(fac_h4.get("ma20"))) / 0.01
    vol_5m = _safe_float(fac_m1.get("vol_5m"), 0.0)

    m1_candles = fac_m1.get("candles") if isinstance(fac_m1.get("candles"), list) else []
    high_3h, low_3h = _extract_candle_ranges(m1_candles, bars=180)
    high_24h, low_24h = _extract_candle_ranges(m1_candles, bars=1440)
    range_3h_pips = max(0.0, (high_3h - low_3h) / 0.01) if high_3h and low_3h else 0.0
    range_24h_pips = max(0.0, (high_24h - low_24h) / 0.01) if high_24h and low_24h else 0.0

    support_price, resistance_price = _extract_levels(story, current_price) if current_price > 0 else (0.0, 0.0)

    forecast_ref = forecast.get("reference") if isinstance(forecast.get("reference"), dict) else {}
    p_up = _safe_float(forecast_ref.get("p_up"), 0.5)
    edge = _safe_float(forecast_ref.get("edge"), 0.0)

    base_trend = (
        0.80 * math.tanh(h4_gap_pips / max(4.0, atr_pips * 1.8, 4.0))
        + 0.50 * math.tanh(m1_gap_pips / max(2.0, atr_pips, 2.0))
    )
    if forecast_ref:
        forecast_bias = (p_up - 0.5) * 2.0
        edge_scale = 0.5 + 0.5 * _clamp(edge, 0.0, 1.0)
        direction_score = 0.55 * base_trend + 0.45 * forecast_bias * edge_scale
    else:
        direction_score = base_trend
    direction_score = _clamp(direction_score, -1.0, 1.0)

    range_position = 0.5
    if current_price > 0 and high_24h > low_24h:
        range_position = _clamp((current_price - low_24h) / (high_24h - low_24h), 0.0, 1.0)

    volatility_state = "high" if atr_pips >= 8.5 else ("low" if atr_pips <= 3.5 else "normal")
    if vol_5m >= 1.45 and volatility_state != "high":
        volatility_state = "high"

    snapshot = {
        "instrument": "USD_JPY",
        "current_price": round(current_price, 3) if current_price > 0 else None,
        "atr_pips": round(atr_pips, 3),
        "range_3h_pips": round(range_3h_pips, 3),
        "range_24h_pips": round(range_24h_pips, 3),
        "range_position_24h": round(range_position, 3),
        "micro_regime": str(fac_m1.get("regime") or ""),
        "macro_regime": str(fac_h4.get("regime") or ""),
        "micro_rsi": round(_safe_float(fac_m1.get("rsi"), 0.0), 3),
        "micro_adx": round(_safe_float(fac_m1.get("adx"), 0.0), 3),
        "macro_adx": round(_safe_float(fac_h4.get("adx"), 0.0), 3),
        "ma_gap_m1_pips": round(m1_gap_pips, 3),
        "ma_gap_h4_pips": round(h4_gap_pips, 3),
        "volatility_state": volatility_state,
        "vol_5m": round(vol_5m, 3),
        "support_price": support_price if support_price > 0 else None,
        "resistance_price": resistance_price if resistance_price > 0 else None,
        "policy_event_lock": bool(policy.get("event_lock", False)),
        "policy_range_mode": bool(policy.get("range_mode", False)),
    }
    return snapshot, direction_score


def _sqlite_rows(path: Path, query: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    con: Optional[sqlite3.Connection] = None
    try:
        con = sqlite3.connect(str(path))
        con.row_factory = sqlite3.Row
        cur = con.execute(query, params)
        rows = cur.fetchall()
        return [dict(row) for row in rows]
    except Exception as exc:
        logging.warning("[OPS_REPORT] sqlite query failed path=%s err=%s", path, exc)
        return []
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:
                pass


def _load_trade_rows(path: Path, *, hours: float) -> list[dict[str, Any]]:
    query = """
        SELECT pocket, strategy_tag, close_reason, units, pl_pips, realized_pl, close_time
        FROM trades
        WHERE close_time IS NOT NULL
          AND julianday(close_time) >= julianday('now', ?)
    """
    return _sqlite_rows(path, query, (f"-{max(0.1, float(hours)):.3f} hours",))


def _calc_bucket_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    trade_count = 0
    wins = 0
    losses = 0
    gross_win_pips = 0.0
    gross_loss_pips = 0.0
    total_pips = 0.0
    total_jpy = 0.0
    for row in rows:
        pips = _safe_float(row.get("pl_pips"), 0.0)
        jpy = _safe_float(row.get("realized_pl"), 0.0)
        trade_count += 1
        total_pips += pips
        total_jpy += jpy
        if pips > 0:
            wins += 1
            gross_win_pips += pips
        elif pips < 0:
            losses += 1
            gross_loss_pips += abs(pips)
    win_rate = (wins / trade_count) if trade_count else 0.0
    if gross_loss_pips > 1e-9:
        pf = gross_win_pips / gross_loss_pips
    else:
        pf = None if gross_win_pips > 0 else 0.0
    return {
        "trade_count": trade_count,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(pf, 4) if isinstance(pf, (int, float)) else pf,
        "total_pips": round(total_pips, 3),
        "total_jpy": round(total_jpy, 3),
    }


def _summarize_trades(rows: list[dict[str, Any]], *, hours: float) -> dict[str, Any]:
    by_pocket_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        pocket = str(row.get("pocket") or "unknown").strip().lower() or "unknown"
        by_pocket_rows.setdefault(pocket, []).append(row)

    by_pocket = {pocket: _calc_bucket_metrics(bucket) for pocket, bucket in by_pocket_rows.items()}
    return {
        "window_hours": round(float(hours), 3),
        "overall": _calc_bucket_metrics(rows),
        "by_pocket": by_pocket,
    }


def _summarize_orders(path: Path, *, hours: float) -> dict[str, Any]:
    rows = _sqlite_rows(
        path,
        """
        SELECT status, error_code
        FROM orders
        WHERE julianday(ts) >= julianday('now', ?)
        """,
        (f"-{max(0.1, float(hours)):.3f} hours",),
    )
    total = len(rows)
    status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    failed = 0
    for row in rows:
        status = str(row.get("status") or "").strip().lower() or "unknown"
        status_counts[status] = status_counts.get(status, 0) + 1
        error_code = str(row.get("error_code") or "").strip().lower()
        is_failed = status in REJECT_STATUSES or bool(error_code)
        if is_failed:
            failed += 1
            reason = error_code or status
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    by_status = [
        {"status": status, "count": count}
        for status, count in sorted(status_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ][:10]
    top_fail_reasons = [
        {"reason": reason, "count": count}
        for reason, count in sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ][:8]

    reject_rate = (failed / total) if total > 0 else 0.0
    return {
        "window_hours": round(float(hours), 3),
        "total_orders": total,
        "failed_orders": failed,
        "reject_rate": round(reject_rate, 4),
        "by_status": by_status,
        "top_fail_reasons": top_fail_reasons,
    }


def _load_events(path: Path, *, now_utc: datetime) -> list[dict[str, Any]]:
    payload = _read_json(path)
    if payload is None:
        return []
    rows_raw = payload.get("events")
    if rows_raw is None and isinstance(payload.get("calendar"), list):
        rows_raw = payload.get("calendar")
    if rows_raw is None and isinstance(payload, dict):
        rows_raw = payload.get("rows")
    if not isinstance(rows_raw, list):
        return []

    normalized: list[dict[str, Any]] = []
    for row in rows_raw:
        if not isinstance(row, dict):
            continue
        event_time = None
        for key, tz in (
            ("time_utc", UTC),
            ("timestamp_utc", UTC),
            ("utc", UTC),
            ("time_jst", JST),
            ("timestamp_jst", JST),
            ("jst", JST),
            ("time", UTC),
            ("timestamp", UTC),
            ("at", UTC),
            ("epoch", UTC),
        ):
            if key not in row:
                continue
            event_time = _parse_iso_datetime(row.get(key), default_tz=tz)
            if event_time is not None:
                break
        if event_time is None:
            continue

        minutes_to_event = int(round((event_time - now_utc).total_seconds() / 60.0))
        if minutes_to_event < -60 or minutes_to_event > 24 * 60:
            continue
        name = (
            str(row.get("name") or row.get("title") or row.get("event") or "event")
            .strip()
            or "event"
        )
        impact = str(row.get("impact") or row.get("level") or "medium").strip().lower()
        normalized.append(
            {
                "name": name,
                "impact": impact,
                "time_utc": _to_iso(event_time),
                "time_jst": _to_jst_label(event_time),
                "minutes_to_event": minutes_to_event,
            }
        )
    normalized.sort(key=lambda item: item["minutes_to_event"])
    return normalized


def _load_policy_overlay(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    return payload if payload is not None else {}


def _event_context(*, events: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    event_lock = bool(policy.get("event_lock", False))
    next_event = None
    soon = False
    active = False
    if events:
        for event in events:
            minutes = _safe_int(event.get("minutes_to_event"), 999999)
            if minutes >= 0 and next_event is None:
                next_event = event
            if 0 <= minutes <= 180:
                soon = True
            if -30 <= minutes <= 45:
                active = True
    if event_lock:
        soon = True
        active = True
    return {
        "event_lock": event_lock,
        "event_soon": soon,
        "event_active_window": active,
        "next_event": next_event,
        "events": events[:6],
    }


def _build_scenarios(
    *,
    direction_score: float,
    snapshot: dict[str, Any],
    forecast: dict[str, Any],
    performance: dict[str, Any],
    order_stats: dict[str, Any],
    event_ctx: dict[str, Any],
) -> list[dict[str, Any]]:
    score = _clamp(direction_score, -1.0, 1.0)
    uncertainty = 1.0 - abs(score)
    uncertainty += min(0.25, _safe_float(order_stats.get("reject_rate"), 0.0) * 0.8)
    if event_ctx.get("event_soon"):
        uncertainty += 0.20
    uncertainty = _clamp(uncertainty, 0.0, 1.0)

    two_way_prob = _clamp(0.16 + uncertainty * 0.42, 0.12, 0.62)
    residual = max(0.0, 1.0 - two_way_prob)
    up_share = _clamp(0.5 + score * 0.45, 0.08, 0.92)
    up_prob = residual * up_share
    down_prob = residual - up_prob

    p_up = _safe_float(((forecast.get("reference") or {}).get("p_up")), 0.5)
    edge = _safe_float(((forecast.get("reference") or {}).get("edge")), 0.0)

    support_price = _safe_float(snapshot.get("support_price"), 0.0)
    resistance_price = _safe_float(snapshot.get("resistance_price"), 0.0)
    current_price = _safe_float(snapshot.get("current_price"), 0.0)

    # Keep percentages stable and sum to 100.0.
    pct_up = round(up_prob * 100.0, 1)
    pct_down = round(down_prob * 100.0, 1)
    pct_two_way = round(max(0.0, 100.0 - pct_up - pct_down), 1)

    scenarios = [
        {
            "key": "continuation_up",
            "title": "A. USD/JPY continuation higher",
            "probability_pct": pct_up,
            "bias": "long_usd_jpy",
            "triggers": [
                f"Price holds above support ({support_price:.3f})",
                f"Forecast p_up stays >= 0.55 (now {p_up:.3f})",
                "Breakout legs are accepted after 5-15m close confirmation",
            ],
            "invalidations": [
                f"5m closes back below support ({support_price:.3f})",
                f"Forecast edge falls below 0.08 (now {edge:.3f})",
            ],
            "short_term_plan": "Prefer pullback-then-resume entries while support is defended.",
            "swing_plan": "Keep bullish bias only while H4 structure remains above support/pivot.",
        },
        {
            "key": "reversal_down",
            "title": "B. USD/JPY pullback / JPY rebound",
            "probability_pct": pct_down,
            "bias": "short_usd_jpy",
            "triggers": [
                f"Price repeatedly fails near resistance ({resistance_price:.3f})",
                "Rebound attempts lose momentum and M1/H1 slope turns down",
                "Risk-off tone or event miss pushes quick downside follow-through",
            ],
            "invalidations": [
                f"5m closes above resistance ({resistance_price:.3f})",
                "Forecast p_up reclaims >= 0.58 with expanding edge",
            ],
            "short_term_plan": "Sell failed breakouts; avoid chasing first spike.",
            "swing_plan": "Only keep bearish swing if H4 momentum stays below neutral slope.",
        },
        {
            "key": "event_two_way",
            "title": "C. Event-driven two-way volatility",
            "probability_pct": pct_two_way,
            "bias": "two_way_wait_for_confirmation",
            "triggers": [
                "Upcoming event window or elevated uncertainty",
                "Large wick behavior and fast mean-revert after initial spike",
                "Execution quality degrades (reject/spread noise rises)",
            ],
            "invalidations": [
                "Volatility compresses and trend leg remains directional for >30m",
            ],
            "short_term_plan": "Reduce size or wait for second move confirmation post spike.",
            "swing_plan": "Stay light until event uncertainty clears and H4 direction stabilizes.",
        },
    ]
    return scenarios


def _build_short_term_playbook(
    *,
    scenarios: list[dict[str, Any]],
    snapshot: dict[str, Any],
    event_ctx: dict[str, Any],
) -> dict[str, Any]:
    primary = max(scenarios, key=lambda row: _safe_float(row.get("probability_pct"), 0.0)) if scenarios else {}
    bias = str(primary.get("bias") or "neutral")
    mode = "post_event_confirmation" if event_ctx.get("event_soon") else "normal"

    support = _safe_float(snapshot.get("support_price"), 0.0)
    resistance = _safe_float(snapshot.get("resistance_price"), 0.0)
    current = _safe_float(snapshot.get("current_price"), 0.0)
    watch_zone = ""
    if support > 0 and resistance > 0:
        watch_zone = f"{support:.3f} - {resistance:.3f}"
    elif current > 0:
        watch_zone = f"{current - 0.15:.3f} - {current + 0.15:.3f}"

    return {
        "horizon": "now_to_72h",
        "mode": mode,
        "bias": bias,
        "primary_scenario": str(primary.get("title") or ""),
        "watch_zone": watch_zone,
        "execution_rules": [
            "Do not chase first spike during event window.",
            "Use close-based confirmation (5m/15m) before breakout follow entries.",
            "Skip mid-range entries; focus on edge zones only.",
        ],
    }


def _build_swing_playbook(
    *,
    scenarios: list[dict[str, Any]],
    snapshot: dict[str, Any],
    performance: dict[str, Any],
) -> dict[str, Any]:
    primary = max(scenarios, key=lambda row: _safe_float(row.get("probability_pct"), 0.0)) if scenarios else {}
    macro_regime = str(snapshot.get("macro_regime") or "")
    volatility_state = str(snapshot.get("volatility_state") or "normal")
    total_pf = (performance.get("overall") or {}).get("profit_factor")
    confidence_note = "stable"
    if isinstance(total_pf, (int, float)) and total_pf < 0.95:
        confidence_note = "cautious"
    elif volatility_state == "high":
        confidence_note = "high_volatility"
    return {
        "horizon": "3d_to_3w",
        "bias": str(primary.get("bias") or "neutral"),
        "macro_regime": macro_regime,
        "confidence_note": confidence_note,
        "management_rules": [
            "Scale in only after H4 confirmation, not from M1 noise.",
            "Reduce carry when volatility_state=high and reject_rate is elevated.",
            "Re-evaluate thesis after each major event or structural level break.",
        ],
    }


def _build_risk_protocol(
    *,
    event_ctx: dict[str, Any],
    order_stats: dict[str, Any],
) -> dict[str, Any]:
    per_trade_loss_pct = _clamp(_safe_float(os.getenv("OPS_PLAYBOOK_MAX_LOSS_PCT", 0.8), 0.8), 0.1, 3.0)
    theme_cap_pct = _clamp(_safe_float(os.getenv("OPS_PLAYBOOK_THEME_CAP_PCT", 2.4), 2.4), 0.5, 8.0)
    max_positions = max(1, _safe_int(os.getenv("OPS_PLAYBOOK_MAX_POSITIONS", 3), 3))
    reject_rate = _safe_float(order_stats.get("reject_rate"), 0.0)
    execution_guard = "normal"
    if reject_rate >= 0.25:
        execution_guard = "tighten_size_and_retry_policy"
    elif reject_rate >= 0.12:
        execution_guard = "slightly_reduce_size"
    return {
        "max_loss_per_trade_pct": round(per_trade_loss_pct, 3),
        "max_theme_exposure_pct": round(theme_cap_pct, 3),
        "max_concurrent_positions": max_positions,
        "event_mode": "reduce_or_wait" if event_ctx.get("event_soon") else "normal",
        "execution_quality_guard": execution_guard,
    }


def build_ops_report(
    *,
    hours: float,
    factors: dict[str, dict[str, Any]],
    forecast: dict[str, Any],
    performance: dict[str, Any],
    order_stats: dict[str, Any],
    policy: dict[str, Any],
    events: list[dict[str, Any]],
    now_utc: Optional[datetime] = None,
) -> dict[str, Any]:
    now_utc = now_utc or _utcnow()
    fac_m1 = factors.get("M1") or {}
    fac_h4 = factors.get("H4") or {}
    story = _load_chart_story(fac_m1, fac_h4)
    snapshot, direction_score = _extract_price_snapshot(
        factors=factors,
        story=story,
        forecast=forecast,
        policy=policy,
    )
    event_ctx = _event_context(events=events, policy=policy)
    scenarios = _build_scenarios(
        direction_score=direction_score,
        snapshot=snapshot,
        forecast=forecast,
        performance=performance,
        order_stats=order_stats,
        event_ctx=event_ctx,
    )
    short_term = _build_short_term_playbook(
        scenarios=scenarios,
        snapshot=snapshot,
        event_ctx=event_ctx,
    )
    swing = _build_swing_playbook(
        scenarios=scenarios,
        snapshot=snapshot,
        performance=performance,
    )
    risk_protocol = _build_risk_protocol(event_ctx=event_ctx, order_stats=order_stats)

    confidence = round((1.0 - _clamp(1.0 - abs(direction_score), 0.0, 1.0)) * 100.0, 1)
    if event_ctx.get("event_soon"):
        confidence = max(20.0, round(confidence - 18.0, 1))

    report = {
        "generated_at": _to_iso(now_utc),
        "llm_disabled": True,
        "hours": round(float(hours), 3),
        "playbook_version": 1,
        "note": "Deterministic market playbook generated from local factors + DB metrics.",
        "direction_score": round(direction_score, 4),
        "direction_confidence_pct": confidence,
        "snapshot": snapshot,
        "forecast": forecast,
        "performance": performance,
        "order_quality": order_stats,
        "event_context": event_ctx,
        "short_term": short_term,
        "swing": swing,
        "scenarios": scenarios,
        "risk_protocol": risk_protocol,
        "data_sources": {
            "factors_ready": bool(fac_m1) and bool(fac_h4),
            "trades_window_count": _safe_int((performance.get("overall") or {}).get("trade_count"), 0),
            "orders_window_count": _safe_int(order_stats.get("total_orders"), 0),
            "events_count": len(events),
            "policy_overlay_present": bool(policy),
        },
    }
    return report


def _render_markdown(report: dict[str, Any]) -> str:
    snapshot = report.get("snapshot") if isinstance(report.get("snapshot"), dict) else {}
    short_term = report.get("short_term") if isinstance(report.get("short_term"), dict) else {}
    swing = report.get("swing") if isinstance(report.get("swing"), dict) else {}
    scenarios = report.get("scenarios") if isinstance(report.get("scenarios"), list) else []
    risk = report.get("risk_protocol") if isinstance(report.get("risk_protocol"), dict) else {}
    event_ctx = report.get("event_context") if isinstance(report.get("event_context"), dict) else {}
    perf = report.get("performance") if isinstance(report.get("performance"), dict) else {}
    perf_overall = perf.get("overall") if isinstance(perf.get("overall"), dict) else {}

    lines: list[str] = []
    lines.append("# Deterministic Market Playbook")
    lines.append("")
    lines.append(f"- generated_at: {report.get('generated_at')}")
    lines.append(f"- direction_score: {report.get('direction_score')}")
    lines.append(f"- direction_confidence_pct: {report.get('direction_confidence_pct')}")
    lines.append("")
    lines.append("## Snapshot")
    lines.append(f"- instrument: {snapshot.get('instrument')}")
    lines.append(f"- current_price: {snapshot.get('current_price')}")
    lines.append(f"- atr_pips: {snapshot.get('atr_pips')}")
    lines.append(f"- range_24h_pips: {snapshot.get('range_24h_pips')}")
    lines.append(f"- macro_regime/micro_regime: {snapshot.get('macro_regime')} / {snapshot.get('micro_regime')}")
    lines.append(f"- support/resistance: {snapshot.get('support_price')} / {snapshot.get('resistance_price')}")
    lines.append("")
    lines.append("## Short-Term (now-72h)")
    lines.append(f"- bias: {short_term.get('bias')}")
    lines.append(f"- primary_scenario: {short_term.get('primary_scenario')}")
    lines.append(f"- watch_zone: {short_term.get('watch_zone')}")
    rules = short_term.get("execution_rules") if isinstance(short_term.get("execution_rules"), list) else []
    for rule in rules:
        lines.append(f"- rule: {rule}")
    lines.append("")
    lines.append("## Swing (3d-3w)")
    lines.append(f"- bias: {swing.get('bias')}")
    lines.append(f"- macro_regime: {swing.get('macro_regime')}")
    lines.append(f"- confidence_note: {swing.get('confidence_note')}")
    mgr = swing.get("management_rules") if isinstance(swing.get("management_rules"), list) else []
    for rule in mgr:
        lines.append(f"- rule: {rule}")
    lines.append("")
    lines.append("## Scenario Map")
    for row in scenarios:
        if not isinstance(row, dict):
            continue
        lines.append(f"- {row.get('title')} ({row.get('probability_pct')}%)")
        lines.append(f"  - bias: {row.get('bias')}")
        triggers = row.get("triggers") if isinstance(row.get("triggers"), list) else []
        if triggers:
            lines.append(f"  - trigger: {triggers[0]}")
    lines.append("")
    lines.append("## Risk Protocol")
    lines.append(f"- max_loss_per_trade_pct: {risk.get('max_loss_per_trade_pct')}")
    lines.append(f"- max_theme_exposure_pct: {risk.get('max_theme_exposure_pct')}")
    lines.append(f"- max_concurrent_positions: {risk.get('max_concurrent_positions')}")
    lines.append(f"- event_mode: {risk.get('event_mode')}")
    lines.append(f"- execution_quality_guard: {risk.get('execution_quality_guard')}")
    lines.append("")
    lines.append("## Performance Window")
    lines.append(f"- trades: {perf_overall.get('trade_count')}")
    lines.append(f"- win_rate: {perf_overall.get('win_rate')}")
    lines.append(f"- profit_factor: {perf_overall.get('profit_factor')}")
    lines.append(f"- total_pips: {perf_overall.get('total_pips')}")
    lines.append("")
    lines.append("## Upcoming Events")
    events = event_ctx.get("events") if isinstance(event_ctx.get("events"), list) else []
    if not events:
        lines.append("- none")
    else:
        for event in events[:5]:
            if not isinstance(event, dict):
                continue
            lines.append(
                f"- {event.get('time_jst')} | {event.get('name')} | impact={event.get('impact')} | t={event.get('minutes_to_event')}m"
            )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Ops report (deterministic playbook, LLM disabled)")
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--output", default="logs/gpt_ops_report.json")
    ap.add_argument("--markdown-output", default=os.getenv("GPT_OPS_REPORT_MD_OUTPUT", "logs/gpt_ops_report.md"))
    ap.add_argument("--trades-db", default=os.getenv("OPS_PLAYBOOK_TRADES_DB", "logs/trades.db"))
    ap.add_argument("--orders-db", default=os.getenv("OPS_PLAYBOOK_ORDERS_DB", "logs/orders.db"))
    ap.add_argument("--overlay-path", default=os.getenv("POLICY_OVERLAY_PATH", "logs/policy_overlay.json"))
    ap.add_argument("--events-path", default=os.getenv("OPS_PLAYBOOK_EVENTS_PATH", "logs/market_events.json"))
    ap.add_argument("--forecast-strategy-tag", default=os.getenv("OPS_PLAYBOOK_FORECAST_STRATEGY_TAG", "scalp_ping_5s_b_live"))
    ap.add_argument("--forecast-pocket", default=os.getenv("OPS_PLAYBOOK_FORECAST_POCKET", "scalp"))
    ap.add_argument("--forecast-units", type=int, default=_safe_int(os.getenv("OPS_PLAYBOOK_FORECAST_UNITS", "10000"), 10000))
    ap.add_argument("--policy", action="store_true")
    ap.add_argument("--policy-output", default="logs/policy_diff_ops.json")
    ap.add_argument("--apply-policy", action="store_true")
    ap.add_argument("--gpt", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args, _ = ap.parse_known_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    now_utc = _utcnow()

    if args.gpt:
        logging.info("[OPS_REPORT] --gpt requested, but LLM execution is disabled. Running deterministic mode.")

    factors = _load_factors()
    forecast = _load_forecast_snapshot(
        strategy_tag=str(args.forecast_strategy_tag or "").strip(),
        pocket=str(args.forecast_pocket or "scalp").strip() or "scalp",
        units=max(1, int(args.forecast_units)),
    )
    performance = _summarize_trades(
        _load_trade_rows(Path(args.trades_db), hours=max(0.1, float(args.hours))),
        hours=max(0.1, float(args.hours)),
    )
    order_stats = _summarize_orders(Path(args.orders_db), hours=max(0.1, float(args.hours)))
    policy = _load_policy_overlay(Path(args.overlay_path))
    events = _load_events(Path(args.events_path), now_utc=now_utc)

    payload = build_ops_report(
        hours=max(0.1, float(args.hours)),
        factors=factors,
        forecast=forecast,
        performance=performance,
        order_stats=order_stats,
        policy=policy,
        events=events,
        now_utc=now_utc,
    )

    out_path = Path(args.output)
    _write_json(out_path, payload)
    logging.info("[OPS_REPORT] wrote %s", out_path)

    md_output = str(args.markdown_output or "").strip().lower()
    if md_output and md_output not in {"off", "none", "-", "false", "0"}:
        md_path = Path(str(args.markdown_output))
        _write_text(md_path, _render_markdown(payload))
        logging.info("[OPS_REPORT] wrote %s", md_path)

    if args.policy or args.apply_policy:
        primary = payload.get("short_term", {}).get("primary_scenario")
        bias = payload.get("short_term", {}).get("bias")
        diff = normalize_policy_diff(
            {
                "no_change": True,
                "source": "ops_playbook",
                "reason": "deterministic_playbook_only",
                "notes": {
                    "generated_at": payload.get("generated_at"),
                    "primary_scenario": primary,
                    "bias": bias,
                },
            },
            source="ops_playbook",
        )
        policy_path = Path(args.policy_output)
        _write_json(policy_path, diff)
        logging.info("[OPS_POLICY] wrote %s", policy_path)
        if args.apply_policy:
            logging.info("[OPS_POLICY] apply requested, but deterministic playbook does not auto-apply policy.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
