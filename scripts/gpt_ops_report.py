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

from analytics.policy_apply import apply_policy_diff_to_paths
from analytics.policy_diff import normalize_policy_diff, validate_policy_diff

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

PAIR_ALIASES = {
    "USD_JPY": {"USD_JPY", "USDJPY"},
    "EUR_USD": {"EUR_USD", "EURUSD"},
    "AUD_JPY": {"AUD_JPY", "AUDJPY"},
    "EUR_JPY": {"EUR_JPY", "EURJPY"},
}
_PLAYBOOK_FACTOR_MAX_AGE_SEC_DEFAULT = 900.0


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


_PLAYBOOK_FACTOR_MAX_AGE_SEC = max(
    30.0,
    _safe_float(os.getenv("OPS_PLAYBOOK_FACTOR_MAX_AGE_SEC", str(_PLAYBOOK_FACTOR_MAX_AGE_SEC_DEFAULT)), _PLAYBOOK_FACTOR_MAX_AGE_SEC_DEFAULT),
)


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


def _safe_float_or_none(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _canonical_pair_key(raw: object) -> str:
    token = str(raw or "").strip().upper().replace("-", "_")
    if not token:
        return ""
    if "_" not in token and len(token) == 6 and token.isalpha():
        token = f"{token[:3]}_{token[3:]}"
    for canonical, aliases in PAIR_ALIASES.items():
        if token in aliases:
            return canonical
    return token


def _find_number(row: dict[str, Any], keys: tuple[str, ...]) -> Optional[float]:
    for key in keys:
        if key not in row:
            continue
        value = _safe_float_or_none(row.get(key))
        if value is not None:
            return value
    return None


def _extract_pair_entry(payload: dict[str, Any], pair: str) -> dict[str, Any]:
    aliases = {pair, pair.replace("_", ""), pair.lower(), pair.replace("_", "").lower()}
    pairs_raw = payload.get("pairs") if isinstance(payload.get("pairs"), dict) else {}
    candidate: object = None

    for key in aliases:
        if isinstance(pairs_raw, dict) and key in pairs_raw:
            candidate = pairs_raw.get(key)
            break
        if key in payload:
            candidate = payload.get(key)
            break
    if candidate is None and isinstance(pairs_raw, dict):
        for key, value in pairs_raw.items():
            if _canonical_pair_key(key) == pair:
                candidate = value
                break

    price = None
    change_pct_24h = None
    if isinstance(candidate, dict):
        price = _find_number(candidate, ("price", "close", "value", "rate", "last"))
        change_pct_24h = _find_number(
            candidate,
            (
                "change_pct_24h",
                "change_24h_pct",
                "change_pct",
                "pct_change",
                "change24h",
            ),
        )
    else:
        price = _safe_float_or_none(candidate)

    return {
        "price": round(price, 6) if isinstance(price, (int, float)) else None,
        "change_pct_24h": round(change_pct_24h, 6) if isinstance(change_pct_24h, (int, float)) else None,
    }


def _extract_rate_entry(payload: dict[str, Any], aliases: tuple[str, ...]) -> Optional[float]:
    rates = payload.get("rates") if isinstance(payload.get("rates"), dict) else {}
    for key in aliases:
        if key in rates:
            value = _safe_float_or_none(rates.get(key))
            if value is not None:
                return value
        if key in payload:
            value = _safe_float_or_none(payload.get(key))
            if value is not None:
                return value
    return None


def _extract_risk_entry(payload: dict[str, Any], aliases: tuple[str, ...]) -> Optional[float]:
    risk = payload.get("risk") if isinstance(payload.get("risk"), dict) else {}
    for key in aliases:
        if key in risk:
            value = _safe_float_or_none(risk.get(key))
            if value is not None:
                return value
        if key in payload:
            value = _safe_float_or_none(payload.get(key))
            if value is not None:
                return value
    return None


def _load_macro_snapshot(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    return payload if payload is not None else {}


def _infer_risk_mode(*, vix: Optional[float], us500_change_pct: Optional[float]) -> str:
    if vix is not None and vix >= 23.0:
        return "risk_off"
    if us500_change_pct is not None and us500_change_pct <= -0.6:
        return "risk_off"
    if vix is not None and vix <= 17.0 and us500_change_pct is not None and us500_change_pct >= 0.4:
        return "risk_on"
    if vix is not None or us500_change_pct is not None:
        return "neutral"
    return "unknown"


def _build_market_context(
    *,
    factors: dict[str, dict[str, Any]],
    events: list[dict[str, Any]],
    now_utc: datetime,
    external_snapshot: dict[str, Any],
    macro_snapshot: dict[str, Any],
) -> dict[str, Any]:
    fac_m1 = factors.get("M1") or {}

    pairs: dict[str, dict[str, Any]] = {}
    for pair in ("USD_JPY", "EUR_USD", "AUD_JPY", "EUR_JPY"):
        row = _extract_pair_entry(external_snapshot, pair)
        source = "external"
        if pair == "USD_JPY" and row.get("price") is None:
            m1_price = _safe_float_or_none(fac_m1.get("close"))
            if m1_price is not None and m1_price > 0.0:
                row["price"] = round(m1_price, 6)
                source = "factor_cache"
        if row.get("price") is None:
            source = "missing"
        row["source"] = source
        pairs[pair.lower()] = row

    dxy_external = _extract_risk_entry(external_snapshot, ("dxy", "DXY"))
    dxy_change = _extract_risk_entry(
        external_snapshot,
        ("dxy_change_pct_24h", "dxy_change_pct", "dxy_24h_change_pct", "dxy_change"),
    )
    dxy_macro = _safe_float_or_none(macro_snapshot.get("dxy"))
    dxy = dxy_external if dxy_external is not None else dxy_macro
    dxy_source = "external" if dxy_external is not None else ("macro_snapshot" if dxy_macro is not None else "missing")

    us10y = _extract_rate_entry(external_snapshot, ("US10Y", "US_10Y", "UST10Y", "us10y", "us_10y"))
    jp10y = _extract_rate_entry(external_snapshot, ("JP10Y", "JP_10Y", "JGB10Y", "jp10y", "jp_10y"))
    us_jp_10y_spread = (us10y - jp10y) if us10y is not None and jp10y is not None else None

    macro_yield = macro_snapshot.get("yield2y") if isinstance(macro_snapshot.get("yield2y"), dict) else {}
    usd2y_proxy = _safe_float_or_none(macro_yield.get("USD"))
    jpy2y_proxy = _safe_float_or_none(macro_yield.get("JPY"))
    us_jp_2y_proxy_spread = (
        (usd2y_proxy - jpy2y_proxy)
        if usd2y_proxy is not None and jpy2y_proxy is not None
        else None
    )

    vix_external = _extract_risk_entry(external_snapshot, ("vix", "VIX"))
    vix_macro = _safe_float_or_none(macro_snapshot.get("vix"))
    vix = vix_external if vix_external is not None else vix_macro
    vix_source = "external" if vix_external is not None else ("macro_snapshot" if vix_macro is not None else "missing")

    us500_change_pct = _extract_risk_entry(
        external_snapshot,
        ("us500_change_pct", "spx_change_pct", "sp500_change_pct", "us_equity_change_pct"),
    )
    risk_mode = _infer_risk_mode(vix=vix, us500_change_pct=us500_change_pct)

    high_impact = [e for e in events if str(e.get("impact") or "").lower() in {"high", "red"}]
    next_event = None
    for event in events:
        minutes = _safe_int(event.get("minutes_to_event"), 999999)
        if minutes >= 0:
            next_event = event
            break

    return {
        "generated_at": _to_iso(now_utc),
        "pairs": pairs,
        "dollar": {
            "dxy": round(dxy, 6) if isinstance(dxy, (int, float)) else None,
            "dxy_change_pct_24h": round(dxy_change, 6) if isinstance(dxy_change, (int, float)) else None,
            "source": dxy_source,
        },
        "rates": {
            "us10y": round(us10y, 6) if isinstance(us10y, (int, float)) else None,
            "jp10y": round(jp10y, 6) if isinstance(jp10y, (int, float)) else None,
            "us_jp_10y_spread": round(us_jp_10y_spread, 6) if isinstance(us_jp_10y_spread, (int, float)) else None,
            "usd2y_proxy": round(usd2y_proxy, 6) if isinstance(usd2y_proxy, (int, float)) else None,
            "jpy2y_proxy": round(jpy2y_proxy, 6) if isinstance(jpy2y_proxy, (int, float)) else None,
            "us_jp_2y_proxy_spread": round(us_jp_2y_proxy_spread, 6)
            if isinstance(us_jp_2y_proxy_spread, (int, float))
            else None,
        },
        "risk": {
            "vix": round(vix, 6) if isinstance(vix, (int, float)) else None,
            "vix_source": vix_source,
            "us500_change_pct": round(us500_change_pct, 6)
            if isinstance(us500_change_pct, (int, float))
            else None,
            "mode": risk_mode,
        },
        "events_summary": {
            "total_events": len(events),
            "high_impact_events": len(high_impact),
            "next_event": next_event,
        },
        "source_flags": {
            "external_snapshot": bool(external_snapshot),
            "macro_snapshot": bool(macro_snapshot),
            "factor_cache": bool(fac_m1),
            "events": bool(events),
        },
    }


def _load_or_build_market_context(
    *,
    context_path: Path,
    factors: dict[str, dict[str, Any]],
    events: list[dict[str, Any]],
    now_utc: datetime,
    external_path: Path,
    macro_snapshot_path: Path,
) -> dict[str, Any]:
    external_snapshot = _read_json(external_path) or {}
    macro_snapshot = _load_macro_snapshot(macro_snapshot_path)
    payload = _build_market_context(
        factors=factors,
        events=events,
        now_utc=now_utc,
        external_snapshot=external_snapshot,
        macro_snapshot=macro_snapshot,
    )
    _write_json(context_path, payload)
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


def _extract_market_pair_price(market_context: Optional[dict[str, Any]], pair: str) -> Optional[float]:
    if not isinstance(market_context, dict):
        return None
    pairs = market_context.get("pairs") if isinstance(market_context.get("pairs"), dict) else {}
    row = pairs.get(pair.lower()) if isinstance(pairs, dict) else None
    if not isinstance(row, dict):
        return None
    return _safe_float_or_none(row.get("price"))


def _factor_timestamp_from_row(row: dict[str, Any]) -> Optional[datetime]:
    for key in ("timestamp", "ts", "time", "datetime", "asof"):
        if key not in row:
            continue
        parsed = _parse_iso_datetime(row.get(key))
        if parsed is not None:
            return parsed
    return None


def _build_factor_freshness(fac_m1: dict[str, Any], *, now_utc: datetime) -> dict[str, Any]:
    ts = _factor_timestamp_from_row(fac_m1)
    age_sec: Optional[float]
    if ts is None:
        age_sec = None
    else:
        age_sec = max(0.0, (now_utc - ts).total_seconds())
    stale = bool(not fac_m1) or age_sec is None or age_sec > _PLAYBOOK_FACTOR_MAX_AGE_SEC
    return {
        "stale": stale,
        "age_sec": round(age_sec, 3) if isinstance(age_sec, (int, float)) else None,
        "timestamp_utc": _to_iso(ts) if ts is not None else None,
        "max_age_sec": _PLAYBOOK_FACTOR_MAX_AGE_SEC,
    }


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
    market_context: Optional[dict[str, Any]],
    factor_freshness: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    fac_m1 = factors.get("M1") or {}
    fac_h4 = factors.get("H4") or {}

    factor_stale = bool(factor_freshness.get("stale"))
    factor_age_sec = _safe_float_or_none(factor_freshness.get("age_sec"))
    factor_timestamp_utc = str(factor_freshness.get("timestamp_utc") or "")
    max_factor_age_sec = _safe_float(factor_freshness.get("max_age_sec"), _PLAYBOOK_FACTOR_MAX_AGE_SEC)

    factor_close = _safe_float_or_none(fac_m1.get("close"))
    market_close = _extract_market_pair_price(market_context, "USD_JPY")
    use_market_price = bool(
        isinstance(market_close, (int, float))
        and market_close > 0.0
        and (factor_stale or not isinstance(factor_close, (int, float)) or factor_close <= 0.0)
    )
    current_price = float(market_close) if use_market_price else _safe_float(fac_m1.get("close"), 0.0)
    current_price_source = "external_snapshot" if use_market_price else ("factor_cache" if current_price > 0 else "missing")

    atr_pips = max(0.0, _safe_float(fac_m1.get("atr_pips"), 0.0)) if not factor_stale else 0.0
    m1_gap_pips = (
        (_safe_float(fac_m1.get("ma10")) - _safe_float(fac_m1.get("ma20"))) / 0.01
        if not factor_stale
        else 0.0
    )
    h4_gap_pips = (
        (_safe_float(fac_h4.get("ma10")) - _safe_float(fac_h4.get("ma20"))) / 0.01
        if not factor_stale
        else 0.0
    )
    vol_5m = _safe_float(fac_m1.get("vol_5m"), 0.0) if not factor_stale else 0.0

    m1_candles = fac_m1.get("candles") if (not factor_stale and isinstance(fac_m1.get("candles"), list)) else []
    high_3h, low_3h = _extract_candle_ranges(m1_candles, bars=180)
    high_24h, low_24h = _extract_candle_ranges(m1_candles, bars=1440)
    range_3h_pips = max(0.0, (high_3h - low_3h) / 0.01) if high_3h and low_3h else 0.0
    range_24h_pips = max(0.0, (high_24h - low_24h) / 0.01) if high_24h and low_24h else 0.0

    support_price, resistance_price = _extract_levels(story, current_price) if (current_price > 0 and not factor_stale) else (0.0, 0.0)
    if current_price > 0 and (support_price <= 0.0 or resistance_price <= 0.0):
        support_price = round(current_price - 0.15, 3)
        resistance_price = round(current_price + 0.15, 3)

    forecast_ref = forecast.get("reference") if isinstance(forecast.get("reference"), dict) else {}
    p_up = _safe_float(forecast_ref.get("p_up"), 0.5)
    edge = _safe_float(forecast_ref.get("edge"), 0.0)

    base_trend = 0.0
    if not factor_stale:
        base_trend = (
            0.80 * math.tanh(h4_gap_pips / max(4.0, atr_pips * 1.8, 4.0))
            + 0.50 * math.tanh(m1_gap_pips / max(2.0, atr_pips, 2.0))
        )
    if forecast_ref and not factor_stale:
        forecast_bias = (p_up - 0.5) * 2.0
        edge_scale = 0.5 + 0.5 * _clamp(edge, 0.0, 1.0)
        direction_score = 0.55 * base_trend + 0.45 * forecast_bias * edge_scale
    elif forecast_ref:
        forecast_bias = (p_up - 0.5) * 2.0
        direction_score = 0.20 * forecast_bias
    else:
        direction_score = base_trend
    direction_score = _clamp(direction_score, -1.0, 1.0)

    range_position = 0.5
    if current_price > 0 and high_24h > low_24h and not factor_stale:
        range_position = _clamp((current_price - low_24h) / (high_24h - low_24h), 0.0, 1.0)

    volatility_state = "unknown" if factor_stale else ("high" if atr_pips >= 8.5 else ("low" if atr_pips <= 3.5 else "normal"))
    if not factor_stale and vol_5m >= 1.45 and volatility_state != "high":
        volatility_state = "high"

    snapshot = {
        "instrument": "USD_JPY",
        "current_price": round(current_price, 3) if current_price > 0 else None,
        "current_price_source": current_price_source,
        "atr_pips": round(atr_pips, 3),
        "range_3h_pips": round(range_3h_pips, 3),
        "range_24h_pips": round(range_24h_pips, 3),
        "range_position_24h": round(range_position, 3),
        "micro_regime": str(fac_m1.get("regime") or "stale"),
        "macro_regime": str(fac_h4.get("regime") or "stale"),
        "micro_rsi": round(_safe_float(fac_m1.get("rsi"), 0.0), 3) if not factor_stale else None,
        "micro_adx": round(_safe_float(fac_m1.get("adx"), 0.0), 3) if not factor_stale else None,
        "macro_adx": round(_safe_float(fac_h4.get("adx"), 0.0), 3) if not factor_stale else None,
        "ma_gap_m1_pips": round(m1_gap_pips, 3),
        "ma_gap_h4_pips": round(h4_gap_pips, 3),
        "volatility_state": volatility_state,
        "vol_5m": round(vol_5m, 3),
        "support_price": support_price if support_price > 0 else None,
        "resistance_price": resistance_price if resistance_price > 0 else None,
        "factor_stale": factor_stale,
        "factor_age_m1_sec": round(factor_age_sec, 1) if isinstance(factor_age_sec, (int, float)) else None,
        "factor_timestamp_utc": factor_timestamp_utc or None,
        "factor_max_age_sec": round(max_factor_age_sec, 1),
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


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def _get_pair_metric(
    market_context: dict[str, Any], pair: str, field: str, default: float = 0.0
) -> float:
    pairs = market_context.get("pairs") if isinstance(market_context.get("pairs"), dict) else {}
    row = pairs.get(pair.lower()) if isinstance(pairs, dict) else None
    if not isinstance(row, dict):
        return default
    return _safe_float(row.get(field), default)


def _build_driver_breakdown(
    *,
    market_context: dict[str, Any],
    event_ctx: dict[str, Any],
) -> dict[str, Any]:
    rates = market_context.get("rates") if isinstance(market_context.get("rates"), dict) else {}
    risk = market_context.get("risk") if isinstance(market_context.get("risk"), dict) else {}
    dollar = market_context.get("dollar") if isinstance(market_context.get("dollar"), dict) else {}

    spread_10y = _safe_float_or_none(rates.get("us_jp_10y_spread"))
    spread_2y = _safe_float_or_none(rates.get("us_jp_2y_proxy_spread"))

    rate_score = 0.0
    rate_detail = "missing"
    if spread_10y is not None:
        rate_score = math.tanh((spread_10y - 1.8) / 1.1)
        rate_detail = f"us_jp_10y_spread={spread_10y:.3f}"
    elif spread_2y is not None:
        rate_score = math.tanh((spread_2y - 0.2) / 0.25)
        rate_detail = f"us_jp_2y_proxy_spread={spread_2y:.3f}"

    jpy_cross_changes = [
        _get_pair_metric(market_context, "USD_JPY", "change_pct_24h", 0.0),
        _get_pair_metric(market_context, "AUD_JPY", "change_pct_24h", 0.0),
        _get_pair_metric(market_context, "EUR_JPY", "change_pct_24h", 0.0),
    ]
    jpy_cross_changes = [x for x in jpy_cross_changes if abs(x) > 1e-9]
    yen_flow = _avg(jpy_cross_changes)

    dxy_change = _safe_float_or_none(dollar.get("dxy_change_pct_24h"))
    eurusd_change = _get_pair_metric(market_context, "EUR_USD", "change_pct_24h", 0.0)
    dollar_signals: list[float] = []
    if dxy_change is not None:
        dollar_signals.append(dxy_change)
    if abs(eurusd_change) > 1e-9:
        dollar_signals.append(-eurusd_change)
    dollar_flow = _avg(dollar_signals)

    contradiction = yen_flow - dollar_flow
    yen_structural = 0.0
    if not jpy_cross_changes:
        usdjpy = _get_pair_metric(market_context, "USD_JPY", "price", 0.0)
        dxy_level = _safe_float_or_none(dollar.get("dxy"))
        if usdjpy >= 154.0 and dxy_level is not None and dxy_level <= 100.0:
            yen_structural = 0.25
    yen_flow_score = _clamp(0.55 * math.tanh(yen_flow / 0.8) + 0.45 * math.tanh(contradiction / 0.8) + yen_structural, -1.0, 1.0)

    risk_mode = str(risk.get("mode") or "unknown").lower()
    us500_change = _safe_float_or_none(risk.get("us500_change_pct"))
    risk_score = 0.0
    if risk_mode == "risk_off":
        risk_score = -0.7
    elif risk_mode == "risk_on":
        risk_score = 0.35
    elif us500_change is not None:
        risk_score = math.tanh(us500_change / 1.0) * 0.35

    if event_ctx.get("event_soon"):
        risk_score *= 0.7

    components = {
        "rate_diff": 0.44 * rate_score,
        "yen_flow": 0.36 * yen_flow_score,
        "risk_sentiment": 0.20 * risk_score,
    }
    net_score = _clamp(sum(components.values()), -1.0, 1.0)
    dominant_key = max(components, key=lambda key: abs(components[key]))
    dominant_sign = "bullish_usd_jpy" if components[dominant_key] >= 0.0 else "bearish_usd_jpy"

    available_count = 0
    if spread_10y is not None or spread_2y is not None:
        available_count += 1
    if jpy_cross_changes or dxy_change is not None or abs(eurusd_change) > 1e-9:
        available_count += 1
    if risk_mode != "unknown" or us500_change is not None:
        available_count += 1
    confidence = round(_clamp(0.35 + 0.22 * available_count, 0.2, 1.0), 3)

    return {
        "dominant_driver": dominant_key,
        "dominant_sign": dominant_sign,
        "net_score": round(net_score, 4),
        "confidence": confidence,
        "components": {k: round(v, 4) for k, v in components.items()},
        "notes": {
            "rate_detail": rate_detail,
            "yen_flow_avg_pct": round(yen_flow, 4),
            "dollar_flow_avg_pct": round(dollar_flow, 4),
            "contradiction_score": round(contradiction, 4),
            "risk_mode": risk_mode,
        },
    }


def _blend_direction_score(*, base_score: float, driver: dict[str, Any]) -> float:
    net = _safe_float(driver.get("net_score"), 0.0)
    confidence = _safe_float(driver.get("confidence"), 0.0)
    blend_weight = _clamp(0.16 + 0.24 * confidence, 0.12, 0.40)
    return _clamp((1.0 - blend_weight) * base_score + blend_weight * net, -1.0, 1.0)


def _build_break_points(
    *,
    snapshot: dict[str, Any],
    event_ctx: dict[str, Any],
    market_context: dict[str, Any],
    driver: dict[str, Any],
    order_stats: dict[str, Any],
) -> list[dict[str, Any]]:
    support = _safe_float(snapshot.get("support_price"), 0.0)
    resistance = _safe_float(snapshot.get("resistance_price"), 0.0)
    rates = market_context.get("rates") if isinstance(market_context.get("rates"), dict) else {}
    spread_10y = _safe_float_or_none(rates.get("us_jp_10y_spread"))
    spread_2y = _safe_float_or_none(rates.get("us_jp_2y_proxy_spread"))
    reject_rate = _safe_float(order_stats.get("reject_rate"), 0.0)

    out: list[dict[str, Any]] = []
    if support > 0.0:
        out.append(
            {
                "key": "support_failure",
                "condition": f"5m close below support {support:.3f}",
                "impact": "invalidate_long_bias",
            }
        )
    if resistance > 0.0:
        out.append(
            {
                "key": "resistance_break",
                "condition": f"5m close above resistance {resistance:.3f}",
                "impact": "invalidate_short_bias",
            }
        )
    if spread_10y is not None:
        out.append(
            {
                "key": "rate_spread_flip",
                "condition": f"US-JP 10Y spread drops below {max(0.0, spread_10y - 0.35):.3f}",
                "impact": "carry_tailwind_weakens",
            }
        )
    elif spread_2y is not None:
        out.append(
            {
                "key": "rate_proxy_flip",
                "condition": f"US-JP 2Y proxy spread drops below {max(0.0, spread_2y - 0.08):.3f}",
                "impact": "macro_tailwind_weakens",
            }
        )
    if reject_rate >= 0.12:
        out.append(
            {
                "key": "execution_degrade",
                "condition": f"reject_rate >= {reject_rate:.3f}",
                "impact": "reduce_size_or_standby",
            }
        )
    next_event = event_ctx.get("next_event") if isinstance(event_ctx.get("next_event"), dict) else {}
    minutes = _safe_int(next_event.get("minutes_to_event"), 999999) if next_event else 999999
    if next_event and minutes <= 180:
        out.append(
            {
                "key": "event_window",
                "condition": f"{next_event.get('name')} in {minutes}m",
                "impact": "avoid_first_spike",
            }
        )

    dominant = str(driver.get("dominant_driver") or "")
    if dominant:
        out.append(
            {
                "key": "driver_shift",
                "condition": f"dominant_driver changes from {dominant}",
                "impact": "rebuild_scenarios",
            }
        )
    if bool(snapshot.get("factor_stale")):
        out.append(
            {
                "key": "factor_stale",
                "condition": "M1 factor cache is stale or missing",
                "impact": "reduce_size_or_wait_for_data_refresh",
            }
        )
    return out


def _build_if_then_rules(
    *,
    scenarios: list[dict[str, Any]],
    snapshot: dict[str, Any],
    event_ctx: dict[str, Any],
    driver: dict[str, Any],
    break_points: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    primary = max(scenarios, key=lambda row: _safe_float(row.get("probability_pct"), 0.0)) if scenarios else {}
    primary_bias = str(primary.get("bias") or "neutral")
    support = _safe_float(snapshot.get("support_price"), 0.0)
    resistance = _safe_float(snapshot.get("resistance_price"), 0.0)
    dominant_driver = str(driver.get("dominant_driver") or "unknown")
    event_soon = bool(event_ctx.get("event_soon"))
    factor_stale = bool(snapshot.get("factor_stale"))

    event_rule = "wait_for_post_spike_confirmation"
    if not event_soon:
        event_rule = "normal_execution"

    bp_key_text = ",".join(str(row.get("key")) for row in break_points[:3] if isinstance(row, dict))

    return [
        {
            "id": "st_data_freshness",
            "horizon": "short_term",
            "if": f"factor_stale={str(factor_stale).lower()}",
            "then": "reduce_size_or_wait_for_fresh_factor",
        },
        {
            "id": "st_event_mode",
            "horizon": "short_term",
            "if": f"event_soon={str(event_soon).lower()}",
            "then": event_rule,
        },
        {
            "id": "st_primary_follow",
            "horizon": "short_term",
            "if": f"primary_bias={primary_bias} and 5m close confirms level",
            "then": "enter_in_primary_direction_with_split_take_profit",
        },
        {
            "id": "st_fail_fast",
            "horizon": "short_term",
            "if": f"any_break_point_hit ({bp_key_text or 'none'})",
            "then": "cut_position_and_recompute",
        },
        {
            "id": "sw_driver_hold",
            "horizon": "swing",
            "if": f"dominant_driver={dominant_driver} persists and H4 structure holds",
            "then": "keep_bias_and_add_on_pullback_only",
        },
        {
            "id": "sw_level_flip",
            "horizon": "swing",
            "if": f"support {support:.3f} or resistance {resistance:.3f} is structurally broken",
            "then": "reduce_or_flip_swing_bias",
        },
    ]


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
    if bool(snapshot.get("factor_stale")):
        uncertainty += 0.24
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
    market_context: Optional[dict[str, Any]] = None,
    now_utc: Optional[datetime] = None,
) -> dict[str, Any]:
    now_utc = now_utc or _utcnow()
    fac_m1 = factors.get("M1") or {}
    fac_h4 = factors.get("H4") or {}
    event_ctx = _event_context(events=events, policy=policy)
    if not isinstance(market_context, dict):
        market_context = _build_market_context(
            factors=factors,
            events=events,
            now_utc=now_utc,
            external_snapshot={},
            macro_snapshot={},
        )
    story = _load_chart_story(fac_m1, fac_h4)
    factor_freshness = _build_factor_freshness(fac_m1, now_utc=now_utc)
    snapshot, direction_score = _extract_price_snapshot(
        factors=factors,
        story=story,
        forecast=forecast,
        policy=policy,
        market_context=market_context,
        factor_freshness=factor_freshness,
    )
    if bool(snapshot.get("factor_stale")):
        direction_score *= 0.45
    driver_breakdown = _build_driver_breakdown(market_context=market_context, event_ctx=event_ctx)
    direction_score = _blend_direction_score(base_score=direction_score, driver=driver_breakdown)
    scenarios = _build_scenarios(
        direction_score=direction_score,
        snapshot=snapshot,
        forecast=forecast,
        performance=performance,
        order_stats=order_stats,
        event_ctx=event_ctx,
    )
    break_points = _build_break_points(
        snapshot=snapshot,
        event_ctx=event_ctx,
        market_context=market_context,
        driver=driver_breakdown,
        order_stats=order_stats,
    )
    if_then_rules = _build_if_then_rules(
        scenarios=scenarios,
        snapshot=snapshot,
        event_ctx=event_ctx,
        driver=driver_breakdown,
        break_points=break_points,
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

    base_conf = 1.0 - _clamp(1.0 - abs(direction_score), 0.0, 1.0)
    driver_conf = _safe_float(driver_breakdown.get("confidence"), 0.0)
    confidence = round(_clamp(base_conf * 0.72 + driver_conf * 0.28, 0.0, 1.0) * 100.0, 1)
    if event_ctx.get("event_soon"):
        confidence = max(20.0, round(confidence - 18.0, 1))
    if bool(snapshot.get("factor_stale")):
        confidence = max(15.0, round(confidence - 22.0, 1))

    report = {
        "generated_at": _to_iso(now_utc),
        "llm_disabled": True,
        "hours": round(float(hours), 3),
        "playbook_version": 2,
        "note": "Deterministic market playbook (driver -> breakpoints -> scenarios -> if-then).",
        "direction_score": round(direction_score, 4),
        "direction_confidence_pct": confidence,
        "market_context": market_context,
        "driver_breakdown": driver_breakdown,
        "snapshot": snapshot,
        "forecast": forecast,
        "performance": performance,
        "order_quality": order_stats,
        "event_context": event_ctx,
        "break_points": break_points,
        "short_term": short_term,
        "swing": swing,
        "scenarios": scenarios,
        "if_then_rules": if_then_rules,
        "risk_protocol": risk_protocol,
        "data_sources": {
            "factors_ready": bool(fac_m1) and bool(fac_h4),
            "factors_m1_stale": bool(snapshot.get("factor_stale")),
            "factors_m1_age_sec": snapshot.get("factor_age_m1_sec"),
            "trades_window_count": _safe_int((performance.get("overall") or {}).get("trade_count"), 0),
            "orders_window_count": _safe_int(order_stats.get("total_orders"), 0),
            "events_count": len(events),
            "policy_overlay_present": bool(policy),
            "market_context_ready": bool(market_context.get("pairs")) if isinstance(market_context, dict) else False,
        },
    }
    return report


def _render_markdown(report: dict[str, Any]) -> str:
    market_context = report.get("market_context") if isinstance(report.get("market_context"), dict) else {}
    driver = report.get("driver_breakdown") if isinstance(report.get("driver_breakdown"), dict) else {}
    snapshot = report.get("snapshot") if isinstance(report.get("snapshot"), dict) else {}
    short_term = report.get("short_term") if isinstance(report.get("short_term"), dict) else {}
    swing = report.get("swing") if isinstance(report.get("swing"), dict) else {}
    scenarios = report.get("scenarios") if isinstance(report.get("scenarios"), list) else []
    break_points = report.get("break_points") if isinstance(report.get("break_points"), list) else []
    if_then_rules = report.get("if_then_rules") if isinstance(report.get("if_then_rules"), list) else []
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
    lines.append(f"- current_price_source: {snapshot.get('current_price_source')}")
    lines.append(f"- atr_pips: {snapshot.get('atr_pips')}")
    lines.append(f"- range_24h_pips: {snapshot.get('range_24h_pips')}")
    lines.append(f"- macro_regime/micro_regime: {snapshot.get('macro_regime')} / {snapshot.get('micro_regime')}")
    lines.append(f"- factor_stale: {snapshot.get('factor_stale')}")
    lines.append(f"- factor_age_m1_sec: {snapshot.get('factor_age_m1_sec')}")
    lines.append(f"- support/resistance: {snapshot.get('support_price')} / {snapshot.get('resistance_price')}")
    pairs = market_context.get("pairs") if isinstance(market_context.get("pairs"), dict) else {}
    usdjpy = pairs.get("usd_jpy") if isinstance(pairs.get("usd_jpy"), dict) else {}
    eurusd = pairs.get("eur_usd") if isinstance(pairs.get("eur_usd"), dict) else {}
    audjpy = pairs.get("aud_jpy") if isinstance(pairs.get("aud_jpy"), dict) else {}
    eurjpy = pairs.get("eur_jpy") if isinstance(pairs.get("eur_jpy"), dict) else {}
    lines.append(
        "- pairs(USD/JPY EUR/USD AUD/JPY EUR/JPY): "
        f"{usdjpy.get('price')} / {eurusd.get('price')} / {audjpy.get('price')} / {eurjpy.get('price')}"
    )
    dollar = market_context.get("dollar") if isinstance(market_context.get("dollar"), dict) else {}
    rates = market_context.get("rates") if isinstance(market_context.get("rates"), dict) else {}
    lines.append(f"- dxy: {dollar.get('dxy')} ({dollar.get('source')})")
    lines.append(
        "- rates(US10Y/JP10Y/spread): "
        f"{rates.get('us10y')} / {rates.get('jp10y')} / {rates.get('us_jp_10y_spread')}"
    )
    lines.append("")
    lines.append("## Driver")
    lines.append(f"- dominant_driver: {driver.get('dominant_driver')}")
    lines.append(f"- dominant_sign: {driver.get('dominant_sign')}")
    lines.append(f"- net_score: {driver.get('net_score')}")
    comp = driver.get("components") if isinstance(driver.get("components"), dict) else {}
    lines.append(
        "- components(rate/yen_flow/risk): "
        f"{comp.get('rate_diff')} / {comp.get('yen_flow')} / {comp.get('risk_sentiment')}"
    )
    notes = driver.get("notes") if isinstance(driver.get("notes"), dict) else {}
    lines.append(
        "- contradiction(yen_flow - dollar_flow): "
        f"{notes.get('yen_flow_avg_pct')} - {notes.get('dollar_flow_avg_pct')} = {notes.get('contradiction_score')}"
    )
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
    lines.append("## Break Points")
    if not break_points:
        lines.append("- none")
    else:
        for row in break_points[:6]:
            if not isinstance(row, dict):
                continue
            lines.append(f"- {row.get('key')}: {row.get('condition')} -> {row.get('impact')}")
    lines.append("")
    lines.append("## If-Then Rules")
    if not if_then_rules:
        lines.append("- none")
    else:
        for row in if_then_rules[:8]:
            if not isinstance(row, dict):
                continue
            lines.append(f"- [{row.get('horizon')}] if {row.get('if')} then {row.get('then')}")
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


def _short_term_bias_to_policy_bias(raw_bias: object) -> str:
    token = str(raw_bias or "").strip().lower()
    if token in {"long_usd_jpy", "long", "buy"}:
        return "long"
    if token in {"short_usd_jpy", "short", "sell"}:
        return "short"
    return "neutral"


def _scenario_probability_map(scenarios: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in scenarios:
        if not isinstance(row, dict):
            continue
        key = str(row.get("key") or "").strip().lower()
        if not key:
            continue
        out[key] = round(_clamp(_safe_float(row.get("probability_pct"), 0.0), 0.0, 100.0), 3)
    return out


def _deep_subset_equal(base: object, patch: object) -> bool:
    if isinstance(patch, dict):
        if not isinstance(base, dict):
            return False
        return all(_deep_subset_equal(base.get(key), value) for key, value in patch.items())
    if isinstance(patch, (int, float)) and isinstance(base, (int, float)):
        return abs(float(base) - float(patch)) <= 1e-9
    return base == patch


def _build_policy_diff_from_report(
    *,
    report: dict[str, Any],
    current_policy: dict[str, Any],
    now_utc: datetime,
) -> dict[str, Any]:
    short_term = report.get("short_term") if isinstance(report.get("short_term"), dict) else {}
    event_ctx = report.get("event_context") if isinstance(report.get("event_context"), dict) else {}
    snapshot = report.get("snapshot") if isinstance(report.get("snapshot"), dict) else {}
    performance = report.get("performance") if isinstance(report.get("performance"), dict) else {}
    order_quality = report.get("order_quality") if isinstance(report.get("order_quality"), dict) else {}
    scenarios = report.get("scenarios") if isinstance(report.get("scenarios"), list) else []

    direction_score = round(_clamp(_safe_float(report.get("direction_score"), 0.0), -1.0, 1.0), 4)
    direction_conf_pct = round(_clamp(_safe_float(report.get("direction_confidence_pct"), 0.0), 0.0, 100.0), 2)
    scenario_probs = _scenario_probability_map(scenarios)
    sorted_probs = sorted(scenario_probs.values(), reverse=True)
    primary_prob = sorted_probs[0] if sorted_probs else 0.0
    secondary_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
    prob_gap = round(max(0.0, primary_prob - secondary_prob), 3)

    event_soon = bool(event_ctx.get("event_soon"))
    event_active = bool(event_ctx.get("event_active_window"))
    factor_stale = bool(snapshot.get("factor_stale"))
    reject_rate = _clamp(_safe_float(order_quality.get("reject_rate"), 0.0), 0.0, 1.0)
    total_pf = _safe_float_or_none(((performance.get("overall") or {}).get("profit_factor")))

    min_conf_pct = _clamp(
        _safe_float(os.getenv("OPS_PLAYBOOK_POLICY_MIN_CONF_PCT", "58"), 58.0),
        0.0,
        100.0,
    )
    min_prob_gap_pct = _clamp(
        _safe_float(os.getenv("OPS_PLAYBOOK_POLICY_MIN_PROB_GAP_PCT", "10"), 10.0),
        0.0,
        100.0,
    )
    reject_warn_threshold = _clamp(
        _safe_float(os.getenv("OPS_PLAYBOOK_POLICY_REJECT_WARN_THRESHOLD", "0.12"), 0.12),
        0.01,
        0.9,
    )
    reject_block_threshold = _clamp(
        _safe_float(os.getenv("OPS_PLAYBOOK_POLICY_REJECT_BLOCK_THRESHOLD", "0.28"), 0.28),
        reject_warn_threshold,
        0.95,
    )

    scenario_two_way = _clamp(_safe_float(scenario_probs.get("event_two_way"), 0.0) / 100.0, 0.0, 1.0)
    uncertainty = _clamp(
        max(
            scenario_two_way,
            1.0 - direction_conf_pct / 100.0,
            min(1.0, reject_rate * 1.6),
            0.25 if factor_stale else 0.0,
            0.18 if event_soon else 0.0,
        ),
        0.0,
        1.0,
    )

    raw_bias = _short_term_bias_to_policy_bias(short_term.get("bias"))
    directional_bias_allowed = (
        raw_bias in {"long", "short"}
        and direction_conf_pct >= min_conf_pct
        and prob_gap >= min_prob_gap_pct
        and not event_soon
        and not factor_stale
    )
    policy_bias = raw_bias if directional_bias_allowed else "neutral"
    if isinstance(total_pf, (int, float)) and total_pf < 0.80:
        policy_bias = "neutral"

    allow_new = not (event_active or factor_stale or reject_rate >= reject_block_threshold)
    require_retest = bool(event_soon or uncertainty >= 0.55)
    spread_ok = reject_rate < reject_warn_threshold
    drift_ok = not factor_stale

    pocket_confidence = _clamp(direction_conf_pct / 100.0, 0.05, 0.98)
    if policy_bias == "neutral":
        pocket_confidence = _clamp(pocket_confidence * 0.78, 0.05, 0.98)
    if isinstance(total_pf, (int, float)) and total_pf < 0.95:
        pocket_confidence = _clamp(pocket_confidence * 0.88, 0.05, 0.98)
        require_retest = True
    pocket_confidence = round(pocket_confidence, 4)

    micro_regime = str(snapshot.get("micro_regime") or "").strip().lower()
    range_mode = bool(
        "range" in micro_regime or policy_bias == "neutral" and uncertainty >= 0.50 or event_soon
    )

    pocket_patch: dict[str, dict[str, Any]] = {}
    for pocket in ("macro", "micro", "scalp"):
        pocket_bias = policy_bias if pocket in {"micro", "scalp"} else "neutral"
        pocket_patch[pocket] = {
            "enabled": True,
            "bias": pocket_bias,
            "confidence": pocket_confidence,
            "entry_gates": {
                "allow_new": allow_new,
                "require_retest": require_retest,
                "spread_ok": spread_ok,
                "drift_ok": drift_ok,
            },
        }

    patch = {
        "air_score": direction_score,
        "uncertainty": round(uncertainty, 4),
        "event_lock": bool(event_active),
        "range_mode": range_mode,
        "pockets": pocket_patch,
    }

    no_change = _deep_subset_equal(current_policy, patch)
    if no_change:
        reason = "playbook_no_delta"
    elif not allow_new:
        reason = "playbook_temporal_lock"
    elif policy_bias == "neutral":
        reason = "playbook_neutral_bias"
    else:
        reason = "playbook_directional_bias"

    diff_input: dict[str, Any] = {
        "source": "ops_playbook",
        "no_change": no_change,
        "reason": reason,
        "notes": {
            "generated_at": report.get("generated_at"),
            "computed_at": _to_iso(now_utc),
            "playbook_version": report.get("playbook_version"),
            "primary_scenario": short_term.get("primary_scenario"),
            "short_term_bias": short_term.get("bias"),
            "policy_bias": policy_bias,
            "direction_confidence_pct": direction_conf_pct,
            "scenario_probabilities_pct": scenario_probs,
            "scenario_primary_gap_pct": prob_gap,
            "event_soon": event_soon,
            "event_active_window": event_active,
            "factor_stale": factor_stale,
            "reject_rate": round(reject_rate, 4),
            "profit_factor": total_pf,
        },
    }
    if not no_change:
        diff_input["patch"] = patch

    diff = normalize_policy_diff(diff_input, source="ops_playbook")
    errors = validate_policy_diff(diff)
    if errors:
        logging.error("[OPS_POLICY] invalid generated policy diff: %s", ", ".join(errors))
        return normalize_policy_diff(
            {
                "source": "ops_playbook",
                "no_change": True,
                "reason": "generated_diff_invalid",
                "notes": {
                    "generated_at": report.get("generated_at"),
                    "errors": errors,
                },
            },
            source="ops_playbook",
        )
    return diff


def main() -> int:
    ap = argparse.ArgumentParser(description="Ops report (deterministic playbook, LLM disabled)")
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--output", default="logs/gpt_ops_report.json")
    ap.add_argument("--markdown-output", default=os.getenv("GPT_OPS_REPORT_MD_OUTPUT", "logs/gpt_ops_report.md"))
    ap.add_argument("--trades-db", default=os.getenv("OPS_PLAYBOOK_TRADES_DB", "logs/trades.db"))
    ap.add_argument("--orders-db", default=os.getenv("OPS_PLAYBOOK_ORDERS_DB", "logs/orders.db"))
    ap.add_argument("--overlay-path", default=os.getenv("POLICY_OVERLAY_PATH", "logs/policy_overlay.json"))
    ap.add_argument("--events-path", default=os.getenv("OPS_PLAYBOOK_EVENTS_PATH", "logs/market_events.json"))
    ap.add_argument(
        "--market-context-path",
        default=os.getenv("OPS_PLAYBOOK_MARKET_CONTEXT_PATH", "logs/market_context_latest.json"),
    )
    ap.add_argument(
        "--market-external-path",
        default=os.getenv("OPS_PLAYBOOK_EXTERNAL_SNAPSHOT_PATH", "logs/market_external_snapshot.json"),
    )
    ap.add_argument(
        "--macro-snapshot-path",
        default=os.getenv("OPS_PLAYBOOK_MACRO_SNAPSHOT_PATH", "fixtures/macro_snapshots/latest.json"),
    )
    ap.add_argument("--forecast-strategy-tag", default=os.getenv("OPS_PLAYBOOK_FORECAST_STRATEGY_TAG", "scalp_ping_5s_b_live"))
    ap.add_argument("--forecast-pocket", default=os.getenv("OPS_PLAYBOOK_FORECAST_POCKET", "scalp"))
    ap.add_argument("--forecast-units", type=int, default=_safe_int(os.getenv("OPS_PLAYBOOK_FORECAST_UNITS", "10000"), 10000))
    ap.add_argument("--policy", action="store_true")
    ap.add_argument("--policy-output", default="logs/policy_diff_ops.json")
    ap.add_argument("--apply-policy", action="store_true")
    ap.add_argument("--policy-history-dir", default=os.getenv("POLICY_HISTORY_DIR", "logs/policy_history"))
    ap.add_argument("--policy-latest-path", default=os.getenv("POLICY_LATEST_PATH", "logs/policy_latest.json"))
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
    market_context = _load_or_build_market_context(
        context_path=Path(args.market_context_path),
        factors=factors,
        events=events,
        now_utc=now_utc,
        external_path=Path(args.market_external_path),
        macro_snapshot_path=Path(args.macro_snapshot_path),
    )

    payload = build_ops_report(
        hours=max(0.1, float(args.hours)),
        factors=factors,
        forecast=forecast,
        performance=performance,
        order_stats=order_stats,
        policy=policy,
        events=events,
        market_context=market_context,
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
        diff = _build_policy_diff_from_report(
            report=payload,
            current_policy=policy if isinstance(policy, dict) else {},
            now_utc=now_utc,
        )
        policy_path = Path(args.policy_output)
        _write_json(policy_path, diff)
        logging.info("[OPS_POLICY] wrote %s", policy_path)
        if args.apply_policy:
            _, changed, flags = apply_policy_diff_to_paths(
                diff,
                overlay_path=Path(args.overlay_path),
                history_dir=Path(args.policy_history_dir),
                latest_path=Path(args.policy_latest_path),
            )
            logging.info(
                "[OPS_POLICY] applied=%s reentry=%s tuning=%s overlay=%s",
                changed,
                flags.get("reentry"),
                flags.get("tuning"),
                args.overlay_path,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
