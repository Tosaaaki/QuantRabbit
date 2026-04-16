#!/usr/bin/env python3
"""
Trend Bot — Automated trend continuation entry bot.

Detects M5/M15/H1 trend alignment, follows band-walk / follow-through moves,
and uses MARKET only when live continuation quality is high, including MICRO reload bites.
Exits remain owned by the trader task and inventory director.

Usage:
    python3 tools/trend_bot.py
    python3 tools/trend_bot.py --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from bot_policy import (
    get_pair_policy,
    global_status_allows_new_entries,
    load_policy,
    pair_policy_worker_block_reason,
)
from market_state import get_market_state
from range_bot import (
    LOG_FILE,
    MAX_BOT_MARGIN_PCT,
    POISON_HOURS_UTC,
    BOT_TAG as RANGE_BOT_TAG,
    BOT_MARKET_TAG as RANGE_BOT_MARKET_TAG,
    PAIR_CURRENCIES,
    build_currency_pulse,
    calculate_units,
    cancel_order,
    current_execution_price,
    describe_trade_conflicts,
    describe_pending_conflicts,
    dynamic_bot_budget_profile,
    discretionary_open_trade_conflicts,
    discretionary_pending_conflicts_for_worker,
    estimate_margin,
    fetch_account,
    fetch_open_trades,
    fetch_pending_orders,
    fetch_prices,
    fetch_recent_worker_close_events,
    format_price,
    get_tag,
    is_entry_pending_order,
    load_config,
    load_technicals,
    oanda_api,
    recent_close_cooldown,
    stop_is_already_broken,
    worker_disaster_stop,
)
from range_scalp_scanner import PAIRS, TYPICAL_SPREADS, get_spread, pip_size, to_pips
import brake_gate
from worker_target_race import (
    build_plan as build_target_race_plan,
    encode_comment as encode_target_race_comment,
    extract_trade_id_from_order_result,
    remember_trade_plan,
)


TREND_BOT_MARKET_TAG = "trend_bot_market"
TREND_H1_ADX_MIN = 18
TREND_M15_ADX_MIN = 22
TREND_M5_ADX_MIN = 17
TREND_H1_DI_GAP = 5
TREND_M15_DI_GAP = 6
TREND_M5_DI_GAP = 4
TREND_M5_BANDWALK_ADX = 24
TREND_M5_BANDWALK_DI_GAP = 6
TREND_PULSE_ALIGN_SCORE = 4.0
TREND_PULSE_BLOCK_SCORE = 4.0
TREND_M1_READY_SCORE = 4
TREND_MAX_SPREAD_MULTIPLE = 1.50
TREND_MICRO_MAX_SPREAD_MULTIPLE = 1.25  # was 1.20 — slightly wider for more MICRO shots
TREND_MIN_RR = 1.25
TREND_MIN_SL_SPREAD_MULTIPLE = 5.0
TREND_MIN_SL_M5_ATR = 1.60
TREND_MIN_SL_H1_ATR = 0.70
TREND_WEAKER_SETUP_SL_BUFFER = 1.10
TREND_BANDWALK_TP_M5_ATR = 2.60
TREND_BANDWALK_TP_H1_ATR = 1.00
TREND_FOLLOWTHROUGH_TP_M5_ATR = 2.10
TREND_FOLLOWTHROUGH_TP_H1_ATR = 0.85
TREND_MIN_TP_RR_MULTIPLE = 1.35
TREND_FAST_TP_RR_MULTIPLE = 1.10
TREND_FAST_TP_M5_ATR = 1.60
TREND_FAST_TP_H1_ATR = 0.55
TREND_FAST_MIN_RR = 1.00
TREND_FAST_ALLOWED_M1_STATES = {"aligned", "reload"}  # was "aligned" only — allow reload for more FAST entries
TREND_MICRO_TP_RR_MULTIPLE = 1.05
TREND_MICRO_TP_M5_ATR = 1.20  # was 1.35 — tighter TP for faster MICRO exits
TREND_MICRO_TP_H1_ATR = 0.40  # was 0.45
TREND_MICRO_MIN_RR = 1.05
TREND_MICRO_ALLOWED_M1_STATES = {"aligned", "reload"}
TREND_MAX_PAIR_TRADES = 2
TREND_MAX_TAG_TRADES = 1
TREND_CANCELABLE_PENDING_TAGS = {RANGE_BOT_TAG, RANGE_BOT_MARKET_TAG}

def pending_margin(order: dict, prices: dict) -> float:
    pair = order.get("instrument", "")
    units = abs(int(order.get("units", 0)))
    price = float(order.get("price", 0))
    if price <= 0:
        quote = prices.get(pair, {})
        price = float(quote.get("ask") or quote.get("bid") or quote.get("mid") or 0)
    return estimate_margin(units, price, pair) if price > 0 and units > 0 else 0.0


def trade_direction(trade: dict) -> str:
    return "BUY" if int(trade.get("currentUnits", 0)) > 0 else "SELL"


def tf_score(direction: str, tf_data: dict, di_gap: float) -> int:
    plus_di = float(tf_data.get("plus_di", 0))
    minus_di = float(tf_data.get("minus_di", 0))
    ema_5 = float(tf_data.get("ema_slope_5", 0))
    ema_10 = float(tf_data.get("ema_slope_10", 0))
    ema_20 = float(tf_data.get("ema_slope_20", 0))
    macd_hist = float(tf_data.get("macd_hist", 0))
    bb_upper = float(tf_data.get("bb_upper", 0))
    bb_lower = float(tf_data.get("bb_lower", 0))
    bb_mid = float(tf_data.get("bb_mid", 0))
    close = float(tf_data.get("close", 0))
    rsi = float(tf_data.get("rsi", 50))

    bb_pos = 0.5
    if bb_upper > bb_lower and close > 0:
        bb_pos = max(0.0, min(1.0, (close - bb_lower) / (bb_upper - bb_lower)))

    score = 0
    if direction == "BUY":
        if plus_di >= minus_di + di_gap:
            score += 3
        elif minus_di >= plus_di + di_gap:
            score -= 3
        score += 1 if ema_5 > 0 else -1 if ema_5 < 0 else 0
        score += 1 if ema_10 > 0 else -1 if ema_10 < 0 else 0
        score += 1 if ema_20 > 0 else -1 if ema_20 < 0 else 0
        score += 1 if macd_hist > 0 else -1 if macd_hist < 0 else 0
        score += 1 if close >= bb_mid else -1
        score += 1 if rsi >= 52 else -1 if rsi <= 48 else 0
        score += 1 if bb_pos >= 0.55 else -1 if bb_pos <= 0.45 else 0
    else:
        if minus_di >= plus_di + di_gap:
            score += 3
        elif plus_di >= minus_di + di_gap:
            score -= 3
        score += 1 if ema_5 < 0 else -1 if ema_5 > 0 else 0
        score += 1 if ema_10 < 0 else -1 if ema_10 > 0 else 0
        score += 1 if ema_20 < 0 else -1 if ema_20 > 0 else 0
        score += 1 if macd_hist < 0 else -1 if macd_hist > 0 else 0
        score += 1 if close <= bb_mid else -1
        score += 1 if rsi <= 48 else -1 if rsi >= 52 else 0
        score += 1 if bb_pos <= 0.45 else -1 if bb_pos >= 0.55 else 0
    return score


def assess_tf_trend(tf_label: str, tf_data: dict | None, min_adx: float, di_gap: float) -> dict:
    result = {
        "status": "missing",
        "direction": None,
        "score": 0,
        "aligned": False,
        "notes": [],
    }
    if not tf_data:
        result["notes"].append(f"{tf_label} missing")
        return result

    adx = float(tf_data.get("adx", 0))
    if adx < min_adx:
        result["status"] = "soft"
        result["notes"].append(f"{tf_label} ADX {adx:.0f} too soft")
        return result

    buy_score = tf_score("BUY", tf_data, di_gap)
    sell_score = tf_score("SELL", tf_data, di_gap)
    if buy_score >= sell_score + 2 and buy_score >= 6:
        result.update(
            status="trend",
            direction="BUY",
            score=buy_score,
            aligned=True,
            notes=[f"{tf_label} trend BUY score={buy_score} ADX={adx:.0f}"],
        )
    elif sell_score >= buy_score + 2 and sell_score >= 6:
        result.update(
            status="trend",
            direction="SELL",
            score=sell_score,
            aligned=True,
            notes=[f"{tf_label} trend SELL score={sell_score} ADX={adx:.0f}"],
        )
    else:
        result.update(
            status="mixed",
            notes=[f"{tf_label} mixed trend score B{buy_score}/S{sell_score} ADX={adx:.0f}"],
        )
    return result


def detect_m5_continuation(direction: str, m5_data: dict | None) -> dict:
    result = {
        "ready": False,
        "setup": "none",
        "strength": 0,
        "notes": [],
    }
    if not m5_data:
        result["notes"].append("M5 missing")
        return result

    adx = float(m5_data.get("adx", 0))
    plus_di = float(m5_data.get("plus_di", 0))
    minus_di = float(m5_data.get("minus_di", 0))
    ema_5 = float(m5_data.get("ema_slope_5", 0))
    ema_10 = float(m5_data.get("ema_slope_10", 0))
    macd_hist = float(m5_data.get("macd_hist", 0))
    bb_upper = float(m5_data.get("bb_upper", 0))
    bb_lower = float(m5_data.get("bb_lower", 0))
    bb_mid = float(m5_data.get("bb_mid", 0))
    close = float(m5_data.get("close", 0))
    upper_wick = float(m5_data.get("upper_wick_avg_pips", 0))
    lower_wick = float(m5_data.get("lower_wick_avg_pips", 0))
    atr_pips = max(float(m5_data.get("atr_pips", 0)), 1.0)

    if bb_upper <= bb_lower or close <= 0:
        result["notes"].append("M5 BB invalid")
        return result

    bb_pos = max(0.0, min(1.0, (close - bb_lower) / (bb_upper - bb_lower)))
    if direction == "BUY":
        band_walk = (
            bb_pos >= 0.78
            and adx >= TREND_M5_BANDWALK_ADX
            and plus_di >= minus_di + TREND_M5_BANDWALK_DI_GAP
            and ema_5 > 0
            and ema_10 >= 0
            and macd_hist >= 0
            and upper_wick <= max(0.2, lower_wick * 1.15)
        )
        follow_through = (
            adx >= TREND_M5_ADX_MIN
            and plus_di >= minus_di + TREND_M5_DI_GAP
            and ema_5 > 0
            and macd_hist >= 0
            and close >= bb_mid
            and bb_pos >= 0.48
        )
    else:
        band_walk = (
            bb_pos <= 0.22
            and adx >= TREND_M5_BANDWALK_ADX
            and minus_di >= plus_di + TREND_M5_BANDWALK_DI_GAP
            and ema_5 < 0
            and ema_10 <= 0
            and macd_hist <= 0
            and lower_wick <= max(0.2, upper_wick * 1.15)
        )
        follow_through = (
            adx >= TREND_M5_ADX_MIN
            and minus_di >= plus_di + TREND_M5_DI_GAP
            and ema_5 < 0
            and macd_hist <= 0
            and close <= bb_mid
            and bb_pos <= 0.52
        )

    if band_walk:
        result.update(ready=True, setup="band_walk", strength=2, notes=[f"M5 {direction} band-walk"])
        return result
    if follow_through:
        result.update(ready=True, setup="follow", strength=1, notes=[f"M5 {direction} follow-through"])
        return result

    result["notes"].append(f"M5 not ready (ADX={adx:.0f}, ATR={atr_pips:.1f})")
    return result


def assess_m1_trend_context(direction: str, m1_data: dict | None) -> dict:
    result = {
        "score": 0,
        "state": "missing",
        "market_ready": False,
        "notes": [],
    }
    if not m1_data:
        result["notes"].append("M1 missing")
        return result

    stoch_rsi = float(m1_data.get("stoch_rsi", 0.5))
    cci = float(m1_data.get("cci", 0))
    rsi = float(m1_data.get("rsi", 50))
    plus_di = float(m1_data.get("plus_di", 0))
    minus_di = float(m1_data.get("minus_di", 0))
    ema_5 = float(m1_data.get("ema_slope_5", 0))
    macd_hist = float(m1_data.get("macd_hist", 0))
    upper_wick = float(m1_data.get("upper_wick_avg_pips", 0))
    lower_wick = float(m1_data.get("lower_wick_avg_pips", 0))
    bb_upper = float(m1_data.get("bb_upper", 0))
    bb_lower = float(m1_data.get("bb_lower", 0))
    bb_mid = float(m1_data.get("bb_mid", 0))
    close = float(m1_data.get("close", 0))

    bb_pos = 0.5
    if bb_upper > bb_lower and close > 0:
        bb_pos = max(0.0, min(1.0, (close - bb_lower) / (bb_upper - bb_lower)))

    score = 0
    notes: list[str] = []
    reload_ready = False
    if direction == "BUY":
        if stoch_rsi >= 0.55:
            score += 1
        elif stoch_rsi <= 0.15:
            score -= 1
        if cci >= 50:
            score += 1
            notes.append(f"M1 CCI={cci:.0f}")
        elif cci <= -100:
            score -= 1
        if rsi >= 52:
            score += 1
        elif rsi <= 42:
            score -= 1
        if plus_di >= minus_di + 3:
            score += 1
        elif minus_di >= plus_di + 5:
            score -= 1
        if ema_5 > 0 and macd_hist > 0:
            score += 1
            notes.append("M1 momentum up")
        elif ema_5 < 0 and macd_hist < 0 and minus_di >= plus_di + 2:
            score -= 1
        if upper_wick > lower_wick * 1.4:
            score -= 1
        elif lower_wick > upper_wick * 1.2:
            score += 1
        reload_ready = (
            bb_pos <= 0.18
            and stoch_rsi <= 0.15
            and cci <= -80
            and rsi <= 45
            and close <= bb_mid
        )
    else:
        if stoch_rsi <= 0.45:
            score += 1
        elif stoch_rsi >= 0.85:
            score -= 1
        if cci <= -50:
            score += 1
            notes.append(f"M1 CCI={cci:.0f}")
        elif cci >= 100:
            score -= 1
        if rsi <= 48:
            score += 1
        elif rsi >= 58:
            score -= 1
        if minus_di >= plus_di + 3:
            score += 1
        elif plus_di >= minus_di + 5:
            score -= 1
        if ema_5 < 0 and macd_hist < 0:
            score += 1
            notes.append("M1 momentum down")
        elif ema_5 > 0 and macd_hist > 0 and plus_di >= minus_di + 2:
            score -= 1
        if lower_wick > upper_wick * 1.4:
            score -= 1
        elif upper_wick > lower_wick * 1.2:
            score += 1
        reload_ready = (
            bb_pos >= 0.82
            and stoch_rsi >= 0.85
            and cci >= 80
            and rsi >= 55
            and close >= bb_mid
        )

    if score >= TREND_M1_READY_SCORE:
        state = "aligned"
        market_ready = True
    elif reload_ready:
        state = "reload"
        market_ready = True
        notes.append(f"M1 {direction} reload zone")
    elif score >= 0:
        state = "mixed"
        market_ready = False
    else:
        state = "against"
        market_ready = False
    if not notes:
        notes.append(f"M1 {state}")
    result.update(score=score, state=state, market_ready=market_ready, notes=notes)
    return result


def assess_currency_trend_context(pair: str, direction: str, pulse: dict[str, dict[str, float]]) -> dict:
    base, quote = PAIR_CURRENCIES[pair]
    tf_scores = {
        tf: pulse.get(base, {}).get(tf, 0.0) - pulse.get(quote, {}).get(tf, 0.0)
        for tf in ("H1", "M15", "M1")
    }
    composite = tf_scores["H1"] * 0.45 + tf_scores["M15"] * 0.35 + tf_scores["M1"] * 0.20
    if direction == "BUY":
        aligned = composite >= TREND_PULSE_ALIGN_SCORE
        blocked = composite <= -TREND_PULSE_BLOCK_SCORE
    else:
        aligned = composite <= -TREND_PULSE_ALIGN_SCORE
        blocked = composite >= TREND_PULSE_BLOCK_SCORE

    if aligned:
        note = (
            f"currency pulse backs trend "
            f"(H1={tf_scores['H1']:+.1f} M15={tf_scores['M15']:+.1f} M1={tf_scores['M1']:+.1f})"
        )
    elif blocked:
        note = (
            f"currency pulse fights trend "
            f"(H1={tf_scores['H1']:+.1f} M15={tf_scores['M15']:+.1f} M1={tf_scores['M1']:+.1f})"
        )
    else:
        note = (
            f"currency pulse mixed "
            f"(H1={tf_scores['H1']:+.1f} M15={tf_scores['M15']:+.1f} M1={tf_scores['M1']:+.1f})"
        )
    return {
        "aligned": aligned,
        "blocked": blocked,
        "score": composite,
        "notes": [note],
    }


def conviction_from_scores(h1: dict, m15: dict, m5_setup: dict, m1: dict, currency_ctx: dict) -> str | None:
    if not (h1["aligned"] and m15["aligned"] and m5_setup["ready"]):
        return None
    if (
        h1["score"] >= 8
        and m15["score"] >= 8
        and m5_setup["setup"] == "band_walk"
        and m1["score"] >= TREND_M1_READY_SCORE + 1
        and currency_ctx["aligned"]
    ):
        return "S"
    if h1["score"] >= 7 and m15["score"] >= 7 and (currency_ctx["aligned"] or m1["score"] >= TREND_M1_READY_SCORE + 1):
        return "A"
    if (
        h1["score"] >= 8
        and m15["score"] >= 8
        and currency_ctx["aligned"]
        and m1["state"] in {"aligned", "reload", "mixed"}
    ):
        return "B"
    if not m1["market_ready"]:
        return None
    return "B"


def place_trend_market(token: str, acct: str, pair: str, direction: str,
                       units: int, tp: float, sl: float | None, comment: str) -> dict:
    """Place trend-bot MARKET order. sl=None = no-SL mode (MICRO/FAST timeout protection only)."""
    signed_units = str(units) if direction == "BUY" else str(-units)
    order: dict = {
        "type": "MARKET",
        "instrument": pair,
        "units": signed_units,
        "timeInForce": "FOK",
        "takeProfitOnFill": {
            "price": format_price(tp, pair),
            "timeInForce": "GTC",
        },
        "clientExtensions": {
            "tag": TREND_BOT_MARKET_TAG,
            "comment": comment,
        },
        "tradeClientExtensions": {
            "tag": TREND_BOT_MARKET_TAG,
            "comment": comment,
        },
    }
    if sl is not None:
        order["stopLossOnFill"] = {
            "price": format_price(sl, pair),
            "timeInForce": "GTC",
        }
    payload = {"order": order}
    try:
        return oanda_api(
            f"/v3/accounts/{acct}/orders", token, acct,
            method="POST", data=payload
        )
    except urllib.error.HTTPError as e:
        body = e.read().decode() if hasattr(e, "read") else str(e)
        return {"error": body, "code": e.code}


def log_entry(pair: str, direction: str, units: int, entry: float, tp: float, sl: float | None, order_id: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    sl_str = str(sl) if sl is not None else "NO_SL"
    line = (
        f"[{now}] TREND_BOT_MARKET {pair} {direction} {units}u "
        f"@{entry} TP={tp} SL={sl_str} id={order_id} tag={TREND_BOT_MARKET_TAG}\n"
    )
    with open(LOG_FILE, "a") as fh:
        fh.write(line)


def log_cancel(pair: str, direction: str, units: int, entry: float, order_id: str, reason: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    line = (
        f"[{now}] TREND_BOT_MARKET_CANCEL {pair} {direction} {units}u "
        f"@{entry} id={order_id} tag={TREND_BOT_MARKET_TAG} reason={reason}\n"
    )
    with open(LOG_FILE, "a") as fh:
        fh.write(line)


def slack_notify(pair: str, direction: str, units: int, entry: float, tp: float, sl: float | None, note: str) -> None:
    side = "LONG" if direction == "BUY" else "SHORT"
    cmd = [
        sys.executable, str(ROOT / "tools" / "slack_trade_notify.py"),
        "entry",
        "--pair", pair,
        "--side", side,
        "--units", str(units),
        "--price", str(entry),
        *(["--sl", str(sl)] if sl is not None else []),
        "--thesis", f"Trend bot market: {note}",
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=10)
    except Exception:
        pass


def build_plan(pair: str, direction: str, prices: dict, m5_data: dict, h1_data: dict,
               setup: str, m1_state: str, tempo: str) -> dict | None:
    current_price = current_execution_price(pair, direction, prices)
    if current_price <= 0:
        return None

    spread = get_spread(pair, prices)
    typical_spread = TYPICAL_SPREADS.get(pair, max(spread, 1.0))
    if spread > typical_spread * TREND_MAX_SPREAD_MULTIPLE:
        return None

    ps = pip_size(pair)
    atr_m5 = max(float(m5_data.get("atr_pips", 0)), max(typical_spread * 4.0, 2.5))
    atr_h1 = max(float(h1_data.get("atr_pips", 0)), atr_m5 * 2.0)

    sl_pips = max(
        typical_spread * TREND_MIN_SL_SPREAD_MULTIPLE,
        atr_m5 * TREND_MIN_SL_M5_ATR,
        atr_h1 * TREND_MIN_SL_H1_ATR,
    )
    tempo = str(tempo or "BALANCED").upper()
    if tempo == "MICRO" and spread > typical_spread * TREND_MICRO_MAX_SPREAD_MULTIPLE:
        return None
    if setup != "band_walk" or m1_state != "aligned":
        sl_pips *= TREND_WEAKER_SETUP_SL_BUFFER
    if setup == "band_walk":
        full_tp_floor = max(
            atr_m5 * TREND_BANDWALK_TP_M5_ATR,
            atr_h1 * TREND_BANDWALK_TP_H1_ATR,
        )
    else:
        full_tp_floor = max(
            atr_m5 * TREND_FOLLOWTHROUGH_TP_M5_ATR,
            atr_h1 * TREND_FOLLOWTHROUGH_TP_H1_ATR,
        )
    if tempo == "MICRO":
        if m1_state not in TREND_MICRO_ALLOWED_M1_STATES:
            return None
        tp1_pips = max(
            typical_spread * 3.5,
            sl_pips * TREND_MICRO_TP_RR_MULTIPLE,
            atr_m5 * TREND_MICRO_TP_M5_ATR,
            atr_h1 * TREND_MICRO_TP_H1_ATR,
        )
        tp2_pips = max(tp1_pips * 1.30, atr_m5 * 1.45, atr_h1 * 0.45)
        min_rr = TREND_MICRO_MIN_RR
    elif tempo == "FAST":
        tp1_pips = max(
            typical_spread * 4.0,
            sl_pips * TREND_FAST_TP_RR_MULTIPLE,
            atr_m5 * TREND_FAST_TP_M5_ATR,
            atr_h1 * TREND_FAST_TP_H1_ATR,
        )
        tp2_pips = max(tp1_pips * 1.25, full_tp_floor)
        min_rr = TREND_FAST_MIN_RR
    else:
        tp2_pips = max(sl_pips * TREND_MIN_TP_RR_MULTIPLE, full_tp_floor)
        tp1_pips = max(
            typical_spread * 4.0,
            atr_m5 * 1.20,
            sl_pips * 0.90,
        )
        tp1_pips = min(tp2_pips * 0.75, tp1_pips)
        min_rr = TREND_MIN_RR
    if tp1_pips <= 0 or tp2_pips <= 0:
        return None
    structural_sl_pips = sl_pips
    if direction == "BUY":
        structural_sl = current_price - structural_sl_pips * ps
        tp1 = current_price + tp1_pips * ps
        tp2 = current_price + tp2_pips * ps
        hold_boundary = current_price + max(
            typical_spread * 2.0,
            min(tp1_pips * 0.28, atr_m5 * 0.70),
        ) * ps
    else:
        structural_sl = current_price + structural_sl_pips * ps
        tp1 = current_price - tp1_pips * ps
        tp2 = current_price - tp2_pips * ps
        hold_boundary = current_price - max(
            typical_spread * 2.0,
            min(tp1_pips * 0.28, atr_m5 * 0.70),
        ) * ps

    if stop_is_already_broken(pair, direction, structural_sl, prices):
        return None

    rr = tp1_pips / structural_sl_pips if structural_sl_pips > 0 else 0.0
    if rr < min_rr:
        return None

    disaster_sl, disaster_sl_pips = worker_disaster_stop(
        pair,
        direction,
        current_price,
        structural_sl,
        tempo=tempo,
        style="trend",
        atr_pips=max(atr_m5, atr_h1 * 0.60),
        spread_pips=spread,
    )

    pace_fast = max(atr_m5 * 0.85, typical_spread * 2.0)
    pace_slow = max(atr_h1 * 0.30, atr_m5 * 0.60, typical_spread * 2.0)
    race_plan = build_target_race_plan(
        style="trend",
        pair=pair,
        direction=direction,
        entry=current_price,
        stop=structural_sl,
        tp1=tp1,
        tp2=tp2,
        hold_boundary=hold_boundary,
        pace_fast_pips=pace_fast,
        pace_slow_pips=pace_slow,
    )

    return {
        "entry": current_price,
        "tp": tp2,
        "tp1": tp1,
        "tp2": tp2,
        "sl": disaster_sl,
        "structural_sl": structural_sl,
        "tp_pips": tp1_pips,
        "tp2_pips": tp2_pips,
        "sl_pips": disaster_sl_pips,
        "structural_sl_pips": structural_sl_pips,
        "rr": rr,
        "rr_runner": (tp2_pips / structural_sl_pips) if structural_sl_pips > 0 else 0.0,
        "spread": spread,
        "race_plan": race_plan,
    }


def scan_trends(prices: dict) -> list[dict]:
    results = []
    all_technicals = {pair: load_technicals(pair) for pair in PAIRS}
    pulse = build_currency_pulse(all_technicals)

    for pair in PAIRS:
        tfs = all_technicals.get(pair, {})
        h1 = assess_tf_trend("H1", tfs.get("H1"), TREND_H1_ADX_MIN, TREND_H1_DI_GAP)
        m15 = assess_tf_trend("M15", tfs.get("M15"), TREND_M15_ADX_MIN, TREND_M15_DI_GAP)
        if not h1["aligned"] or not m15["aligned"] or h1["direction"] != m15["direction"]:
            continue

        direction = h1["direction"]
        m5_setup = detect_m5_continuation(direction, tfs.get("M5"))
        if not m5_setup["ready"]:
            continue

        m1 = assess_m1_trend_context(direction, tfs.get("M1"))
        currency_ctx = assess_currency_trend_context(pair, direction, pulse)
        if currency_ctx["blocked"]:
            continue

        conviction = conviction_from_scores(h1, m15, m5_setup, m1, currency_ctx)
        if conviction is None:
            continue

        balanced_plan = build_plan(
            pair,
            direction,
            prices,
            tfs.get("M5", {}),
            tfs.get("H1", {}),
            m5_setup["setup"],
            m1["state"],
            "BALANCED",
        )
        micro_plan = build_plan(
            pair,
            direction,
            prices,
            tfs.get("M5", {}),
            tfs.get("H1", {}),
            m5_setup["setup"],
            m1["state"],
            "MICRO",
        )
        if balanced_plan is None and micro_plan is None:
            continue

        signal_strength = h1["score"] + m15["score"] + m5_setup["strength"] + m1["score"]
        notes = h1["notes"][:1] + m15["notes"][:1] + m5_setup["notes"][:1] + currency_ctx["notes"][:1] + m1["notes"][:2]
        results.append({
            "pair": pair,
            "direction": direction,
            "conviction": conviction,
            "signal_strength": signal_strength,
            "setup": m5_setup["setup"],
            "m1_state": m1["state"],
            "market_note": " / ".join(notes),
            "triggers": notes,
            "plan_hint": balanced_plan or micro_plan,
            "currency_score": round(currency_ctx["score"], 2),
        })

    order = {"S": 0, "A": 1, "B": 2}
    results.sort(key=lambda item: (order.get(item["conviction"], 9), -item["signal_strength"]))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Trend Bot")
    parser.add_argument("--dry-run", action="store_true", help="Print plans without placing orders")
    args = parser.parse_args()

    now_utc = datetime.now(timezone.utc)
    print(f"=== TREND BOT === {now_utc.strftime('%Y-%m-%d %H:%M UTC')}")

    state, reason = get_market_state(now_utc)
    if state != "OPEN":
        print(f"Market {state}: {reason}")
        return 1

    if now_utc.hour in POISON_HOURS_UTC:
        print(f"Poison hour {now_utc.hour} UTC (19-23 UTC blocked). Skip.")
        return 1

    token, acct = load_config()
    account = fetch_account(token, acct)
    nav = account["nav"]
    margin_used = account["margin_used"]
    margin_pct = margin_used / nav * 100 if nav > 0 else 100
    policy, policy_notes = load_policy()
    allow_new_entries = global_status_allows_new_entries(policy)

    print(f"NAV: {nav:,.0f} JPY | Margin: {margin_pct:.1f}%")
    print(
        f"Policy: {policy['global_status']} | projected_cap={policy['projected_margin_cap']:.2f} "
        f"| panic={policy['panic_margin_cap']:.2f}"
    )
    if policy_notes:
        print(f"Policy notes: {'; '.join(policy_notes)}")
    if not allow_new_entries:
        print("Policy blocks new entries: reduce-only mode")

    prices = fetch_prices(token, acct)
    recent_worker_closes = fetch_recent_worker_close_events(token, acct)
    open_trades = fetch_open_trades(token, acct)
    pending_orders = fetch_pending_orders(token, acct)
    opportunities = scan_trends(prices)
    print(f"\nTrends found: {len(opportunities)}")

    budget_pct, budget_reasons = dynamic_bot_budget_profile(policy, opportunities)
    projected_margin = margin_used + sum(
        pending_margin(order, prices) for order in pending_orders if is_entry_pending_order(order)
    )
    projected_headroom = max(0.0, nav * float(policy["projected_margin_cap"]) - projected_margin)
    budget_remaining = min(nav * budget_pct, projected_headroom)
    print(
        f"Bot margin budget: {budget_remaining:,.0f} JPY "
        f"(dynamic_cap={budget_pct:.2f}, projected_headroom={projected_headroom:,.0f}, max_cap={MAX_BOT_MARGIN_PCT:.2f})"
    )
    if budget_reasons:
        print(f"  Budget reason: {', '.join(budget_reasons)}")

    placed = []
    skipped = []
    cancelled = 0

    for opp in opportunities:
        pair = opp["pair"]
        direction = opp["direction"]
        pair_policy = get_pair_policy(policy, pair)
        tempo = str(pair_policy.get("tempo", "BALANCED")).upper()
        if not allow_new_entries:
            skipped.append(f"{pair}: policy reduce-only")
            continue
        block_reason = pair_policy_worker_block_reason(pair_policy, direction, "MARKET")
        if block_reason:
            skipped.append(f"{pair}: {block_reason}")
            continue
        gate_blocked, gate_reason = brake_gate.check(pair, direction)
        if gate_blocked:
            skipped.append(f"{pair}: brake_gate {gate_reason}")
            continue
        if tempo == "FAST" and opp.get("m1_state") not in TREND_FAST_ALLOWED_M1_STATES:
            skipped.append(
                f"{pair}: FAST requires M1 in {sorted(TREND_FAST_ALLOWED_M1_STATES)}, got {opp.get('m1_state', 'missing')}"
            )
            continue
        if tempo == "MICRO" and opp.get("m1_state") not in TREND_MICRO_ALLOWED_M1_STATES:
            skipped.append(
                f"{pair}: MICRO requires M1 in {sorted(TREND_MICRO_ALLOWED_M1_STATES)}, "
                f"got {opp.get('m1_state', 'missing')}"
            )
            continue

        trade_conflicts = discretionary_open_trade_conflicts(
            open_trades, pair, direction, "MARKET", pair_policy["ownership"]
        )
        if trade_conflicts:
            skipped.append(
                f"{pair}: discretionary open trade blocks ({describe_trade_conflicts(trade_conflicts)})"
            )
            continue

        pending_conflicts = discretionary_pending_conflicts_for_worker(
            pending_orders, pair, direction, "MARKET", pair_policy["ownership"]
        )
        if pending_conflicts:
            skipped.append(
                f"{pair}: discretionary pending entry exists ({describe_pending_conflicts(pending_conflicts)})"
            )
            continue

        cooldown_reason = recent_close_cooldown(
            pair,
            direction,
            now_utc,
            tempo=tempo,
            recent_worker_closes=recent_worker_closes,
        )
        if cooldown_reason:
            skipped.append(f"{pair}: {cooldown_reason}")
            continue

        pair_trades = [trade for trade in open_trades if trade.get("instrument") == pair]
        opposite = [trade for trade in pair_trades if trade_direction(trade) != direction]
        if opposite:
            skipped.append(f"{pair}: opposite open trade exists")
            continue
        same_direction = [trade for trade in pair_trades if trade_direction(trade) == direction]
        same_tag = [trade for trade in same_direction if get_tag(trade) == TREND_BOT_MARKET_TAG]
        if len(same_tag) >= TREND_MAX_TAG_TRADES:
            skipped.append(f"{pair}: trend bot trade already live")
            continue
        if len(same_direction) >= TREND_MAX_PAIR_TRADES:
            skipped.append(f"{pair}: same-direction trades already {len(same_direction)}")
            continue

        opposing_pending = []
        for order in pending_orders:
            if order.get("instrument") != pair:
                continue
            tag = get_tag(order)
            if tag not in TREND_CANCELABLE_PENDING_TAGS:
                continue
            units = int(order.get("units", 0))
            order_direction = "BUY" if units > 0 else "SELL"
            if order_direction != direction:
                opposing_pending.append(order)

        tfs = load_technicals(pair)
        plan = build_plan(
            pair,
            direction,
            prices,
            tfs.get("M5", {}),
            tfs.get("H1", {}),
            opp["setup"],
            opp.get("m1_state", "mixed"),
            tempo,
        )
        if plan is None:
            skipped.append(f"{pair}: no executable plan for {tempo} tempo")
            continue
        entry = plan["entry"]
        tp1 = plan["tp1"]
        tp2 = plan["tp2"]
        # MICRO/FAST: no broker SL — timeout in bot_trade_manager is the primary protection.
        sl = None if tempo in ("MICRO", "FAST") else plan["sl"]
        runner_comment = encode_target_race_comment(plan["race_plan"])
        units = calculate_units(opp["conviction"], nav, entry, pair, order_type="MARKET", tempo=tempo)
        est_margin = estimate_margin(units, entry, pair)
        if est_margin > budget_remaining:
            skipped.append(f"{pair}: margin budget exhausted")
            continue

        side_label = "LONG" if direction == "BUY" else "SHORT"
        print(f"\n  {pair} {opp['conviction']}-{side_label} MARKET {units}u @{format_price(entry, pair)}")
        print(
            f"    TP1={format_price(tp1, pair)} (+{plan['tp_pips']:.1f}pip)"
            f" | TP2={format_price(tp2, pair)} (+{plan['tp2_pips']:.1f}pip)"
        )
        print(
            f"    Thesis line={format_price(plan['structural_sl'], pair)} "
            f"(-{plan['structural_sl_pips']:.1f}pip)"
            + (f" | Disaster SL={format_price(sl, pair)} (-{plan['sl_pips']:.1f}pip)" if sl is not None else " | No-SL (timeout mode)")
        )
        print(
            f"    R:R1={plan['rr']:.1f} | R:R2={plan['rr_runner']:.1f} "
            f"| setup={opp['setup']} | Str={opp['signal_strength']} | {opp['market_note']}"
        )
        print(
            f"    Race: TP1~{plan['race_plan'].get('eta_fast_bars', '?')} bars | "
            f"TP2~{plan['race_plan'].get('eta_slow_bars', '?')} bars | "
            f"hold>{format_price(plan['race_plan']['hold_boundary'], pair)} after TP1"
        )
        if opposing_pending:
            print(f"    Cancel opposing bot pending first: {', '.join(str(order.get('id')) for order in opposing_pending)}")

        if args.dry_run:
            placed.append({
                "pair": pair,
                "direction": direction,
                "units": units,
                "entry": entry,
                "conviction": opp["conviction"],
                "order_type": "MARKET",
            })
            budget_remaining -= est_margin
            continue

        live_policy, live_policy_notes = load_policy()
        if not global_status_allows_new_entries(live_policy):
            note = f"policy switched to {live_policy['global_status']} before submit"
            if live_policy_notes:
                note = f"{note} ({'; '.join(live_policy_notes)})"
            print(f"    BLOCKED before submit: {note}")
            skipped.append(f"{pair}: {note}")
            continue
        live_pair_policy = get_pair_policy(live_policy, pair)
        live_block_reason = pair_policy_worker_block_reason(live_pair_policy, direction, "MARKET")
        if live_block_reason:
            print(f"    BLOCKED before submit: {live_block_reason}")
            skipped.append(f"{pair}: {live_block_reason} before submit")
            continue

        for order in opposing_pending:
            oid = str(order.get("id", "?"))
            if cancel_order(token, acct, oid):
                cancelled += 1

        result = place_trend_market(token, acct, pair, direction, units, tp2, sl, runner_comment)
        if "error" in result:
            print(f"    ERROR: {result['error']}")
            skipped.append(f"{pair}: MARKET API error")
            continue

        cancel_resp = result.get("orderCancelTransaction", {})
        if cancel_resp:
            reason_note = cancel_resp.get("reason", "CANCELLED")
            order_id = cancel_resp.get("orderID") or result.get("orderCreateTransaction", {}).get("id", "?")
            print(f"    CANCELLED by OANDA: {reason_note}")
            log_cancel(pair, direction, units, entry, order_id, reason_note)
            skipped.append(f"{pair}: MARKET cancelled ({reason_note})")
            continue

        reject_resp = result.get("orderRejectTransaction", {})
        if reject_resp:
            reason_note = reject_resp.get("rejectReason", "REJECTED")
            order_id = reject_resp.get("id", "?")
            print(f"    REJECTED by OANDA: {reason_note}")
            log_cancel(pair, direction, units, entry, order_id, reason_note)
            skipped.append(f"{pair}: MARKET rejected ({reason_note})")
            continue

        fill_resp = result.get("orderFillTransaction", {})
        fill_price = float(fill_resp.get("price", entry)) if fill_resp else entry
        order_id = fill_resp.get("id") or result.get("orderCreateTransaction", {}).get("id") or "?"
        print(f"    PLACED id={order_id}")
        trade_id = extract_trade_id_from_order_result(result)
        if trade_id:
            remember_trade_plan(trade_id, plan["race_plan"], units)
        log_entry(pair, direction, units, fill_price, tp2, sl, order_id)
        slack_notify(pair, direction, units, fill_price, tp2, sl, opp["market_note"])

        placed.append({
            "pair": pair,
            "direction": direction,
            "units": units,
            "entry": fill_price,
            "conviction": opp["conviction"],
            "order_id": order_id,
            "order_type": "MARKET",
        })
        budget_remaining -= est_margin

    print(f"\n{'='*50}")
    print("TREND BOT SUMMARY")
    print(f"  Scanned: {len(PAIRS)} pairs")
    print(f"  Trends: {len(opportunities)}")
    print(f"  Placed: {len(placed)}")
    print(f"  Cancelled: {cancelled}")
    if skipped:
        print(f"  Skipped: {', '.join(skipped)}")
    if placed:
        for item in placed:
            side = "LONG" if item["direction"] == "BUY" else "SHORT"
            oid = item.get("order_id", "dry")
            print(
                f"    {item['pair']} {item['conviction']}-{side} MARKET "
                f"{item['units']}u @{format_price(item['entry'], item['pair'])} id={oid}"
            )
    if not placed:
        print("  (no orders placed)")
    print(f"{'='*50}")
    return 0 if placed or cancelled else 1


if __name__ == "__main__":
    raise SystemExit(main())
