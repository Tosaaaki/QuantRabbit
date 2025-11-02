from __future__ import annotations

"""
execution.exit_manager
~~~~~~~~~~~~~~~~~~~~~~

Monitors open trades and improves exit timing by:
- Break-even stop once price moves favorably by N pips
- ATR-based trailing stop for trend/breakout strategies
- Time/RSI-based exit for mean-reversion

Assumptions:
- USD/JPY only (pip = 0.01)
- Strategy metadata stored in clientExtensions.comment (strategy=..., macro=..., micro=...)
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional

import requests

from utils.secrets import get_secret
from indicators.factor_cache import all_factors
from execution.trade_actions import update_trade_orders, close_trade
from analysis.kaizen import get_policy
from utils.exit_events import log_exit_event
from analysis.gpt_exit_advisor import advise_or_fallback
from utils.market_hours import is_market_open


def _auth() -> Tuple[str, Dict[str, str]] | None:
    try:
        token = get_secret("oanda_token")
        account = get_secret("oanda_account_id")  # noqa: F841 (for URL only)
        try:
            pract = get_secret("oanda_practice").lower() == "true"
        except Exception:
            pract = True
        host = (
            "https://api-fxpractice.oanda.com"
            if pract
            else "https://api-fxtrade.oanda.com"
        )
        headers = {"Authorization": f"Bearer {token}"}
        return host, headers
    except Exception:
        return None


def _latest_mid_price() -> float | None:
    api = _auth()
    if not api:
        return None
    host, headers = api
    account = get_secret("oanda_account_id")
    try:
        resp = requests.get(
            f"{host}/v3/accounts/{account}/pricing",
            headers=headers,
            params={"instruments": "USD_JPY"},
            timeout=5,
        )
        resp.raise_for_status()
        prices = resp.json().get("prices", [])
        if not prices:
            return None
        bids = prices[0].get("bids") or []
        asks = prices[0].get("asks") or []
        if bids and asks:
            return (float(bids[0]["price"]) + float(asks[0]["price"])) / 2.0
    except Exception:
        return None
    return None


def _parse_meta(ext: Dict[str, Any] | None) -> Dict[str, str]:
    """Parse strategy/macro/micro from clientExtensions.comment"""
    out: Dict[str, str] = {}
    if not ext:
        return out
    comment = (ext.get("comment") or "").strip()
    for part in comment.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip().lower()] = v.strip()
    tag = (ext.get("tag") or "").strip()
    if tag.startswith("pocket="):
        out.setdefault("pocket", tag.split("=", 1)[1])
    return out


def _get_open_trades() -> list[Dict[str, Any]]:
    env = _auth()
    if not env:
        return []
    REST_HOST, HEADERS = env
    account = get_secret("oanda_account_id")
    url = f"{REST_HOST}/v3/accounts/{account}/openTrades"
    try:
        r = requests.get(url, headers=HEADERS, timeout=7)
        r.raise_for_status()
        trades = r.json().get("trades", [])
        return trades
    except requests.RequestException as e:
        logging.error("[exit_manager] openTrades error: %s", e)
        return []


def _pip() -> float:
    # USD/JPY pip size
    return 0.01


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _minutes_since(iso_str: str) -> float:
    try:
        t = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return (_now_utc() - t).total_seconds() / 60.0
    except Exception:
        return 0.0


ADVICE_TRIGGER_PIPS = float(os.environ.get("EXIT_GPT_TRIGGER_PIPS", "8.0"))
ADVICE_REFRESH_SEC = int(os.environ.get("EXIT_GPT_REFRESH_SEC", "180"))
ADVICE_CLOSE_CONF = float(os.environ.get("EXIT_GPT_CLOSE_CONF", "0.55"))
ADVICE_ADJUST_CONF = float(os.environ.get("EXIT_GPT_ADJUST_CONF", "0.35"))
ADVICE_MAX_WAIT_MIN = int(os.environ.get("EXIT_GPT_MAX_WAIT_MIN", "45"))
BE_BUFFER_DEFAULT = float(os.environ.get("EXIT_BE_BUFFER_PIPS", "2.0"))
BE_BUFFER_STRATEGY = {
    "BB_RSI": float(os.environ.get("EXIT_BE_BUFFER_BB_RSI", "3.0")),
    "NewsSpikeReversal": float(os.environ.get("EXIT_BE_BUFFER_NEWS", "2.5")),
    "TrendMA": float(os.environ.get("EXIT_BE_BUFFER_TRENDMA", "2.0")),
    "Donchian55": float(os.environ.get("EXIT_BE_BUFFER_DONCHIAN", "2.5")),
    "ScalpMeanRevert": float(os.environ.get("EXIT_BE_BUFFER_SCALP", "1.8")),
}

POCKET_TRIGGER_MULT = {
    "micro": float(os.environ.get("EXIT_GPT_TRIGGER_MULT_MICRO", "0.8")),
    "macro": float(os.environ.get("EXIT_GPT_TRIGGER_MULT_MACRO", "1.0")),
    "scalp": float(os.environ.get("EXIT_GPT_TRIGGER_MULT_SCALP", "0.7")),
}

STRATEGY_TRIGGER_OVERRIDES = {
    "BB_RSI": float(os.environ.get("EXIT_GPT_TRIGGER_BB_RSI", "5.0")),
    "NewsSpikeReversal": float(os.environ.get("EXIT_GPT_TRIGGER_NEWS", "4.0")),
    "TrendMA": float(os.environ.get("EXIT_GPT_TRIGGER_TRENDMA", str(ADVICE_TRIGGER_PIPS))),
    "Donchian55": float(os.environ.get("EXIT_GPT_TRIGGER_DONCHIAN", str(ADVICE_TRIGGER_PIPS))),
    "ScalpMeanRevert": float(os.environ.get("EXIT_GPT_TRIGGER_SCALP", "3.8")),
}

# TrendMA early exit tuning (seconds -> minutes calculations later)
TRENDMA_EARLY_WINDOW_MIN = float(os.environ.get("EXIT_TRENDMA_EARLY_WINDOW_MIN", "12"))
TRENDMA_EARLY_PROFIT_PIPS = abs(float(os.environ.get("EXIT_TRENDMA_EARLY_PROFIT_PIPS", "3.0")))
TRENDMA_EARLY_LOSS_PIPS = abs(float(os.environ.get("EXIT_TRENDMA_EARLY_LOSS_PIPS", "6.0")))
TRENDMA_TIMEOUT_MIN = float(os.environ.get("EXIT_TRENDMA_TIMEOUT_MIN", "75"))
TRENDMA_TIMEOUT_TP_PIPS = abs(float(os.environ.get("EXIT_TRENDMA_TIMEOUT_TP_PIPS", "6.0")))
TRENDMA_TIMEOUT_SL_PIPS = abs(float(os.environ.get("EXIT_TRENDMA_TIMEOUT_SL_PIPS", "8.0")))

FORCE_EXIT_LOSS_PIPS = float(os.environ.get("EXIT_FORCE_LOSS_PIPS", "45"))
FORCE_EXIT_ATR_MULT = float(os.environ.get("EXIT_FORCE_ATR_MULT", "4.5"))
FORCE_EXIT_MAX_MIN = float(os.environ.get("EXIT_FORCE_MAX_MIN", "240"))
FORCE_EXIT_TREND_GAP_PIPS = float(os.environ.get("EXIT_FORCE_TREND_GAP_PIPS", "8.0"))
FORCE_EXIT_RSI_LONG_MAX = float(os.environ.get("EXIT_FORCE_RSI_LONG_MAX", "38.0"))
FORCE_EXIT_RSI_SHORT_MIN = float(os.environ.get("EXIT_FORCE_RSI_SHORT_MIN", "62.0"))
FORCE_EXIT_VELOCITY = float(os.environ.get("EXIT_FORCE_VELOCITY", "3.5"))

_ADVICE_CACHE: Dict[str, Dict[str, Any]] = {}


def _order_price(order: Optional[Dict[str, Any]]) -> Optional[float]:
    if not order:
        return None
    price = order.get("price")
    if price is None:
        return None
    try:
        return float(price)
    except (TypeError, ValueError):
        return None


def _should_update_price(current: Optional[float], new_price: float) -> bool:
    if current is None:
        return True
    pip = _pip()
    tolerance = pip * 0.1
    return abs(new_price - current) >= tolerance


_SCALP_PROFIT_STATE: Dict[str, Dict[str, float]] = {}

# Track per-trade microstructure observations so we can detect false breaks
_FALSE_BREAK_STATE: Dict[str, Dict[str, float]] = {}


def _trigger_threshold(strategy: str, pocket: str, atr_pips: float) -> float:
    base = ADVICE_TRIGGER_PIPS
    override = STRATEGY_TRIGGER_OVERRIDES.get(strategy)
    if override is not None:
        base = override
    base *= POCKET_TRIGGER_MULT.get(pocket, 1.0)
    if atr_pips > 0:
        base = min(base, max(4.0, atr_pips * 1.2))
    return max(3.0, base)


def _should_request_advice(move_pips: float, age_min: float, strategy: str, pocket: str, atr_pips: float) -> bool:
    threshold = _trigger_threshold(strategy, pocket, atr_pips)
    if abs(move_pips) >= threshold:
        return True
    if age_min >= ADVICE_MAX_WAIT_MIN and move_pips >= threshold * 0.5:
        return True
    return False


def _false_break_guard(
    trade_id: str,
    move_pips: float,
    atr_pips: float,
    velocity_30s: float,
    tick_range_30s: float,
) -> str | None:
    """Detect situations where the move likely turns into a false break and profits should be protected."""
    if not trade_id:
        return None

    state = _FALSE_BREAK_STATE.get(trade_id)
    if state is None:
        _FALSE_BREAK_STATE[trade_id] = {
            "max_move": move_pips,
            "last_atr": atr_pips,
            "last_vel": velocity_30s,
            "last_range": tick_range_30s,
        }
        return None

    state["max_move"] = max(state.get("max_move", move_pips), move_pips)

    positive_trigger = max(1.5, atr_pips * 0.6)
    atr_drop = False
    range_collapse = False
    vel_flip = False

    last_atr = state.get("last_atr")
    if last_atr and atr_pips:
        atr_drop = atr_pips < last_atr * 0.7

    last_range = state.get("last_range")
    if last_range:
        range_collapse = tick_range_30s < last_range * 0.6 and tick_range_30s <= positive_trigger * 0.5

    last_vel = state.get("last_vel")
    if last_vel is not None:
        vel_flip = last_vel > 0 and velocity_30s < -0.5 or last_vel < 0 and velocity_30s > 0.5

    pullback = state["max_move"] - move_pips
    if (
        state["max_move"] >= positive_trigger
        and move_pips > 0.2
        and pullback >= positive_trigger * 0.5
        and (atr_drop or range_collapse or vel_flip)
    ):
        state["last_atr"] = atr_pips or last_atr
        state["last_vel"] = velocity_30s
        state["last_range"] = tick_range_30s
        return "false_break"

    state["last_atr"] = atr_pips or last_atr
    state["last_vel"] = velocity_30s
    state["last_range"] = tick_range_30s
    return None


def _be_buffer(strategy: str, pocket: str) -> float:
    val = BE_BUFFER_STRATEGY.get(strategy)
    if val is None:
        if pocket == "micro":
            return max(1.5, BE_BUFFER_DEFAULT * 0.8)
        return BE_BUFFER_DEFAULT
    return max(1.0, val)


def _loss_threshold_pips(atr_pips: float) -> float:
    dynamic = atr_pips * FORCE_EXIT_ATR_MULT if atr_pips > 0 else 0.0
    return max(FORCE_EXIT_LOSS_PIPS, dynamic)


def _forced_exit_reason(
    *,
    direction: int,
    move_pips: float,
    age_min: float,
    atr_pips: float,
    price_now: float,
    ma10: float,
    rsi_m1: float,
    velocity_m1: float,
) -> Optional[str]:
    if direction == 0:
        return None
    loss_threshold = _loss_threshold_pips(atr_pips)
    if move_pips <= -loss_threshold:
        return f"hard_loss_{loss_threshold:.1f}"
    if move_pips < 0 and age_min >= FORCE_EXIT_MAX_MIN:
        return "timeout_loss"

    pip = _pip()
    gap_pips = 0.0
    if direction > 0:
        gap_pips = (ma10 - price_now) / pip
        if (
            gap_pips >= FORCE_EXIT_TREND_GAP_PIPS
            and rsi_m1 <= FORCE_EXIT_RSI_LONG_MAX
            and velocity_m1 <= -FORCE_EXIT_VELOCITY
        ):
            return "trend_break_long"
    else:
        gap_pips = (price_now - ma10) / pip
        if (
            gap_pips >= FORCE_EXIT_TREND_GAP_PIPS
            and rsi_m1 >= FORCE_EXIT_RSI_SHORT_MIN
            and velocity_m1 >= FORCE_EXIT_VELOCITY
        ):
            return "trend_break_short"

    return None


def _log_event(
    *,
    trade_id: str,
    strategy: Optional[str],
    pocket: Optional[str],
    version: Optional[str],
    event_type: str,
    action: str,
    advice: Optional[Dict[str, Any]] = None,
    price: Optional[float] = None,
    move_pips: Optional[float] = None,
    note: Optional[str] = None,
) -> None:
    if not trade_id:
        return
    conf = advice.get("confidence") if advice else None
    tp = advice.get("target_tp_pips") if advice else None
    sl = advice.get("target_sl_pips") if advice else None
    note_val = note or (advice.get("note") if advice else None)
    log_exit_event(
        trade_id=trade_id,
        strategy=strategy,
        pocket=pocket,
        version=version,
        event_type=event_type,
        action=action,
        confidence=conf,
        target_tp_pips=tp,
        target_sl_pips=sl,
        note=note_val,
        price=price,
        move_pips=move_pips,
        payload=advice,
    )


async def exit_loop():
    """Periodically review open trades and adjust exits."""
    # Default policy fallback if DB unavailable
    FALLBACK = {
        "micro": {"be": 10.0, "atr": 1.0, "mintrail": 8.0},
        "macro": {"be": 20.0, "atr": 1.5, "mintrail": 15.0},
        "scalp": {"be": 2.5, "atr": 0.8, "mintrail": 4.0},
    }

    market_pause_logged = False

    while True:
        try:
            trades = _get_open_trades()
            if not trades:
                await asyncio.sleep(30)
                continue

            active_ids = {
                str(tr.get("id") or tr.get("tradeID") or tr.get("trade_id") or "")
                for tr in trades
                if tr.get("id") or tr.get("tradeID") or tr.get("trade_id")
            }
            for tid in list(_FALSE_BREAK_STATE.keys()):
                if tid not in active_ids:
                    _FALSE_BREAK_STATE.pop(tid, None)

            factors = all_factors()
            fac_m1 = factors.get("M1") or {}
            fac_h4 = factors.get("H4") or {}
            price_now = float(fac_m1.get("close") or 0.0)
            mid_live = _latest_mid_price()
            if mid_live is not None:
                price_now = mid_live
            atr_m1 = float(fac_m1.get("atr") or 0.0)
            atr_h4 = float(fac_h4.get("atr") or 0.0)
            rsi_m1 = float(fac_m1.get("rsi") or 50.0)
            ma10_m1 = float(fac_m1.get("ma10") or price_now)
            velocity_m1 = float(fac_m1.get("tick_velocity_30s") or 0.0)
            tick_range_m1 = float(fac_m1.get("tick_range_30s") or 0.0)
            atr_m1_pips = atr_m1 / _pip() if atr_m1 else 0.0
            now = _now_utc()
            market_open, age_sec = is_market_open(fac_m1, now=now)
            force_fallback = not market_open
            if force_fallback and not market_pause_logged:
                if age_sec is None:
                    logging.info(
                        "[exit_manager] Market inactive: missing recent M1 candle. GPT exit advice will use fallback."
                    )
                else:
                    logging.info(
                        "[exit_manager] Market inactive for %.1f minutes. GPT exit advice will use fallback.",
                        age_sec / 60.0,
                    )
                market_pause_logged = True
            elif not force_fallback and market_pause_logged:
                logging.info(
                    "[exit_manager] Market activity detected. Restoring GPT exit advice."
                )
                market_pause_logged = False

            for tr in trades:
                trade_id = tr.get("id")
                if not trade_id:
                    continue
                instrument = tr.get("instrument")
                if instrument != "USD_JPY":
                    continue  # scope limited
                units = int(tr.get("initialUnits", 0))
                entry = float(tr.get("price", 0.0))
                ext = tr.get("clientExtensions") or {}
                meta = _parse_meta(ext)
                strategy = meta.get("strategy") or ""
                pocket = meta.get("pocket") or ("macro" if abs(units) >= 100000 else "micro")
                version = meta.get("ver") or meta.get("version")

                direction = 1 if units > 0 else -1
                pip = _pip()

                trade_price_now = price_now
                trade_price = tr.get("currentPrice") or {}
                mid_price = trade_price.get("mid")
                bid = trade_price.get("bid")
                ask = trade_price.get("ask")
                try:
                    if mid_price is not None:
                        trade_price_now = float(mid_price)
                    elif bid is not None and ask is not None:
                        trade_price_now = (float(bid) + float(ask)) / 2.0
                except (TypeError, ValueError):
                    pass

                move_pips = (trade_price_now - entry) * direction / pip
                age_min = _minutes_since(tr.get("openTime", ""))

                sl_price_current = _order_price(tr.get("stopLossOrder"))
                tp_price_current = _order_price(tr.get("takeProfitOrder"))

                force_reason = _forced_exit_reason(
                    direction=direction,
                    move_pips=move_pips,
                    age_min=age_min,
                    atr_pips=atr_m1_pips,
                    price_now=trade_price_now,
                    ma10=ma10_m1,
                    rsi_m1=rsi_m1,
                    velocity_m1=velocity_m1,
                )
                if force_reason:
                    if close_trade(trade_id):
                        logging.warning(
                            "[exit_manager] force close trade=%s reason=%s move=%.1f age=%.1f",
                            trade_id,
                            force_reason,
                            move_pips,
                            age_min,
                        )
                        _log_event(
                            trade_id=trade_id,
                            strategy=strategy,
                            pocket=pocket,
                            version=version,
                            event_type="force_exit",
                            action="close",
                            advice=None,
                            price=trade_price_now,
                            move_pips=move_pips,
                            note=force_reason,
                        )
                        _SCALP_PROFIT_STATE.pop(trade_id, None)
                        _FALSE_BREAK_STATE.pop(str(trade_id), None)
                        continue

                false_break_reason = _false_break_guard(
                    str(trade_id),
                    move_pips,
                    atr_m1_pips,
                    velocity_m1,
                    tick_range_m1,
                )
                if false_break_reason and move_pips > 0:
                    if close_trade(trade_id):
                        logging.info(
                            "[exit_manager] false-break exit trade=%s move=%.1f atr=%.2f range=%.2f",
                            trade_id,
                            move_pips,
                            atr_m1_pips,
                            tick_range_m1,
                        )
                        _log_event(
                            trade_id=trade_id,
                            strategy=strategy,
                            pocket=pocket,
                            version=version,
                            event_type="false_break_exit",
                            action="close",
                            advice=None,
                            price=trade_price_now,
                            move_pips=move_pips,
                            note=false_break_reason,
                        )
                        _FALSE_BREAK_STATE.pop(str(trade_id), None)
                        _SCALP_PROFIT_STATE.pop(trade_id, None)
                        continue

                # Load per (strategy,pocket) policy
                pocket_defaults = FALLBACK.get(pocket, FALLBACK["micro"])
                try:
                    pol = get_policy(strategy or "", pocket)
                    be_trig = float(pol.get("be_trigger_pips", pocket_defaults["be"]))
                    atr_mult = float(pol.get("trail_atr_mult", pocket_defaults["atr"]))
                    min_trail = float(pol.get("min_trail_pips", pocket_defaults["mintrail"]))
                    meanrev_max_min = float(pol.get("meanrev_max_min", 30.0))
                    meanrev_rsi_exit = float(pol.get("meanrev_rsi_exit", 50.0))
                except Exception:
                    be_trig = pocket_defaults["be"]
                    atr_mult = pocket_defaults["atr"]
                    min_trail = pocket_defaults["mintrail"]
                    meanrev_max_min = 30.0
                    meanrev_rsi_exit = 50.0

                advice: Dict[str, Any] | None = None
                if _should_request_advice(move_pips, age_min, strategy, pocket, atr_m1 * 100.0):
                    cache = _ADVICE_CACHE.get(trade_id)
                    too_old = True
                    if cache and cache.get("ts"):
                        try:
                            cached_dt = datetime.fromisoformat(cache["ts"])
                            too_old = (now - cached_dt).total_seconds() > ADVICE_REFRESH_SEC
                        except Exception:
                            too_old = True
                    if too_old:
                        payload = {
                            "trade_id": trade_id,
                            "strategy": strategy,
                            "pocket": pocket,
                            "direction": "long" if direction > 0 else "short",
                            "entry_price": entry,
                            "current_price": trade_price_now,
                            "unrealized_pips": round(move_pips, 2),
                            "atr_m1_pips": round(atr_m1 * 100.0, 2),
                            "atr_h4_pips": round(atr_h4 * 100.0, 2),
                            "macro_regime": meta.get("macro") or "",
                            "micro_regime": meta.get("micro") or "",
                            "age_minutes": round(age_min, 1),
                            "rsi_m1": rsi_m1,
                            "version": version,
                        }
                        logging.info(
                            "[exit_manager] GPT advice request trade=%s move=%.2f age=%.1fm",
                            trade_id,
                            move_pips,
                            age_min,
                        )
                        try:
                            advice = await advise_or_fallback(
                                payload, force_fallback=force_fallback
                            )
                        except Exception as exc:
                            logging.error(
                                "[exit_manager] GPT exit advice failed (fallback disabled): %s",
                                exc,
                            )
                            advice = None
                        if advice is None:
                            continue
                        _ADVICE_CACHE[trade_id] = {
                            "ts": now.isoformat(timespec="seconds"),
                            "payload": payload,
                            "advice": advice,
                        }
                        _log_event(
                            trade_id=trade_id,
                            strategy=strategy,
                            pocket=pocket,
                            version=version,
                            event_type="advice",
                            action="close_now" if advice.get("close_now") else "hold",
                            advice=advice,
                            price=trade_price_now,
                            move_pips=move_pips,
                        )
                    elif cache:
                        advice = cache.get("advice")

                # 1) Break-even move
                if move_pips >= be_trig:
                    buffer = _be_buffer(strategy, pocket)
                    be_price = entry + direction * (buffer * pip)
                    if _should_update_price(sl_price_current, be_price):
                        ok = update_trade_orders(trade_id, sl_price=be_price)
                        if ok:
                            sl_price_current = be_price
                            logging.info(
                                "[exit_manager] BE set trade=%s price=%.3f buffer=%.2f",
                                trade_id,
                                be_price,
                                buffer,
                            )

                # 2) Trailing for trend/breakout
                if strategy in ("TrendMA", "Donchian55"):
                    atr = atr_h4 if pocket == "macro" and atr_h4 else atr_m1
                    dist = max(min_trail * pip, atr_mult * atr)
                    if dist > 0 and price_now > 0:
                        # Do not worsen beyond BE+1pip
                        be_price = entry + direction * (1 * pip)
                        implied_trail_sl = price_now - direction * dist
                        safe_to_trail = (
                            implied_trail_sl >= be_price if direction > 0 else implied_trail_sl <= be_price
                        )
                        if safe_to_trail:
                            ok = update_trade_orders(trade_id, trailing_distance=dist)
                            if ok:
                                logging.info(
                                    "[exit_manager] Trail set trade=%s dist=%.3f pocket=%s strategy=%s",
                                    trade_id,
                                    dist,
                                    pocket,
                                    strategy,
                                )

                # 3) Mean-reversion exit: time/RSI
                if strategy == "BB_RSI":
                    rsi_exit_hit = (
                        rsi_m1 >= meanrev_rsi_exit if direction > 0 else rsi_m1 <= meanrev_rsi_exit
                    )
                    if age_min >= meanrev_max_min or rsi_exit_hit:
                        ok = close_trade(trade_id)
                        if ok:
                            logging.info(
                                "[exit_manager] MeanRev exit trade=%s age=%.1fm rsi=%.1f",
                                trade_id,
                                age_min,
                                rsi_m1,
                            )

                if strategy == "ScalpMeanRevert":
                    state = _SCALP_PROFIT_STATE.get(trade_id)
                    initial_tp = abs((tp_price_current - entry) / pip) if tp_price_current else None
                    initial_sl = abs((entry - sl_price_current) / pip) if sl_price_current else None
                    if not state:
                        state = {
                            "initial_tp": initial_tp or max(be_trig * 1.2, 5.0),
                            "initial_sl": initial_sl or max(be_trig, 3.0),
                            "max_move": max(0.0, move_pips),
                            "last_tp_price": tp_price_current,
                            "last_sl_price": sl_price_current,
                        }
                        _SCALP_PROFIT_STATE[trade_id] = state
                    else:
                        state["max_move"] = max(state.get("max_move", 0.0), move_pips)
                        if tp_price_current is not None:
                            state["last_tp_price"] = tp_price_current
                        if sl_price_current is not None:
                            state["last_sl_price"] = sl_price_current

                    max_move = state["max_move"]
                    init_tp = state.get("initial_tp", max(be_trig, 5.0))
                    init_sl = state.get("initial_sl", max(be_trig, 3.0))

                    if max_move >= max(init_tp * 0.9, init_sl * 1.4):
                        bonus = max(2.0, init_sl * 0.8)
                        desired_move = max_move + bonus
                        desired_tp_price = entry + direction * desired_move * pip
                        current_tp = tp_price_current or state.get("last_tp_price")
                        tp_ok = False
                        if direction > 0:
                            tp_ok = current_tp is None or desired_tp_price > current_tp + pip * 0.5
                        else:
                            tp_ok = current_tp is None or desired_tp_price < current_tp - pip * 0.5
                        if tp_ok:
                            if update_trade_orders(trade_id, tp_price=desired_tp_price):
                                state["last_tp_price"] = desired_tp_price
                                logging.info(
                                    "[exit_manager] Extend TP trade=%s target=%.3f move=%.2f",
                                    trade_id,
                                    desired_tp_price,
                                    desired_move,
                                )

                    if max_move >= max(init_sl * 1.5, 6.0):
                        buffer = max(1.5, init_sl * 0.6)
                        desired_sl_move = max_move - buffer
                        if desired_sl_move > 0:
                            desired_sl_price = entry + direction * desired_sl_move * pip
                            current_sl = sl_price_current or state.get("last_sl_price")
                            sl_ok = False
                            if direction > 0:
                                sl_ok = current_sl is None or desired_sl_price > current_sl + pip * 0.3
                            else:
                                sl_ok = current_sl is None or desired_sl_price < current_sl - pip * 0.3
                            if sl_ok:
                                if update_trade_orders(trade_id, sl_price=desired_sl_price):
                                    state["last_sl_price"] = desired_sl_price
                                    logging.info(
                                        "[exit_manager] Tighten SL trade=%s price=%.3f buffer=%.2f",
                                        trade_id,
                                        desired_sl_price,
                                        buffer,
                                    )

                # 4) News spike reversal: quick BE + modest trail
                if strategy == "NewsSpikeReversal":
                    # quicker BE
                    if move_pips >= 7.0:
                        buffer = _be_buffer(strategy, pocket)
                        be_price = entry + direction * (buffer * pip)
                        update_trade_orders(trade_id, sl_price=be_price)
                    # modest trailing
                    dist = 10.0 * pip
                    update_trade_orders(trade_id, trailing_distance=dist)

                # 5) GPT exit advice adjustments（optional）
                #    TrendMA early-exit heuristics run before GPT adjustments
                if strategy == "TrendMA":
                    early_window = age_min <= TRENDMA_EARLY_WINDOW_MIN
                    timeout_window = age_min >= TRENDMA_TIMEOUT_MIN
                    ma20_m1 = float(fac_m1.get("ma20") or 0.0)
                    stall_signal = False
                    panic_signal = False
                    move_profit_hit = False
                    move_loss_hit = False

                    profit_threshold = TRENDMA_EARLY_PROFIT_PIPS
                    loss_threshold = TRENDMA_EARLY_LOSS_PIPS

                    if direction > 0:
                        if ma20_m1:
                            stall_signal = price_now < ma20_m1
                            panic_signal = price_now < ma20_m1 and rsi_m1 < 50.0
                        stall_signal = stall_signal or rsi_m1 < 55.0
                        move_profit_hit = move_pips >= profit_threshold
                        move_loss_hit = move_pips <= -loss_threshold
                    else:
                        if ma20_m1:
                            stall_signal = price_now > ma20_m1
                            panic_signal = price_now > ma20_m1 and rsi_m1 > 50.0
                        stall_signal = stall_signal or rsi_m1 > 45.0
                        move_profit_hit = move_pips >= profit_threshold
                        move_loss_hit = move_pips <= -loss_threshold

                    if early_window and move_profit_hit and stall_signal:
                        if close_trade(trade_id):
                            logging.info(
                                "[exit_manager] TrendMA early profit exit trade=%s move=%.1f",
                                trade_id,
                                move_pips,
                            )
                            _log_event(
                                trade_id=trade_id,
                                strategy=strategy,
                                pocket=pocket,
                                version=version,
                                event_type="early_exit",
                                action="take_profit",
                                advice=None,
                                price=trade_price_now,
                                move_pips=move_pips,
                                note="trendma_early_profit",
                            )
                            continue

                    if early_window and move_loss_hit and panic_signal:
                        if close_trade(trade_id):
                            logging.info(
                                "[exit_manager] TrendMA early cut trade=%s move=%.1f",
                                trade_id,
                                move_pips,
                            )
                            _log_event(
                                trade_id=trade_id,
                                strategy=strategy,
                                pocket=pocket,
                                version=version,
                                event_type="early_exit",
                                action="panic_cut",
                                advice=None,
                                price=trade_price_now,
                                move_pips=move_pips,
                                note="trendma_early_loss",
                            )
                            continue

                    if timeout_window and abs(move_pips) < max(TRENDMA_EARLY_PROFIT_PIPS, 6.0):
                        tp_pips = max(TRENDMA_TIMEOUT_TP_PIPS, 3.0)
                        sl_pips = abs(TRENDMA_TIMEOUT_SL_PIPS)
                        tp_price = entry + direction * (tp_pips * pip)
                        sl_price = entry - direction * (sl_pips * pip)
                        ok = update_trade_orders(trade_id, tp_price=tp_price, sl_price=sl_price)
                        if ok:
                            logging.info(
                                "[exit_manager] TrendMA timeout tighten trade=%s tp=%.3f sl=%.3f",
                                trade_id,
                                tp_price,
                                sl_price,
                            )
                            _log_event(
                                trade_id=trade_id,
                                strategy=strategy,
                                pocket=pocket,
                                version=version,
                                event_type="timeout_adjust",
                                action="tighten",
                                advice=None,
                                price=trade_price_now,
                                move_pips=move_pips,
                                note="trendma_timeout",
                            )

                if advice:
                    conf = float(advice.get("confidence", 0.0))
                    tp_target = float(advice.get("target_tp_pips", 0.0))
                    sl_target = float(advice.get("target_sl_pips", 0.0))
                    if advice.get("close_now") and conf >= ADVICE_CLOSE_CONF:
                        if close_trade(trade_id):
                            logging.info(
                                "[exit_manager] GPT close trade=%s conf=%.2f note=%s",
                                trade_id,
                                conf,
                                advice.get("note", ""),
                            )
                        _log_event(
                            trade_id=trade_id,
                            strategy=strategy,
                            pocket=pocket,
                            version=version,
                            event_type="close_request",
                            action="close_now",
                            advice=advice,
                            price=trade_price_now,
                            move_pips=move_pips,
                        )
                        continue
                    if conf >= ADVICE_ADJUST_CONF:
                        tp_price = None
                        sl_price = None
                        if tp_target > 0.0:
                            tp_price = entry + direction * (tp_target * pip)
                        if sl_target > 0.0:
                            sl_price = entry - direction * (sl_target * pip)
                        if tp_price is not None:
                            if direction > 0 and tp_price <= entry:
                                tp_price = entry + abs(tp_target) * pip
                            if direction < 0 and tp_price >= entry:
                                tp_price = entry - abs(tp_target) * pip
                        if sl_price is not None:
                            if direction > 0 and sl_price >= entry:
                                sl_price = entry - abs(sl_target) * pip
                            if direction < 0 and sl_price <= entry:
                                sl_price = entry + abs(sl_target) * pip
                        if tp_price is not None or sl_price is not None:
                            ok = update_trade_orders(trade_id, tp_price=tp_price, sl_price=sl_price)
                            if ok:
                                logging.info(
                                    "[exit_manager] GPT adjust trade=%s tp=%s sl=%s conf=%.2f",
                                    trade_id,
                                    tp_price,
                                    sl_price,
                                    conf,
                                )
                                _log_event(
                                    trade_id=trade_id,
                                    strategy=strategy,
                                    pocket=pocket,
                                    version=version,
                                    event_type="adjust",
                                    action="update_tp_sl",
                                    advice=advice,
                                    price=trade_price_now,
                                    move_pips=move_pips,
                                )

            open_trade_ids = {tr.get("id") for tr in trades if tr.get("id")}
            if _SCALP_PROFIT_STATE:
                for stale_id in list(_SCALP_PROFIT_STATE.keys()):
                    if stale_id not in open_trade_ids:
                        _SCALP_PROFIT_STATE.pop(stale_id, None)

        except Exception as e:
            logging.exception("[exit_manager] error: %s", e)

        await asyncio.sleep(30)
