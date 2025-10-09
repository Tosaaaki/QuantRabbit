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
        print(f"[exit_manager] openTrades error: {e}")
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

_ADVICE_CACHE: Dict[str, Dict[str, Any]] = {}


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
    }

    while True:
        try:
            trades = _get_open_trades()
            if not trades:
                await asyncio.sleep(30)
                continue

            factors = all_factors()
            fac_m1 = factors.get("M1") or {}
            fac_h4 = factors.get("H4") or {}
            price_now = float(fac_m1.get("close") or 0.0)  # USD/JPY fallback price
            atr_m1 = float(fac_m1.get("atr") or 0.0)
            atr_h4 = float(fac_h4.get("atr") or 0.0)
            rsi_m1 = float(fac_m1.get("rsi") or 50.0)

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
                try:
                    if mid_price is not None:
                        trade_price_now = float(mid_price)
                    else:
                        bid = trade_price.get("bid")
                        ask = trade_price.get("ask")
                        if bid is not None and ask is not None:
                            trade_price_now = (float(bid) + float(ask)) / 2.0
                except (TypeError, ValueError):
                    pass

                move_pips = (trade_price_now - entry) * direction / pip
                age_min = _minutes_since(tr.get("openTime", ""))

                # Load per (strategy,pocket) policy
                try:
                    pol = get_policy(strategy or "", pocket)
                    be_trig = float(pol.get("be_trigger_pips", FALLBACK[pocket]["be"]))
                    atr_mult = float(pol.get("trail_atr_mult", FALLBACK[pocket]["atr"]))
                    min_trail = float(pol.get("min_trail_pips", FALLBACK[pocket]["mintrail"]))
                    meanrev_max_min = float(pol.get("meanrev_max_min", 30.0))
                    meanrev_rsi_exit = float(pol.get("meanrev_rsi_exit", 50.0))
                except Exception:
                    be_trig = FALLBACK[pocket]["be"]
                    atr_mult = FALLBACK[pocket]["atr"]
                    min_trail = FALLBACK[pocket]["mintrail"]
                    meanrev_max_min = 30.0
                    meanrev_rsi_exit = 50.0

                advice: Dict[str, Any] | None = None
                if abs(move_pips) >= ADVICE_TRIGGER_PIPS:
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
                        advice = await advise_or_fallback(payload)
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
                    be_price = entry + direction * (1 * pip)  # BE + 1 pip
                    ok = update_trade_orders(trade_id, sl_price=be_price)
                    if ok:
                        print(f"[exit_manager] BE set for {trade_id} at {be_price:.3f}")

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
                                print(
                                    f"[exit_manager] Trail set for {trade_id} dist={dist:.3f} ({pocket}, {strategy})"
                                )

                # 3) Mean-reversion exit: time/RSI
                if strategy == "BB_RSI":
                    rsi_exit_hit = (
                        rsi_m1 >= meanrev_rsi_exit if direction > 0 else rsi_m1 <= meanrev_rsi_exit
                    )
                    if age_min >= meanrev_max_min or rsi_exit_hit:
                        ok = close_trade(trade_id)
                        if ok:
                            print(
                                f"[exit_manager] MeanRev exit for {trade_id} age={age_min:.1f}m rsi={rsi_m1:.1f}"
                            )

                # 4) News spike reversal: quick BE + modest trail
                if strategy == "NewsSpikeReversal":
                    # quicker BE
                    if move_pips >= 7.0:
                        be_price = entry + direction * (1 * pip)
                        update_trade_orders(trade_id, sl_price=be_price)
                    # modest trailing
                    dist = 10.0 * pip
                    update_trade_orders(trade_id, trailing_distance=dist)

                # 5) GPT exit advice adjustments（optional）
                if advice:
                    conf = float(advice.get("confidence", 0.0))
                    tp_target = float(advice.get("target_tp_pips", 0.0))
                    sl_target = float(advice.get("target_sl_pips", 0.0))
                    if advice.get("close_now") and conf >= ADVICE_CLOSE_CONF:
                        if close_trade(trade_id):
                            print(
                                f"[exit_manager] GPT close {trade_id} conf={conf:.2f} note={advice.get('note','')}"
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
                                print(
                                    f"[exit_manager] GPT adjust trade={trade_id} tp={tp_price} sl={sl_price} conf={conf:.2f}"
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

        except Exception as e:
            print(f"[exit_manager] error: {e}")

        await asyncio.sleep(30)
