import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

import httpx
import pandas as pd
from flask import Flask

from analysis.kaizen import get_policy
from analysis.regime_classifier import (
    classify,
    THRESH_ADX_TREND,
    THRESH_MA_SLOPE,
)
from execution.trade_actions import close_trade, update_trade_orders
from indicators.calc_core import IndicatorEngine
from market_data.candle_fetcher import fetch_historical_candles
from utils.secrets import get_secret

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
app = Flask(__name__)

INSTRUMENT = "USD_JPY"
PIP_VALUE = 0.01

PRICE_TOL = 0.0005  # ≒0.5 pip on JPY pairs
DIST_TOL = 0.0005


def _auth() -> tuple[str, str, Dict[str, str]]:
    token = get_secret("oanda_token")
    account = get_secret("oanda_account_id")
    try:
        practice = get_secret("oanda_practice").lower() == "true"
    except Exception:
        practice = True
    host = "https://api-fxpractice.oanda.com" if practice else "https://api-fxtrade.oanda.com"
    headers = {"Authorization": f"Bearer {token}"}
    return host, account, headers


def _df_from_candles(candles: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = [{k: c[k] for k in ("open", "high", "low", "close")} for c in candles]
    return pd.DataFrame(rows)


def _parse_meta(ext: Dict[str, Any] | None) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not ext:
        return out
    comment = (ext.get("comment") or "").strip()
    for part in comment.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    tag = (ext.get("tag") or "").strip()
    if tag.startswith("pocket="):
        out.setdefault("pocket", tag.split("=", 1)[1])
    return out


def _fetch_open_trades(host: str, account: str, headers: Dict[str, str]) -> List[Dict[str, Any]]:
    url = f"{host}/v3/accounts/{account}/openTrades"
    try:
        with httpx.Client(timeout=7.0) as client:
            r = client.get(url, headers=headers)
            r.raise_for_status()
            return r.json().get("trades", [])
    except httpx.HTTPError as exc:
        logging.error("[exit_manager] openTrades error: %s", exc)
        return []


def _minutes_since(iso_str: str) -> float:
    if not iso_str:
        return 0.0
    try:
        ts = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    return (datetime.now(timezone.utc) - ts.astimezone(timezone.utc)).total_seconds() / 60.0


def _roughly_equal(a: float | None, b: float | None, tol: float) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def _existing_sl_price(trade: Dict[str, Any]) -> float | None:
    sl = trade.get("stopLossOrder") or {}
    price = sl.get("price")
    if price is None:
        return None
    try:
        return float(price)
    except Exception:
        return None


def _existing_trail_distance(trade: Dict[str, Any]) -> float | None:
    tsl = trade.get("trailingStopLossOrder") or {}
    dist = tsl.get("distance")
    if dist is None:
        return None
    try:
        return float(dist)
    except Exception:
        return None


def _existing_tp_price(trade: Dict[str, Any]) -> float | None:
    tp = trade.get("takeProfitOrder") or {}
    price = tp.get("price")
    if price is None:
        return None
    try:
        return float(price)
    except Exception:
        return None


def _fetch_mid_price(host: str, account: str, headers: Dict[str, str]) -> float | None:
    url = f"{host}/v3/accounts/{account}/pricing"
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(url, headers=headers, params={"instruments": INSTRUMENT})
            r.raise_for_status()
            prices = r.json().get("prices", [])
            if not prices:
                return None
            bids = prices[0].get("bids") or []
            asks = prices[0].get("asks") or []
            if bids and asks:
                return (float(bids[0]["price"]) + float(asks[0]["price"])) / 2.0
    except httpx.HTTPError as exc:
        logging.error("[exit_manager] pricing error: %s", exc)
    except Exception:
        pass
    return None


def _adaptive_trail_pips(
    mfe_pips: float,
    move_pips: float,
    min_trail_pips: float,
    macro_regime: str,
    micro_regime: str,
) -> float:
    """Return trailing cushion in pips based on performance and regimes."""
    effective = max(mfe_pips, move_pips, 0.0)
    # Base cushion: keep ~40% of run, but never below safety floor.
    cushion = max(5.0, effective * 0.4)
    cushion = min(cushion, 30.0)

    if macro_regime == "Trend":
        cushion *= 1.1
    elif macro_regime in ("Range", "Mixed"):
        cushion *= 0.9

    if micro_regime in ("Range", "Mixed"):
        cushion *= 0.75

    cushion = max(cushion, min_trail_pips)
    # Do not let cushion exceed 90% of the traveled distance to avoid over-tightening.
    if effective > 0:
        cushion = min(cushion, max(min_trail_pips, effective * 0.9))
    return cushion


def _compute_factors() -> tuple[Dict[str, float], Dict[str, float], List[Dict[str, Any]]]:
    m1 = asyncio_run(fetch_historical_candles(INSTRUMENT, "M1", 120))
    h4 = asyncio_run(fetch_historical_candles(INSTRUMENT, "H4", 60))
    if len(m1) < 20 or len(h4) < 20:
        raise RuntimeError("insufficient candles for exit manager")
    fac_m1 = IndicatorEngine.compute(_df_from_candles(m1))
    fac_h4 = IndicatorEngine.compute(_df_from_candles(h4))
    fac_m1["close"] = float(m1[-1]["close"])
    fac_h4["close"] = float(h4[-1]["close"])
    return fac_m1, fac_h4, m1


def _mfe_pips_since(entry_price: float, direction: int, opened_iso: str, m1_candles: List[Dict[str, Any]]) -> float:
    """Compute MFE in pips since trade open using available M1 candles.

    For JPY pairs, 1 pip = 0.01.
    """
    try:
        opened = datetime.fromisoformat(opened_iso.replace("Z", "+00:00"))
    except Exception:
        return 0.0
    highs: List[float] = []
    lows: List[float] = []
    for c in m1_candles:
        ts = c.get("time")
        if not ts or not isinstance(ts, datetime):
            continue
        if ts < opened:
            continue
        highs.append(float(c.get("high", 0.0) or 0.0))
        lows.append(float(c.get("low", 0.0) or 0.0))
    if not highs or not lows:
        return 0.0
    PIP = PIP_VALUE
    if direction > 0:
        best = max(highs) - entry_price
        return max(best / PIP, 0.0)
    else:
        best = entry_price - min(lows)
        return max(best / PIP, 0.0)


def asyncio_run(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return asyncio.run_coroutine_threadsafe(coro, loop).result()
        return loop.run_until_complete(coro)


@app.route("/", methods=["GET"])
def run_once():
    try:
        host, account, headers = _auth()
    except Exception as exc:
        logging.error("[exit_manager] missing OANDA credentials: %s", exc)
        return "NO_AUTH", 500

    trades = _fetch_open_trades(host, account, headers)
    if not trades:
        return "NO_TRADES", 200

    try:
        fac_m1, fac_h4, m1_candles = _compute_factors()
    except Exception as exc:
        logging.error("[exit_manager] factor compute failed: %s", exc)
        return "NO_FACTORS", 500

    actions = 0
    rsi_m1 = float(fac_m1.get("rsi", 50.0) or 50.0)
    atr_m1 = float(fac_m1.get("atr", 0.0) or 0.0)
    atr_h4 = float(fac_h4.get("atr", 0.0) or 0.0)
    price_now = float(fac_m1.get("close") or 0.0)
    live_mid = _fetch_mid_price(host, account, headers)
    if live_mid is not None:
        price_now = live_mid

    # Regime snapshot for context-aware exits
    try:
        macro_regime = classify(fac_h4, "H4")
    except Exception:
        macro_regime = "Mixed"
    try:
        micro_regime = classify(fac_m1, "M1")
    except Exception:
        micro_regime = "Mixed"

    ma10_h4 = float(fac_h4.get("ma10", 0.0) or 0.0)
    ma20_h4 = float(fac_h4.get("ma20", 0.0) or 0.0)
    adx_h4 = float(fac_h4.get("adx", 0.0) or 0.0)
    slope_h4 = abs(ma20_h4 - ma10_h4) / ma10_h4 if ma10_h4 else 0.0
    bbw_h4 = float(fac_h4.get("bbw", 0.0) or 0.0)
    upper_h4 = ma20_h4 + ma20_h4 * bbw_h4 / 2 if ma20_h4 and bbw_h4 else None
    lower_h4 = ma20_h4 - ma20_h4 * bbw_h4 / 2 if ma20_h4 and bbw_h4 else None
    ma20_m1 = float(fac_m1.get("ma20", 0.0) or 0.0)
    bbw_m1 = float(fac_m1.get("bbw", 0.0) or 0.0)
    band_span_m1 = ma20_m1 * bbw_m1 / 2 if ma20_m1 and bbw_m1 else 0.0
    upper_m1 = ma20_m1 + band_span_m1 if band_span_m1 else None
    lower_m1 = ma20_m1 - band_span_m1 if band_span_m1 else None

    for tr in trades:
        trade_id = tr.get("id")
        if not trade_id:
            continue
        units = int(tr.get("initialUnits", 0))
        if units == 0:
            continue
        direction = 1 if units > 0 else -1
        entry_price = float(tr.get("price", 0.0) or 0.0)
        if entry_price == 0.0 or price_now == 0.0:
            continue

        ext = tr.get("clientExtensions") or {}
        meta = _parse_meta(ext)
        strategy = meta.get("strategy") or ""
        pocket = meta.get("pocket") or ("macro" if abs(units) >= 100000 else "micro")

        trade_price_now = price_now
        cur = tr.get("currentPrice") or {}
        mid = cur.get("mid")
        bid = cur.get("bid")
        ask = cur.get("ask")
        try:
            if mid is not None:
                trade_price_now = float(mid)
            elif bid is not None and ask is not None:
                trade_price_now = (float(bid) + float(ask)) / 2.0
        except (TypeError, ValueError):
            pass

        price_now = trade_price_now
        move_pips = (price_now - entry_price) * 100.0 * direction
        be_price = entry_price + direction * (1 * PIP_VALUE)

        try:
            pol = get_policy(strategy or "", pocket)
        except Exception:
            pol = {}
        be_trig = float(pol.get("be_trigger_pips", 10.0 if pocket == "micro" else 20.0))
        atr_mult = float(pol.get("trail_atr_mult", 1.0 if pocket == "micro" else 1.5))
        min_trail = float(pol.get("min_trail_pips", 8.0 if pocket == "micro" else 15.0))
        meanrev_max_min = float(pol.get("meanrev_max_min", 30.0))
        meanrev_rsi_exit = float(pol.get("meanrev_rsi_exit", 50.0))

        # 1) Break-even move
        if move_pips >= be_trig:
            existing_sl = _existing_sl_price(tr)
            if not _roughly_equal(existing_sl, be_price, PRICE_TOL):
                if update_trade_orders(trade_id, sl_price=be_price):
                    logging.info("[exit_manager] BE set trade=%s price=%.3f", trade_id, be_price)
                    actions += 1

        # 2) Regime/momentum aware market exits for trend/breakout
        if strategy in ("TrendMA", "Donchian55"):
            # Early regime flip: H4 trend breaks or ADX/slope deteriorates
            regime_break = False
            if direction > 0:
                if (ma10_h4 <= ma20_h4) or (adx_h4 < THRESH_ADX_TREND["H4"] - 2) or (slope_h4 < THRESH_MA_SLOPE["H4"] / 2):
                    regime_break = True
            else:
                if (ma10_h4 >= ma20_h4) or (adx_h4 < THRESH_ADX_TREND["H4"] - 2) or (slope_h4 < THRESH_MA_SLOPE["H4"] / 2):
                    regime_break = True

            # Micro momentum fail: M1 で逆行（MA20とRSIで確認）
            m1_fail = False
            if direction > 0:
                if (price_now < ma20_m1 and rsi_m1 < 48.0 and micro_regime in ("Range", "Mixed")):
                    m1_fail = True
            else:
                if (price_now > ma20_m1 and rsi_m1 > 52.0 and micro_regime in ("Range", "Mixed")):
                    m1_fail = True

            # Gave‑back exit: lock gains after sizeable MFE
            mfe = _mfe_pips_since(entry_price, direction, tr.get("openTime", ""), m1_candles) if tr.get("openTime") else 0.0
            adaptive_trail_pips = _adaptive_trail_pips(mfe, move_pips, min_trail, macro_regime, micro_regime)

            peak_reason = None
            if direction > 0:
                margin_h4 = max(0.0007, abs(upper_h4 - ma20_h4) * 0.18) if upper_h4 else None
                margin_m1 = max(0.0004, abs(upper_m1 - ma20_m1) * 0.2) if upper_m1 else None
                near_upper = False
                if upper_h4 and margin_h4:
                    near_upper = price_now >= upper_h4 - margin_h4
                if not near_upper and upper_m1 and margin_m1:
                    near_upper = price_now >= upper_m1 - margin_m1
                if near_upper and move_pips >= 5.0:
                    peak_reason = "band"
                elif rsi_m1 >= 74.0 and move_pips >= 3.0:
                    peak_reason = "rsi"
            else:
                margin_h4 = max(0.0007, abs(lower_h4 - ma20_h4) * 0.18) if lower_h4 else None
                margin_m1 = max(0.0004, abs(lower_m1 - ma20_m1) * 0.2) if lower_m1 else None
                near_lower = False
                if lower_h4 and margin_h4:
                    near_lower = price_now <= lower_h4 + margin_h4
                if not near_lower and lower_m1 and margin_m1:
                    near_lower = price_now <= lower_m1 + margin_m1
                if near_lower and move_pips >= 5.0:
                    peak_reason = "band"
                elif rsi_m1 <= 26.0 and move_pips >= 3.0:
                    peak_reason = "rsi"

            if peak_reason:
                if close_trade(trade_id):
                    logging.info(
                        "[exit_manager] Peak exit trade=%s reason=%s price=%.3f move=%.1f rsi=%.1f",
                        trade_id,
                        peak_reason,
                        price_now,
                        move_pips,
                        rsi_m1,
                    )
                    actions += 1
                    continue
            gaveback_exit = False
            if mfe >= max(adaptive_trail_pips, 8.0):
                lock = max(5.0, min(12.0, adaptive_trail_pips * 0.6))
                if (mfe - move_pips) >= lock:
                    gaveback_exit = True

            # Stale/timeout exit: very long hold without progress
            age_min = _minutes_since(tr.get("openTime", ""))
            stale_exit = age_min >= 240 and abs(move_pips) < 3.0

            if regime_break or m1_fail or gaveback_exit or stale_exit:
                if close_trade(trade_id):
                    logging.info(
                        "[exit_manager] MARKET EXIT trade=%s reason=%s mfe=%.1f move=%.1f (H4_adx=%.1f slope=%.4f regimes=%s/%s age=%.0fm)",
                        trade_id,
                        "regime_break" if regime_break else ("m1_fail" if m1_fail else ("gaveback" if gaveback_exit else "stale")),
                        mfe,
                        move_pips,
                        adx_h4,
                        slope_h4,
                        macro_regime,
                        micro_regime,
                        age_min,
                    )
                    actions += 1
                    # proceed to next trade (avoid also updating trailing)
                    continue

            desired_tp_price = None
            if direction > 0:
                margin_h4 = max(0.0007, abs(upper_h4 - ma20_h4) * 0.18) if upper_h4 else None
                margin_m1 = max(0.0004, abs(upper_m1 - ma20_m1) * 0.2) if upper_m1 else None
                target = upper_h4 if upper_h4 else upper_m1
                margin = margin_h4 if margin_h4 else margin_m1
                if target and margin:
                    candidate = target - margin / 2
                    if candidate > price_now + PRICE_TOL:
                        desired_tp_price = candidate
            else:
                margin_h4 = max(0.0007, abs(lower_h4 - ma20_h4) * 0.18) if lower_h4 else None
                margin_m1 = max(0.0004, abs(lower_m1 - ma20_m1) * 0.2) if lower_m1 else None
                target = lower_h4 if lower_h4 else lower_m1
                margin = margin_h4 if margin_h4 else margin_m1
                if target and margin:
                    candidate = target + margin / 2
                    if candidate < price_now - PRICE_TOL:
                        desired_tp_price = candidate

            if desired_tp_price is not None:
                existing_tp = _existing_tp_price(tr)
                if not _roughly_equal(existing_tp, desired_tp_price, PRICE_TOL):
                    if update_trade_orders(trade_id, tp_price=round(desired_tp_price, 3)):
                        logging.info(
                            "[exit_manager] Trend TP adjust trade=%s tp=%.3f",
                            trade_id,
                            desired_tp_price,
                        )
                        actions += 1

        # 3) Trailing for trend/breakout (adaptive)
        if strategy in ("TrendMA", "Donchian55"):
            atr = atr_h4 if pocket == "macro" and atr_h4 else atr_m1
            base_trail = max(min_trail * PIP_VALUE, atr_mult * atr)
            adaptive_trail = adaptive_trail_pips * PIP_VALUE
            dist = max(base_trail, adaptive_trail)
            if dist > 0:
                trail_sl = price_now - direction * dist
                safe = trail_sl >= be_price if direction > 0 else trail_sl <= be_price
                existing_dist = _existing_trail_distance(tr)
                if safe and not _roughly_equal(existing_dist, dist, DIST_TOL):
                    if update_trade_orders(trade_id, trailing_distance=dist):
                        logging.info(
                            "[exit_manager] Trail set trade=%s dist=%.3f (%s/%s) move=%.1f",
                            trade_id,
                            dist,
                            pocket,
                            strategy,
                            move_pips,
                        )
                        actions += 1

        # 4) Mean-reversion exit logic
        if strategy == "BB_RSI":
            age_min = _minutes_since(tr.get("openTime", ""))
            band_span = ma20_m1 * bbw_m1 / 2 if ma20_m1 and bbw_m1 else 0.0
            band_margin = max(0.0005, abs(band_span) * 0.15)
            upper_band = ma20_m1 + band_span if band_span else None
            lower_band = ma20_m1 - band_span if band_span else None

            hit_band = False
            if direction > 0 and upper_band is not None:
                hit_band = price_now >= (upper_band - band_margin)
            elif direction < 0 and lower_band is not None:
                hit_band = price_now <= (lower_band + band_margin)

            target_rsi = max(meanrev_rsi_exit, 67.0) if direction > 0 else min(meanrev_rsi_exit, 33.0)
            rsi_exit_hit = (rsi_m1 >= target_rsi) if direction > 0 else (rsi_m1 <= target_rsi)

            atr_push = move_pips >= max(8.0, atr_m1 * 100 * 1.3)
            ttl_hit = age_min >= min(meanrev_max_min, 60.0)

            if hit_band or rsi_exit_hit or atr_push or ttl_hit:
                if close_trade(trade_id):
                    logging.info(
                        "[exit_manager] MeanRev exit trade=%s reason=%s price=%.3f rsi=%.1f move=%.1f",
                        trade_id,
                        "band" if hit_band else ("rsi" if rsi_exit_hit else ("atr" if atr_push else "ttl")),
                        price_now,
                        rsi_m1,
                        move_pips,
                    )
                    actions += 1
                    continue

            desired_tp = None
            if direction > 0 and upper_band is not None:
                desired_tp = upper_band - band_margin / 2
            elif direction < 0 and lower_band is not None:
                desired_tp = lower_band + band_margin / 2
            if desired_tp is not None and not _roughly_equal(_existing_tp_price(tr), desired_tp, PRICE_TOL):
                if update_trade_orders(trade_id, tp_price=round(desired_tp, 3)):
                    logging.info("[exit_manager] MeanRev TP adjust trade=%s tp=%.3f", trade_id, desired_tp)
                    actions += 1

        # 5) News spike reversal quick adjustments
        if strategy == "NewsSpikeReversal":
            if move_pips >= 7.0:
                if update_trade_orders(trade_id, sl_price=be_price):
                    logging.info("[exit_manager] NewsSpike BE trade=%s", trade_id)
                    actions += 1
            dist = 10.0 * PIP_VALUE
            if update_trade_orders(trade_id, trailing_distance=dist):
                logging.info("[exit_manager] NewsSpike trail trade=%s", trade_id)
                actions += 1

    return f"actions={actions}", 200


if __name__ == "__main__":
    import os

    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
