"""London session momentum worker."""

from __future__ import annotations
from analysis.ma_projection import compute_adx_projection, compute_bbw_projection, compute_ma_projection, compute_rsi_projection

import asyncio
import datetime
import logging
from typing import Optional

from execution.order_manager import market_order
from execution.position_manager import PositionManager
from execution.risk_guard import allowed_lot
from execution.stage_tracker import StageTracker
from indicators.factor_cache import all_factors, get_candles_snapshot
from market_data import spread_monitor
from utils.oanda_account import get_account_snapshot

from . import config

import os
_BB_ENTRY_ENABLED = os.getenv("BB_ENTRY_ENABLED", "1").strip().lower() not in {"", "0", "false", "no"}
_BB_ENTRY_REVERT_PIPS = float(os.getenv("BB_ENTRY_REVERT_PIPS", "2.4"))
_BB_ENTRY_REVERT_RATIO = float(os.getenv("BB_ENTRY_REVERT_RATIO", "0.22"))
_BB_ENTRY_TREND_EXT_PIPS = float(os.getenv("BB_ENTRY_TREND_EXT_PIPS", "3.5"))
_BB_ENTRY_TREND_EXT_RATIO = float(os.getenv("BB_ENTRY_TREND_EXT_RATIO", "0.40"))
_BB_ENTRY_SCALP_REVERT_PIPS = float(os.getenv("BB_ENTRY_SCALP_REVERT_PIPS", "2.0"))
_BB_ENTRY_SCALP_REVERT_RATIO = float(os.getenv("BB_ENTRY_SCALP_REVERT_RATIO", "0.20"))
_BB_ENTRY_SCALP_EXT_PIPS = float(os.getenv("BB_ENTRY_SCALP_EXT_PIPS", "2.4"))
_BB_ENTRY_SCALP_EXT_RATIO = float(os.getenv("BB_ENTRY_SCALP_EXT_RATIO", "0.30"))
_BB_PIP = 0.01


def _bb_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bb_levels(fac):
    if not fac:
        return None
    upper = _bb_float(fac.get("bb_upper"))
    lower = _bb_float(fac.get("bb_lower"))
    mid = _bb_float(fac.get("bb_mid")) or _bb_float(fac.get("ma20"))
    bbw = _bb_float(fac.get("bbw")) or 0.0
    if upper is None or lower is None:
        if mid is None or bbw <= 0:
            return None
        half = abs(mid) * bbw / 2.0
        upper = mid + half
        lower = mid - half
    span = upper - lower
    if span <= 0:
        return None
    return upper, mid if mid is not None else (upper + lower) / 2.0, lower, span, span / _BB_PIP


def _bb_entry_allowed(style, side, price, fac_m1, *, range_active=None):
    if not _BB_ENTRY_ENABLED:
        return True
    if price is None or price <= 0:
        return True
    levels = _bb_levels(fac_m1)
    if not levels:
        return True
    upper, mid, lower, span, span_pips = levels
    side_key = str(side or "").lower()
    if side_key in {"buy", "long", "open_long"}:
        direction = "long"
    else:
        direction = "short"
    orig_style = style
    if style == "scalp" and range_active:
        style = "reversion"
    if style == "reversion":
        base_pips = _BB_ENTRY_SCALP_REVERT_PIPS if orig_style == "scalp" else _BB_ENTRY_REVERT_PIPS
        base_ratio = _BB_ENTRY_SCALP_REVERT_RATIO if orig_style == "scalp" else _BB_ENTRY_REVERT_RATIO
        threshold = max(base_pips, span_pips * base_ratio)
        if direction == "long":
            dist = (price - lower) / _BB_PIP
        else:
            dist = (upper - price) / _BB_PIP
        return dist <= threshold
    if direction == "long":
        if price < mid:
            return False
        ext = max(0.0, price - upper) / _BB_PIP
    else:
        if price > mid:
            return False
        ext = max(0.0, lower - price) / _BB_PIP
    max_ext = max(_BB_ENTRY_TREND_EXT_PIPS, span_pips * _BB_ENTRY_TREND_EXT_RATIO)
    if orig_style == "scalp":
        max_ext = max(_BB_ENTRY_SCALP_EXT_PIPS, span_pips * _BB_ENTRY_SCALP_EXT_RATIO)
    return ext <= max_ext

BB_STYLE = "trend"

LOG = logging.getLogger(__name__)
PIP = 0.01



_PROJ_TF_MINUTES = {"M1": 1.0, "M5": 5.0, "H1": 60.0, "H4": 240.0, "D1": 1440.0}


def _projection_mode(pocket, mode_override=None):
    if mode_override:
        return mode_override
    if globals().get("IS_RANGE"):
        return "range"
    if globals().get("IS_PULLBACK"):
        return "pullback"
    if pocket in {"scalp", "scalp_fast"}:
        return "scalp"
    return "trend"


def _projection_tfs(pocket, mode):
    if pocket == "macro":
        return ("H4", "H1")
    if pocket == "micro":
        return ("M5", "M1")
    if pocket in {"scalp", "scalp_fast"}:
        return ("M1",)
    return ("M5", "M1")


def _projection_candles(tfs):
    for tf in tfs:
        candles = get_candles_snapshot(tf, limit=120)
        if candles and len(candles) >= 30:
            return tf, list(candles)
    return None, None


def _score_ma(ma, side, opp_block_bars):
    if ma is None:
        return None
    align = ma.gap_pips >= 0 if side == "long" else ma.gap_pips <= 0
    cross_soon = ma.projected_cross_bars is not None and ma.projected_cross_bars <= opp_block_bars
    if align and not cross_soon:
        return 0.7
    if align and cross_soon:
        return -0.4
    if cross_soon:
        return -0.8
    return -0.5


def _score_rsi(rsi, side, long_target, short_target, overheat_bars):
    if rsi is None:
        return None
    score = 0.0
    if side == "long":
        if rsi.rsi >= long_target and rsi.slope_per_bar > 0:
            score = 0.4
        elif rsi.rsi <= (long_target - 8) and rsi.slope_per_bar < 0:
            score = -0.4
        if rsi.eta_upper_bars is not None and rsi.eta_upper_bars <= overheat_bars:
            score -= 0.2
    else:
        if rsi.rsi <= short_target and rsi.slope_per_bar < 0:
            score = 0.4
        elif rsi.rsi >= (short_target + 8) and rsi.slope_per_bar > 0:
            score = -0.4
        if rsi.eta_lower_bars is not None and rsi.eta_lower_bars <= overheat_bars:
            score -= 0.2
    return score


def _score_adx(adx, trend_mode, threshold):
    if adx is None:
        return None
    if trend_mode:
        if adx.adx >= threshold and adx.slope_per_bar >= 0:
            return 0.4
        if adx.adx <= threshold and adx.slope_per_bar < 0:
            return -0.4
        return 0.0
    if adx.adx >= threshold and adx.slope_per_bar > 0:
        return -0.5
    if adx.adx <= threshold and adx.slope_per_bar < 0:
        return 0.3
    return 0.0


def _score_bbw(bbw, threshold):
    if bbw is None:
        return None
    if bbw.bbw <= threshold and bbw.slope_per_bar <= 0:
        return 0.5
    if bbw.bbw > threshold and bbw.slope_per_bar > 0:
        return -0.5
    return 0.0


def _projection_decision(side, pocket, mode_override=None):
    mode = _projection_mode(pocket, mode_override=mode_override)
    tfs = _projection_tfs(pocket, mode)
    tf, candles = _projection_candles(tfs)
    if not candles:
        return True, 1.0, {}
    minutes = _PROJ_TF_MINUTES.get(tf, 1.0)

    if mode == "trend":
        params = {
            "adx_threshold": 20.0,
            "bbw_threshold": 0.16,
            "opp_block_bars": 5.0,
            "long_target": 52.0,
            "short_target": 48.0,
            "overheat_bars": 3.0,
            "weights": {"ma": 0.45, "rsi": 0.25, "adx": 0.30},
            "block_score": -0.6,
            "size_scale": 0.18,
        }
    elif mode == "pullback":
        params = {
            "adx_threshold": 18.0,
            "bbw_threshold": 0.16,
            "opp_block_bars": 4.0,
            "long_target": 50.0,
            "short_target": 50.0,
            "overheat_bars": 3.0,
            "weights": {"ma": 0.40, "rsi": 0.40, "adx": 0.20},
            "block_score": -0.55,
            "size_scale": 0.15,
        }
    elif mode == "scalp":
        params = {
            "adx_threshold": 18.0,
            "bbw_threshold": 0.16,
            "opp_block_bars": 3.0,
            "long_target": 52.0,
            "short_target": 48.0,
            "overheat_bars": 2.0,
            "weights": {"ma": 0.50, "rsi": 0.30, "adx": 0.20},
            "block_score": -0.6,
            "size_scale": 0.12,
        }
    else:
        params = {
            "adx_threshold": 16.0,
            "bbw_threshold": 0.14,
            "opp_block_bars": 4.0,
            "long_target": 45.0,
            "short_target": 55.0,
            "overheat_bars": 3.0,
            "weights": {"bbw": 0.40, "rsi": 0.35, "adx": 0.25},
            "block_score": -0.5,
            "size_scale": 0.15,
        }

    ma = compute_ma_projection({"candles": candles}, timeframe_minutes=minutes)
    rsi = compute_rsi_projection(candles, timeframe_minutes=minutes)
    adx = compute_adx_projection(candles, timeframe_minutes=minutes, trend_threshold=params["adx_threshold"])
    bbw = None
    if mode == "range":
        bbw = compute_bbw_projection(candles, timeframe_minutes=minutes, squeeze_threshold=params["bbw_threshold"])

    scores = {}
    ma_score = _score_ma(ma, side, params["opp_block_bars"])
    if ma_score is not None and "ma" in params["weights"]:
        scores["ma"] = ma_score
    rsi_score = _score_rsi(rsi, side, params["long_target"], params["short_target"], params["overheat_bars"])
    if rsi_score is not None and "rsi" in params["weights"]:
        scores["rsi"] = rsi_score
    adx_score = _score_adx(adx, mode != "range", params["adx_threshold"])
    if adx_score is not None and "adx" in params["weights"]:
        scores["adx"] = adx_score
    bbw_score = _score_bbw(bbw, params["bbw_threshold"])
    if bbw_score is not None and "bbw" in params["weights"]:
        scores["bbw"] = bbw_score

    weight_sum = 0.0
    score_sum = 0.0
    for key, score in scores.items():
        weight = params["weights"].get(key, 0.0)
        weight_sum += weight
        score_sum += weight * score
    score = score_sum / weight_sum if weight_sum > 0 else 0.0

    allow = score > params["block_score"]
    size_mult = 1.0 + max(0.0, score) * params["size_scale"]
    size_mult = max(0.8, min(1.35, size_mult))

    detail = {
        "mode": mode,
        "tf": tf,
        "score": round(score, 3),
        "size_mult": round(size_mult, 3),
        "scores": {k: round(v, 3) for k, v in scores.items()},
    }
    return allow, size_mult, detail
def _time_in_window(now: datetime.datetime) -> bool:
    start_h, start_m = [int(part) for part in config.SESSION_START_UTC.split(":")]
    end_h, end_m = [int(part) for part in config.SESSION_END_UTC.split(":")]
    start = now.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
    end = now.replace(hour=end_h, minute=end_m, second=0, microsecond=0)
    if end <= start:
        end += datetime.timedelta(days=1)
        if now < start:
            now += datetime.timedelta(days=1)
    return start <= now <= end


async def london_momentum_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s disabled via config", config.LOG_PREFIX)
        return

    LOG.info("%s worker starting (interval=%.1fs)", config.LOG_PREFIX, config.LOOP_INTERVAL_SEC)
    stage_tracker = StageTracker()
    pos_manager = PositionManager()
    try:
        while True:
            try:
                now = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
                if not _time_in_window(now):
                    await asyncio.sleep(max(5.0, config.LOOP_INTERVAL_SEC * 4))
                    continue

                factors = all_factors()
                fac_m1 = dict(factors.get("M1") or {})
                fac_m5 = dict(factors.get("M5") or {})
                fac_h1 = dict(factors.get("H1") or {})
                fac_h4 = dict(factors.get("H4") or {})
                price = fac_m5.get("close")
                ema20_h1 = fac_h1.get("ema20")
                ema50_h1 = fac_h1.get("ema50") or fac_h1.get("ma50")
                ema20_m5 = fac_m5.get("ema20")
                if any(val is None for val in (price, ema20_h1, ema50_h1, ema20_m5)):
                    await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                    continue

                try:
                    spread_blocked, _, spread_state, spread_reason = spread_monitor.is_blocked()
                except Exception:
                    spread_blocked = False
                    spread_state = None
                    spread_reason = ""
                spread_pips = float((spread_state or {}).get("spread_pips", 0.0) or 0.0)
                if spread_blocked or spread_pips > config.MAX_SPREAD_PIPS:
                    if spread_reason:
                        LOG.info("%s spread guard: %s", config.LOG_PREFIX, spread_reason)
                    await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                    continue

                trend_gap = float(ema20_h1) - float(ema50_h1)
                if abs(trend_gap) < config.TREND_GAP_MIN:
                    await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                    continue

                direction = "long" if trend_gap > 0 else "short"
                momentum = float(price) - float(ema20_m5)
                if direction == "short":
                    momentum = -momentum
                if momentum < config.MOMENTUM_MIN:
                    await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                    continue

                adx_m5 = float(fac_m5.get("adx") or 0.0)
                bbw_m5 = float(fac_m5.get("bbw") or 0.0)
                vwap_m5 = fac_m5.get("vwap")
                price_f = float(price)
                range_flag = False
                if bbw_m5 > 0.0 and bbw_m5 < 0.0010 and adx_m5 < 16.0:
                    range_flag = True
                    LOG.debug("%s skip: range/low-vol (adx=%.2f bbw=%.5f)", config.LOG_PREFIX, adx_m5, bbw_m5)
                    await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                    continue
                if vwap_m5 is not None:
                    try:
                        vwap_gap = abs(price_f - float(vwap_m5)) / PIP
                        if vwap_gap < 0.8:
                            LOG.debug("%s skip: vwap too close gap=%.2fp", config.LOG_PREFIX, vwap_gap)
                            await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                            continue
                    except Exception:
                        pass

                atr_pips = fac_m5.get("atr_pips")
                if atr_pips is None:
                    atr_raw = fac_m5.get("atr")
                    atr_pips = (atr_raw or 0.0) * 100.0
                try:
                    atr_pips = float(atr_pips or 0.0)
                except (TypeError, ValueError):
                    atr_pips = 0.0
                if atr_pips < config.MIN_ATR_PIPS:
                    await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                    continue

                pockets = pos_manager.get_open_positions()
                pocket_info = pockets.get(config.POCKET) or {}
                if pocket_info.get("open_trades"):
                    await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                    continue

                blocked, remain, reason = stage_tracker.is_blocked(config.POCKET, direction, now=now)
                if blocked:
                    LOG.debug(
                        "%s cooldown pocket=%s dir=%s remain=%s reason=%s",
                        config.LOG_PREFIX,
                        config.POCKET,
                        direction,
                        remain,
                        reason or "cooldown",
                    )
                    await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                    continue

                try:
                    snapshot = get_account_snapshot()
                    equity = snapshot.nav or snapshot.balance
                    margin_available = snapshot.margin_available
                    margin_rate = snapshot.margin_rate
                except Exception:
                    equity = None
                    margin_available = None
                    margin_rate = None

                base_equity = equity or pocket_info.get("pocket_equity") or 10_000.0
                strategy_tag = "LondonMomentum"
                lot = allowed_lot(
                    base_equity,
                    sl_pips=config.SL_PIPS,
                    margin_available=margin_available,
                    price=price,
                    margin_rate=margin_rate,
                    risk_pct_override=config.RISK_PCT,
                    pocket=config.POCKET,
                    side=direction,
                    strategy_tag=strategy_tag,
                    fac_m1=fac_m1,
                    fac_h4=fac_h4,
                )
                units = int(round(lot * 100000))
                if units < config.MIN_UNITS:
                    await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                    continue
                if direction == "short":
                    units = -units

                # 市況に応じて TP/lot/Cooldown を可変化
                lot_scale = 1.0
                tp_pips = config.TP_PIPS
                if atr_pips <= 2.5:
                    lot_scale *= 0.9
                    tp_pips = max(2.0, tp_pips * 0.7)
                elif atr_pips >= 6.0:
                    lot_scale *= 1.1
                    tp_pips = min(12.0, tp_pips * 1.15)
                if vwap_m5 is not None:
                    try:
                        vwap_gap = abs(price_f - float(vwap_m5)) / PIP
                        if vwap_gap >= 1.6:
                            tp_pips = min(12.0, tp_pips + 0.6)
                        elif vwap_gap <= 0.9:
                            tp_pips = max(1.8, tp_pips * 0.85)
                    except Exception:
                        pass
                if range_flag:
                    lot_scale *= 0.9
                    tp_pips = max(2.0, tp_pips * 0.8)

                units = int(round(units * lot_scale))
                if abs(units) < config.MIN_UNITS:
                    await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                    continue

                sl_delta = config.SL_PIPS * 0.01
                tp_delta = tp_pips * 0.01
                price_float = float(price)
                if direction == "long":
                    sl_price = round(price_float - sl_delta, 3)
                    tp_price = round(price_float + tp_delta, 3)
                else:
                    sl_price = round(price_float + sl_delta, 3)
                    tp_price = round(price_float - tp_delta, 3)

                if not _bb_entry_allowed(BB_STYLE, direction, price_float, fac_m1):
                    await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                    continue

                entry_thesis = {
                    "strategy_tag": "LondonMomentum",
                    "trend_gap": round(trend_gap, 5),
                    "momentum": round(momentum, 5),
                    "atr_pips": round(atr_pips, 2),
                    "spread_pips": round(spread_pips, 2),
                    "tp_pips": round(tp_pips, 2),
                    "sl_pips": round(config.SL_PIPS, 2),
                    "hard_stop_pips": round(config.SL_PIPS, 2),
                }
                proj_allow, proj_mult, proj_detail = _projection_decision(direction, config.POCKET)
                if not proj_allow:
                    await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                    continue
                if proj_detail:
                    entry_thesis["projection"] = proj_detail
                if proj_mult > 1.0:
                    sign = 1 if units > 0 else -1
                    units = int(round(abs(units) * proj_mult)) * sign

                try:
                    candle_allow, candle_mult = _entry_candle_guard("long" if units > 0 else "short")
                    if not candle_allow:
                        continue
                    if candle_mult != 1.0:
                        sign = 1 if units > 0 else -1
                        units = int(round(abs(units) * candle_mult)) * sign
                    trade_id = await market_order(
                        "USD_JPY",
                        units,
                        sl_price=sl_price,
                        tp_price=tp_price,
                        pocket=config.POCKET,
                        client_order_id=f"qr-lm-{int(now.timestamp()*1000)}",
                        entry_thesis=entry_thesis,
                    )
                except Exception as exc:  # pragma: no cover
                    LOG.error("%s order error dir=%s err=%s", config.LOG_PREFIX, direction, exc)
                    trade_id = None

                if not trade_id:
                    await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                    continue

                LOG.info(
                    "%s entry trade=%s dir=%s units=%s sl=%.3f tp=%.3f",
                    config.LOG_PREFIX,
                    trade_id,
                    direction,
                    units,
                    sl_price,
                    tp_price,
                )
                pos_manager.register_open_trade(trade_id, config.POCKET)
                stage_tracker.set_stage(config.POCKET, direction, 1, now=now)
                cooldown = config.COOLDOWN_SEC
                if range_flag:
                    cooldown = max(30.0, cooldown * 0.8)
                elif atr_pips >= 6.0:
                    cooldown = min(cooldown * 0.9, cooldown)
                stage_tracker.set_cooldown(
                    config.POCKET,
                    direction,
                    reason="lm_entry",
                    seconds=cooldown,
                    now=now,
                )

                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                LOG.exception("%s loop error: %s", config.LOG_PREFIX, exc)
                await asyncio.sleep(max(3.0, config.LOOP_INTERVAL_SEC))
    except asyncio.CancelledError:
        LOG.info("%s worker cancelled", config.LOG_PREFIX)
        raise
    finally:
        try:
            stage_tracker.close()
        except Exception:  # pragma: no cover
            pass
        try:
            pos_manager.close()
        except Exception:  # pragma: no cover
            pass


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(london_momentum_worker())


_CANDLE_PIP = 0.01
_CANDLE_MIN_CONF = 0.35
_CANDLE_ENTRY_BLOCK = -0.7
_CANDLE_ENTRY_SCALE = 0.2
_CANDLE_ENTRY_MIN = 0.8
_CANDLE_ENTRY_MAX = 1.2
_CANDLE_WORKER_NAME = (__file__.replace("\\", "/").split("/")[-2] if "/" in __file__ else "").lower()


def _candle_tf_for_worker() -> str:
    name = _CANDLE_WORKER_NAME
    if "macro" in name or "trend_h1" in name or "manual" in name:
        return "H1"
    if "scalp" in name or "s5" in name or "fast" in name:
        return "M1"
    return "M5"


def _extract_candles(raw):
    candles = []
    for candle in raw or []:
        try:
            o = float(candle.get("open", candle.get("o")))
            h = float(candle.get("high", candle.get("h")))
            l = float(candle.get("low", candle.get("l")))
            c = float(candle.get("close", candle.get("c")))
        except Exception:
            continue
        if h <= 0 or l <= 0:
            continue
        candles.append((o, h, l, c))
    return candles


def _detect_candlestick_pattern(candles):
    if len(candles) < 2:
        return None
    o0, h0, l0, c0 = candles[-2]
    o1, h1, l1, c1 = candles[-1]
    body0 = abs(c0 - o0)
    body1 = abs(c1 - o1)
    range1 = max(h1 - l1, _CANDLE_PIP * 0.1)
    upper_wick = h1 - max(o1, c1)
    lower_wick = min(o1, c1) - l1

    if body1 <= range1 * 0.1:
        return {
            "type": "doji",
            "confidence": round(min(1.0, (range1 - body1) / range1), 3),
            "bias": None,
        }

    if (
        c1 > o1
        and c0 < o0
        and c1 >= max(o0, c0)
        and o1 <= min(o0, c0)
        and body1 > body0
    ):
        return {
            "type": "bullish_engulfing",
            "confidence": round(min(1.0, body1 / range1 + 0.3), 3),
            "bias": "up",
        }
    if (
        c1 < o1
        and c0 > o0
        and o1 >= min(o0, c0)
        and c1 <= max(o0, c0)
        and body1 > body0
    ):
        return {
            "type": "bearish_engulfing",
            "confidence": round(min(1.0, body1 / range1 + 0.3), 3),
            "bias": "down",
        }
    if lower_wick > body1 * 2.5 and upper_wick <= body1 * 0.6:
        return {
            "type": "hammer" if c1 >= o1 else "inverted_hammer",
            "confidence": round(min(1.0, lower_wick / range1 + 0.25), 3),
            "bias": "up",
        }
    if upper_wick > body1 * 2.5 and lower_wick <= body1 * 0.6:
        return {
            "type": "shooting_star" if c1 <= o1 else "hanging_man",
            "confidence": round(min(1.0, upper_wick / range1 + 0.25), 3),
            "bias": "down",
        }
    return None


def _score_candle(*, candles, side, min_conf):
    pattern = _detect_candlestick_pattern(_extract_candles(candles))
    if not pattern:
        return None, {}
    bias = pattern.get("bias")
    conf = float(pattern.get("confidence") or 0.0)
    if conf < min_conf:
        return None, {"type": pattern.get("type"), "confidence": round(conf, 3)}
    if bias is None:
        return 0.0, {"type": pattern.get("type"), "confidence": round(conf, 3), "bias": None}
    match = (side == "long" and bias == "up") or (side == "short" and bias == "down")
    score = conf if match else -conf * 0.7
    score = max(-1.0, min(1.0, score))
    return score, {"type": pattern.get("type"), "confidence": round(conf, 3), "bias": bias}


def _entry_candle_guard(side):
    tf = _candle_tf_for_worker()
    candles = get_candles_snapshot(tf, limit=4)
    if not candles:
        return True, 1.0
    score, _detail = _score_candle(candles=candles, side=side, min_conf=_CANDLE_MIN_CONF)
    if score is None:
        return True, 1.0
    if score <= _CANDLE_ENTRY_BLOCK:
        return False, 0.0
    mult = 1.0 + score * _CANDLE_ENTRY_SCALE
    mult = max(_CANDLE_ENTRY_MIN, min(_CANDLE_ENTRY_MAX, mult))
    return True, mult
