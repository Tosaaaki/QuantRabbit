"""Scalp False Break Fade dedicated ENTRY worker."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from analysis.range_guard import detect_range_mode
from analysis.range_model import compute_range_snapshot
from execution.position_manager import PositionManager
from execution.risk_guard import can_trade, clamp_sl_tp
from execution.strategy_entry import market_order
from indicators.factor_cache import all_factors, get_candles_snapshot
from market_data import spread_monitor, tick_window
from utils.market_hours import is_market_open
from workers.common.air_state import adjust_signal, evaluate_air
from workers.common.dyn_cap import compute_cap
from workers.common.dyn_size import compute_units

LOG = logging.getLogger(__name__)

ENV_PREFIX = "SCALP_FALSE_BREAK_FADE"
STRATEGY_TAG = "FalseBreakFade"
PIP = 0.01


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(f"{ENV_PREFIX}_{key}")
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"", "0", "false", "no", "off"}


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(f"{ENV_PREFIX}_{key}")
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(f"{ENV_PREFIX}_{key}")
    if raw is None:
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def _env_str(key: str, default: str) -> str:
    raw = os.getenv(f"{ENV_PREFIX}_{key}")
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip()


def _env_csv(key: str, default: str = "") -> set[str]:
    raw = os.getenv(f"{ENV_PREFIX}_{key}", default)
    return {s.strip() for s in raw.split(",") if s.strip()}


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_probability(value: object) -> float:
    raw = _to_float(value, 0.0)
    if raw > 1.0:
        raw = raw / 100.0
    return max(0.0, min(1.0, raw))


def _confidence_scale(conf: int, *, lo: int, hi: int) -> float:
    if conf <= lo:
        return 0.5
    if conf >= hi:
        return 1.0
    span = (conf - lo) / max(1.0, hi - lo)
    return 0.5 + span * 0.5


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    pct = max(0.0, min(pct, 100.0))
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    rank = pct / 100.0 * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _latest_mid(fallback: float) -> float:
    ticks = tick_window.recent_ticks(seconds=8.0, limit=1)
    if ticks:
        tick = ticks[-1]
        mid_value = tick.get("mid")
        if mid_value is not None:
            try:
                return float(mid_value)
            except (TypeError, ValueError):
                return fallback
        bid = tick.get("bid")
        ask = tick.get("ask")
        try:
            if bid is not None and ask is not None:
                return (float(bid) + float(ask)) / 2.0
        except (TypeError, ValueError):
            return fallback
    return fallback


def _latest_price(fac_m1: Dict[str, object]) -> float:
    try:
        price = float(fac_m1.get("close") or 0.0)
    except Exception:
        price = 0.0
    return _latest_mid(price)


def _atr_pips(fac_m1: Dict[str, object]) -> float:
    return _to_float(fac_m1.get("atr_pips"), 0.0)


def _rsi(fac_m1: Dict[str, object]) -> float:
    return _to_float(fac_m1.get("rsi"), 50.0)


def _adx(fac_m1: Dict[str, object]) -> float:
    return _to_float(fac_m1.get("adx"), 0.0)


def _bbw(fac_m1: Dict[str, object]) -> float:
    return _to_float(fac_m1.get("bbw"), 0.0)


def _ema_slope_pips(fac_m1: Dict[str, object], key: str) -> float:
    return _to_float(fac_m1.get(key), 0.0) / PIP


def _vwap_gap_pips(fac_m1: Dict[str, object]) -> float:
    return _to_float(fac_m1.get("vwap_gap"), 0.0)


def spread_ok(
    *,
    max_pips: Optional[float] = None,
    p25_max: Optional[float] = None,
) -> tuple[bool, Optional[dict[str, float]]]:
    state = spread_monitor.get_state()
    if state is not None:
        try:
            spread_pips = float(state.get("spread_pips") or 0.0)
        except (TypeError, ValueError):
            spread_pips = None
        if spread_pips is not None:
            if max_pips is not None and spread_pips > max_pips:
                return False, {"spread_pips": spread_pips}
            if p25_max is not None and spread_pips > p25_max:
                return False, {"spread_pips": spread_pips, "p25_max": p25_max}
            return True, {"spread_pips": spread_pips}

    ticks = tick_window.recent_ticks(seconds=8.0, limit=120)
    if not ticks:
        return False, None
    spreads: list[float] = []
    for tick in ticks:
        bid = tick.get("bid")
        ask = tick.get("ask")
        try:
            if bid is None or ask is None:
                continue
            spreads.append(max(0.0, float(ask) - float(bid)) / PIP)
        except (TypeError, ValueError):
            continue
    if not spreads:
        return False, None
    spread_pips = sum(spreads) / len(spreads)
    p25 = _percentile(spreads, 25.0)
    if max_pips is not None and spread_pips > max_pips:
        return False, {"spread_pips": spread_pips, "spread_p25": p25}
    if p25_max is not None and p25 > p25_max:
        return False, {"spread_pips": spread_pips, "spread_p25": p25}
    return True, {"spread_pips": spread_pips, "spread_p25": p25}


def tick_snapshot(seconds: float, *, limit: int = 80) -> tuple[list[float], Optional[dict]]:
    raw_ticks = tick_window.recent_ticks(seconds=seconds, limit=limit)
    mids: list[float] = []
    for item in raw_ticks:
        try:
            if item.get("mid") is None:
                bid = item.get("bid")
                ask = item.get("ask")
                if bid is not None and ask is not None:
                    mids.append((float(bid) + float(ask)) / 2.0)
                continue
            mids.append(float(item.get("mid")))
        except (TypeError, ValueError):
            continue
    return mids, None


def tick_reversal(mids: Sequence[float], *, min_ticks: int = 6) -> tuple[bool, Optional[str], float]:
    if len(mids) < min_ticks:
        return False, None, 0.0
    deltas = [mids[i] - mids[i - 1] for i in range(1, len(mids))]
    if len(deltas) < 5:
        return False, None, 0.0
    prev = deltas[-5:-2]
    recent = deltas[-2:]
    prev_sum = sum(prev)
    recent_sum = sum(recent)
    if prev_sum == 0 or recent_sum == 0:
        return False, None, 0.0
    if prev_sum < 0 < recent_sum:
        return True, "long", abs(recent_sum) / max(PIP, abs(prev_sum))
    if prev_sum > 0 > recent_sum:
        return True, "short", abs(recent_sum) / max(PIP, abs(prev_sum))
    return False, None, 0.0


def _client_order_id(tag: str) -> str:
    ts_ms = int(time.time() * 1000)
    sanitized = "".join(ch.lower() for ch in str(tag) if ch.isalnum())[:8] or "fbf"
    digest = hashlib.sha1(f"{ts_ms}-{tag}".encode("utf-8")).hexdigest()[:9]
    return f"qr-{ts_ms}-scalp-fbf-{sanitized}{digest}"


@dataclass
class FalseBreakState:
    direction: str
    side: str
    level: float
    started_ts: float
    expires_ts: float
    extreme: float


_FALSE_BREAK_STATE: Dict[str, FalseBreakState] = {}


ENABLED = _env_bool("ENABLED", True)
POCKET = _env_str("POCKET", "scalp_fast")
LOOP_INTERVAL_SEC = _env_float("LOOP_INTERVAL_SEC", 2.0)
LOG_PREFIX = _env_str("LOG_PREFIX", "[Scalp:FBF]")
CONFIDENCE_FLOOR = _env_int("CONFIDENCE_FLOOR", 30)
CONFIDENCE_CEIL = _env_int("CONFIDENCE_CEIL", 95)
MIN_ENTRY_CONF = _env_int("MIN_ENTRY_CONF", 0)
MAX_SPREAD_PIPS = _env_float("MAX_SPREAD_PIPS", 1.2)
COOLDOWN_SEC = _env_float("COOLDOWN_SEC", 180.0)
MAX_OPEN_TRADES = _env_int("MAX_OPEN_TRADES", 1)
MAX_OPEN_TRADES_GLOBAL = _env_int("MAX_OPEN_TRADES_GLOBAL", 0)
OPEN_TRADES_SCOPE = _env_str("OPEN_TRADES_SCOPE", "tag").strip().lower()
MIN_UNITS = _env_int("MIN_UNITS", 1000)
BASE_ENTRY_UNITS = _env_int("BASE_UNITS", 9000)
MAX_MARGIN_USAGE = _env_float("MAX_MARGIN_USAGE", 0.92)
CAP_MIN = _env_float("CAP_MIN", 0.12)
CAP_MAX = _env_float("CAP_MAX", 0.95)

FBF_BBW_MAX = _env_float("FBF_BBW_MAX", 0.0026)
FBF_ATR_MIN = _env_float("FBF_ATR_MIN", 0.7)
FBF_ATR_MAX = _env_float("FBF_ATR_MAX", 5.5)
FBF_ADX_MAX = _env_float("FBF_ADX_MAX", 32.0)
FBF_LOOKBACK = _env_int("FBF_LOOKBACK", 24)
FBF_HI_PCT = _env_float("FBF_HI_PCT", 95.0)
FBF_LO_PCT = _env_float("FBF_LO_PCT", 5.0)
FBF_BREAKOUT_PIPS = _env_float("FBF_BREAKOUT_PIPS", 0.8)
FBF_RECLAIM_PIPS = _env_float("FBF_RECLAIM_PIPS", 0.4)
FBF_TIMEOUT_SEC = _env_float("FBF_TIMEOUT_SEC", 55.0)
FBF_MIN_SWEEP_PIPS = _env_float("FBF_MIN_SWEEP_PIPS", 0.9)
FBF_RANGE_SCORE_MIN = _env_float("FBF_RANGE_SCORE_MIN", 0.30)
FBF_TICK_WINDOW_SEC = _env_float("FBF_TICK_WINDOW_SEC", 6.0)
FBF_REQUIRE_TICK_REVERSAL = _env_bool("FBF_REQUIRE_TICK_REVERSAL", True)
FBF_SPREAD_P25 = _env_float("FBF_SPREAD_P25", 1.1)
FBF_SIZE_MULT = _env_float("FBF_SIZE_MULT", 1.10)
FBF_MAX_COUNTER_SLOPE_PIPS = _env_float("FBF_MAX_COUNTER_SLOPE_PIPS", 0.0)
FBF_MAX_COUNTER_VWAP_GAP_PIPS = _env_float("FBF_MAX_COUNTER_VWAP_GAP_PIPS", 0.0)
FBF_TICK_MIN_STRENGTH = _env_float("FBF_TICK_MIN_STRENGTH", 0.0)


def _signal_false_break_fade(
    fac_m1: Dict[str, object],
    range_ctx,
    *,
    tag: str,
) -> Optional[Dict[str, object]]:
    price = _latest_price(fac_m1)
    if price <= 0:
        return None

    ok_spread, _ = spread_ok(max_pips=MAX_SPREAD_PIPS, p25_max=FBF_SPREAD_P25)
    if not ok_spread:
        return None

    bbw = _bbw(fac_m1)
    atr = _atr_pips(fac_m1)
    adx = _adx(fac_m1)
    if bbw <= 0.0 or atr <= 0.0:
        return None
    if bbw > FBF_BBW_MAX:
        return None
    if atr < FBF_ATR_MIN or atr > FBF_ATR_MAX:
        return None
    if adx > FBF_ADX_MAX:
        return None

    range_score = 0.0
    range_active = False
    if range_ctx is not None:
        try:
            range_score = float(getattr(range_ctx, "score", 0.0) or 0.0)
            range_active = bool(getattr(range_ctx, "active", False))
        except Exception:
            range_score = 0.0
            range_active = False
    if not range_active and FBF_RANGE_SCORE_MIN > 0.0 and range_score < FBF_RANGE_SCORE_MIN:
        return None

    candles = get_candles_snapshot("M1", limit=max(40, FBF_LOOKBACK + 6))
    snap = compute_range_snapshot(candles or [], lookback=max(10, FBF_LOOKBACK), hi_pct=FBF_HI_PCT, lo_pct=FBF_LO_PCT)
    if not snap:
        return None

    now = time.monotonic()
    state = _FALSE_BREAK_STATE.get(tag)
    if state and now > state.expires_ts:
        _FALSE_BREAK_STATE.pop(tag, None)
        state = None

    if state and now <= state.expires_ts:
        if state.side == "high":
            state.extreme = max(state.extreme, price)
            sweep_pips = (state.extreme - state.level) / PIP
            reclaimed = price <= state.level - FBF_RECLAIM_PIPS * PIP
            want_dir = "short"
        else:
            state.extreme = min(state.extreme, price)
            sweep_pips = (state.level - state.extreme) / PIP
            reclaimed = price >= state.level + FBF_RECLAIM_PIPS * PIP
            want_dir = "long"

        if not reclaimed or sweep_pips < FBF_MIN_SWEEP_PIPS:
            return None

        slope_10_pips = _ema_slope_pips(fac_m1, "ema_slope_10")
        vwap_gap_pips = _vwap_gap_pips(fac_m1)
        if want_dir == "long":
            if FBF_MAX_COUNTER_SLOPE_PIPS > 0.0 and slope_10_pips < -FBF_MAX_COUNTER_SLOPE_PIPS:
                return None
            if FBF_MAX_COUNTER_VWAP_GAP_PIPS > 0.0 and vwap_gap_pips < -FBF_MAX_COUNTER_VWAP_GAP_PIPS:
                return None
        else:
            if FBF_MAX_COUNTER_SLOPE_PIPS > 0.0 and slope_10_pips > FBF_MAX_COUNTER_SLOPE_PIPS:
                return None
            if FBF_MAX_COUNTER_VWAP_GAP_PIPS > 0.0 and vwap_gap_pips > FBF_MAX_COUNTER_VWAP_GAP_PIPS:
                return None

        mids, _ = tick_snapshot(FBF_TICK_WINDOW_SEC, limit=140)
        rev_ok, rev_dir, rev_strength = (
            tick_reversal(mids, min_ticks=6) if mids else (False, None, 0.0)
        )
        if FBF_REQUIRE_TICK_REVERSAL and (not rev_ok or rev_dir != want_dir):
            return None
        if FBF_TICK_MIN_STRENGTH > 0.0:
            try:
                rev_strength_f = float(rev_strength or 0.0)
            except Exception:
                rev_strength_f = 0.0
            if rev_strength_f < FBF_TICK_MIN_STRENGTH:
                return None

        sl = max(1.4, min(2.8, atr * 0.9))
        tp = max(sl * 1.35, min(4.0, atr * 1.25))
        conf = 60 + int(min(12.0, sweep_pips * 3.0)) + int(min(10.0, rev_strength * 4.0))
        conf = int(max(45, min(92, conf)))

        size_mult = max(0.6, min(1.4, FBF_SIZE_MULT * (1.0 + min(0.25, sweep_pips * 0.05))))
        _FALSE_BREAK_STATE.pop(tag, None)
        return {
            "action": "OPEN_LONG" if want_dir == "long" else "OPEN_SHORT",
            "sl_pips": round(sl, 2),
            "tp_pips": round(tp, 2),
            "confidence": conf,
            "tag": tag,
            "reason": "false_break_fade",
            "size_mult": round(size_mult, 3),
            "fade": {
                "side": state.side,
                "sweep_pips": round(sweep_pips, 2),
                "range_score": round(range_score, 3),
            },
        }

    breakout_high = price >= snap.high + FBF_BREAKOUT_PIPS * PIP
    breakout_low = price <= snap.low - FBF_BREAKOUT_PIPS * PIP
    if not breakout_high and not breakout_low:
        return None

    side = "high" if breakout_high else "low"
    level = snap.high if breakout_high else snap.low
    extreme = price
    direction = "short" if breakout_high else "long"
    _FALSE_BREAK_STATE[tag] = FalseBreakState(
        direction=direction,
        side=side,
        level=level,
        started_ts=now,
        expires_ts=now + FBF_TIMEOUT_SEC,
        extreme=extreme,
    )
    return None


def _build_entry_thesis(signal: Dict[str, object], fac_m1: Dict[str, object], range_ctx) -> Dict[str, object]:
    signal_confidence = int(signal.get("confidence", 0) or 0)
    return {
        "strategy_tag": signal.get("tag") or STRATEGY_TAG,
        "env_prefix": ENV_PREFIX,
        "confidence": signal_confidence,
        "entry_probability": _to_probability(signal_confidence),
        "reason": signal.get("reason"),
        "sl_pips": signal.get("sl_pips"),
        "tp_pips": signal.get("tp_pips"),
        "range_active": bool(getattr(range_ctx, "active", False)),
        "range_score": float(getattr(range_ctx, "score", 0.0) or 0.0),
        "range_reason": getattr(range_ctx, "reason", None),
        "range_mode": getattr(range_ctx, "mode", None),
        "rsi": _rsi(fac_m1),
        "adx": _adx(fac_m1),
        "atr_pips": _atr_pips(fac_m1),
        "bbw": _bbw(fac_m1),
        "vwap_gap": _vwap_gap_pips(fac_m1),
        "ema_slope_10_pips": _ema_slope_pips(fac_m1, "ema_slope_10"),
        "air_score": signal.get("air_score"),
        "air_pressure": signal.get("air_pressure"),
        "air_pressure_dir": signal.get("air_pressure_dir"),
    }


async def _place_order(
    signal: Dict[str, object],
    *,
    fac_m1: Dict[str, object],
    fac_h4: Dict[str, object],
    range_ctx,
) -> Optional[str]:
    price = _latest_price(fac_m1)
    if price <= 0:
        return None

    side = "long" if signal.get("action") == "OPEN_LONG" else "short"
    sl_pips = float(signal.get("sl_pips") or 0.0)
    tp_pips = float(signal.get("tp_pips") or 0.0)
    if sl_pips <= 0:
        return None

    conf = int(signal.get("confidence", 0) or 0)
    if MIN_ENTRY_CONF > 0 and conf < MIN_ENTRY_CONF:
        return None
    conf_scale = _confidence_scale(conf, lo=CONFIDENCE_FLOOR, hi=CONFIDENCE_CEIL)
    size_mult = float(signal.get("size_mult", 1.0) or 1.0)
    size_mult = max(0.6, min(1.4, size_mult))

    atr = _atr_pips(fac_m1)
    cap_res = compute_cap(
        atr_pips=atr,
        free_ratio=0.0,
        range_active=bool(getattr(range_ctx, "active", False)),
        perf_pf=None,
        pos_bias=0.0,
        cap_min=CAP_MIN,
        cap_max=CAP_MAX,
        env_prefix=ENV_PREFIX,
    )
    try:
        from utils.oanda_account import get_account_snapshot

        free_ratio = float(get_account_snapshot().free_margin_ratio or 0.0)
    except Exception:
        free_ratio = 0.0
        cap_res = type("CapResult", (), {"cap": 0.0, "reasons": {}})()
    cap = cap_res.cap
    if free_ratio > 0.0 and cap <= 0.0:
        # no extra fallback in this path
        pass
    if cap <= 0.0:
        cap = CAP_MIN

    spread_pips = None
    try:
        state = spread_monitor.get_state()
        if state is not None:
            spread_pips = float(state.get("spread_pips") or 0.0)
    except Exception:
        spread_pips = None

    sizing = compute_units(
        entry_price=price,
        sl_pips=sl_pips,
        base_entry_units=BASE_ENTRY_UNITS,
        min_units=MIN_UNITS,
        max_margin_usage=MAX_MARGIN_USAGE,
        spread_pips=float(spread_pips or 0.0),
        spread_soft_cap=MAX_SPREAD_PIPS,
        adx=_adx(fac_m1),
        signal_score=float(conf) / 100.0,
        pocket=POCKET,
        strategy_tag=str(signal.get("tag") or STRATEGY_TAG),
        env_prefix=ENV_PREFIX,
    )

    units = int(round(sizing.units * cap * size_mult))
    if abs(units) < MIN_UNITS:
        return None
    if side == "short":
        units = -abs(units)
    else:
        units = abs(units)

    if side == "long":
        sl_price = round(price - sl_pips * PIP, 3)
        tp_price = round(price + tp_pips * PIP, 3) if tp_pips > 0 else None
    else:
        sl_price = round(price + sl_pips * PIP, 3)
        tp_price = round(price - tp_pips * PIP, 3) if tp_pips > 0 else None

    sl_price, tp_price = clamp_sl_tp(price=price, sl=sl_price, tp=tp_price, is_buy=(side == "long"))
    client_id = _client_order_id(str(signal.get("tag") or STRATEGY_TAG))
    entry_thesis = _build_entry_thesis(signal, fac_m1, range_ctx)
    entry_thesis["entry_probability"] = _to_probability(conf)
    entry_thesis["entry_units_intent"] = abs(units)
    meta = {
        "cap": round(cap, 3),
        "conf_scale": round(conf_scale, 3),
        "sizing": sizing.factors if hasattr(sizing, "factors") else {},
    }

    return await market_order(
        instrument="USD_JPY",
        units=units,
        sl_price=sl_price,
        tp_price=tp_price,
        pocket=POCKET,
        client_order_id=client_id,
        strategy_tag=str(signal.get("tag") or STRATEGY_TAG),
        entry_thesis=entry_thesis,
        meta=meta,
        confidence=conf,
    )


async def _run_worker() -> None:
    if not ENABLED:
        LOG.info("[disabled] %s", LOG_PREFIX)
        while True:
            await asyncio.sleep(3600.0)

    LOG.info("%s worker start mode=inactive mode=%s", LOG_PREFIX, STRATEGY_TAG)
    pos_manager = PositionManager()
    last_entry_ts = 0.0

    while True:
        await asyncio.sleep(LOOP_INTERVAL_SEC)
        if not is_market_open(datetime.datetime.utcnow()):
            continue
        if not can_trade(POCKET):
            continue

        now_mono = time.monotonic()
        if COOLDOWN_SEC > 0.0 and now_mono - last_entry_ts < COOLDOWN_SEC:
            continue

        if MAX_OPEN_TRADES > 0 or MAX_OPEN_TRADES_GLOBAL > 0:
            try:
                positions = pos_manager.get_open_positions()
                pocket_info = positions.get(POCKET, {})
                open_trades_all = pocket_info.get("open_trades", []) or []
                if MAX_OPEN_TRADES_GLOBAL > 0 and len(open_trades_all) >= MAX_OPEN_TRADES_GLOBAL:
                    continue
                if OPEN_TRADES_SCOPE == "tag":
                    open_trades = [
                        tr
                        for tr in open_trades_all
                        if str(
                            tr.get("strategy_tag")
                            or tr.get("entry_thesis", {}).get("strategy_tag")
                            or tr.get("tag")
                            or tr.get("strategy")
                            or ""
                        ).lower()
                        == STRATEGY_TAG.lower()
                    ]
                else:
                    open_trades = open_trades_all
                if MAX_OPEN_TRADES > 0 and len(open_trades) >= MAX_OPEN_TRADES:
                    continue
            except Exception:
                pass

        factors = all_factors()
        fac_m1 = factors.get("M1", {}) or {}
        fac_h4 = factors.get("H4", {}) or {}
        if not fac_m1:
            continue

        range_ctx = detect_range_mode(fac_m1, fac_h4)
        air = evaluate_air(fac_m1, fac_h4, range_ctx=range_ctx, tag="false_break_fade")
        if getattr(air, "enabled", False) and not bool(getattr(air, "allow_entry", True)):
            continue

        signal = _signal_false_break_fade(fac_m1, range_ctx, tag=STRATEGY_TAG)
        if not signal:
            continue
        signal = adjust_signal(signal, air)
        if not signal:
            continue
        order_id = await _place_order(
            signal,
            fac_m1=fac_m1,
            fac_h4=fac_h4,
            range_ctx=range_ctx,
        )
        if order_id:
            last_entry_ts = now_mono
            LOG.info(
                "%s order placed id=%s side=%s conf=%s",
                LOG_PREFIX,
                order_id,
                signal.get("action"),
                signal.get("confidence"),
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    try:
        asyncio.run(_run_worker())
    except KeyboardInterrupt:
        raise
