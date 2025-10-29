
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

PIP = 0.01
POINT = 0.001


def _to_float(value: object, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _band_stats(ma: Optional[float], bbw: Optional[float]) -> Optional[Tuple[float, float, float, float]]:
    if ma is None or bbw is None:
        return None
    try:
        center = float(ma)
        width_ratio = float(bbw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(center) or not math.isfinite(width_ratio):
        return None
    half_width = center * width_ratio / 2.0
    lower = center - half_width
    upper = center + half_width
    std_approx = abs(center) * width_ratio / 4.0
    if not all(math.isfinite(val) for val in (lower, upper, std_approx)):
        return None
    return lower, center, upper, std_approx


def _recent_range(candles: Sequence[Dict[str, float]], window: int = 5) -> float:
    highs = []
    lows = []
    for candle in candles[-window:]:
        highs.append(_to_float(candle.get('high')))
        lows.append(_to_float(candle.get('low')))
    highs = [h for h in highs if h is not None]
    lows = [l for l in lows if l is not None]
    if not highs or not lows:
        return 0.0
    return (max(highs) - min(lows)) / PIP


def _candle_body_pips(candle: Dict[str, float]) -> Optional[float]:
    open_px = _to_float(candle.get('open'))
    close_px = _to_float(candle.get('close'))
    if open_px is None or close_px is None:
        return None
    return (close_px - open_px) / PIP


def _eta_bonus(eta_value: Optional[float], *, limit: float = 8.0, scale: float = 1.2) -> float:
    if eta_value is None:
        return 0.0
    try:
        val = float(eta_value)
    except (TypeError, ValueError):
        return 0.0
    if val < 0.0:
        return 0.0
    return max(0.0, min(12.0, (limit - min(limit, val)) * scale))


def _squeeze_bias(bbw_eta: Optional[float], bbw_slope: float) -> float:
    if bbw_eta is None:
        return 0.0
    try:
        eta = float(bbw_eta)
    except (TypeError, ValueError):
        return 0.0
    squeeze = max(0.0, 6.0 - min(6.0, eta)) * 0.6
    if bbw_slope > 0.0:
        squeeze *= 0.5
    return squeeze


def _sl_tp_profile(atr_pips: float, range_mode: bool) -> Tuple[float, float]:
    """
    Micro pocket は SL を広めに取り、TP は保険的にタイトへ倒す。
    Exit manager がフォローする前提で RR≒0.4〜0.6 を意図する。
    """
    atr_pips = max(0.0, float(atr_pips))
    if range_mode:
        sl_raw = atr_pips * 2.7
        tp_raw = atr_pips * 0.62
        tp_cap = 1.9
    else:
        sl_raw = atr_pips * 3.05
        tp_raw = atr_pips * 0.58
        tp_cap = 2.3
    sl = max(7.0, min(18.0, sl_raw))
    tp = max(0.9, min(tp_cap, sl * 0.6, tp_raw))
    if tp >= sl:
        tp = max(0.9, sl * 0.55)
    return round(sl, 2), round(tp, 2)


class BBRsi:
    name = "BB_RSI"
    pocket = "micro"

    @staticmethod
    def check(fac: Dict) -> Dict | None:
        rsi = _to_float(fac.get('rsi'))
        bbw = _to_float(fac.get('bbw'))
        ma = _to_float(fac.get('ma20'))
        close = _to_float(fac.get('close'), ma)
        ema = _to_float(fac.get('ema20'), ma)
        ma10 = _to_float(fac.get('ma10'))
        atr_pips = _to_float(fac.get('atr_pips'))
        atr = _to_float(fac.get('atr'))
        adx = _to_float(fac.get('adx'))
        vol_5m = _to_float(fac.get('vol_5m'))
        rsi_eta_up = _to_float(fac.get('rsi_eta_upper_min'))
        rsi_eta_dn = _to_float(fac.get('rsi_eta_lower_min'))
        bbw_eta = _to_float(fac.get('bbw_squeeze_eta_min'))
        bbw_slope = _to_float(fac.get('bbw_slope_per_bar'), 0.0) or 0.0
        candles = fac.get('candles') or []

        if None in (rsi, bbw, ma, close):
            return None

        if atr_pips is None:
            atr_pips = (atr or 0.0) * 100.0
        atr_pips = float(atr_pips or 0.0)
        if atr_pips <= 0.6:
            return None

        if vol_5m is not None and vol_5m < 0.35:
            return None

        stats = _band_stats(ma, bbw)
        if stats is None:
            return None
        lower, basis, upper, std_approx = stats
        band_width = upper - lower
        band_width_pips = band_width / PIP if band_width else 0.0
        if band_width_pips <= 0.2 or std_approx <= 0.0:
            return None

        dist_lower_raw = (close - lower) / PIP
        dist_upper_raw = (upper - close) / PIP
        dist_lower_pips = max(0.0, dist_lower_raw)
        dist_upper_pips = max(0.0, dist_upper_raw)
        bb_z = (close - basis) / std_approx if std_approx else 0.0

        ema_ref = ema if ema is not None else basis
        pullback_long = max(0.0, (ema_ref or basis) - close) / PIP if ema_ref is not None else 0.0
        pullback_short = max(0.0, close - (ema_ref or basis)) / PIP if ema_ref is not None else 0.0

        trend_bias = 0.0
        if ma10 is not None and basis is not None:
            trend_bias = (ma10 - basis) / PIP
        trend_strength = max(-7.5, min(7.5, trend_bias))

        range_active = bool(fac.get('range_mode') or fac.get('range_active'))
        recent_range = _recent_range(candles)
        if recent_range <= 0.0:
            recent_range = atr_pips

        oversold = 34.0 + (trend_strength * 0.85)
        overbought = 66.0 + (trend_strength * 0.85)
        oversold = max(27.0, min(48.0, oversold))
        overbought = max(52.0, min(73.0, overbought))
        if range_active:
            oversold = min(oversold + 3.0, 50.0)
            overbought = max(overbought - 3.0, 50.0)
        oversold_soft = min(oversold + 7.0, 54.0)
        overbought_soft = max(overbought - 7.0, 50.0)

        dynamic_margin = band_width_pips * (0.12 if not range_active else 0.18)
        base_margin = 1.2 if not range_active else 1.6
        band_margin = max(base_margin, dynamic_margin)
        long_band = dist_lower_pips <= band_margin or dist_lower_raw <= 0.0
        short_band = dist_upper_pips <= band_margin or dist_upper_raw <= 0.0
        long_rsi = rsi <= oversold or (rsi <= oversold_soft and long_band)
        short_rsi = rsi >= overbought or (rsi >= overbought_soft and short_band)

        long_momentum_ok = pullback_long >= (0.55 if range_active else 0.85)
        short_momentum_ok = pullback_short >= (0.55 if range_active else 0.85)

        trend_guard_long = trend_strength > -9.0 or (range_active and trend_strength > -11.0)
        trend_guard_short = trend_strength < 9.0 or (range_active and trend_strength < 11.0)

        adx_guard_long = True
        adx_guard_short = True
        if adx is not None:
            if adx >= 40.0 and pullback_long < 1.1:
                adx_guard_long = False
            if adx >= 40.0 and pullback_short < 1.1:
                adx_guard_short = False

        squeeze_adj = _squeeze_bias(bbw_eta, bbw_slope)
        last_body = _candle_body_pips(candles[-1]) if len(candles) >= 1 else None
        prev_body = _candle_body_pips(candles[-2]) if len(candles) >= 2 else None

        def _alignment_ok(side: str) -> bool:
            if last_body is None or prev_body is None:
                return True
            if side == "long":
                return prev_body <= -0.6 and last_body >= 0.4
            return prev_body >= 0.6 and last_body <= -0.4

        long_ready = (
            long_momentum_ok
            and trend_guard_long
            and adx_guard_long
            and (long_rsi or long_band or bb_z <= -0.95)
        )
        short_ready = (
            short_momentum_ok
            and trend_guard_short
            and adx_guard_short
            and (short_rsi or short_band or bb_z >= 0.95)
        )

        if long_ready and not _alignment_ok("long"):
            long_ready = False
        if short_ready and not _alignment_ok("short"):
            short_ready = False

        if not long_ready and not short_ready:
            return None

        def _confidence_base(range_flag: bool) -> float:
            return 51.0 if range_flag else 47.0

        def _confidence_common(base: float, pullback: float) -> float:
            conf = base
            conf += max(0.0, (abs(bb_z) - 0.9) * 6.0)
            span = min(max(band_width_pips, 0.0), 20.0)
            conf += max(0.0, (span - 3.0) * 0.4)
            conf += max(0.0, (atr_pips - 3.5) * 0.35)
            effective_pull = min(max(pullback, 0.0), 8.0)
            conf += max(0.0, (effective_pull - 0.8) * 1.8)
            conf += max(0.0, (min(recent_range, 15.0) - atr_pips) * 0.35)
            conf -= max(0.0, (abs(trend_strength) - 4.5) * 0.9)
            if adx is not None and adx > 35.0:
                conf -= (adx - 35.0) * 0.45
            if bbw is not None and bbw > 0.33:
                conf -= (bbw - 0.33) * 22.0
            conf += squeeze_adj
            return conf

        if long_ready:
            base = _confidence_base(range_active)
            conf = _confidence_common(base, pullback_long)
            conf += _eta_bonus(rsi_eta_up)
            if vol_5m is not None:
                conf += max(0.0, min(5.0, (vol_5m - 0.6) * 4.0))
            conf = max(38.0, min(92.0, conf))

            sl, tp = _sl_tp_profile(atr_pips, range_active)

            return {
                'action': 'OPEN_LONG',
                'sl_pips': sl,
                'tp_pips': tp,
                'confidence': int(round(conf)),
                'tag': f"{BBRsi.name}-long",
                'notes': {
                    'rsi': round(rsi, 2),
                    'bb_z': round(bb_z, 2),
                    'pull': round(pullback_long, 2),
                    'trend': round(trend_strength, 2),
                },
            }

        base = _confidence_base(range_active)
        conf = _confidence_common(base, pullback_short)
        conf += _eta_bonus(rsi_eta_dn)
        if vol_5m is not None:
            conf += max(0.0, min(5.0, (vol_5m - 0.6) * 4.0))
        conf = max(38.0, min(92.0, conf))

        sl, tp = _sl_tp_profile(atr_pips, range_active)

        return {
            'action': 'OPEN_SHORT',
            'sl_pips': sl,
            'tp_pips': tp,
            'confidence': int(round(conf)),
            'tag': f"{BBRsi.name}-short",
            'notes': {
                'rsi': round(rsi, 2),
                'bb_z': round(bb_z, 2),
                'pull': round(pullback_short, 2),
                'trend': round(trend_strength, 2),
            },
        }
