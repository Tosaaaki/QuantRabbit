from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

from analysis.ma_projection import (
    compute_adx_projection,
    compute_bbw_projection,
    compute_ma_projection,
    compute_rsi_projection,
)
from indicators.factor_cache import get_candles_snapshot
from market_data import tick_window, spread_monitor

PIP = 0.01
_PROJ_TF_MINUTES = {"M1": 1.0, "M5": 5.0, "H1": 60.0, "H4": 240.0, "D1": 1440.0}


def _as_float(value: object, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    pct = max(0.0, min(pct, 100.0))
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    if pct == 100.0:
        return sorted_vals[-1]
    rank = pct / 100.0 * (len(sorted_vals) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_vals) - 1)
    frac = rank - lower
    return sorted_vals[lower] * (1.0 - frac) + sorted_vals[upper] * frac


def latest_mid(fallback: float) -> float:
    ticks = tick_window.recent_ticks(seconds=8.0, limit=1)
    if ticks:
        tick = ticks[-1]
        mid_val = tick.get("mid")
        if mid_val is not None:
            try:
                return float(mid_val)
            except Exception:
                pass
        bid = tick.get("bid")
        ask = tick.get("ask")
        if bid is not None and ask is not None:
            try:
                return (float(bid) + float(ask)) / 2.0
            except Exception:
                return fallback
    return fallback


def bb_levels(fac: Dict[str, object]) -> Optional[tuple[float, float, float, float, float]]:
    if not fac:
        return None
    upper = _as_float(fac.get("bb_upper"))
    lower = _as_float(fac.get("bb_lower"))
    mid = _as_float(fac.get("bb_mid")) or _as_float(fac.get("ma20"))
    bbw = _as_float(fac.get("bbw")) or 0.0
    if upper is None or lower is None:
        if mid is None or bbw <= 0:
            return None
        half = abs(mid) * bbw / 2.0
        upper = mid + half
        lower = mid - half
    span = upper - lower
    if span <= 0:
        return None
    mid_val = mid if mid is not None else (upper + lower) / 2.0
    return upper, mid_val, lower, span, span / PIP


def bb_entry_allowed(style: str, side: str, price: float, fac: Dict[str, object], *, range_active: Optional[bool] = None) -> bool:
    levels = bb_levels(fac)
    if price <= 0 or not levels:
        return True
    upper, mid, lower, _, span_pips = levels
    side_key = str(side or "").lower()
    direction = "long" if side_key in {"buy", "long", "open_long"} else "short"
    orig_style = style
    if style == "scalp" and range_active:
        style = "reversion"
    if style == "reversion":
        base_pips = 2.0 if orig_style == "scalp" else 2.4
        base_ratio = 0.2 if orig_style == "scalp" else 0.22
        threshold = max(base_pips, span_pips * base_ratio)
        if direction == "long":
            dist = (price - lower) / PIP
        else:
            dist = (upper - price) / PIP
        return dist <= threshold
    if direction == "long":
        if price < mid:
            return False
        ext = max(0.0, price - upper) / PIP
    else:
        if price > mid:
            return False
        ext = max(0.0, lower - price) / PIP
    max_ext = max(2.4, span_pips * 0.30) if orig_style == "scalp" else max(3.5, span_pips * 0.40)
    return ext <= max_ext


def parse_hours(raw: str) -> set[int]:
    hours: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            try:
                start = int(float(start_s))
                end = int(float(end_s))
            except ValueError:
                continue
            if start > end:
                start, end = end, start
            for h in range(start, end + 1):
                if 0 <= h <= 23:
                    hours.add(h)
            continue
        try:
            h = int(float(token))
        except ValueError:
            continue
        if 0 <= h <= 23:
            hours.add(h)
    return hours


def session_allowed(hour_utc: int, *, allow_hours: set[int], block_hours: set[int], offset: int = 9) -> bool:
    hour_local = (hour_utc + offset) % 24
    if block_hours and hour_local in block_hours:
        return False
    if allow_hours:
        return hour_local in allow_hours
    return True


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
        strength = abs(recent_sum) / max(PIP, abs(prev_sum))
        return True, "long", strength
    if prev_sum > 0 > recent_sum:
        strength = abs(recent_sum) / max(PIP, abs(prev_sum))
        return True, "short", strength
    return False, None, 0.0


@dataclass
class TickImbalance:
    up: int
    down: int
    ratio: float
    momentum_pips: float
    range_pips: float
    span_seconds: float


def tick_imbalance(mids: Sequence[float], span_seconds: float) -> Optional[TickImbalance]:
    if len(mids) < 4:
        return None
    ups = 0
    downs = 0
    for i in range(1, len(mids)):
        if mids[i] > mids[i - 1]:
            ups += 1
        elif mids[i] < mids[i - 1]:
            downs += 1
    total = max(1, ups + downs)
    ratio = ups / total if ups >= downs else downs / total
    momentum_pips = (mids[-1] - mids[0]) / PIP
    range_pips = (max(mids) - min(mids)) / PIP
    return TickImbalance(
        up=ups,
        down=downs,
        ratio=ratio,
        momentum_pips=momentum_pips,
        range_pips=range_pips,
        span_seconds=span_seconds,
    )


def spread_ok(*, max_pips: Optional[float] = None, p25_max: Optional[float] = None) -> tuple[bool, Optional[dict]]:
    state = spread_monitor.get_state()
    if state is None:
        # spread_monitor is process-local; fall back to the cross-process tick cache so
        # standalone scalp workers can still gate on spread.
        ticks = tick_window.recent_ticks(seconds=8.0, limit=120)
        if not ticks:
            return False, None
        spreads: list[float] = []
        for tick in ticks:
            try:
                bid = float(tick.get("bid") or 0.0)
                ask = float(tick.get("ask") or 0.0)
            except Exception:
                continue
            if bid <= 0.0 or ask <= 0.0 or ask < bid:
                continue
            spreads.append((ask - bid) / PIP)
        if not spreads:
            return False, None
        try:
            last_epoch = float(ticks[-1].get("epoch") or 0.0)
        except Exception:
            last_epoch = 0.0
        age_ms = int(max(0.0, (time.time() - (last_epoch or time.time())) * 1000.0))
        state = {
            "spread_pips": spreads[-1],
            "p25_pips": _percentile(spreads, 25.0),
            "median_pips": _percentile(spreads, 50.0),
            "p95_pips": _percentile(spreads, 95.0),
            "samples": len(spreads),
            "age_ms": age_ms,
            "stale": age_ms > 5000,
            "source": "tick_cache",
        }
    if state.get("stale"):
        return False, state
    spread = _as_float(state.get("spread_pips"), 999.0) or 999.0
    if max_pips is not None and spread > max_pips:
        return False, state
    if p25_max is not None:
        p25 = _as_float(state.get("p25_pips"))
        if p25 is None:
            return False, state
        if spread > max(p25_max, p25):
            return False, state
    return True, state


def _projection_candles(tfs: Sequence[str]) -> tuple[Optional[str], Optional[list[dict]]]:
    for tf in tfs:
        candles = get_candles_snapshot(tf, limit=120)
        if candles and len(candles) >= 30:
            return tf, list(candles)
    return None, None


def _score_ma(ma, side: str, opp_block_bars: float) -> Optional[float]:
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


def _score_rsi(rsi, side: str, long_target: float, short_target: float, overheat_bars: float) -> Optional[float]:
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


def _score_adx(adx, trend_mode: bool, threshold: float) -> Optional[float]:
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


def _score_bbw(bbw, threshold: float) -> Optional[float]:
    if bbw is None:
        return None
    if bbw.bbw <= threshold and bbw.slope_per_bar <= 0:
        return 0.5
    if bbw.bbw > threshold and bbw.slope_per_bar > 0:
        return -0.5
    return 0.0


def projection_decision(
    side: str,
    *,
    mode: str = "scalp",
    tfs: Sequence[str] = ("M1",),
) -> tuple[bool, float, dict]:
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
    elif mode == "range":
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
    else:  # scalp/pullback
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

    ma = compute_ma_projection({"candles": candles}, timeframe_minutes=minutes)
    rsi = compute_rsi_projection(candles, timeframe_minutes=minutes)
    adx = compute_adx_projection(candles, timeframe_minutes=minutes, trend_threshold=params["adx_threshold"])
    bbw = None
    if mode == "range":
        bbw = compute_bbw_projection(candles, timeframe_minutes=minutes, squeeze_threshold=params["bbw_threshold"])

    scores: Dict[str, float] = {}
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


def tick_snapshot(seconds: float, *, limit: int = 240) -> tuple[list[float], float]:
    ticks = tick_window.recent_ticks(seconds=seconds, limit=limit)
    if not ticks:
        return [], 0.0
    mids = []
    for t in ticks:
        mid = t.get("mid")
        if mid is None:
            bid = t.get("bid")
            ask = t.get("ask")
            if bid is None or ask is None:
                continue
            mid = (float(bid) + float(ask)) / 2.0
        mids.append(float(mid))
    span = 0.0
    if ticks:
        try:
            span = float(ticks[-1].get("epoch", 0.0)) - float(ticks[0].get("epoch", 0.0))
        except Exception:
            span = 0.0
    return mids, max(0.0, span)
