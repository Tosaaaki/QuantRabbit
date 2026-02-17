"""Micro multi-strategy worker with dynamic cap."""

from __future__ import annotations
from analysis.ma_projection import compute_adx_projection, compute_bbw_projection, compute_ma_projection, compute_rsi_projection
from analysis.technique_engine import evaluate_entry_techniques
from analysis.ma_projection import score_ma_for_side

import asyncio
import datetime
import hashlib
import logging
import pathlib
import sqlite3
import time
from typing import Dict, List, Optional, Tuple

from analysis.range_guard import detect_range_mode
from analysis.range_model import compute_range_snapshot
from indicators.factor_cache import all_factors, get_candles_snapshot, refresh_cache_from_disk
from execution.strategy_entry import market_order
from execution.risk_guard import allowed_lot, can_trade, clamp_sl_tp
from market_data import tick_window
from strategies.micro.momentum_burst import MomentumBurstMicro
from strategies.micro.momentum_stack import MicroMomentumStack
from strategies.micro.pullback_ema import MicroPullbackEMA
from strategies.micro.level_reactor import MicroLevelReactor
from strategies.micro.range_break import MicroRangeBreak
from strategies.micro.vwap_bound_revert import MicroVWAPBound
from strategies.micro_lowvol.micro_vwap_revert import MicroVWAPRevert
from strategies.micro_lowvol.compression_revert import MicroCompressionRevert
from strategies.micro_lowvol.momentum_pulse import MomentumPulse
from strategies.micro.trend_momentum import TrendMomentumMicro
from strategies.micro.trend_retest import MicroTrendRetest
from utils.divergence import apply_divergence_confidence, divergence_bias, divergence_snapshot
from utils.market_hours import is_market_open
from utils.oanda_account import get_account_snapshot, get_position_summary
from utils.metrics_logger import log_metric
from workers.common.dyn_cap import compute_cap
from workers.common.dynamic_alloc import load_strategy_profile
from workers.common import perf_guard
from workers.common.quality_gate import current_regime
from analysis import perf_monitor

from workers.common.size_utils import scale_base_units

from . import config

import os
from utils.env_utils import env_bool, env_float

_BB_ENV_PREFIX = getattr(config, "ENV_PREFIX", "")
_BB_ENTRY_ENABLED = env_bool("BB_ENTRY_ENABLED", True, prefix=_BB_ENV_PREFIX)
_BB_ENTRY_REVERT_PIPS = env_float("BB_ENTRY_REVERT_PIPS", 2.4, prefix=_BB_ENV_PREFIX)
_BB_ENTRY_REVERT_RATIO = env_float("BB_ENTRY_REVERT_RATIO", 0.22, prefix=_BB_ENV_PREFIX)
_BB_ENTRY_TREND_EXT_PIPS = env_float("BB_ENTRY_TREND_EXT_PIPS", 3.5, prefix=_BB_ENV_PREFIX)
_BB_ENTRY_TREND_EXT_RATIO = env_float("BB_ENTRY_TREND_EXT_RATIO", 0.40, prefix=_BB_ENV_PREFIX)
_BB_ENTRY_SCALP_REVERT_PIPS = env_float("BB_ENTRY_SCALP_REVERT_PIPS", 2.0, prefix=_BB_ENV_PREFIX)
_BB_ENTRY_SCALP_REVERT_RATIO = env_float("BB_ENTRY_SCALP_REVERT_RATIO", 0.20, prefix=_BB_ENV_PREFIX)
_BB_ENTRY_SCALP_EXT_PIPS = env_float("BB_ENTRY_SCALP_EXT_PIPS", 2.4, prefix=_BB_ENV_PREFIX)
_BB_ENTRY_SCALP_EXT_RATIO = env_float("BB_ENTRY_SCALP_EXT_RATIO", 0.30, prefix=_BB_ENV_PREFIX)
_BB_PIP = 0.01
try:
    from analysis.pattern_stats import derive_pattern_signature
except Exception:
    derive_pattern_signature = None  # type: ignore

# Pattern gate (pattern_book-driven). Opt-in per strategy to avoid global enforcement.
_MICRO_RANGEBREAK_PATTERN_GATE_OPT_IN = env_bool("MICRO_RANGEBREAK_PATTERN_GATE_OPT_IN", True)
_MICRO_RANGEBREAK_PATTERN_GATE_ALLOW_GENERIC = env_bool("MICRO_RANGEBREAK_PATTERN_GATE_ALLOW_GENERIC", True)
_MICRO_VWAPBOUND_PATTERN_GATE_OPT_IN = env_bool("MICRO_VWAPBOUND_PATTERN_GATE_OPT_IN", True)


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


def _mid_from_tick(tick: Dict) -> Optional[float]:
    mid = tick.get("mid")
    if mid is not None:
        try:
            return float(mid)
        except Exception:
            return None
    bid = tick.get("bid")
    ask = tick.get("ask")
    if bid is not None and ask is not None:
        try:
            return (float(bid) + float(ask)) / 2.0
        except Exception:
            return None
    return None


def _ts_ms_from_tick(tick: Dict) -> Optional[int]:
    ts_ms = tick.get("ts_ms") or tick.get("timestamp")
    if ts_ms is None:
        ts_ms = tick.get("epoch")
    if ts_ms is None:
        return None
    try:
        ts_val = int(float(ts_ms))
    except Exception:
        return None
    if ts_val < 10_000_000_000:
        ts_val *= 1000
    return ts_val


def _build_m1_from_ticks(ticks: List[Dict]) -> Optional[Dict[str, object]]:
    if not ticks:
        return None
    buckets: Dict[int, Dict[str, float]] = {}
    last_tick = None
    for tick in sorted(ticks, key=lambda t: (_ts_ms_from_tick(t) or 0)):
        ts_ms = _ts_ms_from_tick(tick)
        if ts_ms is None:
            continue
        price = _mid_from_tick(tick)
        if price is None:
            continue
        last_tick = tick
        bucket = ts_ms // 60000
        candle = buckets.get(bucket)
        if candle is None:
            buckets[bucket] = {"open": price, "high": price, "low": price, "close": price}
        else:
            candle["high"] = max(candle["high"], price)
            candle["low"] = min(candle["low"], price)
            candle["close"] = price
    if len(buckets) < max(3, config.FRESH_TICKS_MIN_CANDLES):
        return None
    items = sorted(buckets.items())
    candles = []
    for bucket, ohlc in items:
        candles.append(
            {
                "timestamp": int(bucket * 60),
                "open": ohlc["open"],
                "high": ohlc["high"],
                "low": ohlc["low"],
                "close": ohlc["close"],
            }
        )
    try:
        import pandas as pd
        from indicators.calc_core import IndicatorEngine
    except Exception:
        return None
    df = pd.DataFrame(candles)
    fac = IndicatorEngine.compute(df[["open", "high", "low", "close"]])
    last = candles[-1]
    fac.update(
        {
            "open": last["open"],
            "high": last["high"],
            "low": last["low"],
            "close": last["close"],
            "timestamp": datetime.datetime.utcfromtimestamp(last["timestamp"])
            .replace(tzinfo=datetime.timezone.utc)
            .isoformat(),
            "candles": candles,
        }
    )
    if last_tick is not None:
        bid = last_tick.get("bid")
        ask = last_tick.get("ask")
        if bid is not None and ask is not None:
            try:
                fac["spread_pips"] = (float(ask) - float(bid)) / _BB_PIP
            except Exception:
                pass
    return fac


def _trend_snapshot(fac_m1: Dict, fac_m5: Dict, fac_h1: Dict, fac_h4: Dict):
    for tf_name, fac in (("H4", fac_h4), ("H1", fac_h1), ("M5", fac_m5), ("M1", fac_m1)):
        if not fac:
            continue
        ma10 = fac.get("ma10")
        ma20 = fac.get("ma20")
        if ma10 is None or ma20 is None:
            continue
        try:
            gap_pips = (float(ma10) - float(ma20)) / _BB_PIP
        except Exception:
            continue
        if abs(gap_pips) < config.TREND_FLIP_GAP_PIPS:
            continue
        try:
            adx = float(fac.get("adx") or 0.0)
        except Exception:
            adx = 0.0
        direction = "long" if gap_pips > 0 else "short"
        return {
            "tf": tf_name,
            "gap_pips": round(gap_pips, 3),
            "direction": direction,
            "adx": round(adx, 2),
        }
    return None


def _apply_trend_flip(
    side: str,
    signal_tag: str,
    strategy_name: str,
    fac_m1: Dict,
    fac_m5: Dict,
    fac_h1: Dict,
    fac_h4: Dict,
):
    trend = _trend_snapshot(fac_m1, fac_m5, fac_h1, fac_h4)
    if not config.TREND_FLIP_ENABLED or not trend:
        return side, signal_tag, None, 1.0, 1.0, trend
    if config.TREND_FLIP_STRATEGY_ALLOWLIST and strategy_name not in config.TREND_FLIP_STRATEGY_ALLOWLIST:
        return side, signal_tag, None, 1.0, 1.0, trend
    if strategy_name in config.TREND_FLIP_STRATEGY_BLOCKLIST:
        return side, signal_tag, None, 1.0, 1.0, trend
    if trend["direction"] == side:
        return side, signal_tag, None, 1.0, 1.0, trend
    if trend["adx"] < config.TREND_FLIP_ADX_MIN:
        return side, signal_tag, None, 1.0, 1.0, trend
    if not (_is_mr_signal(signal_tag) or strategy_name in _RANGE_STRATEGIES):
        return side, signal_tag, None, 1.0, 1.0, trend
    flipped_side = trend["direction"]
    flip_meta = {
        "from": side,
        "to": flipped_side,
        "tf": trend["tf"],
        "gap_pips": trend["gap_pips"],
        "adx": trend["adx"],
    }
    return flipped_side, f"{signal_tag}-trendflip", flip_meta, config.TREND_FLIP_TP_MULT, config.TREND_FLIP_SL_MULT, trend

LOG = logging.getLogger(__name__)

MR_RANGE_LOOKBACK = 20
MR_RANGE_HI_PCT = 95.0
MR_RANGE_LO_PCT = 5.0
_STRATEGY_LAST_TS: Dict[str, float] = {}
_LAST_FRESH_M1_TS = 0.0
_LOCAL_FRESH_M1: Dict[str, float] | None = None
_TREND_STRATEGIES = {
    MomentumBurstMicro.name,
    MicroMomentumStack.name,
    MicroPullbackEMA.name,
    TrendMomentumMicro.name,
    MicroTrendRetest.name,
}
_PULLBACK_STRATEGIES = {
    MicroPullbackEMA.name,
}
_RANGE_STRATEGIES = {
    MicroRangeBreak.name,
    MicroVWAPBound.name,
    MicroLevelReactor.name,
    MicroVWAPRevert.name,
    MicroCompressionRevert.name,
}
_RANGE_TREND_ALLOWLIST = {name for name in config.RANGE_ONLY_TREND_ALLOWLIST}

_HISTORY_PROFILE_CACHE: Dict[Tuple[str, str, str], tuple[float, Dict[str, object]]] = {}
def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _normalize_tag_key(raw: object) -> str:
    if not raw:
        return ""
    key = str(raw).strip().lower()
    if not key:
        return ""
    if "-" in key:
        key = key.split("-", 1)[0].strip()
    return key


def _normalize_regime_label(raw: object) -> str:
    if not raw:
        return ""
    return str(raw).strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def _history_profile_cache_key(
    strategy_key: str,
    pocket: str,
    regime_label: str,
) -> tuple[str, str, str]:
    return (_normalize_tag_key(strategy_key), str(pocket).strip().lower(), regime_label)


def _query_strategy_history(
    *,
    strategy_key: str,
    pocket: str,
    regime_label: Optional[str],
) -> Dict[str, object]:
    strategy_key = _normalize_tag_key(strategy_key)
    if not strategy_key:
        return dict(n=0, pf=1.0, win_rate=0.0, avg_pips=0.0)
    db_path = pathlib.Path(config.HIST_DB_PATH)
    if not db_path.exists():
        return dict(n=0, pf=1.0, win_rate=0.0, avg_pips=0.0)

    lookback = max(1, int(config.HIST_LOOKBACK_DAYS))
    params: List[object] = [str(pocket).strip().lower(), f"-{lookback} day"]
    conditions = [
        "LOWER(pocket) = ?",
        "close_time IS NOT NULL",
        "datetime(close_time) >= datetime('now', ?)",
    ]

    pattern = f"{strategy_key}-%"
    conditions.append(
        "(LOWER(strategy) = ? OR LOWER(NULLIF(strategy_tag, '')) = ? OR LOWER(NULLIF(strategy_tag, '')) LIKE ?)"
    )
    params.extend([strategy_key, strategy_key, pattern])

    if regime_label:
        normalized_regime = _normalize_regime_label(regime_label)
        if normalized_regime:
            conditions.append(
                "(LOWER(COALESCE(micro_regime, '')) = ? OR LOWER(COALESCE(macro_regime, '')) = ?)"
            )
            params.extend([normalized_regime, normalized_regime])

    where = " AND ".join(conditions)
    con: sqlite3.Connection | None = None
    try:
        con = sqlite3.connect(str(db_path))
        con.row_factory = sqlite3.Row
        row = con.execute(
            f"""
            SELECT
              COUNT(*) AS n,
              SUM(CASE WHEN pl_pips > 0 THEN pl_pips ELSE 0 END) AS profit,
              SUM(CASE WHEN pl_pips < 0 THEN ABS(pl_pips) ELSE 0 END) AS loss,
              SUM(CASE WHEN pl_pips > 0 THEN 1 ELSE 0 END) AS win,
              SUM(pl_pips) AS sum_pips
            FROM trades
            WHERE {where}
            """,
            params,
        ).fetchone()
    except Exception:
        return dict(n=0, pf=1.0, win_rate=0.0, avg_pips=0.0)
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:
                pass
    if not row:
        return dict(n=0, pf=1.0, win_rate=0.0, avg_pips=0.0)

    n = int(row["n"] or 0)
    if n <= 0:
        return dict(n=0, pf=1.0, win_rate=0.0, avg_pips=0.0)
    profit = float(row["profit"] or 0.0)
    loss = float(row["loss"] or 0.0)
    win = float(row["win"] or 0.0)
    sum_pips = float(row["sum_pips"] or 0.0)
    pf = profit / loss if loss > 0 else float("inf")
    win_rate = win / n
    avg_pips = sum_pips / n
    return dict(n=n, pf=pf, win_rate=win_rate, avg_pips=avg_pips)


def _derive_history_score(row: Dict[str, object]) -> float:
    n = int(row.get("n", 0) or 0)
    pf = float(row.get("pf") or 1.0)
    win_rate = float(row.get("win_rate") or 0.0)
    avg_pips = float(row.get("avg_pips") or 0.0)

    if n <= 0:
        return 0.5

    if pf == float("inf"):
        pf_norm = 1.0
    else:
        pf_cap = max(1.001, float(config.HIST_PF_CAP))
        pf_norm = (pf - 1.0) / (pf_cap - 1.0)
        pf_norm = _clamp01(pf_norm)

    win_norm = _clamp01(win_rate)
    avg_norm = _clamp01((avg_pips + 4.0) / 8.0)
    score = 0.40 * pf_norm + 0.45 * win_norm + 0.15 * avg_norm

    if n < config.HIST_MIN_TRADES:
        weight = n / max(1.0, float(config.HIST_MIN_TRADES))
        score = 0.5 + (score - 0.5) * weight
    return score


def _history_profile(
    strategy_key: str,
    pocket: str,
    regime_label: Optional[str],
) -> Dict[str, object]:
    if not config.HIST_ENABLED:
        return {
            "enabled": False,
            "strategy_key": strategy_key,
            "pocket": pocket,
            "used_regime": bool(regime_label),
            "n": 0,
            "pf": 1.0,
            "win_rate": 0.0,
            "avg_pips": 0.0,
            "score": 0.5,
            "lot_multiplier": 1.0,
            "skip": False,
            "source": "disabled",
        }

    normalized_regime = _normalize_regime_label(regime_label)
    cache_key = _history_profile_cache_key(strategy_key, pocket, normalized_regime)
    now = time.time()
    cached = _HISTORY_PROFILE_CACHE.get(cache_key)
    if cached and (now - cached[0]) <= max(1.0, float(config.HIST_TTL_SEC)):
        return dict(cached[1])

    if not strategy_key:
        profile = {
            "enabled": True,
            "strategy_key": strategy_key,
            "pocket": pocket,
            "used_regime": bool(normalized_regime),
            "n": 0,
            "pf": 1.0,
            "win_rate": 0.0,
            "avg_pips": 0.0,
            "score": 0.5,
            "lot_multiplier": 1.0,
            "skip": False,
            "source": "empty_strategy",
        }
        _HISTORY_PROFILE_CACHE[cache_key] = (now, dict(profile))
        return profile

    row = _query_strategy_history(
        strategy_key=strategy_key,
        pocket=pocket,
        regime_label=normalized_regime,
    )
    used_regime = bool(normalized_regime)
    source = "regime"
    if normalized_regime and int(row.get("n", 0) or 0) < max(1, int(config.HIST_REGIME_MIN_TRADES)):
        fallback = _query_strategy_history(
            strategy_key=strategy_key,
            pocket=pocket,
            regime_label=None,
        )
        if int(fallback.get("n", 0) or 0) > 0:
            row = fallback
            used_regime = False
            source = "global"
    score = _derive_history_score(row)
    n = int(row.get("n", 0) or 0)
    pf = float(row.get("pf", 1.0))
    win_rate = float(row.get("win_rate", 0.0))
    avg_pips = float(row.get("avg_pips", 0.0))
    score = _clamp01(score)
    lot_mult = config.HIST_LOT_MIN + (config.HIST_LOT_MAX - config.HIST_LOT_MIN) * score
    lot_mult = max(config.HIST_LOT_MIN, min(config.HIST_LOT_MAX, lot_mult))
    skip = bool(n >= config.HIST_MIN_TRADES and score < config.HIST_SKIP_SCORE)

    profile = {
        "enabled": True,
        "strategy_key": strategy_key,
        "pocket": pocket,
        "used_regime": used_regime,
        "source": source,
        "n": n,
        "pf": pf if pf != float("inf") else float(config.HIST_PF_CAP),
        "win_rate": win_rate,
        "avg_pips": avg_pips,
        "score": score,
        "lot_multiplier": lot_mult,
        "skip": skip,
    }
    _HISTORY_PROFILE_CACHE[cache_key] = (now, dict(profile))
    return dict(profile)



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
    return score_ma_for_side(ma, side, opp_block_bars)


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
def _latest_mid(fallback: float) -> float:
    ticks = tick_window.recent_ticks(seconds=10.0, limit=1)
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


def _client_order_id(tag: str) -> str:
    ts_ms = int(time.time() * 1000)
    sanitized = "".join(ch.lower() for ch in tag if ch.isalnum())[:8] or "micro"
    digest = hashlib.sha1(f"{ts_ms}-{tag}".encode("utf-8")).hexdigest()[:9]
    return f"qr-{ts_ms}-micro-{sanitized}{digest}"


def _confidence_scale(conf: int) -> float:
    lo = config.CONFIDENCE_FLOOR
    hi = config.CONFIDENCE_CEIL
    if conf <= lo:
        return 0.55
    if conf >= hi:
        return 1.0
    span = (conf - lo) / max(1.0, hi - lo)
    return 0.55 + span * 0.45


def _to_confidence_0_100(confidence: object, default: float = 0.0) -> int:
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        conf = float(default)
    if conf < 0.0:
        conf = 0.0
    if conf <= 1.0:
        conf *= 100.0
    if conf > 100.0:
        conf = 100.0
    return int(round(conf))


def _to_probability(
    value: object,
    default_ratio: float = 0.0,
) -> float:
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return max(0.0, min(1.0, float(default_ratio)))
    if raw < 0.0:
        return 0.0
    if raw > 1.0:
        raw /= 100.0
    return max(0.0, min(1.0, raw))


def _compute_cap(*args, **kwargs) -> Tuple[float, Dict[str, float]]:
    kwargs.setdefault("env_prefix", config.ENV_PREFIX)
    res = compute_cap(cap_min=config.CAP_MIN, cap_max=config.CAP_MAX, *args, **kwargs)
    return res.cap, res.reasons


def _allowed_strategies() -> List:
    """
    Return strategy classes filtered by MICRO_STRATEGY_ALLOWLIST.
    Set env MICRO_STRATEGY_ALLOWLIST=\"MicroVWAPBound,TrendMomentumMicro\" to run only those.
    """
    allow_raw = (os.getenv("MICRO_STRATEGY_ALLOWLIST") or "").strip()
    all_classes = [
        MomentumBurstMicro,
        MicroMomentumStack,
        MicroPullbackEMA,
        MicroLevelReactor,
        MicroRangeBreak,
        MicroVWAPBound,
        MicroVWAPRevert,
        MomentumPulse,
        TrendMomentumMicro,
        MicroTrendRetest,
        MicroCompressionRevert,
    ]
    if not allow_raw:
        return all_classes
    allow = {s.strip() for s in allow_raw.split(",") if s.strip()}
    if not allow:
        return all_classes
    filtered = [cls for cls in all_classes if getattr(cls, "name", cls.__name__) in allow]
    if not filtered:
        LOG.warning("%s allowlist empty; using all strategies", config.LOG_PREFIX)
        return all_classes
    LOG.info(
        "%s allowlist applied: %s",
        config.LOG_PREFIX,
        ",".join(getattr(c, "name", c.__name__) for c in filtered),
    )
    return filtered


def _strategy_list() -> List:
    return _allowed_strategies()


def _mlr_strict_range_ok(
    fac_m1: Dict,
    *,
    range_active: bool,
    range_score: float,
) -> tuple[bool, Dict[str, float]]:
    if not config.MLR_STRICT_RANGE_GATE:
        return True, {}

    adx_val = _bb_float(fac_m1.get("adx")) or 0.0
    ma_fast = _bb_float(fac_m1.get("ma10")) or _bb_float(fac_m1.get("ema10"))
    ma_slow = _bb_float(fac_m1.get("ma20")) or _bb_float(fac_m1.get("ema20"))
    ma_gap_pips = 0.0
    if ma_fast is not None and ma_slow is not None:
        ma_gap_pips = abs(ma_fast - ma_slow) / _BB_PIP

    range_ready = bool(range_active) or range_score >= config.MLR_MIN_RANGE_SCORE
    adx_ok = config.MLR_MAX_ADX <= 0.0 or adx_val <= config.MLR_MAX_ADX
    ma_gap_ok = config.MLR_MAX_MA_GAP_PIPS <= 0.0 or ma_gap_pips <= config.MLR_MAX_MA_GAP_PIPS
    allow = bool(range_ready and adx_ok and ma_gap_ok)

    diag = {
        "range_active": 1.0 if range_active else 0.0,
        "range_score": float(range_score),
        "adx": float(adx_val),
        "ma_gap_pips": float(ma_gap_pips),
    }
    return allow, diag


def _is_mr_signal(tag: str) -> bool:
    tag_str = (tag or "").strip()
    if not tag_str:
        return False
    base_tag = tag_str.split("-", 1)[0]
    if base_tag in {"MicroVWAPBound", "MicroVWAPRevert", "MicroCompressionRevert", "BB_RSI"}:
        return True
    lower = tag_str.lower()
    return lower.startswith("mlr-fade") or lower.startswith("mlr-bounce")


def _build_mr_entry_thesis(
    signal: Dict,
    *,
    strategy_tag: str,
    atr_entry: float,
    entry_mean: Optional[float],
) -> Dict:
    thesis: Dict[str, object] = {
        "strategy_tag": strategy_tag,
        "profile": signal.get("profile"),
        "confidence": _to_confidence_0_100(signal.get("confidence", 0)),
        "env_tf": "H1",
        "struct_tf": "M5",
        "range_method": "percentile",
        "range_lookback": MR_RANGE_LOOKBACK,
        "range_hi_pct": MR_RANGE_HI_PCT,
        "range_lo_pct": MR_RANGE_LO_PCT,
        "atr_entry": atr_entry,
        "structure_break": {"buffer_atr": 0.10, "confirm_closes": 2},
        "tp_mode": "soft_zone",
        "tp_target": "entry_mean",
        "tp_pad_atr": 0.05,
        "reversion_failure": {
            "z_ext": 0.55,
            "contraction_min": 0.50,
            "bars_budget": {"k_per_z": 3.5, "min": 2, "max": 12},
            "trend_takeover": {"require_env_trend_bars": 2},
        },
    }
    if entry_mean is not None and entry_mean > 0:
        thesis["entry_mean"] = float(entry_mean)
    candles = get_candles_snapshot("H1", limit=MR_RANGE_LOOKBACK)
    snapshot = compute_range_snapshot(
        candles,
        lookback=MR_RANGE_LOOKBACK,
        method="percentile",
        hi_pct=MR_RANGE_HI_PCT,
        lo_pct=MR_RANGE_LO_PCT,
    )
    if snapshot:
        thesis["range_snapshot"] = snapshot.to_dict()
        thesis.setdefault("entry_mean", snapshot.mid)
    return thesis


def _factor_age_seconds(factors: Dict[str, float]) -> float:
    ts_raw = factors.get("timestamp") if isinstance(factors, dict) else None
    if not ts_raw:
        return float("inf")
    try:
        if isinstance(ts_raw, (int, float)):
            ts_val = float(ts_raw)
            ts_dt = datetime.datetime.utcfromtimestamp(ts_val).replace(tzinfo=datetime.timezone.utc)
        else:
            ts_txt = str(ts_raw)
            if ts_txt.endswith("Z"):
                ts_txt = ts_txt.replace("Z", "+00:00")
            ts_dt = datetime.datetime.fromisoformat(ts_txt)
            if ts_dt.tzinfo is None:
                ts_dt = ts_dt.replace(tzinfo=datetime.timezone.utc)
    except Exception:
        return float("inf")
    now = datetime.datetime.now(datetime.timezone.utc)
    return max(0.0, (now - ts_dt).total_seconds())


def _diversity_bonus(strategy_name: str, now_ts: float) -> float:
    if not config.DIVERSITY_ENABLED:
        return 0.0
    last_ts = _STRATEGY_LAST_TS.get(strategy_name)
    if last_ts is None:
        return config.DIVERSITY_MAX_BONUS
    idle = max(0.0, now_ts - last_ts)
    if idle < config.DIVERSITY_IDLE_SEC:
        return 0.0
    scale = max(1.0, config.DIVERSITY_SCALE_SEC)
    bonus = (idle - config.DIVERSITY_IDLE_SEC) / scale * config.DIVERSITY_MAX_BONUS
    return min(config.DIVERSITY_MAX_BONUS, bonus)


def _strategy_cooldown_active(strategy_name: str, now_ts: float) -> bool:
    cooldown = max(0.0, float(getattr(config, "STRATEGY_COOLDOWN_SEC", 0.0)))
    if cooldown <= 0.0:
        return False
    last_ts = _STRATEGY_LAST_TS.get(strategy_name)
    if last_ts is None:
        return False
    return (now_ts - last_ts) < cooldown


async def micro_multi_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s disabled (idle)", config.LOG_PREFIX)
        try:
            while True:
                await asyncio.sleep(3600.0)
        except asyncio.CancelledError:
            return
    LOG.info("%s worker start (interval=%.1fs)", config.LOG_PREFIX, config.LOOP_INTERVAL_SEC)
    LOG.warning("Application started! %s", config.LOG_PREFIX)
    state = globals()
    last_fresh_m1_ts = float(state.get("_LAST_FRESH_M1_TS", 0.0))
    local_fresh_m1 = state.get("_LOCAL_FRESH_M1")
    last_trend_block_log = 0.0
    last_stale_log = 0.0
    last_stale_scale_log = 0.0
    last_perf_block_log = 0.0
    last_mlr_block_log = 0.0

    while True:
        await asyncio.sleep(config.LOOP_INTERVAL_SEC)
        now = datetime.datetime.utcnow()
        if not is_market_open(now):
            continue
        if not can_trade(config.POCKET):
            continue
        current_hour = now.hour
        now_ts = time.time()
        stale_scale = 1.0

        # 最新キャッシュに更新（他プロセスが書いた factor_cache.json を取り込む）
        try:
            refresh_cache_from_disk()
        except Exception:
            pass
        factors = all_factors()
        fac_m1_disk = factors.get("M1") or {}
        fac_m1 = fac_m1_disk
        if local_fresh_m1 is not None:
            age_disk = _factor_age_seconds(fac_m1_disk)
            age_local = _factor_age_seconds(local_fresh_m1)
            if not fac_m1_disk or age_local < age_disk:
                fac_m1 = local_fresh_m1
        fac_h4 = factors.get("H4") or {}
        fac_h1 = factors.get("H1") or {}
        fac_m5 = factors.get("M5") or {}
        age_m1 = _factor_age_seconds(fac_m1)
        if age_m1 > config.MAX_FACTOR_AGE_SEC:
            # Refresh M1 factors from recent ticks instead of blocking entries.
            if config.FRESH_TICKS_ON_STALE and now_ts - last_fresh_m1_ts >= config.FRESH_TICKS_REFRESH_SEC:
                try:
                    tick_limit = max(1000, int(config.FRESH_TICKS_LOOKBACK_SEC * 5))
                    ticks = tick_window.recent_ticks(
                        seconds=config.FRESH_TICKS_LOOKBACK_SEC,
                        limit=tick_limit,
                    )
                    fresh = _build_m1_from_ticks(ticks)
                except Exception:
                    fresh = None
                if fresh:
                    fac_m1 = fresh
                    age_m1 = _factor_age_seconds(fac_m1)
                    state["_LAST_FRESH_M1_TS"] = now_ts
                    state["_LOCAL_FRESH_M1"] = fac_m1
                    last_fresh_m1_ts = now_ts
                    local_fresh_m1 = fac_m1
                    log_metric(
                        "micro_multi_refresh_m1",
                        float(age_m1),
                        tags={"source": "ticks", "candles": len(fresh.get("candles") or [])},
                        ts=now,
                    )
            if age_m1 > config.MAX_FACTOR_AGE_SEC:
                # 入口を止める代わりにログだけ残して評価を継続（データ欠損で固まらないようにする）
                log_metric(
                    "micro_multi_skip",
                    float(age_m1),
                    tags={"reason": "factor_stale_warn", "tf": "M1"},
                    ts=now,
                )
            else:
                # tick 再構成で復旧できた場合、明示的にはアラートを抑制する。
                local_fresh_m1 = fac_m1
                state["_LOCAL_FRESH_M1"] = fac_m1
            if age_m1 > config.MAX_FACTOR_AGE_SEC:
                hard_age = max(config.MAX_FACTOR_AGE_SEC, config.FRESH_TICKS_STALE_SCALE_HARD_SEC)
                if hard_age > config.MAX_FACTOR_AGE_SEC:
                    stale_ratio = max(0.0, (age_m1 - config.MAX_FACTOR_AGE_SEC) / (hard_age - config.MAX_FACTOR_AGE_SEC))
                    stale_scale = max(
                        config.FRESH_TICKS_STALE_SCALE_MIN,
                        1.0 - stale_ratio * (1.0 - config.FRESH_TICKS_STALE_SCALE_MIN),
                    )
                if stale_scale < 1.0:
                    if now_ts - last_stale_scale_log > 30.0:
                        LOG.info(
                            "%s factor stale scale=%.3f age=%.1fs hard_age=%.1fs",
                            config.LOG_PREFIX,
                            stale_scale,
                            age_m1,
                            hard_age,
                        )
                        last_stale_scale_log = now_ts
                if now_ts - last_stale_log > 30.0:
                    LOG.warning(
                        "%s stale factors age=%.1fs limit=%.1fs (proceeding anyway)",
                        config.LOG_PREFIX,
                        age_m1,
                        config.MAX_FACTOR_AGE_SEC,
                    )
                    last_stale_log = now_ts
        range_ctx = detect_range_mode(fac_m1, fac_h4)
        range_score = 0.0
        try:
            range_score = float(range_ctx.score or 0.0)
        except Exception:
            range_score = 0.0
        range_only = range_ctx.active or range_score >= config.RANGE_ONLY_SCORE
        range_bias = range_score >= config.RANGE_BIAS_SCORE
        fac_m1 = dict(fac_m1)
        fac_m1["range_active"] = bool(range_ctx.active)
        fac_m1["range_score"] = range_score
        fac_m1["range_reason"] = range_ctx.reason
        fac_m1["range_mode"] = range_ctx.mode
        regime_label = _normalize_regime_label(fac_m1.get("regime"))
        if not regime_label:
            regime_label = _normalize_regime_label(current_regime("M1", event_mode=False))
        perf = perf_monitor.snapshot()
        pf = None
        try:
            pf = float((perf.get(config.POCKET) or {}).get("pf"))
        except Exception:
            pf = None

        candidates: List[Tuple[float, int, Dict, str, Dict[str, object], Dict[str, object]]] = []
        for strat in _strategy_list():
            strategy_name = getattr(strat, "name", strat.__name__)
            if _strategy_cooldown_active(strategy_name, now_ts):
                continue
            if (
                strategy_name == TrendMomentumMicro.name
                and current_hour in config.TREND_BLOCK_HOURS_UTC
            ):
                now_mono = time.monotonic()
                if now_mono - last_trend_block_log > 300.0:
                    LOG.info(
                        "%s skip %s at hour=%02d block_hours=%s",
                        config.LOG_PREFIX,
                        TrendMomentumMicro.name,
                        current_hour,
                        sorted(config.TREND_BLOCK_HOURS_UTC),
                    )
                    last_trend_block_log = now_mono
                continue
            if (
                range_only
                and strategy_name not in _RANGE_STRATEGIES
                and strategy_name not in _RANGE_TREND_ALLOWLIST
            ):
                continue
            if strategy_name == MicroLevelReactor.name:
                mlr_ok, mlr_diag = _mlr_strict_range_ok(
                    fac_m1,
                    range_active=bool(range_ctx.active),
                    range_score=range_score,
                )
                if not mlr_ok:
                    now_mono = time.monotonic()
                    if now_mono - last_mlr_block_log > 120.0:
                        LOG.info(
                            "%s mlr_range_gate_block active=%s score=%.3f adx=%.2f ma_gap=%.2f",
                            config.LOG_PREFIX,
                            bool(mlr_diag.get("range_active")),
                            float(mlr_diag.get("range_score", 0.0)),
                            float(mlr_diag.get("adx", 0.0)),
                            float(mlr_diag.get("ma_gap_pips", 0.0)),
                        )
                        last_mlr_block_log = now_mono
                    continue
            cand = strat.check(fac_m1)
            if not cand:
                continue
            perf_decision = perf_guard.is_allowed(strategy_name, config.POCKET, env_prefix=config.ENV_PREFIX)
            if not perf_decision.allowed:
                now_mono = time.monotonic()
                if now_mono - last_perf_block_log > 120.0:
                    LOG.info(
                        "%s perf_block tag=%s reason=%s",
                        config.LOG_PREFIX,
                        strategy_name,
                        perf_decision.reason,
                    )
                    last_perf_block_log = now_mono
                continue
            dyn_profile: Dict[str, object] = {}
            if config.DYN_ALLOC_ENABLED:
                dyn_profile = load_strategy_profile(
                    strategy_name,
                    config.POCKET,
                    path=config.DYN_ALLOC_PATH,
                    ttl_sec=config.DYN_ALLOC_TTL_SEC,
                )
                if config.DYN_ALLOC_LOSER_BLOCK and bool(dyn_profile.get("found")):
                    dyn_trades = int(dyn_profile.get("trades", 0) or 0)
                    dyn_score = float(dyn_profile.get("score", 0.0) or 0.0)
                    if (
                        dyn_trades >= config.DYN_ALLOC_MIN_TRADES
                        and dyn_score <= config.DYN_ALLOC_LOSER_SCORE
                    ):
                        continue
            signal_tag = str(cand.get("tag", strategy_name))
            base_tag = signal_tag.split("-", 1)[0].strip()
            if not base_tag:
                base_tag = strategy_name
            hist_profile = _history_profile(
                strategy_key=base_tag,
                pocket=config.POCKET,
                regime_label=regime_label,
            )
            if bool(hist_profile.get("skip")):
                now_mono = time.monotonic()
                if now_mono - last_perf_block_log > 120.0:
                    LOG.info(
                        "%s hist_block tag=%s strategy=%s n=%s score=%.3f reason=low_recent_score",
                        config.LOG_PREFIX,
                        signal_tag,
                        strategy_name,
                        int(hist_profile.get("n", 0)),
                        float(hist_profile.get("score", 0.0)),
                    )
                    last_perf_block_log = now_mono
                continue
            base_conf = _to_confidence_0_100(cand.get("confidence", 0))
            bonus = _diversity_bonus(strategy_name, now_ts)
            score = base_conf + bonus
            if range_bias:
                if strategy_name in _RANGE_STRATEGIES:
                    score += config.RANGE_STRATEGY_BONUS * range_score
                elif strategy_name in _RANGE_TREND_ALLOWLIST:
                    score -= config.RANGE_TREND_PENALTY * range_score * 0.35
                else:
                    score -= config.RANGE_TREND_PENALTY * range_score
            if config.DYN_ALLOC_ENABLED and bool(dyn_profile.get("found")):
                score += config.DYN_ALLOC_SCORE_BONUS * float(dyn_profile.get("score", 0.0) or 0.0)
            score += config.HIST_CONF_WEIGHT * float(hist_profile.get("score", 0.5))
            candidates.append((score, base_conf, cand, strategy_name, dyn_profile, hist_profile))
        if not candidates:
            continue
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        max_signals = max(1, int(config.MAX_SIGNALS_PER_CYCLE))
        selected = candidates[:max_signals]
        if config.DYN_ALLOC_ENABLED and config.DYN_ALLOC_WINNER_ONLY:
            winners = [
                item
                for item in candidates
                if bool(item[4].get("found"))
                and int(item[4].get("trades", 0) or 0) >= config.DYN_ALLOC_MIN_TRADES
                and float(item[4].get("score", 0.0) or 0.0) >= config.DYN_ALLOC_WINNER_SCORE
            ]
            if winners:
                selected = winners[:max_signals]

        snap = get_account_snapshot()
        equity = float(snap.nav or snap.balance or 0.0)

        balance = float(snap.balance or snap.nav or 0.0)
        free_ratio = float(snap.free_margin_ratio or 0.0) if snap.free_margin_ratio is not None else 0.0
        try:
            atr_pips = float(fac_m1.get("atr_pips") or 0.0)
        except Exception:
            atr_pips = 0.0
        try:
            atr_m5 = float(fac_m5.get("atr_pips") or 0.0)
        except Exception:
            atr_m5 = 0.0
        pos_bias = 0.0
        try:
            open_positions = snap.positions or {}
            micro_pos = open_positions.get("micro") or {}
            pos_bias = abs(float(micro_pos.get("units", 0.0) or 0.0)) / max(1.0, float(snap.nav or 1.0))
        except Exception:
            pos_bias = 0.0

        cap, cap_reason = _compute_cap(
            atr_pips=atr_pips,
            free_ratio=free_ratio,
            range_active=range_ctx.active,
            perf_pf=pf,
            pos_bias=pos_bias,
        )
        if cap <= 0.0:
            continue
        multi_scale = max(
            float(config.MULTI_SIGNAL_MIN_SCALE),
            1.0 / max(1, len(selected)),
        )

        try:
            price = float(fac_m1.get("close") or 0.0)
        except Exception:
            price = 0.0
        price = _latest_mid(price)
        long_units = 0.0
        short_units = 0.0
        try:
            long_units, short_units = get_position_summary("USD_JPY", timeout=3.0)
        except Exception:
            long_units, short_units = 0.0, 0.0
        for _, _, signal, strategy_name, dyn_profile, hist_profile in selected:
            side = "long" if signal["action"] == "OPEN_LONG" else "short"
            orig_side = side
            orig_action = signal.get("action")
            sl_pips = float(signal.get("sl_pips") or 0.0)
            tp_pips = float(signal.get("tp_pips") or 0.0)
            if price <= 0.0 or sl_pips <= 0.0:
                continue
            signal_tag = signal.get("tag", strategy_name)

            trend_flip_meta = None
            trend_snapshot = None
            tp_mult = 1.0
            sl_mult = 1.0
            side, signal_tag, trend_flip_meta, tp_mult, sl_mult, trend_snapshot = _apply_trend_flip(
                side,
                signal_tag,
                strategy_name,
                fac_m1,
                fac_m5,
                fac_h1,
                fac_h4,
            )
            if trend_flip_meta:
                tp_pips = round(tp_pips * tp_mult, 2)
                sl_pips = round(sl_pips * sl_mult, 2)
            signal_action = "OPEN_LONG" if side == "long" else "OPEN_SHORT"

            proj_mode = None
            if strategy_name in _RANGE_STRATEGIES:
                proj_mode = "range"
            elif strategy_name in _PULLBACK_STRATEGIES:
                proj_mode = "pullback"
            elif strategy_name in _TREND_STRATEGIES:
                proj_mode = "trend"
            proj_allow, proj_mult, proj_detail = _projection_decision(
                side,
                config.POCKET,
                mode_override=proj_mode,
            )
            proj_flip_meta = None
            if not proj_allow and config.PROJ_FLIP_ENABLED:
                opp_side = "short" if side == "long" else "long"
                opp_allow, opp_mult, opp_detail = _projection_decision(
                    opp_side,
                    config.POCKET,
                    mode_override=proj_mode,
                )
                if opp_allow:
                    proj_flip_meta = {
                        "from": side,
                        "to": opp_side,
                        "mode": proj_mode,
                    }
                    side = opp_side
                    signal_action = "OPEN_LONG" if side == "long" else "OPEN_SHORT"
                    signal_tag = f"{signal_tag}-projflip"
                    proj_allow, proj_mult, proj_detail = opp_allow, opp_mult, opp_detail
            proj_conflict = None
            if not proj_allow and config.PROJ_CONFLICT_ALLOW:
                proj_conflict = True
                proj_allow = True
                if proj_detail is not None:
                    proj_detail = dict(proj_detail)
                    proj_detail["conflict"] = True
                    proj_detail["conflict_reason"] = "projection_block_override"
            if not proj_allow:
                continue

            signal_mode = str(signal.get("signal_mode") or "").strip().lower()
            if trend_flip_meta or proj_flip_meta:
                bb_style = "trend"
            elif signal_mode == "reversion":
                bb_style = "reversion"
            elif signal_mode == "trend":
                bb_style = "trend"
            elif strategy_name in _RANGE_STRATEGIES or _is_mr_signal(signal_tag):
                bb_style = "reversion"
            elif strategy_name in _PULLBACK_STRATEGIES:
                bb_style = "trend"
            if not _bb_entry_allowed(bb_style, side, price, fac_m1, range_active=range_ctx.active):
                continue

            base_tag = signal_tag.split("-", 1)[0] if signal_tag else ""
            if base_tag in {"MicroVWAPBound", "MicroVWAPRevert"}:
                div_mode = "reversion"
                max_age = 18
                max_bonus = 8.0
                max_penalty = 10.0
                floor = 45.0
                ceil = 92.0
            elif base_tag == "BB_RSI":
                div_mode = "reversion"
                max_age = 16
                max_bonus = 10.0
                max_penalty = 12.0
                floor = 30.0
                ceil = 95.0
            else:
                div_mode = "trend" if strategy_name in _TREND_STRATEGIES else "neutral"
                if strategy_name in _PULLBACK_STRATEGIES:
                    div_mode = "trend"
                max_age = 12
                max_bonus = 6.0
                max_penalty = 8.0
                floor = 40.0
                ceil = 95.0
            div_meta = {}
            div_bias = divergence_bias(
                fac_m1,
                signal_action,
                mode=div_mode,
                max_age_bars=max_age,
            )
            if div_bias:
                base_conf = _to_confidence_0_100(signal.get("confidence", 0))
                signal["confidence"] = _to_confidence_0_100(
                    apply_divergence_confidence(
                        base_conf,
                        div_bias,
                        max_bonus=max_bonus,
                        max_penalty=max_penalty,
                        floor=floor,
                        ceil=ceil,
                    )
                )
            div_meta = divergence_snapshot(fac_m1, max_age_bars=max_age)

            tp_scale = 10.0 / max(1.0, tp_pips)
            tp_scale = max(0.4, min(1.1, tp_scale))
            base_units = int(
                round(
                    scale_base_units(
                        config.BASE_ENTRY_UNITS,
                        equity=balance if balance > 0 else equity,
                        ref_equity=balance,
                        env_prefix=config.ENV_PREFIX,
                    )
                    * tp_scale
                )
            )

            conf_scale = _confidence_scale(_to_confidence_0_100(signal.get("confidence", 50)))
            dyn_mult = 1.0
            dyn_score = 0.0
            dyn_trades = 0
            strategy_units_mult = 1.0
            hist_mult = 1.0
            hist_score = float(hist_profile.get("score", 0.5) if isinstance(hist_profile, dict) else 0.5)
            hist_n = int(hist_profile.get("n", 0) if isinstance(hist_profile, dict) else 0)
            hist_source = str(hist_profile.get("source", "disabled") if isinstance(hist_profile, dict) else "disabled")
            if base_tag:
                strategy_units_mult = float(config.STRATEGY_UNITS_MULT.get(base_tag, 1.0) or 1.0)
            if strategy_units_mult <= 0.0:
                strategy_units_mult = 1.0
            if isinstance(hist_profile, dict):
                hist_mult = float(hist_profile.get("lot_multiplier", 1.0) or 1.0)
                hist_mult = max(config.HIST_LOT_MIN, min(config.HIST_LOT_MAX, hist_mult))
            if config.DYN_ALLOC_ENABLED and bool(dyn_profile.get("found")):
                dyn_mult = float(dyn_profile.get("lot_multiplier", 1.0) or 1.0)
                dyn_mult = max(config.DYN_ALLOC_MULT_MIN, min(config.DYN_ALLOC_MULT_MAX, dyn_mult))
                dyn_score = float(dyn_profile.get("score", 0.0) or 0.0)
                dyn_trades = int(dyn_profile.get("trades", 0) or 0)
            lot = allowed_lot(
                float(snap.nav or 0.0),
                sl_pips,
                margin_available=float(snap.margin_available or 0.0),
                price=price,
                margin_rate=float(snap.margin_rate or 0.0),
                pocket=config.POCKET,
                side=side,
                open_long_units=long_units,
                open_short_units=short_units,
                strategy_tag=signal_tag,
                fac_m1=fac_m1,
                fac_h4=fac_h4,
            )
            units_risk = int(round(lot * 100000))
            units = int(round(base_units * conf_scale))
            units = min(units, units_risk)
            units = int(round(units * cap))
            units = int(round(units * multi_scale))
            units = int(round(units * stale_scale))
            units = int(round(units * hist_mult))
            units = int(round(units * dyn_mult))
            units = int(round(units * strategy_units_mult))
            if units < config.MIN_UNITS:
                continue
            if side == "short":
                units = -abs(units)

            if side == "long":
                sl_price = round(price - sl_pips * 0.01, 3)
                tp_price = round(price + tp_pips * 0.01, 3) if tp_pips > 0 else None
            else:
                sl_price = round(price + sl_pips * 0.01, 3)
                tp_price = round(price - tp_pips * 0.01, 3) if tp_pips > 0 else None

            sl_price, tp_price = clamp_sl_tp(
                price=price,
                sl=sl_price,
                tp=tp_price,
                is_buy=side == "long",
            )
            client_id = _client_order_id(signal_tag)
            signal_conf = _to_confidence_0_100(signal.get("confidence", 0))
            entry_thesis: Dict[str, object] = {
                "strategy_tag": signal_tag,
                "signal_action": orig_action,
                "signal_side": orig_side,
                "exec_action": signal_action,
                "exec_side": side,
                "signal_mode": signal_mode,
                "profile": signal.get("profile"),
                "confidence": signal_conf,
                "entry_probability": round(
                    _to_probability(signal.get("entry_probability"), signal_conf / 100.0),
                    3,
                ),
                "tp_pips": tp_pips,
                "sl_pips": sl_pips,
                "hard_stop_pips": sl_pips,
                "range_active": bool(range_ctx.active),
                "range_score": round(range_score, 3),
                "range_reason": range_ctx.reason,
                "range_mode": range_ctx.mode,
            }
            base_tag = str(signal_tag or "").split("-", 1)[0]
            if base_tag == "MicroRangeBreak":
                entry_thesis["pattern_gate_opt_in"] = bool(_MICRO_RANGEBREAK_PATTERN_GATE_OPT_IN)
                if _MICRO_RANGEBREAK_PATTERN_GATE_ALLOW_GENERIC:
                    entry_thesis["pattern_gate_allow_generic"] = True
            elif base_tag == "MicroVWAPBound":
                entry_thesis["pattern_gate_opt_in"] = bool(_MICRO_VWAPBOUND_PATTERN_GATE_OPT_IN)
            if config.DYN_ALLOC_ENABLED and bool(dyn_profile.get("found")):
                entry_thesis["dynamic_alloc"] = {
                    "strategy_key": dyn_profile.get("strategy_key"),
                    "score": round(dyn_score, 3),
                    "trades": dyn_trades,
                    "lot_multiplier": round(dyn_mult, 3),
                }
            if isinstance(hist_profile, dict):
                entry_thesis["history_perf"] = {
                    "strategy_key": hist_profile.get("strategy_key", base_tag),
                    "source": hist_profile.get("source", "disabled"),
                    "n": int(hist_profile.get("n", 0) or 0),
                    "score": round(float(hist_profile.get("score", 0.5) or 0.5), 3),
                    "lot_multiplier": round(float(hist_profile.get("lot_multiplier", 1.0) or 1.0), 3),
                    "pf": round(float(hist_profile.get("pf", 1.0) or 1.0), 3),
                    "win_rate": round(float(hist_profile.get("win_rate", 0.0) or 0.0), 3),
                    "avg_pips": round(float(hist_profile.get("avg_pips", 0.0) or 0.0), 3),
                }
            if abs(strategy_units_mult - 1.0) > 1e-9:
                entry_thesis["strategy_units_mult"] = round(strategy_units_mult, 3)
            if div_meta:
                entry_thesis["divergence"] = div_meta
            if trend_snapshot:
                entry_thesis["trend_snapshot"] = trend_snapshot
            if trend_flip_meta:
                entry_thesis["trend_flip"] = trend_flip_meta
            if proj_flip_meta:
                entry_thesis["projection_flip"] = proj_flip_meta
            if proj_conflict:
                entry_thesis["projection_conflict"] = True
            if proj_detail:
                entry_thesis["projection"] = proj_detail
            if derive_pattern_signature is not None:
                try:
                    pattern_fac = {
                        "open": fac_m1.get("open"),
                        "high": fac_m1.get("high"),
                        "low": fac_m1.get("low"),
                        "close": fac_m1.get("close"),
                        "ma10": fac_m1.get("ma10"),
                        "ma20": fac_m1.get("ma20"),
                        "rsi": fac_m1.get("rsi"),
                        "atr_pips": fac_m1.get("atr_pips"),
                        "bbw": fac_m1.get("bbw"),
                    }
                    pattern_tag, pattern_meta = derive_pattern_signature(
                        pattern_fac,
                        action=signal_action,
                    )
                    if pattern_tag:
                        entry_thesis["pattern_tag"] = pattern_tag
                    if pattern_meta:
                        entry_thesis["pattern_meta"] = pattern_meta
                except Exception:
                    pass
            if strategy_name in _TREND_STRATEGIES:
                entry_thesis["entry_guard_trend"] = True
                entry_thesis["entry_tf"] = "M5"
            if strategy_name in _PULLBACK_STRATEGIES:
                entry_thesis["entry_guard_pullback"] = True
                entry_thesis["entry_guard_pullback_only"] = True
            if _is_mr_signal(signal_tag):
                entry_thesis["entry_guard_trend"] = False
                entry_mean = None
                base_tag = signal_tag.split("-", 1)[0] if signal_tag else ""
                if base_tag in {"MicroVWAPBound", "MicroVWAPRevert"}:
                    notes = signal.get("notes") or {}
                    if isinstance(notes, dict):
                        try:
                            entry_mean = float(notes.get("vwap"))
                        except Exception:
                            entry_mean = None
                elif base_tag == "BB_RSI":
                    try:
                        entry_mean = float(fac_m1.get("ma20") or fac_m1.get("ma10") or 0.0) or None
                    except Exception:
                        entry_mean = None
                entry_thesis.update(
                    _build_mr_entry_thesis(
                        signal,
                        strategy_tag=signal_tag,
                        atr_entry=atr_m5 or atr_pips or 1.0,
                        entry_mean=entry_mean,
                    )
                )
                if base_tag in {"MicroVWAPBound", "MicroVWAPRevert"}:
                    rf = entry_thesis.get("reversion_failure")
                    if isinstance(rf, dict):
                        rf["z_ext"] = min(float(rf.get("z_ext") or 0.40), 0.40)
                        rf["contraction_min"] = max(float(rf.get("contraction_min") or 0.55), 0.55)
                        bars_budget = rf.get("bars_budget")
                        if not isinstance(bars_budget, dict):
                            bars_budget = {}
                            rf["bars_budget"] = bars_budget
                        bars_budget["k_per_z"] = min(float(bars_budget.get("k_per_z") or 2.6), 2.6)
                        bars_budget["max"] = min(int(bars_budget.get("max") or 8), 8)
                if base_tag == "MicroVWAPBound":
                    notes = signal.get("notes") or {}
                    z_val = None
                    if isinstance(notes, dict):
                        try:
                            z_val = abs(float(notes.get("z")))
                        except Exception:
                            z_val = None
                    rf = entry_thesis.get("reversion_failure")
                    if isinstance(rf, dict) and z_val is not None:
                        bars_budget = rf.get("bars_budget")
                        if not isinstance(bars_budget, dict):
                            bars_budget = {}
                            rf["bars_budget"] = bars_budget
                        if z_val >= 2.5:
                            bars_budget["k_per_z"] = max(float(bars_budget.get("k_per_z") or 0.0), 3.1)
                            bars_budget["max"] = max(int(bars_budget.get("max") or 0), 10)
                        elif z_val <= 1.4:
                            bars_budget["k_per_z"] = min(float(bars_budget.get("k_per_z") or 2.6), 2.3)
                            bars_budget["max"] = min(int(bars_budget.get("max") or 8), 7)
                atr_for_vol = atr_m5 or atr_pips or 0.0
                low_vol = (
                    atr_for_vol > 0
                    and atr_for_vol <= MicroVWAPBound.LOW_VOL_ATR_PIPS
                    and range_score >= MicroVWAPBound.LOW_VOL_RANGE_SCORE
                )
                if low_vol:
                    rf = entry_thesis.get("reversion_failure")
                    if isinstance(rf, dict):
                        bars_budget = rf.get("bars_budget")
                        if not isinstance(bars_budget, dict):
                            bars_budget = {}
                            rf["bars_budget"] = bars_budget
                        bars_budget["k_per_z"] = max(
                            float(bars_budget.get("k_per_z") or 0.0),
                            MicroVWAPBound.LOW_VOL_HOLD_K_PER_Z,
                        )
                        bars_budget["max"] = max(
                            int(bars_budget.get("max") or 0),
                            MicroVWAPBound.LOW_VOL_HOLD_MAX_BARS,
                        )
                    entry_thesis["low_vol"] = True

            if proj_mult > 1.0:
                sign = 1 if units > 0 else -1
                units = int(round(abs(units) * proj_mult)) * sign

            candle_allow, candle_mult = _entry_candle_guard("long" if units > 0 else "short")
            if not candle_allow:
                continue
            if candle_mult != 1.0:
                sign = 1 if units > 0 else -1
                units = int(round(abs(units) * candle_mult)) * sign
            entry_thesis["entry_units_intent"] = abs(int(units))

            entry_thesis.setdefault("env_prefix", config.ENV_PREFIX)
            entry_thesis_ctx = None
            for _name in ("entry_thesis", "thesis"):
                _candidate = locals().get(_name)
                if isinstance(_candidate, dict):
                    entry_thesis_ctx = _candidate
                    break
            if entry_thesis_ctx is None:
                entry_thesis_ctx = {}

            _tech_pocket = str(locals().get("pocket", config.POCKET))
            _tech_side_raw = str(locals().get("side", locals().get("direction", "long"))).lower()
            if _tech_side_raw in {"long", "short"}:
                _tech_side = _tech_side_raw
            else:
                _tech_side = "long"
            _tech_entry_price = locals().get("price")
            if not isinstance(_tech_entry_price, (int, float)):
                _tech_entry_price = locals().get("entry_price")
            if not isinstance(_tech_entry_price, (int, float)):
                _tech_entry_price = 0.0
            try:
                _tech_entry_price = float(_tech_entry_price)
            except (TypeError, ValueError):
                _tech_entry_price = 0.0

            _tech_signal_tag = str(
                locals().get("signal_tag")
                or locals().get("strategy_tag")
                or locals().get("STRATEGY_TAG")
                or getattr(config, "STRATEGY_TAG", "")
            )

            entry_thesis_ctx.setdefault(
                "tech_tfs",
                {"fib": ["H1", "M5"], "median": ["H1", "M5"], "nwave": ["M1", "M5"], "candle": ["M1", "M5"]},
            )
            entry_thesis_ctx.setdefault("technical_context_tfs", ["M1", "M5", "H1", "H4"])
            entry_thesis_ctx.setdefault(
                "technical_context_fields",
                [
                    "ma10",
                    "ma20",
                    "rsi",
                    "atr",
                    "atr_pips",
                    "adx",
                    "macd",
                    "macd_hist",
                    "plus_di",
                    "minus_di",
                    "bbw",
                    "kc_width",
                    "vwap",
                    "ema20",
                    "ema24",
                ],
            )
            entry_thesis_ctx.setdefault("technical_context_ticks", ["latest_bid", "latest_ask", "latest_mid", "spread_pips"])
            entry_thesis_ctx.setdefault("technical_context_candle_counts", {"M1": 120, "M5": 80, "H1": 70, "H4": 60})
            entry_thesis_ctx.setdefault("tech_allow_candle", True)
            entry_thesis_ctx.setdefault(
                "tech_policy",
                {
                    "mode": "balanced",
                    "min_score": 0.0,
                    "min_coverage": 0.0,
                    "weight_fib": 0.25,
                    "weight_median": 0.25,
                    "weight_nwave": 0.25,
                    "weight_candle": 0.25,
                    "require_fib": False,
                    "require_median": False,
                    "require_nwave": False,
                    "require_candle": False,
                    "size_scale": 0.15,
                    "size_min": 0.6,
                    "size_max": 1.25,
                },
            )
            entry_thesis_ctx.setdefault("tech_policy_locked", False)
            entry_thesis_ctx.setdefault("env_tf", "M1")
            entry_thesis_ctx.setdefault("struct_tf", "M1")
            entry_thesis_ctx.setdefault("entry_tf", "M1")
            entry_thesis_ctx.setdefault("forecast_profile", {"timeframe": "M5", "step_bars": 2})
            entry_thesis_ctx.setdefault("forecast_timeframe", "M5")
            entry_thesis_ctx.setdefault("forecast_step_bars", 2)
            entry_thesis_ctx.setdefault("forecast_horizon", "10m")
            entry_thesis_ctx.setdefault("forecast_technical_only", True)

            tech_decision = evaluate_entry_techniques(
                entry_price=_tech_entry_price,
                side=_tech_side,
                pocket=_tech_pocket,
                strategy_tag=_tech_signal_tag,
                entry_thesis=entry_thesis_ctx,
                allow_candle=bool(entry_thesis_ctx.get("tech_allow_candle", False)),
            )
            if not tech_decision.allowed and not getattr(config, "TECH_FAILOPEN", True):
                continue

            entry_thesis_ctx["tech_score"] = round(tech_decision.score, 3) if tech_decision.score is not None else None
            entry_thesis_ctx["tech_coverage"] = (
                round(tech_decision.coverage, 3) if tech_decision.coverage is not None else None
            )
            entry_thesis_ctx["tech_entry"] = tech_decision.debug
            entry_thesis_ctx["tech_reason"] = tech_decision.reason
            entry_thesis_ctx["tech_decision_allowed"] = bool(tech_decision.allowed)
            _tech_tp_mult = max(
                0.2,
                min(2.0, float(getattr(tech_decision, "tp_mult", 1.0) or 1.0)),
            )
            entry_thesis_ctx["tech_tp_mult"] = round(_tech_tp_mult, 3)
            if isinstance(tp_price, (int, float)) and tp_price > 0 and _tech_entry_price > 0:
                _tp_gap = abs(float(tp_price) - float(_tech_entry_price))
                if _tp_gap > 0:
                    _tp_target = (
                        float(_tech_entry_price) + (_tp_gap * _tech_tp_mult)
                        if _tech_side == "long"
                        else float(_tech_entry_price) - (_tp_gap * _tech_tp_mult)
                    )
                    sl_price, tp_price = clamp_sl_tp(
                        price=float(_tech_entry_price),
                        sl=sl_price,
                        tp=round(_tp_target, 3),
                        is_buy=(_tech_side == "long"),
                    )
                    if isinstance(tp_price, (int, float)):
                        entry_thesis_ctx["tp_pips"] = round(
                            abs(float(tp_price) - float(_tech_entry_price)) / 0.01,
                            3,
                        )

            _tech_units_raw = locals().get("units")
            if isinstance(_tech_units_raw, (int, float)):
                _tech_units = int(round(abs(float(_tech_units_raw)) * tech_decision.size_mult))
                if _tech_units <= 0:
                    continue
                units = _tech_units if _tech_side == "long" else -_tech_units
                entry_thesis_ctx["entry_units_intent"] = abs(int(units))

            _tech_conf = locals().get("conf")
            if isinstance(_tech_conf, (int, float)):
                _tech_conf = float(_tech_conf)
                if tech_decision.score is not None:
                    if tech_decision.score >= 0:
                        _tech_conf += tech_decision.score * getattr(config, "TECH_CONF_BOOST", 0.0)
                    else:
                        _tech_conf += tech_decision.score * getattr(config, "TECH_CONF_PENALTY", 0.0)
                conf = _tech_conf


            res = await market_order(
                instrument="USD_JPY",
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                pocket=config.POCKET,
                client_order_id=client_id,
                strategy_tag=signal_tag,
                confidence=signal_conf,
                entry_thesis=entry_thesis,
            )
            _STRATEGY_LAST_TS[strategy_name] = time.time()
            LOG.info(
                "%s strat=%s sent units=%s side=%s price=%.3f sl=%.3f tp=%.3f conf=%.0f cap=%.2f multi=%.2f hist=%.3f(%s,n=%s) dyn=%.2f s_mult=%.2f dyn_score=%.2f dyn_n=%s reasons=%s res=%s",
                config.LOG_PREFIX,
                strategy_name,
                units,
                side,
                price,
                sl_price,
                tp_price,
                signal_conf,
                cap,
                multi_scale,
                hist_mult,
                hist_source,
                hist_n,
                dyn_mult,
                strategy_units_mult,
                dyn_score,
                dyn_trades,
                {**cap_reason, "tp_scale": round(tp_scale, 3)},
                res or "none",
            )




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

if __name__ == "__main__":
    asyncio.run(micro_multi_worker())
