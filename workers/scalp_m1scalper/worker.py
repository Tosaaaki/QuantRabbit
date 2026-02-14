"""M1Scalper dedicated worker with dynamic cap."""

from __future__ import annotations
from analysis.ma_projection import compute_adx_projection, compute_bbw_projection, compute_ma_projection, compute_rsi_projection
from analysis.ma_projection import score_ma_for_side
try:
    from analysis.pattern_stats import derive_pattern_signature
except ModuleNotFoundError:  # pragma: no cover - optional in slim deployments
    derive_pattern_signature = None  # type: ignore

import asyncio
import datetime
import hashlib
import logging
import time
from typing import Dict, Optional, Tuple

from autotune.scalp_trainer import AUTO_INTERVAL_SEC, start_background_autotune
from analysis.range_guard import detect_range_mode
from indicators.factor_cache import all_factors, get_candles_snapshot
from market_data import spread_monitor
from execution.order_manager import limit_order, market_order
from execution.risk_guard import allowed_lot, can_trade, clamp_sl_tp
from market_data import tick_window
from strategies.scalping.m1_scalper import M1Scalper
from utils.market_hours import is_market_open
from utils.oanda_account import get_account_snapshot, get_position_summary
from workers.common.dyn_cap import compute_cap
from workers.common.dynamic_alloc import load_strategy_profile
from workers.common import perf_guard, env_guard
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

_PATTERN_GATE_OPT_IN = env_bool("SCALP_M1SCALPER_PATTERN_GATE_OPT_IN", True)


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


def _candle_float(candle, *keys: str) -> Optional[float]:
    if isinstance(candle, dict):
        for key in keys:
            if key not in candle:
                continue
            raw = candle.get(key)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if value > 0.0:
                return value
        return None
    if isinstance(candle, (list, tuple)) and len(candle) >= 4:
        return _candle_float(
            {"open": candle[0], "high": candle[1], "low": candle[2], "close": candle[3]},
            *keys
        )
    return None


def _recent_low(candles, lookback: int) -> Optional[float]:
    if not candles or lookback <= 0:
        return None
    try:
        n = max(2, int(lookback))
    except (TypeError, ValueError):
        n = 2
    window = list(candles)[-n:] if len(candles) > n else list(candles)
    if not window:
        return None
    lows: list[float] = []
    for candle in window:
        low = _candle_float(candle, "l", "low")
        if low is None:
            continue
        lows.append(low)
    if not lows:
        return None
    return min(lows)


def _detect_usdjpy_setup_mode(side: str, price: float, fac_m1: dict) -> tuple[Optional[str], dict]:
    if not config.USDJPY_SETUP_GATING:
        return None, {}
    if side != "short":
        return None, {}
    if price <= 0.0 or price <= 100.0:
        return None, {}

    levels = _bb_levels(fac_m1)
    bb_mid = None
    if levels:
        _, mid, _, _, _ = levels
        bb_mid = mid
    else:
        return None, {}
    pullback_band = max(config.USDJPY_PULLBACK_BAND_PIPS, 0.0)
    pullback_ok = abs(price - bb_mid) <= pullback_band * _BB_PIP

    m1_lookback = max(2, int(config.USDJPY_BREAK_LOOKBACK_M1))
    m5_lookback = max(2, int(config.USDJPY_BREAK_LOOKBACK_M5))
    m1_candles = get_candles_snapshot("M1", limit=m1_lookback + 3)
    m5_candles = get_candles_snapshot("M5", limit=m5_lookback + 3)
    low_m1 = _recent_low(m1_candles, m1_lookback)
    low_m5 = _recent_low(m5_candles, m5_lookback)
    support_level = None
    if low_m1 is not None and low_m5 is not None:
        support_level = max(low_m1, low_m5)
    elif low_m1 is not None:
        support_level = low_m1
    elif low_m5 is not None:
        support_level = low_m5

    if support_level is not None:
        margin = config.USDJPY_BREAK_MARGIN_PIPS * _BB_PIP
        if price <= (support_level - margin):
            return (
                "breakdown",
                {
                    "support_level": round(support_level, 3),
                    "support_margin_pips": round(config.USDJPY_BREAK_MARGIN_PIPS, 3),
                    "lookback_m1": m1_lookback,
                    "lookback_m5": m5_lookback,
                },
            )
    if pullback_ok:
        return (
            "pullback",
            {
                "bb_mid": round(bb_mid, 3),
                "pullback_band_pips": round(pullback_band, 3),
                "distance_to_mid_pips": round(abs(price - bb_mid) / _BB_PIP, 3),
                "support_level": round(support_level, 3) if support_level is not None else None,
            },
        )
    return None, {}

BB_STYLE = "scalp"

LOG = logging.getLogger(__name__)

_LIMIT_ENTRY_ENABLED = os.getenv("M1SCALP_USE_LIMIT_ENTRY", "0").strip().lower() not in {"", "0", "false", "no"}
_LIMIT_ENTRY_TTL_SEC_DEFAULT = float(os.getenv("M1SCALP_LIMIT_TTL_SEC", "70") or 70.0)
_PENDING_LIMIT_UNTIL_TS: float = 0.0
_PENDING_LIMIT_ORDER_ID: Optional[str] = None


def _htf_trend_state(fac_h1: Dict) -> tuple[str, float, float] | None:
    adx = _bb_float(fac_h1.get("adx"))
    close = _bb_float(fac_h1.get("close"))
    ma20 = _bb_float(fac_h1.get("ma20")) or _bb_float(fac_h1.get("ema20"))
    if adx is None or close is None or ma20 is None or ma20 <= 0:
        return None
    gap_pips = (close - ma20) / _BB_PIP
    if abs(gap_pips) < config.HTF_GAP_PIPS or adx < config.HTF_ADX_MIN:
        return None
    trend_dir = "long" if gap_pips > 0 else "short"
    return trend_dir, gap_pips, adx



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
    ticks = tick_window.recent_ticks(seconds=6.0, limit=1)
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
    sanitized = "".join(ch.lower() for ch in tag if ch.isalnum())[:8] or "m1scalp"
    digest = hashlib.sha1(f"{ts_ms}-{tag}".encode("utf-8")).hexdigest()[:9]
    return f"qr-{ts_ms}-scalp-{sanitized}{digest}"


def _confidence_scale(conf: int) -> float:
    lo = config.CONFIDENCE_FLOOR
    hi = config.CONFIDENCE_CEIL
    if conf <= lo:
        return 0.5
    if conf >= hi:
        return 1.0
    span = (conf - lo) / max(1.0, hi - lo)
    return 0.5 + span * 0.5


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


def _resolve_strategy_tag(signal_tag: str, signal_side: str) -> str:
    override = str(config.STRATEGY_TAG_OVERRIDE or "").strip()
    if not override:
        return signal_tag
    if "{" not in override:
        return override
    try:
        rendered = override.format(tag=signal_tag, side=signal_side)
    except Exception:
        return signal_tag
    rendered = str(rendered or "").strip()
    return rendered or signal_tag


def _compute_cap(*args, **kwargs) -> Tuple[float, Dict[str, float]]:
    kwargs.setdefault("env_prefix", config.ENV_PREFIX)
    res = compute_cap(cap_min=config.CAP_MIN, cap_max=config.CAP_MAX, *args, **kwargs)
    return res.cap, res.reasons


async def scalp_m1_worker() -> None:
    global _PENDING_LIMIT_UNTIL_TS, _PENDING_LIMIT_ORDER_ID
    if not config.ENABLED:
        LOG.info("%s disabled", config.LOG_PREFIX)
        return
    LOG.info(
        "%s worker start (interval=%.1fs side_filter=%s tag_filter=%s strategy_override=%s)",
        config.LOG_PREFIX,
        config.LOOP_INTERVAL_SEC,
        config.SIDE_FILTER or "both",
        ",".join(sorted(config.SIGNAL_TAG_CONTAINS)) if config.SIGNAL_TAG_CONTAINS else "-",
        config.STRATEGY_TAG_OVERRIDE or "-",
    )
    if config.AUTOTUNE_ENABLED:
        start_background_autotune()
        LOG.info(
            "%s scalp_autotune enabled interval_sec=%s",
            config.LOG_PREFIX,
            AUTO_INTERVAL_SEC,
        )
    last_block_log = 0.0
    last_spread_log = 0.0
    last_cap_log = 0.0
    last_conf_log = 0.0

    while True:
        await asyncio.sleep(config.LOOP_INTERVAL_SEC)
        now_epoch = time.time()
        if _PENDING_LIMIT_UNTIL_TS and now_epoch >= _PENDING_LIMIT_UNTIL_TS:
            _PENDING_LIMIT_UNTIL_TS = 0.0
            _PENDING_LIMIT_ORDER_ID = None
        now = datetime.datetime.utcnow()
        if not is_market_open(now):
            continue
        if not can_trade(config.POCKET):
            continue
        if config.BLOCK_HOURS_ENABLED and now.hour in config.BLOCK_HOURS_UTC:
            now_mono = time.monotonic()
            if now_mono - last_block_log > 300.0:
                LOG.info(
                    "%s blocked by hour hour=%02d block_hours=%s",
                    config.LOG_PREFIX,
                    now.hour,
                    sorted(config.BLOCK_HOURS_UTC),
                )
                last_block_log = now_mono
            continue
        if _PENDING_LIMIT_UNTIL_TS and now_epoch < _PENDING_LIMIT_UNTIL_TS:
            LOG.debug(
                "%s skip: pending_limit order_id=%s until=%.0f",
                config.LOG_PREFIX,
                _PENDING_LIMIT_ORDER_ID or "-",
                _PENDING_LIMIT_UNTIL_TS,
            )
            continue
        blocked, remain, spread_state, spread_reason = spread_monitor.is_blocked()
        spread_pips = float(spread_state.get("spread_pips") or 0.0) if spread_state else 0.0
        spread_stale = bool(spread_state.get("stale")) if spread_state else False
        if blocked or spread_stale or spread_pips > config.MAX_SPREAD_PIPS:
            now_mono = time.monotonic()
            if now_mono - last_spread_log > 60.0:
                LOG.info(
                    "%s blocked by spread spread=%.2fp stale=%s remain=%ss reason=%s",
                    config.LOG_PREFIX,
                    spread_pips,
                    spread_stale,
                    remain,
                    spread_reason or "guard_active",
                )
                last_spread_log = now_mono
            continue

        factors = all_factors()
        fac_m1 = factors.get("M1") or {}
        fac_h1 = factors.get("H1") or {}
        fac_h4 = factors.get("H4") or {}
        range_ctx = detect_range_mode(fac_m1, fac_h4)
        range_score = 0.0
        try:
            range_score = float(range_ctx.score or 0.0)
        except Exception:
            range_score = 0.0
        fac_m1 = dict(fac_m1)
        fac_m1["range_active"] = bool(range_ctx.active)
        fac_m1["range_score"] = range_score
        fac_m1["range_reason"] = range_ctx.reason
        fac_m1["range_mode"] = range_ctx.mode
        if config.ALLOWED_REGIMES:
            regime = str(fac_m1.get("regime") or "").strip().lower()
            if regime not in config.ALLOWED_REGIMES:
                now_mono = time.monotonic()
                if now_mono - last_block_log > 300.0:
                    LOG.info(
                        "%s blocked by regime regime=%s allowed=%s",
                        config.LOG_PREFIX,
                        regime or "-",
                        sorted(config.ALLOWED_REGIMES),
                    )
                    last_block_log = now_mono
                continue
        perf = perf_monitor.snapshot()
        pf = None
        try:
            pf = float((perf.get(config.POCKET) or {}).get("pf"))
        except Exception:
            pf = None

        signal = M1Scalper.check(fac_m1)
        if not signal:
            continue
        signal_tag = str(signal.get("tag") or M1Scalper.name)
        signal_tag_l = signal_tag.lower()
        signal_action = str(signal.get("action") or "").upper()
        if signal_action not in {"OPEN_LONG", "OPEN_SHORT"}:
            continue
        signal_side = "long" if signal_action == "OPEN_LONG" else "short"
        if config.SIGNAL_TAG_CONTAINS and not any(
            token in signal_tag_l for token in config.SIGNAL_TAG_CONTAINS
        ):
            now_mono = time.monotonic()
            if now_mono - last_block_log > 120.0:
                LOG.info(
                    "%s tag_filter_block tag=%s allow=%s",
                    config.LOG_PREFIX,
                    signal_tag,
                    sorted(config.SIGNAL_TAG_CONTAINS),
                )
                last_block_log = now_mono
            continue
        if config.SIDE_FILTER and signal_side != config.SIDE_FILTER:
            now_mono = time.monotonic()
            if now_mono - last_block_log > 120.0:
                LOG.info(
                    "%s side_filter_block side=%s allow=%s tag=%s",
                    config.LOG_PREFIX,
                    signal_side,
                    config.SIDE_FILTER,
                    signal_tag,
                )
                last_block_log = now_mono
            continue
        conf_val = _to_confidence_0_100(signal.get("confidence", 0))
        if conf_val < config.CONFIDENCE_FLOOR:
            now_mono = time.monotonic()
            if now_mono - last_conf_log > 120.0:
                LOG.info(
                    "%s conf_block conf=%s floor=%s tag=%s",
                    config.LOG_PREFIX,
                    conf_val,
                    config.CONFIDENCE_FLOOR,
                    signal_tag,
                )
                last_conf_log = now_mono
            continue
        is_reversion = (
            "buy-dip" in signal_tag_l
            or "sell-rally" in signal_tag_l
            or "reversion" in signal_tag_l
        )
        if is_reversion and config.REVERSION_REQUIRE_STRONG_RANGE:
            range_mode = str(range_ctx.mode or "").strip().lower()
            range_mode_ok = (
                not config.REVERSION_ALLOWED_RANGE_MODES
                or range_mode in config.REVERSION_ALLOWED_RANGE_MODES
            )
            range_ready = bool(range_ctx.active) or range_score >= config.REVERSION_MIN_RANGE_SCORE
            adx_val = 0.0
            try:
                adx_val = float(fac_m1.get("adx") or 0.0)
            except Exception:
                adx_val = 0.0
            adx_ok = config.REVERSION_MAX_ADX <= 0.0 or adx_val <= config.REVERSION_MAX_ADX
            if not (range_mode_ok and range_ready and adx_ok):
                now_mono = time.monotonic()
                if now_mono - last_block_log > 120.0:
                    LOG.info(
                        "%s reversion_range_block tag=%s mode=%s active=%s score=%.3f adx=%.2f",
                        config.LOG_PREFIX,
                        signal_tag,
                        range_mode or "-",
                        bool(range_ctx.active),
                        range_score,
                        adx_val,
                    )
                    last_block_log = now_mono
                continue
        if is_reversion and not config.ALLOW_REVERSION:
            now_mono = time.monotonic()
            if now_mono - last_block_log > 120.0:
                LOG.info(
                    "%s reversion_disabled tag=%s",
                    config.LOG_PREFIX,
                    signal_tag,
                )
                last_block_log = now_mono
            continue
        if (not is_reversion) and (not config.ALLOW_TREND):
            now_mono = time.monotonic()
            if now_mono - last_block_log > 120.0:
                LOG.info(
                    "%s trend_disabled tag=%s",
                    config.LOG_PREFIX,
                    signal_tag,
                )
                last_block_log = now_mono
            continue
        if config.ENV_GUARD_ENABLED and is_reversion:
            ticks = tick_window.recent_ticks(
                seconds=max(config.ENV_RETURN_WINDOW_SEC, 5.0),
                limit=int(max(config.ENV_RETURN_WINDOW_SEC, 5.0) * 12),
            )
            allowed_env, env_reason = env_guard.mean_reversion_allowed(
                spread_p50_limit=config.ENV_SPREAD_P50_LIMIT,
                return_pips_limit=config.ENV_RETURN_PIPS_LIMIT,
                return_window_sec=config.ENV_RETURN_WINDOW_SEC,
                instant_move_limit=config.ENV_INSTANT_MOVE_LIMIT,
                tick_gap_ms_limit=config.ENV_TICK_GAP_MS_LIMIT,
                tick_gap_move_pips=config.ENV_TICK_GAP_MOVE_PIPS,
                ticks=ticks,
            )
            if not allowed_env:
                now_mono = time.monotonic()
                if now_mono - last_block_log > 120.0:
                    LOG.info(
                        "%s env_guard_block tag=%s reason=%s",
                        config.LOG_PREFIX,
                        signal_tag,
                        env_reason,
                    )
                    last_block_log = now_mono
                continue

        perf_tag = str(signal_tag or M1Scalper.name)
        perf_decision = perf_guard.is_allowed(perf_tag, config.POCKET, env_prefix=config.ENV_PREFIX)
        if not perf_decision.allowed:
            now_mono = time.monotonic()
            if now_mono - last_block_log > 120.0:
                LOG.info(
                    "%s perf_block tag=%s reason=%s",
                    config.LOG_PREFIX,
                    perf_tag,
                    perf_decision.reason,
                )
                last_block_log = now_mono
            continue
        dyn_profile: Dict[str, object] = {}
        if config.DYN_ALLOC_ENABLED:
            dyn_profile = load_strategy_profile(
                str(signal_tag),
                config.POCKET,
                path=config.DYN_ALLOC_PATH,
                ttl_sec=config.DYN_ALLOC_TTL_SEC,
            )
            if not bool(dyn_profile.get("found")):
                dyn_profile = load_strategy_profile(
                    M1Scalper.name,
                    config.POCKET,
                    path=config.DYN_ALLOC_PATH,
                    ttl_sec=config.DYN_ALLOC_TTL_SEC,
                )
            if config.DYN_ALLOC_LOSER_BLOCK and bool(dyn_profile.get("found")):
                dyn_trades = int(dyn_profile.get("trades", 0) or 0)
                dyn_score = float(dyn_profile.get("score", 0.0) or 0.0)
                if dyn_trades >= config.DYN_ALLOC_MIN_TRADES and dyn_score <= config.DYN_ALLOC_LOSER_SCORE:
                    now_mono = time.monotonic()
                    if now_mono - last_block_log > 120.0:
                        LOG.info(
                            "%s dyn_alloc_block tag=%s score=%.3f trades=%s",
                            config.LOG_PREFIX,
                            signal_tag,
                            dyn_score,
                            dyn_trades,
                        )
                        last_block_log = now_mono
                    continue

        snap = get_account_snapshot()
        equity = float(snap.nav or snap.balance or 0.0)

        balance = float(snap.balance or snap.nav or 0.0)
        free_ratio = float(snap.free_margin_ratio or 0.0) if snap.free_margin_ratio is not None else 0.0
        margin_usage_ratio = float(snap.margin_used or 0.0) / max(1.0, equity)
        if (
            free_ratio < config.MIN_FREE_MARGIN_RATIO_HARD
            or margin_usage_ratio > config.MARGIN_USAGE_HARD
        ):
            now_mono = time.monotonic()
            if now_mono - last_block_log > 30.0:
                LOG.warning(
                    "%s margin_hard_block tag=%s free_ratio=%.3f(min=%.3f) usage=%.3f(max=%.3f)",
                    config.LOG_PREFIX,
                    signal_tag,
                    free_ratio,
                    config.MIN_FREE_MARGIN_RATIO_HARD,
                    margin_usage_ratio,
                    config.MARGIN_USAGE_HARD,
                )
                last_block_log = now_mono
            continue
        try:
            atr_pips = float(fac_m1.get("atr_pips") or 0.0)
        except Exception:
            atr_pips = 0.0
        pos_bias = 0.0
        try:
            open_positions = snap.positions or {}
            scalp_pos = open_positions.get("scalp") or {}
            pos_bias = abs(float(scalp_pos.get("units", 0.0) or 0.0)) / max(1.0, float(snap.nav or 1.0))
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

        now_mono = time.monotonic()
        if (
            cap >= config.CAP_MAX * 0.85
            or free_ratio < 0.12
            or pos_bias > 0.25
            or range_ctx.active
        ) and now_mono - last_cap_log > 90.0:
            LOG.info(
                "%s cap=%.3f reasons=%s free_ratio=%.3f pos_bias=%.3f pf=%s range=%s",
                config.LOG_PREFIX,
                cap,
                cap_reason,
                free_ratio,
                pos_bias,
                pf,
                range_ctx.active,
            )
            last_cap_log = now_mono

        try:
            price = float(fac_m1.get("close") or 0.0)
        except Exception:
            price = 0.0
        # Prefer fresh bid/ask from tick cache for limit passivity checks.
        current_bid = None
        current_ask = None
        ticks_latest = tick_window.recent_ticks(seconds=15.0, limit=1)
        if ticks_latest:
            tick = ticks_latest[-1]
            current_bid = _bb_float(tick.get("bid"))
            current_ask = _bb_float(tick.get("ask"))
        price = _latest_mid(price)
        side = signal_side
        sl_pips = float(signal.get("sl_pips") or 0.0)
        tp_pips = float(signal.get("tp_pips") or 0.0)
        if price <= 0.0 or sl_pips <= 0.0:
            continue
        proj_flip = False
        proj_allow, proj_mult, proj_detail = _projection_decision(side, config.POCKET)
        if config.PROJ_FLIP_ENABLED and proj_detail:
            opp_side = "short" if side == "long" else "long"
            opp_allow, opp_mult, opp_detail = _projection_decision(opp_side, config.POCKET)
            if opp_detail and opp_allow:
                score = float(proj_detail.get("score", 0.0))
                opp_score = float(opp_detail.get("score", 0.0))
                if (
                    opp_score >= config.PROJ_FLIP_MIN_SCORE
                    and score <= config.PROJ_FLIP_MAX_SCORE
                    and (opp_score - score) >= config.PROJ_FLIP_MARGIN
                ):
                    side = opp_side
                    proj_flip = True
                    proj_allow, proj_mult, proj_detail = opp_allow, opp_mult, opp_detail
        if not proj_allow:
            continue
        if config.SIDE_FILTER and side != config.SIDE_FILTER:
            now_mono = time.monotonic()
            if now_mono - last_block_log > 120.0:
                LOG.info(
                    "%s side_filter_postproj_block side=%s allow=%s tag=%s",
                    config.LOG_PREFIX,
                    side,
                    config.SIDE_FILTER,
                    signal_tag,
                )
                last_block_log = now_mono
            continue
        htf = _htf_trend_state(fac_h1)
        if htf and config.HTF_BLOCK_COUNTER:
            htf_dir, htf_gap, htf_adx = htf
            if side != htf_dir:
                LOG.info(
                    "%s htf_block tag=%s side=%s h1_dir=%s gap=%.2fp adx=%.1f",
                    config.LOG_PREFIX,
                    signal_tag,
                    side,
                    htf_dir,
                    htf_gap,
                    htf_adx,
                )
                continue
        bb_style = "reversion" if is_reversion else BB_STYLE
        if not _bb_entry_allowed(bb_style, side, price, fac_m1, range_active=range_ctx.active):
            continue

        usdjpy_setup_mode: Optional[str] = None
        usdjpy_setup_detail: dict = {}
        usdjpy_setup_mult = 1.0
        if config.USDJPY_SETUP_GATING and side == "short":
            usdjpy_setup_mode, usdjpy_setup_detail = _detect_usdjpy_setup_mode(side, price, fac_m1)
            if not usdjpy_setup_mode:
                now_mono = time.monotonic()
                if now_mono - last_block_log > 120.0:
                    LOG.info(
                        "%s usdjpy_setup_block side=%s tag=%s price=%.3f",
                        config.LOG_PREFIX,
                        side,
                        signal_tag,
                        price,
                    )
                    last_block_log = now_mono
                continue
            if usdjpy_setup_mode == "pullback":
                usdjpy_setup_mult = config.USDJPY_PULLBACK_SIZE_MULT
            elif usdjpy_setup_mode == "breakdown":
                usdjpy_setup_mult = config.USDJPY_BREAK_SIZE_MULT

        tp_scale = 5.0 / max(1.0, tp_pips)
        tp_scale = max(0.45, min(1.15, tp_scale))
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

        conf_scale = _confidence_scale(conf_val)
        signal_tag = signal_tag or M1Scalper.name
        strategy_tag = _resolve_strategy_tag(signal_tag, side)
        entry_kind = "market"
        entry_ref_price = price
        entry_signal_price = None
        entry_tolerance_pips = None
        limit_ttl_sec = None
        if _LIMIT_ENTRY_ENABLED:
            entry_type = str(signal.get("entry_type") or "").strip().lower()
            entry_price = _bb_float(signal.get("entry_price"))
            entry_tol = _bb_float(signal.get("entry_tolerance_pips")) or 0.0
            ttl_raw = _bb_float(signal.get("limit_expiry_seconds"))
            ttl = float(ttl_raw) if ttl_raw is not None else _LIMIT_ENTRY_TTL_SEC_DEFAULT
            ttl = max(1.0, min(300.0, ttl))
            if (
                entry_type == "limit"
                and entry_price is not None
                and entry_price > 0
                and ttl > 0.0
                and current_bid is not None
                and current_ask is not None
            ):
                entry_signal_price = float(entry_price)
                entry_tolerance_pips = float(entry_tol)
                limit_ttl_sec = float(ttl)
                # If we're already close enough, take a market fill. Otherwise, stage a passive limit entry.
                if side == "long" and current_ask <= (entry_signal_price + entry_tolerance_pips * _BB_PIP):
                    entry_kind = "market"
                    entry_ref_price = price
                elif side == "short" and current_bid >= (entry_signal_price - entry_tolerance_pips * _BB_PIP):
                    entry_kind = "market"
                    entry_ref_price = price
                else:
                    entry_kind = "limit"
                    entry_ref_price = entry_signal_price
        long_units = 0.0
        short_units = 0.0
        try:
            long_units, short_units = get_position_summary("USD_JPY", timeout=3.0)
        except Exception:
            long_units, short_units = 0.0, 0.0
        lot = allowed_lot(
            float(snap.nav or 0.0),
            sl_pips,
            margin_available=float(snap.margin_available or 0.0),
            price=entry_ref_price,
            margin_rate=float(snap.margin_rate or 0.0),
            pocket=config.POCKET,
            side=side,
            open_long_units=long_units,
            open_short_units=short_units,
            strategy_tag=strategy_tag,
            fac_m1=fac_m1,
            fac_h4=fac_h4,
        )
        units_risk = int(round(lot * 100000))
        units = int(round(base_units * conf_scale))
        units = min(units, units_risk)
        units = int(round(units * cap))
        setup_size_mult = usdjpy_setup_mult
        dyn_mult = 1.0
        dyn_score = 0.0
        dyn_trades = 0
        if config.DYN_ALLOC_ENABLED and bool(dyn_profile.get("found")):
            dyn_mult = float(dyn_profile.get("lot_multiplier", 1.0) or 1.0)
            dyn_mult = max(config.DYN_ALLOC_MULT_MIN, min(config.DYN_ALLOC_MULT_MAX, dyn_mult))
            dyn_score = float(dyn_profile.get("score", 0.0) or 0.0)
            dyn_trades = int(dyn_profile.get("trades", 0) or 0)
        total_size_mult = dyn_mult * setup_size_mult
        if total_size_mult != 1.0:
            units = int(round(units * total_size_mult))
        if units < config.MIN_UNITS:
            continue
        if side == "short":
            units = -abs(units)

        if side == "long":
            sl_price = round(entry_ref_price - sl_pips * 0.01, 3)
            tp_price = round(entry_ref_price + tp_pips * 0.01, 3) if tp_pips > 0 else None
        else:
            sl_price = round(entry_ref_price + sl_pips * 0.01, 3)
            tp_price = round(entry_ref_price - tp_pips * 0.01, 3) if tp_pips > 0 else None

        sl_price, tp_price = clamp_sl_tp(
            price=entry_ref_price,
            sl=sl_price,
            tp=tp_price,
            is_buy=side == "long",
        )
        client_id = _client_order_id(strategy_tag)
        entry_thesis = {
            "strategy_tag": strategy_tag,
            "source_signal_tag": signal_tag,
            "env_prefix": config.ENV_PREFIX,
            "pattern_gate_opt_in": bool(_PATTERN_GATE_OPT_IN),
            "signal_side": signal_side,
            "exec_side": side,
            "confidence": conf_val,
            "entry_probability": round(
                _to_probability(signal.get("entry_probability"), conf_val / 100.0),
                3,
            ),
            "sl_pips": round(sl_pips, 2),
            "tp_pips": round(tp_pips, 2),
            "hard_stop_pips": round(sl_pips, 2),
            "fast_cut_pips": signal.get("fast_cut_pips"),
            "fast_cut_time_sec": signal.get("fast_cut_time_sec"),
        }
        if config.DYN_ALLOC_ENABLED and bool(dyn_profile.get("found")):
            entry_thesis["dynamic_alloc"] = {
                "strategy_key": dyn_profile.get("strategy_key"),
                "score": round(dyn_score, 3),
                "trades": dyn_trades,
                "lot_multiplier": round(dyn_mult, 3),
            }
        if config.USDJPY_SETUP_GATING and usdjpy_setup_mode:
            entry_thesis["usdjpy_setup_mode"] = usdjpy_setup_mode
            if usdjpy_setup_detail:
                entry_thesis["usdjpy_setup_detail"] = usdjpy_setup_detail
            entry_thesis["setup_mult"] = round(setup_size_mult, 4)
        if derive_pattern_signature is not None and isinstance(entry_thesis, dict):
            pattern_tag, pattern_meta = derive_pattern_signature(
                fac_m1, action="OPEN_LONG" if units > 0 else "OPEN_SHORT"
            )
            if pattern_tag:
                entry_thesis["pattern_tag"] = pattern_tag
            if pattern_meta:
                entry_thesis["pattern_meta"] = pattern_meta

        if proj_flip:
            entry_thesis["projection_flip"] = True
        if proj_detail:
            entry_thesis["projection"] = proj_detail
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
        if entry_kind == "limit" and limit_ttl_sec is not None:
            entry_thesis["entry_type"] = "limit"
            entry_thesis["entry_price"] = round(float(entry_ref_price), 3)
            if entry_tolerance_pips is not None:
                entry_thesis["entry_tolerance_pips"] = round(float(entry_tolerance_pips), 2)
            entry_thesis["limit_ttl_sec"] = round(max(1.0, limit_ttl_sec), 1)
            trade_id, order_id = await limit_order(
                instrument="USD_JPY",
                units=units,
                price=float(entry_ref_price),
                sl_price=sl_price,
                tp_price=tp_price,
                pocket=config.POCKET,
                current_bid=float(current_bid) if current_bid is not None else None,
                current_ask=float(current_ask) if current_ask is not None else None,
                client_order_id=client_id,
                ttl_ms=max(1.0, limit_ttl_sec) * 1000.0,
                entry_thesis=entry_thesis,
            )
            if order_id and not trade_id:
                _PENDING_LIMIT_ORDER_ID = order_id
                _PENDING_LIMIT_UNTIL_TS = time.time() + max(1.0, limit_ttl_sec)
            LOG.info(
                "%s sent(limit) units=%s side=%s ref=%.3f mid=%.3f sl=%.3f tp=%s conf=%.0f cap=%.2f dyn=%.2f setup=%s setup_mult=%.2f dyn_score=%.2f dyn_n=%s reasons=%s trade_id=%s order_id=%s",
                    config.LOG_PREFIX,
                    units,
                    side,
                    entry_ref_price,
                price,
                sl_price,
                    f"{tp_price:.3f}" if tp_price is not None else "NA",
                    conf_val,
                    cap,
                    dyn_mult,
                    usdjpy_setup_mode or "none",
                    setup_size_mult,
                    dyn_score,
                dyn_trades,
                {**cap_reason, "tp_scale": round(tp_scale, 3)},
                trade_id or "none",
                order_id or "none",
            )
        else:
            res = await market_order(
                instrument="USD_JPY",
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                pocket=config.POCKET,
                client_order_id=client_id,
                strategy_tag=strategy_tag,
                confidence=conf_val,
                entry_thesis=entry_thesis,
            )
            LOG.info(
                "%s sent units=%s side=%s price=%.3f sl=%.3f tp=%.3f conf=%.0f cap=%.2f dyn=%.2f setup=%s setup_mult=%.2f dyn_score=%.2f dyn_n=%s reasons=%s res=%s",
                config.LOG_PREFIX,
                units,
                side,
                price,
                sl_price,
                tp_price,
                conf_val,
                cap,
                dyn_mult,
                usdjpy_setup_mode or "none",
                setup_size_mult,
                dyn_score,
                dyn_trades,
                {**cap_reason, "tp_scale": round(tp_scale, 3)},
                res if res else "none",
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

if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    asyncio.run(scalp_m1_worker())
