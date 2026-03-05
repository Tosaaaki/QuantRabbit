from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from . import config as strategy_config
from utils.metrics_logger import log_metric
from indicators.factor_cache import all_factors

@dataclass
class ReentryConfig:
    enabled: bool
    shadow: bool
    revert_min: float
    trend_min: float
    trend_max: float
    edge_min: float
    min_adverse_pips: float
    min_adverse_atr: float
    log_interval_sec: float
    name: str
    atr_low_pips: float
    atr_high_pips: float
    revert_min_low_atr: float
    revert_min_high_atr: float
    trend_min_low_atr: float
    trend_min_high_atr: float
    trend_max_low_atr: float
    trend_max_high_atr: float
    edge_min_low_atr: float
    edge_min_high_atr: float
    range_revert_bonus_low_atr: float
    range_revert_bonus_high_atr: float
    range_trend_penalty_low_atr: float
    range_trend_penalty_high_atr: float
    revert_weights_low_atr: Tuple[float, float, float, float]
    revert_weights_high_atr: Tuple[float, float, float, float]
    trend_weights_low_atr: Tuple[float, float, float, float]
    trend_weights_high_atr: Tuple[float, float, float, float]
    rsi_short_min: float
    rsi_short_max: float
    rsi_long_neutral: float
    rsi_long_extreme: float
    adx_revert_neutral: float
    adx_revert_extreme: float
    bbw_revert_neutral: float
    bbw_revert_extreme: float
    vwap_revert_min: float
    vwap_revert_max: float
    adx_trend_min: float
    adx_trend_max: float
    atr_trend_min: float
    atr_trend_max: float
    vwap_trend_min: float
    vwap_trend_max: float
    edge_base_adverse_atr: float
    edge_scale_atr: float
    edge_atr_fallback: float

@dataclass
class ReentryDecision:
    action: Optional[str]
    shadow: bool
    revert_score: Optional[float]
    trend_score: Optional[float]
    edge: float
    min_adverse: float
    enabled: bool

_LAST_LOG_TS: dict[str, float] = {}

def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes"}

def _env_bool_opt(name: str) -> Optional[bool]:
    raw = os.getenv(name)
    if raw is None:
        return None
    return str(raw).strip().lower() in {"1", "true", "yes"}

def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default

def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp01(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _norm(value: Optional[float], low: float, high: float) -> Optional[float]:
    if value is None:
        return None
    if high <= low:
        return 0.0
    return _clamp01((float(value) - low) / (high - low))


def _mix(low: float, high: float, ratio: float) -> float:
    return float(low) + (float(high) - float(low)) * float(ratio)


def _atr_ratio(*, atr_pips: Optional[float], atr_low_pips: float, atr_high_pips: float) -> float:
    ratio = _norm(atr_pips, atr_low_pips, atr_high_pips)
    if ratio is None:
        return 0.5
    return float(ratio)


def _blend_by_atr(
    *,
    atr_pips: Optional[float],
    atr_low_pips: float,
    atr_high_pips: float,
    low_value: float,
    high_value: float,
) -> float:
    ratio = _atr_ratio(atr_pips=atr_pips, atr_low_pips=atr_low_pips, atr_high_pips=atr_high_pips)
    return _mix(low_value, high_value, ratio)


def _weighted_score(items: Sequence[tuple[Optional[float], float]]) -> Optional[float]:
    total = 0.0
    weight = 0.0
    for value, w in items:
        if value is None:
            continue
        total += float(value) * float(w)
        weight += float(w)
    if weight <= 0.0:
        return None
    return _clamp01(total / weight)


def _reentry_scores(
    *,
    side: str,
    rsi: Optional[float],
    adx: Optional[float],
    atr_pips: Optional[float],
    bbw: Optional[float],
    vwap_gap: Optional[float],
    ma_pair: Optional[Tuple[float, float]],
    range_active: bool,
    cfg: ReentryConfig,
) -> tuple[Optional[float], Optional[float]]:
    side_key = str(side).lower()
    is_short = side_key in {"short", "sell"}
    atr_ratio = _atr_ratio(
        atr_pips=atr_pips,
        atr_low_pips=cfg.atr_low_pips,
        atr_high_pips=cfg.atr_high_pips,
    )

    rsi_score = None
    if rsi is not None:
        if is_short:
            rsi_score = _norm(rsi, cfg.rsi_short_min, cfg.rsi_short_max)
        else:
            rsi_score = _norm(
                cfg.rsi_long_neutral - rsi,
                0.0,
                cfg.rsi_long_neutral - cfg.rsi_long_extreme,
            )

    adx_revert = (
        _norm(
            cfg.adx_revert_neutral - adx,
            0.0,
            cfg.adx_revert_neutral - cfg.adx_revert_extreme,
        )
        if adx is not None
        else None
    )
    bbw_revert = (
        _norm(
            cfg.bbw_revert_neutral - bbw,
            0.0,
            cfg.bbw_revert_neutral - cfg.bbw_revert_extreme,
        )
        if bbw is not None
        else None
    )

    vwap_revert = None
    if vwap_gap is not None:
        gap = vwap_gap if is_short else -vwap_gap
        vwap_revert = _norm(gap, cfg.vwap_revert_min, cfg.vwap_revert_max)

    revert_weights = (
        _mix(cfg.revert_weights_low_atr[0], cfg.revert_weights_high_atr[0], atr_ratio),
        _mix(cfg.revert_weights_low_atr[1], cfg.revert_weights_high_atr[1], atr_ratio),
        _mix(cfg.revert_weights_low_atr[2], cfg.revert_weights_high_atr[2], atr_ratio),
        _mix(cfg.revert_weights_low_atr[3], cfg.revert_weights_high_atr[3], atr_ratio),
    )

    revert_score = _weighted_score(
        [
            (rsi_score, revert_weights[0]),
            (adx_revert, revert_weights[1]),
            (bbw_revert, revert_weights[2]),
            (vwap_revert, revert_weights[3]),
        ]
    )

    adx_trend = _norm(adx, cfg.adx_trend_min, cfg.adx_trend_max) if adx is not None else None
    atr_trend = _norm(atr_pips, cfg.atr_trend_min, cfg.atr_trend_max) if atr_pips is not None else None
    ma_trend = None
    if ma_pair is not None:
        ma10, ma20 = ma_pair
        if is_short:
            ma_trend = 1.0 if ma10 > ma20 else 0.0
        else:
            ma_trend = 1.0 if ma10 < ma20 else 0.0
    vwap_trend = None
    if vwap_gap is not None:
        gap = vwap_gap if is_short else -vwap_gap
        vwap_trend = _norm(gap, cfg.vwap_trend_min, cfg.vwap_trend_max)

    trend_weights = (
        _mix(cfg.trend_weights_low_atr[0], cfg.trend_weights_high_atr[0], atr_ratio),
        _mix(cfg.trend_weights_low_atr[1], cfg.trend_weights_high_atr[1], atr_ratio),
        _mix(cfg.trend_weights_low_atr[2], cfg.trend_weights_high_atr[2], atr_ratio),
        _mix(cfg.trend_weights_low_atr[3], cfg.trend_weights_high_atr[3], atr_ratio),
    )

    trend_score = _weighted_score(
        [
            (adx_trend, trend_weights[0]),
            (atr_trend, trend_weights[1]),
            (ma_trend, trend_weights[2]),
            (vwap_trend, trend_weights[3]),
        ]
    )

    if range_active:
        revert_bonus = _mix(
            cfg.range_revert_bonus_low_atr,
            cfg.range_revert_bonus_high_atr,
            atr_ratio,
        )
        trend_penalty = _mix(
            cfg.range_trend_penalty_low_atr,
            cfg.range_trend_penalty_high_atr,
            atr_ratio,
        )
        if revert_score is not None:
            revert_score = _clamp01(revert_score + revert_bonus)
        if trend_score is not None:
            trend_score = _clamp01(trend_score - trend_penalty)

    return revert_score, trend_score

def _reentry_edge(adverse_pips: float, atr_pips: Optional[float], cfg: ReentryConfig) -> float:
    if adverse_pips <= 0:
        return 0.0
    atr_base = atr_pips if atr_pips is not None else cfg.edge_atr_fallback
    base = adverse_pips / max(atr_base, 0.1)
    edge_scale = max(abs(cfg.edge_scale_atr), 1e-6)
    return float(_clamp01((base - cfg.edge_base_adverse_atr) / edge_scale) or 0.0)

def _load_config(prefix: str) -> ReentryConfig:
    name = prefix.strip().upper()
    enabled = _env_bool_opt(f"{name}_REENTRY_ENABLE")
    shadow = _env_bool_opt(f"{name}_REENTRY_SHADOW")
    if enabled is None:
        enabled = _env_bool_opt("REENTRY_ENABLE_ALL")
    if enabled is None:
        enabled = False
    if shadow is None:
        shadow = _env_bool_opt("REENTRY_SHADOW_ALL")
    if shadow is None:
        shadow = True
    base_revert_min = _env_float(f"{name}_REENTRY_REVERT_MIN", strategy_config.REENTRY_REVERT_MIN_LOW_ATR)
    base_trend_min = _env_float(f"{name}_REENTRY_TREND_MIN", strategy_config.REENTRY_TREND_MIN_LOW_ATR)
    base_trend_max = _env_float(f"{name}_REENTRY_TREND_MAX", strategy_config.REENTRY_TREND_MAX_LOW_ATR)
    base_edge_min = _env_float(f"{name}_REENTRY_EDGE_MIN", strategy_config.REENTRY_EDGE_MIN_LOW_ATR)
    legacy_revert_min_set = os.getenv(f"{name}_REENTRY_REVERT_MIN") is not None
    legacy_trend_min_set = os.getenv(f"{name}_REENTRY_TREND_MIN") is not None
    legacy_trend_max_set = os.getenv(f"{name}_REENTRY_TREND_MAX") is not None
    legacy_edge_min_set = os.getenv(f"{name}_REENTRY_EDGE_MIN") is not None

    def _prefixed_float(var: str, default: float) -> float:
        return _env_float(f"{name}_{var}", default)

    return ReentryConfig(
        enabled=enabled,
        shadow=shadow,
        revert_min=base_revert_min,
        trend_min=base_trend_min,
        trend_max=base_trend_max,
        edge_min=base_edge_min,
        min_adverse_pips=_env_float(f"{name}_REENTRY_MIN_ADVERSE_PIPS", 2.5),
        min_adverse_atr=_env_float(f"{name}_REENTRY_MIN_ADVERSE_ATR", 1.0),
        log_interval_sec=_env_float(f"{name}_REENTRY_LOG_INTERVAL_SEC", 8.0),
        name=name,
        atr_low_pips=_prefixed_float("REENTRY_ATR_LOW_PIPS", strategy_config.REENTRY_ATR_LOW_PIPS),
        atr_high_pips=_prefixed_float("REENTRY_ATR_HIGH_PIPS", strategy_config.REENTRY_ATR_HIGH_PIPS),
        revert_min_low_atr=_prefixed_float("REENTRY_REVERT_MIN_ATR_LOW", base_revert_min),
        revert_min_high_atr=_prefixed_float(
            "REENTRY_REVERT_MIN_ATR_HIGH",
            base_revert_min if legacy_revert_min_set else strategy_config.REENTRY_REVERT_MIN_HIGH_ATR,
        ),
        trend_min_low_atr=_prefixed_float("REENTRY_TREND_MIN_ATR_LOW", base_trend_min),
        trend_min_high_atr=_prefixed_float(
            "REENTRY_TREND_MIN_ATR_HIGH",
            base_trend_min if legacy_trend_min_set else strategy_config.REENTRY_TREND_MIN_HIGH_ATR,
        ),
        trend_max_low_atr=_prefixed_float("REENTRY_TREND_MAX_ATR_LOW", base_trend_max),
        trend_max_high_atr=_prefixed_float(
            "REENTRY_TREND_MAX_ATR_HIGH",
            base_trend_max if legacy_trend_max_set else strategy_config.REENTRY_TREND_MAX_HIGH_ATR,
        ),
        edge_min_low_atr=_prefixed_float("REENTRY_EDGE_MIN_ATR_LOW", base_edge_min),
        edge_min_high_atr=_prefixed_float(
            "REENTRY_EDGE_MIN_ATR_HIGH",
            base_edge_min if legacy_edge_min_set else strategy_config.REENTRY_EDGE_MIN_HIGH_ATR,
        ),
        range_revert_bonus_low_atr=_prefixed_float(
            "REENTRY_RANGE_REVERT_BONUS_LOW_ATR",
            strategy_config.REENTRY_RANGE_REVERT_BONUS_LOW_ATR,
        ),
        range_revert_bonus_high_atr=_prefixed_float(
            "REENTRY_RANGE_REVERT_BONUS_HIGH_ATR",
            strategy_config.REENTRY_RANGE_REVERT_BONUS_HIGH_ATR,
        ),
        range_trend_penalty_low_atr=_prefixed_float(
            "REENTRY_RANGE_TREND_PENALTY_LOW_ATR",
            strategy_config.REENTRY_RANGE_TREND_PENALTY_LOW_ATR,
        ),
        range_trend_penalty_high_atr=_prefixed_float(
            "REENTRY_RANGE_TREND_PENALTY_HIGH_ATR",
            strategy_config.REENTRY_RANGE_TREND_PENALTY_HIGH_ATR,
        ),
        revert_weights_low_atr=(
            _prefixed_float("REENTRY_REVERT_WEIGHT_RSI_LOW_ATR", strategy_config.REENTRY_REVERT_WEIGHT_RSI_LOW_ATR),
            _prefixed_float("REENTRY_REVERT_WEIGHT_ADX_LOW_ATR", strategy_config.REENTRY_REVERT_WEIGHT_ADX_LOW_ATR),
            _prefixed_float("REENTRY_REVERT_WEIGHT_BBW_LOW_ATR", strategy_config.REENTRY_REVERT_WEIGHT_BBW_LOW_ATR),
            _prefixed_float("REENTRY_REVERT_WEIGHT_VWAP_LOW_ATR", strategy_config.REENTRY_REVERT_WEIGHT_VWAP_LOW_ATR),
        ),
        revert_weights_high_atr=(
            _prefixed_float("REENTRY_REVERT_WEIGHT_RSI_HIGH_ATR", strategy_config.REENTRY_REVERT_WEIGHT_RSI_HIGH_ATR),
            _prefixed_float("REENTRY_REVERT_WEIGHT_ADX_HIGH_ATR", strategy_config.REENTRY_REVERT_WEIGHT_ADX_HIGH_ATR),
            _prefixed_float("REENTRY_REVERT_WEIGHT_BBW_HIGH_ATR", strategy_config.REENTRY_REVERT_WEIGHT_BBW_HIGH_ATR),
            _prefixed_float("REENTRY_REVERT_WEIGHT_VWAP_HIGH_ATR", strategy_config.REENTRY_REVERT_WEIGHT_VWAP_HIGH_ATR),
        ),
        trend_weights_low_atr=(
            _prefixed_float("REENTRY_TREND_WEIGHT_ADX_LOW_ATR", strategy_config.REENTRY_TREND_WEIGHT_ADX_LOW_ATR),
            _prefixed_float("REENTRY_TREND_WEIGHT_ATR_LOW_ATR", strategy_config.REENTRY_TREND_WEIGHT_ATR_LOW_ATR),
            _prefixed_float("REENTRY_TREND_WEIGHT_MA_LOW_ATR", strategy_config.REENTRY_TREND_WEIGHT_MA_LOW_ATR),
            _prefixed_float("REENTRY_TREND_WEIGHT_VWAP_LOW_ATR", strategy_config.REENTRY_TREND_WEIGHT_VWAP_LOW_ATR),
        ),
        trend_weights_high_atr=(
            _prefixed_float("REENTRY_TREND_WEIGHT_ADX_HIGH_ATR", strategy_config.REENTRY_TREND_WEIGHT_ADX_HIGH_ATR),
            _prefixed_float("REENTRY_TREND_WEIGHT_ATR_HIGH_ATR", strategy_config.REENTRY_TREND_WEIGHT_ATR_HIGH_ATR),
            _prefixed_float("REENTRY_TREND_WEIGHT_MA_HIGH_ATR", strategy_config.REENTRY_TREND_WEIGHT_MA_HIGH_ATR),
            _prefixed_float("REENTRY_TREND_WEIGHT_VWAP_HIGH_ATR", strategy_config.REENTRY_TREND_WEIGHT_VWAP_HIGH_ATR),
        ),
        rsi_short_min=_prefixed_float("REENTRY_RSI_SHORT_MIN", strategy_config.REENTRY_RSI_SHORT_MIN),
        rsi_short_max=_prefixed_float("REENTRY_RSI_SHORT_MAX", strategy_config.REENTRY_RSI_SHORT_MAX),
        rsi_long_neutral=_prefixed_float("REENTRY_RSI_LONG_NEUTRAL", strategy_config.REENTRY_RSI_LONG_NEUTRAL),
        rsi_long_extreme=_prefixed_float("REENTRY_RSI_LONG_EXTREME", strategy_config.REENTRY_RSI_LONG_EXTREME),
        adx_revert_neutral=_prefixed_float("REENTRY_ADX_REVERT_NEUTRAL", strategy_config.REENTRY_ADX_REVERT_NEUTRAL),
        adx_revert_extreme=_prefixed_float("REENTRY_ADX_REVERT_EXTREME", strategy_config.REENTRY_ADX_REVERT_EXTREME),
        bbw_revert_neutral=_prefixed_float("REENTRY_BBW_REVERT_NEUTRAL", strategy_config.REENTRY_BBW_REVERT_NEUTRAL),
        bbw_revert_extreme=_prefixed_float("REENTRY_BBW_REVERT_EXTREME", strategy_config.REENTRY_BBW_REVERT_EXTREME),
        vwap_revert_min=_prefixed_float("REENTRY_VWAP_REVERT_MIN", strategy_config.REENTRY_VWAP_REVERT_MIN),
        vwap_revert_max=_prefixed_float("REENTRY_VWAP_REVERT_MAX", strategy_config.REENTRY_VWAP_REVERT_MAX),
        adx_trend_min=_prefixed_float("REENTRY_ADX_TREND_MIN", strategy_config.REENTRY_ADX_TREND_MIN),
        adx_trend_max=_prefixed_float("REENTRY_ADX_TREND_MAX", strategy_config.REENTRY_ADX_TREND_MAX),
        atr_trend_min=_prefixed_float("REENTRY_ATR_TREND_MIN", strategy_config.REENTRY_ATR_TREND_MIN),
        atr_trend_max=_prefixed_float("REENTRY_ATR_TREND_MAX", strategy_config.REENTRY_ATR_TREND_MAX),
        vwap_trend_min=_prefixed_float("REENTRY_VWAP_TREND_MIN", strategy_config.REENTRY_VWAP_TREND_MIN),
        vwap_trend_max=_prefixed_float("REENTRY_VWAP_TREND_MAX", strategy_config.REENTRY_VWAP_TREND_MAX),
        edge_base_adverse_atr=_prefixed_float(
            "REENTRY_EDGE_BASE_ADVERSE_ATR",
            strategy_config.REENTRY_EDGE_BASE_ADVERSE_ATR,
        ),
        edge_scale_atr=_prefixed_float("REENTRY_EDGE_SCALE_ATR", strategy_config.REENTRY_EDGE_SCALE_ATR),
        edge_atr_fallback=_prefixed_float("REENTRY_EDGE_ATR_FALLBACK", strategy_config.REENTRY_EDGE_ATR_FALLBACK),
    )

def _log_decision(prefix: str, decision: str, tags: dict, interval_sec: float) -> None:
    now = time.monotonic()
    last = _LAST_LOG_TS.get(prefix, 0.0)
    if now - last < interval_sec:
        return
    _LAST_LOG_TS[prefix] = now
    payload = dict(tags)
    payload["decision"] = decision
    payload["name"] = prefix
    log_metric("reentry_decision", 1.0, tags=payload)

def decide_reentry_from_factors(
    *,
    prefix: str,
    side: str,
    pnl_pips: float,
    tf: str,
    range_active: bool,
    log_tags: Optional[dict] = None,
    factors: Optional[dict] = None,
) -> ReentryDecision:
    fac = factors or (all_factors().get(tf) or {})
    rsi = _safe_float(fac.get("rsi"))
    adx = _safe_float(fac.get("adx"))
    atr_pips = _safe_float(fac.get("atr_pips"))
    bbw = _safe_float(fac.get("bbw"))
    vwap_gap = _safe_float(fac.get("vwap_gap"))
    ma10 = _safe_float(fac.get("ma10"))
    ma20 = _safe_float(fac.get("ma20"))
    ma_pair = (ma10, ma20) if ma10 is not None and ma20 is not None else None
    return decide_reentry(
        prefix=prefix,
        side=side,
        pnl_pips=pnl_pips,
        rsi=rsi,
        adx=adx,
        atr_pips=atr_pips,
        bbw=bbw,
        vwap_gap=vwap_gap,
        ma_pair=ma_pair,
        range_active=range_active,
        log_tags=log_tags,
    )

def decide_reentry(
    *,
    prefix: str,
    side: str,
    pnl_pips: float,
    rsi: Optional[float],
    adx: Optional[float],
    atr_pips: Optional[float],
    bbw: Optional[float],
    vwap_gap: Optional[float],
    ma_pair: Optional[Tuple[float, float]],
    range_active: bool,
    log_tags: Optional[dict] = None,
) -> ReentryDecision:
    cfg = _load_config(prefix)
    if not cfg.enabled:
        return ReentryDecision(
            action=None,
            shadow=cfg.shadow,
            revert_score=None,
            trend_score=None,
            edge=0.0,
            min_adverse=cfg.min_adverse_pips,
            enabled=False,
        )

    adverse_pips = abs(float(pnl_pips))
    min_adverse = cfg.min_adverse_pips
    if atr_pips is not None:
        min_adverse = max(min_adverse, atr_pips * cfg.min_adverse_atr)
    if adverse_pips < min_adverse:
        return ReentryDecision(
            action=None,
            shadow=cfg.shadow,
            revert_score=None,
            trend_score=None,
            edge=0.0,
            min_adverse=min_adverse,
            enabled=True,
        )

    revert_score, trend_score = _reentry_scores(
        side=side,
        rsi=rsi,
        adx=adx,
        atr_pips=atr_pips,
        bbw=bbw,
        vwap_gap=vwap_gap,
        ma_pair=ma_pair,
        range_active=range_active,
        cfg=cfg,
    )
    edge = _reentry_edge(adverse_pips, atr_pips, cfg)
    revert_min = _blend_by_atr(
        atr_pips=atr_pips,
        atr_low_pips=cfg.atr_low_pips,
        atr_high_pips=cfg.atr_high_pips,
        low_value=cfg.revert_min_low_atr,
        high_value=cfg.revert_min_high_atr,
    )
    trend_min = _blend_by_atr(
        atr_pips=atr_pips,
        atr_low_pips=cfg.atr_low_pips,
        atr_high_pips=cfg.atr_high_pips,
        low_value=cfg.trend_min_low_atr,
        high_value=cfg.trend_min_high_atr,
    )
    trend_max = _blend_by_atr(
        atr_pips=atr_pips,
        atr_low_pips=cfg.atr_low_pips,
        atr_high_pips=cfg.atr_high_pips,
        low_value=cfg.trend_max_low_atr,
        high_value=cfg.trend_max_high_atr,
    )
    edge_min = _blend_by_atr(
        atr_pips=atr_pips,
        atr_low_pips=cfg.atr_low_pips,
        atr_high_pips=cfg.atr_high_pips,
        low_value=cfg.edge_min_low_atr,
        high_value=cfg.edge_min_high_atr,
    )

    action = None
    if revert_score is not None and trend_score is not None:
        if revert_score >= revert_min and trend_score <= trend_max:
            action = "hold"
        elif trend_score >= trend_min and edge >= edge_min:
            action = "exit_reentry"

    if action:
        tags = {
            "side": str(side),
            "revert": f"{revert_score:.2f}" if revert_score is not None else "na",
            "trend": f"{trend_score:.2f}" if trend_score is not None else "na",
            "edge": f"{edge:.2f}",
            "pnl": f"{pnl_pips:.2f}",
            "revert_min": f"{revert_min:.2f}",
            "trend_min": f"{trend_min:.2f}",
            "trend_max": f"{trend_max:.2f}",
            "edge_min": f"{edge_min:.2f}",
        }
        if log_tags:
            tags.update(log_tags)
        _log_decision(cfg.name, action, tags, cfg.log_interval_sec)

    return ReentryDecision(
        action=action,
        shadow=cfg.shadow,
        revert_score=revert_score,
        trend_score=trend_score,
        edge=edge,
        min_adverse=min_adverse,
        enabled=True,
    )


def test_reentry_thresholds_change_continuously_with_atr() -> None:
    low = _blend_by_atr(
        atr_pips=6.0,
        atr_low_pips=6.0,
        atr_high_pips=14.0,
        low_value=0.55,
        high_value=0.75,
    )
    mid = _blend_by_atr(
        atr_pips=10.0,
        atr_low_pips=6.0,
        atr_high_pips=14.0,
        low_value=0.55,
        high_value=0.75,
    )
    high = _blend_by_atr(
        atr_pips=14.0,
        atr_low_pips=6.0,
        atr_high_pips=14.0,
        low_value=0.55,
        high_value=0.75,
    )
    assert low == 0.55
    assert mid == 0.65
    assert high == 0.75
