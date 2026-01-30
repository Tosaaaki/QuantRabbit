from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

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
) -> tuple[Optional[float], Optional[float]]:
    side_key = str(side).lower()
    is_short = side_key in {"short", "sell"}

    rsi_score = None
    if rsi is not None:
        if is_short:
            rsi_score = _norm(rsi, 55.0, 70.0)
        else:
            rsi_score = _norm(45.0 - rsi, 0.0, 15.0)

    adx_revert = _norm(25.0 - adx, 0.0, 10.0) if adx is not None else None
    bbw_revert = _norm(0.22 - bbw, 0.0, 0.12) if bbw is not None else None

    vwap_revert = None
    if vwap_gap is not None:
        gap = vwap_gap if is_short else -vwap_gap
        vwap_revert = _norm(gap, 0.6, 2.4)

    revert_score = _weighted_score(
        [
            (rsi_score, 0.35),
            (adx_revert, 0.25),
            (bbw_revert, 0.25),
            (vwap_revert, 0.15),
        ]
    )

    adx_trend = _norm(adx, 18.0, 35.0) if adx is not None else None
    atr_trend = _norm(atr_pips, 6.0, 14.0) if atr_pips is not None else None
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
        vwap_trend = _norm(gap, 0.6, 2.6)

    trend_score = _weighted_score(
        [
            (adx_trend, 0.35),
            (atr_trend, 0.30),
            (ma_trend, 0.25),
            (vwap_trend, 0.10),
        ]
    )

    if range_active:
        if revert_score is not None:
            revert_score = _clamp01(revert_score + 0.15)
        if trend_score is not None:
            trend_score = _clamp01(trend_score - 0.15)

    return revert_score, trend_score


def _reentry_edge(adverse_pips: float, atr_pips: Optional[float]) -> float:
    if adverse_pips <= 0:
        return 0.0
    base = adverse_pips / max(atr_pips or 6.0, 0.1)
    return float(_clamp01((base - 0.8) / 1.4) or 0.0)


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
    return ReentryConfig(
        enabled=enabled,
        shadow=shadow,
        revert_min=_env_float(f"{name}_REENTRY_REVERT_MIN", 0.65),
        trend_min=_env_float(f"{name}_REENTRY_TREND_MIN", 0.60),
        trend_max=_env_float(f"{name}_REENTRY_TREND_MAX", 0.45),
        edge_min=_env_float(f"{name}_REENTRY_EDGE_MIN", 0.55),
        min_adverse_pips=_env_float(f"{name}_REENTRY_MIN_ADVERSE_PIPS", 2.5),
        min_adverse_atr=_env_float(f"{name}_REENTRY_MIN_ADVERSE_ATR", 1.0),
        log_interval_sec=_env_float(f"{name}_REENTRY_LOG_INTERVAL_SEC", 8.0),
        name=name,
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
    )
    edge = _reentry_edge(adverse_pips, atr_pips)

    action = None
    if revert_score is not None and trend_score is not None:
        if revert_score >= cfg.revert_min and trend_score <= cfg.trend_max:
            action = "hold"
        elif trend_score >= cfg.trend_min and edge >= cfg.edge_min:
            action = "exit_reentry"

    if action:
        tags = {
            "side": str(side),
            "revert": f"{revert_score:.2f}" if revert_score is not None else "na",
            "trend": f"{trend_score:.2f}" if trend_score is not None else "na",
            "edge": f"{edge:.2f}",
            "pnl": f"{pnl_pips:.2f}",
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
