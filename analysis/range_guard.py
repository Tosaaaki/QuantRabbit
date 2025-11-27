"""
analysis.range_guard
~~~~~~~~~~~~~~~~~~~~
簡易なレンジ判定ロジックを集約するモジュール。

低ボラティリティやADXの低迷が続く場合は「range_mode」を有効にし、
メインループや各コンポーネントにレンジ特化の挙動を促す。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(slots=True)
class RangeContext:
    active: bool
    reason: str
    score: float
    metrics: Dict[str, float]


def _score_component(value: float, threshold: float, reverse: bool = False) -> float:
    """閾値との差分から 0.0〜1.0 のスコアを算出する。"""
    if threshold <= 0.0:
        return 0.0
    ratio = value / threshold
    if reverse:
        ratio = threshold / value if value else 0.0
    score = max(0.0, min(1.0, 1.0 - ratio))
    return score


def _safe_get(fac: Dict[str, float], key: str, default: float = 0.0) -> float:
    value = fac.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):  # noqa: PERF203
        return default


def detect_range_mode(
    fac_m1: Dict[str, float],
    fac_h4: Dict[str, float],
    *,
    adx_threshold: float = 24.0,
    bbw_threshold: float = 0.24,
    atr_threshold: float = 7.0,
) -> RangeContext:
    """
    M1/H4 の因子からレンジモードを検知する。

    Returns
    -------
    RangeContext
        active: レンジモードかどうか
        reason: 主因
        score : 0〜1 の確信度（0.6 以上でアクティブ判定）
        metrics: 参考値
    """

    adx_m1 = _safe_get(fac_m1, "adx", 0.0)
    bbw_m1 = _safe_get(fac_m1, "bbw", 1.0)
    atr_m1 = _safe_get(fac_m1, "atr_pips", 10.0)
    adx_h4 = _safe_get(fac_h4, "adx", 0.0)
    slope_h4 = abs(_safe_get(fac_h4, "ma20", 0.0) - _safe_get(fac_h4, "ma10", 0.0))

    compression_ratio = 0.0
    if bbw_threshold > 0.0:
        compression_ratio = max(
            0.0,
            min(
                1.0,
                1.0 - (bbw_m1 / max(bbw_threshold, 1e-6)),
            ),
        )
    volatility_ratio = 0.0
    if atr_threshold > 0.0:
        volatility_ratio = max(
            0.0,
            min(
                1.0,
                1.0 - (atr_m1 / max(atr_threshold, 1e-6)),
            ),
        )

    relax_mix_m1 = (compression_ratio * 0.7) + (volatility_ratio * 0.5)
    relax_factor_m1 = max(0.18, 1.0 - min(0.9, relax_mix_m1))
    effective_adx_m1 = adx_m1 * relax_factor_m1

    relax_factor_h4 = max(0.3, 1.0 - min(0.6, compression_ratio * 0.4))
    effective_adx_h4 = adx_h4 * relax_factor_h4

    is_low_adx = adx_m1 <= adx_threshold
    is_narrow_band = bbw_m1 <= bbw_threshold
    is_low_atr = atr_m1 <= atr_threshold
    if not is_low_adx:
        is_low_adx = effective_adx_m1 <= adx_threshold
        if not is_low_adx and volatility_ratio >= 0.6:
            is_low_adx = effective_adx_m1 <= (adx_threshold + 4.0)
    h4_trend_weak = (
        effective_adx_h4 <= (adx_threshold + 3.0) and slope_h4 <= 0.00045
    )

    components = [
        _score_component(effective_adx_m1, adx_threshold),
        _score_component(bbw_m1, bbw_threshold),
        _score_component(atr_m1, atr_threshold),
        _score_component(effective_adx_h4, adx_threshold + 3.0),
        min(1.0, (0.00045 / slope_h4) if slope_h4 else 1.0),
        compression_ratio,
        volatility_ratio,
    ]
    composite = sum(components) / len(components)

    compression_trigger = (
        compression_ratio >= 0.65
        and volatility_ratio >= 0.50
        and effective_adx_m1 <= (adx_threshold + 3.0)
    )
    active = (is_low_adx and is_narrow_band and is_low_atr) or (
        composite >= 0.66 and h4_trend_weak
    )
    if not active and compression_trigger:
        active = True

    if active:
        if is_low_atr and is_narrow_band:
            reason = "volatility_compression"
        elif is_low_adx:
            reason = "adx_squeeze"
        elif compression_trigger:
            reason = "compression_override"
        else:
            reason = "trend_weaken"
    else:
        reason = "trend_ok"

    metrics = {
        "adx_m1": adx_m1,
        "bbw_m1": bbw_m1,
        "atr_pips": atr_m1,
        "adx_h4": adx_h4,
        "slope_h4": slope_h4,
        "composite": round(composite, 3),
        "low_adx": float(is_low_adx),
        "narrow_band": float(is_narrow_band),
        "low_atr": float(is_low_atr),
        "effective_adx_m1": effective_adx_m1,
        "effective_adx_h4": effective_adx_h4,
        "relax_factor_m1": relax_factor_m1,
        "relax_factor_h4": relax_factor_h4,
        "compression_ratio": compression_ratio,
        "volatility_ratio": volatility_ratio,
        "compression_trigger": float(compression_trigger),
        "adx_threshold": adx_threshold,
    }

    return RangeContext(active=active, reason=reason, score=composite, metrics=metrics)
