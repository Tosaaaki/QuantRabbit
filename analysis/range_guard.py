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
    adx_threshold: float = 22.0,
    bbw_threshold: float = 0.20,
    atr_threshold: float = 6.0,
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

    is_low_adx = adx_m1 <= adx_threshold
    is_narrow_band = bbw_m1 <= bbw_threshold
    is_low_atr = atr_m1 <= atr_threshold
    h4_trend_weak = adx_h4 <= (adx_threshold + 3.0) and slope_h4 <= 0.00045

    components = [
        _score_component(adx_m1, adx_threshold),
        _score_component(bbw_m1, bbw_threshold),
        _score_component(atr_m1, atr_threshold),
        _score_component(adx_h4, adx_threshold + 3.0),
        min(1.0, (0.00045 / slope_h4) if slope_h4 else 1.0),
    ]
    composite = sum(components) / len(components)

    active = (is_low_adx and is_narrow_band and is_low_atr) or (
        composite >= 0.65 and h4_trend_weak
    )

    if active:
        if is_low_atr and is_narrow_band:
            reason = "volatility_compression"
        elif is_low_adx:
            reason = "adx_squeeze"
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
    }

    return RangeContext(active=active, reason=reason, score=composite, metrics=metrics)
