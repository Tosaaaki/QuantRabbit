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
    mode: str = "UNKNOWN"  # RANGE / NEUTRAL / TREND
    env_tf: str = "M1"
    macro_tf: str = "H4"


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


def _resolve_mode(
    active: bool,
    score: float,
    *,
    range_score_threshold: float,
    trend_score_threshold: float,
) -> str:
    if active:
        return "RANGE"
    if score <= trend_score_threshold:
        return "TREND"
    return "NEUTRAL"


def detect_range_mode(
    fac_m1: Dict[str, float],
    fac_h4: Dict[str, float],
    *,
    env_tf: str = "M1",
    macro_tf: str = "H4",
    adx_threshold: float = 24.0,
    # BBW is (upper-lower)/mid ratio. For USD/JPY M1 it's typically ~0.0002..0.0020.
    # Expressed as ratio (e.g. 0.0025 == 0.25%).
    bbw_threshold: float = 0.0025,
    # atr_pips is already converted to "pips" (USD/JPY: 0.01 == 1 pip).
    # 7.0p is almost always true for M1 and makes range_mode stick ON; use a low-vol threshold.
    atr_threshold: float = 3.0,
    bbw_pips_threshold: float = 0.0,
    vol_5m_threshold: float = 0.8,
    range_score_threshold: float = 0.66,
    trend_score_threshold: float = 0.45,
) -> RangeContext:
    """
    M1/H4 の因子からレンジモードを検知する。
    env_tf / macro_tf を渡すと判定系の文脈を識別できる。

    Returns
    -------
    RangeContext
        active: レンジモードかどうか
        reason: 主因
        score : 0〜1 の確信度（高いほどレンジ寄り）
        metrics: 参考値
    """

    adx_m1 = _safe_get(fac_m1, "adx", 0.0)
    bbw_m1 = _safe_get(fac_m1, "bbw", 1.0)
    # ATR が欠損した場合に 10p と誤判定しないよう、穏当なデフォルトに寄せる
    atr_m1 = _safe_get(fac_m1, "atr_pips", 1.8)
    vol_5m = _safe_get(fac_m1, "vol_5m", 0.0)
    if str(env_tf).upper() != "M1":
        # vol_5m は 5 bars 依存のため、非 M1 ではスケールが崩れる
        vol_5m_threshold = 0.0
        vol_5m = 0.0
    anchor_price = _safe_get(fac_m1, "ma20", 0.0) or _safe_get(fac_m1, "ma10", 0.0)
    if anchor_price <= 0.0:
        anchor_price = 155.0  # fallback for width→pips換算
    adx_h4 = _safe_get(fac_h4, "adx", 0.0)
    slope_h4 = abs(_safe_get(fac_h4, "ma20", 0.0) - _safe_get(fac_h4, "ma10", 0.0))

    # BBW を pips 基準に換算し、実際の値幅で「圧縮」を見る
    bbw_pips = bbw_m1 * anchor_price / 0.01 if anchor_price > 0 else 0.0
    bbw_thresh_pips = bbw_pips_threshold
    if bbw_thresh_pips <= 0.0 and anchor_price > 0:
        bbw_thresh_pips = bbw_threshold * anchor_price / 0.01

    compression_ratio = 0.0
    if bbw_thresh_pips > 0.0:
        compression_ratio = max(
            0.0,
            min(
                1.0,
                1.0 - (bbw_pips / max(bbw_thresh_pips, 1e-6)),
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

    relax_cap = 1.6  # raw ADX が閾値から大きく乖離している場合は緩和を抑制
    is_low_adx_raw = adx_m1 <= adx_threshold
    is_low_adx_relaxed = (
        not is_low_adx_raw
        and adx_m1 <= (adx_threshold * relax_cap)
        and effective_adx_m1 <= adx_threshold
    )
    # 極端な圧縮時は やや緩めに許容するが、生ADXが高止まりならトレンド継続とみなす
    is_low_adx = is_low_adx_raw or is_low_adx_relaxed
    is_narrow_band = (
        bbw_pips <= bbw_thresh_pips if bbw_thresh_pips > 0 else bbw_m1 <= bbw_threshold
    )
    is_low_atr = atr_m1 <= atr_threshold
    is_low_vol5m = vol_5m <= vol_5m_threshold if vol_5m_threshold > 0 else False
    if (
        not is_low_adx
        and volatility_ratio >= 0.6
        and adx_m1 <= (adx_threshold * (relax_cap + 0.2))
    ):
        is_low_adx = effective_adx_m1 <= (adx_threshold + 2.0)
    h4_trend_weak = (
        effective_adx_h4 <= (adx_threshold + 3.0) and slope_h4 <= 0.00045
    )

    components = [
        _score_component(effective_adx_m1, adx_threshold),
        _score_component(bbw_pips, bbw_thresh_pips if bbw_thresh_pips > 0 else bbw_threshold),
        _score_component(atr_m1, atr_threshold),
        _score_component(effective_adx_h4, adx_threshold + 3.0),
        min(1.0, (0.00045 / slope_h4) if slope_h4 else 1.0),
        compression_ratio,
        volatility_ratio,
        _score_component(vol_5m, vol_5m_threshold) if vol_5m_threshold > 0 else 0.0,
    ]
    composite = sum(components) / len(components)

    composite_threshold = range_score_threshold  # require stronger confluence before range=ON
    compression_trigger = (
        compression_ratio >= 0.65
        and volatility_ratio >= 0.50
        and effective_adx_m1 <= (adx_threshold + 3.0)
    )
    active = (is_low_adx and is_narrow_band and (is_low_atr or is_low_vol5m)) or (
        composite >= composite_threshold and h4_trend_weak
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

    mode = _resolve_mode(
        active,
        composite,
        range_score_threshold=composite_threshold,
        trend_score_threshold=trend_score_threshold,
    )

    metrics = {
        "adx_m1": adx_m1,
        "bbw_m1": bbw_m1,
        "bbw_pips": bbw_pips,
        "bbw_thresh_pips": bbw_thresh_pips,
        "atr_pips": atr_m1,
        "adx_h4": adx_h4,
        "slope_h4": slope_h4,
        "composite": round(composite, 3),
        "low_adx": float(is_low_adx),
        "narrow_band": float(is_narrow_band),
        "low_atr": float(is_low_atr),
        "low_vol5m": float(is_low_vol5m),
        "vol_5m": vol_5m,
        "effective_adx_m1": effective_adx_m1,
        "effective_adx_h4": effective_adx_h4,
        "relax_factor_m1": relax_factor_m1,
        "relax_factor_h4": relax_factor_h4,
        "relax_cap": relax_cap,
        "compression_ratio": compression_ratio,
        "volatility_ratio": volatility_ratio,
        "compression_trigger": float(compression_trigger),
        "adx_threshold": adx_threshold,
        "adx_m1_raw_low": float(is_low_adx_raw),
        "adx_m1_relaxed_low": float(is_low_adx_relaxed),
    }

    return RangeContext(
        active=active,
        reason=reason,
        score=composite,
        metrics=metrics,
        mode=mode,
        env_tf=env_tf,
        macro_tf=macro_tf,
    )


def detect_range_mode_for_tf(
    factors: Dict[str, Dict[str, float]],
    env_tf: str,
    *,
    macro_tf: str = "H4",
    **kwargs,
) -> RangeContext:
    """Factor cache から TF を選んでレンジ判定する薄いラッパー。"""
    fac_env = factors.get(env_tf, {}) if isinstance(factors, dict) else {}
    fac_macro = factors.get(macro_tf, {}) if isinstance(factors, dict) else {}
    return detect_range_mode(
        fac_env,
        fac_macro,
        env_tf=env_tf,
        macro_tf=macro_tf,
        **kwargs,
    )
