"""
analysis.local_decider
~~~~~~~~~~~~~~~~~~~~~~
LLM なしで動作するヒューリスティック判定。
市場レジームとテクニカル指標から簡易的に focus / macro/micro/scalp の重み / strategies を生成する。
"""

from __future__ import annotations

import math
import os
from typing import Dict, Iterable, List, Optional

_ALLOWED_STRATEGIES = (
    "TrendMA",
    "H1Momentum",
    "BB_RSI",
    "MicroVWAPBound",
    "MomentumPulse",
    "VolCompressionBreak",
    "BB_RSI_Fast",
    "MicroVWAPRevert",
    "MomentumBurst",
    "TrendMomentumMicro",
    "MicroMomentumStack",
    "MicroPullbackEMA",
    "MicroRangeBreak",
    "MicroLevelReactor",
)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _score_linear(value: float, bad: float, ref: float) -> float:
    if value <= bad:
        return 0.0
    if value >= ref:
        return 1.0
    span = max(1e-6, ref - bad)
    return _clamp((value - bad) / span, 0.0, 1.0)


_DECIDER_PERF_ENABLED = _env_bool("DECIDER_PERF_ENABLED", True)
_DECIDER_PERF_HOURLY_ENABLED = _env_bool("DECIDER_PERF_HOURLY_ENABLED", True)
_DECIDER_PERF_MIN_TRADES = max(5, int(_env_float("DECIDER_PERF_MIN_TRADES", 12)))
_DECIDER_PERF_PF_BAD = _env_float("DECIDER_PERF_PF_BAD", 0.9)
_DECIDER_PERF_PF_REF = _env_float("DECIDER_PERF_PF_REF", 1.15)
_DECIDER_PERF_WIN_BAD = _env_float("DECIDER_PERF_WIN_BAD", 0.48)
_DECIDER_PERF_WIN_REF = _env_float("DECIDER_PERF_WIN_REF", 0.55)
_DECIDER_PERF_MULT_MIN = _env_float("DECIDER_PERF_MULT_MIN", 0.85)
_DECIDER_PERF_MULT_MAX = _env_float("DECIDER_PERF_MULT_MAX", 1.1)

_DECIDER_PERF_HOURLY_MIN_TRADES = max(
    5, int(_env_float("DECIDER_PERF_HOURLY_MIN_TRADES", 8))
)
_DECIDER_PERF_HOURLY_PF_BAD = _env_float("DECIDER_PERF_HOURLY_PF_BAD", 0.85)
_DECIDER_PERF_HOURLY_PF_REF = _env_float("DECIDER_PERF_HOURLY_PF_REF", 1.1)
_DECIDER_PERF_HOURLY_WIN_BAD = _env_float("DECIDER_PERF_HOURLY_WIN_BAD", 0.46)
_DECIDER_PERF_HOURLY_WIN_REF = _env_float("DECIDER_PERF_HOURLY_WIN_REF", 0.54)
_DECIDER_PERF_HOURLY_MULT_MIN = _env_float("DECIDER_PERF_HOURLY_MULT_MIN", 0.8)
_DECIDER_PERF_HOURLY_MULT_MAX = _env_float("DECIDER_PERF_HOURLY_MULT_MAX", 1.05)
_DECIDER_PERF_TOTAL_MULT_MIN = _env_float("DECIDER_PERF_TOTAL_MULT_MIN", 0.7)
_DECIDER_PERF_TOTAL_MULT_MAX = _env_float("DECIDER_PERF_TOTAL_MULT_MAX", 1.15)
_DECIDER_FORECAST_ENABLED = _env_bool("DECIDER_FORECAST_ENABLED", True)
_DECIDER_FORECAST_EDGE_MIN = _clamp(_env_float("DECIDER_FORECAST_EDGE_MIN", 0.08), 0.0, 0.8)
_DECIDER_FORECAST_WEIGHT_MAX = _clamp(_env_float("DECIDER_FORECAST_WEIGHT_MAX", 0.16), 0.0, 0.35)


def _atr_pips(factors: Optional[Dict]) -> float:
    if not factors:
        return 0.0
    if "atr_pips" in factors:
        return _safe_float(factors.get("atr_pips"))
    atr = factors.get("atr")
    if atr is not None:
        return max(0.0, _safe_float(atr)) * 100.0
    return 0.0


def _enqueue_unique(seq: List[str], names: Iterable[str]) -> None:
    allowed = set(_ALLOWED_STRATEGIES)
    for name in names:
        if name in allowed and name not in seq:
            seq.append(name)


def _pocket_pf(perf: Dict, pocket: str, default: float = 1.0) -> float:
    """Accepts nested ({'macro': {'pf':1.2}}) or flattened ('macro_pf':1.2) layouts."""
    if not perf:
        return default
    direct = perf.get(pocket)
    if isinstance(direct, dict):
        return _safe_float(direct.get("pf"), default)
    return _safe_float(perf.get(f"{pocket}_pf"), default)


def _pocket_metric(perf: Dict, pocket: str, key: str, default: float = 0.0) -> float:
    """Fetch pocket metric from nested or flattened dicts."""
    if not perf:
        return default
    direct = perf.get(pocket)
    if isinstance(direct, dict):
        return _safe_float(direct.get(key), default)
    return _safe_float(perf.get(f"{pocket}_{key}"), default)


def _perf_multiplier(
    pf: float,
    win_rate: float,
    sample: float,
    *,
    min_trades: int,
    pf_bad: float,
    pf_ref: float,
    win_bad: float,
    win_ref: float,
    mult_min: float,
    mult_max: float,
) -> float:
    if sample < min_trades:
        return 1.0
    pf_score = _score_linear(pf, pf_bad, pf_ref)
    win_score = _score_linear(win_rate, win_bad, win_ref)
    combined = _clamp(0.6 * pf_score + 0.4 * win_score, 0.0, 1.0)
    return _clamp(mult_min + combined * (mult_max - mult_min), mult_min, mult_max)


def _sigmoid(x: float) -> float:
    x = _clamp(float(x), -30.0, 30.0)
    return 1.0 / (1.0 + math.exp(-x))


def _technical_forecast_bias(factors_m1: Dict, factors_h4: Dict) -> Dict[str, float]:
    atr_pips = max(0.5, _atr_pips(factors_m1))
    vol_5m = max(0.1, _safe_float(factors_m1.get("vol_5m"), 1.0))
    micro_rsi = _safe_float(factors_m1.get("rsi"), 50.0)
    micro_adx = max(0.0, _safe_float(factors_m1.get("adx")))
    macro_adx = max(0.0, _safe_float(factors_h4.get("adx")))

    ma10_m1 = _safe_float(factors_m1.get("ma10"))
    ma20_m1 = _safe_float(factors_m1.get("ma20"))
    ma10_h4 = _safe_float(factors_h4.get("ma10"))
    ma20_h4 = _safe_float(factors_h4.get("ma20"))

    m1_gap_pips = (ma10_m1 - ma20_m1) / 0.01 if ma10_m1 and ma20_m1 else 0.0
    h4_gap_pips = (ma10_h4 - ma20_h4) / 0.01 if ma10_h4 and ma20_h4 else 0.0
    m1_gap_norm = m1_gap_pips / max(1.0, atr_pips)
    h4_gap_norm = h4_gap_pips / max(1.5, atr_pips * 2.2)

    rsi_term = (micro_rsi - 50.0) / 16.0
    adx_term = _clamp((micro_adx - 16.0) / 18.0, 0.0, 1.0)
    macro_adx_term = _clamp((macro_adx - 18.0) / 16.0, 0.0, 1.0)
    vol_term = _clamp((vol_5m - 0.9) / 0.7, -1.0, 1.0)

    trend_score = (
        1.10 * math.tanh(m1_gap_norm * 1.25)
        + 0.88 * math.tanh(h4_gap_norm * 1.0)
        + 0.42 * rsi_term
        + 0.34 * adx_term
        + 0.24 * macro_adx_term
        + 0.18 * vol_term
    )
    mean_score = (
        -0.72 * math.tanh(m1_gap_norm * 1.35)
        - 0.44 * rsi_term
        + 0.22 * _clamp(1.0 - vol_term, 0.0, 1.0)
    )
    trend_strength = _clamp(
        0.2
        + 0.45 * abs(math.tanh(m1_gap_norm * 1.4))
        + 0.2 * adx_term
        + 0.15 * macro_adx_term
        + 0.1 * max(0.0, vol_term),
        0.0,
        1.0,
    )
    range_pressure = _clamp(1.0 - trend_strength, 0.0, 1.0)
    combo = trend_score * (1.0 - 0.5 * range_pressure) + 0.55 * mean_score * range_pressure
    p_up = _clamp(_sigmoid(combo * 1.15), 0.04, 0.96)
    edge = abs(p_up - 0.5) * 2.0

    return {
        "p_up": round(p_up, 6),
        "edge": round(edge, 6),
        "trend_strength": round(trend_strength, 6),
        "range_pressure": round(range_pressure, 6),
        "m1_gap_pips": round(m1_gap_pips, 4),
        "h4_gap_pips": round(h4_gap_pips, 4),
        "direction": 1.0 if p_up >= 0.5 else -1.0,
    }


def heuristic_decision(
    payload: Dict,
    last_decision: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """
    GPT 応答エラー時に利用する簡易フォールバック。

    Args:
        payload: main ループから渡される GPT 入力データ
        last_decision: 前回の決定（存在すればバイアスの維持に使用）
    Returns:
        focus_tag / weight_macro / weight_scalp / ranked_strategies を含む dict
    """
    factors_m1 = payload.get("factors_m1") or {}
    factors_h4 = payload.get("factors_h4") or {}

    event_soon = bool(payload.get("event_soon"))
    perf = payload.get("perf") or {}
    perf_hourly = payload.get("perf_hourly") or {}

    macro_adx = _safe_float(factors_h4.get("adx"))
    macro_gap = abs(_safe_float(factors_h4.get("ma10")) - _safe_float(factors_h4.get("ma20")))
    micro_adx = _safe_float(factors_m1.get("adx"))
    micro_rsi = _safe_float(factors_m1.get("rsi"), 50.0)
    atr_pips = _atr_pips(factors_m1)
    vol_5m = _safe_float(factors_m1.get("vol_5m"), 1.0)
    ma10_m1 = _safe_float(factors_m1.get("ma10"))
    ma20_m1 = _safe_float(factors_m1.get("ma20"))
    ma_gap_pips = abs(ma10_m1 - ma20_m1) / 0.01 if ma10_m1 or ma20_m1 else 0.0

    focus_tag = "hybrid"
    weight_macro = 0.5

    if event_soon:
        focus_tag = "event"
        weight_macro = 0.35
    elif macro_adx >= 24 and macro_gap >= 0.045:
        focus_tag = "macro"
        weight_macro = min(0.85, 0.45 + macro_adx / 140 + macro_gap * 4.0)
    elif micro_adx <= 18 or atr_pips <= 5.5:
        focus_tag = "micro"
        weight_macro = 0.22
    else:
        focus_tag = "hybrid"
        weight_macro = min(0.7, 0.4 + max(0.0, (macro_adx - 20) / 120))

    # パフォーマンスが悪化している pocket は重みを抑える
    macro_pf = _pocket_pf(perf, "macro", 1.0)
    micro_pf = _pocket_pf(perf, "micro", 1.0)
    scalp_pf = _pocket_pf(perf, "scalp", 1.0)
    if focus_tag in {"macro", "hybrid"} and macro_pf < 0.9:
        weight_macro = min(weight_macro, 0.4)
    if focus_tag in {"micro", "hybrid"} and micro_pf < 0.9:
        weight_macro = min(weight_macro, 0.35)

    if focus_tag == "micro":
        weight_macro = min(weight_macro, 0.28)
    elif focus_tag == "macro":
        weight_macro = max(weight_macro, 0.34)

    # スキャル pocket の重みを決定
    weight_scalp = 0.1
    if atr_pips >= 7.0 or vol_5m >= 1.35:
        weight_scalp = 0.18
    elif atr_pips <= 3.4 or vol_5m <= 0.85:
        weight_scalp = 0.06
    if atr_pips <= 2.2 or vol_5m <= 0.7:
        weight_scalp = 0.0
    if event_soon:
        weight_scalp = min(weight_scalp, 0.05)
    if focus_tag == "macro":
        weight_scalp = min(weight_scalp, 0.08)
    elif focus_tag == "event":
        weight_scalp = min(weight_scalp, 0.04)

    if scalp_pf < 0.85:
        weight_scalp = min(weight_scalp, 0.06)
    elif scalp_pf > 1.15:
        weight_scalp = max(weight_scalp, 0.14)

    forecast_bias: Dict[str, float] | None = None
    if _DECIDER_FORECAST_ENABLED:
        forecast_bias = _technical_forecast_bias(factors_m1, factors_h4)
        edge = _safe_float(forecast_bias.get("edge"))
        trend_strength = _safe_float(forecast_bias.get("trend_strength"))
        range_pressure = _safe_float(forecast_bias.get("range_pressure"))

        if edge >= _DECIDER_FORECAST_EDGE_MIN:
            edge_strength = _clamp(
                (edge - _DECIDER_FORECAST_EDGE_MIN) / max(1e-6, 1.0 - _DECIDER_FORECAST_EDGE_MIN),
                0.0,
                1.0,
            )
            macro_shift = _DECIDER_FORECAST_WEIGHT_MAX * edge_strength * (trend_strength - 0.5) * 2.0
            weight_macro = _clamp(weight_macro + macro_shift, 0.0, 0.9)

            if trend_strength >= 0.65 and focus_tag == "micro":
                focus_tag = "hybrid"
            elif range_pressure >= 0.72 and focus_tag == "macro":
                focus_tag = "hybrid"

            if edge <= 0.16 and range_pressure >= 0.6:
                weight_scalp = min(weight_scalp, 0.12)
            elif edge >= 0.28 and trend_strength >= 0.55 and atr_pips >= 4.0:
                weight_scalp = max(weight_scalp, 0.12)

    # Performance-aware weight scaling (overall + hourly)
    if _DECIDER_PERF_ENABLED:
        macro_win = _pocket_metric(perf, "macro", "win_rate", 0.5)
        micro_win = _pocket_metric(perf, "micro", "win_rate", 0.5)
        scalp_win = _pocket_metric(perf, "scalp", "win_rate", 0.5)
        macro_n = _pocket_metric(perf, "macro", "sample", 0.0)
        micro_n = _pocket_metric(perf, "micro", "sample", 0.0)
        scalp_n = _pocket_metric(perf, "scalp", "sample", 0.0)

        def _combined_mult(pocket: str, pf: float, win: float, sample: float) -> float:
            mult = _perf_multiplier(
                pf,
                win,
                sample,
                min_trades=_DECIDER_PERF_MIN_TRADES,
                pf_bad=_DECIDER_PERF_PF_BAD,
                pf_ref=_DECIDER_PERF_PF_REF,
                win_bad=_DECIDER_PERF_WIN_BAD,
                win_ref=_DECIDER_PERF_WIN_REF,
                mult_min=_DECIDER_PERF_MULT_MIN,
                mult_max=_DECIDER_PERF_MULT_MAX,
            )
            if _DECIDER_PERF_HOURLY_ENABLED and perf_hourly:
                h_pf = _pocket_metric(perf_hourly, pocket, "pf", pf)
                h_win = _pocket_metric(perf_hourly, pocket, "win_rate", win)
                h_n = _pocket_metric(perf_hourly, pocket, "sample", 0.0)
                h_mult = _perf_multiplier(
                    h_pf,
                    h_win,
                    h_n,
                    min_trades=_DECIDER_PERF_HOURLY_MIN_TRADES,
                    pf_bad=_DECIDER_PERF_HOURLY_PF_BAD,
                    pf_ref=_DECIDER_PERF_HOURLY_PF_REF,
                    win_bad=_DECIDER_PERF_HOURLY_WIN_BAD,
                    win_ref=_DECIDER_PERF_HOURLY_WIN_REF,
                    mult_min=_DECIDER_PERF_HOURLY_MULT_MIN,
                    mult_max=_DECIDER_PERF_HOURLY_MULT_MAX,
                )
                mult *= h_mult
            return _clamp(mult, _DECIDER_PERF_TOTAL_MULT_MIN, _DECIDER_PERF_TOTAL_MULT_MAX)

        base_macro = max(0.0, weight_macro)
        base_scalp = max(0.0, weight_scalp)
        base_micro = max(0.0, 1.0 - base_macro - base_scalp)

        macro_mult = _combined_mult("macro", macro_pf, macro_win, macro_n)
        micro_mult = _combined_mult("micro", micro_pf, micro_win, micro_n)
        scalp_mult = _combined_mult("scalp", scalp_pf, scalp_win, scalp_n)

        macro_w = base_macro * macro_mult
        scalp_w = base_scalp * scalp_mult
        micro_w = base_micro * micro_mult
        total = macro_w + scalp_w + micro_w
        if total > 0:
            weight_macro = macro_w / total
            weight_scalp = scalp_w / total

    max_total = 0.9
    total_weight = weight_macro + weight_scalp
    if total_weight > max_total:
        excess = total_weight - max_total
        if weight_scalp >= excess:
            weight_scalp -= excess
        else:
            remainder = excess - weight_scalp
            weight_scalp = 0.0
            weight_macro = max(0.0, weight_macro - remainder)

    weight_macro = max(0.0, min(1.0, round(weight_macro, 2)))
    weight_scalp = max(0.0, min(0.3, round(weight_scalp, 2)))

    ranked: List[str] = []
    if forecast_bias is not None:
        edge = _safe_float(forecast_bias.get("edge"))
        trend_strength = _safe_float(forecast_bias.get("trend_strength"))
        range_pressure = _safe_float(forecast_bias.get("range_pressure"))
        if edge >= _DECIDER_FORECAST_EDGE_MIN:
            if trend_strength >= range_pressure:
                _enqueue_unique(
                    ranked,
                    (
                        "H1Momentum",
                        "TrendMA",
                        "TrendMomentumMicro",
                        "MicroMomentumStack",
                        "MomentumBurst",
                    ),
                )
            else:
                _enqueue_unique(
                    ranked,
                    (
                        "BB_RSI",
                        "BB_RSI_Fast",
                        "MicroVWAPRevert",
                        "MicroLevelReactor",
                        "MicroPullbackEMA",
                    ),
                )

    if focus_tag == "event":
        _enqueue_unique(ranked, ("BB_RSI",))

    if focus_tag in {"macro", "hybrid", "event"}:
        if macro_adx >= 19 or macro_gap >= 0.03:
            _enqueue_unique(ranked, ("H1Momentum",))
        if macro_adx >= 20 or macro_gap >= 0.032:
            _enqueue_unique(ranked, ("TrendMA",))

    if focus_tag in {"micro", "hybrid"}:
        if micro_rsi >= 62 or micro_rsi <= 38 or micro_adx <= 22:
            _enqueue_unique(ranked, ("BB_RSI",))
        if atr_pips >= 6.0 or vol_5m >= 1.15:
            _enqueue_unique(ranked, ("MomentumPulse", "VolCompressionBreak"))
        if atr_pips <= 3.8 and vol_5m <= 1.3:
            _enqueue_unique(
                ranked,
                (
                    "MomentumPulse",
                    "VolCompressionBreak",
                    "BB_RSI_Fast",
                    "MicroVWAPRevert",
                ),
            )
        if micro_adx >= 20 and ma_gap_pips >= 0.45:
            _enqueue_unique(
                ranked,
                (
                    "TrendMomentumMicro",
                    "MicroMomentumStack",
                    "MomentumBurst",
                ),
            )
        if micro_adx <= 26 and vol_5m <= 1.2:
            _enqueue_unique(ranked, ("MicroRangeBreak", "MicroPullbackEMA"))
        if micro_adx <= 26 and vol_5m <= 1.1:
            _enqueue_unique(ranked, ("MicroLevelReactor",))

    # 偏り防止: 最低限のスキャル/レンジ系を常に候補に含める
    _enqueue_unique(ranked, ("BB_RSI", "MomentumPulse"))
    # macro 側のデフォルトも1本は入れておく
    _enqueue_unique(ranked, ("TrendMA",))

    # 前回決定の上位戦略を尊重し、再追加
    if last_decision:
        previous = last_decision.get("ranked_strategies") or []
        _enqueue_unique(ranked, previous[:2])

    if not ranked:
        fallback = ("BB_RSI",) if focus_tag in {"micro", "event"} else ("TrendMA",)
        _enqueue_unique(ranked, fallback)

    return {
        "focus_tag": focus_tag,
        "weight_macro": weight_macro,
        "weight_scalp": weight_scalp,
        "ranked_strategies": ranked,
        "forecast_bias": forecast_bias,
        "reason": "heuristic_fallback",
    }


__all__ = ["heuristic_decision"]
