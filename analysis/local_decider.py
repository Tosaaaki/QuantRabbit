"""
analysis.local_decider
~~~~~~~~~~~~~~~~~~~~~~
OpenAI が利用できない場合に備えたフォールバック用ヒューリスティック判定。
市場レジームとテクニカル指標から簡易的に focus / macro/micro/scalp の重み / strategies を生成する。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

_ALLOWED_STRATEGIES = (
    "TrendMA",
    "H1Momentum",
    "Donchian55",
    "BB_RSI",
    "MicroVWAPBound",
    "M1Scalper",
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

    if focus_tag == "event":
        _enqueue_unique(ranked, ("BB_RSI",))

    if focus_tag in {"macro", "hybrid", "event"}:
        if macro_adx >= 19 or macro_gap >= 0.03:
            _enqueue_unique(ranked, ("H1Momentum",))
        if macro_adx >= 20 or macro_gap >= 0.032:
            _enqueue_unique(ranked, ("TrendMA",))
        if macro_adx >= 25 or macro_gap >= 0.05:
            _enqueue_unique(ranked, ("Donchian55",))

    if focus_tag in {"micro", "hybrid"}:
        if micro_rsi >= 62 or micro_rsi <= 38 or micro_adx <= 22:
            _enqueue_unique(ranked, ("BB_RSI",))
        if atr_pips >= 6.0 or vol_5m >= 1.15:
            _enqueue_unique(ranked, ("M1Scalper",))
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

    # 低ボラ時はスキャル戦略を抑制
    if atr_pips < 4.0 and "M1Scalper" in ranked:
        ranked.remove("M1Scalper")

    # 偏り防止: 最低限のスキャル/レンジ系を常に候補に含める
    _enqueue_unique(ranked, ("M1Scalper", "BB_RSI"))
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
        "reason": "heuristic_fallback",
    }


__all__ = ["heuristic_decision"]
