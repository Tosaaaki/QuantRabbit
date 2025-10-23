"""
analysis.local_decider
~~~~~~~~~~~~~~~~~~~~~~
OpenAI が利用できない場合に備えたフォールバック用ヒューリスティック判定。
市場レジームとテクニカル指標から簡易的に focus / weight / strategies を生成する。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

_ALLOWED_STRATEGIES = (
    "TrendMA",
    "Donchian55",
    "BB_RSI",
    "NewsSpikeReversal",
    "M1Scalper",
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
        focus_tag / weight_macro / ranked_strategies を含む dict
    """
    factors_m1 = payload.get("factors_m1") or {}
    factors_h4 = payload.get("factors_h4") or {}

    event_soon = bool(payload.get("event_soon"))
    news_short = payload.get("news_short") or []
    perf = payload.get("perf") or {}

    macro_adx = _safe_float(factors_h4.get("adx"))
    macro_gap = abs(_safe_float(factors_h4.get("ma10")) - _safe_float(factors_h4.get("ma20")))
    micro_adx = _safe_float(factors_m1.get("adx"))
    micro_rsi = _safe_float(factors_m1.get("rsi"), 50.0)
    atr_pips = _atr_pips(factors_m1)
    vol_5m = _safe_float(factors_m1.get("vol_5m"), 1.0)

    focus_tag = "hybrid"
    weight_macro = 0.5

    if event_soon or any(news_short):
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
    macro_pf = _safe_float(perf.get("macro_pf"), 1.0)
    micro_pf = _safe_float(perf.get("micro_pf"), 1.0)
    if focus_tag in {"macro", "hybrid"} and macro_pf < 0.9:
        weight_macro = min(weight_macro, 0.4)
    if focus_tag in {"micro", "hybrid"} and micro_pf < 0.9:
        weight_macro = min(weight_macro, 0.35)

    if focus_tag == "micro":
        weight_macro = min(weight_macro, 0.28)
    elif focus_tag == "macro":
        weight_macro = max(weight_macro, 0.34)

    weight_macro = max(0.0, min(1.0, round(weight_macro, 2)))

    ranked: List[str] = []

    if focus_tag == "event":
        _enqueue_unique(ranked, ("NewsSpikeReversal", "BB_RSI"))

    if focus_tag in {"macro", "hybrid", "event"}:
        if macro_adx >= 20 or macro_gap >= 0.03:
            _enqueue_unique(ranked, ("TrendMA",))
        if macro_adx >= 27 or macro_gap >= 0.055:
            _enqueue_unique(ranked, ("Donchian55",))

    if focus_tag in {"micro", "hybrid"}:
        if micro_rsi >= 62 or micro_rsi <= 38 or micro_adx <= 22:
            _enqueue_unique(ranked, ("BB_RSI",))
        if atr_pips >= 6.0 or vol_5m >= 1.15:
            _enqueue_unique(ranked, ("M1Scalper",))

    # 低ボラ時はスキャル戦略を抑制
    if atr_pips < 4.0 and "M1Scalper" in ranked:
        ranked.remove("M1Scalper")

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
        "ranked_strategies": ranked,
        "reason": "heuristic_fallback",
    }


__all__ = ["heuristic_decision"]
