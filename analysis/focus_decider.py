"""
analysis.focus_decider
~~~~~~~~~~~~~~~~~~~~~~
Macro / Micro 2 つのレジームと補助情報を入力し、

    focus_tag   : "micro" | "macro" | "hybrid" | "event"
    weight_macro: 0.0〜1.0  (lot の配分比率)

を決定する。
"""

from __future__ import annotations
from typing import Literal

Focus = Literal["micro", "macro", "hybrid", "event"]


def decide_focus(
    macro_regime: str,
    micro_regime: str,
    *,
    event_soon: bool = False,
    macro_pf: float | None = None,
    micro_pf: float | None = None,
) -> tuple[Focus, float]:
    """
    Parameters
    ----------
    macro_regime : H4/D1 判定結果
    micro_regime : M1 判定結果
    event_soon   : 経済指標 <=30min
    macro_pf     : 直近 PF  (optional)
    micro_pf     : 直近 PF  (optional)

    Returns
    -------
    focus_tag, weight_macro
    """

    # Event 優先
    if event_soon:
        return "event", 0.0

    # Trend 方向が一致
    if macro_regime == "Trend" and micro_regime == "Trend":
        return "macro", 0.8

    # Macro Trend だが Micro は Range  -> 押し目取りチャンス
    if macro_regime == "Trend" and micro_regime == "Range":
        return "micro", 0.3

    # Macro Range だが Micro Breakout 始動
    if macro_regime in ("Range", "Mixed") and micro_regime == "Breakout":
        return "micro", 0.4

    # どちらも Range
    if macro_regime == "Range" and micro_regime == "Range":
        return "micro", 0.2

    # 迷ったらハイブリッド
    base = 0.5
    if macro_pf is not None and micro_pf is not None:
        # 成績差で重みを 0.3〜0.7 に微調整
        diff = max(min((macro_pf - micro_pf) / 2.0, 0.2), -0.2)
        base += diff
    return "hybrid", round(base, 2)