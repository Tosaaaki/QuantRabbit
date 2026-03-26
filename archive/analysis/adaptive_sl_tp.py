"""
analysis.adaptive_sl_tp
~~~~~~~~~~~~~~~~~~~~~~~
市況レジームに応じた動的 SL/TP 計算。

各 strategy worker が strategy-local で呼び出す。
SL/TP を固定倍率ではなく、レジーム・スプレッド・ATR に応じて動的に決定する。

AGENTS.md: 「トレード判断・ロット・利確/損切り・保有調整は固定値運用を避け、
市場状態に応じて常時動的に更新する」に準拠。

Usage:
    from analysis.adaptive_sl_tp import compute_adaptive_sl_tp
    from analysis.market_regime import classify_regime

    regime = classify_regime()
    sl_tp = compute_adaptive_sl_tp(
        regime=regime,
        atr_pips=2.0,
        spread_pips=0.4,
        strategy_type="scalp_reversal",
    )
    if sl_tp:
        sl_pips = sl_tp["sl_pips"]
        tp_pips = sl_tp["tp_pips"]
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

from analysis.market_regime import MarketRegime, RegimeSnapshot

PIP = 0.01

# --- 設定パラメータ（env override 可能） ---

# SL の絶対下限（ノイズ刈り防止）
_SL_ABSOLUTE_FLOOR_PIPS = max(
    0.0, float(os.getenv("ADAPTIVE_SL_FLOOR_PIPS", "2.0") or 2.0)
)
# SL はスプレッドの何倍以上にするか
_SL_SPREAD_MULT_MIN = max(
    1.0, float(os.getenv("ADAPTIVE_SL_SPREAD_MULT", "3.0") or 3.0)
)


# --- レジーム別のSL/TP係数 ---
# key: (regime, strategy_type)
# value: (sl_atr_mult, tp_atr_mult)
_REGIME_PARAMS: Dict[tuple, tuple] = {
    # scalp_reversal: 逆張り系（extrema_reversal, wick_reversal_blend, DroughtRevert, PrecisionLowVol）
    (MarketRegime.RANGE_TIGHT, "scalp_reversal"): (1.2, 1.8),  # レンジ: SL狭め、TP近め
    (MarketRegime.RANGE_WIDE, "scalp_reversal"): (1.5, 2.2),  # 広めレンジ: やや広げる
    (MarketRegime.TRENDING_UP, "scalp_reversal"): (
        1.8,
        2.5,
    ),  # トレンド順方向: SL広め、TP遠め
    (MarketRegime.TRENDING_DOWN, "scalp_reversal"): (1.8, 2.5),
    (MarketRegime.VOLATILE, "scalp_reversal"): (2.5, 3.5),  # ボラ高: 大きめSL/TP
    (MarketRegime.CHOPPY, "scalp_reversal"): (0.0, 0.0),  # エントリー禁止
    # scalp_breakout: ブレイクアウト系（M1Scalper, session_open_breakout）
    (MarketRegime.RANGE_TIGHT, "scalp_breakout"): (1.5, 2.0),
    (MarketRegime.RANGE_WIDE, "scalp_breakout"): (1.5, 2.5),
    (MarketRegime.TRENDING_UP, "scalp_breakout"): (1.5, 3.0),  # トレンドに乗る: TP遠め
    (MarketRegime.TRENDING_DOWN, "scalp_breakout"): (1.5, 3.0),
    (MarketRegime.VOLATILE, "scalp_breakout"): (2.0, 3.5),
    (MarketRegime.CHOPPY, "scalp_breakout"): (0.0, 0.0),
    # momentum: モメンタム系（MomentumBurst）
    (MarketRegime.RANGE_TIGHT, "momentum"): (0.0, 0.0),  # レンジ: 不適
    (MarketRegime.RANGE_WIDE, "momentum"): (0.0, 0.0),
    (MarketRegime.TRENDING_UP, "momentum"): (1.5, 3.5),  # トレンド: TP最大
    (MarketRegime.TRENDING_DOWN, "momentum"): (1.5, 3.5),
    (MarketRegime.VOLATILE, "momentum"): (2.0, 4.0),
    (MarketRegime.CHOPPY, "momentum"): (0.0, 0.0),
    # range_fade: レンジフェード系（RangeFader）
    (MarketRegime.RANGE_TIGHT, "range_fade"): (1.0, 1.5),  # タイトレンジ: 最小SL/TP
    (MarketRegime.RANGE_WIDE, "range_fade"): (1.5, 2.0),
    (MarketRegime.TRENDING_UP, "range_fade"): (0.0, 0.0),  # トレンド: 不適
    (MarketRegime.TRENDING_DOWN, "range_fade"): (0.0, 0.0),
    (MarketRegime.VOLATILE, "range_fade"): (0.0, 0.0),
    (MarketRegime.CHOPPY, "range_fade"): (0.0, 0.0),
    # trend_follow: トレンドフォロー系（MicroTrendRetest）
    (MarketRegime.RANGE_TIGHT, "trend_follow"): (0.0, 0.0),
    (MarketRegime.RANGE_WIDE, "trend_follow"): (1.5, 2.0),
    (MarketRegime.TRENDING_UP, "trend_follow"): (1.5, 3.0),
    (MarketRegime.TRENDING_DOWN, "trend_follow"): (1.5, 3.0),
    (MarketRegime.VOLATILE, "trend_follow"): (2.0, 3.5),
    (MarketRegime.CHOPPY, "trend_follow"): (0.0, 0.0),
    # ping_5s: 超短期スキャルピング系
    (MarketRegime.RANGE_TIGHT, "ping_5s"): (1.2, 1.5),
    (MarketRegime.RANGE_WIDE, "ping_5s"): (1.5, 2.0),
    (MarketRegime.TRENDING_UP, "ping_5s"): (1.5, 2.5),
    (MarketRegime.TRENDING_DOWN, "ping_5s"): (1.5, 2.5),
    (MarketRegime.VOLATILE, "ping_5s"): (0.0, 0.0),  # ボラ高: 不適
    (MarketRegime.CHOPPY, "ping_5s"): (0.0, 0.0),
    # micro_level: レベルリアクター系
    (MarketRegime.RANGE_TIGHT, "micro_level"): (1.5, 2.0),
    (MarketRegime.RANGE_WIDE, "micro_level"): (1.8, 2.5),
    (MarketRegime.TRENDING_UP, "micro_level"): (2.0, 3.0),
    (MarketRegime.TRENDING_DOWN, "micro_level"): (2.0, 3.0),
    (MarketRegime.VOLATILE, "micro_level"): (2.5, 3.5),
    (MarketRegime.CHOPPY, "micro_level"): (0.0, 0.0),
}

# デフォルトパラメータ（未定義の組み合わせ用）
_DEFAULT_PARAMS = (1.5, 2.0)


def compute_adaptive_sl_tp(
    regime: RegimeSnapshot,
    atr_pips: float,
    spread_pips: float,
    strategy_type: str,
) -> Optional[Dict[str, float]]:
    """
    市況レジームに応じて SL/TP を動的に計算する。

    Args:
        regime: RegimeSnapshot from classify_regime()
        atr_pips: 現在の ATR (pips)
        spread_pips: 現在のスプレッド (pips)
        strategy_type: ストラテジータイプ（_REGIME_PARAMS のキー）

    Returns:
        {"sl_pips": float, "tp_pips": float, "rr": float, "regime": str}
        エントリー禁止の場合は None
    """
    key = (regime.regime, strategy_type)
    sl_mult, tp_mult = _REGIME_PARAMS.get(key, _DEFAULT_PARAMS)

    # sl_mult == 0.0 はエントリー禁止
    if sl_mult <= 0.0 or tp_mult <= 0.0:
        logging.debug(
            "[ADAPTIVE_SL_TP] blocked: regime=%s strategy=%s",
            regime.regime.value,
            strategy_type,
        )
        return None

    # ATR ベースの SL/TP
    sl_pips = atr_pips * sl_mult
    tp_pips = atr_pips * tp_mult

    # SL 下限: スプレッドの N 倍以上 + 絶対下限
    sl_floor = max(
        _SL_ABSOLUTE_FLOOR_PIPS,
        spread_pips * _SL_SPREAD_MULT_MIN,
    )
    if sl_pips < sl_floor:
        # SL を引き上げた分、RR を維持するため TP も比例拡大
        original_rr = tp_pips / max(sl_pips, 1e-6)
        sl_pips = sl_floor
        tp_pips = sl_pips * original_rr

    rr = tp_pips / max(sl_pips, 1e-6)

    result = {
        "sl_pips": round(sl_pips, 2),
        "tp_pips": round(tp_pips, 2),
        "rr": round(rr, 3),
        "regime": regime.regime.value,
        "sl_mult": sl_mult,
        "tp_mult": tp_mult,
        "atr_pips": round(atr_pips, 2),
        "spread_pips": round(spread_pips, 3),
    }

    logging.debug(
        "[ADAPTIVE_SL_TP] regime=%s strategy=%s sl=%.2f tp=%.2f rr=%.3f",
        regime.regime.value,
        strategy_type,
        sl_pips,
        tp_pips,
        rr,
    )

    return result


def map_strategy_tag_to_type(strategy_tag: str) -> str:
    """
    strategy_tag からストラテジータイプにマッピングする。
    各 worker が直接 strategy_type を指定してもよいが、
    利便性のためにタグからの自動マッピングも提供する。
    """
    tag = (strategy_tag or "").lower()

    # 逆張り系
    if any(
        k in tag
        for k in (
            "extrema_reversal",
            "wick_reversal",
            "droughtrevert",
            "drought_revert",
            "precisionlowvol",
            "precision_lowvol",
            "vwapreverts",
            "vwaprevert",
            "wickreversalblend",
            "falsebreakfade",
            "false_break_fade",
        )
    ):
        return "scalp_reversal"

    # ブレイクアウト系
    if any(
        k in tag
        for k in (
            "m1scalper",
            "session_open",
            "squeezepulsebreak",
        )
    ):
        return "scalp_breakout"

    # モメンタム系
    if any(
        k in tag
        for k in (
            "momentumburst",
            "momentum_burst",
        )
    ):
        return "momentum"

    # レンジフェード系
    if any(
        k in tag
        for k in (
            "rangefader",
            "range_fader",
        )
    ):
        return "range_fade"

    # トレンドフォロー系
    if any(
        k in tag
        for k in (
            "microtrendretest",
            "trend_retest",
            "micropullbackema",
        )
    ):
        return "trend_follow"

    # ping_5s系
    if "ping_5s" in tag:
        return "ping_5s"

    # micro_level系
    if any(
        k in tag
        for k in (
            "microlevelreactor",
            "level_reactor",
            "microrangebreak",
            "range_break",
            "microvwaprevert",
            "microvwapbound",
            "microcompressionrevert",
        )
    ):
        return "micro_level"

    # デフォルト
    return "scalp_reversal"
