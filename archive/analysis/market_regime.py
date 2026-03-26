"""
analysis.market_regime
~~~~~~~~~~~~~~~~~~~~~~
リアルタイム市況レジーム判定。

factor_cache の M1/M5 指標から「今の相場タイプ」を統合判定し、
各 strategy worker が strategy-local で参照して entry 可否を判断する。
（AGENTS.md: 共通レイヤは強制的に戦略を選別しない。各workerが自分で判断する。）

Usage:
    from analysis.market_regime import classify_regime, MarketRegime

    regime = classify_regime()
    if regime == MarketRegime.CHOPPY:
        return  # エントリー見送り
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

PIP = 0.01


class MarketRegime(str, Enum):
    """市況レジームの分類"""

    TRENDING_UP = "trending_up"  # ADX>25, DI+ > DI-, 上昇トレンド
    TRENDING_DOWN = "trending_down"  # ADX>25, DI- > DI+, 下降トレンド
    RANGE_TIGHT = "range_tight"  # ADX<20, BBW<0.001, 狭いレンジ
    RANGE_WIDE = "range_wide"  # ADX<20, BBW>0.002, 広いレンジ
    VOLATILE = "volatile"  # ATR急上昇, 高ボラ
    CHOPPY = "choppy"  # ADX<15, DI gap<8, 方向性なし（最も危険）
    UNKNOWN = "unknown"  # 判定不能


@dataclass(frozen=True)
class RegimeSnapshot:
    """レジーム判定結果と根拠指標のスナップショット"""

    regime: MarketRegime
    adx: float
    rsi: float
    atr_pips: float
    bbw: float
    di_gap: float
    di_plus: float
    di_minus: float
    ema_slope_10: float
    confidence: float  # 判定の確信度 0.0-1.0

    def is_tradeable(self) -> bool:
        """トレード可能なレジームか"""
        return (
            self.regime != MarketRegime.CHOPPY and self.regime != MarketRegime.UNKNOWN
        )

    def favors_long(self) -> bool:
        """ロング有利なレジームか"""
        return self.regime == MarketRegime.TRENDING_UP

    def favors_short(self) -> bool:
        """ショート有利なレジームか"""
        return self.regime == MarketRegime.TRENDING_DOWN

    def is_range(self) -> bool:
        """レンジ相場か"""
        return self.regime in (MarketRegime.RANGE_TIGHT, MarketRegime.RANGE_WIDE)

    def is_trending(self) -> bool:
        """トレンド相場か"""
        return self.regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN)


# --- 設定パラメータ（env override 可能） ---
_ADX_CHOPPY_MAX = float(os.getenv("REGIME_ADX_CHOPPY_MAX", "15.0"))
_ADX_RANGE_MAX = float(os.getenv("REGIME_ADX_RANGE_MAX", "20.0"))
_ADX_TREND_MIN = float(os.getenv("REGIME_ADX_TREND_MIN", "25.0"))
_DI_GAP_CHOPPY_MAX = float(os.getenv("REGIME_DI_GAP_CHOPPY_MAX", "8.0"))
_DI_GAP_TREND_MIN = float(os.getenv("REGIME_DI_GAP_TREND_MIN", "12.0"))
_BBW_TIGHT_MAX = float(os.getenv("REGIME_BBW_TIGHT_MAX", "0.0010"))
_BBW_WIDE_MIN = float(os.getenv("REGIME_BBW_WIDE_MIN", "0.0020"))
_ATR_VOLATILE_MULT = float(os.getenv("REGIME_ATR_VOLATILE_MULT", "1.8"))
_ATR_ROLLING_WINDOW = int(os.getenv("REGIME_ATR_ROLLING_WINDOW", "60"))  # M1 60本=1時間


def classify_regime(
    factors_m1: Optional[Dict[str, float]] = None,
    factors_m5: Optional[Dict[str, float]] = None,
    *,
    atr_rolling_mean: Optional[float] = None,
) -> RegimeSnapshot:
    """
    M1/M5 の factor_cache 指標から市況レジームを判定する。

    Args:
        factors_m1: factor_cache の M1 dict。None なら内部で取得を試みる。
        factors_m5: factor_cache の M5 dict（補助）。
        atr_rolling_mean: ATR の直近1時間平均。None なら factors_m1 の atr_pips をそのまま使う。

    Returns:
        RegimeSnapshot: 判定結果と根拠指標
    """
    if factors_m1 is None:
        try:
            from indicators.factor_cache import all_factors

            all_f = all_factors()
            factors_m1 = all_f.get("M1", {})
            if factors_m5 is None:
                factors_m5 = all_f.get("M5", {})
        except Exception:
            return RegimeSnapshot(
                regime=MarketRegime.UNKNOWN,
                adx=0.0,
                rsi=0.0,
                atr_pips=0.0,
                bbw=0.0,
                di_gap=0.0,
                di_plus=0.0,
                di_minus=0.0,
                ema_slope_10=0.0,
                confidence=0.0,
            )

    if not factors_m1:
        return RegimeSnapshot(
            regime=MarketRegime.UNKNOWN,
            adx=0.0,
            rsi=0.0,
            atr_pips=0.0,
            bbw=0.0,
            di_gap=0.0,
            di_plus=0.0,
            di_minus=0.0,
            ema_slope_10=0.0,
            confidence=0.0,
        )

    adx = float(factors_m1.get("adx", 0.0) or 0.0)
    rsi = float(factors_m1.get("rsi", 50.0) or 50.0)
    atr_pips = float(factors_m1.get("atr_pips", 0.0) or 0.0)
    bbw = float(factors_m1.get("bbw", 0.0) or 0.0)
    di_plus = float(factors_m1.get("plus_di", 0.0) or 0.0)
    di_minus = float(factors_m1.get("minus_di", 0.0) or 0.0)
    ema_slope = float(factors_m1.get("ema_slope_10", 0.0) or 0.0)
    di_gap = abs(di_plus - di_minus)

    # M5 の ADX/DI で補強（あれば）
    m5_adx = 0.0
    m5_di_gap = 0.0
    if factors_m5:
        m5_adx = float(factors_m5.get("adx", 0.0) or 0.0)
        m5_di_plus = float(factors_m5.get("plus_di", 0.0) or 0.0)
        m5_di_minus = float(factors_m5.get("minus_di", 0.0) or 0.0)
        m5_di_gap = abs(m5_di_plus - m5_di_minus)

    # --- レジーム判定ロジック ---
    regime = MarketRegime.UNKNOWN
    confidence = 0.5

    # 1. チョッピー判定（最優先: 最も危険な相場）
    if adx < _ADX_CHOPPY_MAX and di_gap < _DI_GAP_CHOPPY_MAX:
        regime = MarketRegime.CHOPPY
        # M5 も方向性がなければ確信度UP
        confidence = 0.8 if m5_adx < _ADX_RANGE_MAX else 0.6

    # 2. ボラ急変判定
    elif (
        atr_rolling_mean
        and atr_rolling_mean > 0
        and atr_pips > atr_rolling_mean * _ATR_VOLATILE_MULT
    ):
        regime = MarketRegime.VOLATILE
        confidence = min(0.9, 0.5 + (atr_pips / atr_rolling_mean - 1.0) * 0.3)

    # 3. トレンド判定
    elif adx >= _ADX_TREND_MIN and di_gap >= _DI_GAP_TREND_MIN:
        if di_plus > di_minus:
            regime = MarketRegime.TRENDING_UP
        else:
            regime = MarketRegime.TRENDING_DOWN
        # M5 もトレンド一致なら高確信
        if m5_adx >= _ADX_RANGE_MAX and m5_di_gap >= _DI_GAP_TREND_MIN:
            confidence = 0.85
        else:
            confidence = 0.65

    # 4. レンジ判定
    elif adx < _ADX_RANGE_MAX:
        if bbw < _BBW_TIGHT_MAX:
            regime = MarketRegime.RANGE_TIGHT
            confidence = 0.75
        elif bbw >= _BBW_WIDE_MIN:
            regime = MarketRegime.RANGE_WIDE
            confidence = 0.7
        else:
            # BBW が中間帯: ADX の低さで判定
            regime = MarketRegime.RANGE_TIGHT if adx < 17 else MarketRegime.RANGE_WIDE
            confidence = 0.55

    # 5. ADX 20-25 の遷移帯: EMA slope で方向を補助判定
    else:
        if abs(ema_slope) > 0.2 and di_gap >= 10:
            if ema_slope > 0 and di_plus > di_minus:
                regime = MarketRegime.TRENDING_UP
            elif ema_slope < 0 and di_minus > di_plus:
                regime = MarketRegime.TRENDING_DOWN
            else:
                regime = MarketRegime.RANGE_WIDE
            confidence = 0.5
        else:
            regime = MarketRegime.RANGE_WIDE
            confidence = 0.45

    result = RegimeSnapshot(
        regime=regime,
        adx=adx,
        rsi=rsi,
        atr_pips=atr_pips,
        bbw=bbw,
        di_gap=di_gap,
        di_plus=di_plus,
        di_minus=di_minus,
        ema_slope_10=ema_slope,
        confidence=confidence,
    )

    logging.debug(
        "[REGIME] %s adx=%.1f rsi=%.1f atr=%.2f bbw=%.5f di_gap=%.1f conf=%.2f",
        regime.value,
        adx,
        rsi,
        atr_pips,
        bbw,
        di_gap,
        confidence,
    )

    return result


# --- ストラテジー別の推奨アクション ---


def should_enter(
    regime: RegimeSnapshot,
    strategy_type: str,
    side: str,
) -> tuple[bool, str]:
    """
    レジームに基づいてエントリー可否を判定する。
    各 worker が strategy-local で呼び出す（shared gate ではない）。

    Args:
        regime: RegimeSnapshot
        strategy_type: "scalp_reversal" | "scalp_breakout" | "momentum" | "range_fade" | "trend_follow"
        side: "long" | "short"

    Returns:
        (allowed, reason)
    """
    r = regime.regime

    # チョッピー: 全面拒否
    if r == MarketRegime.CHOPPY:
        return False, "regime_choppy_block"

    # UNKNOWN: 判定不能時は通す（fail-open）
    if r == MarketRegime.UNKNOWN:
        return True, "regime_unknown_allow"

    # ストラテジータイプ別
    if strategy_type == "scalp_reversal":
        # リバーサル系: トレンド逆張りは拒否
        if r == MarketRegime.TRENDING_UP and side == "short":
            return False, "regime_trending_up_short_block"
        if r == MarketRegime.TRENDING_DOWN and side == "long":
            return False, "regime_trending_down_long_block"
        if r == MarketRegime.VOLATILE:
            return False, "regime_volatile_reversal_block"
        return True, "regime_ok"

    elif strategy_type == "scalp_breakout":
        # ブレイクアウト系: タイトレンジでは不利
        if r == MarketRegime.RANGE_TIGHT:
            return False, "regime_tight_range_breakout_block"
        return True, "regime_ok"

    elif strategy_type == "momentum":
        # モメンタム系: レンジでは不利、トレンド順張りが最適
        if r in (MarketRegime.RANGE_TIGHT, MarketRegime.RANGE_WIDE):
            return False, "regime_range_momentum_block"
        if r == MarketRegime.CHOPPY:
            return False, "regime_choppy_momentum_block"
        # トレンド方向と一致するか
        if r == MarketRegime.TRENDING_UP and side == "short":
            return False, "regime_trend_contra_momentum_block"
        if r == MarketRegime.TRENDING_DOWN and side == "long":
            return False, "regime_trend_contra_momentum_block"
        return True, "regime_ok"

    elif strategy_type == "range_fade":
        # レンジフェード系: トレンド相場では不利
        if r in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
            return False, "regime_trending_range_fade_block"
        if r == MarketRegime.VOLATILE:
            return False, "regime_volatile_range_fade_block"
        return True, "regime_ok"

    elif strategy_type == "trend_follow":
        # トレンドフォロー系: レンジ・チョッピーでは不利
        if r in (MarketRegime.RANGE_TIGHT, MarketRegime.RANGE_WIDE):
            return False, "regime_range_trend_follow_block"
        if r == MarketRegime.TRENDING_UP and side == "short":
            return False, "regime_trend_contra_block"
        if r == MarketRegime.TRENDING_DOWN and side == "long":
            return False, "regime_trend_contra_block"
        return True, "regime_ok"

    # デフォルト: 通す
    return True, "regime_default_allow"
