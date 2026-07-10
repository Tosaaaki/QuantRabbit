from __future__ import annotations

import math
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from quant_rabbit.analysis.candles import Candle
from quant_rabbit.analysis.families import (
    FamilyScores,
    compute_family_scores,
)
from quant_rabbit.analysis.indicators import IndicatorSet, compute_indicators


def _make_candles(closes: list[float], pair: str = "USD_JPY") -> list[Candle]:
    base = datetime(2026, 5, 4, tzinfo=timezone.utc)
    out: list[Candle] = []
    prev_close = closes[0]
    for i, c in enumerate(closes):
        h = max(prev_close, c) + 0.05
        lo = min(prev_close, c) - 0.05
        out.append(
            Candle(
                timestamp_utc=base + timedelta(minutes=5 * i),
                open=prev_close,
                high=h,
                low=lo,
                close=c,
                volume=1000,
                complete=True,
            )
        )
        prev_close = c
    return out


def _empty_indicator_set() -> IndicatorSet:
    """Build a minimum IndicatorSet with all-None numeric fields."""
    return IndicatorSet(
        pair="USD_JPY",
        granularity="M5",
        candles_count=0,
        pip_size=0.01,
        close=150.0,
        valid=False,
        sma_20=None,
        ema_12=None,
        ema_20=None,
        ema_50=None,
        ema_slope_5=None,
        ema_slope_20=None,
        rsi_14=None,
        stoch_rsi=None,
        macd=None,
        macd_signal=None,
        macd_hist=None,
        roc_5=None,
        roc_10=None,
        cci_14=None,
        atr_14=None,
        atr_pips=None,
        bb_upper=None,
        bb_middle=None,
        bb_lower=None,
        bb_width=None,
        bb_span_pips=None,
        keltner_width=None,
        donchian_high=None,
        donchian_low=None,
        donchian_width_pips=None,
        adx_14=None,
        plus_di_14=None,
        minus_di_14=None,
        vwap=None,
        vwap_gap_pips=None,
        swing_high=None,
        swing_low=None,
        swing_distance_high_pips=None,
        swing_distance_low_pips=None,
        ichimoku_tenkan=None,
        ichimoku_kijun=None,
        ichimoku_span_a=None,
        ichimoku_span_b=None,
        ichimoku_cloud_pos=0,
    )


class FamilyScoresEdgeTest(unittest.TestCase):
    def test_empty_indicator_set_returns_zero_scores(self) -> None:
        ind = _empty_indicator_set()
        scores = compute_family_scores(ind)
        self.assertIsInstance(scores, FamilyScores)
        self.assertEqual(scores.trend_score, 0.0)
        self.assertEqual(scores.mean_rev_score, 0.0)
        self.assertEqual(scores.breakout_score, 0.0)
        self.assertEqual(scores.trend_components, {})
        self.assertEqual(scores.mean_rev_components, {})
        self.assertEqual(scores.breakout_components, {})

    def test_dataclass_is_frozen(self) -> None:
        scores = FamilyScores(trend_score=0.0, mean_rev_score=0.0, breakout_score=0.0)
        with self.assertRaises(Exception):
            scores.trend_score = 1.0  # type: ignore[misc]


class FamilyScoresTrendingMarketTest(unittest.TestCase):
    def test_uptrend_yields_positive_trend_negative_mean_rev(self) -> None:
        # Monotonic ramp -> close > EMA50, MACD-hist > 0, RSI > 50.
        prices = [156.0 + i * 0.05 for i in range(200)]
        candles = _make_candles(prices)
        ind = compute_indicators("USD_JPY", "M5", candles)
        self.assertTrue(ind.valid)
        scores = compute_family_scores(ind)
        self.assertGreater(scores.trend_score, 0.0)
        # In an uptrend, RSI > 50 and price extended above VWAP -> mean-rev
        # score should be negative (mean-rev SHORT bias).
        self.assertLess(scores.mean_rev_score, 0.0)

    def test_downtrend_yields_negative_trend_positive_mean_rev(self) -> None:
        prices = [160.0 - i * 0.05 for i in range(200)]
        candles = _make_candles(prices)
        ind = compute_indicators("USD_JPY", "M5", candles)
        self.assertTrue(ind.valid)
        scores = compute_family_scores(ind)
        self.assertLess(scores.trend_score, 0.0)
        # Down-extended -> mean-rev LONG bias -> positive mean_rev_score.
        self.assertGreater(scores.mean_rev_score, 0.0)


class FamilyScoresMeanReversionTest(unittest.TestCase):
    def test_stoch_rsi_uses_zero_to_one_scale_and_correct_mean_reversion_side(self) -> None:
        overbought = replace(_empty_indicator_set(), stoch_rsi=1.0)
        neutral = replace(_empty_indicator_set(), stoch_rsi=0.5)
        oversold = replace(_empty_indicator_set(), stoch_rsi=0.0)

        overbought_scores = compute_family_scores(overbought)
        neutral_scores = compute_family_scores(neutral)
        oversold_scores = compute_family_scores(oversold)

        self.assertLess(overbought_scores.mean_rev_score, 0.0)
        self.assertEqual(neutral_scores.mean_rev_score, 0.0)
        self.assertGreater(oversold_scores.mean_rev_score, 0.0)
        self.assertAlmostEqual(
            overbought_scores.mean_rev_score,
            -oversold_scores.mean_rev_score,
        )

    def test_anti_persistent_market_has_smaller_trend_score(self) -> None:
        # An AR(1)-with-negative-correlation series mean-reverts; trend
        # score should be small in absolute terms vs a clean ramp.
        import random
        random.seed(11)
        prev = 150.0
        mean_lvl = 150.0
        prices: list[float] = []
        for _ in range(200):
            eps = random.gauss(0, 0.05)
            cur = mean_lvl - 0.5 * (prev - mean_lvl) + eps
            prices.append(cur)
            prev = cur
        candles = _make_candles(prices)
        ind = compute_indicators("USD_JPY", "M5", candles)
        self.assertTrue(ind.valid)
        scores = compute_family_scores(ind)
        # Trend score absolute value should be modest — far smaller than
        # a clean monotonic ramp would produce (~1.5+).
        self.assertLess(abs(scores.trend_score), 0.7)


class FamilyScoresBreakoutTest(unittest.TestCase):
    def test_caller_supplied_atr_percentile_is_used(self) -> None:
        ind = _empty_indicator_set()
        # Low ATR percentile = squeeze = positive breakout pressure.
        scores = compute_family_scores(ind, atr_percentile=10.0, bb_width_percentile=10.0)
        self.assertGreater(scores.breakout_score, 0.0)
        self.assertIn("atr_squeeze", scores.breakout_components)
        self.assertIn("bb_width_squeeze", scores.breakout_components)

    def test_high_percentile_yields_negative_breakout_score(self) -> None:
        ind = _empty_indicator_set()
        scores = compute_family_scores(ind, atr_percentile=90.0, bb_width_percentile=90.0)
        self.assertLess(scores.breakout_score, 0.0)


class FamilyScoresDisagreementTest(unittest.TestCase):
    def test_disagreement_is_zero_when_all_zero(self) -> None:
        ind = _empty_indicator_set()
        scores = compute_family_scores(ind)
        self.assertEqual(scores.disagreement, 0.0)

    def test_disagreement_is_positive_when_signs_differ(self) -> None:
        # Trending uptrend usually yields trend>0, mean_rev<0 -> sign mix.
        prices = [156.0 + i * 0.05 for i in range(200)]
        candles = _make_candles(prices)
        ind = compute_indicators("USD_JPY", "M5", candles)
        scores = compute_family_scores(ind, atr_percentile=10.0, bb_width_percentile=10.0)
        # trend_score > 0, mean_rev_score < 0, breakout_score > 0
        # signs: +, -, + -> stdev > 0
        self.assertGreater(scores.disagreement, 0.0)


if __name__ == "__main__":
    unittest.main()
