from __future__ import annotations

import math
import unittest
from datetime import datetime, timedelta, timezone

from quant_rabbit.analysis.candles import Candle
from quant_rabbit.analysis.indicators import (
    compute_adx_series,
    compute_atr_pips_series,
    compute_ema_spread_pips_series,
    compute_indicators,
)


def _make_candles(closes: list[float], pair: str = "USD_JPY") -> list[Candle]:
    base = datetime(2026, 5, 4, tzinfo=timezone.utc)
    out: list[Candle] = []
    prev_close = closes[0]
    for i, c in enumerate(closes):
        h = max(prev_close, c) + 0.05
        low_value = min(prev_close, c) - 0.05
        out.append(
            Candle(
                timestamp_utc=base + timedelta(minutes=5 * i),
                open=prev_close,
                high=h,
                low=low_value,
                close=c,
                volume=1000,
                complete=True,
            )
        )
        prev_close = c
    return out


class IndicatorSetTest(unittest.TestCase):
    def test_short_series_returns_invalid_set_without_raising(self) -> None:
        candles = _make_candles([157.0, 157.1, 157.2])
        result = compute_indicators("USD_JPY", "M5", candles)
        self.assertFalse(result.valid)
        self.assertEqual(result.candles_count, 3)
        self.assertEqual(result.pip_size, 0.01)
        self.assertIsNone(result.atr_14)
        self.assertIsNone(result.rsi_14)

    def test_uptrend_series_yields_long_leaning_indicators(self) -> None:
        prices = [156.0 + i * 0.05 for i in range(100)]
        candles = _make_candles(prices)
        result = compute_indicators("USD_JPY", "M5", candles)
        self.assertTrue(result.valid)
        self.assertGreater(result.ema_12, result.ema_50)
        self.assertGreater(result.rsi_14, 60)
        self.assertGreater(result.macd_hist, 0)
        self.assertGreater(result.adx_14, 20)
        self.assertGreater(result.plus_di_14, result.minus_di_14)
        self.assertGreater(result.ema_slope_5, 0)
        self.assertEqual(result.ichimoku_cloud_pos, 1)

    def test_downtrend_series_yields_short_leaning_indicators(self) -> None:
        prices = [160.0 - i * 0.05 for i in range(100)]
        candles = _make_candles(prices)
        result = compute_indicators("USD_JPY", "M5", candles)
        self.assertTrue(result.valid)
        self.assertLess(result.ema_12, result.ema_50)
        self.assertLess(result.rsi_14, 40)
        # On a perfectly linear ramp, MACD line and signal converge to the same constant
        # so the histogram floats at the edge of float precision. `<= 1e-9` covers that
        # while still catching any real positive bias.
        self.assertLessEqual(result.macd_hist, 1e-9)
        self.assertGreater(result.minus_di_14, result.plus_di_14)
        self.assertEqual(result.ichimoku_cloud_pos, -1)

    def test_pip_size_distinguishes_jpy_and_non_jpy_pairs(self) -> None:
        usd_jpy = compute_indicators("USD_JPY", "M5", _make_candles([157.0] * 60))
        eur_usd = compute_indicators("EUR_USD", "M5", _make_candles([1.1700 + i * 0.0001 for i in range(60)]))
        self.assertEqual(usd_jpy.pip_size, 0.01)
        self.assertEqual(eur_usd.pip_size, 0.0001)

    def test_donchian_and_swing_bands_are_consistent(self) -> None:
        prices = [157.0 + math.sin(i / 5.0) * 0.30 for i in range(80)]
        candles = _make_candles(prices)
        result = compute_indicators("USD_JPY", "M5", candles)
        self.assertIsNotNone(result.donchian_high)
        self.assertIsNotNone(result.donchian_low)
        self.assertGreaterEqual(result.donchian_high, result.donchian_low)
        self.assertGreaterEqual(result.swing_high, result.swing_low)

    def test_adx_series_exposes_bounded_directionless_strength_history(self) -> None:
        prices = [156.0 + i * 0.01 + (i / 100.0) ** 2 for i in range(140)]
        candles = _make_candles(prices)

        series = compute_adx_series(
            [candle.high for candle in candles],
            [candle.low for candle in candles],
            [candle.close for candle in candles],
            count=12,
        )

        self.assertEqual(len(series), 12)
        self.assertTrue(all(math.isfinite(value) and value >= 0.0 for value in series))

    def test_volatility_and_ema_cross_histories_are_bounded_and_in_pips(self) -> None:
        prices = [156.0 - i * 0.01 for i in range(70)] + [155.3 + i * 0.04 for i in range(70)]
        candles = _make_candles(prices)

        atr = compute_atr_pips_series(
            [candle.high for candle in candles],
            [candle.low for candle in candles],
            [candle.close for candle in candles],
            pip_size=0.01,
            count=15,
        )
        ema_spread = compute_ema_spread_pips_series(
            [candle.close for candle in candles],
            pip_size=0.01,
            count=30,
        )

        self.assertEqual(len(atr), 15)
        self.assertTrue(all(value > 0.0 for value in atr))
        self.assertEqual(len(ema_spread), 30)
        self.assertLess(min(ema_spread), max(ema_spread))
        self.assertGreater(ema_spread[-1], 0.0)


if __name__ == "__main__":
    unittest.main()
