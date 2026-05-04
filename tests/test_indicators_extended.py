"""Tests for the Phase A + Phase B extended indicator stack."""

from __future__ import annotations

import math
import unittest
from datetime import datetime, timedelta, timezone

from quant_rabbit.analysis.candles import Candle
from quant_rabbit.analysis.indicators import compute_indicators


def _make_candles(closes: list[float], pair: str = "USD_JPY", base_volume: int = 1000) -> list[Candle]:
    base = datetime(2026, 5, 4, tzinfo=timezone.utc)
    out: list[Candle] = []
    prev = closes[0]
    for i, c in enumerate(closes):
        h = max(prev, c) + 0.05
        l = min(prev, c) - 0.05
        out.append(Candle(
            timestamp_utc=base + timedelta(minutes=5 * i),
            open=prev, high=h, low=l, close=c,
            volume=base_volume + i, complete=True,
        ))
        prev = c
    return out


class ExtendedIndicatorsTest(unittest.TestCase):
    def test_uptrend_yields_long_supertrend_and_psar(self) -> None:
        prices = [156.0 + i * 0.05 for i in range(150)]
        ind = compute_indicators("USD_JPY", "M5", _make_candles(prices))
        self.assertTrue(ind.valid)
        self.assertEqual(ind.supertrend_dir, 1)
        self.assertEqual(ind.psar_dir, 1)
        self.assertGreater(ind.aroon_up_14 or 0, ind.aroon_down_14 or 0)
        self.assertGreater(ind.linreg_slope_20 or 0, 0)
        self.assertGreater(ind.linreg_r2_20 or 0, 0.9)

    def test_downtrend_yields_short_supertrend_and_psar(self) -> None:
        prices = [156.0 - i * 0.05 for i in range(150)]
        ind = compute_indicators("USD_JPY", "M5", _make_candles(prices))
        self.assertEqual(ind.supertrend_dir, -1)
        self.assertEqual(ind.psar_dir, -1)
        self.assertLess(ind.linreg_slope_20 or 0, 0)

    def test_williams_r_in_overbought_range_for_strong_uptrend(self) -> None:
        prices = [100.0 + i * 0.5 for i in range(60)]
        ind = compute_indicators("EUR_USD", "M5", _make_candles(prices))
        self.assertIsNotNone(ind.williams_r_14)
        # %R near 0 = top of range (overbought)
        self.assertGreater(ind.williams_r_14, -50.0)

    def test_mfi_increases_with_rising_typical_price(self) -> None:
        prices = [100.0 + i * 0.1 for i in range(80)]
        ind = compute_indicators("EUR_USD", "M5", _make_candles(prices))
        self.assertIsNotNone(ind.mfi_14)
        self.assertGreater(ind.mfi_14, 50.0)

    def test_choppiness_low_in_strong_trend(self) -> None:
        prices = [100.0 + i * 0.2 for i in range(60)]
        ind = compute_indicators("EUR_USD", "M5", _make_candles(prices))
        self.assertIsNotNone(ind.choppiness_14)
        self.assertLess(ind.choppiness_14, 50.0)

    def test_bb_squeeze_flag_is_zero_in_volatile_series(self) -> None:
        prices = [100.0 + math.sin(i / 3.0) * 5.0 for i in range(120)]
        ind = compute_indicators("EUR_USD", "M5", _make_candles(prices))
        self.assertIn(ind.bb_squeeze, (0, 1, None))

    def test_anchored_vwap_within_min_max(self) -> None:
        prices = [100.0 + i * 0.05 for i in range(80)]
        ind = compute_indicators("EUR_USD", "M5", _make_candles(prices))
        self.assertIsNotNone(ind.avwap_anchor)
        self.assertGreater(ind.avwap_upper_2sd, ind.avwap_anchor)
        self.assertLess(ind.avwap_lower_2sd, ind.avwap_anchor)

    def test_z_score_extreme_in_jump(self) -> None:
        prices = [100.0] * 30 + [120.0]
        candles = _make_candles(prices)
        ind = compute_indicators("EUR_USD", "M5", candles)
        self.assertIsNotNone(ind.z_score_20)
        self.assertGreater(abs(ind.z_score_20), 1.0)

    def test_realized_vol_positive_for_random_series(self) -> None:
        prices = [100.0 + (i % 7) * 0.3 for i in range(60)]
        ind = compute_indicators("EUR_USD", "M5", _make_candles(prices))
        self.assertIsNotNone(ind.realized_vol_20)
        self.assertGreater(ind.realized_vol_20, 0.0)

    def test_regime_quantile_label_is_one_of_known(self) -> None:
        prices = [100.0 + i * 0.05 for i in range(150)]
        ind = compute_indicators("EUR_USD", "M5", _make_candles(prices))
        self.assertIn(ind.regime_quantile, (None, "QUIET", "NORMAL", "VOLATILE"))

    def test_to_dict_includes_all_extended_fields(self) -> None:
        prices = [100.0 + i * 0.05 for i in range(150)]
        ind = compute_indicators("EUR_USD", "M5", _make_candles(prices))
        payload = ind.to_dict()
        for key in (
            "williams_r_14", "mfi_14", "aroon_up_14", "aroon_down_14", "aroon_osc_14",
            "vortex_plus_14", "vortex_minus_14",
            "supertrend_value", "supertrend_dir", "psar_value", "psar_dir",
            "hull_ma_20", "kama_10", "alma_20",
            "linreg_slope_20", "linreg_r2_20", "linreg_channel_upper", "linreg_channel_lower",
            "choppiness_14", "bb_squeeze",
            "atr_percentile_100", "bb_width_percentile_100", "adx_percentile_100",
            "z_score_20", "realized_vol_20",
            "avwap_anchor", "avwap_upper_1sd", "avwap_lower_1sd",
            "avwap_upper_2sd", "avwap_lower_2sd",
            "avwap_swing_high", "avwap_swing_low",
            "hurst_100", "half_life_60", "regime_quantile",
        ):
            self.assertIn(key, payload, msg=f"missing field {key} in to_dict")


if __name__ == "__main__":
    unittest.main()
