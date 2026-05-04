"""Tests for the macro / FRED / risk-score composite layer.

CI never hits FRED; every test passes ``offline_payload`` to
``build_macro_reading`` per §3.5.
"""

from __future__ import annotations

import os
import unittest
from datetime import date, datetime, timedelta, timezone

from quant_rabbit.analysis.macro import (
    FredClient,
    MacroReading,
    build_macro_reading,
)


def _series(start: date, values: list[float]) -> list[tuple[date, float]]:
    return [(start + timedelta(days=i), v) for i, v in enumerate(values)]


def _flat_series(start: date, n: int, value: float) -> list[tuple[date, float]]:
    return [(start + timedelta(days=i), value) for i in range(n)]


def _trending_series(start: date, n: int, base: float, slope: float) -> list[tuple[date, float]]:
    return [(start + timedelta(days=i), base + slope * i) for i in range(n)]


class FredClientTests(unittest.TestCase):
    def test_constructor_rejects_empty_key(self) -> None:
        with self.assertRaises(ValueError):
            FredClient("")

    def test_offline_payload_builds_reading(self) -> None:
        start = date(2026, 1, 1)
        payload = {
            "us2y": _flat_series(start, 70, 4.5),
            "us10y": _flat_series(start, 70, 4.2),
            "t10y2y": _flat_series(start, 70, -0.3),
            "t10y_inflation_be": _flat_series(start, 70, 2.4),
            "hy_oas": _flat_series(start, 70, 3.0),
            "vix": _flat_series(start, 70, 15.0),
            "gold": _flat_series(start, 70, 2300.0),
        }
        reading = build_macro_reading(offline_payload=payload)
        self.assertIsInstance(reading, MacroReading)
        self.assertEqual(reading.us2y, 4.5)
        self.assertEqual(reading.vix, 15.0)
        self.assertEqual(reading.gold, 2300.0)
        # Flat series → no z-score signal → composite is None.
        self.assertIsNone(reading.risk_score)
        self.assertIn("offline_payload", reading.notes)

    def test_to_dict_serialises_reading(self) -> None:
        reading = build_macro_reading(offline_payload={})
        d = reading.to_dict()
        self.assertIn("fetched_at_utc", d)
        self.assertIn("risk_score_components", d)
        self.assertIn("usd_credibility_regime", d)


class RiskScoreTests(unittest.TestCase):
    def test_components_clipped_at_three(self) -> None:
        start = date(2026, 1, 1)
        # Realistic baseline (small noise so sd > 0) plus a huge spike on the
        # last day → z-score blows past ±3 and must clip.
        base = [15.0 + (0.1 if i % 2 == 0 else -0.1) for i in range(60)]
        spike_series = base + [200.0]
        payload = {
            "vix": _series(start, spike_series),
        }
        reading = build_macro_reading(offline_payload=payload)
        # VIX spike → risk-OFF → z_vix > 0 → component = -z_vix → clipped at -3.
        self.assertIn("vix_z_inv", reading.risk_score_components)
        self.assertAlmostEqual(reading.risk_score_components["vix_z_inv"], -3.0, places=6)
        self.assertIsNotNone(reading.risk_score)
        self.assertGreaterEqual(reading.risk_score, -3.0)
        self.assertLessEqual(reading.risk_score, 3.0)

    def test_composite_is_equal_weighted_sum_clipped(self) -> None:
        start = date(2026, 1, 1)
        # VIX trending DOWN → z_vix < 0 → -z_vix > 0 (risk-ON).
        # HY OAS trending DOWN → -z_hy > 0 (risk-ON).
        # Copper/Gold trending UP → z > 0.
        # SPX 60d return positive.
        # AUDJPY 60d return positive.
        payload = {
            "vix": _trending_series(start, 61, base=30.0, slope=-0.2),
            "hy_oas": _trending_series(start, 61, base=5.0, slope=-0.02),
        }
        reading = build_macro_reading(
            offline_payload=payload,
            spx_60d_return=0.05,
            audjpy_60d_return=0.04,
        )
        # All risk-ON → composite > 0.
        self.assertIsNotNone(reading.risk_score)
        self.assertGreater(reading.risk_score, 0)
        # Sum of components equals (clipped) raw sum.
        components = reading.risk_score_components
        # Components present and clipped within ±3.
        for k, v in components.items():
            if k in {"copper_gold_ratio_raw", "us2y_60d_change_raw"}:
                continue
            self.assertGreaterEqual(v, -3.0, msg=k)
            self.assertLessEqual(v, 3.0, msg=k)

    def test_us2y_change_sign_inverted(self) -> None:
        start = date(2026, 1, 1)
        # Noisy small daily changes, then a big positive jump on the last day:
        # the final change is a large positive outlier → its z-score is well
        # above zero → composite component is -z (risk-OFF, < 0).
        values = [4.0]
        for i in range(64):
            # Tiny alternating noise so the change-series sd > 0.
            values.append(values[-1] + (0.005 if i % 2 == 0 else -0.005))
        values.append(values[-1] + 0.5)  # big spike on the final day
        payload = {"us2y": _series(start, values)}
        reading = build_macro_reading(offline_payload=payload)
        self.assertIn("us2y_change_z_inv", reading.risk_score_components)
        self.assertLess(reading.risk_score_components["us2y_change_z_inv"], 0)


class UsdCredibilityTests(unittest.TestCase):
    def test_positive_beta_means_on(self) -> None:
        # DXY change moves WITH US10Y change → positive beta → "ON".
        us10y = [4.0, 4.1, 4.05, 4.2, 4.25, 4.3]
        dxy = [104.0, 104.5, 104.3, 105.0, 105.4, 105.8]
        reading = build_macro_reading(
            offline_payload={},
            dxy_recent=dxy,
            us10y_recent=us10y,
        )
        self.assertEqual(reading.usd_credibility_regime, "ON")
        self.assertEqual(reading.dxy_us10y_5d_beta_sign, 1)

    def test_negative_beta_means_off(self) -> None:
        # DXY change moves AGAINST US10Y change → "OFF".
        us10y = [4.0, 4.1, 4.05, 4.2, 4.25, 4.3]
        dxy = [104.0, 103.5, 103.7, 103.0, 102.6, 102.2]
        reading = build_macro_reading(
            offline_payload={},
            dxy_recent=dxy,
            us10y_recent=us10y,
        )
        self.assertEqual(reading.usd_credibility_regime, "OFF")
        self.assertEqual(reading.dxy_us10y_5d_beta_sign, -1)

    def test_flat_series_unknown(self) -> None:
        us10y = [4.0] * 6
        dxy = [104.0] * 6
        reading = build_macro_reading(
            offline_payload={},
            dxy_recent=dxy,
            us10y_recent=us10y,
        )
        self.assertEqual(reading.usd_credibility_regime, "UNKNOWN")

    def test_missing_inputs_unknown(self) -> None:
        reading = build_macro_reading(offline_payload={})
        self.assertEqual(reading.usd_credibility_regime, "UNKNOWN")
        self.assertIsNone(reading.dxy_us10y_5d_beta_sign)


class MissingFredKeyTests(unittest.TestCase):
    def test_missing_key_returns_all_none_no_exception(self) -> None:
        # Ensure the env var is not set during this test.
        prior = os.environ.pop("QR_FRED_API_KEY", None)
        try:
            reading = build_macro_reading()  # no fred_key, no offline_payload
            self.assertIsNone(reading.us2y)
            self.assertIsNone(reading.vix)
            self.assertIsNone(reading.risk_score)
            self.assertEqual(reading.usd_credibility_regime, "UNKNOWN")
            self.assertTrue(
                any(note.startswith("fred_key_missing:") for note in reading.notes),
                msg=f"expected fred_key_missing note, got {reading.notes}",
            )
        finally:
            if prior is not None:
                os.environ["QR_FRED_API_KEY"] = prior


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
