"""Unit tests for strategy/reversal_signal.py."""

from __future__ import annotations

import os
import unittest

from quant_rabbit.strategy.reversal_signal import (
    REVERSAL_BONUS,
    REVERSAL_HIGH_PERCENTILE_THRESHOLD,
    REVERSAL_LOW_PERCENTILE_THRESHOLD,
    detect_reversal,
)


def _chart(
    pair: str,
    *,
    pctile: float | None = None,
    chart_story: str = "",
) -> dict:
    return {
        "pair": pair,
        "confluence": {"price_percentile_24h": pctile} if pctile is not None else {},
        "chart_story": chart_story,
    }


class ReversalSignalTest(unittest.TestCase):
    def setUp(self) -> None:
        if "QR_DISABLE_REVERSAL_SIGNAL" in os.environ:
            del os.environ["QR_DISABLE_REVERSAL_SIGNAL"]

    def test_no_chart_returns_none(self) -> None:
        self.assertIsNone(detect_reversal(None, "LONG"))

    def test_no_percentile_returns_none(self) -> None:
        chart = _chart("EUR_USD", chart_story="M15(struct=BOS_UP@1.1700)")
        self.assertIsNone(detect_reversal(chart, "LONG"))

    def test_long_at_extreme_low_with_bos_up_fires(self) -> None:
        chart = _chart(
            "GBP_USD",
            pctile=0.05,  # extreme low
            chart_story="M15(RANGE struct=BOS_UP@1.3500); H4(...)",
        )
        sig = detect_reversal(chart, "LONG")
        self.assertIsNotNone(sig)
        self.assertEqual(sig.side, "LONG")
        self.assertEqual(sig.struct_tf, "M15")
        self.assertEqual(sig.struct_kind, "BOS")
        self.assertAlmostEqual(sig.bonus, REVERSAL_BONUS)

    def test_short_at_extreme_high_with_bos_down_fires(self) -> None:
        chart = _chart(
            "USD_JPY",
            pctile=0.95,
            chart_story="M5(TREND struct=BOS_DOWN@159.50)",
        )
        sig = detect_reversal(chart, "SHORT")
        self.assertIsNotNone(sig)
        self.assertEqual(sig.side, "SHORT")
        self.assertEqual(sig.struct_tf, "M5")

    def test_long_mid_range_no_fire(self) -> None:
        chart = _chart("EUR_USD", pctile=0.5, chart_story="M15(struct=BOS_UP@1.17)")
        self.assertIsNone(detect_reversal(chart, "LONG"))

    def test_long_at_extreme_low_without_struct_no_fire(self) -> None:
        chart = _chart("GBP_USD", pctile=0.05, chart_story="M15(RANGE no_struct)")
        self.assertIsNone(detect_reversal(chart, "LONG"))

    def test_long_at_extreme_low_with_bos_down_no_fire(self) -> None:
        """Bottom but structure prints DOWN = continued downtrend, not reversal."""
        chart = _chart("GBP_USD", pctile=0.05, chart_story="M15(struct=BOS_DOWN@1.34)")
        self.assertIsNone(detect_reversal(chart, "LONG"))

    def test_wick_confirmed_break_does_not_fire(self) -> None:
        """`:wick` suffix means structure was wick-only — must NOT trigger."""
        chart = _chart("GBP_USD", pctile=0.05, chart_story="M15(struct=BOS_UP@1.34:wick)")
        self.assertIsNone(detect_reversal(chart, "LONG"))

    def test_choch_also_qualifies(self) -> None:
        chart = _chart("EUR_USD", pctile=0.10, chart_story="M5(struct=CHOCH_UP@1.17)")
        sig = detect_reversal(chart, "LONG")
        self.assertIsNotNone(sig)
        self.assertEqual(sig.struct_kind, "CHOCH")

    def test_h4_struct_does_not_fire(self) -> None:
        """Only M5/M15 structure events count for short-term reversal."""
        chart = _chart("EUR_USD", pctile=0.10, chart_story="H4(struct=BOS_UP@1.17)")
        self.assertIsNone(detect_reversal(chart, "LONG"))

    def test_kill_switch_disables(self) -> None:
        os.environ["QR_DISABLE_REVERSAL_SIGNAL"] = "1"
        try:
            chart = _chart("GBP_USD", pctile=0.05, chart_story="M15(struct=BOS_UP@1.34)")
            self.assertIsNone(detect_reversal(chart, "LONG"))
        finally:
            del os.environ["QR_DISABLE_REVERSAL_SIGNAL"]


if __name__ == "__main__":
    unittest.main()
