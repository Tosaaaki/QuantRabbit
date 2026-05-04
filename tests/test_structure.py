"""Tests for the SMC / price-action structural analyzer."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from quant_rabbit.analysis.candles import Candle
from quant_rabbit.analysis.structure import analyze_structure


def _make(values: list[tuple[float, float, float]]) -> list[Candle]:
    """values = [(open, high, low) ...]; close = midpoint of high/low for simplicity."""
    base = datetime(2026, 5, 4, tzinfo=timezone.utc)
    out: list[Candle] = []
    for i, (o, h, l) in enumerate(values):
        c = (h + l) / 2.0
        out.append(Candle(
            timestamp_utc=base + timedelta(minutes=5 * i),
            open=o, high=h, low=l, close=c, volume=1000, complete=True,
        ))
    return out


class StructureTest(unittest.TestCase):
    def test_short_series_returns_empty_reading(self) -> None:
        candles = _make([(100.0, 100.5, 99.5)] * 5)
        r = analyze_structure(candles, pivot_strength=3)
        self.assertEqual(len(r.swings), 0)
        self.assertIsNone(r.last_event)

    def test_uptrend_produces_bos_up_events(self) -> None:
        # Construct alternating swings stepping up
        candles: list[Candle] = []
        base_price = 100.0
        for i in range(40):
            wave = (i % 8) - 4  # -4..3
            o = base_price + i * 0.1
            h = o + max(0.0, wave) * 0.1 + 0.05
            l = o - max(0.0, -wave) * 0.1 - 0.05
            candles.append(Candle(
                timestamp_utc=datetime(2026, 5, 4, tzinfo=timezone.utc) + timedelta(minutes=5 * i),
                open=o, high=h, low=l, close=(h + l) / 2.0,
                volume=1000, complete=True,
            ))
        r = analyze_structure(candles, pivot_strength=2)
        # Expect at least one BOS_UP somewhere
        kinds = {ev.kind for ev in r.structure_events}
        self.assertTrue(any("UP" in k for k in kinds))

    def test_fvg_detected_in_clear_gap(self) -> None:
        candles = _make([(100.0, 100.5, 99.5)] * 30)
        # Inject an upward gap: candle[i] low > candle[i-2] high
        candles[20] = Candle(
            timestamp_utc=candles[20].timestamp_utc, open=102.0, high=103.0, low=101.5,
            close=102.5, volume=1000, complete=True,
        )
        candles[21] = Candle(
            timestamp_utc=candles[21].timestamp_utc, open=102.7, high=103.2, low=102.5,
            close=102.9, volume=1000, complete=True,
        )
        r = analyze_structure(candles, pivot_strength=3)
        ups = [f for f in r.fair_value_gaps if f.direction == "UP"]
        self.assertGreaterEqual(len(ups), 1)

    def test_to_dict_round_trips_keys(self) -> None:
        candles = _make([(100.0 + i * 0.01, 100.0 + i * 0.01 + 0.1, 100.0 + i * 0.01 - 0.1) for i in range(60)])
        r = analyze_structure(candles, pivot_strength=3)
        out = r.to_dict()
        self.assertIn("swings", out)
        self.assertIn("structure_events", out)
        self.assertIn("order_blocks", out)
        self.assertIn("fair_value_gaps", out)
        self.assertIn("liquidity", out)


if __name__ == "__main__":
    unittest.main()
