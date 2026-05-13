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
        # close_confirmed must be serialised on every event so downstream
        # callers (Gate A in gpt_trader) can filter wick-only breaks.
        for ev in out["structure_events"]:
            self.assertIn("close_confirmed", ev)
            self.assertIsInstance(ev["close_confirmed"], bool)

    def test_wick_only_break_marks_close_unconfirmed(self) -> None:
        # Build a fixture where the breaking candle's HIGH exceeds the
        # prior pivot but its CLOSE prints back inside the range. This is
        # the classic stop-hunt pattern that 2026-05-13 AUD_JPY exposed
        # (M15 BOS_UP@114.146 with only a 0.4-pip wick).
        base = datetime(2026, 5, 4, tzinfo=timezone.utc)
        candles: list[Candle] = []

        def push(o: float, h: float, l: float, c: float) -> None:
            candles.append(Candle(
                timestamp_utc=base + timedelta(minutes=5 * len(candles)),
                open=o, high=h, low=l, close=c, volume=1000, complete=True,
            ))

        # First pivot HIGH at index 3 = 101.0 (surrounded by lower highs).
        push(99.5, 99.8, 99.4, 99.6)   # 0
        push(99.7, 100.0, 99.5, 99.9)  # 1
        push(99.9, 100.2, 99.7, 100.1) # 2
        push(100.1, 101.0, 100.0, 100.8)  # 3  <- pivot HIGH (101.0)
        push(100.5, 100.7, 100.2, 100.4)  # 4
        push(100.3, 100.5, 100.0, 100.2)  # 5
        # Second pivot HIGH at index 8 = 101.5 (breaks 101.0). Close ABOVE
        # 101.0 -> close_confirmed True.
        push(100.5, 100.9, 100.3, 100.7)  # 6
        push(100.8, 101.2, 100.6, 101.1)  # 7
        push(101.0, 101.5, 100.8, 101.4)  # 8  <- pivot HIGH (101.5), close > 101.0
        push(101.2, 101.4, 100.9, 101.1)  # 9
        push(100.9, 101.1, 100.6, 100.8)  # 10
        # Third pivot HIGH at index 13 = 101.8 (wick only). Close BELOW
        # the prior pivot 101.5 -> close_confirmed False.
        push(100.8, 101.1, 100.6, 100.9)  # 11
        push(100.9, 101.3, 100.7, 101.1)  # 12
        push(101.1, 101.8, 101.0, 101.2)  # 13 <- new HIGH 101.8 but close 101.2 < 101.5 prior pivot
        push(101.0, 101.2, 100.7, 100.9)  # 14
        push(100.8, 101.0, 100.5, 100.7)  # 15

        r = analyze_structure(candles, pivot_strength=2)
        events = list(r.structure_events)
        self.assertGreaterEqual(len(events), 2)

        # The first UP event (pivot index 8 breaks pivot index 3 at 101.0)
        # is a CHOCH_UP (no prior UP trend) and should be close_confirmed
        # because candle[8].close=101.4 > 101.0.
        first_up = next(
            (e for e in events if "UP" in e.kind and abs(e.broken_pivot_price - 101.0) < 1e-6),
            None,
        )
        self.assertIsNotNone(first_up, f"events={events}")
        self.assertTrue(first_up.close_confirmed)

        # The second UP event (pivot index 13 breaks pivot index 8 at 101.5)
        # is wick-only because candle[13].close=101.2 < 101.5.
        wick_up = next(
            (e for e in events if "UP" in e.kind and abs(e.broken_pivot_price - 101.5) < 1e-6),
            None,
        )
        self.assertIsNotNone(wick_up, f"events={events}")
        self.assertFalse(wick_up.close_confirmed)


if __name__ == "__main__":
    unittest.main()
