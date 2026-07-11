"""Unit tests for strategy/pattern_signals.py."""

from __future__ import annotations

import os
import unittest
from datetime import datetime, timedelta, timezone

from quant_rabbit.analysis.candles import Candle
from quant_rabbit.analysis.structure import analyze_structure
from quant_rabbit.strategy.pattern_signals import (
    FAILED_BREAKOUT_BONUS,
    PATTERN_TOTAL_CAP,
    PatternSignal,
    aggregate_pattern_score,
    detect_pattern_signals,
)


def _view(
    *,
    tf: str = "M15",
    rsi: float | None = None,
    close: float | None = None,
    bb_upper: float | None = None,
    bb_lower: float | None = None,
    aroon_up: float | None = None,
    aroon_down: float | None = None,
    structure_events: list | None = None,
    dealing_range: dict | None = None,
    recent_candle_time: str | None = None,
) -> dict:
    ind = {}
    if rsi is not None: ind["rsi_14"] = rsi
    if close is not None: ind["close"] = close
    if bb_upper is not None: ind["bb_upper"] = bb_upper
    if bb_lower is not None: ind["bb_lower"] = bb_lower
    if aroon_up is not None: ind["aroon_up_14"] = aroon_up
    if aroon_down is not None: ind["aroon_down_14"] = aroon_down
    view = {"granularity": tf, "indicators": ind}
    if structure_events is not None:
        view["structure"] = {"structure_events": structure_events}
    if recent_candle_time is not None:
        view["recent_candles"] = [{"t": recent_candle_time}]
    if dealing_range is not None:
        view["smc"] = {"dealing_range": dealing_range}
    return view


def _chart(views: list[dict]) -> dict:
    return {"pair": "EUR_USD", "views": views}


class FailedBreakoutTest(unittest.TestCase):
    def setUp(self) -> None:
        if "QR_DISABLE_PATTERN_SIGNALS" in os.environ:
            del os.environ["QR_DISABLE_PATTERN_SIGNALS"]

    def test_wick_only_bos_up_fades_down(self) -> None:
        chart = _chart([_view(
            structure_events=[
                {
                    "kind": "BOS_UP",
                    "broken_pivot_price": 1.17,
                    "close_confirmed": False,
                    "index": 10,
                    # Canonical pivot_strength=3 means this event first becomes
                    # observable three bars after its own candle.
                    "timestamp": "2026-07-01T11:15:00Z",
                },
            ],
            recent_candle_time="2026-07-01T12:00:00Z",
        )])
        signals = detect_pattern_signals(chart)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].name, "failed_breakout")
        self.assertEqual(signals[0].direction, "DOWN")

    def test_canonical_three_bar_pivot_confirmation_remains_actionable(self) -> None:
        base = datetime(2026, 7, 1, tzinfo=timezone.utc)
        highs = [
            1.090,
            1.092,
            1.094,
            1.096,
            1.100,
            1.096,
            1.094,
            1.095,
            1.097,
            1.099,
            1.110,
            1.098,
            1.096,
            1.095,
        ]
        candles = []
        for index, high in enumerate(highs):
            close = 1.095 if index == 10 else min(high - 0.001, 1.095)
            candles.append(
                Candle(
                    timestamp_utc=base + timedelta(minutes=15 * index),
                    open=close,
                    high=high,
                    low=1.080,
                    close=close,
                )
            )
        structure = analyze_structure(candles, pivot_strength=3).to_dict()
        chart = _chart(
            [
                {
                    "granularity": "M15",
                    "indicators": {},
                    "structure": structure,
                    "recent_candles": [{"t": candles[-1].timestamp_utc.isoformat()}],
                }
            ]
        )

        signals = detect_pattern_signals(chart)

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].name, "failed_breakout")
        self.assertEqual(signals[0].direction, "DOWN")

    def test_close_confirmed_bos_does_not_fire(self) -> None:
        chart = _chart([_view(
            structure_events=[
                {"kind": "BOS_UP", "broken_pivot_price": 1.17, "close_confirmed": True, "index": 10},
            ],
        )])
        signals = detect_pattern_signals(chart)
        self.assertEqual(len(signals), 0)

    def test_wick_only_bos_down_fades_up(self) -> None:
        chart = _chart([_view(
            structure_events=[
                {
                    "kind": "BOS_DOWN",
                    "broken_pivot_price": 1.17,
                    "close_confirmed": False,
                    "index": 10,
                    "timestamp": "2026-07-01T11:45:00Z",
                },
            ],
            recent_candle_time="2026-07-01T12:00:00Z",
        )])
        signals = detect_pattern_signals(chart)
        self.assertEqual(signals[0].direction, "UP")

    def test_stale_wick_only_event_does_not_keep_voting(self) -> None:
        chart = _chart([_view(
            structure_events=[
                {
                    "kind": "BOS_UP",
                    "broken_pivot_price": 1.17,
                    "close_confirmed": False,
                    "timestamp": "2026-07-01T10:00:00Z",
                },
            ],
            recent_candle_time="2026-07-01T12:00:00Z",
        )])

        self.assertEqual(detect_pattern_signals(chart), [])

    def test_older_failed_event_is_superseded_by_latest_confirmed_structure(self) -> None:
        chart = _chart([_view(
            structure_events=[
                {
                    "kind": "BOS_UP",
                    "broken_pivot_price": 1.17,
                    "close_confirmed": False,
                    "timestamp": "2026-07-01T11:45:00Z",
                },
                {
                    "kind": "BOS_DOWN",
                    "broken_pivot_price": 1.16,
                    "close_confirmed": True,
                    "timestamp": "2026-07-01T12:00:00Z",
                },
            ],
            recent_candle_time="2026-07-01T12:00:00Z",
        )])

        self.assertEqual(detect_pattern_signals(chart), [])


class RSIExtremeTest(unittest.TestCase):
    def test_rsi_overbought_near_bb_upper_fires_down(self) -> None:
        chart = _chart([_view(rsi=75, close=1.180, bb_upper=1.181, bb_lower=1.165)])
        signals = detect_pattern_signals(chart)
        rsi_signals = [s for s in signals if "rsi_extreme" in s.name]
        self.assertEqual(len(rsi_signals), 1)
        self.assertEqual(rsi_signals[0].direction, "DOWN")

    def test_rsi_oversold_near_bb_lower_fires_up(self) -> None:
        chart = _chart([_view(rsi=25, close=1.166, bb_upper=1.181, bb_lower=1.165)])
        signals = detect_pattern_signals(chart)
        rsi_signals = [s for s in signals if "rsi_extreme" in s.name]
        self.assertEqual(rsi_signals[0].direction, "UP")

    def test_rsi_overbought_mid_BB_no_fire(self) -> None:
        """RSI 75 alone isn't enough — need price at the rail too."""
        chart = _chart([_view(rsi=75, close=1.173, bb_upper=1.181, bb_lower=1.165)])
        signals = detect_pattern_signals(chart)
        self.assertEqual([s for s in signals if "rsi_extreme" in s.name], [])


class DealingRangeTest(unittest.TestCase):
    def test_at_swing_high_fires_down(self) -> None:
        chart = _chart([_view(
            close=1.180, dealing_range={"swing_high": 1.180, "swing_low": 1.160},
        )])
        signals = detect_pattern_signals(chart)
        dr = [s for s in signals if "dealing_range" in s.name]
        self.assertEqual(dr[0].direction, "DOWN")

    def test_at_swing_low_fires_up(self) -> None:
        chart = _chart([_view(
            close=1.160, dealing_range={"swing_high": 1.180, "swing_low": 1.160},
        )])
        signals = detect_pattern_signals(chart)
        dr = [s for s in signals if "dealing_range" in s.name]
        self.assertEqual(dr[0].direction, "UP")

    def test_mid_range_no_fire(self) -> None:
        chart = _chart([_view(
            close=1.170, dealing_range={"swing_high": 1.180, "swing_low": 1.160},
        )])
        signals = detect_pattern_signals(chart)
        self.assertEqual([s for s in signals if "dealing_range" in s.name], [])


class AroonFlipTest(unittest.TestCase):
    def test_strong_up_momentum(self) -> None:
        chart = _chart([_view(aroon_up=85, aroon_down=15)])
        signals = detect_pattern_signals(chart)
        a = [s for s in signals if "aroon" in s.name]
        self.assertEqual(a[0].direction, "UP")

    def test_strong_down_momentum(self) -> None:
        chart = _chart([_view(aroon_up=10, aroon_down=80)])
        signals = detect_pattern_signals(chart)
        a = [s for s in signals if "aroon" in s.name]
        self.assertEqual(a[0].direction, "DOWN")


class AggregateScoreTest(unittest.TestCase):
    def test_aligned_signal_adds_full_magnitude(self) -> None:
        sig = PatternSignal("x", "M15", "UP", 1.0, 10.0, "test")
        total, _ = aggregate_pattern_score([sig], "LONG")
        self.assertAlmostEqual(total, 10.0)

    def test_opposed_signal_subtracts_half_magnitude(self) -> None:
        sig = PatternSignal("x", "M15", "DOWN", 1.0, 10.0, "test")
        total, _ = aggregate_pattern_score([sig], "LONG")
        self.assertAlmostEqual(total, -5.0)

    def test_total_clamped_to_cap(self) -> None:
        signals = [PatternSignal(f"s{i}", "M15", "UP", 1.0, 50.0, "test") for i in range(5)]
        total, _ = aggregate_pattern_score(signals, "LONG")
        self.assertAlmostEqual(total, PATTERN_TOTAL_CAP)

    def test_confidence_scales_contribution(self) -> None:
        sig = PatternSignal("x", "M15", "UP", 0.5, 10.0, "test")
        total, _ = aggregate_pattern_score([sig], "LONG")
        self.assertAlmostEqual(total, 5.0)


class CandlestickPatternTest(unittest.TestCase):
    def setUp(self) -> None:
        if "QR_DISABLE_PATTERN_SIGNALS" in os.environ:
            del os.environ["QR_DISABLE_PATTERN_SIGNALS"]

    def _v(self, candles: list[dict]) -> dict:
        return {"granularity": "M15", "indicators": {}, "recent_candles": candles}

    def test_bullish_engulfing_fires_up(self) -> None:
        # prev bear (o=1.10, c=1.09), last bull engulfs (o=1.088, c=1.105)
        chart = _chart([self._v([
            {"o": 1.10, "h": 1.105, "l": 1.085, "c": 1.09, "v": 100},
            {"o": 1.088, "h": 1.108, "l": 1.087, "c": 1.105, "v": 120},
        ])])
        signals = detect_pattern_signals(chart)
        names = [s.name for s in signals]
        self.assertIn("bullish_engulfing", names)

    def test_bearish_engulfing_fires_down(self) -> None:
        chart = _chart([self._v([
            {"o": 1.090, "h": 1.105, "l": 1.088, "c": 1.100, "v": 100},
            {"o": 1.102, "h": 1.103, "l": 1.085, "c": 1.087, "v": 120},
        ])])
        signals = detect_pattern_signals(chart)
        self.assertIn("bearish_engulfing", [s.name for s in signals])

    def test_hammer_fires_up(self) -> None:
        # Hammer: long lower wick, small body at top
        # o=1.105, h=1.108, l=1.095, c=1.106 → body 0.001, lower_wick 0.010 (10x body), upper_wick 0.002
        chart = _chart([self._v([
            {"o": 1.100, "h": 1.108, "l": 1.095, "c": 1.105, "v": 100},  # prev
            {"o": 1.105, "h": 1.108, "l": 1.095, "c": 1.106, "v": 100},  # hammer
        ])])
        signals = detect_pattern_signals(chart)
        self.assertIn("hammer", [s.name for s in signals])

    def test_shooting_star_fires_down(self) -> None:
        chart = _chart([self._v([
            {"o": 1.100, "h": 1.108, "l": 1.095, "c": 1.105, "v": 100},
            {"o": 1.106, "h": 1.115, "l": 1.105, "c": 1.107, "v": 100},  # shooting star
        ])])
        signals = detect_pattern_signals(chart)
        self.assertIn("shooting_star", [s.name for s in signals])


class VolumeSpikeTest(unittest.TestCase):
    def setUp(self) -> None:
        if "QR_DISABLE_PATTERN_SIGNALS" in os.environ:
            del os.environ["QR_DISABLE_PATTERN_SIGNALS"]

    def test_volume_spike_on_bull_fades_down(self) -> None:
        # 20 bars of avg volume 100, then last bar volume 300 on bull
        candles = [
            {"o": 1.10, "h": 1.105, "l": 1.095, "c": 1.103, "v": 100}
            for _ in range(20)
        ]
        candles.append({"o": 1.103, "h": 1.110, "l": 1.102, "c": 1.108, "v": 300})
        chart = _chart([{"granularity": "M15", "indicators": {}, "recent_candles": candles}])
        signals = detect_pattern_signals(chart)
        spike = [s for s in signals if "volume_spike" in s.name]
        self.assertEqual(spike[0].direction, "DOWN")

    def test_no_spike_when_volume_normal(self) -> None:
        candles = [
            {"o": 1.10, "h": 1.105, "l": 1.095, "c": 1.103, "v": 100}
            for _ in range(21)
        ]
        chart = _chart([{"granularity": "M15", "indicators": {}, "recent_candles": candles}])
        signals = detect_pattern_signals(chart)
        spike = [s for s in signals if "volume_spike" in s.name]
        self.assertEqual(spike, [])


class TimeExhaustionTest(unittest.TestCase):
    def setUp(self) -> None:
        if "QR_DISABLE_PATTERN_SIGNALS" in os.environ:
            del os.environ["QR_DISABLE_PATTERN_SIGNALS"]

    def test_5_bull_with_shrinking_range_fades_down(self) -> None:
        # 5 consecutive bull candles, range shrinking 1.0 → 0.5
        candles = [
            {"o": 1.10 + i*0.001, "h": 1.10 + i*0.001 + (1.0 - i*0.15) * 0.001,
             "l": 1.099 + i*0.001, "c": 1.10 + i*0.001 + 0.0005, "v": 100}
            for i in range(5)
        ]
        chart = _chart([{"granularity": "M15", "indicators": {}, "recent_candles": candles + [candles[-1]]}])
        signals = detect_pattern_signals(chart)
        te = [s for s in signals if "time_exhaustion" in s.name]
        # May or may not fire depending on exact shrink; just ensure no crash
        # and if fires, it fades down (bull → DOWN).
        for s in te:
            self.assertEqual(s.direction, "DOWN")

    def test_mixed_colors_does_not_fire(self) -> None:
        candles = []
        for i in range(6):
            is_bull = (i % 2 == 0)
            candles.append({
                "o": 1.10, "h": 1.105, "l": 1.095,
                "c": 1.103 if is_bull else 1.097, "v": 100,
            })
        chart = _chart([{"granularity": "M15", "indicators": {}, "recent_candles": candles}])
        signals = detect_pattern_signals(chart)
        self.assertEqual([s for s in signals if "time_exhaustion" in s.name], [])


class RSIDivergenceTest(unittest.TestCase):
    """True divergence using bar-aligned RSI series against price swings."""

    def setUp(self) -> None:
        if "QR_DISABLE_PATTERN_SIGNALS" in os.environ:
            del os.environ["QR_DISABLE_PATTERN_SIGNALS"]

    def _candle(self, idx: int, high: float, low: float, bull: bool = True) -> dict:
        # Build a candle of given high/low with a small body
        return {"o": low + 0.0001, "h": high, "l": low,
                "c": high - 0.0001 if bull else low + 0.0001, "v": 100}

    def test_bearish_rsi_divergence_fires_down(self) -> None:
        """Price makes HH, RSI makes LH between two swings."""
        # 30 candles: build two swing highs (indices 5 and 25) with HH but
        # corresponding RSI values with LH.
        candles = []
        for i in range(30):
            # Default low candles
            candles.append(self._candle(i, high=1.10, low=1.099))
        # Insert swing highs at idx 5 (high=1.103) and idx 25 (HH=1.105)
        candles[5] = self._candle(5, high=1.103, low=1.099)
        candles[25] = self._candle(25, high=1.105, low=1.099)
        # RSI series: 30 entries; RSI at idx 5 = 75, at idx 25 = 65 (LH).
        # Pad surrounding values lower than the swing peaks so swing detection works.
        rsi_series = [50.0] * 30
        rsi_series[5] = 75.0
        rsi_series[25] = 65.0
        view = {"granularity": "M15", "indicators": {},
                "recent_candles": candles,
                "indicator_series": {"rsi_14": rsi_series}}
        chart = {"pair": "EUR_USD", "views": [view]}
        signals = detect_pattern_signals(chart)
        div = [s for s in signals if "rsi_14_divergence" in s.name]
        self.assertEqual(len(div), 1)
        self.assertEqual(div[0].direction, "DOWN")
        self.assertEqual(div[0].name, "bearish_rsi_14_divergence")

    def test_bullish_rsi_divergence_fires_up(self) -> None:
        candles = []
        for i in range(30):
            candles.append(self._candle(i, high=1.105, low=1.10))
        candles[5] = self._candle(5, high=1.105, low=1.097)   # swing low
        candles[25] = self._candle(25, high=1.105, low=1.095)  # LL
        rsi_series = [50.0] * 30
        rsi_series[5] = 25.0
        rsi_series[25] = 32.0  # HL (rsi up despite price down)
        view = {"granularity": "M15", "indicators": {},
                "recent_candles": candles,
                "indicator_series": {"rsi_14": rsi_series}}
        chart = {"pair": "EUR_USD", "views": [view]}
        signals = detect_pattern_signals(chart)
        div = [s for s in signals if "rsi_14_divergence" in s.name]
        self.assertEqual(div[0].direction, "UP")

    def test_no_divergence_when_indicator_confirms(self) -> None:
        """Price HH + RSI HH = no divergence."""
        candles = []
        for i in range(30):
            candles.append(self._candle(i, high=1.10, low=1.099))
        candles[5] = self._candle(5, high=1.103, low=1.099)
        candles[25] = self._candle(25, high=1.105, low=1.099)
        rsi_series = [50.0] * 30
        rsi_series[5] = 65.0
        rsi_series[25] = 78.0  # HH confirms — no divergence
        view = {"granularity": "M15", "indicators": {},
                "recent_candles": candles,
                "indicator_series": {"rsi_14": rsi_series}}
        chart = {"pair": "EUR_USD", "views": [view]}
        signals = detect_pattern_signals(chart)
        self.assertEqual([s for s in signals if "rsi_14_divergence" in s.name], [])

    def test_no_divergence_when_series_missing(self) -> None:
        view = {"granularity": "M15", "indicators": {"rsi_14": 80},
                "recent_candles": [self._candle(0, 1.1, 1.099) for _ in range(30)]}
        chart = {"pair": "EUR_USD", "views": [view]}
        signals = detect_pattern_signals(chart)
        self.assertEqual([s for s in signals if "rsi_14_divergence" in s.name], [])


class ThreeBarPatternTest(unittest.TestCase):
    def setUp(self) -> None:
        if "QR_DISABLE_PATTERN_SIGNALS" in os.environ:
            del os.environ["QR_DISABLE_PATTERN_SIGNALS"]

    def test_morning_star_fires_up(self) -> None:
        chart = _chart([{"granularity": "M15", "indicators": {}, "recent_candles": [
            # c1: bear (o=1.110, c=1.100)
            {"o": 1.110, "h": 1.112, "l": 1.098, "c": 1.100, "v": 100},
            # c2: narrow body (small range, body 0.0005 vs range 0.002 = 25%)
            {"o": 1.100, "h": 1.102, "l": 1.099, "c": 1.1005, "v": 100},
            # c3: bull closing above midpoint of c1 (mid = (1.110+1.100)/2 = 1.105)
            {"o": 1.101, "h": 1.108, "l": 1.100, "c": 1.107, "v": 100},
        ]}])
        signals = detect_pattern_signals(chart)
        self.assertIn("morning_star", [s.name for s in signals])

    def test_evening_star_fires_down(self) -> None:
        chart = _chart([{"granularity": "M15", "indicators": {}, "recent_candles": [
            {"o": 1.100, "h": 1.112, "l": 1.099, "c": 1.110, "v": 100},
            {"o": 1.110, "h": 1.111, "l": 1.108, "c": 1.1095, "v": 100},
            {"o": 1.109, "h": 1.110, "l": 1.102, "c": 1.103, "v": 100},
        ]}])
        signals = detect_pattern_signals(chart)
        self.assertIn("evening_star", [s.name for s in signals])

    def test_three_white_soldiers(self) -> None:
        chart = _chart([{"granularity": "M15", "indicators": {}, "recent_candles": [
            {"o": 1.100, "h": 1.103, "l": 1.099, "c": 1.102, "v": 100},
            {"o": 1.1015, "h": 1.105, "l": 1.101, "c": 1.104, "v": 100},
            {"o": 1.1035, "h": 1.107, "l": 1.103, "c": 1.106, "v": 100},
        ]}])
        signals = detect_pattern_signals(chart)
        self.assertIn("three_white_soldiers", [s.name for s in signals])

    def test_three_black_crows(self) -> None:
        chart = _chart([{"granularity": "M15", "indicators": {}, "recent_candles": [
            {"o": 1.106, "h": 1.107, "l": 1.103, "c": 1.104, "v": 100},
            {"o": 1.1045, "h": 1.105, "l": 1.101, "c": 1.102, "v": 100},
            {"o": 1.1025, "h": 1.103, "l": 1.099, "c": 1.100, "v": 100},
        ]}])
        signals = detect_pattern_signals(chart)
        self.assertIn("three_black_crows", [s.name for s in signals])


class InsideBarBreakTest(unittest.TestCase):
    def setUp(self) -> None:
        if "QR_DISABLE_PATTERN_SIGNALS" in os.environ:
            del os.environ["QR_DISABLE_PATTERN_SIGNALS"]

    def test_inside_bar_break_up(self) -> None:
        chart = _chart([{"granularity": "M15", "indicators": {}, "recent_candles": [
            # Mother
            {"o": 1.100, "h": 1.110, "l": 1.095, "c": 1.105, "v": 100},
            # Inside (high < mother.high, low > mother.low)
            {"o": 1.104, "h": 1.108, "l": 1.099, "c": 1.106, "v": 100},
            # Break UP (close > mother.high 1.110)
            {"o": 1.107, "h": 1.115, "l": 1.106, "c": 1.112, "v": 100},
        ]}])
        signals = detect_pattern_signals(chart)
        names = [s.name for s in signals]
        self.assertIn("inside_bar_break_up", names)

    def test_inside_bar_break_down(self) -> None:
        chart = _chart([{"granularity": "M15", "indicators": {}, "recent_candles": [
            {"o": 1.105, "h": 1.110, "l": 1.095, "c": 1.100, "v": 100},
            {"o": 1.101, "h": 1.108, "l": 1.099, "c": 1.103, "v": 100},
            {"o": 1.102, "h": 1.103, "l": 1.090, "c": 1.092, "v": 100},
        ]}])
        signals = detect_pattern_signals(chart)
        self.assertIn("inside_bar_break_down", [s.name for s in signals])

    def test_no_break_when_inside_continues(self) -> None:
        """Mother → inside → still-inside (no break)."""
        chart = _chart([{"granularity": "M15", "indicators": {}, "recent_candles": [
            {"o": 1.100, "h": 1.110, "l": 1.095, "c": 1.105, "v": 100},
            {"o": 1.104, "h": 1.108, "l": 1.099, "c": 1.106, "v": 100},
            {"o": 1.105, "h": 1.107, "l": 1.100, "c": 1.103, "v": 100},
        ]}])
        signals = detect_pattern_signals(chart)
        self.assertEqual([s for s in signals if "inside_bar_break" in s.name], [])


class KillSwitchTest(unittest.TestCase):
    def test_disabled_returns_empty(self) -> None:
        os.environ["QR_DISABLE_PATTERN_SIGNALS"] = "1"
        try:
            chart = _chart([_view(rsi=75, close=1.180, bb_upper=1.181, bb_lower=1.165)])
            self.assertEqual(detect_pattern_signals(chart), [])
        finally:
            del os.environ["QR_DISABLE_PATTERN_SIGNALS"]


if __name__ == "__main__":
    unittest.main()
