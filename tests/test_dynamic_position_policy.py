"""Unit tests for strategy/dynamic_position_policy.py."""

from __future__ import annotations

import unittest

from quant_rabbit.strategy.dynamic_position_policy import (
    ADVERSE_TRIGGER_BASE,
    ADVERSE_TRIGGER_CEILING,
    ADVERSE_TRIGGER_FLOOR,
    PARTIAL_FRACTION_BASE,
    PARTIAL_FRACTION_CEILING,
    PARTIAL_FRACTION_FLOOR,
    TRAILING_LOCK_BASE,
    TRAILING_LOCK_CEILING,
    TRAILING_LOCK_FLOOR,
    TRAILING_TRIGGER_BASE,
    TRAILING_TRIGGER_CEILING,
    TRAILING_TRIGGER_FLOOR,
    adverse_trigger_mult,
    partial_close_fraction,
    trailing_lock_behind_mult,
    trailing_trigger_mult,
)


class TrailingTriggerTest(unittest.TestCase):
    def test_empty_context_returns_base(self) -> None:
        v, _ = trailing_trigger_mult(None)
        self.assertAlmostEqual(v, TRAILING_TRIGGER_BASE)

    def test_strong_trend_trails_sooner(self) -> None:
        v, _ = trailing_trigger_mult({"h1_adx": 30})
        self.assertLess(v, TRAILING_TRIGGER_BASE)

    def test_choppy_waits_longer(self) -> None:
        v, _ = trailing_trigger_mult({"h1_adx": 12})
        self.assertGreater(v, TRAILING_TRIGGER_BASE)

    def test_session_overlap_trails_sooner(self) -> None:
        v, _ = trailing_trigger_mult({"session_current_tag": "LONDON_NY_OVERLAP"})
        self.assertLess(v, TRAILING_TRIGGER_BASE)

    def test_off_hours_waits_longer(self) -> None:
        v, _ = trailing_trigger_mult({"session_current_tag": "OFF_HOURS"})
        self.assertGreater(v, TRAILING_TRIGGER_BASE)

    def test_clamped_to_bounds(self) -> None:
        v, _ = trailing_trigger_mult({"h1_adx": 30, "session_current_tag": "LONDON_NY_OVERLAP"})
        self.assertGreaterEqual(v, TRAILING_TRIGGER_FLOOR)
        v2, _ = trailing_trigger_mult({"h1_adx": 10, "session_current_tag": "OFF_HOURS"})
        self.assertLessEqual(v2, TRAILING_TRIGGER_CEILING)


class TrailingLockTest(unittest.TestCase):
    def test_high_atr_widens_lock(self) -> None:
        v, _ = trailing_lock_behind_mult({"confluence": {"atr_percentile_24h": 0.85}})
        self.assertGreater(v, TRAILING_LOCK_BASE)

    def test_low_atr_tightens_lock(self) -> None:
        v, _ = trailing_lock_behind_mult({"confluence": {"atr_percentile_24h": 0.15}})
        self.assertLess(v, TRAILING_LOCK_BASE)

    def test_24h_expansion_outlier_tightens_lock(self) -> None:
        v, _ = trailing_lock_behind_mult({"confluence": {
            "range_24h_expansion_ratio": 9.0,
            "range_24h_expansion_outlier": True,
        }})
        self.assertLess(v, TRAILING_LOCK_BASE)

    def test_strong_adx_widens_lock(self) -> None:
        v, _ = trailing_lock_behind_mult({"h1_adx": 30})
        self.assertGreater(v, TRAILING_LOCK_BASE)


class AdverseTriggerTest(unittest.TestCase):
    def test_higher_tf_against_closes_earlier(self) -> None:
        """LONG when higher_tf is LONG_LEAN = WITH us → +0.5.
        LONG when higher_tf is SHORT_LEAN = AGAINST us → -0.5."""
        ctx = {"confluence": {"higher_tf_alignment": "SHORT_LEAN"}}
        v, _ = adverse_trigger_mult(ctx, "LONG")
        self.assertLess(v, ADVERSE_TRIGGER_BASE)

    def test_higher_tf_with_gives_room(self) -> None:
        ctx = {"confluence": {"higher_tf_alignment": "LONG_LEAN"}}
        v, _ = adverse_trigger_mult(ctx, "LONG")
        self.assertGreater(v, ADVERSE_TRIGGER_BASE)

    def test_choppy_regime_closes_earlier(self) -> None:
        ctx = {"h1_adx": 12}
        v, _ = adverse_trigger_mult(ctx, "LONG")
        self.assertLess(v, ADVERSE_TRIGGER_BASE)

    def test_clamped_to_bounds(self) -> None:
        # Stack negative adjusters: against + choppy + sigma exhausted
        ctx = {
            "confluence": {
                "higher_tf_alignment": "SHORT_LEAN",
                "range_24h_expansion_ratio": 9.0,
                "range_24h_expansion_outlier": True,
            },
            "h1_adx": 10,
        }
        v, _ = adverse_trigger_mult(ctx, "LONG")
        self.assertGreaterEqual(v, ADVERSE_TRIGGER_FLOOR)


class PartialFractionTest(unittest.TestCase):
    def test_higher_tf_against_reduces_more(self) -> None:
        ctx = {"confluence": {"higher_tf_alignment": "SHORT_LEAN"}}
        v, _ = partial_close_fraction(ctx, "LONG")
        self.assertGreater(v, PARTIAL_FRACTION_BASE)

    def test_higher_tf_with_keeps_more(self) -> None:
        ctx = {"confluence": {"higher_tf_alignment": "LONG_LEAN"}}
        v, _ = partial_close_fraction(ctx, "LONG")
        self.assertLess(v, PARTIAL_FRACTION_BASE)

    def test_choppy_reduces_more(self) -> None:
        ctx = {"h1_adx": 12}
        v, _ = partial_close_fraction(ctx, "LONG")
        self.assertGreater(v, PARTIAL_FRACTION_BASE)

    def test_clamped_to_ceiling(self) -> None:
        ctx = {"confluence": {"higher_tf_alignment": "SHORT_LEAN"}, "h1_adx": 10}
        v, _ = partial_close_fraction(ctx, "LONG")
        self.assertLessEqual(v, PARTIAL_FRACTION_CEILING)

    def test_clamped_to_floor(self) -> None:
        ctx = {"confluence": {"higher_tf_alignment": "LONG_LEAN"}, "h1_adx": 30}
        v, _ = partial_close_fraction(ctx, "LONG")
        self.assertGreaterEqual(v, PARTIAL_FRACTION_FLOOR)


if __name__ == "__main__":
    unittest.main()
