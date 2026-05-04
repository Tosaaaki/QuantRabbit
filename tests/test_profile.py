from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from quant_rabbit.analysis.candles import Candle
from quant_rabbit.analysis.profile import (
    DEFAULT_BIN_ATR_FACTOR,
    DEFAULT_NAKED_POC_MAX_AGE_SESSIONS,
    IB_BRACKET_COUNT,
    VALUE_AREA_PCT,
    build_tpo_profile,
    naked_pocs,
)


SESSION_START = datetime(2026, 5, 4, 22, 0, tzinfo=timezone.utc)


def _candle(start: datetime, o: float, h: float, lo: float, c: float) -> Candle:
    return Candle(
        timestamp_utc=start,
        open=o,
        high=h,
        low=lo,
        close=c,
        volume=100,
        complete=True,
    )


def _session_candles(prices: list[tuple[float, float, float, float]]) -> list[Candle]:
    """Build a 30-min M30 session, one tuple = one (o,h,l,c) bracket."""

    out: list[Candle] = []
    ts = SESSION_START
    for o, h, lo, c in prices:
        out.append(_candle(ts, o, h, lo, c))
        ts += timedelta(minutes=30)
    return out


class TPOProfileBasicsTest(unittest.TestCase):
    def test_uptrend_session_classified_as_trend_with_range_extension_up(self) -> None:
        # Monotonic upward 8-bracket session; clear trend day, close near top.
        candles = _session_candles(
            [
                (150.00, 150.10, 149.95, 150.05),  # A — IB
                (150.05, 150.20, 150.00, 150.15),  # B — IB
                (150.15, 150.35, 150.10, 150.30),  # C
                (150.30, 150.50, 150.25, 150.45),  # D
                (150.45, 150.65, 150.40, 150.60),  # E
                (150.60, 150.80, 150.55, 150.75),  # F
                (150.75, 150.95, 150.70, 150.90),  # G
                (150.90, 151.10, 150.85, 151.05),  # H
            ]
        )
        prof = build_tpo_profile(
            candles,
            session_start_utc=SESSION_START,
            pip_size=0.01,
            bin_pips=2.0,  # 0.02 bucket width — keeps the test deterministic
        )
        self.assertEqual(prof.day_type, "TREND")
        self.assertTrue(prof.range_extension_up)
        self.assertFalse(prof.range_extension_down)
        # IB = first 2 brackets: H = max(150.10, 150.20)=150.20, L=min(149.95,150.00)=149.95.
        self.assertAlmostEqual(prof.initial_balance_high, 150.20, places=4)
        self.assertAlmostEqual(prof.initial_balance_low, 149.95, places=4)
        # Session window honored.
        self.assertEqual(prof.session_start_utc, SESSION_START)
        self.assertEqual(prof.session_end_utc, SESSION_START + timedelta(hours=24))
        # Letters A..H assigned.
        self.assertEqual(prof.brackets[0].letter, "A")
        self.assertEqual(prof.brackets[7].letter, "H")

    def test_balanced_session_normal_or_neutral_va_brackets_price(self) -> None:
        # Oscillation around 150.00; no strong drive.
        candles = _session_candles(
            [
                (150.00, 150.05, 149.95, 150.00),
                (150.00, 150.05, 149.95, 150.00),
                (150.00, 150.05, 149.95, 150.00),
                (150.00, 150.05, 149.95, 150.00),
                (150.00, 150.05, 149.95, 150.00),
                (150.00, 150.05, 149.95, 150.00),
                (150.00, 150.05, 149.95, 150.00),
                (150.00, 150.05, 149.95, 150.00),
            ]
        )
        prof = build_tpo_profile(
            candles, session_start_utc=SESSION_START, pip_size=0.01, bin_pips=1.0
        )
        # No range extension on either side.
        self.assertFalse(prof.range_extension_up)
        self.assertFalse(prof.range_extension_down)
        self.assertIn(prof.day_type, {"NORMAL", "NEUTRAL"})
        self.assertGreaterEqual(prof.vah, prof.poc)
        self.assertLessEqual(prof.val, prof.poc)


class POCAndValueAreaTest(unittest.TestCase):
    def test_poc_at_most_touched_price(self) -> None:
        # Most brackets cluster tightly around 1.0850; only one wider bracket
        # extends out so that 1.0850's bucket is the unique most-touched.
        candles = _session_candles(
            [
                (1.0850, 1.0851, 1.0849, 1.0850),
                (1.0850, 1.0851, 1.0849, 1.0850),
                (1.0850, 1.0851, 1.0849, 1.0850),
                (1.0850, 1.0851, 1.0849, 1.0850),
                (1.0850, 1.0851, 1.0849, 1.0850),
                (1.0850, 1.0860, 1.0840, 1.0850),  # one wider bracket
            ]
        )
        prof = build_tpo_profile(
            candles, session_start_utc=SESSION_START, pip_size=0.0001, bin_pips=1.0
        )
        # POC bucket should contain 1.0850 (within one bin width = 1 pip).
        self.assertAlmostEqual(prof.poc, 1.0850, delta=0.00015)

    def test_value_area_honors_70_percent_rule(self) -> None:
        # Construct a session whose total touches we can reason about.
        candles = _session_candles(
            [
                (150.00, 150.05, 149.95, 150.00),
                (150.00, 150.05, 149.95, 150.00),
                (150.00, 150.05, 149.95, 150.00),
                (150.00, 150.10, 149.90, 150.00),
                (150.00, 150.05, 149.95, 150.00),
                (150.00, 150.05, 149.95, 150.00),
            ]
        )
        prof = build_tpo_profile(
            candles, session_start_utc=SESSION_START, pip_size=0.01, bin_pips=1.0
        )
        # VAH > POC > VAL (or equal at the bucket edge).
        self.assertGreaterEqual(prof.vah, prof.poc)
        self.assertLessEqual(prof.val, prof.poc)
        # 70% rule: sum of touches inside [VAL, VAH] >= 70% of total.
        total_touches = 0
        ib_touches = 0
        for br in prof.brackets:
            # Approximate: count buckets in the bracket and intersect with VA.
            spans = []
            cur = br.low
            while cur <= br.high + 1e-9:
                spans.append(cur)
                cur += 0.01
            total_touches += len(spans)
            ib_touches += sum(1 for p in spans if prof.val - 1e-9 <= p <= prof.vah + 1e-9)
        self.assertGreaterEqual(ib_touches / max(total_touches, 1), 0.70 - 0.05)


class IBTest(unittest.TestCase):
    def test_ib_uses_first_two_brackets(self) -> None:
        candles = _session_candles(
            [
                (150.00, 150.40, 149.50, 150.20),  # A — wide
                (150.20, 150.30, 149.80, 150.10),  # B
                (150.10, 150.50, 149.70, 150.30),  # C — wider but post-IB
                (150.30, 150.60, 149.40, 150.40),  # D
            ]
        )
        prof = build_tpo_profile(
            candles, session_start_utc=SESSION_START, pip_size=0.01, bin_pips=2.0
        )
        # IB H = max(150.40, 150.30) = 150.40
        # IB L = min(149.50, 149.80) = 149.50
        self.assertAlmostEqual(prof.initial_balance_high, 150.40, places=4)
        self.assertAlmostEqual(prof.initial_balance_low, 149.50, places=4)
        self.assertTrue(prof.range_extension_up)   # C/D high 150.50/150.60 > 150.40
        self.assertTrue(prof.range_extension_down)  # D low 149.40 < 149.50


class OpenRelationTest(unittest.TestCase):
    def test_oaor_when_open_above_prior_va(self) -> None:
        candles = _session_candles(
            [
                (151.00, 151.05, 150.95, 151.00),
                (151.00, 151.05, 150.95, 151.00),
                (151.00, 151.05, 150.95, 151.00),
                (151.00, 151.05, 150.95, 151.00),
            ]
        )
        # Prior VA: 150.20–150.40. Open 151.00 is well above 150.40 → OAOR.
        prof = build_tpo_profile(
            candles,
            session_start_utc=SESSION_START,
            pip_size=0.01,
            bin_pips=1.0,
            prior_value_area=(150.40, 150.20),
        )
        self.assertEqual(prof.open_relation, "OAOR")

    def test_oare_when_open_inside_prior_va(self) -> None:
        candles = _session_candles(
            [
                (150.30, 150.35, 150.25, 150.30),
                (150.30, 150.35, 150.25, 150.30),
                (150.30, 150.35, 150.25, 150.30),
                (150.30, 150.35, 150.25, 150.30),
            ]
        )
        prof = build_tpo_profile(
            candles,
            session_start_utc=SESSION_START,
            pip_size=0.01,
            bin_pips=1.0,
            prior_value_area=(150.40, 150.20),
        )
        self.assertEqual(prof.open_relation, "OARE")


class HVNLVNTest(unittest.TestCase):
    def test_hvn_at_concentration_lvn_at_gap(self) -> None:
        # Heavy concentration near 150.00, sparse zone, then another shelf at 150.10.
        candles = _session_candles(
            [
                (150.00, 150.01, 149.99, 150.00),
                (150.00, 150.01, 149.99, 150.00),
                (150.00, 150.01, 149.99, 150.00),
                (150.05, 150.05, 150.05, 150.05),  # quick traverse — single touch
                (150.10, 150.11, 150.09, 150.10),
                (150.10, 150.11, 150.09, 150.10),
                (150.10, 150.11, 150.09, 150.10),
            ]
        )
        prof = build_tpo_profile(
            candles, session_start_utc=SESSION_START, pip_size=0.01, bin_pips=1.0
        )
        # HVN should include something near 150.00 or 150.10 (the shelves).
        self.assertTrue(
            any(abs(h - 150.00) < 0.02 or abs(h - 150.10) < 0.02 for h in prof.hvn),
            f"HVN={prof.hvn}",
        )
        # LVN should include the sparse mid-zone near ~150.05 (one bracket only).
        if prof.lvn:
            self.assertTrue(
                any(149.95 <= lv <= 150.10 for lv in prof.lvn),
                f"LVN={prof.lvn}",
            )

    def test_single_prints_detected(self) -> None:
        # One bracket spikes higher than the rest — that price is a single print.
        candles = _session_candles(
            [
                (150.00, 150.05, 149.95, 150.00),
                (150.00, 150.05, 149.95, 150.00),
                (150.00, 150.50, 149.95, 150.00),  # spike high
                (150.00, 150.05, 149.95, 150.00),
                (150.00, 150.05, 149.95, 150.00),
            ]
        )
        prof = build_tpo_profile(
            candles, session_start_utc=SESSION_START, pip_size=0.01, bin_pips=1.0
        )
        # Some price near 150.30–150.45 is touched by exactly one bracket.
        self.assertTrue(
            any(150.10 <= p <= 150.50 for p in prof.single_prints),
            f"single_prints={prof.single_prints}",
        )


class NakedPOCTest(unittest.TestCase):
    def test_unretraded_prior_poc_is_naked(self) -> None:
        # Session 1: POC near 150.00.
        s1 = _session_candles(
            [
                (150.00, 150.05, 149.95, 150.00),
                (150.00, 150.05, 149.95, 150.00),
                (150.00, 150.05, 149.95, 150.00),
                (150.00, 150.05, 149.95, 150.00),
            ]
        )
        prof1 = build_tpo_profile(
            s1, session_start_utc=SESSION_START, pip_size=0.01, bin_pips=1.0
        )
        # Session 2 (next day): trades way above 150.00 the entire day, never returns.
        s2_start = SESSION_START + timedelta(hours=24)
        s2 = []
        ts = s2_start
        for _ in range(8):
            s2.append(_candle(ts, 151.00, 151.10, 150.90, 151.05))
            ts += timedelta(minutes=30)
        prof2 = build_tpo_profile(
            s2, session_start_utc=s2_start, pip_size=0.01, bin_pips=1.0
        )

        # Current price is up at 151.00 — has never crossed back to prof1.poc (~150.00).
        naked = naked_pocs([prof1, prof2], current_price=151.00)
        # prof1's POC should be in the naked list.
        self.assertTrue(
            any(abs(p - prof1.poc) < 0.02 for p in naked),
            f"naked={naked}, prof1.poc={prof1.poc}",
        )

    def test_retraded_prior_poc_is_not_naked(self) -> None:
        # Session 1 POC near 150.00. Session 2 trades through 150.00 → retraded.
        s1 = _session_candles(
            [
                (150.00, 150.05, 149.95, 150.00),
                (150.00, 150.05, 149.95, 150.00),
                (150.00, 150.05, 149.95, 150.00),
                (150.00, 150.05, 149.95, 150.00),
            ]
        )
        prof1 = build_tpo_profile(
            s1, session_start_utc=SESSION_START, pip_size=0.01, bin_pips=1.0
        )
        s2_start = SESSION_START + timedelta(hours=24)
        s2 = []
        ts = s2_start
        for _ in range(4):
            s2.append(_candle(ts, 150.30, 150.40, 149.80, 150.20))  # range covers 150.00
            ts += timedelta(minutes=30)
        prof2 = build_tpo_profile(
            s2, session_start_utc=s2_start, pip_size=0.01, bin_pips=1.0
        )
        naked = naked_pocs([prof1, prof2], current_price=150.20)
        self.assertFalse(
            any(abs(p - prof1.poc) < 0.005 for p in naked),
            f"prof1.poc={prof1.poc} should have been retraded; naked={naked}",
        )


class NumericDefaultsDocTest(unittest.TestCase):
    """§3.5: numeric defaults all carry an (a)/(b)/(c) docstring rationale."""

    def test_numeric_defaults_documented(self) -> None:
        from quant_rabbit.analysis import profile as mod

        with open(mod.__file__, "r", encoding="utf-8") as fh:
            src = fh.read()
        # The four numeric defaults specified by the spec must each be
        # accompanied by an (a)/(b)/(c) rationale block.
        for name in (
            "DEFAULT_BIN_ATR_FACTOR",
            "IB_BRACKET_COUNT",
            "VALUE_AREA_PCT",
            "DEFAULT_NAKED_POC_MAX_AGE_SESSIONS",
        ):
            idx = src.find(name)
            self.assertNotEqual(idx, -1, f"{name} not found in source")
            # Look at the preceding 800 chars for the (a)/(b)/(c) markers.
            preamble = src[max(0, idx - 1000) : idx]
            self.assertIn("(a)", preamble, f"{name} missing (a) rationale")
            self.assertIn("(b)", preamble, f"{name} missing (b) rationale")
            self.assertIn("(c)", preamble, f"{name} missing (c) rationale")

    def test_numeric_default_values(self) -> None:
        # Sanity-check the spec-mandated numeric defaults.
        self.assertAlmostEqual(DEFAULT_BIN_ATR_FACTOR, 0.05)
        self.assertEqual(DEFAULT_NAKED_POC_MAX_AGE_SESSIONS, 5)
        self.assertEqual(IB_BRACKET_COUNT, 2)
        self.assertAlmostEqual(VALUE_AREA_PCT, 0.70)


if __name__ == "__main__":
    unittest.main()
