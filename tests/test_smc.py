"""Tests for SMC/ICT primitive detectors."""

from __future__ import annotations

import inspect
import unittest
from datetime import datetime, timedelta, timezone

from quant_rabbit.analysis.candles import Candle
from quant_rabbit.analysis import smc
from quant_rabbit.analysis.smc import (
    OTE_LOWER_FIB,
    OTE_SWEET_SPOT_FIB,
    OTE_UPPER_FIB,
    SwingPivot,
    analyze_smc,
    analyze_structure,
    compute_dealing_range,
    compute_premium_discount,
    detect_breakers,
    detect_displacement,
    detect_fair_value_gaps,
    detect_inversion_fvgs,
    detect_mitigations,
    detect_sweeps,
    detect_swing_pivots,
)


def _candle(
    i: int,
    o: float,
    h: float,
    low: float,
    c: float,
    *,
    base: datetime | None = None,
) -> Candle:
    base = base or datetime(2026, 5, 4, tzinfo=timezone.utc)
    return Candle(
        timestamp_utc=base + timedelta(minutes=5 * i),
        open=o,
        high=h,
        low=low,
        close=c,
        volume=1000,
        complete=True,
    )


def _series(rows: list[tuple[float, float, float, float]]) -> list[Candle]:
    return [_candle(i, o, h, low, c) for i, (o, h, low, c) in enumerate(rows)]


class SweepDetectorTest(unittest.TestCase):
    def test_sweep_high_when_wick_exceeds_pivot_but_close_back(self) -> None:
        # Build a swing-high pivot at index 4, then a sweep wick at index 9
        rows = [
            (100.0, 100.5, 99.5, 100.4),  # 0
            (100.4, 100.7, 100.0, 100.6),  # 1
            (100.6, 101.2, 100.4, 101.0),  # 2
            (101.0, 101.5, 100.7, 101.3),  # 3
            (101.3, 102.0, 101.0, 101.8),  # 4 <- swing high
            (101.8, 101.9, 101.0, 101.2),  # 5
            (101.2, 101.4, 100.5, 100.8),  # 6
            (100.8, 100.9, 100.3, 100.5),  # 7
            (100.5, 100.7, 100.0, 100.3),  # 8
            (100.3, 102.5, 100.2, 101.5),  # 9 <- sweep wick > 102.0, close < 102.0
            (101.5, 101.7, 100.8, 101.0),  # 10 (post bar so pivot 4 confirms early)
        ]
        candles = _series(rows)
        pivots = detect_swing_pivots(candles, strength=2)
        self.assertTrue(any(p.kind == "HIGH" and p.price == 102.0 for p in pivots))
        sweeps = detect_sweeps(candles, pivots)
        sweep_highs = [s for s in sweeps if s.side == "SWEEP_HIGH"]
        self.assertTrue(sweep_highs, "expected at least one SWEEP_HIGH")
        s = sweep_highs[0]
        self.assertGreater(s.wick_extreme, s.swept_pivot_price)
        self.assertTrue(s.close_back_inside)

    def test_sweep_low_symmetric(self) -> None:
        rows = [
            (105.0, 105.3, 104.5, 104.8),
            (104.8, 105.0, 104.5, 104.6),
            (104.6, 104.8, 104.2, 104.4),
            (104.4, 104.6, 104.0, 104.2),
            (104.2, 104.5, 103.5, 104.0),  # 4 swing low at 103.5
            (104.0, 104.4, 103.8, 104.2),
            (104.2, 104.7, 104.0, 104.5),
            (104.5, 104.9, 104.2, 104.7),
            (104.7, 104.9, 104.3, 104.4),
            (104.4, 104.6, 102.9, 104.1),  # 9 wick below 103.5, close back above
            (104.1, 104.3, 103.7, 104.0),
        ]
        candles = _series(rows)
        pivots = detect_swing_pivots(candles, strength=2)
        sweeps = detect_sweeps(candles, pivots)
        sweep_lows = [s for s in sweeps if s.side == "SWEEP_LOW"]
        self.assertTrue(sweep_lows)
        self.assertLess(sweep_lows[0].wick_extreme, sweep_lows[0].swept_pivot_price)


class BreakerDetectorTest(unittest.TestCase):
    def test_breaker_requires_ob_then_sweep_then_opposite_choch(self) -> None:
        # Canonical breaker: up-leg → BULL OB → swing-low pivot inside OB band →
        # wick below that swing low AND close back above (SWEEP_LOW that taps OB) →
        # BOS_DOWN below an even earlier swing low.
        rows = [
            # initial pump → swing high pivot at index 3
            (100.0, 100.4, 99.95, 100.3),     # 0
            (100.3, 100.6, 100.25, 100.55),   # 1
            (100.55, 100.85, 100.5, 100.8),   # 2
            (100.8, 100.95, 100.75, 100.9),   # 3 swing high (100.95)
            (100.9, 100.92, 100.6, 100.65),   # 4 fall
            (100.65, 100.7, 100.4, 100.45),   # 5
            (100.45, 100.5, 100.2, 100.3),    # 6 swing low (100.2)
            (100.3, 100.5, 100.28, 100.45),   # 7
            (100.45, 100.6, 100.4, 100.55),   # 8
            (100.55, 100.7, 100.5, 100.5),    # 9 bearish — BULL OB candidate (low=100.5, high=100.7)
            (100.5, 102.0, 100.5, 101.9),     # 10 displacement up; close 101.9 > 100.95 → CHOCH_UP
            (101.9, 102.3, 101.85, 102.2),    # 11
            (102.2, 102.5, 102.15, 102.4),    # 12 swing high (102.5)
            (102.4, 102.45, 102.1, 102.2),    # 13
            (102.2, 102.25, 101.9, 102.0),    # 14
            (102.0, 102.05, 101.7, 101.8),    # 15
            (101.8, 101.85, 101.5, 101.6),    # 16
            (101.6, 101.65, 101.3, 101.4),    # 17
            (101.4, 101.45, 101.05, 101.15),  # 18
            (101.15, 101.2, 100.95, 101.0),   # 19
            (101.0, 101.05, 100.8, 100.85),   # 20 swing low candidate (100.8)
            (100.85, 100.95, 100.82, 100.9),  # 21
            (100.9, 100.95, 100.85, 100.88),  # 22
            # Sweep: wick to 100.5 (< 100.8 prior low) AND traverses BULL OB (100.5-100.7).
            # Close 100.85 back above 100.8.
            (100.88, 100.95, 100.5, 100.85),  # 23 SWEEP_LOW
            (100.85, 100.9, 100.4, 100.5),    # 24
            (100.5, 100.55, 100.1, 100.15),   # 25
            (100.15, 100.2, 99.5, 99.6),      # 26 close 99.6 < 100.2 prior swing low → BOS_DOWN
            (99.6, 99.7, 99.3, 99.4),         # 27
        ]
        candles = _series(rows)
        structure = analyze_structure(candles)
        sweeps = detect_sweeps(candles, structure.pivots)
        breakers = detect_breakers(structure, sweeps)
        self.assertTrue(
            breakers,
            f"expected breaker; pivots={[(p.kind,p.index,p.price) for p in structure.pivots]}, "
            f"events={[(e.kind,e.index) for e in structure.events]}, "
            f"sweeps={[(s.side,s.index) for s in sweeps]}, "
            f"OBs={[(o.side,o.index,o.low,o.high) for o in structure.order_blocks]}",
        )
        b = breakers[0]
        self.assertEqual(b.base_ob.side, "BULL")
        self.assertEqual(b.sweep.side, "SWEEP_LOW")
        self.assertIn(b.shift_event.kind, ("CHOCH_DOWN", "BOS_DOWN"))


class MitigationDetectorTest(unittest.TestCase):
    def test_mitigation_when_ob_then_opposite_choch_without_sweep(self) -> None:
        # Pattern: up-leg → swing high → BEAR OB → CHOCH_DOWN → swing low →
        # CHOCH_UP, with NO sweep above the swing high before CHOCH_UP.
        rows = [
            (100.0, 100.2, 99.5, 99.6),      # 0 dip
            (99.6, 99.65, 99.4, 99.5),       # 1
            (99.5, 99.55, 99.3, 99.4),       # 2 swing low candidate (99.3)
            (99.4, 99.7, 99.35, 99.6),       # 3
            (99.6, 99.9, 99.55, 99.8),       # 4
            (99.8, 100.2, 99.75, 100.1),     # 5
            (100.1, 100.5, 100.05, 100.4),   # 6
            (100.4, 100.8, 100.35, 100.7),   # 7
            (100.7, 101.1, 100.65, 101.0),   # 8 swing high candidate (101.1)
            (101.0, 101.05, 100.7, 100.8),   # 9
            (100.8, 100.95, 100.75, 100.9),  # 10 bullish — BEAR OB candidate
            (100.9, 100.95, 99.1, 99.2),     # 11 displacement down. close < 99.3 swing low → CHOCH_DOWN
            (99.2, 99.3, 98.7, 98.8),        # 12
            (98.8, 98.9, 98.4, 98.5),        # 13 swing low candidate (98.4)
            (98.5, 98.7, 98.45, 98.65),      # 14
            (98.65, 98.85, 98.6, 98.8),      # 15
            (98.8, 99.1, 98.75, 99.0),       # 16
            (99.0, 99.4, 98.95, 99.3),       # 17
            (99.3, 99.7, 99.25, 99.6),       # 18
            (99.6, 99.85, 99.55, 99.75),     # 19 swing high candidate (99.85)
            (99.75, 99.8, 99.5, 99.6),       # 20
            (99.6, 99.7, 99.4, 99.55),       # 21
            (99.55, 100.2, 99.5, 100.1),     # 22 close > 99.85 (last high) → BOS_UP/CHOCH_UP. high < 101.1 (no sweep above swing high). high also < 100.95 OB top
            (100.1, 100.3, 100.05, 100.2),   # 23
        ]
        candles = _series(rows)
        structure = analyze_structure(candles)
        sweeps = detect_sweeps(candles, structure.pivots)
        # No SWEEP_HIGH that traverses any BEAR OB
        ob_traversing = [
            s for s in sweeps
            if s.side == "SWEEP_HIGH" and any(
                ob.side == "BEAR" and ob.low <= s.wick_extreme <= ob.high
                for ob in structure.order_blocks
            )
        ]
        self.assertFalse(
            ob_traversing, f"fixture leaked OB-traversing sweep: {ob_traversing}"
        )
        mitigations = detect_mitigations(structure)
        self.assertTrue(
            mitigations,
            f"expected mitigation; pivots={[(p.kind,p.index,p.price) for p in structure.pivots]}, "
            f"events={[(e.kind,e.index) for e in structure.events]}, "
            f"OBs={[(o.side, o.index) for o in structure.order_blocks]}",
        )
        self.assertIn("lower probability", mitigations[0].note)


class InversionFVGTest(unittest.TestCase):
    def test_ifvg_when_filled_then_rejected(self) -> None:
        # UP-FVG between candle 1 high and candle 3 low; later filled then rejected
        rows = [
            (100.0, 100.2, 99.8, 100.0),   # 0
            (100.0, 100.5, 99.9, 100.4),   # 1 prev_high=100.5
            (100.4, 101.5, 100.4, 101.4),  # 2 displacement
            (101.4, 101.6, 101.0, 101.5),  # 3 next_low=101.0 -> FVG (100.5..101.0)
            (101.5, 101.7, 101.2, 101.6),  # 4
            (101.6, 101.8, 100.4, 100.6),  # 5 fills the gap (low <= 100.5)
            (100.6, 100.9, 100.3, 100.7),  # 6
            (100.7, 101.2, 100.6, 100.4),  # 7 retests gap (high>=100.5) and closes <100.5 -> rejection
        ]
        candles = _series(rows)
        fvgs = detect_fair_value_gaps(candles)
        ifvgs = detect_inversion_fvgs(candles, fvgs)
        self.assertTrue(
            ifvgs,
            f"expected iFVG; fvgs={[(f.direction, f.lower, f.upper, f.filled) for f in fvgs]}",
        )
        ifv = ifvgs[0]
        self.assertGreater(ifv.rejected_at_index, ifv.filled_at_index)


class DisplacementDetectorTest(unittest.TestCase):
    def test_large_body_flagged(self) -> None:
        # 30 small bars then one huge body
        rows: list[tuple[float, float, float, float]] = []
        price = 100.0
        for i in range(30):
            o = price
            c = price + 0.05
            rows.append((o, max(o, c) + 0.05, min(o, c) - 0.05, c))
            price = c
        # huge displacement bar
        o = price
        c = price + 5.0
        rows.append((o, c + 0.1, o - 0.05, c))
        candles = _series(rows)
        disp = detect_displacement(candles, atr_window=14, body_atr_threshold=1.5)
        self.assertTrue(disp)
        self.assertGreater(disp[-1].body_atr_ratio, 1.5)
        self.assertEqual(disp[-1].direction, "UP")


class DealingRangeTest(unittest.TestCase):
    def _build_up_then_down(self) -> list[Candle]:
        rows: list[tuple[float, float, float, float]] = []
        # rise from 100 to 110 over 30 bars
        for i in range(30):
            base = 100.0 + i * (10.0 / 29.0)
            rows.append((base, base + 0.2, base - 0.2, base + 0.1))
        # peak bar at i=30 with sharp top
        rows.append((110.1, 112.0, 110.0, 111.5))  # 30 — pivot high candidate
        # fall back to 101 over 30 bars
        for i in range(30):
            base = 111.5 - i * (10.5 / 29.0)
            rows.append((base, base + 0.2, base - 0.2, base - 0.1))
        # trough bar
        rows.append((101.0, 101.2, 99.5, 100.0))  # 61 — pivot low candidate
        # recovery bars so the trough is confirmed
        for i in range(8):
            base = 100.0 + (i + 1) * 0.3
            rows.append((base, base + 0.2, base - 0.1, base + 0.1))
        return _series(rows)

    def test_dealing_range_extremes_and_classification(self) -> None:
        candles = self._build_up_then_down()
        dr = compute_dealing_range(candles, lookback=200, swing_strength=3)
        self.assertIsNotNone(dr)
        assert dr is not None
        self.assertGreater(dr.swing_high.price, dr.swing_low.price)
        # Equilibrium = midpoint
        self.assertAlmostEqual(
            dr.equilibrium,
            (dr.swing_high.price + dr.swing_low.price) / 2.0,
            places=6,
        )
        # Premium / discount classification
        self.assertEqual(
            compute_premium_discount(dr.swing_high.price - 0.01, dr), "PREMIUM"
        )
        self.assertEqual(
            compute_premium_discount(dr.swing_low.price + 0.01, dr), "DISCOUNT"
        )
        self.assertEqual(compute_premium_discount(dr.equilibrium, dr), "EQUILIBRIUM")

    def test_ote_zone_within_62_to_79_retracement(self) -> None:
        candles = self._build_up_then_down()
        dr = compute_dealing_range(candles, lookback=200, swing_strength=3)
        assert dr is not None
        span = dr.swing_high.price - dr.swing_low.price
        # Mirror smc.compute_dealing_range: up-leg branch when low precedes high
        if dr.swing_low.index < dr.swing_high.index:
            expected_lo = dr.swing_high.price - span * OTE_UPPER_FIB
            expected_hi = dr.swing_high.price - span * OTE_LOWER_FIB
            expected_sweet = dr.swing_high.price - span * OTE_SWEET_SPOT_FIB
        else:
            expected_lo = dr.swing_low.price + span * OTE_LOWER_FIB
            expected_hi = dr.swing_low.price + span * OTE_UPPER_FIB
            expected_sweet = dr.swing_low.price + span * OTE_SWEET_SPOT_FIB
        lo, hi = dr.ote_zone
        self.assertAlmostEqual(lo, min(expected_lo, expected_hi), places=6)
        self.assertAlmostEqual(hi, max(expected_lo, expected_hi), places=6)
        self.assertAlmostEqual(dr.ote_sweet_spot, expected_sweet, places=6)
        # sweet spot lies inside zone
        self.assertGreaterEqual(dr.ote_sweet_spot, lo)
        self.assertLessEqual(dr.ote_sweet_spot, hi)


class OrchestratorTest(unittest.TestCase):
    def test_analyze_smc_returns_aggregate(self) -> None:
        rows: list[tuple[float, float, float, float]] = []
        price = 100.0
        for i in range(60):
            price += 0.1 if i < 30 else -0.1
            rows.append((price - 0.05, price + 0.15, price - 0.15, price))
        candles = _series(rows)
        reading = analyze_smc(candles)
        self.assertIsNotNone(reading.structure)
        # all aggregate fields should be tuples or None
        for fld in (
            "sweeps",
            "breakers",
            "mitigations",
            "inversion_fvgs",
            "displacements",
        ):
            self.assertIsInstance(getattr(reading, fld), tuple)


class HardcodeDocstringTest(unittest.TestCase):
    """§3.5: every numeric default in the public API must be documented
    with (a)/(b)/(c) — what / why constant / what to replace it with.
    """

    DOCUMENTED_NAMES = {
        "detect_swing_pivots": {"strength"},
        "detect_liquidity_clusters": {"tolerance"},
        "detect_sweeps": {"close_back_inside"},
        "detect_displacement": {"atr_window", "body_atr_threshold"},
        "compute_dealing_range": {"lookback", "swing_strength"},
    }

    def test_each_default_is_documented(self) -> None:
        for fn_name, kwargs in self.DOCUMENTED_NAMES.items():
            fn = getattr(smc, fn_name)
            doc = inspect.getdoc(fn) or ""
            sig = inspect.signature(fn)
            for kw in kwargs:
                self.assertIn(
                    kw,
                    sig.parameters,
                    f"{fn_name} missing kwarg {kw}",
                )
                self.assertIn(
                    kw,
                    doc,
                    f"{fn_name} docstring must mention `{kw}` per §3.5",
                )
            # contract markers (a)/(b)/(c)
            for marker in ("(a)", "(b)", "(c)"):
                self.assertIn(
                    marker,
                    doc,
                    f"{fn_name} docstring missing contract marker {marker}",
                )

    def test_ote_constants_documented(self) -> None:
        src = inspect.getsource(smc)
        for marker in ("(a)", "(b)", "(c)"):
            self.assertIn(marker, src)
        for name in ("OTE_LOWER_FIB", "OTE_UPPER_FIB", "OTE_SWEET_SPOT_FIB"):
            self.assertIn(name, src)
        # sanity: classical ICT values
        self.assertEqual(OTE_LOWER_FIB, 0.62)
        self.assertEqual(OTE_UPPER_FIB, 0.79)
        self.assertEqual(OTE_SWEET_SPOT_FIB, 0.705)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
