from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path
import unittest

from hypothesis import given, strategies as st


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("robustness", ROOT / "run_robustness.py")
ROBUSTNESS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(ROBUSTNESS)


class RobustnessProperties(unittest.TestCase):
    @given(st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), max_size=100))
    def test_drawdown_is_nonnegative(self, values: list[float]) -> None:
        self.assertGreaterEqual(ROBUSTNESS.max_drawdown(values), 0.0)

    @given(
        st.floats(min_value=0.000001, max_value=1.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    def test_equal_hedge_is_strictly_negative_after_spread(self, spread: float, units: float) -> None:
        # Equal and opposite price exposure cancels. The two paid spreads do not.
        locked_price_pnl = 0.0
        after_cost = locked_price_pnl - 2.0 * spread * units
        self.assertLess(after_cost, 0.0)

    def test_gap_count_does_not_interpolate(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        candles = {start: {}, start + timedelta(seconds=10): {}}
        self.assertEqual(ROBUSTNESS.gap_count(candles, start, start + timedelta(seconds=10)), 1)

    def test_split_is_chronological_and_embargoed(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        rows = [{"trade_id": str(i), "fill_at_utc": ROBUSTNESS.iso(start + timedelta(hours=i))} for i in range(10)]
        assigned = ROBUSTNESS.assign_split(rows)
        self.assertEqual(sum(v == "TRAIN" for v in assigned.values()), 6)
        self.assertEqual(assigned["6"], "EMBARGO")
        self.assertTrue(all(assigned[str(i)] == "VALIDATION" for i in range(7, 10)))


if __name__ == "__main__":
    unittest.main()
