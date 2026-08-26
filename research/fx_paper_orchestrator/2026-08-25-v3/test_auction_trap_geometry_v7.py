import unittest
from datetime import datetime, timedelta, timezone

from causal_composite_indicators_v3 import Bar
from run_auction_trap_geometry_v7 import event_energy_ratio, qualifies, score


def bar(index, close, high=None, low=None, spread=0.0002):
    high = close + 0.0003 if high is None else high
    low = close - 0.0003 if low is None else low
    stamp = (datetime(2026, 5, 1, tzinfo=timezone.utc) + timedelta(minutes=5 * index)).isoformat().replace("+00:00", "Z")
    return Bar(
        "EUR_USD", stamp,
        close - spread / 2, high - spread / 2, low - spread / 2, close - spread / 2,
        close + spread / 2, high + spread / 2, low + spread / 2, close + spread / 2, 100,
    )


class AuctionTrapGeometryTest(unittest.TestCase):
    def fixture(self):
        bars = [bar(i, 1.0 + (i % 2) * 0.0001) for i in range(24)]
        bars.extend([
            bar(24, 1.0000, high=1.0015, low=0.9998, spread=0.0004),
            bar(25, 0.9997, high=1.0002, low=0.9995, spread=0.0002),
        ])
        bars.extend(bar(i, 0.9997) for i in range(26, 52))
        event = {
            "signal_id": "x", "pair": "EUR_USD", "breakout_index": 24,
            "breakout_time": bars[24].time, "escape_side": 1,
            "workers": ["IMMEDIATE_ESCAPE", "CONFIRMED_REJECTION"],
            "boundary_crowding": 2 / 24, "rail_escape_energy": 0.2,
            "next_boundary_distance": -0.2,
        }
        return bars, event

    def test_gate_uses_only_completed_event_and_response(self):
        bars, event = self.fixture()
        before = qualifies(bars, event)
        bars[-1] = bar(51, 9.0)
        self.assertEqual(before, qualifies(bars, event))

    def test_energy_is_causal_and_finite(self):
        bars, _ = self.fixture()
        self.assertGreaterEqual(event_energy_ratio(bars, 24), 1.0)

    def test_all_cost_arms_share_gross_and_signal_timing(self):
        bars, event = self.fixture()
        rows = [score(bars, event, arm) for arm in ("RAW_SIGNAL", "EXECUTABLE_BASE", "ADVERSE_STRESS")]
        self.assertTrue(all(row is not None for row in rows))
        self.assertEqual(len({row["gross_return"] for row in rows}), 1)
        self.assertGreater(rows[0]["net_return"], rows[1]["net_return"])
        self.assertGreater(rows[1]["net_return"], rows[2]["net_return"])


if __name__ == "__main__":
    unittest.main()
