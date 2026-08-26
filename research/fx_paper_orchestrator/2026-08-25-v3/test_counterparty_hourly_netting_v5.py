import unittest
from datetime import datetime, timezone

from causal_composite_indicators_v3 import Bar
from run_counterparty_hourly_netting_v5 import build_targets, next_hour, simulate


class HourlyNettingTest(unittest.TestCase):
    def row(self, signal, fill_time, direction):
        return {
            "candidate_id": "FX_CRS_MULTINOMIAL_RESPONSE_H12_V4",
            "source_signal_id": signal,
            "pair": "EUR_USD",
            "fill_time": fill_time,
            "direction": direction,
            "expected_order": True,
        }

    def test_exact_hour_signal_waits_until_following_hour(self):
        self.assertEqual(next_hour("2026-05-01T10:00:00+00:00").hour, 11)

    def test_opposite_votes_net_to_flat_without_cost_gate(self):
        targets, audit = build_targets([
            self.row("a", "2026-05-01T10:05:00+00:00", 1),
            self.row("b", "2026-05-01T10:10:00+00:00", -1),
        ])
        checkpoint = next(iter(targets))
        self.assertEqual(targets[checkpoint]["EUR_USD"], 0)
        self.assertEqual(audit["eligible_source_orders"], 2)
        self.assertEqual(audit["tie_flat_targets"], 1)

    def test_cost_never_removes_source_vote(self):
        _, audit = build_targets([
            self.row("a", "2026-05-01T10:05:00+00:00", 1),
            self.row("b", "2026-05-01T10:10:00+00:00", 1),
            self.row("c", "2026-05-01T10:15:00+00:00", -1),
        ])
        self.assertEqual(audit["eligible_source_orders"], 3)

    def test_raw_portfolio_has_zero_execution_cost(self):
        bars = [
            Bar("EUR_USD", f"2026-05-01T{hour:02d}:00:00.000000000Z",
                1.0 + hour * .001, 1.001 + hour * .001, .999 + hour * .001, 1.0 + hour * .001,
                1.0002 + hour * .001, 1.0012 + hour * .001, .9992 + hour * .001, 1.0002 + hour * .001, 100)
            for hour in range(3)
        ]
        stamps = [datetime(2026, 5, 1, hour, tzinfo=timezone.utc) for hour in range(3)]
        targets = {stamps[0]: {"EUR_USD": 1}, stamps[1]: {"EUR_USD": 1}, stamps[2]: {"EUR_USD": 0}}
        result = simulate({"EUR_USD": bars}, {"EUR_USD": stamps}, targets,
                          "RAW_SIGNAL", "2026-05-01", "2026-05-02")
        self.assertEqual(result["cost_drag_nav_additive"], 0.0)
        self.assertEqual(result["terminal_open_inventory"], 0)

    def test_short_uses_exact_inverse_price_ratio(self):
        bars = [
            Bar("EUR_USD", "2026-05-01T00:00:00.000000000Z",
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100),
            Bar("EUR_USD", "2026-05-01T01:00:00.000000000Z",
                2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 100),
            Bar("EUR_USD", "2026-05-01T02:00:00.000000000Z",
                2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 100),
        ]
        stamps = [datetime(2026, 5, 1, hour, tzinfo=timezone.utc) for hour in range(3)]
        targets = {
            stamps[0]: {"EUR_USD": -1},
            stamps[1]: {"EUR_USD": 0},
            stamps[2]: {"EUR_USD": 0},
        }
        result = simulate({"EUR_USD": bars}, {"EUR_USD": stamps}, targets,
                          "RAW_SIGNAL", "2026-05-01", "2026-05-02")
        self.assertAlmostEqual(result["gross_pnl_nav_additive"], -0.5, places=12)

    def test_finite_persistence_reduces_no_signal_churn(self):
        eur = [
            Bar("EUR_USD", f"2026-05-01T{hour:02d}:00:00.000000000Z",
                1.0, 1.001, .999, 1.0, 1.0002, 1.0012, .9992, 1.0002, 100)
            for hour in range(4)
        ]
        gbp = [
            Bar("GBP_USD", f"2026-05-01T{hour:02d}:00:00.000000000Z",
                1.2, 1.201, 1.199, 1.2, 1.2002, 1.2012, 1.1992, 1.2002, 100)
            for hour in range(4)
        ]
        stamps = [datetime(2026, 5, 1, hour, tzinfo=timezone.utc) for hour in range(4)]
        targets = {
            stamps[0]: {"EUR_USD": 1},
            stamps[1]: {"GBP_USD": 1},
            stamps[2]: {"GBP_USD": 1},
            stamps[3]: {"GBP_USD": 0},
        }
        corpus = {"EUR_USD": eur, "GBP_USD": gbp}
        index = {"EUR_USD": stamps, "GBP_USD": stamps}
        flat = simulate(corpus, index, targets, "RAW_SIGNAL", "2026-05-01", "2026-05-02")
        held = simulate(corpus, index, targets, "RAW_SIGNAL", "2026-05-01", "2026-05-02", persistence_hours=12)
        self.assertLess(held["target_changes"], flat["target_changes"])
        self.assertEqual(held["terminal_open_inventory"], 0)


if __name__ == "__main__":
    unittest.main()
