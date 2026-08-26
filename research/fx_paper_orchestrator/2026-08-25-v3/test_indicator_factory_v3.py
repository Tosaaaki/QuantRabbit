import unittest

from indicator_factory_v3 import (
    build_conditions,
    currency_exposure_diagnostics,
    FEATURES_BY_WORKER,
    lower_bound,
    qvalue,
    transform_features,
)


class IndicatorFactoryTest(unittest.TestCase):
    def test_quantile_is_deterministic_order_statistic(self):
        self.assertEqual(qvalue([5, 1, 4, 2, 3], 0.7), 3)

    def test_cluster_lower_penalizes_instability(self):
        stable = {f"w{i}": [1.0, 1.0] for i in range(8)}
        unstable = {f"w{i}": [10.0 if i % 2 else -8.0] for i in range(8)}
        self.assertGreater(lower_bound(stable, 1.96), lower_bound(unstable, 1.96))

    def test_feature_transform_does_not_use_cost_or_future_outcome(self):
        raw = {
            "rail_escape_energy": 1, "boundary_acceptance": -1,
            "rejection_curvature": 2, "geodesic_efficiency": .5,
            "price_spread_loop_area": -.2, "session_spread_strain": 1.1,
            "boundary_crowding": .3, "pre_break_compression": 1.2,
            "wick_rejection_ratio": .4, "tick_volume_shock": 1.5,
            "liquidity_sweep_geometry": .02, "currency_propagation": -2,
            "currency_breadth": .6, "currency_propagation_concentration": .3,
            "future_return": 999, "cost": 999,
        }
        result = transform_features(raw)
        self.assertEqual(result["abs_price_spread_loop_area"], .2)
        self.assertEqual(result["abs_currency_propagation"], 2)
        self.assertNotIn("future_return", result)
        self.assertNotIn("cost", result)

    def test_semantically_identical_quantile_conditions_are_deduplicated(self):
        events = {
            f"s{i}": {"features": {feature: 1.0 for feature in (
                "rail_escape_energy", "boundary_acceptance", "rejection_curvature",
                "geodesic_efficiency", "abs_price_spread_loop_area", "session_spread_strain",
                "boundary_crowding", "pre_break_compression", "wick_rejection_ratio",
                "tick_volume_shock", "liquidity_sweep_geometry", "currency_propagation",
                "abs_currency_propagation", "currency_breadth",
                "currency_propagation_concentration",
            )}} for i in range(4)
        }
        conditions = build_conditions(events, set(events), set(events))
        per_feature = [c for c in conditions if c["feature"] == "rail_escape_energy"]
        self.assertEqual(len(per_feature), 2)  # one <= rule and one >= rule

    def test_currency_exposure_reports_ticket_dependence(self):
        rows = [
            {"pair": "EUR_USD", "direction": 1, "fill_time": "2026-01-01T00:00:00Z"},
            {"pair": "GBP_USD", "direction": 1, "fill_time": "2026-01-01T04:00:00Z"},
        ]
        result = currency_exposure_diagnostics(rows)
        self.assertEqual(result["max_gross_currency_share"], 0.5)
        self.assertLess(result["effective_currency_nodes"], 4.0)
        self.assertEqual(result["net_currency_exposure_units"]["USD"], -2.0)

    def test_next_bar_feature_is_forbidden_for_immediate_worker(self):
        self.assertNotIn("rejection_curvature", FEATURES_BY_WORKER["IMMEDIATE_ESCAPE"])
        self.assertIn("rejection_curvature", FEATURES_BY_WORKER["CONFIRMED_REJECTION"])


if __name__ == "__main__":
    unittest.main()
