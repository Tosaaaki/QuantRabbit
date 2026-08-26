import math
import sys
import unittest
from pathlib import Path

from causal_composite_indicators_v3 import ALL_FEATURES, CURRENT_FEATURES, enrich_event
from composite_factory_v3 import FEATURES_BY_WORKER


V2_DIR = Path(__file__).resolve().parents[1] / "2026-08-25-v2"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))
from fx_original_indicators import Bar  # noqa: E402


class CompositeIndicatorTest(unittest.TestCase):
    def bars(self):
        output = []
        for i in range(32):
            close = 1.0 + i * .0001
            if i == 24:
                close += .001
            if i == 25:
                close -= .0008
            output.append(Bar(
                "EUR_USD", f"2026-01-01T{i:02d}:00:00.000000000Z",
                close, close + .0003, close - .0003, close,
                close + .0001, close + .0004, close - .0002, close + .0001,
                100 + i,
            ))
        return output

    def test_composites_are_finite_and_complete(self):
        event = {
            "breakout_index": 24, "escape_side": 1, "scale": .0005,
            "rail_escape_energy": 2.0, "session_spread_strain": 1.2,
            "currency_propagation": 1.0, "currency_breadth": .7,
            "currency_propagation_concentration": .2,
            "price_spread_loop_area": .03, "next_boundary_distance": -.5,
            "wick_rejection_ratio": .4, "boundary_crowding": .2,
            "boundary_acceptance": -.2, "tick_volume_shock": 1.5,
        }
        result = enrich_event(event, self.bars(), 24)
        self.assertTrue(all(name in result for name in ALL_FEATURES))
        self.assertTrue(all(math.isfinite(result[name]) for name in ALL_FEATURES))
        self.assertGreater(result["post_break_failure_velocity"], 0)

    def test_delayed_features_cannot_reach_immediate_worker(self):
        self.assertEqual(tuple(FEATURES_BY_WORKER["IMMEDIATE_ESCAPE"]), CURRENT_FEATURES)
        self.assertNotIn("post_break_failure_velocity", FEATURES_BY_WORKER["IMMEDIATE_ESCAPE"])
        self.assertIn("post_break_failure_velocity", FEATURES_BY_WORKER["SWEEP_RECOVERY"])


if __name__ == "__main__":
    unittest.main()
