import math
import unittest

from causal_composite_indicators_v3 import Bar
from counterparty_response_v4 import FEATURES, counterparty_features


class CounterpartyResponseTest(unittest.TestCase):
    def bars(self):
        rows = []
        for i in range(36):
            close = 1.1 + i * 0.0001
            if i == 24:
                close += 0.0010
            if i == 25:
                close -= 0.0008
            rows.append(Bar(
                "EUR_USD", f"2026-01-01T{i:02d}:00:00.000000000Z",
                close - 0.0001, close + 0.0004, close - 0.0003, close,
                close, close + 0.0005, close - 0.0002, close + 0.0001,
                100 + i,
            ))
        return rows

    def event(self):
        return {
            "breakout_index": 24,
            "escape_side": 1,
            "scale": 0.0005,
            "rail_escape_energy": 2.0,
            "next_boundary_distance": -0.5,
            "currency_propagation": 0.7,
        }

    def test_feature_set_is_complete_and_finite(self):
        result = counterparty_features(self.event(), self.bars())
        self.assertEqual(result["response_completed_index"], 25)
        self.assertTrue(all(name in result for name in FEATURES))
        self.assertTrue(all(math.isfinite(result[name]) for name in FEATURES))

    def test_bars_after_response_cannot_change_features(self):
        bars = self.bars()
        baseline = counterparty_features(self.event(), bars)
        bars[26] = Bar(
            "EUR_USD", "2026-01-02T02:00:00.000000000Z",
            9.0, 10.0, 8.0, 9.5, 9.1, 10.1, 8.1, 9.6, 999999,
        )
        changed = counterparty_features(self.event(), bars)
        self.assertEqual(
            {name: baseline[name] for name in FEATURES},
            {name: changed[name] for name in FEATURES},
        )

    def test_invalid_direction_fails_closed(self):
        event = self.event()
        event["escape_side"] = 0
        with self.assertRaises(ValueError):
            counterparty_features(event, self.bars())


if __name__ == "__main__":
    unittest.main()
