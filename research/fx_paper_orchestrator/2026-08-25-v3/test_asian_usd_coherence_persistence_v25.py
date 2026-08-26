import inspect
import unittest
from datetime import datetime, timedelta, timezone

from causal_composite_indicators_v3 import Bar
from run_asian_usd_coherence_persistence_v25 import UNIVERSE, USD_BASE, detect_day_signals


def bar(pair, index, close, spread=0.0002):
    stamp = (datetime(2026, 3, 11, tzinfo=timezone.utc) + timedelta(minutes=5 * index)).isoformat().replace(
        "+00:00", ".000000000Z"
    )
    high, low = close + 0.0001, close - 0.0001
    return Bar(
        pair, stamp,
        close - spread / 2, high - spread / 2, low - spread / 2, close - spread / 2,
        close + spread / 2, high + spread / 2, low + spread / 2, close + spread / 2, 100,
    )


def fixture(usd_votes):
    result = {}
    for pair in sorted(UNIVERSE):
        native_sign = usd_votes[pair] if pair in USD_BASE else -usd_votes[pair]
        result[pair] = [
            bar(pair, index, 1.0 + native_sign * min(index, 71) * 0.00001)
            for index in range(144)
        ]
    return result


class AsianUsdCoherencePersistenceTest(unittest.TestCase):
    def test_all_aligned_usd_strength_emits_seven_native_directions(self):
        signals = detect_day_signals(fixture({pair: 1 for pair in UNIVERSE}))
        self.assertEqual(len(signals), 7)
        by_pair = {signal["pair"]: signal for signal in signals}
        for pair in UNIVERSE:
            self.assertEqual(by_pair[pair]["direction"], 1 if pair in USD_BASE else -1)
            self.assertEqual(by_pair[pair]["diagnostics"]["aligned_pairs"], 7)

    def test_five_of_seven_emits_only_aligned_pairs(self):
        votes = {pair: 1 for pair in UNIVERSE}
        for pair in sorted(UNIVERSE)[:2]:
            votes[pair] = -1
        signals = detect_day_signals(fixture(votes))
        self.assertEqual(len(signals), 5)
        self.assertTrue(all(votes[signal["pair"]] == 1 for signal in signals))

    def test_four_of_seven_is_not_coherent(self):
        votes = {pair: 1 for pair in UNIVERSE}
        for pair in sorted(UNIVERSE)[:3]:
            votes[pair] = -1
        self.assertEqual(detect_day_signals(fixture(votes)), [])

    def test_future_fill_and_exit_prices_cannot_change_decision(self):
        bars = fixture({pair: 1 for pair in UNIVERSE})
        before = detect_day_signals(bars)
        for pair in UNIVERSE:
            bars[pair][72] = bar(pair, 72, 8.0)
            bars[pair][143] = bar(pair, 143, 0.2)
        self.assertEqual(before, detect_day_signals(bars))

    def test_missing_pair_or_completed_bar_fails_closed(self):
        bars = fixture({pair: 1 for pair in UNIVERSE})
        missing_pair = dict(bars)
        missing_pair.pop(sorted(UNIVERSE)[0])
        self.assertEqual(detect_day_signals(missing_pair), [])
        missing_bar = fixture({pair: 1 for pair in UNIVERSE})
        del missing_bar[sorted(UNIVERSE)[0]][12]
        self.assertEqual(detect_day_signals(missing_bar), [])

    def test_signal_detector_has_no_cost_or_outcome_parameter(self):
        parameters = set(inspect.signature(detect_day_signals).parameters)
        self.assertEqual(parameters, {"pair_day_bars"})


if __name__ == "__main__":
    unittest.main()
