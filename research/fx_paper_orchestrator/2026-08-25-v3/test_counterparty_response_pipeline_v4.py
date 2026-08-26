import unittest

from causal_composite_indicators_v3 import Bar
from run_counterparty_response_models_v4 import response_label
from run_counterparty_response_study_v4 import score_response


class CounterpartyPipelineTest(unittest.TestCase):
    def bars(self):
        rows = []
        for i in range(60):
            close = 1.1 + i * 0.0001
            rows.append(Bar(
                "EUR_USD", f"2026-01-{1 + i // 24:02d}T{i % 24:02d}:00:00.000000000Z",
                close, close + 0.0002, close - 0.0002, close,
                close + 0.0001, close + 0.0003, close - 0.0001, close + 0.0001,
                100,
            ))
        return rows

    def test_fill_is_strictly_after_completed_response_bar(self):
        event = {"signal_id": "s", "pair": "EUR_USD", "breakout_index": 24,
                 "breakout_time": self.bars()[24].time, "escape_side": 1}
        result = score_response(self.bars(), event, "CONTINUATION_RESPONSE", 3, "RAW_SIGNAL")
        self.assertEqual(result["response_completed_bar_time"], self.bars()[25].time)
        self.assertEqual(result["fill_time"], self.bars()[26].time)

    def test_raw_signal_does_not_consult_cost(self):
        event = {"signal_id": "s", "pair": "EUR_USD", "breakout_index": 24,
                 "breakout_time": self.bars()[24].time, "escape_side": 1}
        raw = score_response(self.bars(), event, "CONTINUATION_RESPONSE", 3, "RAW_SIGNAL")
        base = score_response(self.bars(), event, "CONTINUATION_RESPONSE", 3, "EXECUTABLE_BASE")
        self.assertEqual(raw["slippage_pips_per_side"], 0.0)
        self.assertLess(base["net_return"], raw["gross_return"])

    def test_response_label_can_abstain_without_cost(self):
        row = {"roles": {
            "CONTINUATION_RESPONSE": {"returns": {"RAW_SIGNAL": -0.01}},
            "FAILED_AUCTION_REVERSAL": {"returns": {"RAW_SIGNAL": -0.02}},
        }}
        self.assertEqual(response_label(row), "UNRESOLVED_NO_ORDER")


if __name__ == "__main__":
    unittest.main()
