import hashlib
import inspect
import json
import unittest
from pathlib import Path

from run_causal_tail_excess_representative_v34 import apply_rule, causal_tail_excess_score


def row(signal_id, day, pair, displacement, threshold):
    return {
        "signal_id": signal_id,
        "utc_day": day,
        "pair": pair,
        "diagnostics": {
            "native_asian_log_displacement": displacement,
            "training_abs_displacement_q75": threshold,
        },
    }


class CausalTailExcessRepresentativeV34Test(unittest.TestCase):
    def test_one_max_normalized_tail_excess_is_selected_per_day(self):
        rows = [
            row("a", "2026-05-01", "AUD_USD", .002, .001),
            row("b", "2026-05-01", "EUR_USD", -.003, .001),
            row("c", "2026-05-02", "GBP_USD", .004, .002),
            row("d", "2026-05-02", "USD_JPY", -.0015, .001),
        ]
        self.assertEqual(apply_rule(rows), {"b", "c"})

    def test_tie_break_is_pair_then_signal_id(self):
        rows = [
            row("z", "2026-05-01", "EUR_USD", .002, .001),
            row("a", "2026-05-01", "AUD_USD", -.004, .002),
        ]
        self.assertEqual(apply_rule(rows), {"a"})

    def test_score_uses_only_predecision_displacement_and_frozen_threshold(self):
        value = row("a", "2026-05-01", "AUD_USD", -.004, .002)
        self.assertEqual(causal_tail_excess_score(value), 2.0)
        self.assertEqual(set(inspect.signature(causal_tail_excess_score).parameters), {"row"})

    def test_sealed_v33_parent_and_v34_work_order_hashes_are_fixed(self):
        root = Path(__file__).resolve().parent
        expected = {
            "evidence/run_asian_displacement_handoff_fade_v33_official_001/result_asian_displacement_handoff_fade_v33.json":
                "80ac9cf09680f50aec45eb36a29ed21528246d87bda30e1af049fee8722bd611",
            "evidence/run_asian_displacement_handoff_fade_v33_official_001/proposal_ledger_asian_displacement_handoff_fade_v33.jsonl":
                "6498f917839ed1bd13beb36e8e9eb650fc5aa972d4c1b9f073f52e624a8b9dd4",
            "evidence/orchestrator_state_v2/next_hypothesis_work_order_v34.json":
                "c74effb4d7be63152abdd756b225f89309d939d46d6f5b487c625febdcaee060",
        }
        for relative, digest in expected.items():
            self.assertEqual(hashlib.sha256((root / relative).read_bytes()).hexdigest(), digest)
        work_order = json.loads((root / "evidence/orchestrator_state_v2/next_hypothesis_work_order_v34.json").read_text())
        self.assertEqual(work_order["reason_code"], "FX_SESSION_HANDOFF_FADE_COST_DOMINANT")
        self.assertEqual(
            work_order["single_next_changed_variable"],
            "one_preregistered_turnover_reduction_rule_preserving_all_v32_raw_signals",
        )
        self.assertEqual(work_order["authority"]["external_orders"], 0)


if __name__ == "__main__":
    unittest.main()
