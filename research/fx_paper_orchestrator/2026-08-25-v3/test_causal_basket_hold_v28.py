from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import run_causal_basket_hold_v28 as v28
import run_causal_min_spread_representative_v26 as frozen_v26


ROOT = Path(__file__).resolve().parent
INPUT_ROOT = Path("/Users/tossaki/App/QuantRabbit/logs/replay/oanda_history/20260715T115624Z")
PARENT_LEDGER = ROOT / "evidence/run_asian_usd_coherence_persistence_v25_official_001/proposal_ledger_asian_usd_coherence_persistence_v25.jsonl"
PARENT_RESULT = ROOT / "evidence/run_asian_usd_coherence_persistence_v25_official_001/result_asian_usd_coherence_persistence_v25.json"


def bar(pair: str, stamp: str, mid: float = 1.1) -> frozen_v26.Bar:
    return frozen_v26.Bar(
        pair, stamp,
        mid - 0.0001, mid, mid - 0.0002, mid - 0.00005,
        mid + 0.0001, mid + 0.0002, mid, mid + 0.00015,
    )


def signal(signal_id: str, stamp: str, direction: int) -> dict:
    return {
        "signal_id": signal_id,
        "pair": "EUR_USD",
        "utc_day": stamp[:10],
        "decision_time": stamp.replace("06:00:00", "05:55:00"),
        "fill_time": stamp,
        "exit_time": stamp.replace("06:00:00", "11:55:00"),
        "direction": direction,
    }


class CausalBasketHoldRuleTest(unittest.TestCase):
    def setUp(self) -> None:
        v28.runtime_v27.install_timestamp_compatibility()

    def test_same_direction_signal_holds_without_add_or_expiry_extension(self):
        rows = [
            signal("S1", "2026-05-04T06:00:00.000000000Z", 1),
            signal("S2", "2026-05-05T06:00:00.000000000Z", 1),
        ]
        bars = [
            bar("EUR_USD", "2026-05-04T06:00:00.000000000Z"),
            bar("EUR_USD", "2026-05-05T06:00:00.000000000Z", 1.101),
            bar("EUR_USD", "2026-05-06T06:00:00.000000000Z", 1.102),
        ]
        plan = v28.build_pair_plan("EUR_USD", bars, rows, "2026-05-01", "2026-06-01")
        self.assertEqual(
            [event["action"] for event in plan["signal_events"]],
            ["OPEN_FIXED_ONE_SEVENTH", "HOLD_EXISTING_NO_ADD_NO_EXPIRY_EXTENSION"],
        )
        self.assertEqual(len(plan["episodes"]), 1)
        self.assertEqual(plan["episodes"][0]["source_signal_ids"], ["S1", "S2"])
        self.assertEqual(plan["episodes"][0]["inventory_age_seconds"], v28.TARGET_HOLD_SECONDS)

    def test_opposite_signal_closes_and_reverses_at_same_completed_open(self):
        rows = [
            signal("S1", "2026-05-04T06:00:00.000000000Z", 1),
            signal("S2", "2026-05-05T06:00:00.000000000Z", -1),
        ]
        bars = [
            bar("EUR_USD", "2026-05-04T06:00:00.000000000Z"),
            bar("EUR_USD", "2026-05-05T06:00:00.000000000Z", 1.101),
            bar("EUR_USD", "2026-05-07T06:00:00.000000000Z", 1.099),
        ]
        plan = v28.build_pair_plan("EUR_USD", bars, rows, "2026-05-01", "2026-06-01")
        self.assertEqual(plan["signal_events"][1]["action"], "REVERSE_FIXED_ONE_SEVENTH")
        self.assertEqual(plan["episodes"][0]["close_reason"], "OPPOSITE_SIGNAL_CLOSE")
        self.assertTrue(plan["episodes"][0]["exit_at_open"])
        self.assertEqual(plan["episodes"][0]["exit_time"], rows[1]["fill_time"])

    def test_missing_target_bar_uses_first_completed_open_within_hard_cap(self):
        rows = [signal("S1", "2026-03-12T06:00:00.000000000Z", 1)]
        bars = [
            bar("EUR_USD", "2026-03-12T06:00:00.000000000Z"),
            bar("EUR_USD", "2026-03-15T21:00:00.000000000Z", 1.102),
        ]
        plan = v28.build_pair_plan("EUR_USD", bars, rows, "2026-03-01", "2026-04-01")
        episode = plan["episodes"][0]
        self.assertEqual(episode["close_reason"], "MAX_AGE_CLOSE")
        self.assertEqual(episode["inventory_age_seconds"], 313200.0)
        self.assertLessEqual(episode["inventory_age_seconds"], v28.HARD_MAX_AGE_SECONDS)

    def test_hard_age_violation_fails_closed(self):
        rows = [signal("S1", "2026-05-01T06:00:00.000000000Z", 1)]
        bars = [
            bar("EUR_USD", "2026-05-01T06:00:00.000000000Z"),
            bar("EUR_USD", "2026-05-06T06:05:00.000000000Z"),
        ]
        with self.assertRaisesRegex(ValueError, "hard inventory age exceeded"):
            v28.build_pair_plan("EUR_USD", bars, rows, "2026-05-01", "2026-06-01")

    def test_terminal_inventory_is_liquidated_and_mtm_is_realized(self):
        rows = [signal("S1", "2026-05-30T06:00:00.000000000Z", 1)]
        bars = [
            bar("EUR_USD", "2026-05-30T06:00:00.000000000Z"),
            bar("EUR_USD", "2026-05-31T23:55:00.000000000Z", 1.103),
        ]
        plan = v28.build_pair_plan("EUR_USD", bars, rows, "2026-05-01", "2026-06-01")
        self.assertEqual(plan["episodes"][0]["close_reason"], "TERMINAL_LIQUIDATION")
        self.assertFalse(plan["episodes"][0]["exit_at_open"])
        marks, active, _directions, returns = v28._pair_marks(plan, "RAW_SIGNAL")
        self.assertEqual(active[bars[-1].time], 0)
        self.assertEqual(len(returns), 1)
        self.assertTrue(math.isfinite(marks[bars[-1].time]))


class FrozenEvidenceAndActualLedgerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        v28.runtime_v27.install_timestamp_compatibility()
        cls.parent, cls.parent_rows = frozen_v26.load_parent(PARENT_RESULT, PARENT_LEDGER)
        cls.corpus, _ = frozen_v26.load_corpus(INPUT_ROOT)
        cls.rows = v28.build_execution_ledger(cls.parent_rows, cls.corpus)

    def test_v25_and_v27_evidence_hashes_remain_frozen(self):
        self.assertEqual(frozen_v26.sha256_file(PARENT_RESULT), v28.PARENT_RESULT_SHA256)
        self.assertEqual(frozen_v26.sha256_file(PARENT_LEDGER), v28.PARENT_LEDGER_SHA256)
        self.assertEqual(
            frozen_v26.sha256_file(ROOT / "evidence/run_causal_min_spread_representative_v27_official_001/result_causal_min_spread_representative_v27.json"),
            v28.V27_RESULT_SHA256,
        )
        self.assertEqual(
            frozen_v26.sha256_file(ROOT / "evidence/run_causal_min_spread_representative_v27_official_001/proposal_ledger_causal_min_spread_representative_v27.jsonl"),
            v28.V27_LEDGER_SHA256,
        )

    def test_all_500_raw_signals_and_frozen_identity_are_preserved(self):
        identity = ("signal_id", "pair", "utc_day", "direction", "decision_time", "fill_time", "exit_time")
        self.assertEqual(len(self.rows), 500)
        self.assertEqual(
            [[row[field] for field in identity] for row in self.rows],
            [[row[field] for field in identity] for row in self.parent_rows],
        )
        self.assertEqual(frozen_v26.signal_id_set_hash(self.rows), v28.PARENT_SIGNAL_ID_SET_SHA256)

    def test_every_signal_has_identical_action_across_all_cost_arms(self):
        for row in self.rows:
            self.assertEqual(set(row["arm_actions"]), set(v28.ARMS))
            self.assertEqual(len(set(row["arm_actions"].values())), 1)
            self.assertTrue(row["execution_selected"])

    def test_preregistered_training_structure_uses_one_candidate_and_no_outcome(self):
        prereg = json.loads((ROOT / "CAUSAL_BASKET_HOLD_PREREGISTRATION_V28.json").read_text())
        selection = prereg["training_only_rule_selection"]
        self.assertEqual(selection["candidate_rules_compared"], 1)
        self.assertEqual(selection["training_signals"], 202)
        self.assertEqual(selection["training_effective_days"], 33)
        self.assertFalse(selection["return_outcome_consulted"])
        self.assertFalse(selection["cost_consulted"])
        self.assertEqual(prereg["execution_rule"]["hard_max_age_seconds"], 345600)


if __name__ == "__main__":
    unittest.main()
