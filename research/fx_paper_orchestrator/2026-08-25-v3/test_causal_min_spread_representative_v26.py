from __future__ import annotations

import copy
import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fx_original_indicators import Bar
import run_causal_min_spread_representative_v26 as v26


ROOT = Path(__file__).resolve().parent
PARENT_LEDGER = ROOT / "evidence/run_asian_usd_coherence_persistence_v25_official_001/proposal_ledger_asian_usd_coherence_persistence_v25.jsonl"
PARENT_RESULT = ROOT / "evidence/run_asian_usd_coherence_persistence_v25_official_001/result_asian_usd_coherence_persistence_v25.json"


def make_bars(pair: str, spread: float, future_spread: float = 0.5) -> list[Bar]:
    start = datetime(2026, 4, 1, 4, 20, tzinfo=timezone.utc)
    bars = []
    for index in range(21):
        stamp = start + timedelta(minutes=5 * index)
        current_spread = future_spread if index == 20 else spread
        bid = 1.0
        ask = bid + current_spread
        bars.append(Bar(
            pair=pair,
            time=stamp.isoformat(timespec="seconds").replace("+00:00", "Z"),
            bid_o=bid, bid_h=bid, bid_l=bid, bid_c=bid,
            ask_o=ask, ask_h=ask, ask_l=ask, ask_c=ask,
        ))
    return bars


def row(pair: str) -> dict:
    return {
        "signal_id": f"S::{pair}",
        "pair": pair,
        "utc_day": "2026-04-01",
        "decision_time": "2026-04-01T05:55:00Z",
        "fill_time": "2026-04-01T06:00:00Z",
        "exit_time": "2026-04-01T11:55:00Z",
        "direction": 1,
    }


class ParentSealTest(unittest.TestCase):
    def test_actual_v25_parent_hashes_and_signal_set_are_frozen(self):
        _, rows = v26.load_parent(PARENT_RESULT, PARENT_LEDGER)
        self.assertEqual(len(rows), 500)
        self.assertEqual(v26.signal_id_set_hash(rows), v26.PARENT_SIGNAL_ID_SET_SHA256)


class DeterministicRuleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [row("EUR_USD"), row("USD_JPY")]
        self.corpus = {
            "EUR_USD": make_bars("EUR_USD", 0.0001),
            "USD_JPY": make_bars("USD_JPY", 0.0010),
        }

    def test_exactly_one_causally_cheapest_representative_is_selected(self):
        result = v26.apply_rule(copy.deepcopy(self.rows), self.corpus)
        selected = [item for item in result if item["execution_selected"]]
        self.assertEqual([item["pair"] for item in selected], ["EUR_USD"])
        self.assertEqual(len(selected), 1)

    def test_same_raw_ids_times_directions_and_arm_mask_are_preserved(self):
        result = v26.apply_rule(copy.deepcopy(self.rows), self.corpus)
        fields = ("signal_id", "pair", "utc_day", "decision_time", "fill_time", "exit_time", "direction")
        self.assertEqual(
            [[item[field] for field in fields] for item in result],
            [[item[field] for field in fields] for item in self.rows],
        )
        for item in result:
            self.assertEqual(set(item["arm_actions"]), set(ARMS := v26.ARMS))
            self.assertEqual(len({item["arm_actions"][arm] for arm in ARMS}), 1)

    def test_post_decision_spread_cannot_change_selection(self):
        first = v26.apply_rule(copy.deepcopy(self.rows), self.corpus)
        changed = copy.deepcopy(self.corpus)
        changed["EUR_USD"] = make_bars("EUR_USD", 0.0001, future_spread=0.9)
        changed["USD_JPY"] = make_bars("USD_JPY", 0.0010, future_spread=0.000001)
        second = v26.apply_rule(copy.deepcopy(self.rows), changed)
        self.assertEqual(
            [item["signal_id"] for item in first if item["execution_selected"]],
            [item["signal_id"] for item in second if item["execution_selected"]],
        )

    def test_missing_completed_lookback_fails_closed(self):
        short = {pair: bars[-19:] for pair, bars in self.corpus.items()}
        with self.assertRaisesRegex(ValueError, "missing causal cost lookback"):
            v26.apply_rule(copy.deepcopy(self.rows), short)


class PreregistrationTest(unittest.TestCase):
    def test_preregistration_is_one_variable_and_zero_authority(self):
        prereg = json.loads((ROOT / "CAUSAL_MIN_SPREAD_REPRESENTATIVE_PREREGISTRATION_V26.json").read_text())
        self.assertEqual(prereg["hypothesis_contract"]["changed_variable_count"], 1)
        self.assertEqual(prereg["training_only_rule_selection"]["candidate_rules_evaluated"], 1)
        self.assertFalse(prereg["training_only_rule_selection"]["outcome_fields_read"])
        self.assertEqual(prereg["authority"], v26.AUTHORITY)
        self.assertEqual(prereg["evaluation_contract"]["holdout"]["state"], "UNOPENED")
        self.assertFalse(prereg["evaluation_contract"]["leverage_changed"])


if __name__ == "__main__":
    unittest.main()
