from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

import run_causal_basket_consensus_release_v29 as v29
import run_causal_min_spread_representative_v26 as frozen_v26


ROOT = Path(__file__).resolve().parent
INPUT_ROOT = Path("/Users/tossaki/App/QuantRabbit/logs/replay/oanda_history/20260715T115624Z")
PARENT_LEDGER = ROOT / "evidence/run_asian_usd_coherence_persistence_v25_official_001/proposal_ledger_asian_usd_coherence_persistence_v25.jsonl"
PARENT_RESULT = ROOT / "evidence/run_asian_usd_coherence_persistence_v25_official_001/result_asian_usd_coherence_persistence_v25.json"


def bar(pair: str, stamp: str, mid: float = 1.1) -> frozen_v26.Bar:
    return frozen_v26.Bar(pair, stamp, mid - .0001, mid, mid - .0002, mid - .00005,
                          mid + .0001, mid + .0002, mid, mid + .00015)


def signal(signal_id: str, pair: str, stamp: str, direction: int) -> dict:
    return {"signal_id": signal_id, "pair": pair, "utc_day": stamp[:10],
            "decision_time": stamp.replace("06:00:00", "05:55:00"), "fill_time": stamp,
            "exit_time": stamp.replace("06:00:00", "11:55:00"), "direction": direction}


class BasketConsensusFormulaTest(unittest.TestCase):
    def setUp(self) -> None:
        v29.frozen_v28.runtime_v27.install_timestamp_compatibility()

    def test_usd_vote_units_cover_base_and_quote(self):
        self.assertEqual(v29.implied_usd_direction("USD_JPY", 1), 1)
        self.assertEqual(v29.implied_usd_direction("USD_JPY", -1), -1)
        self.assertEqual(v29.implied_usd_direction("EUR_USD", 1), -1)
        self.assertEqual(v29.implied_usd_direction("EUR_USD", -1), 1)

    def test_two_unanimous_opposite_peers_release_without_own_signal(self):
        t0 = "2026-05-04T06:00:00.000000001Z"
        t1 = "2026-05-05T06:00:00.000000789Z"
        rows = [signal("E1", "EUR_USD", t0, 1),
                signal("J1", "USD_JPY", t1, 1), signal("C1", "USD_CAD", t1, 1)]
        plan = v29.build_pair_plan("EUR_USD", [bar("EUR_USD", t0), bar("EUR_USD", t1)],
                                   rows, "2026-05-01", "2026-06-01")
        self.assertEqual(plan["episodes"][0]["close_reason"], "BASKET_CONSENSUS_RELEASE")
        self.assertEqual(plan["episodes"][0]["exit_time"], t1)
        self.assertTrue(plan["episodes"][0]["exit_at_open"])
        self.assertEqual(plan["episodes"][0]["consensus_audit"]["peer_count"], 2)

    def test_tie_insufficient_and_own_signal_use_unchanged_v28_default(self):
        t0 = "2026-05-04T06:00:00.000000000Z"
        t1 = "2026-05-05T06:00:00.000000000Z"
        rows = [signal("E1", "EUR_USD", t0, 1), signal("J1", "USD_JPY", t1, 1)]
        plan = v29.build_pair_plan("EUR_USD", [bar("EUR_USD", t0), bar("EUR_USD", t1)],
                                   rows, "2026-05-01", "2026-06-01")
        self.assertEqual(plan["episodes"][0]["close_reason"], "TERMINAL_LIQUIDATION")

        own = rows + [signal("E2", "EUR_USD", t1, 1), signal("C1", "USD_CAD", t1, 1)]
        plan = v29.build_pair_plan("EUR_USD", [bar("EUR_USD", t0), bar("EUR_USD", t1)],
                                   own, "2026-05-01", "2026-06-01")
        self.assertEqual(plan["signal_events"][-1]["action"], "HOLD_EXISTING_NO_ADD_NO_EXPIRY_EXTENSION")
        self.assertFalse(any(e["event_type"] == "BASKET_CONSENSUS_RELEASE" for e in plan["close_events"]))

    def test_max_age_has_precedence_over_consensus(self):
        t0 = "2026-05-01T06:00:00.000000000Z"
        t1 = "2026-05-03T06:00:00.000000000Z"
        rows = [signal("E1", "EUR_USD", t0, 1),
                signal("J1", "USD_JPY", t1, 1), signal("C1", "USD_CAD", t1, 1)]
        plan = v29.build_pair_plan("EUR_USD", [bar("EUR_USD", t0), bar("EUR_USD", t1)],
                                   rows, "2026-05-01", "2026-06-01")
        self.assertEqual(plan["episodes"][0]["close_reason"], "MAX_AGE_CLOSE")


class ActualTrainingStructureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        v29.frozen_v28.runtime_v27.install_timestamp_compatibility()
        _, cls.parent_rows = frozen_v26.load_parent(PARENT_RESULT, PARENT_LEDGER)
        cls.corpus, _ = frozen_v26.load_corpus(INPUT_ROOT)
        cls.plans = v29.build_period_plans(cls.corpus, cls.parent_rows, "2026-03-11", "2026-05-01")
        cls.rows = v29.build_execution_ledger(cls.parent_rows, cls.corpus)

    def test_preregistered_training_decomposition_matches_code_without_outcomes(self):
        prereg = json.loads((ROOT / "CAUSAL_BASKET_CONSENSUS_RELEASE_PREREGISTRATION_V29.json").read_text())
        selection = prereg["training_only_rule_selection"]
        closes = Counter(e["event_type"] for p in self.plans.values() for e in p["close_events"])
        pairs = Counter(e["pair"] for p in self.plans.values() for e in p["close_events"]
                        if e["event_type"] == "BASKET_CONSENSUS_RELEASE")
        self.assertEqual(closes["BASKET_CONSENSUS_RELEASE"], selection["selected_rule_structural_release_count"])
        self.assertEqual(dict(sorted(pairs.items())), selection["selected_rule_structural_release_pairs"])
        self.assertEqual(selection["candidate_rules_compared"], 1)
        self.assertFalse(selection["return_outcome_consulted"])
        self.assertFalse(selection["cost_consulted"])

    def test_all_500_v25_signals_and_arm_actions_are_preserved(self):
        identity = ("signal_id", "pair", "utc_day", "direction", "decision_time", "fill_time", "exit_time")
        self.assertEqual(len(self.rows), 500)
        self.assertEqual([[r[f] for f in identity] for r in self.rows],
                         [[r[f] for f in identity] for r in self.parent_rows])
        self.assertEqual(frozen_v26.signal_id_set_hash(self.rows), v29.PARENT_SIGNAL_ID_SET_SHA256)
        for row in self.rows:
            self.assertTrue(row["execution_selected"])
            self.assertEqual(set(row["arm_actions"]), set(v29.ARMS))
            self.assertEqual(len(set(row["arm_actions"].values())), 1)

    def test_v28_reference_hashes_are_frozen(self):
        result = ROOT / "evidence/run_causal_basket_hold_v28_official_001/result_causal_basket_hold_v28.json"
        ledger = ROOT / "evidence/run_causal_basket_hold_v28_official_001/proposal_ledger_causal_basket_hold_v28.jsonl"
        self.assertEqual(frozen_v26.sha256_file(result), v29.V28_RESULT_SHA256)
        self.assertEqual(frozen_v26.sha256_file(ledger), v29.V28_LEDGER_SHA256)

    def test_currency_inventory_cap_uses_base_and_quote_once(self):
        metrics = v29.arm_metrics(self.plans, "RAW_SIGNAL")
        self.assertLessEqual(metrics["max_currency_abs_exposure_nav"], 1.0)
        self.assertEqual(metrics["terminal_open_inventory"], 0)
        self.assertEqual(metrics["terminal_inventory_mtm"], 0.0)


if __name__ == "__main__":
    unittest.main()
