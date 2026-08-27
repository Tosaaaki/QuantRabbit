from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

import run_causal_consensus_release_persistence_v31 as v31
import run_causal_min_spread_representative_v26 as frozen_v26


ROOT = Path(__file__).resolve().parent
INPUT_ROOT = Path("/Users/tossaki/App/QuantRabbit/logs/replay/oanda_history/20260715T115624Z")
PARENT_LEDGER = ROOT / "evidence/run_asian_usd_coherence_persistence_v25_official_001/proposal_ledger_asian_usd_coherence_persistence_v25.jsonl"
PARENT_RESULT = ROOT / "evidence/run_asian_usd_coherence_persistence_v25_official_001/result_asian_usd_coherence_persistence_v25.json"


def bar(pair: str, stamp: str, mid: float = 1.1) -> frozen_v26.Bar:
    return frozen_v26.Bar(pair, stamp, mid - .0001, mid, mid - .0002, mid - .00005,
                          mid + .0001, mid + .0002, mid, mid + .00015)


def signal(signal_id: str, pair: str, stamp: str, direction: int) -> dict:
    return {
        "signal_id": signal_id, "pair": pair, "utc_day": stamp[:10],
        "decision_time": stamp.replace("06:00:00", "05:55:00"),
        "fill_time": stamp, "exit_time": stamp, "direction": direction,
    }


def corpus(*stamps: str) -> dict[str, list]:
    return {pair: [bar(pair, stamp) for stamp in stamps] for pair in sorted(v31.UNIVERSE)}


def persistence_rows(t0: str, t1: str, t2: str) -> list[dict]:
    return [
        signal("E0", "EUR_USD", t0, 1),
        signal("J0", "USD_JPY", t0, -1),
        signal("G0", "GBP_USD", t0, 1),
        signal("C0", "USD_CAD", t0, -1),
        signal("H0", "USD_CHF", t0, -1),
        signal("J1", "USD_JPY", t1, 1),
        signal("G1", "GBP_USD", t1, -1),
        signal("C2", "USD_CAD", t2, 1),
        signal("H2", "USD_CHF", t2, 1),
    ]


class PersistenceFormulaTest(unittest.TestCase):
    def setUp(self) -> None:
        v31.frozen_v30.frozen_v29.frozen_v28.runtime_v27.install_timestamp_compatibility()

    def test_first_confirmation_only_arms_and_next_completed_event_releases(self):
        t0 = "2026-05-04T06:00:00.000000001Z"
        t1 = "2026-05-04T12:00:00.000000123Z"
        t2 = "2026-05-04T18:00:00.000000789Z"
        plans = v31.build_period_plans(corpus(t0, t1, t2), persistence_rows(t0, t1, t2),
                                       "2026-05-01", "2026-06-01")
        target = plans["EUR_USD"]
        armed = [e for e in target["persistence_events"] if e["event_type"] == "PERSISTENCE_ARMED"]
        releases = [e for e in target["close_events"]
                    if e["event_type"] == "BASKET_CONSENSUS_PERSISTENCE_RELEASE"]
        self.assertEqual([e["time"] for e in armed], [t1])
        self.assertEqual([e["time"] for e in releases], [t2])
        self.assertEqual(releases[0]["consensus_audit"]["prior_confirmation"]["time"], t1)

    def test_intervening_peer_shortage_clears_confirmation_without_retroactivity(self):
        t0 = "2026-05-04T06:00:00.000000001Z"
        t1 = "2026-05-04T12:00:00.000000123Z"
        t2 = "2026-05-04T15:00:00.000000456Z"
        t3 = "2026-05-04T18:00:00.000000789Z"
        rows = persistence_rows(t0, t1, t3)
        rows.append(signal("A2", "AUD_USD", t2, 1))
        plans = v31.build_period_plans(corpus(t0, t1, t2, t3), rows,
                                       "2026-05-01", "2026-06-01")
        target = plans["EUR_USD"]
        self.assertFalse(any(e["event_type"] == "BASKET_CONSENSUS_PERSISTENCE_RELEASE"
                             for e in target["close_events"]))
        resets = [e for e in target["persistence_events"] if e["event_type"] == "PERSISTENCE_RESET"]
        self.assertTrue(any(e["time"] == t2 and e["reason"] == "MISSING_TIE_OR_PEER_SHORTAGE"
                            for e in resets))

    def test_finite_max_age_precedes_second_confirmation(self):
        t0 = "2026-05-01T06:00:00.000000001Z"
        t1 = "2026-05-02T06:00:00.000000123Z"
        t2 = "2026-05-03T06:00:00.000000789Z"
        plans = v31.build_period_plans(corpus(t0, t1, t2), persistence_rows(t0, t1, t2),
                                       "2026-05-01", "2026-06-01")
        target = plans["EUR_USD"]
        self.assertEqual(target["episodes"][0]["close_reason"], "MAX_AGE_CLOSE")
        self.assertFalse(any(e["event_type"] == "BASKET_CONSENSUS_PERSISTENCE_RELEASE"
                             for e in target["close_events"]))

    def test_own_signal_clears_pending_and_uses_unchanged_hold_action(self):
        t0 = "2026-05-04T06:00:00.000000001Z"
        t1 = "2026-05-04T12:00:00.000000123Z"
        t2 = "2026-05-04T18:00:00.000000789Z"
        rows = persistence_rows(t0, t1, t2)
        rows.append(signal("E2", "EUR_USD", t2, 1))
        plans = v31.build_period_plans(corpus(t0, t1, t2), rows,
                                       "2026-05-01", "2026-06-01")
        target = plans["EUR_USD"]
        self.assertEqual(target["signal_events"][-1]["action"],
                         "HOLD_EXISTING_NO_ADD_NO_EXPIRY_EXTENSION")
        self.assertFalse(any(e["event_type"] == "BASKET_CONSENSUS_PERSISTENCE_RELEASE"
                             for e in target["close_events"]))


class ActualTrainingStructureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        v31.frozen_v30.frozen_v29.frozen_v28.runtime_v27.install_timestamp_compatibility()
        _, cls.parent_rows = frozen_v26.load_parent(PARENT_RESULT, PARENT_LEDGER)
        cls.corpus, _ = frozen_v26.load_corpus(INPUT_ROOT)
        cls.plans = v31.build_period_plans(cls.corpus, cls.parent_rows,
                                          "2026-03-11", "2026-05-01")
        cls.rows = v31.build_execution_ledger(cls.parent_rows, cls.corpus)

    def test_preregistered_one_candidate_training_structure_matches_without_outcomes(self):
        prereg = json.loads((ROOT / "CAUSAL_CONSENSUS_RELEASE_PERSISTENCE_PREREGISTRATION_V31.json").read_text())
        selection = prereg["training_only_persistence_selection"]
        persistence = [e for p in self.plans.values() for e in p["persistence_events"]]
        closes = [e for p in self.plans.values() for e in p["close_events"]]
        actions = [e for p in self.plans.values() for e in p["signal_events"]]
        self.assertEqual(selection["candidate_rules_preregistered"], 1)
        self.assertEqual(selection["candidate_confirmation_counts_compared_by_outcome"], 0)
        self.assertEqual(selection["required_consecutive_confirmations"], 2)
        self.assertEqual(selection["first_confirmations_armed"],
                         sum(e["event_type"] == "PERSISTENCE_ARMED" for e in persistence))
        self.assertEqual(selection["persistence_confirmed_release_count"],
                         sum(e["event_type"] == "PERSISTENCE_CONFIRMED" for e in persistence))
        self.assertEqual(selection["persistence_reset_count"],
                         sum(e["event_type"] == "PERSISTENCE_RESET" for e in persistence))
        self.assertEqual(selection["selected_rule_state_action_counts"],
                         dict(sorted(Counter(e["action"] for e in actions).items())))
        self.assertEqual(selection["selected_rule_close_counts"],
                         dict(sorted(Counter(e["event_type"] for e in closes).items())))
        self.assertFalse(selection["price_consulted"])
        self.assertFalse(selection["return_outcome_consulted"])
        self.assertFalse(selection["cost_consulted"])

    def test_all_500_v25_signals_and_arm_actions_are_preserved(self):
        identity = ("signal_id", "pair", "utc_day", "direction", "decision_time", "fill_time", "exit_time")
        self.assertEqual(len(self.rows), 500)
        self.assertEqual([[row[field] for field in identity] for row in self.rows],
                         [[row[field] for field in identity] for row in self.parent_rows])
        self.assertEqual(frozen_v26.signal_id_set_hash(self.rows), v31.PARENT_SIGNAL_ID_SET_SHA256)
        for row in self.rows:
            self.assertTrue(row["execution_selected"])
            self.assertEqual(set(row["arm_actions"]), set(v31.ARMS))
            self.assertEqual(len(set(row["arm_actions"].values())), 1)

    def test_v30_reference_hashes_are_frozen(self):
        result = ROOT / "evidence/run_causal_consensus_release_scope_v30_official_001/result_causal_consensus_release_scope_v30.json"
        ledger = ROOT / "evidence/run_causal_consensus_release_scope_v30_official_001/proposal_ledger_causal_consensus_release_scope_v30.jsonl"
        self.assertEqual(frozen_v26.sha256_file(result), v31.V30_RESULT_SHA256)
        self.assertEqual(frozen_v26.sha256_file(ledger), v31.V30_LEDGER_SHA256)

    def test_currency_inventory_cap_and_terminal_liquidation(self):
        metrics = v31.arm_metrics(self.plans, "RAW_SIGNAL")
        self.assertLessEqual(metrics["max_currency_abs_exposure_nav"], 1.0)
        self.assertEqual(metrics["terminal_open_inventory"], 0)
        self.assertEqual(metrics["terminal_inventory_mtm"], 0.0)


if __name__ == "__main__":
    unittest.main()
