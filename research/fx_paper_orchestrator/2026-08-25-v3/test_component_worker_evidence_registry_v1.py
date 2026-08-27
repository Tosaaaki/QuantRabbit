import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import component_worker_evidence_registry_v1 as registry
import component_worker_evidence_registry_v2 as sign_aware_registry
import paper_research_orchestrator_v2 as orchestrator


ROOT = Path(__file__).resolve().parent


class ComponentWorkerEvidenceRegistryV1Test(unittest.TestCase):
    def test_policy_keeps_candidates_unadmitted_and_profit_gate_unchanged(self):
        policy = json.loads((ROOT / registry.POLICY_PATH).read_text())
        self.assertEqual(policy["candidate_status"], "RESEARCH_COMPONENT_CANDIDATE_UNADMITTED")
        self.assertFalse(policy["provisional_only_gate"]["monthly_2x_or_holdout_reproduction_may_be_inferred"])
        self.assertFalse(policy["provisional_only_gate"]["strategy_adoption_may_be_inferred"])
        self.assertEqual(policy["authority"], registry.AUTHORITY)

    def test_independence_is_currency_time_cluster_not_pair_ticket_count(self):
        policy = json.loads((ROOT / registry.POLICY_PATH).read_text())
        self.assertEqual(policy["deduplication"]["cluster_unit"],
                         "SIGNED_CURRENCY_BY_COMPLETED_M5_TIMESTAMP")
        self.assertTrue(policy["deduplication"]["pair_ticket_count_is_not_independence_evidence"])
        self.assertFalse(policy["deduplication"]["v39_may_count_as_independent_worker"])

    def test_validate_rejects_changed_embedded_registry_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / registry.REGISTRY_PATH
            path.parent.mkdir(parents=True)
            payload = {
                "registry_sha256": "wrong", "authority": registry.AUTHORITY,
                "existing_profit_gate_changed": False, "strategy_adoption_authorized": False,
                "v41_artifact_hashes_before_and_after": {"unchanged": True}, "candidates": [],
            }
            path.write_text(json.dumps(payload))
            with self.assertRaises(registry.ComponentEvidenceError):
                registry.validate(root)

    def test_actual_registry_keeps_v38_v40_provisional_and_deduplicates_v39(self):
        payload = registry.validate(ROOT)
        self.assertEqual(payload["positive_provisional_candidate_count"], 2)
        self.assertEqual({item["cycle_id"] for item in payload["candidates"]}, {"V38", "V40"})
        self.assertTrue(all(item["qualification"] == "PROVISIONAL" for item in payload["candidates"]))
        self.assertFalse(payload["portfolio_composition_proposal_allowed"])
        self.assertFalse(payload["strategy_adoption_authorized"])
        self.assertFalse(next(item for item in payload["deduplicated_variants"]
                              if item["cycle_id"] == "V39")["counted_as_independent_worker"])

    def test_v1_checkpoint_used_absolute_correlation_and_stopped_portfolio(self):
        payload = registry.validate(ROOT)
        self.assertFalse(payload["portfolio_composition_proposal_allowed"])
        pair = payload["pairwise_currency_time_independence"][0]
        self.assertGreater(abs(pair["daily_base_return_correlation"]), 0.35)
        self.assertFalse(pair["independence_gate_passed"])

    def test_future_portfolio_proposal_requires_two_candidates_and_independence(self):
        eligible = {
            "positive_provisional_candidate_count": 2,
            "portfolio_composition_proposal_allowed": True,
            "strategy_adoption_authorized": False,
            "existing_profit_gate_changed": False,
        }
        with mock.patch.object(sign_aware_registry, "validate", return_value=eligible):
            self.assertTrue(orchestrator.component_portfolio_proposal_eligible(ROOT))
        insufficient = {**eligible, "positive_provisional_candidate_count": 1}
        with mock.patch.object(sign_aware_registry, "validate", return_value=insufficient):
            self.assertFalse(orchestrator.component_portfolio_proposal_eligible(ROOT))

    def test_v41_signal_action_result_and_seal_hashes_are_unchanged(self):
        payload = registry.validate(ROOT)
        binding = payload["v41_artifact_hashes_before_and_after"]
        self.assertTrue(binding["unchanged"])
        self.assertEqual(binding["before"], binding["after"])


if __name__ == "__main__":
    unittest.main()
