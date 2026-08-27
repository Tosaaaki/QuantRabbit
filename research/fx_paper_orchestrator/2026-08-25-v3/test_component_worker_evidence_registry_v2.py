import json
import unittest
from pathlib import Path

import component_worker_evidence_registry_v2 as registry
import paper_research_orchestrator_v2 as orchestrator


ROOT = Path(__file__).resolve().parent


class ComponentWorkerEvidenceRegistryV2Test(unittest.TestCase):
    def test_policy_routes_negative_correlation_to_complementarity_review(self):
        policy = json.loads((ROOT / registry.POLICY_PATH).read_text())
        self.assertEqual(policy["correlation_routing"]["negative_complementarity_review_if_strictly_less_than"], -0.35)
        self.assertTrue(policy["correlation_routing"]["absolute_correlation_rejection_forbidden"])
        self.assertEqual(policy["admission_boundary"]["qualification"], "PROVISIONAL")
        self.assertFalse(policy["admission_boundary"]["strategy_adoption_authorized"])

    def test_actual_negative_pair_receives_full_sign_aware_review(self):
        payload = registry.validate(ROOT)
        review = payload["sign_aware_pair_reviews"][0]
        self.assertLess(review["daily_base_return_correlation"], -0.35)
        self.assertEqual(review["routing"], "COMPLEMENTARITY_REVIEW")
        self.assertIn("fixed_equal_sleeve_cvar_contribution", review["adverse_tail"])
        for field in (
            "signal_timestamp_jaccard_overlap", "pair_event_jaccard_overlap",
            "currency_exposure_sign_inversion_rate_at_common_timestamps",
            "daily_downside_co_loss_rate", "adverse_tail", "currency_time_cluster_n_eff",
        ):
            self.assertIn(field, review)

    def test_diagnostic_never_infers_profit_or_holdout_pass(self):
        payload = registry.validate(ROOT)
        diagnostic = payload["achievability_diagnostic"]
        self.assertFalse(diagnostic["oracle_used"])
        self.assertFalse(diagnostic["post_hoc_leverage_used"])
        self.assertFalse(diagnostic["profit_gate_pass_inferred"])
        self.assertFalse(diagnostic["holdout_reproduction_inferred"])
        for month in diagnostic["monthly"].values():
            for arm in month.values():
                self.assertTrue(arm["linear_worker_count_is_diagnostic_only"])
                self.assertTrue(arm["linear_worker_count_may_not_pass_profit_gate"])
                self.assertEqual(arm["current_independent_worker_count"], 2)
                self.assertEqual(arm["required_independent_worker_count_linear_diagnostic"],
                                 arm["linear_additive_uncapped_worker_lower_bound"])

    def test_v1_v41_and_v42_artifacts_are_unchanged(self):
        payload = registry.validate(ROOT)
        protected = payload["protected_strategy_artifact_hashes"]
        self.assertTrue(protected["unchanged"])
        self.assertEqual(protected["before"], protected["after"])

    def test_future_generator_may_propose_portfolio_without_rewriting_v42(self):
        payload = registry.validate(ROOT)
        self.assertTrue(payload["portfolio_composition_proposal_allowed"])
        self.assertTrue(orchestrator.component_portfolio_proposal_eligible(ROOT))
        self.assertEqual(payload["protected_strategy_artifact_hashes"]["before"]["v42_work_order"],
                         "29d541646f57efffe543007d94ce0958a2fa3cc68180cf521101886a8b09b524")


if __name__ == "__main__":
    unittest.main()
