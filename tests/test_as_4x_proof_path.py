from __future__ import annotations

import unittest

from tools import build_as_4x_proof_path as proof_path
from tools import build_as_live_ready_evidence_loop as evidence_loop


class As4xProofPathTests(unittest.TestCase):
    def test_historical_or_negative_bidask_replay_is_not_live_grade_proof(self) -> None:
        missing = evidence_loop._missing_proof_map(
            {
                "source_evidence": {
                    "historical_only": True,
                    "bidask_rule_status": "LIVE_BLOCK_NEGATIVE_EXPECTANCY",
                },
                "exact_proof_gaps": [],
                "current_blockers": [],
                "status": "DRY_RUN_BLOCKED",
                "risk_allowed": False,
            }
        )

        self.assertFalse(missing["fresh_744h_replay"])
        self.assertFalse(missing["s5_bidask_spread_included_replay"])

    def test_fresh_direction_evidence_marks_under_sampled_gap(self) -> None:
        replay = {
            "price_truth_coverage": {
                "status": "PRICE_TRUTH_OK",
                "adoption_level": "PAIR_LOCAL_RANK_ONLY",
            },
            "forecast_sample_coverage": {
                "pairs": [{"pair": "GBP_USD", "evaluated_samples": 13}],
                "under_sampled_pair_directions": [
                    {
                        "pair": "GBP_USD",
                        "direction": "UP",
                        "evaluated_samples": 2,
                        "evaluated_active_days": 2,
                    }
                ],
            },
            "precision_rules": {
                "edge_rules": [],
                "daily_stable_edge_rules": [],
                "negative_rules": [],
                "contrarian_edge_rules": [],
                "rejected_daily_stability_segments": [],
            },
            "segments": {"by_pair_direction": []},
        }

        evidence = proof_path.fresh_direction_evidence(replay, "GBP_USD", "UP")

        self.assertEqual(evidence["status"], "EVIDENCE_GAP_UNDER_SAMPLED")
        self.assertFalse(evidence["live_grade_support"])
        self.assertFalse(evidence["can_create_live_permission"])


if __name__ == "__main__":
    unittest.main()
