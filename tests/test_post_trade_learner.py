from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quant_rabbit.learning import PostTradeLearner


class PostTradeLearnerTest(unittest.TestCase):
    def test_loss_beyond_cap_creates_blocker_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outcome = root / "outcome.json"
            live_order = root / "live_order.json"
            trader_decision = root / "decision.json"
            outcome.write_text(
                json.dumps(
                    {
                        "lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                        "pair": "EUR_USD",
                        "direction": "LONG",
                        "realized_pl_jpy": -900.0,
                        "close_reason": "SL slipped through old geometry",
                    }
                )
            )
            live_order.write_text(
                json.dumps(
                    {
                        "status": "SENT",
                        "lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                        "sent": True,
                        "order_request": {"instrument": "EUR_USD"},
                    }
                )
            )
            trader_decision.write_text(
                json.dumps({"selected_lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION"})
            )

            summary = PostTradeLearner(
                outcome_path=outcome,
                live_order_path=live_order,
                position_execution_path=root / "position_execution.json",
                trader_decision_path=trader_decision,
                gpt_decision_path=root / "gpt.json",
                output_path=root / "learning.json",
                report_path=root / "learning.md",
                max_loss_jpy=500,
            ).run()

            self.assertEqual(summary.status, "BLOCKED")
            self.assertEqual(summary.profile_update_candidates, 1)
            payload = json.loads((root / "learning.json").read_text())
            self.assertEqual(payload["candidates"][0]["recommendation"], "BLOCK_UNTIL_NEW_EVIDENCE")
            self.assertIn("breached current 500 JPY cap", payload["candidates"][0]["reason"])

    def test_observational_receipt_does_not_mutate_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            live_order = root / "live_order.json"
            live_order.write_text(
                json.dumps(
                    {
                        "status": "STAGED",
                        "lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                        "sent": False,
                        "order_request": {"instrument": "EUR_USD"},
                    }
                )
            )

            summary = PostTradeLearner(
                live_order_path=live_order,
                position_execution_path=root / "position_execution.json",
                trader_decision_path=root / "decision.json",
                gpt_decision_path=root / "gpt.json",
                output_path=root / "learning.json",
                report_path=root / "learning.md",
            ).run()

            self.assertEqual(summary.status, "READY_FOR_REVIEW")
            payload = json.loads((root / "learning.json").read_text())
            self.assertEqual(payload["candidates"][0]["recommendation"], "NO_PROFILE_CHANGE")


if __name__ == "__main__":
    unittest.main()
