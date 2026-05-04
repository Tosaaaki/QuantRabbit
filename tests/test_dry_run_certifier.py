from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quant_rabbit.certification import DryRunCertifier


class DryRunCertifierTest(unittest.TestCase):
    def test_certifies_complete_dry_run_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)

            summary = DryRunCertifier(
                coverage_path=files["coverage"],
                execution_replay_path=files["execution"],
                post_trade_learning_path=files["learning"],
                order_intents_path=files["intents"],
                live_order_path=files["live_order"],
                position_execution_path=files["position_execution"],
                gpt_decision_path=files["gpt"],
                output_path=root / "cert.json",
                report_path=root / "cert.md",
            ).run()

            self.assertEqual(summary.status, "CERTIFIED")
            self.assertEqual(summary.blockers, 0)
            self.assertIn("Certification Contract", (root / "cert.md").read_text())

    def test_blocks_when_dry_run_contains_send(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["live_order"].write_text(json.dumps({"sent": True, "send_requested": True}))

            summary = DryRunCertifier(
                coverage_path=files["coverage"],
                execution_replay_path=files["execution"],
                post_trade_learning_path=files["learning"],
                order_intents_path=files["intents"],
                live_order_path=files["live_order"],
                position_execution_path=files["position_execution"],
                gpt_decision_path=files["gpt"],
                output_path=root / "cert.json",
                report_path=root / "cert.md",
            ).run()

            self.assertEqual(summary.status, "BLOCKED")
            payload = json.loads((root / "cert.json").read_text())
            self.assertTrue(any("entry send" in item for item in payload["blockers"]))

    def test_blocks_certifiable_coverage_status_with_remaining_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["coverage"].write_text(
                json.dumps(
                    {
                        "status": "TARGET_REACHED_PROTECT",
                        "blockers": ["daily target state is missing; run daily-target-state or plan-campaign"],
                    }
                )
            )

            summary = DryRunCertifier(
                coverage_path=files["coverage"],
                execution_replay_path=files["execution"],
                post_trade_learning_path=files["learning"],
                order_intents_path=files["intents"],
                live_order_path=files["live_order"],
                position_execution_path=files["position_execution"],
                gpt_decision_path=files["gpt"],
                output_path=root / "cert.json",
                report_path=root / "cert.md",
            ).run()

            self.assertEqual(summary.status, "BLOCKED")
            payload = json.loads((root / "cert.json").read_text())
            self.assertTrue(any("coverage optimization still has blockers" in item for item in payload["blockers"]))


def _fixtures(root: Path) -> dict[str, Path]:
    files = {
        "coverage": root / "coverage.json",
        "execution": root / "execution.json",
        "learning": root / "learning.json",
        "intents": root / "intents.json",
        "live_order": root / "live_order.json",
        "position_execution": root / "position_execution.json",
        "gpt": root / "gpt.json",
    }
    files["coverage"].write_text(json.dumps({"status": "LIVE_READY_COVERAGE_READY"}))
    files["execution"].write_text(json.dumps({"status": "TARGET_HIT", "target_jpy": 150, "target_hit": True}))
    files["learning"].write_text(json.dumps({"status": "READY_FOR_REVIEW"}))
    files["intents"].write_text(json.dumps({"results": [_intent()]}))
    files["live_order"].write_text(json.dumps({"sent": False, "send_requested": False}))
    files["position_execution"].write_text(json.dumps({"sent": False, "send_requested": False}))
    files["gpt"].write_text(json.dumps({"status": "ACCEPTED"}))
    return files


def _intent() -> dict:
    return {
        "lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
        "status": "LIVE_READY",
        "risk_metrics": {
            "risk_jpy": 78.5,
            "reward_jpy": 157.0,
            "reward_risk": 2.0,
            "spread_pips": 0.8,
        },
        "intent": {
            "pair": "EUR_USD",
            "side": "LONG",
            "order_type": "STOP-ENTRY",
            "units": 1000,
            "entry": 1.1000,
            "tp": 1.1010,
            "sl": 1.0995,
            "thesis": "test",
            "market_context": {
                "regime": "TREND_CONTINUATION campaign lane",
                "narrative": "trend continuation pressure",
                "chart_story": "trend staircase",
                "method": "TREND_CONTINUATION",
                "invalidation": "SL trades",
            },
        },
    }


if __name__ == "__main__":
    unittest.main()
