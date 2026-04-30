from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.strategy.intent_generator import IntentGenerator


class IntentGeneratorTest(unittest.TestCase):
    def test_requires_snapshot_before_pricing_intents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = _campaign(root)
            strategy = _strategy(root)
            output = root / "intents.json"
            report = root / "intents.md"

            summary = IntentGenerator(
                campaign_plan=campaign,
                strategy_profile=strategy,
                output_path=output,
                report_path=report,
            ).run()

            self.assertEqual(summary.generated, 0)
            self.assertEqual(summary.needs_snapshot, 1)
            self.assertIn("NEEDS_BROKER_SNAPSHOT", report.read_text())

    def test_generates_and_risk_checks_priced_intent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = _campaign(root)
            strategy = _strategy(root)
            snapshot = _snapshot(root)
            output = root / "intents.json"
            report = root / "intents.md"

            summary = IntentGenerator(
                campaign_plan=campaign,
                strategy_profile=strategy,
                output_path=output,
                report_path=report,
            ).run(snapshot_path=snapshot)

            self.assertEqual(summary.generated, 1)
            self.assertEqual(summary.dry_run_passed, 1)
            self.assertEqual(summary.live_ready, 0)
            payload = json.loads(output.read_text())
            result = payload["results"][0]
            self.assertEqual(result["status"], "DRY_RUN_PASSED")
            self.assertEqual(result["intent"]["pair"], "EUR_USD")
            self.assertEqual(result["intent"]["market_context"]["method"], "TREND_CONTINUATION")
            self.assertTrue(result["live_blockers"])


def _campaign(root: Path) -> Path:
    path = root / "campaign.json"
    path.write_text(
        json.dumps(
            {
                "lanes": [
                    {
                        "desk": "trend_trader",
                        "pair": "EUR_USD",
                        "direction": "LONG",
                        "method": "TREND_CONTINUATION",
                        "adoption": "RISK_REPAIR_DRY_RUN",
                        "campaign_role": "NOW_IF_REPAIRED",
                        "reason": "trend continuation pressure",
                        "required_receipt": "dry-run under loss cap",
                        "blockers": ["old sizing broke the loss cap"],
                        "story_examples": ["quality_audit: green staircase into upper band"],
                    },
                    {
                        "desk": "event_risk_trader",
                        "pair": "EUR_USD",
                        "direction": "BOTH",
                        "method": "EVENT_RISK",
                        "adoption": "RISK_OVERLAY",
                    },
                ]
            }
        )
    )
    return path


def _strategy(root: Path) -> Path:
    path = root / "strategy.json"
    path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "pair": "EUR_USD",
                        "direction": "LONG",
                        "status": "RISK_REPAIR_CANDIDATE",
                        "required_fix": "edge exists but old sizing broke the loss cap",
                    }
                ]
            }
        )
    )
    return path


def _snapshot(root: Path) -> Path:
    path = root / "snapshot.json"
    now = datetime.now(timezone.utc).isoformat()
    path.write_text(
        json.dumps(
            {
                "fetched_at_utc": now,
                "positions": [],
                "orders": [],
                "quotes": {
                    "EUR_USD": {"bid": 1.17322, "ask": 1.17330, "timestamp_utc": now},
                    "USD_JPY": {"bid": 156.64, "ask": 156.648, "timestamp_utc": now},
                },
            }
        )
    )
    return path


if __name__ == "__main__":
    unittest.main()
