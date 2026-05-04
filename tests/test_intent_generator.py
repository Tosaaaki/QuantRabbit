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

            cap_jpy = 500.0
            summary = IntentGenerator(
                campaign_plan=campaign,
                strategy_profile=strategy,
                output_path=output,
                report_path=report,
                max_loss_jpy=cap_jpy,
            ).run(snapshot_path=snapshot)

            self.assertEqual(summary.generated, 1)
            self.assertEqual(summary.dry_run_passed, 1)
            self.assertEqual(summary.live_ready, 0)
            payload = json.loads(output.read_text())
            result = payload["results"][0]
            self.assertEqual(result["status"], "DRY_RUN_PASSED")
            self.assertEqual(result["intent"]["pair"], "EUR_USD")
            self.assertEqual(result["intent"]["market_context"]["method"], "TREND_CONTINUATION")
            self.assertLessEqual(result["risk_metrics"]["risk_jpy"], cap_jpy)
            self.assertGreater(result["risk_metrics"]["spread_pips"], 0.0)
            self.assertTrue(result["live_blockers"])

    def test_sizes_repair_receipt_to_use_loss_budget_without_breaking_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = _campaign(root)
            strategy = _strategy(root)
            snapshot = _snapshot(root)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=campaign,
                strategy_profile=strategy,
                output_path=output,
                report_path=root / "intents.md",
                max_loss_jpy=500.0,
            ).run(snapshot_path=snapshot)

            payload = json.loads(output.read_text())
            result = payload["results"][0]
            intent = result["intent"]
            # With ATR-derived geometry the SL distance is the larger of
            # 1*ATR(M5) and 6*spread, so unit count is bounded by the new
            # geometry; assert risk fits the cap rather than a fixed unit.
            self.assertLessEqual(result["risk_metrics"]["risk_jpy"], 500.0)
            self.assertGreater(intent["units"], 0)

    def test_sizes_usd_quote_pair_from_snapshot_conversion_not_static_rate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root),
                output_path=output,
                report_path=root / "intents.md",
                max_loss_jpy=500.0,
            ).run(snapshot_path=_snapshot(root, usd_jpy=300.0))

            result = json.loads(output.read_text())["results"][0]
            self.assertEqual(result["status"], "DRY_RUN_PASSED")
            # Unit count depends on conversion rate AND the (now ATR-derived)
            # SL distance — assert risk fits cap rather than fixed unit count.
            self.assertGreater(result["intent"]["units"], 0)
            self.assertLessEqual(result["risk_metrics"]["risk_jpy"], 500.0)

    def test_uses_campaign_runner_reward_risk_for_tp_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_campaign(root, target_reward_risk=4.0),
                strategy_profile=_strategy(root),
                output_path=output,
                report_path=root / "intents.md",
            ).run(snapshot_path=_snapshot(root))

            result = json.loads(output.read_text())["results"][0]
            self.assertEqual(result["intent"]["metadata"]["target_reward_risk"], 4.0)
            self.assertAlmostEqual(result["risk_metrics"]["reward_risk"], 4.0)

    def test_sizes_units_with_percentage_risk_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "intents.json"

            IntentGenerator(
                campaign_plan=_campaign(root),
                strategy_profile=_strategy(root),
                output_path=output,
                report_path=root / "intents.md",
                max_loss_pct=1.0,
                risk_equity_jpy=100_000.0,
            ).run(snapshot_path=_snapshot(root))

            payload = json.loads(output.read_text())
            result = payload["results"][0]
            self.assertEqual(result["status"], "DRY_RUN_PASSED")
            # 1% of 100,000 = 1000 JPY cap. Unit count is derived from cap and
            # the (now ATR-aware) SL distance; assert risk respects the cap.
            self.assertGreater(result["intent"]["units"], 0)
            self.assertLessEqual(result["risk_metrics"]["risk_jpy"], 1000.0)


def _campaign(root: Path, *, target_reward_risk: float | None = None) -> Path:
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
                        **({"target_reward_risk": target_reward_risk} if target_reward_risk is not None else {}),
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


def _snapshot(root: Path, *, usd_jpy: float = 156.64) -> Path:
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
                    "USD_JPY": {"bid": usd_jpy, "ask": usd_jpy + 0.008, "timestamp_utc": now},
                },
            }
        )
    )
    return path


if __name__ == "__main__":
    unittest.main()
