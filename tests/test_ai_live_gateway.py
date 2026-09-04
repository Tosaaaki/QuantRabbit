from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from quant_rabbit.ai_live_gateway import (
    AILiveGatewayError,
    build_live_gateway_artifacts,
)


class AILiveGatewayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "data").mkdir()
        (self.root / "data" / "broker_snapshot.json").write_text(
            json.dumps({"fetched_at_utc": "2026-09-04T00:00:00+00:00"})
        )
        (self.root / "data" / "guardian_action_receipt.json").write_text(
            json.dumps({"action": "NO_ACTION", "selected_pair": "EUR_USD"})
        )
        (self.root / "data" / "order_intents.json").write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "lane_id": "bot:lane",
                            "intent": {
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "order_type": "LIMIT",
                                "units": 99999,
                                "entry": 9.99,
                                "tp": 10.0,
                                "sl": 9.0,
                                "market_context": {
                                    "method": "TREND_CONTINUATION"
                                },
                                "metadata": {
                                    "capture_exact_vehicle_net_scope": "PAIR_SIDE_METHOD_VEHICLE",
                                    "capture_exact_vehicle_net_scope_key": "EUR_USD|LONG|TREND_CONTINUATION|LIMIT|ALL_AUDITED_EXITS",
                                    "capture_exact_vehicle_net_vehicle": "LIMIT",
                                    "capture_exact_vehicle_net_metrics_source": "data/execution_ledger.db:exact_vehicle_net",
                                    "capture_exact_vehicle_net_exit_scope": "ALL_AUDITED_EXITS",
                                    "capture_exact_vehicle_net_trades": 20,
                                    "capture_exact_vehicle_net_wins": 12,
                                    "capture_exact_vehicle_net_losses": 8,
                                    "capture_exact_vehicle_net_jpy": 1000.0,
                                    "capture_exact_vehicle_net_expectancy_jpy": 50.0,
                                    "capture_exact_vehicle_net_avg_win_jpy": 150.0,
                                    "capture_exact_vehicle_net_avg_loss_jpy": 100.0,
                                    "capture_exact_vehicle_net_unresolved_realized_trades": 0,
                                    "capture_exact_vehicle_net_unresolved_realized_net_jpy": 0.0,
                                    "attach_take_profit_on_fill": True,
                                    "forecast_target_price": 10.0,
                                },
                            },
                        }
                    ]
                }
            )
        )
        self.order = {
            "decision_id": "ai-order-1",
            "pair": "EUR_USD",
            "side": "LONG",
            "method": "TREND_CONTINUATION",
            "vehicle": "LIMIT",
            "order_type": "LIMIT",
            "entry": 1.10,
            "take_profit": 1.12,
            "stop_loss": 1.09,
            "units": 1200,
            "allocation_multiplier": 0.75,
            "rationale": "AI-authored price geometry.",
            "extensions": {},
        }
        self.candidate = {
            "model": "gpt-5.6-luna",
            "reasoning_effort": "max",
            "source_digest": "a" * 64,
            "confidence": 0.75,
            "thesis": "AI thesis",
            "evidence_refs": ["market:data/pair_charts.json"],
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    @patch("quant_rabbit.ai_live_gateway.execution_cost_floor_from_surface")
    @patch("quant_rabbit.ai_live_gateway.read_exact_vehicle_allocation_surface")
    def test_artifact_uses_ai_numbers_and_only_audited_bot_metadata(
        self, read_surface, cost_floor
    ) -> None:
        read_surface.return_value = {
            "parse_status": "VALID",
            "exact_vehicle_net": [
                {
                    "pair": "EUR_USD",
                    "side": "LONG",
                    "method": "TREND_CONTINUATION",
                    "vehicle": "LIMIT",
                    "trades": 20,
                    "unresolved_realized_trades": 0,
                    "net_jpy": 1000.0,
                    "expectancy_jpy_per_trade": 50.0,
                }
            ],
            "exact_vehicle_take_profit": [],
        }
        cost_floor.return_value = {"proof_sha256": "b" * 64}
        built = build_live_gateway_artifacts(
            repo_root=self.root,
            order=self.order,
            candidate=self.candidate,
            run_id="intraday-test",
            generated_at=datetime(2026, 9, 4, tzinfo=timezone.utc),
        )
        intent = built["intents"]["results"][0]["intent"]
        self.assertEqual(intent["entry"], 1.10)
        self.assertEqual(intent["tp"], 1.12)
        self.assertEqual(intent["sl"], 1.09)
        self.assertEqual(intent["units"], 1600)
        self.assertNotIn("forecast_target_price", intent["metadata"])
        self.assertEqual(
            built["verified_decision"]["decision"]["capital_allocation"]["selected_units"],
            1200,
        )

    def test_missing_exact_execution_evidence_fails_closed(self) -> None:
        self.order["pair"] = "GBP_USD"
        with self.assertRaisesRegex(AILiveGatewayError, "no current audited"):
            build_live_gateway_artifacts(
                repo_root=self.root,
                order=self.order,
                candidate=self.candidate,
                run_id="intraday-test",
                generated_at=datetime(2026, 9, 4, tzinfo=timezone.utc),
            )


if __name__ == "__main__":
    unittest.main()
