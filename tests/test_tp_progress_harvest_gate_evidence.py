from __future__ import annotations

import json
import unittest
from pathlib import Path

from quant_rabbit.tp_progress_harvest_gate_evidence import build_tp_progress_harvest_gate_evidence


class TPProgressHarvestGateEvidenceTest(unittest.TestCase):
    def test_classifies_executable_noise_floor_and_manual_rows(self) -> None:
        payload = build_tp_progress_harvest_gate_evidence(
            {
                "generated_at_utc": "2026-06-26T00:00:00+00:00",
                "lookback_hours": 744,
                "summary": {
                    "tp_progress_repair_live_evidence_boundary_utc": "2026-06-22T17:54:26Z",
                    "loss_close_repair_replay_actual_pl_jpy": -600.0,
                    "loss_close_repair_replay_counterfactual_pl_jpy": -150.0,
                    "loss_close_repair_replay_delta_jpy": 450.0,
                },
                "loss_close_regrets": [
                    {
                        "trade_id": "t-trigger",
                        "pair": "USD_JPY",
                        "side": "SHORT",
                        "lane_id": "range_trader:USD_JPY:SHORT:RANGE_ROTATION",
                        "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
                        "fill_at_utc": "2026-06-21T00:00:00+00:00",
                        "close_at_utc": "2026-06-21T00:30:00+00:00",
                        "profit_capture_missed_before_loss_close": True,
                        "repair_replay_triggered_before_loss_close": True,
                        "repair_replay_trigger_at_utc": "2026-06-21T00:10:00+00:00",
                        "repair_replay_exit": "TP_PROGRESS_PRODUCTION_GATE_REPLAY",
                        "repair_replay_profit_pips": 2.4,
                        "repair_replay_noise_floor_pips": 1.6,
                        "repair_replay_tp_progress": 0.32,
                        "repair_replay_progress_gate": 0.3,
                        "realized_pl_jpy": -400.0,
                        "repair_replay_counterfactual_pl_jpy": 120.0,
                    },
                    {
                        "trade_id": "t-noise",
                        "pair": "AUD_NZD",
                        "side": "SHORT",
                        "lane_id": "range_trader:AUD_NZD:SHORT:RANGE_ROTATION",
                        "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
                        "fill_at_utc": "2026-06-21T00:00:00+00:00",
                        "close_at_utc": "2026-06-21T00:30:00+00:00",
                        "profit_capture_missed_before_loss_close": True,
                        "repair_replay_triggered_before_loss_close": False,
                        "repair_replay_block_reason": "BELOW_NOISE_FLOOR",
                        "realized_pl_jpy": -200.0,
                    },
                    {
                        "trade_id": "472987",
                        "pair": "EUR_USD",
                        "side": "SHORT",
                        "lane_id": None,
                        "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
                        "fill_at_utc": "2026-06-21T00:00:00+00:00",
                        "close_at_utc": "2026-06-21T00:30:00+00:00",
                        "profit_capture_missed_before_loss_close": False,
                        "repair_replay_triggered_before_loss_close": False,
                        "realized_pl_jpy": -100.0,
                    },
                ],
            },
            generated_at_utc="2026-06-26T01:00:00+00:00",
        )

        self.assertEqual(payload["metrics"]["historical_missed_capture_count"], 2)
        self.assertEqual(payload["metrics"]["current_rule_trigger_count"], 1)
        self.assertEqual(payload["metrics"]["executable_profit_capture_before_loss_close_count"], 1)
        self.assertEqual(payload["metrics"]["below_noise_floor_count"], 1)
        self.assertEqual(payload["attribution_counts"], {"SYSTEM_GATEWAY": 2})
        proposed = payload["month_scale_replay"]["after_proposed_gates"]
        self.assertEqual(proposed["excluded_trade_ids"], ["472987"])
        self.assertEqual(proposed["improved_pl_jpy"], -80.0)

    def test_current_execution_timing_table_has_14_misses_and_13_current_rule_triggers(self) -> None:
        timing_path = Path("data/execution_timing_audit.json")
        if not timing_path.exists():
            self.skipTest("current execution timing audit fixture is not present")
        payload = build_tp_progress_harvest_gate_evidence(json.loads(timing_path.read_text()))

        self.assertEqual(payload["metrics"]["historical_missed_capture_count"], 14)
        self.assertEqual(payload["metrics"]["current_rule_trigger_count"], 13)
        self.assertEqual(payload["metrics"]["executable_profit_capture_before_loss_close_count"], 13)
        self.assertEqual(payload["metrics"]["below_noise_floor_count"], 1)
        self.assertFalse(
            payload["month_scale_replay"]["month_scale_tp_progress_replay_still_negative_clears"]
        )


if __name__ == "__main__":
    unittest.main()
