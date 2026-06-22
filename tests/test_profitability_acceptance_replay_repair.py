from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quant_rabbit.execution_timing_contracts import (
    TP_PROGRESS_REPAIR_REPLAY_CONTRACT,
    TP_PROGRESS_REPAIR_REPLAY_FIELD,
)
from quant_rabbit.profitability_acceptance import (
    _execution_timing_loss_close_labels,
    _profit_capture_replay_repair_findings,
)


class ProfitabilityAcceptanceReplayRepairTest(unittest.TestCase):
    def test_blocks_when_candle_replay_shows_tp_progress_profit_capture_miss(self) -> None:
        metrics, findings = _profit_capture_replay_repair_findings(
            {
                "loaded": True,
                "generated_at_utc": "2026-06-22T17:10:24+00:00",
                "loss_closes_profit_capture_missed": 2,
                "loss_close_counterfactual_profit_capture_delta_jpy": 1054.8654,
                "loss_close_counterfactual_profit_capture_jpy": 475.1863,
                "top_profit_capture_misses": [
                    {
                        "trade_id": "472792",
                        "pair": "USD_JPY",
                        "side": "SHORT",
                        "counterfactual_jpy": 105.84,
                        "counterfactual_delta_jpy": 446.04,
                    }
                ],
            },
            self_metrics={"p0_codes": ["LOSS_CLOSE_PROFIT_CAPTURE_MISSED"]},
        )

        self.assertFalse(metrics["replay_repair_proved"])
        self.assertEqual(metrics["loss_closes_profit_capture_missed"], 2)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["priority"], "P0")
        self.assertEqual(findings[0]["code"], "TP_PROGRESS_REPAIR_REPLAY_CONTRACT_MISSING")
        self.assertEqual(
            findings[0]["evidence"]["top_profit_capture_misses"][0]["trade_id"],
            "472792",
        )

    def test_passes_when_active_timing_audit_has_no_tp_progress_misses(self) -> None:
        metrics, findings = _profit_capture_replay_repair_findings(
            {
                "loaded": True,
                "generated_at_utc": "2026-06-22T17:10:24+00:00",
                "repair_replay_contract_present": True,
                "loss_closes_profit_capture_missed": 0,
                "loss_close_counterfactual_profit_capture_delta_jpy": 0.0,
                "loss_close_counterfactual_profit_capture_jpy": 0.0,
            },
            self_metrics={"p0_codes": ["LOSS_CLOSE_PROFIT_CAPTURE_MISSED"]},
        )

        self.assertTrue(metrics["replay_repair_proved"])
        self.assertEqual(findings, [])

    def test_blocks_from_production_gate_replay_not_only_raw_mfe_threshold(self) -> None:
        metrics, findings = _profit_capture_replay_repair_findings(
            {
                "loaded": True,
                "generated_at_utc": "2026-06-22T17:10:24+00:00",
                "repair_replay_contract_present": True,
                "loss_closes_profit_capture_missed": 1,
                "loss_closes_repair_replay_triggered": 1,
                "loss_close_repair_replay_delta_jpy": 82.0,
                "loss_close_repair_replay_profit_capture_jpy": 32.0,
                "top_repair_replay_triggers": [
                    {
                        "trade_id": "t-noisy-cleared",
                        "repair_replay_exit": "TP_PROGRESS_PRODUCTION_GATE_REPLAY",
                        "repair_counterfactual_delta_jpy": 82.0,
                    }
                ],
            },
            self_metrics={"p0_codes": ["LOSS_CLOSE_PROFIT_CAPTURE_MISSED"]},
        )

        self.assertFalse(metrics["replay_repair_proved"])
        self.assertEqual(metrics["loss_closes_repair_replay_triggered"], 1)
        self.assertEqual(findings[0]["code"], "TP_PROGRESS_REPLAY_REPAIR_UNPROVED")
        self.assertEqual(
            findings[0]["evidence"]["top_repair_replay_triggers"][0]["trade_id"],
            "t-noisy-cleared",
        )

    def test_marks_replay_repair_not_deployed_when_guardian_is_inactive(self) -> None:
        metrics, findings = _profit_capture_replay_repair_findings(
            {
                "loaded": True,
                "generated_at_utc": "2026-06-22T17:10:24+00:00",
                "repair_replay_contract_present": True,
                "repair_replay_contract": "TP_PROGRESS_PRODUCTION_GATE_REPLAY_V1",
                "loss_closes_profit_capture_missed": 1,
                "loss_closes_repair_replay_triggered": 1,
                "loss_close_repair_replay_delta_jpy": 466.2,
                "top_repair_replay_triggers": [
                    {
                        "trade_id": "472792",
                        "repair_replay_exit": "TP_PROGRESS_PRODUCTION_GATE_REPLAY",
                    }
                ],
            },
            self_metrics={
                "p0_codes": [
                    "LOSS_CLOSE_PROFIT_CAPTURE_MISSED",
                    "POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE",
                ]
            },
        )

        codes = [item["code"] for item in findings]
        self.assertFalse(metrics["replay_repair_proved"])
        self.assertTrue(metrics["guardian_profit_capture_inactive"])
        self.assertIn("TP_PROGRESS_REPAIR_REPLAY_NOT_DEPLOYED", codes)
        self.assertIn("TP_PROGRESS_REPLAY_REPAIR_UNPROVED", codes)
        deploy_finding = next(
            item for item in findings if item["code"] == "TP_PROGRESS_REPAIR_REPLAY_NOT_DEPLOYED"
        )
        self.assertTrue(deploy_finding["evidence"]["guardian_profit_capture_inactive"])
        self.assertEqual(
            deploy_finding["evidence"]["top_repair_replay_triggers"][0]["trade_id"],
            "472792",
        )

    def test_raw_tp_progress_miss_without_production_gate_replay_is_diagnostic_only(self) -> None:
        metrics, findings = _profit_capture_replay_repair_findings(
            {
                "loaded": True,
                "generated_at_utc": "2026-06-22T17:10:24+00:00",
                "repair_replay_contract_present": True,
                "loss_closes_profit_capture_missed": 1,
                "loss_closes_repair_replay_triggered": 0,
                "loss_close_counterfactual_profit_capture_delta_jpy": 607.5,
                "top_profit_capture_misses": [
                    {
                        "trade_id": "raw-only-noise",
                        "counterfactual_delta_jpy": 607.5,
                    }
                ],
            },
            self_metrics={"p0_codes": ["LOSS_CLOSE_PROFIT_CAPTURE_MISSED"]},
        )

        self.assertTrue(metrics["replay_repair_proved"])
        self.assertEqual(metrics["loss_closes_profit_capture_missed"], 1)
        self.assertEqual(metrics["loss_closes_repair_replay_triggered"], 0)
        self.assertEqual(findings, [])

    def test_does_not_mix_unrelated_default_timing_into_acceptance_without_self_context(self) -> None:
        metrics, findings = _profit_capture_replay_repair_findings(
            {
                "loaded": True,
                "generated_at_utc": "2026-06-22T17:10:24+00:00",
                "loss_closes_profit_capture_missed": 2,
                "loss_close_counterfactual_profit_capture_delta_jpy": 1054.8654,
            },
            self_metrics={"p0_codes": []},
        )

        self.assertFalse(metrics["self_improvement_profit_capture_context"])
        self.assertEqual(findings, [])

    def test_requires_month_scale_replay_when_tp_positive_but_market_close_negative(self) -> None:
        metrics, findings = _profit_capture_replay_repair_findings(
            {
                "loaded": True,
                "generated_at_utc": "2026-06-22T17:10:24+00:00",
                "window_lookback_hours": 168.0,
                "repair_replay_contract_present": True,
                "loss_closes_profit_capture_missed": 0,
                "loss_closes_repair_replay_triggered": 0,
                "loss_close_repair_replay_counterfactual_pl_jpy": 42.0,
            },
            capture_metrics={
                "take_profit": {"net_jpy": 48804.0},
                "market_close": {"net_jpy": -81147.0},
            },
        )

        self.assertFalse(metrics["month_scale_replay_loaded"])
        self.assertFalse(metrics["replay_repair_proved"])
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["code"], "MONTH_SCALE_LOSS_CLOSE_REPLAY_REQUIRED")
        self.assertEqual(findings[0]["priority"], "P0")

    def test_blocks_when_month_scale_replay_improves_but_stays_negative(self) -> None:
        residual_groups = [
            {
                "pair": "GBP_USD",
                "side": "LONG",
                "method": "BREAKOUT_FAILURE",
                "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
                "loss_closes": 1,
                "repair_replay_pl_jpy": -2981.8961,
            }
        ]
        metrics, findings = _profit_capture_replay_repair_findings(
            {
                "loaded": True,
                "generated_at_utc": "2026-06-22T17:10:24+00:00",
                "window_lookback_hours": 744.0,
                "repair_replay_contract_present": True,
                "loss_closes_profit_capture_missed": 14,
                "loss_closes_repair_replay_triggered": 13,
                "loss_close_repair_replay_counterfactual_pl_jpy": -13824.5957,
                "loss_close_repair_replay_delta_jpy": 18775.1646,
                "loss_close_repair_replay_profit_capture_jpy": 3830.5491,
                "top_repair_replay_residual_groups": residual_groups,
            },
            self_metrics={"p0_codes": ["LOSS_CLOSE_PROFIT_CAPTURE_MISSED"]},
            capture_metrics={
                "take_profit": {"net_jpy": 48804.0},
                "market_close": {"net_jpy": -81147.0},
            },
        )

        codes = [item["code"] for item in findings]
        self.assertTrue(metrics["month_scale_replay_loaded"])
        self.assertFalse(metrics["replay_repair_proved"])
        self.assertIn("MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE", codes)
        self.assertIn("TP_PROGRESS_REPLAY_REPAIR_UNPROVED", codes)
        month_finding = next(
            item
            for item in findings
            if item["code"] == "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE"
        )
        self.assertEqual(
            month_finding["evidence"]["top_repair_replay_residual_groups"],
            residual_groups,
        )

    def test_month_scale_replay_non_negative_can_clear_month_specific_gate(self) -> None:
        metrics, findings = _profit_capture_replay_repair_findings(
            {
                "loaded": True,
                "generated_at_utc": "2026-06-22T17:10:24+00:00",
                "window_lookback_hours": 744.0,
                "repair_replay_contract_present": True,
                "loss_closes_profit_capture_missed": 2,
                "loss_closes_repair_replay_triggered": 0,
                "loss_close_repair_replay_counterfactual_pl_jpy": 15.5,
            },
            capture_metrics={
                "take_profit": {"net_jpy": 48804.0},
                "market_close": {"net_jpy": -81147.0},
            },
        )

        self.assertTrue(metrics["month_scale_replay_loaded"])
        self.assertTrue(metrics["replay_repair_proved"])
        self.assertEqual(findings, [])

    def test_timing_audit_surfaces_residual_loss_groups_after_repair_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "execution_timing_audit.json"
            path.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-06-22T17:10:24+00:00",
                        "window": {"lookback_hours": 744.0},
                        "precision": {
                            TP_PROGRESS_REPAIR_REPLAY_FIELD: (
                                TP_PROGRESS_REPAIR_REPLAY_CONTRACT
                            )
                        },
                        "summary": {
                            "loss_close_actual_pl_jpy": -3481.8961,
                            "loss_close_repair_replay_counterfactual_pl_jpy": -2981.8961,
                        },
                        "loss_close_regrets": [
                            {
                                "trade_id": "472071",
                                "pair": "GBP_USD",
                                "side": "LONG",
                                "lane_id": "failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE",
                                "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
                                "realized_pl_jpy": -2981.8961,
                                "repair_replay_counterfactual_pl_jpy": -2981.8961,
                                "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
                            },
                            {
                                "trade_id": "would-clear",
                                "pair": "USD_JPY",
                                "side": "SHORT",
                                "lane_id": "range_trader:USD_JPY:SHORT:RANGE_ROTATION",
                                "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
                                "realized_pl_jpy": -500.0,
                                "repair_replay_counterfactual_pl_jpy": 42.0,
                                "repair_replay_triggered_before_loss_close": True,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            _labels, metrics = _execution_timing_loss_close_labels(path)

        groups = metrics["top_repair_replay_residual_groups"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["pair"], "GBP_USD")
        self.assertEqual(groups[0]["side"], "LONG")
        self.assertEqual(groups[0]["method"], "BREAKOUT_FAILURE")
        self.assertEqual(groups[0]["exit_reason"], "MARKET_ORDER_TRADE_CLOSE")
        self.assertEqual(groups[0]["loss_closes"], 1)
        self.assertEqual(groups[0]["repair_replay_pl_jpy"], -2981.8961)
        self.assertEqual(groups[0]["block_reasons"], {"BELOW_TP_PROGRESS_GATE": 1})


if __name__ == "__main__":
    unittest.main()
