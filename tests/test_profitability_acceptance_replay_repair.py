from __future__ import annotations

import unittest

from quant_rabbit.profitability_acceptance import _profit_capture_replay_repair_findings


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


if __name__ == "__main__":
    unittest.main()
