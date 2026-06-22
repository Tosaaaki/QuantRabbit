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
