from __future__ import annotations

import json
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from quant_rabbit.cli import main
from quant_rabbit.gpt_trader import GPTTraderBrain, StaticTraderProvider


LANE_ID = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION"


class GPTTraderBrainTest(unittest.TestCase):
    def test_accepts_schema_valid_evidence_cited_live_ready_trade(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            brain = _brain(root, files, _trade_decision())

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            self.assertTrue(summary.allowed)
            self.assertEqual(summary.action, "TRADE")
            self.assertEqual(summary.selected_lane_id, LANE_ID)
            payload = json.loads((root / "gpt_decision.json").read_text())
            self.assertEqual(payload["verification_issues"], [])
            self.assertIn("GPT Trader Decision Report", (root / "gpt_decision.md").read_text())

    def test_rejects_trade_receipt_that_predates_broker_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            snapshot = json.loads(files["snapshot"].read_text())
            decision = _trade_decision()
            decision["generated_at_utc"] = (
                datetime.fromisoformat(snapshot["fetched_at_utc"]) - timedelta(seconds=1)
            ).isoformat()
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("STALE_DECISION_RECEIPT", codes)

    def test_rejects_trade_receipt_without_generated_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision = _trade_decision()
            decision.pop("generated_at_utc")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("MISSING_DECISION_TIMESTAMP", codes)

    def test_rejects_trade_receipt_that_predates_order_intents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            snapshot = json.loads(files["snapshot"].read_text())
            snapshot_ts = datetime.fromisoformat(snapshot["fetched_at_utc"])
            decision_ts = snapshot_ts + timedelta(seconds=1)
            intents_ts = snapshot_ts + timedelta(seconds=2)
            intents = json.loads(files["intents"].read_text())
            intents["generated_at_utc"] = intents_ts.isoformat()
            files["intents"].write_text(json.dumps(intents))
            decision = _trade_decision()
            decision["generated_at_utc"] = decision_ts.isoformat()
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            stale_messages = [
                issue["message"]
                for issue in payload["verification_issues"]
                if issue["code"] == "STALE_DECISION_RECEIPT"
            ]
            self.assertTrue(any("order intents" in message for message in stale_messages))

    def test_rejects_trade_receipt_that_predates_market_context_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            snapshot = json.loads(files["snapshot"].read_text())
            snapshot_ts = datetime.fromisoformat(snapshot["fetched_at_utc"])
            decision_ts = snapshot_ts + timedelta(seconds=1)
            matrix_ts = snapshot_ts + timedelta(seconds=2)
            matrix = json.loads(files["market_context_matrix"].read_text())
            matrix["generated_at_utc"] = matrix_ts.isoformat()
            files["market_context_matrix"].write_text(json.dumps(matrix))
            decision = _trade_decision()
            decision["generated_at_utc"] = decision_ts.isoformat()
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            stale_messages = [
                issue["message"]
                for issue in payload["verification_issues"]
                if issue["code"] == "STALE_DECISION_RECEIPT"
            ]
            self.assertTrue(any("market_context_matrix" in message for message in stale_messages))

    def test_rejects_trade_without_twenty_minute_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision = _trade_decision()
            decision.pop("twenty_minute_plan")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("SHALLOW_DECISION_HORIZON", codes)

    def test_input_packet_includes_predictive_limit_timing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["predictive_limits"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-05-18T13:10:20+00:00",
                        "dry_run": True,
                        "orders": [
                            {
                                "pair": "EUR_USD",
                                "side": "SHORT",
                                "grade": "B",
                                "limit_price": 1.16495,
                                "take_profit_price": 1.16363,
                                "units": 2500,
                                "source": "liquidity_sweep_fade",
                                "gtd_utc": "2026-05-18T14:40:20Z",
                                "rationale": "liquidity sweep fade",
                            }
                        ],
                    }
                )
            )
            brain = _brain(root, files, _trade_decision())

            brain.run(snapshot_path=files["snapshot"])

            payload = json.loads((root / "gpt_decision.json").read_text())
            self.assertEqual(payload["input_packet"]["predictive_limits"]["orders_count"], 1)
            self.assertEqual(
                payload["input_packet"]["predictive_limits"]["orders"][0]["evidence_ref"],
                "predictive:limit:EUR_USD:SHORT",
            )
            self.assertIn("predictive:limits", payload["input_packet"]["allowed_evidence_refs"])

    def test_input_packet_includes_market_status_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            brain = _brain(root, files, _trade_decision())

            brain.run(snapshot_path=files["snapshot"])

            payload = json.loads((root / "gpt_decision.json").read_text())
            self.assertEqual(payload["input_packet"]["market_status"]["evidence_ref"], "market:status")
            self.assertTrue(payload["input_packet"]["market_status"]["is_fx_open"])
            self.assertIn("market:status", payload["input_packet"]["allowed_evidence_refs"])

    def test_input_packet_includes_strategy_seat_pnl_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            strategy = json.loads(files["strategy"].read_text())
            strategy["profiles"][0]["seat_pl_n"] = 12
            strategy["profiles"][0]["seat_net_jpy"] = -3000.0
            strategy["profiles"][0]["seat_win_rate_pct"] = 16.7
            files["strategy"].write_text(json.dumps(strategy))
            brain = _brain(root, files, _trade_decision())

            brain.run(snapshot_path=files["snapshot"])

            payload = json.loads((root / "gpt_decision.json").read_text())
            lane_strategy = payload["input_packet"]["lanes"][0]["strategy"]
            self.assertEqual(lane_strategy["seat_pl_n"], 12)
            self.assertEqual(lane_strategy["seat_net_jpy"], -3000.0)
            self.assertEqual(lane_strategy["seat_win_rate_pct"], 16.7)

    def test_input_packet_includes_verification_ledger_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["verification_ledger"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-06-04T00:00:00+00:00",
                        "status": "OK",
                        "db_path": str(root / "execution_ledger.db"),
                        "report_path": str(root / "verification_ledger.md"),
                        "blocking_observations": 0,
                        "missing_observations": 0,
                        "effect_metrics": {
                            "window_hours": 168.0,
                            "closed_trades": 42,
                            "net_jpy": 1200.0,
                            "profit_factor": 1.6,
                            "win_rate": 0.57,
                            "expectancy_jpy": 28.57,
                        },
                        "blocking_evidence": [],
                        "missing_artifacts": [],
                        "learning_evidence": [
                            {
                                "evidence_ref": "verification:learning_audit:learning_audit_status:LEARNING_AUDIT_WARN",
                                "source": "learning_audit",
                                "subject_type": "learning_audit",
                                "subject_id": "LEARNING_AUDIT_WARN",
                                "check_name": "learning_audit_status",
                                "status": "WARN",
                                "severity": "WARN",
                            }
                        ],
                        "measurements": [
                            {
                                "evidence_ref": "verification:effect:all:net_jpy",
                                "segment": "all",
                                "metric_name": "net_jpy",
                                "metric_value": 1200.0,
                                "metric_unit": "JPY",
                                "sample_size": 42,
                            }
                        ],
                        "contract": {
                            "read_only": True,
                            "live_permission": False,
                            "json_packet_is_trader_readable": True,
                            "markdown_report_is_operator_readable": True,
                            "learning_cannot_override_risk_or_gateway_gates": True,
                        },
                    }
                )
            )
            decision = _trade_decision()
            decision["evidence_refs"].extend(
                [
                    "verification:ledger",
                    "verification:learning_audit:learning_audit_status:LEARNING_AUDIT_WARN",
                    "verification:effect:all:net_jpy",
                ]
            )
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            packet = payload["input_packet"]["verification_ledger"]
            self.assertEqual(packet["status"], "OK")
            self.assertEqual(packet["effect_metrics"]["closed_trades"], 42)
            self.assertIn("verification:ledger", payload["input_packet"]["allowed_evidence_refs"])
            self.assertIn(
                "verification:learning_audit:learning_audit_status:LEARNING_AUDIT_WARN",
                payload["input_packet"]["allowed_evidence_refs"],
            )
            self.assertIn("verification:effect:all:net_jpy", payload["input_packet"]["allowed_evidence_refs"])

    def test_input_packet_accepts_chart_refs_for_open_position_pairs_without_lanes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        **_position(),
                        "trade_id": "202",
                        "pair": "GBP_JPY",
                        "entry_price": 215.104,
                        "take_profit": 215.276,
                        "stop_loss": 214.8,
                    }
                ],
            )
            pair_charts = json.loads(files["pair_charts"].read_text())
            pair_charts["charts"].append(
                {
                    "pair": "GBP_JPY",
                    "dominant_regime": "RANGE",
                    "chart_story": "GBP_JPY position-management chart story",
                    "long_score": 0.4,
                    "short_score": 0.6,
                    "views": _chart_views(),
                }
            )
            files["pair_charts"].write_text(json.dumps(pair_charts))
            decision = _trade_decision()
            decision["evidence_refs"].append("chart:GBP_JPY:M5")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            self.assertIn("chart:GBP_JPY:M5", payload["input_packet"]["allowed_evidence_refs"])
            self.assertIn("GBP_JPY", payload["input_packet"]["market_context"]["pairs"])
            self.assertEqual(
                payload["input_packet"]["market_context"]["pairs"]["GBP_JPY"]["chart"]["chart_story"],
                "GBP_JPY position-management chart story",
            )

    def test_rejects_trade_when_self_improvement_profitability_p0_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["self_improvement_audit"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "SELF_IMPROVEMENT_BLOCKED",
                        "p0_findings": 1,
                        "p1_findings": 2,
                        "p2_findings": 0,
                        "effect_metrics": {
                            "closed_trades": 28,
                            "net_jpy": -6571.91,
                            "profit_factor": 0.508,
                            "expectancy_jpy": -234.71,
                            "avg_win_jpy": 356.47,
                            "avg_loss_jpy_abs": 1482.76,
                        },
                        "findings": [
                            {
                                "priority": "P0",
                                "layer": "profitability",
                                "code": "PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED",
                                "message": "profitability discipline has failed for 19 consecutive audit run(s)",
                                "next_action": "Block new-risk confidence until execution_ledger.db worst segments prove repaired close discipline.",
                                "evidence": {
                                    "current_streak": 19,
                                    "system_defect_evidence": {
                                        "profit_factor": 0.508,
                                        "expectancy_jpy": -234.71,
                                        "avg_win_jpy": 356.47,
                                        "avg_loss_jpy_abs": 1482.76,
                                        "worst_segments": [
                                            {
                                                "pair": "EUR_USD",
                                                "side": "SHORT",
                                                "trades": 6,
                                                "net_jpy": -2977.0,
                                                "expectancy_jpy": -496.17,
                                            }
                                        ],
                                    },
                                },
                            }
                        ],
                    }
                )
            )
            decision = _trade_decision()
            decision["evidence_refs"].extend(
                [
                    "self_improvement:audit",
                    "self_improvement:profitability",
                    "self_improvement:finding:PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED",
                ]
            )
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("SELF_IMPROVEMENT_P0_BLOCKS_TRADE", codes)
            packet = payload["input_packet"]["self_improvement_audit"]
            self.assertEqual(packet["profitability_blockers"][0]["current_streak"], 19)
            self.assertIn(
                "self_improvement:finding:PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED",
                payload["input_packet"]["allowed_evidence_refs"],
            )

    def test_allows_trade_when_only_self_improvement_p0_is_stale_prior_gpt_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["self_improvement_audit"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "SELF_IMPROVEMENT_BLOCKED",
                        "findings": [
                            {
                                "priority": "P0",
                                "layer": "decision_history",
                                "code": "LATEST_GPT_DECISION_STALE",
                                "message": "latest GPT decision receipt predates the current broker snapshot",
                            }
                        ],
                    }
                )
            )
            decision = _trade_decision()
            decision["evidence_refs"].extend(
                [
                    "self_improvement:audit",
                    "self_improvement:decision_history",
                    "self_improvement:finding:LATEST_GPT_DECISION_STALE",
                ]
            )
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertNotIn("SELF_IMPROVEMENT_P0_BLOCKS_TRADE", codes)

    def test_report_contract_does_not_treat_receipt_close_flag_as_gate_b(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            brain = _brain(root, files, _trade_decision())

            brain.run(snapshot_path=files["snapshot"])

            report = (root / "gpt_decision.md").read_text()
            self.assertIn("fresh `data/.operator_close_token`", report)
            self.assertIn("`operator_close_authorized` field is advisory only", report)
            self.assertNotIn("`operator_close_authorized=true` or", report)

    def test_close_gate_b_docs_match_env_or_token_authorization(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        prompt = (repo / "docs" / "trader_prompts" / "35_position_management.md").read_text()
        source = (repo / "src" / "quant_rabbit" / "gpt_trader.py").read_text()

        for text in (prompt, source):
            self.assertIn("QR_OPERATOR_CLOSE_OVERRIDE=1", text)
            self.assertIn("data/.operator_close_token", text)
            self.assertIn("advisory", text)
            self.assertNotIn("operator-authorize-close", text)
            self.assertNotIn("operator_close_authorized=true` or", text)

    def test_position_prompt_does_not_allow_margin_pressure_close(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        prompt = (repo / "docs" / "trader_prompts" / "35_position_management.md").read_text()
        contract = (repo / "docs" / "AGENT_CONTRACT.md").read_text()

        for text in (prompt, contract):
            self.assertIn("Margin pressure is not a CLOSE trigger", text)
        self.assertIn("blocks new entries", prompt)
        self.assertIn("cancel", prompt)
        self.assertIn("CLOSE still needs", prompt)
        self.assertNotIn("Structural margin pressure", prompt)
        self.assertNotIn("All five triggers", prompt)

    def test_accepts_batch_trade_when_selected_lanes_are_live_ready_and_cited(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            market_lane = f"{LANE_ID}:MARKET"
            files["intents"].write_text(
                json.dumps({"results": [_result(), _result(lane_id=market_lane)]})
            )
            brain = _brain(root, files, _batch_trade_decision([LANE_ID, market_lane]))

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            self.assertEqual(summary.selected_lane_ids, (LANE_ID, market_lane))
            payload = json.loads((root / "gpt_decision.json").read_text())
            self.assertEqual(payload["verification_issues"], [])

    def test_rejects_trade_when_selected_lane_contradicts_forecast_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            result = _result()
            result["intent"]["metadata"] = {
                "forecast_direction": "DOWN",
                "forecast_confidence": 0.91,
                "forecast_target_price": 1.1712,
                "forecast_invalidation_price": 1.1742,
            }
            files["intents"].write_text(json.dumps({"results": [result]}))
            brain = _brain(root, files, _trade_decision())

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("FORECAST_DIRECTION_CONFLICT", codes)
            self.assertEqual(
                payload["input_packet"]["lanes"][0]["forecast"]["forecast_direction"],
                "DOWN",
            )

    def test_rejects_batch_trade_when_selected_lane_is_not_cited(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            market_lane = f"{LANE_ID}:MARKET"
            files["intents"].write_text(
                json.dumps({"results": [_result(), _result(lane_id=market_lane)]})
            )
            decision = _batch_trade_decision([LANE_ID, market_lane])
            decision["evidence_refs"].remove(f"intent:{market_lane}")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("SELECTED_LANE_EVIDENCE_MISSING", codes)

    def test_accepts_trade_when_existing_position_is_protected_and_trader_owned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, positions=[_position()])
            brain = _brain(root, files, _trade_decision())

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            self.assertTrue(summary.allowed)

    def test_accepts_trade_when_existing_position_is_sl_free_tp_less_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        **_position(stop_loss=None),
                        "take_profit": None,
                        "trade_id": "471232",
                    }
                ],
            )
            brain = _brain(root, files, _trade_decision())

            prior_sl = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            prior_tp = os.environ.get("QR_ENABLE_MISSING_TP_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            os.environ.pop("QR_ENABLE_MISSING_TP_REPAIR", None)
            try:
                summary = brain.run(snapshot_path=files["snapshot"])
            finally:
                if prior_sl is None:
                    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
                else:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl
                if prior_tp is None:
                    os.environ.pop("QR_ENABLE_MISSING_TP_REPAIR", None)
                else:
                    os.environ["QR_ENABLE_MISSING_TP_REPAIR"] = prior_tp

            self.assertEqual(summary.status, "ACCEPTED")
            self.assertTrue(summary.allowed)

    def test_accepts_trade_with_operator_manual_position_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "manual-470201",
                        "pair": "USD_JPY",
                        "side": "LONG",
                        "units": 25000,
                        "entry_price": 155.962,
                        "take_profit": None,
                        "stop_loss": None,
                        "owner": "unknown",
                    }
                ],
            )
            brain = _brain(root, files, _trade_decision())

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            self.assertTrue(summary.allowed)

    def test_accepts_trade_with_operator_manual_pending_order_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, orders=[{**_pending_order(), "order_id": "manual-pending", "owner": "unknown"}])
            brain = _brain(root, files, _trade_decision())

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            self.assertTrue(summary.allowed)

    def test_accepts_trade_with_trader_pending_entry_for_gateway_basket_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, orders=[_pending_order()])
            brain = _brain(root, files, _trade_decision())

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            self.assertTrue(summary.allowed)

    def test_accepts_trade_with_current_pending_cancel_order_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, orders=[_pending_order()])
            decision = _trade_decision()
            decision["cancel_order_ids"] = ["pending-1"]
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            self.assertEqual(summary.cancel_order_ids, ("pending-1",))
            payload = json.loads((root / "gpt_decision.json").read_text())
            self.assertEqual(payload["verification_issues"], [])

    def test_accepts_cancel_order_id_for_trader_pending_entry_beyond_order_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attached_orders = [
                {
                    "order_id": f"protective-{idx}",
                    "pair": None,
                    "order_type": "STOP_LOSS" if idx % 2 else "TAKE_PROFIT",
                    "trade_id": f"trade-{idx}",
                    "price": 1.17 + idx * 0.0001,
                    "state": "PENDING",
                    "units": None,
                    "owner": "unknown",
                }
                for idx in range(5)
            ]
            files = _fixtures(root, orders=[*attached_orders, _pending_order()])
            decision = _trade_decision()
            decision["cancel_order_ids"] = ["pending-1"]
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            self.assertEqual(summary.cancel_order_ids, ("pending-1",))
            payload = json.loads((root / "gpt_decision.json").read_text())
            packet_order_ids = [
                item.get("order_id")
                for item in payload["input_packet"]["broker_snapshot"]["pending_orders"]
            ]
            self.assertIn("pending-1", packet_order_ids)
            self.assertEqual(payload["verification_issues"], [])

    def test_rejects_trade_with_unknown_pending_cancel_order_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, orders=[_pending_order()])
            decision = _trade_decision()
            decision["cancel_order_ids"] = ["missing-order"]
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("UNKNOWN_CANCEL_ORDER_ID", codes)

    def test_rejects_trade_when_broker_exposure_is_not_layerable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, positions=[_position(stop_loss=None)])
            brain = _brain(root, files, _trade_decision())

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("EXPOSURE_BLOCKS_TRADE", codes)

    def test_rejects_hallucinated_evidence_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision = _trade_decision()
            decision["evidence_refs"] = ["broker:snapshot", "legacy:invented-row"]
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("UNKNOWN_EVIDENCE_REF", codes)

    def test_rejects_disabled_option_skew_evidence_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision = _trade_decision()
            decision["evidence_refs"].append("option:skew:unknown")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("UNKNOWN_EVIDENCE_REF", codes)

    def test_accepts_extended_pair_chart_timeframe_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision = _trade_decision()
            decision["evidence_refs"].extend(["chart:EUR_USD:M1", "chart:EUR_USD:M30", "chart:EUR_USD:H4", "chart:EUR_USD:D"])
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            self.assertEqual(payload["verification_issues"], [])

    def test_accepts_read_only_specialist_review_with_packet_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision = _trade_decision()
            decision["specialist_reviews"] = [
                {
                    "role": "macro_news",
                    "lane_id": LANE_ID,
                    "method": "TREND_CONTINUATION",
                    "verdict": "SUPPORTS",
                    "summary": "Macro review supports the lane but does not grant execution authority.",
                    "cited_evidence_refs": ["broker:snapshot", "cross:dxy"],
                    "hard_gate_codes": [],
                    "read_only": True,
                    "live_permission": False,
                }
            ]
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            self.assertEqual(payload["verification_issues"], [])

    def test_rejects_specialist_review_that_grants_live_permission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision = _trade_decision()
            decision["specialist_reviews"] = [
                {
                    "role": "macro_news",
                    "verdict": "SUPPORTS",
                    "summary": "A specialist must never authorize live execution.",
                    "cited_evidence_refs": ["broker:snapshot"],
                    "read_only": False,
                    "live_permission": True,
                }
            ]
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("SPECIALIST_REVIEW_NOT_READ_ONLY", codes)
            self.assertIn("SPECIALIST_REVIEW_LIVE_PERMISSION", codes)

    def test_rejects_stale_request_evidence_when_live_ready_lane_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            brain = _brain(root, files, _request_evidence_decision())

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("REQUEST_EVIDENCE_WITH_LIVE_READY_LANES", codes)

    def test_rejects_wait_when_flat_target_open_and_live_ready_lane_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            brain = _brain(root, files, _wait_decision())

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            self.assertFalse(summary.allowed)
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("CAMPAIGN_EXPOSURE_REQUIRED", codes)

    def test_accepts_wait_when_trader_exposure_is_already_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, positions=[{**_position(), "take_profit": 1.185}])
            brain = _brain(root, files, _wait_decision())

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            self.assertTrue(summary.allowed)
            payload = json.loads((root / "gpt_decision.json").read_text())
            self.assertEqual(payload["verification_issues"], [])

    def test_rejects_wait_when_tp_rebalance_sidecar_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "471292",
                        "pair": "EUR_USD",
                        "side": "SHORT",
                        "units": -22000,
                        "entry_price": 1.16077,
                        "take_profit": 1.15640,
                        "stop_loss": None,
                        "owner": "trader",
                        "unrealized_pl_jpy": 3000.0,
                    }
                ],
            )
            snapshot = json.loads(files["snapshot"].read_text())
            snapshot["quotes"]["EUR_USD"] = {
                "bid": 1.15937,
                "ask": 1.15947,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            files["snapshot"].write_text(json.dumps(snapshot))
            files["pair_charts"].write_text(json.dumps(_tp_rebalance_pair_charts()))
            (root / "forecast_history.jsonl").write_text(
                json.dumps(
                    {
                        "pair": "EUR_USD",
                        "direction": "UNCLEAR",
                        "confidence": 0.24,
                        "horizon_min": 0,
                    }
                )
                + "\n"
            )
            brain = _brain(root, files, _wait_decision())

            prior_sl = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                summary = brain.run(snapshot_path=files["snapshot"])
            finally:
                if prior_sl is None:
                    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
                else:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("TP_REBALANCE_REQUIRED", codes)
            self.assertTrue(payload["input_packet"]["protection_sidecars"]["tp_rebalance"]["required"])

    def test_rejects_trade_when_entry_thesis_is_unverifiable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, positions=[_position()])
            _write_entry_thesis_blocker(root, files, trade_id="101")
            brain = _brain(root, files, _trade_decision())

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("ENTRY_THESIS_REPAIR_REQUIRED", codes)
            blockers = payload["input_packet"]["protection_sidecars"]["entry_thesis_blockers"]
            self.assertEqual(blockers[0]["trade_id"], "101")
            self.assertEqual(blockers[0]["verdict"], "REQUIRE_THESIS_REPAIR")

    def test_rejects_wait_when_entry_thesis_is_unverifiable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, positions=[_position()])
            _write_entry_thesis_blocker(root, files, trade_id="101")
            brain = _brain(root, files, _wait_decision())

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("ENTRY_THESIS_REPAIR_REQUIRED", codes)

    def test_rejects_cancel_pending_without_order_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, orders=[_pending_order()])
            brain = _brain(root, files, _cancel_pending_decision())

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("MISSING_CANCEL_ORDER_IDS", codes)

    def test_rejects_cancel_pending_when_current_live_ready_lane_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, orders=[_pending_order()])
            brain = _brain(root, files, _cancel_pending_decision(cancel_order_ids=["pending-1"]))

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("CANCEL_PENDING_WITH_LIVE_READY_LANES", codes)
            self.assertEqual(payload["decision"]["cancel_order_ids"], ["pending-1"])

    def test_accepts_cancel_pending_when_no_current_live_ready_lane_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, orders=[_pending_order()])
            blocked_result = _result()
            blocked_result["status"] = "DRY_RUN_BLOCKED"
            blocked_result["live_blockers"] = ["forecast no longer backs this entry"]
            files["intents"].write_text(json.dumps({"results": [blocked_result]}))
            brain = _brain(root, files, _cancel_pending_decision(cancel_order_ids=["pending-1"]))

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            self.assertEqual(payload["verification_issues"], [])

    def test_cli_uses_external_decision_response_without_model_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision_response = root / "codex_decision_response.json"
            decision_response.write_text(json.dumps(_trade_decision()))

            with redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "gpt-trader-decision",
                        "--snapshot",
                        str(files["snapshot"]),
                        "--intents",
                        str(files["intents"]),
                        "--campaign-plan",
                        str(files["campaign"]),
                        "--strategy-profile",
                        str(files["strategy"]),
                        "--market-story-profile",
                        str(files["story"]),
                        "--target-state",
                        str(files["target"]),
                        "--attack-advice",
                        str(files["attack_advice"]),
                        "--self-improvement-audit",
                        str(files["self_improvement_audit"]),
                        "--decision-response",
                        str(decision_response),
                        "--output",
                        str(root / "cli_decision.json"),
                        "--report",
                        str(root / "cli_decision.md"),
                    ]
                )

            self.assertEqual(exit_code, 0)
            payload = json.loads((root / "cli_decision.json").read_text())
            self.assertEqual(payload["status"], "ACCEPTED")
            self.assertEqual(payload["decision"]["selected_lane_id"], LANE_ID)

    def test_default_packet_includes_range_lane_beyond_first_eight_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            range_lane = "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            filler = [
                _result(lane_id=f"candidate_{idx}:EUR_USD:LONG:TREND_CONTINUATION")
                for idx in range(9)
            ]
            files["intents"].write_text(
                json.dumps({"results": [*filler, _result(lane_id=range_lane, method="RANGE_ROTATION")]})
            )
            brain = _brain(root, files, _trade_decision(lane_id=range_lane, method="RANGE_ROTATION"))

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            self.assertEqual(summary.selected_lane_id, range_lane)

    def test_live_ready_lane_beyond_packet_cap_is_still_verifiable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            filler = [
                _result(lane_id=f"candidate_{idx}:EUR_USD:LONG:TREND_CONTINUATION")
                for idx in range(13)
            ]
            files["intents"].write_text(json.dumps({"results": [*filler, _result()]}))
            brain = _brain(root, files, _trade_decision(), max_lanes=12)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            self.assertEqual(summary.selected_lane_id, LANE_ID)
            payload = json.loads((root / "gpt_decision.json").read_text())
            packet_lane_ids = {lane["lane_id"] for lane in payload["input_packet"]["lanes"]}
            self.assertIn(LANE_ID, packet_lane_ids)
            self.assertGreater(len(packet_lane_ids), 12)

    def test_packet_includes_market_context_payloads_not_only_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            brain = _brain(root, files, _trade_decision())

            brain.run(snapshot_path=files["snapshot"])

            payload = json.loads((root / "gpt_decision.json").read_text())
            market_context = payload["input_packet"]["market_context"]
            eur = market_context["pairs"]["EUR_USD"]

            self.assertEqual(eur["chart"]["dominant_regime"], "TREND_UP")
            self.assertEqual(eur["chart"]["views"]["M1"]["last_jump_bars_ago"], 12)
            self.assertEqual(eur["chart"]["views"]["M5"]["atr_pips"], 5.3)
            self.assertEqual(eur["chart"]["views"]["M5"]["regime_state"], "TREND_STRONG")
            self.assertEqual(eur["chart"]["views"]["H4"]["regime_state"], "TREND_WEAK")
            self.assertEqual(eur["chart"]["views"]["D"]["regime_state"], "RANGE")
            self.assertEqual(eur["flow"]["spread"]["stress_flag"], "NORMAL")
            self.assertEqual(eur["levels"]["pdh"], 1.18)
            self.assertEqual(eur["matrix"]["LONG"]["evidence_ref"], "matrix:EUR_USD:LONG")
            self.assertGreaterEqual(eur["matrix"]["LONG"]["support_count"], 1)
            self.assertIn("matrix:EUR_USD:LONG", payload["input_packet"]["allowed_evidence_refs"])
            self.assertFalse(eur["calendar"]["in_window"])
            self.assertEqual(market_context["currency_strength"]["USD"]["rank"], 2)
            self.assertEqual(market_context["cot"]["USD"]["leveraged_net"], 1234)
            xau = market_context["context_assets"]["assets"]["XAU_USD"]
            self.assertEqual(xau["chart"]["dominant_regime"], "TREND_DOWN")
            self.assertFalse(xau["broker_tradeable"])
            self.assertEqual(market_context["broker_tradeability"]["context_assets_not_tradeable"], ["XAU_USD"])
            self.assertIn("context_asset:XAU_USD", payload["input_packet"]["allowed_evidence_refs"])
            self.assertIn("broker:instruments", payload["input_packet"]["allowed_evidence_refs"])

    def test_packet_includes_coverage_gap_profitable_bucket_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["coverage_optimization"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-06-07T14:40:00+00:00",
                        "status": "COVERAGE_GAP",
                        "remaining_target_jpy": 22977.7,
                        "live_ready_reward_jpy": 0.0,
                        "potential_reward_jpy": 0.0,
                        "coverage_pct": 0.0,
                        "artifact_diagnostics": {
                            "spread_normalized_candidate_count": 8,
                            "spread_normalized_no_live_blocker_count": 2,
                            "profitable_bucket_coverage": {
                                "source_status": "RESEARCH_PROFITABLE_NOT_CERTIFIED",
                                "live_permission": False,
                                "positive_pair_directions": 8,
                                "positive_managed_net_jpy": 33312.35,
                                "positive_trade_count": 221,
                                "state_counts": {"SURFACED_BUT_BLOCKED": 5},
                                "top_edges": [
                                    {
                                        "pair": "EUR_USD",
                                        "direction": "LONG",
                                        "coverage_state": "SURFACED_BUT_BLOCKED",
                                        "managed_net_jpy": 17650.08,
                                        "raw_net_jpy": 16098.53,
                                        "trades": 64,
                                        "days": 13,
                                        "current_lane_count": 7,
                                        "current_best_reward_jpy": 4881.68,
                                        "top_blockers": [
                                            "EUR_USD LONG current pair forecast is UNCLEAR conf=0.03",
                                            "HARVEST_TP_STRUCTURE_MISSING",
                                        ],
                                        "strategy_profile_status": "BLOCK_UNTIL_NEW_EVIDENCE",
                                        "strategy_profile_required_fix": "live execution and pretrade feedback are negative",
                                        "strategy_profile_blocks_live": True,
                                        "strategy_profile_live_net_jpy": -1200.0,
                                        "strategy_profile_pretrade_net_jpy": -350.0,
                                        "strategy_profile_seat_net_jpy": -7000.0,
                                        "strategy_profile_seat_win_rate_pct": 21.0,
                                        "matrix_ref": "matrix:EUR_USD:LONG",
                                        "matrix_support_count": 0,
                                        "matrix_reject_count": 5,
                                        "matrix_warning_count": 8,
                                        "matrix_strongest_reject": "EUR_USD confluence score_balance=SHORT_LEAN",
                                        "matrix_cross_asset_context": [
                                            "GOLD_CONTEXT_TECHNICAL_DIRECTION: XAU_USD maps to SHORT",
                                            "DXY_24H_DIRECTION: synthetic DXY maps to SHORT",
                                        ],
                                    }
                                ],
                                "matrix_supported_repair_queue": [
                                    {
                                        "pair": "AUD_JPY",
                                        "direction": "SHORT",
                                        "coverage_state": "SURFACED_BUT_BLOCKED",
                                        "managed_net_jpy": 6192.92,
                                        "top_blockers": [
                                            "AUD_JPY SHORT is BLOCK_UNTIL_NEW_EVIDENCE",
                                            "forecast confidence below live floor",
                                        ],
                                        "strategy_profile_status": "BLOCK_UNTIL_NEW_EVIDENCE",
                                        "matrix_ref": "matrix:AUD_JPY:SHORT",
                                        "matrix_support_count": 11,
                                        "matrix_reject_count": 1,
                                        "matrix_support_layers": ["chart", "cross_asset", "context_asset_chart"],
                                        "matrix_support_context": [
                                            "RISK_ASSET_JPY_CROSS_DIRECTION: SPX down maps to SHORT",
                                            "US10Y_JPY_CROSS_DIRECTION: US10Y down maps to SHORT",
                                        ],
                                    }
                                ],
                            },
                        },
                        "action_items": ["repair historical-profitable bucket coverage before widening discovery"],
                    }
                )
            )
            decision = _trade_decision()
            decision["evidence_refs"].extend(
                ["coverage:optimization", "coverage:profitable_bucket:EUR_USD:LONG"]
            )
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            packet = payload["input_packet"]["coverage_optimization"]
            self.assertEqual(packet["status"], "COVERAGE_GAP")
            self.assertFalse(packet["live_permission"])
            bucket = packet["profitable_bucket_coverage"]
            self.assertEqual(bucket["source_status"], "RESEARCH_PROFITABLE_NOT_CERTIFIED")
            edge = bucket["top_edges"][0]
            self.assertEqual(edge["evidence_ref"], "coverage:profitable_bucket:EUR_USD:LONG")
            self.assertEqual(edge["strategy_profile_status"], "BLOCK_UNTIL_NEW_EVIDENCE")
            self.assertTrue(edge["strategy_profile_blocks_live"])
            self.assertEqual(edge["strategy_profile_live_net_jpy"], -1200.0)
            self.assertEqual(edge["matrix_reject_count"], 5)
            self.assertIn("GOLD_CONTEXT_TECHNICAL_DIRECTION: XAU_USD maps to SHORT", edge["matrix_cross_asset_context"])
            repair_queue = bucket["matrix_supported_repair_queue"]
            self.assertEqual(repair_queue[0]["evidence_ref"], "coverage:profitable_bucket:AUD_JPY:SHORT")
            self.assertEqual(repair_queue[0]["matrix_support_count"], 11)
            self.assertIn("RISK_ASSET_JPY_CROSS_DIRECTION", repair_queue[0]["matrix_support_context"][0])
            self.assertIn("coverage:optimization", payload["input_packet"]["allowed_evidence_refs"])
            self.assertIn("coverage:profitable_bucket:EUR_USD:LONG", payload["input_packet"]["allowed_evidence_refs"])
            self.assertIn("coverage:profitable_bucket:AUD_JPY:SHORT", payload["input_packet"]["allowed_evidence_refs"])

    def test_accepts_attack_advice_evidence_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["attack_advice"].write_text(
                json.dumps(
                    {
                        "status": "ATTACK_PARTIAL",
                        "read_only": True,
                        "live_permission": False,
                        "coverage_pct": 49.0,
                        "recommended_now_lane_ids": [LANE_ID],
                        "recommended_now_reward_jpy": 900.0,
                        "recommended_now_risk_jpy": 300.0,
                    }
                )
            )
            decision = _trade_decision()
            decision["evidence_refs"].extend(["attack:advice", f"attack:lane:{LANE_ID}"])
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            self.assertEqual(payload["input_packet"]["ai_attack_advice"]["recommended_now_lane_ids"], [LANE_ID])
            self.assertFalse(payload["input_packet"]["ai_attack_advice"]["live_permission"])

    def test_rejects_learning_influenced_trade_without_learning_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["attack_advice"].write_text(json.dumps(_learning_influenced_attack_advice()))
            files["learning_audit"].write_text(json.dumps({}))
            decision = _trade_decision()
            decision["evidence_refs"].extend(["attack:advice", f"attack:lane:{LANE_ID}"])
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("LEARNING_AUDIT_REQUIRED", codes)

    def test_rejects_learning_influenced_trade_when_audit_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["attack_advice"].write_text(json.dumps(_learning_influenced_attack_advice()))
            files["learning_audit"].write_text(
                json.dumps(_learning_audit_payload(status="LEARNING_AUDIT_BLOCKED", blockers=["recent learned lane effect is negative"]))
            )
            decision = _trade_decision()
            decision["evidence_refs"].extend(
                ["attack:advice", f"attack:lane:{LANE_ID}", "learning:audit", f"learning:lane:{LANE_ID}"]
            )
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("LEARNING_AUDIT_BLOCKED", codes)

    def test_rejects_learning_influenced_trade_without_learning_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["attack_advice"].write_text(json.dumps(_learning_influenced_attack_advice()))
            files["learning_audit"].write_text(json.dumps(_learning_audit_payload(status="LEARNING_AUDIT_WARN")))
            decision = _trade_decision()
            decision["evidence_refs"].extend(["attack:advice", f"attack:lane:{LANE_ID}"])
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("LEARNING_AUDIT_EVIDENCE_MISSING", codes)
            self.assertIn("LEARNING_LANE_EVIDENCE_MISSING", codes)

    def test_accepts_learning_influenced_trade_with_warn_audit_and_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["attack_advice"].write_text(json.dumps(_learning_influenced_attack_advice()))
            files["learning_audit"].write_text(json.dumps(_learning_audit_payload(status="LEARNING_AUDIT_WARN")))
            decision = _trade_decision()
            decision["evidence_refs"].extend(
                ["attack:advice", f"attack:lane:{LANE_ID}", "learning:audit", f"learning:lane:{LANE_ID}"]
            )
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            self.assertEqual(payload["verification_issues"], [])
            self.assertEqual(payload["input_packet"]["learning_audit"]["status"], "LEARNING_AUDIT_WARN")

    def test_input_packet_exposes_learning_exit_reason_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["learning_audit"].write_text(
                json.dumps(
                    _learning_audit_payload(
                        status="LEARNING_AUDIT_WARN",
                        exit_reason_metrics={
                            "MARKET_ORDER_TRADE_CLOSE": {
                                "closed_trades": 10,
                                "net_jpy": -13314.65,
                                "gross_profit_jpy": 25.0,
                                "gross_loss_jpy": -13339.65,
                                "profit_factor": 0.0019,
                                "win_rate": 0.1,
                                "expectancy_jpy": -1331.465,
                            },
                            "TAKE_PROFIT_ORDER": {
                                "closed_trades": 11,
                                "net_jpy": 4758.54,
                                "profit_factor": None,
                                "win_rate": 1.0,
                                "expectancy_jpy": 432.594,
                            },
                        },
                    )
                )
            )
            decision = _trade_decision()
            decision["evidence_refs"].append("learning:exit_reason:MARKET_ORDER_TRADE_CLOSE")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            packet = payload["input_packet"]["learning_audit"]
            market_close = packet["effect_metrics"]["exit_reason_metrics"]["MARKET_ORDER_TRADE_CLOSE"]
            self.assertEqual(market_close["evidence_ref"], "learning:exit_reason:MARKET_ORDER_TRADE_CLOSE")
            self.assertEqual(market_close["closed_trades"], 10)
            self.assertEqual(market_close["net_jpy"], -13314.65)
            self.assertIn(
                "learning:exit_reason:MARKET_ORDER_TRADE_CLOSE",
                payload["input_packet"]["allowed_evidence_refs"],
            )
            self.assertEqual(payload["verification_issues"], [])

    def test_rejects_wait_when_attack_advice_recommends_lane_even_with_trader_exposure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, positions=[_position()])
            files["attack_advice"].write_text(
                json.dumps(
                    {
                        "status": "ATTACK_PARTIAL",
                        "read_only": True,
                        "live_permission": False,
                        "recommended_now_lane_ids": [LANE_ID],
                    }
                )
            )
            decision = _wait_decision()
            decision["evidence_refs"].extend(["attack:advice", f"attack:lane:{LANE_ID}"])
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("ATTACK_ADVICE_REQUIRES_TRADE", codes)

    def test_wait_is_allowed_when_self_improvement_p0_blocks_attack_advice_trade(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["attack_advice"].write_text(
                json.dumps(
                    {
                        "status": "ATTACK_PARTIAL",
                        "read_only": True,
                        "live_permission": False,
                        "recommended_now_lane_ids": [LANE_ID],
                    }
                )
            )
            files["self_improvement_audit"].write_text(json.dumps(_self_improvement_profitability_p0()))
            decision = _wait_decision()
            decision["evidence_refs"].extend(
                [
                    "attack:advice",
                    f"attack:lane:{LANE_ID}",
                    "self_improvement:audit",
                    "self_improvement:profitability",
                    "self_improvement:finding:PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED",
                ]
            )
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertNotIn("ATTACK_ADVICE_REQUIRES_TRADE", codes)
            self.assertNotIn("CAMPAIGN_EXPOSURE_REQUIRED", codes)
            packet = payload["input_packet"]["self_improvement_audit"]
            self.assertEqual(packet["profitability_blockers"][0]["current_streak"], 19)

    def test_wait_is_allowed_when_self_improvement_projection_p0_blocks_attack_advice_trade(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["attack_advice"].write_text(
                json.dumps(
                    {
                        "status": "ATTACK_PARTIAL",
                        "read_only": True,
                        "live_permission": False,
                        "recommended_now_lane_ids": [LANE_ID],
                    }
                )
            )
            files["self_improvement_audit"].write_text(json.dumps(_self_improvement_projection_p0()))
            decision = _wait_decision()
            decision["evidence_refs"].extend(
                [
                    "attack:advice",
                    f"attack:lane:{LANE_ID}",
                    "self_improvement:audit",
                    "self_improvement:forecast",
                    "self_improvement:finding:PROJECTION_LEDGER_EXPIRED_PENDING",
                ]
            )
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertNotIn("ATTACK_ADVICE_REQUIRES_TRADE", codes)
            self.assertNotIn("CAMPAIGN_EXPOSURE_REQUIRED", codes)
            packet = payload["input_packet"]["self_improvement_audit"]
            self.assertEqual(packet["p0_blockers"][0]["code"], "PROJECTION_LEDGER_EXPIRED_PENDING")

    def test_wait_with_soft_close_sidecar_still_must_trade_attack_advice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="LONG", m15_dir="UP", h4_dir="UP")
            _write_fresh_forecast_close_recommendation(root, files, side="LONG")
            files["attack_advice"].write_text(
                json.dumps(
                    {
                        "status": "ATTACK_PARTIAL",
                        "read_only": True,
                        "live_permission": False,
                        "recommended_now_lane_ids": [LANE_ID],
                    }
                )
            )
            decision = _wait_decision()
            decision["evidence_refs"].extend(
                ["position:persistence:555", "attack:advice", f"attack:lane:{LANE_ID}"]
            )
            brain = _brain(root, files, decision)

            with patch.dict(os.environ, {"QR_TRADER_DISABLE_SL_REPAIR": "1"}, clear=False), patch(
                "quant_rabbit.gpt_trader._operator_close_gate_authorized", return_value=False
            ):
                summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("ATTACK_ADVICE_REQUIRES_TRADE", codes)
            self.assertNotIn("CLOSE_OPERATOR_AUTH_REQUIRED", codes)
            self.assertNotIn("WAIT_MISSING_LIVE_READY_REJECTION", codes)
            message = "\n".join(issue["message"] for issue in payload["verification_issues"])
            self.assertIn("choose TRADE", message)

    def test_trade_with_soft_close_sidecar_and_no_operator_token_is_allowed(self) -> None:
        """Soft close review is not a blanket blocker for separate entries.

        The same sidecar remains Gate A evidence if the trader chooses CLOSE,
        but a TP-managed existing position should not freeze all-horizon
        participation when current LIVE_READY lanes exist.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="LONG", m15_dir="UP", h4_dir="UP")
            _write_fresh_forecast_close_recommendation(root, files, side="LONG")
            decision = _trade_decision()
            decision["evidence_refs"].append("position:persistence:555")
            brain = _brain(root, files, decision)

            with patch.dict(os.environ, {"QR_TRADER_DISABLE_SL_REPAIR": "1"}, clear=False), patch(
                "quant_rabbit.gpt_trader._operator_close_gate_authorized", return_value=False
            ):
                summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertNotIn("CLOSE_OPERATOR_AUTH_REQUIRED", codes)
            self.assertEqual(codes, set())

    def test_wait_with_authorized_close_sidecar_must_emit_close_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="DOWN", h4_dir="DOWN")
            _write_fresh_forecast_close_recommendation(root, files)
            decision = _wait_decision()
            decision["evidence_refs"].append("position:persistence:555")
            brain = _brain(root, files, decision)

            with patch("quant_rabbit.gpt_trader._operator_close_gate_authorized", return_value=True):
                summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("POSITION_CLOSE_REQUIRED", codes)
            self.assertNotIn("CLOSE_OPERATOR_AUTH_REQUIRED", codes)

    def test_protect_with_soft_close_sidecar_and_no_operator_token_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="DOWN", h4_dir="DOWN")
            _write_fresh_forecast_close_recommendation(root, files)
            decision = _protect_decision()
            decision["evidence_refs"].append("position:persistence:555")
            brain = _brain(root, files, decision)

            with patch("quant_rabbit.gpt_trader._operator_close_gate_authorized", return_value=False):
                summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertNotIn("CLOSE_OPERATOR_AUTH_REQUIRED", codes)
            self.assertEqual(codes, set())

    def test_tighten_sl_with_authorized_close_sidecar_must_emit_close_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="DOWN", h4_dir="DOWN")
            _write_fresh_forecast_close_recommendation(root, files)
            decision = _tighten_sl_decision()
            decision["evidence_refs"].append("position:persistence:555")
            brain = _brain(root, files, decision)

            with patch("quant_rabbit.gpt_trader._operator_close_gate_authorized", return_value=True):
                summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("POSITION_CLOSE_REQUIRED", codes)
            self.assertNotIn("CLOSE_OPERATOR_AUTH_REQUIRED", codes)

    def test_protect_with_hard_close_sidecar_must_emit_close_first_without_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="DOWN", h4_dir="DOWN")
            _write_fresh_thesis_evolution_close_recommendation(root, files)
            decision = _protect_decision()
            decision["evidence_refs"].append("position:evolution:555")
            brain = _brain(root, files, decision)

            with patch("quant_rabbit.gpt_trader._operator_close_gate_authorized", return_value=False):
                summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("POSITION_CLOSE_REQUIRED", codes)
            self.assertNotIn("CLOSE_OPERATOR_AUTH_REQUIRED", codes)

    def test_rejects_trade_that_ignores_attack_advice_recommended_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            alternative_lane = "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE"
            files["intents"].write_text(
                json.dumps(
                    {
                        "results": [
                            _result(),
                            _result(lane_id=alternative_lane, method="BREAKOUT_FAILURE"),
                        ]
                    }
                )
            )
            files["attack_advice"].write_text(
                json.dumps(
                    {
                        "status": "ATTACK_PARTIAL",
                        "read_only": True,
                        "live_permission": False,
                        "recommended_now_lane_ids": [LANE_ID],
                    }
                )
            )
            decision = _trade_decision(lane_id=alternative_lane, method="BREAKOUT_FAILURE")
            decision["evidence_refs"].append("attack:advice")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("ATTACK_ADVICE_IGNORED", codes)

    def test_rejects_trade_that_skips_first_attack_priority_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            priority_lane = "trend_trader:EUR_USD:SHORT:TREND_CONTINUATION:MARKET"
            files["intents"].write_text(
                json.dumps(
                    {
                        "results": [
                            _result(lane_id=priority_lane),
                            _result(),
                        ]
                    }
                )
            )
            files["attack_advice"].write_text(
                json.dumps(
                    {
                        "status": "ATTACK_PARTIAL",
                        "read_only": True,
                        "live_permission": False,
                        "recommended_now_lane_ids": [priority_lane, LANE_ID],
                    }
                )
            )
            decision = _trade_decision()
            decision["evidence_refs"].extend(["attack:advice", f"attack:lane:{LANE_ID}"])
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("ATTACK_PRIORITY_SKIPPED", codes)

    def test_warns_single_pair_basket_when_advice_covers_multiple_pairs(self) -> None:
        """Regression: single-pair GPT JSON should not discard a valid trade.

        The autotrade cycle expands accepted GPT trades into a deterministic
        gateway basket, so this verifier issue is advisory instead of a send
        blocker.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            primary_lane = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET"
            second_pair_lane = "trend_trader:EUR_JPY:LONG:TREND_CONTINUATION:MARKET"
            third_pair_lane = "trend_trader:GBP_USD:LONG:TREND_CONTINUATION:MARKET"
            files["intents"].write_text(
                json.dumps(
                    {
                        "results": [
                            _result(lane_id=primary_lane),
                            _result(lane_id=second_pair_lane),
                            _result(lane_id=third_pair_lane),
                        ]
                    }
                )
            )
            files["attack_advice"].write_text(
                json.dumps(
                    {
                        "status": "ATTACK_PARTIAL",
                        "read_only": True,
                        "live_permission": False,
                        "recommended_now_lane_ids": [
                            primary_lane,
                            second_pair_lane,
                            third_pair_lane,
                        ],
                    }
                )
            )
            decision = _trade_decision(lane_id=primary_lane)
            decision["evidence_refs"].extend(
                ["attack:advice", f"attack:lane:{primary_lane}"]
            )
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            issues = {
                issue["code"]: issue
                for issue in payload["verification_issues"]
            }
            self.assertEqual(issues["BASKET_PAIR_COVERAGE_INCOMPLETE"]["severity"], "WARN")

    def test_accepts_basket_covering_every_advised_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            primary_lane = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET"
            second_pair_lane = "trend_trader:EUR_JPY:LONG:TREND_CONTINUATION:MARKET"
            third_pair_lane = "trend_trader:GBP_USD:LONG:TREND_CONTINUATION:MARKET"
            advised = [primary_lane, second_pair_lane, third_pair_lane]
            files["intents"].write_text(
                json.dumps({"results": [_result(lane_id=l) for l in advised]})
            )
            files["attack_advice"].write_text(
                json.dumps(
                    {
                        "status": "ATTACK_PARTIAL",
                        "read_only": True,
                        "live_permission": False,
                        "recommended_now_lane_ids": advised,
                    }
                )
            )
            decision = _batch_trade_decision(advised)
            decision["evidence_refs"].extend(
                ["attack:advice"] + [f"attack:lane:{l}" for l in advised]
            )
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            self.assertEqual(payload["verification_issues"], [])

    def test_accepts_single_pair_basket_when_other_pairs_are_below_rank_ceiling(self) -> None:
        """High-conviction concentrated attack should not be blocked by basket
        coverage when the other advised pairs only appear below
        PRIMARY_ATTACK_RANK_CEILING. The rank gap itself is the deterministic
        conviction gate per AGENT_CONTRACT §5–§6 — see
        feedback_high_conviction_execution.md.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            primary_lane = "trend_trader:EUR_USD:SHORT:TREND_CONTINUATION:MARKET"
            # Pad ranks 2..4 with EUR_USD lanes so the rank ceiling stays
            # within a single pair; AUD_JPY/GBP_USD only appear at rank 5+.
            eur_filler_2 = "range_trader:EUR_USD:SHORT:RANGE_ROTATION:MARKET"
            eur_filler_3 = "trend_trader:EUR_USD:SHORT:TREND_CONTINUATION"
            eur_filler_4 = "range_trader:EUR_USD:SHORT:RANGE_ROTATION"
            low_rank_aud = "trend_trader:AUD_JPY:LONG:TREND_CONTINUATION:MARKET"
            low_rank_gbp = "trend_trader:GBP_USD:LONG:TREND_CONTINUATION:MARKET"
            files["intents"].write_text(
                json.dumps(
                    {
                        "results": [
                            _result(lane_id=primary_lane),
                            _result(lane_id=eur_filler_2),
                            _result(lane_id=eur_filler_3),
                            _result(lane_id=eur_filler_4),
                            _result(lane_id=low_rank_aud),
                            _result(lane_id=low_rank_gbp),
                        ]
                    }
                )
            )
            files["attack_advice"].write_text(
                json.dumps(
                    {
                        "status": "ATTACK_PARTIAL",
                        "read_only": True,
                        "live_permission": False,
                        "recommended_now_lane_ids": [
                            primary_lane,
                            eur_filler_2,
                            eur_filler_3,
                            eur_filler_4,
                            low_rank_aud,
                            low_rank_gbp,
                        ],
                    }
                )
            )
            decision = _trade_decision(lane_id=primary_lane)
            decision["evidence_refs"].extend(
                ["attack:advice", f"attack:lane:{primary_lane}"]
            )
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertNotIn("BASKET_PAIR_COVERAGE_INCOMPLETE", codes)

    def test_accepts_attack_priority_lane_in_selected_basket(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            priority_lane = "trend_trader:EUR_USD:SHORT:TREND_CONTINUATION:MARKET"
            files["intents"].write_text(
                json.dumps(
                    {
                        "results": [
                            _result(lane_id=priority_lane),
                            _result(),
                        ]
                    }
                )
            )
            files["attack_advice"].write_text(
                json.dumps(
                    {
                        "status": "ATTACK_PARTIAL",
                        "read_only": True,
                        "live_permission": False,
                        "recommended_now_lane_ids": [priority_lane, LANE_ID],
                    }
                )
            )
            decision = _batch_trade_decision([priority_lane, LANE_ID])
            decision["evidence_refs"].extend(
                ["attack:advice", f"attack:lane:{priority_lane}", f"attack:lane:{LANE_ID}"]
            )
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            self.assertEqual(payload["verification_issues"], [])

    def test_rejects_recommended_trade_without_attack_advice_evidence_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["attack_advice"].write_text(
                json.dumps(
                    {
                        "status": "ATTACK_PARTIAL",
                        "read_only": True,
                        "live_permission": False,
                        "recommended_now_lane_ids": [LANE_ID],
                    }
                )
            )
            brain = _brain(root, files, _trade_decision())

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("ATTACK_ADVICE_EVIDENCE_MISSING", codes)
            self.assertIn("ATTACK_ADVICE_LANE_EVIDENCE_MISSING", codes)

    def test_accepts_read_only_specialist_reviews(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision = _trade_decision()
            decision["specialist_reviews"] = [_specialist_review()]
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            self.assertEqual(payload["verification_issues"], [])

    def test_rejects_specialist_review_that_claims_live_permission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision = _trade_decision()
            decision["specialist_reviews"] = [_specialist_review(live_permission=True)]
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("SPECIALIST_REVIEW_LIVE_PERMISSION", codes)

    def test_rejects_specialist_review_that_is_not_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision = _trade_decision()
            decision["specialist_reviews"] = [_specialist_review(read_only=False)]
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("SPECIALIST_REVIEW_NOT_READ_ONLY", codes)

    def test_rejects_specialist_review_with_unknown_evidence_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision = _trade_decision()
            decision["specialist_reviews"] = [
                _specialist_review(cited_evidence_refs=["chart:EUR_USD:M1", "external:invented"])
            ]
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("UNKNOWN_SPECIALIST_REVIEW_REF", codes)

    def test_rejects_specialist_review_with_unknown_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision = _trade_decision()
            decision["specialist_reviews"] = [_specialist_review(lane_id="unknown:EUR_USD:LONG:TREND_CONTINUATION")]
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("UNKNOWN_SPECIALIST_REVIEW_LANE", codes)

    def test_rejects_specialist_review_method_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision = _trade_decision()
            decision["specialist_reviews"] = [_specialist_review(method="RANGE_ROTATION")]
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("SPECIALIST_REVIEW_METHOD_MISMATCH", codes)

    def test_rejects_specialist_review_with_execution_authority_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision = _trade_decision()
            review = _specialist_review()
            review["action"] = "TRADE"
            review["selected_lane_id"] = LANE_ID
            decision["specialist_reviews"] = [review]
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("SPECIALIST_REVIEW_AUTHORITY_FIELD", codes)

    def test_rejects_strategy_review_that_uses_wrong_method_for_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision = _trade_decision()
            decision["strategy_reviews"] = [
                {
                    "lane_id": LANE_ID,
                    "method": "RANGE_ROTATION",
                    "verdict": "SUPPORTS",
                    "summary": "wrong review method should not authorize the selected trend lane",
                }
            ]
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("STRATEGY_REVIEW_METHOD_MISMATCH", codes)


def _brain(root: Path, files: dict[str, Path], decision: dict, *, max_lanes: int | None = None) -> GPTTraderBrain:
    return GPTTraderBrain(
        provider=StaticTraderProvider(decision),
        intents_path=files["intents"],
        campaign_plan_path=files["campaign"],
        strategy_profile_path=files["strategy"],
        market_story_profile_path=files["story"],
        market_status_path=files["market_status"],
        target_state_path=files["target"],
        output_path=root / "gpt_decision.json",
        report_path=root / "gpt_decision.md",
        pair_charts_path=files["pair_charts"],
        context_asset_charts_path=files["context_asset_charts"],
        broker_instruments_path=files["broker_instruments"],
        cross_asset_path=files["cross_asset"],
        flow_path=files["flow"],
        currency_strength_path=files["currency_strength"],
        levels_path=files["levels"],
        market_context_matrix_path=files["market_context_matrix"],
        calendar_path=files["calendar"],
        cot_path=files["cot"],
        option_skew_path=files["option_skew"],
        attack_advice_path=files["attack_advice"],
        coverage_optimization_path=files["coverage_optimization"],
        learning_audit_path=files["learning_audit"],
        verification_ledger_path=files["verification_ledger"],
        self_improvement_audit_path=files["self_improvement_audit"],
        predictive_limits_path=files["predictive_limits"],
        **({"max_lanes": max_lanes} if max_lanes is not None else {}),
    )


def _write_entry_thesis_blocker(root: Path, files: dict[str, Path], *, trade_id: str) -> None:
    snapshot = json.loads(files["snapshot"].read_text())
    generated_at = (
        datetime.fromisoformat(snapshot["fetched_at_utc"]) + timedelta(seconds=1)
    ).isoformat()
    (root / "thesis_evolution_report.json").write_text(
        json.dumps(
            {
                "generated_at_utc": generated_at,
                "evolutions": [
                    {
                        "trade_id": trade_id,
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "status": "UNVERIFIABLE",
                        "verdict": "REQUIRE_THESIS_REPAIR",
                        "rationale": "missing entry_thesis_ledger row",
                    }
                ],
            }
        )
    )


def _fixtures(root: Path, *, positions: list[dict] | None = None, orders: list[dict] | None = None) -> dict[str, Path]:
    files = {
        "snapshot": root / "snapshot.json",
        "intents": root / "intents.json",
        "campaign": root / "campaign.json",
        "strategy": root / "strategy.json",
        "story": root / "story.json",
        "target": root / "target.json",
        "market_status": root / "market_status.json",
        "pair_charts": root / "pair_charts.json",
        "context_asset_charts": root / "context_asset_charts.json",
        "broker_instruments": root / "broker_instruments.json",
        "cross_asset": root / "cross_asset.json",
        "flow": root / "flow.json",
        "currency_strength": root / "currency_strength.json",
        "levels": root / "levels.json",
        "market_context_matrix": root / "market_context_matrix.json",
        "calendar": root / "calendar.json",
        "cot": root / "cot.json",
        "option_skew": root / "option_skew.json",
        "attack_advice": root / "attack_advice.json",
        "coverage_optimization": root / "coverage_optimization.json",
        "learning_audit": root / "learning_audit.json",
        "verification_ledger": root / "verification_ledger.json",
        "self_improvement_audit": root / "self_improvement_audit.json",
        "predictive_limits": root / "predictive_limits.json",
    }
    now = datetime.now(timezone.utc).isoformat()
    files["snapshot"].write_text(
        json.dumps(
            {
                "fetched_at_utc": now,
                "positions": positions or [],
                "orders": orders or [],
                "quotes": {"EUR_USD": {"bid": 1.172, "ask": 1.1721, "timestamp_utc": now}},
            }
        )
    )
    files["intents"].write_text(json.dumps({"results": [_result()]}))
    files["campaign"].write_text(
        json.dumps(
            {
                "lanes": [
                    {
                        "desk": "trend_trader",
                        "pair": "EUR_USD",
                        "direction": "LONG",
                        "method": "TREND_CONTINUATION",
                        "adoption": "ORDER_INTENT_REQUIRED",
                        "campaign_role": "NOW_IF_CLEAN",
                        "required_receipt": "live-ready continuation receipt",
                    }
                ]
            }
        )
    )
    files["strategy"].write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "pair": "EUR_USD",
                        "direction": "LONG",
                        "status": "CANDIDATE",
                        "pretrade_net_jpy": 5200,
                        "live_net_jpy": 1800,
                        "live_worst_jpy": -350,
                    }
                ]
            }
        )
    )
    files["story"].write_text(
        json.dumps(
            {
                "pair_profiles": [
                    {
                        "pair": "EUR_USD",
                        "methods": {"TREND_CONTINUATION": 12},
                        "themes": {"momentum": 4},
                        "examples": ["EUR_USD trend-bull staircase continuation"],
                    }
                ]
            }
        )
    )
    files["target"].write_text(
        json.dumps(
            {
                "status": "PURSUE_TARGET",
                "target_jpy": 22278.1,
                "progress_jpy": 0.0,
                "remaining_target_jpy": 22278.1,
                "remaining_risk_budget_jpy": 500.0,
            }
        )
    )
    files["market_status"].write_text(
        json.dumps(
            {
                "generated_at_utc": now,
                "evidence_ref": "market:status",
                "weekday": "Monday",
                "weekday_index": 0,
                "is_fx_open": True,
                "closed_reason": None,
                "active_sessions": ["London", "New_York"],
                "minutes_to_next_open": None,
                "minutes_to_next_close": 1800,
                "contract": {"live_permission": False, "must_not_override_broker_truth": True},
            }
        )
    )
    files["pair_charts"].write_text(
        json.dumps(
            {
                "generated_at_utc": now,
                "charts": [
                    {
                        "pair": "EUR_USD",
                        "dominant_regime": "TREND_UP",
                        "chart_story": "EUR_USD trend-up test story",
                        "long_score": 0.8,
                        "short_score": 0.2,
                        "session": {
                            "current_tag": "NY_AM_KILLZONE",
                            "jp_holiday": False,
                            "judas_armed": False,
                            "ny_midnight_open_price": 1.17,
                        },
                        "views": _chart_views(),
                    }
                ],
            }
        )
    )
    files["context_asset_charts"].write_text(
        json.dumps(
            {
                "generated_at_utc": now,
                "role": "NON_FX_CONTEXT_TECHNICALS_NOT_TRADE_PERMISSION",
                "charts": [
                    {
                        "pair": "XAU_USD",
                        "dominant_regime": "TREND_DOWN",
                        "chart_story": "XAU_USD trend-down context story",
                        "long_score": 0.15,
                        "short_score": 0.85,
                        "views": _chart_views(),
                    }
                ],
                "issues": [],
            }
        )
    )
    files["broker_instruments"].write_text(
        json.dumps(
            {
                "generated_at_utc": now,
                "status": "OK",
                "tradeability_policy": "BROKER_ACCOUNT_INSTRUMENTS_REQUIRED_FOR_LIVE_TRADE_UNIVERSE",
                "tradeable_instruments": ["EUR_USD"],
                "context_assets_tradeable": [],
                "context_assets_not_tradeable": ["XAU_USD"],
                "trader_pairs_missing": [],
                "specs": {"EUR_USD": {"type": "CURRENCY"}},
                "issues": [],
            }
        )
    )
    files["cross_asset"].write_text(
        json.dumps(
            {
                "generated_at_utc": now,
                "synthetic_dxy": {"last_value": 98.1, "change_pct_24h": -0.2},
                "yield_spreads": [{"name": "US10Y_minus_US2Y", "spread_last": 7.4}],
                "assets": [{"instrument": "USB10Y_USD", "trend_label": "UP", "last_price": 110.5}],
                "correlations": {"EUR_USD": {"USB10Y_USD": 0.15}},
                "issues": [],
            }
        )
    )
    files["flow"].write_text(
        json.dumps(
            {
                "generated_at_utc": now,
                "spreads": [
                    {
                        "instrument": "EUR_USD",
                        "current_pips": 0.8,
                        "median_pips": 1.2,
                        "p90_pips": 1.7,
                        "stress_flag": "NORMAL",
                    }
                ],
                "issues": [],
            }
        )
    )
    files["currency_strength"].write_text(
        json.dumps(
            {
                "generated_at_utc": now,
                "scores": [
                    {"currency": "EUR", "rank": 1, "score_pct": 0.4},
                    {"currency": "USD", "rank": 2, "score_pct": 0.2},
                ],
                "strongest_pair_suggestion": "EUR_USD",
                "issues": [],
            }
        )
    )
    files["levels"].write_text(
        json.dumps(
            {
                "generated_at_utc": now,
                "pairs": [
                    {
                        "pair": "EUR_USD",
                        "pdh": 1.18,
                        "pdl": 1.16,
                        "pdc": 1.17,
                        "daily_open": 1.171,
                        "pivots": [{"style": "STANDARD", "pp": 1.17, "r1": 1.18, "s1": 1.16}],
                        "round_numbers": [{"price": 1.18, "distance_pips": 8.0}],
                    }
                ],
                "issues": [],
            }
        )
    )
    files["market_context_matrix"].write_text(
        json.dumps(
            {
                "generated_at_utc": now,
                "trade_count_policy": "ADVISORY_ONLY_DOES_NOT_BLOCK_OR_DEMOTE_LANES",
                "pairs": {
                    "EUR_USD": {
                        "LONG": {
                            "evidence_ref": "matrix:EUR_USD:LONG",
                            "support_count": 3,
                            "reject_count": 1,
                            "warning_count": 1,
                            "missing_count": 1,
                            "strongest_support": "EUR_USD chart and strength support LONG",
                            "strongest_reject": "COT longer-term conflicts LONG",
                            "supports": [
                                {
                                    "code": "BASE_STRENGTH_EXCEEDS_QUOTE",
                                    "layer": "strength",
                                    "message": "EUR stronger than USD",
                                    "evidence_refs": ["strength:EUR", "strength:USD"],
                                }
                            ],
                            "rejects": [],
                            "warnings": [],
                        },
                        "SHORT": {
                            "evidence_ref": "matrix:EUR_USD:SHORT",
                            "support_count": 1,
                            "reject_count": 3,
                            "warning_count": 1,
                            "missing_count": 1,
                            "strongest_support": "COT longer-term aligns SHORT",
                            "strongest_reject": "EUR_USD chart and strength reject SHORT",
                            "supports": [],
                            "rejects": [
                                {
                                    "code": "BASE_STRENGTH_EXCEEDS_QUOTE",
                                    "layer": "strength",
                                    "message": "EUR stronger than USD",
                                    "evidence_refs": ["strength:EUR", "strength:USD"],
                                }
                            ],
                            "warnings": [],
                        },
                    }
                },
                "issues": [],
            }
        )
    )
    files["calendar"].write_text(
        json.dumps(
            {
                "generated_at_utc": now,
                "pair_windows": [
                    {
                        "pair": "EUR_USD",
                        "in_window": False,
                        "reason": "next event outside window",
                        "next_event": {"currency": "USD", "impact": "Medium", "title": "ADP"},
                    }
                ],
                "issues": [],
            }
        )
    )
    files["cot"].write_text(
        json.dumps(
            {
                "generated_at_utc": now,
                "reports": [
                    {"currency": "USD", "leveraged_net": 1234, "week_change_leveraged_net": 56},
                    {"currency": "EUR", "leveraged_net": -789, "week_change_leveraged_net": -12},
                ],
                "issues": [],
            }
        )
    )
    files["option_skew"].write_text(
        json.dumps(
            {
                "generated_at_utc": now,
                "provider": None,
                "enabled": False,
                "disabled_reason": "NO_OPTION_SKEW_PROVIDER",
                "readings": [],
                "issues": [],
            }
        )
    )
    files["attack_advice"].write_text(json.dumps({}))
    files["coverage_optimization"].write_text(json.dumps({"status": "OK"}))
    files["learning_audit"].write_text(json.dumps({}))
    files["self_improvement_audit"].write_text(json.dumps({}))
    files["predictive_limits"].write_text(json.dumps({"dry_run": True, "orders": []}))
    return files


def _self_improvement_profitability_p0() -> dict:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SELF_IMPROVEMENT_BLOCKED",
        "p0_findings": 1,
        "p1_findings": 2,
        "p2_findings": 0,
        "effect_metrics": {
            "closed_trades": 28,
            "net_jpy": -6571.91,
            "profit_factor": 0.508,
            "expectancy_jpy": -234.71,
            "avg_win_jpy": 356.47,
            "avg_loss_jpy_abs": 1482.76,
        },
        "findings": [
            {
                "priority": "P0",
                "layer": "profitability",
                "code": "PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED",
                "message": "profitability discipline has failed for 19 consecutive audit run(s)",
                "next_action": "Block new-risk confidence until execution_ledger.db worst segments prove repaired.",
                "evidence": {
                    "current_streak": 19,
                    "system_defect_evidence": {
                        "profit_factor": 0.508,
                        "expectancy_jpy": -234.71,
                        "avg_win_jpy": 356.47,
                        "avg_loss_jpy_abs": 1482.76,
                        "worst_segments": [
                            {
                                "pair": "EUR_USD",
                                "side": "SHORT",
                                "method": "BREAKOUT_FAILURE",
                                "trades": 6,
                                "net_jpy": -2977.0,
                                "expectancy_jpy": -496.17,
                            }
                        ],
                    },
                },
            }
        ],
    }


def _self_improvement_projection_p0() -> dict:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SELF_IMPROVEMENT_BLOCKED",
        "p0_findings": 1,
        "p1_findings": 1,
        "p2_findings": 0,
        "effect_metrics": {
            "closed_trades": 29,
            "net_jpy": -6138.23,
            "profit_factor": 0.54,
            "expectancy_jpy": -211.66,
        },
        "findings": [
            {
                "priority": "P0",
                "layer": "forecast",
                "code": "PROJECTION_LEDGER_EXPIRED_PENDING",
                "message": "projection ledger has 49 expired PENDING projection(s)",
                "next_action": "Run verify-projections and learn from HIT/MISS/TIMEOUT before new risk.",
                "evidence": {
                    "count": 49,
                    "examples": [
                        {
                            "pair": "AUD_CAD",
                            "signal_name": "directional_forecast",
                            "timestamp_emitted_utc": "2026-06-08T00:41:09.769570Z",
                            "resolution_window_min": 180.0,
                        }
                    ],
                },
            }
        ],
    }


def _tp_rebalance_pair_charts() -> dict:
    return {
        "charts": [
            {
                "pair": "EUR_USD",
                "confluence": {
                    "h4_atr_pips": 19.2,
                    "price_percentile_7d": 0.0,
                    "tf_agreement_score": 0.33,
                    "range_24h_sigma_multiple": 6.7,
                },
                "views": [
                    {
                        "granularity": "M15",
                        "indicators": {
                            "atr_pips": 19.2,
                            "stoch_rsi": 0.13,
                            "williams_r_14": -85.0,
                            "close": 1.15970,
                            "bb_lower": 1.15972,
                        },
                        "structure": {
                            "liquidity": [
                                {
                                    "side": "EQ_LOW",
                                    "price": 1.15875,
                                    "indices": [1, 2, 3, 4],
                                }
                            ]
                        },
                    }
                ],
            }
        ]
    }


def _chart_views() -> list[dict]:
    return [
        _chart_view("M1", atr_pips=1.2, state="TREND_WEAK", last_jump_bars_ago=12),
        _chart_view("M5", atr_pips=5.3, state="TREND_STRONG", last_jump_bars_ago=8),
        _chart_view("M15", atr_pips=9.1, state="TREND_STRONG", last_jump_bars_ago=18),
        _chart_view("M30", atr_pips=13.4, state="TREND_WEAK", last_jump_bars_ago=24),
        _chart_view("H1", atr_pips=18.2, state="TREND_WEAK", last_jump_bars_ago=31),
        _chart_view("H4", atr_pips=35.8, state="TREND_WEAK", last_jump_bars_ago=40),
        _chart_view("D", atr_pips=76.4, state="RANGE", last_jump_bars_ago=55),
    ]


def _chart_view(granularity: str, *, atr_pips: float, state: str, last_jump_bars_ago: int) -> dict:
    return {
        "granularity": granularity,
        "indicators": {
            "atr_pips": atr_pips,
            "adx_14": 42.0,
            "rsi_14": 61.0,
            "choppiness_14": 39.0,
            "bb_width_percentile_100": 0.62,
            "atr_percentile_100": 0.71,
        },
        "regime_reading": {
            "state": state,
            "confidence": 0.82,
            "hurst": 0.58,
        },
        "family_scores": {
            "trend_score": 1.2,
            "mean_rev_score": -0.4,
            "breakout_score": 0.3,
            "disagreement": 0.35,
        },
        "stat_filters": {
            "last_jump_bars_ago": last_jump_bars_ago,
            "lag1_autocorr": 0.12,
        },
    }


def _result(*, lane_id: str = LANE_ID, method: str = "TREND_CONTINUATION") -> dict:
    return {
        "lane_id": lane_id,
        "status": "LIVE_READY",
        "risk_allowed": True,
        "risk_issues": [],
        "strategy_issues": [],
        "live_blockers": [],
        "intent": {
            "pair": "EUR_USD",
            "side": "LONG",
            "order_type": "STOP-ENTRY",
            "units": 1000,
            "entry": 1.1725,
            "tp": 1.1737,
            "sl": 1.1717,
            "thesis": "EUR_USD continuation can pay before daily target window closes.",
            "owner": "trader",
            "market_context": {
                "regime": f"{method} campaign lane",
                "narrative": "Dollar pressure and momentum theme favor EUR_USD continuation.",
                "chart_story": "Higher lows are pressing into the trigger shelf.",
                "method": method,
                "invalidation": "Invalid if the shelf breaks before entry.",
                "event_risk": "",
                "session": "test",
            },
        },
    }


def _trade_decision(*, lane_id: str = LANE_ID, method: str = "TREND_CONTINUATION") -> dict:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "action": "TRADE",
        "selected_lane_id": lane_id,
        "confidence": "HIGH",
        "thesis": "The live-ready EUR_USD continuation lane has current story and positive mined evidence.",
        "method": method,
        "narrative": "Momentum and campaign role align with a controlled stop-entry.",
        "chart_story": "Higher lows press into the trigger shelf.",
        "invalidation": "Do not trade if the shelf fails before entry or the SL level trades.",
        "rejected_alternatives": ["WAIT rejected because the target gap remains open and lane is clean."],
        "risk_notes": ["Use only the lane units, TP, and SL already verified by the dry-run receipt."],
        "evidence_refs": [
            "broker:snapshot",
            "target:daily",
            f"intent:{lane_id}",
            f"campaign:{lane_id}",
            "strategy:EUR_USD:LONG",
            "story:EUR_USD",
            "chart:EUR_USD:M5",
            "chart:EUR_USD:M15",
        ],
        "twenty_minute_plan": _twenty_minute_plan(lane_ids=[lane_id]),
        "operator_summary": "Accept the verified EUR_USD continuation lane.",
    }


def _batch_trade_decision(lane_ids: list[str]) -> dict:
    decision = _trade_decision(lane_id=lane_ids[0])
    decision["selected_lane_ids"] = lane_ids
    refs = list(decision["evidence_refs"])
    for lane_id in lane_ids[1:]:
        refs.extend([f"intent:{lane_id}", f"campaign:{lane_id}"])
    decision["evidence_refs"] = refs
    decision["twenty_minute_plan"] = _twenty_minute_plan(lane_ids=lane_ids)
    decision["operator_summary"] = "Accept the verified EUR_USD continuation basket."
    return decision


def _twenty_minute_plan(*, lane_ids: list[str] | None = None, pair: str = "EUR_USD") -> dict:
    refs = [f"chart:{pair}:M5", f"chart:{pair}:M15"]
    for lane_id in lane_ids or []:
        refs.append(f"intent:{lane_id}")
    return {
        "horizon_minutes": 20,
        "primary_path": f"{pair} should hold the M5 shelf and press toward the selected trigger before the next cycle.",
        "failure_path": "A close back through the shelf or a newly named packet blocker makes the idea wrong.",
        "entry_or_hold_trigger": "Use only the current LIVE_READY intent trigger or hold WAIT if that trigger is absent.",
        "invalidation_or_cancel_trigger": "Cancel the idea if the invalidation shelf breaks or the selected intent leaves LIVE_READY.",
        "counterargument": "M15 can still fade the move; the trade is only acceptable because current chart refs keep the shelf intact.",
        "next_cycle_check": "First re-check broker truth, the selected lane status, and M5/M15 structure before extending the thesis.",
        "evidence_refs": refs,
    }


def _learning_influenced_attack_advice(*, lane_id: str = LANE_ID) -> dict:
    return {
        "status": "ATTACK_PARTIAL",
        "read_only": True,
        "live_permission": False,
        "recommended_now_lane_ids": [lane_id],
        "recommended_now_reward_jpy": 900.0,
        "recommended_now_risk_jpy": 300.0,
        "lanes": [
            {
                "lane_id": lane_id,
                "score": 44.0,
                "learning_influences": ["ai_backtest_research_positive_edge"],
                "learning_score_delta": 8.0,
                "learning_influence_details": [
                    {
                        "influence": "ai_backtest_research_positive_edge",
                        "source": "ai_backtest",
                        "reason": "profitable research edge, reduced weight",
                        "score_delta": 8.0,
                    }
                ],
            }
        ],
    }


def _learning_audit_payload(
    *,
    status: str,
    lane_id: str = LANE_ID,
    blockers: list[str] | None = None,
    exit_reason_metrics: dict[str, dict] | None = None,
) -> dict:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "blockers": blockers or [],
        "warnings": ["research edge is not target-coverage certified"] if status == "LEARNING_AUDIT_WARN" else [],
        "learning_influence": {
            "influenced_lanes": 1,
            "total_learning_score_delta": 8.0,
            "lanes": [
                {
                    "lane_id": lane_id,
                    "learning_influences": ["ai_backtest_research_positive_edge"],
                    "learning_score_delta": 8.0,
                }
            ],
        },
        "effect_metrics": {
            "closed_trades": 30,
            "net_jpy": 1200.0,
            "profit_factor": 1.2,
            "expectancy_jpy": 40.0,
            "exit_reason_metrics": exit_reason_metrics or {},
        },
    }


def _specialist_review(
    *,
    role: str = "indicator",
    lane_id: str | None = LANE_ID,
    method: str | None = "TREND_CONTINUATION",
    verdict: str = "SUPPORTS",
    cited_evidence_refs: list[str] | None = None,
    read_only: bool = True,
    live_permission: bool = False,
) -> dict:
    return {
        "role": role,
        "lane_id": lane_id,
        "method": method,
        "verdict": verdict,
        "summary": "M1 has no fresh jump and H4/D do not contradict the continuation lane.",
        "cited_evidence_refs": cited_evidence_refs or [
            "chart:EUR_USD:M1",
            "chart:EUR_USD:M5",
            "chart:EUR_USD:H4",
            "chart:EUR_USD:D",
        ],
        "hard_gate_codes": [],
        "read_only": read_only,
        "live_permission": live_permission,
    }


def _request_evidence_decision() -> dict:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "action": "REQUEST_EVIDENCE",
        "selected_lane_id": None,
        "confidence": "HIGH",
        "thesis": "Request more evidence because the packet appears to have no executable lanes.",
        "method": "EVENT_RISK",
        "narrative": "The daily target is open, but the operator believes live-ready coverage is absent.",
        "chart_story": "No clean chart story is accepted yet.",
        "invalidation": "Refresh when current broker truth produces a live-ready lane.",
        "rejected_alternatives": ["TRADE rejected because no current lane appears executable."],
        "risk_notes": ["Stay flat until executable evidence appears."],
        "evidence_refs": ["broker:snapshot", "target:daily", "chart:EUR_USD:M5"],
        "twenty_minute_plan": _twenty_minute_plan(),
        "operator_summary": "Do not trade from this stale evidence request.",
    }


def _wait_decision() -> dict:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "action": "WAIT",
        "selected_lane_id": None,
        "confidence": "MEDIUM",
        "thesis": "Wait despite the live-ready lane because discretionary timing is not clean enough.",
        "method": "EVENT_RISK",
        "narrative": "The lane is executable, but event timing argues for patience this cycle.",
        "chart_story": "The trigger shelf exists, but confirmation has not printed yet.",
        "invalidation": "Reconsider if the shelf holds and spread remains inside the receipt.",
        "rejected_alternatives": [f"{LANE_ID} rejected for this cycle because timing confirmation is incomplete."],
        "risk_notes": ["No exposure is open; waiting adds no risk."],
        "evidence_refs": [
            "broker:snapshot",
            "target:daily",
            f"intent:{LANE_ID}",
            f"campaign:{LANE_ID}",
            "strategy:EUR_USD:LONG",
            "story:EUR_USD",
            "chart:EUR_USD:M5",
        ],
        "twenty_minute_plan": _twenty_minute_plan(lane_ids=[LANE_ID]),
        "operator_summary": "Wait with an explicit rejection of the current executable lane.",
    }


def _protect_decision() -> dict:
    decision = _wait_decision()
    decision.update(
        {
            "action": "PROTECT",
            "confidence": "HIGH",
            "thesis": "Keep exposure protected while the position-management gateway runs.",
            "method": "POSITION_MANAGEMENT",
            "narrative": "Open trader exposure needs protection review before fresh entries.",
            "chart_story": "Position-management context is active.",
            "invalidation": "Switch to CLOSE if fresh sidecars prove the recovery edge is broken.",
            "operator_summary": "Run protection only.",
        }
    )
    return decision


def _tighten_sl_decision() -> dict:
    decision = _protect_decision()
    decision.update(
        {
            "action": "TIGHTEN_SL",
            "thesis": "Tighten eligible broker-side stop only when protection rules allow it.",
            "operator_summary": "Tighten stop only.",
        }
    )
    return decision


def _cancel_pending_decision(*, cancel_order_ids: list[str] | None = None) -> dict:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "action": "CANCEL_PENDING",
        "selected_lane_id": None,
        "cancel_order_ids": cancel_order_ids or [],
        "confidence": "HIGH",
        "thesis": "The pending entry is stale relative to current broker truth and should be cleared before new risk.",
        "method": "POSITION_MANAGEMENT",
        "narrative": "A pending order blocks clean discretionary comparison.",
        "chart_story": "The original trigger has drifted away from the current executable lane.",
        "invalidation": "Do not cancel if the order id is not present in current broker truth.",
        "rejected_alternatives": ["TRADE rejected until pending exposure is resolved."],
        "risk_notes": ["Canceling a pending entry reduces possible future exposure."],
        "evidence_refs": ["broker:snapshot", "target:daily"],
        "operator_summary": "Clear the stale pending order before considering another entry.",
    }


def _pending_order() -> dict:
    return {
        "order_id": "pending-1",
        "pair": "EUR_USD",
        "order_type": "STOP",
        "price": 1.173,
        "state": "PENDING",
        "units": 1000,
        "owner": "trader",
    }


def _position(*, stop_loss: float | None = 1.17) -> dict:
    return {
        "trade_id": "101",
        "pair": "EUR_USD",
        "side": "LONG",
        "units": 1000,
        "entry_price": 1.171,
        "unrealized_pl_jpy": 120.0,
        "take_profit": 1.173,
        "stop_loss": stop_loss,
        "owner": "trader",
    }


# Helpers for CLOSE-discipline tests (2026-05-12, feedback_no_unilateral_close.md).

import os as _os
from quant_rabbit.gpt_trader import _parse_struct_events, _close_thesis_invalidated


def _chart_story_with_struct(pair: str, m15_dir: str = "UP", h4_dir: str = "UP") -> str:
    """Return a chart_story snippet matching chart_reader's emit format with
    a controllable M15 and H4 struct event so tests can flip thesis-valid
    vs thesis-invalidated. Other TFs print neutral non-counter events so
    they never coincidentally satisfy the gate."""
    return (
        f"{pair} RANGE; "
        f"M1(RANGE, ADX=15 RSI=50 ATR=1.0p struct=BOS_UP@1.0000); "
        f"M5(RANGE, ADX=15 RSI=50 ATR=2.0p struct=BOS_UP@1.0000); "
        f"M15(RANGE, ADX=20 RSI=50 ATR=3.0p struct=BOS_{m15_dir}@1.1000); "
        f"M30(RANGE, ADX=20 RSI=50 ATR=4.0p struct=BOS_UP@1.0000); "
        f"H1(RANGE, ADX=20 RSI=50 ATR=5.0p struct=BOS_UP@1.0000); "
        f"H4(RANGE, ADX=25 RSI=50 ATR=8.0p struct=CHOCH_{h4_dir}@1.2000); "
        f"D(RANGE, ADX=15 RSI=50 ATR=15.0p struct=BOS_UP@1.0000)"
    )


def _close_decision(
    *,
    trade_ids: list[str],
    operator_close_authorized: bool = False,
    invalidation_price: float | None = None,
    invalidation_tf: str | None = None,
) -> dict:
    """Decision payload for action=CLOSE with the new discipline fields."""
    decision = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "action": "CLOSE",
        "selected_lane_id": None,
        "selected_lane_ids": [],
        "cancel_order_ids": [],
        "close_trade_ids": trade_ids,
        "confidence": "MEDIUM",
        "thesis": "Close per operator-cited invalidation.",
        "method": "POSITION_MANAGEMENT",
        "narrative": "Operator-directed close on trader-owned position(s).",
        "chart_story": "See pair_charts for current structure.",
        "invalidation": "See `invalidation_price` and `invalidation_tf` if cited.",
        "rejected_alternatives": [],
        "risk_notes": [],
        "evidence_refs": ["broker:snapshot", "target:daily"],
        "operator_summary": "Close trader-owned positions per gate-A + gate-B authorization.",
    }
    if invalidation_price is not None:
        decision["invalidation_price"] = invalidation_price
    if invalidation_tf is not None:
        decision["invalidation_tf"] = invalidation_tf
    if operator_close_authorized:
        decision["operator_close_authorized"] = True
    return decision


def _close_tech_views(move: str = "UP") -> list[dict]:
    direction = move.upper()
    if direction == "UP":
        return [
            {
                "granularity": tf,
                "regime": "TREND_UP",
                "indicators": {
                    "rsi_14": 70.0,
                    "macd_hist": 0.0002,
                    "supertrend_dir": 1,
                    "ichimoku_cloud_pos": 1,
                    "plus_di_14": 35.0,
                    "minus_di_14": 10.0,
                },
                "structure": {"last_event": {"kind": "CHOCH_UP", "close_confirmed": True}},
            }
            for tf in ("M5", "M15")
        ]
    return [
        {
            "granularity": tf,
            "regime": "TREND_DOWN",
            "indicators": {
                "rsi_14": 30.0,
                "macd_hist": -0.0002,
                "supertrend_dir": -1,
                "ichimoku_cloud_pos": -1,
                "plus_di_14": 10.0,
                "minus_di_14": 35.0,
            },
            "structure": {"last_event": {"kind": "CHOCH_DOWN", "close_confirmed": True}},
        }
        for tf in ("M5", "M15")
    ]


def _close_fixtures(
    root: Path,
    *,
    position_side: str = "SHORT",
    m15_dir: str = "UP",
    h4_dir: str = "UP",
    quote_bid: float = 1.176,
    quote_ask: float = 1.1761,
    unrealized_pl_jpy: float = -800.0,
) -> dict[str, Path]:
    """Build minimal fixtures for a CLOSE-discipline test.

    Defaults: one trader-owned EUR_USD SHORT position whose chart_story
    prints UP-direction structure (i.e. thesis-invalidated against
    SHORT). Override `m15_dir`/`h4_dir` to flip the gate.
    """
    pos = {
        "trade_id": "555",
        "pair": "EUR_USD",
        "side": position_side,
        "units": 9000,
        "entry_price": 1.17708 if position_side == "SHORT" else 1.17400,
        "unrealized_pl_jpy": unrealized_pl_jpy,
        "take_profit": 1.17060 if position_side == "SHORT" else 1.18000,
        "stop_loss": None,
        "owner": "trader",
    }
    files = _fixtures(root, positions=[pos])
    # Override snapshot quotes to control invalidation_price hit testing.
    snap = json.loads(files["snapshot"].read_text())
    snap["quotes"] = {
        "EUR_USD": {"bid": quote_bid, "ask": quote_ask, "timestamp_utc": snap["fetched_at_utc"]},
    }
    files["snapshot"].write_text(json.dumps(snap))
    # Override pair_charts chart_story so the CLOSE gate sees the
    # M15/H4 structural events we want.
    pc = json.loads(files["pair_charts"].read_text())
    pc["charts"][0]["chart_story"] = _chart_story_with_struct("EUR_USD", m15_dir=m15_dir, h4_dir=h4_dir)
    pc["charts"][0]["views"] = _close_tech_views("UP" if position_side == "SHORT" else "DOWN")
    files["pair_charts"].write_text(json.dumps(pc))
    return files


def _write_fresh_forecast_close_recommendation(
    root: Path,
    files: dict[str, Path],
    *,
    trade_id: str = "555",
    pair: str = "EUR_USD",
    side: str = "SHORT",
) -> None:
    snapshot = json.loads(files["snapshot"].read_text())
    generated_at = (
        datetime.fromisoformat(snapshot["fetched_at_utc"]) + timedelta(seconds=1)
    ).isoformat()
    (root / "forecast_persistence_report.json").write_text(
        json.dumps(
            {
                "generated_at_utc": generated_at,
                "verdicts": [
                    {
                        "trade_id": trade_id,
                        "pair": pair,
                        "side": side,
                        "verdict": "RECOMMEND_CLOSE",
                        "reason": "fresh forecast persistence no longer supports recovery",
                    }
                ],
            }
        )
    )


def _write_fresh_position_thesis_close_recommendation(
    root: Path,
    files: dict[str, Path],
    *,
    trade_id: str = "555",
    pair: str = "EUR_USD",
    side: str = "SHORT",
    rationale_lines: list[str] | None = None,
    context_notes: list[str] | None = None,
) -> None:
    snapshot = json.loads(files["snapshot"].read_text())
    generated_at = (
        datetime.fromisoformat(snapshot["fetched_at_utc"]) + timedelta(seconds=1)
    ).isoformat()
    rationale_lines = rationale_lines or ["soft position thesis review says recovery edge is weak"]
    context_notes = context_notes or []
    (root / "position_thesis_report.json").write_text(
        json.dumps(
            {
                "generated_at_utc": generated_at,
                "assessments": [
                    {
                        "trade_id": trade_id,
                        "pair": pair,
                        "side": side,
                        "verdict": "REVIEW_CLOSE",
                        "rationale_lines": rationale_lines,
                        "context_notes": context_notes,
                    }
                ],
            }
        )
    )


def _write_same_direction_context_asset_matrix_support(
    files: dict[str, Path],
    *,
    pair: str = "EUR_USD",
    side: str = "LONG",
) -> None:
    side_upper = side.upper()
    matrix = json.loads(files["market_context_matrix"].read_text())
    pair_matrix = matrix.setdefault("pairs", {}).setdefault(pair, {})
    reading = pair_matrix.setdefault(side_upper, {})
    reading.update(
        {
            "evidence_ref": f"matrix:{pair}:{side_upper}",
            "support_count": 1,
            "reject_count": 0,
            "warning_count": 0,
            "missing_count": 0,
            "strongest_support": "XAU_USD context asset chart supports the open position side",
            "strongest_reject": None,
            "supports": [
                {
                    "code": "CONTEXT_ASSET_SUPPORTS_OPEN_SIDE",
                    "layer": "context_asset_chart",
                    "message": "XAU_USD chart pressure still supports the open EUR_USD side",
                    "evidence_refs": ["context_asset:XAU_USD", f"matrix:{pair}:{side_upper}"],
                }
            ],
            "rejects": [],
            "warnings": [],
        }
    )
    files["market_context_matrix"].write_text(json.dumps(matrix))


def _write_fresh_thesis_evolution_close_recommendation(
    root: Path,
    files: dict[str, Path],
    *,
    trade_id: str = "555",
    pair: str = "EUR_USD",
    side: str = "SHORT",
) -> None:
    snapshot = json.loads(files["snapshot"].read_text())
    generated_at = (
        datetime.fromisoformat(snapshot["fetched_at_utc"]) + timedelta(seconds=1)
    ).isoformat()
    (root / "thesis_evolution_report.json").write_text(
        json.dumps(
            {
                "generated_at_utc": generated_at,
                "evolutions": [
                    {
                        "trade_id": trade_id,
                        "pair": pair,
                        "side": side,
                        "status": "BROKEN",
                        "verdict": "RECOMMEND_CLOSE",
                        "rationale": "invalidation hit and technical invalidation confirmed against the entry thesis",
                    }
                ],
            }
        )
    )


def _write_recent_position_management_close_recommendation(
    root: Path,
    files: dict[str, Path],
    *,
    trade_id: str = "555",
    pair: str = "EUR_USD",
    side: str = "SHORT",
    reasons: list[str] | None = None,
) -> None:
    snapshot = json.loads(files["snapshot"].read_text())
    generated_at = (
        datetime.fromisoformat(snapshot["fetched_at_utc"]) - timedelta(minutes=20)
    ).isoformat()
    if reasons is None:
        reasons = [
            "score context before structural review",
            "loss-cut: structural OB broken across 2 TFs (M15@1.17000, H1@1.17100) (-1900 JPY)",
        ]
    (root / "position_management.json").write_text(
        json.dumps(
            {
                "generated_at_utc": generated_at,
                "action": "REVIEW_EXIT",
                "positions": [
                    {
                        "trade_id": trade_id,
                        "pair": pair,
                        "side": side,
                        "action": "REVIEW_EXIT",
                        "reasons": reasons,
                    }
                ],
            }
        )
    )


class CloseDisciplineTest(unittest.TestCase):
    """Coverage for 2026-05-12 CLOSE two-gate discipline added in
    response to the 2026-05-11 18:17 UTC mass-close regression where the
    GPT trader autonomously closed four valid SHORT positions for
    -3,291 JPY. Mirrors `feedback_no_unilateral_close.md` and the
    AGENT_CONTRACT §10 CLOSE discipline section.
    """

    def setUp(self) -> None:
        self._prior_override = _os.environ.pop("QR_OPERATOR_CLOSE_OVERRIDE", None)

    def tearDown(self) -> None:
        if self._prior_override is None:
            _os.environ.pop("QR_OPERATOR_CLOSE_OVERRIDE", None)
        else:
            _os.environ["QR_OPERATOR_CLOSE_OVERRIDE"] = self._prior_override

    def test_close_rejected_when_thesis_still_valid_even_with_operator_auth(self) -> None:
        # SHORT position + chart_story shows BOS_UP on M15/H4? No — both
        # set to UP would invalidate. Force both to DOWN so neither
        # counter-direction event prints against SHORT.
        # Gate B via env override (J hardening 2026-05-13) so the only
        # reason to reject is Gate A (thesis still valid).
        _os.environ["QR_OPERATOR_CLOSE_OVERRIDE"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="DOWN", h4_dir="DOWN")
            decision = _close_decision(trade_ids=["555"])
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            self.assertFalse(summary.allowed)
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("CLOSE_THESIS_STILL_VALID", codes)

    def test_close_accepted_when_m15_bos_against_side_and_operator_authorized(self) -> None:
        # SHORT position + M15 prints BOS_UP (against SHORT) → gate A
        # passes via structural lens. Gate B via env override (J hardening
        # 2026-05-13: receipt's operator_close_authorized field is
        # advisory-only and no longer satisfies Gate B by itself).
        _os.environ["QR_OPERATOR_CLOSE_OVERRIDE"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="UP", h4_dir="DOWN")
            decision = _close_decision(trade_ids=["555"])
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED", msg=summary)
            self.assertTrue(summary.allowed)

    def test_close_accepted_when_h4_choch_against_side(self) -> None:
        # SHORT position + H4 prints CHOCH_UP (against SHORT), M15 neutral.
        # Gate B via env override (J hardening 2026-05-13).
        _os.environ["QR_OPERATOR_CLOSE_OVERRIDE"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="DOWN", h4_dir="UP")
            decision = _close_decision(trade_ids=["555"])
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")

    def test_close_accepted_when_invalidation_price_hit_on_broker_truth(self) -> None:
        # SHORT @ 1.17708, TP 1.17060, no structural counter-event.
        # Receipt cites invalidation_price=1.1750 + tf=H1; broker ask
        # 1.1761 clears the anti-wick buffer above 1.1750 → gate A passes.
        # Gate B via env override (J hardening 2026-05-13).
        _os.environ["QR_OPERATOR_CLOSE_OVERRIDE"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(
                root,
                position_side="SHORT",
                m15_dir="DOWN",
                h4_dir="DOWN",
                quote_bid=1.1760,
                quote_ask=1.1761,
            )
            decision = _close_decision(
                trade_ids=["555"],
                invalidation_price=1.1750,
                invalidation_tf="H1",
            )
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")

    def test_close_rejected_when_invalidation_only_wicks_inside_buffer(self) -> None:
        # A shallow touch of the invalidation level is not enough. The close
        # gate requires price to clear the anti-wick buffer so tiny stop hunts
        # do not authorize loss cuts.
        _os.environ["QR_OPERATOR_CLOSE_OVERRIDE"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(
                root,
                position_side="SHORT",
                m15_dir="DOWN",
                h4_dir="DOWN",
                quote_bid=1.1750,
                quote_ask=1.1751,
            )
            decision = _close_decision(
                trade_ids=["555"],
                invalidation_price=1.1750,
                invalidation_tf="H1",
            )
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("CLOSE_THESIS_STILL_VALID", codes)

    def test_close_accepted_without_operator_token_when_structural_invalidation_is_hard(self) -> None:
        # Structural invalidation passes (M15 BOS_UP vs SHORT). The user's
        # standing directive allows justified loss-cuts, so hard Gate A does
        # not need the 5-minute token.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="UP", h4_dir="UP")
            decision = _close_decision(
                trade_ids=["555"],
                operator_close_authorized=False,
            )
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            self.assertTrue(summary.allowed)
            payload = json.loads((root / "gpt_decision.json").read_text())
            self.assertEqual(payload["verification_issues"], [])

    def test_close_accepted_when_qr_operator_close_override_env_set(self) -> None:
        # Emergency override path: env QR_OPERATOR_CLOSE_OVERRIDE=1
        # bypasses gate B even when receipt lacks operator_close_authorized.
        # Gate A still required.
        _os.environ["QR_OPERATOR_CLOSE_OVERRIDE"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="UP", h4_dir="UP")
            decision = _close_decision(trade_ids=["555"], operator_close_authorized=False)
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")

    def test_trade_receipt_cannot_close_and_reenter_in_same_receipt(self) -> None:
        # Loss-cut and re-entry must be separate receipts. Otherwise the
        # trader can close a broken thesis and immediately chase a new lane
        # without a refreshed broker snapshot / margin / intent packet.
        _os.environ["QR_OPERATOR_CLOSE_OVERRIDE"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="UP", h4_dir="UP")
            decision = _trade_decision()
            decision["close_trade_ids"] = ["555"]
            decision["operator_summary"] = "Close the broken SHORT and immediately re-enter via the selected LONG lane."
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("CLOSE_REENTRY_SAME_RECEIPT", codes)

    def test_close_accepted_when_fresh_forecast_persistence_recommends_close_and_operator_authorized(self) -> None:
        # Sidecar recommendations are Gate A only: they can prove the
        # position thesis no longer has recovery edge, but Gate B remains
        # operator-controlled.
        _os.environ["QR_OPERATOR_CLOSE_OVERRIDE"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="DOWN", h4_dir="DOWN")
            snapshot = json.loads(files["snapshot"].read_text())
            generated_at = (
                datetime.fromisoformat(snapshot["fetched_at_utc"]) + timedelta(seconds=1)
            ).isoformat()
            (root / "forecast_persistence_report.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": generated_at,
                        "verdicts": [
                            {
                                "trade_id": "555",
                                "pair": "EUR_USD",
                                "side": "SHORT",
                                "verdict": "RECOMMEND_CLOSE",
                                "reason": "last 3 forecasts flipped to UP",
                            }
                        ],
                    }
                )
            )
            decision = _close_decision(trade_ids=["555"])
            decision["evidence_refs"].append("position:persistence:555")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED", msg=summary)
            self.assertTrue(summary.allowed)
            payload = json.loads((root / "gpt_decision.json").read_text())
            sidecar = payload["input_packet"]["protection_sidecars"]["position_close_recommendations"][0]
            self.assertEqual(sidecar["source"], "forecast_persistence")
            self.assertIn("position:persistence:555", payload["input_packet"]["allowed_evidence_refs"])

    def test_close_rejected_without_operator_token_when_only_forecast_persistence_sidecar(self) -> None:
        # Forecast persistence is useful Gate A evidence, but it is softer than
        # structural invalidation / thesis_evolution BROKEN. It still needs
        # explicit env/token Gate B.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="DOWN", h4_dir="DOWN")
            _write_fresh_forecast_close_recommendation(root, files)
            decision = _close_decision(trade_ids=["555"])
            decision["evidence_refs"].append("position:persistence:555")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("CLOSE_OPERATOR_AUTH_REQUIRED", codes)
            self.assertNotIn("CLOSE_SAME_DIRECTION_MARKET_SUPPORT", codes)
            self.assertNotIn("CLOSE_THESIS_STILL_VALID", codes)

    def test_close_rejected_without_operator_token_when_only_soft_position_thesis_sidecar(self) -> None:
        # Score-only position_thesis review is Gate A evidence, but not hard
        # standing loss-cut authorization.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="DOWN", h4_dir="DOWN")
            _write_fresh_position_thesis_close_recommendation(root, files)
            decision = _close_decision(trade_ids=["555"])
            decision["evidence_refs"].append("position:thesis:555")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("CLOSE_OPERATOR_AUTH_REQUIRED", codes)
            self.assertNotIn("CLOSE_SAME_DIRECTION_MARKET_SUPPORT", codes)
            self.assertNotIn("CLOSE_THESIS_STILL_VALID", codes)
            recs = payload["input_packet"]["protection_sidecars"]["position_close_recommendations"]
            self.assertFalse(recs[0]["gate_b_standing_authorized"])

    def test_close_rejected_when_soft_sidecar_conflicts_with_same_direction_context_asset_matrix(self) -> None:
        # A soft position_thesis review plus operator Gate B is still not enough
        # when the directional market stack still supports the open side. This
        # pins AGENT_CONTRACT §10: same-direction recovery edge should become
        # HOLD/reprice/TP rebalance, not GPT-driven loss close.
        _os.environ["QR_OPERATOR_CLOSE_OVERRIDE"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="LONG", m15_dir="UP", h4_dir="UP")
            _write_same_direction_context_asset_matrix_support(files, side="LONG")
            _write_fresh_position_thesis_close_recommendation(root, files, side="LONG")
            decision = _close_decision(trade_ids=["555"])
            decision["evidence_refs"].extend(
                ["position:thesis:555", "matrix:EUR_USD:LONG", "context_asset:XAU_USD"]
            )
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED", msg=summary)
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("CLOSE_SAME_DIRECTION_MARKET_SUPPORT", codes)
            self.assertNotIn("CLOSE_OPERATOR_AUTH_REQUIRED", codes)
            self.assertNotIn("CLOSE_THESIS_STILL_VALID", codes)

    def test_profit_side_soft_close_not_blocked_by_same_direction_matrix_support(self) -> None:
        # The same-direction matrix blocker is for loss-side soft closes. A
        # profitable operator-authorized close can still pass if Gate A sidecar
        # evidence is present.
        _os.environ["QR_OPERATOR_CLOSE_OVERRIDE"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(
                root,
                position_side="LONG",
                m15_dir="UP",
                h4_dir="UP",
                unrealized_pl_jpy=350.0,
            )
            _write_same_direction_context_asset_matrix_support(files, side="LONG")
            _write_fresh_position_thesis_close_recommendation(root, files, side="LONG")
            decision = _close_decision(trade_ids=["555"])
            decision["evidence_refs"].extend(
                ["position:thesis:555", "matrix:EUR_USD:LONG", "context_asset:XAU_USD"]
            )
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED", msg=summary)
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertNotIn("CLOSE_SAME_DIRECTION_MARKET_SUPPORT", codes)

    def test_close_rejected_without_operator_token_when_only_soft_position_management_review_exit(self) -> None:
        # PositionManager REVIEW_EXIT is carried into Gate A, but score/advisory
        # reasons are not standing loss-cut authorization.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="DOWN", h4_dir="DOWN")
            _write_recent_position_management_close_recommendation(
                root,
                files,
                reasons=["score weakened but no structural loss-cut reason"],
            )
            decision = _close_decision(trade_ids=["555"])
            decision["evidence_refs"].append("position:management:555")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("CLOSE_OPERATOR_AUTH_REQUIRED", codes)
            self.assertNotIn("CLOSE_THESIS_STILL_VALID", codes)
            recs = payload["input_packet"]["protection_sidecars"]["position_close_recommendations"]
            self.assertEqual(recs[0]["source"], "position_management")
            self.assertFalse(recs[0]["gate_b_standing_authorized"])

    def test_close_rejected_without_operator_token_when_position_thesis_adverse_loss_only(self) -> None:
        # Legacy/no-ledger adverse-entry-buffer loss is soft. It may be useful
        # evidence, but without invalidation-hit / structural-break evidence it
        # must not become standing permission to realize a loss.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="DOWN", h4_dir="DOWN")
            _write_fresh_position_thesis_close_recommendation(
                root,
                files,
                rationale_lines=[
                    "patterns +30.0",
                    "forward-proj +25.0",
                    "chart-tech -8.0",
                ],
                context_notes=[
                    "adverse technical loss: no entry thesis; current ask 1.16310 >= entry-buffer 1.15891",
                    "technical invalidation confirmed against SHORT: H1 RSI=65.3; M15 BOS_UP; M30 MACD+; M5 ST+",
                ],
            )
            decision = _close_decision(trade_ids=["555"])
            decision["evidence_refs"].append("position:thesis:555")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED", msg=summary)
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("CLOSE_OPERATOR_AUTH_REQUIRED", codes)
            self.assertNotIn("CLOSE_SAME_DIRECTION_MARKET_SUPPORT", codes)
            self.assertNotIn("CLOSE_THESIS_STILL_VALID", codes)
            recs = payload["input_packet"]["protection_sidecars"]["position_close_recommendations"]
            self.assertEqual(recs[0]["source"], "position_thesis")
            self.assertFalse(recs[0]["gate_b_standing_authorized"])

    def test_close_accepted_without_operator_token_when_position_thesis_invalidation_hit(self) -> None:
        # Position-thesis can still hard-authorize no-ledger loss-cut when the
        # sidecar records a machine-checkable invalidation hit plus multi-TF
        # confirmation.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="DOWN", h4_dir="DOWN")
            _write_fresh_position_thesis_close_recommendation(
                root,
                files,
                rationale_lines=[
                    "patterns +30.0",
                    "forward-proj +25.0",
                    "chart-tech -8.0",
                ],
                context_notes=[
                    "invalidation hit: current ask 1.16310 >= invalidation price 1.16290 plus anti-wick buffer",
                    "technical invalidation confirmed against SHORT: H1 RSI=65.3; M15 trend up; M30 MACD+; M5 ST+",
                ],
            )
            decision = _close_decision(trade_ids=["555"])
            decision["evidence_refs"].append("position:thesis:555")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED", msg=summary)
            self.assertTrue(summary.allowed)
            payload = json.loads((root / "gpt_decision.json").read_text())
            self.assertEqual(payload["verification_issues"], [])
            recs = payload["input_packet"]["protection_sidecars"]["position_close_recommendations"]
            self.assertEqual(recs[0]["source"], "position_thesis")
            self.assertTrue(recs[0]["gate_b_standing_authorized"])

    def test_hard_close_ignores_same_direction_context_asset_matrix_support(self) -> None:
        # Hard invalidation evidence is allowed to close even if the advisory
        # matrix still has stale same-direction support; otherwise a delayed
        # context-asset artifact could block a deterministic invalidation hit.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="LONG", m15_dir="UP", h4_dir="UP")
            _write_same_direction_context_asset_matrix_support(files, side="LONG")
            _write_fresh_position_thesis_close_recommendation(
                root,
                files,
                side="LONG",
                rationale_lines=[
                    "patterns +30.0",
                    "forward-proj +25.0",
                    "chart-tech -8.0",
                ],
                context_notes=[
                    "invalidation hit: current bid 1.16310 <= invalidation price 1.16330 minus anti-wick buffer",
                    "technical invalidation confirmed against LONG: H1 RSI=34.2; M15 trend down; M30 MACD-; M5 ST-",
                ],
            )
            decision = _close_decision(trade_ids=["555"])
            decision["evidence_refs"].extend(
                ["position:thesis:555", "matrix:EUR_USD:LONG", "context_asset:XAU_USD"]
            )
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED", msg=summary)
            payload = json.loads((root / "gpt_decision.json").read_text())
            self.assertEqual(payload["verification_issues"], [])
            recs = payload["input_packet"]["protection_sidecars"]["position_close_recommendations"]
            self.assertTrue(recs[0]["gate_b_standing_authorized"])

    def test_close_accepted_without_operator_token_when_position_management_structural_review_exit(self) -> None:
        # Regression for 471817: a deterministic structural REVIEW_EXIT must not
        # disappear before GPT can verify the CLOSE receipt.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="DOWN", h4_dir="DOWN")
            _write_recent_position_management_close_recommendation(root, files)
            decision = _close_decision(trade_ids=["555"])
            decision["evidence_refs"].append("position:management:555")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED", msg=summary)
            self.assertTrue(summary.allowed)
            payload = json.loads((root / "gpt_decision.json").read_text())
            self.assertEqual(payload["verification_issues"], [])
            recs = payload["input_packet"]["protection_sidecars"]["position_close_recommendations"]
            self.assertEqual(recs[0]["source"], "position_management")
            self.assertTrue(recs[0]["gate_b_standing_authorized"])
            self.assertIn("position:management:555", payload["input_packet"]["allowed_evidence_refs"])

    def test_close_rejected_without_operator_token_when_position_management_entry_invalidation_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="LONG", m15_dir="UP", h4_dir="UP")
            _write_recent_position_management_close_recommendation(
                root,
                files,
                side="LONG",
                reasons=[
                    "score context before entry-invalidation review",
                    (
                        "loss-cut: entry thesis invalidation hit: current bid 1.34392 <= "
                        "buffered invalidation 1.34659 (raw 1.34679, buffer 2.0p); "
                        "technical invalidation confirmed against LONG: H1 BOS_DOWN; H4 BOS_DOWN"
                    ),
                ],
            )
            decision = _close_decision(trade_ids=["555"])
            decision["evidence_refs"].append("position:management:555")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED", msg=summary)
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("CLOSE_SAME_DIRECTION_MARKET_SUPPORT", codes)
            self.assertNotIn("CLOSE_OPERATOR_AUTH_REQUIRED", codes)
            self.assertNotIn("CLOSE_THESIS_STILL_VALID", codes)
            recs = payload["input_packet"]["protection_sidecars"]["position_close_recommendations"]
            self.assertEqual(recs[0]["source"], "position_management")
            self.assertFalse(recs[0]["gate_b_standing_authorized"])

    def test_close_accepted_without_operator_token_when_thesis_evolution_is_broken(self) -> None:
        # thesis_evolution BROKEN / RECOMMEND_CLOSE is generated from entry
        # thesis invalidation plus technical confirmation, so it is hard Gate A.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="DOWN", h4_dir="DOWN")
            _write_fresh_thesis_evolution_close_recommendation(root, files)
            decision = _close_decision(trade_ids=["555"])
            decision["evidence_refs"].append("position:evolution:555")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED", msg=summary)
            payload = json.loads((root / "gpt_decision.json").read_text())
            sidecar = payload["input_packet"]["protection_sidecars"]["position_close_recommendations"][0]
            self.assertEqual(sidecar["source"], "thesis_evolution")
            self.assertTrue(sidecar["gate_b_standing_authorized"])

    def test_hard_sidecar_takes_precedence_over_soft_sidecar_for_same_trade(self) -> None:
        # Live packets can contain position_thesis REVIEW_CLOSE before
        # thesis_evolution RECOMMEND_CLOSE for the same trade. The verifier must
        # not let the earlier soft sidecar hide standing hard authorization.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="DOWN", h4_dir="DOWN")
            _write_fresh_position_thesis_close_recommendation(root, files)
            _write_fresh_thesis_evolution_close_recommendation(root, files)
            decision = _close_decision(trade_ids=["555"])
            decision["evidence_refs"].extend(["position:thesis:555", "position:evolution:555"])
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED", msg=summary)
            payload = json.loads((root / "gpt_decision.json").read_text())
            recs = payload["input_packet"]["protection_sidecars"]["position_close_recommendations"]
            self.assertEqual([item["source"] for item in recs], ["position_thesis", "thesis_evolution"])

    def test_stale_forecast_persistence_close_recommendation_does_not_pass_gate_a(self) -> None:
        _os.environ["QR_OPERATOR_CLOSE_OVERRIDE"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="DOWN", h4_dir="DOWN")
            snapshot = json.loads(files["snapshot"].read_text())
            generated_at = (
                datetime.fromisoformat(snapshot["fetched_at_utc"]) - timedelta(seconds=1)
            ).isoformat()
            (root / "forecast_persistence_report.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": generated_at,
                        "verdicts": [
                            {
                                "trade_id": "555",
                                "pair": "EUR_USD",
                                "side": "SHORT",
                                "verdict": "RECOMMEND_CLOSE",
                                "reason": "old forecast flip from a prior snapshot",
                            }
                        ],
                    }
                )
            )
            decision = _close_decision(trade_ids=["555"])
            decision["evidence_refs"].append("position:persistence:555")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("CLOSE_THESIS_STILL_VALID", codes)
            self.assertEqual(
                payload["input_packet"]["protection_sidecars"]["position_close_recommendations"],
                [],
            )

    def test_18_17_mass_close_regression(self) -> None:
        # Reproduce the 2026-05-11 18:17 UTC GPT close: SHORT positions
        # in EUR_USD/AUD_JPY whose chart_story did NOT show structural
        # counter-events. The model emitted CLOSE; the new gate rejects.
        # Gate B via env override (J hardening 2026-05-13) so the test
        # exercises Gate A in isolation.
        _os.environ["QR_OPERATOR_CLOSE_OVERRIDE"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 2 SHORT positions, no struct counter events anywhere.
            positions = [
                {
                    "trade_id": "470719", "pair": "EUR_USD", "side": "SHORT",
                    "units": 8425, "entry_price": 1.17708,
                    "unrealized_pl_jpy": -1060.9, "take_profit": 1.1706,
                    "stop_loss": None, "owner": "trader",
                },
                {
                    "trade_id": "470749", "pair": "AUD_JPY", "side": "SHORT",
                    "units": 13650, "entry_price": 113.905,
                    "unrealized_pl_jpy": -1173.9, "take_profit": 113.396,
                    "stop_loss": None, "owner": "trader",
                },
            ]
            files = _fixtures(root, positions=positions)
            # Both pairs in pair_charts with thesis-still-valid structure.
            snap = json.loads(files["snapshot"].read_text())
            snap["quotes"] = {
                "EUR_USD": {"bid": 1.17784, "ask": 1.17786, "timestamp_utc": snap["fetched_at_utc"]},
                "AUD_JPY": {"bid": 113.98, "ask": 113.99, "timestamp_utc": snap["fetched_at_utc"]},
            }
            files["snapshot"].write_text(json.dumps(snap))
            pc = json.loads(files["pair_charts"].read_text())
            pc["charts"] = [
                {**pc["charts"][0], "pair": "EUR_USD",
                 "chart_story": _chart_story_with_struct("EUR_USD", m15_dir="DOWN", h4_dir="DOWN")},
                {**pc["charts"][0], "pair": "AUD_JPY",
                 "chart_story": _chart_story_with_struct("AUD_JPY", m15_dir="DOWN", h4_dir="DOWN")},
            ]
            files["pair_charts"].write_text(json.dumps(pc))
            decision = _close_decision(
                trade_ids=["470719", "470749"],
                # No operator_close_authorized — gate B passes via env
                # override above; gate A is what blocks.
            )
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("CLOSE_THESIS_STILL_VALID", codes)


class StructEventParserTest(unittest.TestCase):
    def test_parses_all_seven_timeframes(self) -> None:
        story = (
            "EUR_USD RANGE; "
            "M1(RANGE struct=BOS_UP@1.17); M5(RANGE struct=CHOCH_DOWN@1.18); "
            "M15(RANGE struct=BOS_DOWN@1.19); M30(RANGE struct=BOS_UP@1.20); "
            "H1(RANGE struct=CHOCH_UP@1.21); H4(RANGE struct=BOS_DOWN@1.22); "
            "D(RANGE struct=BOS_UP@1.23)"
        )
        events = _parse_struct_events(story)
        # Default (no `:wick` suffix) means close-confirmed.
        self.assertEqual(events["M15"], ("BOS", "DOWN", 1.19, True))
        self.assertEqual(events["H4"], ("BOS", "DOWN", 1.22, True))
        self.assertEqual(len(events), 7)

    def test_parses_wick_suffix_as_not_close_confirmed(self) -> None:
        story = (
            "AUD_JPY RANGE; "
            "M15(RANGE struct=BOS_UP@114.1460:wick); "
            "H4(RANGE struct=BOS_UP@113.5870)"
        )
        events = _parse_struct_events(story)
        self.assertEqual(events["M15"], ("BOS", "UP", 114.146, False))
        self.assertEqual(events["H4"], ("BOS", "UP", 113.587, True))

    def test_invalidated_when_m15_against_long(self) -> None:
        packet = {
            "market_context": {
                "pairs": {
                    "EUR_USD": {
                        "chart": {
                            "chart_story": (
                                "EUR_USD RANGE; M15(RANGE struct=BOS_DOWN@1.19); "
                                "H4(RANGE struct=BOS_UP@1.20)"
                            ),
                        }
                    }
                }
            }
        }
        ok, reason = _close_thesis_invalidated(packet, "EUR_USD", "LONG")
        self.assertTrue(ok)
        self.assertIn("M15", reason)
        self.assertIn("close-confirmed", reason)

    def test_not_invalidated_when_struct_aligned_with_side(self) -> None:
        packet = {
            "market_context": {
                "pairs": {
                    "EUR_USD": {
                        "chart": {
                            "chart_story": (
                                "EUR_USD RANGE; M15(RANGE struct=BOS_UP@1.19); "
                                "H4(RANGE struct=BOS_UP@1.20)"
                            ),
                        }
                    }
                }
            }
        }
        ok, _ = _close_thesis_invalidated(packet, "EUR_USD", "LONG")
        self.assertFalse(ok)

    def test_not_invalidated_when_only_event_is_wick_only_break(self) -> None:
        # Stop-hunt regression (2026-05-13). The wick of a new swing pivot
        # taps the prior pivot but the candle closes back inside the
        # range. Gate A must NOT fire on this signal alone.
        packet = {
            "market_context": {
                "pairs": {
                    "AUD_JPY": {
                        "chart": {
                            "chart_story": (
                                "AUD_JPY RANGE; "
                                "M15(RANGE struct=BOS_UP@114.1460:wick); "
                                "H4(RANGE struct=BOS_UP@113.5870:wick)"
                            ),
                        }
                    }
                }
            }
        }
        ok, _ = _close_thesis_invalidated(packet, "AUD_JPY", "SHORT")
        self.assertFalse(ok)

    def test_h4_close_confirmed_still_fires_when_m15_is_wick_only(self) -> None:
        # The 2026-05-13 AUD_JPY scenario: M15 BOS_UP@114.146 was a
        # 0.4-pip wick break (no close confirmation), but H4
        # BOS_UP@113.587 was a 46-pip clean structural break. Gate A
        # should fire on the H4 close-confirmed event and ignore the
        # M15 wick.
        packet = {
            "market_context": {
                "pairs": {
                    "AUD_JPY": {
                        "chart": {
                            "chart_story": (
                                "AUD_JPY RANGE; "
                                "M15(RANGE struct=BOS_UP@114.1460:wick); "
                                "H4(RANGE struct=BOS_UP@113.5870)"
                            ),
                        }
                    }
                }
            }
        }
        ok, reason = _close_thesis_invalidated(packet, "AUD_JPY", "SHORT")
        self.assertTrue(ok)
        self.assertIn("H4", reason)
        self.assertIn("close-confirmed", reason)


class OperatorCloseTokenFreshnessTest(unittest.TestCase):
    """Coverage for the J (2026-05-13) Gate B hardening: the receipt's
    `operator_close_authorized` JSON field is no longer accepted on its
    own. Authorization must come from either `QR_OPERATOR_CLOSE_OVERRIDE`
    in the operator shell or a fresh `data/.operator_close_token` file.
    """

    def test_missing_token_returns_false(self) -> None:
        from quant_rabbit.gpt_trader import _operator_close_token_fresh
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            root.mkdir()
            self.assertFalse(_operator_close_token_fresh(data_root=root))

    def test_fresh_token_returns_true(self) -> None:
        from quant_rabbit.gpt_trader import _operator_close_token_fresh
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            root.mkdir()
            (root / ".operator_close_token").write_text("ok")
            self.assertTrue(_operator_close_token_fresh(data_root=root))

    def test_stale_token_returns_false(self) -> None:
        import os as _os_mod
        from quant_rabbit.gpt_trader import (
            _operator_close_token_fresh,
            OPERATOR_CLOSE_TOKEN_FRESH_SECONDS,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            root.mkdir()
            token = root / ".operator_close_token"
            token.write_text("stale")
            # Push mtime past the freshness window.
            old = datetime.now(timezone.utc).timestamp() - (OPERATOR_CLOSE_TOKEN_FRESH_SECONDS + 60)
            _os_mod.utime(token, (old, old))
            self.assertFalse(_operator_close_token_fresh(data_root=root))


class NoiseResistantSLGeometryTest(unittest.TestCase):
    """Coverage for the F (2026-05-13) noise-resistant SL geometry:
    new-entry SL is floored at `H4_ATR * NEW_ENTRY_SL_H4_ATR_MULT`,
    widened by session multiplier for thin/off-hours liquidity. Activated
    via `QR_NEW_ENTRY_INITIAL_SL=1` (which the SL-free bootstrap sets
    automatically — see cli._SL_FREE_RUNTIME_DEFAULTS).
    """

    def test_session_widening_mult_off_hours(self) -> None:
        from quant_rabbit.strategy.intent_generator import (
            _session_widening_mult,
            NEW_ENTRY_SL_OFF_HOURS_MULT,
        )
        self.assertAlmostEqual(_session_widening_mult("OFF_HOURS"), NEW_ENTRY_SL_OFF_HOURS_MULT)

    def test_session_widening_mult_tokyo_thin(self) -> None:
        from quant_rabbit.strategy.intent_generator import (
            _session_widening_mult,
            NEW_ENTRY_SL_THIN_SESSION_MULT,
        )
        # TOKYO_KILLZONE is treated as thin.
        self.assertAlmostEqual(
            _session_widening_mult("TOKYO_KILLZONE"),
            NEW_ENTRY_SL_THIN_SESSION_MULT,
        )

    def test_session_widening_mult_deep_liquidity_no_widen(self) -> None:
        from quant_rabbit.strategy.intent_generator import _session_widening_mult
        # London/NY overlap is deep — no widening.
        self.assertAlmostEqual(_session_widening_mult("LONDON_NY_OVERLAP"), 1.0)

    def test_session_widening_mult_asia_alias_is_thin(self) -> None:
        from quant_rabbit.strategy.intent_generator import (
            _session_widening_mult,
            NEW_ENTRY_SL_THIN_SESSION_MULT,
        )
        self.assertAlmostEqual(_session_widening_mult("ASIA"), NEW_ENTRY_SL_THIN_SESSION_MULT)

    def test_session_widening_mult_unknown_tag_no_widen(self) -> None:
        from quant_rabbit.strategy.intent_generator import _session_widening_mult
        self.assertAlmostEqual(_session_widening_mult(None), 1.0)
        self.assertAlmostEqual(_session_widening_mult("UNKNOWN_TAG"), 1.0)

    def test_new_entry_initial_sl_active_respects_env(self) -> None:
        import os as _os_mod
        from quant_rabbit.strategy.intent_generator import _new_entry_initial_sl_active
        prior = _os_mod.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
        try:
            _os_mod.environ["QR_NEW_ENTRY_INITIAL_SL"] = "1"
            self.assertTrue(_new_entry_initial_sl_active())
            _os_mod.environ["QR_NEW_ENTRY_INITIAL_SL"] = "0"
            self.assertFalse(_new_entry_initial_sl_active())
        finally:
            if prior is None:
                _os_mod.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
            else:
                _os_mod.environ["QR_NEW_ENTRY_INITIAL_SL"] = prior


class SessionAwareSpreadCapTest(unittest.TestCase):
    """Coverage for the I (2026-05-13) session-aware spread tolerance:
    `RiskPolicy.max_spread_multiple` is multiplied by a session-tag
    factor before the spread check. Deep sessions (London/NY overlap)
    tighten; thin sessions (Tokyo, off-hours, JP holiday) loosen.
    """

    def test_off_hours_loosens_spread_cap(self) -> None:
        from quant_rabbit.risk import _SPREAD_SESSION_MULTS
        self.assertGreater(_SPREAD_SESSION_MULTS["OFF_HOURS"], 1.0)
        self.assertGreater(_SPREAD_SESSION_MULTS["JP_HOLIDAY"], 1.0)

    def test_london_ny_overlap_tightens_spread_cap(self) -> None:
        from quant_rabbit.risk import _SPREAD_SESSION_MULTS
        self.assertLess(_SPREAD_SESSION_MULTS["LONDON_NY_OVERLAP"], 1.0)

    def test_tokyo_loosens_spread_cap(self) -> None:
        from quant_rabbit.risk import _SPREAD_SESSION_MULTS
        self.assertGreater(_SPREAD_SESSION_MULTS["TOKYO_KILLZONE"], 1.0)

    def test_spread_session_multiplier_reads_session_current_tag(self) -> None:
        from quant_rabbit.risk import _spread_session_multiplier, _SPREAD_SESSION_MULTS

        class _Stub:
            metadata = {"session_current_tag": "OFF_HOURS"}

        self.assertAlmostEqual(
            _spread_session_multiplier(_Stub()),
            _SPREAD_SESSION_MULTS["OFF_HOURS"],
        )

    def test_spread_session_multiplier_falls_back_to_session_bucket(self) -> None:
        from quant_rabbit.risk import _spread_session_multiplier, _SPREAD_SESSION_MULTS

        class _Stub:
            metadata = {"session_bucket": "TOKYO_KILLZONE"}

        self.assertAlmostEqual(
            _spread_session_multiplier(_Stub()),
            _SPREAD_SESSION_MULTS["TOKYO_KILLZONE"],
        )

    def test_spread_session_multiplier_default_when_missing(self) -> None:
        from quant_rabbit.risk import _spread_session_multiplier

        class _Stub:
            metadata = {}

        self.assertAlmostEqual(_spread_session_multiplier(_Stub()), 1.0)


if __name__ == "__main__":
    unittest.main()
