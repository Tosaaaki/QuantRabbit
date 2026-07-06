from __future__ import annotations

import unittest
from unittest.mock import patch

from tools import build_as_4x_proof_path as proof_path
from tools import build_as_live_ready_evidence_loop as evidence_loop


class As4xProofPathTests(unittest.TestCase):
    def test_as_board_overwrites_stale_order_intent_timestamp_and_total_lanes(self) -> None:
        firepower = evidence_loop._build_firepower_board(
            generated_at="2026-07-06T15:04:28Z",
            order_intents={
                "generated_at_utc": "2026-07-06T15:02:44.070184+00:00",
                "results": [],
            },
            daily={
                "funding_adjusted_equity": 183428.3742,
                "required_calendar_daily_return_funding_adjusted": 5.898554,
            },
            broker={},
            blocked={
                "market_close_trade_ids": set(),
                "residual_trade_ids": set(),
                "residual_keys": set(),
                "market_close_family_key": ("EUR_USD", "LONG", "BREAKOUT_FAILURE"),
            },
            p0_decomposition={"rows": []},
        )

        board = evidence_loop._update_as_board(
            generated_at="2026-07-06T15:04:28Z",
            board={
                "order_intents_generated_at_utc": "2026-07-05T17:17:54.720362+00:00",
                "total_lanes": 73,
            },
            daily={},
            p0_decomposition={"rows": [], "dependency_graph": []},
            firepower=firepower,
            proof_queue={"queue": []},
        )

        self.assertEqual(board["order_intents_generated_at_utc"], "2026-07-06T15:02:44.070184+00:00")
        self.assertEqual(board["total_lanes"], 0)

    def test_historical_or_negative_bidask_replay_is_not_live_grade_proof(self) -> None:
        missing = evidence_loop._missing_proof_map(
            {
                "source_evidence": {
                    "historical_only": True,
                    "bidask_rule_status": "LIVE_BLOCK_NEGATIVE_EXPECTANCY",
                },
                "exact_proof_gaps": [],
                "current_blockers": [],
                "status": "DRY_RUN_BLOCKED",
                "risk_allowed": False,
            }
        )

        self.assertFalse(missing["fresh_744h_replay"])
        self.assertFalse(missing["s5_bidask_spread_included_replay"])

    def test_fresh_direction_evidence_marks_under_sampled_gap(self) -> None:
        replay = {
            "price_truth_coverage": {
                "status": "PRICE_TRUTH_OK",
                "adoption_level": "PAIR_LOCAL_RANK_ONLY",
            },
            "forecast_sample_coverage": {
                "pairs": [{"pair": "GBP_USD", "evaluated_samples": 13}],
                "under_sampled_pair_directions": [
                    {
                        "pair": "GBP_USD",
                        "direction": "UP",
                        "evaluated_samples": 2,
                        "evaluated_active_days": 2,
                    }
                ],
            },
            "precision_rules": {
                "edge_rules": [],
                "daily_stable_edge_rules": [],
                "negative_rules": [],
                "contrarian_edge_rules": [],
                "rejected_daily_stability_segments": [],
            },
            "segments": {"by_pair_direction": []},
        }

        evidence = proof_path.fresh_direction_evidence(replay, "GBP_USD", "UP")

        self.assertEqual(evidence["status"], "EVIDENCE_GAP_UNDER_SAMPLED")
        self.assertFalse(evidence["live_grade_support"])
        self.assertFalse(evidence["can_create_live_permission"])

    def test_portfolio_planner_excludes_negative_bidask_from_mathematical_basket(self) -> None:
        negative_bidask = {
            "lane_id": "trend_trader:USD_CHF:LONG:TREND_CONTINUATION",
            "pair": "USD_CHF",
            "side": "LONG",
            "method": "TREND_CONTINUATION",
            "order_type": "STOP-ENTRY",
            "source_evidence": {
                "historical_only": True,
                "bidask_rule_status": "LIVE_BLOCK_NEGATIVE_EXPECTANCY",
            },
            "expected_daily_return_pct_on_funding_adjusted_equity": 34.3,
            "expected_jpy_per_trade": 2263.4,
            "estimated_trades_per_day_available": 27.3,
            "realistic_units": 1000,
            "margin_requirement_realistic_size_jpy": 6500.0,
            "risk_allowed": False,
            "proof_gap_count": 7,
            "current_blockers": [
                "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
                "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
            ],
        }
        repair_candidate = {
            "lane_id": "range_trader:CAD_JPY:SHORT:RANGE_ROTATION",
            "pair": "CAD_JPY",
            "side": "SHORT",
            "method": "RANGE_ROTATION",
            "order_type": "LIMIT",
            "source_evidence": {"historical_only": False},
            "expected_daily_return_pct_on_funding_adjusted_equity": 6.2,
            "expected_jpy_per_trade": 420.0,
            "estimated_trades_per_day_available": 26.7,
            "realistic_units": 3000,
            "margin_requirement_realistic_size_jpy": 13700.0,
            "risk_allowed": False,
            "proof_gap_count": 6,
            "current_blockers": [
                "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
            ],
        }
        ctx = {
            "proof_by_lane": {},
            "firepower": {
                "target_math": {
                    "required_calendar_daily_return_funding_adjusted_pct": 5.0,
                }
            },
            "acceptance": {},
            "support": {},
            "memory": {},
            "broker": {},
        }

        with patch.object(proof_path, "all_firepower_candidates", return_value=[negative_bidask, repair_candidate]):
            payload = proof_path.build_portfolio_planner("2026-07-06T00:00:00Z", ctx)

        self.assertEqual(payload["summary"]["non_hard_excluded_candidates"], 2)
        self.assertEqual(payload["summary"]["math_candidate_eligible_candidates"], 1)
        self.assertEqual(payload["summary"]["planner_rejected_candidates"], 1)
        self.assertEqual(payload["standalone_math_candidates"][0]["lane_id"], repair_candidate["lane_id"])
        self.assertEqual(payload["fastest_mathematical_basket"]["components"][0]["lane_id"], repair_candidate["lane_id"])
        rejected = next(row for row in payload["candidate_rankings"] if row["lane_id"] == negative_bidask["lane_id"])
        self.assertEqual(rejected["proof_classification"], "REJECTED")
        self.assertFalse(rejected["math_candidate_eligible"])
        self.assertIn("spread_included_bidask_replay_negative_for_exact_lane", rejected["math_exclusion_reasons"])
        self.assertIn("packaged_bidask_rule_live_block_negative_expectancy", rejected["math_exclusion_reasons"])

    def test_gbpusd_breakout_failure_market_close_is_blocked_family_repair(self) -> None:
        rows = [
            evidence_loop.Outcome(
                ts_utc="2026-07-01T00:00:00Z",
                trade_id="TEST_GBP_CLOSE_1",
                pair="GBP_USD",
                side="LONG",
                lane_id="failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE",
                method="BREAKOUT_FAILURE",
                exit_reason="MARKET_ORDER_TRADE_CLOSE",
                realized_pl_jpy=-1000.0,
            ),
            evidence_loop.Outcome(
                ts_utc="2026-07-01T00:05:00Z",
                trade_id="TEST_GBP_CLOSE_2",
                pair="GBP_USD",
                side="LONG",
                lane_id="failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE",
                method="BREAKOUT_FAILURE",
                exit_reason="MARKET_ORDER_TRADE_CLOSE",
                realized_pl_jpy=-500.0,
            ),
        ]

        table = evidence_loop._build_post_gate_gap_family_repair_table(
            generated_at="2026-07-06T00:00:00Z",
            attributed=rows,
            blocked={
                "market_close_trade_ids": set(),
                "residual_trade_ids": set(),
                "residual_keys": set(),
                "market_close_family_key": ("EUR_USD", "LONG", "BREAKOUT_FAILURE"),
            },
        )

        self.assertEqual(table["summary"]["largest_remaining_adverse_family"], "GBP_USD|LONG|BREAKOUT_FAILURE|MARKET_ORDER_TRADE_CLOSE")
        self.assertEqual(table["rows"][0]["action"], "MARKET_CLOSE_PATH_BLOCK")
        self.assertFalse(table["rows"][0]["market_close_path_allowed"])
        self.assertFalse(table["rows"][0]["can_create_live_permission"])
        self.assertEqual(table["containment_filter_trade_ids"], ["TEST_GBP_CLOSE_1", "TEST_GBP_CLOSE_2"])

    def test_manual_position_safety_reports_replaced_expected_tp_chain(self) -> None:
        broker = {
            "positions": [
                {
                    "trade_id": "472987",
                    "owner": "operator_manual",
                    "pair": "EUR_USD",
                    "side": "SHORT",
                    "units": 30000,
                    "take_profit": 1.13968,
                    "raw": {
                        "takeProfitOrder": {
                            "id": "472996",
                            "price": "1.13968",
                            "replacesOrderID": "472994",
                            "createTime": "2026-07-06T02:07:40Z",
                        }
                    },
                    "operator_manual_position": {
                        "management_intent": "KEEP",
                        "system_pl_counted": False,
                        "system_occupancy_counted": False,
                    },
                }
            ],
            "orders": [
                {
                    "order_id": "472996",
                    "trade_id": "472987",
                    "price": 1.13968,
                    "state": "PENDING",
                }
            ],
        }
        ledger_events = [
            {
                "raw_type": "TAKE_PROFIT_ORDER",
                "event_type": "PROTECTION_CREATED",
                "source": "oanda",
                "order_id": "472988",
                "trade_id": "472987",
                "price": "1.13880",
                "time": "2026-07-02T10:30:48Z",
                "reason": "CLIENT_ORDER",
            },
            {
                "raw_type": "ORDER_CANCEL",
                "event_type": "ORDER_CANCELED",
                "source": "oanda",
                "order_id": "472988",
                "replaced_by_order_id": "472994",
                "time": "2026-07-06T01:27:42Z",
                "reason": "CLIENT_REQUEST_REPLACED",
            },
            {
                "raw_type": "TAKE_PROFIT_ORDER",
                "event_type": "PROTECTION_CREATED",
                "source": "oanda",
                "order_id": "472994",
                "trade_id": "472987",
                "price": "1.13600",
                "time": "2026-07-06T01:27:42Z",
                "reason": "REPLACEMENT",
                "replaces_order_id": "472988",
            },
            {
                "raw_type": "ORDER_CANCEL",
                "event_type": "ORDER_CANCELED",
                "source": "oanda",
                "order_id": "472994",
                "replaced_by_order_id": "472996",
                "time": "2026-07-06T02:07:40Z",
                "reason": "CLIENT_REQUEST_REPLACED",
            },
            {
                "raw_type": "TAKE_PROFIT_ORDER",
                "event_type": "PROTECTION_CREATED",
                "source": "oanda",
                "order_id": "472996",
                "trade_id": "472987",
                "price": "1.13968",
                "time": "2026-07-06T02:07:40Z",
                "reason": "REPLACEMENT",
                "replaces_order_id": "472994",
            },
        ]

        safety = proof_path.manual_position_safety(broker, ledger_events=ledger_events, gateway_receipts={})

        self.assertEqual(safety["expected_manual_tp_order_id"], "472988")
        self.assertEqual(safety["audit_manual_tp_order_id"], "472994")
        self.assertEqual(safety["current_manual_tp_order_id"], "472996")
        self.assertFalse(safety["expected_tp_order_present"])
        self.assertTrue(safety["expected_tp_replaced_in_broker_truth"])
        self.assertEqual(safety["audit_tp_lifecycle"], "REPLACED")
        self.assertEqual(safety["audit_tp_replaced_by_order_id"], "472996")
        self.assertEqual(safety["audit_tp_provenance_classification"], "PROVENANCE_UNKNOWN_BLOCK_AUTOMATION")
        self.assertEqual(safety["current_tp_provenance_classification"], "PROVENANCE_UNKNOWN_BLOCK_AUTOMATION")
        self.assertTrue(safety["automation_blocked"])
        self.assertFalse(safety["auto_tp_modify_allowed"])
        self.assertTrue(safety["untouched_by_this_run"])

    def test_audjpy_limit_proof_pack_uses_repair_required_classification_only(self) -> None:
        lane_id = "failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT"
        ctx = {
            "candidate_by_lane": {
                lane_id: {
                    "current_blockers": [
                        "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
                        "SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH",
                        "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
                        "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
                    ],
                    "expected_jpy_per_trade": 992.7,
                    "estimated_trades_per_day_available": 6.7,
                    "expected_daily_return_pct_on_funding_adjusted_equity": 3.7972,
                    "realistic_units": 1000,
                    "margin_requirement_realistic_size_jpy": 3000,
                    "margin_requirement_min_lot_jpy": 3000,
                    "broker_margin_context": {},
                    "risk_allowed": False,
                    "order_type": "LIMIT",
                    "sample_count": 1361,
                    "active_days": 39,
                    "source_evidence": {"historical_only": False},
                }
            },
            "proof_by_lane": {
                lane_id: {
                    "missing_proof": {
                        "fresh_744h_replay": True,
                        "s5_bidask_spread_included_replay": False,
                        "sample_count_floor": True,
                        "active_day_floor": True,
                        "daily_stability_floor": True,
                        "positive_day_rate_floor": True,
                        "forecast_executable_proof": True,
                        "geometry_proof": True,
                        "attached_tp_proof": True,
                        "market_close_absence_or_close_gate_proof": True,
                        "residual_family_absence": True,
                        "risk_engine_pass": False,
                        "live_order_gateway_pass": False,
                        "gpt_verifier_pass": False,
                        "no_stale_packaged_rule": True,
                        "no_guardian_operator_review_blocker": False,
                    }
                }
            },
            "intent_by_lane": {
                lane_id: {
                    "risk_metrics": {
                        "reward_pips": 33.7,
                        "loss_pips": 12.5,
                        "reward_risk": 2.696,
                        "spread_pips": 1.2,
                        "risk_jpy": 125.0,
                    },
                    "intent": {
                        "entry": 112.169,
                        "tp": 111.832,
                        "sl": 112.294,
                        "metadata": {
                            "forecast_direction": "RANGE",
                            "forecast_confidence": 0.5867,
                            "forecast_market_support_ok": False,
                            "attach_take_profit_on_fill": True,
                            "capture_take_profit_scope_key": "AUD_JPY|SHORT|BREAKOUT_FAILURE|TAKE_PROFIT_ORDER",
                            "capture_take_profit_trades": 6,
                            "capture_take_profit_wins": 6,
                            "capture_take_profit_expectancy_jpy": 992.7,
                            "positive_rotation_mode": "TP_PROOF_COLLECTION_HARVEST",
                            "positive_rotation_proof_collection_ready": True,
                        },
                    },
                }
            },
            "fresh_replay": {
                "forecast_sample_coverage": {
                    "min_directional_samples_for_precision_rule": 30,
                    "min_active_days_for_daily_stability": 3,
                    "under_sampled_pair_directions": [
                        {
                            "pair": "AUD_JPY",
                            "direction": "DOWN",
                            "evaluated_samples": 9,
                            "evaluated_active_days": 5,
                        }
                    ],
                },
                "price_truth_coverage": {},
                "precision_rules": {
                    "edge_rules": [],
                    "daily_stable_edge_rules": [],
                    "negative_rules": [],
                    "contrarian_edge_rules": [
                        {
                            "pair": "AUD_JPY",
                            "direction": "DOWN",
                            "daily_stability_status": "INSUFFICIENT_ACTIVE_DAYS",
                        }
                    ],
                    "rejected_daily_stability_segments": [],
                },
                "segments": {"by_pair_direction": []},
            },
            "fresh_replay_relpath": "logs/reports/forecast_improvement/oanda_history_replay_validate_latest.json",
            "timing": {},
            "firepower": {"target_math": {"required_calendar_daily_return_funding_adjusted_pct": 5.4}},
        }

        payload = proof_path.build_audjpy_limit_proof_pack("2026-07-06T00:00:00Z", ctx)

        self.assertEqual(payload["classification_values"], ["PROOF_READY", "REPAIR_REQUIRED", "EVIDENCE_GAP", "REJECTED"])
        self.assertEqual(payload["classification"], "REPAIR_REQUIRED")
        self.assertNotIn("PORTFOLIO_COMPONENT_REPAIR_REQUIRED", payload["classification_values"])
        self.assertTrue(payload["portfolio_component_possible_after_repair"])
        matrix = {row["proof"]: row["status"] for row in payload["required_proof_matrix"]}
        self.assertEqual(matrix["S5 samples"], "MISSING")
        self.assertEqual(matrix["active days"], "MISSING")
        self.assertEqual(matrix["forecast executable proof"], "MISSING")
        self.assertEqual(matrix["RiskEngine"], "MISSING")
        self.assertEqual(matrix["Gateway"], "MISSING")
        self.assertEqual(matrix["GPT verifier"], "MISSING")
        self.assertEqual(matrix["guardian/operator review"], "MISSING")
        self.assertEqual(matrix["geometry proof"], "PRESENT_BUT_NOT_PERMISSION")
        self.assertEqual(matrix["attached TP proof"], "PRESENT_BUT_NOT_PERMISSION")

    def test_profitability_reconciliation_keeps_routing_blocked(self) -> None:
        broker_mutation_audit = {
            "conclusion": {
                "manual_trade_472987_untouched": True,
                "position_manager_gateway_bypass_fixed": True,
                "tp_rebalance_incident_contained": True,
            },
            "broker_truth": {
                "dev": {
                    "last_transaction_id": "472996",
                    "orders": [
                        {
                            "order_id": "472996",
                            "order_type": "TAKE_PROFIT",
                            "state": "PENDING",
                            "trade_id": "472987",
                        }
                    ],
                },
                "live": {
                    "last_transaction_id": "472996",
                    "orders": [
                        {
                            "order_id": "472996",
                            "order_type": "TAKE_PROFIT",
                            "state": "PENDING",
                            "trade_id": "472987",
                        }
                    ],
                },
            },
        }
        ctx = {
            "acceptance": {
                "status": "PROFITABILITY_ACCEPTANCE_BLOCKED",
                "blockers": [
                    "SELF_IMPROVEMENT_P0_PRESENT: self-improvement audit still has 1 P0 finding(s)",
                    "NEGATIVE_EXPECTANCY_ACTIVE: capture economics is still NEGATIVE_EXPECTANCY",
                    "MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE: leakage",
                    "MARKET_CLOSE_LEAK_FAMILY_BLOCKED: family",
                    "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE: replay",
                ],
                "findings": [
                    {"code": "SELF_IMPROVEMENT_P0_PRESENT", "message": "p0", "priority": "P0", "evidence": {}},
                    {"code": "NEGATIVE_EXPECTANCY_ACTIVE", "message": "negative", "priority": "P0", "evidence": {}},
                    {"code": "MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE", "message": "leak", "priority": "P0", "evidence": {}},
                    {"code": "MARKET_CLOSE_LEAK_FAMILY_BLOCKED", "message": "family", "priority": "P0", "evidence": {}},
                    {"code": "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE", "message": "replay", "priority": "P0", "evidence": {}},
                    {"code": "NO_LIVE_READY_TARGET_COVERAGE", "message": "none", "priority": "P1", "evidence": {}},
                ],
            },
            "self_audit": {
                "status": "SELF_IMPROVEMENT_BLOCKED",
                "findings": [{"code": "MEMORY_HEALTH_BLOCKED", "priority": "P0", "message": "memory blocked"}],
            },
            "memory": {
                "status": "MEMORY_HEALTH_BLOCKED",
                "blockers": ["forecast_history latest row predates broker snapshot"],
                "metrics": {
                    "execution_ledger": {
                        "last_oanda_transaction_id": "472996",
                        "snapshot_last_transaction_id": "472996",
                    }
                },
            },
        }

        with patch.object(proof_path, "_load_json", return_value=broker_mutation_audit):
            payload = proof_path.build_profitability_acceptance_blocker_reconciliation("2026-07-06T00:00:00Z", ctx)
        rows = {row["code"]: row for row in payload["rows"]}

        self.assertEqual(payload["status"], "PROFITABILITY_ACCEPTANCE_BLOCKED")
        self.assertFalse(payload["summary"]["routing_allowed"])
        self.assertFalse(payload["summary"]["as_live_ready_path_exists"])
        self.assertFalse(payload["summary"]["can_create_live_permission"])
        self.assertEqual(rows["SELF_IMPROVEMENT_P0_PRESENT"]["classification"], "ACTIVE_BLOCKER")
        self.assertEqual(rows["MARKET_CLOSE_LEAK_FAMILY_BLOCKED"]["classification"], "TAXONOMY_DUPLICATE")
        self.assertTrue(rows["MARKET_CLOSE_LEAK_FAMILY_BLOCKED"]["still_blocks_fresh_entries"])
        self.assertEqual(rows["EXECUTION_LEDGER_STALE"]["classification"], "STALE_SUPERSEDED")
        manual_tp_summary = rows["OPERATOR_MANUAL_TP_OPT_OUT_BYPASS"]["evidence_summary"]
        self.assertEqual(manual_tp_summary["last_transaction_id"], "472996")
        self.assertEqual(manual_tp_summary["active_take_profit_order"], "472996")


if __name__ == "__main__":
    unittest.main()
