from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from copy import deepcopy
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import quant_rabbit.broker.execution as execution_module
import quant_rabbit.decision_execution_lineage as lineage_module
from quant_rabbit.broker.execution import LiveOrderGateway
from quant_rabbit.execution_ledger import ExecutionLedger
from quant_rabbit.forecast_precision import (
    bidask_replay_precision_rule_digest,
    canonical_bidask_replay_precision_rule,
)
from quant_rabbit.market_close_leak_gate import MARKET_CLOSE_LEAK_FAMILY_BLOCK_CODE
from quant_rabbit.models import AccountSummary, BrokerOrder, BrokerPosition, BrokerSnapshot, MarketContext, OrderIntent, OrderType, Owner, Quote, Side, TradeMethod
from quant_rabbit.predictive_scout import predictive_scout_sizing_digest
from quant_rabbit.risk import OANDA_JP_RETAIL_FX_MARGIN_RATE


TEST_PER_TRADE_RISK_JPY = 10_000.0
TEST_DAILY_RISK_BUDGET_JPY = 50_000.0
PREDICTIVE_SCOUT_LANE_ID = "predictive_scout:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT"
TP_PROVEN_RANGE_LANE_ID = "range_trader:EUR_USD:LONG:RANGE_ROTATION:LIMIT"
PREDICTIVE_SCOUT_RULE_NAME = (
    "USD_CAD_DOWN_H31_60m_C0p50_0p65_FADE_TO_UP_"
    "S5_BIDASK_CONTRARIAN_HARVEST_TP10_SL7"
)


def _synthetic_execution_cost_floor(
    exact_key: tuple[str, str, str, str] = (
        "EUR_USD",
        "LONG",
        "TREND_CONTINUATION",
        "MARKET",
    ),
) -> dict[str, Any]:
    material = {
        "contract": execution_module.EXECUTION_COST_FLOOR_CONTRACT,
        "status": "PASSED",
        "reason": "TEST_DYNAMIC_LEDGER_EXECUTION_COST_FLOOR",
        "failed_checks": [],
        "scope_key": "|".join(exact_key),
        "market_entry_adverse_p95_pips": 0.2,
        "audited_protected_exit_adverse_p95_pips": 0.4,
        "financing_adverse_stress_jpy_per_unit": 0.0001,
        "spread_double_count_forbidden": True,
    }
    return {
        **material,
        "proof_sha256": execution_module._canonical_json_sha256(material),
    }


def _bypass_pre_post_reconciliation(self, **kwargs):
    return execution_module._PrePostReconciliationResult(
        intent=kwargs["intent"],
        snapshot=kwargs["snapshot"],
        risk=kwargs["risk"],
        order_request=kwargs["order_request"],
        attached_stop_metrics=kwargs["attached_stop_metrics"],
        max_loss_jpy=kwargs["max_loss_jpy"],
        portfolio_loss_cap_jpy=kwargs["portfolio_loss_cap_jpy"],
        size_multiple=kwargs["size_multiple"],
        order_build_issues=kwargs["order_build_issues"],
        issues=(),
        evidence={"status": "TEST_BYPASS"},
    )


def _bypass_final_pre_post_boundary(self, **kwargs):
    issues = []
    receipt_issue = execution_module._verified_decision_receipt_change_issue(
        self.verified_decision_path,
        expected_sha256=kwargs["expected_receipt_sha256"],
    )
    if receipt_issue is not None:
        issues.append(
            execution_module.RiskIssue(
                receipt_issue["code"],
                receipt_issue["message"],
                receipt_issue.get("severity", "BLOCK"),
            )
        )
    current_order_sha = execution_module._canonical_order_request_sha256(
        kwargs["order_request"]
    )
    if current_order_sha != kwargs["expected_order_request_sha256"]:
        issues.append(
            execution_module.RiskIssue(
                "FINAL_PRE_POST_ORDER_REQUEST_CHANGED_AFTER_RESERVATION",
                "test boundary observed a changed broker order request",
            )
        )
    return execution_module._FinalPrePostBoundaryResult(
        evidence={
            "status": "BLOCKED" if issues else "TEST_BYPASS",
            "post_attempted": False,
        },
        issues=tuple(issues),
    )


class LiveOrderGatewayTest(unittest.TestCase):
    def setUp(self) -> None:
        self._guardian_tmp = tempfile.TemporaryDirectory()
        self._original_per_trade_reader = execution_module._per_trade_risk_from_state
        self._original_daily_budget_reader = execution_module._daily_risk_budget_from_state
        self._original_target_trades_reader = execution_module._target_trades_per_day_from_state
        self._original_pre_post_reconcile = execution_module.LiveOrderGateway._pre_post_reconcile
        self._original_final_pre_post_boundary = (
            execution_module.LiveOrderGateway._final_pre_post_boundary
        )
        self._original_gpt_live_send_issues = (
            execution_module._gpt_verified_decision_live_send_issues
        )
        self._enforce_missing_gpt_receipt = False

        def preserve_legacy_gate_test_focus(
            verified_decision_path: Path | None,
            **kwargs: Any,
        ) -> list[dict[str, str]]:
            # Most tests in this long-lived suite isolate a different broker
            # boundary.  Keep their historical no-receipt fixture focused on
            # that boundary, while SCOUT and the explicit receipt regressions
            # exercise the production gate below.  This is a test mock, not a
            # production bypass flag.
            if (
                verified_decision_path is None
                and not kwargs.get("predictive_scout_trade_only")
                and not self._enforce_missing_gpt_receipt
            ):
                return []
            return self._original_gpt_live_send_issues(
                verified_decision_path,
                **kwargs,
            )

        self._gpt_live_send_issues_patch = patch.object(
            execution_module,
            "_gpt_verified_decision_live_send_issues",
            side_effect=preserve_legacy_gate_test_focus,
        )
        self._gpt_live_send_issues_patch.start()
        self._pre_post_reconcile_patch = patch.object(
            execution_module.LiveOrderGateway,
            "_pre_post_reconcile",
            _bypass_pre_post_reconciliation,
        )
        self._pre_post_reconcile_patch.start()
        self._final_pre_post_boundary_patch = patch.object(
            execution_module.LiveOrderGateway,
            "_final_pre_post_boundary",
            _bypass_final_pre_post_boundary,
        )
        self._final_pre_post_boundary_patch.start()
        self._execution_cost_floor_patch = patch.object(
            execution_module,
            "execution_cost_floor_from_surface",
            side_effect=lambda _surface, *, exact_key, as_of: (
                _synthetic_execution_cost_floor(exact_key)
            ),
        )
        self._execution_cost_floor_patch.start()
        self._prior_guardian_watchdog = os.environ.get("QR_GUARDIAN_RECEIPT_WATCHDOG_PATH")
        self._prior_guardian_consumption = os.environ.get("QR_GUARDIAN_RECEIPT_CONSUMPTION_PATH")
        self._prior_guardian_operator_review = os.environ.get("QR_GUARDIAN_RECEIPT_OPERATOR_REVIEW_PATH")
        self._prior_guardian_broker_snapshot = os.environ.get("QR_GUARDIAN_RECEIPT_BROKER_SNAPSHOT_PATH")
        execution_module._per_trade_risk_from_state = lambda: TEST_PER_TRADE_RISK_JPY
        execution_module._daily_risk_budget_from_state = lambda path=None: TEST_DAILY_RISK_BUDGET_JPY
        execution_module._target_trades_per_day_from_state = lambda path=None: None
        guardian_paths = _write_empty_guardian_artifacts(Path(self._guardian_tmp.name))
        os.environ["QR_GUARDIAN_RECEIPT_WATCHDOG_PATH"] = str(guardian_paths["watchdog"])
        os.environ["QR_GUARDIAN_RECEIPT_CONSUMPTION_PATH"] = str(guardian_paths["consumption"])
        os.environ["QR_GUARDIAN_RECEIPT_OPERATOR_REVIEW_PATH"] = str(guardian_paths["operator_review"])
        os.environ["QR_GUARDIAN_RECEIPT_BROKER_SNAPSHOT_PATH"] = str(guardian_paths["broker_snapshot"])

    def tearDown(self) -> None:
        self._gpt_live_send_issues_patch.stop()
        self._execution_cost_floor_patch.stop()
        self._final_pre_post_boundary_patch.stop()
        self._pre_post_reconcile_patch.stop()
        execution_module._per_trade_risk_from_state = self._original_per_trade_reader
        execution_module._daily_risk_budget_from_state = self._original_daily_budget_reader
        execution_module._target_trades_per_day_from_state = self._original_target_trades_reader
        if self._prior_guardian_watchdog is None:
            os.environ.pop("QR_GUARDIAN_RECEIPT_WATCHDOG_PATH", None)
        else:
            os.environ["QR_GUARDIAN_RECEIPT_WATCHDOG_PATH"] = self._prior_guardian_watchdog
        if self._prior_guardian_consumption is None:
            os.environ.pop("QR_GUARDIAN_RECEIPT_CONSUMPTION_PATH", None)
        else:
            os.environ["QR_GUARDIAN_RECEIPT_CONSUMPTION_PATH"] = self._prior_guardian_consumption
        if self._prior_guardian_operator_review is None:
            os.environ.pop("QR_GUARDIAN_RECEIPT_OPERATOR_REVIEW_PATH", None)
        else:
            os.environ["QR_GUARDIAN_RECEIPT_OPERATOR_REVIEW_PATH"] = self._prior_guardian_operator_review
        if self._prior_guardian_broker_snapshot is None:
            os.environ.pop("QR_GUARDIAN_RECEIPT_BROKER_SNAPSHOT_PATH", None)
        else:
            os.environ["QR_GUARDIAN_RECEIPT_BROKER_SNAPSHOT_PATH"] = self._prior_guardian_broker_snapshot
        self._guardian_tmp.cleanup()

    def _run_real_numeric_gateway(
        self,
        *,
        root: Path,
        client: ReconciliationExecutionClient,
        batch: bool,
        metadata: dict[str, Any] | None = None,
        tp: float | None = None,
        sl: float | None = None,
        mutate_price_bound_after_reservation: bool = False,
    ) -> tuple[Any, dict[str, Any], Path]:
        lane_id = "lane:EUR_USD:LONG"
        target_state, target_report, ledger_path, ledger_report = (
            _reconciliation_files(root, edge_vehicle="MARKET")
        )
        resolved_metadata = (
            dict(metadata)
            if metadata is not None
            else _ordinary_claim_metadata(
                vehicle="MARKET",
                numeric_forecast=True,
            )
        )
        if "forecast_directional_economic_hit_rate" in resolved_metadata:
            quote_time = client.snapshot_value.quotes[
                "EUR_USD"
            ].timestamp_utc
            emitted_at = (quote_time - timedelta(seconds=1)).isoformat()
            resolved_metadata["forecast_cycle_id"] = (
                f"pre-entry-forecast-refresh:{emitted_at}:{emitted_at}"
            )
        intents = _intents(
            root,
            lane_id=lane_id,
            order_type="MARKET",
            metadata=resolved_metadata,
            tp=tp,
            sl=sl,
        )
        gateway = LiveOrderGateway(
            client=client,
            strategy_profile=_profile(root),
            output_path=root / "request.json",
            report_path=root / "report.md",
            target_state_path=target_state,
            target_report_path=target_report,
            execution_ledger_db_path=ledger_path,
            execution_ledger_report_path=ledger_report,
            verified_decision_path=_write_ordinary_verified_decision(
                root,
                lane_id=lane_id,
            ),
            live_enabled=True,
        )
        original_reserve = LiveOrderGateway._reserve_ordinary_entry_post

        def reserve_then_optionally_mutate(gateway_self, **kwargs):
            result = original_reserve(gateway_self, **kwargs)
            if (
                mutate_price_bound_after_reservation
                and result.issue is None
                and kwargs["order_request"].get("priceBound")
            ):
                current = float(kwargs["order_request"]["priceBound"])
                kwargs["order_request"]["priceBound"] = (
                    execution_module._price(
                        "EUR_USD",
                        current + execution_module._price_tick("EUR_USD"),
                    )
                )
            return result

        with (
            patch.object(
                LiveOrderGateway,
                "_pre_post_reconcile",
                self._original_pre_post_reconcile,
            ),
            patch.object(
                LiveOrderGateway,
                "_final_pre_post_boundary",
                self._original_final_pre_post_boundary,
            ),
            patch.object(
                LiveOrderGateway,
                "_reserve_ordinary_entry_post",
                reserve_then_optionally_mutate,
            ),
        ):
            summary = (
                gateway.run_batch(
                    intents_path=intents,
                    lane_ids=(lane_id,),
                    send=True,
                    confirm_live=True,
                )
                if batch
                else gateway.run(
                    intents_path=intents,
                    lane_id=lane_id,
                    send=True,
                    confirm_live=True,
                )
            )
        payload = json.loads((root / "request.json").read_text())
        order = payload["orders"][0] if batch else payload
        return summary, order, ledger_path

    def _run_real_tp_proven_range_gateway(
        self,
        *,
        root: Path,
        client: ReconciliationExecutionClient,
        edge_basis: str = "EXACT_VEHICLE_TAKE_PROFIT",
        batch: bool = False,
    ) -> tuple[Any, dict[str, Any], Path]:
        target_state, target_report, ledger_path, ledger_report = (
            _reconciliation_files(root, gross_loss_jpy=300.0)
        )
        _insert_exact_tp_outcomes(
            ledger_path,
            losses=0,
            method="RANGE_ROTATION",
        )
        intents = _tp_proven_range_limit_intents(root, client=client)
        cost_floor = _synthetic_execution_cost_floor(
            ("EUR_USD", "LONG", "RANGE_ROTATION", "LIMIT")
        )
        gateway = LiveOrderGateway(
            client=client,
            strategy_profile=_profile(root),
            output_path=root / "request.json",
            report_path=root / "report.md",
            target_state_path=target_state,
            target_report_path=target_report,
            execution_ledger_db_path=ledger_path,
            execution_ledger_report_path=ledger_report,
            verified_decision_path=_write_ordinary_verified_decision(
                root,
                lane_id=TP_PROVEN_RANGE_LANE_ID,
                direction="RANGE",
                edge_basis=edge_basis,
                execution_cost_floor=cost_floor,
            ),
            live_enabled=True,
            max_loss_jpy=3_000.0,
        )
        with (
            patch.object(
                LiveOrderGateway,
                "_pre_post_reconcile",
                self._original_pre_post_reconcile,
            ),
            patch.object(
                LiveOrderGateway,
                "_final_pre_post_boundary",
                self._original_final_pre_post_boundary,
            ),
            patch.object(
                execution_module,
                "DailyTargetLedger",
                FixedReconciliationTargetLedger,
            ),
        ):
            summary = (
                gateway.run_batch(
                    intents_path=intents,
                    lane_ids=(TP_PROVEN_RANGE_LANE_ID,),
                    send=True,
                    confirm_live=True,
                )
                if batch
                else gateway.run(
                    intents_path=intents,
                    lane_id=TP_PROVEN_RANGE_LANE_ID,
                    send=True,
                    confirm_live=True,
                )
            )
        payload = json.loads((root / "request.json").read_text())
        return (
            summary,
            payload["orders"][0] if batch else payload,
            ledger_path,
        )

    def test_codex_capital_allocation_binds_preclip_units_and_allows_only_later_reduction(self) -> None:
        lane_id = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:STOP"
        board_sha = "a" * 64
        allocation = {
            "decision": "ALLOCATE",
            "lane_id": lane_id,
            "size_multiple": 0.75,
            "selected_units": 1500,
            "allocation_board_sha256": board_sha,
            "rationale": "Direction-specific forecast edge supports bounded exposure.",
        }
        allocation_sha = execution_module.hashlib.sha256(
            json.dumps(
                allocation,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        decision = {
            "action": "TRADE",
            "selected_lane_id": lane_id,
            "selected_lane_ids": [lane_id],
            "capital_allocation": allocation,
            "decision_provenance": {
                "schema_version": 2,
                "author_kind": "CODEX_MARKET_READ",
                "capital_allocation_edge_basis": "EXACT_VEHICLE_ALL_EXIT_NET",
                "capital_allocation_sha256": allocation_sha,
                "capital_allocation_board_sha256": board_sha,
                "authorized_size_multiple": 0.75,
                "authorized_units": 1500,
            },
        }
        intent = SimpleNamespace(side=Side.LONG)

        exact = execution_module._codex_capital_allocation_live_send_issues(
            decision=decision,
            selected_lane_id=lane_id,
            base_units=2001,
            authorized_size_multiple=0.75,
            authorized_units=1500,
            final_units=1500,
            order_request={"units": "1500"},
            intent=intent,
        )
        reduced = execution_module._codex_capital_allocation_live_send_issues(
            decision=decision,
            selected_lane_id=lane_id,
            base_units=2001,
            authorized_size_multiple=0.75,
            authorized_units=1500,
            final_units=1200,
            order_request={"units": "1200"},
            intent=intent,
        )

        self.assertEqual(exact, [])
        self.assertEqual(reduced, [])

    def test_verified_trade_size_multiple_reads_only_exact_accepted_allocation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "verified.json"
            payload = {
                "status": "ACCEPTED",
                "decision": {
                    "action": "TRADE",
                    "capital_allocation": {
                        "decision": "ALLOCATE",
                        "size_multiple": 0.5,
                    },
                },
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(
                execution_module.verified_trade_size_multiple(path),
                0.5,
            )

            payload["decision"]["capital_allocation"]["size_multiple"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNone(execution_module.verified_trade_size_multiple(path))

            payload["decision"]["capital_allocation"]["size_multiple"] = 0.6
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNone(execution_module.verified_trade_size_multiple(path))

    def test_codex_capital_allocation_rejects_caller_and_final_unit_mismatch(self) -> None:
        lane_id = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:STOP"
        allocation = {
            "decision": "ALLOCATE",
            "lane_id": lane_id,
            "size_multiple": 0.5,
            "selected_units": 1000,
            "allocation_board_sha256": "b" * 64,
            "rationale": "Use half size.",
        }
        allocation_sha = execution_module.hashlib.sha256(
            json.dumps(
                allocation,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        decision = {
            "action": "TRADE",
            "selected_lane_id": lane_id,
            "selected_lane_ids": [lane_id],
            "capital_allocation": allocation,
            "decision_provenance": {
                "schema_version": 2,
                "author_kind": "CODEX_MARKET_READ",
                "capital_allocation_edge_basis": "EXACT_VEHICLE_ALL_EXIT_NET",
                "capital_allocation_sha256": allocation_sha,
                "capital_allocation_board_sha256": "b" * 64,
                "authorized_size_multiple": 0.5,
                "authorized_units": 1000,
            },
        }

        issues = execution_module._codex_capital_allocation_live_send_issues(
            decision=decision,
            selected_lane_id=lane_id,
            base_units=2000,
            authorized_size_multiple=1.0,
            authorized_units=2000,
            final_units=1001,
            order_request={"units": "1001"},
            intent=SimpleNamespace(side=Side.LONG),
        )
        codes = {issue.code for issue in issues}

        self.assertIn("GPT_CAPITAL_ALLOCATION_MULTIPLE_MISMATCH", codes)
        self.assertIn("GPT_CAPITAL_ALLOCATION_PRECLIP_MISMATCH", codes)
        self.assertIn("GPT_CAPITAL_ALLOCATION_FINAL_UNITS_EXCEEDED", codes)

    def test_verified_allocation_receipt_change_is_blocked_before_post(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gpt_decision.json"
            path.write_text('{"status":"ACCEPTED"}')
            expected = execution_module._file_sha256(path)
            path.write_text('{"status":"REPLACED"}')

            issue = execution_module._verified_decision_receipt_change_issue(
                path,
                expected_sha256=expected,
            )

            self.assertIsNotNone(issue)
            self.assertEqual(
                issue["code"],
                "GPT_CAPITAL_ALLOCATION_RECEIPT_CHANGED_BEFORE_POST",
            )

    def test_live_allocation_validator_rejects_receipt_bytes_swapped_after_entry(self) -> None:
        original_freeze = (
            execution_module._freeze_verified_decision_receipt
        )
        for batch in (False, True):
            with self.subTest(batch=batch), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                lane_id = "lane:EUR_USD:LONG"
                verified = _write_ordinary_verified_decision(
                    root,
                    lane_id=lane_id,
                    size_multiple=0.5,
                )
                replacement = _write_ordinary_verified_decision(
                    root,
                    lane_id=lane_id,
                    size_multiple=1.0,
                    suffix="replacement",
                ).read_bytes()
                client = FakeExecutionClient()

                def freeze_then_swap(path):
                    result = original_freeze(path)
                    verified.write_bytes(replacement)
                    return result

                gateway = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    verified_decision_path=verified,
                    execution_ledger_db_path=root / "execution_ledger.db",
                    live_enabled=True,
                )
                with patch.object(
                    execution_module,
                    "_freeze_verified_decision_receipt",
                    side_effect=freeze_then_swap,
                ):
                    if batch:
                        summary = gateway.run_batch(
                            intents_path=_intents(root, lane_id=lane_id),
                            lane_ids=(lane_id,),
                            size_multiples={lane_id: 1.0},
                            send=True,
                            confirm_live=True,
                        )
                    else:
                        summary = gateway.run(
                            intents_path=_intents(root, lane_id=lane_id),
                            lane_id=lane_id,
                            size_multiple=1.0,
                            send=True,
                            confirm_live=True,
                        )

                self.assertFalse(summary.sent)
                self.assertEqual(client.orders, [])
                payload = json.loads((root / "request.json").read_text())
                risk_issues = (
                    payload["orders"][0]["risk_issues"]
                    if batch
                    else payload["risk_issues"]
                )
                self.assertIn(
                    "GPT_VERIFIED_RECEIPT_BYTES_MISMATCH_FOR_LIVE_SEND",
                    {issue["code"] for issue in risk_issues},
                )

    def test_receipt_freeze_keeps_same_byte_basis_and_cost_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_cost = _synthetic_execution_cost_floor()
            first = _write_ordinary_verified_decision(
                root,
                lane_id="lane:EUR_USD:LONG",
                edge_basis="EXACT_VEHICLE_ALL_EXIT_NET",
                execution_cost_floor=first_cost,
            )
            frozen = execution_module._freeze_verified_decision_receipt(first)
            replacement_cost = {
                **first_cost,
                "proof_sha256": "d" * 64,
            }
            replacement = _write_ordinary_verified_decision(
                root,
                lane_id="lane:EUR_USD:LONG",
                edge_basis="EXACT_VEHICLE_TAKE_PROFIT",
                execution_cost_floor=replacement_cost,
                suffix="replacement-freeze",
            ).read_bytes()
            first.write_bytes(replacement)

        self.assertTrue(frozen.numeric_allocation_required)
        self.assertEqual(
            frozen.capital_allocation_edge_basis,
            "EXACT_VEHICLE_ALL_EXIT_NET",
        )
        self.assertEqual(
            frozen.execution_cost_floor_sha256,
            first_cost["proof_sha256"],
        )
        self.assertNotEqual(
            frozen.sha256,
            execution_module.hashlib.sha256(replacement).hexdigest(),
        )

    def test_required_signed_basis_and_cost_never_fall_back_to_intent(self) -> None:
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=1000,
            entry=1.17306,
            tp=1.17450,
            sl=1.17250,
            thesis="signed evidence must remain exact",
            market_context=MarketContext(
                regime="TREND",
                narrative="trend",
                chart_story="trend",
                method=TradeMethod.TREND_CONTINUATION,
                invalidation="below structure",
            ),
            metadata=_ordinary_claim_metadata(
                vehicle="MARKET",
                numeric_forecast=True,
            ),
        )
        evidence, issue = execution_module._capital_allocation_edge_pre_post_recheck(
            intent,
            ledger_path=Path("not-read.db"),
            expected_edge_basis=None,
            expected_execution_cost_floor_sha256="a" * 64,
            execution_cost_floor_required=True,
        )
        self.assertIsNotNone(issue)
        self.assertEqual(
            issue.code,
            "PRE_POST_GPT_ALLOCATION_SIGNED_EDGE_BASIS_UNREADABLE",
        )
        self.assertEqual(evidence["status"], "BLOCKED")

        evidence, issue = execution_module._capital_allocation_edge_pre_post_recheck(
            intent,
            ledger_path=Path("not-read.db"),
            expected_edge_basis="EXACT_VEHICLE_ALL_EXIT_NET",
            expected_execution_cost_floor_sha256=None,
            execution_cost_floor_required=True,
        )
        self.assertIsNotNone(issue)
        self.assertEqual(
            issue.code,
            "PRE_POST_GPT_ALLOCATION_SIGNED_COST_FLOOR_UNREADABLE",
        )
        self.assertEqual(evidence["status"], "BLOCKED")

    def test_missing_verified_receipt_reports_single_change_issue_direct_and_batch(self) -> None:
        for batch in (False, True):
            with self.subTest(batch=batch), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                lane_id = "lane:EUR_USD:LONG"
                client = FakeExecutionClient()
                gateway = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    verified_decision_path=root / "missing_gpt_decision.json",
                    execution_ledger_db_path=root / "execution_ledger.db",
                    live_enabled=True,
                )

                if batch:
                    summary = gateway.run_batch(
                        intents_path=_intents(root, lane_id=lane_id),
                        lane_ids=(lane_id,),
                        send=True,
                        confirm_live=True,
                    )
                else:
                    summary = gateway.run(
                        intents_path=_intents(root, lane_id=lane_id),
                        lane_id=lane_id,
                        send=True,
                        confirm_live=True,
                    )

                self.assertFalse(summary.sent)
                self.assertEqual(client.orders, [])
                payload = json.loads((root / "request.json").read_text())
                risk_issues = (
                    payload["orders"][0]["risk_issues"]
                    if batch
                    else payload["risk_issues"]
                )
                self.assertEqual(
                    sum(
                        issue["code"]
                        == "GPT_CAPITAL_ALLOCATION_RECEIPT_CHANGED_BEFORE_POST"
                        for issue in risk_issues
                    ),
                    1,
                )

    def test_live_fresh_entry_without_verified_receipt_is_blocked_direct_and_batch(self) -> None:
        self._enforce_missing_gpt_receipt = True
        for batch in (False, True):
            with self.subTest(batch=batch), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                lane_id = "lane:EUR_USD:LONG"
                client = FakeExecutionClient()
                gateway = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    execution_ledger_db_path=root / "execution_ledger.db",
                    live_enabled=True,
                )

                if batch:
                    summary = gateway.run_batch(
                        intents_path=_intents(root, lane_id=lane_id),
                        lane_ids=(lane_id,),
                        send=True,
                        confirm_live=True,
                    )
                else:
                    summary = gateway.run(
                        intents_path=_intents(root, lane_id=lane_id),
                        lane_id=lane_id,
                        send=True,
                        confirm_live=True,
                    )

                self.assertFalse(summary.sent)
                self.assertEqual(client.orders, [])
                payload = json.loads((root / "request.json").read_text())
                risk_issues = (
                    payload["orders"][0]["risk_issues"]
                    if batch
                    else payload["risk_issues"]
                )
                self.assertEqual(
                    sum(
                        issue["code"]
                        == "GPT_FRESH_TRADE_RECEIPT_REQUIRED_FOR_LIVE_SEND"
                        for issue in risk_issues
                    ),
                    1,
                )

    def test_accepted_trade_without_codex_allocation_is_blocked_before_direct_and_batch_post(self) -> None:
        for batch in (False, True):
            with self.subTest(batch=batch), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                lane_id = "lane:EUR_USD:LONG"
                verified = _write_ordinary_verified_decision(root, lane_id=lane_id)
                receipt = json.loads(verified.read_text())
                receipt["decision"].pop("capital_allocation", None)
                receipt["decision"].pop("decision_provenance", None)
                receipt.pop("market_read_prediction", None)
                verified.write_text(json.dumps(receipt) + "\n")
                client = FakeExecutionClient()
                gateway = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    verified_decision_path=verified,
                    execution_ledger_db_path=root / "execution_ledger.db",
                    live_enabled=True,
                )

                if batch:
                    summary = gateway.run_batch(
                        intents_path=_intents(root, lane_id=lane_id),
                        lane_ids=(lane_id,),
                        send=True,
                        confirm_live=True,
                    )
                else:
                    summary = gateway.run(
                        intents_path=_intents(root, lane_id=lane_id),
                        lane_id=lane_id,
                        send=True,
                        confirm_live=True,
                    )

                self.assertFalse(summary.sent)
                self.assertEqual(client.orders, [])
                payload = json.loads((root / "request.json").read_text())
                risk_issues = (
                    payload["orders"][0]["risk_issues"]
                    if batch
                    else payload["risk_issues"]
                )
                codes = {issue["code"] for issue in risk_issues}
                self.assertIn("GPT_CAPITAL_ALLOCATION_AUTHOR_MISMATCH", codes)
                self.assertIn("GPT_CAPITAL_ALLOCATION_REQUIRED_FOR_LIVE_SEND", codes)

    def test_add_action_receipt_cannot_bypass_trade_capital_allocation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lane_id = "lane:EUR_USD:LONG"
            verified = _write_ordinary_verified_decision(
                root,
                lane_id=lane_id,
                action="ADD",
            )
            client = FakeExecutionClient()

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                verified_decision_path=verified,
                execution_ledger_db_path=root / "execution_ledger.db",
                live_enabled=True,
            ).run(
                intents_path=_intents(root, lane_id=lane_id),
                lane_id=lane_id,
                send=True,
                confirm_live=True,
            )

            self.assertFalse(summary.sent)
            self.assertEqual(client.orders, [])
            payload = json.loads((root / "request.json").read_text())
            self.assertIn(
                "GPT_FRESH_TRADE_RECEIPT_REQUIRED_FOR_LIVE_SEND",
                {issue["code"] for issue in payload["risk_issues"]},
            )

    def test_verified_receipt_mutation_during_ordinary_reservation_blocks_direct_and_batch_post(self) -> None:
        original_reserve = LiveOrderGateway._reserve_ordinary_entry_post
        for batch in (False, True):
            with self.subTest(batch=batch), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                lane_id = "lane:EUR_USD:LONG"
                verified = _write_ordinary_verified_decision(root, lane_id=lane_id)
                client = FakeExecutionClient()

                def mutate_after_reservation(gateway_self, **kwargs):
                    result = original_reserve(gateway_self, **kwargs)
                    receipt = json.loads(verified.read_text())
                    receipt["decision"]["capital_allocation"]["selected_units"] += 1
                    verified.write_text(json.dumps(receipt) + "\n")
                    return result

                gateway = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    verified_decision_path=verified,
                    execution_ledger_db_path=root / "execution_ledger.db",
                    live_enabled=True,
                )
                with patch.object(
                    LiveOrderGateway,
                    "_reserve_ordinary_entry_post",
                    mutate_after_reservation,
                ):
                    if batch:
                        summary = gateway.run_batch(
                            intents_path=_intents(
                                root,
                                lane_id=lane_id,
                                metadata=_ordinary_claim_metadata(),
                            ),
                            lane_ids=(lane_id,),
                            send=True,
                            confirm_live=True,
                        )
                    else:
                        summary = gateway.run(
                            intents_path=_intents(
                                root,
                                lane_id=lane_id,
                                metadata=_ordinary_claim_metadata(),
                            ),
                            lane_id=lane_id,
                            send=True,
                            confirm_live=True,
                        )

                self.assertFalse(summary.sent)
                self.assertEqual(client.orders, [])
                payload = json.loads((root / "request.json").read_text())
                risk_issues = (
                    payload["orders"][0]["risk_issues"]
                    if batch
                    else payload["risk_issues"]
                )
                self.assertIn(
                    "GPT_CAPITAL_ALLOCATION_RECEIPT_CHANGED_BEFORE_POST",
                    {issue["code"] for issue in risk_issues},
                )

    def test_transaction_advance_during_reservation_blocks_direct_and_batch_post(self) -> None:
        original_reserve = LiveOrderGateway._reserve_ordinary_entry_post
        for batch in (False, True):
            with self.subTest(batch=batch), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                lane_id = "lane:EUR_USD:LONG"
                target_state, target_report, ledger_path, ledger_report = (
                    _reconciliation_files(root, edge_vehicle="MARKET")
                )
                intents = _intents(
                    root,
                    lane_id=lane_id,
                    order_type="MARKET",
                    metadata=_ordinary_claim_metadata(
                        vehicle="MARKET",
                        numeric_forecast=True,
                    ),
                )
                verified = _write_ordinary_verified_decision(
                    root,
                    lane_id=lane_id,
                )
                client = PostReservationTransactionAdvanceClient()

                def reserve_then_advance(gateway_self, **kwargs):
                    result = original_reserve(gateway_self, **kwargs)
                    if result.issue is None:
                        client.advance_after_reservation = True
                    return result

                gateway = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    target_state_path=target_state,
                    target_report_path=target_report,
                    execution_ledger_db_path=ledger_path,
                    execution_ledger_report_path=ledger_report,
                    verified_decision_path=verified,
                    live_enabled=True,
                )
                with (
                    patch.object(
                        LiveOrderGateway,
                        "_pre_post_reconcile",
                        self._original_pre_post_reconcile,
                    ),
                    patch.object(
                        LiveOrderGateway,
                        "_final_pre_post_boundary",
                        self._original_final_pre_post_boundary,
                    ),
                    patch.object(
                        LiveOrderGateway,
                        "_reserve_ordinary_entry_post",
                        reserve_then_advance,
                    ),
                ):
                    if batch:
                        summary = gateway.run_batch(
                            intents_path=intents,
                            lane_ids=(lane_id,),
                            send=True,
                            confirm_live=True,
                        )
                    else:
                        summary = gateway.run(
                            intents_path=intents,
                            lane_id=lane_id,
                            send=True,
                            confirm_live=True,
                        )

                self.assertFalse(summary.sent)
                self.assertEqual(client.orders, [])
                payload = json.loads((root / "request.json").read_text())
                order = payload["orders"][0] if batch else payload
                codes = {issue["code"] for issue in order["risk_issues"]}
                self.assertIn(
                    "FINAL_PRE_POST_TRANSACTION_ID_CHANGED_AFTER_RESERVATION",
                    codes,
                )
                boundary = order["pre_post_reconciliation"][
                    "final_post_reservation_boundary"
                ]
                self.assertEqual(boundary["status"], "BLOCKED")
                self.assertEqual(boundary["early_broker_last_transaction_id"], "100")
                self.assertEqual(boundary["final_ledger_last_transaction_id"], "101")
                self.assertEqual(boundary["final_broker_last_transaction_id"], "101")
                self.assertFalse(boundary["transaction_id_unchanged"])
                self.assertEqual(
                    boundary["capital_allocation_edge_recheck"]["status"],
                    "PASSED",
                )
                self.assertFalse(
                    order["sizing_evidence"]["capital_allocation_validated"]
                )
                self.assertEqual(
                    order["ordinary_entry_claim"]["status"],
                    "FINAL_BOUNDARY_BLOCKED_RESERVATION_RETAINED",
                )
                with closing(sqlite3.connect(ledger_path)) as conn:
                    claim_status = conn.execute(
                        "SELECT status FROM ordinary_live_entry_signal_claims"
                    ).fetchone()[0]
                self.assertEqual(
                    claim_status,
                    "FINAL_BOUNDARY_BLOCKED_RESERVATION_RETAINED",
                )

    def test_unchanged_transaction_passes_final_boundary_direct_and_batch(self) -> None:
        for batch in (False, True):
            with self.subTest(batch=batch), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                lane_id = "lane:EUR_USD:LONG"
                target_state, target_report, ledger_path, ledger_report = (
                    _reconciliation_files(root, edge_vehicle="MARKET")
                )
                intents = _intents(
                    root,
                    lane_id=lane_id,
                    order_type="MARKET",
                    metadata=_ordinary_claim_metadata(
                        vehicle="MARKET",
                        numeric_forecast=True,
                    ),
                )
                client = ReconciliationExecutionClient()
                gateway = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    target_state_path=target_state,
                    target_report_path=target_report,
                    execution_ledger_db_path=ledger_path,
                    execution_ledger_report_path=ledger_report,
                    verified_decision_path=_write_ordinary_verified_decision(
                        root,
                        lane_id=lane_id,
                    ),
                    live_enabled=True,
                )
                with (
                    patch.object(
                        LiveOrderGateway,
                        "_pre_post_reconcile",
                        self._original_pre_post_reconcile,
                    ),
                    patch.object(
                        LiveOrderGateway,
                        "_final_pre_post_boundary",
                        self._original_final_pre_post_boundary,
                    ),
                ):
                    if batch:
                        summary = gateway.run_batch(
                            intents_path=intents,
                            lane_ids=(lane_id,),
                            send=True,
                            confirm_live=True,
                        )
                    else:
                        summary = gateway.run(
                            intents_path=intents,
                            lane_id=lane_id,
                            send=True,
                            confirm_live=True,
                        )

                self.assertTrue(summary.sent)
                self.assertEqual(len(client.orders), 1)
                payload = json.loads((root / "request.json").read_text())
                order = payload["orders"][0] if batch else payload
                boundary = order["pre_post_reconciliation"][
                    "final_post_reservation_boundary"
                ]
                self.assertEqual(boundary["status"], "PASSED")
                self.assertTrue(boundary["transaction_id_unchanged"])
                self.assertTrue(boundary["verified_decision_sha_matches"])
                self.assertTrue(boundary["order_request_sha_matches"])
                self.assertTrue(
                    order["sizing_evidence"]["capital_allocation_validated"]
                )
                self.assertEqual(
                    order["ordinary_entry_claim"]["status"],
                    "BROKER_RESPONSE_RECORDED",
                )
                early_bound = order["pre_post_reconciliation"][
                    "capital_allocation_numeric_recheck"
                ]["market_price_bound"]
                self.assertEqual(
                    order["order_request"]["priceBound"],
                    early_bound["price_bound_text"],
                )
                self.assertEqual(
                    client.orders[0]["priceBound"],
                    early_bound["price_bound_text"],
                )
                final_bound = boundary[
                    "capital_allocation_numeric_recheck"
                ]["market_price_bound"]
                self.assertEqual(
                    final_bound["reserved_price_bound_text"],
                    early_bound["price_bound_text"],
                )
                self.assertEqual(
                    boundary["post_s5_account_fence"]["status"],
                    "PASSED",
                )

    def test_frozen_numeric_proof_rejects_forged_net_cost_identities(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary, order, _ = self._run_real_numeric_gateway(
                root=Path(tmp),
                client=ReconciliationExecutionClient(),
                batch=False,
            )
        self.assertTrue(summary.sent)
        proof = order["pre_post_reconciliation"][
            "capital_allocation_numeric_recheck"
        ]
        order_request = order["order_request"]
        expected_intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=1000,
            entry=1.17306,
            tp=1.17450,
            sl=1.17253,
            thesis="frozen cost proof",
            market_context=MarketContext(
                regime="TREND_CONTINUATION campaign lane",
                narrative="trend continuation pressure",
                chart_story="trend staircase",
                method=TradeMethod.TREND_CONTINUATION,
                invalidation="technical invalidation",
            ),
        )

        def rehash(candidate: dict[str, Any]) -> None:
            bound = candidate["market_price_bound"]
            bound_material = dict(bound)
            bound_material.pop("proof_sha256", None)
            bound["proof_sha256"] = (
                execution_module._canonical_json_sha256(bound_material)
            )
            material = dict(candidate)
            material.pop("proof_sha256", None)
            candidate["proof_sha256"] = (
                execution_module._canonical_json_sha256(material)
            )

        def frozen(candidate: dict[str, Any]) -> bool:
            return execution_module._capital_allocation_numeric_proof_is_frozen(
                candidate,
                order_request=order_request,
                required=True,
                expected_intent=expected_intent,
            )

        self.assertTrue(frozen(proof))
        bound = proof["market_price_bound"]
        self.assertGreater(bound["outcome_cost_jpy"], 0.0)
        self.assertAlmostEqual(
            bound["outcome_cost_jpy"],
            bound["exit_execution_stress_jpy"]
            + bound["financing_stress_jpy"],
        )
        self.assertAlmostEqual(
            bound["worst_fill_net_risk_jpy"],
            bound["worst_fill_risk_jpy"] + bound["outcome_cost_jpy"],
        )
        self.assertAlmostEqual(
            bound["worst_fill_net_reward_jpy"],
            bound["worst_fill_reward_jpy"] - bound["outcome_cost_jpy"],
        )

        def cost_exact_once(candidate: dict[str, Any]) -> None:
            forged = candidate["market_price_bound"]
            forged["outcome_cost_jpy"] += 1.0
            forged["worst_fill_net_risk_jpy"] += 1.0
            forged["worst_fill_net_reward_jpy"] -= 1.0
            forged["next_adverse_tick_net_risk_jpy"] += 1.0
            forged["next_adverse_tick_net_reward_jpy"] -= 1.0

        def wrong_cost_contract(candidate: dict[str, Any]) -> None:
            cost = deepcopy(
                candidate["numeric_ceiling"]["execution_cost_floor"]
            )
            cost["contract"] = "FORGED_COST_CONTRACT"
            material = dict(cost)
            material.pop("proof_sha256", None)
            cost["proof_sha256"] = (
                execution_module._canonical_json_sha256(material)
            )
            candidate["numeric_ceiling"]["execution_cost_floor"] = deepcopy(
                cost
            )
            candidate["market_price_bound"]["execution_cost_floor"] = deepcopy(
                cost
            )

        def wrong_cost_scope(candidate: dict[str, Any]) -> None:
            cost = deepcopy(
                candidate["numeric_ceiling"]["execution_cost_floor"]
            )
            wrong = "EUR_USD|LONG|BREAKOUT_FAILURE|MARKET"
            cost["scope_key"] = wrong
            material = dict(cost)
            material.pop("proof_sha256", None)
            cost["proof_sha256"] = (
                execution_module._canonical_json_sha256(material)
            )
            candidate["numeric_ceiling"]["execution_cost_floor"] = deepcopy(
                cost
            )
            forged_bound = candidate["market_price_bound"]
            forged_bound["execution_cost_floor"] = deepcopy(cost)
            forged_bound["method"] = "BREAKOUT_FAILURE"
            forged_bound["expected_execution_cost_scope_key"] = wrong

        mutations = {
            "cost_counted_twice": cost_exact_once,
            "worst_net_risk_identity": lambda candidate: candidate[
                "market_price_bound"
            ].__setitem__(
                "worst_fill_net_risk_jpy",
                candidate["market_price_bound"]["worst_fill_net_risk_jpy"]
                + 1.0,
            ),
            "next_net_reward_identity": lambda candidate: candidate[
                "market_price_bound"
            ].__setitem__(
                "next_adverse_tick_net_reward_jpy",
                candidate["market_price_bound"][
                    "next_adverse_tick_net_reward_jpy"
                ]
                - 1.0,
            ),
            "boolean_net_risk": lambda candidate: candidate[
                "market_price_bound"
            ].__setitem__("worst_fill_net_risk_jpy", True),
            "nan_net_reward": lambda candidate: candidate[
                "market_price_bound"
            ].__setitem__("worst_fill_net_reward_jpy", float("nan")),
            "portfolio_below_net_risk": lambda candidate: candidate[
                "market_price_bound"
            ].__setitem__(
                "portfolio_loss_remaining_jpy",
                candidate["market_price_bound"]["worst_fill_net_risk_jpy"]
                - 0.01,
            ),
            "wrong_cost_contract": wrong_cost_contract,
            "wrong_cost_scope": wrong_cost_scope,
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                forged = deepcopy(proof)
                mutate(forged)
                rehash(forged)
                self.assertFalse(frozen(forged))

    def test_price_bound_mutation_after_reservation_blocks_direct_and_batch(self) -> None:
        for batch in (False, True):
            with self.subTest(batch=batch), tempfile.TemporaryDirectory() as tmp:
                client = ReconciliationExecutionClient()
                summary, order, ledger_path = self._run_real_numeric_gateway(
                    root=Path(tmp),
                    client=client,
                    batch=batch,
                    mutate_price_bound_after_reservation=True,
                )
                self.assertFalse(summary.sent)
                self.assertEqual(client.orders, [])
                codes = {issue["code"] for issue in order["risk_issues"]}
                self.assertIn(
                    "FINAL_PRE_POST_ORDER_REQUEST_CHANGED_AFTER_RESERVATION",
                    codes,
                )
                boundary = order["pre_post_reconciliation"][
                    "final_post_reservation_boundary"
                ]
                self.assertFalse(boundary["order_request_sha_matches"])
                self.assertFalse(boundary["post_attempted"])
                self.assertEqual(
                    order["ordinary_entry_claim"]["status"],
                    "FINAL_BOUNDARY_BLOCKED_RESERVATION_RETAINED",
                )
                with closing(sqlite3.connect(ledger_path)) as conn:
                    claim_status = conn.execute(
                        "SELECT status FROM ordinary_live_entry_signal_claims"
                    ).fetchone()[0]
                self.assertEqual(
                    claim_status,
                    "FINAL_BOUNDARY_BLOCKED_RESERVATION_RETAINED",
                )

    def test_final_s5_path_corruption_retains_reservation_direct_and_batch(self) -> None:
        expected_reasons = {
            "target_touch": "FORECAST_S5_TP_OR_TARGET_TOUCHED",
            "invalidation_touch": "FORECAST_S5_INVALIDATION_TOUCHED",
            "both_touch": "FORECAST_S5_BOTH_BARRIERS_TOUCHED",
            "gap": "FORECAST_S5_COVERAGE_INCOMPLETE",
            "malformed": "FORECAST_S5_CANDLE_INVALID",
            "fetch_failure": "FORECAST_S5_FETCH_FAILED",
            "timeout": "FORECAST_S5_FETCH_FAILED",
            "wrong_instrument": "FORECAST_S5_PAYLOAD_INVALID",
        }
        for batch in (False, True):
            for scenario, expected_reason in expected_reasons.items():
                with (
                    self.subTest(batch=batch, scenario=scenario),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    client = FinalS5ScenarioExecutionClient(scenario)
                    summary, order, ledger_path = self._run_real_numeric_gateway(
                        root=Path(tmp),
                        client=client,
                        batch=batch,
                    )
                    self.assertFalse(summary.sent)
                    self.assertEqual(client.orders, [])
                    codes = {issue["code"] for issue in order["risk_issues"]}
                    self.assertIn(
                        "FINAL_PRE_POST_GPT_ALLOCATION_FORECAST_S5_PATH_REPROOF_FAILED",
                        codes,
                    )
                    boundary = order["pre_post_reconciliation"][
                        "final_post_reservation_boundary"
                    ]
                    path_proof = boundary[
                        "capital_allocation_numeric_recheck"
                    ]["forecast_s5_path_proof"]
                    self.assertEqual(path_proof["reason"], expected_reason)
                    self.assertEqual(boundary["status"], "BLOCKED")
                    self.assertFalse(boundary["post_attempted"])
                    self.assertEqual(
                        order["ordinary_entry_claim"]["status"],
                        "FINAL_BOUNDARY_BLOCKED_RESERVATION_RETAINED",
                    )
                    with closing(sqlite3.connect(ledger_path)) as conn:
                        claim_status = conn.execute(
                            "SELECT status FROM ordinary_live_entry_signal_claims"
                        ).fetchone()[0]
                    self.assertEqual(
                        claim_status,
                        "FINAL_BOUNDARY_BLOCKED_RESERVATION_RETAINED",
                    )

    def test_tp_proven_range_passive_limit_send_freezes_rail_and_fresh_tp_basis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = ReconciliationExecutionClient()
            summary, order, _ = self._run_real_tp_proven_range_gateway(
                root=Path(tmp),
                client=client,
            )

        self.assertTrue(summary.sent, json.dumps(order, sort_keys=True))
        self.assertEqual(len(client.orders), 1)
        request = order["order_request"]
        self.assertEqual(request["type"], "LIMIT")
        self.assertEqual(request["price"], "1.17100")
        self.assertNotIn("priceBound", request)
        reconciliation = order["pre_post_reconciliation"]
        numeric = reconciliation["capital_allocation_numeric_recheck"]
        self.assertEqual(
            numeric["numeric_ceiling"]["reason"],
            "TP_PROVEN_RANGE_NONMARKET_PREBOUNDED_CONTRACT",
        )
        self.assertTrue(numeric["range_tp_prebounded_claimed"])
        self.assertTrue(numeric["range_tp_geometry_frozen"])
        self.assertEqual(
            numeric["range_tp_fresh_edge_basis"]["basis"],
            "EXACT_TP_PROVEN_HARVEST",
        )
        self.assertEqual(
            numeric["forecast_s5_path_proof"]["barrier_basis"],
            "RANGE_RAILS",
        )
        self.assertFalse(
            numeric["forecast_s5_path_proof"]["order_entry_touched"]
        )
        self.assertEqual(
            reconciliation["capital_allocation_edge_recheck"]["basis"],
            "EXACT_VEHICLE_TAKE_PROFIT",
        )
        boundary = reconciliation["final_post_reservation_boundary"]
        self.assertEqual(boundary["status"], "PASSED")
        self.assertEqual(client.orders[0]["price"], "1.17100")
        self.assertNotIn("priceBound", client.orders[0])

    def test_tp_proven_range_final_s5_entry_touch_retains_reservation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = FinalRangeEntryTouchExecutionClient()
            summary, order, ledger_path = (
                self._run_real_tp_proven_range_gateway(
                    root=Path(tmp),
                    client=client,
                )
            )
            with closing(sqlite3.connect(ledger_path)) as conn:
                claim_status = conn.execute(
                    "SELECT status FROM ordinary_live_entry_signal_claims"
                ).fetchone()[0]

        self.assertFalse(summary.sent)
        self.assertEqual(client.orders, [])
        self.assertEqual(client.s5_calls, 2)
        self.assertIn(
            "FINAL_PRE_POST_GPT_ALLOCATION_FORECAST_S5_PATH_REPROOF_FAILED",
            {issue["code"] for issue in order["risk_issues"]},
        )
        boundary = order["pre_post_reconciliation"][
            "final_post_reservation_boundary"
        ]
        proof = boundary["capital_allocation_numeric_recheck"][
            "forecast_s5_path_proof"
        ]
        self.assertEqual(
            proof["reason"],
            "FORECAST_S5_RANGE_ENTRY_OR_SL_TOUCHED",
        )
        self.assertTrue(proof["order_entry_touched"])
        self.assertFalse(boundary["post_attempted"])
        self.assertEqual(
            order["ordinary_entry_claim"]["status"],
            "FINAL_BOUNDARY_BLOCKED_RESERVATION_RETAINED",
        )
        self.assertEqual(
            claim_status,
            "FINAL_BOUNDARY_BLOCKED_RESERVATION_RETAINED",
        )

    def test_tp_proven_range_signed_edge_basis_switch_blocks_before_post(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = ReconciliationExecutionClient()
            summary, order, _ = self._run_real_tp_proven_range_gateway(
                root=Path(tmp),
                client=client,
                edge_basis="EXACT_VEHICLE_ALL_EXIT_NET",
            )

        self.assertFalse(summary.sent)
        self.assertEqual(client.orders, [])
        self.assertIn(
            "PRE_POST_GPT_ALLOCATION_EDGE_BASIS_UNPROVEN",
            {issue["code"] for issue in order["risk_issues"]},
        )
        self.assertEqual(
            order["pre_post_reconciliation"]["status"],
            "BLOCKED",
        )

    def test_gateway_uses_frozen_receipt_basis_and_cost_without_path_reread(self) -> None:
        for batch in (False, True):
            with self.subTest(batch=batch), tempfile.TemporaryDirectory() as tmp:
                client = ReconciliationExecutionClient()
                with (
                    patch.object(
                        execution_module,
                        "_verified_capital_allocation_edge_basis",
                        side_effect=AssertionError(
                            "mutable receipt path must not select an edge basis"
                        ),
                    ),
                    patch.object(
                        execution_module,
                        "_verified_execution_cost_floor_sha256",
                        side_effect=AssertionError(
                            "mutable receipt path must not select a cost proof"
                        ),
                    ),
                ):
                    summary, order, _ = (
                        self._run_real_tp_proven_range_gateway(
                            root=Path(tmp),
                            client=client,
                            batch=batch,
                        )
                    )

                self.assertTrue(summary.sent, json.dumps(order, sort_keys=True))
                self.assertEqual(len(client.orders), 1)
                self.assertEqual(
                    order["pre_post_reconciliation"][
                        "capital_allocation_edge_recheck"
                    ]["basis"],
                    "EXACT_VEHICLE_TAKE_PROFIT",
                )

    def test_tp_proven_range_post_s5_quote_cross_retains_reservation_direct_and_batch(self) -> None:
        for batch in (False, True):
            with self.subTest(batch=batch), tempfile.TemporaryDirectory() as tmp:
                client = PostS5RangeQuoteCrossExecutionClient()
                summary, order, ledger_path = (
                    self._run_real_tp_proven_range_gateway(
                        root=Path(tmp),
                        client=client,
                        batch=batch,
                    )
                )
                with closing(sqlite3.connect(ledger_path)) as conn:
                    claim_status = conn.execute(
                        "SELECT status FROM ordinary_live_entry_signal_claims"
                    ).fetchone()[0]

            self.assertFalse(summary.sent)
            self.assertEqual(client.orders, [])
            self.assertEqual(client.s5_calls, 2)
            self.assertEqual(client.snapshot_reads_after_final_s5, 1)
            codes = {issue["code"] for issue in order["risk_issues"]}
            self.assertIn(
                "FINAL_PRE_POST_TP_PROVEN_RANGE_TRIGGER_CROSSED",
                codes,
            )
            self.assertNotIn("LIMIT_ENTRY_REPRICED_PASSIVE", codes)
            boundary = order["pre_post_reconciliation"][
                "final_post_reservation_boundary"
            ]
            quote_fence = boundary[
                "post_s5_tp_proven_range_quote_fence"
            ]
            self.assertEqual(quote_fence["status"], "BLOCKED")
            self.assertEqual(
                quote_fence["reason"],
                "POST_S5_RANGE_LIMIT_ENTRY_CROSSED",
            )
            self.assertTrue(quote_fence["entry_touched"])
            self.assertFalse(quote_fence["repriced"])
            broker_fence = boundary["post_s5_broker_fence"]
            self.assertTrue(broker_fence["account_and_quote_same_snapshot"])
            self.assertTrue(broker_fence["account_fence_passed"])
            self.assertFalse(boundary["post_attempted"])
            self.assertEqual(
                order["ordinary_entry_claim"]["status"],
                "FINAL_BOUNDARY_BLOCKED_RESERVATION_RETAINED",
            )
            self.assertEqual(
                claim_status,
                "FINAL_BOUNDARY_BLOCKED_RESERVATION_RETAINED",
            )

    def test_post_s5_account_deterioration_blocks_direct_and_batch(self) -> None:
        for batch in (False, True):
            for scenario in ("nav", "margin_used", "margin_available"):
                with (
                    self.subTest(batch=batch, scenario=scenario),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    client = PostS5AccountDeteriorationClient(scenario)
                    summary, order, _ = self._run_real_numeric_gateway(
                        root=Path(tmp),
                        client=client,
                        batch=batch,
                    )
                    self.assertFalse(summary.sent)
                    self.assertEqual(client.orders, [])
                    self.assertEqual(client.s5_calls, 2)
                    self.assertEqual(client.snapshot_reads_after_final_s5, 1)
                    codes = {issue["code"] for issue in order["risk_issues"]}
                    self.assertIn("FINAL_PRE_POST_ACCOUNT_FENCE_FAILED", codes)
                    boundary = order["pre_post_reconciliation"][
                        "final_post_reservation_boundary"
                    ]
                    fence = boundary["post_s5_account_fence"]
                    self.assertEqual(fence["status"], "BLOCKED")
                    self.assertEqual(
                        fence["read_order"],
                        "AFTER_FINAL_S5_AND_WORST_FILL_REPROOF_SAME_BROKER_SNAPSHOT",
                    )
                    self.assertTrue(
                        fence[f"{scenario}_worsened"]
                        if scenario != "margin_available"
                        else fence["margin_available_worsened"]
                    )
                    self.assertEqual(
                        order["ordinary_entry_claim"]["status"],
                        "FINAL_BOUNDARY_BLOCKED_RESERVATION_RETAINED",
                    )

    def test_market_price_bound_rounding_and_next_adverse_tick_fail_closed(self) -> None:
        snapshot = ReconciliationExecutionClient().snapshot_value
        long_metadata = _ordinary_claim_metadata(
            vehicle="MARKET",
            numeric_forecast=True,
        )
        long_intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=1000,
            entry=1.17306,
            tp=1.17450,
            sl=1.17250,
            thesis="long bound",
            metadata=long_metadata,
        )
        long_proof, long_issue = (
            execution_module._capital_allocation_market_price_bound(
                intent=long_intent,
                snapshot=snapshot,
                effective_max_loss_jpy=2_000.0,
                execution_cost_floor=_synthetic_execution_cost_floor(
                    ("EUR_USD", "LONG", "UNKNOWN", "MARKET")
                ),
            )
        )
        self.assertIsNone(long_issue)
        self.assertEqual(long_proof["status"], "PASSED")
        self.assertTrue(long_proof["next_adverse_tick_fails"])
        tick = execution_module._price_tick("EUR_USD")
        self.assertAlmostEqual(
            long_proof["price_bound"] / tick,
            round(long_proof["price_bound"] / tick),
        )
        next_long, next_long_issue = (
            execution_module._capital_allocation_market_price_bound(
                intent=long_intent,
                snapshot=snapshot,
                effective_max_loss_jpy=2_000.0,
                execution_cost_floor=_synthetic_execution_cost_floor(
                    ("EUR_USD", "LONG", "UNKNOWN", "MARKET")
                ),
                reserved_price_bound=execution_module._price(
                    "EUR_USD",
                    long_proof["price_bound"] + tick,
                ),
            )
        )
        self.assertIsNotNone(next_long_issue)
        self.assertEqual(next_long["status"], "BLOCKED")

        short_metadata = dict(long_metadata)
        short_metadata.update(
            {
                "forecast_direction": "DOWN",
                "forecast_directional_calibration_name": (
                    "directional_forecast_down"
                ),
                "forecast_target_price": 1.17142,
                "forecast_invalidation_price": 1.17342,
            }
        )
        short_intent = replace(
            long_intent,
            side=Side.SHORT,
            entry=1.17298,
            tp=1.17150,
            sl=1.17350,
            thesis="short bound",
            metadata=short_metadata,
        )
        short_proof, short_issue = (
            execution_module._capital_allocation_market_price_bound(
                intent=short_intent,
                snapshot=snapshot,
                effective_max_loss_jpy=2_000.0,
                execution_cost_floor=_synthetic_execution_cost_floor(
                    ("EUR_USD", "SHORT", "UNKNOWN", "MARKET")
                ),
            )
        )
        self.assertIsNone(short_issue)
        self.assertEqual(short_proof["status"], "PASSED")
        self.assertTrue(short_proof["next_adverse_tick_fails"])
        self.assertAlmostEqual(
            short_proof["price_bound"] / tick,
            round(short_proof["price_bound"] / tick),
        )
        next_short, next_short_issue = (
            execution_module._capital_allocation_market_price_bound(
                intent=short_intent,
                snapshot=snapshot,
                effective_max_loss_jpy=2_000.0,
                execution_cost_floor=_synthetic_execution_cost_floor(
                    ("EUR_USD", "SHORT", "UNKNOWN", "MARKET")
                ),
                reserved_price_bound=execution_module._price(
                    "EUR_USD",
                    short_proof["price_bound"] - tick,
                ),
            )
        )
        self.assertIsNotNone(next_short_issue)
        self.assertEqual(next_short["status"], "BLOCKED")

        # Regression for a concrete raw-float false positive: the old helper
        # found +EV on unrounded rails while OANDA's 5dp TP/SL made the same
        # bound negative. The helper must derive a different safe bound from
        # the exact transport rails.
        off_grid_metadata = dict(long_metadata)
        off_grid_metadata.update(
            {
                "forecast_directional_economic_hit_rate": 0.70,
                "forecast_directional_economic_samples": 100,
            }
        )
        off_grid_intent = replace(
            long_intent,
            tp=1.1738024885932492,
            sl=1.1724109430942786,
            metadata=off_grid_metadata,
        )
        off_grid_proof, off_grid_issue = (
            execution_module._capital_allocation_market_price_bound(
                intent=off_grid_intent,
                snapshot=snapshot,
                effective_max_loss_jpy=2_000.0,
                execution_cost_floor=_synthetic_execution_cost_floor(
                    ("EUR_USD", "LONG", "UNKNOWN", "MARKET")
                ),
            )
        )
        self.assertIsNone(off_grid_issue)
        self.assertEqual(off_grid_proof["tp"], 1.17380)
        self.assertEqual(off_grid_proof["sl"], 1.17241)
        self.assertEqual(
            off_grid_proof["dependent_price_basis"],
            "EXACT_OANDA_TRANSPORT_PRECISION",
        )
        self.assertTrue(off_grid_proof["worst_fill_ev_strictly_positive"])

    def test_s5_validator_recomputes_rows_instead_of_trusting_forged_hashes(self) -> None:
        client = ReconciliationExecutionClient()
        snapshot = client.snapshot_value
        metadata = _ordinary_claim_metadata(
            vehicle="MARKET",
            numeric_forecast=True,
        )
        emitted_at = (
            snapshot.quotes["EUR_USD"].timestamp_utc
            - timedelta(seconds=1)
        ).isoformat()
        metadata["forecast_cycle_id"] = (
            f"pre-entry-forecast-refresh:{emitted_at}:{emitted_at}"
        )
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=1000,
            entry=1.17306,
            tp=1.17450,
            sl=1.17250,
            thesis="S5 hash forgery regression",
            metadata=metadata,
        )
        proof = execution_module._forecast_s5_no_touch_proof(
            client,
            intent=intent,
            snapshot=snapshot,
        )
        self.assertEqual(proof["status"], "PASSED")
        forged = json.loads(json.dumps(proof))
        forged["candle_rows"][-1]["h"] = 1.17460
        forged["max_high"] = 1.17460
        forged["candle_rows_sha256"] = (
            execution_module._canonical_json_sha256(
                forged["candle_rows"]
            )
        )
        forged_material = dict(forged)
        forged_material.pop("proof_sha256", None)
        forged["proof_sha256"] = execution_module._canonical_json_sha256(
            forged_material
        )
        self.assertFalse(
            execution_module._forecast_s5_path_proof_valid(
                forged,
                intent=intent,
                snapshot=snapshot,
                bypass_reason=None,
            )
        )
        missing_rail_metadata = dict(metadata)
        missing_rail_metadata.pop("forecast_target_price")
        self.assertFalse(
            execution_module._forecast_s5_path_proof_valid(
                proof,
                intent=replace(intent, metadata=missing_rail_metadata),
                snapshot=snapshot,
                bypass_reason=None,
            )
        )
        original_get_json = client.get_json

        def boolean_ohlc(path, query=None):
            payload = original_get_json(path, query)
            payload["candles"][-1]["mid"]["o"] = True
            return payload

        with patch.object(client, "get_json", side_effect=boolean_ohlc):
            boolean_proof = execution_module._forecast_s5_no_touch_proof(
                client,
                intent=intent,
                snapshot=snapshot,
            )
        self.assertEqual(boolean_proof["status"], "BLOCKED")
        self.assertEqual(
            boolean_proof["reason"],
            "FORECAST_S5_CANDLE_INVALID",
        )

        multi_metadata = dict(metadata)
        multi_emitted_at = (
            snapshot.quotes["EUR_USD"].timestamp_utc
            - timedelta(seconds=6)
        ).isoformat()
        multi_metadata["forecast_cycle_id"] = (
            f"pre-entry-forecast-refresh:{multi_emitted_at}:{multi_emitted_at}"
        )

        def incomplete_interior(path, query=None):
            payload = original_get_json(path, query)
            self.assertGreaterEqual(len(payload["candles"]), 2)
            payload["candles"][0]["complete"] = False
            return payload

        with patch.object(client, "get_json", side_effect=incomplete_interior):
            incomplete_proof = execution_module._forecast_s5_no_touch_proof(
                client,
                intent=replace(intent, metadata=multi_metadata),
                snapshot=snapshot,
            )
        self.assertEqual(incomplete_proof["status"], "BLOCKED")
        self.assertEqual(
            incomplete_proof["reason"],
            "FORECAST_S5_COVERAGE_INCOMPLETE",
        )
        stale_metadata = dict(metadata)
        stale_emitted_at = (
            snapshot.quotes["EUR_USD"].timestamp_utc
            - timedelta(seconds=901)
        ).isoformat()
        stale_metadata["forecast_cycle_id"] = (
            f"pre-entry-forecast-refresh:{stale_emitted_at}:{stale_emitted_at}"
        )
        stale_proof = execution_module._forecast_s5_no_touch_proof(
            client,
            intent=replace(intent, metadata=stale_metadata),
            snapshot=snapshot,
        )
        self.assertEqual(stale_proof["status"], "BLOCKED")
        self.assertEqual(
            stale_proof["reason"],
            "FORECAST_S5_WINDOW_OUT_OF_RANGE",
        )

    def test_off_grid_dependent_rails_block_before_post_direct_and_batch(self) -> None:
        metadata = _ordinary_claim_metadata(
            vehicle="MARKET",
            numeric_forecast=True,
        )
        raw_tp = 1.1739024885932492
        raw_sl = 1.1724109430942786
        for batch in (False, True):
            with self.subTest(batch=batch), tempfile.TemporaryDirectory() as tmp:
                client = ReconciliationExecutionClient()
                summary, order, _ = self._run_real_numeric_gateway(
                    root=Path(tmp),
                    client=client,
                    batch=batch,
                    metadata=metadata,
                    tp=raw_tp,
                    sl=raw_sl,
                )
                self.assertFalse(summary.sent)
                self.assertEqual(client.orders, [])
                self.assertIn(
                    "BROKER_PRICE_PRECISION_MISMATCH",
                    {issue["code"] for issue in order["risk_issues"]},
                )
                request = order["order_request"]
                self.assertEqual(request["takeProfitOnFill"]["price"], "1.17390")
                self.assertEqual(request["stopLossOnFill"]["price"], "1.17241")
                self.assertEqual(
                    order["pre_post_reconciliation"]["status"],
                    "NOT_RUN",
                )

    def test_nonnumeric_off_grid_market_and_pending_prices_block_direct_and_batch(self) -> None:
        cases = (
            {
                "name": "market_attached_tp",
                "order_type": "MARKET",
                "entry": None,
                "tp": 1.1745001,
                "metadata": {
                    "desk": "trend_trader",
                    "campaign_role": "NOW",
                },
            },
            {
                "name": "pending_entry",
                "order_type": "STOP-ENTRY",
                "entry": 1.1733001,
                "tp": 1.17450,
                "metadata": _ordinary_claim_metadata(),
            },
        )
        for batch in (False, True):
            for case in cases:
                with (
                    self.subTest(batch=batch, case=case["name"]),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    root = Path(tmp)
                    lane_id = "lane:EUR_USD:LONG"
                    intents = _intents(
                        root,
                        lane_id=lane_id,
                        order_type=case["order_type"],
                        entry=case["entry"],
                        tp=case["tp"],
                        metadata=case["metadata"],
                    )
                    client = FakeExecutionClient()
                    gateway = LiveOrderGateway(
                        client=client,
                        strategy_profile=_profile(root),
                        output_path=root / "request.json",
                        report_path=root / "report.md",
                        live_enabled=True,
                    )
                    summary = (
                        gateway.run_batch(
                            intents_path=intents,
                            lane_ids=(lane_id,),
                            send=True,
                            confirm_live=True,
                        )
                        if batch
                        else gateway.run(
                            intents_path=intents,
                            lane_id=lane_id,
                            send=True,
                            confirm_live=True,
                        )
                    )
                    self.assertFalse(summary.sent)
                    self.assertEqual(client.orders, [])
                    payload = json.loads((root / "request.json").read_text())
                    order = payload["orders"][0] if batch else payload
                    self.assertIn(
                        "BROKER_PRICE_PRECISION_MISMATCH",
                        {issue["code"] for issue in order["risk_issues"]},
                    )


    def test_final_sync_failure_retains_reservation_without_direct_or_batch_post(self) -> None:
        original_reserve = LiveOrderGateway._reserve_ordinary_entry_post
        for batch in (False, True):
            with self.subTest(batch=batch), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                lane_id = "lane:EUR_USD:LONG"
                target_state, target_report, ledger_path, ledger_report = (
                    _reconciliation_files(root, edge_vehicle="MARKET")
                )
                intents = _intents(
                    root,
                    lane_id=lane_id,
                    order_type="MARKET",
                    metadata=_ordinary_claim_metadata(
                        vehicle="MARKET",
                        numeric_forecast=True,
                    ),
                )
                client = PostReservationLedgerFailureClient()

                def reserve_then_fail(gateway_self, **kwargs):
                    result = original_reserve(gateway_self, **kwargs)
                    if result.issue is None:
                        client.fail_after_reservation = True
                    return result

                gateway = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    target_state_path=target_state,
                    target_report_path=target_report,
                    execution_ledger_db_path=ledger_path,
                    execution_ledger_report_path=ledger_report,
                    verified_decision_path=_write_ordinary_verified_decision(
                        root,
                        lane_id=lane_id,
                    ),
                    live_enabled=True,
                )
                with (
                    patch.object(
                        LiveOrderGateway,
                        "_pre_post_reconcile",
                        self._original_pre_post_reconcile,
                    ),
                    patch.object(
                        LiveOrderGateway,
                        "_final_pre_post_boundary",
                        self._original_final_pre_post_boundary,
                    ),
                    patch.object(
                        LiveOrderGateway,
                        "_reserve_ordinary_entry_post",
                        reserve_then_fail,
                    ),
                ):
                    if batch:
                        summary = gateway.run_batch(
                            intents_path=intents,
                            lane_ids=(lane_id,),
                            send=True,
                            confirm_live=True,
                        )
                    else:
                        summary = gateway.run(
                            intents_path=intents,
                            lane_id=lane_id,
                            send=True,
                            confirm_live=True,
                        )

                self.assertFalse(summary.sent)
                self.assertEqual(client.orders, [])
                payload = json.loads((root / "request.json").read_text())
                order = payload["orders"][0] if batch else payload
                self.assertIn(
                    "FINAL_PRE_POST_LEDGER_SYNC_FAILED",
                    {issue["code"] for issue in order["risk_issues"]},
                )
                self.assertEqual(
                    order["ordinary_entry_claim"]["status"],
                    "FINAL_BOUNDARY_BLOCKED_RESERVATION_RETAINED",
                )

    def test_staged_request_does_not_claim_gpt_authorization_without_verified_allocation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = LiveOrderGateway(
                client=FakeExecutionClient(),
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
            ).run(intents_path=_intents(root), send=False)

            self.assertEqual(summary.status, "STAGED")
            evidence = json.loads((root / "request.json").read_text())["sizing_evidence"]
            self.assertEqual(evidence["caller_size_multiple"], 1.0)
            self.assertEqual(evidence["caller_preclip_units"], 1000)
            self.assertFalse(evidence["capital_allocation_validated"])
            self.assertIsNone(evidence.get("authorized_size_multiple"))
            self.assertIsNone(evidence.get("authorized_units"))
            self.assertIsNone(evidence.get("allocation_status"))

    def test_stages_oanda_stop_order_without_sending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
            ).run(intents_path=_intents(root))

            self.assertEqual(summary.status, "STAGED")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders, [])
            payload = json.loads((root / "request.json").read_text())
            order = payload["order_request"]
            self.assertEqual(order["type"], "STOP")
            self.assertEqual(order["instrument"], "EUR_USD")
            self.assertEqual(order["units"], "1000")
            self.assertEqual(order["price"], "1.17330")
            self.assertEqual(order["takeProfitOnFill"]["price"], "1.17450")
            self.assertEqual(order["stopLossOnFill"]["price"], "1.17250")
            self.assertTrue(order["clientExtensions"]["id"].startswith("qrv1-EURUSD-L-"))
            self.assertEqual(order["clientExtensions"]["tag"], "trader")
            self.assertIn("lane=lane:EUR_USD:LONG", order["clientExtensions"]["comment"])
            self.assertEqual(order["tradeClientExtensions"]["tag"], "trader")
            self.assertIn("lane=lane:EUR_USD:LONG", order["tradeClientExtensions"]["comment"])

    def test_stages_oanda_market_order_without_entry_price(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
            ).run(intents_path=_intents(root, order_type="MARKET"), lane_id="lane:EUR_USD:LONG")

            self.assertEqual(summary.status, "STAGED")
            self.assertFalse(summary.sent)
            payload = json.loads((root / "request.json").read_text())
            order = payload["order_request"]
            self.assertEqual(order["type"], "MARKET")
            self.assertEqual(order["timeInForce"], "FOK")
            self.assertNotIn("price", order)

    def test_selected_lane_metadata_mismatch_blocks_before_broker_post(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            selected_lane = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT"
            submitted_sibling_lane = (
                "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET"
            )

            with patch(
                "quant_rabbit.risk.resolve_forecast_confidence_floor_state",
                return_value={
                    "status": "ACTIVE_OVERRIDE",
                    "resolved_value": 0.70,
                    "override": {"lane_id": selected_lane},
                },
            ) as resolve_floor:
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    live_enabled=True,
                ).run(
                    intents_path=_intents(
                        root,
                        order_type="LIMIT",
                        lane_id=selected_lane,
                        metadata={
                            "desk": "trend_trader",
                            "campaign_role": "NOW",
                            "lane_id": submitted_sibling_lane,
                            "forecast_confidence": 0.66,
                        },
                    ),
                    lane_id=selected_lane,
                    send=True,
                    confirm_live=True,
                )

            self.assertEqual(summary.status, "BLOCKED")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders, [])
            result = json.loads((root / "request.json").read_text())
            codes = {issue["code"] for issue in result["risk_issues"]}
            self.assertIn("GUARDIAN_TUNING_FORECAST_CONFIDENCE_FLOOR", codes)
            self.assertTrue(
                any(
                    invocation.kwargs.get("lane_id") == selected_lane
                    for invocation in resolve_floor.call_args_list
                )
            )
            mismatch = next(
                issue
                for issue in result["risk_issues"]
                if issue["code"] == "SELECTED_LANE_METADATA_MISMATCH"
            )
            self.assertIn(selected_lane, mismatch["message"])
            self.assertIn(submitted_sibling_lane, mismatch["message"])

    def test_selected_lane_metadata_exact_match_remains_stageable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected_lane = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT"

            with patch(
                "quant_rabbit.risk.resolve_forecast_confidence_floor_state",
                side_effect=lambda **kwargs: {
                    "status": "NO_OVERRIDE",
                    "resolved_value": kwargs["fallback"],
                },
            ):
                summary = LiveOrderGateway(
                    client=FakeExecutionClient(),
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                ).run(
                    intents_path=_intents(
                        root,
                        order_type="LIMIT",
                        lane_id=selected_lane,
                        metadata={
                            "desk": "trend_trader",
                            "campaign_role": "NOW",
                            "lane_id": selected_lane,
                        },
                    ),
                    lane_id=selected_lane,
                )

            self.assertEqual(summary.status, "STAGED")
            result = json.loads((root / "request.json").read_text())
            self.assertNotIn(
                "SELECTED_LANE_METADATA_MISMATCH",
                {issue["code"] for issue in result["risk_issues"]},
            )

    def test_batch_selected_lane_metadata_mismatch_blocks_before_broker_post(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            selected_lane = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT"

            with patch(
                "quant_rabbit.risk.resolve_forecast_confidence_floor_state",
                side_effect=lambda **kwargs: {
                    "status": "NO_OVERRIDE",
                    "resolved_value": kwargs["fallback"],
                },
            ):
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    live_enabled=True,
                ).run_batch(
                    intents_path=_intents(
                        root,
                        order_type="LIMIT",
                        lane_id=selected_lane,
                        metadata={
                            "desk": "trend_trader",
                            "campaign_role": "NOW",
                            "lane_id": (
                                "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET"
                            ),
                        },
                    ),
                    lane_ids=(selected_lane,),
                    send=True,
                    confirm_live=True,
                )

            self.assertEqual(summary.status, "BLOCKED")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders, [])
            result = json.loads((root / "request.json").read_text())
            self.assertIn(
                "SELECTED_LANE_METADATA_MISMATCH",
                {issue["code"] for issue in result["orders"][0]["risk_issues"]},
            )

    def test_unresolved_guardian_receipt_issue_blocks_send_before_broker_post(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            watchdog = root / "watchdog.json"
            consumption = root / "guardian_receipt_consumption.json"
            operator_review = root / "guardian_receipt_operator_review.json"
            broker_snapshot = root / "broker_snapshot.json"
            _write_gateway_guardian_watchdog_issue(watchdog)
            os.environ["QR_GUARDIAN_RECEIPT_WATCHDOG_PATH"] = str(watchdog)
            os.environ["QR_GUARDIAN_RECEIPT_CONSUMPTION_PATH"] = str(consumption)
            os.environ["QR_GUARDIAN_RECEIPT_OPERATOR_REVIEW_PATH"] = str(operator_review)
            os.environ["QR_GUARDIAN_RECEIPT_BROKER_SNAPSHOT_PATH"] = str(broker_snapshot)

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
                qr_trader_run_watchdog_path=watchdog,
                guardian_receipt_consumption_path=consumption,
                guardian_receipt_operator_review_path=operator_review,
                broker_snapshot_path=broker_snapshot,
            ).run(
                intents_path=_intents(
                    root,
                    pair="EUR_USD",
                    lane_id="campaign_exposure_recovery:EUR_USD:LONG:TREND_CONTINUATION",
                    metadata={"desk": "campaign_exposure_recovery", "campaign_role": "NOW"},
                ),
                lane_id="campaign_exposure_recovery:EUR_USD:LONG:TREND_CONTINUATION",
                send=True,
                confirm_live=True,
            )

            self.assertEqual(summary.status, "BLOCKED")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders, [])
            payload = json.loads((root / "request.json").read_text())
            codes = {issue["code"] for issue in payload["risk_issues"]}
            self.assertIn("GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED", codes)
            self.assertIn(
                "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
                (root / "report.md").read_text(),
            )

    def test_market_close_leak_family_blocks_campaign_recovery_send_before_broker_post(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            lane_id = "campaign_exposure_recovery:EUR_USD:LONG:BREAKOUT_FAILURE"
            intents = _intents(
                root,
                order_type="LIMIT",
                lane_id=lane_id,
                metadata={
                    "desk": "campaign_exposure_recovery",
                    "campaign_role": "NOW",
                    "positive_rotation_mode": "TP_PROVEN_HARVEST",
                    "capture_take_profit_scope_key": "EUR_USD|LONG|BREAKOUT_FAILURE|TAKE_PROFIT_ORDER",
                },
            )
            payload = json.loads(intents.read_text())
            payload["results"][0]["intent"]["market_context"]["method"] = "BREAKOUT_FAILURE"
            payload["results"][0]["intent"]["market_context"]["regime"] = "BREAKOUT_FAILURE leak family"
            intents.write_text(json.dumps(payload))

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
            ).run(intents_path=intents, lane_id=lane_id, send=True, confirm_live=True)

            self.assertEqual(summary.status, "BLOCKED")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders, [])
            staged = json.loads((root / "request.json").read_text())
            codes = {issue["code"] for issue in staged["risk_issues"]}
            self.assertIn(MARKET_CLOSE_LEAK_FAMILY_BLOCK_CODE, codes)

    def test_missing_quote_blocks_send_without_guardian_artifact_leak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root, pair="AUD_USD"),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
            ).run(
                intents_path=_intents(root, pair="AUD_USD", lane_id="lane:AUD_USD:LONG"),
                lane_id="lane:AUD_USD:LONG",
                send=True,
                confirm_live=True,
            )

            self.assertEqual(summary.status, "BLOCKED")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders, [])
            payload = json.loads((root / "request.json").read_text())
            codes = {issue["code"] for issue in payload["risk_issues"]}
            self.assertIn("MISSING_QUOTE", codes)
            self.assertNotIn("GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED", codes)
            self.assertNotIn("GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER_BLOCKS_NEW_ENTRY", codes)

    def test_verified_wait_receipt_blocks_fresh_entry_send_before_broker_post(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            verified = root / "gpt_decision.json"
            verified.write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "ACCEPTED",
                        "decision": {"action": "WAIT", "selected_lane_id": None},
                        "verification_issues": [],
                    }
                )
                + "\n"
            )
            client = FakeExecutionClient()

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
                verified_decision_path=verified,
            ).run(
                intents_path=_intents(root),
                lane_id="lane:EUR_USD:LONG",
                send=True,
                confirm_live=True,
            )

            self.assertEqual(summary.status, "BLOCKED")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders, [])
            payload = json.loads((root / "request.json").read_text())
            codes = {issue["code"] for issue in payload["risk_issues"]}
            self.assertIn("GPT_FRESH_TRADE_RECEIPT_REQUIRED_FOR_LIVE_SEND", codes)

    def test_guardian_wake_hourly_schedule_alone_cannot_stage_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = root / "guardian_action_receipt.json"
            receipt.write_text(
                json.dumps(
                    {
                        "action": "TRADE",
                        "new_information": True,
                        "event_id": "event-hourly",
                        "pair": "EUR_USD",
                        "thesis": "trend continuation",
                        "thesis_state": "ALIVE",
                        "reason": "scheduled hour arrived, so place the trade",
                        "invalidation": "break back below support",
                        "harvest_trigger": "upper rail",
                        "gateway_required": True,
                    }
                )
            )

            summary = LiveOrderGateway(
                client=FakeExecutionClient(),
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                guardian_action_receipt_path=receipt,
            ).run(
                intents_path=_intents(
                    root,
                    metadata={
                        "desk": "trend_trader",
                        "campaign_role": "NOW",
                        "guardian_event_id": "event-hourly",
                        "guardian_event_wake": True,
                    },
                    order_type="MARKET",
                ),
                lane_id="lane:EUR_USD:LONG",
            )

            self.assertEqual(summary.status, "BLOCKED")
            payload = json.loads((root / "request.json").read_text())
            self.assertIn("GUARDIAN_ACTION_SCHEDULE_ONLY", {issue["code"] for issue in payload["risk_issues"]})

    def test_guardian_wake_intent_requires_action_receipt_before_gateway(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = LiveOrderGateway(
                client=FakeExecutionClient(),
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                guardian_action_receipt_path=root / "missing_guardian_action_receipt.json",
            ).run(
                intents_path=_intents(
                    root,
                    metadata={
                        "desk": "trend_trader",
                        "campaign_role": "NOW",
                        "guardian_event_id": "event-missing",
                        "guardian_event_wake": True,
                    },
                    order_type="MARKET",
                ),
                lane_id="lane:EUR_USD:LONG",
            )

            self.assertEqual(summary.status, "BLOCKED")
            payload = json.loads((root / "request.json").read_text())
            self.assertIn("GUARDIAN_ACTION_RECEIPT_REQUIRED", {issue["code"] for issue in payload["risk_issues"]})

    def test_sl_lint_blocks_jpy_major_figure_battle_zone_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            now = datetime.now(timezone.utc)
            account = client.snapshot_value.account
            assert account is not None
            client.snapshot_value = BrokerSnapshot(
                fetched_at_utc=now,
                positions=(),
                orders=(),
                quotes={
                    "USD_JPY": Quote("USD_JPY", bid=161.884, ask=161.894, timestamp_utc=now),
                },
                account=account,
            )
            intents = _intents(
                root,
                order_type="MARKET",
                metadata={
                    "desk": "trend_trader",
                    "campaign_role": "NOW",
                    "tp_atr_pips": 6.0,
                    "level_cluster_radius_pips": 8.0,
                    "nearest_levels_above": [
                        {"price": 162.0, "source": "levels:round_number"}
                    ],
                    "event_risk": "JPY intervention risk near 162.00",
                },
            )
            payload = json.loads(intents.read_text())
            result = payload["results"][0]
            result["lane_id"] = "lane:USD_JPY:SHORT"
            intent = result["intent"]
            intent.update(
                {
                    "pair": "USD_JPY",
                    "side": "SHORT",
                    "entry": 161.884,
                    "tp": 161.720,
                    "sl": 161.941,
                    "thesis": "JPY strength reversal fade below 162.00",
                    "market_context": {
                        "regime": "TREND_CONTINUATION campaign lane",
                        "narrative": "JPY intervention risk and reversal pressure near 162.00",
                        "chart_story": "fade major figure stop run",
                        "method": "TREND_CONTINUATION",
                        "invalidation": "clean acceptance above the major figure",
                    },
                }
            )
            intents.write_text(json.dumps(payload))

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root, direction="SHORT", pair="USD_JPY"),
                output_path=root / "request.json",
                report_path=root / "report.md",
            ).run(intents_path=intents, lane_id="lane:USD_JPY:SHORT")

            self.assertEqual(summary.status, "BLOCKED")
            staged = json.loads((root / "request.json").read_text())
            self.assertEqual(staged["sl_lint"]["status"], "BLOCK")
            codes = {issue["code"] for issue in staged["risk_issues"]}
            self.assertIn("SL_LINT_MAJOR_FIGURE_BATTLE_ZONE", codes)
            self.assertIn("SL_LINT_EVENT_INTERVENTION_ZONE", codes)
            self.assertEqual(client.orders, [])

    def test_sl_lint_blocks_same_jpy_theme_without_theme_level_invalidation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            now = datetime.now(timezone.utc)
            account = client.snapshot_value.account
            assert account is not None
            client.snapshot_value = BrokerSnapshot(
                fetched_at_utc=now,
                positions=(
                    BrokerPosition(
                        trade_id="usd-jpy-short",
                        pair="USD_JPY",
                        side=Side.SHORT,
                        units=1000,
                        entry_price=161.800,
                        unrealized_pl_jpy=-120.0,
                        stop_loss=162.200,
                        owner=Owner.TRADER,
                    ),
                ),
                orders=(),
                quotes={
                    "EUR_JPY": Quote("EUR_JPY", bid=185.500, ask=185.510, timestamp_utc=now),
                    "USD_JPY": Quote("USD_JPY", bid=161.884, ask=161.894, timestamp_utc=now),
                },
                account=account,
            )
            intents = _intents(
                root,
                order_type="MARKET",
                metadata={
                    "desk": "trend_trader",
                    "campaign_role": "NOW",
                    "tp_atr_pips": 6.0,
                    "level_cluster_radius_pips": 8.0,
                },
            )
            payload = json.loads(intents.read_text())
            result = payload["results"][0]
            result["lane_id"] = "lane:EUR_JPY:SHORT"
            intent = result["intent"]
            intent.update(
                {
                    "pair": "EUR_JPY",
                    "side": "SHORT",
                    "entry": 185.500,
                    "tp": 185.200,
                    "sl": 185.610,
                    "thesis": "same JPY strength reversal theme on EUR_JPY",
                    "market_context": {
                        "regime": "TREND_CONTINUATION campaign lane",
                        "narrative": "JPY strength reversal theme already active",
                        "chart_story": "JPY crosses fading together",
                        "method": "TREND_CONTINUATION",
                        "invalidation": "theme acceptance against JPY strength",
                    },
                }
            )
            intents.write_text(json.dumps(payload))

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root, direction="SHORT", pair="EUR_JPY"),
                output_path=root / "request.json",
                report_path=root / "report.md",
            ).run(intents_path=intents, lane_id="lane:EUR_JPY:SHORT")

            self.assertEqual(summary.status, "BLOCKED")
            staged = json.loads((root / "request.json").read_text())
            self.assertEqual(staged["sl_lint"]["theme_group"], "JPY_STRENGTH_REVERSAL")
            codes = {issue["code"] for issue in staged["risk_issues"]}
            self.assertIn("SL_LINT_JPY_THEME_INVALIDATION_REQUIRED", codes)
            self.assertEqual(client.orders, [])

    def test_long_limit_crossed_favorably_reprices_passive_instead_of_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
            ).run(intents_path=_intents(root, order_type="LIMIT"), lane_id="lane:EUR_USD:LONG")

            self.assertEqual(summary.status, "STAGED")
            payload = json.loads((root / "request.json").read_text())
            self.assertEqual(payload["order_request"]["type"], "LIMIT")
            self.assertEqual(payload["order_request"]["price"], "1.17305")
            issue_codes = {issue["code"] for issue in payload["risk_issues"]}
            self.assertIn("LIMIT_ENTRY_REPRICED_PASSIVE", issue_codes)
            self.assertNotIn("LIMIT_ENTRY_NOT_BELOW_MARKET", issue_codes)

    def test_short_limit_crossed_favorably_reprices_passive_instead_of_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            intents = _intents(root, order_type="LIMIT")
            payload = json.loads(intents.read_text())
            result = payload["results"][0]
            result["lane_id"] = "lane:EUR_USD:SHORT"
            intent = result["intent"]
            intent["side"] = "SHORT"
            intent["entry"] = 1.17290
            intent["tp"] = 1.17140
            intent["sl"] = 1.17380
            intent["thesis"] = "bear trend continuation"
            intent["market_context"] = {
                "regime": "TREND_CONTINUATION campaign lane",
                "narrative": "downtrend continuation pressure",
                "chart_story": "lower highs and continuation pressure",
                "method": "TREND_CONTINUATION",
                "invalidation": "SL trades",
            }
            intents.write_text(json.dumps(payload))

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root, direction="SHORT"),
                output_path=root / "request.json",
                report_path=root / "report.md",
            ).run(intents_path=intents, lane_id="lane:EUR_USD:SHORT")

            self.assertEqual(summary.status, "STAGED")
            payload = json.loads((root / "request.json").read_text())
            self.assertEqual(payload["order_request"]["type"], "LIMIT")
            self.assertEqual(payload["order_request"]["units"], "-1000")
            self.assertEqual(payload["order_request"]["price"], "1.17299")
            issue_codes = {issue["code"] for issue in payload["risk_issues"]}
            self.assertIn("LIMIT_ENTRY_REPRICED_PASSIVE", issue_codes)
            self.assertNotIn("LIMIT_ENTRY_NOT_ABOVE_MARKET", issue_codes)

    def test_predictive_scout_crossed_limit_expires_instead_of_repricing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"QR_PREDICTIVE_SCOUT_LIVE_ENABLED": "1"},
        ):
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            _write_predictive_scout_policy(root)
            now = datetime.now(timezone.utc)
            intents = _predictive_scout_intents(data_root, now=now, crossed=True)

            summary = LiveOrderGateway(
                client=_predictive_scout_client(),
                strategy_profile=_profile(root, pair="USD_CAD"),
                output_path=root / "request.json",
                report_path=root / "report.md",
            ).run(intents_path=intents, lane_id=PREDICTIVE_SCOUT_LANE_ID)

            self.assertEqual(summary.status, "BLOCKED")
            payload = json.loads((root / "request.json").read_text())
            self.assertEqual(payload["order_request"]["price"], "1.41610")
            issue_codes = {issue["code"] for issue in payload["risk_issues"]}
            self.assertIn("PREDICTIVE_SCOUT_TRIGGER_CROSSED", issue_codes)
            self.assertNotIn("LIMIT_ENTRY_REPRICED_PASSIVE", issue_codes)

    def test_tp_proven_range_crossed_limit_expires_instead_of_repricing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            intents = _tp_proven_range_limit_intents(
                root,
                client=client,
                crossed=True,
            )

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
            ).run(
                intents_path=intents,
                lane_id=TP_PROVEN_RANGE_LANE_ID,
            )
            payload = json.loads((root / "request.json").read_text())

        self.assertEqual(summary.status, "BLOCKED")
        self.assertEqual(payload["order_request"]["price"], "1.17310")
        issue_codes = {issue["code"] for issue in payload["risk_issues"]}
        self.assertIn("TP_PROVEN_RANGE_TRIGGER_CROSSED", issue_codes)
        self.assertNotIn("LIMIT_ENTRY_REPRICED_PASSIVE", issue_codes)

    def test_predictive_scout_gateway_rechecks_policy_and_uses_gtd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "QR_PREDICTIVE_SCOUT_LIVE_ENABLED": "1",
                "QR_TRADER_DISABLE_SL_REPAIR": "1",
                "QR_NEW_ENTRY_INITIAL_SL": "0",
            },
        ):
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            _write_predictive_scout_policy(root)
            now = datetime.now(timezone.utc)
            metadata = _predictive_scout_metadata(now)
            intents = _predictive_scout_intents(data_root, now=now)

            summary = LiveOrderGateway(
                client=_predictive_scout_client(),
                strategy_profile=_profile(root, pair="USD_CAD"),
                output_path=root / "request.json",
                report_path=root / "report.md",
            ).run(intents_path=intents, lane_id=PREDICTIVE_SCOUT_LANE_ID)

            self.assertEqual(summary.status, "STAGED")
            staged = json.loads((root / "request.json").read_text())
            order = staged["order_request"]
            self.assertEqual(order["timeInForce"], "GTD")
            self.assertEqual(order["positionFill"], "OPEN_ONLY")
            self.assertEqual(order["stopLossOnFill"]["price"], "1.41520")
            expected_expiry = datetime.fromisoformat(
                metadata["predictive_scout_expires_at_utc"]
            ).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
            self.assertEqual(order["gtdTime"], expected_expiry)
            self.assertTrue(staged["predictive_scout"])
            scout_receipt = staged["predictive_scout_receipt"]
            self.assertTrue(scout_receipt["predictive_scout"])
            self.assertTrue(scout_receipt["predictive_scout_vehicle_id"].startswith("psv-"))
            self.assertTrue(scout_receipt["predictive_scout_experiment_id"].startswith("psx-"))
            self.assertEqual(scout_receipt["predictive_scout_risk_tier"], "DISCOVERY")
            self.assertEqual(scout_receipt["predictive_scout_nav_jpy_at_sizing"], 200_000.0)
            self.assertEqual(scout_receipt["predictive_scout_max_risk_pct_nav"], 0.10)
            self.assertEqual(
                scout_receipt["predictive_scout_planned_initial_risk_jpy"],
                75.6,
            )
            self.assertAlmostEqual(
                scout_receipt["predictive_scout_fresh_actual_initial_risk_jpy"],
                75.6,
                places=6,
            )
            self.assertEqual(
                scout_receipt["predictive_scout_active_initial_risk_jpy"],
                0.0,
            )
            self.assertAlmostEqual(
                scout_receipt["predictive_scout_aggregate_initial_risk_jpy"],
                75.6,
                places=6,
            )
            self.assertEqual(
                scout_receipt["predictive_scout_concurrent_risk_cap_jpy"],
                4000.0,
            )
            self.assertTrue(
                scout_receipt["predictive_scout_sizing_digest"].startswith("psd-")
            )
            self.assertEqual(
                {
                    "pair": scout_receipt["pair"],
                    "side": scout_receipt["side"],
                    "order_type": scout_receipt["order_type"],
                    "units": scout_receipt["units"],
                    "entry": scout_receipt["entry"],
                    "take_profit": scout_receipt["take_profit"],
                    "stop_loss": scout_receipt["stop_loss"],
                },
                {
                    "pair": "USD_CAD",
                    "side": "LONG",
                    "order_type": "LIMIT",
                    "units": 1000,
                    "entry": 1.41590,
                    "take_profit": 1.41690,
                    "stop_loss": 1.41520,
                },
            )
            self.assertEqual(
                scout_receipt["predictive_scout_source"],
                "BIDASK_REPLAY_PRECISION",
            )
            self.assertEqual(
                scout_receipt["forecast_cycle_id"],
                "test-usdcad-down-c050-065-h31-60",
            )
            self.assertEqual(
                scout_receipt["predictive_scout_rule_digest"],
                metadata["predictive_scout_rule_digest"],
            )
            self.assertEqual(
                scout_receipt["bidask_replay_precision_seed_rule"]["name"],
                PREDICTIVE_SCOUT_RULE_NAME,
            )
            comment = order["clientExtensions"]["comment"]
            self.assertLess(
                comment.index("role=BIDASK_REPLAY_CONTRARIAN_SCOUT"),
                comment.index(f"vehicle={scout_receipt['predictive_scout_vehicle_id']}"),
            )
            self.assertIn("lane=predictive_scout:", comment)
            self.assertIn(
                scout_receipt["predictive_scout_signal_id"],
                order["clientExtensions"]["id"],
            )
            self.assertIn(
                scout_receipt["predictive_scout_signal_id"],
                order["tradeClientExtensions"]["id"],
            )

    def test_predictive_scout_rejects_ai_size_multiple(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"QR_PREDICTIVE_SCOUT_LIVE_ENABLED": "1"},
        ):
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            _write_predictive_scout_policy(root)
            intents = _predictive_scout_intents(
                data_root,
                now=datetime.now(timezone.utc),
            )

            summary = LiveOrderGateway(
                client=_predictive_scout_client(),
                strategy_profile=_profile(root, pair="USD_CAD"),
                output_path=root / "request.json",
                report_path=root / "report.md",
            ).run(
                intents_path=intents,
                lane_id=PREDICTIVE_SCOUT_LANE_ID,
                size_multiple=1.25,
            )

            self.assertEqual(summary.status, "BLOCKED")
            staged = json.loads((root / "request.json").read_text())
            issue_codes = {issue["code"] for issue in staged["risk_issues"]}
            self.assertIn("PREDICTIVE_SCOUT_SIZE_MULTIPLE_FORBIDDEN", issue_codes)
            self.assertEqual(staged["requested_units"], 1000)
            self.assertEqual(staged["scaled_units"], 1000)

    def test_predictive_scout_gateway_blocks_forged_packaged_rule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"QR_PREDICTIVE_SCOUT_LIVE_ENABLED": "1"},
        ):
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            _write_predictive_scout_policy(root)
            intents = _predictive_scout_intents(
                data_root,
                now=datetime.now(timezone.utc),
            )
            payload = json.loads(intents.read_text())
            embedded_rule = payload["results"][0]["intent"]["metadata"][
                "bidask_replay_precision_seed_rule"
            ]
            embedded_rule["samples"] = int(embedded_rule["samples"]) + 1
            intents.write_text(json.dumps(payload))

            summary = LiveOrderGateway(
                client=_predictive_scout_client(),
                strategy_profile=_profile(root, pair="USD_CAD"),
                output_path=root / "request.json",
                report_path=root / "report.md",
            ).run(intents_path=intents, lane_id=PREDICTIVE_SCOUT_LANE_ID)

            self.assertEqual(summary.status, "BLOCKED")
            staged = json.loads((root / "request.json").read_text())
            issue_codes = {issue["code"] for issue in staged["risk_issues"]}
            self.assertIn("PREDICTIVE_SCOUT_RULE_NOT_LIVE_GRADE", issue_codes)

    def test_predictive_scout_gateway_blocks_method_and_desk_relabel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"QR_PREDICTIVE_SCOUT_LIVE_ENABLED": "1"},
        ):
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            _write_predictive_scout_policy(root)
            intents = _predictive_scout_intents(
                data_root,
                now=datetime.now(timezone.utc),
            )
            payload = json.loads(intents.read_text())
            intent = payload["results"][0]["intent"]
            intent["metadata"]["desk"] = "range_trader"
            intent["market_context"]["method"] = "RANGE_ROTATION"
            intents.write_text(json.dumps(payload))

            summary = LiveOrderGateway(
                client=_predictive_scout_client(),
                strategy_profile=_profile(root, pair="USD_CAD"),
                output_path=root / "request.json",
                report_path=root / "report.md",
            ).run(intents_path=intents, lane_id=PREDICTIVE_SCOUT_LANE_ID)
            staged = json.loads((root / "request.json").read_text())

        self.assertEqual(summary.status, "BLOCKED")
        issue_codes = {issue["code"] for issue in staged["risk_issues"]}
        self.assertIn("PREDICTIVE_SCOUT_METHOD_REQUIRED", issue_codes)
        self.assertIn("PREDICTIVE_SCOUT_DESK_REQUIRED", issue_codes)

    def test_predictive_scout_cannot_be_downgraded_by_stripping_flags(self) -> None:
        for strip_mode in ("marker_only", "reserved", "reserved_and_rule"):
            with self.subTest(strip_mode=strip_mode), tempfile.TemporaryDirectory() as tmp, patch.dict(
                os.environ,
                {"QR_PREDICTIVE_SCOUT_LIVE_ENABLED": "1"},
            ):
                root = Path(tmp)
                data_root = root / "data"
                data_root.mkdir()
                _write_predictive_scout_policy(root)
                intents = _predictive_scout_intents(
                    data_root,
                    now=datetime.now(timezone.utc),
                )
                payload = json.loads(intents.read_text())
                metadata = payload["results"][0]["intent"]["metadata"]
                if strip_mode != "marker_only":
                    for key in list(metadata):
                        if key.startswith("predictive_scout"):
                            metadata.pop(key)
                    metadata["campaign_role"] = "NOW"
                else:
                    metadata.pop("predictive_scout")
                if strip_mode == "reserved_and_rule":
                    metadata.pop("bidask_replay_precision_seed_rule")
                intents.write_text(json.dumps(payload))

                summary = LiveOrderGateway(
                    client=_predictive_scout_client(),
                    strategy_profile=_profile(root, pair="USD_CAD"),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                ).run(intents_path=intents, lane_id=PREDICTIVE_SCOUT_LANE_ID)

                self.assertEqual(summary.status, "BLOCKED")
                staged = json.loads((root / "request.json").read_text())
                issue_codes = {issue["code"] for issue in staged["risk_issues"]}
                self.assertIn("PREDICTIVE_SCOUT_MARKER_REQUIRED", issue_codes)

    def test_predictive_scout_blocks_when_global_tp_kill_switch_removes_attached_tp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "QR_PREDICTIVE_SCOUT_LIVE_ENABLED": "1",
                "QR_DISABLE_AUTO_TP": "1",
            },
        ):
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            _write_predictive_scout_policy(root)
            intents = _predictive_scout_intents(
                data_root,
                now=datetime.now(timezone.utc),
            )

            summary = LiveOrderGateway(
                client=_predictive_scout_client(),
                strategy_profile=_profile(root, pair="USD_CAD"),
                output_path=root / "request.json",
                report_path=root / "report.md",
            ).run(intents_path=intents, lane_id=PREDICTIVE_SCOUT_LANE_ID)

            self.assertEqual(summary.status, "BLOCKED")
            staged = json.loads((root / "request.json").read_text())
            self.assertNotIn("takeProfitOnFill", staged["order_request"])
            issue_codes = {issue["code"] for issue in staged["risk_issues"]}
            self.assertIn("PREDICTIVE_SCOUT_ATTACHED_TP_MISMATCH", issue_codes)

    def test_predictive_scout_live_send_requires_verified_ai_trade_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "QR_PREDICTIVE_SCOUT_LIVE_ENABLED": "1",
                "QR_REQUIRE_POSITION_GUARDIAN_ACTIVE": "0",
            },
        ):
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            _write_predictive_scout_policy(root)
            intents = _predictive_scout_intents(
                data_root,
                now=datetime.now(timezone.utc),
            )
            client = _predictive_scout_client()

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root, pair="USD_CAD"),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
                predictive_scout_canonical_ledger_db_path=data_root / "execution_ledger.db",
            ).run(
                intents_path=intents,
                lane_id=PREDICTIVE_SCOUT_LANE_ID,
                send=True,
                confirm_live=True,
            )
            staged = json.loads((root / "request.json").read_text())
            with closing(sqlite3.connect(data_root / "execution_ledger.db")) as con, con:
                claims = con.execute(
                    "SELECT COUNT(*) FROM predictive_scout_signal_claims"
                ).fetchone()[0]

        self.assertFalse(summary.sent)
        self.assertEqual(client.orders, [])
        self.assertEqual(claims, 0)
        self.assertIn(
            "GPT_FRESH_TRADE_RECEIPT_REQUIRED_FOR_LIVE_SEND",
            {issue["code"] for issue in staged["risk_issues"]},
        )

    def test_predictive_scout_ai_receipt_is_trade_signal_and_market_bound(self) -> None:
        cases = (
            ("single_add", "single", "ADD", "LONG", False, 0, "GPT_FRESH_TRADE_RECEIPT_REQUIRED_FOR_LIVE_SEND"),
            ("batch_add", "batch", "ADD", "LONG", False, 0, "GPT_FRESH_TRADE_RECEIPT_REQUIRED_FOR_LIVE_SEND"),
            ("wrong_market_side", "single", "TRADE", "SHORT", False, 0, "GPT_SCOUT_MARKET_READ_MISMATCH_FOR_LIVE_SEND"),
            ("wrong_cycle", "single", "TRADE", "LONG", True, 0, "GPT_SCOUT_SIGNAL_MISMATCH_FOR_LIVE_SEND"),
            ("future_receipt", "single", "TRADE", "LONG", False, 30, "GPT_SCOUT_RECEIPT_FROM_FUTURE"),
        )
        for (
            label,
            mode,
            action,
            market_direction,
            wrong_cycle,
            future_days,
            expected_code,
        ) in cases:
            with (
                self.subTest(label=label),
                tempfile.TemporaryDirectory() as tmp,
                patch.dict(
                    os.environ,
                    {
                        "QR_PREDICTIVE_SCOUT_LIVE_ENABLED": "1",
                        "QR_REQUIRE_POSITION_GUARDIAN_ACTIVE": "0",
                    },
                ),
            ):
                root = Path(tmp)
                data_root = root / "data"
                data_root.mkdir()
                _write_predictive_scout_policy(root)
                intents = _predictive_scout_intents(
                    data_root,
                    now=datetime.now(timezone.utc),
                )
                generated_at = (
                    datetime.now(timezone.utc) + timedelta(days=future_days)
                ).isoformat()
                verified = _write_predictive_scout_verified_decision(
                    root,
                    intents_path=intents,
                    action=action,
                    market_direction=market_direction,
                    generated_at_utc=generated_at,
                )
                if wrong_cycle:
                    verified_payload = json.loads(verified.read_text())
                    verified_payload["input_packet"]["lanes"][0]["predictive_scout"][
                        "forecast_cycle_id"
                    ] = "stale-forecast-cycle"
                    verified.write_text(json.dumps(verified_payload), encoding="utf-8")
                client = _predictive_scout_client()
                gateway = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root, pair="USD_CAD"),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    live_enabled=True,
                    verified_decision_path=verified,
                    predictive_scout_canonical_ledger_db_path=(
                        data_root / "execution_ledger.db"
                    ),
                )
                if mode == "batch":
                    summary = gateway.run_batch(
                        intents_path=intents,
                        lane_ids=(PREDICTIVE_SCOUT_LANE_ID,),
                        send=True,
                        confirm_live=True,
                    )
                else:
                    summary = gateway.run(
                        intents_path=intents,
                        lane_id=PREDICTIVE_SCOUT_LANE_ID,
                        send=True,
                        confirm_live=True,
                    )
                staged = json.loads((root / "request.json").read_text())
                risk_issues = (
                    staged["orders"][0]["risk_issues"]
                    if mode == "batch"
                    else staged["risk_issues"]
                )

                self.assertFalse(summary.sent)
                self.assertEqual(client.orders, [])
                self.assertIn(expected_code, {issue["code"] for issue in risk_issues})

    def test_predictive_scout_manual_same_pair_blocks_gateway_post(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "QR_PREDICTIVE_SCOUT_LIVE_ENABLED": "1",
                "QR_REQUIRE_POSITION_GUARDIAN_ACTIVE": "0",
            },
        ):
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            _write_predictive_scout_policy(root)
            intents = _predictive_scout_intents(
                data_root,
                now=datetime.now(timezone.utc),
            )
            verified = _write_predictive_scout_verified_decision(
                root,
                intents_path=intents,
            )
            client = _predictive_scout_client()
            snapshot = client.snapshot_value
            client.snapshot_value = BrokerSnapshot(
                fetched_at_utc=snapshot.fetched_at_utc,
                positions=(
                    BrokerPosition(
                        trade_id="manual-usdcad-short",
                        pair="USD_CAD",
                        side=Side.SHORT,
                        units=10_000,
                        entry_price=1.42,
                        owner=Owner.OPERATOR_MANUAL,
                    ),
                ),
                orders=snapshot.orders,
                quotes=snapshot.quotes,
                account=snapshot.account,
                home_conversions=snapshot.home_conversions,
            )

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root, pair="USD_CAD"),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
                verified_decision_path=verified,
                predictive_scout_canonical_ledger_db_path=data_root / "execution_ledger.db",
            ).run(
                intents_path=intents,
                lane_id=PREDICTIVE_SCOUT_LANE_ID,
                send=True,
                confirm_live=True,
            )
            staged = json.loads((root / "request.json").read_text())

        self.assertFalse(summary.sent)
        self.assertEqual(client.orders, [])
        self.assertIn(
            "PREDICTIVE_SCOUT_MANUAL_PAIR_BLOCKED",
            {issue["code"] for issue in staged["risk_issues"]},
        )

    def test_predictive_scout_tagless_same_pair_pending_blocks_gateway_post(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "QR_PREDICTIVE_SCOUT_LIVE_ENABLED": "1",
                "QR_REQUIRE_POSITION_GUARDIAN_ACTIVE": "0",
            },
        ):
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            _write_predictive_scout_policy(root)
            intents = _predictive_scout_intents(
                data_root,
                now=datetime.now(timezone.utc),
            )
            verified = _write_predictive_scout_verified_decision(
                root,
                intents_path=intents,
            )
            client = _predictive_scout_client()
            snapshot = client.snapshot_value
            client.snapshot_value = BrokerSnapshot(
                fetched_at_utc=snapshot.fetched_at_utc,
                positions=snapshot.positions,
                orders=(
                    BrokerOrder(
                        order_id="oanda-ui-tagless-usdcad-limit",
                        pair="USD_CAD",
                        order_type="LIMIT",
                        owner=Owner.UNKNOWN,
                        raw={
                            "id": "oanda-ui-tagless-usdcad-limit",
                            "instrument": "USD_CAD",
                            "type": "LIMIT_ORDER",
                            "units": "1000",
                        },
                    ),
                ),
                quotes=snapshot.quotes,
                account=snapshot.account,
                home_conversions=snapshot.home_conversions,
            )

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root, pair="USD_CAD"),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
                verified_decision_path=verified,
                predictive_scout_canonical_ledger_db_path=data_root / "execution_ledger.db",
            ).run(
                intents_path=intents,
                lane_id=PREDICTIVE_SCOUT_LANE_ID,
                send=True,
                confirm_live=True,
            )
            staged = json.loads((root / "request.json").read_text())

        self.assertFalse(summary.sent)
        self.assertEqual(client.orders, [])
        self.assertIn(
            "PREDICTIVE_SCOUT_MANUAL_PAIR_BLOCKED",
            {issue["code"] for issue in staged["risk_issues"]},
        )

    def test_predictive_scout_expiry_is_rechecked_immediately_before_post(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "QR_PREDICTIVE_SCOUT_LIVE_ENABLED": "1",
                "QR_REQUIRE_POSITION_GUARDIAN_ACTIVE": "0",
            },
        ):
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            _write_predictive_scout_policy(root)
            intents = _predictive_scout_intents(
                data_root,
                now=datetime.now(timezone.utc),
            )
            verified = _write_predictive_scout_verified_decision(
                root,
                intents_path=intents,
            )
            client = _predictive_scout_client()

            with patch.object(
                execution_module,
                "_predictive_scout_pre_post_issues",
                return_value=[
                    {
                        "severity": "BLOCK",
                        "code": "PREDICTIVE_SCOUT_EXPIRED_BEFORE_POST",
                        "message": "expired at final pre-POST check",
                    }
                ],
            ) as final_expiry_check:
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root, pair="USD_CAD"),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    live_enabled=True,
                    verified_decision_path=verified,
                    predictive_scout_canonical_ledger_db_path=data_root / "execution_ledger.db",
                ).run(
                    intents_path=intents,
                    lane_id=PREDICTIVE_SCOUT_LANE_ID,
                    send=True,
                    confirm_live=True,
                )

            self.assertEqual(summary.status, "BLOCKED")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders, [])
            final_expiry_check.assert_called_once()
            staged = json.loads((root / "request.json").read_text())
            issue_codes = {issue["code"] for issue in staged["risk_issues"]}
            self.assertIn("PREDICTIVE_SCOUT_EXPIRED_BEFORE_POST", issue_codes)

    def test_predictive_scout_accepts_preverified_sub_1000_units_within_nav_risk_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"QR_PREDICTIVE_SCOUT_LIVE_ENABLED": "1"},
        ):
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            _write_predictive_scout_policy(root)
            intents = _predictive_scout_intents(
                data_root,
                now=datetime.now(timezone.utc),
            )
            payload = json.loads(intents.read_text())
            intent = payload["results"][0]["intent"]
            intent["units"] = 500
            intent["metadata"]["predictive_scout_units"] = 500
            intent["metadata"]["predictive_scout_planned_initial_risk_jpy"] = 37.8
            intent["metadata"]["predictive_scout_planned_initial_risk_pct_nav"] = 0.0189
            intent["metadata"]["predictive_scout_sizing_digest"] = (
                predictive_scout_sizing_digest(
                    execution_module._intent_from_json(intent)
                )
            )
            intents.write_text(json.dumps(payload), encoding="utf-8")

            summary = LiveOrderGateway(
                client=_predictive_scout_client(),
                strategy_profile=_profile(root, pair="USD_CAD"),
                output_path=root / "request.json",
                report_path=root / "report.md",
            ).run(intents_path=intents, lane_id=PREDICTIVE_SCOUT_LANE_ID)

            self.assertEqual(summary.status, "STAGED")
            staged = json.loads((root / "request.json").read_text())
            self.assertEqual(staged["order_request"]["units"], "500")
            self.assertEqual(staged["predictive_scout_receipt"]["units"], 500)

    def test_sent_predictive_scout_is_indexed_before_gateway_returns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "QR_PREDICTIVE_SCOUT_LIVE_ENABLED": "1",
                "QR_REQUIRE_POSITION_GUARDIAN_ACTIVE": "0",
            },
        ):
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            _write_predictive_scout_policy(root)
            intents = _predictive_scout_intents(
                data_root,
                now=datetime.now(timezone.utc),
            )
            verified = _write_predictive_scout_verified_decision(
                root,
                intents_path=intents,
            )
            client = _predictive_scout_client()

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root, pair="USD_CAD"),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
                verified_decision_path=verified,
                predictive_scout_canonical_ledger_db_path=data_root / "execution_ledger.db",
            ).run(
                intents_path=intents,
                lane_id=PREDICTIVE_SCOUT_LANE_ID,
                send=True,
                confirm_live=True,
            )

            self.assertTrue(summary.sent)
            staged = json.loads((root / "request.json").read_text())
            self.assertTrue(staged["predictive_scout_receipt_durable"])
            with closing(sqlite3.connect(data_root / "execution_ledger.db")) as con, con:
                row = con.execute(
                    "SELECT payload_json FROM gateway_receipts WHERE sent = 1"
                ).fetchone()
            self.assertIsNotNone(row)
            self.assertIn("predictive_scout_vehicle_id", str(row[0]))

    def test_predictive_scout_reservation_survives_output_write_failure_after_post(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "QR_PREDICTIVE_SCOUT_LIVE_ENABLED": "1",
                "QR_REQUIRE_POSITION_GUARDIAN_ACTIVE": "0",
            },
        ):
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            _write_predictive_scout_policy(root)
            intents = _predictive_scout_intents(
                data_root,
                now=datetime.now(timezone.utc),
            )
            verified = _write_predictive_scout_verified_decision(
                root,
                intents_path=intents,
            )
            client = _predictive_scout_client()
            gateway = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root, pair="USD_CAD"),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
                verified_decision_path=verified,
                predictive_scout_canonical_ledger_db_path=data_root / "execution_ledger.db",
            )

            with patch.object(gateway, "_write_result", side_effect=OSError("disk failed")):
                with self.assertRaises(OSError):
                    gateway.run(
                        intents_path=intents,
                        lane_id=PREDICTIVE_SCOUT_LANE_ID,
                        send=True,
                        confirm_live=True,
                    )

            self.assertEqual(len(client.orders), 1)
            self.assertFalse((root / "request.json").exists())
            with closing(sqlite3.connect(data_root / "execution_ledger.db")) as con, con:
                row = con.execute(
                    """
                    SELECT event_type, client_order_id, raw_json
                    FROM execution_events
                    WHERE event_type = 'GATEWAY_ORDER_STAGED'
                    """
                ).fetchone()
            self.assertIsNotNone(row)
            self.assertTrue(str(row[1]).startswith("qrv1-USDCAD-L-"))
            self.assertIn("predictive_scout_vehicle_id", str(row[2]))

    def test_same_vehicle_forecast_signal_cannot_post_twice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "QR_PREDICTIVE_SCOUT_LIVE_ENABLED": "1",
                "QR_REQUIRE_POSITION_GUARDIAN_ACTIVE": "0",
            },
        ):
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            _write_predictive_scout_policy(root)
            intents = _predictive_scout_intents(data_root, now=datetime.now(timezone.utc))
            verified = _write_predictive_scout_verified_decision(
                root,
                intents_path=intents,
            )
            client = _predictive_scout_client()
            gateway = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root, pair="USD_CAD"),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
                verified_decision_path=verified,
                predictive_scout_canonical_ledger_db_path=data_root / "execution_ledger.db",
            )

            first = gateway.run(
                intents_path=intents,
                lane_id=PREDICTIVE_SCOUT_LANE_ID,
                send=True,
                confirm_live=True,
            )
            second = gateway.run(
                intents_path=intents,
                lane_id=PREDICTIVE_SCOUT_LANE_ID,
                send=True,
                confirm_live=True,
            )
            payload = json.loads((root / "request.json").read_text())
            with closing(sqlite3.connect(data_root / "execution_ledger.db")) as con, con:
                claims = con.execute(
                    "SELECT COUNT(*) FROM predictive_scout_signal_claims"
                ).fetchone()[0]

        self.assertTrue(first.sent)
        self.assertFalse(second.sent)
        self.assertEqual(len(client.orders), 1)
        self.assertEqual(claims, 1)
        self.assertIn(
            "PREDICTIVE_SCOUT_EXPERIMENT_ALREADY_RESERVED",
            {issue["code"] for issue in payload["risk_issues"]},
        )

    def test_distinct_predictive_scout_gateways_cannot_race_past_global_slots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "QR_PREDICTIVE_SCOUT_LIVE_ENABLED": "1",
                "QR_REQUIRE_POSITION_GUARDIAN_ACTIVE": "0",
            },
        ):
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            _write_predictive_scout_policy(root)
            base_intents = _predictive_scout_intents(
                data_root,
                now=datetime.now(timezone.utc),
            )
            profile_path = _profile(root, pair="USD_CAD")
            gateways = []
            clients = []
            intent_paths = []
            for index in range(3):
                payload = json.loads(base_intents.read_text())
                payload["results"][0]["intent"]["metadata"]["forecast_cycle_id"] = (
                    f"parallel-forecast-cycle-{index}"
                )
                parallel_intent = payload["results"][0]["intent"]
                parallel_intent["metadata"]["predictive_scout_sizing_digest"] = (
                    predictive_scout_sizing_digest(
                        execution_module._intent_from_json(parallel_intent)
                    )
                )
                intents_path = data_root / f"parallel-intents-{index}.json"
                intents_path.write_text(json.dumps(payload), encoding="utf-8")
                client = _predictive_scout_client()
                clients.append(client)
                intent_paths.append(intents_path)
                gateways.append(
                    LiveOrderGateway(
                        client=client,
                        strategy_profile=profile_path,
                        output_path=root / f"request-{index}.json",
                        report_path=root / f"report-{index}.md",
                        live_enabled=True,
                        verified_decision_path=_write_predictive_scout_verified_decision(
                            root,
                            intents_path=intents_path,
                            suffix=str(index),
                        ),
                        predictive_scout_canonical_ledger_db_path=(
                            data_root / "execution_ledger.db"
                        ),
                        execution_ledger_report_path=root / f"ledger-{index}.md",
                    )
                )

            def send(index: int):
                return gateways[index].run(
                    intents_path=intent_paths[index],
                    lane_id=PREDICTIVE_SCOUT_LANE_ID,
                    send=True,
                    confirm_live=True,
                )

            with ThreadPoolExecutor(max_workers=3) as executor:
                results = list(executor.map(send, range(3)))
            with closing(sqlite3.connect(data_root / "execution_ledger.db")) as con, con:
                claim_count = con.execute(
                    "SELECT COUNT(*) FROM predictive_scout_signal_claims"
                ).fetchone()[0]
            issue_codes = []
            for index in range(3):
                staged = json.loads((root / f"request-{index}.json").read_text())
                issue_codes.extend(issue["code"] for issue in staged["risk_issues"])

        self.assertEqual(sum(result.sent for result in results), 2)
        self.assertEqual(sum(len(client.orders) for client in clients), 2)
        self.assertEqual(claim_count, 2)
        self.assertIn(
            "PREDICTIVE_SCOUT_CONCURRENT_CAP_REACHED_AT_RESERVATION",
            issue_codes,
        )

    def test_predictive_scout_cannot_redirect_atomic_claim_to_custom_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "QR_PREDICTIVE_SCOUT_LIVE_ENABLED": "1",
                "QR_REQUIRE_POSITION_GUARDIAN_ACTIVE": "0",
            },
        ):
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            _write_predictive_scout_policy(root)
            intents = _predictive_scout_intents(data_root, now=datetime.now(timezone.utc))
            client = _predictive_scout_client()
            custom_ledger = root / "redirected" / "execution_ledger.db"

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root, pair="USD_CAD"),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
                execution_ledger_db_path=custom_ledger,
                execution_ledger_report_path=root / "redirected" / "report.md",
                predictive_scout_canonical_ledger_db_path=data_root / "execution_ledger.db",
            ).run(
                intents_path=intents,
                lane_id=PREDICTIVE_SCOUT_LANE_ID,
                send=True,
                confirm_live=True,
            )
            payload = json.loads((root / "request.json").read_text())

        self.assertFalse(summary.sent)
        self.assertEqual(client.orders, [])
        self.assertFalse(custom_ledger.exists())
        self.assertIn(
            "PREDICTIVE_SCOUT_CANONICAL_LEDGER_REQUIRED",
            {issue["code"] for issue in payload["risk_issues"]},
        )

    def test_single_predictive_scout_uses_batch_api_without_becoming_a_basket(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"QR_PREDICTIVE_SCOUT_LIVE_ENABLED": "1"},
        ):
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            _write_predictive_scout_policy(root)
            now = datetime.now(timezone.utc)
            intents = _predictive_scout_intents(data_root, now=now)

            summary = LiveOrderGateway(
                client=_predictive_scout_client(),
                strategy_profile=_profile(root, pair="USD_CAD"),
                output_path=root / "request.json",
                report_path=root / "report.md",
            ).run_batch(
                intents_path=intents,
                lane_ids=(PREDICTIVE_SCOUT_LANE_ID,),
            )

            self.assertEqual(summary.status, "STAGED")
            staged = json.loads((root / "request.json").read_text())
            issue_codes = {
                issue["code"]
                for issue in staged["orders"][0]["risk_issues"]
            }
            self.assertNotIn("PREDICTIVE_SCOUT_SINGLE_ORDER_ONLY", issue_codes)
            self.assertEqual(len(staged["orders"]), 1)

    def test_predictive_scout_blocks_entire_mixed_batch_before_post(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            _write_predictive_scout_policy(root)
            intents = _predictive_scout_intents(
                data_root,
                now=datetime.now(timezone.utc),
            )
            payload = json.loads(intents.read_text())
            normal = json.loads(json.dumps(payload["results"][0]))
            normal["lane_id"] = "lane:EUR_USD:LONG"
            normal_intent = normal["intent"]
            normal_intent.update(
                {
                    "pair": "EUR_USD",
                    "entry": 1.17330,
                    "tp": 1.17450,
                    "sl": 1.17250,
                    "metadata": {"desk": "trend_trader", "campaign_role": "NOW"},
                }
            )
            payload["results"].append(normal)
            intents.write_text(json.dumps(payload))
            client = _predictive_scout_client()

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root, pair="USD_CAD"),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
            ).run_batch(
                intents_path=intents,
                lane_ids=(PREDICTIVE_SCOUT_LANE_ID, "lane:EUR_USD:LONG"),
                send=True,
                confirm_live=True,
            )

            self.assertEqual(summary.status, "BLOCKED")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders, [])
            staged = json.loads((root / "request.json").read_text())
            self.assertEqual(staged["blocked_count"], 2)
            self.assertTrue(
                all(
                    "PREDICTIVE_SCOUT_SINGLE_ORDER_ONLY"
                    in {issue["code"] for issue in item["risk_issues"]}
                    for item in staged["orders"]
                )
            )

    def test_runner_intent_omits_broker_take_profit_on_fill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
            ).run(
                intents_path=_intents(
                    root,
                    metadata={
                        "desk": "trend_trader",
                        "campaign_role": "RUNNER",
                        "attach_take_profit_on_fill": False,
                        "tp_execution_mode": "RUNNER_NO_BROKER_TP",
                    },
                ),
                lane_id="lane:EUR_USD:LONG",
            )

            self.assertEqual(summary.status, "STAGED")
            payload = json.loads((root / "request.json").read_text())
            order = payload["order_request"]
            self.assertNotIn("takeProfitOnFill", order)

    def test_report_surfaces_loss_asymmetry_sizing_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                max_loss_jpy=1000.0,
            ).run(
                intents_path=_intents(
                    root,
                    metadata={
                        "desk": "trend_trader",
                        "campaign_role": "NOW",
                        "capture_economics_status": "NEGATIVE_EXPECTANCY",
                        "capture_avg_win_jpy": 600.0,
                        "capture_avg_loss_jpy": 1100.0,
                        "loss_asymmetry_guard_active": True,
                        "loss_asymmetry_guard_loss_cap_jpy": 600.0,
                        "loss_asymmetry_guard_base_max_loss_jpy": 1000.0,
                        "loss_asymmetry_guard_effective_max_loss_jpy": 600.0,
                    },
                ),
                lane_id="lane:EUR_USD:LONG",
            )

            self.assertEqual(summary.status, "STAGED")
            payload = json.loads((root / "request.json").read_text())
            self.assertEqual(payload["sizing_evidence"]["loss_asymmetry_guard_loss_cap_jpy"], 600.0)
            report = (root / "report.md").read_text()
            self.assertIn("sizing guard: `LOSS_ASYMMETRY`", report)
            self.assertIn("units=`1000`", report)
            self.assertIn("cap=`600.0 JPY`", report)
            self.assertIn("avg_win/avg_loss=`600.0 JPY`/`1100.0 JPY`", report)

    def test_sl_free_disaster_stop_reports_attached_tail_risk_separately(self) -> None:
        prior_sl_free = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        prior_initial_sl = os.environ.get("QR_NEW_ENTRY_INITIAL_SL")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                client = FakeExecutionClient()
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                ).run(
                    intents_path=_intents(
                        root,
                        metadata={
                            "desk": "trend_trader",
                            "campaign_role": "NOW",
                            "disaster_sl": 1.17000,
                        },
                    ),
                    lane_id="lane:EUR_USD:LONG",
                )

                self.assertEqual(summary.status, "STAGED")
                payload = json.loads((root / "request.json").read_text())
                self.assertEqual(payload["order_request"]["stopLossOnFill"]["price"], "1.17000")
                self.assertEqual(payload["attached_stop_risk_metrics"]["basis"], "DISASTER_SL")
                self.assertGreater(
                    payload["attached_stop_risk_metrics"]["risk_jpy"],
                    payload["risk_metrics"]["risk_jpy"],
                )
                report = (root / "report.md").read_text()
                self.assertIn("intent risk", report)
                self.assertIn("attached broker SL: `DISASTER_SL`", report)
        finally:
            if prior_sl_free is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl_free
            if prior_initial_sl is None:
                os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
            else:
                os.environ["QR_NEW_ENTRY_INITIAL_SL"] = prior_initial_sl

    def test_sl_free_firepower_route_attaches_intent_stop_for_measured_risk(self) -> None:
        prior_sl_free = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        prior_initial_sl = os.environ.get("QR_NEW_ENTRY_INITIAL_SL")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                client = FakeExecutionClient()
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                ).run(
                    intents_path=_intents(
                        root,
                        metadata={
                            "desk": "range_trader",
                            "campaign_role": "OANDA_FIREPOWER_ROUTE",
                            "positive_rotation_oanda_campaign_firepower_vehicle_match": True,
                            "disaster_sl": 1.17000,
                        },
                    ),
                    lane_id="lane:EUR_USD:LONG",
                )

                self.assertEqual(summary.status, "STAGED")
                payload = json.loads((root / "request.json").read_text())
                self.assertEqual(payload["order_request"]["stopLossOnFill"]["price"], "1.17250")
                self.assertEqual(payload["attached_stop_risk_metrics"]["basis"], "INTENT_SL")
                self.assertAlmostEqual(
                    payload["attached_stop_risk_metrics"]["risk_jpy"],
                    payload["risk_metrics"]["risk_jpy"],
                )
                self.assertAlmostEqual(
                    payload["attached_stop_risk_metrics"]["loss_delta_pips"],
                    0.0,
                )
        finally:
            if prior_sl_free is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl_free
            if prior_initial_sl is None:
                os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
            else:
                os.environ["QR_NEW_ENTRY_INITIAL_SL"] = prior_initial_sl

    def test_sl_free_disaster_stop_does_not_block_on_per_trade_tail_cap(self) -> None:
        prior_sl_free = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        prior_initial_sl = os.environ.get("QR_NEW_ENTRY_INITIAL_SL")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                client = FakeExecutionClient()
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    max_loss_jpy=300.0,
                    live_enabled=True,
                ).run(
                    intents_path=_intents(
                        root,
                        metadata={
                            "desk": "trend_trader",
                            "campaign_role": "NOW",
                            "disaster_sl": 1.17000,
                        },
                    ),
                    lane_id="lane:EUR_USD:LONG",
                    send=True,
                    confirm_live=True,
                )

                self.assertEqual(summary.status, "SENT")
                self.assertTrue(summary.sent)
                self.assertEqual(len(client.orders), 1)
                payload = json.loads((root / "request.json").read_text())
                self.assertEqual(payload["order_request"]["stopLossOnFill"]["price"], "1.17000")
                self.assertGreater(payload["attached_stop_risk_metrics"]["risk_jpy"], 300.0)
                self.assertNotIn(
                    "ATTACHED_STOP_LOSS_CAP_BELOW_MIN_LOT",
                    {issue["code"] for issue in payload["risk_issues"]},
                )
        finally:
            if prior_sl_free is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl_free
            if prior_initial_sl is None:
                os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
            else:
                os.environ["QR_NEW_ENTRY_INITIAL_SL"] = prior_initial_sl

    def test_sl_free_disaster_stop_does_not_clip_units_to_attached_tail_cap(self) -> None:
        prior_sl_free = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        prior_initial_sl = os.environ.get("QR_NEW_ENTRY_INITIAL_SL")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                client = FakeExecutionClient()
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    max_loss_jpy=1100.0,
                ).run(
                    intents_path=_intents(
                        root,
                        units=3000,
                        metadata={
                            "desk": "trend_trader",
                            "campaign_role": "NOW",
                            "disaster_sl": 1.17000,
                        },
                    ),
                    lane_id="lane:EUR_USD:LONG",
                )

                self.assertEqual(summary.status, "STAGED")
                payload = json.loads((root / "request.json").read_text())
                scaled_units = int(payload["order_request"]["units"])
                self.assertEqual(scaled_units, 3000)
                self.assertEqual(payload["scaled_units"], scaled_units)
                self.assertGreater(payload["attached_stop_risk_metrics"]["risk_jpy"], 1100.0)
                self.assertEqual(payload["attached_stop_risk_metrics"]["basis"], "DISASTER_SL")
                self.assertNotIn(
                    "SIZE_MULTIPLE_CLIPPED_TO_ATTACHED_STOP_CAP",
                    {issue["code"] for issue in payload["risk_issues"]},
                )
        finally:
            if prior_sl_free is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl_free
            if prior_initial_sl is None:
                os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
            else:
                os.environ["QR_NEW_ENTRY_INITIAL_SL"] = prior_initial_sl

    def test_attached_tail_risk_counts_against_portfolio_remaining_cap(self) -> None:
        prior_sl_free = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        prior_initial_sl = os.environ.get("QR_NEW_ENTRY_INITIAL_SL")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                client = FakeExecutionClient()
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    max_loss_jpy=10_000.0,
                    portfolio_loss_cap_jpy=300.0,
                ).run(
                    intents_path=_intents(
                        root,
                        metadata={
                            "desk": "trend_trader",
                            "campaign_role": "NOW",
                            "disaster_sl": 1.17000,
                        },
                    ),
                    lane_id="lane:EUR_USD:LONG",
                    send=True,
                    confirm_live=True,
                )

                self.assertEqual(summary.status, "BLOCKED")
                self.assertFalse(summary.sent)
                self.assertEqual(client.orders, [])
                payload = json.loads((root / "request.json").read_text())
                self.assertLessEqual(payload["risk_metrics"]["risk_jpy"], 300.0)
                self.assertGreater(payload["attached_stop_risk_metrics"]["risk_jpy"], 300.0)
                self.assertIn(
                    "DISASTER_STOP_PORTFOLIO_CAP_EXCEEDED",
                    {issue["code"] for issue in payload["risk_issues"]},
                )
        finally:
            if prior_sl_free is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl_free
            if prior_initial_sl is None:
                os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
            else:
                os.environ["QR_NEW_ENTRY_INITIAL_SL"] = prior_initial_sl

    def test_batch_counts_attached_tail_risk_in_cumulative_portfolio_cap(self) -> None:
        prior_sl_free = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        prior_initial_sl = os.environ.get("QR_NEW_ENTRY_INITIAL_SL")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                client = FakeExecutionClient()
                intents = _intents(
                    root,
                    metadata={
                        "desk": "trend_trader",
                        "campaign_role": "NOW",
                        "disaster_sl": 1.17000,
                    },
                )
                payload = json.loads(intents.read_text())
                second = json.loads(json.dumps(payload["results"][0]))
                second["lane_id"] = "lane:EUR_USD:LONG:reload"
                second["intent"]["entry"] = 1.17360
                second["intent"]["tp"] = 1.17480
                second["intent"]["sl"] = 1.17280
                payload["results"].append(second)
                intents.write_text(json.dumps(payload))

                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    max_loss_jpy=10_000.0,
                    portfolio_loss_cap_jpy=800.0,
                ).run_batch(
                    intents_path=intents,
                    lane_ids=("lane:EUR_USD:LONG", "lane:EUR_USD:LONG:reload"),
                )

                self.assertEqual(summary.status, "STAGED")
                self.assertEqual(summary.sent_count, 0)
                result = json.loads((root / "request.json").read_text())
                self.assertEqual(result["staged_count"], 1)
                self.assertEqual(result["blocked_count"], 1)
                first, second = result["orders"]
                self.assertEqual(first["status"], "STAGED")
                self.assertGreater(
                    first["attached_stop_risk_metrics"]["risk_jpy"],
                    first["risk_metrics"]["risk_jpy"],
                )
                self.assertEqual(second["status"], "BLOCKED")
                self.assertLessEqual(second["risk_metrics"]["risk_jpy"], 800.0)
                self.assertIn(
                    "DISASTER_STOP_PORTFOLIO_CAP_EXCEEDED",
                    {issue["code"] for issue in second["risk_issues"]},
                )
        finally:
            if prior_sl_free is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl_free
            if prior_initial_sl is None:
                os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
            else:
                os.environ["QR_NEW_ENTRY_INITIAL_SL"] = prior_initial_sl

    def test_gateway_clips_units_to_intent_metadata_loss_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
            ).run(
                intents_path=_intents(
                    root,
                    units=3000,
                    metadata={
                        "desk": "trend_trader",
                        "campaign_role": "NOW",
                        "max_loss_jpy": 250.0,
                    },
                ),
                lane_id="lane:EUR_USD:LONG",
            )

            self.assertEqual(summary.status, "STAGED")
            payload = json.loads((root / "request.json").read_text())
            scaled_units = int(payload["order_request"]["units"])
            self.assertGreaterEqual(scaled_units, 1)
            self.assertLess(scaled_units, 3000)
            self.assertEqual(payload["scaled_units"], scaled_units)
            self.assertLessEqual(payload["risk_metrics"]["risk_jpy"], 250.0)
            self.assertIn(
                "SIZE_MULTIPLE_CLIPPED_TO_LOSS_CAP",
                {issue["code"] for issue in payload["risk_issues"]},
            )

    def test_high_confidence_macro_units_already_fit_fresh_gateway_cap_without_clip(self) -> None:
        from quant_rabbit.strategy.intent_generator import (
            _macro_event_sizing_plan,
            _risk_budgeted_units,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            snapshot = client.snapshot_value
            assert snapshot.account is not None
            effective_cap, confidence_metadata = _macro_event_sizing_plan(
                {
                    "forecast_market_support": {
                        "ok": True,
                        "direction": "UP",
                        "signals": [
                            {
                                "name": "event_surprise_followthrough",
                                "direction": "UP",
                                "confidence": 0.92,
                            }
                        ],
                    }
                },
                side=Side.LONG,
                base_max_loss_jpy=500.0,
                portfolio_loss_cap=1_000.0,
                position_metadata={},
                sizing_nav_jpy=snapshot.account.nav_jpy,
            )
            units = _risk_budgeted_units(
                "EUR_USD",
                1.17330,
                1.17250,
                max_loss_jpy=effective_cap,
                snapshot=snapshot,
                side=Side.LONG,
                loss_budget_target=True,
            )

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                max_loss_jpy=500.0,
            ).run(
                intents_path=_intents(
                    root,
                    units=units,
                    metadata={
                        "desk": "trend_trader",
                        "campaign_role": "NOW",
                        "max_loss_jpy": effective_cap,
                        **confidence_metadata,
                    },
                ),
                lane_id="lane:EUR_USD:LONG",
            )

            self.assertEqual(summary.status, "STAGED")
            payload = json.loads((root / "request.json").read_text())
            self.assertEqual(effective_cap, 500.0)
            self.assertEqual(payload["requested_units"], units)
            self.assertEqual(payload["scaled_units"], units)
            self.assertLessEqual(payload["risk_metrics"]["risk_jpy"], 500.0)
            self.assertNotIn(
                "SIZE_MULTIPLE_CLIPPED_TO_LOSS_CAP",
                {issue["code"] for issue in payload["risk_issues"]},
            )

    def test_stage_receipt_persists_market_context_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
            ).run(
                intents_path=_intents(
                    root,
                    metadata={
                        "desk": "trend_trader",
                        "campaign_role": "NOW",
                        "market_context_matrix_ref": "matrix:EUR_USD:LONG",
                        "matrix_support_count": 4,
                        "matrix_reject_count": 0,
                        "matrix_support_layers": ["chart", "cross_asset", "context_asset_chart", "flow"],
                        "matrix_support_refs": [
                            "matrix:EUR_USD:LONG",
                            "cross:XAU_USD",
                            "context_asset:WTICO_USD",
                            "news:macro_event",
                        ],
                        "news_digest_ref": "news:macro_event",
                    },
                ),
                lane_id="lane:EUR_USD:LONG",
            )

            self.assertEqual(summary.status, "STAGED")
            payload = json.loads((root / "request.json").read_text())
            evidence = payload["context_evidence"]
            self.assertEqual(evidence["market_context_matrix_ref"], "matrix:EUR_USD:LONG")
            self.assertEqual(evidence["matrix_support_layers"], ["chart", "cross_asset", "context_asset_chart", "flow"])
            self.assertIn("cross:XAU_USD", evidence["context_asset_refs"])
            self.assertIn("context_asset:WTICO_USD", evidence["context_asset_refs"])
            self.assertIn("news:macro_event", evidence["evidence_refs"])

    def test_send_requires_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
            ).run(intents_path=_intents(root), lane_id="lane:EUR_USD:LONG", send=True)

            self.assertEqual(summary.status, "BLOCKED")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders, [])
            self.assertIn("LIVE_CONFIRMATION_REQUIRED", (root / "report.md").read_text())

    def test_live_send_retries_stale_quote_before_blocking(self) -> None:
        prior_attempts = os.environ.get("QR_GATEWAY_STALE_QUOTE_RETRY_ATTEMPTS")
        prior_sleep = os.environ.get("QR_GATEWAY_STALE_QUOTE_RETRY_SLEEP_SECONDS")
        os.environ["QR_GATEWAY_STALE_QUOTE_RETRY_ATTEMPTS"] = "2"
        os.environ["QR_GATEWAY_STALE_QUOTE_RETRY_SLEEP_SECONDS"] = "0"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                now = datetime.now(timezone.utc)
                stale = now - timedelta(seconds=45)
                client = SequenceExecutionClient(
                    (
                        _gateway_snapshot(fetched_at=now, eur_usd_quote_time=stale),
                        _gateway_snapshot(fetched_at=now, eur_usd_quote_time=now),
                    )
                )

                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    live_enabled=True,
                ).run(
                    intents_path=_intents(root),
                    lane_id="lane:EUR_USD:LONG",
                    send=True,
                    confirm_live=True,
                )

                self.assertEqual(summary.status, "SENT")
                self.assertTrue(summary.sent)
                self.assertEqual(len(client.orders), 1)
                self.assertGreaterEqual(len(client.snapshot_calls), 2)
                payload = json.loads((root / "request.json").read_text())
                self.assertEqual(payload["quote_refresh_attempts"], 1)
                self.assertNotIn("STALE_QUOTE", {issue["code"] for issue in payload["risk_issues"]})
        finally:
            if prior_attempts is None:
                os.environ.pop("QR_GATEWAY_STALE_QUOTE_RETRY_ATTEMPTS", None)
            else:
                os.environ["QR_GATEWAY_STALE_QUOTE_RETRY_ATTEMPTS"] = prior_attempts
            if prior_sleep is None:
                os.environ.pop("QR_GATEWAY_STALE_QUOTE_RETRY_SLEEP_SECONDS", None)
            else:
                os.environ["QR_GATEWAY_STALE_QUOTE_RETRY_SLEEP_SECONDS"] = prior_sleep

    def test_self_improvement_p0_blocks_live_order_staging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "self_improvement.json"
            audit.write_text(
                json.dumps(
                    {
                        "findings": [
                            {
                                "priority": "P0",
                                "code": "PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED",
                                "message": "profitability discipline has failed for 50 consecutive audit runs",
                            }
                        ]
                    }
                )
            )
            client = FakeExecutionClient()
            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                self_improvement_audit=audit,
            ).run(intents_path=_intents(root), lane_id="lane:EUR_USD:LONG")

            self.assertEqual(summary.status, "BLOCKED")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders, [])
            payload = json.loads((root / "request.json").read_text())
            codes = {issue["code"] for issue in payload["risk_issues"]}
            self.assertIn("SELF_IMPROVEMENT_P0_BLOCKS_LIVE_ORDER", codes)

    def test_self_improvement_p0_allows_verified_repair_lane_staging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "self_improvement.json"
            audit.write_text(
                json.dumps(
                    {
                        "findings": [
                            {
                                "priority": "P0",
                                "code": "PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED",
                                "message": "profitability discipline has failed for 50 consecutive audit runs",
                            }
                        ]
                    }
                )
            )
            intents = _intents(
                root,
                metadata={
                    "desk": "range_trader",
                    "campaign_role": "NOW",
                    "self_improvement_p0_repair_live_ready": True,
                    "self_improvement_p0_repair_mode": "TP_HARVEST_REPAIR",
                },
            )
            client = FakeExecutionClient()
            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                self_improvement_audit=audit,
            ).run(intents_path=intents, lane_id="lane:EUR_USD:LONG")

            self.assertEqual(summary.status, "STAGED")
            self.assertFalse(summary.sent)
            payload = json.loads((root / "request.json").read_text())
            codes = {issue["code"] for issue in payload["risk_issues"]}
            self.assertNotIn("SELF_IMPROVEMENT_P0_BLOCKS_LIVE_ORDER", codes)

    def test_profitability_p0_allows_only_supported_min_lot_predictive_scout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"QR_PREDICTIVE_SCOUT_LIVE_ENABLED": "1"},
        ):
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            _write_predictive_scout_policy(root)
            audit = root / "self_improvement.json"
            audit.write_text(
                json.dumps(
                    {
                        "findings": [
                            {
                                "priority": "P0",
                                "code": "PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED",
                                "message": "profitability discipline remains red",
                            }
                        ]
                    }
                )
            )
            intents = _predictive_scout_intents(
                data_root,
                now=datetime.now(timezone.utc),
                metadata_updates={
                    "self_improvement_p0_repair_live_ready": True,
                    "self_improvement_p0_repair_mode": "PREDICTIVE_SCOUT_FORWARD_EVIDENCE",
                },
            )

            summary = LiveOrderGateway(
                client=_predictive_scout_client(),
                strategy_profile=_profile(root, pair="USD_CAD"),
                output_path=root / "request.json",
                report_path=root / "report.md",
                self_improvement_audit=audit,
            ).run(intents_path=intents, lane_id=PREDICTIVE_SCOUT_LANE_ID)

            self.assertEqual(summary.status, "STAGED")
            staged = json.loads((root / "request.json").read_text())
            issue_codes = {issue["code"] for issue in staged["risk_issues"]}
            self.assertNotIn("SELF_IMPROVEMENT_P0_BLOCKS_LIVE_ORDER", issue_codes)

    def test_predictive_scout_does_not_exempt_pending_cancel_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"QR_PREDICTIVE_SCOUT_LIVE_ENABLED": "1"},
        ):
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            _write_predictive_scout_policy(root)
            now = datetime.now(timezone.utc)
            audit = root / "self_improvement.json"
            audit.write_text(
                json.dumps(
                    {
                        "generated_at_utc": (now - timedelta(minutes=1)).isoformat(),
                        "findings": [
                            {
                                "priority": "P0",
                                "code": "PENDING_ENTRY_CANCEL_REVIEW_REQUIRED",
                                "message": "pending entry requires explicit replacement",
                                "evidence": {"cancel_review_order_ids": ["pending-1"]},
                            }
                        ],
                    }
                )
            )
            verified = root / "gpt_decision.json"
            verified.write_text(
                json.dumps(
                    {
                        "generated_at_utc": now.isoformat(),
                        "status": "ACCEPTED",
                        "decision": {
                            "action": "TRADE",
                            "selected_lane_id": PREDICTIVE_SCOUT_LANE_ID,
                            "selected_lane_ids": [PREDICTIVE_SCOUT_LANE_ID],
                            "cancel_order_ids": ["pending-1"],
                        },
                        "verification_issues": [],
                    }
                )
            )
            intents = _predictive_scout_intents(
                data_root,
                now=now,
                metadata_updates={
                    "self_improvement_p0_repair_live_ready": True,
                    "self_improvement_p0_repair_mode": "PREDICTIVE_SCOUT_FORWARD_EVIDENCE",
                },
            )

            summary = LiveOrderGateway(
                client=_predictive_scout_client(),
                strategy_profile=_profile(root, pair="USD_CAD"),
                output_path=root / "request.json",
                report_path=root / "report.md",
                self_improvement_audit=audit,
                verified_decision_path=verified,
            ).run(intents_path=intents, lane_id=PREDICTIVE_SCOUT_LANE_ID)

            self.assertEqual(summary.status, "BLOCKED")
            staged = json.loads((root / "request.json").read_text())
            issue_codes = {issue["code"] for issue in staged["risk_issues"]}
            self.assertIn("SELF_IMPROVEMENT_P0_BLOCKS_LIVE_ORDER", issue_codes)

    def test_pending_cancel_review_p0_requires_verified_trade_cancel_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "self_improvement.json"
            audit.write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "findings": [
                            {
                                "priority": "P0",
                                "code": "PENDING_ENTRY_CANCEL_REVIEW_REQUIRED",
                                "message": "1 trader-owned pending entry order(s) need cancel review",
                                "evidence": {"cancel_review_order_ids": ["pending-1"]},
                            }
                        ],
                    }
                )
            )
            intents = _intents(
                root,
                metadata={
                    "desk": "failure_trader",
                    "campaign_role": "NOW",
                    "self_improvement_p0_repair_live_ready": True,
                    "self_improvement_p0_repair_mode": "TP_HARVEST_REPAIR",
                },
            )
            client = FakeExecutionClient()
            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                self_improvement_audit=audit,
            ).run(intents_path=intents, lane_id="lane:EUR_USD:LONG")

            self.assertEqual(summary.status, "BLOCKED")
            payload = json.loads((root / "request.json").read_text())
            codes = {issue["code"] for issue in payload["risk_issues"]}
            self.assertIn("SELF_IMPROVEMENT_P0_BLOCKS_LIVE_ORDER", codes)

    def test_pending_cancel_review_p0_allows_verified_trade_replacement_staging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            audit = root / "self_improvement.json"
            audit.write_text(
                json.dumps(
                    {
                        "generated_at_utc": (now - timedelta(minutes=1)).isoformat(),
                        "findings": [
                            {
                                "priority": "P0",
                                "code": "PENDING_ENTRY_CANCEL_REVIEW_REQUIRED",
                                "message": "1 trader-owned pending entry order(s) need cancel review",
                                "evidence": {"cancel_review_order_ids": ["pending-1"]},
                            }
                        ],
                    }
                )
            )
            verified = root / "gpt_decision.json"
            verified.write_text(
                json.dumps(
                    {
                        "generated_at_utc": now.isoformat(),
                        "status": "ACCEPTED",
                        "decision": {
                            "action": "TRADE",
                            "selected_lane_id": "lane:EUR_USD:LONG",
                            "selected_lane_ids": ["lane:EUR_USD:LONG"],
                            "cancel_order_ids": ["pending-1"],
                        },
                        "verification_issues": [],
                    }
                )
            )
            intents = _intents(
                root,
                metadata={
                    "desk": "failure_trader",
                    "campaign_role": "NOW",
                    "self_improvement_p0_repair_live_ready": True,
                    "self_improvement_p0_repair_mode": "TP_HARVEST_REPAIR",
                },
            )
            client = FakeExecutionClient()
            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                self_improvement_audit=audit,
                verified_decision_path=verified,
            ).run(intents_path=intents, lane_id="lane:EUR_USD:LONG")

            self.assertEqual(summary.status, "STAGED")
            payload = json.loads((root / "request.json").read_text())
            codes = {issue["code"] for issue in payload["risk_issues"]}
            self.assertNotIn("SELF_IMPROVEMENT_P0_BLOCKS_LIVE_ORDER", codes)

    def test_profit_capture_miss_p0_allows_tp_harvest_repair_lane_staging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "self_improvement.json"
            audit.write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "SELF_IMPROVEMENT_BLOCKED",
                        "findings": [
                            {
                                "priority": "P0",
                                "layer": "execution_quality",
                                "code": "LOSS_CLOSE_PROFIT_CAPTURE_MISSED",
                                "message": (
                                    "13 losing close(s) had production-gate replay proof "
                                    "that TP-progress capture was executable before closing red"
                                ),
                            }
                        ],
                    }
                )
            )
            intents = _intents(
                root,
                metadata={
                    "desk": "range_trader",
                    "campaign_role": "NOW",
                    "positive_rotation_mode": "TP_PROOF_COLLECTION_HARVEST",
                    "positive_rotation_pessimistic_expectancy_jpy": 215.6,
                    "self_improvement_p0_repair_live_ready": True,
                    "self_improvement_p0_repair_mode": "TP_HARVEST_REPAIR",
                    "self_improvement_p0_repair_blocker_code": (
                        "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION"
                    ),
                    "capture_take_profit_trades": 6,
                    "capture_take_profit_wins": 6,
                    "capture_take_profit_losses": 0,
                    "capture_take_profit_expectancy_jpy": 992.7,
                },
            )
            client = FakeExecutionClient()
            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                self_improvement_audit=audit,
            ).run(intents_path=intents, lane_id="lane:EUR_USD:LONG")

            self.assertEqual(summary.status, "STAGED")
            self.assertFalse(summary.sent)
            payload = json.loads((root / "request.json").read_text())
            codes = {issue["code"] for issue in payload["risk_issues"]}
            self.assertNotIn("SELF_IMPROVEMENT_P0_BLOCKS_LIVE_ORDER", codes)

    def test_self_improvement_p0_blocks_underpowered_oanda_repair_lane_staging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "self_improvement.json"
            audit.write_text(
                json.dumps(
                    {
                        "findings": [
                            {
                                "priority": "P0",
                                "code": "PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED",
                                "message": "profitability discipline has failed for 50 consecutive audit runs",
                            }
                        ]
                    }
                )
            )
            intents = _intents(
                root,
                metadata={
                    "desk": "range_trader",
                    "campaign_role": "OANDA_FIREPOWER_ROUTE",
                    "self_improvement_p0_repair_live_ready": True,
                    "self_improvement_p0_repair_mode": "TP_HARVEST_REPAIR",
                    "positive_rotation_mode": "OANDA_CAMPAIGN_FIREPOWER_HARVEST",
                    "positive_rotation_minimum_floor_reachable": False,
                    "positive_rotation_minimum_floor_reach_basis": (
                        "OANDA_CAMPAIGN_FIREPOWER_CURRENT_RISK_UNDERPOWERED"
                    ),
                    "positive_rotation_oanda_campaign_current_risk_minimum_floor_reachable": False,
                },
            )
            client = FakeExecutionClient()
            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                self_improvement_audit=audit,
            ).run(intents_path=intents, lane_id="lane:EUR_USD:LONG")

            self.assertEqual(summary.status, "BLOCKED")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders, [])
            payload = json.loads((root / "request.json").read_text())
            codes = {issue["code"] for issue in payload["risk_issues"]}
            self.assertIn("SELF_IMPROVEMENT_P0_BLOCKS_LIVE_ORDER", codes)

    def test_self_improvement_p0_rejects_repair_lane_on_named_worst_segment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "self_improvement.json"
            audit.write_text(
                json.dumps(
                    {
                        "findings": [
                            {
                                "priority": "P0",
                                "layer": "profitability",
                                "code": "PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED",
                                "message": "profitability discipline has failed for 62 consecutive audit runs",
                                "evidence": {
                                    "system_defect_evidence": {
                                        "worst_segments": [
                                            {
                                                "pair": "EUR_USD",
                                                "side": "LONG",
                                                "method": "TREND_CONTINUATION",
                                                "trades": 2,
                                                "net_jpy": -1937.49,
                                            }
                                        ]
                                    }
                                },
                            }
                        ]
                    }
                )
            )
            intents = _intents(
                root,
                metadata={
                    "desk": "trend_trader",
                    "campaign_role": "NOW",
                    "self_improvement_p0_repair_live_ready": True,
                    "self_improvement_p0_repair_mode": "TP_HARVEST_REPAIR",
                },
            )
            client = FakeExecutionClient()
            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                self_improvement_audit=audit,
            ).run(intents_path=intents, lane_id="lane:EUR_USD:LONG")

            self.assertEqual(summary.status, "BLOCKED")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders, [])
            payload = json.loads((root / "request.json").read_text())
            codes = {issue["code"] for issue in payload["risk_issues"]}
            self.assertIn("SELF_IMPROVEMENT_P0_BLOCKS_LIVE_ORDER", codes)

    def test_forecast_adverse_path_blocks_live_order_staging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "self_improvement.json"
            audit.write_text(
                json.dumps(
                    {
                        "root_cause_focus": {
                            "primary": {
                                "family": "FORECAST_ADVERSE_PATH",
                                "confidence": "HIGH",
                                "priority": "P1",
                                "process_loop_streak": 16,
                                "supporting_codes": [
                                    "DIRECTIONAL_FORECAST_HIT_RATE_WEAK",
                                    "DIRECTIONAL_FORECAST_INVALIDATION_FIRST_DOMINANT",
                                ],
                                "metrics": {
                                    "directional_hit_rate": 0.261,
                                    "invalidation_first_rate": 0.739,
                                    "profit_factor": 0.891,
                                },
                            }
                        },
                        "findings": [
                            {
                                "priority": "P1",
                                "layer": "forecast",
                                "code": "DIRECTIONAL_FORECAST_HIT_RATE_WEAK",
                                "message": "directional_forecast HIT rate is weak",
                            }
                        ],
                    }
                )
            )
            client = FakeExecutionClient()
            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                self_improvement_audit=audit,
            ).run(intents_path=_intents(root), lane_id="lane:EUR_USD:LONG")

            self.assertEqual(summary.status, "BLOCKED")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders, [])
            payload = json.loads((root / "request.json").read_text())
            codes = {issue["code"] for issue in payload["risk_issues"]}
            self.assertIn("SELF_IMPROVEMENT_P0_BLOCKS_LIVE_ORDER", codes)
            self.assertIn("SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH", payload["risk_issues"][-1]["message"])

    def test_forecast_adverse_path_allows_tp_proven_repair_lane_staging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "self_improvement.json"
            audit.write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "root_cause_focus": {
                            "primary": {
                                "family": "FORECAST_ADVERSE_PATH",
                                "confidence": "HIGH",
                                "priority": "P1",
                                "process_loop_streak": 16,
                                "supporting_codes": [
                                    "DIRECTIONAL_FORECAST_HIT_RATE_WEAK",
                                    "DIRECTIONAL_FORECAST_INVALIDATION_FIRST_DOMINANT",
                                ],
                            }
                        },
                    }
                )
            )
            intents = _intents(
                root,
                order_type="LIMIT",
                metadata={
                    "desk": "failure_trader",
                    "campaign_role": "NOW",
                    "forecast_direction": "RANGE",
                    "forecast_confidence": 0.62,
                    "attach_take_profit_on_fill": True,
                    "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                    "tp_target_intent": "HARVEST",
                    "opportunity_mode": "HARVEST",
                    "positive_rotation_live_ready": True,
                    "positive_rotation_mode": "TP_PROVEN_HARVEST",
                    "positive_rotation_pessimistic_expectancy_jpy": 180.0,
                    "capture_take_profit_scope": "PAIR_SIDE_METHOD",
                    "capture_take_profit_scope_key": (
                        "EUR_USD|LONG|BREAKOUT_FAILURE|TAKE_PROFIT_ORDER"
                    ),
                    "capture_take_profit_trades": 20,
                    "capture_take_profit_losses": 0,
                    "capture_take_profit_expectancy_jpy": 591.5,
                    "self_improvement_forecast_adverse_path_repair_live_ready": True,
                    "self_improvement_forecast_adverse_path_repair_mode": "TP_HARVEST_REPAIR",
                    "self_improvement_forecast_adverse_path_repair_blocker_code": (
                        "SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH"
                    ),
                },
            )
            payload = json.loads(intents.read_text())
            intent = payload["results"][0]["intent"]
            intent["thesis"] = "tp-proven failed-break fade"
            intent["market_context"]["method"] = "BREAKOUT_FAILURE"
            intent["market_context"]["regime"] = "RANGE current; BREAKOUT_FAILURE campaign lane"
            intents.write_text(json.dumps(payload))
            client = FakeExecutionClient()

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                self_improvement_audit=audit,
            ).run(intents_path=intents, lane_id="lane:EUR_USD:LONG")

            self.assertEqual(summary.status, "STAGED")
            self.assertFalse(summary.sent)
            payload = json.loads((root / "request.json").read_text())
            codes = {issue["code"] for issue in payload["risk_issues"]}
            self.assertNotIn("SELF_IMPROVEMENT_P0_BLOCKS_LIVE_ORDER", codes)

    def test_stale_prior_gpt_decision_p0_does_not_block_live_order_staging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "self_improvement.json"
            audit.write_text(
                json.dumps(
                    {
                        "findings": [
                            {
                                "priority": "P0",
                                "code": "LATEST_GPT_DECISION_STALE",
                                "message": "latest GPT decision receipt predates the current broker snapshot",
                            }
                        ]
                    }
                )
            )
            client = FakeExecutionClient()
            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                self_improvement_audit=audit,
            ).run(intents_path=_intents(root), lane_id="lane:EUR_USD:LONG")

            self.assertEqual(summary.status, "STAGED")
            self.assertFalse(summary.sent)
            payload = json.loads((root / "request.json").read_text())
            codes = {issue["code"] for issue in payload["risk_issues"]}
            self.assertNotIn("SELF_IMPROVEMENT_P0_BLOCKS_LIVE_ORDER", codes)

    def test_persistent_stale_prior_gpt_decision_p0_blocks_live_order_staging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "self_improvement.json"
            audit.write_text(
                json.dumps(
                    {
                        "findings": [
                            {
                                "priority": "P0",
                                "code": "LATEST_GPT_DECISION_STALE",
                                "message": "latest GPT decision receipt predates the current broker snapshot",
                                "evidence": {"current_streak": 2},
                            }
                        ]
                    }
                )
            )
            client = FakeExecutionClient()
            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                self_improvement_audit=audit,
            ).run(intents_path=_intents(root), lane_id="lane:EUR_USD:LONG")

            self.assertEqual(summary.status, "BLOCKED")
            self.assertFalse(summary.sent)
            payload = json.loads((root / "request.json").read_text())
            codes = {issue["code"] for issue in payload["risk_issues"]}
            self.assertIn("SELF_IMPROVEMENT_P0_BLOCKS_LIVE_ORDER", codes)

    def test_persistent_stale_decision_p0_exempted_when_verified_receipt_postdates_audit(self) -> None:
        # Mirrors gpt_trader._self_improvement_trade_blockers: an ACCEPTED
        # verification produced AFTER the audit ran proves the stale-decision
        # finding is already repaired. Without this, the 20-minute audit
        # cadence re-blocks the first staging attempt of every fresh receipt
        # whenever the decision cadence is slower than two audit runs.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            audit = root / "self_improvement.json"
            audit.write_text(
                json.dumps(
                    {
                        "generated_at_utc": (now - timedelta(minutes=10)).isoformat(),
                        "findings": [
                            {
                                "priority": "P0",
                                "code": "LATEST_GPT_DECISION_STALE",
                                "message": "latest GPT decision receipt predates the current broker snapshot",
                                "evidence": {"current_streak": 21},
                            }
                        ],
                    }
                )
            )
            verified = root / "gpt_decision.json"
            verified.write_text(
                json.dumps(
                    {
                        "generated_at_utc": now.isoformat(),
                        "status": "ACCEPTED",
                        "decision": {"action": "TRADE"},
                        "verification_issues": [],
                    }
                )
            )
            client = FakeExecutionClient()
            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                self_improvement_audit=audit,
                verified_decision_path=verified,
            ).run(intents_path=_intents(root), lane_id="lane:EUR_USD:LONG")

            self.assertEqual(summary.status, "STAGED")
            payload = json.loads((root / "request.json").read_text())
            codes = {issue["code"] for issue in payload["risk_issues"]}
            self.assertNotIn("SELF_IMPROVEMENT_P0_BLOCKS_LIVE_ORDER", codes)

    def test_persistent_stale_decision_p0_still_blocks_when_verification_predates_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            audit = root / "self_improvement.json"
            audit.write_text(
                json.dumps(
                    {
                        "generated_at_utc": now.isoformat(),
                        "findings": [
                            {
                                "priority": "P0",
                                "code": "LATEST_GPT_DECISION_STALE",
                                "message": "latest GPT decision receipt predates the current broker snapshot",
                                "evidence": {"current_streak": 21},
                            }
                        ],
                    }
                )
            )
            verified = root / "gpt_decision.json"
            verified.write_text(
                json.dumps(
                    {
                        "generated_at_utc": (now - timedelta(minutes=10)).isoformat(),
                        "status": "ACCEPTED",
                        "decision": {"action": "TRADE"},
                        "verification_issues": [],
                    }
                )
            )
            client = FakeExecutionClient()
            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                self_improvement_audit=audit,
                verified_decision_path=verified,
            ).run(intents_path=_intents(root), lane_id="lane:EUR_USD:LONG")

            self.assertEqual(summary.status, "BLOCKED")
            payload = json.loads((root / "request.json").read_text())
            codes = {issue["code"] for issue in payload["risk_issues"]}
            self.assertIn("SELF_IMPROVEMENT_P0_BLOCKS_LIVE_ORDER", codes)

    def test_send_posts_only_after_live_validation_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            with patch.object(
                execution_module,
                "clear_guardian_tuning_validation_cache",
                wraps=execution_module.clear_guardian_tuning_validation_cache,
            ) as clear_validation_cache:
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    live_enabled=True,
                ).run(
                    intents_path=_intents(root),
                    lane_id="lane:EUR_USD:LONG",
                    send=True,
                    confirm_live=True,
                )

            self.assertEqual(summary.status, "SENT")
            self.assertTrue(summary.sent)
            self.assertEqual(len(client.orders), 1)
            self.assertEqual(clear_validation_cache.call_count, 1)

    def test_target_path_live_send_defaults_to_disabled_even_when_gateway_live_is_enabled(self) -> None:
        prior = os.environ.get("QR_TARGET_PATH_LIVE_ENABLED")
        os.environ.pop("QR_TARGET_PATH_LIVE_ENABLED", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                client = FakeExecutionClient()

                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    live_enabled=True,
                ).run(
                    intents_path=_intents(root, metadata=_target_path_metadata(grade="A")),
                    lane_id="lane:EUR_USD:LONG",
                    send=True,
                    confirm_live=True,
                )

                self.assertEqual(summary.status, "BLOCKED")
                self.assertFalse(summary.sent)
                self.assertEqual(client.orders, [])
                result = json.loads((root / "request.json").read_text())
                self.assertIn("TARGET_PATH_LIVE_DISABLED", {issue["code"] for issue in result["risk_issues"]})
                self.assertFalse(result["target_path_receipt"]["live_order_sent"])
                self.assertFalse(result["target_path_receipt"]["target_path_live_enabled"])
        finally:
            _restore_env("QR_TARGET_PATH_LIVE_ENABLED", prior)

    def test_target_path_live_send_blocks_b0_even_with_explicit_flag(self) -> None:
        prior = os.environ.get("QR_TARGET_PATH_LIVE_ENABLED")
        os.environ["QR_TARGET_PATH_LIVE_ENABLED"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                client = FakeExecutionClient()

                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    live_enabled=True,
                ).run(
                    intents_path=_intents(root, metadata=_target_path_metadata(grade="B0", valid="NO")),
                    lane_id="lane:EUR_USD:LONG",
                    send=True,
                    confirm_live=True,
                )

                self.assertEqual(summary.status, "BLOCKED")
                self.assertEqual(client.orders, [])
                result = json.loads((root / "request.json").read_text())
                self.assertIn("TARGET_PATH_GRADE_TOO_LOW", {issue["code"] for issue in result["risk_issues"]})
        finally:
            _restore_env("QR_TARGET_PATH_LIVE_ENABLED", prior)

    def test_target_path_live_send_allows_a_grade_with_receipt(self) -> None:
        prior = os.environ.get("QR_TARGET_PATH_LIVE_ENABLED")
        os.environ["QR_TARGET_PATH_LIVE_ENABLED"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                client = FakeExecutionClient()

                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    live_enabled=True,
                ).run(
                    intents_path=_intents(root, metadata=_target_path_metadata(grade="A")),
                    lane_id="lane:EUR_USD:LONG",
                    send=True,
                    confirm_live=True,
                )

                self.assertEqual(summary.status, "SENT")
                self.assertTrue(summary.sent)
                self.assertEqual(len(client.orders), 1)
                result = json.loads((root / "request.json").read_text())
                receipt = result["target_path_receipt"]
                self.assertTrue(receipt["live_order_sent"])
                self.assertTrue(receipt["target_path_live_enabled"])
                self.assertEqual(receipt["target_path_live_mode"], "LIVE_LEARNING")
                self.assertEqual(receipt["daily_target_mode"], "ATTACK")
                self.assertEqual(receipt["five_pct_path_role"], "HERO")
                self.assertEqual(receipt["attack_stack_slot"], "NOW")
                self.assertEqual(receipt["grade"], "A")
                self.assertTrue(str(receipt["live_order_gateway_receipt_id"]).startswith("qrv1-EURUSD-L-"))
                report = (root / "report.md").read_text()
                self.assertIn("target-path receipt", report)
        finally:
            _restore_env("QR_TARGET_PATH_LIVE_ENABLED", prior)

    def test_target_path_live_send_allows_b_plus_support_reload(self) -> None:
        prior = os.environ.get("QR_TARGET_PATH_LIVE_ENABLED")
        os.environ["QR_TARGET_PATH_LIVE_ENABLED"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                client = FakeExecutionClient()

                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    live_enabled=True,
                ).run(
                    intents_path=_intents(
                        root,
                        metadata=_target_path_metadata(grade="B+", role="SUPPORT", slot="RELOAD"),
                    ),
                    lane_id="lane:EUR_USD:LONG",
                    send=True,
                    confirm_live=True,
                )

                self.assertEqual(summary.status, "SENT")
                self.assertEqual(len(client.orders), 1)
                result = json.loads((root / "request.json").read_text())
                self.assertEqual(result["target_path_receipt"]["five_pct_path_role"], "SUPPORT")
                self.assertEqual(result["target_path_receipt"]["attack_stack_slot"], "RELOAD")
                self.assertEqual(result["target_path_receipt"]["grade"], "B+")
        finally:
            _restore_env("QR_TARGET_PATH_LIVE_ENABLED", prior)

    def test_batch_send_posts_multiple_live_ready_orders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            intents = _intents(root)
            payload = json.loads(intents.read_text())
            second = json.loads(json.dumps(payload["results"][0]))
            second["lane_id"] = "lane:EUR_USD:LONG:reload"
            second["intent"]["entry"] = 1.17360
            second["intent"]["tp"] = 1.17480
            second["intent"]["sl"] = 1.17280
            payload["results"].append(second)
            intents.write_text(json.dumps(payload))

            with patch.object(
                execution_module,
                "clear_guardian_tuning_validation_cache",
                wraps=execution_module.clear_guardian_tuning_validation_cache,
            ) as clear_validation_cache:
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    live_enabled=True,
                ).run_batch(
                    intents_path=intents,
                    lane_ids=("lane:EUR_USD:LONG", "lane:EUR_USD:LONG:reload"),
                    send=True,
                    confirm_live=True,
                )

            self.assertEqual(summary.status, "SENT")
            self.assertEqual(summary.sent_count, 2)
            self.assertTrue(summary.sent)
            self.assertEqual(len(client.orders), 2)
            self.assertEqual(clear_validation_cache.call_count, 2)
            result = json.loads((root / "request.json").read_text())
            self.assertEqual(len(result["orders"]), 2)

    def test_batch_blocks_duplicate_parent_lane_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            intents = _intents(root)
            payload = json.loads(intents.read_text())
            market = json.loads(json.dumps(payload["results"][0]))
            market["lane_id"] = "lane:EUR_USD:LONG:MARKET"
            market["intent"]["order_type"] = "MARKET"
            market["intent"]["entry"] = 1.17306
            payload["results"].append(market)
            intents.write_text(json.dumps(payload))

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
            ).run_batch(
                intents_path=intents,
                lane_ids=("lane:EUR_USD:LONG", "lane:EUR_USD:LONG:MARKET"),
                send=True,
                confirm_live=True,
            )

            self.assertEqual(summary.status, "PARTIAL_SENT")
            self.assertEqual(summary.sent_count, 1)
            self.assertEqual(len(client.orders), 1)
            result = json.loads((root / "request.json").read_text())
            second = result["orders"][1]
            self.assertEqual(second["status"], "BLOCKED")
            self.assertIn("BASKET_DUPLICATE_PARENT_LANE", {issue["code"] for issue in second["risk_issues"]})

    def test_batch_blocks_same_pair_opposite_side_without_explicit_hedge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            intents = _intents(root, order_type="MARKET")
            payload = json.loads(intents.read_text())
            short = json.loads(json.dumps(payload["results"][0]))
            short["lane_id"] = "lane:EUR_USD:SHORT"
            short["intent"]["side"] = "SHORT"
            short["intent"]["entry"] = 1.17298
            short["intent"]["tp"] = 1.17180
            short["intent"]["sl"] = 1.17360
            short["intent"]["thesis"] = "opposite side should not share basket"
            payload["results"].append(short)
            intents.write_text(json.dumps(payload))

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
            ).run_batch(
                intents_path=intents,
                lane_ids=("lane:EUR_USD:LONG", "lane:EUR_USD:SHORT"),
                send=True,
                confirm_live=True,
            )

            self.assertEqual(summary.status, "PARTIAL_SENT")
            self.assertEqual(summary.sent_count, 1)
            self.assertEqual(len(client.orders), 1)
            result = json.loads((root / "request.json").read_text())
            second = result["orders"][1]
            self.assertEqual(second["status"], "BLOCKED")
            self.assertIn("BASKET_OPPOSING_PAIR_SIDE", {issue["code"] for issue in second["risk_issues"]})

    def test_batch_send_does_not_double_count_sent_margin_from_fresh_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = MutatingExecutionClient(margin_used_jpy=168_000.0)
            intents = _intents(root, order_type="MARKET")
            payload = json.loads(intents.read_text())
            second = json.loads(json.dumps(payload["results"][0]))
            second["lane_id"] = "lane:EUR_USD:LONG:reload"
            second["intent"]["tp"] = 1.17480
            payload["results"].append(second)
            intents.write_text(json.dumps(payload))

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
            ).run_batch(
                intents_path=intents,
                lane_ids=("lane:EUR_USD:LONG", "lane:EUR_USD:LONG:reload"),
                send=True,
                confirm_live=True,
            )

            self.assertEqual(summary.status, "SENT")
            self.assertEqual(summary.sent_count, 2)
            result = json.loads((root / "request.json").read_text())
            second_result = result["orders"][1]
            self.assertEqual(second_result["scaled_units"], 1000)
            self.assertNotIn(
                "BASKET_DOWNSIZED_FOR_CAPACITY",
                {issue["code"] for issue in second_result["risk_issues"]},
            )

    def test_capacity_downsize_floors_units_instead_of_rounding_up(self) -> None:
        scaled, issues = execution_module._scaled_units(4000, 0.28849)

        self.assertEqual(scaled, 1153)
        self.assertEqual(issues, [])

    def test_score_size_multiple_preserves_integer_unit_downsize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
            ).run_batch(
                intents_path=_intents(root),
                lane_ids=("lane:EUR_USD:LONG",),
                size_multiples={"lane:EUR_USD:LONG": 0.95},
            )

            self.assertEqual(summary.status, "STAGED")
            payload = json.loads((root / "request.json").read_text())
            order_result = payload["orders"][0]
            self.assertEqual(order_result["requested_units"], 1000)
            self.assertEqual(order_result["scaled_units"], 950)
            self.assertEqual(order_result["order_request"]["units"], "950")
            issue_codes = {issue["code"] for issue in order_result["risk_issues"]}
            self.assertNotIn("SIZE_MULTIPLE_CLAMPED_TO_MIN_LOT", issue_codes)
            self.assertNotIn("MIN_LOT_VIOLATION", issue_codes)

    def test_capacity_downsize_accepts_positive_sub_1000_units(self) -> None:
        scaled, issues = execution_module._scaled_units(
            1000,
            0.95,
            sub_min_lot_mode="block",
        )

        self.assertEqual(scaled, 950)
        self.assertEqual(issues, [])

    def test_portfolio_position_cap_scales_with_target_trade_pace(self) -> None:
        execution_module._target_trades_per_day_from_state = lambda path=None: 12
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "daily_target_state.json"
            state.write_text(json.dumps({"target_trades_per_day": 12}))

            cap = execution_module._portfolio_position_cap_from_state(
                state,
                policy=SimpleNamespace(max_portfolio_positions=2),
            )

            self.assertEqual(cap, 4)

    def test_capacity_downsize_leaves_integer_margin_headroom(self) -> None:
        now = datetime.now(timezone.utc)
        intent = execution_module._intent_from_json(
            {
                "pair": "EUR_USD",
                "side": "LONG",
                "order_type": "MARKET",
                "units": 4000,
                "entry": 1.17306,
                "tp": 1.17450,
                "sl": 1.17250,
                "thesis": "trend continuation",
                "owner": "trader",
                "market_context": {
                    "regime": "TREND_CONTINUATION campaign lane",
                    "narrative": "trend continuation pressure",
                    "chart_story": "trend staircase",
                    "method": "TREND_CONTINUATION",
                    "invalidation": "SL trades",
                },
            }
        )
        snapshot = BrokerSnapshot(
            fetched_at_utc=now,
            positions=(),
            orders=(),
            quotes={
                "EUR_USD": Quote("EUR_USD", bid=1.17298, ask=1.17306, timestamp_utc=now),
                "AUD_USD": Quote("AUD_USD", bid=0.66210, ask=0.66218, timestamp_utc=now),
                "USD_JPY": Quote("USD_JPY", bid=157.0, ask=157.01, timestamp_utc=now),
            },
            account=AccountSummary(
                nav_jpy=200_000.0,
                balance_jpy=200_000.0,
                margin_used_jpy=175_510.0,
                margin_available_jpy=200_000.0,
                last_transaction_id="100",
                fetched_at_utc=now,
            ),
            home_conversions={"USD": 157.0},
        )
        metrics = SimpleNamespace(risk_jpy=400.0, estimated_margin_jpy=29_465.0)

        scale, issue = execution_module._basket_size_multiple(
            intent=intent,
            snapshot=snapshot,
            metrics=metrics,
            portfolio_loss_cap=None,
            cumulative_risk_jpy=0.0,
            cumulative_margin_jpy=0.0,
        )
        scaled, scale_issues = execution_module._scaled_units(intent.units, scale)

        self.assertIsNotNone(issue)
        self.assertEqual(issue.code, "BASKET_DOWNSIZED_FOR_CAPACITY")
        self.assertEqual(scale_issues, [])
        self.assertLessEqual((metrics.estimated_margin_jpy / intent.units) * scaled, 8_490.0)
        self.assertLess(scaled, 1153)

    def test_default_risk_cap_reads_daily_target_state_before_policy_literal(self) -> None:
        original_reader = execution_module._per_trade_risk_from_state
        execution_module._per_trade_risk_from_state = lambda: 100.0
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                client = FakeExecutionClient()

                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    live_enabled=True,
                ).run(intents_path=_intents(root), lane_id="lane:EUR_USD:LONG")

                self.assertEqual(summary.status, "STAGED")
                payload = json.loads((root / "request.json").read_text())
                self.assertLess(payload["scaled_units"], payload["requested_units"])
                self.assertLessEqual(payload["risk_metrics"]["risk_jpy"], 100.0)
                self.assertIn(
                    "SIZE_MULTIPLE_CLIPPED_TO_LOSS_CAP",
                    {issue["code"] for issue in payload["risk_issues"]},
                )
        finally:
            execution_module._per_trade_risk_from_state = original_reader

    def test_send_blocks_when_candidate_exceeds_margin_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            now = client.snapshot_value.fetched_at_utc
            client.snapshot_value = BrokerSnapshot(
                fetched_at_utc=now,
                positions=(),
                orders=(),
                quotes=client.snapshot_value.quotes,
                account=AccountSummary(
                    nav_jpy=220_145.7765,
                    balance_jpy=208_945.7765,
                    margin_used_jpy=156_414.0,
                    margin_available_jpy=63_831.7765,
                    fetched_at_utc=now,
                ),
            )
            intents = _intents(root, order_type="MARKET")
            payload = json.loads(intents.read_text())
            payload["results"][0]["intent"]["units"] = 13_000
            intents.write_text(json.dumps(payload))

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
                max_loss_jpy=2_000.0,
            ).run(intents_path=intents, lane_id="lane:EUR_USD:LONG", send=True, confirm_live=True)

            self.assertEqual(summary.status, "BLOCKED")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders, [])
            result = json.loads((root / "request.json").read_text())
            self.assertIn("MARGIN_UTILIZATION_CAP_EXCEEDED", {issue["code"] for issue in result["risk_issues"]})

    def test_send_blocks_when_required_position_guardian_is_inactive(self) -> None:
        prior_required = os.environ.get("QR_REQUIRE_POSITION_GUARDIAN_ACTIVE")
        prior_active = os.environ.get("QR_POSITION_GUARDIAN_ACTIVE")
        os.environ["QR_REQUIRE_POSITION_GUARDIAN_ACTIVE"] = "1"
        os.environ["QR_POSITION_GUARDIAN_ACTIVE"] = "0"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                client = FakeExecutionClient()

                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    live_enabled=True,
                ).run(intents_path=_intents(root), lane_id="lane:EUR_USD:LONG", send=True, confirm_live=True)

                self.assertEqual(summary.status, "BLOCKED")
                self.assertFalse(summary.sent)
                self.assertEqual(client.orders, [])
                result = json.loads((root / "request.json").read_text())
                self.assertIn(
                    "POSITION_GUARDIAN_INACTIVE_FOR_SEND",
                    {issue["code"] for issue in result["risk_issues"]},
                )
        finally:
            _restore_env("QR_REQUIRE_POSITION_GUARDIAN_ACTIVE", prior_required)
            _restore_env("QR_POSITION_GUARDIAN_ACTIVE", prior_active)

    def test_send_position_guardian_requirement_has_explicit_operator_override(self) -> None:
        prior_required = os.environ.get("QR_REQUIRE_POSITION_GUARDIAN_ACTIVE")
        prior_active = os.environ.get("QR_POSITION_GUARDIAN_ACTIVE")
        os.environ["QR_REQUIRE_POSITION_GUARDIAN_ACTIVE"] = "0"
        os.environ["QR_POSITION_GUARDIAN_ACTIVE"] = "0"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                client = FakeExecutionClient()

                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    live_enabled=True,
                ).run(intents_path=_intents(root), lane_id="lane:EUR_USD:LONG", send=True, confirm_live=True)

                self.assertEqual(summary.status, "SENT")
                self.assertTrue(summary.sent)
                self.assertEqual(len(client.orders), 1)
                result = json.loads((root / "request.json").read_text())
                self.assertNotIn(
                    "POSITION_GUARDIAN_INACTIVE_FOR_SEND",
                    {issue["code"] for issue in result["risk_issues"]},
                )
        finally:
            _restore_env("QR_REQUIRE_POSITION_GUARDIAN_ACTIVE", prior_required)
            _restore_env("QR_POSITION_GUARDIAN_ACTIVE", prior_active)

    def test_direct_send_fallback_blocks_when_loaded_guardian_lacks_fresh_heartbeat(self) -> None:
        env_keys = (
            "PATH",
            "QR_REQUIRE_POSITION_GUARDIAN_ACTIVE",
            "QR_POSITION_GUARDIAN_ACTIVE",
            "QR_POSITION_GUARDIAN_LABEL",
            "QR_POSITION_GUARDIAN_PLIST",
            "QR_POSITION_GUARDIAN_EXECUTION",
            "QR_POSITION_GUARDIAN_HEARTBEAT",
            "QR_POSITION_GUARDIAN_INTERVAL",
            "QR_POSITION_GUARDIAN_HEARTBEAT_MAX_AGE_SECONDS",
            "QR_POSITION_GUARDIAN_REQUIRE_HEARTBEAT",
            "QR_FAKE_POSITION_GUARDIAN_LOADED",
        )
        prior = {key: os.environ.get(key) for key in env_keys}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                plist = _write_guardian_plist(root)
                _install_fake_launchctl(root, loaded=True)
                os.environ["QR_REQUIRE_POSITION_GUARDIAN_ACTIVE"] = "1"
                os.environ.pop("QR_POSITION_GUARDIAN_ACTIVE", None)
                os.environ["QR_POSITION_GUARDIAN_LABEL"] = "com.quantrabbit.position-guardian"
                os.environ["QR_POSITION_GUARDIAN_PLIST"] = str(plist)
                os.environ["QR_POSITION_GUARDIAN_EXECUTION"] = str(root / "missing_execution.json")
                os.environ["QR_POSITION_GUARDIAN_HEARTBEAT"] = str(root / "missing_heartbeat.json")
                os.environ["QR_POSITION_GUARDIAN_INTERVAL"] = "30"
                os.environ["QR_POSITION_GUARDIAN_REQUIRE_HEARTBEAT"] = "1"

                client = FakeExecutionClient()
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    live_enabled=True,
                ).run(intents_path=_intents(root), lane_id="lane:EUR_USD:LONG", send=True, confirm_live=True)

                self.assertEqual(summary.status, "BLOCKED")
                self.assertFalse(summary.sent)
                self.assertEqual(client.orders, [])
                result = json.loads((root / "request.json").read_text())
                self.assertIn(
                    "POSITION_GUARDIAN_INACTIVE_FOR_SEND",
                    {issue["code"] for issue in result["risk_issues"]},
                )
        finally:
            for key, value in prior.items():
                _restore_env(key, value)

    def test_direct_send_fallback_allows_loaded_guardian_with_fresh_heartbeat(self) -> None:
        env_keys = (
            "PATH",
            "QR_REQUIRE_POSITION_GUARDIAN_ACTIVE",
            "QR_POSITION_GUARDIAN_ACTIVE",
            "QR_POSITION_GUARDIAN_LABEL",
            "QR_POSITION_GUARDIAN_PLIST",
            "QR_POSITION_GUARDIAN_EXECUTION",
            "QR_POSITION_GUARDIAN_HEARTBEAT",
            "QR_POSITION_GUARDIAN_INTERVAL",
            "QR_POSITION_GUARDIAN_HEARTBEAT_MAX_AGE_SECONDS",
            "QR_POSITION_GUARDIAN_REQUIRE_HEARTBEAT",
            "QR_FAKE_POSITION_GUARDIAN_LOADED",
        )
        prior = {key: os.environ.get(key) for key in env_keys}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                plist = _write_guardian_plist(root)
                heartbeat = _write_guardian_heartbeat(root)
                _install_fake_launchctl(root, loaded=True)
                os.environ["QR_REQUIRE_POSITION_GUARDIAN_ACTIVE"] = "1"
                os.environ.pop("QR_POSITION_GUARDIAN_ACTIVE", None)
                os.environ["QR_POSITION_GUARDIAN_LABEL"] = "com.quantrabbit.position-guardian"
                os.environ["QR_POSITION_GUARDIAN_PLIST"] = str(plist)
                os.environ["QR_POSITION_GUARDIAN_EXECUTION"] = str(heartbeat)
                os.environ["QR_POSITION_GUARDIAN_HEARTBEAT"] = str(root / "missing_heartbeat.json")
                os.environ["QR_POSITION_GUARDIAN_INTERVAL"] = "30"
                os.environ["QR_POSITION_GUARDIAN_REQUIRE_HEARTBEAT"] = "1"

                client = FakeExecutionClient()
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    live_enabled=True,
                ).run(intents_path=_intents(root), lane_id="lane:EUR_USD:LONG", send=True, confirm_live=True)

                self.assertEqual(summary.status, "SENT")
                self.assertTrue(summary.sent)
                self.assertEqual(len(client.orders), 1)
                result = json.loads((root / "request.json").read_text())
                self.assertNotIn(
                    "POSITION_GUARDIAN_INACTIVE_FOR_SEND",
                    {issue["code"] for issue in result["risk_issues"]},
                )
        finally:
            for key, value in prior.items():
                _restore_env(key, value)

    def test_send_allows_same_pair_hedge_when_margin_cap_is_full(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            now = client.snapshot_value.fetched_at_utc
            client.snapshot_value = BrokerSnapshot(
                fetched_at_utc=now,
                positions=(
                    BrokerPosition(
                        trade_id="101",
                        pair="EUR_USD",
                        side=Side.LONG,
                        units=22_000,
                        entry_price=1.16688,
                        take_profit=1.17100,
                        stop_loss=1.16600,
                        owner=Owner.TRADER,
                    ),
                ),
                orders=(),
                quotes=client.snapshot_value.quotes,
                account=AccountSummary(
                    nav_jpy=175_988.7367,
                    balance_jpy=192_275.8359,
                    margin_used_jpy=162_740.16,
                    margin_available_jpy=13_436.9823,
                    hedging_enabled=True,
                    fetched_at_utc=now,
                ),
            )
            intents = _intents(
                root,
                metadata={
                    "desk": "failure_trader",
                    "campaign_role": "NOW",
                    "parent_lane_id": "lane:EUR_USD:SHORT",
                    "forecast_cycle_id": "hedge-forecast-cycle-1",
                    "position_intent": "HEDGE",
                    "position_fill": "OPEN_ONLY",
                    "hedge_timing_class": "OPPOSITE_EXPOSURE",
                    "hedge_unwind_plan_required": True,
                    "hedge_review_trigger": "next_m15_close_or_structure_change",
                },
            )
            payload = json.loads(intents.read_text())
            result = payload["results"][0]
            result["lane_id"] = "lane:EUR_USD:SHORT"
            intent = result["intent"]
            intent["side"] = "SHORT"
            intent["entry"] = 1.17270
            intent["tp"] = 1.17120
            intent["sl"] = 1.17350
            intent["thesis"] = "same-pair hedge against existing long"
            intent["market_context"] = {
                "regime": "BREAKOUT_FAILURE reject/retest current",
                "narrative": "short hedge against trapped long exposure",
                "chart_story": "failed reclaim and rejection below trigger",
                "method": "BREAKOUT_FAILURE",
                "invalidation": "SL trades",
            }
            intents.write_text(json.dumps(payload))

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root, direction="SHORT"),
                output_path=root / "request.json",
                report_path=root / "report.md",
                verified_decision_path=_write_ordinary_verified_decision(
                    root,
                    lane_id="lane:EUR_USD:SHORT",
                    direction="SHORT",
                ),
                live_enabled=True,
                max_loss_jpy=2_000.0,
            ).run(intents_path=intents, lane_id="lane:EUR_USD:SHORT", send=True, confirm_live=True)

            self.assertEqual(
                summary.status,
                "SENT",
                msg=(root / "request.json").read_text(),
            )
            self.assertTrue(summary.sent)
            self.assertEqual(len(client.orders), 1)
            self.assertEqual(client.orders[0]["units"], "-1000")
            self.assertEqual(client.orders[0]["positionFill"], "OPEN_ONLY")
            request = json.loads((root / "request.json").read_text())
            self.assertEqual(request["risk_metrics"]["estimated_margin_jpy"], 0.0)
            self.assertNotIn("MARGIN_UTILIZATION_CAP_EXCEEDED", {issue["code"] for issue in request["risk_issues"]})

    def test_hedge_intent_uses_open_only_position_fill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            now = client.snapshot_value.fetched_at_utc
            client.snapshot_value = BrokerSnapshot(
                fetched_at_utc=now,
                positions=(
                    BrokerPosition(
                        trade_id="101",
                        pair="EUR_USD",
                        side=Side.SHORT,
                        units=22_000,
                        entry_price=1.17000,
                        take_profit=1.16400,
                        stop_loss=1.17300,
                        owner=Owner.TRADER,
                    ),
                ),
                orders=(),
                quotes=client.snapshot_value.quotes,
                account=AccountSummary(
                    nav_jpy=200_000.0,
                    balance_jpy=200_000.0,
                    margin_used_jpy=50_000.0,
                    margin_available_jpy=150_000.0,
                    hedging_enabled=True,
                    fetched_at_utc=now,
                ),
            )
            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
            ).run(
                intents_path=_intents(
                    root,
                    metadata={
                        "desk": "range_trader",
                        "campaign_role": "BACKUP_OR_RELOAD",
                        "position_intent": "HEDGE",
                        "hedge_timing_class": "OPPOSITE_EXPOSURE",
                        "hedge_unwind_plan_required": True,
                        "hedge_review_trigger": "next_m15_close_or_structure_change",
                    },
                )
            )

            self.assertEqual(summary.status, "STAGED")
            payload = json.loads((root / "request.json").read_text())
            self.assertEqual(payload["order_request"]["positionFill"], "OPEN_ONLY")

    def test_existing_protected_position_blocks_when_portfolio_budget_exceeded(self) -> None:
        # Per AGENT_CONTRACT §3.5: portfolio cap is the WHOLE-DAY risk budget,
        # not the per-trade slice. We assert the gateway blocks new entries
        # only when open_risk + candidate_risk exceeds the day's budget — set
        # an explicit small portfolio cap to force that condition without
        # relying on the per-trade resolver.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            client.snapshot_value = BrokerSnapshot(
                fetched_at_utc=client.snapshot_value.fetched_at_utc,
                positions=(
                    BrokerPosition(
                        trade_id="101",
                        pair="EUR_USD",
                        side=Side.LONG,
                        units=5000,
                        entry_price=1.1710,
                        take_profit=1.1750,
                        stop_loss=1.1690,
                        owner=Owner.TRADER,
                    ),
                ),
                orders=(),
                quotes=client.snapshot_value.quotes,
                account=client.snapshot_value.account,
            )

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
                portfolio_loss_cap_jpy=500.0,
            ).run(
                intents_path=_intents(root),
                lane_id="lane:EUR_USD:LONG",
                send=True,
                confirm_live=True,
            )

            self.assertEqual(summary.status, "BLOCKED")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders, [])
            payload = json.loads((root / "request.json").read_text())
            self.assertIn("PORTFOLIO_LOSS_CAP_EXCEEDED", {issue["code"] for issue in payload["risk_issues"]})

    def test_gateway_uses_loss_capacity_before_open_without_double_counting_state_open_risk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_state = root / "daily_target_state.json"
            target_state.write_text(
                json.dumps(
                    {
                        "daily_risk_budget_jpy": 1_000.0,
                        "realized_loss_spent_jpy": 800.0,
                        "daily_loss_capacity_before_open_jpy": 200.0,
                        # This already subtracts the broker position below and
                        # must not be used as the gateway portfolio cap.
                        "open_risk_jpy": 91.0,
                        "remaining_risk_budget_jpy": 109.0,
                        "base_per_trade_risk_budget_jpy": 100.0,
                        "per_trade_risk_budget_jpy": 100.0,
                    }
                )
            )
            client = FakeExecutionClient()
            client.snapshot_value = BrokerSnapshot(
                fetched_at_utc=client.snapshot_value.fetched_at_utc,
                positions=(
                    BrokerPosition(
                        trade_id="loss-cap-open",
                        pair="EUR_USD",
                        side=Side.LONG,
                        units=700,
                        entry_price=1.1710,
                        take_profit=1.1750,
                        stop_loss=1.1700,
                        owner=Owner.TRADER,
                    ),
                ),
                orders=(),
                quotes=client.snapshot_value.quotes,
                account=client.snapshot_value.account,
            )

            with patch.object(
                execution_module,
                "_daily_risk_budget_from_state",
                self._original_daily_budget_reader,
            ):
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    target_state_path=target_state,
                    live_enabled=True,
                ).run(
                    intents_path=_intents(root),
                    lane_id="lane:EUR_USD:LONG",
                    send=False,
                )

            self.assertEqual(summary.status, "STAGED")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders, [])
            payload = json.loads((root / "request.json").read_text())
            self.assertLess(payload["scaled_units"], payload["requested_units"])
            self.assertLessEqual(payload["risk_metrics"]["risk_jpy"], 91.0)
            self.assertIn(
                "SIZE_MULTIPLE_CLIPPED_TO_ATTACHED_STOP_CAP",
                {issue["code"] for issue in payload["risk_issues"]},
            )

    def test_ordinary_gpt_receipt_claim_blocks_duplicate_direct_post(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lane_id = "lane:EUR_USD:LONG"
            ledger_path = root / "execution_ledger.db"
            intents = _intents(
                root,
                lane_id=lane_id,
                metadata=_ordinary_claim_metadata(),
            )
            verified = _write_ordinary_verified_decision(root, lane_id=lane_id)
            client = FakeExecutionClient()
            gateway = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                execution_ledger_db_path=ledger_path,
                verified_decision_path=verified,
                live_enabled=True,
            )

            first = gateway.run(
                intents_path=intents,
                lane_id=lane_id,
                send=True,
                confirm_live=True,
            )
            second = gateway.run(
                intents_path=intents,
                lane_id=lane_id,
                send=True,
                confirm_live=True,
            )
            result = json.loads((root / "request.json").read_text())
            with closing(sqlite3.connect(ledger_path)) as conn, conn:
                rows = conn.execute(
                    "SELECT status, parent_lane_id, forecast_cycle_id FROM ordinary_live_entry_signal_claims"
                ).fetchall()

            self.assertTrue(first.sent)
            self.assertFalse(second.sent)
            self.assertEqual(len(client.orders), 1)
            self.assertEqual(rows, [("BROKER_RESPONSE_RECORDED", lane_id, "forecast-cycle-1")])
            self.assertIn(
                "ORDINARY_ENTRY_RECEIPT_ALREADY_CLAIMED",
                {issue["code"] for issue in result["risk_issues"]},
            )

    def test_accepted_gpt_trade_lineage_reaches_request_claim_receipt_and_explicit_fill_link(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lane_id = "lane:EUR_USD:LONG"
            ledger_path = root / "execution_ledger.db"
            links_path = root / "market_read_execution_links.jsonl"
            intents = _intents(
                root,
                lane_id=lane_id,
                metadata=_ordinary_claim_metadata(),
            )
            verified = _write_lineaged_verified_decision(root, lane_id=lane_id)
            expected = json.loads(verified.read_text())["market_read_prediction"]

            class ExplicitFillClient(FakeExecutionClient):
                def post_order_json(self, order_request: dict[str, Any]) -> dict[str, Any]:
                    self.orders.append(order_request)
                    return {
                        "orderCreateTransaction": {"id": "101"},
                        "orderFillTransaction": {
                            "id": "102",
                            "orderID": "101",
                            "tradeOpened": {"tradeID": "200"},
                        },
                        "relatedTransactionIDs": ["101", "102"],
                        "lastTransactionID": "102",
                    }

            client = ExplicitFillClient()
            record_result = SimpleNamespace(
                status="RECORDED",
                issue=None,
                to_dict=lambda: {"status": "RECORDED", "trade_id": "200"},
            )
            with patch(
                "quant_rabbit.strategy.entry_thesis_ledger.record_entry_thesis_from_response_result",
                return_value=record_result,
            ):
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    execution_ledger_db_path=ledger_path,
                    verified_decision_path=verified,
                    market_read_execution_links_path=links_path,
                    live_enabled=True,
                ).run(
                    intents_path=intents,
                    lane_id=lane_id,
                    send=True,
                    confirm_live=True,
                )

            self.assertEqual(summary.status, "SENT")
            self.assertTrue(summary.sent)
            self.assertEqual(len(client.orders), 1)
            sent_request = client.orders[0]
            client_id = sent_request["clientExtensions"]["id"]
            self.assertLessEqual(len(client_id), 128)
            self.assertEqual(sent_request["clientExtensions"]["tag"], "trader")
            self.assertIn("lane=lane:EUR_USD:LONG", sent_request["clientExtensions"]["comment"])

            receipt = json.loads((root / "request.json").read_text())
            lineage = receipt["decision_lineage"]
            self.assertEqual(lineage["decision_receipt_id"], expected["decision_receipt_id"])
            self.assertEqual(lineage["market_read_prediction_id"], expected["prediction_id"])
            self.assertIn(lineage["lineage_token"], client_id)
            self.assertEqual(
                receipt["ordinary_entry_claim"]["decision_lineage"]["decision_receipt_id"],
                expected["decision_receipt_id"],
            )
            self.assertEqual(receipt["market_read_execution_link"]["status"], "RECORDED")
            links = lineage_module.read_execution_links(links_path)
            self.assertEqual(len(links), 1)
            self.assertEqual(links[0]["broker_ids"]["order_ids"], ["101"])
            self.assertEqual(links[0]["broker_ids"]["fill_transaction_ids"], ["102"])
            self.assertEqual(links[0]["broker_ids"]["trade_ids"], ["200"])
            self.assertFalse(links[0]["pair_or_time_inference_used"])
            with closing(sqlite3.connect(ledger_path)) as conn, conn:
                claim_ids = conn.execute(
                    """
                    SELECT decision_receipt_id, market_read_prediction_id,
                           decision_lineage_token, status
                    FROM ordinary_live_entry_signal_claims
                    """
                ).fetchone()
            self.assertEqual(
                claim_ids,
                (
                    expected["decision_receipt_id"],
                    expected["prediction_id"],
                    lineage["lineage_token"],
                    "BROKER_RESPONSE_RECORDED",
                ),
            )

    def test_batch_lineage_write_gap_stays_sent_and_one_shot_claim_prevents_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lane_id = "lane:EUR_USD:LONG"
            ledger_path = root / "execution_ledger.db"
            intents = _intents(
                root,
                lane_id=lane_id,
                metadata=_ordinary_claim_metadata(),
            )
            verified = _write_lineaged_verified_decision(root, lane_id=lane_id)

            class ExplicitOrderClient(FakeExecutionClient):
                def post_order_json(self, order_request: dict[str, Any]) -> dict[str, Any]:
                    self.orders.append(order_request)
                    return {
                        "orderCreateTransaction": {"id": "101"},
                        "relatedTransactionIDs": ["101"],
                    }

            client = ExplicitOrderClient()
            gateway = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                execution_ledger_db_path=ledger_path,
                verified_decision_path=verified,
                market_read_execution_links_path=root / "links.jsonl",
                live_enabled=True,
            )
            with patch.object(
                execution_module,
                "append_execution_link",
                side_effect=OSError("simulated lineage disk failure"),
            ):
                first = gateway.run_batch(
                    intents_path=intents,
                    lane_ids=(lane_id,),
                    send=True,
                    confirm_live=True,
                )

            self.assertEqual(first.status, "SENT_WITH_DECISION_LINEAGE_GAP")
            self.assertTrue(first.sent)
            self.assertEqual(first.sent_count, 1)
            self.assertEqual(len(client.orders), 1)
            first_receipt = json.loads((root / "request.json").read_text())
            self.assertEqual(
                first_receipt["orders"][0]["market_read_execution_link"]["status"],
                "WRITE_FAILED",
            )

            retry = gateway.run_batch(
                intents_path=intents,
                lane_ids=(lane_id,),
                send=True,
                confirm_live=True,
            )
            self.assertEqual(retry.status, "BLOCKED")
            self.assertFalse(retry.sent)
            self.assertEqual(len(client.orders), 1)
            retry_receipt = json.loads((root / "request.json").read_text())
            self.assertIn(
                "ORDINARY_ENTRY_RECEIPT_ALREADY_CLAIMED",
                {
                    issue["code"]
                    for issue in retry_receipt["orders"][0]["risk_issues"]
                },
            )

    def test_ordinary_gpt_receipt_claim_blocks_duplicate_batch_post(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lane_id = "lane:EUR_USD:LONG"
            ledger_path = root / "execution_ledger.db"
            intents = _intents(
                root,
                lane_id=lane_id,
                metadata=_ordinary_claim_metadata(),
            )
            verified = _write_ordinary_verified_decision(root, lane_id=lane_id)
            client = FakeExecutionClient()
            gateway = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                execution_ledger_db_path=ledger_path,
                verified_decision_path=verified,
                live_enabled=True,
            )

            first = gateway.run_batch(
                intents_path=intents,
                lane_ids=(lane_id,),
                send=True,
                confirm_live=True,
            )
            second = gateway.run_batch(
                intents_path=intents,
                lane_ids=(lane_id,),
                send=True,
                confirm_live=True,
            )
            result = json.loads((root / "request.json").read_text())

            self.assertTrue(first.sent)
            self.assertFalse(second.sent)
            self.assertEqual(len(client.orders), 1)
            self.assertIn(
                "ORDINARY_ENTRY_RECEIPT_ALREADY_CLAIMED",
                {issue["code"] for issue in result["orders"][0]["risk_issues"]},
            )

    def test_ordinary_add_requires_new_receipt_and_new_forecast_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lane_id = "lane:EUR_USD:LONG"
            ledger_path = root / "execution_ledger.db"
            intents = _intents(
                root,
                lane_id=lane_id,
                metadata=_ordinary_claim_metadata(),
            )
            client = FakeExecutionClient()

            first_gateway = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                execution_ledger_db_path=ledger_path,
                verified_decision_path=_write_ordinary_verified_decision(
                    root,
                    lane_id=lane_id,
                    suffix="trade",
                ),
                live_enabled=True,
            )
            first = first_gateway.run(
                intents_path=intents,
                lane_id=lane_id,
                send=True,
                confirm_live=True,
            )

            add_receipt = _write_ordinary_verified_decision(
                root,
                lane_id=lane_id,
                suffix="add",
            )
            add_gateway = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                execution_ledger_db_path=ledger_path,
                verified_decision_path=add_receipt,
                live_enabled=True,
            )
            same_cycle = add_gateway.run(
                intents_path=intents,
                lane_id=lane_id,
                send=True,
                confirm_live=True,
            )
            same_cycle_result = json.loads((root / "request.json").read_text())

            intent_payload = json.loads(intents.read_text())
            intent_payload["results"][0]["intent"]["metadata"][
                "forecast_cycle_id"
            ] = "forecast-cycle-2"
            intents.write_text(json.dumps(intent_payload), encoding="utf-8")
            new_signal = add_gateway.run(
                intents_path=intents,
                lane_id=lane_id,
                send=True,
                confirm_live=True,
            )
            replayed_add = add_gateway.run(
                intents_path=intents,
                lane_id=lane_id,
                send=True,
                confirm_live=True,
            )
            replay_result = json.loads((root / "request.json").read_text())
            with closing(sqlite3.connect(ledger_path)) as conn, conn:
                claim_count = conn.execute(
                    "SELECT COUNT(*) FROM ordinary_live_entry_signal_claims"
                ).fetchone()[0]

            self.assertTrue(first.sent)
            self.assertFalse(same_cycle.sent)
            self.assertIn(
                "ORDINARY_ENTRY_FORECAST_CYCLE_ALREADY_CLAIMED",
                {issue["code"] for issue in same_cycle_result["risk_issues"]},
            )
            self.assertTrue(new_signal.sent)
            self.assertFalse(replayed_add.sent)
            self.assertEqual(len(client.orders), 2)
            self.assertEqual(claim_count, 2)
            self.assertIn(
                "ORDINARY_ENTRY_RECEIPT_ALREADY_CLAIMED",
                {issue["code"] for issue in replay_result["risk_issues"]},
            )

    def test_ordinary_claim_survives_process_crash_before_post_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lane_id = "lane:EUR_USD:LONG"
            ledger_path = root / "execution_ledger.db"
            intents = _intents(
                root,
                lane_id=lane_id,
                metadata=_ordinary_claim_metadata(),
            )
            verified = _write_ordinary_verified_decision(root, lane_id=lane_id)
            crashing_gateway = LiveOrderGateway(
                client=CrashBeforePostExecutionClient(),
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                execution_ledger_db_path=ledger_path,
                verified_decision_path=verified,
                live_enabled=True,
            )

            with self.assertRaises(SystemExit):
                crashing_gateway.run(
                    intents_path=intents,
                    lane_id=lane_id,
                    send=True,
                    confirm_live=True,
                )

            retry_client = FakeExecutionClient()
            retry = LiveOrderGateway(
                client=retry_client,
                strategy_profile=_profile(root),
                output_path=root / "retry.json",
                report_path=root / "retry.md",
                execution_ledger_db_path=ledger_path,
                verified_decision_path=verified,
                live_enabled=True,
            ).run(
                intents_path=intents,
                lane_id=lane_id,
                send=True,
                confirm_live=True,
            )
            with closing(sqlite3.connect(ledger_path)) as conn, conn:
                claim_status = conn.execute(
                    "SELECT status FROM ordinary_live_entry_signal_claims"
                ).fetchone()[0]

            self.assertFalse(retry.sent)
            self.assertEqual(retry_client.orders, [])
            self.assertEqual(claim_status, "RESERVED_PRE_POST")
            retry_result = json.loads((root / "retry.json").read_text())
            self.assertIn(
                "ORDINARY_ENTRY_RECEIPT_ALREADY_CLAIMED",
                {issue["code"] for issue in retry_result["risk_issues"]},
            )

    def test_pre_post_reconciliation_blocks_direct_fresh_duplicate_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_state, target_report, ledger_path, ledger_report = _reconciliation_files(root)
            pending = _trader_pending_order(
                order_id="fresh-duplicate-direct",
                lane_id="lane:EUR_USD:LONG:STOP",
            )
            client = FreshPendingReconciliationExecutionClient((pending,))

            with patch.object(
                LiveOrderGateway,
                "_pre_post_reconcile",
                self._original_pre_post_reconcile,
            ):
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    target_state_path=target_state,
                    target_report_path=target_report,
                    execution_ledger_db_path=ledger_path,
                    execution_ledger_report_path=ledger_report,
                    live_enabled=True,
                ).run(
                    intents_path=_intents(root),
                    lane_id="lane:EUR_USD:LONG",
                    send=True,
                    confirm_live=True,
                )

            self.assertFalse(summary.sent)
            self.assertEqual(client.orders, [])
            result = json.loads((root / "request.json").read_text())
            codes = {issue["code"] for issue in result["risk_issues"]}
            self.assertIn("PRE_POST_DUPLICATE_PENDING_GEOMETRY", codes)
            self.assertIn("PRE_POST_DUPLICATE_PENDING_PARENT_LANE", codes)
            self.assertEqual(
                result["pre_post_reconciliation"]["fresh_pending_reconciliation"]["status"],
                "BLOCKED",
            )

    def test_pre_post_reconciliation_revokes_same_vehicle_edge_direct_and_batch(self) -> None:
        for batch in (False, True):
            with self.subTest(batch=batch), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                target_state, target_report, ledger_path, ledger_report = (
                    _reconciliation_files(root)
                )
                _insert_exact_stop_all_exit_outcomes(
                    ledger_path,
                    outcomes=[-5000.0],
                    prefix="new-selected-loss",
                    age_days=0,
                )
                client = ReconciliationExecutionClient()
                gateway = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    target_state_path=target_state,
                    target_report_path=target_report,
                    execution_ledger_db_path=ledger_path,
                    execution_ledger_report_path=ledger_report,
                    live_enabled=True,
                )

                with patch.object(
                    LiveOrderGateway,
                    "_pre_post_reconcile",
                    self._original_pre_post_reconcile,
                ):
                    if batch:
                        summary = gateway.run_batch(
                            intents_path=_intents(root),
                            lane_ids=("lane:EUR_USD:LONG",),
                            send=True,
                            confirm_live=True,
                        )
                    else:
                        summary = gateway.run(
                            intents_path=_intents(root),
                            lane_id="lane:EUR_USD:LONG",
                            send=True,
                            confirm_live=True,
                        )

                self.assertFalse(summary.sent)
                self.assertEqual(client.orders, [])
                payload = json.loads((root / "request.json").read_text())
                order = payload["orders"][0] if batch else payload
                self.assertIn(
                    "PRE_POST_GPT_ALLOCATION_EDGE_STALE",
                    {issue["code"] for issue in order["risk_issues"]},
                )
                recheck = order["pre_post_reconciliation"][
                    "capital_allocation_edge_recheck"
                ]
                self.assertEqual(recheck["status"], "BLOCKED")
                self.assertEqual(recheck["basis"], "EXACT_VEHICLE_ALL_EXIT_NET")
                self.assertIn(
                    "ALL_EXIT_EDGE_NO_LONGER_PROVEN",
                    recheck["failed_checks"],
                )

    def test_pre_post_reconciliation_revokes_unresolved_selected_reduction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_state, target_report, ledger_path, ledger_report = (
                _reconciliation_files(root)
            )
            _insert_exact_stop_unresolved_reduction(ledger_path)
            client = ReconciliationExecutionClient()

            with patch.object(
                LiveOrderGateway,
                "_pre_post_reconcile",
                self._original_pre_post_reconcile,
            ):
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    target_state_path=target_state,
                    target_report_path=target_report,
                    execution_ledger_db_path=ledger_path,
                    execution_ledger_report_path=ledger_report,
                    live_enabled=True,
                ).run(
                    intents_path=_intents(root),
                    lane_id="lane:EUR_USD:LONG",
                    send=True,
                    confirm_live=True,
                )

            self.assertFalse(summary.sent)
            self.assertEqual(client.orders, [])
            payload = json.loads((root / "request.json").read_text())
            self.assertIn(
                "PRE_POST_GPT_ALLOCATION_EDGE_STALE",
                {issue["code"] for issue in payload["risk_issues"]},
            )
            recheck = payload["pre_post_reconciliation"][
                "capital_allocation_edge_recheck"
            ]
            self.assertEqual(recheck["status"], "BLOCKED")
            self.assertEqual(recheck["basis"], "EXACT_VEHICLE_ALL_EXIT_NET")
            self.assertEqual(
                recheck["current_all_exit"]["unresolved_realized_trades"],
                1,
            )
            self.assertIn(
                "ALL_EXIT_EDGE_NO_LONGER_PROVEN",
                recheck["failed_checks"],
            )

    def test_pre_post_reconciliation_blocks_batch_fresh_duplicate_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_state, target_report, ledger_path, ledger_report = _reconciliation_files(root)
            pending = _trader_pending_order(
                order_id="fresh-duplicate-batch",
                lane_id="lane:EUR_USD:LONG:STOP",
            )
            client = FreshPendingReconciliationExecutionClient(
                (pending,),
                initial_snapshot_calls=2,
            )

            with patch.object(
                LiveOrderGateway,
                "_pre_post_reconcile",
                self._original_pre_post_reconcile,
            ):
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    target_state_path=target_state,
                    target_report_path=target_report,
                    execution_ledger_db_path=ledger_path,
                    execution_ledger_report_path=ledger_report,
                    live_enabled=True,
                ).run_batch(
                    intents_path=_intents(root),
                    lane_ids=("lane:EUR_USD:LONG",),
                    send=True,
                    confirm_live=True,
                )

            self.assertFalse(summary.sent)
            self.assertEqual(client.orders, [])
            result = json.loads((root / "request.json").read_text())
            order = result["orders"][0]
            codes = {issue["code"] for issue in order["risk_issues"]}
            self.assertIn("PRE_POST_DUPLICATE_PENDING_GEOMETRY", codes)
            self.assertIn("PRE_POST_DUPLICATE_PENDING_PARENT_LANE", codes)

    def test_pre_post_reconciliation_rebuilds_fresh_pending_occupancy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_state, target_report, ledger_path, ledger_report = _reconciliation_files(root)
            pending_orders = tuple(
                _trader_pending_order(
                    order_id=f"fresh-occupancy-{index}",
                    lane_id=f"other:EUR_USD:LONG:TREND_CONTINUATION:{index}",
                    price=1.17000 + index * 0.00010,
                    tp=1.17150 + index * 0.00010,
                    sl=1.16900 + index * 0.00010,
                    units=1,
                )
                for index in range(4)
            )
            client = FreshPendingReconciliationExecutionClient(pending_orders)

            with patch.object(
                LiveOrderGateway,
                "_pre_post_reconcile",
                self._original_pre_post_reconcile,
            ):
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    target_state_path=target_state,
                    target_report_path=target_report,
                    execution_ledger_db_path=ledger_path,
                    execution_ledger_report_path=ledger_report,
                    live_enabled=True,
                ).run(
                    intents_path=_intents(root),
                    lane_id="lane:EUR_USD:LONG",
                    send=True,
                    confirm_live=True,
                )

            self.assertFalse(summary.sent)
            result = json.loads((root / "request.json").read_text())
            self.assertIn(
                "PRE_POST_PORTFOLIO_POSITION_LIMIT",
                {issue["code"] for issue in result["risk_issues"]},
            )
            fresh = result["pre_post_reconciliation"]["fresh_pending_reconciliation"]
            self.assertEqual(fresh["trader_entry_occupancy"], 4)
            self.assertEqual(fresh["portfolio_position_cap"], 4)

    def test_pre_post_reconciliation_tightens_macro_event_to_fresh_nav_cap_direct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_state, target_report, ledger_path, ledger_report = _reconciliation_files(root)
            _write_macro_reconciliation_target_state(target_state)
            client = FreshNavReconciliationExecutionClient(
                initial_nav_jpy=200_000.0,
                fresh_nav_jpy=100_000.0,
            )
            metadata = _ordinary_claim_metadata()
            metadata.update(
                {
                    "macro_event_confidence_sizing": True,
                    "macro_event_nav_jpy_at_sizing": 200_000.0,
                    "macro_event_nav_cap_jpy": 2_000.0,
                    "max_loss_jpy": 2_000.0,
                }
            )

            with patch.object(
                LiveOrderGateway,
                "_pre_post_reconcile",
                self._original_pre_post_reconcile,
            ):
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    target_state_path=target_state,
                    target_report_path=target_report,
                    execution_ledger_db_path=ledger_path,
                    execution_ledger_report_path=ledger_report,
                    live_enabled=True,
                    max_loss_jpy=3_000.0,
                ).run(
                    intents_path=_intents(root, units=15_000, metadata=metadata),
                    lane_id="lane:EUR_USD:LONG",
                    send=True,
                    confirm_live=True,
                )

            self.assertTrue(summary.sent)
            result = json.loads((root / "request.json").read_text())
            reconciliation = result["pre_post_reconciliation"]
            macro = reconciliation["macro_event_fresh_nav_recheck"]
            self.assertEqual(macro["status"], "APPLIED")
            self.assertEqual(
                macro["max_risk_pct_nav"],
                execution_module.MACRO_EVENT_MAX_RISK_PCT_NAV,
            )
            self.assertEqual(macro["fresh_nav_cap_jpy"], 1_000.0)
            self.assertEqual(reconciliation["final_max_loss_jpy"], 1_000.0)
            self.assertLess(result["scaled_units"], 15_000)
            self.assertLessEqual(result["risk_metrics"]["risk_jpy"], 1_000.0)

    def test_pre_post_reconciliation_caps_every_fresh_entry_to_one_pct_nav_direct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_state, target_report, ledger_path, ledger_report = _reconciliation_files(root)
            _write_macro_reconciliation_target_state(target_state)
            client = FreshNavReconciliationExecutionClient(
                initial_nav_jpy=200_000.0,
                fresh_nav_jpy=100_000.0,
            )

            with patch.object(
                LiveOrderGateway,
                "_pre_post_reconcile",
                self._original_pre_post_reconcile,
            ), patch.object(
                execution_module,
                "DailyTargetLedger",
                HighRiskReconciliationTargetLedger,
            ):
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    target_state_path=target_state,
                    target_report_path=target_report,
                    execution_ledger_db_path=ledger_path,
                    execution_ledger_report_path=ledger_report,
                    live_enabled=True,
                    max_loss_jpy=3_000.0,
                ).run(
                    intents_path=_intents(
                        root,
                        units=50_000,
                        metadata=_ordinary_claim_metadata(),
                    ),
                    lane_id="lane:EUR_USD:LONG",
                    send=True,
                    confirm_live=True,
                )

            self.assertTrue(summary.sent)
            result = json.loads((root / "request.json").read_text())
            reconciliation = result["pre_post_reconciliation"]
            self.assertEqual(reconciliation["fresh_nav_recheck"]["fresh_nav_cap_jpy"], 1_000.0)
            self.assertEqual(reconciliation["final_max_loss_jpy"], 1_000.0)
            self.assertLessEqual(result["risk_metrics"]["risk_jpy"], 1_000.0)

    def test_pre_post_reconciliation_caps_every_fresh_entry_to_one_pct_nav_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_state, target_report, ledger_path, ledger_report = _reconciliation_files(root)
            _write_macro_reconciliation_target_state(target_state)
            client = FreshNavReconciliationExecutionClient(
                initial_nav_jpy=200_000.0,
                fresh_nav_jpy=100_000.0,
                initial_snapshot_calls=2,
            )

            with patch.object(
                LiveOrderGateway,
                "_pre_post_reconcile",
                self._original_pre_post_reconcile,
            ), patch.object(
                execution_module,
                "DailyTargetLedger",
                HighRiskReconciliationTargetLedger,
            ):
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    target_state_path=target_state,
                    target_report_path=target_report,
                    execution_ledger_db_path=ledger_path,
                    execution_ledger_report_path=ledger_report,
                    live_enabled=True,
                    max_loss_jpy=3_000.0,
                ).run_batch(
                    intents_path=_intents(
                        root,
                        units=50_000,
                        metadata=_ordinary_claim_metadata(),
                    ),
                    lane_ids=("lane:EUR_USD:LONG",),
                    send=True,
                    confirm_live=True,
                )

            self.assertTrue(summary.sent)
            result = json.loads((root / "request.json").read_text())
            order = result["orders"][0]
            reconciliation = order["pre_post_reconciliation"]
            self.assertEqual(reconciliation["fresh_nav_recheck"]["fresh_nav_cap_jpy"], 1_000.0)
            self.assertEqual(reconciliation["final_max_loss_jpy"], 1_000.0)
            self.assertLessEqual(order["risk_metrics"]["risk_jpy"], 1_000.0)

    def test_pre_post_reconciliation_tightens_macro_event_to_fresh_nav_cap_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_state, target_report, ledger_path, ledger_report = _reconciliation_files(root)
            _write_macro_reconciliation_target_state(target_state)
            client = FreshNavReconciliationExecutionClient(
                initial_nav_jpy=200_000.0,
                fresh_nav_jpy=100_000.0,
                initial_snapshot_calls=2,
            )
            metadata = _ordinary_claim_metadata()
            metadata.update(
                {
                    "macro_event_confidence_sizing": True,
                    "macro_event_nav_jpy_at_sizing": 200_000.0,
                    "macro_event_nav_cap_jpy": 2_000.0,
                    "max_loss_jpy": 2_000.0,
                }
            )

            with patch.object(
                LiveOrderGateway,
                "_pre_post_reconcile",
                self._original_pre_post_reconcile,
            ):
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    target_state_path=target_state,
                    target_report_path=target_report,
                    execution_ledger_db_path=ledger_path,
                    execution_ledger_report_path=ledger_report,
                    live_enabled=True,
                    max_loss_jpy=3_000.0,
                ).run_batch(
                    intents_path=_intents(root, units=15_000, metadata=metadata),
                    lane_ids=("lane:EUR_USD:LONG",),
                    send=True,
                    confirm_live=True,
                )

            self.assertTrue(summary.sent)
            result = json.loads((root / "request.json").read_text())
            order = result["orders"][0]
            macro = order["pre_post_reconciliation"][
                "macro_event_fresh_nav_recheck"
            ]
            self.assertEqual(macro["status"], "APPLIED")
            self.assertEqual(macro["fresh_absolute_cap_jpy"], 1_000.0)
            self.assertLess(order["scaled_units"], 15_000)
            self.assertLessEqual(order["risk_metrics"]["risk_jpy"], 1_000.0)

    def test_pre_post_reconciliation_reproves_exact_tp_relaxation_from_fresh_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_state, target_report, ledger_path, ledger_report = _reconciliation_files(root)
            _write_macro_reconciliation_target_state(target_state)
            _insert_exact_tp_outcomes(ledger_path, losses=0)
            client = FreshNavReconciliationExecutionClient(
                initial_nav_jpy=100_000.0,
                fresh_nav_jpy=100_000.0,
            )
            lane_id = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT"
            intents = _tp_relaxed_limit_intents(root, lane_id=lane_id, units=10_000)

            with patch.object(
                LiveOrderGateway,
                "_pre_post_reconcile",
                self._original_pre_post_reconcile,
            ), patch.object(
                execution_module,
                "DailyTargetLedger",
                FixedReconciliationTargetLedger,
            ):
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    target_state_path=target_state,
                    target_report_path=target_report,
                    execution_ledger_db_path=ledger_path,
                    execution_ledger_report_path=ledger_report,
                    live_enabled=True,
                    max_loss_jpy=3_000.0,
                ).run(
                    intents_path=intents,
                    lane_id=lane_id,
                    send=True,
                    confirm_live=True,
                )

            self.assertTrue(summary.sent)
            result = json.loads((root / "request.json").read_text())
            recheck = result["pre_post_reconciliation"]["tp_proven_fresh_recheck"]
            self.assertEqual(recheck["status"], "PASSED")
            self.assertEqual(
                recheck["proof_key"],
                ["EUR_USD", "LONG", "TREND_CONTINUATION", "LIMIT"],
            )
            self.assertEqual(recheck["tp_trades"], 20)
            self.assertEqual(recheck["tp_losses"], 0)
            self.assertEqual(
                result["sizing_evidence"]["loss_asymmetry_guard_mode"],
                "TP_PROVEN_RELAXED",
            )

    def test_pre_post_reconciliation_clips_revoked_tp_proof_to_fresh_avg_win(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_state, target_report, ledger_path, ledger_report = _reconciliation_files(root)
            _write_macro_reconciliation_target_state(target_state)
            _insert_exact_tp_outcomes(ledger_path, losses=1)
            client = FreshNavReconciliationExecutionClient(
                initial_nav_jpy=100_000.0,
                fresh_nav_jpy=100_000.0,
            )
            lane_id = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT"
            intents = _tp_relaxed_limit_intents(root, lane_id=lane_id, units=50_000)

            with patch.object(
                LiveOrderGateway,
                "_pre_post_reconcile",
                self._original_pre_post_reconcile,
            ), patch.object(
                execution_module,
                "DailyTargetLedger",
                FixedReconciliationTargetLedger,
            ):
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    target_state_path=target_state,
                    target_report_path=target_report,
                    execution_ledger_db_path=ledger_path,
                    execution_ledger_report_path=ledger_report,
                    live_enabled=True,
                    max_loss_jpy=3_000.0,
                ).run(
                    intents_path=intents,
                    lane_id=lane_id,
                    send=True,
                    confirm_live=True,
                )

            self.assertTrue(summary.sent)
            result = json.loads((root / "request.json").read_text())
            reconciliation = result["pre_post_reconciliation"]
            recheck = reconciliation["tp_proven_fresh_recheck"]
            self.assertEqual(recheck["status"], "CLIPPED_TO_AVG_WIN")
            self.assertIn("TP_LOSS_PRESENT", recheck["failed_checks"])
            self.assertEqual(recheck["fallback_avg_win_cap_jpy"], 100.0)
            self.assertEqual(reconciliation["final_max_loss_jpy"], 100.0)
            self.assertLess(result["scaled_units"], 50_000)
            self.assertLessEqual(result["risk_metrics"]["risk_jpy"], 100.0)
            self.assertEqual(
                result["sizing_evidence"]["loss_asymmetry_guard_mode"],
                "CAP_AVG_WIN",
            )
            self.assertIn(
                "PRE_POST_TP_PROOF_REVOKED_CLIPPED_TO_AVG_WIN",
                {issue["code"] for issue in result["risk_issues"]},
            )

    def test_pre_post_reconciliation_counts_daily_financing_in_proof_and_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_state, target_report, ledger_path, ledger_report = _reconciliation_files(root)
            _write_macro_reconciliation_target_state(target_state)
            _insert_exact_tp_outcomes(
                ledger_path,
                losses=0,
                first_trade_daily_financing_jpy=-200.0,
            )
            client = FreshNavReconciliationExecutionClient(
                initial_nav_jpy=100_000.0,
                fresh_nav_jpy=100_000.0,
            )
            lane_id = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT"
            intents = _tp_relaxed_limit_intents(root, lane_id=lane_id, units=50_000)

            with patch.object(
                LiveOrderGateway,
                "_pre_post_reconcile",
                self._original_pre_post_reconcile,
            ), patch.object(
                execution_module,
                "DailyTargetLedger",
                FixedReconciliationTargetLedger,
            ):
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    target_state_path=target_state,
                    target_report_path=target_report,
                    execution_ledger_db_path=ledger_path,
                    execution_ledger_report_path=ledger_report,
                    live_enabled=True,
                    max_loss_jpy=3_000.0,
                ).run(
                    intents_path=intents,
                    lane_id=lane_id,
                    send=True,
                    confirm_live=True,
                )

            self.assertTrue(summary.sent)
            result = json.loads((root / "request.json").read_text())
            recheck = result["pre_post_reconciliation"]["tp_proven_fresh_recheck"]
            self.assertEqual(recheck["status"], "CLIPPED_TO_AVG_WIN")
            self.assertEqual(recheck["tp_losses"], 1)
            self.assertEqual(recheck["fresh_capture"]["losses"], 1)
            self.assertEqual(recheck["fallback_avg_win_cap_jpy"], 100.0)
            self.assertLessEqual(result["risk_metrics"]["risk_jpy"], 100.0)

    def test_pre_post_reconciliation_blocks_batch_when_tp_proof_cannot_be_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_state, target_report, ledger_path, ledger_report = _reconciliation_files(root)
            _write_macro_reconciliation_target_state(target_state)
            _insert_exact_tp_outcomes(ledger_path, losses=0)
            client = FreshNavReconciliationExecutionClient(
                initial_nav_jpy=100_000.0,
                fresh_nav_jpy=100_000.0,
                initial_snapshot_calls=2,
            )
            lane_id = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT"
            intents = _tp_relaxed_limit_intents(root, lane_id=lane_id, units=10_000)

            with (
                patch.object(
                    LiveOrderGateway,
                    "_pre_post_reconcile",
                    self._original_pre_post_reconcile,
                ),
                patch.object(execution_module, "_exact_vehicle_take_profit_metrics", return_value=None),
            ):
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    target_state_path=target_state,
                    target_report_path=target_report,
                    execution_ledger_db_path=ledger_path,
                    execution_ledger_report_path=ledger_report,
                    live_enabled=True,
                    max_loss_jpy=3_000.0,
                ).run_batch(
                    intents_path=intents,
                    lane_ids=(lane_id,),
                    send=True,
                    confirm_live=True,
                )

            self.assertFalse(summary.sent)
            self.assertEqual(client.orders, [])
            result = json.loads((root / "request.json").read_text())
            order = result["orders"][0]
            self.assertIn(
                "PRE_POST_TP_PROOF_LEDGER_READ_FAILED",
                {issue["code"] for issue in order["risk_issues"]},
            )
            self.assertEqual(
                order["pre_post_reconciliation"]["tp_proven_fresh_recheck"]["status"],
                "BLOCKED",
            )

    def test_pre_post_reconciliation_blocks_direct_send_when_fresh_capacity_is_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_state, target_report, ledger_path, ledger_report = _reconciliation_files(
                root,
                gross_loss_jpy=1_000.0,
            )
            client = ReconciliationExecutionClient()

            with patch.object(
                LiveOrderGateway,
                "_pre_post_reconcile",
                self._original_pre_post_reconcile,
            ):
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    target_state_path=target_state,
                    target_report_path=target_report,
                    execution_ledger_db_path=ledger_path,
                    execution_ledger_report_path=ledger_report,
                    live_enabled=True,
                ).run(
                    intents_path=_intents(root),
                    lane_id="lane:EUR_USD:LONG",
                    send=True,
                    confirm_live=True,
                )

            self.assertEqual(summary.status, "BLOCKED")
            self.assertEqual(client.orders, [])
            payload = json.loads((root / "request.json").read_text())
            self.assertEqual(payload["pre_post_reconciliation"]["ledger_sync_status"], "SYNCED")
            self.assertEqual(payload["pre_post_reconciliation"]["daily_loss_capacity_before_open_jpy"], 0.0)
            self.assertIn(
                "PRE_POST_DAILY_LOSS_CAPACITY_EXHAUSTED",
                {issue["code"] for issue in payload["risk_issues"]},
            )

    def test_pre_post_reconciliation_blocks_transaction_id_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_state, target_report, ledger_path, ledger_report = _reconciliation_files(root)
            client = ReconciliationExecutionClient(broker_transaction_id="101")

            with patch.object(
                LiveOrderGateway,
                "_pre_post_reconcile",
                self._original_pre_post_reconcile,
            ):
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    target_state_path=target_state,
                    target_report_path=target_report,
                    execution_ledger_db_path=ledger_path,
                    execution_ledger_report_path=ledger_report,
                    live_enabled=True,
                ).run(
                    intents_path=_intents(root),
                    lane_id="lane:EUR_USD:LONG",
                    send=True,
                    confirm_live=True,
                )

            self.assertEqual(summary.status, "BLOCKED")
            self.assertEqual(client.orders, [])
            payload = json.loads((root / "request.json").read_text())
            self.assertIn(
                "PRE_POST_TRANSACTION_ID_MISMATCH",
                {issue["code"] for issue in payload["risk_issues"]},
            )

    def test_pre_post_reconciliation_blocks_missing_ledger_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = ReconciliationExecutionClient()
            with patch.object(
                LiveOrderGateway,
                "_pre_post_reconcile",
                self._original_pre_post_reconcile,
            ):
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    live_enabled=True,
                ).run(
                    intents_path=_intents(root),
                    lane_id="lane:EUR_USD:LONG",
                    send=True,
                    confirm_live=True,
                )

            self.assertEqual(summary.status, "BLOCKED")
            payload = json.loads((root / "request.json").read_text())
            self.assertIn(
                "PRE_POST_LEDGER_PATH_MISSING",
                {issue["code"] for issue in payload["risk_issues"]},
            )

    def test_pre_post_reconciliation_blocks_baseline_only_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_state, target_report, ledger_path, ledger_report = _reconciliation_files(root)
            with closing(sqlite3.connect(ledger_path)) as conn, conn:
                conn.execute("DELETE FROM sync_state WHERE key = 'last_oanda_transaction_id'")
            client = ReconciliationExecutionClient()

            with patch.object(
                LiveOrderGateway,
                "_pre_post_reconcile",
                self._original_pre_post_reconcile,
            ):
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    target_state_path=target_state,
                    target_report_path=target_report,
                    execution_ledger_db_path=ledger_path,
                    execution_ledger_report_path=ledger_report,
                    live_enabled=True,
                ).run(
                    intents_path=_intents(root),
                    lane_id="lane:EUR_USD:LONG",
                    send=True,
                    confirm_live=True,
                )

            self.assertEqual(summary.status, "BLOCKED")
            self.assertEqual(client.orders, [])
            payload = json.loads((root / "request.json").read_text())
            self.assertIn(
                "PRE_POST_LEDGER_NOT_SYNCED",
                {issue["code"] for issue in payload["risk_issues"]},
            )

    def test_pre_post_reconciliation_blocks_client_without_sync_capability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_state, target_report, ledger_path, ledger_report = _reconciliation_files(root)
            client = FakeExecutionClient()
            with patch.object(
                LiveOrderGateway,
                "_pre_post_reconcile",
                self._original_pre_post_reconcile,
            ):
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    target_state_path=target_state,
                    target_report_path=target_report,
                    execution_ledger_db_path=ledger_path,
                    execution_ledger_report_path=ledger_report,
                    live_enabled=True,
                ).run(
                    intents_path=_intents(root),
                    lane_id="lane:EUR_USD:LONG",
                    send=True,
                    confirm_live=True,
                )

            self.assertEqual(summary.status, "BLOCKED")
            self.assertEqual(client.orders, [])
            payload = json.loads((root / "request.json").read_text())
            self.assertIn(
                "PRE_POST_LEDGER_SYNC_UNAVAILABLE",
                {issue["code"] for issue in payload["risk_issues"]},
            )

    def test_pre_post_reconciliation_blocks_legacy_target_that_cannot_upgrade(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_state, target_report, ledger_path, ledger_report = _reconciliation_files(root)
            target_state.write_text(json.dumps({"daily_risk_budget_jpy": 1_000.0}))
            client = ReconciliationExecutionClient()
            fake_ledger = SimpleNamespace(run=lambda **kwargs: SimpleNamespace(status="PURSUE_TARGET"))

            with patch.object(
                LiveOrderGateway,
                "_pre_post_reconcile",
                self._original_pre_post_reconcile,
            ), patch.object(execution_module, "DailyTargetLedger", return_value=fake_ledger):
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    target_state_path=target_state,
                    target_report_path=target_report,
                    execution_ledger_db_path=ledger_path,
                    execution_ledger_report_path=ledger_report,
                    live_enabled=True,
                ).run(
                    intents_path=_intents(root),
                    lane_id="lane:EUR_USD:LONG",
                    send=True,
                    confirm_live=True,
                )

            self.assertEqual(summary.status, "BLOCKED")
            self.assertEqual(client.orders, [])
            payload = json.loads((root / "request.json").read_text())
            self.assertIn(
                "PRE_POST_TARGET_STATE_LEGACY",
                {issue["code"] for issue in payload["risk_issues"]},
            )

    def test_batch_resyncs_each_candidate_and_blocks_second_after_first_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_state, target_report, ledger_path, ledger_report = _reconciliation_files(root)
            intents = _intents(root, lane_id="lane:EUR_USD:LONG:first")
            payload = json.loads(intents.read_text())
            second = json.loads(json.dumps(payload["results"][0]))
            second["lane_id"] = "lane:EUR_USD:LONG:second"
            second["intent"]["entry"] = 1.1736
            second["intent"]["tp"] = 1.1748
            second["intent"]["sl"] = 1.1728
            payload["results"].append(second)
            intents.write_text(json.dumps(payload))
            client = LossAfterFirstPostExecutionClient()

            with patch.object(
                LiveOrderGateway,
                "_pre_post_reconcile",
                self._original_pre_post_reconcile,
            ):
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    target_state_path=target_state,
                    target_report_path=target_report,
                    execution_ledger_db_path=ledger_path,
                    execution_ledger_report_path=ledger_report,
                    live_enabled=True,
                ).run_batch(
                    intents_path=intents,
                    lane_ids=("lane:EUR_USD:LONG:first", "lane:EUR_USD:LONG:second"),
                    send=True,
                    confirm_live=True,
                )

            self.assertEqual(summary.status, "PARTIAL_SENT")
            self.assertEqual(summary.sent_count, 1)
            self.assertEqual(len(client.orders), 1)
            result = json.loads((root / "request.json").read_text())
            self.assertEqual(result["orders"][0]["pre_post_reconciliation"]["status"], "PASSED")
            self.assertEqual(result["orders"][1]["status"], "BLOCKED")
            self.assertIn(
                "PRE_POST_DAILY_LOSS_CAPACITY_EXHAUSTED",
                {issue["code"] for issue in result["orders"][1]["risk_issues"]},
            )

    def test_pre_post_reconciliation_never_resizes_verified_predictive_scout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "QR_PREDICTIVE_SCOUT_LIVE_ENABLED": "1",
                "QR_REQUIRE_POSITION_GUARDIAN_ACTIVE": "0",
            },
        ):
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            _write_predictive_scout_policy(root)
            target_state, target_report, ledger_path, ledger_report = _reconciliation_files(root)
            target_payload = json.loads(target_state.read_text())
            target_payload["daily_risk_budget_jpy"] = 500.0
            target_state.write_text(json.dumps(target_payload))
            intents = _predictive_scout_intents(
                data_root,
                now=datetime.now(timezone.utc),
            )
            verified = _write_predictive_scout_verified_decision(
                root,
                intents_path=intents,
            )
            client = ReconciliationExecutionClient()
            snapshot = client.snapshot_value
            quotes = dict(snapshot.quotes)
            quotes["USD_CAD"] = Quote(
                "USD_CAD",
                bid=1.41600,
                ask=1.41608,
                timestamp_utc=snapshot.fetched_at_utc,
            )
            client.snapshot_value = replace(
                snapshot,
                quotes=quotes,
                home_conversions={"CAD": 108.0},
            )

            with patch.object(
                LiveOrderGateway,
                "_pre_post_reconcile",
                self._original_pre_post_reconcile,
            ):
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root, pair="USD_CAD"),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    target_state_path=target_state,
                    target_report_path=target_report,
                    execution_ledger_db_path=ledger_path,
                    execution_ledger_report_path=ledger_report,
                    predictive_scout_canonical_ledger_db_path=ledger_path,
                    verified_decision_path=verified,
                    live_enabled=True,
                ).run(
                    intents_path=intents,
                    lane_id=PREDICTIVE_SCOUT_LANE_ID,
                    send=True,
                    confirm_live=True,
                )

            self.assertEqual(summary.status, "BLOCKED")
            self.assertEqual(client.orders, [])
            result = json.loads((root / "request.json").read_text())
            self.assertEqual(result["pre_post_reconciliation"]["status"], "BLOCKED")
            self.assertTrue(result["pre_post_reconciliation"]["final_check_failed"])
            self.assertIn(
                "PREDICTIVE_SCOUT_PRE_POST_MUTATION_REQUIRES_REVERIFY",
                {issue["code"] for issue in result["risk_issues"]},
            )

    def test_pre_post_reconciliation_rebuilds_ordinary_order_after_final_resize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_state, target_report, ledger_path, ledger_report = _reconciliation_files(root)
            target_payload = json.loads(target_state.read_text())
            target_payload["daily_risk_budget_jpy"] = 500.0
            target_state.write_text(json.dumps(target_payload))
            client = ReconciliationExecutionClient()

            with patch.object(
                LiveOrderGateway,
                "_pre_post_reconcile",
                self._original_pre_post_reconcile,
            ):
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    target_state_path=target_state,
                    target_report_path=target_report,
                    execution_ledger_db_path=ledger_path,
                    execution_ledger_report_path=ledger_report,
                    live_enabled=True,
                ).run(
                    intents_path=_intents(root),
                    lane_id="lane:EUR_USD:LONG",
                    send=True,
                    confirm_live=True,
                )

            self.assertEqual(summary.status, "SENT")
            self.assertEqual(len(client.orders), 1)
            result = json.loads((root / "request.json").read_text())
            final_units = result["pre_post_reconciliation"]["final_units"]
            self.assertLess(final_units, result["requested_units"])
            self.assertEqual(int(result["order_request"]["units"]), final_units)
            self.assertEqual(int(client.orders[0]["units"]), final_units)
            self.assertEqual(result["pre_post_reconciliation"]["status"], "PASSED")

    def test_pre_post_reconciliation_rechecks_target_path_units_after_direct_resize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"QR_TARGET_PATH_LIVE_ENABLED": "1"},
        ):
            root = Path(tmp)
            target_state, target_report, ledger_path, ledger_report = _reconciliation_files(root)
            target_payload = json.loads(target_state.read_text())
            target_payload["daily_risk_budget_jpy"] = 500.0
            target_state.write_text(json.dumps(target_payload))
            client = ReconciliationExecutionClient()

            with patch.object(
                LiveOrderGateway,
                "_pre_post_reconcile",
                self._original_pre_post_reconcile,
            ):
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    target_state_path=target_state,
                    target_report_path=target_report,
                    execution_ledger_db_path=ledger_path,
                    execution_ledger_report_path=ledger_report,
                    live_enabled=True,
                ).run(
                    intents_path=_intents(root, metadata=_target_path_metadata(grade="A")),
                    lane_id="lane:EUR_USD:LONG",
                    send=True,
                    confirm_live=True,
                )

            self.assertEqual(summary.status, "BLOCKED")
            self.assertEqual(client.orders, [])
            result = json.loads((root / "request.json").read_text())
            reconciliation = result["pre_post_reconciliation"]
            self.assertLess(reconciliation["final_units"], 500)
            self.assertEqual(reconciliation["target_path_final_recheck"]["status"], "BLOCKED")
            self.assertIn(
                "TARGET_PATH_UNITS_UNDER_SIZED",
                {issue["code"] for issue in result["risk_issues"]},
            )

    def test_pre_post_reconciliation_rechecks_target_path_units_after_batch_resize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"QR_TARGET_PATH_LIVE_ENABLED": "1"},
        ):
            root = Path(tmp)
            target_state, target_report, ledger_path, ledger_report = _reconciliation_files(root)
            target_payload = json.loads(target_state.read_text())
            target_payload["daily_risk_budget_jpy"] = 500.0
            target_state.write_text(json.dumps(target_payload))
            client = ReconciliationExecutionClient()

            with patch.object(
                LiveOrderGateway,
                "_pre_post_reconcile",
                self._original_pre_post_reconcile,
            ):
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    target_state_path=target_state,
                    target_report_path=target_report,
                    execution_ledger_db_path=ledger_path,
                    execution_ledger_report_path=ledger_report,
                    live_enabled=True,
                ).run_batch(
                    intents_path=_intents(root, metadata=_target_path_metadata(grade="A")),
                    lane_ids=("lane:EUR_USD:LONG",),
                    send=True,
                    confirm_live=True,
                )

            self.assertEqual(summary.status, "BLOCKED")
            self.assertEqual(client.orders, [])
            result = json.loads((root / "request.json").read_text())
            order = result["orders"][0]
            reconciliation = order["pre_post_reconciliation"]
            self.assertLess(reconciliation["final_units"], 500)
            self.assertEqual(reconciliation["target_path_final_recheck"]["status"], "BLOCKED")
            self.assertIn(
                "TARGET_PATH_UNITS_UNDER_SIZED",
                {issue["code"] for issue in order["risk_issues"]},
            )

    def test_pre_post_reconciliation_blocks_direct_stale_b_plus_after_five_pct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"QR_TARGET_PATH_LIVE_ENABLED": "1"},
        ):
            root = Path(tmp)
            target_state, target_report, ledger_path, ledger_report = _reconciliation_files(root)
            client = ReconciliationExecutionClient()
            _insert_reconciliation_profit_close(
                ledger_path,
                ts_utc=client.snapshot_value.fetched_at_utc.isoformat(),
                realized_profit_jpy=11_000.0,
            )
            assert client.snapshot_value.account is not None
            client.snapshot_value = replace(
                client.snapshot_value,
                account=replace(
                    client.snapshot_value.account,
                    nav_jpy=211_000.0,
                    balance_jpy=211_000.0,
                    margin_available_jpy=211_000.0,
                ),
            )

            with patch.object(
                LiveOrderGateway,
                "_pre_post_reconcile",
                self._original_pre_post_reconcile,
            ):
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    target_state_path=target_state,
                    target_report_path=target_report,
                    execution_ledger_db_path=ledger_path,
                    execution_ledger_report_path=ledger_report,
                    live_enabled=True,
                ).run(
                    intents_path=_intents(
                        root,
                        metadata=_target_path_metadata(
                            grade="B+",
                            role="SUPPORT",
                            slot="RELOAD",
                        ),
                    ),
                    lane_id="lane:EUR_USD:LONG",
                    send=True,
                    confirm_live=True,
                )

            self.assertEqual(summary.status, "BLOCKED")
            self.assertEqual(client.orders, [])
            result = json.loads((root / "request.json").read_text())
            reconciliation = result["pre_post_reconciliation"]
            self.assertEqual(reconciliation["target_status"], "PURSUE_TARGET")
            self.assertEqual(
                reconciliation["target_path_final_recheck"]["remaining_minimum_jpy"],
                0.0,
            )
            self.assertGreaterEqual(
                reconciliation["target_path_final_recheck"]["minimum_progress_pct"],
                100.0,
            )
            self.assertIn(
                "BASE_TARGET_REACHED_B_RISK_BLOCKED",
                {issue["code"] for issue in result["risk_issues"]},
            )

    def test_pre_post_reconciliation_blocks_batch_stale_b_plus_after_five_pct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"QR_TARGET_PATH_LIVE_ENABLED": "1"},
        ):
            root = Path(tmp)
            target_state, target_report, ledger_path, ledger_report = _reconciliation_files(root)
            client = ReconciliationExecutionClient()
            _insert_reconciliation_profit_close(
                ledger_path,
                ts_utc=client.snapshot_value.fetched_at_utc.isoformat(),
                realized_profit_jpy=11_000.0,
            )
            assert client.snapshot_value.account is not None
            client.snapshot_value = replace(
                client.snapshot_value,
                account=replace(
                    client.snapshot_value.account,
                    nav_jpy=211_000.0,
                    balance_jpy=211_000.0,
                    margin_available_jpy=211_000.0,
                ),
            )

            with patch.object(
                LiveOrderGateway,
                "_pre_post_reconcile",
                self._original_pre_post_reconcile,
            ):
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    target_state_path=target_state,
                    target_report_path=target_report,
                    execution_ledger_db_path=ledger_path,
                    execution_ledger_report_path=ledger_report,
                    live_enabled=True,
                ).run_batch(
                    intents_path=_intents(
                        root,
                        metadata=_target_path_metadata(
                            grade="B+",
                            role="SUPPORT",
                            slot="RELOAD",
                        ),
                    ),
                    lane_ids=("lane:EUR_USD:LONG",),
                    send=True,
                    confirm_live=True,
                )

            self.assertEqual(summary.status, "BLOCKED")
            self.assertEqual(client.orders, [])
            result = json.loads((root / "request.json").read_text())
            order = result["orders"][0]
            reconciliation = order["pre_post_reconciliation"]
            self.assertEqual(reconciliation["target_status"], "PURSUE_TARGET")
            self.assertEqual(
                reconciliation["target_path_final_recheck"]["remaining_minimum_jpy"],
                0.0,
            )
            self.assertGreaterEqual(
                reconciliation["target_path_final_recheck"]["minimum_progress_pct"],
                100.0,
            )
            self.assertIn(
                "BASE_TARGET_REACHED_B_RISK_BLOCKED",
                {issue["code"] for issue in order["risk_issues"]},
            )

    def test_pre_post_reconciliation_allows_a_hero_after_five_pct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"QR_TARGET_PATH_LIVE_ENABLED": "1"},
        ):
            root = Path(tmp)
            target_state, target_report, ledger_path, ledger_report = _reconciliation_files(root)
            client = ReconciliationExecutionClient()
            _insert_reconciliation_profit_close(
                ledger_path,
                ts_utc=client.snapshot_value.fetched_at_utc.isoformat(),
                realized_profit_jpy=11_000.0,
            )
            assert client.snapshot_value.account is not None
            client.snapshot_value = replace(
                client.snapshot_value,
                account=replace(
                    client.snapshot_value.account,
                    nav_jpy=211_000.0,
                    balance_jpy=211_000.0,
                    margin_available_jpy=211_000.0,
                ),
            )

            with patch.object(
                LiveOrderGateway,
                "_pre_post_reconcile",
                self._original_pre_post_reconcile,
            ):
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    target_state_path=target_state,
                    target_report_path=target_report,
                    execution_ledger_db_path=ledger_path,
                    execution_ledger_report_path=ledger_report,
                    live_enabled=True,
                ).run(
                    intents_path=_intents(root, metadata=_target_path_metadata(grade="A")),
                    lane_id="lane:EUR_USD:LONG",
                    send=True,
                    confirm_live=True,
                )

            self.assertEqual(summary.status, "SENT")
            self.assertEqual(len(client.orders), 1)
            result = json.loads((root / "request.json").read_text())
            reconciliation = result["pre_post_reconciliation"]
            self.assertEqual(reconciliation["target_status"], "PURSUE_TARGET")
            self.assertEqual(reconciliation["target_path_final_recheck"]["status"], "PASSED")
            self.assertEqual(result["target_path_receipt"]["remaining_to_5pct"], 0.0)
            self.assertNotIn(
                "BASE_TARGET_REACHED_B_RISK_BLOCKED",
                {issue["code"] for issue in result["risk_issues"]},
            )

    def test_pre_post_reconciliation_metadata_cap_cannot_widen_fresh_direct_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_state, target_report, ledger_path, ledger_report = _reconciliation_files(root)
            client = ReconciliationExecutionClient()
            intents = _intents(
                root,
                metadata={
                    "desk": "trend_trader",
                    "campaign_role": "NOW",
                    "max_loss_jpy": 1_000.0,
                },
            )

            with patch.object(
                LiveOrderGateway,
                "_pre_post_reconcile",
                self._original_pre_post_reconcile,
            ):
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    target_state_path=target_state,
                    target_report_path=target_report,
                    execution_ledger_db_path=ledger_path,
                    execution_ledger_report_path=ledger_report,
                    live_enabled=True,
                ).run(
                    intents_path=intents,
                    lane_id="lane:EUR_USD:LONG",
                    send=True,
                    confirm_live=True,
                )

            self.assertEqual(summary.status, "SENT")
            result = json.loads((root / "request.json").read_text())
            final = result["pre_post_reconciliation"]
            self.assertEqual(final["final_max_loss_jpy"], 100.0)
            self.assertLess(final["final_units"], result["requested_units"])
            self.assertLessEqual(result["risk_metrics"]["risk_jpy"], 100.0)
            self.assertEqual(int(client.orders[0]["units"]), final["final_units"])

    def test_pre_post_reconciliation_metadata_cap_cannot_widen_fresh_batch_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_state, target_report, ledger_path, ledger_report = _reconciliation_files(root)
            client = ReconciliationExecutionClient()
            intents = _intents(
                root,
                metadata={
                    "desk": "trend_trader",
                    "campaign_role": "NOW",
                    "max_loss_jpy": 1_000.0,
                },
            )

            with patch.object(
                LiveOrderGateway,
                "_pre_post_reconcile",
                self._original_pre_post_reconcile,
            ):
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    target_state_path=target_state,
                    target_report_path=target_report,
                    execution_ledger_db_path=ledger_path,
                    execution_ledger_report_path=ledger_report,
                    live_enabled=True,
                ).run_batch(
                    intents_path=intents,
                    lane_ids=("lane:EUR_USD:LONG",),
                    send=True,
                    confirm_live=True,
                )

            self.assertEqual(summary.status, "SENT")
            result = json.loads((root / "request.json").read_text())
            order = result["orders"][0]
            final = order["pre_post_reconciliation"]
            self.assertEqual(final["final_max_loss_jpy"], 100.0)
            self.assertLess(final["final_units"], order["requested_units"])
            self.assertLessEqual(order["risk_metrics"]["risk_jpy"], 100.0)
            self.assertEqual(int(client.orders[0]["units"]), final["final_units"])

    def test_pre_post_reconciliation_blocks_direct_stale_packet_after_target_reached(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_state, target_report, ledger_path, ledger_report = _reconciliation_files(root)
            client = ReconciliationExecutionClient()
            _insert_reconciliation_profit_close(
                ledger_path,
                ts_utc=client.snapshot_value.fetched_at_utc.isoformat(),
                realized_profit_jpy=25_000.0,
            )
            assert client.snapshot_value.account is not None
            client.snapshot_value = replace(
                client.snapshot_value,
                account=replace(
                    client.snapshot_value.account,
                    nav_jpy=225_000.0,
                    balance_jpy=225_000.0,
                    margin_available_jpy=225_000.0,
                ),
            )

            with patch.object(
                LiveOrderGateway,
                "_pre_post_reconcile",
                self._original_pre_post_reconcile,
            ):
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    target_state_path=target_state,
                    target_report_path=target_report,
                    execution_ledger_db_path=ledger_path,
                    execution_ledger_report_path=ledger_report,
                    live_enabled=True,
                ).run(
                    intents_path=_intents(root),
                    lane_id="lane:EUR_USD:LONG",
                    send=True,
                    confirm_live=True,
                )

            self.assertEqual(summary.status, "BLOCKED")
            self.assertEqual(client.orders, [])
            result = json.loads((root / "request.json").read_text())
            self.assertEqual(
                result["pre_post_reconciliation"]["target_status"],
                "TARGET_REACHED_PROTECT",
            )
            self.assertIn(
                "PRE_POST_TARGET_STATUS_BLOCKS_FRESH_ENTRY",
                {issue["code"] for issue in result["risk_issues"]},
            )

    def test_pre_post_reconciliation_blocks_batch_stale_packet_after_target_reached(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_state, target_report, ledger_path, ledger_report = _reconciliation_files(root)
            client = ReconciliationExecutionClient()
            _insert_reconciliation_profit_close(
                ledger_path,
                ts_utc=client.snapshot_value.fetched_at_utc.isoformat(),
                realized_profit_jpy=25_000.0,
            )
            assert client.snapshot_value.account is not None
            client.snapshot_value = replace(
                client.snapshot_value,
                account=replace(
                    client.snapshot_value.account,
                    nav_jpy=225_000.0,
                    balance_jpy=225_000.0,
                    margin_available_jpy=225_000.0,
                ),
            )

            with patch.object(
                LiveOrderGateway,
                "_pre_post_reconcile",
                self._original_pre_post_reconcile,
            ):
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    target_state_path=target_state,
                    target_report_path=target_report,
                    execution_ledger_db_path=ledger_path,
                    execution_ledger_report_path=ledger_report,
                    live_enabled=True,
                ).run_batch(
                    intents_path=_intents(root),
                    lane_ids=("lane:EUR_USD:LONG",),
                    send=True,
                    confirm_live=True,
                )

            self.assertEqual(summary.status, "BLOCKED")
            self.assertEqual(client.orders, [])
            result = json.loads((root / "request.json").read_text())
            order = result["orders"][0]
            self.assertEqual(
                order["pre_post_reconciliation"]["target_status"],
                "TARGET_REACHED_PROTECT",
            )
            self.assertIn(
                "PRE_POST_TARGET_STATUS_BLOCKS_FRESH_ENTRY",
                {issue["code"] for issue in order["risk_issues"]},
            )

    def test_existing_break_even_trader_position_allows_portfolio_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            client.snapshot_value = BrokerSnapshot(
                fetched_at_utc=client.snapshot_value.fetched_at_utc,
                positions=(
                    BrokerPosition(
                        trade_id="101",
                        pair="EUR_USD",
                        side=Side.LONG,
                        units=3000,
                        entry_price=1.1710,
                        take_profit=1.1750,
                        stop_loss=1.1710,
                        owner=Owner.TRADER,
                    ),
                ),
                orders=(),
                quotes=client.snapshot_value.quotes,
                account=client.snapshot_value.account,
            )

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
            ).run(intents_path=_intents(root), lane_id="lane:EUR_USD:LONG")

            self.assertEqual(summary.status, "STAGED")
            self.assertFalse(summary.sent)
            payload = json.loads((root / "request.json").read_text())
            self.assertNotIn("OPEN_POSITION_EXISTS", {issue["code"] for issue in payload["risk_issues"]})

    def test_batch_send_retries_stale_quote_before_blocking(self) -> None:
        prior_attempts = os.environ.get("QR_GATEWAY_STALE_QUOTE_RETRY_ATTEMPTS")
        prior_sleep = os.environ.get("QR_GATEWAY_STALE_QUOTE_RETRY_SLEEP_SECONDS")
        os.environ["QR_GATEWAY_STALE_QUOTE_RETRY_ATTEMPTS"] = "2"
        os.environ["QR_GATEWAY_STALE_QUOTE_RETRY_SLEEP_SECONDS"] = "0"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                now = datetime.now(timezone.utc)
                stale = now - timedelta(seconds=45)
                client = SequenceExecutionClient(
                    (
                        _gateway_snapshot(fetched_at=now, eur_usd_quote_time=now),
                        _gateway_snapshot(fetched_at=now, eur_usd_quote_time=stale),
                        _gateway_snapshot(fetched_at=now, eur_usd_quote_time=now),
                    )
                )

                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    live_enabled=True,
                ).run_batch(
                    intents_path=_intents(root),
                    lane_ids=("lane:EUR_USD:LONG",),
                    send=True,
                    confirm_live=True,
                )

                self.assertEqual(summary.status, "SENT")
                self.assertTrue(summary.sent)
                self.assertEqual(len(client.orders), 1)
                payload = json.loads((root / "request.json").read_text())
                self.assertEqual(payload["orders"][0]["quote_refresh_attempts"], 1)
                self.assertNotIn("STALE_QUOTE", {issue["code"] for issue in payload["risk_issues"]})
        finally:
            if prior_attempts is None:
                os.environ.pop("QR_GATEWAY_STALE_QUOTE_RETRY_ATTEMPTS", None)
            else:
                os.environ["QR_GATEWAY_STALE_QUOTE_RETRY_ATTEMPTS"] = prior_attempts
            if prior_sleep is None:
                os.environ.pop("QR_GATEWAY_STALE_QUOTE_RETRY_SLEEP_SECONDS", None)
            else:
                os.environ["QR_GATEWAY_STALE_QUOTE_RETRY_SLEEP_SECONDS"] = prior_sleep

    def test_batch_position_cap_scales_with_target_trades_per_day(self) -> None:
        execution_module._target_trades_per_day_from_state = lambda path=None: 30
        prior_pair_cap = os.environ.get("QR_MAX_SAME_PAIR_TRADER_POSITIONS")
        os.environ["QR_MAX_SAME_PAIR_TRADER_POSITIONS"] = "10"
        with tempfile.TemporaryDirectory() as tmp:
            try:
                root = Path(tmp)
                client = FakeExecutionClient()
                now = client.snapshot_value.fetched_at_utc
                client.snapshot_value = BrokerSnapshot(
                    fetched_at_utc=now,
                    positions=tuple(
                        BrokerPosition(
                            trade_id=str(200 + index),
                            pair="EUR_USD",
                            side=Side.LONG,
                            units=1000,
                            entry_price=1.1710 + index * 0.0001,
                            take_profit=1.1750,
                            stop_loss=1.1710 + index * 0.0001,
                            owner=Owner.TRADER,
                        )
                        for index in range(4)
                    ),
                    orders=(),
                    quotes=client.snapshot_value.quotes,
                    account=client.snapshot_value.account,
                )

                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    live_enabled=True,
                ).run_batch(
                    intents_path=_intents(root, order_type="MARKET"),
                    lane_ids=("lane:EUR_USD:LONG",),
                )

                self.assertEqual(summary.status, "STAGED")
                payload = json.loads((root / "request.json").read_text())
                self.assertEqual(payload["portfolio_position_cap"], 10)
                issue_codes = {issue["code"] for issue in payload["risk_issues"]}
                self.assertNotIn("BASKET_PORTFOLIO_POSITION_LIMIT", issue_codes)
                self.assertNotIn("PORTFOLIO_POSITION_LIMIT", issue_codes)
            finally:
                if prior_pair_cap is None:
                    os.environ.pop("QR_MAX_SAME_PAIR_TRADER_POSITIONS", None)
                else:
                    os.environ["QR_MAX_SAME_PAIR_TRADER_POSITIONS"] = prior_pair_cap

    def test_non_live_ready_intent_is_not_staged_even_if_fresh_risk_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
            ).run(
                intents_path=_intents(root, status="DRY_RUN_BLOCKED"),
                lane_id="lane:EUR_USD:LONG",
                send=True,
                confirm_live=True,
            )

            self.assertEqual(summary.status, "BLOCKED")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders, [])
            payload = json.loads((root / "request.json").read_text())
            self.assertIn("INTENT_NOT_LIVE_READY", {issue["code"] for issue in payload["risk_issues"]})

    def test_expired_projection_pending_blocks_stale_live_ready_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            emitted = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat().replace("+00:00", "Z")
            (root / "projection_ledger.jsonl").write_text(
                json.dumps(
                    {
                        "timestamp_emitted_utc": emitted,
                        "pair": "EUR_USD",
                        "signal_name": "directional_forecast",
                        "direction": "UP",
                        "resolution_window_min": 30,
                        "resolution_status": "PENDING",
                        "cycle_id": "expired-cycle",
                    }
                )
                + "\n"
            )

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
            ).run(
                intents_path=_intents(root),
                lane_id="lane:EUR_USD:LONG",
                send=True,
                confirm_live=True,
            )

            self.assertEqual(summary.status, "BLOCKED")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders, [])
            payload = json.loads((root / "request.json").read_text())
            self.assertIn(
                "TELEMETRY_PROJECTION_PENDING_EXPIRED_FOR_LIVE",
                {issue["code"] for issue in payload["risk_issues"]},
            )

    def test_expired_projection_pending_blocks_batch_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            emitted = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat().replace("+00:00", "Z")
            (root / "projection_ledger.jsonl").write_text(
                json.dumps(
                    {
                        "timestamp_emitted_utc": emitted,
                        "pair": "EUR_USD",
                        "signal_name": "directional_forecast",
                        "direction": "UP",
                        "resolution_window_min": 30,
                        "resolution_status": "PENDING",
                        "cycle_id": "expired-cycle",
                    }
                )
                + "\n"
            )

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
            ).run_batch(
                intents_path=_intents(root, order_type="MARKET"),
                lane_ids=("lane:EUR_USD:LONG",),
                send=True,
                confirm_live=True,
            )

            self.assertEqual(summary.status, "BLOCKED")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders, [])
            payload = json.loads((root / "request.json").read_text())
            self.assertEqual(payload["blocked_count"], 1)
            self.assertIn(
                "TELEMETRY_PROJECTION_PENDING_EXPIRED_FOR_LIVE",
                {issue["code"] for issue in payload["risk_issues"]},
            )

    def test_invalid_pending_entry_writes_blocked_receipt_instead_of_raising(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            intents = _intents(root)
            payload = json.loads(intents.read_text())
            payload["results"][0]["intent"]["entry"] = None
            intents.write_text(json.dumps(payload))

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
            ).run(
                intents_path=intents,
                lane_id="lane:EUR_USD:LONG",
                send=True,
                confirm_live=True,
            )

            self.assertEqual(summary.status, "BLOCKED")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders, [])
            result = json.loads((root / "request.json").read_text())
            self.assertIsNone(result["order_request"])
            self.assertIn("ORDER_REQUEST_INVALID", {issue["code"] for issue in result["risk_issues"]})

    def test_sl_free_pending_order_without_broker_sl_has_synthetic_basket_risk(self) -> None:
        prior_sl_free = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        prior_initial_sl = os.environ.get("QR_NEW_ENTRY_INITIAL_SL")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                client = FakeExecutionClient()
                client.snapshot_value = BrokerSnapshot(
                    fetched_at_utc=client.snapshot_value.fetched_at_utc,
                    positions=(),
                    orders=(
                        BrokerOrder(
                            order_id="471248",
                            pair="EUR_USD",
                            order_type="LIMIT",
                            price=1.16512,
                            units=1000,
                            owner=Owner.TRADER,
                            raw={
                                "id": "471248",
                                "instrument": "EUR_USD",
                                "type": "LIMIT_ORDER",
                                "price": "1.16512",
                                "units": "1000",
                                "takeProfitOnFill": {"price": "1.16826"},
                            },
                        ),
                    ),
                    quotes=client.snapshot_value.quotes,
                    account=client.snapshot_value.account,
                )

                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    live_enabled=True,
                ).run_batch(
                    intents_path=_intents(root),
                    lane_ids=("lane:EUR_USD:LONG",),
                )

                self.assertEqual(summary.status, "STAGED")
                payload = json.loads((root / "request.json").read_text())
                issue_codes = {issue["code"] for issue in payload["risk_issues"]}
                self.assertNotIn("PENDING_RISK_UNKNOWN", issue_codes)
                self.assertNotIn("stopLossOnFill", payload["orders"][0]["order_request"])
        finally:
            if prior_sl_free is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl_free
            if prior_initial_sl is None:
                os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
            else:
                os.environ["QR_NEW_ENTRY_INITIAL_SL"] = prior_initial_sl

    def test_batch_blocks_existing_pending_geometry_with_disaster_stop(self) -> None:
        prior_sl_free = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        prior_initial_sl = os.environ.get("QR_NEW_ENTRY_INITIAL_SL")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                client = FakeExecutionClient()
                client.snapshot_value = BrokerSnapshot(
                    fetched_at_utc=client.snapshot_value.fetched_at_utc,
                    positions=(),
                    orders=(
                        BrokerOrder(
                            order_id="471249",
                            pair="EUR_USD",
                            order_type="LIMIT",
                            price=1.17330,
                            units=1000,
                            owner=Owner.TRADER,
                            raw={
                                "id": "471249",
                                "instrument": "EUR_USD",
                                "type": "LIMIT_ORDER",
                                "price": "1.17330",
                                "units": "1000",
                                "takeProfitOnFill": {"price": "1.17450"},
                                "stopLossOnFill": {"price": "1.17000"},
                            },
                        ),
                    ),
                    quotes=client.snapshot_value.quotes,
                    account=client.snapshot_value.account,
                )

                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    live_enabled=True,
                ).run_batch(
                    intents_path=_intents(
                        root,
                        order_type="LIMIT",
                        metadata={
                            "desk": "trend_trader",
                            "campaign_role": "NOW",
                            "disaster_sl": 1.17000,
                        },
                    ),
                    lane_ids=("lane:EUR_USD:LONG",),
                    send=True,
                    confirm_live=True,
                )

                self.assertEqual(summary.status, "BLOCKED")
                self.assertEqual(summary.sent_count, 0)
                self.assertEqual(client.orders, [])
                payload = json.loads((root / "request.json").read_text())
                self.assertIn("BASKET_DUPLICATE_GEOMETRY", {issue["code"] for issue in payload["risk_issues"]})
        finally:
            if prior_sl_free is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl_free
            if prior_initial_sl is None:
                os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
            else:
                os.environ["QR_NEW_ENTRY_INITIAL_SL"] = prior_initial_sl

    def test_batch_blocks_existing_pending_parent_lane_with_drifted_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            client.snapshot_value = BrokerSnapshot(
                fetched_at_utc=client.snapshot_value.fetched_at_utc,
                positions=(),
                orders=(
                    BrokerOrder(
                        order_id="472480",
                        pair="EUR_USD",
                        order_type="LIMIT",
                        price=1.17336,
                        units=1000,
                        owner=Owner.TRADER,
                        raw={
                            "id": "472480",
                            "instrument": "EUR_USD",
                            "type": "LIMIT_ORDER",
                            "price": "1.17336",
                            "units": "1000",
                            "clientExtensions": {
                                "comment": "qr-vnext lane=lane:EUR_USD:LONG desk=trend_trader role=NOW",
                                "id": "qrv1-EURUSD-L-existing",
                                "tag": "trader",
                            },
                            "tradeClientExtensions": {
                                "comment": "qr-vnext lane=lane:EUR_USD:LONG desk=trend_trader role=NOW",
                                "id": "qrv1-EURUSD-L-existing-trade",
                                "tag": "trader",
                            },
                            "takeProfitOnFill": {"price": "1.17456"},
                            "stopLossOnFill": {"price": "1.17244"},
                        },
                    ),
                ),
                quotes=client.snapshot_value.quotes,
                account=client.snapshot_value.account,
            )

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
            ).run_batch(
                intents_path=_intents(root, order_type="LIMIT"),
                lane_ids=("lane:EUR_USD:LONG",),
                send=True,
                confirm_live=True,
            )

            self.assertEqual(summary.status, "BLOCKED")
            self.assertEqual(summary.sent_count, 0)
            self.assertEqual(client.orders, [])
            payload = json.loads((root / "request.json").read_text())
            self.assertIn("BASKET_DUPLICATE_PARENT_LANE", {issue["code"] for issue in payload["risk_issues"]})


class FixedReconciliationTargetLedger:
    """Keep target capacity open while gateway-proof tests isolate TP evidence."""

    def __init__(
        self,
        *,
        state_path: Path,
        report_path: Path,
        execution_ledger_path: Path,
    ) -> None:
        self.state_path = state_path

    def run(self, *, snapshot: BrokerSnapshot, now_utc: datetime):
        self.state_path.write_text(
            json.dumps(
                {
                    "realized_loss_spent_jpy": 0.0,
                    "daily_loss_capacity_before_open_jpy": 3_000.0,
                    "base_per_trade_risk_budget_jpy": 1_000.0,
                    "per_trade_risk_budget_jpy": 1_000.0,
                    "remaining_minimum_jpy": 5_000.0,
                    "minimum_progress_pct": 0.0,
                }
            )
        )
        return SimpleNamespace(status="PURSUE_TARGET")


class HighRiskReconciliationTargetLedger:
    """Expose a 3% target cap so the independent fresh-NAV cap is tested."""

    def __init__(
        self,
        *,
        state_path: Path,
        report_path: Path,
        execution_ledger_path: Path,
    ) -> None:
        self.state_path = state_path

    def run(self, *, snapshot: BrokerSnapshot, now_utc: datetime):
        self.state_path.write_text(
            json.dumps(
                {
                    "realized_loss_spent_jpy": 0.0,
                    "daily_loss_capacity_before_open_jpy": 3_000.0,
                    "base_per_trade_risk_budget_jpy": 3_000.0,
                    "per_trade_risk_budget_jpy": 3_000.0,
                    "remaining_minimum_jpy": 5_000.0,
                    "minimum_progress_pct": 0.0,
                }
            )
        )
        return SimpleNamespace(status="PURSUE_TARGET")


class FakeExecutionClient:
    def __init__(self) -> None:
        now = datetime.now(timezone.utc)
        self.snapshot_value = BrokerSnapshot(
            fetched_at_utc=now,
            positions=(),
            orders=(),
            quotes={
                "EUR_USD": Quote("EUR_USD", bid=1.17298, ask=1.17306, timestamp_utc=now),
                "USD_JPY": Quote("USD_JPY", bid=157.0, ask=157.01, timestamp_utc=now),
            },
            account=AccountSummary(
                nav_jpy=200_000.0,
                balance_jpy=200_000.0,
                margin_used_jpy=0.0,
                margin_available_jpy=200_000.0,
                last_transaction_id="100",
                fetched_at_utc=now,
            ),
            home_conversions={"USD": 157.0},
        )
        self.orders: list[dict[str, Any]] = []

    def snapshot(self, pairs: tuple[str, ...]) -> BrokerSnapshot:
        return self.snapshot_value

    def get_json(
        self,
        path: str,
        query: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if not path.endswith("/candles") or not isinstance(query, dict):
            raise AssertionError(f"unexpected read-only broker query: {path}")
        start = datetime.fromisoformat(query["from"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(query["to"].replace("Z", "+00:00"))
        candles = []
        cursor = start
        while cursor < end:
            candles.append(
                {
                    "time": cursor.isoformat().replace("+00:00", "Z"),
                    "complete": cursor + timedelta(seconds=5) < end,
                    "mid": {
                        "o": "1.17302",
                        "h": "1.17304",
                        "l": "1.17300",
                        "c": "1.17302",
                    },
                }
            )
            cursor += timedelta(seconds=5)
        instrument = path.split("/instruments/", 1)[1].split("/", 1)[0]
        return {
            "instrument": instrument,
            "granularity": query.get("granularity"),
            "candles": candles,
        }

    def post_order_json(self, order_request: dict[str, Any]) -> dict[str, Any]:
        self.orders.append(order_request)
        return {"orderCreateTransaction": {"id": "1"}, "relatedTransactionIDs": ["1"]}


class ReconciliationExecutionClient(FakeExecutionClient):
    def __init__(self, *, broker_transaction_id: str = "100") -> None:
        super().__init__()
        account = self.snapshot_value.account
        assert account is not None
        self.snapshot_value = BrokerSnapshot(
            fetched_at_utc=self.snapshot_value.fetched_at_utc,
            positions=self.snapshot_value.positions,
            orders=self.snapshot_value.orders,
            quotes=self.snapshot_value.quotes,
            account=AccountSummary(
                nav_jpy=account.nav_jpy,
                balance_jpy=account.balance_jpy,
                margin_used_jpy=account.margin_used_jpy,
                margin_available_jpy=account.margin_available_jpy,
                last_transaction_id=broker_transaction_id,
                fetched_at_utc=account.fetched_at_utc,
            ),
            home_conversions=self.snapshot_value.home_conversions,
        )
        self.transaction_id = "100"

    def account_summary(self, *, now_utc=None):
        assert self.snapshot_value.account is not None
        return self.snapshot_value.account

    def transactions_since_id(self, transaction_id: str) -> dict[str, Any]:
        return {"transactions": [], "lastTransactionID": self.transaction_id}


class FinalS5ScenarioExecutionClient(ReconciliationExecutionClient):
    """Keep the early path clean, then corrupt only the reserved final path."""

    def __init__(self, scenario: str) -> None:
        super().__init__()
        self.scenario = scenario
        self.s5_calls = 0

    def get_json(
        self,
        path: str,
        query: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        self.s5_calls += 1
        if self.s5_calls < 2:
            return super().get_json(path, query)
        if self.scenario == "fetch_failure":
            raise RuntimeError("simulated final S5 outage")
        if self.scenario == "timeout":
            raise TimeoutError("simulated final S5 timeout")
        payload = super().get_json(path, query)
        candles = payload["candles"]
        if self.scenario == "target_touch":
            candles[-1]["mid"]["h"] = "1.17460"
            candles[-1]["mid"]["c"] = "1.17302"
        elif self.scenario == "invalidation_touch":
            candles[-1]["mid"]["l"] = "1.17250"
            candles[-1]["mid"]["c"] = "1.17302"
        elif self.scenario == "both_touch":
            candles[-1]["mid"]["h"] = "1.17460"
            candles[-1]["mid"]["l"] = "1.17250"
        elif self.scenario == "gap":
            payload["candles"] = []
        elif self.scenario == "malformed":
            candles[-1]["complete"] = "false"
        elif self.scenario == "wrong_instrument":
            payload["instrument"] = "GBP_USD"
        else:
            raise AssertionError(f"unsupported final S5 scenario: {self.scenario}")
        return payload


class FinalRangeEntryTouchExecutionClient(ReconciliationExecutionClient):
    """Keep the first RANGE proof clean, then touch its frozen LIMIT entry."""

    def __init__(self) -> None:
        super().__init__()
        self.s5_calls = 0

    def get_json(
        self,
        path: str,
        query: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        self.s5_calls += 1
        payload = super().get_json(path, query)
        if self.s5_calls >= 2:
            payload["candles"][-1]["mid"]["l"] = "1.17100"
        return payload


class PostS5RangeQuoteCrossExecutionClient(ReconciliationExecutionClient):
    """Cross the frozen RANGE LIMIT only after the final S5 response."""

    def __init__(self) -> None:
        super().__init__()
        self.s5_calls = 0
        self.snapshot_reads_after_final_s5 = 0

    def get_json(
        self,
        path: str,
        query: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        self.s5_calls += 1
        return super().get_json(path, query)

    def snapshot(self, pairs: tuple[str, ...]) -> BrokerSnapshot:
        snapshot = super().snapshot(pairs)
        if self.s5_calls < 2:
            return snapshot
        self.snapshot_reads_after_final_s5 += 1
        quote = snapshot.quotes["EUR_USD"]
        return replace(
            snapshot,
            quotes={
                **snapshot.quotes,
                "EUR_USD": replace(
                    quote,
                    bid=1.17098,
                    ask=1.17100,
                ),
            },
        )


class PostS5AccountDeteriorationClient(ReconciliationExecutionClient):
    def __init__(self, scenario: str) -> None:
        super().__init__()
        self.scenario = scenario
        self.s5_calls = 0
        self.snapshot_reads_after_final_s5 = 0

    def get_json(
        self,
        path: str,
        query: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        self.s5_calls += 1
        return super().get_json(path, query)

    def snapshot(self, pairs: tuple[str, ...]) -> BrokerSnapshot:
        snapshot = super().snapshot(pairs)
        if self.s5_calls < 2:
            return snapshot
        self.snapshot_reads_after_final_s5 += 1
        account = snapshot.account
        assert account is not None
        if self.scenario == "nav":
            changed_account = replace(account, nav_jpy=account.nav_jpy - 1.0)
        elif self.scenario == "margin_used":
            changed_account = replace(
                account,
                margin_used_jpy=account.margin_used_jpy + 1.0,
            )
        elif self.scenario == "margin_available":
            changed_account = replace(
                account,
                margin_available_jpy=account.margin_available_jpy - 1.0,
            )
        else:
            raise AssertionError(f"unsupported account scenario: {self.scenario}")
        return replace(snapshot, account=changed_account)


class PostReservationTransactionAdvanceClient(ReconciliationExecutionClient):
    """Advance broker truth only after the gateway's durable claim commits."""

    def __init__(self) -> None:
        super().__init__()
        self.advance_after_reservation = False

    def snapshot(self, pairs: tuple[str, ...]) -> BrokerSnapshot:
        snapshot = super().snapshot(pairs)
        if not self.advance_after_reservation or snapshot.account is None:
            return snapshot
        return replace(
            snapshot,
            fetched_at_utc=datetime.now(timezone.utc),
            account=replace(
                snapshot.account,
                last_transaction_id="101",
                fetched_at_utc=datetime.now(timezone.utc),
            ),
        )

    def account_summary(self, *, now_utc=None):
        account = super().account_summary(now_utc=now_utc)
        return replace(
            account,
            last_transaction_id=(
                "101" if self.advance_after_reservation else "100"
            ),
            fetched_at_utc=now_utc or account.fetched_at_utc,
        )

    def transactions_since_id(self, transaction_id: str) -> dict[str, Any]:
        if not self.advance_after_reservation:
            return {"transactions": [], "lastTransactionID": "100"}
        now = datetime.now(timezone.utc).isoformat()
        return {
            "transactions": [
                {
                    "id": "101",
                    "type": "ORDER_CANCEL",
                    "time": now,
                    "orderID": "unrelated-late-cancel",
                    "reason": "CLIENT_REQUEST",
                }
            ],
            "lastTransactionID": "101",
        }


class PostReservationLedgerFailureClient(ReconciliationExecutionClient):
    def __init__(self) -> None:
        super().__init__()
        self.fail_after_reservation = False

    def transactions_since_id(self, transaction_id: str) -> dict[str, Any]:
        if self.fail_after_reservation:
            raise TimeoutError("final ledger sync timed out")
        return super().transactions_since_id(transaction_id)


class FreshPendingReconciliationExecutionClient(ReconciliationExecutionClient):
    """Expose new pending broker truth only after pre-POST ledger sync begins."""

    def __init__(
        self,
        orders: tuple[BrokerOrder, ...],
        *,
        initial_snapshot_calls: int = 1,
    ) -> None:
        super().__init__()
        self.initial_snapshot = self.snapshot_value
        self.fresh_snapshot = replace(self.snapshot_value, orders=orders)
        self.snapshot_value = self.fresh_snapshot
        self.initial_snapshot_calls = initial_snapshot_calls
        self.snapshot_call_count = 0

    def snapshot(self, pairs: tuple[str, ...]) -> BrokerSnapshot:
        self.snapshot_call_count += 1
        return (
            self.initial_snapshot
            if self.snapshot_call_count <= self.initial_snapshot_calls
            else self.fresh_snapshot
        )


class FreshNavReconciliationExecutionClient(ReconciliationExecutionClient):
    def __init__(
        self,
        *,
        initial_nav_jpy: float,
        fresh_nav_jpy: float,
        initial_snapshot_calls: int = 1,
    ) -> None:
        super().__init__()
        assert self.snapshot_value.account is not None
        base_account = self.snapshot_value.account
        self.initial_snapshot = replace(
            self.snapshot_value,
            account=replace(
                base_account,
                nav_jpy=initial_nav_jpy,
                balance_jpy=fresh_nav_jpy,
                margin_available_jpy=initial_nav_jpy,
            ),
        )
        self.fresh_snapshot = replace(
            self.snapshot_value,
            account=replace(
                base_account,
                nav_jpy=fresh_nav_jpy,
                balance_jpy=fresh_nav_jpy,
                margin_available_jpy=fresh_nav_jpy,
            ),
        )
        self.snapshot_value = self.fresh_snapshot
        self.initial_snapshot_calls = initial_snapshot_calls
        self.snapshot_call_count = 0

    def snapshot(self, pairs: tuple[str, ...]) -> BrokerSnapshot:
        self.snapshot_call_count += 1
        return (
            self.initial_snapshot
            if self.snapshot_call_count <= self.initial_snapshot_calls
            else self.fresh_snapshot
        )


class CrashBeforePostExecutionClient(FakeExecutionClient):
    def post_order_json(self, order_request: dict[str, Any]) -> dict[str, Any]:
        raise SystemExit("simulated process crash after durable claim")


class LossAfterFirstPostExecutionClient(ReconciliationExecutionClient):
    def __init__(self) -> None:
        super().__init__()
        self.loss_transactions: list[dict[str, Any]] = []

    def post_order_json(self, order_request: dict[str, Any]) -> dict[str, Any]:
        response = super().post_order_json(order_request)
        if len(self.orders) == 1:
            now = datetime.now(timezone.utc).isoformat()
            self.loss_transactions = [
                {
                    "id": "101",
                    "type": "ORDER_FILL",
                    "time": now,
                    "orderID": "first-order",
                    "instrument": "EUR_USD",
                    "units": "1000",
                    "price": "1.17330",
                    "reason": "STOP_ORDER",
                    "clientExtensions": {"comment": "lane=lane:EUR_USD:LONG:first"},
                    "tradeOpened": {
                        "tradeID": "first-trade",
                        "units": "1000",
                        "price": "1.17330",
                    },
                },
                {
                    "id": "102",
                    "type": "ORDER_FILL",
                    "time": now,
                    "orderID": "loss-close",
                    "instrument": "EUR_USD",
                    "units": "-1000",
                    "price": "1.16690",
                    "reason": "MARKET_ORDER_TRADE_CLOSE",
                    "tradesClosed": [
                        {
                            "tradeID": "first-trade",
                            "units": "1000",
                            "price": "1.16690",
                            "realizedPL": "-1000.0",
                            "financing": "0.0",
                        }
                    ],
                },
            ]
            self.transaction_id = "102"
            account = self.snapshot_value.account
            assert account is not None
            self.snapshot_value = BrokerSnapshot(
                fetched_at_utc=datetime.now(timezone.utc),
                positions=(),
                orders=(),
                quotes=self.snapshot_value.quotes,
                account=AccountSummary(
                    nav_jpy=account.nav_jpy - 1000.0,
                    balance_jpy=account.balance_jpy - 1000.0,
                    margin_used_jpy=0.0,
                    margin_available_jpy=account.margin_available_jpy - 1000.0,
                    last_transaction_id="102",
                    fetched_at_utc=datetime.now(timezone.utc),
                ),
            )
        return response

    def transactions_since_id(self, transaction_id: str) -> dict[str, Any]:
        transactions = self.loss_transactions if str(transaction_id) == "100" else []
        return {"transactions": transactions, "lastTransactionID": self.transaction_id}


def _restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def _write_guardian_plist(root: Path) -> Path:
    plist = root / "com.quantrabbit.position-guardian.plist"
    plist.write_text("<plist><dict><key>Label</key><string>com.quantrabbit.position-guardian</string></dict></plist>\n")
    return plist


def _write_guardian_heartbeat(root: Path, *, generated_at: datetime | None = None) -> Path:
    path = root / "position_guardian_execution.json"
    path.write_text(
        json.dumps(
            {
                "generated_at_utc": (generated_at or datetime.now(timezone.utc)).isoformat(),
                "status": "NO_ACTION",
                "sent": False,
            }
        )
        + "\n"
    )
    return path


def _install_fake_launchctl(root: Path, *, loaded: bool) -> None:
    bin_dir = root / "bin"
    bin_dir.mkdir(exist_ok=True)
    script = bin_dir / "launchctl"
    script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "cmd=\"${1:-}\"",
                "label=\"${2:-}\"",
                "if [[ \"$cmd\" == \"list\" && \"$label\" == \"com.quantrabbit.position-guardian\" ]]; then",
                "  if [[ \"${QR_FAKE_POSITION_GUARDIAN_LOADED:-0}\" == \"1\" ]]; then",
                "    printf '123\\t0\\tcom.quantrabbit.position-guardian\\n'",
                "    exit 0",
                "  fi",
                "  exit 113",
                "fi",
                "if [[ \"$cmd\" == \"print\" && \"$label\" == gui/*/com.quantrabbit.position-guardian ]]; then",
                "  if [[ \"${QR_FAKE_POSITION_GUARDIAN_LOADED:-0}\" == \"1\" ]]; then",
                "    printf 'com.quantrabbit.position-guardian = { active = 1 }\\n'",
                "    exit 0",
                "  fi",
                "  exit 113",
                "fi",
                "printf 'unsupported fake launchctl command: %s %s\\n' \"$cmd\" \"$label\" >&2",
                "exit 64",
            ]
        )
        + "\n"
    )
    script.chmod(0o755)
    os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    os.environ["QR_FAKE_POSITION_GUARDIAN_LOADED"] = "1" if loaded else "0"


class SequenceExecutionClient(FakeExecutionClient):
    def __init__(self, snapshots: tuple[BrokerSnapshot, ...]) -> None:
        self.snapshots = snapshots
        self.snapshot_calls: list[tuple[str, ...]] = []
        self.orders: list[dict[str, Any]] = []

    def snapshot(self, pairs: tuple[str, ...]) -> BrokerSnapshot:
        self.snapshot_calls.append(tuple(pairs))
        index = min(len(self.snapshot_calls) - 1, len(self.snapshots) - 1)
        return self.snapshots[index]


class MutatingExecutionClient(FakeExecutionClient):
    def __init__(self, *, margin_used_jpy: float) -> None:
        super().__init__()
        account = self.snapshot_value.account
        assert account is not None
        self.snapshot_value = BrokerSnapshot(
            fetched_at_utc=self.snapshot_value.fetched_at_utc,
            positions=self.snapshot_value.positions,
            orders=self.snapshot_value.orders,
            quotes=self.snapshot_value.quotes,
            account=AccountSummary(
                nav_jpy=account.nav_jpy,
                balance_jpy=account.balance_jpy,
                margin_used_jpy=margin_used_jpy,
                margin_available_jpy=max(0.0, account.nav_jpy - margin_used_jpy),
                fetched_at_utc=account.fetched_at_utc,
            ),
        )

    def post_order_json(self, order_request: dict[str, Any]) -> dict[str, Any]:
        response = super().post_order_json(order_request)
        units = int(order_request["units"])
        quote = self.snapshot_value.quotes["EUR_USD"]
        entry = quote.ask if units > 0 else quote.bid
        margin = abs(units) * entry * self.snapshot_value.quotes["USD_JPY"].bid * OANDA_JP_RETAIL_FX_MARGIN_RATE
        account = self.snapshot_value.account
        assert account is not None
        self.snapshot_value = BrokerSnapshot(
            fetched_at_utc=datetime.now(timezone.utc),
            positions=(
                *self.snapshot_value.positions,
                BrokerPosition(
                    trade_id=str(len(self.orders)),
                    pair="EUR_USD",
                    side=Side.LONG if units > 0 else Side.SHORT,
                    units=abs(units),
                    entry_price=entry,
                    take_profit=float(order_request["takeProfitOnFill"]["price"]),
                    stop_loss=float(order_request["stopLossOnFill"]["price"]),
                    owner=Owner.TRADER,
                ),
            ),
            orders=self.snapshot_value.orders,
            quotes=self.snapshot_value.quotes,
            account=AccountSummary(
                nav_jpy=account.nav_jpy,
                balance_jpy=account.balance_jpy,
                margin_used_jpy=account.margin_used_jpy + margin,
                margin_available_jpy=max(0.0, account.margin_available_jpy - margin),
                fetched_at_utc=datetime.now(timezone.utc),
            ),
        )
        return response


def _gateway_snapshot(*, fetched_at: datetime, eur_usd_quote_time: datetime) -> BrokerSnapshot:
    return BrokerSnapshot(
        fetched_at_utc=fetched_at,
        positions=(),
        orders=(),
        quotes={
            "EUR_USD": Quote("EUR_USD", bid=1.17298, ask=1.17306, timestamp_utc=eur_usd_quote_time),
            "USD_JPY": Quote("USD_JPY", bid=157.0, ask=157.01, timestamp_utc=fetched_at),
        },
        account=AccountSummary(
            nav_jpy=200_000.0,
            balance_jpy=200_000.0,
            margin_used_jpy=0.0,
            margin_available_jpy=200_000.0,
            fetched_at_utc=fetched_at,
        ),
        home_conversions={"USD": 157.0},
    )


def _profile(root: Path, *, direction: str = "LONG", pair: str = "EUR_USD") -> Path:
    path = root / "profile.json"
    path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "pair": pair,
                        "direction": direction,
                        "status": "CANDIDATE",
                        "required_fix": "eligible",
                    }
                ]
            }
        )
    )
    return path


def _reconciliation_files(
    root: Path,
    *,
    gross_loss_jpy: float = 0.0,
    edge_vehicle: str = "STOP",
) -> tuple[Path, Path, Path, Path]:
    data_root = root / "data"
    data_root.mkdir(exist_ok=True)
    ledger_path = data_root / "execution_ledger.db"
    ledger_report = root / "execution_ledger_report.md"
    target_state = data_root / "daily_target_state.json"
    target_report = root / "daily_target_report.md"
    ExecutionLedger(db_path=ledger_path, report_path=ledger_report)._init_db()
    now = datetime.now(timezone.utc).isoformat()
    with closing(sqlite3.connect(ledger_path)) as conn, conn:
        conn.execute(
            """
            INSERT INTO sync_state(key, value, updated_at_utc)
            VALUES ('last_oanda_transaction_id', '100', ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at_utc=excluded.updated_at_utc
            """,
            (now,),
        )
        conn.execute(
            """
            INSERT INTO sync_state(key, value, updated_at_utc)
            VALUES ('oanda_transaction_coverage_start_utc', ?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at_utc=excluded.updated_at_utc
            """,
            ((datetime.now(timezone.utc) - timedelta(days=30)).isoformat(), now),
        )
        if gross_loss_jpy > 0.0:
            entry_raw = json.dumps(
                {
                    "type": "ORDER_FILL",
                    "time": now,
                    "instrument": "EUR_USD",
                    "orderID": "entry-order",
                    "units": "1000",
                    "reason": "MARKET_ORDER",
                    "tradeOpened": {
                        "tradeID": "loss-trade",
                        "units": "1000",
                    },
                }
            )
            rows = (
                (
                    "test:entry",
                    now,
                    "ORDER_FILLED",
                    "loss_trader:EUR_USD:LONG:LOSS:MARKET",
                    "entry-order",
                    "loss-trade",
                    0.0,
                    "MARKET_ORDER",
                    entry_raw,
                ),
                (
                    "test:close",
                    now,
                    "TRADE_CLOSED",
                    None,
                    "close-order",
                    "loss-trade",
                    -gross_loss_jpy,
                    "MARKET_ORDER_TRADE_CLOSE",
                    json.dumps(
                        {
                            "type": "ORDER_FILL",
                            "time": now,
                            "instrument": "EUR_USD",
                            "orderID": "close-order",
                            "reason": "MARKET_ORDER_TRADE_CLOSE",
                            "pl": str(-gross_loss_jpy),
                            "financing": "0.0",
                            "commission": "0.0",
                            "guaranteedExecutionFee": "0.0",
                            "tradesClosed": [
                                {
                                    "tradeID": "loss-trade",
                                    "realizedPL": str(-gross_loss_jpy),
                                    "financing": "0.0",
                                }
                            ],
                        }
                    ),
                ),
            )
            conn.executemany(
                """
                INSERT INTO execution_events(
                    event_uid, ts_utc, source, event_type, lane_id, order_id, trade_id,
                    client_order_id, pair, side, units, price, tp, sl,
                    realized_pl_jpy, financing_jpy, exit_reason, oanda_transaction_id,
                    related_transaction_ids_json, raw_json, inserted_at_utc
                )
                VALUES (?, ?, 'test', ?, ?, ?, ?, NULL, 'EUR_USD', 'LONG', 1000,
                        1.1733, NULL, NULL, ?, 0.0, ?, NULL, '[]', ?, ?)
                """,
                [(*row, now) for row in rows],
            )
    _insert_exact_stop_all_exit_outcomes(
        ledger_path,
        vehicle=edge_vehicle,
    )
    target_state.write_text(
        json.dumps(
            {
                "campaign_day_jst": datetime.now(timezone.utc).date().isoformat(),
                "start_balance_jpy": 200_000.0 + gross_loss_jpy,
                "current_equity_jpy": 200_000.0,
                "target_return_pct": 10.0,
                "daily_risk_budget_jpy": 1_000.0,
                "target_trades_per_day": 10,
                "target_trades_per_day_source": "cli",
            }
        )
    )
    return target_state, target_report, ledger_path, ledger_report


def _insert_exact_stop_all_exit_outcomes(
    ledger_path: Path,
    *,
    outcomes: list[float] | None = None,
    prefix: str = "allocation-edge",
    age_days: int = 7,
    vehicle: str = "STOP",
) -> None:
    normalized_vehicle = str(vehicle).strip().upper()
    if normalized_vehicle not in {"MARKET", "STOP"}:
        raise AssertionError(f"unsupported exact edge fixture vehicle: {vehicle}")
    entry_reason = f"{normalized_vehicle}_ORDER"
    lane_id = (
        "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:"
        f"{normalized_vehicle}"
    )
    now = datetime.now(timezone.utc)
    rows: list[tuple[Any, ...]] = []
    for index, realized in enumerate(outcomes or [100.0] * 20):
        trade_id = f"{prefix}-trade-{index}"
        order_id = f"{prefix}-order-{index}"
        entry_ts = (now - timedelta(days=age_days, minutes=60 - index)).isoformat()
        close_ts = (now - timedelta(days=age_days, minutes=30 - index)).isoformat()
        exit_reason = (
            "TAKE_PROFIT_ORDER"
            if realized > 0
            else "MARKET_ORDER_TRADE_CLOSE"
        )
        entry_raw = {
            "type": "ORDER_FILL",
            "time": entry_ts,
            "instrument": "EUR_USD",
            "orderID": order_id,
            "units": "1000",
            "reason": entry_reason,
            "tradeOpened": {"tradeID": trade_id, "units": "1000"},
        }
        close_raw = {
            "type": "ORDER_FILL",
            "time": close_ts,
            "instrument": "EUR_USD",
            "orderID": f"{prefix}-close-{index}",
            "reason": exit_reason,
            "commission": "0.0",
            "guaranteedExecutionFee": "0.0",
            "tradesClosed": [
                {
                    "tradeID": trade_id,
                    "realizedPL": str(realized),
                    "financing": "0.0",
                }
            ],
        }
        rows.extend(
            (
                (
                    f"{prefix}:gateway:{index}", entry_ts,
                    "GATEWAY_ORDER_SENT", lane_id, order_id, trade_id, None,
                    entry_reason, json.dumps({"type": entry_reason}),
                ),
                (
                    f"{prefix}:fill:{index}", entry_ts, "ORDER_FILLED",
                    lane_id, order_id, trade_id, None, entry_reason,
                    json.dumps(entry_raw),
                ),
                (
                    f"{prefix}:close:{index}", close_ts, "TRADE_CLOSED",
                    lane_id, f"{prefix}-close-{index}", trade_id,
                    realized, exit_reason, json.dumps(close_raw),
                ),
            )
        )
    inserted_at = now.isoformat()
    with closing(sqlite3.connect(ledger_path)) as conn, conn:
        conn.executemany(
            """
            INSERT INTO execution_events(
                event_uid, ts_utc, source, event_type, lane_id, order_id, trade_id,
                client_order_id, pair, side, units, price, tp, sl,
                realized_pl_jpy, financing_jpy, exit_reason, oanda_transaction_id,
                related_transaction_ids_json, raw_json, inserted_at_utc
            )
            VALUES (?, ?, 'test', ?, ?, ?, ?, NULL, 'EUR_USD', 'LONG', 1000,
                    1.1733, 1.1745, 1.1725, ?, 0.0, ?, NULL, '[]', ?, ?)
            """,
            [(*row, inserted_at) for row in rows],
        )


def _insert_exact_stop_unresolved_reduction(ledger_path: Path) -> None:
    now = datetime.now(timezone.utc)
    entry_ts = (now - timedelta(minutes=2)).isoformat()
    reduction_ts = (now - timedelta(minutes=1)).isoformat()
    trade_id = "allocation-unresolved-trade"
    order_id = "allocation-unresolved-entry"
    lane_id = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:STOP"
    entry_raw = {
        "type": "ORDER_FILL",
        "time": entry_ts,
        "instrument": "EUR_USD",
        "orderID": order_id,
        "units": "1000",
        "reason": "STOP_ORDER",
        "tradeOpened": {"tradeID": trade_id, "units": "1000"},
    }
    reduction_raw = {
        "type": "ORDER_FILL",
        "time": reduction_ts,
        "instrument": "EUR_USD",
        "orderID": "allocation-unresolved-reduction",
        "reason": "MARKET_ORDER_TRADE_CLOSE",
        "commission": "0.0",
        "guaranteedExecutionFee": "0.0",
        "tradeReduced": {
            "tradeID": trade_id,
            "realizedPL": "50.0",
            "financing": "0.0",
        },
    }
    rows = (
        (
            "allocation-unresolved:gateway",
            entry_ts,
            "GATEWAY_ORDER_SENT",
            lane_id,
            order_id,
            trade_id,
            None,
            "STOP_ORDER",
            json.dumps({"type": "STOP_ORDER"}),
        ),
        (
            "allocation-unresolved:fill",
            entry_ts,
            "ORDER_FILLED",
            lane_id,
            order_id,
            trade_id,
            None,
            "STOP_ORDER",
            json.dumps(entry_raw),
        ),
        (
            "allocation-unresolved:reduction",
            reduction_ts,
            "TRADE_REDUCED",
            lane_id,
            "allocation-unresolved-reduction",
            trade_id,
            50.0,
            "MARKET_ORDER_TRADE_CLOSE",
            json.dumps(reduction_raw),
        ),
    )
    inserted_at = now.isoformat()
    with closing(sqlite3.connect(ledger_path)) as conn, conn:
        conn.executemany(
            """
            INSERT INTO execution_events(
                event_uid, ts_utc, source, event_type, lane_id, order_id, trade_id,
                client_order_id, pair, side, units, price, tp, sl,
                realized_pl_jpy, financing_jpy, exit_reason, oanda_transaction_id,
                related_transaction_ids_json, raw_json, inserted_at_utc
            )
            VALUES (?, ?, 'test', ?, ?, ?, ?, NULL, 'EUR_USD', 'LONG', 1000,
                    1.1733, 1.1745, 1.1725, ?, 0.0, ?, NULL, '[]', ?, ?)
            """,
            [(*row, inserted_at) for row in rows],
        )


def _write_ordinary_verified_decision(
    root: Path,
    *,
    lane_id: str,
    action: str = "TRADE",
    direction: str = "LONG",
    size_multiple: float = 1.0,
    edge_basis: str = "EXACT_VEHICLE_ALL_EXIT_NET",
    execution_cost_floor: dict[str, Any] | None = None,
    suffix: str = "",
) -> Path:
    generated_at = datetime.now(timezone.utc).isoformat()
    path = root / f"gpt_verified_ordinary{('-' + suffix) if suffix else ''}.json"
    decision: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "action": action,
        "selected_lane_id": lane_id,
        "selected_lane_ids": [lane_id],
        "market_read_first": {
            "next_30m_prediction": {
                "pair": "EUR_USD",
                "direction": direction,
            }
        },
    }
    if action == "TRADE":
        allocation = {
            "decision": "ALLOCATE",
            "lane_id": lane_id,
            "size_multiple": size_multiple,
            "selected_units": int(1000 * size_multiple),
            "allocation_board_sha256": "c" * 64,
            "rationale": "The verified lane supports the risk-capped baseline units.",
        }
        allocation_sha = execution_module.hashlib.sha256(
            json.dumps(
                allocation,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        decision["capital_allocation"] = allocation
        decision["decision_provenance"] = {
            "schema_version": 2,
            "author_kind": "CODEX_MARKET_READ",
            "capital_allocation_edge_basis": edge_basis,
            "capital_allocation_sha256": allocation_sha,
            "capital_allocation_board_sha256": "c" * 64,
            "authorized_size_multiple": size_multiple,
            "authorized_units": int(1000 * size_multiple),
            "execution_cost_floor_sha256": (
                (
                    execution_cost_floor
                    or _synthetic_execution_cost_floor()
                )["proof_sha256"]
            ),
        }
    payload = {
        "generated_at_utc": generated_at,
        "status": "ACCEPTED",
        "decision": decision,
        "verification_issues": [],
        "input_packet": {
            "broker_snapshot": {"fetched_at_utc": generated_at},
        },
    }
    if action == "TRADE":
        _attach_market_read_prediction_lineage(payload, prediction_hex="c")
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def _write_lineaged_verified_decision(root: Path, *, lane_id: str) -> Path:
    path = _write_ordinary_verified_decision(
        root,
        lane_id=lane_id,
        suffix="lineaged",
    )
    payload = json.loads(path.read_text())
    allocation = {
        "decision": "ALLOCATE",
        "lane_id": lane_id,
        "size_multiple": 1.0,
        "selected_units": 1000,
        "allocation_board_sha256": "d" * 64,
        "rationale": "The verified lane supports the risk-capped baseline units.",
    }
    allocation_sha = execution_module.hashlib.sha256(
        json.dumps(
            allocation,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    payload["decision"]["capital_allocation"] = allocation
    payload["decision"]["decision_provenance"] = {
        "schema_version": 2,
        "author_kind": "CODEX_MARKET_READ",
        "capital_allocation_edge_basis": "EXACT_VEHICLE_ALL_EXIT_NET",
        "model": "gpt-5.5",
        "reasoning_effort": "high",
        "capital_allocation_sha256": allocation_sha,
        "capital_allocation_board_sha256": "d" * 64,
        "authorized_size_multiple": 1.0,
        "authorized_units": 1000,
        "execution_cost_floor_sha256": (
            _synthetic_execution_cost_floor()["proof_sha256"]
        ),
    }
    payload["input_packet"] = {
        "broker_snapshot": {
            "fetched_at_utc": payload["generated_at_utc"],
        }
    }
    decision_receipt_id = lineage_module._expected_decision_receipt_id(payload)
    assert decision_receipt_id is not None
    payload["market_read_prediction"] = {
        "status": "RECORDED",
        "schema_version": 2,
        "prediction_id": "mr2:" + "b" * 64,
        "decision_receipt_id": decision_receipt_id,
        "read_only": True,
        "live_permission": False,
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def _attach_market_read_prediction_lineage(
    payload: dict[str, Any],
    *,
    prediction_hex: str,
) -> None:
    decision_receipt_id = lineage_module._expected_decision_receipt_id(payload)
    assert decision_receipt_id is not None
    payload["market_read_prediction"] = {
        "status": "RECORDED",
        "schema_version": 2,
        "prediction_id": "mr2:" + prediction_hex * 64,
        "decision_receipt_id": decision_receipt_id,
        "read_only": True,
        "live_permission": False,
    }


def _ordinary_claim_metadata(
    *,
    parent_lane_id: str = "lane:EUR_USD:LONG",
    forecast_cycle_id: str = "forecast-cycle-1",
    vehicle: str = "STOP",
    numeric_forecast: bool = False,
) -> dict[str, Any]:
    metadata = {
        "desk": "trend_trader",
        "campaign_role": "NOW",
        "parent_lane_id": parent_lane_id,
        "forecast_cycle_id": forecast_cycle_id,
        **_ordinary_exact_all_exit_edge_metadata(vehicle=vehicle),
    }
    if numeric_forecast:
        emitted_at = datetime.now(timezone.utc).isoformat()
        metadata.update(
            {
                "forecast_cycle_id": (
                    f"pre-entry-forecast-refresh:{emitted_at}:{emitted_at}"
                ),
                "forecast_direction": "UP",
                "forecast_confidence": 1.0,
                "forecast_raw_confidence": 1.0,
                "forecast_calibration_multiplier": 1.0,
                "forecast_directional_calibration_name": (
                    "directional_forecast_up"
                ),
                "forecast_directional_economic_hit_rate": 1.0,
                "forecast_directional_economic_samples": 100,
                "forecast_directional_hit_rate": 1.0,
                "forecast_directional_samples": 100,
                "forecast_directional_timeout_rate": 0.0,
                "forecast_horizon_min": 120,
                "forecast_current_price": 1.17302,
                "forecast_target_price": 1.17458,
                "forecast_invalidation_price": 1.17258,
            }
        )
    return metadata


def _ordinary_exact_all_exit_edge_metadata(
    *,
    vehicle: str = "STOP",
) -> dict[str, Any]:
    normalized_vehicle = str(vehicle).strip().upper()
    return {
        "capture_exact_vehicle_net_scope": "PAIR_SIDE_METHOD_VEHICLE",
        "capture_exact_vehicle_net_scope_key": (
            "EUR_USD|LONG|TREND_CONTINUATION|"
            f"{normalized_vehicle}|ALL_AUDITED_EXITS"
        ),
        "capture_exact_vehicle_net_vehicle": normalized_vehicle,
        "capture_exact_vehicle_net_metrics_source": (
            "data/execution_ledger.db:exact_vehicle_net"
        ),
        "capture_exact_vehicle_net_exit_scope": "ALL_AUDITED_EXITS",
        "capture_exact_vehicle_net_trades": 20,
        "capture_exact_vehicle_net_wins": 20,
        "capture_exact_vehicle_net_losses": 0,
        "capture_exact_vehicle_net_jpy": 2000.0,
        "capture_exact_vehicle_net_expectancy_jpy": 100.0,
        "capture_exact_vehicle_net_avg_win_jpy": 100.0,
        "capture_exact_vehicle_net_avg_loss_jpy": 0.0,
        "capture_exact_vehicle_net_unresolved_realized_trades": 0,
        "capture_exact_vehicle_net_unresolved_realized_net_jpy": 0.0,
        "capture_exact_vehicle_net_unresolved_trade_ids_sha256": (
            execution_module.hashlib.sha256(b"[]").hexdigest()
        ),
    }


def _write_macro_reconciliation_target_state(target_state: Path) -> None:
    target_state.write_text(
        json.dumps(
            {
                "campaign_day_jst": datetime.now(timezone.utc).date().isoformat(),
                "start_balance_jpy": 100_000.0,
                "current_equity_jpy": 100_000.0,
                "target_return_pct": 10.0,
                "daily_risk_budget_jpy": 3_000.0,
                "target_trades_per_day": 1,
                "target_trades_per_day_source": "cli",
            }
        ),
        encoding="utf-8",
    )


def _tp_proven_range_limit_intents(
    root: Path,
    *,
    client: FakeExecutionClient,
    crossed: bool = False,
    metadata_updates: dict[str, Any] | None = None,
) -> Path:
    quote = client.snapshot_value.quotes["EUR_USD"]
    emitted_at = (quote.timestamp_utc - timedelta(seconds=1)).isoformat()
    trades = 20
    wins = 20
    loss_proxy = 300.0
    avg_win = 100.0
    wilson = execution_module.hit_rate_wilson_lower(
        wins / trades,
        trades,
    )
    if wilson is None:
        raise AssertionError("TP-proven RANGE fixture has no Wilson lower bound")
    pessimistic = wilson * avg_win - (1.0 - wilson) * loss_proxy
    metadata: dict[str, Any] = {
        "desk": "range_trader",
        "campaign_role": "NOW",
        "parent_lane_id": TP_PROVEN_RANGE_LANE_ID.rsplit(":", 1)[0],
        "forecast_cycle_id": (
            f"pre-entry-forecast-refresh:{emitted_at}:{emitted_at}"
        ),
        "forecast_direction": "RANGE",
        "forecast_confidence": 1.0,
        "forecast_raw_confidence": 1.0,
        "forecast_calibration_multiplier": 1.0,
        "forecast_directional_calibration_name": (
            "directional_forecast_range"
        ),
        "forecast_directional_economic_hit_rate": 1.0,
        "forecast_directional_economic_samples": 100,
        "forecast_directional_hit_rate": 1.0,
        "forecast_directional_samples": 100,
        "forecast_directional_timeout_rate": 0.0,
        "forecast_horizon_min": 120,
        "forecast_current_price": (quote.bid + quote.ask) / 2.0,
        "forecast_range_low_price": 1.17000,
        "forecast_range_high_price": 1.17600,
        "forecast_target_price": 1.17600,
        "forecast_invalidation_price": 1.17000,
        "range_support": 1.17000,
        "range_resistance": 1.17600,
        "range_entry_side": "SUPPORT",
        "geometry_model": "RANGE_RAIL_LIMIT",
        "range_tp_is_inside_box": True,
        "range_sl_outside_box": True,
        "attach_take_profit_on_fill": True,
        "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
        "tp_target_intent": "HARVEST",
        "opportunity_mode": "HARVEST",
        "positive_rotation_live_ready": True,
        "positive_rotation_mode": "TP_PROVEN_HARVEST",
        "positive_rotation_tp_trades": trades,
        "positive_rotation_tp_wins": wins,
        "positive_rotation_loss_proxy_jpy": loss_proxy,
        "positive_rotation_tp_win_rate_lower": round(wilson, 6),
        "positive_rotation_pessimistic_expectancy_jpy": round(
            pessimistic,
            4,
        ),
        "loss_asymmetry_guard_active": True,
        "loss_asymmetry_guard_mode": "TP_PROVEN_RELAXED",
        "loss_asymmetry_guard_relaxed": True,
        "loss_asymmetry_guard_loss_cap_jpy": avg_win,
        "loss_asymmetry_guard_base_max_loss_jpy": 400.0,
        "loss_asymmetry_guard_effective_max_loss_jpy": 400.0,
        "capture_economics_status": "NEGATIVE_EXPECTANCY",
        "capture_avg_win_jpy": avg_win,
        "capture_avg_loss_jpy": loss_proxy,
        "capture_take_profit_scope": "PAIR_SIDE_METHOD_VEHICLE",
        "capture_take_profit_scope_key": (
            "EUR_USD|LONG|RANGE_ROTATION|LIMIT|TAKE_PROFIT_ORDER"
        ),
        "capture_take_profit_exact_vehicle_required": True,
        "capture_take_profit_vehicle": "LIMIT",
        "capture_take_profit_metrics_source": (
            "data/execution_ledger.db:exact_vehicle_take_profit"
        ),
        "capture_take_profit_trades": trades,
        "capture_take_profit_wins": wins,
        "capture_take_profit_losses": 0,
        "capture_take_profit_net_jpy": trades * avg_win,
        "capture_take_profit_expectancy_jpy": avg_win,
        "capture_take_profit_avg_win_jpy": avg_win,
        "capture_take_profit_avg_loss_jpy": 0.0,
        "max_loss_jpy": 400.0,
        "tp_atr_pips": 14.0,
    }
    metadata.update(metadata_updates or {})
    intents = _intents(
        root,
        order_type="LIMIT",
        units=1000,
        lane_id=TP_PROVEN_RANGE_LANE_ID,
        entry=1.17310 if crossed else 1.17100,
        tp=1.17500,
        sl=1.16900,
        metadata=metadata,
    )
    payload = json.loads(intents.read_text())
    intent = payload["results"][0]["intent"]
    intent["thesis"] = "fresh TP-proven passive range support rotation"
    intent["market_context"] = {
        "regime": "RANGE",
        "narrative": "support rail rotation toward the inner premium rail",
        "chart_story": "passive support LIMIT remains below the current ask",
        "method": "RANGE_ROTATION",
        "invalidation": "range support acceptance fails below the outer stop",
    }
    intents.write_text(json.dumps(payload) + "\n")
    return intents


def _tp_relaxed_limit_intents(
    root: Path,
    *,
    lane_id: str,
    units: int,
) -> Path:
    metadata = _ordinary_claim_metadata(parent_lane_id=lane_id.rsplit(":", 1)[0])
    metadata.update(
        {
            "attach_take_profit_on_fill": True,
            "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
            "tp_target_intent": "HARVEST",
            "opportunity_mode": "HARVEST",
            "loss_asymmetry_guard_active": True,
            "loss_asymmetry_guard_mode": "TP_PROVEN_RELAXED",
            "loss_asymmetry_guard_relaxed": True,
            "loss_asymmetry_guard_loss_cap_jpy": 100.0,
            "loss_asymmetry_guard_base_max_loss_jpy": 3_000.0,
            "loss_asymmetry_guard_effective_max_loss_jpy": 3_000.0,
            "capture_economics_status": "NEGATIVE_EXPECTANCY",
            "capture_avg_win_jpy": 100.0,
            "capture_avg_loss_jpy": 500.0,
            "capture_take_profit_scope": "PAIR_SIDE_METHOD_VEHICLE",
            "capture_take_profit_scope_key": (
                "EUR_USD|LONG|TREND_CONTINUATION|LIMIT|TAKE_PROFIT_ORDER"
            ),
            "capture_take_profit_exact_vehicle_required": True,
            "capture_take_profit_vehicle": "LIMIT",
            "capture_take_profit_metrics_source": (
                "data/execution_ledger.db:exact_vehicle_take_profit"
            ),
            "capture_take_profit_trades": 20,
            "capture_take_profit_wins": 20,
            "capture_take_profit_losses": 0,
            "capture_take_profit_net_jpy": 2000.0,
            "capture_take_profit_expectancy_jpy": 100.0,
            "capture_take_profit_avg_win_jpy": 100.0,
            "capture_take_profit_avg_loss_jpy": 0.0,
            "capture_exact_vehicle_net_scope": "PAIR_SIDE_METHOD_VEHICLE",
            "capture_exact_vehicle_net_scope_key": (
                "EUR_USD|LONG|TREND_CONTINUATION|LIMIT|ALL_AUDITED_EXITS"
            ),
            "capture_exact_vehicle_net_vehicle": "LIMIT",
            "capture_exact_vehicle_net_metrics_source": (
                "data/execution_ledger.db:exact_vehicle_net"
            ),
            "capture_exact_vehicle_net_exit_scope": "ALL_AUDITED_EXITS",
            "capture_exact_vehicle_net_trades": 20,
            "capture_exact_vehicle_net_wins": 20,
            "capture_exact_vehicle_net_losses": 0,
            "capture_exact_vehicle_net_jpy": 2000.0,
            "capture_exact_vehicle_net_expectancy_jpy": 100.0,
            "capture_exact_vehicle_net_avg_win_jpy": 100.0,
            "capture_exact_vehicle_net_avg_loss_jpy": 0.0,
            "capture_exact_vehicle_net_unresolved_realized_trades": 0,
            "capture_exact_vehicle_net_unresolved_realized_net_jpy": 0.0,
            "capture_exact_vehicle_net_unresolved_trade_ids_sha256": (
                execution_module.hashlib.sha256(b"[]").hexdigest()
            ),
        }
    )
    path = _intents(
        root,
        order_type="LIMIT",
        units=units,
        lane_id=lane_id,
        metadata=metadata,
    )
    payload = json.loads(path.read_text())
    intent = payload["results"][0]["intent"]
    intent["entry"] = 1.17290
    intent["sl"] = 1.17210
    intent["market_context"]["method"] = "TREND_CONTINUATION"
    intent["market_context"]["regime"] = "TREND_CONTINUATION"
    path.write_text(json.dumps(payload))
    return path


def _insert_exact_tp_outcomes(
    ledger_path: Path,
    *,
    losses: int,
    method: str = "TREND_CONTINUATION",
    first_trade_daily_financing_jpy: float | None = None,
) -> None:
    desk = "trend_trader" if method == "TREND_CONTINUATION" else "range_trader"
    lane_id = f"{desk}:EUR_USD:LONG:{method}:LIMIT"
    now = datetime.now(timezone.utc)
    rows: list[tuple[Any, ...]] = []
    for index in range(20):
        trade_id = f"tp-proof-trade-{index}"
        order_id = f"tp-proof-order-{index}"
        ts_utc = (now - timedelta(minutes=20 - index)).isoformat()
        realized = -50.0 if index < losses else 100.0
        rows.extend(
            (
                (
                    f"tp-proof:gateway:{index}",
                    ts_utc,
                    "GATEWAY_ORDER_SENT",
                    lane_id,
                    order_id,
                    trade_id,
                    None,
                    "LIMIT_ORDER",
                    json.dumps({"type": "LIMIT_ORDER"}),
                ),
                (
                    f"tp-proof:fill:{index}",
                    ts_utc,
                    "ORDER_FILLED",
                    lane_id,
                    order_id,
                    trade_id,
                    None,
                    "LIMIT_ORDER",
                    json.dumps(
                        {
                            "type": "ORDER_FILL",
                            "time": ts_utc,
                            "instrument": "EUR_USD",
                            "orderID": order_id,
                            "units": "1000",
                            "reason": "LIMIT_ORDER",
                            "tradeOpened": {
                                "tradeID": trade_id,
                                "units": "1000",
                            },
                        }
                    ),
                ),
                (
                    f"tp-proof:close:{index}",
                    ts_utc,
                    "TRADE_CLOSED",
                    lane_id,
                    f"tp-proof-close-{index}",
                    trade_id,
                    realized,
                    "TAKE_PROFIT_ORDER",
                    json.dumps(
                        {
                            "type": "ORDER_FILL",
                            "time": ts_utc,
                            "reason": "TAKE_PROFIT_ORDER",
                            "commission": "0.0",
                            "guaranteedExecutionFee": "0.0",
                            "tradesClosed": [
                                {
                                    "tradeID": trade_id,
                                    "realizedPL": str(realized),
                                    "financing": "0.0",
                                }
                            ],
                        }
                    ),
                ),
            )
        )
    inserted_at = now.isoformat()
    with closing(sqlite3.connect(ledger_path)) as conn, conn:
        conn.executemany(
            """
            INSERT INTO execution_events(
                event_uid, ts_utc, source, event_type, lane_id, order_id, trade_id,
                client_order_id, pair, side, units, price, tp, sl,
                realized_pl_jpy, financing_jpy, exit_reason, oanda_transaction_id,
                related_transaction_ids_json, raw_json, inserted_at_utc
            )
            VALUES (?, ?, 'test', ?, ?, ?, ?, NULL, 'EUR_USD', 'LONG', 1000,
                    1.1729, 1.1745, 1.1721, ?, 0.0, ?, NULL, '[]', ?, ?)
            """,
            [(*row, inserted_at) for row in rows],
        )
        if first_trade_daily_financing_jpy is not None:
            financing_transaction_id = "9000"
            financing_raw = {
                "id": financing_transaction_id,
                "type": "DAILY_FINANCING",
                "financing": str(first_trade_daily_financing_jpy),
                "positionFinancings": [
                    {
                        "openTradeFinancings": [
                            {
                                "tradeID": "tp-proof-trade-0",
                                "financing": str(first_trade_daily_financing_jpy),
                            }
                        ]
                    }
                ],
            }
            conn.execute(
                """
                INSERT INTO execution_events(
                    event_uid, ts_utc, source, event_type, lane_id, order_id, trade_id,
                    client_order_id, pair, side, units, price, tp, sl,
                    realized_pl_jpy, financing_jpy, exit_reason, oanda_transaction_id,
                    related_transaction_ids_json, raw_json, inserted_at_utc
                )
                VALUES (?, ?, 'test', 'OANDA_TRANSACTION', NULL, NULL, NULL,
                        NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                        NULL, ?, NULL, ?, '[]', ?, ?)
                """,
                (
                    "tp-proof:financing:0",
                    (now - timedelta(minutes=1)).isoformat(),
                    first_trade_daily_financing_jpy,
                    financing_transaction_id,
                    json.dumps(financing_raw),
                    inserted_at,
                ),
            )
            conn.execute(
                """
                INSERT INTO oanda_transactions(
                    transaction_id, type, time_utc, batch_id, request_id,
                    raw_json, inserted_at_utc
                )
                VALUES (?, 'DAILY_FINANCING', ?, NULL, NULL, ?, ?)
                """,
                (
                    financing_transaction_id,
                    (now - timedelta(minutes=1)).isoformat(),
                    json.dumps(financing_raw),
                    inserted_at,
                ),
            )


def _trader_pending_order(
    *,
    order_id: str,
    lane_id: str,
    price: float = 1.17330,
    tp: float = 1.17450,
    sl: float = 1.17250,
    units: int = 1000,
) -> BrokerOrder:
    parent_lane_id = execution_module._parent_lane_id_from_lane_id(lane_id)
    return BrokerOrder(
        order_id=order_id,
        pair="EUR_USD",
        order_type="STOP",
        price=price,
        units=units,
        owner=Owner.TRADER,
        raw={
            "id": order_id,
            "instrument": "EUR_USD",
            "type": "STOP_ORDER",
            "price": str(price),
            "units": str(units),
            "takeProfitOnFill": {"price": str(tp)},
            "stopLossOnFill": {"price": str(sl)},
            "clientExtensions": {
                "comment": f"qr-vnext parent={parent_lane_id} lane={lane_id} desk=trend_trader"
            },
        },
    )


def _insert_reconciliation_profit_close(
    ledger_path: Path,
    *,
    ts_utc: str,
    realized_profit_jpy: float,
) -> None:
    raw_close = json.dumps(
        {
            "type": "ORDER_FILL",
            "time": ts_utc,
            "instrument": "EUR_USD",
            "orderID": "target-win-close-order",
            "reason": "MARKET_ORDER_TRADE_CLOSE",
            "pl": str(realized_profit_jpy),
            "financing": "0.0",
            "commission": "0.0",
            "guaranteedExecutionFee": "0.0",
            "tradesClosed": [
                {
                    "tradeID": "target-win-trade",
                    "realizedPL": str(realized_profit_jpy),
                    "financing": "0.0",
                }
            ],
        }
    )
    raw_entry = json.dumps(
        {
            "type": "ORDER_FILL",
            "time": ts_utc,
            "instrument": "EUR_USD",
            "orderID": "target-win-entry-order",
            "units": "1000",
            "reason": "MARKET_ORDER",
            "tradeOpened": {
                "tradeID": "target-win-trade",
                "units": "1000",
            },
        }
    )
    with closing(sqlite3.connect(ledger_path)) as conn, conn:
        conn.executemany(
            """
            INSERT INTO execution_events(
                event_uid, ts_utc, source, event_type, lane_id, order_id, trade_id,
                client_order_id, pair, side, units, price, tp, sl,
                realized_pl_jpy, financing_jpy, exit_reason, oanda_transaction_id,
                related_transaction_ids_json, raw_json, inserted_at_utc
            )
            VALUES (?, ?, 'test', ?, ?, ?, ?, NULL, 'EUR_USD', 'LONG', 1000,
                    1.1733, NULL, NULL, ?, 0.0, ?, NULL, '[]', ?, ?)
            """,
            (
                (
                    "test:target-win-entry",
                    ts_utc,
                    "ORDER_FILLED",
                    "target_trader:EUR_USD:LONG:TARGET_WIN:MARKET",
                    "target-win-entry-order",
                    "target-win-trade",
                    0.0,
                    "MARKET_ORDER",
                    raw_entry,
                    ts_utc,
                ),
                (
                    "test:target-win-close",
                    ts_utc,
                    "TRADE_CLOSED",
                    None,
                    "target-win-close-order",
                    "target-win-trade",
                    realized_profit_jpy,
                    "MARKET_ORDER_TRADE_CLOSE",
                    raw_close,
                    ts_utc,
                ),
            ),
        )


def _target_path_metadata(*, grade: str, role: str = "HERO", slot: str = "NOW", valid: str = "YES") -> dict[str, Any]:
    return {
        "desk": "trend_trader",
        "campaign_role": slot,
        "daily_target_mode": "ATTACK",
        "remaining_to_5pct_yen": 3000.0,
        "remaining_to_10pct_yen": 8000.0,
        "target_path_role": role,
        "path_board_slot": role,
        "path_board_available": True,
        "five_pct_path_available": True,
        "attack_stack_available": True,
        "attack_stack_slot": slot,
        "maps_to_attack_stack": True,
        "conviction_grade": grade,
        "valid_as_target_path": valid,
        "suggested_units": 1000,
        "risk_yen": 87.92,
        "risk_pct": 0.04,
        "target_yen": 226.08,
        "contribution_to_5pct": 226.08,
        "extension_gate": "NO",
        "exact_pretrade_passed": True,
        "spread_guard_passed": True,
        "pricing_probe_passed": True,
        "fill_guard_passed": True,
        "same_thesis_lost_recently": False,
        "vehicle_unchanged_after_loss": False,
        "target_path_live_mode": "LIVE_LEARNING",
    }


def _predictive_scout_metadata(now: datetime) -> dict[str, Any]:
    rule = canonical_bidask_replay_precision_rule(PREDICTIVE_SCOUT_RULE_NAME)
    if rule is None:
        raise AssertionError("canonical USD_CAD predictive SCOUT rule is missing")
    return {
        "desk": "failure_trader",
        "campaign_role": "BIDASK_REPLAY_CONTRARIAN_SCOUT",
        "attach_take_profit_on_fill": True,
        "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
        "tp_target_intent": "HARVEST",
        "opportunity_mode": "HARVEST",
        "broker_stop_loss_mode": "INTENT_SL",
        "forecast_direction": "DOWN",
        "forecast_confidence": 0.55,
        "forecast_horizon_min": 45,
        "forecast_cycle_id": "test-usdcad-down-c050-065-h31-60",
        "predictive_scout": True,
        "predictive_scout_source": "BIDASK_REPLAY_PRECISION",
        "predictive_scout_rule_name": PREDICTIVE_SCOUT_RULE_NAME,
        "predictive_scout_rule_digest": bidask_replay_precision_rule_digest(rule),
        "predictive_scout_rule_is_vehicle_proof": False,
        "predictive_scout_vehicle_proof_status": "UNPROVEN_PASSIVE_LIMIT",
        "predictive_scout_hypothesis": "REPRODUCIBLE_FORECAST_FAILURE_CONTRARIAN",
        "predictive_scout_generated_at_utc": now.isoformat(),
        "predictive_scout_expires_at_utc": (now + timedelta(minutes=45)).isoformat(),
        "predictive_scout_ttl_minutes": 45,
        "predictive_scout_promotion_allowed": False,
        "disaster_sl": 1.40000,
        "bidask_replay_precision_seed_rule": dict(rule),
    }


def _predictive_scout_client() -> FakeExecutionClient:
    client = FakeExecutionClient()
    snapshot = client.snapshot_value
    quotes = dict(snapshot.quotes)
    quotes["USD_CAD"] = Quote(
        "USD_CAD",
        bid=1.41600,
        ask=1.41608,
        timestamp_utc=snapshot.fetched_at_utc,
    )
    client.snapshot_value = BrokerSnapshot(
        fetched_at_utc=snapshot.fetched_at_utc,
        positions=snapshot.positions,
        orders=snapshot.orders,
        quotes=quotes,
        account=snapshot.account,
        home_conversions={"CAD": 108.0},
    )
    return client


def _predictive_scout_intents(
    data_root: Path,
    *,
    now: datetime,
    crossed: bool = False,
    metadata_updates: dict[str, Any] | None = None,
) -> Path:
    metadata = _predictive_scout_metadata(now)
    metadata.update(metadata_updates or {})
    intents = _intents(
        data_root,
        order_type="LIMIT",
        metadata=metadata,
        pair="USD_CAD",
        side="LONG",
        lane_id=PREDICTIVE_SCOUT_LANE_ID,
    )
    payload = json.loads(intents.read_text())
    payload["generated_at_utc"] = now.isoformat()
    intent = payload["results"][0]["intent"]
    intent["entry"] = 1.41610 if crossed else 1.41590
    intent["tp"] = 1.41690
    intent["sl"] = 1.41520
    intent["market_context"] = {
        "regime": "BREAKOUT_FAILURE contrarian forecast-failure scout",
        "narrative": "daily-stable USD_CAD DOWN forecast failure is tested through a passive LONG LIMIT",
        "chart_story": "bounded forward evidence at the replay entry geometry",
        "method": "BREAKOUT_FAILURE",
        "invalidation": "attached replay stop trades",
    }
    intent["metadata"].update(
        {
            "predictive_scout_risk_tier": "DISCOVERY",
            "predictive_scout_nav_jpy_at_sizing": 200_000.0,
            "predictive_scout_max_risk_pct_nav": 0.10,
            "predictive_scout_max_loss_jpy": 200.0,
            "predictive_scout_effective_max_loss_jpy": 200.0,
            "predictive_scout_planned_initial_risk_jpy": 75.6,
            "predictive_scout_planned_initial_risk_pct_nav": 0.0378,
        }
    )
    intent["metadata"]["predictive_scout_sizing_digest"] = (
        predictive_scout_sizing_digest(execution_module._intent_from_json(intent))
    )
    intents.write_text(json.dumps(payload))
    return intents


def _write_predictive_scout_verified_decision(
    root: Path,
    *,
    intents_path: Path,
    suffix: str = "",
    action: str = "TRADE",
    market_pair: str = "USD_CAD",
    market_direction: str = "LONG",
    generated_at_utc: str | None = None,
) -> Path:
    name_suffix = f"-{suffix}" if suffix else ""
    path = root / f"gpt_verified_predictive_scout{name_suffix}.json"
    intents_payload = json.loads(intents_path.read_text())
    intent = intents_payload["results"][0]["intent"]
    metadata = intent["metadata"]
    selected_units = abs(int(intent["units"]))
    lane_packet = {
        "lane_id": PREDICTIVE_SCOUT_LANE_ID,
        "pair": intent["pair"],
        "direction": intent["side"],
        "order_type": intent["order_type"],
        "entry": intent["entry"],
        "tp": intent["tp"],
        "sl": intent["sl"],
        "units": intent["units"],
        "predictive_scout": {
            "forecast_cycle_id": metadata["forecast_cycle_id"],
            "rule_digest": metadata["predictive_scout_rule_digest"],
            "risk_tier": metadata["predictive_scout_risk_tier"],
            "planned_initial_risk_jpy": metadata[
                "predictive_scout_planned_initial_risk_jpy"
            ],
            "sizing_digest": metadata["predictive_scout_sizing_digest"],
            "generated_at_utc": metadata["predictive_scout_generated_at_utc"],
            "expires_at_utc": metadata["predictive_scout_expires_at_utc"],
        },
    }
    decision: dict[str, Any] = {
        "action": action,
        "selected_lane_id": PREDICTIVE_SCOUT_LANE_ID,
        "selected_lane_ids": [PREDICTIVE_SCOUT_LANE_ID],
        "market_read_first": {
            "next_30m_prediction": {
                "pair": market_pair,
                "direction": market_direction,
            },
            "next_2h_prediction": {
                "pair": market_pair,
                "direction": market_direction,
            },
            "best_trade_if_forced": {
                "pair": market_pair,
                "direction": market_direction,
            },
        },
    }
    if action == "TRADE":
        allocation = {
            "decision": "ALLOCATE",
            "lane_id": PREDICTIVE_SCOUT_LANE_ID,
            "size_multiple": 1.0,
            "selected_units": selected_units,
            "allocation_board_sha256": "e" * 64,
            "rationale": "Predictive SCOUT remains fixed at its current-NAV risk-sized units.",
        }
        allocation_sha = execution_module.hashlib.sha256(
            json.dumps(
                allocation,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        decision["capital_allocation"] = allocation
        decision["decision_provenance"] = {
            "schema_version": 2,
            "author_kind": "CODEX_MARKET_READ",
            "capital_allocation_edge_basis": "PREDICTIVE_SCOUT_FORWARD_EVIDENCE",
            "capital_allocation_sha256": allocation_sha,
            "capital_allocation_board_sha256": "e" * 64,
            "authorized_size_multiple": 1.0,
            "authorized_units": selected_units,
        }
    payload = {
        "generated_at_utc": generated_at_utc
        or datetime.now(timezone.utc).isoformat(),
        "status": "ACCEPTED",
        "decision": decision,
        "verification_issues": [],
        "input_packet": {"lanes": [lane_packet]},
    }
    if action == "TRADE":
        _attach_market_read_prediction_lineage(payload, prediction_hex="e")
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def _write_predictive_scout_policy(root: Path) -> Path:
    config_root = root / "config"
    config_root.mkdir(exist_ok=True)
    path = config_root / "predictive_scout_policy.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "enabled": True,
                "mode": "FORWARD_EVIDENCE_ONLY",
                "allowed_sources": ["BIDASK_REPLAY_PRECISION"],
                "order_types": ["LIMIT"],
                "risk_tiers": {
                    "DISCOVERY": {"max_risk_pct_nav": 0.10},
                    "EMERGING": {"max_risk_pct_nav": 0.25},
                    "ESTABLISHED": {"max_risk_pct_nav": 0.50},
                    "STRONG": {"max_risk_pct_nav": 0.75},
                    "PROVEN": {"max_risk_pct_nav": 1.00},
                },
                "max_per_trade_risk_pct_nav": 1.0,
                "max_concurrent_risk_pct_nav": 2.0,
                "max_concurrent": 2,
                "max_sent_per_campaign_day": 8,
                "max_ttl_minutes": 90,
                "minimum_replay_samples": 30,
                "minimum_active_days": 5,
                "minimum_profit_factor": 1.2,
                "minimum_positive_day_rate": 2 / 3,
                "loss_cooldown_hours": 6,
                "quarantine_after_resolved_losses": 3,
                "quarantine_requires_negative_net": True,
                "promotion_min_resolved_exits": 30,
                "promotion_min_active_days": 5,
                "promotion_min_profit_factor": 1.2,
                "promotion_min_positive_day_rate": 0.6667,
                "promotion_one_sided_confidence": 0.95,
                "promotion_requires_all_resolved_exit_expectancy_lower_bound_positive": True,
            }
        )
    )
    ExecutionLedger(
        db_path=root / "data" / "execution_ledger.db",
        report_path=root / "execution_ledger_report.md",
    )._init_db()
    with closing(sqlite3.connect(root / "data" / "execution_ledger.db")) as con, con:
        con.execute(
            "INSERT INTO sync_state(key, value, updated_at_utc) VALUES ('last_oanda_transaction_id', '100', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at_utc=excluded.updated_at_utc",
            (datetime.now(timezone.utc).isoformat(),),
        )
    return path


def _write_empty_guardian_artifacts(root: Path) -> dict[str, Path]:
    paths = {
        "watchdog": root / "watchdog.json",
        "consumption": root / "guardian_receipt_consumption.json",
        "operator_review": root / "guardian_receipt_operator_review.json",
        "broker_snapshot": root / "broker_snapshot.json",
    }
    paths["watchdog"].write_text(
        json.dumps({"status": "OK", "issue_status": "OK", "guardian_receipt": {"issues": []}}),
        encoding="utf-8",
    )
    paths["consumption"].write_text(
        json.dumps({"status": "OK", "normal_routing_allowed": True, "classifications": []}),
        encoding="utf-8",
    )
    paths["operator_review"].write_text(
        json.dumps({"status": "OK", "normal_routing_allowed": True, "reviews": []}),
        encoding="utf-8",
    )
    paths["broker_snapshot"].write_text(
        json.dumps({"generated_at_utc": datetime.now(timezone.utc).isoformat(), "positions": [], "orders": []}),
        encoding="utf-8",
    )
    return paths


def _write_gateway_guardian_watchdog_issue(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "BLOCKED",
                "severity": "P1",
                "guardian_receipt": {
                    "issues": [
                        {
                            "code": "GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER",
                            "severity": "P1",
                            "message": "receipt_lifecycle=EXPIRED while consumed_by_trader=false",
                            "receipt_event_id": "receipt-stale-reduce",
                            "receipt_action": "REDUCE",
                            "receipt_lifecycle": "EXPIRED",
                            "consumed_by_trader": False,
                            "normal_routing_allowed": False,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )


def _intents(
    root: Path,
    *,
    status: str = "LIVE_READY",
    metadata: dict[str, Any] | None = None,
    order_type: str = "STOP-ENTRY",
    units: int = 1000,
    pair: str = "EUR_USD",
    side: str = "LONG",
    lane_id: str = "lane:EUR_USD:LONG",
    entry: float | None = None,
    tp: float | None = None,
    sl: float | None = None,
) -> Path:
    path = root / "intents.json"
    resolved_metadata = dict(
        metadata
        or {
            "desk": "trend_trader",
            "campaign_role": "NOW",
        }
    )
    if (
        pair == "EUR_USD"
        and side == "LONG"
        and order_type in {"STOP", "STOP-ENTRY", "STOP_ENTRY"}
        and not resolved_metadata.get("predictive_scout")
    ):
        for key, value in _ordinary_exact_all_exit_edge_metadata().items():
            resolved_metadata.setdefault(key, value)
    path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "lane_id": lane_id,
                        "status": status,
                        "risk_allowed": True,
                        "intent": {
                            "pair": pair,
                            "side": side,
                            "order_type": order_type,
                            "units": units,
                            "entry": (
                                entry
                                if entry is not None
                                else 1.17306
                                if order_type == "MARKET"
                                else 0.66240
                                if pair == "AUD_USD"
                                else 1.17330
                            ),
                            "tp": (
                                tp
                                if tp is not None
                                else 0.66360
                                if pair == "AUD_USD"
                                else 1.17450
                            ),
                            "sl": (
                                sl
                                if sl is not None
                                else 0.66160
                                if pair == "AUD_USD"
                                # The ordinary MARKET numeric fixture must
                                # leave room for the independently audited
                                # exit/financing floor plus the entry-p95
                                # priceBound allowance under its 100 JPY cap.
                                else 1.17253
                                if order_type == "MARKET"
                                and "forecast_directional_economic_hit_rate"
                                in resolved_metadata
                                else 1.17250
                            ),
                            "thesis": "trend continuation",
                            "owner": "trader",
                            "market_context": {
                                "regime": "TREND_CONTINUATION campaign lane",
                                "narrative": "trend continuation pressure",
                                "chart_story": "trend staircase",
                                "method": "TREND_CONTINUATION",
                                "invalidation": "SL trades",
                            },
                            "metadata": resolved_metadata,
                        },
                    }
                ]
            }
        )
    )
    return path


class DisasterStopOrderRequestTest(unittest.TestCase):
    """§3.5-K disaster stop attach in _oanda_order_request (2026-06-11)."""

    SL_KEYS = ("QR_NEW_ENTRY_INITIAL_SL", "QR_TRADER_DISABLE_SL_REPAIR")

    def setUp(self) -> None:
        self._prior = {k: os.environ.get(k) for k in self.SL_KEYS}

    def tearDown(self) -> None:
        for k, v in self._prior.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    @staticmethod
    def _intent(metadata: dict | None = None):
        from quant_rabbit.models import OrderIntent, OrderType, Side

        return OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=5000,
            tp=1.1620,
            sl=1.1470,
            thesis="test",
            entry=1.1500,
            metadata=metadata or {},
        )

    def test_sl_free_mode_attaches_disaster_stop_from_metadata(self) -> None:
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
        order = execution_module._oanda_order_request(
            self._intent({"disaster_sl": 1.1380, "disaster_sl_pips": 120.0})
        )
        self.assertEqual(order["stopLossOnFill"]["price"], "1.13800")

    def test_sl_free_mode_without_metadata_stays_sl_free(self) -> None:
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
        order = execution_module._oanda_order_request(self._intent({"disaster_sl_missing": "H4_ATR_MISSING"}))
        self.assertNotIn("stopLossOnFill", order)

    def test_initial_sl_mode_keeps_intent_sl_over_disaster(self) -> None:
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ["QR_NEW_ENTRY_INITIAL_SL"] = "1"
        order = execution_module._oanda_order_request(self._intent({"disaster_sl": 1.1380}))
        self.assertEqual(order["stopLossOnFill"]["price"], "1.14700")

    def test_firepower_route_keeps_intent_sl_over_disaster(self) -> None:
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
        order = execution_module._oanda_order_request(
            self._intent(
                {
                    "campaign_role": "OANDA_FIREPOWER_ROUTE",
                    "positive_rotation_oanda_campaign_firepower_vehicle_match": True,
                    "disaster_sl": 1.1380,
                }
            )
        )
        self.assertEqual(order["stopLossOnFill"]["price"], "1.14700")

    def test_normal_sl_mode_unchanged(self) -> None:
        os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
        os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
        order = execution_module._oanda_order_request(self._intent({"disaster_sl": 1.1380}))
        self.assertEqual(order["stopLossOnFill"]["price"], "1.14700")


if __name__ == "__main__":
    unittest.main()
