from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from quant_rabbit.cli import main
from quant_rabbit.trader_prompts import (
    BRANCH_ENTRY,
    BRANCH_POSITION,
    BRANCH_REFRESH,
    BRANCH_VERIFY,
    _fresh_close_recommendations,
    route_trader_prompts,
)


class TraderPromptRouteTest(unittest.TestCase):
    def test_routes_missing_artifacts_to_refresh_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            route = route_trader_prompts(
                snapshot_path=root / "missing_snapshot.json",
                target_state_path=root / "missing_target.json",
                intents_path=root / "missing_intents.json",
                decision_response_path=None,
            )

        self.assertEqual(route.branch, BRANCH_REFRESH)
        self.assertIn("10_precheck_refresh.md", _read_paths(route)[-1])

    def test_routes_flat_open_target_with_live_ready_lanes_to_entry_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_ENTRY)
        read_paths = _read_paths(route)
        self.assertTrue(any(path.endswith("30_entry_decision.md") for path in read_paths))
        self.assertTrue(any(path.endswith("90_decision_receipt_schema.md") for path in read_paths))

    def test_stale_trader_overrides_routes_open_target_to_refresh_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            snapshot = json.loads(files["snapshot"].read_text())
            fetched_at = datetime.fromisoformat(snapshot["fetched_at_utc"])
            files["trader_overrides"].write_text(
                json.dumps(
                    {
                        "expires_at_utc": (fetched_at - timedelta(seconds=1)).isoformat(),
                        "bias_overrides": {"EUR_USD": {"LONG": -20.0}},
                        "blocked_lanes": [],
                    }
                )
            )

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_REFRESH)
        self.assertTrue(any("daily-review feedback stale" in reason for reason in route.reasons))

    def test_missing_trader_overrides_routes_open_target_to_refresh_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["trader_overrides"].unlink()

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_REFRESH)
        self.assertTrue(any("daily-review feedback missing" in reason for reason in route.reasons))

    def test_stale_trader_overrides_does_not_preempt_position_management(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "101",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit": 1.18,
                        "stop_loss": None,
                        "owner": "trader",
                    }
                ],
            )
            snapshot = json.loads(files["snapshot"].read_text())
            fetched_at = datetime.fromisoformat(snapshot["fetched_at_utc"])
            files["trader_overrides"].write_text(
                json.dumps({"expires_at_utc": (fetched_at - timedelta(seconds=1)).isoformat()})
            )

            prior = os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            try:
                route = route_trader_prompts(**_route_paths(files), decision_response_path=None)
            finally:
                if prior is not None:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

        self.assertEqual(route.branch, BRANCH_POSITION)
        self.assertIn("needs protection repair", route.reasons[0])

    def test_routes_unprotected_trader_position_to_position_management(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "101",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit": 1.18,
                        "stop_loss": None,
                        "owner": "trader",
                    }
                ],
            )

            prior = os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            try:
                route = route_trader_prompts(**_route_paths(files), decision_response_path=None)
            finally:
                if prior is not None:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

        self.assertEqual(route.branch, BRANCH_POSITION)
        self.assertIn("needs protection repair", route.reasons[0])

    def test_sl_free_trader_tp_only_position_does_not_force_position_branch(self) -> None:
        # Under QR_TRADER_DISABLE_SL_REPAIR=1 the operator directive 「SLいらない」
        # makes trader-owned SL=None intentional. Routing must let the operator
        # reach BRANCH_ENTRY for fresh entries instead of being trapped in
        # BRANCH_POSITION repair mode every cycle.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "470395",
                        "pair": "EUR_USD",
                        "side": "SHORT",
                        "take_profit": 1.17026,
                        "stop_loss": None,
                        "owner": "trader",
                    }
                ],
            )

            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                route = route_trader_prompts(**_route_paths(files), decision_response_path=None)
            finally:
                if prior is None:
                    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
                else:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

        self.assertEqual(route.branch, BRANCH_ENTRY)

    def test_sl_free_profitable_position_with_tp_rebalance_routes_to_position_management(self) -> None:
        """A WAIT-eligible SL-free position still routes to protection when
        the deterministic TP rebalancer has a live adjustment."""
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
            files["pair_charts"].write_text(
                json.dumps(
                    {
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
                )
            )
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

            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                route = route_trader_prompts(**_route_paths(files), decision_response_path=None)
            finally:
                if prior is None:
                    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
                else:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

        self.assertEqual(route.branch, BRANCH_POSITION)
        self.assertTrue(any("TP rebalance required" in reason for reason in route.reasons))
        self.assertTrue(any("forecast_harvest" in reason for reason in route.reasons))

    def test_sl_free_trader_no_broker_tp_runner_does_not_force_position_branch(self) -> None:
        # Missing broker TP is preserved as a no-broker-TP runner under the
        # SL-free runtime unless explicit TP repair is enabled. It must not
        # trap the trader in BRANCH_POSITION and starve fresh entries.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "999",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit": None,
                        "stop_loss": None,
                        "owner": "trader",
                    }
                ],
            )

            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            prior_tp = os.environ.get("QR_ENABLE_MISSING_TP_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            os.environ.pop("QR_ENABLE_MISSING_TP_REPAIR", None)
            try:
                route = route_trader_prompts(**_route_paths(files), decision_response_path=None)
            finally:
                if prior is None:
                    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
                else:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior
                if prior_tp is None:
                    os.environ.pop("QR_ENABLE_MISSING_TP_REPAIR", None)
                else:
                    os.environ["QR_ENABLE_MISSING_TP_REPAIR"] = prior_tp

        self.assertEqual(route.branch, BRANCH_ENTRY)

    def test_fresh_forecast_persistence_recommend_close_routes_to_position_management(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "555",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit": 1.18,
                        "stop_loss": None,
                        "owner": "trader",
                        "unrealized_pl_jpy": -1200.0,
                    }
                ],
            )
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
                                "side": "LONG",
                                "verdict": "RECOMMEND_CLOSE",
                                "reason": "last 3 forecasts flipped to DOWN",
                            }
                        ],
                    }
                )
            )

            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                route = route_trader_prompts(**_route_paths(files), decision_response_path=None)
            finally:
                if prior is None:
                    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
                else:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

        self.assertEqual(route.branch, BRANCH_POSITION)
        self.assertTrue(any("loss-cut review required" in reason for reason in route.reasons))
        self.assertTrue(any("forecast_persistence RECOMMEND_CLOSE" in reason for reason in route.reasons))

    def test_position_thesis_recommend_close_routes_with_context_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "471739",
                        "pair": "EUR_USD",
                        "side": "SHORT",
                        "take_profit": 1.16056,
                        "stop_loss": None,
                        "owner": "trader",
                        "unrealized_pl_jpy": -57.5,
                    }
                ],
            )
            snapshot = json.loads(files["snapshot"].read_text())
            generated_at = (
                datetime.fromisoformat(snapshot["fetched_at_utc"]) + timedelta(seconds=1)
            ).isoformat()
            (root / "position_thesis_report.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": generated_at,
                        "assessments": [
                            {
                                "trade_id": "471739",
                                "pair": "EUR_USD",
                                "side": "SHORT",
                                "verdict": "REVIEW_CLOSE",
                                "aggregate_score": 21.32,
                                "rationale_lines": ["synthetic detector support"],
                                "context_notes": [
                                    "adverse technical loss: entry thesis lacks invalidation_price",
                                    "technical invalidation confirmed against SHORT",
                                ],
                            }
                        ],
                    }
                )
            )

            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                route = route_trader_prompts(**_route_paths(files), decision_response_path=None)
            finally:
                if prior is None:
                    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
                else:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

        self.assertEqual(route.branch, BRANCH_POSITION)
        self.assertTrue(any("position_thesis REVIEW_CLOSE" in reason for reason in route.reasons))
        self.assertTrue(any("adverse technical loss" in reason for reason in route.reasons))

    def test_position_thesis_adverse_technical_loss_is_standing_authorized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "471414",
                        "pair": "EUR_USD",
                        "side": "SHORT",
                        "take_profit": 1.16056,
                        "stop_loss": None,
                        "owner": "trader",
                        "unrealized_pl_jpy": -2109.8,
                    }
                ],
            )
            snapshot = json.loads(files["snapshot"].read_text())
            generated_at = (
                datetime.fromisoformat(snapshot["fetched_at_utc"]) + timedelta(seconds=1)
            ).isoformat()
            (root / "position_thesis_report.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": generated_at,
                        "assessments": [
                            {
                                "trade_id": "471414",
                                "pair": "EUR_USD",
                                "side": "SHORT",
                                "verdict": "REVIEW_CLOSE",
                                "aggregate_score": 47.0,
                                "rationale_lines": [
                                    "patterns +30.0",
                                    "forward-proj +25.0",
                                    "chart-tech -8.0",
                                ],
                                "context_notes": [
                                    "adverse technical loss: no entry thesis; current ask 1.16310 >= entry-buffer 1.15891",
                                    "technical invalidation confirmed against SHORT: H1 RSI=65.3; M15 trend up; M30 MACD+; M5 ST+",
                                ],
                            }
                        ],
                    }
                )
            )

            recs = _fresh_close_recommendations(snapshot, data_root=root)

        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["source"], "position_thesis")
        self.assertTrue(recs[0]["gate_b_standing_authorized"])
        self.assertIn("adverse technical loss", recs[0]["reason"])
        self.assertIn("technical invalidation confirmed", recs[0]["reason"])

    def test_position_thesis_score_only_review_is_not_standing_authorized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "471414",
                        "pair": "EUR_USD",
                        "side": "SHORT",
                        "take_profit": 1.16056,
                        "stop_loss": None,
                        "owner": "trader",
                    }
                ],
            )
            snapshot = json.loads(files["snapshot"].read_text())
            generated_at = (
                datetime.fromisoformat(snapshot["fetched_at_utc"]) + timedelta(seconds=1)
            ).isoformat()
            (root / "position_thesis_report.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": generated_at,
                        "assessments": [
                            {
                                "trade_id": "471414",
                                "pair": "EUR_USD",
                                "side": "SHORT",
                                "verdict": "REVIEW_CLOSE",
                                "aggregate_score": 12.0,
                                "rationale_lines": ["score weakened but no invalidation hit"],
                                "context_notes": [],
                            }
                        ],
                    }
                )
            )

            recs = _fresh_close_recommendations(snapshot, data_root=root)

        self.assertEqual(len(recs), 1)
        self.assertFalse(recs[0]["gate_b_standing_authorized"])

    def test_position_management_review_exit_carryforward_routes_to_position_management(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "471817",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit": 1.1800,
                        "stop_loss": None,
                        "owner": "trader",
                        "unrealized_pl_jpy": -1900.0,
                    }
                ],
            )
            snapshot = json.loads(files["snapshot"].read_text())
            generated_at = (
                datetime.fromisoformat(snapshot["fetched_at_utc"]) - timedelta(minutes=20)
            ).isoformat()
            (root / "position_management.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": generated_at,
                        "action": "REVIEW_EXIT",
                        "positions": [
                            {
                                "trade_id": "471817",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "action": "REVIEW_EXIT",
                                "reasons": [
                                    "score context before structural review",
                                    "loss-cut: structural OB broken across 2 TFs (M15@1.17000, H1@1.17100) (-1900 JPY)",
                                ],
                            }
                        ],
                    }
                )
            )

            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                route = route_trader_prompts(**_route_paths(files), decision_response_path=None)
                recs = _fresh_close_recommendations(snapshot, data_root=root)
            finally:
                if prior is None:
                    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
                else:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

        self.assertEqual(route.branch, BRANCH_POSITION)
        self.assertTrue(any("position_management REVIEW_EXIT" in reason for reason in route.reasons))
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["source"], "position_management")
        self.assertEqual(recs[0]["evidence_ref"], "position:management:471817")
        self.assertTrue(recs[0]["gate_b_standing_authorized"])
        self.assertIn("loss-cut: structural OB broken", recs[0]["reason"])

    def test_stale_position_management_review_exit_does_not_route_to_position_management(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "471817",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit": 1.1800,
                        "stop_loss": None,
                        "owner": "trader",
                        "unrealized_pl_jpy": -1900.0,
                    }
                ],
            )
            snapshot = json.loads(files["snapshot"].read_text())
            generated_at = (
                datetime.fromisoformat(snapshot["fetched_at_utc"]) - timedelta(minutes=10)
            ).isoformat()
            (root / "position_management.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": generated_at,
                        "positions": [
                            {
                                "trade_id": "471817",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "action": "REVIEW_EXIT",
                                "reasons": [
                                    "loss-cut: structural OB broken across 2 TFs (M15@1.17000, H1@1.17100) (-1900 JPY)",
                                ],
                            }
                        ],
                    }
                )
            )

            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                with patch("quant_rabbit.trader_prompts.POSITION_MANAGEMENT_REVIEW_EXIT_TTL_SECONDS", 60.0):
                    route = route_trader_prompts(**_route_paths(files), decision_response_path=None)
                    recs = _fresh_close_recommendations(snapshot, data_root=root)
            finally:
                if prior is None:
                    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
                else:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

        self.assertEqual(route.branch, BRANCH_ENTRY)
        self.assertEqual(recs, ())
        self.assertFalse(any("position_management REVIEW_EXIT" in reason for reason in route.reasons))

    def test_entry_thesis_blocker_routes_without_close_recommendation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "471910",
                        "pair": "AUD_USD",
                        "side": "SHORT",
                        "take_profit": 0.7164,
                        "stop_loss": 0.7195,
                        "owner": "trader",
                    }
                ],
            )
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
                                "trade_id": "471910",
                                "pair": "AUD_USD",
                                "side": "SHORT",
                                "status": "UNVERIFIABLE",
                                "verdict": "REQUIRE_THESIS_REPAIR",
                                "rationale": "missing entry_thesis_ledger row",
                            }
                        ],
                    }
                )
            )

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_POSITION)
        self.assertTrue(any("entry-thesis repair required" in reason for reason in route.reasons))
        self.assertFalse(any("loss-cut review required" in reason for reason in route.reasons))

    def test_stale_forecast_persistence_recommend_close_does_not_route_to_position_management(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "555",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit": 1.18,
                        "stop_loss": None,
                        "owner": "trader",
                        "unrealized_pl_jpy": -1200.0,
                    }
                ],
            )
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
                                "side": "LONG",
                                "verdict": "RECOMMEND_CLOSE",
                                "reason": "old forecast flip from a prior snapshot",
                            }
                        ],
                    }
                )
            )

            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                route = route_trader_prompts(**_route_paths(files), decision_response_path=None)
            finally:
                if prior is None:
                    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
                else:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

        self.assertEqual(route.branch, BRANCH_ENTRY)
        self.assertFalse(any("loss-cut review required" in reason for reason in route.reasons))

    def test_fresh_close_recommendation_for_closed_trade_does_not_route_to_position_management(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "555",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit": 1.18,
                        "stop_loss": None,
                        "owner": "trader",
                    }
                ],
            )
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
                                "trade_id": "closed-444",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "verdict": "RECOMMEND_CLOSE",
                                "reason": "report belongs to a trade no longer open",
                            }
                        ],
                    }
                )
            )

            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                route = route_trader_prompts(**_route_paths(files), decision_response_path=None)
            finally:
                if prior is None:
                    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
                else:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

        self.assertEqual(route.branch, BRANCH_ENTRY)
        self.assertFalse(any("closed-444" in reason for reason in route.reasons))

    def test_sl_free_missing_tp_repair_opt_in_routes_to_position_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "999",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit": None,
                        "stop_loss": None,
                        "owner": "trader",
                    }
                ],
            )

            prior_sl = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            prior_tp = os.environ.get("QR_ENABLE_MISSING_TP_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            os.environ["QR_ENABLE_MISSING_TP_REPAIR"] = "1"
            try:
                route = route_trader_prompts(**_route_paths(files), decision_response_path=None)
            finally:
                if prior_sl is None:
                    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
                else:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl
                if prior_tp is None:
                    os.environ.pop("QR_ENABLE_MISSING_TP_REPAIR", None)
                else:
                    os.environ["QR_ENABLE_MISSING_TP_REPAIR"] = prior_tp

        self.assertEqual(route.branch, BRANCH_POSITION)

    def test_manual_position_missing_tp_routes_to_tp_only_position_management(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "manual-999",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit": None,
                        "stop_loss": None,
                        "owner": "unknown",
                        "unrealized_pl_jpy": 120.0,
                    }
                ],
            )

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_POSITION)
        self.assertIn("TP-only profit management", route.reasons[0])

    def test_underwater_manual_position_missing_tp_does_not_force_position_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "manual-998",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit": None,
                        "stop_loss": None,
                        "owner": "unknown",
                        "unrealized_pl_jpy": -2625.0,
                    }
                ],
            )

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_ENTRY)

    def test_manual_position_with_tp_missing_sl_does_not_force_position_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "manual-1000",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "take_profit": 1.1800,
                        "stop_loss": None,
                        "owner": "manual",
                    }
                ],
            )

            route = route_trader_prompts(**_route_paths(files), decision_response_path=None)

        self.assertEqual(route.branch, BRANCH_ENTRY)

    def test_routes_existing_decision_response_to_verify_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision_response = root / "codex_trader_decision_response.json"
            decision_response.write_text(json.dumps({"action": "WAIT"}))

            route = route_trader_prompts(**_route_paths(files), decision_response_path=decision_response)

        self.assertEqual(route.branch, BRANCH_VERIFY)
        self.assertTrue(any(path.endswith("40_verify_execute.md") for path in _read_paths(route)))

    def test_routes_rejected_decision_response_back_to_entry_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision_response = root / "codex_trader_decision_response.json"
            decision_response.write_text(json.dumps({"action": "TRADE"}))
            files["gpt_decision"].write_text(
                json.dumps(
                    {
                        "status": "REJECTED",
                        "decision": {"action": "TRADE"},
                        "verification_issues": [{"code": "UNKNOWN_LANE", "severity": "BLOCK"}],
                    }
                )
            )
            _set_mtime(decision_response, 100.0)
            _set_mtime(files["snapshot"], 99.0)
            _set_mtime(files["intents"], 99.0)
            _set_mtime(files["gpt_decision"], 101.0)

            route = route_trader_prompts(**_route_paths(files), decision_response_path=decision_response)

        self.assertEqual(route.branch, BRANCH_ENTRY)
        self.assertIn("already verified as REJECTED", route.reasons[0])

    def test_routes_consumed_decision_response_back_to_entry_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision_response = root / "codex_trader_decision_response.json"
            decision_response.write_text(json.dumps({"action": "TRADE"}))
            files["live_order"].write_text(json.dumps({"status": "SENT", "sent": True}))
            _set_mtime(decision_response, 100.0)
            _set_mtime(files["snapshot"], 99.0)
            _set_mtime(files["intents"], 99.0)
            _set_mtime(files["live_order"], 101.0)

            route = route_trader_prompts(**_route_paths(files), decision_response_path=decision_response)

        self.assertEqual(route.branch, BRANCH_ENTRY)
        self.assertIn("already consumed by live order gateway receipt", route.reasons[0])

    def test_routes_accepted_trade_without_newer_gateway_to_verify_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision_response = root / "codex_trader_decision_response.json"
            decision_response.write_text(json.dumps({"action": "TRADE"}))
            files["autotrade_report"].write_text("# stale no-send cycle\n")
            files["gpt_decision"].write_text(
                json.dumps(
                    {
                        "status": "ACCEPTED",
                        "decision": {
                            "action": "TRADE",
                            "selected_lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                            "selected_lane_ids": ["trend_trader:EUR_USD:LONG:TREND_CONTINUATION"],
                        },
                    }
                )
            )
            _set_mtime(decision_response, 100.0)
            _set_mtime(files["snapshot"], 99.0)
            _set_mtime(files["intents"], 99.0)
            _set_mtime(files["autotrade_report"], 101.0)
            _set_mtime(files["gpt_decision"], 102.0)

            route = route_trader_prompts(**_route_paths(files), decision_response_path=decision_response)

        self.assertEqual(route.branch, BRANCH_VERIFY)
        self.assertIn("accepted TRADE decision has no newer gateway receipt", route.reasons[0])

    def test_routes_accepted_trade_with_newer_gateway_back_to_entry_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision_response = root / "codex_trader_decision_response.json"
            decision_response.write_text(json.dumps({"action": "TRADE"}))
            files["gpt_decision"].write_text(
                json.dumps(
                    {
                        "status": "ACCEPTED",
                        "decision": {
                            "action": "TRADE",
                            "selected_lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                            "selected_lane_ids": ["trend_trader:EUR_USD:LONG:TREND_CONTINUATION"],
                        },
                    }
                )
            )
            files["live_order"].write_text(json.dumps({"status": "SENT", "sent": True}))
            _set_mtime(decision_response, 100.0)
            _set_mtime(files["snapshot"], 99.0)
            _set_mtime(files["intents"], 99.0)
            _set_mtime(files["gpt_decision"], 101.0)
            _set_mtime(files["live_order"], 102.0)

            route = route_trader_prompts(**_route_paths(files), decision_response_path=decision_response)

        self.assertEqual(route.branch, BRANCH_ENTRY)
        self.assertIn("accepted TRADE decision already consumed by live order gateway receipt", route.reasons[0])

    def test_routes_decision_older_than_refreshed_broker_truth_back_to_entry_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision_response = root / "codex_trader_decision_response.json"
            decision_response.write_text(json.dumps({"action": "TRADE"}))
            _set_mtime(decision_response, 100.0)
            _set_mtime(files["snapshot"], 101.0)
            _set_mtime(files["intents"], 101.0)

            route = route_trader_prompts(**_route_paths(files), decision_response_path=decision_response)

        self.assertEqual(route.branch, BRANCH_ENTRY)
        self.assertIn("predates refreshed broker snapshot", route.reasons[0])

    def test_routes_same_timestamp_decision_response_to_verify_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision_response = root / "codex_trader_decision_response.json"
            decision_response.write_text(json.dumps({"action": "TRADE"}))
            _set_mtime(decision_response, 100.0)
            _set_mtime(files["snapshot"], 100.0)
            _set_mtime(files["intents"], 100.0)

            route = route_trader_prompts(**_route_paths(files), decision_response_path=decision_response)

        self.assertEqual(route.branch, BRANCH_VERIFY)

    def test_cli_prints_prompt_route_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                code = main(
                    [
                        "trader-prompt-route",
                        "--snapshot",
                        str(files["snapshot"]),
                        "--target-state",
                        str(files["target"]),
                        "--intents",
                        str(files["intents"]),
                        "--pair-charts",
                        str(files["pair_charts"]),
                        "--cross-asset",
                        str(files["cross_asset"]),
                        "--flow",
                        str(files["flow"]),
                        "--currency-strength",
                        str(files["currency_strength"]),
                        "--levels",
                        str(files["levels"]),
                        "--calendar",
                        str(files["calendar"]),
                        "--cot",
                        str(files["cot"]),
                        "--option-skew",
                        str(files["option_skew"]),
                        "--attack-advice",
                        str(files["attack_advice"]),
                        "--learning-audit",
                        str(files["learning_audit"]),
                        "--trader-overrides",
                        str(files["trader_overrides"]),
                        "--decision-response",
                        str(root / "missing_decision_response.json"),
                        "--live-order",
                        str(files["live_order"]),
                        "--position-execution",
                        str(files["position_execution"]),
                        "--autotrade-report",
                        str(files["autotrade_report"]),
                    ]
                )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["branch"], BRANCH_ENTRY)
        self.assertTrue(any(item["path"].endswith("30_entry_decision.md") for item in payload["read_order"]))


def _route_paths(files: dict[str, Path]) -> dict[str, Path]:
    return {
        "snapshot_path": files["snapshot"],
        "target_state_path": files["target"],
        "intents_path": files["intents"],
        "pair_charts_path": files["pair_charts"],
        "cross_asset_path": files["cross_asset"],
        "flow_path": files["flow"],
        "currency_strength_path": files["currency_strength"],
        "levels_path": files["levels"],
        "calendar_path": files["calendar"],
        "cot_path": files["cot"],
        "option_skew_path": files["option_skew"],
        "attack_advice_path": files["attack_advice"],
        "learning_audit_path": files["learning_audit"],
        "trader_overrides_path": files["trader_overrides"],
        "gpt_decision_path": files["gpt_decision"],
        "live_order_path": files["live_order"],
        "position_execution_path": files["position_execution"],
        "autotrade_report_path": files["autotrade_report"],
    }


def _read_paths(route) -> list[str]:
    return [str(doc.path) for doc in route.read_order]


def _fixtures(root: Path, *, positions: list[dict] | None = None) -> dict[str, Path]:
    files = {
        "snapshot": root / "broker_snapshot.json",
        "target": root / "daily_target_state.json",
        "intents": root / "order_intents.json",
        "pair_charts": root / "pair_charts.json",
        "cross_asset": root / "cross_asset_snapshot.json",
        "flow": root / "flow_snapshot.json",
        "currency_strength": root / "currency_strength.json",
        "levels": root / "levels_snapshot.json",
        "calendar": root / "economic_calendar.json",
        "cot": root / "cot_snapshot.json",
        "option_skew": root / "option_skew_snapshot.json",
        "attack_advice": root / "ai_attack_advice.json",
        "learning_audit": root / "learning_audit.json",
        "trader_overrides": root / "trader_overrides.json",
        "gpt_decision": root / "gpt_trader_decision.json",
        "live_order": root / "live_order_request.json",
        "position_execution": root / "position_execution.json",
        "autotrade_report": root / "autotrade_cycle_report.md",
    }
    now = datetime.now(timezone.utc).isoformat()
    files["snapshot"].write_text(
        json.dumps(
            {
                "fetched_at_utc": now,
                "positions": positions or [],
                "orders": [],
                "quotes": {"EUR_USD": {"bid": 1.17, "ask": 1.1701, "timestamp_utc": now}},
            }
        )
    )
    files["target"].write_text(
        json.dumps(
            {
                "status": "PURSUE_TARGET",
                "remaining_target_jpy": 1000.0,
                "progress_jpy": 0.0,
            }
        )
    )
    files["intents"].write_text(
        json.dumps(
            {
                "results": [
                    {
                        "lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                        "status": "LIVE_READY",
                        "risk_issues": [],
                        "strategy_issues": [],
                        "live_blockers": [],
                    }
                ]
            }
        )
    )
    for key in (
        "pair_charts",
        "cross_asset",
        "flow",
        "currency_strength",
        "levels",
        "calendar",
        "cot",
        "option_skew",
        "attack_advice",
        "learning_audit",
    ):
        files[key].write_text(json.dumps({}))
    files["trader_overrides"].write_text(
        json.dumps(
            {
                "expires_at_utc": (
                    datetime.fromisoformat(now) + timedelta(hours=1)
                ).isoformat(),
                "bias_overrides": {},
                "blocked_lanes": [],
            }
        )
    )
    return files


def _set_mtime(path: Path, value: float) -> None:
    os.utime(path, (value, value))


if __name__ == "__main__":
    unittest.main()
