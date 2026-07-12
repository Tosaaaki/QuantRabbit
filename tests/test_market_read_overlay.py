from __future__ import annotations

import json
import os
import hashlib
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime, timedelta, timezone
from pathlib import Path

import quant_rabbit.market_read_overlay as market_read_overlay_module
from quant_rabbit.forecast_precision import (
    bidask_replay_precision_rule_digest,
    canonical_bidask_replay_precision_rule,
    hit_rate_wilson_lower,
)
from quant_rabbit.market_read_overlay import (
    CAPITAL_ALLOCATION_FORECAST_MIN_SAMPLES,
    CODEX_MARKET_READ_AUTHOR,
    MARKET_READ_OVERLAY_SCHEMA_VERSION,
    MarketReadOverlayError,
    apply_codex_market_read_overlay,
    baseline_core_payload,
    canonical_json_sha256,
    execution_envelope_payload,
    prepare_market_read_baseline,
    revalidate_codex_market_read_artifacts,
    validate_codex_market_read_provenance,
)


NOW = datetime(2026, 7, 11, 3, 0, tzinfo=timezone.utc)
LANE_ID = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET"


class MarketReadOverlayTest(unittest.TestCase):
    def setUp(self) -> None:
        cost_patch = patch(
            "quant_rabbit.capture_economics.read_execution_cost_surface",
            side_effect=lambda _path: _synthetic_execution_cost_surface(),
        )
        cost_patch.start()
        self.addCleanup(cost_patch.stop)

    def test_missing_ledger_source_cannot_be_replaced_by_cwd_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _prepared_paths(root)
            sentinel = root / "__missing_execution_ledger_for_market_read__"
            _write_exact_vehicle_ledger(sentinel, [320.0] * 8)
            baseline = json.loads(paths["baseline"].read_text())
            baseline.pop("decision_provenance", None)
            paths["baseline"].write_text(json.dumps(baseline))
            sources = _sources(paths)
            sources.pop("execution_ledger")
            original_cwd = Path.cwd()
            try:
                os.chdir(root)
                prepare_market_read_baseline(
                    baseline_path=paths["baseline"],
                    packet_path=paths["packet"],
                    evidence_sources=sources,
                    now=NOW,
                )
            finally:
                os.chdir(original_cwd)

            packet = json.loads(paths["packet"].read_text())
            self.assertEqual(
                packet["execution_ledger_allocation_surface"]["parse_status"],
                "MISSING",
            )
            self.assertFalse(
                packet["capital_allocation_board"]["selected_lane"][
                    "allocation_eligible"
                ]
            )

    def test_nonsemantic_source_metadata_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            _write_overlay(paths)
            packet = json.loads(paths["packet"].read_text())
            packet["source_metadata"]["broker_snapshot"][
                "generated_at_utc"
            ] = "2099-01-01T00:00:00+00:00"
            paths["packet"].write_text(json.dumps(packet))

            with self.assertRaisesRegex(
                MarketReadOverlayError,
                "MARKET_READ_EVIDENCE_PACKET_BODY_STALE",
            ):
                _apply(paths)

    def test_uncheckpointed_selected_loss_stales_but_checkpoint_only_does_not(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            _write_overlay(paths)
            with sqlite3.connect(paths["execution_ledger"]) as writer:
                self.assertEqual(writer.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower(), "wal")
                writer.execute("PRAGMA wal_autocheckpoint=0")
                main_sha_before = hashlib.sha256(
                    paths["execution_ledger"].read_bytes()
                ).hexdigest()
                _append_vehicle_trade(
                    paths["execution_ledger"],
                    pair="EUR_USD",
                    side="LONG",
                    method="TREND_CONTINUATION",
                    vehicle="MARKET",
                    realized=-5000.0,
                    index=200,
                    connection=writer,
                )
                writer.commit()
                wal_path = Path(str(paths["execution_ledger"]) + "-wal")
                self.assertTrue(wal_path.exists())
                self.assertGreater(wal_path.stat().st_size, 0)
                self.assertEqual(
                    hashlib.sha256(paths["execution_ledger"].read_bytes()).hexdigest(),
                    main_sha_before,
                )
                with self.assertRaisesRegex(
                    MarketReadOverlayError,
                    "MARKET_READ_EVIDENCE_PACKET_STALE",
                ):
                    _apply(paths)

        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            with sqlite3.connect(paths["execution_ledger"]) as writer:
                self.assertEqual(writer.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower(), "wal")
                writer.execute("PRAGMA wal_autocheckpoint=0")
                _append_vehicle_trade(
                    paths["execution_ledger"],
                    pair="GBP_USD",
                    side="SHORT",
                    method="RANGE_ROTATION",
                    vehicle="LIMIT",
                    realized=-5000.0,
                    index=201,
                    connection=writer,
                )
                writer.commit()
                _reprepare(paths)
                _write_overlay(paths)
                writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")

            self.assertEqual(_apply(paths).action, "TRADE")

    def test_mature_negative_all_exit_surface_suppresses_positive_tp_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            intents = json.loads(paths["intents"].read_text())
            metadata = intents["results"][0]["intent"]["metadata"]
            metadata.update(
                {
                    "capture_exact_vehicle_net_trades": 20,
                    "capture_exact_vehicle_net_wins": 8,
                    "capture_exact_vehicle_net_losses": 12,
                    "capture_exact_vehicle_net_jpy": -9440.0,
                    "capture_exact_vehicle_net_expectancy_jpy": -472.0,
                    "capture_exact_vehicle_net_avg_win_jpy": 320.0,
                    "capture_exact_vehicle_net_avg_loss_jpy": 1000.0,
                }
            )
            paths["intents"].write_text(json.dumps(intents))
            _write_exact_vehicle_ledger(
                paths["execution_ledger"],
                [320.0] * 8 + [-1000.0] * 12,
            )
            _reprepare(paths)

            lane = json.loads(paths["packet"].read_text())[
                "capital_allocation_board"
            ]["selected_lane"]

            self.assertFalse(lane["allocation_eligible"])
            self.assertFalse(lane["positive_edge_proven"])
            self.assertEqual(
                lane["edge_basis"],
                "EXACT_VEHICLE_ALL_EXIT_CONTRADICTS_TP",
            )
            self.assertTrue(
                lane["capture"]["exact_vehicle_all_exit"][
                    "blocks_tp_exception"
                ]
            )

    def test_tp_proof_binds_explicit_nondivisible_net_before_rounded_averages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            outcomes = [100.0] * 5 + [100.0001]
            _write_exact_vehicle_ledger(paths["execution_ledger"], outcomes)
            intents = json.loads(paths["intents"].read_text())
            metadata = intents["results"][0]["intent"]["metadata"]
            metadata.update(
                {
                    "capture_take_profit_expectancy_jpy": 100.0,
                    "capture_take_profit_net_jpy": 600.0001,
                    "capture_take_profit_trades": 6,
                    "capture_take_profit_wins": 6,
                    "capture_take_profit_losses": 0,
                    "capture_take_profit_avg_win_jpy": 100.0,
                    "capture_take_profit_avg_loss_jpy": 0.0,
                    "capture_exact_vehicle_net_trades": 6,
                    "capture_exact_vehicle_net_wins": 6,
                    "capture_exact_vehicle_net_losses": 0,
                    "capture_exact_vehicle_net_jpy": 600.0001,
                    "capture_exact_vehicle_net_expectancy_jpy": 100.0,
                    "capture_exact_vehicle_net_avg_win_jpy": 100.0,
                    "capture_exact_vehicle_net_avg_loss_jpy": 0.0,
                }
            )
            paths["intents"].write_text(json.dumps(intents))
            _reprepare(paths)

            lane = json.loads(paths["packet"].read_text())[
                "capital_allocation_board"
            ]["selected_lane"]

            self.assertTrue(lane["allocation_eligible"])
            self.assertEqual(lane["edge_basis"], "EXACT_VEHICLE_TAKE_PROFIT")
            self.assertEqual(
                lane["capture"]["exact_vehicle_all_exit"]["net_jpy"],
                600.0001,
            )
            _write_overlay(paths)
            self.assertEqual(_apply(paths).action, "TRADE")

    def test_selected_vehicle_loss_stales_packet_but_unrelated_vehicle_does_not(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            _write_overlay(paths)
            _append_vehicle_trade(
                paths["execution_ledger"],
                pair="GBP_USD",
                side="SHORT",
                method="RANGE_ROTATION",
                vehicle="LIMIT",
                realized=-5000.0,
                index=100,
            )

            self.assertEqual(_apply(paths).action, "TRADE")

        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            _write_overlay(paths)
            _append_vehicle_trade(
                paths["execution_ledger"],
                pair="EUR_USD",
                side="LONG",
                method="TREND_CONTINUATION",
                vehicle="MARKET",
                realized=-5000.0,
                index=100,
            )

            with self.assertRaisesRegex(
                MarketReadOverlayError,
                "MARKET_READ_EVIDENCE_PACKET_STALE",
            ):
                _apply(paths)

    def test_global_cost_change_is_material_when_unrelated_edge_row_is_not(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            baseline = json.loads(paths["baseline"].read_text())
            intents = json.loads(paths["intents"].read_text())
        selected = {
            "pair": "EUR_USD",
            "side": "LONG",
            "method": "TREND_CONTINUATION",
            "vehicle": "MARKET",
            "trades": 20,
        }
        surface = {
            "contract": "QR_EXACT_VEHICLE_ALLOCATION_SURFACE_V2",
            "parse_status": "VALID",
            "coverage_start_utc": "2026-01-01T00:00:00Z",
            "exact_vehicle_net": [selected],
            "exact_vehicle_take_profit": [],
            "execution_cost": _synthetic_execution_cost_surface(),
        }
        project = (
            market_read_overlay_module
            ._selected_execution_ledger_allocation_surface
        )
        original = project(
            surface,
            baseline=baseline,
            order_intents=intents,
        )

        unrelated = json.loads(json.dumps(surface))
        unrelated["exact_vehicle_net"].append(
            {
                "pair": "GBP_USD",
                "side": "SHORT",
                "method": "RANGE_ROTATION",
                "vehicle": "LIMIT",
                "trades": 999,
            }
        )
        unrelated_projection = project(
            unrelated,
            baseline=baseline,
            order_intents=intents,
        )
        self.assertEqual(
            original["allocation_surface_sha256"],
            unrelated_projection["allocation_surface_sha256"],
        )

        cost_changed = json.loads(json.dumps(surface))
        cost = cost_changed["execution_cost"]
        cost["market_entry"]["adverse_p95_pips"] = 0.2
        cost_material = dict(cost)
        cost_material.pop("execution_cost_surface_sha256", None)
        cost["execution_cost_surface_sha256"] = canonical_json_sha256(
            cost_material
        )
        changed_projection = project(
            cost_changed,
            baseline=baseline,
            order_intents=intents,
        )
        self.assertNotEqual(
            original["allocation_surface_sha256"],
            changed_projection["allocation_surface_sha256"],
        )

    def test_selected_unresolved_reduction_stales_and_blocks_tp_exception(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            _write_overlay(paths)
            _append_vehicle_trade(
                paths["execution_ledger"],
                pair="EUR_USD",
                side="LONG",
                method="TREND_CONTINUATION",
                vehicle="MARKET",
                realized=50.0,
                index=300,
                terminal=False,
            )

            with self.assertRaisesRegex(
                MarketReadOverlayError,
                "MARKET_READ_EVIDENCE_PACKET_STALE",
            ):
                _apply(paths)

            _reprepare(paths)
            lane = json.loads(paths["packet"].read_text())[
                "capital_allocation_board"
            ]["selected_lane"]
            all_exit = lane["capture"]["exact_vehicle_all_exit"]
            self.assertFalse(lane["allocation_eligible"])
            self.assertEqual(
                lane["edge_basis"],
                "EXACT_VEHICLE_ALL_EXIT_CONTRADICTS_TP",
            )
            self.assertEqual(all_exit["unresolved_realized_trades"], 1)
            self.assertEqual(all_exit["unresolved_realized_net_jpy"], 50.0)
            self.assertTrue(all_exit["blocks_tp_exception"])

    def test_stored_allocation_board_and_prediction_body_tampering_is_rejected(self) -> None:
        for mutation, expected_code in (
            ("board", "MARKET_READ_CAPITAL_ALLOCATION_BOARD_STALE"),
            ("resolved_predictions", "MARKET_READ_EVIDENCE_PACKET_BODY_STALE"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                paths = _prepared_paths(Path(tmp))
                _write_overlay(paths)
                packet = json.loads(paths["packet"].read_text())
                if mutation == "board":
                    packet["capital_allocation_board"]["selected_lane"]["capture"][
                        "take_profit_expectancy_jpy"
                    ] = 999_999.0
                else:
                    packet["recent_resolved_predictions"].append(
                        {"prediction_id": "mr2:" + "f" * 64, "verdict": "FORGED"}
                    )
                paths["packet"].write_text(json.dumps(packet))

                with self.assertRaisesRegex(MarketReadOverlayError, expected_code):
                    _apply(paths)

    def test_unknown_or_broad_capture_edge_cannot_receive_allocation(self) -> None:
        for mutation, expected_basis in (
            ("unknown", "UNKNOWN_OR_NON_EXACT_EDGE"),
            ("positive_broad", "UNKNOWN_OR_NON_EXACT_EDGE"),
            ("broad", "UNKNOWN_OR_NON_EXACT_EDGE"),
            ("invalid_scout", "INVALID_PREDICTIVE_SCOUT_CLAIM"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                paths = _prepared_paths(Path(tmp))
                intents = json.loads(paths["intents"].read_text())
                metadata = intents["results"][0]["intent"]["metadata"]
                for key in list(metadata):
                    if key.startswith("capture_take_profit") or key in {
                        "attach_take_profit_on_fill",
                        "tp_execution_mode",
                    }:
                        metadata.pop(key, None)
                metadata.pop("capture_economics_status", None)
                if mutation == "positive_broad":
                    metadata.update(
                        {
                            "capture_economics_status": "POSITIVE_EXPECTANCY",
                            "capture_expectancy_jpy": 100.0,
                        }
                    )
                elif mutation == "broad":
                    metadata.update(
                        {
                            "attach_take_profit_on_fill": True,
                            "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                            "capture_take_profit_scope": "PAIR_SIDE_METHOD",
                            "capture_take_profit_scope_key": (
                                "EUR_USD|LONG|TREND_CONTINUATION|TAKE_PROFIT_ORDER"
                            ),
                            "capture_take_profit_expectancy_jpy": 900.0,
                            "capture_take_profit_trades": 100,
                            "capture_take_profit_losses": 0,
                            "capture_take_profit_avg_win_jpy": 900.0,
                        }
                    )
                elif mutation == "invalid_scout":
                    metadata["predictive_scout"] = True
                    metadata["predictive_scout_source"] = "UNVERIFIED_FIXTURE"
                paths["intents"].write_text(json.dumps(intents))
                _reprepare(paths)
                packet = json.loads(paths["packet"].read_text())
                lane = packet["capital_allocation_board"]["selected_lane"]
                self.assertFalse(lane["allocation_eligible"])
                self.assertFalse(lane["positive_edge_proven"])
                self.assertEqual(lane["edge_basis"], expected_basis)
                _write_overlay(paths)

                with self.assertRaisesRegex(
                    MarketReadOverlayError,
                    "MARKET_READ_CAPITAL_ALLOCATION_EDGE_NOT_PROVEN",
                ):
                    _apply(paths)

    def test_exact_vehicle_all_exit_net_edge_can_receive_allocation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            intents = json.loads(paths["intents"].read_text())
            metadata = intents["results"][0]["intent"]["metadata"]
            for key in list(metadata):
                if key.startswith("capture_take_profit"):
                    metadata.pop(key, None)
            metadata.update(
                {
                    "capture_economics_status": "POSITIVE_EXPECTANCY",
                    "capture_exact_vehicle_net_scope": "PAIR_SIDE_METHOD_VEHICLE",
                    "capture_exact_vehicle_net_scope_key": (
                        "EUR_USD|LONG|TREND_CONTINUATION|MARKET|ALL_AUDITED_EXITS"
                    ),
                    "capture_exact_vehicle_net_vehicle": "MARKET",
                    "capture_exact_vehicle_net_metrics_source": (
                        "data/execution_ledger.db:exact_vehicle_net"
                    ),
                    "capture_exact_vehicle_net_exit_scope": "ALL_AUDITED_EXITS",
                    "capture_exact_vehicle_net_trades": 20,
                    "capture_exact_vehicle_net_wins": 18,
                    "capture_exact_vehicle_net_losses": 2,
                    "capture_exact_vehicle_net_jpy": 1780.0,
                    "capture_exact_vehicle_net_expectancy_jpy": 89.0,
                    "capture_exact_vehicle_net_avg_win_jpy": 100.0,
                    "capture_exact_vehicle_net_avg_loss_jpy": 10.0,
                    "capture_exact_vehicle_net_unresolved_realized_trades": 0,
                    "capture_exact_vehicle_net_unresolved_realized_net_jpy": 0.0,
                    "capture_exact_vehicle_net_unresolved_trade_ids_sha256": (
                        hashlib.sha256(b"[]").hexdigest()
                    ),
                }
            )
            paths["intents"].write_text(json.dumps(intents))
            _write_exact_vehicle_ledger(
                paths["execution_ledger"],
                [100.0] * 18 + [-10.0] * 2,
            )
            _reprepare(paths)

            lane = json.loads(paths["packet"].read_text())["capital_allocation_board"][
                "selected_lane"
            ]
            self.assertTrue(lane["allocation_eligible"])
            self.assertTrue(lane["positive_edge_proven"])
            self.assertEqual(lane["edge_basis"], "EXACT_VEHICLE_ALL_EXIT_NET")
            self.assertGreater(
                lane["capture"]["exact_vehicle_all_exit"][
                    "wilson_stressed_expectancy_jpy"
                ],
                0.0,
            )

            _write_overlay(paths)
            summary = _apply(paths)
            self.assertEqual(summary.action, "TRADE")

    def test_metadata_method_cannot_redirect_typed_market_context_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            intents = json.loads(paths["intents"].read_text())
            intent = intents["results"][0]["intent"]
            self.assertEqual(
                intent["market_context"]["method"],
                "TREND_CONTINUATION",
            )
            intent["metadata"]["method"] = "BREAKOUT_FAILURE"
            paths["intents"].write_text(json.dumps(intents))
            _reprepare(paths)

            packet = json.loads(paths["packet"].read_text())
            lane = packet["capital_allocation_board"]["selected_lane"]
            self.assertFalse(lane["method_scope_consistent"])
            self.assertFalse(lane["positive_edge_proven"])
            self.assertFalse(lane["allocation_eligible"])
            self.assertEqual(lane["allowed_size_multiples"], [])
            self.assertEqual(lane["edge_basis"], "METHOD_SCOPE_MISMATCH")
            self.assertEqual(
                lane["numeric_ceiling"]["reason"],
                "METHOD_SCOPE_MISMATCH",
            )
            self.assertEqual(lane["numeric_ceiling"]["max_multiple"], 0.0)
            self.assertIsNone(
                packet["execution_ledger_allocation_surface"][
                    "selected_scope_key"
                ]
            )

    def test_exact_vehicle_all_exit_net_edge_rejects_weak_or_mismatched_proof(self) -> None:
        for mutation in (
            "thin_sample",
            "wrong_source",
            "negative_stress",
            "zero_avg_loss",
            "missing_avg_loss",
            "negative_avg_loss",
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                paths = _prepared_paths(Path(tmp))
                intents = json.loads(paths["intents"].read_text())
                metadata = intents["results"][0]["intent"]["metadata"]
                for key in list(metadata):
                    if key.startswith("capture_take_profit"):
                        metadata.pop(key, None)
                metadata.update(
                    {
                        "capture_exact_vehicle_net_scope": "PAIR_SIDE_METHOD_VEHICLE",
                        "capture_exact_vehicle_net_scope_key": (
                            "EUR_USD|LONG|TREND_CONTINUATION|MARKET|ALL_AUDITED_EXITS"
                        ),
                        "capture_exact_vehicle_net_vehicle": "MARKET",
                        "capture_exact_vehicle_net_metrics_source": (
                            "data/execution_ledger.db:exact_vehicle_net"
                        ),
                        "capture_exact_vehicle_net_exit_scope": "ALL_AUDITED_EXITS",
                        "capture_exact_vehicle_net_trades": 20,
                        "capture_exact_vehicle_net_wins": 18,
                        "capture_exact_vehicle_net_losses": 2,
                        "capture_exact_vehicle_net_jpy": 1780.0,
                        "capture_exact_vehicle_net_expectancy_jpy": 89.0,
                        "capture_exact_vehicle_net_avg_win_jpy": 100.0,
                        "capture_exact_vehicle_net_avg_loss_jpy": 10.0,
                    }
                )
                if mutation == "thin_sample":
                    metadata["capture_exact_vehicle_net_trades"] = 19
                    metadata["capture_exact_vehicle_net_wins"] = 17
                elif mutation == "wrong_source":
                    metadata["capture_exact_vehicle_net_metrics_source"] = "handwritten"
                elif mutation == "negative_stress":
                    metadata["capture_exact_vehicle_net_avg_loss_jpy"] = 1000.0
                elif mutation == "zero_avg_loss":
                    metadata["capture_exact_vehicle_net_avg_loss_jpy"] = 0.0
                elif mutation == "missing_avg_loss":
                    metadata.pop("capture_exact_vehicle_net_avg_loss_jpy")
                else:
                    metadata["capture_exact_vehicle_net_avg_loss_jpy"] = -10.0
                paths["intents"].write_text(json.dumps(intents))
                _reprepare(paths)

                lane = json.loads(paths["packet"].read_text())[
                    "capital_allocation_board"
                ]["selected_lane"]
                self.assertFalse(lane["allocation_eligible"])
                self.assertFalse(
                    lane["capture"]["exact_vehicle_all_exit"]["proven"]
                )

    def test_allocation_schema_rejects_non_string_rationale_and_near_enum_multiple(self) -> None:
        for mutation in ("bool_rationale", "nan_rationale", "near_multiple"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                paths = _prepared_paths(Path(tmp))
                overlay = _overlay(paths)
                if mutation == "bool_rationale":
                    overlay["capital_allocation"]["rationale"] = True
                elif mutation == "nan_rationale":
                    overlay["capital_allocation"]["rationale"] = float("nan")
                else:
                    overlay["capital_allocation"]["size_multiple"] = 0.5000000000005
                    overlay["capital_allocation"]["selected_units"] = 600
                paths["overlay"].write_text(json.dumps(overlay))

                with self.assertRaisesRegex(
                    MarketReadOverlayError,
                    "MARKET_READ_CAPITAL_ALLOCATION_INVALID|MARKET_READ_CAPITAL_ALLOCATION_MULTIPLE_INVALID",
                ):
                    _apply(paths)

    def test_one_unit_lane_advertises_only_executable_allocation_choice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            intents = json.loads(paths["intents"].read_text())
            intents["results"][0]["intent"]["units"] = 1
            intents["results"][0]["risk_metrics"].update(
                {
                    "jpy_per_pip": 0.01,
                    "risk_jpy": 0.22,
                    "reward_jpy": 0.38,
                }
            )
            paths["intents"].write_text(json.dumps(intents))
            _reprepare(paths)
            packet = json.loads(paths["packet"].read_text())

            self.assertEqual(
                packet["capital_allocation_board"]["selected_lane"][
                    "allowed_size_multiples"
                ],
                [1.0],
            )
            _write_overlay(paths, size_multiple=1.0)
            _apply(paths)
            final = json.loads(paths["output"].read_text())
            self.assertEqual(final["capital_allocation"]["selected_units"], 1)

    def test_veto_requires_schema_v2_and_exact_artifact_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            _write_overlay(paths, disposition="VETO_WAIT")
            _apply(paths)
            final = json.loads(paths["output"].read_text())
            final["decision_provenance"]["schema_version"] = 1
            issues = validate_codex_market_read_provenance(
                action="WAIT",
                market_read=final["market_read_first"],
                provenance=final["decision_provenance"],
                review=final["market_read_review"],
                counterargument=final["market_read_counterargument"],
                change_summary=final["market_read_change_summary"],
                disposition=final["market_read_disposition"],
                veto_reason=final["market_read_veto_reason"],
                vetoed_lane_ids=tuple(final["market_read_vetoed_lane_ids"]),
                capital_allocation=final["capital_allocation"],
                execution_envelope_sha256=canonical_json_sha256(
                    execution_envelope_payload(final)
                ),
                now=NOW,
            )
            self.assertIn("AI_MARKET_READ_PROVENANCE_INVALID", {code for code, _ in issues})
            artifact_issues = revalidate_codex_market_read_artifacts(
                final_payload=final,
                baseline_path=paths["baseline"],
                packet_path=paths["packet"],
                overlay_path=paths["overlay"],
                evidence_sources=_sources(paths),
            )
            self.assertIn(
                "AI_MARKET_READ_ARTIFACT_FINAL_MISMATCH",
                {code for code, _ in artifact_issues},
            )

    def test_trade_allocation_is_bound_to_numeric_lane_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            packet = json.loads(paths["packet"].read_text())
            board = packet["capital_allocation_board"]

            self.assertEqual(board["selected_lane"]["base_units"], 1200)
            self.assertEqual(
                board["selected_lane"]["forecast"]["calibration_name"],
                "directional_forecast_up",
            )
            self.assertEqual(
                board["selected_lane"]["capture"]["take_profit_expectancy_jpy"],
                320.0,
            )
            self.assertTrue(board["selected_lane"]["allocation_eligible"])
            _write_overlay(paths, size_multiple=0.75)

            _apply(paths)

            final = json.loads(paths["output"].read_text())
            allocation = final["capital_allocation"]
            self.assertEqual(allocation["decision"], "ALLOCATE")
            self.assertEqual(allocation["lane_id"], LANE_ID)
            self.assertEqual(allocation["size_multiple"], 0.75)
            self.assertEqual(allocation["selected_units"], 900)
            self.assertEqual(
                final["decision_provenance"]["authorized_units"],
                900,
            )

    def test_numeric_ceiling_fails_closed_on_missing_or_invalid_inputs(self) -> None:
        cases = (
            ("missing_economic_rate", "ECONOMIC_HIT_RATE_MISSING_OR_INVALID"),
            ("bool_economic_rate", "ECONOMIC_HIT_RATE_MISSING_OR_INVALID"),
            ("missing_economic_samples", "ECONOMIC_SAMPLE_FLOOR_NOT_MET"),
            ("bool_economic_samples", "ECONOMIC_SAMPLE_FLOOR_NOT_MET"),
            ("thin_economic_samples", "ECONOMIC_SAMPLE_FLOOR_NOT_MET"),
            ("missing_nav", "BROKER_NAV_MISSING_OR_INVALID"),
            ("bool_nav", "BROKER_NAV_MISSING_OR_INVALID"),
            ("nan_nav", "BROKER_NAV_MISSING_OR_INVALID"),
            ("economic_rate_over_one", "ECONOMIC_HIT_RATE_MISSING_OR_INVALID"),
            ("wrong_calibration", "FORECAST_CALIBRATION_IDENTITY_MISMATCH"),
            ("reward_risk_mismatch", "RISK_REWARD_JPY_GEOMETRY_INCONSISTENT"),
        )
        for mutation, expected_reason in cases:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                paths = _prepared_paths(Path(tmp))
                intents = json.loads(paths["intents"].read_text())
                result = intents["results"][0]
                metadata = result["intent"]["metadata"]
                if mutation == "missing_economic_rate":
                    metadata.pop("forecast_directional_economic_hit_rate")
                elif mutation == "bool_economic_rate":
                    metadata["forecast_directional_economic_hit_rate"] = True
                elif mutation == "missing_economic_samples":
                    metadata.pop("forecast_directional_economic_samples")
                elif mutation == "bool_economic_samples":
                    metadata["forecast_directional_economic_samples"] = True
                elif mutation == "thin_economic_samples":
                    metadata["forecast_directional_economic_samples"] = (
                        CAPITAL_ALLOCATION_FORECAST_MIN_SAMPLES - 1
                    )
                elif mutation in {"missing_nav", "bool_nav", "nan_nav"}:
                    snapshot = json.loads(paths["snapshot"].read_text())
                    if mutation == "missing_nav":
                        snapshot.pop("account")
                    elif mutation == "bool_nav":
                        snapshot["account"]["nav_jpy"] = True
                    else:
                        snapshot["account"]["nav_jpy"] = float("nan")
                    paths["snapshot"].write_text(json.dumps(snapshot))
                elif mutation == "economic_rate_over_one":
                    metadata["forecast_directional_economic_hit_rate"] = 1.01
                elif mutation == "wrong_calibration":
                    metadata["forecast_directional_calibration_name"] = (
                        "directional_forecast_down"
                    )
                else:
                    result["risk_metrics"]["reward_risk"] = 2.0
                paths["intents"].write_text(json.dumps(intents))
                _reprepare(paths)

                lane = json.loads(paths["packet"].read_text())[
                    "capital_allocation_board"
                ]["selected_lane"]
                self.assertTrue(lane["positive_edge_proven"])
                self.assertFalse(lane["allocation_eligible"])
                self.assertEqual(lane["allowed_size_multiples"], [])
                self.assertEqual(lane["numeric_ceiling"]["reason"], expected_reason)

    def test_headline_precision_cannot_hide_negative_economic_ev(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            intents = json.loads(paths["intents"].read_text())
            metadata = intents["results"][0]["intent"]["metadata"]
            metadata.update(
                {
                    "forecast_directional_hit_rate": 1.0,
                    "forecast_directional_samples": 100,
                    "forecast_directional_economic_hit_rate": 0.30,
                    "forecast_directional_economic_samples": 100,
                    "forecast_directional_timeout_rate": 0.70,
                }
            )
            paths["intents"].write_text(json.dumps(intents))
            _reprepare(paths)

            lane = json.loads(paths["packet"].read_text())[
                "capital_allocation_board"
            ]["selected_lane"]
            ceiling = lane["numeric_ceiling"]
            self.assertEqual(
                ceiling["inputs"]["headline_hit_rate_context_only"],
                1.0,
            )
            self.assertLess(
                ceiling["probability"]["economic_wilson95_lower"],
                0.30,
            )
            self.assertFalse(ceiling["ev_lower"]["positive"])
            self.assertEqual(ceiling["reason"], "CONSERVATIVE_EV_NOT_POSITIVE")
            self.assertEqual(lane["allowed_size_multiples"], [])
            self.assertFalse(lane["allocation_eligible"])

    def test_numeric_ceiling_rejects_direction_or_forecast_rail_mismatch(self) -> None:
        cases = (
            ("direction", "FORECAST_DIRECTION_SIDE_MISMATCH"),
            ("target", "FORECAST_RAIL_DOES_NOT_CONSERVATIVELY_CONTAIN_ORDER"),
            ("invalidation", "FORECAST_RAIL_DOES_NOT_CONSERVATIVELY_CONTAIN_ORDER"),
        )
        for mutation, expected_reason in cases:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                paths = _prepared_paths(Path(tmp))
                intents = json.loads(paths["intents"].read_text())
                metadata = intents["results"][0]["intent"]["metadata"]
                if mutation == "direction":
                    metadata["forecast_direction"] = "DOWN"
                elif mutation == "target":
                    metadata["forecast_target_price"] = 1.1030
                else:
                    metadata["forecast_invalidation_price"] = 1.0970
                paths["intents"].write_text(json.dumps(intents))
                _reprepare(paths)

                lane = json.loads(paths["packet"].read_text())[
                    "capital_allocation_board"
                ]["selected_lane"]
                self.assertEqual(lane["allowed_size_multiples"], [])
                self.assertEqual(lane["numeric_ceiling"]["reason"], expected_reason)

    def test_numeric_ceiling_rejects_forecast_current_outside_rails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            intents = json.loads(paths["intents"].read_text())
            metadata = intents["results"][0]["intent"]["metadata"]
            # Current broker mid remains 1.1001, but the half-spread-adjusted
            # invalidation is now above it, so the forecast rail is already
            # invalid before this order can be sized.
            metadata["forecast_invalidation_price"] = 1.1003
            paths["intents"].write_text(json.dumps(intents))
            _reprepare(paths)

            lane = json.loads(paths["packet"].read_text())[
                "capital_allocation_board"
            ]["selected_lane"]
            ceiling = lane["numeric_ceiling"]
            self.assertTrue(
                ceiling["geometry"]["forecast_current_matches_broker_mid"]
            )
            self.assertFalse(
                ceiling["geometry"][
                    "forecast_outcome_conservatively_contains_order"
                ]
            )
            self.assertEqual(
                ceiling["reason"],
                "FORECAST_RAIL_DOES_NOT_CONSERVATIVELY_CONTAIN_ORDER",
            )
            self.assertEqual(lane["allowed_size_multiples"], [])

    def test_numeric_ceiling_rejects_half_spread_and_broker_metric_tampering(self) -> None:
        cases = (
            ("half_spread_boundary", "FORECAST_RAIL_DOES_NOT_CONSERVATIVELY_CONTAIN_ORDER"),
            ("current_mid", "FORECAST_CURRENT_BROKER_MID_MISMATCH"),
            ("spread", "BROKER_SPREAD_RISK_METRICS_MISMATCH"),
            ("reward_pips", "RISK_REWARD_JPY_GEOMETRY_INCONSISTENT"),
            ("scaled_jpy_metrics", "RISK_REWARD_JPY_GEOMETRY_INCONSISTENT"),
        )
        for mutation, expected_reason in cases:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                paths = _prepared_paths(Path(tmp))
                intents = json.loads(paths["intents"].read_text())
                result = intents["results"][0]
                intent = result["intent"]
                metadata = intent["metadata"]
                risk_metrics = result["risk_metrics"]
                if mutation == "half_spread_boundary":
                    # Raw SL<=invalidation passes, but the bid-side
                    # invalidation after half-spread does not protect the SL.
                    metadata["forecast_invalidation_price"] = (
                        intent["sl"] + 0.00005
                    )
                elif mutation == "current_mid":
                    metadata["forecast_current_price"] = 1.10011
                elif mutation == "spread":
                    risk_metrics["spread_pips"] = 2.1
                elif mutation == "reward_pips":
                    risk_metrics["reward_pips"] = 39.0
                else:
                    risk_metrics["jpy_per_pip"] *= 2.0
                    risk_metrics["risk_jpy"] *= 2.0
                    risk_metrics["reward_jpy"] *= 2.0
                paths["intents"].write_text(json.dumps(intents))
                _reprepare(paths)

                lane = json.loads(paths["packet"].read_text())[
                    "capital_allocation_board"
                ]["selected_lane"]
                self.assertEqual(lane["numeric_ceiling"]["reason"], expected_reason)
                self.assertEqual(lane["allowed_size_multiples"], [])

    def test_pending_entry_is_unbound_and_mismatch_is_explicit(self) -> None:
        for mutation, expected_reason in (
            ("matched", "FORECAST_ECONOMIC_PROBABILITY_ENTRY_VEHICLE_UNBOUND"),
            ("mismatched", "ORDER_ENTRY_RISK_METRICS_BINDING_INVALID"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                paths = _prepared_paths(Path(tmp))
                intents = json.loads(paths["intents"].read_text())
                intent = intents["results"][0]["intent"]
                intent["order_type"] = "STOP"
                intent["entry"] = (
                    intents["results"][0]["risk_metrics"]["entry_price"]
                    if mutation == "matched"
                    else 1.1003
                )
                paths["intents"].write_text(json.dumps(intents))
                _reprepare(paths)

                lane = json.loads(paths["packet"].read_text())[
                    "capital_allocation_board"
                ]["selected_lane"]
                ceiling = lane["numeric_ceiling"]
                self.assertEqual(ceiling["reason"], expected_reason)
                self.assertEqual(lane["allowed_size_multiples"], [])
                if mutation == "matched":
                    self.assertTrue(ceiling["geometry"]["entry_binding_passed"])
                else:
                    self.assertFalse(ceiling["geometry"]["entry_binding_passed"])

    def test_market_entry_must_equal_fresh_broker_executable_side(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            lane = json.loads(paths["packet"].read_text())[
                "capital_allocation_board"
            ]["selected_lane"]
            ceiling = lane["numeric_ceiling"]
            self.assertEqual(
                ceiling["geometry"]["entry_binding_basis"],
                "MARKET_RISK_ENTRY_EQUALS_FRESH_BROKER_ASK_OR_BID",
            )
            self.assertTrue(ceiling["geometry"]["entry_binding_passed"])

            intents = json.loads(paths["intents"].read_text())
            intents["results"][0]["risk_metrics"]["entry_price"] = 1.1003
            paths["intents"].write_text(json.dumps(intents))
            _reprepare(paths)
            lane = json.loads(paths["packet"].read_text())[
                "capital_allocation_board"
            ]["selected_lane"]
            self.assertEqual(
                lane["numeric_ceiling"]["reason"],
                "ORDER_ENTRY_RISK_METRICS_BINDING_INVALID",
            )
            self.assertEqual(lane["allowed_size_multiples"], [])

    def test_gross_break_even_ev_is_net_negative_after_cost_floor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            intents = json.loads(paths["intents"].read_text())
            result = intents["results"][0]
            intent = result["intent"]
            risk_metrics = result["risk_metrics"]
            metadata = intent["metadata"]
            p_lower = hit_rate_wilson_lower(
                metadata["forecast_directional_economic_hit_rate"],
                metadata["forecast_directional_economic_samples"],
            )
            self.assertIsNotNone(p_lower)
            assert p_lower is not None
            break_even_rr = (1.0 - p_lower) / p_lower
            loss_pips = risk_metrics["loss_pips"]
            reward_pips = loss_pips * break_even_rr
            intent["tp"] = risk_metrics["entry_price"] + reward_pips / 10_000.0
            risk_metrics["reward_pips"] = reward_pips
            risk_metrics["reward_jpy"] = reward_pips * risk_metrics["jpy_per_pip"]
            risk_metrics["reward_risk"] = break_even_rr
            paths["intents"].write_text(json.dumps(intents))
            _reprepare(paths)

            lane = json.loads(paths["packet"].read_text())[
                "capital_allocation_board"
            ]["selected_lane"]
            ceiling = lane["numeric_ceiling"]
            gross_ev = (
                p_lower * risk_metrics["reward_jpy"]
                - (1.0 - p_lower) * risk_metrics["risk_jpy"]
            )
            self.assertAlmostEqual(
                gross_ev,
                0.0,
                places=8,
            )
            self.assertGreater(
                ceiling["ev_lower"]["additional_cost_jpy"],
                0.0,
            )
            self.assertAlmostEqual(
                ceiling["ev_lower"]["value_jpy_snapshot"],
                -ceiling["ev_lower"]["additional_cost_jpy"],
                places=8,
            )
            self.assertFalse(ceiling["ev_lower"]["positive"])
            self.assertEqual(ceiling["reason"], "CONSERVATIVE_EV_NOT_POSITIVE")
            self.assertEqual(lane["allowed_size_multiples"], [])

    def test_short_numeric_ceiling_uses_symmetric_forecast_rail(self) -> None:
        intent = {
            "pair": "EUR_USD",
            "side": "SHORT",
            "order_type": "MARKET",
            "units": 1000,
            "tp": 1.0960,
            "sl": 1.1030,
            "market_context": {"method": "TREND_CONTINUATION"},
        }
        metadata = {
            "forecast_direction": "DOWN",
            "forecast_directional_calibration_name": "directional_forecast_down",
            "forecast_current_price": 1.1001,
            "forecast_target_price": 1.0950,
            "forecast_invalidation_price": 1.1020,
            "forecast_directional_economic_hit_rate": 0.70,
            "forecast_directional_economic_samples": 100,
            "forecast_directional_hit_rate": 0.75,
            "forecast_directional_samples": 100,
            "forecast_directional_timeout_rate": 0.10,
        }
        risk_metrics = {
            "entry_price": 1.1000,
            "risk_jpy": 300.0,
            "reward_jpy": 400.0,
            "loss_pips": 30.0,
            "reward_pips": 40.0,
            "jpy_per_pip": 10.0,
            "reward_risk": 4.0 / 3.0,
            "spread_pips": 2.0,
        }
        evidence, max_multiple = (
            market_read_overlay_module._capital_allocation_numeric_ceiling(
                intent=intent,
                metadata=metadata,
                risk_metrics=risk_metrics,
                account_nav_jpy=100_000.0,
                broker_bid=1.1000,
                broker_ask=1.1002,
                broker_quote_to_jpy=100.0,
                predictive_scout=False,
                hedge=False,
                execution_cost_floor=_synthetic_execution_cost_floor(
                    scope_key="EUR_USD|SHORT|TREND_CONTINUATION|MARKET"
                ),
            )
        )
        self.assertTrue(evidence["geometry"]["passed"])
        self.assertEqual(max_multiple, 1.0)
        self.assertEqual(
            evidence["ev_lower"]["additional_cost_jpy"],
            20.0,
        )
        self.assertEqual(evidence["ev_lower"]["net_risk_jpy_snapshot"], 320.0)
        self.assertEqual(evidence["ev_lower"]["net_reward_jpy_snapshot"], 380.0)
        p_lower = evidence["probability"]["economic_wilson95_lower"]
        self.assertAlmostEqual(
            evidence["ev_lower"]["value_jpy_snapshot"],
            p_lower * 400.0 - (1.0 - p_lower) * 300.0 - 20.0,
            places=8,
        )

        wrong_scope_evidence, wrong_scope_multiple = (
            market_read_overlay_module._capital_allocation_numeric_ceiling(
                intent=intent,
                metadata=metadata,
                risk_metrics=risk_metrics,
                account_nav_jpy=100_000.0,
                broker_bid=1.1000,
                broker_ask=1.1002,
                broker_quote_to_jpy=100.0,
                predictive_scout=False,
                hedge=False,
                execution_cost_floor=_synthetic_execution_cost_floor(
                    scope_key="EUR_USD|LONG|UNKNOWN|MARKET"
                ),
            )
        )
        self.assertEqual(wrong_scope_multiple, 0.0)
        self.assertEqual(
            wrong_scope_evidence["reason"],
            "NET_EXECUTION_COST_FLOOR_MISSING_INVALID_OR_STALE",
        )

        metadata["forecast_target_price"] = 1.0970
        evidence, max_multiple = (
            market_read_overlay_module._capital_allocation_numeric_ceiling(
                intent=intent,
                metadata=metadata,
                risk_metrics=risk_metrics,
                account_nav_jpy=100_000.0,
                broker_bid=1.1000,
                broker_ask=1.1002,
                broker_quote_to_jpy=100.0,
                predictive_scout=False,
                hedge=False,
                execution_cost_floor=_synthetic_execution_cost_floor(
                    scope_key="EUR_USD|SHORT|TREND_CONTINUATION|MARKET"
                ),
            )
        )
        self.assertFalse(evidence["geometry"]["passed"])
        self.assertEqual(max_multiple, 0.0)

    def test_quarter_kelly_nav_ratio_can_cap_lane_at_half_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            snapshot = json.loads(paths["snapshot"].read_text())
            snapshot["account"]["nav_jpy"] = 4_000.0
            paths["snapshot"].write_text(json.dumps(snapshot))
            _reprepare(paths)

            lane = json.loads(paths["packet"].read_text())[
                "capital_allocation_board"
            ]["selected_lane"]
            ceiling = lane["numeric_ceiling"]
            self.assertTrue(lane["allocation_eligible"])
            self.assertEqual(lane["allowed_size_multiples"], [0.5])
            self.assertEqual(ceiling["kelly"]["decision_basis"], "NAV_PERCENT_RATIO")
            self.assertGreater(ceiling["max_multiple"], 0.5)
            self.assertLess(ceiling["max_multiple"], 0.75)
            self.assertGreater(
                ceiling["kelly"]["base_risk_nav_pct"],
                ceiling["kelly"]["quarter_kelly_risk_nav_pct"],
            )

            _write_overlay(paths, size_multiple=0.75)
            with self.assertRaisesRegex(
                MarketReadOverlayError,
                "MARKET_READ_CAPITAL_ALLOCATION_MULTIPLE_INVALID",
            ):
                _apply(paths)
            _write_overlay(paths, size_multiple=0.5)
            self.assertEqual(_apply(paths).action, "TRADE")

    def test_quarter_kelly_cap_below_half_advertises_no_trade_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            snapshot = json.loads(paths["snapshot"].read_text())
            snapshot["account"]["nav_jpy"] = 2_000.0
            paths["snapshot"].write_text(json.dumps(snapshot))
            _reprepare(paths)

            lane = json.loads(paths["packet"].read_text())[
                "capital_allocation_board"
            ]["selected_lane"]
            ceiling = lane["numeric_ceiling"]
            self.assertGreater(ceiling["max_multiple"], 0.0)
            self.assertLess(ceiling["max_multiple"], 0.5)
            self.assertEqual(
                ceiling["reason"],
                "QUARTER_KELLY_CAP_BELOW_MINIMUM_MULTIPLE",
            )
            self.assertEqual(lane["allowed_size_multiples"], [])
            self.assertFalse(lane["allocation_eligible"])

    def test_strong_numeric_ceiling_allows_full_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            lane = json.loads(paths["packet"].read_text())[
                "capital_allocation_board"
            ]["selected_lane"]
            ceiling = lane["numeric_ceiling"]
            self.assertEqual(
                json.loads(paths["packet"].read_text())[
                    "capital_allocation_board"
                ]["schema_version"],
                2,
            )
            self.assertEqual(lane["allowed_size_multiples"], [0.5, 0.75, 1.0])
            self.assertEqual(ceiling["max_multiple"], 1.0)
            self.assertTrue(ceiling["ev_lower"]["positive"])
            self.assertGreater(
                ceiling["kelly"]["quarter_kelly_risk_nav_pct"],
                ceiling["kelly"]["base_risk_nav_pct"],
            )
            _write_overlay(paths, size_multiple=1.0)
            self.assertEqual(_apply(paths).action, "TRADE")

    def test_hedge_and_valid_predictive_scout_keep_prebounded_full_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            intents = json.loads(paths["intents"].read_text())
            metadata = intents["results"][0]["intent"]["metadata"]
            metadata["position_intent"] = "HEDGE"
            for key in (
                "forecast_directional_economic_hit_rate",
                "forecast_directional_economic_samples",
                "forecast_target_price",
                "forecast_invalidation_price",
            ):
                metadata.pop(key, None)
            paths["intents"].write_text(json.dumps(intents))
            snapshot = json.loads(paths["snapshot"].read_text())
            snapshot.pop("account")
            paths["snapshot"].write_text(json.dumps(snapshot))
            _reprepare(paths)
            lane = json.loads(paths["packet"].read_text())[
                "capital_allocation_board"
            ]["selected_lane"]
            self.assertEqual(lane["allowed_size_multiples"], [1.0])
            self.assertTrue(lane["allocation_eligible"])
            self.assertEqual(
                lane["numeric_ceiling"]["reason"],
                "HEDGE_RISK_REDUCTION_PREBOUNDED_CONTRACT",
            )

        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            intents = json.loads(paths["intents"].read_text())
            intent = intents["results"][0]["intent"]
            metadata = intent["metadata"]
            rule_name = (
                "USD_CAD_DOWN_H31_60m_C0p50_0p65_FADE_TO_UP_S5_"
                "BIDASK_CONTRARIAN_HARVEST_TP10_SL7"
            )
            rule = canonical_bidask_replay_precision_rule(rule_name)
            self.assertIsNotNone(rule)
            assert rule is not None
            intent["order_type"] = "LIMIT"
            intent["market_context"]["method"] = "BREAKOUT_FAILURE"
            metadata.update(
                {
                    "predictive_scout": True,
                    "predictive_scout_source": "BIDASK_REPLAY_PRECISION",
                    "predictive_scout_hypothesis": (
                        "REPRODUCIBLE_FORECAST_FAILURE_CONTRARIAN"
                    ),
                    "predictive_scout_vehicle_proof_status": (
                        "UNPROVEN_PASSIVE_LIMIT"
                    ),
                    "predictive_scout_rule_is_vehicle_proof": False,
                    "predictive_scout_rule_digest": (
                        bidask_replay_precision_rule_digest(rule)
                    ),
                    "bidask_replay_precision_seed_rule": rule,
                    "desk": "failure_trader",
                    "campaign_role": "BIDASK_REPLAY_CONTRARIAN_SCOUT",
                    "attach_take_profit_on_fill": True,
                    "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                }
            )
            for key in (
                "forecast_directional_economic_hit_rate",
                "forecast_directional_economic_samples",
                "forecast_target_price",
                "forecast_invalidation_price",
            ):
                metadata.pop(key, None)
            paths["intents"].write_text(json.dumps(intents))
            snapshot = json.loads(paths["snapshot"].read_text())
            snapshot.pop("account")
            paths["snapshot"].write_text(json.dumps(snapshot))
            _reprepare(paths)
            lane = json.loads(paths["packet"].read_text())[
                "capital_allocation_board"
            ]["selected_lane"]
            self.assertTrue(lane["predictive_scout"])
            self.assertEqual(lane["allowed_size_multiples"], [1.0])
            self.assertTrue(lane["allocation_eligible"])
            self.assertEqual(
                lane["numeric_ceiling"]["reason"],
                "PREDICTIVE_SCOUT_PREBOUNDED_CONTRACT",
            )

    def test_trade_allocation_rejects_expansion_units_and_stale_board(self) -> None:
        for mutation, expected_code in (
            ("expansion", "MARKET_READ_CAPITAL_ALLOCATION_MULTIPLE_INVALID"),
            ("units", "MARKET_READ_CAPITAL_ALLOCATION_UNITS_MISMATCH"),
            ("board", "MARKET_READ_CAPITAL_ALLOCATION_BOARD_STALE"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                paths = _prepared_paths(Path(tmp))
                overlay = _overlay(paths)
                allocation = overlay["capital_allocation"]
                if mutation == "expansion":
                    allocation["size_multiple"] = 1.25
                    allocation["selected_units"] = 1500
                elif mutation == "units":
                    allocation["selected_units"] += 1
                else:
                    allocation["allocation_board_sha256"] = "0" * 64
                paths["overlay"].write_text(json.dumps(overlay))

                with self.assertRaisesRegex(MarketReadOverlayError, expected_code):
                    _apply(paths)

    def test_nontrade_allocation_cannot_smuggle_units(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            overlay = _overlay(paths, disposition="VETO_WAIT")
            overlay["capital_allocation"].update(
                {
                    "decision": "ALLOCATE",
                    "lane_id": LANE_ID,
                    "size_multiple": 0.5,
                    "selected_units": 600,
                }
            )
            paths["overlay"].write_text(json.dumps(overlay))

            with self.assertRaisesRegex(
                MarketReadOverlayError,
                "MARKET_READ_CAPITAL_ALLOCATION_NONTRADE_REQUIRED",
            ):
                _apply(paths)

    def test_explicit_negative_lane_edge_requires_zero_allocation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            intents = json.loads(paths["intents"].read_text())
            metadata = intents["results"][0]["intent"]["metadata"]
            metadata["capture_take_profit_expectancy_jpy"] = -25.0
            metadata["capture_take_profit_wins"] = 2
            metadata["capture_take_profit_losses"] = 2
            paths["intents"].write_text(json.dumps(intents))
            baseline = json.loads(paths["baseline"].read_text())
            baseline.pop("decision_provenance", None)
            paths["baseline"].write_text(json.dumps(baseline))
            prepare_market_read_baseline(
                baseline_path=paths["baseline"],
                packet_path=paths["packet"],
                evidence_sources=_sources(paths),
                now=NOW,
            )
            _write_overlay(paths)

            with self.assertRaisesRegex(
                MarketReadOverlayError,
                "MARKET_READ_CAPITAL_ALLOCATION_EDGE_NOT_PROVEN",
            ):
                _apply(paths)

    def test_accept_preserves_the_deterministic_execution_envelope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            baseline = json.loads(paths["baseline"].read_text())
            _write_overlay(paths, disposition="ACCEPT_BASELINE")

            summary = _apply(paths)

            final = json.loads(paths["output"].read_text())
            self.assertEqual(summary.action, "TRADE")
            self.assertEqual(final["action"], baseline["action"])
            self.assertEqual(final["selected_lane_id"], baseline["selected_lane_id"])
            self.assertEqual(final["selected_lane_ids"], baseline["selected_lane_ids"])
            self.assertEqual(final["cancel_order_ids"], baseline["cancel_order_ids"])
            self.assertEqual(final["risk_notes"], baseline["risk_notes"])
            provenance = final["decision_provenance"]
            self.assertEqual(provenance["author_kind"], CODEX_MARKET_READ_AUTHOR)
            self.assertEqual(provenance["baseline_action"], "TRADE")
            self.assertEqual(provenance["final_action"], "TRADE")
            self.assertFalse(provenance["action_downgrade_only"])
            self.assertTrue(provenance["execution_fields_preserved"])
            self.assertTrue(provenance["risk_envelope_not_expanded"])
            self.assertFalse(provenance["live_permission_granted"])

    def test_veto_can_only_downgrade_trade_and_clears_selected_lanes(self) -> None:
        for disposition, expected_action in (
            ("VETO_WAIT", "WAIT"),
            ("VETO_REQUEST_EVIDENCE", "REQUEST_EVIDENCE"),
        ):
            with self.subTest(disposition=disposition), tempfile.TemporaryDirectory() as tmp:
                paths = _prepared_paths(Path(tmp))
                baseline = json.loads(paths["baseline"].read_text())
                _write_overlay(paths, disposition=disposition)

                _apply(paths)

                final = json.loads(paths["output"].read_text())
                self.assertEqual(final["action"], expected_action)
                self.assertIsNone(final["selected_lane_id"])
                self.assertEqual(final["selected_lane_ids"], [])
                self.assertEqual(final["market_read_vetoed_lane_ids"], [LANE_ID])
                self.assertEqual(final["cancel_order_ids"], baseline["cancel_order_ids"])
                self.assertEqual(final["risk_notes"], baseline["risk_notes"])
                provenance = final["decision_provenance"]
                self.assertEqual(provenance["baseline_action"], "TRADE")
                self.assertEqual(provenance["final_action"], expected_action)
                self.assertTrue(provenance["action_downgrade_only"])

    def test_accept_rejects_crafted_multi_lane_trade_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            second_lane = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:STOP"
            paths = _prepared_paths(
                Path(tmp),
                baseline=_baseline(lane_ids=[LANE_ID, second_lane]),
            )
            _write_overlay(paths, disposition="ACCEPT_BASELINE")

            with self.assertRaisesRegex(
                MarketReadOverlayError,
                "MARKET_READ_BASELINE_SINGLE_LANE_REQUIRED",
            ):
                _apply(paths)

            self.assertFalse(paths["output"].exists())

    def test_multi_lane_trade_baseline_can_still_publish_nontrade_veto(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            second_lane = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:STOP"
            paths = _prepared_paths(
                Path(tmp),
                baseline=_baseline(lane_ids=[LANE_ID, second_lane]),
            )
            _write_overlay(paths, disposition="VETO_WAIT")

            summary = _apply(paths)

            final = json.loads(paths["output"].read_text())
            self.assertEqual(summary.action, "WAIT")
            self.assertEqual(
                final["market_read_vetoed_lane_ids"],
                [LANE_ID, second_lane],
            )

    def test_nontrade_baseline_cannot_be_changed_to_any_other_disposition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = _baseline(action="WAIT", lane_ids=[])
            paths = _prepared_paths(root, baseline=baseline)
            _write_overlay(paths, disposition="VETO_WAIT")

            with self.assertRaisesRegex(MarketReadOverlayError, "MARKET_READ_NONTRADE_UPGRADE_FORBIDDEN"):
                _apply(paths)

            self.assertFalse(paths["output"].exists())

    def test_accept_close_baseline_preserves_exact_close_trade_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = _baseline(action="CLOSE", lane_ids=[])
            baseline["cancel_order_ids"] = []
            baseline["close_trade_ids"] = ["555"]
            baseline["method"] = "POSITION_MANAGEMENT"
            paths = _prepared_paths(root, baseline=baseline)
            _write_overlay(paths, disposition="ACCEPT_BASELINE")

            summary = _apply(paths)

            final = json.loads(paths["output"].read_text())
            self.assertEqual(summary.action, "CLOSE")
            self.assertEqual(final["action"], "CLOSE")
            self.assertEqual(final["close_trade_ids"], ["555"])
            self.assertEqual(final["selected_lane_ids"], [])
            self.assertEqual(final["cancel_order_ids"], [])
            self.assertEqual(
                final["capital_allocation"],
                {
                    "decision": "NO_TRADE",
                    "lane_id": None,
                    "size_multiple": 0.0,
                    "selected_units": 0,
                    "allocation_board_sha256": final["capital_allocation"][
                        "allocation_board_sha256"
                    ],
                    "rationale": "No fresh entry capital is authorized for this receipt.",
                },
            )
            self.assertEqual(
                final["decision_provenance"]["baseline_action"],
                "CLOSE",
            )
            self.assertTrue(
                final["decision_provenance"]["execution_fields_preserved"]
            )

    def test_close_baseline_rejects_allocate_capital_smuggling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = _baseline(action="CLOSE", lane_ids=[])
            baseline["cancel_order_ids"] = []
            baseline["close_trade_ids"] = ["555"]
            baseline["method"] = "POSITION_MANAGEMENT"
            paths = _prepared_paths(root, baseline=baseline)
            overlay = _overlay(paths, disposition="ACCEPT_BASELINE")
            overlay["capital_allocation"].update(
                {
                    "decision": "ALLOCATE",
                    "lane_id": LANE_ID,
                    "size_multiple": 1.0,
                    "selected_units": 1,
                }
            )
            paths["overlay"].write_text(json.dumps(overlay))

            with self.assertRaisesRegex(
                MarketReadOverlayError,
                "MARKET_READ_CAPITAL_ALLOCATION_NONTRADE_REQUIRED",
            ):
                _apply(paths)

    def test_accept_rejects_multi_target_close_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = _baseline(action="CLOSE", lane_ids=[])
            baseline["cancel_order_ids"] = []
            baseline["close_trade_ids"] = ["555", "556"]
            baseline["method"] = "POSITION_MANAGEMENT"
            paths = _prepared_paths(root, baseline=baseline)
            _write_overlay(paths, disposition="ACCEPT_BASELINE")

            with self.assertRaisesRegex(
                MarketReadOverlayError,
                "MARKET_READ_BASELINE_SINGLE_CLOSE_REQUIRED",
            ):
                _apply(paths)

            self.assertFalse(paths["output"].exists())

    def test_accept_rejects_close_baseline_with_entry_or_cancel_scope(self) -> None:
        for field, value in (
            ("selected_lane_id", LANE_ID),
            ("selected_lane_ids", [LANE_ID]),
            ("cancel_order_ids", ["123"]),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                baseline = _baseline(action="CLOSE", lane_ids=[])
                baseline["cancel_order_ids"] = []
                baseline["close_trade_ids"] = ["555"]
                baseline["method"] = "POSITION_MANAGEMENT"
                baseline[field] = value
                paths = _prepared_paths(root, baseline=baseline)
                _write_overlay(paths, disposition="ACCEPT_BASELINE")

                with self.assertRaisesRegex(
                    MarketReadOverlayError,
                    "MARKET_READ_BASELINE_CLOSE_SCOPE_INVALID",
                ):
                    _apply(paths)

                self.assertFalse(paths["output"].exists())

    def test_overlay_rejects_any_execution_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            overlay = _overlay(paths)
            overlay["action"] = "TRADE"
            paths["overlay"].write_text(json.dumps(overlay))

            with self.assertRaisesRegex(MarketReadOverlayError, "MARKET_READ_OVERLAY_SCHEMA_INVALID"):
                _apply(paths)

    def test_baseline_or_evidence_mutation_rejects_stale_overlay(self) -> None:
        for mutation, expected_code in (
            ("baseline", "MARKET_READ_BASELINE_SHA_STALE"),
            ("evidence", "MARKET_READ_EVIDENCE_PACKET_STALE"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                paths = _prepared_paths(Path(tmp))
                _write_overlay(paths)
                if mutation == "baseline":
                    baseline = json.loads(paths["baseline"].read_text())
                    baseline["risk_notes"] = ["mutated after AI review"]
                    paths["baseline"].write_text(json.dumps(baseline))
                else:
                    snapshot = json.loads(paths["snapshot"].read_text())
                    snapshot["quotes"]["EUR_USD"]["ask"] = 1.1010
                    paths["snapshot"].write_text(json.dumps(snapshot))

                with self.assertRaisesRegex(MarketReadOverlayError, expected_code):
                    _apply(paths)

    def test_equivalent_relative_and_absolute_source_paths_share_one_evidence_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            relative_snapshot = Path(os.path.relpath(paths["snapshot"], Path.cwd()))
            relative_sources = {**_sources(paths), "broker_snapshot": relative_snapshot}
            prepare_market_read_baseline(
                baseline_path=paths["baseline"],
                packet_path=paths["packet"],
                evidence_sources=relative_sources,
                now=NOW,
            )
            _write_overlay(paths)

            summary = _apply(paths)

            packet = json.loads(paths["packet"].read_text())
            self.assertEqual(summary.action, "TRADE")
            self.assertEqual(
                packet["source_paths"]["broker_snapshot"],
                str(paths["snapshot"].resolve()),
            )

    def test_identical_bytes_at_a_different_source_path_do_not_share_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            _write_overlay(paths)
            copied_snapshot = paths["snapshot"].with_name("copied_snapshot.json")
            copied_snapshot.write_bytes(paths["snapshot"].read_bytes())
            relocated_sources = {**_sources(paths), "broker_snapshot": copied_snapshot}

            with self.assertRaisesRegex(
                MarketReadOverlayError,
                "MARKET_READ_EVIDENCE_PACKET_STALE",
            ):
                apply_codex_market_read_overlay(
                    baseline_path=paths["baseline"],
                    packet_path=paths["packet"],
                    overlay_path=paths["overlay"],
                    output_path=paths["output"],
                    evidence_sources=relocated_sources,
                    now=NOW,
                )

    def test_watchdog_observation_clock_rewrite_does_not_stale_ai_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            _write_overlay(paths)
            watchdog = json.loads(paths["watchdog"].read_text())
            watchdog["generated_at_utc"] = (NOW + timedelta(minutes=1)).isoformat()
            watchdog["minutes_since_last_run"] = 61.0
            watchdog["last_trader_run_at"] = NOW.isoformat()
            watchdog["last_trader_run_source"] = "decision_response.generated_at_utc"
            watchdog["last_decision_artifact_at"] = NOW.isoformat()
            watchdog["guardian_receipt"].update(
                {
                    "action": "NO_ACTION",
                    "expired_before_trader_run": True,
                    "next_run_window_missed": False,
                    "receipt_after_last_trader_run": True,
                    "receipt_lifecycle": "EXPIRED",
                    "receipt_status": "ACCEPTED",
                    "terminal_lifecycle": True,
                    "will_expire_before_next_run": False,
                    "receipt_summaries": [
                        {
                            "action": "REDUCE",
                            "active": False,
                            "canonical_present": False,
                            "emergency_or_margin_risk": False,
                            "event_id": "historical-event",
                            "high_urgency_action": True,
                            "identity": "event|historical-event|REDUCE",
                            "receipt_lifecycle": "SUPERSEDED",
                        }
                    ],
                }
            )
            watchdog["weekend_pause"]["now_jst"] = "2026-07-11T12:01:00+09:00"
            watchdog["automation_config"]["weekend_pause"]["now_jst"] = (
                "2026-07-11T12:01:00+09:00"
            )
            watchdog["codex_logs"]["queried_at_utc"] = (
                NOW + timedelta(minutes=1)
            ).isoformat()
            watchdog["guardian_receipt"]["review_excerpt"] = (
                "Generated at a later observation clock"
            )
            paths["watchdog"].write_text(json.dumps(watchdog))

            summary = _apply(paths)

            self.assertEqual(summary.action, "TRADE")

    def test_watchdog_material_safety_change_stales_ai_review(self) -> None:
        for mutation in ("status", "receipt_identity"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                paths = _prepared_paths(Path(tmp))
                _write_overlay(paths)
                watchdog = json.loads(paths["watchdog"].read_text())
                if mutation == "status":
                    watchdog["status"] = "BROKEN"
                    watchdog["runtime_status"] = "BROKEN"
                    watchdog["issues"] = ["MISSED_TRADER_RUN"]
                else:
                    watchdog["guardian_receipt"]["receipt_summaries"] = [
                        {
                            "action": "REDUCE",
                            "active": True,
                            "canonical_present": True,
                            "event_id": "new-event",
                            "identity": "event|new-event|REDUCE",
                            "high_urgency_action": True,
                            "receipt_lifecycle": "ACTIVE",
                            "receipt_status": "ACCEPTED",
                        }
                    ]
                paths["watchdog"].write_text(json.dumps(watchdog))

                with self.assertRaisesRegex(
                    MarketReadOverlayError,
                    "MARKET_READ_EVIDENCE_PACKET_STALE",
                ):
                    _apply(paths)

    def test_stale_overlay_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            _write_overlay(paths, authored_at=NOW - timedelta(minutes=16))

            with self.assertRaisesRegex(MarketReadOverlayError, "MARKET_READ_OVERLAY_STALE"):
                _apply(paths)

    def test_latest_truly_resolved_v2_prediction_must_be_reviewed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prediction_id = "mr2:" + "a" * 64
            predictions = root / "market_read_predictions.jsonl"
            rows = [
                _v2_prediction("mr2:" + "b" * 64, resolution_status="UNRESOLVED"),
                {
                    **_v2_prediction("mr2:" + "c" * 64),
                    "source_snapshot_conflict": True,
                    "score_eligible": False,
                },
                _v2_prediction(prediction_id),
            ]
            predictions.write_text("".join(json.dumps(row) + "\n" for row in rows))
            paths = _prepared_paths(root, predictions_path=predictions)
            _write_overlay(paths)

            with self.assertRaisesRegex(
                MarketReadOverlayError,
                "MARKET_READ_PRIOR_PREDICTION_NOT_REVIEWED",
            ):
                _apply(paths)

            _write_overlay(paths, prior_prediction_ids=[prediction_id])
            _apply(paths)
            final = json.loads(paths["output"].read_text())
            self.assertEqual(
                final["market_read_review"]["prior_prediction_ids"],
                [prediction_id],
            )

    def test_directional_read_requires_numeric_geometry_around_current_quote(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            read = _market_read()
            read["next_30m_prediction"]["target_zone"] = "higher after 30m"
            _write_overlay(paths, market_read=read)

            with self.assertRaisesRegex(MarketReadOverlayError, "AI_MARKET_READ_GEOMETRY_INCOMPLETE"):
                _apply(paths)

    def test_directional_read_rejects_any_target_or_invalidation_rail_on_wrong_side(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            read = _market_read()
            read["next_30m_prediction"]["target_zone"] = "1.0990 to 1.1030"
            _write_overlay(paths, market_read=read)

            with self.assertRaisesRegex(MarketReadOverlayError, "AI_MARKET_READ_GEOMETRY_CONFLICT"):
                _apply(paths)

            read = _market_read()
            read["best_trade_if_forced"]["tp"] = "1.0990 to 1.1040"
            _write_overlay(paths, market_read=read)
            with self.assertRaisesRegex(
                MarketReadOverlayError,
                "AI_MARKET_READ_FORCED_GEOMETRY_CONFLICT",
            ):
                _apply(paths)

    def test_trade_source_uses_five_minute_ai_read_window_not_post_quote_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(
                Path(tmp),
                snapshot_at=NOW - timedelta(seconds=30),
            )
            _write_overlay(paths)

            summary = _apply(paths)

            self.assertEqual(summary.action, "TRADE")

    def test_trade_rejects_source_older_than_read_only_snapshot_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(
                Path(tmp),
                snapshot_at=NOW - timedelta(minutes=5, seconds=1),
            )
            _write_overlay(paths)

            with self.assertRaisesRegex(MarketReadOverlayError, "MARKET_READ_SOURCE_STALE"):
                _apply(paths)

    def test_stale_trade_baseline_can_still_publish_a_nontrade_veto(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            baseline = _baseline()
            baseline["generated_at_utc"] = (NOW - timedelta(minutes=10)).isoformat()
            paths = _prepared_paths(
                Path(tmp),
                baseline=baseline,
                snapshot_at=NOW - timedelta(minutes=10),
            )
            _write_overlay(paths, disposition="VETO_WAIT")

            summary = _apply(paths)

            self.assertEqual(summary.action, "WAIT")

    def test_range_read_requires_bracketed_targets_and_outer_invalidations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepared_paths(Path(tmp))
            _write_overlay(paths, market_read=_range_market_read())
            _apply(paths)

            read = _range_market_read()
            read["next_2h_prediction"]["invalidation"] = "1.1000 to 1.1030"
            _write_overlay(paths, market_read=read)
            with self.assertRaisesRegex(MarketReadOverlayError, "AI_MARKET_READ_RANGE_GEOMETRY_CONFLICT"):
                _apply(paths)


def _synthetic_execution_cost_surface() -> dict:
    """Stable, valid global cost material for tests outside cost calibration.

    The allocation tests use small semantic ledgers to exercise exact-lane
    binding and WAL behavior. They intentionally do not manufacture the 20
    independently audited entry/TP/SL transport receipts required by the live
    cost contract; this fixture keeps that orthogonal concern constant.
    """

    observed_at = (NOW - timedelta(days=1)).isoformat()

    def transport_section(label: str) -> dict:
        rows_sha256 = hashlib.sha256(
            f"synthetic-{label}-cost-cohort".encode()
        ).hexdigest()
        return {
            "samples": 20,
            "adverse_p95_pips": 0.0,
            "adverse_max_pips": 0.0,
            "oldest_fill_utc": observed_at,
            "latest_fill_utc": observed_at,
            "rows_sha256": rows_sha256,
        }

    material = {
        "contract": "QR_NET_EXECUTION_COST_FLOOR_V1",
        "parse_status": "VALID",
        "scope": "SYSTEM_GATEWAY_ATTRIBUTED_ALL_PAIRS_SIDES_METHODS",
        "minimum_samples": 20,
        "maximum_sample_age_seconds": 90 * 24 * 60 * 60,
        "market_entry": transport_section("market-entry"),
        "take_profit_exit": transport_section("take-profit-exit"),
        "stop_loss_exit": transport_section("stop-loss-exit"),
        "global_financing": {
            "observation_trades": 20,
            "adverse_trades": 1,
            "entry_units_total": 20_000.0,
            "adverse_total_jpy": 0.002,
            "adverse_mean_jpy_per_unit": 0.000002,
            "adverse_occurrence_wilson95_upper": 0.236131193,
            "adverse_stress_jpy_per_unit": 0.000000472262,
            "oldest_observation_utc": observed_at,
            "latest_observation_utc": observed_at,
        },
    }
    return {
        **material,
        "execution_cost_surface_sha256": canonical_json_sha256(material),
    }


def _synthetic_execution_cost_floor(
    *,
    scope_key: str = "EUR_USD|LONG|TREND_CONTINUATION|MARKET",
) -> dict:
    """Minimal content-addressed proof accepted by the pure numeric helper."""

    material = {
        "contract": "QR_NET_EXECUTION_COST_FLOOR_V1",
        "status": "PASSED",
        "market_entry_adverse_p95_pips": 0.0,
        "audited_protected_exit_adverse_p95_pips": 0.0,
        "financing_adverse_stress_jpy_per_unit": 0.0,
        "scope_key": scope_key,
        "spread_double_count_forbidden": True,
    }
    return {**material, "proof_sha256": canonical_json_sha256(material)}


def _prepared_paths(
    root: Path,
    *,
    baseline: dict | None = None,
    predictions_path: Path | None = None,
    snapshot_at: datetime = NOW,
) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "baseline": root / "baseline.json",
        "packet": root / "packet.json",
        "overlay": root / "overlay.json",
        "output": root / "output.json",
        "snapshot": root / "broker_snapshot.json",
        "intents": root / "order_intents.json",
        "predictions": predictions_path or root / "market_read_predictions.jsonl",
        "watchdog": root / "qr_trader_run_watchdog.json",
        "execution_ledger": root / "execution_ledger.db",
    }
    _write_exact_vehicle_ledger(
        paths["execution_ledger"],
        [320.0] * 8,
    )
    paths["baseline"].write_text(json.dumps(baseline or _baseline()))
    paths["snapshot"].write_text(
        json.dumps(
            {
                "fetched_at_utc": snapshot_at.isoformat(),
                "account": {
                    "nav_jpy": 100_000.0,
                    "fetched_at_utc": snapshot_at.isoformat(),
                },
                "home_conversions": {"USD": 100.0, "JPY": 1.0},
                "quotes": {
                    "EUR_USD": {
                        "bid": 1.1000,
                        "ask": 1.1002,
                        "timestamp_utc": snapshot_at.isoformat(),
                    },
                },
            }
        )
    )
    paths["intents"].write_text(
        json.dumps(
            {
                "generated_at_utc": NOW.isoformat(),
                "results": [
                    {
                        "lane_id": LANE_ID,
                        "status": "LIVE_READY",
                        "risk_allowed": True,
                        "live_blocker_codes": [],
                        "intent": {
                            "pair": "EUR_USD",
                            "side": "LONG",
                            "order_type": "MARKET",
                            "units": 1200,
                            "entry": 1.1002,
                            "tp": 1.1040,
                            "sl": 1.0980,
                            "market_context": {"method": "TREND_CONTINUATION"},
                            "metadata": {
                                "attach_take_profit_on_fill": True,
                                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                                "capture_economics_status": "NEGATIVE_EXPECTANCY",
                                "capture_take_profit_exact_vehicle_required": True,
                                "capture_take_profit_scope": "PAIR_SIDE_METHOD_VEHICLE",
                                "capture_take_profit_scope_key": (
                                    "EUR_USD|LONG|TREND_CONTINUATION|MARKET|TAKE_PROFIT_ORDER"
                                ),
                                "capture_take_profit_vehicle": "MARKET",
                                "capture_take_profit_metrics_source": (
                                    "data/execution_ledger.db:exact_vehicle_take_profit"
                                ),
                                "capture_take_profit_expectancy_jpy": 320.0,
                                "capture_take_profit_net_jpy": 2560.0,
                                "capture_take_profit_trades": 8,
                                "capture_take_profit_wins": 8,
                                "capture_take_profit_losses": 0,
                                "capture_take_profit_avg_win_jpy": 320.0,
                                "capture_take_profit_avg_loss_jpy": 0.0,
                                "capture_exact_vehicle_net_scope": (
                                    "PAIR_SIDE_METHOD_VEHICLE"
                                ),
                                "capture_exact_vehicle_net_scope_key": (
                                    "EUR_USD|LONG|TREND_CONTINUATION|MARKET|ALL_AUDITED_EXITS"
                                ),
                                "capture_exact_vehicle_net_vehicle": "MARKET",
                                "capture_exact_vehicle_net_metrics_source": (
                                    "data/execution_ledger.db:exact_vehicle_net"
                                ),
                                "capture_exact_vehicle_net_exit_scope": (
                                    "ALL_AUDITED_EXITS"
                                ),
                                "capture_exact_vehicle_net_trades": 8,
                                "capture_exact_vehicle_net_wins": 8,
                                "capture_exact_vehicle_net_losses": 0,
                                "capture_exact_vehicle_net_jpy": 2560.0,
                                "capture_exact_vehicle_net_expectancy_jpy": 320.0,
                                "capture_exact_vehicle_net_avg_win_jpy": 320.0,
                                "capture_exact_vehicle_net_avg_loss_jpy": 0.0,
                                "capture_exact_vehicle_net_unresolved_realized_trades": 0,
                                "capture_exact_vehicle_net_unresolved_realized_net_jpy": 0.0,
                                "capture_exact_vehicle_net_unresolved_trade_ids_sha256": (
                                    hashlib.sha256(b"[]").hexdigest()
                                ),
                                "capture_market_close_expectancy_jpy": -120.0,
                                "capture_avg_win_jpy": 500.0,
                                "capture_avg_loss_jpy": 300.0,
                                "forecast_direction": "UP",
                                "forecast_confidence": 0.72,
                                "forecast_raw_confidence": 0.80,
                                "forecast_current_price": 1.1001,
                                "forecast_target_price": 1.1050,
                                "forecast_invalidation_price": 1.0985,
                                "forecast_directional_calibration_name": "directional_forecast_up",
                                "forecast_calibration_multiplier": 0.90,
                                "forecast_directional_economic_hit_rate": 0.61,
                                "forecast_directional_economic_samples": 100,
                                "forecast_directional_hit_rate": 0.67,
                                "forecast_directional_samples": 72,
                                "forecast_directional_timeout_rate": 0.28,
                                "max_loss_jpy": 400.0,
                            },
                        },
                        "risk_metrics": {
                            "entry_price": 1.1002,
                            "loss_pips": 22.0,
                            "reward_pips": 38.0,
                            "risk_jpy": 264.0,
                            "reward_jpy": 456.0,
                            "reward_risk": 38.0 / 22.0,
                            "spread_pips": 2.0,
                            "jpy_per_pip": 12.0,
                            "estimated_margin_jpy": 1200.0,
                        },
                    }
                ],
            }
        )
    )
    if not paths["predictions"].exists():
        paths["predictions"].write_text("")
    paths["watchdog"].write_text(json.dumps(_watchdog()))
    prepare_market_read_baseline(
        baseline_path=paths["baseline"],
        packet_path=paths["packet"],
        evidence_sources=_sources(paths),
        now=NOW,
    )
    return paths


def _sources(paths: dict[str, Path]) -> dict[str, Path]:
    return {
        "broker_snapshot": paths["snapshot"],
        "order_intents": paths["intents"],
        "market_read_predictions": paths["predictions"],
        "qr_trader_run_watchdog": paths["watchdog"],
        "execution_ledger": paths["execution_ledger"],
    }


def _write_exact_vehicle_ledger(path: Path, outcomes: list[float]) -> None:
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE execution_events (
                event_uid TEXT PRIMARY KEY,
                ts_utc TEXT,
                event_type TEXT,
                lane_id TEXT,
                order_id TEXT,
                trade_id TEXT,
                pair TEXT,
                side TEXT,
                units INTEGER,
                realized_pl_jpy REAL,
                financing_jpy REAL,
                exit_reason TEXT,
                raw_json TEXT
            )
            """
        )
        conn.execute(
            "CREATE TABLE sync_state (key TEXT PRIMARY KEY, value TEXT, updated_at_utc TEXT)"
        )
        conn.execute(
            "INSERT INTO sync_state VALUES (?, ?, ?)",
            (
                "oanda_transaction_coverage_start_utc",
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
            ),
        )
        lane = LANE_ID
        for index, realized in enumerate(outcomes):
            trade_id = f"trade-{index}"
            order_id = f"entry-{index}"
            entry_ts = f"2026-07-01T00:{index:02d}:00Z"
            close_ts = f"2026-07-01T01:{index:02d}:00Z"
            entry_raw = {
                "id": f"fill-{index}",
                "time": entry_ts,
                "type": "ORDER_FILL",
                "orderID": order_id,
                "instrument": "EUR_USD",
                "units": "1000",
                "reason": "MARKET_ORDER",
                "tradeOpened": {
                    "tradeID": trade_id,
                    "units": "1000",
                },
            }
            exit_reason = (
                "TAKE_PROFIT_ORDER"
                if realized > 0
                else "MARKET_ORDER_TRADE_CLOSE"
            )
            close_raw = {
                "id": f"close-{index}",
                "time": close_ts,
                "type": "ORDER_FILL",
                "instrument": "EUR_USD",
                "orderID": f"close-{index}",
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
            conn.execute(
                "INSERT INTO execution_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"gateway-{index}", entry_ts, "GATEWAY_ORDER_SENT", lane,
                    order_id, trade_id, "EUR_USD", "LONG", 1000, None, 0.0,
                    "MARKET_ORDER", json.dumps({"type": "MARKET_ORDER"}),
                ),
            )
            conn.execute(
                "INSERT INTO execution_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"fill-{index}", entry_ts, "ORDER_FILLED", lane,
                    order_id, trade_id, "EUR_USD", "LONG", 1000, None, 0.0,
                    "MARKET_ORDER", json.dumps(entry_raw),
                ),
            )
            conn.execute(
                "INSERT INTO execution_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"close-{index}", close_ts, "TRADE_CLOSED", None,
                    f"close-{index}", trade_id, "EUR_USD", "SHORT", 1000,
                    realized, 0.0, exit_reason, json.dumps(close_raw),
                ),
            )


def _append_vehicle_trade(
    path: Path,
    *,
    pair: str,
    side: str,
    method: str,
    vehicle: str,
    realized: float,
    index: int,
    connection: sqlite3.Connection | None = None,
    terminal: bool = True,
) -> None:
    trade_id = f"appended-trade-{index}"
    order_id = f"appended-entry-{index}"
    entry_ts = "2026-07-02T00:00:00Z"
    close_ts = "2026-07-02T01:00:00Z"
    signed_units = 1000 if side == "LONG" else -1000
    entry_reason = {
        "LIMIT": "LIMIT_ORDER",
        "STOP": "STOP_ORDER",
        "MARKET": "MARKET_ORDER",
    }[vehicle]
    lane = f"test_trader:{pair}:{side}:{method}:{vehicle}"
    entry_raw = {
        "id": f"appended-fill-{index}",
        "time": entry_ts,
        "type": "ORDER_FILL",
        "orderID": order_id,
        "instrument": pair,
        "units": str(signed_units),
        "reason": entry_reason,
        "tradeOpened": {"tradeID": trade_id, "units": str(signed_units)},
    }
    exit_reason = (
        "TAKE_PROFIT_ORDER" if realized > 0 else "MARKET_ORDER_TRADE_CLOSE"
    )
    close_raw = {
        "id": f"appended-close-{index}",
        "time": close_ts,
        "type": "ORDER_FILL",
        "instrument": pair,
        "orderID": f"appended-close-{index}",
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
    if not terminal:
        close_raw.pop("tradesClosed")
        close_raw["tradeReduced"] = {
            "tradeID": trade_id,
            "realizedPL": str(realized),
            "financing": "0.0",
        }
    rows = [
        (
            f"appended-gateway-{index}", entry_ts, "GATEWAY_ORDER_SENT", lane,
            order_id, trade_id, pair, side, signed_units, None, 0.0,
            entry_reason, json.dumps({"type": entry_reason}),
        ),
        (
            f"appended-fill-{index}", entry_ts, "ORDER_FILLED", lane,
            order_id, trade_id, pair, side, signed_units, None, 0.0,
            entry_reason, json.dumps(entry_raw),
        ),
        (
            f"appended-close-{index}", close_ts,
            "TRADE_CLOSED" if terminal else "TRADE_REDUCED", None,
            f"appended-close-{index}", trade_id, pair,
            "SHORT" if side == "LONG" else "LONG", abs(signed_units),
            realized, 0.0, exit_reason, json.dumps(close_raw),
        ),
    ]
    if connection is not None:
        connection.executemany(
            "INSERT INTO execution_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    else:
        with sqlite3.connect(path) as conn:
            conn.executemany(
                "INSERT INTO execution_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
def _reprepare(paths: dict[str, Path]) -> None:
    baseline = json.loads(paths["baseline"].read_text())
    baseline.pop("decision_provenance", None)
    paths["baseline"].write_text(json.dumps(baseline))
    prepare_market_read_baseline(
        baseline_path=paths["baseline"],
        packet_path=paths["packet"],
        evidence_sources=_sources(paths),
        now=NOW,
    )


def _watchdog() -> dict:
    weekend = {
        "active": True,
        "automation_status": "PAUSED",
        "exists": True,
        "in_weekend_pause_window": True,
        "mode": "paused",
        "now_jst": "2026-07-11T12:00:00+09:00",
        "qr_trader_managed": True,
        "reason": "weekend guard",
    }
    return {
        "generated_at_utc": NOW.isoformat(),
        "status": "OK",
        "runtime_status": "OK",
        "issue_status": "OK",
        "overall_status": "OK",
        "severity": "OK",
        "issues": [],
        "missed_expected_window": False,
        "minutes_since_last_run": 60.0,
        "last_trader_run_at": (NOW - timedelta(hours=1)).isoformat(),
        "last_trader_run_source": "trader_journal.ts",
        "automation_config": {
            "exists": True,
            "issues": [],
            "model": "gpt-5.5",
            "reasoning_effort": "high",
            "cadence_minutes": 60,
            "cwd": "/runtime",
            "cwds": ["/runtime"],
            "status": "PAUSED",
            "weekend_pause": dict(weekend),
        },
        "weekend_pause": dict(weekend),
        "guardian_receipt": {
            "action": None,
            "active": False,
            "exists": False,
            "issues": [],
            "receipt_summaries": [],
            "review_excerpt": "Generated at the original observation clock",
        },
        "execution_boundary": {
            "broker_writes_enabled": False,
            "no_live_side_effects": True,
            "read_only": True,
        },
        "environment": {"QR_TRADER_WATCHDOG_CAN_WAKE": "0"},
        "codex_logs": {
            "available": True,
            "queried_at_utc": NOW.isoformat(),
            "entries": [],
        },
    }


def _baseline(*, action: str = "TRADE", lane_ids: list[str] | None = None) -> dict:
    selected = [LANE_ID] if lane_ids is None else lane_ids
    return {
        "generated_at_utc": NOW.isoformat(),
        "market_read_first": _market_read(),
        "action": action,
        "selected_lane_id": selected[0] if selected else None,
        "selected_lane_ids": selected,
        "cancel_order_ids": ["old-pending-1"],
        "confidence": "HIGH",
        "method": "TREND_CONTINUATION",
        "risk_notes": ["deterministic units and stops are immutable"],
        "evidence_refs": [f"intent:{LANE_ID}", "broker:snapshot"],
        "operator_summary": "deterministic baseline",
    }


def _market_read() -> dict:
    return {
        "naked_read": {
            "currency_bought": "EUR",
            "currency_sold": "USD",
            "cleanest_pair_expression": "EUR_USD",
            "is_cleanest_currency_theme": "YES",
            "location_24h": "MIDDLE",
            "h1_h4_alignment": "H1/H4 aligned long",
            "tape_state": "TREND",
            "known_winning_trade_shape_match": "partial match",
            "proposed_building_style_allowed": "single entry only",
            "thesis_state": "ALIVE",
            "what_price_is_trying_to_do_now": "break the 1.1015 shelf",
        },
        "next_30m_prediction": {
            "pair": "EUR_USD",
            "direction": "LONG",
            "expected_path": "hold 1.1000 then test 1.1020",
            "target_zone": "1.1020 to 1.1030",
            "invalidation": "1.0985",
        },
        "next_2h_prediction": {
            "pair": "EUR_USD",
            "direction": "LONG",
            "expected_path": "extend through 1.1030",
            "target_zone": "1.1040 to 1.1050",
            "invalidation": "1.0975",
        },
        "best_trade_if_forced": {
            "pair": "EUR_USD",
            "direction": "LONG",
            "vehicle": "MARKET",
            "entry": "1.1002",
            "tp": "1.1040",
            "sl": "1.0980",
            "why_this_pays": "target is above entry and invalidation is below it",
        },
    }


def _range_market_read() -> dict:
    read = _market_read()
    for key in ("next_30m_prediction", "next_2h_prediction"):
        read[key]["direction"] = "RANGE"
        read[key]["target_zone"] = "1.0995 to 1.1010"
        read[key]["invalidation"] = "1.0980 to 1.1025"
    return read


def _overlay(
    paths: dict[str, Path],
    *,
    disposition: str = "ACCEPT_BASELINE",
    authored_at: datetime = NOW,
    market_read: dict | None = None,
    prior_prediction_ids: list[str] | None = None,
    size_multiple: float = 0.75,
) -> dict:
    baseline = json.loads(paths["baseline"].read_text())
    packet = json.loads(paths["packet"].read_text())
    trade_allocation = (
        str(baseline.get("action") or "").upper() == "TRADE"
        and disposition == "ACCEPT_BASELINE"
    )
    base_units = int(
        ((packet.get("capital_allocation_board") or {}).get("selected_lane") or {}).get(
            "base_units"
        )
        or 0
    )
    return {
        "schema_version": MARKET_READ_OVERLAY_SCHEMA_VERSION,
        "author_kind": CODEX_MARKET_READ_AUTHOR,
        "model": "gpt-5.5",
        "reasoning_effort": "high",
        "authored_at_utc": authored_at.isoformat(),
        "baseline_sha256": canonical_json_sha256(baseline_core_payload(baseline)),
        "evidence_packet_sha256": packet["evidence_packet_sha256"],
        "baseline_disposition": disposition,
        "market_read_first": market_read or _market_read(),
        "market_read_review": {
            "prior_prediction_ids": prior_prediction_ids or [],
            "what_failed": "Reviewed the latest resolved path" if prior_prediction_ids else "NO_RESOLVED_PRIOR",
            "adjustment": "Use numeric quote-relative geometry and veto if the counterargument dominates.",
            "no_change_reason": "",
        },
        "market_read_counterargument": "The apparent breakout can fail back into the prior range.",
        "market_read_change_summary": "Rebuilt the directional path from the current broker quote.",
        "market_read_veto_reason": (
            "Current numeric forecast contradicts the deterministic entry trigger."
            if disposition.startswith("VETO_")
            else ""
        ),
        "capital_allocation": {
            "decision": "ALLOCATE" if trade_allocation else "NO_TRADE",
            "lane_id": baseline.get("selected_lane_id") if trade_allocation else None,
            "size_multiple": size_multiple if trade_allocation else 0.0,
            "selected_units": int(base_units * size_multiple) if trade_allocation else 0,
            "allocation_board_sha256": packet["capital_allocation_board_sha256"],
            "rationale": (
                "Direction-specific economic precision and exact-vehicle TP expectancy support bounded exposure."
                if trade_allocation
                else "No fresh entry capital is authorized for this receipt."
            ),
        },
    }


def _write_overlay(paths: dict[str, Path], **kwargs: object) -> None:
    paths["overlay"].write_text(json.dumps(_overlay(paths, **kwargs)))


def _apply(paths: dict[str, Path]):
    return apply_codex_market_read_overlay(
        baseline_path=paths["baseline"],
        packet_path=paths["packet"],
        overlay_path=paths["overlay"],
        output_path=paths["output"],
        evidence_sources=_sources(paths),
        now=NOW,
    )


def _v2_prediction(prediction_id: str, *, resolution_status: str = "RESOLVED_MID_CANDLE_DIAGNOSTIC") -> dict:
    result = {
        "resolution_status": resolution_status,
        "direction_status": "WRONG" if resolution_status.startswith("RESOLVED") else "UNRESOLVED",
        "target_completion_status": "NOT_TOUCHED" if resolution_status.startswith("RESOLVED") else "UNRESOLVED",
        "invalidation_status": "TOUCHED" if resolution_status.startswith("RESOLVED") else "UNRESOLVED",
        "first_touch_status": "INVALIDATION_FIRST" if resolution_status.startswith("RESOLVED") else "UNRESOLVED",
        "full_read_status": "INVALIDATION_FIRST" if resolution_status.startswith("RESOLVED") else "UNRESOLVED",
    }
    return {
        "schema_version": 2,
        "prediction_id": prediction_id,
        "generated_at_utc": (NOW - timedelta(hours=3)).isoformat(),
        "pair": "EUR_USD",
        "direction": "LONG",
        "action": "TRADE",
        "score_eligible": True,
        "source_snapshot_conflict": False,
        "verdict": "FULL_READ_INCOMPLETE" if resolution_status.startswith("RESOLVED") else "UNRESOLVED",
        "horizon_results": {"30m": result, "2h": {**result, "resolution_status": "UNRESOLVED"}},
    }


if __name__ == "__main__":
    unittest.main()
