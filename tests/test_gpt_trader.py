from __future__ import annotations

import json
import io
import os
import hashlib
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from quant_rabbit.cli import main
from quant_rabbit.execution_timing_contracts import (
    TP_PROGRESS_REPAIR_LIVE_EVIDENCE_BOUNDARY_UTC,
)
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS
from quant_rabbit.market_close_leak_gate import MARKET_CLOSE_LEAK_FAMILY_BLOCK_CODE
from quant_rabbit.market_read_overlay import (
    CODEX_MARKET_READ_AUTHOR,
    GUARDIAN_ACTION_RECEIPT_MATERIAL_CONTRACT,
    apply_codex_market_read_overlay,
    baseline_core_payload,
    canonical_json_sha256,
    execution_envelope_payload,
    market_read_sha256,
    prepare_market_read_baseline,
    projection_calibration_scopes_from_lanes,
)
from quant_rabbit.month_scale_residual_gate import (
    MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCK_CODE,
)
from quant_rabbit.gpt_trader import (
    GPT_TRADER_SCHEMA,
    GPTTraderBrain,
    StaticTraderProvider,
    _draft_candidate_lane_ids,
    _draft_margin_aware_basket,
    _guardian_receipt_consumption_trade_routing_issues,
    _lane_forecast_packet,
    _projection_ledger_packet,
    _qr_trader_run_watchdog_packet,
    draft_trader_decision,
    post_stop_thesis_review,
)
from quant_rabbit.strategy.forecast_technical_context import (
    CONFIDENCE_SEMANTICS,
    MAX_EVIDENCE_BYTES,
    build_forecast_technical_context,
    build_forecast_technical_context_evidence,
)
from quant_rabbit.strategy.projection_ledger import LedgerEntry, write_ledger


LANE_ID = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION"


def _forecast_context_evidence(
    pair: str,
    current_price: float,
    *,
    direction: str = "UP",
    spread_pips: float = 0.2,
    pair_chart: dict | None = None,
    calendar_path: Path | None = None,
    strategy_profile_path: Path | None = None,
    now_utc: datetime | None = None,
) -> dict:
    up = str(direction).upper() == "UP"
    chart = pair_chart or {
        "confluence": {
            "dominant_regime": "TREND_UP" if up else "TREND_DOWN",
            "price_percentile_24h": 0.7 if up else 0.3,
            "price_percentile_7d": 0.6 if up else 0.4,
        },
        "views": [
            {
                "granularity": "M5",
                "regime_reading": {
                    "state": "TREND_STRONG",
                    "atr_percentile": 60.0,
                },
                "indicators": {"atr_pips": 2.0},
                "family_scores": {
                    "trend_score": 1.0 if up else -1.0,
                    "mean_rev_score": 0.0,
                    "breakout_score": 0.0,
                    "disagreement": 0.0,
                },
                "structure": {
                    "structure_events": [
                        {
                            "kind": "BOS_UP" if up else "BOS_DOWN",
                            "index": 1,
                            "close_confirmed": True,
                        }
                    ]
                },
            }
        ],
    }
    context = build_forecast_technical_context(
        chart,
        pair=pair,
        current_price=current_price,
        spread_pips=spread_pips,
        calendar_path=calendar_path,
        strategy_profile_path=strategy_profile_path,
        now_utc=now_utc,
    )
    return build_forecast_technical_context_evidence(
        context,
        pair=pair,
        current_price=current_price,
    )


def _forecast_weighting_metadata(evidence: dict) -> dict:
    body = evidence.get("technical_context_v1") or {}
    receipt = body.get("regime_family_weighting") or {}
    source = receipt.get("source_identity") or {}
    aggregate = receipt.get("aggregate") or {}
    return {
        "forecast_regime_family_weighting_sha256": receipt.get("receipt_sha256"),
        "forecast_regime_family_selected_method": source.get("selected_method"),
        "forecast_regime_family_direction": aggregate.get("direction"),
    }


class GPTTraderBrainTest(unittest.TestCase):
    def setUp(self) -> None:
        # These verifier tests exercise decision/artifact semantics with tiny
        # exact-lane ledgers. Keep the independent 20-sample transport/carry
        # calibration fixed and content-addressed instead of making every
        # fixture manufacture broker request/create/fill/protection cohorts.
        cost_patch = patch(
            "quant_rabbit.capture_economics.read_execution_cost_surface",
            return_value=_synthetic_execution_cost_surface(),
        )
        cost_patch.start()
        self.addCleanup(cost_patch.stop)

    def _run_m15_recovery_trade_with_metadata(
        self,
        root: Path,
        recovery_intent,
        metadata: dict,
    ) -> tuple[object, dict]:
        """Run the real GPT packet/verifier path for one recovery mutation."""

        files = _fixtures(root)
        lane_id = "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE"
        reward = float(recovery_intent.entry) - float(recovery_intent.tp)
        loss = float(recovery_intent.sl) - float(recovery_intent.entry)
        result = _result(
            lane_id=lane_id,
            method="BREAKOUT_FAILURE",
            pair="EUR_USD",
            side="SHORT",
            metadata=metadata,
        )
        result["intent"].update(
            {
                "order_type": recovery_intent.order_type.value,
                "units": recovery_intent.units,
                "entry": recovery_intent.entry,
                "tp": recovery_intent.tp,
                "sl": recovery_intent.sl,
                "thesis": recovery_intent.thesis,
                "market_context": {
                    "regime": recovery_intent.market_context.regime,
                    "narrative": recovery_intent.market_context.narrative,
                    "chart_story": recovery_intent.market_context.chart_story,
                    "method": recovery_intent.market_context.method.value,
                    "invalidation": recovery_intent.market_context.invalidation,
                },
            }
        )
        result["risk_metrics"] = {
            "entry_price": recovery_intent.entry,
            "loss_pips": loss * 10_000,
            "reward_pips": reward * 10_000,
            "risk_jpy": 100.0,
            "reward_jpy": 100.0 * (reward / loss),
            "reward_risk": reward / loss,
            "spread_pips": 0.8,
            "jpy_per_pip": 0.08,
            "estimated_margin_jpy": 10.0,
        }
        files["intents"].write_text(json.dumps({"results": [result]}))
        files["campaign"].write_text(
            json.dumps(
                {
                    "lanes": [
                        {
                            "desk": "failure_trader",
                            "pair": "EUR_USD",
                            "direction": "SHORT",
                            "method": "BREAKOUT_FAILURE",
                            "adoption": "ORDER_INTENT_REQUIRED",
                            "campaign_role": "NOW",
                            "required_receipt": "verified M15 recovery",
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
                            "direction": "SHORT",
                            "status": "CANDIDATE",
                            "pretrade_net_jpy": 1000,
                            "live_net_jpy": 1000,
                            "live_worst_jpy": -100,
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
                            "methods": {"BREAKOUT_FAILURE": 10},
                            "themes": {"breakout_failure": 4},
                            "examples": ["M15 upper-rail rejection"],
                        }
                    ]
                }
            )
        )
        decision = _trade_decision(
            lane_id=lane_id,
            method="BREAKOUT_FAILURE",
            pair="EUR_USD",
            direction="SHORT",
        )
        summary = _brain(root, files, decision).run(
            snapshot_path=files["snapshot"]
        )
        return summary, json.loads((root / "gpt_decision.json").read_text())

    def test_lane_packet_preserves_verified_context_and_drops_tampered_body(self) -> None:
        evidence = _forecast_context_evidence("EUR_USD", 1.1001)
        metadata = {
            "forecast_direction": "UP",
            "forecast_confidence": 0.72,
            "forecast_current_price": 1.1001,
            "forecast_technical_context": evidence,
        }

        packet = _lane_forecast_packet(metadata, pair="EUR_USD")
        self.assertEqual(packet["technical_context"], evidence)
        self.assertEqual(
            packet["technical_context"]["confidence_semantics"],
            CONFIDENCE_SEMANTICS,
        )

        tampered = json.loads(json.dumps(evidence))
        tampered["technical_context_v1"]["regime"]["dominant"] = "RANGE"
        metadata["forecast_technical_context"] = tampered
        packet = _lane_forecast_packet(metadata, pair="EUR_USD")
        self.assertEqual(packet["technical_context"]["status"], "UNKNOWN")
        self.assertEqual(
            packet["technical_context"]["reason"],
            "TECHNICAL_CONTEXT_EVIDENCE_HASH_MISMATCH",
        )
        self.assertIsNone(packet["technical_context"]["technical_context_v1"])
        self.assertFalse(packet["technical_context"]["live_permission"])

    def test_lane_packet_exposes_verified_m15_recovery_without_faking_standard_context(self) -> None:
        from tests.test_risk_engine import _m15_recovery_fixture

        with tempfile.TemporaryDirectory() as tmp:
            _now, _chart_path, recovery_intent, _broker = (
                _m15_recovery_fixture(Path(tmp))
            )
            intent = {
                "pair": recovery_intent.pair,
                "side": recovery_intent.side.value,
                "order_type": recovery_intent.order_type.value,
                "units": recovery_intent.units,
                "entry": recovery_intent.entry,
                "tp": recovery_intent.tp,
                "sl": recovery_intent.sl,
                "market_context": {
                    "method": recovery_intent.market_context.method.value,
                },
                "metadata": recovery_intent.metadata,
            }

            packet = _lane_forecast_packet(
                recovery_intent.metadata,
                pair="EUR_USD",
                intent=intent,
            )

            self.assertEqual(packet["technical_context"]["status"], "UNKNOWN")
            recovery = packet["m15_recovery"]
            self.assertEqual(recovery["status"], "VERIFIED")
            self.assertEqual(
                recovery["source_timeframes"],
                ["M15", "M30", "H1", "H4", "D"],
            )
            self.assertEqual(recovery["forecast"]["direction"], "DOWN")
            self.assertEqual(recovery["forecast"]["raw_confidence"], 0.758)
            self.assertEqual(
                recovery["exact_tp_proof"]["mode"],
                "TP_PROOF_COLLECTION_HARVEST",
            )
            self.assertTrue(recovery["risk_revalidated"])
            self.assertFalse(recovery["live_permission"])
            self.assertEqual(len(recovery["packet_sha256"]), 64)

            limit_packet = _lane_forecast_packet(
                recovery_intent.metadata,
                pair="EUR_USD",
                intent={**intent, "order_type": "LIMIT"},
            )["m15_recovery"]
            self.assertEqual(limit_packet["status"], "INVALID")
            self.assertIn(
                "M15_RECOVERY_INTENT_SCOPE_INVALID",
                limit_packet["validation_errors"],
            )
            self.assertIn(
                "M15_RECOVERY_LANE_BINDING_CONTRACT_INVALID",
                limit_packet["validation_errors"],
            )

            tampered_metadata = json.loads(
                json.dumps(recovery_intent.metadata)
            )
            tampered_metadata["m15_recovery_micro_risk_revalidated"] = False
            tampered_intent = {**intent, "metadata": tampered_metadata}
            tampered = _lane_forecast_packet(
                tampered_metadata,
                pair="EUR_USD",
                intent=tampered_intent,
            )
            self.assertEqual(tampered["m15_recovery"]["status"], "INVALID")
            self.assertIn(
                "M15_RECOVERY_RISK_OR_CONTEXT_SCOPE_INVALID",
                tampered["m15_recovery"]["validation_errors"],
            )

            for mutation in ("empty", "missing"):
                with self.subTest(mutation=mutation):
                    partial_metadata = json.loads(
                        json.dumps(recovery_intent.metadata)
                    )
                    for key in (
                        "m15_recovery_micro_receipt",
                        "forecast_m15_recovery_binding",
                        "m15_recovery_lane_binding",
                    ):
                        if mutation == "empty":
                            partial_metadata[key] = {}
                        else:
                            partial_metadata.pop(key, None)
                    partial_intent = {**intent, "metadata": partial_metadata}
                    partial = _lane_forecast_packet(
                        partial_metadata,
                        pair="EUR_USD",
                        intent=partial_intent,
                    )["m15_recovery"]
                    self.assertEqual(partial["status"], "INVALID")
                    self.assertNotEqual(partial["status"], "NOT_APPLICABLE")
                    self.assertIn(
                        "M15_RECOVERY_BINDING_MISSING",
                        partial["validation_errors"],
                    )

    def test_rejects_trade_when_selected_m15_recovery_packet_is_invalid(self) -> None:
        from tests.test_risk_engine import _m15_recovery_fixture

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            _now, _chart_path, recovery_intent, _broker = (
                _m15_recovery_fixture(root)
            )
            lane_id = "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE"
            metadata = json.loads(json.dumps(recovery_intent.metadata))
            # Keep the upstream result deceptively LIVE_READY while breaking
            # the recovery authorization chain.  The GPT verifier must consume
            # the INVALID packet instead of treating it as display-only data.
            metadata["m15_recovery_micro_risk_revalidated"] = False
            reward = float(recovery_intent.entry) - float(recovery_intent.tp)
            loss = float(recovery_intent.sl) - float(recovery_intent.entry)
            result = {
                "lane_id": lane_id,
                "status": "LIVE_READY",
                "risk_allowed": True,
                "risk_issues": [],
                "strategy_issues": [],
                "live_blockers": [],
                "intent": {
                    "pair": recovery_intent.pair,
                    "side": recovery_intent.side.value,
                    "order_type": recovery_intent.order_type.value,
                    "units": recovery_intent.units,
                    "entry": recovery_intent.entry,
                    "tp": recovery_intent.tp,
                    "sl": recovery_intent.sl,
                    "thesis": recovery_intent.thesis,
                    "owner": "trader",
                    "market_context": {
                        "regime": recovery_intent.market_context.regime,
                        "narrative": recovery_intent.market_context.narrative,
                        "chart_story": recovery_intent.market_context.chart_story,
                        "method": recovery_intent.market_context.method.value,
                        "invalidation": recovery_intent.market_context.invalidation,
                    },
                    "metadata": metadata,
                },
                "risk_metrics": {
                    "entry_price": recovery_intent.entry,
                    "loss_pips": loss * 10_000,
                    "reward_pips": reward * 10_000,
                    "risk_jpy": 100.0,
                    "reward_jpy": 100.0 * (reward / loss),
                    "reward_risk": reward / loss,
                    "spread_pips": 0.8,
                    "jpy_per_pip": 0.08,
                    "estimated_margin_jpy": 10.0,
                },
            }
            files["intents"].write_text(json.dumps({"results": [result]}))
            files["campaign"].write_text(
                json.dumps(
                    {
                        "lanes": [
                            {
                                "desk": "failure_trader",
                                "pair": "EUR_USD",
                                "direction": "SHORT",
                                "method": "BREAKOUT_FAILURE",
                                "adoption": "ORDER_INTENT_REQUIRED",
                                "campaign_role": "NOW",
                                "required_receipt": "verified M15 recovery",
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
                                "direction": "SHORT",
                                "status": "CANDIDATE",
                                "pretrade_net_jpy": 1000,
                                "live_net_jpy": 1000,
                                "live_worst_jpy": -100,
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
                                "methods": {"BREAKOUT_FAILURE": 10},
                                "themes": {"breakout_failure": 4},
                                "examples": ["M15 upper-rail rejection"],
                            }
                        ]
                    }
                )
            )
            decision = _trade_decision(
                lane_id=lane_id,
                method="BREAKOUT_FAILURE",
                pair="EUR_USD",
                direction="SHORT",
            )
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {
                issue["code"] for issue in payload["verification_issues"]
            }
            self.assertIn("M15_RECOVERY_PACKET_NOT_VERIFIED", codes)
            recovery_packet = payload["input_packet"]["lanes"][0][
                "forecast"
            ]["m15_recovery"]
            self.assertEqual(recovery_packet["status"], "INVALID")
            self.assertFalse(recovery_packet["risk_revalidated"])

    def test_rejects_trade_when_m15_recovery_bindings_are_empty_or_missing(
        self,
    ) -> None:
        from tests.test_risk_engine import _m15_recovery_fixture

        for mutation in ("empty", "missing"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                _now, _chart_path, recovery_intent, _broker = (
                    _m15_recovery_fixture(root)
                )
                metadata = json.loads(json.dumps(recovery_intent.metadata))
                for key in (
                    "m15_recovery_micro_receipt",
                    "forecast_m15_recovery_binding",
                    "m15_recovery_lane_binding",
                ):
                    if mutation == "empty":
                        metadata[key] = {}
                    else:
                        metadata.pop(key, None)

                summary, payload = self._run_m15_recovery_trade_with_metadata(
                    root,
                    recovery_intent,
                    metadata,
                )

                self.assertEqual(summary.status, "REJECTED")
                codes = {
                    issue["code"] for issue in payload["verification_issues"]
                }
                self.assertIn("M15_RECOVERY_PACKET_NOT_VERIFIED", codes)
                recovery_packet = payload["input_packet"]["lanes"][0][
                    "forecast"
                ]["m15_recovery"]
                self.assertEqual(recovery_packet["status"], "INVALID")
                self.assertIn(
                    "M15_RECOVERY_BINDING_MISSING",
                    recovery_packet["validation_errors"],
                )

    def test_projection_scope_requires_explicit_canonical_forecast_lineage(self) -> None:
        lane = {
            "pair": "EUR_USD",
            "direction": "LONG",
            "status": "LIVE_READY",
            "forecast": {},
        }
        self.assertEqual(projection_calibration_scopes_from_lanes([lane]), [])

        metadata = {
            "forecast_direction": "UP",
            "forecast_directional_calibration_name": "directional_forecast_up",
        }
        lane["forecast"] = _lane_forecast_packet(metadata, pair="EUR_USD")
        self.assertEqual(
            projection_calibration_scopes_from_lanes([lane]),
            [{"pair": "EUR_USD", "direction": "UP", "regime": None}],
        )

        metadata["forecast_directional_calibration_name"] = (
            "directional_forecast_down"
        )
        lane["forecast"] = _lane_forecast_packet(metadata, pair="EUR_USD")
        self.assertEqual(projection_calibration_scopes_from_lanes([lane]), [])

        metadata["forecast_directional_calibration_name"] = (
            "directional_forecast_up"
        )
        lane["forecast"] = _lane_forecast_packet(metadata, pair="EUR_USD")
        duplicate_lanes = [dict(lane) for _index in range(4)]
        gbp_lane = {**lane, "pair": "GBP_USD"}
        self.assertEqual(
            [
                item["pair"]
                for item in projection_calibration_scopes_from_lanes(
                    [*duplicate_lanes, gbp_lane]
                )
            ],
            ["EUR_USD", "GBP_USD"],
        )
        blocked = [{**lane, "status": "BLOCK"} for _index in range(4)]
        self.assertEqual(
            projection_calibration_scopes_from_lanes([*blocked, gbp_lane]),
            [{"pair": "GBP_USD", "direction": "UP", "regime": None}],
        )

    def test_lane_packet_does_not_forward_rehashed_oversized_unknown_reason(self) -> None:
        evidence = _forecast_context_evidence("EUR_USD", 1.1001)
        marker = "DO_NOT_FORWARD_TO_GPT"
        evidence.update(
            {
                "status": "UNKNOWN",
                "reason": marker * MAX_EVIDENCE_BYTES,
                "technical_context_v1": None,
                "context_sha256": None,
            }
        )
        evidence["evidence_sha256"] = canonical_json_sha256(
            {
                key: item
                for key, item in evidence.items()
                if key != "evidence_sha256"
            }
        )
        packet = _lane_forecast_packet(
            {
                "forecast_direction": "UP",
                "forecast_confidence": 0.72,
                "forecast_current_price": 1.1001,
                "forecast_technical_context": evidence,
            },
            pair="EUR_USD",
        )

        self.assertEqual(packet["technical_context"]["status"], "UNKNOWN")
        self.assertEqual(
            packet["technical_context"]["reason"],
            "TECHNICAL_CONTEXT_EVIDENCE_TOO_LARGE",
        )
        self.assertNotIn(marker, json.dumps(packet))

    def test_trade_revalidates_real_market_read_artifacts_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision, artifacts = _apply_real_market_read_artifacts(
                root,
                files,
                _trade_decision(),
            )
            brain = _brain(
                root,
                files,
                decision,
                market_read_artifact_validation_required=True,
                market_read_artifact_paths=artifacts,
            )

            with patch(
                "quant_rabbit.gpt_trader.refresh_market_read_measurements",
                return_value={
                    "status": "NO_CHANGE",
                    "read_only_measurement": True,
                    "live_permission": False,
                    "may_change_execution_permission": False,
                },
            ):
                summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            self.assertEqual(payload["verification_issues"], [])

    def test_trade_rejects_fake_self_declared_digests_even_with_real_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision, artifacts = _apply_real_market_read_artifacts(
                root,
                files,
                _trade_decision(),
            )
            decision["decision_provenance"]["baseline_sha256"] = "a" * 64
            brain = _brain(
                root,
                files,
                decision,
                market_read_artifact_validation_required=True,
                market_read_artifact_paths=artifacts,
            )

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("AI_MARKET_READ_ARTIFACT_FINAL_MISMATCH", codes)

    def test_trade_rejects_missing_market_read_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision = _trade_decision()
            _stamp_codex_market_read(decision)
            artifacts = {
                "baseline": root / "missing-baseline.json",
                "packet": root / "missing-packet.json",
                "overlay": root / "missing-overlay.json",
            }
            brain = _brain(
                root,
                files,
                decision,
                market_read_artifact_validation_required=True,
                market_read_artifact_paths=artifacts,
            )

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("MARKET_READ_ARTIFACT_MISSING", codes)

    def test_trade_rejects_evidence_mutated_after_overlay_application(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision, artifacts = _apply_real_market_read_artifacts(
                root,
                files,
                _trade_decision(),
            )
            snapshot = json.loads(files["snapshot"].read_text())
            snapshot["quotes"]["EUR_USD"]["ask"] = 1.1722
            files["snapshot"].write_text(json.dumps(snapshot))
            brain = _brain(
                root,
                files,
                decision,
                market_read_artifact_validation_required=True,
                market_read_artifact_paths=artifacts,
            )

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("MARKET_READ_EVIDENCE_PACKET_STALE", codes)

    def test_verifier_snapshot_path_overrides_packet_broker_copy_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            packet_snapshot = root / "packet-selected-fresh-copy.json"
            packet_snapshot.write_bytes(files["snapshot"].read_bytes())
            decision, artifacts = _apply_real_market_read_artifacts(
                root,
                files,
                _trade_decision(),
                broker_snapshot_source_path=packet_snapshot,
            )
            brain = _brain(
                root,
                files,
                decision,
                market_read_artifact_validation_required=True,
                market_read_artifact_paths=artifacts,
            )

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("MARKET_READ_EVIDENCE_PACKET_STALE", codes)

    def test_trade_rejects_overlay_that_skips_latest_resolved_prediction_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision, artifacts = _apply_real_market_read_artifacts(
                root,
                files,
                _trade_decision(),
            )
            prediction_id = "mr2:" + "9" * 64
            resolved = {
                "schema_version": 2,
                "prediction_id": prediction_id,
                "score_eligible": True,
                "source_snapshot_conflict": False,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "pair": "EUR_USD",
                "direction": "LONG",
                "action": "TRADE",
                "verdict": "FULL_READ_INCOMPLETE",
                "horizon_results": {
                    "30m": {
                        "resolution_status": "RESOLVED_MID_CANDLE_DIAGNOSTIC",
                        "direction_status": "WRONG",
                        "target_completion_status": "NOT_TOUCHED",
                        "invalidation_status": "TOUCHED",
                        "first_touch_status": "INVALIDATION_ONLY",
                        "full_read_status": "INVALIDATION_ONLY",
                    },
                    "2h": {"resolution_status": "UNRESOLVED"},
                },
            }
            predictions = root / "market_read_predictions.jsonl"
            predictions.write_text(json.dumps(resolved) + "\n", encoding="utf-8")
            sources = _market_read_artifact_sources(root, files)
            refreshed_at = datetime.now(timezone.utc)
            prepare_market_read_baseline(
                baseline_path=artifacts["baseline"],
                packet_path=artifacts["packet"],
                evidence_sources=sources,
                now=refreshed_at,
            )
            packet = json.loads(artifacts["packet"].read_text())
            overlay = json.loads(artifacts["overlay"].read_text())
            overlay["evidence_packet_sha256"] = packet["evidence_packet_sha256"]
            artifacts["overlay"].write_text(json.dumps(overlay))
            brain = _brain(
                root,
                files,
                decision,
                market_read_artifact_validation_required=True,
                market_read_artifact_paths=artifacts,
            )

            with patch(
                "quant_rabbit.gpt_trader.refresh_market_read_measurements",
                return_value={
                    "status": "NO_CHANGE",
                    "read_only_measurement": True,
                    "live_permission": False,
                    "may_change_execution_permission": False,
                },
            ):
                summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("MARKET_READ_PRIOR_PREDICTION_NOT_REVIEWED", codes)

    def test_real_apply_trade_rejects_overlay_pair_that_differs_from_primary_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            snapshot = json.loads(files["snapshot"].read_text())
            snapshot["quotes"]["GBP_USD"] = {
                "bid": 1.3000,
                "ask": 1.3002,
                "timestamp_utc": snapshot["fetched_at_utc"],
            }
            files["snapshot"].write_text(json.dumps(snapshot))
            foreign_read = _market_read_first(pair="GBP_USD", direction="LONG")
            foreign_read["next_30m_prediction"].update(
                {"target_zone": "1.3020", "invalidation": "1.2980"}
            )
            foreign_read["next_2h_prediction"].update(
                {"target_zone": "1.3040", "invalidation": "1.2970"}
            )
            foreign_read["best_trade_if_forced"].update(
                {"entry": "1.3005", "tp": "1.3040", "sl": "1.2970"}
            )
            decision, artifacts = _apply_real_market_read_artifacts(
                root,
                files,
                _trade_decision(),
                market_read=foreign_read,
            )
            brain = _brain(
                root,
                files,
                decision,
                market_read_artifact_validation_required=True,
                market_read_artifact_paths=artifacts,
            )

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("MARKET_READ_PAIR_ACTION_CONFLICT", codes)

    def test_real_apply_range_read_rejects_trend_continuation_trade(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            range_read = _market_read_first(pair="EUR_USD", direction="LONG")
            for key in ("next_30m_prediction", "next_2h_prediction"):
                range_read[key]["direction"] = "RANGE"
                range_read[key]["target_zone"] = "1.1710 to 1.1730"
                range_read[key]["invalidation"] = "1.1690 to 1.1750"
            decision, artifacts = _apply_real_market_read_artifacts(
                root,
                files,
                _trade_decision(),
                market_read=range_read,
            )
            brain = _brain(
                root,
                files,
                decision,
                market_read_artifact_validation_required=True,
                market_read_artifact_paths=artifacts,
            )

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("MARKET_READ_RANGE_ACTION_CONFLICT", codes)

    def test_real_apply_trade_rejects_naked_currencies_opposite_primary_side(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            market_read = _market_read_first(pair="EUR_USD", direction="LONG")
            market_read["naked_read"]["currency_bought"] = "USD"
            market_read["naked_read"]["currency_sold"] = "EUR"
            decision, artifacts = _apply_real_market_read_artifacts(
                root,
                files,
                _trade_decision(),
                market_read=market_read,
            )
            brain = _brain(
                root,
                files,
                decision,
                market_read_artifact_validation_required=True,
                market_read_artifact_paths=artifacts,
            )

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("MARKET_READ_CURRENCY_ACTION_CONFLICT", codes)

    def test_real_apply_trade_rejects_forced_vehicle_and_execution_geometry_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            market_read = _market_read_first(pair="EUR_USD", direction="LONG")
            market_read["best_trade_if_forced"].update(
                {
                    "vehicle": "STOP",
                    "entry": "1.1726",
                    "tp": "1.1740",
                    "sl": "1.1716",
                }
            )
            decision, artifacts = _apply_real_market_read_artifacts(
                root,
                files,
                _trade_decision(),
                market_read=market_read,
            )
            brain = _brain(
                root,
                files,
                decision,
                market_read_artifact_validation_required=True,
                market_read_artifact_paths=artifacts,
            )

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("MARKET_READ_VEHICLE_ACTION_CONFLICT", codes)
            self.assertIn("MARKET_READ_FORCED_EXECUTION_GEOMETRY_CONFLICT", codes)

    def test_draft_never_mixes_predictive_scout_with_normal_or_scout_basket(self) -> None:
        ordinary_id = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:LIMIT"
        scout_a = "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT"
        scout_b = "failure_trader:USD_JPY:LONG:BREAKOUT_FAILURE:LIMIT"
        scout_packet = {
            "enabled": True,
            "canonical_rule_supported": True,
            "promotion_allowed": False,
            "vehicle_proof_status": "UNPROVEN_PASSIVE_LIMIT",
            "rule_is_vehicle_proof": False,
        }
        packet = {
            "lanes": [
                {
                    "lane_id": ordinary_id,
                    "pair": "EUR_USD",
                    "status": "LIVE_READY",
                    "predictive_scout": {},
                },
                {
                    "lane_id": scout_a,
                    "pair": "USD_CAD",
                    "status": "LIVE_READY",
                    "predictive_scout": scout_packet,
                },
                {
                    "lane_id": scout_b,
                    "pair": "USD_JPY",
                    "status": "LIVE_READY",
                    "predictive_scout": scout_packet,
                },
            ]
        }

        self.assertEqual(
            _draft_candidate_lane_ids(packet, [scout_a, ordinary_id, scout_b]),
            [ordinary_id],
        )
        self.assertEqual(
            _draft_candidate_lane_ids(packet, [scout_a, scout_b]),
            [scout_a],
        )
        lanes_by_id = {str(item["lane_id"]): item for item in packet["lanes"]}
        self.assertEqual(
            _draft_margin_aware_basket(packet, [scout_a, scout_b], lanes_by_id),
            (scout_a,),
        )

    def test_post_stop_thesis_review_marks_noise_stop_for_reentry(self) -> None:
        review = post_stop_thesis_review(
            {
                "trade_id": "t-stop",
                "pair": "USD_JPY",
                "side": "SHORT",
                "gateway_action": "STOP_LOSS_ORDER",
                "realized_pl_jpy": -180.0,
                "post_close_favorable_pips": 18.0,
                "sl_lint": {
                    "issues": [
                        {"code": "SL_LINT_MAJOR_FIGURE_BATTLE_ZONE", "severity": "BLOCK"}
                    ]
                },
            }
        )

        self.assertFalse(review["thesis_failed"])
        self.assertTrue(review["price_later_moved_intended_direction"])
        self.assertTrue(review["broker_sl_failure"])
        self.assertTrue(review["sl_inside_noise_or_battle_zone"])
        self.assertEqual(review["next_cycle_action"], "RE_ENTER")

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
            self.assertEqual(summary.capital_allocation_size_multiple, 1.0)
            payload = json.loads((root / "gpt_decision.json").read_text())
            self.assertEqual(payload["verification_issues"], [])
            self.assertIn("GPT Trader Decision Report", (root / "gpt_decision.md").read_text())

    def test_rejects_deterministic_trade_without_codex_market_read_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision = _trade_decision()
            decision["decision_provenance"] = None
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("AI_MARKET_READ_REQUIRED", codes)

    def test_rejects_missing_or_invalid_guardian_receipt_provenance_fields(self) -> None:
        cases = (
            ("guardian_action_receipt_material_contract", "missing", None),
            (
                "guardian_action_receipt_material_contract",
                "invalid_value",
                "WRONG_CONTRACT",
            ),
            ("guardian_action_receipt_baseline_pairs", "missing", None),
            ("guardian_action_receipt_baseline_pairs", "invalid_type", "EUR_USD"),
            (
                "guardian_action_receipt_baseline_pairs",
                "selected_pair_mismatch",
                ["GBP_USD"],
            ),
            ("guardian_action_receipt_scope_state_sha256", "missing", None),
            (
                "guardian_action_receipt_scope_state_sha256",
                "invalid_value",
                "not-a-sha256",
            ),
        )
        for field, mutation, invalid_value in cases:
            with (
                self.subTest(field=field, mutation=mutation),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                files = _fixtures(root)
                decision = _trade_decision()
                _stamp_codex_market_read(decision)
                provenance = decision["decision_provenance"]
                if mutation == "missing":
                    provenance.pop(field)
                else:
                    provenance[field] = invalid_value
                brain = _brain(root, files, decision)

                summary = brain.run(snapshot_path=files["snapshot"])

                self.assertEqual(summary.status, "REJECTED")
                payload = json.loads((root / "gpt_decision.json").read_text())
                issues = payload["verification_issues"]
                self.assertIn(
                    "AI_MARKET_READ_PROVENANCE_INVALID",
                    {issue["code"] for issue in issues},
                )
                self.assertTrue(
                    any(field in issue["message"] for issue in issues),
                    msg=issues,
                )

    def test_gpt_schema_allows_guardian_receipt_provenance_fields(self) -> None:
        provenance_schema = GPT_TRADER_SCHEMA["properties"]["decision_provenance"]
        self.assertFalse(provenance_schema["additionalProperties"])
        properties = provenance_schema["properties"]
        self.assertEqual(
            properties["guardian_action_receipt_material_contract"],
            {"type": "string"},
        )
        self.assertEqual(
            properties["guardian_action_receipt_baseline_pairs"],
            {"type": "array", "items": {"type": "string"}},
        )
        self.assertEqual(
            properties["guardian_action_receipt_scope_state_sha256"],
            {"type": "string"},
        )

    def test_accepts_codex_market_read_veto_of_a_deterministic_trade(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision = _wait_decision()
            _stamp_codex_market_read(
                decision,
                disposition="VETO_WAIT",
                baseline_action="TRADE",
                baseline_lane_ids=[LANE_ID],
            )
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            self.assertEqual(payload["verification_issues"], [])
            self.assertEqual(payload["decision"]["market_read_disposition"], "VETO_WAIT")

    def test_records_market_read_prediction_every_verified_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            brain = _brain(root, files, _trade_decision())

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            rows = [
                json.loads(line)
                for line in (root / "market_read_predictions.jsonl").read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["schema_version"], 2)
            self.assertEqual(rows[0]["pair"], "EUR_USD")
            self.assertEqual(rows[0]["direction"], "LONG")
            self.assertIn(
                rows[0]["verdict"],
                {"UNRESOLVED", "NOT_APPLICABLE_CLOSED_MARKET_WINDOW"},
            )
            self.assertEqual(rows[0]["truth_source"], "MID_CANDLE_DIAGNOSTIC")
            self.assertFalse(rows[0]["live_permission"])
            payload = json.loads((root / "gpt_decision.json").read_text())
            self.assertTrue(
                payload["market_read_prediction"]["decision_receipt_id"].startswith("gptd:")
            )
            report = (root / "market_read_score_report.md").read_text()
            self.assertIn("Market Read Score Report", report)
            self.assertIn("Direction accuracy", report)

    def test_next_input_packet_includes_read_only_market_read_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            brain = _brain(root, files, _trade_decision())
            brain.run(snapshot_path=files["snapshot"])

            packet = brain._input_packet(files["snapshot"])

            feedback = packet["market_read_feedback"]
            self.assertEqual(feedback["metrics"]["rows"], 1)
            self.assertTrue(feedback["read_only"])
            self.assertTrue(feedback["advisory_only"])
            self.assertFalse(feedback["live_permission"])
            self.assertFalse(feedback["may_change_execution_permission"])
            self.assertIn(feedback["measurement_refresh"]["status"], {"NO_CHANGE", "REFRESHED"})
            self.assertFalse(feedback["measurement_refresh"]["live_permission"])
            self.assertIn("market_read:feedback", packet["allowed_evidence_refs"])

    def test_input_packet_exposes_sibling_market_read_artifact_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            brain = _brain(root, files, _trade_decision())

            packet = brain._input_packet(files["snapshot"])

            contract = packet["market_read_artifact_contract"]
            self.assertEqual(contract["required_for_action"], "TRADE_CLOSE_OR_VETO")
            self.assertEqual(
                contract["required_for_actions"],
                ["TRADE", "CLOSE", "WAIT", "REQUEST_EVIDENCE"],
            )
            self.assertEqual(
                contract["baseline_path"],
                str(root / "trader_decision_baseline.json"),
            )
            self.assertEqual(
                contract["evidence_packet_path"],
                str(root / "market_read_evidence_packet.json"),
            )
            self.assertEqual(
                contract["overlay_path"],
                str(root / "codex_market_read_overlay.json"),
            )
            self.assertEqual(
                contract["evidence_source_paths"]["broker_snapshot"],
                str(files["snapshot"]),
            )
            self.assertEqual(
                contract["evidence_source_paths"]["order_intents"],
                str(files["intents"]),
            )
            self.assertEqual(
                contract["evidence_source_paths"]["guardian_action_receipt"],
                str(files["guardian_action_receipt"]),
            )
            self.assertFalse(contract["live_permission"])
            self.assertFalse(contract["may_grant_gateway_permission"])

    def test_input_packet_keeps_broker_quote_when_no_lane_or_position_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["intents"].write_text(json.dumps({"results": []}))
            files["campaign"].write_text(json.dumps({"lanes": []}))
            brain = _brain(root, files, _wait_decision())

            packet = brain._input_packet(files["snapshot"])

            self.assertEqual(packet["lanes"], [])
            self.assertEqual(packet["broker_snapshot"]["positions"], 0)
            self.assertEqual(
                packet["broker_snapshot"]["quotes"]["EUR_USD"]["bid"],
                1.172,
            )
            self.assertIn("EUR_USD", packet["market_context"]["pairs"])

    def test_rejects_decision_without_market_read_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision = _trade_decision()
            decision.pop("market_read_first")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("MARKET_READ_FIRST_MISSING", codes)

    def test_rejects_blocker_before_market_read_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision = _trade_decision()
            generated_at = decision.pop("generated_at_utc")
            market_read = decision.pop("market_read_first")
            decision.pop("risk_notes")
            blocker_first = {
                "generated_at_utc": generated_at,
                "risk_notes": ["NO_LIVE_READY_LANES blocker stated before the market read."],
                "market_read_first": market_read,
            }
            blocker_first.update(decision)
            brain = _brain(root, files, blocker_first)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("BLOCKER_BEFORE_MARKET_READ", codes)

    def test_rejects_trade_when_market_read_short_has_long_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision = _trade_decision()
            decision["market_read_first"] = _market_read_first(pair="EUR_USD", direction="SHORT")
            decision["thesis"] = (
                "MARKET READ FIRST next 30m/next 2h EUR_USD SHORT path conflicts with the selected LONG lane."
            )
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("MARKET_READ_DIRECTION_ACTION_CONFLICT", codes)
            self.assertIn("MARKET_READ_TARGET_GEOMETRY_CONFLICT", codes)
            self.assertIn("MARKET_READ_INVALIDATION_GEOMETRY_CONFLICT", codes)
            self.assertIn("MARKET_READ_FORCED_TRADE_GEOMETRY_CONFLICT", codes)

    def test_rejects_request_evidence_when_market_read_geometry_is_inverted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision = _wait_decision()
            decision["action"] = "REQUEST_EVIDENCE"
            decision["market_read_first"] = _market_read_first(
                pair="EUR_USD",
                direction="SHORT",
            )
            decision["thesis"] = (
                "MARKET READ FIRST next 30m/next 2h EUR_USD SHORT is recorded before requesting evidence."
            )
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("MARKET_READ_TARGET_GEOMETRY_CONFLICT", codes)
            self.assertIn("MARKET_READ_INVALIDATION_GEOMETRY_CONFLICT", codes)
            self.assertIn("MARKET_READ_FORCED_TRADE_GEOMETRY_CONFLICT", codes)
            rows = [
                json.loads(line)
                for line in (root / "market_read_predictions.jsonl").read_text().splitlines()
                if line.strip()
            ]
            self.assertFalse(rows[-1]["score_eligible"])

    def test_rejects_live_ready_zero_without_naked_market_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision = _wait_decision()
            decision.pop("market_read_first")
            decision["risk_notes"] = ["LIVE_READY=0 is why we wait."]
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("MARKET_READ_FIRST_MISSING", codes)
            self.assertIn("LIVE_READY_ZERO_WITHOUT_MARKET_READ", codes)

    def test_rejects_negative_expectancy_as_market_prediction_substitute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision = _wait_decision()
            decision.pop("market_read_first")
            decision["thesis"] = "NEGATIVE_EXPECTANCY blocks this entry, so no market prediction is needed."
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("NEGATIVE_EXPECTANCY_WITHOUT_MARKET_READ", codes)

    def test_accepts_user_alpha_matching_trade_when_continuation_is_cited(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["trader_overrides"].write_text(json.dumps(_user_alpha_overrides()))
            decision = _trade_decision()
            decision["evidence_refs"].extend(
                ["user_alpha:continuation", "user_alpha:latest", "user_alpha:EUR_USD:LONG"]
            )
            decision["twenty_minute_plan"]["evidence_refs"].append("user_alpha:continuation")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertNotIn("USER_ALPHA_CONTINUATION_UNADDRESSED", codes)
            self.assertTrue(payload["input_packet"]["user_alpha_continuation"]["active"])
            report = (root / "gpt_decision.md").read_text()
            self.assertIn("## USER ALPHA CONTINUATION", report)
            self.assertIn("`OPERATOR_ALPHA` `EUR_USD` `LONG`", report)

    def test_rejects_negative_expectancy_wait_that_ignores_user_alpha(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["trader_overrides"].write_text(json.dumps(_user_alpha_overrides()))
            files["capture_economics"].write_text(json.dumps({"status": "NEGATIVE_EXPECTANCY"}))
            brain = _brain(root, files, _wait_decision())

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("USER_ALPHA_CONTINUATION_EVIDENCE_MISSING", codes)
            self.assertIn("USER_ALPHA_CONTINUATION_UNADDRESSED", codes)
            self.assertTrue(payload["input_packet"]["user_alpha_continuation"]["active"])

    def test_drafts_live_ready_trade_receipt_for_scheduled_trader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)

            summary = _draft(root, files)

            self.assertEqual(summary.status, "DRAFT_REQUIRES_OPERATOR_REVIEW")
            self.assertIn("AI_MARKET_READ_REQUIRED", summary.verification_issues)
            self.assertEqual(summary.action, "TRADE")
            self.assertEqual(summary.selected_lane_ids, (LANE_ID,))
            decision = json.loads((root / "codex_trader_decision_response.json").read_text())
            self.assertIn("news:health", decision["evidence_refs"])
            self.assertIn("news:items", decision["evidence_refs"])
            brain = _brain(root, files, decision)
            verified = brain.run(snapshot_path=files["snapshot"])
            self.assertEqual(verified.status, "ACCEPTED")

    def test_draft_with_multiple_live_ready_pairs_binds_one_primary_market_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            eur_jpy_lane = "trend_trader:EUR_JPY:LONG:TREND_CONTINUATION"
            gbp_usd_lane = "trend_trader:GBP_USD:SHORT:TREND_CONTINUATION"
            files["intents"].write_text(
                json.dumps(
                    {
                        "results": [
                            _result(),
                            _result(lane_id=eur_jpy_lane, pair="EUR_JPY"),
                            _result(
                                lane_id=gbp_usd_lane,
                                pair="GBP_USD",
                                side="SHORT",
                            ),
                        ]
                    }
                )
            )

            summary = _draft(root, files)

            self.assertEqual(summary.action, "TRADE")
            self.assertEqual(summary.selected_lane_id, LANE_ID)
            self.assertEqual(summary.selected_lane_ids, (LANE_ID,))
            baseline = json.loads(
                (root / "codex_trader_decision_response.json").read_text()
            )
            self.assertEqual(baseline["selected_lane_ids"], [LANE_ID])
            self.assertEqual(
                baseline["market_read_first"]["best_trade_if_forced"]["pair"],
                "EUR_USD",
            )

            decision, artifacts = _apply_real_market_read_artifacts(
                root,
                files,
                baseline,
            )
            brain = _brain(
                root,
                files,
                decision,
                market_read_artifact_validation_required=True,
                market_read_artifact_paths=artifacts,
            )
            with patch(
                "quant_rabbit.gpt_trader.refresh_market_read_measurements",
                return_value={
                    "status": "NO_CHANGE",
                    "read_only_measurement": True,
                    "live_permission": False,
                    "may_change_execution_permission": False,
                },
            ):
                verified = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(verified.status, "ACCEPTED")
            self.assertEqual(verified.selected_lane_ids, (LANE_ID,))
            payload = json.loads((root / "gpt_decision.json").read_text())
            blocking_codes = {
                issue["code"]
                for issue in payload["verification_issues"]
                if issue.get("severity", "BLOCK") == "BLOCK"
            }
            self.assertEqual(blocking_codes, set())

    def test_draft_uses_active_non_eurusd_lane_for_no_live_ready_market_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            aud_lane = _result(
                lane_id="failure_trader:AUD_CHF:SHORT:BREAKOUT_FAILURE:LIMIT",
                method="BREAKOUT_FAILURE",
                pair="AUD_CHF",
                side="SHORT",
            )
            aud_lane["status"] = "DRY_RUN_BLOCKED"
            aud_lane["live_blockers"] = ["NO_LIVE_READY_LANES"]
            usd_cad_lane_id = "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT"
            files["intents"].write_text(json.dumps({"results": [aud_lane]}))
            snapshot = json.loads(files["snapshot"].read_text())
            snapshot["quotes"]["USD_CAD"] = {
                "bid": 1.3712,
                "ask": 1.3714,
                "timestamp_utc": snapshot["fetched_at_utc"],
            }
            files["snapshot"].write_text(json.dumps(snapshot))
            active_lane = {
                "lane_id": usd_cad_lane_id,
                "pair": "USD_CAD",
                "direction": "LONG",
                "strategy_family": "BREAKOUT_FAILURE",
                "vehicle": "LIMIT",
                "status": "BLOCKED_EVIDENCE_ACQUISITION",
                "entry": 1.3712,
                "tp": 1.3748,
                "sl": 1.3691,
                "blockers": [
                    "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
                    "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                ],
                "next_action": "Refresh USD_CAD pattern and TP proof before live permission.",
            }
            files["active_opportunity_board"].write_text(
                json.dumps(
                    {
                        "status": "BOARD_BUILT_ACTIVE_PATH_AVAILABLE_READ_ONLY",
                        "live_permission_allowed": False,
                        "top_lane": active_lane,
                        "next_active_path": "ENTRY_FREQUENCY_RECOVERY",
                    }
                )
            )
            files["active_trader_contract"].write_text(
                json.dumps(
                    {
                        "status": "ACTIVE_PATH_SELECTED_REPLAY_PASSED_STILL_BLOCKED",
                        "selected_active_path": "EVIDENCE_ACQUISITION",
                        "live_permission_allowed": False,
                        "next_trade_enabling_action": "Use latest active board top lane USD_CAD LONG LIMIT.",
                    }
                )
            )
            files["non_eurusd_frontier"].write_text(
                json.dumps(
                    {
                        "status": "ONLY_EURUSD_FRONTIER_FOUND",
                        "live_permission_allowed": False,
                        "top_non_eurusd_lane": active_lane,
                        "next_active_path": "TP_PROOF_COLLECTION_USD_CAD_LONG_LIMIT",
                    }
                )
            )

            summary = _draft(root, files)

            self.assertEqual(summary.action, "REQUEST_EVIDENCE")
            decision = json.loads((root / "codex_trader_decision_response.json").read_text())
            self.assertEqual(decision["market_read_first"]["best_trade_if_forced"]["pair"], "USD_CAD")
            self.assertEqual(decision["market_read_first"]["best_trade_if_forced"]["direction"], "LONG")
            self.assertIn("active:board", decision["evidence_refs"])
            self.assertIn(f"active:lane:{usd_cad_lane_id}", decision["evidence_refs"])
            self.assertIn(usd_cad_lane_id, " ".join(decision["risk_notes"]))
            self.assertIsNone(decision["specialist_reviews"][0]["lane_id"])

    def test_draft_uses_forecast_direction_when_blocked_lane_points_the_other_way(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            lane = _result(
                lane_id="trend_trader:AUD_JPY:SHORT:TREND_CONTINUATION",
                method="TREND_CONTINUATION",
                pair="AUD_JPY",
                side="SHORT",
                metadata={
                    "forecast_direction": "UP",
                    "forecast_confidence": 0.20,
                    "forecast_target_price": 112.444,
                    "forecast_invalidation_price": 112.203,
                },
            )
            lane["status"] = "DRY_RUN_BLOCKED"
            lane["live_blockers"] = ["FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE"]
            lane["intent"].update(
                {
                    "entry": 112.284,
                    "tp": 111.423,
                    "sl": 112.706,
                }
            )
            files["intents"].write_text(json.dumps({"results": [lane]}))
            snapshot = json.loads(files["snapshot"].read_text())
            snapshot["quotes"]["AUD_JPY"] = {
                "bid": 112.323,
                "ask": 112.325,
                "timestamp_utc": snapshot["fetched_at_utc"],
            }
            files["snapshot"].write_text(json.dumps(snapshot))

            summary = _draft(root, files)

            self.assertEqual(summary.action, "REQUEST_EVIDENCE")
            decision = json.loads((root / "codex_trader_decision_response.json").read_text())
            market_read = decision["market_read_first"]
            self.assertEqual(market_read["naked_read"]["currency_bought"], "AUD")
            self.assertEqual(market_read["naked_read"]["currency_sold"], "JPY")
            self.assertEqual(market_read["next_30m_prediction"]["direction"], "LONG")
            self.assertEqual(market_read["next_30m_prediction"]["target_zone"], "112.444")
            self.assertEqual(market_read["next_30m_prediction"]["invalidation"], "112.203")
            forced = market_read["best_trade_if_forced"]
            self.assertEqual(forced["direction"], "LONG")
            self.assertEqual(forced["vehicle"], "MARKET")
            self.assertEqual(forced["entry"], "112.324")
            self.assertEqual(forced["tp"], "112.444")
            self.assertEqual(forced["sl"], "112.203")

    def test_draft_focuses_parallel_non_eurusd_frontier_when_board_top_is_eurusd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            eur_lane = _result(
                lane_id="failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
                method="BREAKOUT_FAILURE",
                pair="EUR_USD",
                side="LONG",
            )
            eur_lane["status"] = "DRY_RUN_BLOCKED"
            eur_lane["live_blockers"] = ["NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION"]
            files["intents"].write_text(json.dumps({"results": [eur_lane]}))
            snapshot = json.loads(files["snapshot"].read_text())
            snapshot["quotes"]["AUD_CAD"] = {
                "bid": 0.8891,
                "ask": 0.8893,
                "timestamp_utc": snapshot["fetched_at_utc"],
            }
            files["snapshot"].write_text(json.dumps(snapshot))
            board_lane = {
                "lane_id": "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
                "pair": "EUR_USD",
                "direction": "LONG",
                "strategy_family": "BREAKOUT_FAILURE",
                "vehicle": "LIMIT",
                "status": "EVIDENCE_ACQUISITION",
                "blockers": ["NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION"],
                "next_action": "Collect exact local TAKE_PROFIT_ORDER proof for EUR_USD LONG LIMIT.",
            }
            frontier_lane = {
                "lane_id": "range_trader:AUD_CAD:SHORT:RANGE_ROTATION",
                "pair": "AUD_CAD",
                "direction": "SHORT",
                "strategy_family": "RANGE_ROTATION",
                "vehicle": "LIMIT",
                "status": "EVIDENCE_ACQUISITION",
                "bidask_status": "NEGATIVE",
                "blockers": [
                    "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
                    "LOCAL_TP_PROOF_ZERO_TRADES",
                ],
                "next_action": "Repair bid/ask-negative pattern before rerunning replay.",
            }
            files["active_opportunity_board"].write_text(
                json.dumps(
                    {
                        "status": "BOARD_BUILT_ACTIVE_PATH_AVAILABLE_READ_ONLY",
                        "live_permission_allowed": False,
                        "top_lane": board_lane,
                        "next_active_path": "EVIDENCE_ACQUISITION: EUR_USD proof collection.",
                    }
                )
            )
            files["non_eurusd_frontier"].write_text(
                json.dumps(
                    {
                        "status": "NON_EURUSD_FRONTIER_FOUND",
                        "live_permission_allowed": False,
                        "top_lane": board_lane,
                        "top_non_eurusd_lane": frontier_lane,
                        "next_evidence_lane": frontier_lane,
                        "next_active_path": "BIDASK_NEGATIVE_PATTERN_REPAIR: AUD_CAD.",
                    }
                )
            )
            files["active_trader_contract"].write_text(
                json.dumps(
                    {
                        "status": "ACTIVE_PATH_SELECTED_REPLAY_PASSED_STILL_BLOCKED",
                        "selected_active_path": "EVIDENCE_ACQUISITION",
                        "live_permission_allowed": False,
                        "next_trade_enabling_action": (
                            "Use active board top lane EUR_USD, and parallel "
                            "non_eurusd_live_grade_frontier evidence lane "
                            "range_trader:AUD_CAD:SHORT:RANGE_ROTATION; keep blockers visible."
                        ),
                        "current_state": {
                            "active_opportunity_board": {"top_lane": board_lane},
                            "non_eurusd_live_grade_frontier": {
                                "top_non_eurusd_lane": frontier_lane,
                                "next_evidence_lane": frontier_lane,
                            },
                        },
                    }
                )
            )

            summary = _draft(root, files)

            self.assertEqual(summary.action, "REQUEST_EVIDENCE")
            decision = json.loads((root / "codex_trader_decision_response.json").read_text())
            self.assertEqual(decision["market_read_first"]["best_trade_if_forced"]["pair"], "AUD_CAD")
            self.assertEqual(decision["market_read_first"]["best_trade_if_forced"]["direction"], "SHORT")
            self.assertIn("active:lane:range_trader:AUD_CAD:SHORT:RANGE_ROTATION", decision["evidence_refs"])
            self.assertIn("range_trader:AUD_CAD:SHORT:RANGE_ROTATION", " ".join(decision["risk_notes"]))
            self.assertEqual(len(decision["specialist_reviews"]), 2)

    def test_draft_ignores_operator_manual_position_for_scheduled_trader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(
                root,
                positions=[
                    {
                        "trade_id": "472987",
                        "pair": "EUR_USD",
                        "side": "SHORT",
                        "units": 30000,
                        "entry_price": 1.14048,
                        "take_profit": 1.1361,
                        "stop_loss": None,
                        "owner": "operator_manual",
                    }
                ],
            )

            summary = _draft(root, files)

            self.assertEqual(summary.status, "DRAFT_REQUIRES_OPERATOR_REVIEW")
            self.assertIn("AI_MARKET_READ_REQUIRED", summary.verification_issues)
            self.assertEqual(summary.action, "TRADE")
            report = (root / "trader_decision_draft.md").read_text()
            self.assertNotIn("non-layerable position EUR_USD SHORT id=472987", report)

    def test_draft_emits_exact_single_close_for_hard_thesis_evolution_sidecar(self) -> None:
        with patch("quant_rabbit.gpt_trader._operator_close_gate_authorized", return_value=False):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                files = _close_fixtures(
                    root,
                    position_side="SHORT",
                    m15_dir="DOWN",
                    h4_dir="DOWN",
                )
                _write_fresh_thesis_evolution_close_recommendation(root, files)

                summary = _draft(root, files)

                self.assertEqual(summary.status, "DRAFT_ACCEPTED", msg=summary)
                self.assertEqual(summary.action, "CLOSE")
                self.assertTrue(summary.verification_allowed)
                decision = json.loads(
                    (root / "codex_trader_decision_response.json").read_text()
                )
                self.assertEqual(decision["close_trade_ids"], ["555"])
                self.assertEqual(decision["selected_lane_ids"], [])
                self.assertEqual(decision["cancel_order_ids"], [])
                self.assertIn("position:evolution:555", decision["evidence_refs"])
                self.assertEqual(
                    decision["market_read_first"]["next_30m_prediction"]["pair"],
                    "EUR_USD",
                )
                self.assertFalse(decision["operator_close_authorized"])
                authored_market_read = _market_read_first(
                    pair="EUR_USD",
                    direction="LONG",
                )
                authored_market_read["next_30m_prediction"].update(
                    {"target_zone": "1.1780", "invalidation": "1.1740"}
                )
                authored_market_read["next_2h_prediction"].update(
                    {"target_zone": "1.1800", "invalidation": "1.1730"}
                )
                authored_market_read["best_trade_if_forced"].update(
                    {
                        "entry": "1.1761",
                        "tp": "1.1780",
                        "sl": "1.1740",
                    }
                )
                applied, artifacts = _apply_real_market_read_artifacts(
                    root,
                    files,
                    decision,
                    market_read=authored_market_read,
                )
                self.assertEqual(applied["action"], "CLOSE")
                self.assertEqual(applied["close_trade_ids"], ["555"])
                verified = _brain(
                    root,
                    files,
                    applied,
                    market_read_artifact_validation_required=True,
                    market_read_artifact_paths=artifacts,
                ).run(snapshot_path=files["snapshot"])
                verification_payload = json.loads(
                    (root / "gpt_decision.json").read_text()
                )
                self.assertEqual(
                    verified.status,
                    "ACCEPTED",
                    msg=verification_payload["verification_issues"],
                )

    def test_close_requires_real_market_read_artifacts_at_final_boundary(self) -> None:
        with patch("quant_rabbit.gpt_trader._operator_close_gate_authorized", return_value=False):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                files = _close_fixtures(
                    root,
                    position_side="SHORT",
                    m15_dir="DOWN",
                    h4_dir="DOWN",
                )
                _write_fresh_thesis_evolution_close_recommendation(root, files)
                draft_summary = _draft(root, files)
                self.assertEqual(draft_summary.action, "CLOSE")
                decision = json.loads(
                    (root / "codex_trader_decision_response.json").read_text()
                )

                verified = _brain(
                    root,
                    files,
                    decision,
                    market_read_artifact_validation_required=True,
                ).run(snapshot_path=files["snapshot"])

                payload = json.loads((root / "gpt_decision.json").read_text())
                codes = {issue["code"] for issue in payload["verification_issues"]}
                self.assertEqual(verified.status, "REJECTED")
                self.assertIn(
                    "AI_MARKET_READ_ARTIFACT_PROVENANCE_MISSING",
                    codes,
                )

    def test_close_rejects_artifact_bound_target_mutation(self) -> None:
        with patch("quant_rabbit.gpt_trader._operator_close_gate_authorized", return_value=False):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                files = _close_fixtures(
                    root,
                    position_side="SHORT",
                    m15_dir="DOWN",
                    h4_dir="DOWN",
                )
                _write_fresh_thesis_evolution_close_recommendation(root, files)
                _draft(root, files)
                baseline = json.loads(
                    (root / "codex_trader_decision_response.json").read_text()
                )
                authored_market_read = _market_read_first(
                    pair="EUR_USD",
                    direction="LONG",
                )
                authored_market_read["next_30m_prediction"].update(
                    {"target_zone": "1.1780", "invalidation": "1.1740"}
                )
                authored_market_read["next_2h_prediction"].update(
                    {"target_zone": "1.1800", "invalidation": "1.1730"}
                )
                authored_market_read["best_trade_if_forced"].update(
                    {
                        "entry": "1.1761",
                        "tp": "1.1780",
                        "sl": "1.1740",
                    }
                )
                applied, artifacts = _apply_real_market_read_artifacts(
                    root,
                    files,
                    baseline,
                    market_read=authored_market_read,
                )
                applied["close_trade_ids"] = ["555", "555"]

                verified = _brain(
                    root,
                    files,
                    applied,
                    market_read_artifact_validation_required=True,
                    market_read_artifact_paths=artifacts,
                ).run(snapshot_path=files["snapshot"])

                payload = json.loads((root / "gpt_decision.json").read_text())
                codes = {issue["code"] for issue in payload["verification_issues"]}
                self.assertEqual(verified.status, "REJECTED")
                self.assertIn("AI_MARKET_READ_ARTIFACT_FINAL_MISMATCH", codes)
                self.assertIn("CLOSE_SINGLE_TRADE_REQUIRED", codes)

    def test_close_rejects_multi_target_entry_and_cancel_scope_without_artifacts(self) -> None:
        with patch("quant_rabbit.gpt_trader._operator_close_gate_authorized", return_value=False):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                files = _close_fixtures(
                    root,
                    position_side="SHORT",
                    m15_dir="DOWN",
                    h4_dir="DOWN",
                )
                _write_fresh_thesis_evolution_close_recommendation(root, files)
                _draft(root, files)
                decision = json.loads(
                    (root / "codex_trader_decision_response.json").read_text()
                )
                decision["close_trade_ids"] = ["555", "555"]
                decision["selected_lane_id"] = LANE_ID
                decision["selected_lane_ids"] = [LANE_ID]
                decision["cancel_order_ids"] = ["123"]

                verified = _brain(root, files, decision).run(
                    snapshot_path=files["snapshot"]
                )

                payload = json.loads((root / "gpt_decision.json").read_text())
                codes = {issue["code"] for issue in payload["verification_issues"]}
                self.assertEqual(verified.status, "REJECTED")
                self.assertIn("CLOSE_SINGLE_TRADE_REQUIRED", codes)
                self.assertIn("CLOSE_SELECTED_LANE_FORBIDDEN", codes)
                self.assertIn("CLOSE_CANCEL_ORDER_IDS_FORBIDDEN", codes)

    def test_draft_emits_exact_single_close_for_soft_sidecar_with_explicit_gate_b(self) -> None:
        with patch("quant_rabbit.gpt_trader._operator_close_gate_authorized", return_value=True):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                files = _close_fixtures(
                    root,
                    position_side="SHORT",
                    m15_dir="DOWN",
                    h4_dir="DOWN",
                )
                _write_fresh_forecast_close_recommendation(root, files)

                summary = _draft(root, files)

                self.assertEqual(summary.status, "DRAFT_ACCEPTED", msg=summary)
                self.assertEqual(summary.action, "CLOSE")
                decision = json.loads(
                    (root / "codex_trader_decision_response.json").read_text()
                )
                self.assertEqual(decision["close_trade_ids"], ["555"])
                self.assertIn("position:persistence:555", decision["evidence_refs"])
                self.assertTrue(decision["operator_close_authorized"])

    def test_draft_does_not_close_soft_sidecar_without_explicit_gate_b(self) -> None:
        with patch("quant_rabbit.gpt_trader._operator_close_gate_authorized", return_value=False):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                files = _close_fixtures(
                    root,
                    position_side="SHORT",
                    m15_dir="DOWN",
                    h4_dir="DOWN",
                )
                _write_fresh_forecast_close_recommendation(root, files)

                summary = _draft(root, files)

                self.assertNotEqual(summary.action, "CLOSE")
                decision = json.loads(
                    (root / "codex_trader_decision_response.json").read_text()
                )
                self.assertEqual(decision["close_trade_ids"], [])

    def test_draft_does_not_close_hard_sidecar_downgraded_by_hold_conflict(self) -> None:
        with patch("quant_rabbit.gpt_trader._operator_close_gate_authorized", return_value=False):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                files = _close_fixtures(
                    root,
                    position_side="SHORT",
                    m15_dir="DOWN",
                    h4_dir="DOWN",
                )
                _write_fresh_position_hold_support(root, files)
                _write_fresh_thesis_evolution_close_recommendation(root, files)

                summary = _draft(root, files)

                self.assertNotEqual(summary.action, "CLOSE")
                decision = json.loads(
                    (root / "codex_trader_decision_response.json").read_text()
                )
                self.assertEqual(decision["close_trade_ids"], [])

    def test_draft_never_targets_operator_manual_position_from_hard_sidecar(self) -> None:
        with patch("quant_rabbit.gpt_trader._operator_close_gate_authorized", return_value=False):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                files = _fixtures(
                    root,
                    positions=[
                        {
                            "trade_id": "472987",
                            "pair": "EUR_USD",
                            "side": "SHORT",
                            "units": 30000,
                            "entry_price": 1.14048,
                            "take_profit": 1.1361,
                            "stop_loss": None,
                            "owner": "operator_manual",
                        }
                    ],
                )
                _write_fresh_thesis_evolution_close_recommendation(
                    root,
                    files,
                    trade_id="472987",
                )

                summary = _draft(root, files)

                self.assertNotEqual(summary.action, "CLOSE")
                decision = json.loads(
                    (root / "codex_trader_decision_response.json").read_text()
                )
                self.assertEqual(decision["close_trade_ids"], [])

    def test_draft_refuses_deterministic_mass_close_when_two_trades_are_authorized(self) -> None:
        with patch("quant_rabbit.gpt_trader._operator_close_gate_authorized", return_value=False):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                files = _close_fixtures(
                    root,
                    position_side="SHORT",
                    m15_dir="DOWN",
                    h4_dir="DOWN",
                )
                snapshot = json.loads(files["snapshot"].read_text())
                second = dict(snapshot["positions"][0])
                second["trade_id"] = "556"
                snapshot["positions"].append(second)
                files["snapshot"].write_text(json.dumps(snapshot))
                generated_at = (
                    datetime.fromisoformat(snapshot["fetched_at_utc"])
                    + timedelta(seconds=1)
                ).isoformat()
                (root / "thesis_evolution_report.json").write_text(
                    json.dumps(
                        {
                            "generated_at_utc": generated_at,
                            "evolutions": [
                                {
                                    "trade_id": trade_id,
                                    "pair": "EUR_USD",
                                    "side": "SHORT",
                                    "status": "BROKEN",
                                    "verdict": "RECOMMEND_CLOSE",
                                    "rationale": (
                                        "invalidation hit: current ask 1.16310 >= buffered "
                                        "invalidation 1.16290 (raw 1.16270, buffer 2.0p); "
                                        "technical invalidation confirmed against SHORT: "
                                        "H1 BOS_UP; H4 BOS_UP"
                                    ),
                                }
                                for trade_id in ("555", "556")
                            ],
                        }
                    )
                )

                summary = _draft(root, files)

                self.assertNotEqual(summary.action, "CLOSE")
                self.assertIn(
                    "CLOSE_BASELINE_SINGLE_TRADE_REQUIRED",
                    " ".join(summary.blockers),
                )
                decision = json.loads(
                    (root / "codex_trader_decision_response.json").read_text()
                )
                self.assertEqual(decision["close_trade_ids"], [])

    def test_draft_classifies_expired_guardian_receipt_before_trade_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            _write_watchdog_guardian_issue(
                files["qr_trader_run_watchdog"],
                lifecycle="EXPIRED",
                action="HOLD",
                event_id="receipt-expired",
                emergency_or_margin_risk=False,
            )

            summary = _draft(root, files)

            self.assertEqual(summary.status, "DRAFT_REQUIRES_OPERATOR_REVIEW")
            self.assertIn("AI_MARKET_READ_REQUIRED", summary.verification_issues)
            self.assertEqual(summary.action, "TRADE")
            consumption = json.loads(files["guardian_receipt_consumption"].read_text())
            self.assertTrue(consumption["normal_routing_allowed"])
            self.assertEqual(consumption["classifications"][0]["classification"], "EXPIRED_ACKNOWLEDGED")
            self.assertFalse(consumption["classifications"][0]["consumed_by_trader"])
            self.assertTrue(files["guardian_receipt_consumption_report"].exists())

    def test_draft_records_reduce_receipt_operator_review_before_normal_routing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            _write_watchdog_guardian_issue(
                files["qr_trader_run_watchdog"],
                lifecycle="EXPIRED",
                action="REDUCE",
                event_id="receipt-expired-reduce",
                emergency_or_margin_risk=False,
            )

            summary = _draft(root, files)

            self.assertEqual(summary.status, "DRAFT_REQUIRES_OPERATOR_REVIEW")
            self.assertEqual(summary.action, "WAIT")
            self.assertIn("GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED", " ".join(summary.blockers))
            consumption = json.loads(files["guardian_receipt_consumption"].read_text())
            self.assertFalse(consumption["normal_routing_allowed"])
            self.assertEqual(consumption["classifications"][0]["classification"], "NEEDS_OPERATOR_REVIEW")
            self.assertEqual(consumption["classifications"][0]["operator_review_status"], "OPERATOR_REVIEW_MISSING")
            self.assertIn(
                "Guardian receipt operator review",
                (root / "trader_decision_draft.md").read_text(),
            )

    def test_draft_blocks_normal_trade_when_guardian_receipt_needs_operator_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            _write_watchdog_guardian_issue(
                files["qr_trader_run_watchdog"],
                lifecycle="EXPIRED",
                action="HOLD",
                event_id="receipt-review",
                emergency_or_margin_risk=True,
            )

            summary = _draft(root, files)

            self.assertEqual(summary.status, "DRAFT_REQUIRES_OPERATOR_REVIEW")
            self.assertEqual(summary.action, "WAIT")
            self.assertIn("GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED", " ".join(summary.blockers))
            consumption = json.loads(files["guardian_receipt_consumption"].read_text())
            self.assertFalse(consumption["normal_routing_allowed"])
            self.assertEqual(consumption["classifications"][0]["classification"], "NEEDS_OPERATOR_REVIEW")
            self.assertFalse(consumption["classifications"][0]["consumed_by_trader"])

    def test_gpt_selected_ordinary_lane_observes_exact_p1_margin_warning_only_below_cap(self) -> None:
        issue = {
            "code": "GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER",
            "severity": "P1",
            "message": "current P1 margin warning",
            "receipt_event_id": "margin-p1-event",
            "receipt_action": "HOLD",
            "receipt_lifecycle": "EXPIRED",
            "consumed_by_trader": False,
            "emergency_or_margin_risk": True,
            "normal_routing_allowed": False,
            "event_type": "MARGIN_PRESSURE",
            "event_severity": "P1",
            "event_action_hint": "HOLD",
            "event_details": {
                "nav_jpy": 100_000.0,
                "margin_used_jpy": 92_000.0,
                "margin_available_jpy": 8_000.0,
                "max_margin_utilization_pct": 95.0,
                "fresh_entry_risk_block_active": False,
                "fresh_entry_risk_block_reason": "MARGIN_PRESSURE",
                "fresh_entry_risk_observation_only": True,
                "fresh_entry_margin_contract": "QR_GUARDIAN_P1_MARGIN_WARNING_V1",
            },
        }
        packet = {
            "qr_trader_run_watchdog": {
                "status": "BLOCKED",
                "runtime_status": "OK",
                "issue_status": "P1",
                "severity": "P1",
                "guardian_receipt_issues": [issue],
            },
            "guardian_receipt_consumption": {
                "status": "CURRENT_P1_BLOCKS_ROUTING",
                "normal_routing_allowed": False,
                "current_p0_p1_blocks_routing": True,
                "classifications": [],
            },
            "guardian_receipt_operator_review": {},
            "broker_snapshot": {
                "account": {
                    "nav_jpy": 100_000.0,
                    "margin_used_jpy": 92_000.0,
                    "margin_available_jpy": 8_000.0,
                }
            },
            "lanes": [
                {
                    "lane_id": LANE_ID,
                    "status": "LIVE_READY",
                    "risk_metrics": {
                        "margin_utilization_after_pct": 94.9,
                        "max_margin_utilization_pct": 95.0,
                    },
                }
            ],
        }

        observed = _guardian_receipt_consumption_trade_routing_issues(
            packet,
            selected_lane_ids=(LANE_ID,),
        )

        self.assertEqual(
            [(item["code"], item["severity"]) for item in observed],
            [("GUARDIAN_P1_MARGIN_PRESSURE_OBSERVED", "WARN")],
        )

        packet["lanes"][0]["risk_metrics"][
            "margin_utilization_after_pct"
        ] = 95.0001
        blocked = _guardian_receipt_consumption_trade_routing_issues(
            packet,
            selected_lane_ids=(LANE_ID,),
        )
        self.assertTrue(any(item["severity"] == "BLOCK" for item in blocked))

        compact = _qr_trader_run_watchdog_packet(
            {
                "status": "BLOCKED",
                "runtime_status": "OK",
                "issue_status": "P1",
                "severity": "P1",
                "issues": [issue],
                "guardian_receipt": {"issues": [issue]},
            }
        )
        for compact_issue in (
            compact["issues"][0],
            compact["guardian_receipt_issues"][0],
        ):
            self.assertEqual(compact_issue["event_type"], "MARGIN_PRESSURE")
            self.assertEqual(compact_issue["event_severity"], "P1")
            self.assertEqual(compact_issue["event_action_hint"], "HOLD")
            self.assertEqual(
                compact_issue["event_details"]["fresh_entry_margin_contract"],
                "QR_GUARDIAN_P1_MARGIN_WARNING_V1",
            )

    def test_rejects_trade_when_active_guardian_receipt_issue_has_no_classification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            _write_watchdog_guardian_issue(
                files["qr_trader_run_watchdog"],
                lifecycle="EXPIRED",
                action="REDUCE",
                event_id="receipt-unclassified",
                emergency_or_margin_risk=False,
            )
            brain = _brain(root, files, _trade_decision())

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED", codes)

    def test_regression_blocks_aud_usd_campaign_recovery_when_reduce_receipt_unconsumed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aud_lane = "campaign_exposure_recovery:AUD_USD:LONG:TREND_CONTINUATION"
            files = _fixtures(root)
            files["intents"].write_text(
                json.dumps(
                    {
                        "results": [
                            _result(
                                lane_id=aud_lane,
                                pair="AUD_USD",
                                side="LONG",
                                metadata={
                                    "desk": "campaign_exposure_recovery",
                                    "campaign_role": "NOW",
                                    "campaign_exposure_recovery": True,
                                },
                            )
                        ]
                    }
                )
            )
            _write_watchdog_guardian_issue(
                files["qr_trader_run_watchdog"],
                lifecycle="EXPIRED",
                action="REDUCE",
                event_id="receipt-stale-reduce",
                emergency_or_margin_risk=False,
            )
            decision = _trade_decision(lane_id=aud_lane, pair="AUD_USD", direction="LONG")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED", codes)
            report = (root / "gpt_decision.md").read_text()
            self.assertIn("GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED", report)

    def test_draft_cites_user_alpha_when_selected_lane_continues_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["trader_overrides"].write_text(json.dumps(_user_alpha_overrides()))

            summary = _draft(root, files)

            self.assertEqual(summary.status, "DRAFT_REQUIRES_OPERATOR_REVIEW")
            self.assertIn("AI_MARKET_READ_REQUIRED", summary.verification_issues)
            decision = json.loads((root / "codex_trader_decision_response.json").read_text())
            self.assertIn("user_alpha:continuation", decision["evidence_refs"])
            self.assertIn("user_alpha:continuation", decision["twenty_minute_plan"]["evidence_refs"])
            self.assertIn("## USER ALPHA CONTINUATION", (root / "trader_decision_draft.md").read_text())

    def test_draft_refuses_trade_when_news_health_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["news_health"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "BLOCK",
                        "issues": ["BLOCK:news_digest_freshness: curated digest is stale"],
                    }
                )
            )

            summary = _draft(root, files)

            self.assertEqual(summary.status, "DRAFT_ACCEPTED")
            self.assertEqual(summary.action, "WAIT")
            self.assertEqual(summary.selected_lane_ids, ())
            decision = json.loads((root / "codex_trader_decision_response.json").read_text())
            self.assertEqual(decision["selected_lane_ids"], [])
            self.assertIn("news:health", decision["evidence_refs"])

    def test_draft_replaces_pending_cancel_review_with_trade_cancel_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, orders=[_pending_order()])
            files["self_improvement_audit"].write_text(
                json.dumps(_self_improvement_pending_cancel_review_p0())
            )

            summary = _draft(root, files)

            self.assertEqual(summary.status, "DRAFT_REQUIRES_OPERATOR_REVIEW")
            self.assertIn("AI_MARKET_READ_REQUIRED", summary.verification_issues)
            self.assertEqual(summary.action, "TRADE")
            self.assertEqual(summary.selected_lane_ids, (LANE_ID,))
            decision = json.loads((root / "codex_trader_decision_response.json").read_text())
            self.assertEqual(decision["cancel_order_ids"], ["pending-1"])
            self.assertIn("self_improvement:audit", decision["evidence_refs"])
            brain = _brain(root, files, decision)
            verified = brain.run(snapshot_path=files["snapshot"])
            self.assertEqual(verified.status, "ACCEPTED")

    def test_draft_cancels_pending_review_when_no_live_ready_replacement_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, orders=[_pending_order()])
            files["intents"].write_text(json.dumps({"results": []}))
            files["self_improvement_audit"].write_text(
                json.dumps(_self_improvement_pending_cancel_review_p0())
            )

            summary = _draft(root, files)

            self.assertEqual(summary.status, "DRAFT_ACCEPTED")
            self.assertEqual(summary.action, "CANCEL_PENDING")
            self.assertEqual(summary.selected_lane_ids, ())
            decision = json.loads((root / "codex_trader_decision_response.json").read_text())
            self.assertEqual(decision["cancel_order_ids"], ["pending-1"])
            self.assertIn("self_improvement:audit", decision["evidence_refs"])
            self.assertIn(
                "self_improvement:finding:PENDING_ENTRY_CANCEL_REVIEW_REQUIRED",
                decision["evidence_refs"],
            )
            brain = _brain(root, files, decision)
            verified = brain.run(snapshot_path=files["snapshot"])
            self.assertEqual(verified.status, "ACCEPTED")

    def test_rejects_trade_when_news_health_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["news_health"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "BLOCK",
                        "issues": ["BLOCK:market_story_news_sync: story is older than news"],
                    }
                )
            )
            brain = _brain(root, files, _trade_decision())

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("NEWS_HEALTH_BLOCKS_TRADE", codes)

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

    def test_rejects_trade_when_order_intents_predate_daily_target_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            snapshot = json.loads(files["snapshot"].read_text())
            snapshot_ts = datetime.fromisoformat(snapshot["fetched_at_utc"])
            intents_ts = snapshot_ts + timedelta(seconds=1)
            target_ts = snapshot_ts + timedelta(seconds=2)
            decision_ts = snapshot_ts + timedelta(seconds=3)
            intents = json.loads(files["intents"].read_text())
            intents["generated_at_utc"] = intents_ts.isoformat()
            files["intents"].write_text(json.dumps(intents))
            target = json.loads(files["target"].read_text())
            target["generated_at_utc"] = target_ts.isoformat()
            files["target"].write_text(json.dumps(target))
            decision = _trade_decision()
            decision["generated_at_utc"] = decision_ts.isoformat()
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            issues = {
                issue["code"]: issue
                for issue in payload["verification_issues"]
            }
            self.assertIn("STALE_ORDER_INTENTS_PACKET", issues)
            self.assertEqual(issues["STALE_ORDER_INTENTS_PACKET"]["severity"], "BLOCK")

    def test_rejects_trade_when_campaign_plan_predates_daily_target_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            snapshot = json.loads(files["snapshot"].read_text())
            snapshot_ts = datetime.fromisoformat(snapshot["fetched_at_utc"])
            campaign_ts = snapshot_ts + timedelta(seconds=1)
            target_ts = snapshot_ts + timedelta(seconds=2)
            intents_ts = snapshot_ts + timedelta(seconds=3)
            decision_ts = snapshot_ts + timedelta(seconds=4)
            campaign = json.loads(files["campaign"].read_text())
            campaign["generated_at_utc"] = campaign_ts.isoformat()
            files["campaign"].write_text(json.dumps(campaign))
            target = json.loads(files["target"].read_text())
            target["generated_at_utc"] = target_ts.isoformat()
            files["target"].write_text(json.dumps(target))
            intents = json.loads(files["intents"].read_text())
            intents["generated_at_utc"] = intents_ts.isoformat()
            files["intents"].write_text(json.dumps(intents))
            decision = _trade_decision()
            decision["generated_at_utc"] = decision_ts.isoformat()
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            issues = {
                issue["code"]: issue
                for issue in payload["verification_issues"]
            }
            self.assertIn("STALE_CAMPAIGN_PLAN_PACKET", issues)
            self.assertEqual(issues["STALE_CAMPAIGN_PLAN_PACKET"]["severity"], "BLOCK")

    def test_allows_cancel_pending_when_order_intents_predate_daily_target_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, orders=[_pending_order()])
            snapshot = json.loads(files["snapshot"].read_text())
            snapshot_ts = datetime.fromisoformat(snapshot["fetched_at_utc"])
            campaign_ts = snapshot_ts + timedelta(seconds=1)
            intents_ts = snapshot_ts + timedelta(seconds=2)
            target_ts = snapshot_ts + timedelta(seconds=3)
            decision_ts = snapshot_ts + timedelta(seconds=4)
            campaign = json.loads(files["campaign"].read_text())
            campaign["generated_at_utc"] = campaign_ts.isoformat()
            files["campaign"].write_text(json.dumps(campaign))
            files["intents"].write_text(
                json.dumps({"generated_at_utc": intents_ts.isoformat(), "results": []})
            )
            target = json.loads(files["target"].read_text())
            target["generated_at_utc"] = target_ts.isoformat()
            files["target"].write_text(json.dumps(target))
            decision = _cancel_pending_decision(cancel_order_ids=["pending-1"])
            decision["generated_at_utc"] = decision_ts.isoformat()
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            self.assertTrue(summary.allowed)
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertNotIn("STALE_ORDER_INTENTS_PACKET", codes)
            self.assertNotIn("STALE_CAMPAIGN_PLAN_PACKET", codes)

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

    def test_rejects_trade_receipt_that_predates_attack_advice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            snapshot = json.loads(files["snapshot"].read_text())
            snapshot_ts = datetime.fromisoformat(snapshot["fetched_at_utc"])
            decision_ts = snapshot_ts + timedelta(seconds=1)
            attack_ts = snapshot_ts + timedelta(seconds=2)
            files["attack_advice"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": attack_ts.isoformat(),
                        "status": "ATTACK_PARTIAL",
                        "recommended_now_lane_ids": [LANE_ID],
                    }
                )
            )
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
            self.assertTrue(any("ai_attack_advice" in message for message in stale_messages))

    def test_rejects_attack_advice_packet_that_predates_order_intents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            snapshot = json.loads(files["snapshot"].read_text())
            snapshot_ts = datetime.fromisoformat(snapshot["fetched_at_utc"])
            attack_ts = snapshot_ts + timedelta(seconds=1)
            intents_ts = snapshot_ts + timedelta(seconds=2)
            decision_ts = snapshot_ts + timedelta(seconds=3)
            intents = json.loads(files["intents"].read_text())
            intents["generated_at_utc"] = intents_ts.isoformat()
            files["intents"].write_text(json.dumps(intents))
            files["attack_advice"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": attack_ts.isoformat(),
                        "status": "NO_ATTACK_ADVICE",
                        "live_ready_lanes": 0,
                        "recommended_now_lane_ids": [],
                    }
                )
            )
            decision = _trade_decision()
            decision["generated_at_utc"] = decision_ts.isoformat()
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("STALE_ATTACK_ADVICE_PACKET", codes)

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

    def test_input_packet_exposes_uncapped_trade_pace_without_changing_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            target = json.loads(files["target"].read_text())
            target.update(
                {
                    "uncapped_required_trades_per_day": 173,
                    "uncapped_required_trades_per_day_basis_return_pct": 10.0,
                    "selected_basis_uncapped_required_trades_per_day": 87,
                    "selected_basis_return_pct": 5.0,
                    "operating_pace_trades_per_day": 30,
                    "automated_operating_cap_trades_per_day": 30,
                    "observed_trades_per_day": 4.7955,
                    "observed_expectancy_jpy_per_trade": 168.6658,
                    "frequency_multiple_required": 36.0755,
                    "planned_reward_at_operating_pace_jpy": 5059.974,
                    "stretch_required_minus_operating_gap_trades_per_day": 143,
                    "selected_required_minus_operating_gap_trades_per_day": 57,
                    "trade_pace_feasible_within_operating_pace": False,
                    "trade_pace_feasibility": "INFEASIBLE_AT_OPERATING_PACE",
                }
            )
            files["target"].write_text(json.dumps(target))

            summary = _brain(root, files, _trade_decision()).run(
                snapshot_path=files["snapshot"]
            )
            payload = json.loads((root / "gpt_decision.json").read_text())
            packet = payload["input_packet"]["daily_target"]

        self.assertEqual(summary.status, "ACCEPTED")
        self.assertTrue(summary.allowed)
        self.assertEqual(packet["uncapped_required_trades_per_day"], 173)
        self.assertEqual(packet["selected_basis_uncapped_required_trades_per_day"], 87)
        self.assertEqual(packet["operating_pace_trades_per_day"], 30)
        self.assertEqual(packet["observed_expectancy_jpy_per_trade"], 168.6658)
        self.assertEqual(packet["planned_reward_at_operating_pace_jpy"], 5059.974)
        self.assertEqual(packet["stretch_required_minus_operating_gap_trades_per_day"], 143)
        self.assertEqual(packet["selected_required_minus_operating_gap_trades_per_day"], 57)
        self.assertEqual(packet["trade_pace_feasibility"], "INFEASIBLE_AT_OPERATING_PACE")

    def test_input_packet_includes_manual_precedent_context_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["operator_precedent"].write_text(json.dumps(_operator_precedent_audit([LANE_ID])))
            files["manual_market_context"].write_text(json.dumps(_manual_market_context_audit()))
            decision = _trade_decision()
            decision["evidence_refs"].extend(["operator:precedent", "manual:market_context"])
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            self.assertEqual(payload["input_packet"]["operator_precedent"]["evidence_ref"], "operator:precedent")
            self.assertEqual(
                payload["input_packet"]["manual_market_context"]["evidence_ref"],
                "manual:market_context",
            )
            self.assertIn("operator:precedent", payload["input_packet"]["allowed_evidence_refs"])
            self.assertIn("manual:market_context", payload["input_packet"]["allowed_evidence_refs"])
            self.assertEqual(
                payload["input_packet"]["manual_market_context"]["guidance"]["prefer_when_citing_precedent"][
                    "h1_alignment"
                ],
                "AGAINST_H1_TREND",
            )
            building = payload["input_packet"]["manual_market_context"]["position_building"]
            self.assertEqual(building["adverse_adds"]["clusters"], 8)
            self.assertEqual(building["adverse_adds"]["net_jpy"], 102564.0)
            self.assertTrue(building["contract"]["nanpin_is_not_live_permission"])

    def test_rejects_trade_citing_operator_precedent_without_manual_context_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["operator_precedent"].write_text(json.dumps(_operator_precedent_audit([LANE_ID])))
            files["manual_market_context"].write_text(json.dumps(_manual_market_context_audit()))
            decision = _trade_decision()
            decision["evidence_refs"].append("operator:precedent")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("MANUAL_CONTEXT_EVIDENCE_MISSING", codes)

    def test_rejects_trade_citing_operator_precedent_for_unaligned_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["operator_precedent"].write_text(
                json.dumps(_operator_precedent_audit(["trend_trader:USD_JPY:LONG:TREND_CONTINUATION"]))
            )
            files["manual_market_context"].write_text(json.dumps(_manual_market_context_audit()))
            decision = _trade_decision()
            decision["evidence_refs"].extend(["operator:precedent", "manual:market_context"])
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("OPERATOR_PRECEDENT_SELECTED_LANE_NOT_ALIGNED", codes)

    def test_rejects_trade_citing_operator_precedent_when_manual_technical_context_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            precedent = _operator_precedent_audit([LANE_ID])
            precedent["runtime_alignment"]["manual_context_alignment"] = {
                "status": "MANUAL_CONTEXT_ALIGNMENT_READY",
                "compatible_lanes": [],
                "conflicting_lanes": [
                    {
                        "lane_id": LANE_ID,
                        "conflicting_buckets": [
                            "by_h1_alignment:WITH_H1_TREND",
                            "by_side_entry_location_24h:LONG_UPPER_THIRD_24H",
                        ],
                    }
                ],
                "conflicting_aligned_lanes": 1,
            }
            files["operator_precedent"].write_text(json.dumps(precedent))
            files["manual_market_context"].write_text(json.dumps(_manual_market_context_audit()))
            decision = _trade_decision()
            decision["evidence_refs"].extend(["operator:precedent", "manual:market_context"])
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("OPERATOR_PRECEDENT_TECHNICAL_CONTEXT_CONFLICT", codes)

    def test_rejects_trade_citing_operator_precedent_for_with_move_same_pair_add(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["operator_precedent"].write_text(json.dumps(_operator_precedent_audit([LANE_ID])))
            files["manual_market_context"].write_text(json.dumps(_manual_market_context_audit()))
            result = _result()
            result["intent"]["metadata"] = {
                "position_intent": "PYRAMID",
                "position_fill": "OPEN_ONLY",
                "same_pair_add_type": "PYRAMID_WITH_MOVE",
                "same_pair_existing_entries": 1,
                "same_pair_existing_units": 10_000,
                "same_pair_existing_avg_entry": 1.1000,
                "same_pair_add_entry": 1.1008,
                "same_pair_with_move_add_pips": 8.0,
            }
            files["intents"].write_text(json.dumps({"results": [result]}))
            decision = _trade_decision()
            decision["evidence_refs"].extend(["operator:precedent", "manual:market_context"])
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("OPERATOR_PRECEDENT_POSITION_BUILDING_CONFLICT", codes)

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

    def test_accepts_trade_for_self_improvement_profitability_p0_repair_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            intents = json.loads(files["intents"].read_text())
            metadata = intents["results"][0]["intent"].setdefault("metadata", {})
            metadata.update(
                {
                    "opportunity_mode": "HARVEST",
                    "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                    "tp_target_intent": "HARVEST",
                    "tp_target_source": "RANGE_RAIL",
                    "self_improvement_p0_repair_live_ready": True,
                    "self_improvement_p0_repair_mode": "TP_HARVEST_REPAIR",
                    "self_improvement_p0_repair_blocker_code": "SELF_IMPROVEMENT_P0_PROFITABILITY_DISCIPLINE",
                }
            )
            files["intents"].write_text(json.dumps(intents))
            files["self_improvement_audit"].write_text(json.dumps(_self_improvement_profitability_p0()))
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

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertNotIn("SELF_IMPROVEMENT_P0_BLOCKS_TRADE", codes)
            lane = payload["input_packet"]["lanes"][0]
            self.assertTrue(lane["self_improvement"]["self_improvement_p0_repair_live_ready"])

    def test_accepts_trade_for_profit_capture_miss_p0_repair_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            intents = json.loads(files["intents"].read_text())
            intent = intents["results"][0]["intent"]
            intent["order_type"] = "LIMIT"
            intent["market_context"]["method"] = "BREAKOUT_FAILURE"
            metadata = intent.setdefault("metadata", {})
            metadata.update(
                {
                    "opportunity_mode": "HARVEST",
                    "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                    "tp_target_intent": "HARVEST",
                    "tp_target_source": "OPERATING_HARVEST_FLOOR",
                    "positive_rotation_mode": "TP_PROOF_COLLECTION_HARVEST",
                    "positive_rotation_pessimistic_expectancy_jpy": 215.6,
                    "self_improvement_p0_repair_live_ready": True,
                    "self_improvement_p0_repair_mode": "TP_HARVEST_REPAIR",
                    "self_improvement_p0_repair_blocker_code": (
                        "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION"
                    ),
                    "capture_take_profit_scope": "PAIR_SIDE_METHOD",
                    "capture_take_profit_scope_key": "EUR_USD|LONG|BREAKOUT_FAILURE|TAKE_PROFIT_ORDER",
                    "capture_take_profit_trades": 6,
                    "capture_take_profit_wins": 6,
                    "capture_take_profit_losses": 0,
                    "capture_take_profit_expectancy_jpy": 992.7,
                    "market_close_leak_family_close_gate_proof": True,
                    "market_close_leak_family_contained_risk_timing_evidence": True,
                    "market_close_leak_family_tp_proven_exception": True,
                }
            )
            files["intents"].write_text(json.dumps(intents))
            files["self_improvement_audit"].write_text(json.dumps(_self_improvement_profit_capture_miss_p0()))
            decision = _trade_decision(method="BREAKOUT_FAILURE")
            decision["market_read_first"]["best_trade_if_forced"]["vehicle"] = "LIMIT"
            decision["evidence_refs"].extend(
                [
                    "self_improvement:audit",
                    "self_improvement:finding:LOSS_CLOSE_PROFIT_CAPTURE_MISSED",
                ]
            )
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertNotIn("SELF_IMPROVEMENT_P0_BLOCKS_TRADE", codes)
            lane = payload["input_packet"]["lanes"][0]
            self.assertTrue(lane["self_improvement"]["self_improvement_p0_repair_live_ready"])

    def test_rejects_eurusd_breakout_failure_trade_without_market_close_leak_exception_proof(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            lane_id = "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE"
            files["intents"].write_text(
                json.dumps({"results": [_result(lane_id=lane_id, method="BREAKOUT_FAILURE")]})
            )
            decision = _trade_decision(lane_id=lane_id, method="BREAKOUT_FAILURE")
            decision["market_read_first"]["best_trade_if_forced"]["vehicle"] = "LIMIT"
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn(MARKET_CLOSE_LEAK_FAMILY_BLOCK_CODE, codes)

    def test_accepts_eurusd_breakout_failure_tp_proven_harvest_exception_from_lane_packet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            lane_id = "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT"
            result = _result(
                lane_id=lane_id,
                method="BREAKOUT_FAILURE",
                metadata={
                    "opportunity_mode": "HARVEST",
                    "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                    "tp_target_intent": "HARVEST",
                    "positive_rotation_mode": "TP_PROVEN_HARVEST",
                    "positive_rotation_live_ready": True,
                    "positive_rotation_pessimistic_expectancy_jpy": 332.5961,
                    "capture_take_profit_scope": "PAIR_SIDE_METHOD",
                    "capture_take_profit_scope_key": (
                        "EUR_USD|LONG|BREAKOUT_FAILURE|TAKE_PROFIT_ORDER"
                    ),
                    "capture_take_profit_trades": 20,
                    "capture_take_profit_losses": 0,
                    "capture_take_profit_expectancy_jpy": 591.5,
                    "self_improvement_p0_repair_live_ready": True,
                    "self_improvement_p0_repair_mode": "TP_HARVEST_REPAIR",
                },
            )
            result["intent"]["order_type"] = "LIMIT"
            files["intents"].write_text(json.dumps({"results": [result]}))
            decision = _trade_decision(lane_id=lane_id, method="BREAKOUT_FAILURE")
            decision["market_read_first"]["best_trade_if_forced"]["vehicle"] = "LIMIT"
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertNotIn(MARKET_CLOSE_LEAK_FAMILY_BLOCK_CODE, codes)

    def test_rejects_live_ready_lane_with_month_scale_residual_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            lane_id = "range_trader:EUR_USD:LONG:RANGE_ROTATION"
            files["intents"].write_text(
                json.dumps(
                    {
                        "results": [
                            _result(
                                lane_id=lane_id,
                                method="RANGE_ROTATION",
                                metadata={
                                    "opportunity_mode": "RUNNER",
                                    "month_scale_residual_loss_repair_blocked": True,
                                    "month_scale_residual_loss_group": {
                                        "pair": "EUR_USD",
                                        "side": "LONG",
                                        "method": "RANGE_ROTATION",
                                        "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
                                        "repair_replay_pl_jpy": -2333.8215,
                                        "loss_closes": 1,
                                        "trade_ids": ["471817"],
                                    },
                                },
                            )
                        ]
                    }
                )
            )
            decision = _trade_decision(lane_id=lane_id, method="RANGE_ROTATION")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn(MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCK_CODE, codes)

    def test_rejects_underpowered_oanda_self_improvement_repair_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            intents = json.loads(files["intents"].read_text())
            metadata = intents["results"][0]["intent"].setdefault("metadata", {})
            metadata.update(
                {
                    "opportunity_mode": "HARVEST",
                    "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                    "tp_target_intent": "HARVEST",
                    "tp_target_source": "RANGE_RAIL",
                    "self_improvement_p0_repair_live_ready": True,
                    "self_improvement_p0_repair_mode": "TP_HARVEST_REPAIR",
                    "self_improvement_p0_repair_blocker_code": "SELF_IMPROVEMENT_P0_PROFITABILITY_DISCIPLINE",
                    "positive_rotation_mode": "OANDA_CAMPAIGN_FIREPOWER_HARVEST",
                    "positive_rotation_minimum_floor_reachable": False,
                    "positive_rotation_minimum_floor_reach_basis": (
                        "OANDA_CAMPAIGN_FIREPOWER_CURRENT_RISK_UNDERPOWERED"
                    ),
                    "positive_rotation_oanda_campaign_current_risk_minimum_floor_reachable": False,
                }
            )
            files["intents"].write_text(json.dumps(intents))
            files["self_improvement_audit"].write_text(json.dumps(_self_improvement_profitability_p0()))
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
            lane = payload["input_packet"]["lanes"][0]
            self.assertFalse(lane["self_improvement"]["positive_rotation_minimum_floor_reachable"])

    def test_rejects_self_improvement_repair_lane_on_named_worst_segment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            intents = json.loads(files["intents"].read_text())
            metadata = intents["results"][0]["intent"].setdefault("metadata", {})
            metadata.update(
                {
                    "opportunity_mode": "HARVEST",
                    "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                    "tp_target_intent": "HARVEST",
                    "tp_target_source": "RANGE_RAIL",
                    "self_improvement_p0_repair_live_ready": True,
                    "self_improvement_p0_repair_mode": "TP_HARVEST_REPAIR",
                    "self_improvement_p0_repair_blocker_code": "SELF_IMPROVEMENT_P0_PROFITABILITY_DISCIPLINE",
                }
            )
            files["intents"].write_text(json.dumps(intents))
            audit = _self_improvement_profitability_p0()
            audit["findings"][0]["evidence"]["system_defect_evidence"]["worst_segments"] = [
                {
                    "pair": "EUR_USD",
                    "side": "LONG",
                    "method": "TREND_CONTINUATION",
                    "trades": 2,
                    "net_jpy": -1937.49,
                }
            ]
            files["self_improvement_audit"].write_text(json.dumps(audit))
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

    def test_rejects_trade_when_forecast_adverse_path_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["self_improvement_audit"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "SELF_IMPROVEMENT_BLOCKED",
                        "p0_findings": 0,
                        "p1_findings": 3,
                        "p2_findings": 0,
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
                                "next_action": "Repair directional forecast buckets before expanding exposure.",
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
            decision = _trade_decision()
            decision["evidence_refs"].extend(
                [
                    "self_improvement:audit",
                    "self_improvement:forecast",
                    "self_improvement:root_cause:FORECAST_ADVERSE_PATH",
                    "self_improvement:finding:DIRECTIONAL_FORECAST_HIT_RATE_WEAK",
                ]
            )
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("SELF_IMPROVEMENT_P0_BLOCKS_TRADE", codes)
            packet = payload["input_packet"]["self_improvement_audit"]
            self.assertEqual(
                packet["new_risk_blockers"][0]["code"],
                "SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH",
            )
            self.assertIn(
                "self_improvement:root_cause:FORECAST_ADVERSE_PATH",
                payload["input_packet"]["allowed_evidence_refs"],
            )

    def test_allows_tp_proven_repair_trade_when_forecast_adverse_path_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["self_improvement_audit"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "SELF_IMPROVEMENT_BLOCKED",
                        "p0_findings": 0,
                        "p1_findings": 3,
                        "p2_findings": 0,
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
            intents = json.loads(files["intents"].read_text())
            intent = intents["results"][0]["intent"]
            intent["order_type"] = "LIMIT"
            intent["thesis"] = "tp-proven failed-break fade"
            intent["market_context"]["method"] = "BREAKOUT_FAILURE"
            intent["market_context"]["regime"] = "RANGE current; BREAKOUT_FAILURE campaign lane"
            intent["metadata"].update(
                {
                    "opportunity_mode": "HARVEST",
                    "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                    "tp_target_intent": "HARVEST",
                    "forecast_direction": "RANGE",
                    "forecast_confidence": 0.62,
                    "attach_take_profit_on_fill": True,
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
                    "market_close_leak_family_close_gate_proof": True,
                    "market_close_leak_family_contained_risk_timing_evidence": True,
                    "market_close_leak_family_tp_proven_exception": True,
                    "self_improvement_forecast_adverse_path_repair_live_ready": True,
                    "self_improvement_forecast_adverse_path_repair_mode": "TP_HARVEST_REPAIR",
                    "self_improvement_forecast_adverse_path_repair_blocker_code": (
                        "SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH"
                    ),
                }
            )
            files["intents"].write_text(json.dumps(intents))
            decision = _trade_decision(method="BREAKOUT_FAILURE")
            decision["market_read_first"]["best_trade_if_forced"]["vehicle"] = "LIMIT"
            decision["evidence_refs"].extend(
                [
                    "self_improvement:audit",
                    "self_improvement:forecast",
                    "self_improvement:root_cause:FORECAST_ADVERSE_PATH",
                ]
            )
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertNotIn("SELF_IMPROVEMENT_P0_BLOCKS_TRADE", codes)

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

    def test_rejects_trade_when_stale_prior_gpt_decision_p0_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["self_improvement_audit"].write_text(
                json.dumps(
                    {
                        # The audit postdates the receipt under verification, so
                        # its persistent stale-decision verdict still applies to
                        # that receipt.
                        "generated_at_utc": (
                            datetime.now(timezone.utc) + timedelta(minutes=10)
                        ).isoformat(),
                        "status": "SELF_IMPROVEMENT_BLOCKED",
                        "findings": [
                            {
                                "priority": "P0",
                                "layer": "decision_history",
                                "code": "LATEST_GPT_DECISION_STALE",
                                "message": "latest GPT decision receipt predates the current broker snapshot",
                                "evidence": {"current_streak": 2},
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

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("SELF_IMPROVEMENT_P0_BLOCKS_TRADE", codes)

    def test_allows_trade_when_receipt_postdates_stale_decision_audit(self) -> None:
        # The persistent LATEST_GPT_DECISION_STALE streak is repaired by
        # writing one current receipt. When the receipt under verification was
        # generated AFTER the audit ran, the audit's stale verdict is about an
        # older receipt and must not reject the repair receipt itself — the
        # verifier's own STALE_DECISION_RECEIPT freshness gate still rejects
        # receipts that predate broker snapshot or order intents.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["self_improvement_audit"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": (
                            datetime.now(timezone.utc) - timedelta(minutes=10)
                        ).isoformat(),
                        "status": "SELF_IMPROVEMENT_BLOCKED",
                        "findings": [
                            {
                                "priority": "P0",
                                "layer": "decision_history",
                                "code": "LATEST_GPT_DECISION_STALE",
                                "message": "latest GPT decision receipt predates the current broker snapshot",
                                "evidence": {"current_streak": 17},
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

    def test_rejects_artifact_bound_trade_with_multiple_selected_lanes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            market_lane = f"{LANE_ID}:MARKET"
            files["intents"].write_text(
                json.dumps({"results": [_result(), _result(lane_id=market_lane)]})
            )
            brain = _brain(root, files, _batch_trade_decision([LANE_ID, market_lane]))

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            self.assertEqual(summary.selected_lane_ids, (LANE_ID, market_lane))
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("AI_MARKET_READ_SINGLE_LANE_REQUIRED", codes)

    def test_rejects_artifact_bound_trade_with_missing_or_mismatched_primary(self) -> None:
        cases = ("missing_primary", "mismatched_primary", "missing_lane_list")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                files = _fixtures(root)
                decision = _trade_decision()
                if case == "missing_primary":
                    decision["selected_lane_id"] = None
                elif case == "mismatched_primary":
                    decision["selected_lane_id"] = f"{LANE_ID}:OTHER"
                else:
                    decision.pop("selected_lane_ids")
                brain = _brain(root, files, decision)

                summary = brain.run(snapshot_path=files["snapshot"])

                self.assertEqual(summary.status, "REJECTED")
                payload = json.loads((root / "gpt_decision.json").read_text())
                codes = {issue["code"] for issue in payload["verification_issues"]}
                self.assertIn("AI_MARKET_READ_SINGLE_LANE_REQUIRED", codes)

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

    def test_input_packet_includes_same_pair_position_building_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            result = _result()
            result["intent"]["metadata"] = {
                "position_intent": "PYRAMID",
                "position_fill": "OPEN_ONLY",
                "same_pair_add_type": "AVERAGE_INTO_ADVERSE",
                "same_pair_existing_entries": 2,
                "same_pair_existing_units": 12_000,
                "same_pair_existing_avg_entry": 1.1003,
                "same_pair_add_entry": 1.0996,
                "same_pair_add_distance_from_avg_pips": -7.0,
                "same_pair_adverse_add_pips": 7.0,
                "same_pair_with_move_add_pips": 0.0,
            }
            files["intents"].write_text(json.dumps({"results": [result]}))
            brain = _brain(root, files, _trade_decision())

            brain.run(snapshot_path=files["snapshot"])

            payload = json.loads((root / "gpt_decision.json").read_text())
            position_building = payload["input_packet"]["lanes"][0]["position_building"]
            self.assertEqual(position_building["same_pair_add_type"], "AVERAGE_INTO_ADVERSE")
            self.assertEqual(position_building["same_pair_adverse_add_pips"], 7.0)
            self.assertEqual(position_building["same_pair_with_move_add_pips"], 0.0)

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

    def test_accepts_trade_replacing_self_improvement_pending_cancel_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, orders=[_pending_order()])
            files["self_improvement_audit"].write_text(
                json.dumps(_self_improvement_pending_cancel_review_p0())
            )
            decision = _trade_decision()
            decision["cancel_order_ids"] = ["pending-1"]
            decision["evidence_refs"].extend(
                [
                    "self_improvement:audit",
                    "self_improvement:execution_quality",
                    "self_improvement:finding:PENDING_ENTRY_CANCEL_REVIEW_REQUIRED",
                ]
            )
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            self.assertEqual(summary.cancel_order_ids, ("pending-1",))
            payload = json.loads((root / "gpt_decision.json").read_text())
            self.assertEqual(payload["verification_issues"], [])

    def test_rejects_trade_when_self_improvement_cancel_review_not_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, orders=[_pending_order()])
            files["self_improvement_audit"].write_text(
                json.dumps(_self_improvement_pending_cancel_review_p0())
            )
            decision = _trade_decision()
            decision["evidence_refs"].extend(
                [
                    "self_improvement:audit",
                    "self_improvement:execution_quality",
                    "self_improvement:finding:PENDING_ENTRY_CANCEL_REVIEW_REQUIRED",
                ]
            )
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("SELF_IMPROVEMENT_P0_BLOCKS_TRADE", codes)

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

    def test_accepts_known_profitability_acceptance_replay_repair_ref_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["profitability_acceptance"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "PROFITABILITY_ACCEPTANCE_BLOCKED",
                        "findings": [
                            {
                                "priority": "P0",
                                "code": "TP_PROGRESS_REPLAY_REPAIR_UNPROVED",
                                "message": "TP-progress repair is still unproved",
                            }
                        ],
                    }
                )
            )
            decision = _request_evidence_decision()
            decision["evidence_refs"].append(
                "profitability:acceptance:TP_PROGRESS_REPAIR_REPLAY_UNPROVED"
            )
            brain = _brain(root, files, decision)

            brain.run(snapshot_path=files["snapshot"])

            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            refs = payload["input_packet"]["allowed_evidence_refs"]
            p0_findings = payload["input_packet"]["profitability_acceptance"]["p0_findings"]
            self.assertNotIn("UNKNOWN_EVIDENCE_REF", codes)
            self.assertIn(
                "profitability:acceptance:TP_PROGRESS_REPAIR_REPLAY_UNPROVED",
                refs,
            )
            self.assertIn(
                "profitability:acceptance:TP_PROGRESS_REPAIR_REPLAY_UNPROVED",
                p0_findings[0]["evidence_ref_aliases"],
            )

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

    def test_accepts_wait_when_rolling_policy_only_has_b_grade_pace_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            _enable_rolling_policy(files, pace_state="BEHIND")
            _set_lane_target_grade(files, "B0")
            brain = _brain(root, files, _wait_decision())

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertNotIn("CAMPAIGN_EXPOSURE_REQUIRED", codes)
            self.assertNotIn("WAIT_MISSING_LIVE_READY_REJECTION", codes)
            self.assertNotIn("REQUEST_EVIDENCE_WITH_LIVE_READY_LANES", codes)

    def test_rejects_wait_when_rolling_policy_has_a_grade_pace_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            _enable_rolling_policy(files, pace_state="BEHIND")
            _set_lane_target_grade(files, "A")
            brain = _brain(root, files, _wait_decision())

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
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

    def test_rejects_wait_justified_only_by_session_timing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, positions=[{**_position(), "take_profit": 1.185}])
            decision = _wait_decision()
            decision["thesis"] = "Asia is a quiet session, so wait for London/NY liquidity."
            decision["narrative"] = "Stay flat because the session is quiet."
            decision["chart_story"] = "No chart blocker is named."
            decision["invalidation"] = "Reconsider when London starts."
            decision["rejected_alternatives"] = [f"{LANE_ID} rejected only because this is the Asian session."]
            decision["risk_notes"] = ["No concrete risk gate is cited."]
            decision["operator_summary"] = "Wait for London/NY."
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("SESSION_ONLY_WAIT_REJECTED", codes)

    def test_allows_wait_that_mentions_session_with_concrete_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, positions=[{**_position(), "take_profit": 1.185}])
            decision = _wait_decision()
            decision["thesis"] = "Off-hours conditions are acceptable to monitor, but the spread gate is too wide."
            decision["narrative"] = "Wait only because spread exceeds the current receipt cap, not because of clock time."
            decision["rejected_alternatives"] = [f"{LANE_ID} rejected this cycle by SPREAD_TOO_WIDE."]
            decision["risk_notes"] = ["SPREAD_TOO_WIDE blocks the lane until the next broker refresh."]
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertNotIn("SESSION_ONLY_WAIT_REJECTED", codes)

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

    def test_rejects_wait_when_self_improvement_pending_cancel_review_is_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, orders=[_pending_order()])
            files["self_improvement_audit"].write_text(
                json.dumps(_self_improvement_pending_cancel_review_p0())
            )
            decision = _wait_decision()
            decision["evidence_refs"].extend(
                [
                    "self_improvement:audit",
                    "self_improvement:execution_quality",
                    "self_improvement:finding:PENDING_ENTRY_CANCEL_REVIEW_REQUIRED",
                ]
            )
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("SELF_IMPROVEMENT_PENDING_CANCEL_REVIEW_REQUIRED", codes)
            packet = payload["input_packet"]["self_improvement_audit"]
            self.assertEqual(
                packet["p0_blockers"][0]["cancel_review_order_ids"], ["pending-1"]
            )

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

    def test_accepts_cancel_pending_when_live_ready_lane_is_learning_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, orders=[_pending_order()])
            files["attack_advice"].write_text(json.dumps(_learning_influenced_attack_advice()))
            files["learning_audit"].write_text(
                json.dumps(_learning_audit_payload(status="LEARNING_AUDIT_BLOCKED", blockers=["recent learned lane effect is negative"]))
            )
            decision = _cancel_pending_decision(cancel_order_ids=["pending-1"])
            decision["evidence_refs"].extend(["attack:advice", f"attack:lane:{LANE_ID}", "learning:audit"])
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            self.assertEqual(payload["verification_issues"], [])
            self.assertEqual(payload["decision"]["cancel_order_ids"], ["pending-1"])

    def test_rejects_cancel_pending_when_learning_influenced_lane_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, orders=[_pending_order()])
            files["attack_advice"].write_text(json.dumps(_learning_influenced_attack_advice()))
            files["learning_audit"].write_text(json.dumps(_learning_audit_payload(status="LEARNING_AUDIT_WARN")))
            decision = _cancel_pending_decision(cancel_order_ids=["pending-1"])
            decision["evidence_refs"].extend(["attack:advice", f"attack:lane:{LANE_ID}", "learning:audit"])
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("CANCEL_PENDING_WITH_LIVE_READY_LANES", codes)

    def test_accepts_cancel_pending_when_no_current_live_ready_lane_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, orders=[_pending_order()])
            blocked_result = _result()
            blocked_result["lane_id"] = "range_trader:USD_JPY:SHORT:RANGE_ROTATION"
            blocked_result["intent"]["pair"] = "USD_JPY"
            blocked_result["intent"]["side"] = "SHORT"
            blocked_result["status"] = "DRY_RUN_BLOCKED"
            blocked_result["live_blockers"] = ["forecast no longer backs this entry"]
            files["intents"].write_text(json.dumps({"results": [blocked_result]}))
            brain = _brain(root, files, _cancel_pending_decision(cancel_order_ids=["pending-1"]))

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            self.assertEqual(payload["verification_issues"], [])

    def test_cancel_pending_requires_timing_audit_when_same_shape_cancel_regret_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, orders=[_pending_order()])
            blocked_result = _result()
            blocked_result["lane_id"] = "range_trader:USD_JPY:SHORT:RANGE_ROTATION"
            blocked_result["intent"]["pair"] = "USD_JPY"
            blocked_result["intent"]["side"] = "SHORT"
            blocked_result["status"] = "DRY_RUN_BLOCKED"
            blocked_result["live_blockers"] = ["forecast no longer backs this entry"]
            files["intents"].write_text(json.dumps({"results": [blocked_result]}))
            files["execution_timing_audit"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "OK",
                        "canceled_order_regrets": [
                            {
                                "order_id": "prior-eurusd-stop",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "order_type": "STOP_ORDER",
                                "entry_touched_after_cancel": True,
                                "tp_touched_after_cancel": False,
                                "sl_touched_after_cancel": False,
                                "mfe_pips_after_cancel_entry": 3.2,
                            }
                        ],
                    }
                )
            )
            brain = _brain(root, files, _cancel_pending_decision(cancel_order_ids=["pending-1"]))

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("PENDING_CANCEL_TIMING_AUDIT_REQUIRED", codes)
            self.assertNotIn("CANCEL_PENDING_CURRENT_THESIS_VISIBLE", codes)

    def test_cancel_pending_with_timing_audit_ref_uses_normal_cancel_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, orders=[_pending_order()])
            blocked_result = _result()
            blocked_result["lane_id"] = "range_trader:USD_JPY:SHORT:RANGE_ROTATION"
            blocked_result["intent"]["pair"] = "USD_JPY"
            blocked_result["intent"]["side"] = "SHORT"
            blocked_result["status"] = "DRY_RUN_BLOCKED"
            blocked_result["live_blockers"] = ["forecast no longer backs this entry"]
            files["intents"].write_text(json.dumps({"results": [blocked_result]}))
            files["execution_timing_audit"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "OK",
                        "canceled_order_regrets": [
                            {
                                "order_id": "prior-eurusd-stop",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "order_type": "STOP_ORDER",
                                "entry_touched_after_cancel": True,
                                "tp_touched_after_cancel": False,
                                "sl_touched_after_cancel": False,
                                "mfe_pips_after_cancel_entry": 3.2,
                            }
                        ],
                    }
                )
            )
            decision = _cancel_pending_decision(cancel_order_ids=["pending-1"])
            decision["evidence_refs"].append("timing:audit")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED", msg=summary)
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertNotIn("PENDING_CANCEL_TIMING_AUDIT_REQUIRED", codes)

    def test_accepts_cancel_pending_for_self_improvement_pending_cancel_review_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, orders=[_pending_order()])
            blocked_result = _result()
            blocked_result["status"] = "DRY_RUN_BLOCKED"
            blocked_result["live_blockers"] = ["pending parent lane is no longer LIVE_READY"]
            files["intents"].write_text(json.dumps({"results": [blocked_result]}))
            files["self_improvement_audit"].write_text(
                json.dumps(_self_improvement_pending_cancel_review_p0())
            )
            decision = _cancel_pending_decision(cancel_order_ids=["pending-1"])
            decision["evidence_refs"].extend(
                [
                    "self_improvement:audit",
                    "self_improvement:execution_quality",
                    "self_improvement:finding:PENDING_ENTRY_CANCEL_REVIEW_REQUIRED",
                ]
            )
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            self.assertEqual(payload["verification_issues"], [])
            self.assertEqual(payload["decision"]["cancel_order_ids"], ["pending-1"])

    def test_rejects_cancel_pending_when_same_pair_thesis_still_visible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root, orders=[_pending_order()])
            blocked_result = _result()
            blocked_result["status"] = "DRY_RUN_BLOCKED"
            blocked_result["live_blockers"] = ["temporary spread block; thesis remains visible"]
            files["intents"].write_text(json.dumps({"results": [blocked_result]}))
            brain = _brain(root, files, _cancel_pending_decision(cancel_order_ids=["pending-1"]))

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("CANCEL_PENDING_CURRENT_THESIS_VISIBLE", codes)

    def test_cli_uses_external_decision_response_without_model_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            decision_response = root / "codex_decision_response.json"
            decision, artifacts = _apply_real_market_read_artifacts(
                root,
                files,
                _trade_decision(),
            )
            decision_response.write_text(json.dumps(decision))

            with patch(
                "quant_rabbit.cli.GPTTraderBrain",
                new=_local_cli_brain_factory(root, files),
            ), redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "gpt-trader-decision",
                        "--snapshot",
                        str(files["snapshot"]),
                        "--intents",
                        str(files["intents"]),
                        "--execution-ledger-db",
                        str(files["execution_ledger"]),
                        "--campaign-plan",
                        str(files["campaign"]),
                        "--strategy-profile",
                        str(files["strategy"]),
                        "--market-story-profile",
                        str(files["story"]),
                        "--market-status",
                        str(files["market_status"]),
                        "--target-state",
                        str(files["target"]),
                        "--attack-advice",
                        str(files["attack_advice"]),
                        "--learning-audit",
                        str(files["learning_audit"]),
                        "--self-improvement-audit",
                        str(files["self_improvement_audit"]),
                        "--projection-ledger",
                        str(files["projection_ledger"]),
                        "--market-context-matrix",
                        str(files["market_context_matrix"]),
                        "--trader-overrides",
                        str(files["trader_overrides"]),
                        "--qr-trader-run-watchdog",
                        str(files["qr_trader_run_watchdog"]),
                        "--guardian-action-receipt",
                        str(files["guardian_action_receipt"]),
                        "--guardian-receipt-consumption",
                        str(files["guardian_receipt_consumption"]),
                        "--guardian-receipt-operator-review",
                        str(files["guardian_receipt_operator_review"]),
                        "--active-trader-contract",
                        str(files["active_trader_contract"]),
                        "--active-opportunity-board",
                        str(files["active_opportunity_board"]),
                        "--non-eurusd-live-grade-frontier",
                        str(files["non_eurusd_frontier"]),
                        "--range-rail-geometry-repair",
                        str(files["range_rail_geometry_repair"]),
                        "--decision-response",
                        str(decision_response),
                        "--market-read-baseline",
                        str(artifacts["baseline"]),
                        "--market-read-evidence-packet",
                        str(artifacts["packet"]),
                        "--market-read-overlay",
                        str(artifacts["overlay"]),
                        "--market-read-predictions",
                        str(root / "market_read_predictions.jsonl"),
                        "--market-read-score-report",
                        str(root / "market_read_score_report.md"),
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
            self.assertIn("cross:WTICO_USD", payload["input_packet"]["allowed_evidence_refs"])

    def test_accepts_and_packets_current_news_evidence_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["news_items"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "issues": [],
                        "items": [
                            {
                                "source": "MarketPulse",
                                "published_at_utc": "2026-06-17T06:59:00+00:00",
                                "title": "Fed decision keeps dollar pairs in focus",
                                "pairs": ["EUR_USD"],
                                "currencies": ["EUR", "USD"],
                                "topics": ["central_bank"],
                                "categories": ["FX_EURUSD", "FX_USD"],
                                "link": "https://example.test/fed",
                            }
                        ],
                    }
                )
            )
            decision = _trade_decision()
            decision["evidence_refs"].extend(["news:items", "news:health", "news:EUR_USD", "news:USD"])
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            refs = payload["input_packet"]["allowed_evidence_refs"]
            self.assertIn("news:items", refs)
            self.assertIn("news:health", refs)
            self.assertIn("news:EUR_USD", refs)
            self.assertIn("news:USD", refs)
            self.assertEqual(payload["input_packet"]["news"]["health"]["status"], "OK")
            news_item = payload["input_packet"]["news"]["relevant_items"][0]
            self.assertEqual(news_item["evidence_refs"], ["news:EUR", "news:EUR_USD", "news:USD", "news:items"])
            self.assertEqual(news_item["title"], "Fed decision keeps dollar pairs in focus")

    def test_accepts_capture_economics_and_named_cross_asset_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["capture_economics"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "NEGATIVE_EXPECTANCY",
                        "overall": {
                            "trades": 193,
                            "win_rate": 0.5803,
                            "payoff_ratio": 0.398,
                            "breakeven_payoff_at_win_rate": 0.723,
                            "expectancy_jpy_per_trade": -197.8,
                            "net_jpy": -38183.1,
                        },
                        "by_exit_reason": {
                            "MARKET_ORDER_TRADE_CLOSE": {
                                "trades": 72,
                                "win_rate": 0.0972,
                                "payoff_ratio": 0.11,
                                "expectancy_jpy_per_trade": -1005.6,
                                "net_jpy": -72400.0,
                            }
                        },
                        "repair_summary": {
                            "dominant_loss_exit_reason": "MARKET_ORDER_TRADE_CLOSE",
                            "dominant_loss_exit_net_jpy": -72400.0,
                            "payoff_gap_to_breakeven": 0.325,
                            "strongest_positive_exit_reason": "TAKE_PROFIT_ORDER",
                            "strongest_positive_exit_net_jpy": 46874.8,
                        },
                        "segment_repair_priorities": {
                            "basis": "trader-attributed realized outcomes grouped by pair|side|method",
                            "scoped_tp_proof_min_exit_trades": 20,
                            "items": [
                                {
                                    "evidence_ref": "capture:segment:EUR_USD:LONG:BREAKOUT_FAILURE",
                                    "pair": "EUR_USD",
                                    "side": "LONG",
                                    "method": "BREAKOUT_FAILURE",
                                    "priority_class": "PRESERVE_TP_PROVEN_REPAIR_MARKET_CLOSE_LEAK",
                                    "next_action": "preserve broker TP and repair market-close leakage",
                                    "trades": 23,
                                    "win_rate": 0.87,
                                    "expectancy_jpy_per_trade": 304.3,
                                    "net_jpy": 7000.0,
                                    "take_profit_trades": 20,
                                    "take_profit_proof_gap_trades": 0,
                                    "take_profit_proven": True,
                                    "take_profit_expectancy_jpy": 500.0,
                                    "market_close_trades": 3,
                                    "market_close_losses": 3,
                                    "market_close_expectancy_jpy": -1000.0,
                                    "market_close_net_jpy": -3000.0,
                                }
                            ],
                        },
                        "action_items": [
                            "contain MARKET_ORDER_TRADE_CLOSE drag: prefer TP and hard Gate A/B closes"
                        ],
                    }
                )
            )
            decision = _trade_decision()
            decision["evidence_refs"].extend(
                [
                    "capture:economics",
                    "capture:segment:EUR_USD:LONG:BREAKOUT_FAILURE",
                    "cross:WTICO_USD",
                ]
            )
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            refs = payload["input_packet"]["allowed_evidence_refs"]
            self.assertIn("capture:economics", refs)
            self.assertIn("capture:exit_reason:MARKET_ORDER_TRADE_CLOSE", refs)
            self.assertIn("capture:segment:EUR_USD:LONG:BREAKOUT_FAILURE", refs)
            self.assertIn("cross:WTICO_USD", refs)
            capture = payload["input_packet"]["capture_economics"]
            self.assertEqual(capture["evidence_ref"], "capture:economics")
            self.assertEqual(capture["status"], "NEGATIVE_EXPECTANCY")
            self.assertEqual(
                capture["by_exit_reason"]["MARKET_ORDER_TRADE_CLOSE"]["net_jpy"],
                -72400.0,
            )
            self.assertEqual(
                capture["repair_summary"]["dominant_loss_exit_reason"],
                "MARKET_ORDER_TRADE_CLOSE",
            )
            self.assertEqual(
                capture["segment_repair_priorities"]["items"][0]["priority_class"],
                "PRESERVE_TP_PROVEN_REPAIR_MARKET_CLOSE_LEAK",
            )
            self.assertTrue(
                capture["segment_repair_priorities"]["items"][0]["take_profit_proven"]
            )
            self.assertIn("MARKET_ORDER_TRADE_CLOSE drag", capture["action_items"][0])
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertNotIn("UNKNOWN_EVIDENCE_REF", codes)

    def test_packet_includes_execution_timing_market_close_counterfactuals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            files["execution_timing_audit"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "OK",
                        "window": {"lookback_hours": 24.0, "post_close_hours": 6.0},
                        "summary": {
                            "market_closes_audited": 2,
                            "profit_market_closes_left_runner_upside": 1,
                            "loss_market_closes_contained_risk": 1,
                            "market_close_estimated_followthrough_jpy": 420.0,
                            "market_close_estimated_avoided_adverse_jpy": 900.0,
                            "loss_closes_audited": 3,
                            "loss_closes_profit_capture_missed": 1,
                            "loss_close_actual_pl_jpy": -120.0,
                            "loss_close_counterfactual_profit_capture_pl_jpy": 45.0,
                            "loss_close_counterfactual_profit_capture_delta_jpy": 165.0,
                            "loss_close_counterfactual_profit_capture_jpy": 45.0,
                        },
                        "canceled_order_regrets": [
                            {
                                "order_id": "o1",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "order_type": "LIMIT_ORDER",
                                "entry_touched_after_cancel": True,
                                "mfe_pips_after_cancel_entry": 3.1,
                            }
                        ],
                        "canceled_order_regret_by_shape": {
                            "items": [
                                {
                                    "evidence_ref": "timing:canceled_shape:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT_ORDER",
                                    "pair": "EUR_USD",
                                    "side": "LONG",
                                    "method": "BREAKOUT_FAILURE",
                                    "order_type": "LIMIT_ORDER",
                                    "priority_class": "PRESERVE_PENDING_THESIS_TP_TOUCHED",
                                    "next_action": "review cancel rule/TTL before canceling this pending shape",
                                    "orders": 2,
                                    "entry_touched_after_cancel": 2,
                                    "entry_touch_after_cancel_rate": 1.0,
                                    "positive_after_cancel_entry": 2,
                                    "positive_after_cancel_entry_rate": 1.0,
                                    "tp_touched_after_cancel": 2,
                                    "tp_touched_after_cancel_rate": 1.0,
                                    "estimated_missed_mfe_jpy": 240.0,
                                }
                            ]
                        },
                        "loss_close_regrets": [
                            {
                                "trade_id": "t2",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "gateway_action": "GPT_CLOSE",
                                "realized_pl_jpy": -120.0,
                                "had_positive_mfe_before_loss_close": True,
                                "tp_progress_before_loss_close": 0.42,
                                "profit_capture_missed_before_loss_close": True,
                                "profit_capture_counterfactual_jpy": 45.0,
                                "profit_capture_counterfactual_net_improvement_jpy": 165.0,
                                "profit_capture_counterfactual_pl_jpy": 45.0,
                            }
                        ],
                        "market_close_counterfactuals": [
                            {
                                "trade_id": "t1",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "gateway_action": "TAKE_PROFIT_MARKET",
                                "realized_pl_jpy": 80.0,
                                "post_close_path_label": "PROFIT_CLOSE_LEFT_RUNNER_UPSIDE",
                                "post_close_favorable_pips": 4.2,
                                "estimated_post_close_favorable_jpy": 420.0,
                            }
                        ],
                    }
                )
            )
            decision = _trade_decision()
            decision["evidence_refs"].extend(
                [
                    "timing:audit",
                    "timing:canceled_order:o1",
                    "timing:canceled_shape:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT_ORDER",
                    "timing:loss_close:t2",
                    "timing:market_close:t1",
                ]
            )
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            refs = payload["input_packet"]["allowed_evidence_refs"]
            self.assertIn("timing:audit", refs)
            self.assertIn(
                "timing:canceled_shape:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT_ORDER",
                refs,
            )
            self.assertIn("timing:market_close:t1", refs)
            timing = payload["input_packet"]["execution_timing_audit"]
            self.assertEqual(timing["evidence_ref"], "timing:audit")
            self.assertEqual(timing["summary"]["market_closes_audited"], 2)
            self.assertEqual(timing["summary"]["loss_closes_profit_capture_missed"], 1)
            self.assertEqual(
                timing["summary"]["loss_close_counterfactual_profit_capture_delta_jpy"],
                165.0,
            )
            self.assertEqual(
                timing["loss_close_regrets"][0]["profit_capture_counterfactual_pl_jpy"],
                45.0,
            )
            self.assertEqual(
                timing["canceled_order_regret_by_shape"][0]["priority_class"],
                "PRESERVE_PENDING_THESIS_TP_TOUCHED",
            )
            self.assertEqual(
                timing["market_close_counterfactuals"][0]["post_close_path_label"],
                "PROFIT_CLOSE_LEFT_RUNNER_UPSIDE",
            )
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertNotIn("UNKNOWN_EVIDENCE_REF", codes)

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
                        "opportunity_modes": {
                            "HARVEST": {
                                "lanes": 6,
                                "live_ready_lanes": 0,
                                "promotion_candidate_lanes": 0,
                                "reward_jpy": 7400.0,
                                "live_ready_reward_jpy": 0.0,
                                "potential_reward_jpy": 0.0,
                                "coverage_pct": 0.0,
                                "potential_coverage_pct": 0.0,
                                "top_issue_codes": [{"code": "SPREAD_TOO_WIDE", "count": 4}],
                                "top_live_blocker_codes": [{"code": "SPREAD_TOO_WIDE", "count": 4}],
                                "top_blockers": [{"label": "spread too wide", "count": 4}],
                                "top_lanes": [
                                    {
                                        "lane_id": "range_trader:EUR_USD:LONG:RANGE_ROTATION",
                                        "status": "DRY_RUN_BLOCKED",
                                        "reward_jpy": 1200.0,
                                        "reward_risk": 1.2,
                                    }
                                ],
                            },
                            "RUNNER": {
                                "lanes": 3,
                                "live_ready_lanes": 0,
                                "promotion_candidate_lanes": 0,
                                "reward_jpy": 11600.0,
                                "live_ready_reward_jpy": 0.0,
                                "potential_reward_jpy": 0.0,
                                "coverage_pct": 0.0,
                                "potential_coverage_pct": 0.0,
                                "top_issue_codes": [{"code": "FORECAST_REQUIRED", "count": 2}],
                                "top_live_blocker_codes": [{"code": "FORECAST_REQUIRED", "count": 2}],
                                "top_blockers": [{"label": "fresh runner forecast is missing", "count": 2}],
                                "top_lanes": [
                                    {
                                        "lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                                        "status": "DRY_RUN_BLOCKED",
                                        "reward_jpy": 3000.0,
                                        "reward_risk": 3.0,
                                    }
                                ],
                            },
                        },
                        "runner_candidate_diagnostics": {
                            "status": "RUNNER_CANDIDATES_DEMOTED_TO_HARVEST",
                            "trend_candidate_lanes": 3,
                            "runner_qualified_lanes": 0,
                            "attached_harvest_lanes": 3,
                            "status_counts": {"DRY_RUN_BLOCKED": 3},
                            "top_demotion_reasons": [
                                {
                                    "reason": "UNCLEAR regime is not a clean runner trend",
                                    "count": 2,
                                }
                            ],
                            "top_issue_codes": [
                                {
                                    "code": "TREND_MARKET_NOT_OPERATING_TREND",
                                    "count": 2,
                                }
                            ],
                            "top_live_blocker_codes": [
                                {
                                    "code": "TREND_MARKET_NOT_OPERATING_TREND",
                                    "count": 2,
                                }
                            ],
                            "top_lanes": [
                                {
                                    "lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                                    "status": "DRY_RUN_BLOCKED",
                                    "opportunity_mode": "HARVEST",
                                    "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                                    "tp_attach_reason": "UNCLEAR regime is not a clean runner trend",
                                    "reward_risk": 2.8,
                                }
                            ],
                        },
                        "perspective_alignment_diagnostics": {
                            "status": "RANGE_METHOD_MISMATCH_REPAIR_REQUIRED",
                            "pair_direction_groups": 4,
                            "range_forecast_method_mismatch_groups": 1,
                            "range_forecast_method_mismatch_lanes": 2,
                            "range_forecast_method_mismatch_top": [
                                {
                                    "pair": "EUR_USD",
                                    "direction": "SHORT",
                                    "method_mismatch_lanes": 2,
                                    "method_mismatch_reward_jpy": 2800.0,
                                    "range_rotation_lanes": 1,
                                    "range_rotation_live_ready_lanes": 0,
                                    "method_counts": [
                                        {"code": "BREAKOUT_FAILURE", "count": 2},
                                        {"code": "RANGE_ROTATION", "count": 1},
                                    ],
                                    "forecast_direction_counts": [{"code": "RANGE", "count": 3}],
                                    "chart_direction_bias_counts": [{"code": "SHORT", "count": 3}],
                                    "range_rotation_top_live_blocker_codes": [
                                        {"code": "RANGE_ROTATION_BROADER_LOCATION_CHASE", "count": 1}
                                    ],
                                    "range_rotation_absence_reason": "OPPOSITE_RAIL_SIDE_SURFACED",
                                    "range_rotation_other_side_lanes": 1,
                                    "range_rotation_other_side_directions": [
                                        {"code": "LONG", "count": 1}
                                    ],
                                    "range_rotation_other_side_top_live_blocker_codes": [
                                        {"code": "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE", "count": 1}
                                    ],
                                    "range_rotation_other_side_top_blockers": [
                                        {"label": "opposite rail still below live confidence", "count": 1}
                                    ],
                                    "top_live_blocker_codes": [
                                        {"code": "RANGE_FORECAST_REQUIRES_RANGE_ROTATION", "count": 2}
                                    ],
                                    "top_lanes": [
                                        {
                                            "lane_id": "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT",
                                            "status": "DRY_RUN_PASSED",
                                            "method": "BREAKOUT_FAILURE",
                                            "forecast_direction": "RANGE",
                                            "chart_direction_bias": "SHORT",
                                            "reward_jpy": 1400.0,
                                            "reward_risk": 2.6,
                                        }
                                    ],
                                }
                            ],
                        },
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
            self.assertEqual(packet["opportunity_modes"]["HARVEST"]["lanes"], 6)
            self.assertEqual(packet["opportunity_modes"]["RUNNER"]["top_issue_codes"][0]["code"], "FORECAST_REQUIRED")
            self.assertEqual(
                packet["opportunity_modes"]["RUNNER"]["top_live_blocker_codes"][0]["code"],
                "FORECAST_REQUIRED",
            )
            runner_diagnostics = packet["runner_candidate_diagnostics"]
            self.assertEqual(runner_diagnostics["status"], "RUNNER_CANDIDATES_DEMOTED_TO_HARVEST")
            self.assertEqual(runner_diagnostics["trend_candidate_lanes"], 3)
            self.assertEqual(runner_diagnostics["runner_qualified_lanes"], 0)
            self.assertEqual(runner_diagnostics["top_demotion_reasons"][0]["reason"], "UNCLEAR regime is not a clean runner trend")
            self.assertEqual(
                runner_diagnostics["top_live_blocker_codes"][0]["code"],
                "TREND_MARKET_NOT_OPERATING_TREND",
            )
            self.assertEqual(runner_diagnostics["top_lanes"][0]["opportunity_mode"], "HARVEST")
            perspective = packet["perspective_alignment_diagnostics"]
            self.assertEqual(perspective["status"], "RANGE_METHOD_MISMATCH_REPAIR_REQUIRED")
            self.assertEqual(perspective["range_forecast_method_mismatch_lanes"], 2)
            self.assertEqual(perspective["range_forecast_method_mismatch_top"][0]["pair"], "EUR_USD")
            self.assertEqual(
                perspective["range_forecast_method_mismatch_top"][0]["range_rotation_top_live_blocker_codes"][0]["code"],
                "RANGE_ROTATION_BROADER_LOCATION_CHASE",
            )
            self.assertEqual(
                perspective["range_forecast_method_mismatch_top"][0]["range_rotation_other_side_directions"][0]["code"],
                "LONG",
            )
            self.assertEqual(
                perspective["range_forecast_method_mismatch_top"][0]["range_rotation_other_side_top_live_blocker_codes"][0]["code"],
                "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
            )
            self.assertEqual(
                perspective["range_forecast_method_mismatch_top"][0]["range_rotation_other_side_top_blockers"][0]["label"],
                "opposite rail still below live confidence",
            )
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
            self.assertEqual(payload["input_packet"]["lanes"][0]["opportunity"]["opportunity_mode"], "RUNNER")

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

    def test_input_packet_exposes_required_learning_lane_refs(self) -> None:
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
            requirements = payload["input_packet"]["decision_requirements"]["learning_influenced_lane_evidence"]
            self.assertEqual(len(requirements), 1)
            self.assertEqual(requirements[0]["lane_id"], LANE_ID)
            self.assertTrue(requirements[0]["covered_by_learning_audit"])
            self.assertEqual(
                requirements[0]["required_evidence_refs"],
                ["learning:audit", f"learning:lane:{LANE_ID}"],
            )
            self.assertIn(f"learning:lane:{LANE_ID}", payload["input_packet"]["allowed_evidence_refs"])

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

    def test_trade_rejected_when_projection_ledger_has_expired_pending_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            emitted = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
            files["projection_ledger"].write_text(
                json.dumps(
                    {
                        "timestamp_emitted_utc": emitted,
                        "pair": "EUR_USD",
                        "signal_name": "directional_forecast",
                        "direction": "UP",
                        "resolution_window_min": 30.0,
                        "resolution_status": "PENDING",
                        "cycle_id": "cycle-expired-projection",
                    }
                )
                + "\n"
            )
            brain = _brain(root, files, _trade_decision())

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("PROJECTION_LEDGER_EXPIRED_PENDING_BLOCKS_TRADE", codes)
            projection = payload["input_packet"]["projection_ledger"]
            self.assertEqual(projection["expired_pending_count"], 1)
            self.assertIn("projection:expired_pending", payload["input_packet"]["allowed_evidence_refs"])

    def test_projection_packet_uses_independent_score_not_raw_row_majority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            base = now - timedelta(hours=2)

            def trial(minute: int, status: str, cycle_id: str) -> LedgerEntry:
                return LedgerEntry(
                    timestamp_emitted_utc=(base + timedelta(minutes=minute)).isoformat(),
                    pair="EUR_USD",
                    signal_name="directional_forecast",
                    direction="UP",
                    lead_time_min=60,
                    confidence=0.24,
                    raw_confidence=0.80,
                    calibration_multiplier=0.30,
                    entry_price=1.1000,
                    predicted_target_price=1.1020,
                    predicted_invalidation_price=1.0990,
                    resolution_window_min=60,
                    resolution_status=status,
                    resolved_at_utc=(
                        base + timedelta(minutes=minute + 60)
                    ).isoformat(),
                    resolution_evidence=(
                        "target touched before invalidation"
                        if status == "HIT"
                        else "invalidation touched before target"
                    ),
                    regime_at_emission="TREND",
                    cycle_id=cycle_id,
                )

            entries = [trial(0, "MISS", "first-independent-miss")]
            entries.extend(
                trial(minute, "HIT", f"overlapping-hit-{minute}")
                for minute in range(1, 60)
            )
            entries.append(trial(60, "MISS", "boundary-independent-miss"))
            write_ledger(entries, root)

            packet = _projection_ledger_packet(
                root / "projection_ledger.jsonl",
                now=now,
                calibration_scopes=[
                    {"pair": "EUR_USD", "direction": "UP", "regime": "TREND"}
                ],
            )

            self.assertEqual(packet["status_counts"], {"MISS": 2, "HIT": 59})
            calibration = packet["calibration_evidence"]
            self.assertEqual(
                calibration["statistics_source"],
                "quant_rabbit.strategy.projection_ledger.compute_hit_rates",
            )
            row = calibration["rows"][0]
            self.assertEqual(row["specific_bucket"]["samples"], 2)
            self.assertEqual(row["specific_bucket"]["hit_rate"], 0.0)
            self.assertIsNone(row["selected_bucket"])
            self.assertEqual(row["edge_status"], "INSUFFICIENT_SAMPLES_NO_EDGE")
            self.assertFalse(calibration["selection_contract"]["raw_ledger_rows_exposed"])
            self.assertNotIn("overlapping-hit-1", json.dumps(calibration))
            self.assertFalse(calibration["usage_policy"]["live_permission"])

    def test_wait_with_projection_ref_can_pause_attack_trade_for_expired_pending_projection(self) -> None:
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
            emitted = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
            files["projection_ledger"].write_text(
                json.dumps(
                    {
                        "timestamp_emitted_utc": emitted,
                        "pair": "EUR_USD",
                        "signal_name": "directional_forecast",
                        "direction": "UP",
                        "resolution_window_min": 30.0,
                        "resolution_status": "PENDING",
                        "cycle_id": "cycle-expired-projection",
                    }
                )
                + "\n"
            )
            decision = _wait_decision()
            decision["evidence_refs"].extend(["attack:advice", f"attack:lane:{LANE_ID}", "projection:ledger"])
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertNotIn("ATTACK_ADVICE_REQUIRES_TRADE", codes)
            self.assertNotIn("CAMPAIGN_EXPOSURE_REQUIRED", codes)
            self.assertNotIn("PROJECTION_LEDGER_EVIDENCE_MISSING", codes)

    def test_wait_rejects_attack_advice_when_stale_decision_p0_is_only_blocker(self) -> None:
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
            files["self_improvement_audit"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": (
                            datetime.now(timezone.utc) + timedelta(minutes=1)
                        ).isoformat(),
                        "status": "SELF_IMPROVEMENT_BLOCKED",
                        "p0_findings": 1,
                        "findings": [
                            {
                                "priority": "P0",
                                "layer": "decision_history",
                                "code": "LATEST_GPT_DECISION_STALE",
                                "message": "latest GPT decision receipt predates the current broker snapshot",
                                "evidence": {"current_streak": 7},
                            }
                        ],
                    }
                )
            )
            decision = _wait_decision()
            decision["evidence_refs"].extend(
                [
                    "attack:advice",
                    f"attack:lane:{LANE_ID}",
                    "self_improvement:audit",
                    "self_improvement:decision_history",
                    "self_improvement:finding:LATEST_GPT_DECISION_STALE",
                ]
            )
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("ATTACK_ADVICE_REQUIRES_TRADE", codes)
            self.assertNotIn("CAMPAIGN_EXPOSURE_REQUIRED", codes)

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
            decision["market_read_first"]["next_30m_prediction"].update(
                {"target_zone": "1.1780", "invalidation": "1.1750"}
            )
            decision["market_read_first"]["next_2h_prediction"].update(
                {"target_zone": "1.1800", "invalidation": "1.1750"}
            )
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

    def test_rejects_multi_pair_basket_until_pair_scoped_reads_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _fixtures(root)
            primary_lane = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET"
            second_pair_lane = "trend_trader:EUR_JPY:LONG:TREND_CONTINUATION:MARKET"
            third_pair_lane = "trend_trader:GBP_USD:LONG:TREND_CONTINUATION:MARKET"
            advised = [primary_lane, second_pair_lane, third_pair_lane]
            files["intents"].write_text(
                json.dumps(
                    {
                        "results": [
                            _result(lane_id=primary_lane, pair="EUR_USD"),
                            _result(lane_id=second_pair_lane, pair="EUR_JPY"),
                            _result(lane_id=third_pair_lane, pair="GBP_USD"),
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

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("MARKET_READ_PAIR_ACTION_CONFLICT", codes)

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

    def test_accepts_single_attack_priority_lane(self) -> None:
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
            decision = _trade_decision(lane_id=priority_lane)
            decision["evidence_refs"].extend(
                ["attack:advice", f"attack:lane:{priority_lane}"]
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


def _brain(
    root: Path,
    files: dict[str, Path],
    decision: dict,
    *,
    max_lanes: int | None = None,
    market_read_artifact_validation_required: bool = False,
    market_read_artifact_paths: dict[str, Path] | None = None,
) -> GPTTraderBrain:
    if decision.get("action") == "TRADE" and "decision_provenance" not in decision:
        _stamp_codex_market_read(decision)
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
        capture_economics_path=files["capture_economics"],
        profitability_acceptance_path=files["profitability_acceptance"],
        execution_timing_audit_path=files["execution_timing_audit"],
        coverage_optimization_path=files["coverage_optimization"],
        learning_audit_path=files["learning_audit"],
        verification_ledger_path=files["verification_ledger"],
        self_improvement_audit_path=files["self_improvement_audit"],
        projection_ledger_path=files["projection_ledger"],
        operator_precedent_path=files["operator_precedent"],
        manual_market_context_path=files["manual_market_context"],
        trader_overrides_path=files["trader_overrides"],
        predictive_limits_path=files["predictive_limits"],
        news_items_path=files["news_items"],
        news_health_path=files["news_health"],
        qr_trader_run_watchdog_path=files["qr_trader_run_watchdog"],
        guardian_action_receipt_path=files["guardian_action_receipt"],
        guardian_receipt_consumption_path=files["guardian_receipt_consumption"],
        guardian_receipt_operator_review_path=files["guardian_receipt_operator_review"],
        active_trader_contract_path=files["active_trader_contract"],
        active_opportunity_board_path=files["active_opportunity_board"],
        non_eurusd_live_grade_frontier_path=files["non_eurusd_frontier"],
        range_rail_geometry_repair_path=files["range_rail_geometry_repair"],
        execution_ledger_path=files["execution_ledger"],
        market_read_artifact_validation_required=market_read_artifact_validation_required,
        **(
            {
                "market_read_baseline_path": market_read_artifact_paths["baseline"],
                "market_read_evidence_packet_path": market_read_artifact_paths["packet"],
                "market_read_overlay_path": market_read_artifact_paths["overlay"],
            }
            if market_read_artifact_paths is not None
            else {}
        ),
        **({"max_lanes": max_lanes} if max_lanes is not None else {}),
    )


def _local_cli_brain_factory(root: Path, files: dict[str, Path]):
    """Bind CLI-only verifier coverage to temporary runtime artifacts."""

    def factory(**kwargs: object) -> GPTTraderBrain:
        kwargs.setdefault("execution_ledger_path", files["execution_ledger"])
        return GPTTraderBrain(
            pair_charts_path=files["pair_charts"],
            context_asset_charts_path=files["context_asset_charts"],
            broker_instruments_path=files["broker_instruments"],
            cross_asset_path=files["cross_asset"],
            flow_path=files["flow"],
            currency_strength_path=files["currency_strength"],
            levels_path=files["levels"],
            calendar_path=files["calendar"],
            cot_path=files["cot"],
            option_skew_path=files["option_skew"],
            capture_economics_path=files["capture_economics"],
            profitability_acceptance_path=files["profitability_acceptance"],
            execution_timing_audit_path=files["execution_timing_audit"],
            coverage_optimization_path=files["coverage_optimization"],
            verification_ledger_path=files["verification_ledger"],
            operator_precedent_path=files["operator_precedent"],
            manual_market_context_path=files["manual_market_context"],
            predictive_limits_path=files["predictive_limits"],
            news_items_path=files["news_items"],
            news_health_path=files["news_health"],
            **kwargs,
        )

    return factory


def _market_read_artifact_sources(
    root: Path,
    files: dict[str, Path],
) -> dict[str, Path]:
    predictions = root / "market_read_predictions.jsonl"
    if not predictions.exists():
        predictions.write_text("", encoding="utf-8")
    return {
        "broker_snapshot": files["snapshot"],
        "order_intents": files["intents"],
        "campaign_plan": files["campaign"],
        "strategy_profile": files["strategy"],
        "market_story_profile": files["story"],
        "market_status": files["market_status"],
        "daily_target_state": files["target"],
        "pair_charts": files["pair_charts"],
        "context_asset_charts": files["context_asset_charts"],
        "broker_instruments": files["broker_instruments"],
        "cross_asset": files["cross_asset"],
        "flow": files["flow"],
        "currency_strength": files["currency_strength"],
        "levels": files["levels"],
        "market_context_matrix": files["market_context_matrix"],
        "calendar": files["calendar"],
        "cot": files["cot"],
        "option_skew": files["option_skew"],
        "attack_advice": files["attack_advice"],
        "capture_economics": files["capture_economics"],
        "execution_timing_audit": files["execution_timing_audit"],
        "coverage_optimization": files["coverage_optimization"],
        "learning_audit": files["learning_audit"],
        "verification_ledger": files["verification_ledger"],
        "self_improvement_audit": files["self_improvement_audit"],
        "projection_ledger": files["projection_ledger"],
        "operator_precedent": files["operator_precedent"],
        "manual_market_context": files["manual_market_context"],
        "profitability_acceptance": files["profitability_acceptance"],
        "news_items": files["news_items"],
        "news_health": files["news_health"],
        "trader_overrides": files["trader_overrides"],
        "predictive_limits": files["predictive_limits"],
        "qr_trader_run_watchdog": files["qr_trader_run_watchdog"],
        "guardian_action_receipt": files["guardian_action_receipt"],
        "guardian_receipt_consumption": files["guardian_receipt_consumption"],
        "guardian_receipt_operator_review": files["guardian_receipt_operator_review"],
        "active_trader_contract": files["active_trader_contract"],
        "active_opportunity_board": files["active_opportunity_board"],
        "non_eurusd_live_grade_frontier": files["non_eurusd_frontier"],
        "range_rail_geometry_repair": files["range_rail_geometry_repair"],
        "execution_ledger": files["execution_ledger"],
        "market_read_predictions": predictions,
    }


def _synthetic_execution_cost_surface() -> dict:
    observed_at = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()

    def transport_section(label: str) -> dict:
        return {
            "samples": 20,
            "adverse_p95_pips": 0.0,
            "adverse_max_pips": 0.0,
            "oldest_fill_utc": observed_at,
            "latest_fill_utc": observed_at,
            "rows_sha256": hashlib.sha256(
                f"gpt-trader-{label}-cost-cohort".encode()
            ).hexdigest(),
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


def _apply_real_market_read_artifacts(
    root: Path,
    files: dict[str, Path],
    baseline_decision: dict,
    *,
    market_read: dict | None = None,
    broker_snapshot_source_path: Path | None = None,
) -> tuple[dict, dict[str, Path]]:
    paths = {
        "baseline": root / "trader_decision_baseline.json",
        "packet": root / "market_read_evidence_packet.json",
        "overlay": root / "codex_market_read_overlay.json",
        "output": root / "codex_trader_decision_response.json",
    }
    baseline = json.loads(json.dumps(baseline_decision))
    baseline.pop("decision_provenance", None)
    for key in (
        "market_read_review",
        "market_read_counterargument",
        "market_read_change_summary",
        "market_read_disposition",
        "market_read_veto_reason",
        "market_read_vetoed_lane_ids",
        "capital_allocation",
    ):
        baseline.pop(key, None)
    paths["baseline"].write_text(json.dumps(baseline), encoding="utf-8")
    if str(baseline.get("action") or "").upper() == "TRADE":
        _ensure_real_artifact_exact_vehicle_edge(
            files["intents"],
            ledger_path=files["execution_ledger"],
            selected_lane_id=str(baseline.get("selected_lane_id") or ""),
            snapshot_path=(broker_snapshot_source_path or files["snapshot"]),
            pair_charts_path=files["pair_charts"],
            calendar_path=files["calendar"],
            strategy_profile_path=files["strategy"],
        )
        refreshed_intents = json.loads(files["intents"].read_text(encoding="utf-8"))
        selected_intent = next(
            item["intent"]
            for item in refreshed_intents.get("results", [])
            if isinstance(item, dict)
            and str(item.get("lane_id") or "")
            == str(baseline.get("selected_lane_id") or "")
        )
        forced = (baseline.get("market_read_first") or {}).get(
            "best_trade_if_forced"
        )
        if isinstance(forced, dict):
            forced.update(
                {
                    "vehicle": "MARKET",
                    "entry": str(selected_intent["entry"]),
                    "tp": str(selected_intent["tp"]),
                    "sl": str(selected_intent["sl"]),
                }
            )
        paths["baseline"].write_text(json.dumps(baseline), encoding="utf-8")
    sources = _market_read_artifact_sources(root, files)
    if broker_snapshot_source_path is not None:
        sources["broker_snapshot"] = broker_snapshot_source_path
    applied_at = datetime.now(timezone.utc)
    prepare_market_read_baseline(
        baseline_path=paths["baseline"],
        packet_path=paths["packet"],
        evidence_sources=sources,
        now=applied_at,
    )
    stamped = json.loads(paths["baseline"].read_text(encoding="utf-8"))
    packet = json.loads(paths["packet"].read_text(encoding="utf-8"))
    selected_lane = (
        (packet.get("capital_allocation_board") or {}).get("selected_lane") or {}
    )
    allocate = str(baseline.get("action") or "").upper() == "TRADE"
    overlay = {
        "schema_version": packet["schema_version"],
        "author_kind": CODEX_MARKET_READ_AUTHOR,
        "model": "gpt-5.5",
        "reasoning_effort": "high",
        "authored_at_utc": applied_at.isoformat(),
        "baseline_sha256": canonical_json_sha256(baseline_core_payload(stamped)),
        "evidence_packet_sha256": packet["evidence_packet_sha256"],
        "baseline_disposition": "ACCEPT_BASELINE",
        "market_read_first": market_read or baseline["market_read_first"],
        "market_read_review": {
            "prior_prediction_ids": [],
            "what_failed": "NO_RESOLVED_PRIOR",
            "adjustment": "Use current quote-relative numeric geometry.",
            "no_change_reason": "",
        },
        "market_read_counterargument": "The forecast can fail before reaching its target.",
        "market_read_change_summary": "Rebuilt the forecast from current broker truth.",
        "market_read_veto_reason": "",
        "capital_allocation": {
            "decision": "ALLOCATE" if allocate else "NO_TRADE",
            "lane_id": baseline.get("selected_lane_id") if allocate else None,
            "size_multiple": 1.0 if allocate else 0.0,
            "selected_units": int(selected_lane.get("base_units") or 0) if allocate else 0,
            "allocation_board_sha256": packet["capital_allocation_board_sha256"],
            "rationale": (
                "The content-addressed lane edge supports the risk-capped baseline units."
                if allocate
                else "No fresh entry capital is authorized by this receipt."
            ),
        },
    }
    paths["overlay"].write_text(json.dumps(overlay), encoding="utf-8")
    apply_codex_market_read_overlay(
        baseline_path=paths["baseline"],
        packet_path=paths["packet"],
        overlay_path=paths["overlay"],
        output_path=paths["output"],
        evidence_sources=sources,
        now=applied_at,
    )
    return json.loads(paths["output"].read_text(encoding="utf-8")), paths


def _ensure_real_artifact_exact_vehicle_edge(
    intents_path: Path,
    *,
    ledger_path: Path,
    selected_lane_id: str,
    snapshot_path: Path,
    pair_charts_path: Path,
    calendar_path: Path,
    strategy_profile_path: Path,
) -> None:
    payload = json.loads(intents_path.read_text(encoding="utf-8"))
    result = next(
        (
            item
            for item in payload.get("results", []) or []
            if isinstance(item, dict)
            and str(item.get("lane_id") or "") == selected_lane_id
        ),
        None,
    )
    if not isinstance(result, dict):
        raise AssertionError(
            f"real artifact fixture lacks selected order intent {selected_lane_id}"
        )
    intent = result.get("intent") if isinstance(result.get("intent"), dict) else {}
    metadata = intent.setdefault("metadata", {})
    market_context = (
        intent.get("market_context")
        if isinstance(intent.get("market_context"), dict)
        else {}
    )
    pair = str(intent.get("pair") or "").upper()
    side = str(intent.get("side") or "").upper()
    method = str(metadata.get("method") or market_context.get("method") or "").upper()
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    quote = (snapshot.get("quotes") or {}).get(pair) or {}
    bid = float(quote["bid"])
    ask = float(quote["ask"])
    entry = ask if side == "LONG" else bid
    intent["order_type"] = "MARKET"
    intent["entry"] = entry
    raw_order_type = str(intent.get("order_type") or "").upper()
    vehicle = {
        "STOP_ENTRY": "STOP",
        "STOP-ENTRY": "STOP",
        "STOP_ORDER": "STOP",
        "LIMIT_ORDER": "LIMIT",
        "MARKET_ORDER": "MARKET",
    }.get(raw_order_type, raw_order_type)
    pip_factor = 100 if pair.endswith("_JPY") else 10_000
    loss_pips = (
        (entry - float(intent["sl"])) * pip_factor
        if side == "LONG"
        else (float(intent["sl"]) - entry) * pip_factor
    )
    reward_pips = (
        (float(intent["tp"]) - entry) * pip_factor
        if side == "LONG"
        else (entry - float(intent["tp"])) * pip_factor
    )
    quote_currency = pair.split("_", 1)[1]
    quote_to_jpy = (
        1.0
        if quote_currency == "JPY"
        else float((snapshot.get("home_conversions") or {})[quote_currency])
    )
    jpy_per_pip = abs(int(intent["units"])) / pip_factor * quote_to_jpy
    result["risk_metrics"] = {
        "entry_price": entry,
        "loss_pips": loss_pips,
        "reward_pips": reward_pips,
        "risk_jpy": loss_pips * jpy_per_pip,
        "reward_jpy": reward_pips * jpy_per_pip,
        "reward_risk": reward_pips / loss_pips,
        "spread_pips": (ask - bid) * pip_factor,
        "jpy_per_pip": jpy_per_pip,
        "estimated_margin_jpy": 1000.0,
    }
    half_spread = (ask - bid) / 2.0
    if side == "LONG":
        forecast_target = float(intent["tp"]) + half_spread * 2.0
        forecast_invalidation = float(intent["sl"]) + half_spread * 2.0
        forecast_direction = "UP"
    else:
        forecast_target = float(intent["tp"]) - half_spread * 2.0
        forecast_invalidation = float(intent["sl"]) - half_spread * 2.0
        forecast_direction = "DOWN"
    pair_charts = json.loads(pair_charts_path.read_text(encoding="utf-8"))
    matching_charts = [
        item
        for item in pair_charts.get("charts", []) or []
        if isinstance(item, dict)
        and str(item.get("pair") or "").upper() == pair
    ]
    if len(matching_charts) != 1:
        raise AssertionError(
            f"real artifact fixture requires exactly one {pair} pair-chart row"
        )
    pair_chart = json.loads(json.dumps(matching_charts[0]))
    pair_chart.setdefault("generated_at_utc", pair_charts.get("generated_at_utc"))
    evaluated_at = datetime.fromisoformat(
        str(snapshot["fetched_at_utc"]).replace("Z", "+00:00")
    )
    forecast_context_evidence = _forecast_context_evidence(
        pair,
        (bid + ask) / 2.0,
        direction=forecast_direction,
        spread_pips=(ask - bid) * pip_factor,
        pair_chart=pair_chart,
        calendar_path=calendar_path,
        strategy_profile_path=strategy_profile_path,
        now_utc=evaluated_at,
    )
    metadata.update(
        {
            "attach_take_profit_on_fill": True,
            "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
            "capture_economics_status": "NEGATIVE_EXPECTANCY",
            "capture_take_profit_exact_vehicle_required": True,
            "capture_take_profit_scope": "PAIR_SIDE_METHOD_VEHICLE",
            "capture_take_profit_scope_key": (
                f"{pair}|{side}|{method}|{vehicle}|TAKE_PROFIT_ORDER"
            ),
            "capture_take_profit_vehicle": vehicle,
            "capture_take_profit_metrics_source": (
                "data/execution_ledger.db:exact_vehicle_take_profit"
            ),
            "capture_take_profit_expectancy_jpy": 250.0,
            "capture_take_profit_net_jpy": 2000.0,
            "capture_take_profit_trades": 8,
            "capture_take_profit_wins": 8,
            "capture_take_profit_losses": 0,
            "capture_take_profit_avg_win_jpy": 250.0,
            "capture_take_profit_avg_loss_jpy": 0.0,
            "capture_exact_vehicle_net_scope": "PAIR_SIDE_METHOD_VEHICLE",
            "capture_exact_vehicle_net_scope_key": (
                f"{pair}|{side}|{method}|{vehicle}|ALL_AUDITED_EXITS"
            ),
            "capture_exact_vehicle_net_vehicle": vehicle,
            "capture_exact_vehicle_net_metrics_source": (
                "data/execution_ledger.db:exact_vehicle_net"
            ),
            "capture_exact_vehicle_net_exit_scope": "ALL_AUDITED_EXITS",
            "capture_exact_vehicle_net_trades": 8,
            "capture_exact_vehicle_net_wins": 8,
            "capture_exact_vehicle_net_losses": 0,
            "capture_exact_vehicle_net_jpy": 2000.0,
            "capture_exact_vehicle_net_expectancy_jpy": 250.0,
            "capture_exact_vehicle_net_avg_win_jpy": 250.0,
            "capture_exact_vehicle_net_avg_loss_jpy": 0.0,
            "capture_exact_vehicle_net_unresolved_realized_trades": 0,
            "capture_exact_vehicle_net_unresolved_realized_net_jpy": 0.0,
            "capture_exact_vehicle_net_unresolved_trade_ids_sha256": (
                hashlib.sha256(b"[]").hexdigest()
            ),
            "capture_market_close_expectancy_jpy": -100.0,
            "forecast_direction": forecast_direction,
            "forecast_current_price": (bid + ask) / 2.0,
            "forecast_technical_context": forecast_context_evidence,
            **_forecast_weighting_metadata(forecast_context_evidence),
            "forecast_target_price": forecast_target,
            "forecast_invalidation_price": forecast_invalidation,
            "forecast_directional_calibration_name": (
                f"directional_forecast_{forecast_direction.lower()}"
            ),
            "forecast_directional_hit_rate": 0.75,
            "forecast_directional_samples": 100,
            "forecast_directional_economic_hit_rate": 0.70,
            "forecast_directional_economic_samples": 100,
            "forecast_directional_timeout_rate": 0.05,
        }
    )
    intents_path.write_text(json.dumps(payload), encoding="utf-8")
    _write_exact_vehicle_ledger(
        ledger_path,
        pair=pair,
        side=side,
        method=method,
        vehicle=vehicle,
        outcomes=[250.0] * 8,
    )


def _write_exact_vehicle_ledger(
    path: Path,
    *,
    pair: str,
    side: str,
    method: str,
    vehicle: str,
    outcomes: list[float],
) -> None:
    if path.exists():
        path.unlink()
    entry_reason = {
        "LIMIT": "LIMIT_ORDER",
        "STOP": "STOP_ORDER",
        "MARKET": "MARKET_ORDER",
    }[vehicle]
    signed_units = 1000 if side == "LONG" else -1000
    lane = f"fixture_trader:{pair}:{side}:{method}:{vehicle}"
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
        for index, realized in enumerate(outcomes):
            trade_id = f"artifact-trade-{index}"
            order_id = f"artifact-entry-{index}"
            entry_ts = f"2026-07-01T00:{index:02d}:00Z"
            close_ts = f"2026-07-01T01:{index:02d}:00Z"
            entry_raw = {
                "id": f"artifact-fill-{index}",
                "time": entry_ts,
                "type": "ORDER_FILL",
                "orderID": order_id,
                "instrument": pair,
                "units": str(signed_units),
                "reason": entry_reason,
                "tradeOpened": {
                    "tradeID": trade_id,
                    "units": str(signed_units),
                },
            }
            exit_reason = (
                "TAKE_PROFIT_ORDER"
                if realized > 0
                else "MARKET_ORDER_TRADE_CLOSE"
            )
            close_raw = {
                "id": f"artifact-close-{index}",
                "time": close_ts,
                "type": "ORDER_FILL",
                "instrument": pair,
                "orderID": f"artifact-close-{index}",
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
            rows = [
                (
                    f"artifact-gateway-{index}", entry_ts, "GATEWAY_ORDER_SENT",
                    lane, order_id, trade_id, pair, side, signed_units, None,
                    0.0, entry_reason, json.dumps({"type": entry_reason}),
                ),
                (
                    f"artifact-fill-{index}", entry_ts, "ORDER_FILLED", lane,
                    order_id, trade_id, pair, side, signed_units, None, 0.0,
                    entry_reason, json.dumps(entry_raw),
                ),
                (
                    f"artifact-close-{index}", close_ts, "TRADE_CLOSED", None,
                    f"artifact-close-{index}", trade_id, pair,
                    "SHORT" if side == "LONG" else "LONG", abs(signed_units),
                    realized, 0.0, exit_reason, json.dumps(close_raw),
                ),
            ]
            conn.executemany(
                "INSERT INTO execution_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )


def _draft(root: Path, files: dict[str, Path]):
    return draft_trader_decision(
        snapshot_path=files["snapshot"],
        intents_path=files["intents"],
        campaign_plan_path=files["campaign"],
        strategy_profile_path=files["strategy"],
        market_story_profile_path=files["story"],
        market_status_path=files["market_status"],
        target_state_path=files["target"],
        output_path=root / "codex_trader_decision_response.json",
        report_path=root / "trader_decision_draft.md",
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
        capture_economics_path=files["capture_economics"],
        profitability_acceptance_path=files["profitability_acceptance"],
        execution_timing_audit_path=files["execution_timing_audit"],
        coverage_optimization_path=files["coverage_optimization"],
        learning_audit_path=files["learning_audit"],
        verification_ledger_path=files["verification_ledger"],
        self_improvement_audit_path=files["self_improvement_audit"],
        projection_ledger_path=files["projection_ledger"],
        operator_precedent_path=files["operator_precedent"],
        manual_market_context_path=files["manual_market_context"],
        trader_overrides_path=files["trader_overrides"],
        predictive_limits_path=files["predictive_limits"],
        news_items_path=files["news_items"],
        news_health_path=files["news_health"],
        qr_trader_run_watchdog_path=files["qr_trader_run_watchdog"],
        guardian_action_receipt_path=files["guardian_action_receipt"],
        guardian_receipt_consumption_path=files["guardian_receipt_consumption"],
        guardian_receipt_consumption_report_path=files["guardian_receipt_consumption_report"],
        guardian_receipt_operator_review_path=files["guardian_receipt_operator_review"],
        active_trader_contract_path=files["active_trader_contract"],
        active_opportunity_board_path=files["active_opportunity_board"],
        non_eurusd_live_grade_frontier_path=files["non_eurusd_frontier"],
        range_rail_geometry_repair_path=files["range_rail_geometry_repair"],
        execution_ledger_path=files["execution_ledger"],
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


def _write_watchdog_guardian_issue(
    path: Path,
    *,
    lifecycle: str,
    action: str,
    event_id: str,
    emergency_or_margin_risk: bool,
) -> None:
    issue = {
        "code": "GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER",
        "severity": "P0" if emergency_or_margin_risk else "WARN",
        "message": f"receipt_lifecycle={lifecycle} while consumed_by_trader=false",
        "receipt_event_id": event_id,
        "receipt_action": action,
        "receipt_lifecycle": lifecycle,
        "consumed_by_trader": False,
        "emergency_or_margin_risk": emergency_or_margin_risk,
        "normal_routing_allowed": False,
    }
    path.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "OK",
                "severity": issue["severity"],
                "guardian_receipt": {"issues": [issue]},
            }
        ),
        encoding="utf-8",
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
        "capture_economics": root / "capture_economics.json",
        "profitability_acceptance": root / "profitability_acceptance.json",
        "execution_timing_audit": root / "execution_timing_audit.json",
        "coverage_optimization": root / "coverage_optimization.json",
        "learning_audit": root / "learning_audit.json",
        "verification_ledger": root / "verification_ledger.json",
        "self_improvement_audit": root / "self_improvement_audit.json",
        "projection_ledger": root / "projection_ledger.jsonl",
        "operator_precedent": root / "operator_precedent.json",
        "manual_market_context": root / "manual_market_context.json",
        "trader_overrides": root / "trader_overrides.json",
        "predictive_limits": root / "predictive_limits.json",
        "news_items": root / "news_items.json",
        "news_health": root / "news_health.json",
        "qr_trader_run_watchdog": root / "qr_trader_run_watchdog.json",
        "guardian_action_receipt": root / "guardian_action_receipt.json",
        "guardian_receipt_consumption": root / "guardian_receipt_consumption.json",
        "guardian_receipt_consumption_report": root / "guardian_receipt_consumption_report.md",
        "guardian_receipt_operator_review": root / "guardian_receipt_operator_review.json",
        "active_trader_contract": root / "active_trader_contract.json",
        "active_opportunity_board": root / "active_opportunity_board.json",
        "non_eurusd_frontier": root / "non_eurusd_live_grade_frontier.json",
        "range_rail_geometry_repair": root / "range_rail_geometry_repair.json",
        "execution_ledger": root / "execution_ledger.db",
    }
    now = datetime.now(timezone.utc).isoformat()
    files["snapshot"].write_text(
        json.dumps(
            {
                "fetched_at_utc": now,
                "account": {"nav_jpy": 100_000.0, "fetched_at_utc": now},
                "home_conversions": {"USD": 100.0, "JPY": 1.0},
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
                        "confluence": {
                            "dominant_regime": "TREND_UP",
                            "price_percentile_24h": 0.7,
                            "price_percentile_7d": 0.6,
                        },
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
    files["capture_economics"].write_text(json.dumps({}))
    files["profitability_acceptance"].write_text(json.dumps({}))
    files["execution_timing_audit"].write_text(json.dumps({}))
    files["coverage_optimization"].write_text(json.dumps({"status": "OK"}))
    files["learning_audit"].write_text(json.dumps({}))
    files["self_improvement_audit"].write_text(json.dumps({}))
    files["projection_ledger"].write_text("")
    files["operator_precedent"].write_text(json.dumps({}))
    files["manual_market_context"].write_text(json.dumps({}))
    files["trader_overrides"].write_text(json.dumps({}))
    files["predictive_limits"].write_text(json.dumps({"dry_run": True, "orders": []}))
    files["news_items"].write_text(json.dumps({"generated_at_utc": now, "issues": [], "items": []}))
    files["news_health"].write_text(
        json.dumps(
            {
                "generated_at_utc": now,
                "status": "OK",
                "market_window": "ACTIVE",
                "issues": [],
            }
        )
    )
    files["active_trader_contract"].write_text(json.dumps({}))
    files["active_opportunity_board"].write_text(json.dumps({}))
    files["non_eurusd_frontier"].write_text(json.dumps({}))
    files["range_rail_geometry_repair"].write_text(json.dumps({}))
    _write_exact_vehicle_ledger(
        files["execution_ledger"],
        pair="EUR_USD",
        side="LONG",
        method="TREND_CONTINUATION",
        vehicle="MARKET",
        outcomes=[],
    )
    return files


def _enable_rolling_policy(files: dict[str, Path], *, pace_state: str = "BEHIND") -> None:
    target = json.loads(files["target"].read_text())
    target.update(
        {
            "rolling_30d_policy": "ROLLING_30D_4X",
            "rolling_30d_start_equity": 100_000.0,
            "current_equity": 110_000.0,
            "current_30d_multiplier": 1.1,
            "remaining_to_4x": 290_000.0,
            "required_calendar_daily_return": 4.25,
            "required_active_day_return": 5.92,
            "pace_state": pace_state,
            "remaining_target_jpy": 4_000.0,
        }
    )
    files["target"].write_text(json.dumps(target))


def _set_lane_target_grade(files: dict[str, Path], grade: str, *, lane_id: str = LANE_ID) -> None:
    intents = json.loads(files["intents"].read_text())
    for result in intents.get("results", []) or []:
        if str(result.get("lane_id") or "") != lane_id:
            continue
        metadata = result.setdefault("intent", {}).setdefault("metadata", {})
        metadata.update(
            {
                "daily_target_mode": "BUILD",
                "target_path_role": "MAIN",
                "attack_stack_slot": "NOW",
                "grade": grade,
                "valid_as_target_path": "YES" if grade in {"A", "S"} else "NO",
                "remaining_to_5pct_yen": 4_000.0,
            }
        )
    files["intents"].write_text(json.dumps(intents))


def _operator_precedent_audit(aligned_lane_ids: list[str]) -> dict:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "OPERATOR_PRECEDENT_PASS",
        "operator_claim": {
            "claim": "funding-adjusted 30-calendar-day return exceeded 200%",
            "required_return_pct": 200.0,
            "verified": True,
        },
        "precedent": {
            "winning_shape": {
                "primary_pair": "USD_JPY",
                "primary_direction": "LONG",
                "primary_sessions": ["LONDON_AM", "NY_OVERLAP"],
                "positive_sessions": ["LONDON_AM", "NY_OVERLAP"],
                "expectancy_jpy_per_exit": 649.2,
                "median_hold_hours": 0.48,
                "payoff": 1.3,
            },
            "failure_shape": {
                "margin_closeout": {
                    "trades": 24,
                    "net_jpy": -217327.8,
                    "win_rate": 0.042,
                    "median_hold_hours": 12.38,
                }
            },
        },
        "runtime_alignment": {
            "live_ready_lanes": len(aligned_lane_ids),
            "aligned_live_ready_lanes": len(aligned_lane_ids),
            "aligned_lanes": [
                {
                    "lane_id": lane_id,
                    "pair": "USD_JPY",
                    "side": "LONG",
                    "method": "TREND_CONTINUATION",
                    "order_type": "STOP-ENTRY",
                    "session": "LONDON_AM",
                }
                for lane_id in aligned_lane_ids
            ],
            "manual_context_alignment": {
                "status": "MANUAL_CONTEXT_ALIGNMENT_READY",
                "compatible_lanes": [],
                "conflicting_lanes": [],
                "conflicting_aligned_lanes": 0,
            },
            "manual_exit_events_per_calendar_day": 9.24,
            "target_trades_per_day": 30.0,
            "alignment_contract": {
                "aligned_precedent_is_advisory": True,
                "absence_of_alignment_is_not_a_trade_blocker": True,
                "current_risk_geometry_remains_authority": True,
            },
        },
        "warnings": [],
        "blockers": [],
    }


def _manual_market_context_audit() -> dict:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "MANUAL_MARKET_CONTEXT_PASS",
        "sample": {
            "pair": "USD_JPY",
            "manual_trades": 411,
            "analyzed_trades": 411,
            "coverage_pct": 100.0,
        },
        "guidance": {
            "basis": "bounded_replay_lt_12h_excluding_margin_closeout",
            "prefer_when_citing_precedent": {
                "h1_alignment": "AGAINST_H1_TREND",
                "session_jst": "LONDON_AM",
            },
            "require_extra_current_reason_when_conflicting": {
                "h1_alignment": "WITH_H1_TREND",
                "hold_bucket": "2H_12H",
            },
            "operator_precedent_usage_gate": (
                "Current lanes may cite the 2025 manual precedent only when comparable."
            ),
        },
        "bounded_replay_profile": {
            "overall": {"trades": 323, "net_jpy": 105621.2},
            "by_h1_alignment": [
                {
                    "bucket": "AGAINST_H1_TREND",
                    "trades": 131,
                    "net_jpy": 96729.5,
                    "win_rate": 0.466,
                    "expectancy_jpy": 738.4,
                    "median_hold_hours": 0.23,
                    "avg_h1_adx": 27.5,
                }
            ],
            "by_side_h1_alignment": [],
            "by_side_entry_location_24h": [],
            "by_session_jst": [],
        },
        "excluded_tail_profile": {
            "by_hold_bucket": [
                {
                    "bucket": "GE_12H",
                    "trades": 76,
                    "net_jpy": 186253.7,
                    "win_rate": 0.75,
                    "expectancy_jpy": 2450.7,
                    "median_hold_hours": 121.38,
                    "avg_h1_adx": 29.7,
                }
            ],
            "by_close_reason": [],
        },
        "position_building_profile": {
            "basis": "same-pair same-side overlapping open/close windows reconstructed from manual OANDA exit rows",
            "overall": {
                "clusters": 264,
                "multi_entry_clusters": 18,
                "entries": 384,
                "net_jpy": 266815.9,
                "win_rate": 0.428,
                "expectancy_jpy": 1010.7,
                "max_entries": 55,
                "adverse_adds": 71,
                "pyramid_adds": 49,
                "avg_adverse_add_pips": 50.7,
            },
            "bounded_lt_12h_excluding_margin_closeout": {
                "clusters": 255,
                "multi_entry_clusters": 10,
                "entries": 279,
                "net_jpy": 108343.7,
                "win_rate": 0.431,
                "expectancy_jpy": 424.9,
                "max_entries": 8,
                "adverse_adds": 14,
                "pyramid_adds": 10,
                "avg_adverse_add_pips": 6.45,
            },
            "adverse_adds": {
                "clusters": 8,
                "entries": 24,
                "net_jpy": 102564.0,
                "win_rate": 0.875,
                "expectancy_jpy": 12820.5,
                "max_entries": 4,
                "adverse_adds": 14,
                "avg_adverse_add_pips": 6.45,
            },
            "bounded_by_build_type": [
                {
                    "bucket": "AVERAGE_INTO_ADVERSE",
                    "clusters": 7,
                    "multi_entry_clusters": 7,
                    "entries": 20,
                    "net_jpy": 38234.0,
                    "win_rate": 0.857,
                    "expectancy_jpy": 5462.0,
                    "median_entries": 3,
                    "max_entries": 4,
                    "adverse_adds": 13,
                    "pyramid_adds": 0,
                    "avg_adverse_add_pips": 6.89,
                }
            ],
            "examples": {
                "largest_adverse_add_winners": [
                    {
                        "cluster_id": "USD_JPY:SHORT:2025-06-16T10:25:00.820230+00:00",
                        "side": "SHORT",
                        "build_type": "AVERAGE_INTO_ADVERSE",
                        "entries": 3,
                        "trade_ids": ["1863", "1866", "1868"],
                        "session_jst": "LONDON_AM",
                        "hold_hours": 2.972,
                        "realized_pl": 11346.0,
                        "initial_price": 144.125,
                        "final_weighted_avg": 144.12972,
                        "adverse_add_count": 2,
                        "pyramid_add_count": 0,
                        "close_reasons": ["MARKET_ORDER_TRADE_CLOSE", "TAKE_PROFIT_ORDER"],
                    }
                ]
            },
            "contract": {
                "advisory_only": True,
                "nanpin_is_not_live_permission": True,
                "requires_current_basket_risk_validation": True,
                "forbidden_to_use_for_unbounded_martingale": True,
            },
        },
        "contract": {
            "advisory_only": True,
            "may_gate_use_of_operator_precedent_as_aggression_reason": True,
            "does_not_override_current_risk_geometry": True,
            "does_not_grant_live_permission": True,
        },
        "warnings": [],
        "blockers": [],
    }


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


def _self_improvement_profit_capture_miss_p0() -> dict:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SELF_IMPROVEMENT_BLOCKED",
        "p0_findings": 1,
        "findings": [
            {
                "priority": "P0",
                "layer": "execution_quality",
                "code": "LOSS_CLOSE_PROFIT_CAPTURE_MISSED",
                "message": (
                    "13 losing close(s) had production-gate replay proof that "
                    "TP-progress capture was executable before closing red"
                ),
                "evidence": {
                    "loss_closes_profit_capture_missed": 13,
                    "loss_closes_repair_replay_triggered": 13,
                    "loss_close_repair_replay_delta_jpy": 18768.834,
                },
            }
        ],
    }


def _profitability_acceptance_close_leak_p0() -> dict:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "BLOCKED",
        "blockers": [
            "RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK: 7 loss-side gateway MARKET_ORDER_TRADE_CLOSE event(s) remain inside the 7-day acceptance window"
        ],
        "findings": [
            {
                "priority": "P0",
                "code": "RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK",
                "message": "7 loss-side gateway MARKET_ORDER_TRADE_CLOSE event(s) remain inside the 7-day acceptance window",
                "next_action": "Keep profitability acceptance red until a full recent window shows no new loss-side gateway market-close leakage.",
                "evidence": {
                    "recent_loss_closes": 7,
                    "recent_loss_net_jpy": -4567.1974,
                    "latest_loss_close_ts_utc": "2026-06-19T14:22:08.785628+00:00",
                    "examples": [
                        {
                            "trade_id": "472743",
                            "pair": "NZD_USD",
                            "side": "LONG",
                            "close_provenance": "GATEWAY_TRADE_CLOSE_SENT",
                            "realized_pl_jpy": -1380.8008,
                        }
                    ],
                },
            }
        ],
    }


def _profitability_acceptance_close_gate_evidence_missing_p0() -> dict:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "BLOCKED",
        "blockers": [
            "LOSS_CLOSE_GATE_EVIDENCE_MISSING: 1 recent GPT loss-side market close lacks passing durable close_gate_evidence"
        ],
        "findings": [
            {
                "priority": "P0",
                "code": "LOSS_CLOSE_GATE_EVIDENCE_MISSING",
                "message": (
                    "1 recent GPT loss-side market close lacks passing durable "
                    "close_gate_evidence in verification_observations"
                ),
                "next_action": (
                    "Persist gpt-trader close_gate_evidence before acceptance can "
                    "clear the loss-side market-close leak."
                ),
                "evidence": {
                    "recent_close_gate_unverified_loss_closes": 1,
                    "recent_close_gate_unverified_loss_net_jpy": -1380.8008,
                    "examples": [
                        {
                            "trade_id": "472743",
                            "pair": "NZD_USD",
                            "side": "LONG",
                            "realized_pl_jpy": -1380.8008,
                        }
                    ],
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


def _self_improvement_pending_cancel_review_p0() -> dict:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SELF_IMPROVEMENT_BLOCKED",
        "p0_findings": 1,
        "p1_findings": 1,
        "p2_findings": 0,
        "effect_metrics": {
            "closed_trades": 32,
            "net_jpy": -845.32,
            "profit_factor": 0.894,
            "expectancy_jpy": -26.416,
        },
        "findings": [
            {
                "priority": "P0",
                "layer": "execution_quality",
                "code": "PENDING_ENTRY_CANCEL_REVIEW_REQUIRED",
                "message": "1 trader-owned pending entry order(s) need cancel review",
                "next_action": (
                    "Write a CANCEL_PENDING receipt for these order ids when no current "
                    "LIVE_READY replacement exists, or write a TRADE receipt with "
                    "cancel_order_ids when replacing them with a current verified basket."
                ),
                "evidence": {
                    "cancel_review_order_ids": ["pending-1"],
                    "orders": [
                        {
                            "order_id": "pending-1",
                            "pair": "EUR_USD",
                            "side": "LONG",
                            "method": "TREND_CONTINUATION",
                            "review_reasons": [
                                {"code": "PENDING_CURRENT_CANDIDATE_NOT_LIVE_READY"}
                            ],
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
    view = {
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
    if granularity == "M5":
        view["structure"] = {
            "structure_events": [
                {
                    "kind": "BOS_UP",
                    "index": 1,
                    "close_confirmed": True,
                }
            ]
        }
    return view


def _result(
    *,
    lane_id: str = LANE_ID,
    method: str = "TREND_CONTINUATION",
    pair: str = "EUR_USD",
    side: str = "LONG",
    metadata: dict | None = None,
) -> dict:
    return {
        "lane_id": lane_id,
        "status": "LIVE_READY",
        "risk_allowed": True,
        "risk_issues": [],
        "strategy_issues": [],
        "live_blockers": [],
        "intent": {
            "pair": pair,
            "side": side,
            "order_type": "MARKET",
            "units": 1000,
            "entry": 1.1721,
            "tp": 1.1737,
            "sl": 1.1717,
            "thesis": f"{pair} continuation can pay before daily target window closes.",
            "owner": "trader",
            "market_context": {
                "regime": f"{method} campaign lane",
                "narrative": f"Momentum theme favors {pair} continuation.",
                "chart_story": "Higher lows are pressing into the trigger shelf.",
                "method": method,
                "invalidation": "Invalid if the shelf breaks before entry.",
                "event_risk": "",
                "session": "test",
            },
            "metadata": metadata or {
                "opportunity_mode": "RUNNER",
                "opportunity_mode_reason": "tp_target_intent=EXTEND",
                "opportunity_mode_reward_risk": 2.4,
                "tp_execution_mode": "RUNNER_NO_BROKER_TP",
                "tp_target_intent": "EXTEND",
                "tp_target_source": "STRUCTURAL_EXTEND",
                "capture_economics_status": "NEGATIVE_EXPECTANCY",
                "capture_take_profit_expectancy_jpy": 250.0,
                "capture_take_profit_trades": 4,
                "capture_take_profit_wins": 4,
                "capture_take_profit_losses": 0,
                "forecast_direction": "UP",
                "forecast_current_price": 1.17205,
                "forecast_directional_calibration_name": "directional_forecast_up",
                "forecast_directional_hit_rate": 0.75,
                "forecast_directional_samples": 100,
                "forecast_directional_economic_hit_rate": 0.70,
                "forecast_directional_economic_samples": 100,
                "forecast_directional_timeout_rate": 0.05,
            },
        },
        "risk_metrics": {
            "entry_price": 1.1721,
            "loss_pips": 4.0,
            "reward_pips": 16.0,
            "risk_jpy": 40.0,
            "reward_jpy": 160.0,
            "reward_risk": 4.0,
            "spread_pips": 1.0,
            "jpy_per_pip": 10.0,
            "estimated_margin_jpy": 1000.0,
        },
    }


def _trade_decision(
    *,
    lane_id: str = LANE_ID,
    method: str = "TREND_CONTINUATION",
    pair: str = "EUR_USD",
    direction: str = "LONG",
) -> dict:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "market_read_first": _market_read_first(pair=pair, direction=direction),
        "action": "TRADE",
        "selected_lane_id": lane_id,
        "selected_lane_ids": [lane_id],
        "confidence": "HIGH",
        "thesis": f"MARKET READ FIRST next 30m/next 2h {pair} {direction} path supports the live-ready continuation lane.",
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
            f"strategy:{pair}:{direction}",
            f"story:{pair}",
            f"chart:{pair}:M5",
            f"chart:{pair}:M15",
            "news:health",
            "news:items",
        ],
        "twenty_minute_plan": _twenty_minute_plan(lane_ids=[lane_id], pair=pair),
        "operator_summary": f"Accept the verified {pair} continuation lane after the next 30m and next 2h market read.",
    }


def _stamp_codex_market_read(
    decision: dict,
    *,
    disposition: str = "ACCEPT_BASELINE",
    baseline_action: str = "TRADE",
    baseline_lane_ids: list[str] | None = None,
) -> None:
    applied_at = datetime.now(timezone.utc).isoformat()
    lane_ids = baseline_lane_ids
    if lane_ids is None:
        lane_ids = list(decision.get("selected_lane_ids") or [])
        if not lane_ids and decision.get("selected_lane_id"):
            lane_ids = [str(decision["selected_lane_id"])]
    veto = disposition.startswith("VETO_")
    decision["market_read_review"] = {
        "prior_prediction_ids": [],
        "what_failed": "NO_RESOLVED_PRIOR",
        "adjustment": "Use current quote-relative numeric geometry.",
        "no_change_reason": "",
    }
    decision["market_read_counterargument"] = "The forecast can fail before reaching its target."
    decision["market_read_change_summary"] = "Rebuilt the forecast from current broker truth."
    decision["market_read_disposition"] = disposition
    decision["market_read_veto_reason"] = (
        "The current forecast contradicts the deterministic entry trigger." if veto else ""
    )
    decision["market_read_vetoed_lane_ids"] = list(lane_ids) if veto else []
    allocate = (
        str(decision.get("action") or "").upper() == "TRADE"
        and disposition == "ACCEPT_BASELINE"
    )
    capital_allocation = {
        "decision": "ALLOCATE" if allocate else "NO_TRADE",
        "lane_id": lane_ids[0] if allocate and len(lane_ids) == 1 else None,
        "size_multiple": 1.0 if allocate else 0.0,
        "selected_units": 1000 if allocate else 0,
        "allocation_board_sha256": "e" * 64,
        "rationale": (
            "The verified direction-specific edge supports the risk-capped baseline units."
            if allocate
            else "No fresh entry capital is authorized by this receipt."
        ),
    }
    decision["capital_allocation"] = capital_allocation
    guardian_baseline_pairs = _synthetic_guardian_baseline_pairs(
        decision,
        lane_ids=lane_ids,
    )
    guardian_scope_material = {
        "parse_status": "MISSING",
        "baseline_pairs": guardian_baseline_pairs,
    }
    execution_sha = canonical_json_sha256(execution_envelope_payload(decision))
    decision["decision_provenance"] = {
        "schema_version": 2,
        "author_kind": CODEX_MARKET_READ_AUTHOR,
        "model": "gpt-5.5",
        "reasoning_effort": "high",
        "authored_at_utc": applied_at,
        "applied_at_utc": applied_at,
        "baseline_sha256": "a" * 64,
        "evidence_packet_sha256": "b" * 64,
        "overlay_sha256": "c" * 64,
        "market_read_sha256": market_read_sha256(decision.get("market_read_first")),
        "execution_envelope_sha256": execution_sha,
        "baseline_execution_envelope_sha256": "d" * 64,
        "final_execution_envelope_sha256": execution_sha,
        "baseline_action": baseline_action,
        "final_action": decision.get("action"),
        "baseline_selected_lane_ids": list(lane_ids),
        "baseline_disposition": disposition,
        "action_downgrade_only": veto,
        "capital_allocation_sha256": canonical_json_sha256(capital_allocation),
        "capital_allocation_board_sha256": "e" * 64,
        "capital_allocation_edge_basis": "EXACT_VEHICLE_ALL_EXIT_NET",
        "execution_cost_floor_sha256": "f" * 64,
        "guardian_action_receipt_material_contract": (
            GUARDIAN_ACTION_RECEIPT_MATERIAL_CONTRACT
        ),
        "guardian_action_receipt_baseline_pairs": guardian_baseline_pairs,
        "guardian_action_receipt_scope_state_sha256": canonical_json_sha256(
            guardian_scope_material
        ),
        "authorized_size_multiple": capital_allocation["size_multiple"],
        "authorized_units": capital_allocation["selected_units"],
        "execution_fields_preserved": True,
        "risk_envelope_not_expanded": True,
        "live_permission_granted": False,
    }


def _synthetic_guardian_baseline_pairs(
    decision: dict,
    *,
    lane_ids: list[str],
) -> list[str]:
    pairs: set[str] = set()
    for lane_id in lane_ids:
        if not isinstance(lane_id, str):
            continue
        pairs.update(
            token.strip().upper()
            for token in lane_id.split(":")
            if token.strip().upper() in DEFAULT_TRADER_PAIRS
        )

    market_read = decision.get("market_read_first")
    market_read = market_read if isinstance(market_read, dict) else {}
    naked = market_read.get("naked_read")
    if isinstance(naked, dict):
        pair = str(naked.get("cleanest_pair_expression") or "").strip().upper()
        if pair in DEFAULT_TRADER_PAIRS:
            pairs.add(pair)
    for section_name in (
        "next_30m_prediction",
        "next_2h_prediction",
        "best_trade_if_forced",
    ):
        section = market_read.get(section_name)
        if not isinstance(section, dict):
            continue
        pair = str(section.get("pair") or "").strip().upper()
        if pair in DEFAULT_TRADER_PAIRS:
            pairs.add(pair)
    return sorted(pairs)


def _market_read_first(*, pair: str = "EUR_USD", direction: str = "LONG") -> dict:
    base, quote = pair.split("_", 1) if "_" in pair else (pair, "USD")
    bought = base if direction == "LONG" else quote
    sold = quote if direction == "LONG" else base
    return {
        "naked_read": {
            "currency_bought": bought,
            "currency_sold": sold,
            "cleanest_pair_expression": pair,
            "is_cleanest_currency_theme": f"YES - {pair} is the cleanest current expression.",
            "location_24h": "LOWER",
            "h1_h4_alignment": "H1=WITH_H1_TREND; H4=WITH_H4_TREND",
            "tape_state": "TREND",
            "known_winning_trade_shape_match": "MATCH - generalized 2025 operator trade shape.",
            "proposed_building_style_allowed": "YES - SINGLE",
            "thesis_state": "ALIVE",
            "what_price_is_trying_to_do_now": f"{pair} is pressing {direction} from the current shelf before execution filters.",
        },
        "next_30m_prediction": {
            "pair": pair,
            "direction": direction,
            "expected_path": f"Next 30m {pair} should hold the shelf and push toward 1.1740.",
            "target_zone": "1.1740",
            "invalidation": "1.1700",
        },
        "next_2h_prediction": {
            "pair": pair,
            "direction": direction,
            "expected_path": f"Next 2h {pair} should extend into 1.1760 if the shelf holds.",
            "target_zone": "1.1760",
            "invalidation": "1.1700",
        },
        "best_trade_if_forced": {
            "pair": pair,
            "direction": direction,
            "vehicle": "MARKET",
            "entry": "1.1721",
            "tp": "1.1737",
            "sl": "1.1717",
            "why_this_pays": "The forced trade only pays if the naked read reaches target before invalidation.",
        },
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
        "horizon_minutes": 60,
        "primary_path": f"{pair} should hold the M5 shelf and press toward the selected trigger before the next cycle.",
        "failure_path": "A close back through the shelf or a newly named packet blocker makes the idea wrong.",
        "entry_or_hold_trigger": "Use only the current LIVE_READY intent trigger or hold WAIT if that trigger is absent.",
        "invalidation_or_cancel_trigger": "Cancel the idea if the invalidation shelf breaks or the selected intent leaves LIVE_READY.",
        "counterargument": "M15 can still fade the move; the trade is only acceptable because current chart refs keep the shelf intact.",
        "next_cycle_check": "First re-check broker truth, the selected lane status, and M5/M15 structure before extending the thesis.",
        "evidence_refs": refs,
    }


def _user_alpha_overrides() -> dict:
    latest = {
        "edge_source": "USER_ALPHA",
        "classification": "OPERATOR_ALPHA",
        "discovered_by": "OPERATOR",
        "system_discovered": False,
        "system_tp_managed": True,
        "outcome_id": "manual-eurusd-long",
        "trade_id": "manual-eurusd-long",
        "pair": "EUR_USD",
        "direction": "LONG",
        "entry": 1.172,
        "tp": 1.174,
        "realized_pl_jpy": 2300.0,
        "max_favorable_excursion": None,
        "time_to_tp_seconds": 7200,
        "thesis": "operator found EUR_USD long continuation before the system reload",
        "closed_at_utc": "2026-06-29T04:00:00Z",
        "continuation_required": True,
    }
    return {
        "expires_at_utc": "2099-01-01T00:00:00Z",
        "user_alpha_trades": [latest],
        "user_alpha_continuation": {
            "status": "ACTIVE",
            "active": True,
            "edge_source": "USER_ALPHA",
            "latest_trade": latest,
            "five_pct_path_board_candidate": {
                "source": "USER_ALPHA",
                "pair": "EUR_USD",
                "direction": "LONG",
                "candidate_roles": ["RELOAD", "SECOND_SHOT"],
                "target_layer": "PACE_5",
            },
            "required_trader_answers": [
                "thesis_alive",
                "reload_candidate",
                "second_shot_candidate",
                "exact_blocker_if_no_continuation",
                "next_trigger",
            ],
            "if_no_continuation_requires_exact_blocker": True,
        },
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
        "market_read_first": _market_read_first(pair="EUR_USD", direction="LONG"),
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
        "market_read_first": _market_read_first(pair="EUR_USD", direction="LONG"),
        "action": "WAIT",
        "selected_lane_id": None,
        "confidence": "MEDIUM",
        "thesis": "MARKET READ FIRST next 30m/next 2h EUR_USD LONG path is noted, but wait because timing is not clean enough.",
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
        "operator_summary": "Wait with an explicit rejection of the current executable lane after the next 30m and next 2h market read.",
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
        "market_read_first": _market_read_first(pair="EUR_USD", direction="LONG"),
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
        "market_read_first": _market_read_first(pair="EUR_USD", direction="LONG"),
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
    snapshot_at = datetime.fromisoformat(snap["fetched_at_utc"])
    raw = snap["positions"][0].get("raw")
    snap["positions"][0]["raw"] = {
        **(raw if isinstance(raw, dict) else {}),
        "openTime": (snapshot_at - timedelta(hours=8)).isoformat(),
    }
    snap["quotes"] = {
        "EUR_USD": {"bid": quote_bid, "ask": quote_ask, "timestamp_utc": snap["fetched_at_utc"]},
    }
    files["snapshot"].write_text(json.dumps(snap))
    # Override pair_charts chart_story so the CLOSE gate sees the
    # M15/H4 structural events we want.
    pc = json.loads(files["pair_charts"].read_text())
    pc["charts"][0]["chart_story"] = _chart_story_with_struct("EUR_USD", m15_dir=m15_dir, h4_dir=h4_dir)
    views = _close_tech_views("UP" if position_side == "SHORT" else "DOWN")
    views.append(
        {
            "granularity": "H4",
            "regime": "TREND_UP" if h4_dir == "UP" else "TREND_DOWN",
            "indicators": {},
            "structure": {
                "last_event": {
                    "kind": f"CHOCH_{h4_dir}",
                    "close_confirmed": True,
                    "broken_pivot_price": 1.2,
                    "timestamp": (snapshot_at - timedelta(hours=4)).isoformat(),
                }
            },
        }
    )
    pc["charts"][0]["views"] = views
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


def _write_fresh_position_hold_support(
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
    (root / "position_thesis_report.json").write_text(
        json.dumps(
            {
                "generated_at_utc": generated_at,
                "assessments": [
                    {
                        "trade_id": trade_id,
                        "pair": pair,
                        "side": side,
                        "verdict": "EXTEND",
                        "aggregate_score": 20.55,
                        "rationale_lines": ["position thesis still supports same-side carry"],
                        "context_notes": [],
                    }
                ],
            }
        )
    )
    (root / "thesis_evolution_report.json").write_text(
        json.dumps(
            {
                "generated_at_utc": generated_at,
                "evolutions": [
                    {
                        "trade_id": trade_id,
                        "pair": pair,
                        "side": side,
                        "status": "STILL_VALID",
                        "verdict": "HOLD",
                        "rationale": "entry forecast remains aligned with current forecast",
                    }
                ],
            }
        )
    )
    (root / "forecast_persistence_report.json").write_text(
        json.dumps(
            {
                "generated_at_utc": generated_at,
                "verdicts": [
                    {
                        "trade_id": trade_id,
                        "pair": pair,
                        "side": side,
                        "verdict": "EXTEND",
                        "reason": "last forecasts aligned with the open position",
                    }
                ],
            }
        )
    )


def _write_fresh_evolution_and_persistence_hold_support(
    root: Path,
    files: dict[str, Path],
    *,
    trade_id: str = "555",
    pair: str = "EUR_USD",
    side: str = "LONG",
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
                        "status": "WEAKENED",
                        "verdict": "HOLD",
                        "rationale": "current forecast still supports the open position side",
                    }
                ],
            }
        )
    )
    (root / "forecast_persistence_report.json").write_text(
        json.dumps(
            {
                "generated_at_utc": generated_at,
                "verdicts": [
                    {
                        "trade_id": trade_id,
                        "pair": pair,
                        "side": side,
                        "verdict": "EXTEND",
                        "reason": "recent forecasts still support the open position side",
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


def _write_entry_thesis(
    root: Path,
    *,
    trade_id: str = "555",
    pair: str = "EUR_USD",
    side: str = "SHORT",
    invalidation_price: float | None = None,
    timestamp_utc: str | None = None,
) -> None:
    (root / "entry_thesis_ledger.jsonl").write_text(
        json.dumps(
            {
                "timestamp_utc": timestamp_utc or datetime.now(timezone.utc).isoformat(),
                "trade_id": trade_id,
                "pair": pair,
                "side": side,
                "entry_price": 1.17500,
                "forecast_direction": "RANGE",
                "forecast_confidence": 0.72,
                "regime": "RANGE",
                "invalidation_price": invalidation_price,
                "target_price": None,
                "key_drivers": ["test-entry-thesis"],
                "context_evidence": {},
                "horizon_hours": 6.0,
            }
        )
        + "\n"
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
    rationale: str | None = None,
) -> None:
    if rationale is None:
        if side.upper() == "LONG":
            rationale = (
                "invalidation hit: current bid 1.16900 <= buffered invalidation "
                "1.16930 (raw 1.16950, buffer 2.0p); technical invalidation "
                "confirmed against LONG: H1 BOS_DOWN; M15 MACD-"
            )
        else:
            rationale = (
                "invalidation hit: current ask 1.16310 >= buffered invalidation "
                "1.16290 (raw 1.16270, buffer 2.0p); technical invalidation "
                "confirmed against SHORT: H1 BOS_UP; M15 MACD+"
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
                        "trade_id": trade_id,
                        "pair": pair,
                        "side": side,
                        "status": "BROKEN",
                        "verdict": "RECOMMEND_CLOSE",
                        "rationale": rationale,
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
    path_name: str = "position_management.json",
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
    (root / path_name).write_text(
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


def _write_recent_position_management_hold_support(
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
            "latest forecast RANGE conf=0.78",
            "TP/SL present and current thesis is not contradicted enough to force exit",
            "remaining reward about 1235 JPY",
        ]
    (root / "position_management.json").write_text(
        json.dumps(
            {
                "generated_at_utc": generated_at,
                "action": "HOLD_PROTECTED",
                "positions": [
                    {
                        "trade_id": trade_id,
                        "pair": pair,
                        "side": side,
                        "action": "HOLD_PROTECTED",
                        "reasons": reasons,
                    }
                ],
            }
        )
    )


def _pre_repair_only_tp_progress_timing_audit() -> dict:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "OK",
        "summary": {
            "loss_closes_audited": 14,
            "loss_closes_profit_capture_missed": 14,
            "loss_closes_repair_replay_triggered": 13,
            "loss_close_actual_pl_jpy": -5188.197,
            "loss_close_counterfactual_profit_capture_pl_jpy": -4134.026,
            "loss_close_counterfactual_profit_capture_delta_jpy": 1054.171,
            "loss_close_counterfactual_profit_capture_jpy": 474.341,
            "tp_progress_repair_live_evidence_boundary_utc": (
                TP_PROGRESS_REPAIR_LIVE_EVIDENCE_BOUNDARY_UTC
            ),
            "tp_progress_repair_live_evidence_status": "WAITING_FOR_POST_REPAIR_SAMPLE",
            "pre_repair_historical_loss_closes_audited": 14,
            "pre_repair_historical_loss_closes_profit_capture_missed": 14,
            "pre_repair_historical_loss_closes_repair_replay_triggered": 13,
            "post_repair_live_evidence_loss_closes_audited": 0,
            "post_repair_live_evidence_loss_closes_profit_capture_missed": 0,
            "post_repair_live_evidence_loss_closes_repair_replay_triggered": 0,
        },
    }


class CloseDisciplineTest(unittest.TestCase):
    """Coverage for 2026-05-12 CLOSE two-gate discipline added in
    response to the 2026-05-11 18:17 UTC mass-close regression where the
    GPT trader autonomously closed four valid SHORT positions for
    -3,291 JPY. Mirrors `feedback_no_unilateral_close.md` and the
    AGENT_CONTRACT §10 CLOSE discipline section.
    """

    def setUp(self) -> None:
        self._prior_override = _os.environ.pop("QR_OPERATOR_CLOSE_OVERRIDE", None)
        self._prior_spread_override = _os.environ.pop("QR_POSITION_CLOSE_SPREAD_OVERRIDE", None)

    def tearDown(self) -> None:
        if self._prior_override is None:
            _os.environ.pop("QR_OPERATOR_CLOSE_OVERRIDE", None)
        else:
            _os.environ["QR_OPERATOR_CLOSE_OVERRIDE"] = self._prior_override
        if self._prior_spread_override is None:
            _os.environ.pop("QR_POSITION_CLOSE_SPREAD_OVERRIDE", None)
        else:
            _os.environ["QR_POSITION_CLOSE_SPREAD_OVERRIDE"] = self._prior_spread_override

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

    def test_loss_close_rejected_when_negative_pl_is_only_reason(self) -> None:
        _os.environ["QR_OPERATOR_CLOSE_OVERRIDE"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(
                root,
                position_side="SHORT",
                m15_dir="DOWN",
                h4_dir="DOWN",
                unrealized_pl_jpy=-200.0,
            )
            decision = _close_decision(trade_ids=["555"])
            decision["thesis"] = "Close because unrealized P/L is negative, not because structure failed."
            decision["risk_notes"] = ["negative P/L on a red position is the close reason."]
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("CLOSE_THESIS_STILL_VALID", codes)
            self.assertIn("THESIS_INVALIDATION_EXIT_REQUIRED", codes)

    def test_loss_close_rejected_when_negative_expectancy_is_only_reason(self) -> None:
        _os.environ["QR_OPERATOR_CLOSE_OVERRIDE"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(
                root,
                position_side="SHORT",
                m15_dir="DOWN",
                h4_dir="DOWN",
                unrealized_pl_jpy=-200.0,
            )
            decision = _close_decision(trade_ids=["555"])
            decision["risk_notes"] = [
                "NEGATIVE_EXPECTANCY says the execution shape is weak; close this trade."
            ]
            decision["evidence_refs"].append("self_improvement:finding:NEGATIVE_EXPECTANCY")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("CLOSE_THESIS_STILL_VALID", codes)
            self.assertIn("THESIS_INVALIDATION_EXIT_REQUIRED", codes)

    def test_close_accepted_when_m15_bos_against_side_and_operator_authorized(self) -> None:
        # SHORT position + M15 prints BOS_UP (against SHORT) → soft Gate A.
        # Explicit env Gate B is required because M15-only structure is too
        # local to be unattended standing loss-cut authorization.
        _os.environ["QR_OPERATOR_CLOSE_OVERRIDE"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="UP", h4_dir="DOWN")
            decision = _close_decision(trade_ids=["555"])
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED", msg=summary)
            self.assertTrue(summary.allowed)

    def test_m15_bos_against_side_requires_explicit_gate_b(self) -> None:
        # Repeated live loss closes showed M15-only BOS/CHOCH is often an
        # internal pullback. It is Gate A evidence, but not a no-token hard
        # authorization unless H4 / recorded invalidation / hard sidecar also
        # confirms.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="UP", h4_dir="DOWN")
            decision = _close_decision(trade_ids=["555"])
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED", msg=summary)
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("CLOSE_OPERATOR_AUTH_REQUIRED", codes)
            self.assertNotIn("CLOSE_SAME_DIRECTION_MARKET_SUPPORT", codes)
            self.assertNotIn("CLOSE_THESIS_STILL_VALID", codes)
            evidence = payload["close_gate_evidence"][0]
            self.assertEqual(evidence["trade_id"], "555")
            self.assertEqual(evidence["pair"], "EUR_USD")
            self.assertEqual(evidence["side"], "SHORT")
            self.assertTrue(evidence["gate_a_invalidated"])
            self.assertIn("M15", evidence["gate_a_reason"])
            self.assertFalse(evidence["gate_b_standing_authorized"])
            self.assertTrue(evidence["explicit_gate_b_required"])
            self.assertFalse(evidence["gate_b_explicit_operator_authorized"])

    def test_close_rejected_when_m15_break_conflicts_with_hold_sidecars(self) -> None:
        # Regression for 2026-06-12 USD_CHF: M15 can flip during an internal
        # pullback while the position thesis, thesis evolution, and forecast
        # persistence still support the swing. That must not become another
        # low-quality loss-side market close.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="LONG", m15_dir="DOWN", h4_dir="UP")
            _write_fresh_position_hold_support(root, files, side="LONG")
            decision = _close_decision(trade_ids=["555"])
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED", msg=summary)
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("CLOSE_THESIS_STILL_VALID", codes)
            self.assertEqual(
                len(payload["input_packet"]["protection_sidecars"]["position_hold_support"]),
                3,
            )

    def test_h4_break_still_closes_even_when_m15_has_hold_sidecars(self) -> None:
        # The support veto is intentionally scoped to M15 internal pullbacks.
        # H4 close-confirmed structure beyond the current reward target remains
        # a hard close.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="DOWN", h4_dir="UP")
            _write_fresh_position_hold_support(root, files, side="SHORT")
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
            evidence = payload["close_gate_evidence"][0]
            self.assertEqual(evidence["trade_id"], "555")
            self.assertTrue(evidence["loss_side_close"])
            self.assertTrue(evidence["gate_a_invalidated"])
            self.assertIn("H4", evidence["gate_a_reason"])
            self.assertTrue(evidence["gate_b_standing_authorized"])
            self.assertFalse(evidence["explicit_gate_b_required"])
            self.assertFalse(evidence["gate_b_explicit_operator_authorized"])
            h4_event = payload["input_packet"]["market_context"]["pairs"]["EUR_USD"]["chart"]["views"]["H4"]["structure"]["last_event"]
            self.assertIsNotNone(h4_event["timestamp"])
            position = payload["input_packet"]["broker_snapshot"]["position_summaries"][0]
            self.assertIsNotNone(position["open_time_utc"])

    def test_pre_entry_h4_structure_is_soft_gate_a_without_operator_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="DOWN", h4_dir="UP")
            snapshot = json.loads(files["snapshot"].read_text())
            opened_at = datetime.fromisoformat(snapshot["positions"][0]["raw"]["openTime"])
            charts = json.loads(files["pair_charts"].read_text())
            h4 = next(view for view in charts["charts"][0]["views"] if view["granularity"] == "H4")
            h4["structure"]["last_event"]["timestamp"] = (
                opened_at - timedelta(hours=4)
            ).isoformat()
            files["pair_charts"].write_text(json.dumps(charts))
            brain = _brain(root, files, _close_decision(trade_ids=["555"]))

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED", msg=summary)
            payload = json.loads((root / "gpt_decision.json").read_text())
            evidence = payload["close_gate_evidence"][0]
            self.assertTrue(evidence["gate_a_invalidated"])
            self.assertFalse(evidence["gate_b_standing_authorized"])
            self.assertTrue(evidence["explicit_gate_b_required"])
            self.assertIn("does not postdate", evidence["gate_a_reason"])
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("CLOSE_OPERATOR_AUTH_REQUIRED", codes)

    def test_h4_structure_without_timestamp_is_soft_gate_a(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="DOWN", h4_dir="UP")
            charts = json.loads(files["pair_charts"].read_text())
            h4 = next(view for view in charts["charts"][0]["views"] if view["granularity"] == "H4")
            h4["structure"]["last_event"].pop("timestamp")
            files["pair_charts"].write_text(json.dumps(charts))
            brain = _brain(root, files, _close_decision(trade_ids=["555"]))

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED", msg=summary)
            payload = json.loads((root / "gpt_decision.json").read_text())
            evidence = payload["close_gate_evidence"][0]
            self.assertTrue(evidence["gate_a_invalidated"])
            self.assertFalse(evidence["gate_b_standing_authorized"])
            self.assertIn("timestamp is missing or invalid", evidence["gate_a_reason"])

    def test_pre_entry_h4_structure_can_close_only_with_explicit_gate_b(self) -> None:
        with patch("quant_rabbit.gpt_trader._operator_close_gate_authorized", return_value=True):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                files = _close_fixtures(root, position_side="SHORT", m15_dir="DOWN", h4_dir="UP")
                snapshot = json.loads(files["snapshot"].read_text())
                opened_at = datetime.fromisoformat(snapshot["positions"][0]["raw"]["openTime"])
                charts = json.loads(files["pair_charts"].read_text())
                h4 = next(view for view in charts["charts"][0]["views"] if view["granularity"] == "H4")
                h4["structure"]["last_event"]["timestamp"] = (
                    opened_at - timedelta(hours=4)
                ).isoformat()
                files["pair_charts"].write_text(json.dumps(charts))
                brain = _brain(root, files, _close_decision(trade_ids=["555"]))

                summary = brain.run(snapshot_path=files["snapshot"])

                self.assertEqual(summary.status, "ACCEPTED", msg=summary)
                payload = json.loads((root / "gpt_decision.json").read_text())
                evidence = payload["close_gate_evidence"][0]
                self.assertFalse(evidence["gate_b_standing_authorized"])
                self.assertTrue(evidence["gate_b_explicit_operator_authorized"])

    def test_h4_structure_uses_entry_thesis_timestamp_when_broker_open_time_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="DOWN", h4_dir="UP")
            snapshot = json.loads(files["snapshot"].read_text())
            snapshot_at = datetime.fromisoformat(snapshot["fetched_at_utc"])
            snapshot["positions"][0].pop("raw", None)
            files["snapshot"].write_text(json.dumps(snapshot))
            thesis_at = snapshot_at - timedelta(hours=6)
            _write_entry_thesis(root, timestamp_utc=thesis_at.isoformat())
            brain = _brain(root, files, _close_decision(trade_ids=["555"]))

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED", msg=summary)
            payload = json.loads((root / "gpt_decision.json").read_text())
            context = payload["input_packet"]["protection_sidecars"]["entry_thesis_close_context"][0]
            self.assertEqual(context["entry_thesis_timestamp_utc"], thesis_at.isoformat())
            evidence = payload["close_gate_evidence"][0]
            self.assertTrue(evidence["gate_b_standing_authorized"])
            self.assertIn("entry-thesis timestamp", evidence["gate_a_reason"])

    def test_h4_structure_rejects_malformed_present_broker_open_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="DOWN", h4_dir="UP")
            snapshot = json.loads(files["snapshot"].read_text())
            snapshot_at = datetime.fromisoformat(snapshot["fetched_at_utc"])
            snapshot["positions"][0]["raw"]["openTime"] = "not-a-timestamp"
            files["snapshot"].write_text(json.dumps(snapshot))
            _write_entry_thesis(
                root,
                timestamp_utc=(snapshot_at - timedelta(hours=6)).isoformat(),
            )
            brain = _brain(root, files, _close_decision(trade_ids=["555"]))

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED", msg=summary)
            payload = json.loads((root / "gpt_decision.json").read_text())
            evidence = payload["close_gate_evidence"][0]
            self.assertFalse(evidence["gate_b_standing_authorized"])
            self.assertIn("broker openTime is present but invalid", evidence["gate_a_reason"])

    def test_h4_structure_rejects_malformed_present_thesis_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="DOWN", h4_dir="UP")
            _write_entry_thesis(root, timestamp_utc="not-a-timestamp")
            brain = _brain(root, files, _close_decision(trade_ids=["555"]))

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED", msg=summary)
            payload = json.loads((root / "gpt_decision.json").read_text())
            evidence = payload["close_gate_evidence"][0]
            self.assertFalse(evidence["gate_b_standing_authorized"])
            self.assertIn(
                "entry-thesis timestamp is present but invalid",
                evidence["gate_a_reason"],
            )

    def test_h4_structure_at_exact_open_time_is_soft_gate_a(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="DOWN", h4_dir="UP")
            snapshot = json.loads(files["snapshot"].read_text())
            opened_at = datetime.fromisoformat(snapshot["positions"][0]["raw"]["openTime"])
            charts = json.loads(files["pair_charts"].read_text())
            h4 = next(view for view in charts["charts"][0]["views"] if view["granularity"] == "H4")
            h4["structure"]["last_event"]["timestamp"] = opened_at.isoformat()
            files["pair_charts"].write_text(json.dumps(charts))
            brain = _brain(root, files, _close_decision(trade_ids=["555"]))

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED", msg=summary)
            payload = json.loads((root / "gpt_decision.json").read_text())
            evidence = payload["close_gate_evidence"][0]
            self.assertFalse(evidence["gate_b_standing_authorized"])
            self.assertIn("does not postdate", evidence["gate_a_reason"])

    def test_h4_structure_parses_oanda_nanosecond_open_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="DOWN", h4_dir="UP")
            snapshot = json.loads(files["snapshot"].read_text())
            snapshot_at = datetime.fromisoformat(snapshot["fetched_at_utc"])
            opened_at = snapshot_at - timedelta(hours=8)
            snapshot["positions"][0]["raw"]["openTime"] = (
                opened_at.strftime("%Y-%m-%dT%H:%M:%S.")
                + f"{opened_at.microsecond:06d}789Z"
            )
            files["snapshot"].write_text(json.dumps(snapshot))
            brain = _brain(root, files, _close_decision(trade_ids=["555"]))

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED", msg=summary)
            payload = json.loads((root / "gpt_decision.json").read_text())
            evidence = payload["close_gate_evidence"][0]
            self.assertTrue(evidence["gate_b_standing_authorized"])
            self.assertIn("broker openTime", evidence["gate_a_reason"])

    def test_loss_close_requires_timing_audit_after_premature_close_regrets(self) -> None:
        # Regression for 2026-06-18: recent GPT_CLOSE losses later touched TP
        # or continued favorably, but a new hard-close receipt could ignore
        # execution_timing_audit entirely. Hard invalidation may still close,
        # but the receipt must show it weighed the timing-regret evidence.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="UP", h4_dir="UP")
            files["execution_timing_audit"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "OK",
                        "summary": {
                            "loss_market_closes_may_have_been_premature": 2,
                            "market_close_estimated_followthrough_jpy": 3027.25,
                        },
                    }
                )
            )
            decision = _close_decision(trade_ids=["555"], operator_close_authorized=False)
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("CLOSE_TIMING_AUDIT_REQUIRED", codes)

    def test_loss_close_requires_timing_audit_after_profit_capture_miss_counterfactuals(self) -> None:
        # Regression for 2026-06-22: candle replay proved loss closes had
        # bankable TP-progress profit before turning red. A new underwater
        # CLOSE must cite that evidence instead of repeating the leak.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(
                root,
                position_side="SHORT",
                m15_dir="UP",
                h4_dir="UP",
            )
            files["execution_timing_audit"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "OK",
                        "summary": {
                            "loss_closes_audited": 9,
                            "loss_closes_profit_capture_missed": 2,
                            "loss_close_actual_pl_jpy": -5188.197,
                            "loss_close_counterfactual_profit_capture_pl_jpy": -4134.026,
                            "loss_close_counterfactual_profit_capture_delta_jpy": 1054.171,
                            "loss_close_counterfactual_profit_capture_jpy": 474.341,
                        },
                    }
                )
            )
            decision = _close_decision(trade_ids=["555"], operator_close_authorized=False)
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            issues = {
                issue["code"]: issue["message"]
                for issue in payload["verification_issues"]
            }
            self.assertIn("CLOSE_TIMING_AUDIT_REQUIRED", issues)
            self.assertIn(
                "profit-capture counterfactual delta 1054.17 JPY",
                issues["CLOSE_TIMING_AUDIT_REQUIRED"],
            )

    def test_loss_close_does_not_require_timing_audit_for_pre_repair_only_tp_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(
                root,
                position_side="SHORT",
                m15_dir="UP",
                h4_dir="UP",
            )
            files["execution_timing_audit"].write_text(
                json.dumps(_pre_repair_only_tp_progress_timing_audit())
            )
            decision = _close_decision(trade_ids=["555"], operator_close_authorized=False)
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED", msg=summary)
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertNotIn("CLOSE_TIMING_AUDIT_REQUIRED", codes)

    def test_loss_close_with_timing_audit_ref_uses_normal_close_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="UP", h4_dir="UP")
            files["execution_timing_audit"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "OK",
                        "summary": {
                            "loss_market_closes_may_have_been_premature": 2,
                            "market_close_estimated_followthrough_jpy": 3027.25,
                        },
                    }
                )
            )
            decision = _close_decision(trade_ids=["555"], operator_close_authorized=False)
            decision["evidence_refs"].append("timing:audit")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED", msg=summary)
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertNotIn("CLOSE_TIMING_AUDIT_REQUIRED", codes)

    def test_soft_loss_close_with_timing_ref_still_requires_hard_gate_when_premature_leak_active(self) -> None:
        # A generic timing:audit citation must not launder another soft
        # operator-token market close while the timing audit says recent
        # loss-side MARKET_ORDER_TRADE_CLOSE exits were premature. M15-only
        # structure remains Gate A evidence, but it needs either no active
        # premature-loss timing guard or hard Gate A confirmation.
        _os.environ["QR_OPERATOR_CLOSE_OVERRIDE"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="UP", h4_dir="DOWN")
            files["execution_timing_audit"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "OK",
                        "summary": {
                            "loss_market_closes_audited": 6,
                            "loss_market_closes_may_have_been_premature": 3,
                            "loss_market_closes_contained_risk": 3,
                            "market_close_estimated_followthrough_jpy": 8236.17,
                        },
                    }
                )
            )
            decision = _close_decision(trade_ids=["555"])
            decision["evidence_refs"].append("timing:audit")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("CLOSE_PREMATURE_TIMING_HARD_GATE_REQUIRED", codes)
            self.assertNotIn("CLOSE_TIMING_AUDIT_REQUIRED", codes)
            self.assertNotIn("CLOSE_OPERATOR_AUTH_REQUIRED", codes)

    def test_soft_loss_close_with_profit_capture_timing_ref_still_requires_hard_gate(self) -> None:
        # A generic timing citation acknowledges the missed-capture replay, but
        # soft M15-only Gate A plus operator token must not authorize another
        # underwater market close while that loss-close leak is active.
        _os.environ["QR_OPERATOR_CLOSE_OVERRIDE"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(
                root,
                position_side="SHORT",
                m15_dir="UP",
                h4_dir="DOWN",
            )
            files["execution_timing_audit"].write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "OK",
                        "summary": {
                            "loss_closes_audited": 9,
                            "loss_closes_profit_capture_missed": 2,
                            "loss_close_actual_pl_jpy": -5188.197,
                            "loss_close_counterfactual_profit_capture_pl_jpy": -4134.026,
                            "loss_close_counterfactual_profit_capture_delta_jpy": 1054.171,
                            "loss_close_counterfactual_profit_capture_jpy": 474.341,
                        },
                    }
                )
            )
            decision = _close_decision(trade_ids=["555"])
            decision["evidence_refs"].append("timing:audit")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("CLOSE_PREMATURE_TIMING_HARD_GATE_REQUIRED", codes)
            self.assertNotIn("CLOSE_TIMING_AUDIT_REQUIRED", codes)
            self.assertNotIn("CLOSE_OPERATOR_AUTH_REQUIRED", codes)

    def test_soft_loss_close_with_timing_ref_ignores_pre_repair_only_tp_history(self) -> None:
        _os.environ["QR_OPERATOR_CLOSE_OVERRIDE"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(
                root,
                position_side="SHORT",
                m15_dir="UP",
                h4_dir="DOWN",
            )
            files["execution_timing_audit"].write_text(
                json.dumps(_pre_repair_only_tp_progress_timing_audit())
            )
            decision = _close_decision(trade_ids=["555"])
            decision["evidence_refs"].append("timing:audit")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED", msg=summary)
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertNotIn("CLOSE_PREMATURE_TIMING_HARD_GATE_REQUIRED", codes)
            self.assertNotIn("CLOSE_TIMING_AUDIT_REQUIRED", codes)
            self.assertNotIn("CLOSE_OPERATOR_AUTH_REQUIRED", codes)

    def test_close_spread_cap_uses_pair_chart_session_tag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(
                root,
                position_side="SHORT",
                m15_dir="UP",
                h4_dir="UP",
                quote_bid=1.17600,
                quote_ask=1.17617,
            )
            charts = json.loads(files["pair_charts"].read_text())
            charts["charts"][0]["session"]["current_tag"] = "OFF_HOURS"
            files["pair_charts"].write_text(json.dumps(charts))
            decision = _close_decision(trade_ids=["555"], operator_close_authorized=False)
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            self.assertTrue(summary.allowed)
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertNotIn("POSITION_CLOSE_SPREAD_TOO_WIDE", codes)

    def test_close_rejected_when_flow_spread_exceeds_close_cap(self) -> None:
        # Hard Gate A still cannot authorize paying a stressed spread. This
        # reproduces the 2026-06-11 GBP_CHF GPT_CLOSE path where the receipt
        # cited flow stress, then paid the same stressed market close.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="UP", h4_dir="UP")
            flow = json.loads(files["flow"].read_text())
            flow["spreads"][0].update(
                {
                    "current_pips": 30.0,
                    "median_pips": 2.65,
                    "p90_pips": 2.8,
                    "stress_flag": "STRESSED",
                }
            )
            files["flow"].write_text(json.dumps(flow))
            decision = _close_decision(trade_ids=["555"], operator_close_authorized=False)
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED")
            self.assertFalse(summary.allowed)
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("POSITION_CLOSE_FLOW_SPREAD_TOO_WIDE", codes)
            self.assertNotIn("CLOSE_OPERATOR_AUTH_REQUIRED", codes)
            self.assertNotIn("CLOSE_THESIS_STILL_VALID", codes)

    def test_close_spread_override_allows_flow_stressed_close(self) -> None:
        _os.environ["QR_POSITION_CLOSE_SPREAD_OVERRIDE"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="UP", h4_dir="UP")
            flow = json.loads(files["flow"].read_text())
            flow["spreads"][0].update(
                {
                    "current_pips": 30.0,
                    "median_pips": 2.65,
                    "p90_pips": 2.8,
                    "stress_flag": "STRESSED",
                }
            )
            files["flow"].write_text(json.dumps(flow))
            decision = _close_decision(trade_ids=["555"], operator_close_authorized=False)
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED")
            self.assertTrue(summary.allowed)

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

    def test_thesis_evolution_close_rejected_when_matrix_still_supports_open_side(self) -> None:
        # Regression for 2026-06-15 EUR_CHF 472445: thesis_evolution BROKEN
        # was treated as standing hard loss-cut authorization even though the
        # receipt itself cited H4 still with the LONG side and matrix support
        # 4:1 for the open direction. That is a soft Gate A conflict, not an
        # unattended market-loss close.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="LONG", m15_dir="DOWN", h4_dir="UP")
            _write_same_direction_context_asset_matrix_support(files, side="LONG")
            _write_fresh_thesis_evolution_close_recommendation(
                root,
                files,
                side="LONG",
                rationale=(
                    "invalidation hit: current bid 1.16900 <= buffered invalidation "
                    "1.16930 (raw 1.16950, buffer 2.0p); technical invalidation "
                    "confirmed against LONG: M5/M15/M30/H1; H4 still supports "
                    "the open side"
                ),
            )
            decision = _close_decision(trade_ids=["555"])
            decision["evidence_refs"].extend(["position:evolution:555", "matrix:EUR_USD:LONG"])
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED", msg=summary)
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("CLOSE_SAME_DIRECTION_MARKET_SUPPORT", codes)
            self.assertNotIn("CLOSE_THESIS_STILL_VALID", codes)

    def test_h4_counter_structure_above_long_tp_does_not_hard_authorize_loss_close(self) -> None:
        # Regression for 2026-06-19 NZD_USD 472743: a LONG range-harvest
        # position with TP still below an H4 CHOCH_DOWN price was loss-closed
        # as if that old overhead structure invalidated the still-reachable TP.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(
                root,
                position_side="LONG",
                m15_dir="UP",
                h4_dir="DOWN",
                quote_bid=1.17600,
                quote_ask=1.17610,
            )
            decision = _close_decision(trade_ids=["555"])
            decision["evidence_refs"].extend(["chart:EUR_USD:H4", "position:management:555"])
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED", msg=summary)
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("CLOSE_THESIS_STILL_VALID", codes)
            self.assertNotIn("CLOSE_OPERATOR_AUTH_REQUIRED", codes)

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

    def test_forecast_persistence_close_rejected_when_position_management_still_holds_loss_side(self) -> None:
        # Regression for 2026-06-17 AUD_NZD 472632: forecast_persistence
        # RECOMMEND_CLOSE overrode a fresh PositionManager HOLD_PROTECTED read,
        # and the trade later touched TP after the red market close.
        prior = _os.environ.get("QR_OPERATOR_CLOSE_OVERRIDE")
        _os.environ["QR_OPERATOR_CLOSE_OVERRIDE"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                files = _close_fixtures(root, position_side="SHORT", m15_dir="DOWN", h4_dir="DOWN")
                _write_fresh_forecast_close_recommendation(root, files)
                _write_recent_position_management_hold_support(root, files)
                decision = _close_decision(trade_ids=["555"])
                decision["evidence_refs"].extend(["position:persistence:555", "position:management:555"])
                brain = _brain(root, files, decision)

                summary = brain.run(snapshot_path=files["snapshot"])

                self.assertEqual(summary.status, "REJECTED", msg=summary)
                payload = json.loads((root / "gpt_decision.json").read_text())
                codes = {issue["code"] for issue in payload["verification_issues"]}
                self.assertIn("CLOSE_SAME_DIRECTION_MARKET_SUPPORT", codes)
                self.assertNotIn("CLOSE_OPERATOR_AUTH_REQUIRED", codes)
                support = payload["input_packet"]["protection_sidecars"]["position_hold_support"]
                self.assertTrue(any(row.get("source") == "position_management" for row in support))
                rec = payload["input_packet"]["protection_sidecars"]["position_close_recommendations"][0]
                self.assertFalse(rec["blocks_non_close_actions"])
                self.assertIn("position_management", rec.get("non_blocking_reason", ""))
        finally:
            if prior is None:
                _os.environ.pop("QR_OPERATOR_CLOSE_OVERRIDE", None)
            else:
                _os.environ["QR_OPERATOR_CLOSE_OVERRIDE"] = prior

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

    def test_soft_loss_close_under_profitability_p0_must_cite_self_improvement_context(self) -> None:
        # Regression for 2026-06-19 NZD_USD 472743: a soft position_thesis
        # REVIEW_CLOSE during persistent MARKET_ORDER_TRADE_CLOSE leakage must
        # explicitly account for the active profitability P0 instead of
        # laundering another red market close through generic operator Gate B.
        prior = _os.environ.get("QR_OPERATOR_CLOSE_OVERRIDE")
        _os.environ["QR_OPERATOR_CLOSE_OVERRIDE"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                files = _close_fixtures(root, position_side="SHORT", m15_dir="DOWN", h4_dir="DOWN")
                files["self_improvement_audit"].write_text(json.dumps(_self_improvement_profitability_p0()))
                _write_fresh_position_thesis_close_recommendation(root, files)
                decision = _close_decision(trade_ids=["555"])
                decision["evidence_refs"].append("position:thesis:555")
                brain = _brain(root, files, decision)

                summary = brain.run(snapshot_path=files["snapshot"])

                self.assertEqual(summary.status, "REJECTED")
                payload = json.loads((root / "gpt_decision.json").read_text())
                codes = {issue["code"] for issue in payload["verification_issues"]}
                self.assertIn("CLOSE_PROFITABILITY_P0_CONTEXT_REQUIRED", codes)
                self.assertNotIn("CLOSE_OPERATOR_AUTH_REQUIRED", codes)
                evidence = payload["close_gate_evidence"][0]
                self.assertTrue(evidence["profitability_p0_context_required"])
                self.assertFalse(evidence["profitability_p0_context_cited"])
                self.assertTrue(evidence["gate_b_explicit_operator_authorized"])
        finally:
            if prior is None:
                _os.environ.pop("QR_OPERATOR_CLOSE_OVERRIDE", None)
            else:
                _os.environ["QR_OPERATOR_CLOSE_OVERRIDE"] = prior

    def test_soft_loss_close_under_profitability_p0_accepts_explicit_repair_context(self) -> None:
        prior = _os.environ.get("QR_OPERATOR_CLOSE_OVERRIDE")
        _os.environ["QR_OPERATOR_CLOSE_OVERRIDE"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                files = _close_fixtures(root, position_side="SHORT", m15_dir="DOWN", h4_dir="DOWN")
                files["self_improvement_audit"].write_text(json.dumps(_self_improvement_profitability_p0()))
                _write_fresh_position_thesis_close_recommendation(root, files)
                decision = _close_decision(trade_ids=["555"])
                decision["evidence_refs"].extend(
                    [
                        "position:thesis:555",
                        "self_improvement:finding:PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED",
                    ]
                )
                brain = _brain(root, files, decision)

                summary = brain.run(snapshot_path=files["snapshot"])

                self.assertEqual(summary.status, "ACCEPTED", msg=summary)
                payload = json.loads((root / "gpt_decision.json").read_text())
                codes = {issue["code"] for issue in payload["verification_issues"]}
                self.assertNotIn("CLOSE_PROFITABILITY_P0_CONTEXT_REQUIRED", codes)
        finally:
            if prior is None:
                _os.environ.pop("QR_OPERATOR_CLOSE_OVERRIDE", None)
            else:
                _os.environ["QR_OPERATOR_CLOSE_OVERRIDE"] = prior

    def test_loss_close_under_profitability_acceptance_p0_must_cite_acceptance_context(self) -> None:
        # Regression for 2026-06-19 NZD_USD 472743: the close receipt cited
        # timing/capture context but did not account for the red profitability
        # acceptance gate showing recent loss-side gateway close leakage.
        prior = _os.environ.get("QR_OPERATOR_CLOSE_OVERRIDE")
        _os.environ["QR_OPERATOR_CLOSE_OVERRIDE"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                files = _close_fixtures(root, position_side="LONG", m15_dir="DOWN", h4_dir="DOWN")
                files["self_improvement_audit"].write_text(json.dumps(_self_improvement_profitability_p0()))
                files["profitability_acceptance"].write_text(json.dumps(_profitability_acceptance_close_leak_p0()))
                _write_fresh_position_thesis_close_recommendation(root, files, side="LONG")
                decision = _close_decision(trade_ids=["555"])
                decision["evidence_refs"].extend(
                    [
                        "position:thesis:555",
                        "self_improvement:finding:PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED",
                        "timing:audit",
                    ]
                )
                brain = _brain(root, files, decision)

                summary = brain.run(snapshot_path=files["snapshot"])

                self.assertEqual(summary.status, "REJECTED")
                payload = json.loads((root / "gpt_decision.json").read_text())
                codes = {issue["code"] for issue in payload["verification_issues"]}
                self.assertIn("CLOSE_PROFITABILITY_ACCEPTANCE_P0_REQUIRED", codes)
                self.assertIn(
                    "profitability:acceptance",
                    payload["input_packet"]["allowed_evidence_refs"],
                )
        finally:
            if prior is None:
                _os.environ.pop("QR_OPERATOR_CLOSE_OVERRIDE", None)
            else:
                _os.environ["QR_OPERATOR_CLOSE_OVERRIDE"] = prior

    def test_loss_close_gate_evidence_missing_p0_must_cite_acceptance_context(self) -> None:
        # A missing durable close_gate_evidence P0 is itself a loss-close leak:
        # without citing it, another underwater CLOSE can keep producing
        # unverifiable market-close losses that the acceptance audit cannot clear.
        prior = _os.environ.get("QR_OPERATOR_CLOSE_OVERRIDE")
        _os.environ["QR_OPERATOR_CLOSE_OVERRIDE"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                files = _close_fixtures(root, position_side="LONG", m15_dir="DOWN", h4_dir="DOWN")
                files["self_improvement_audit"].write_text(json.dumps(_self_improvement_profitability_p0()))
                files["profitability_acceptance"].write_text(
                    json.dumps(_profitability_acceptance_close_gate_evidence_missing_p0())
                )
                _write_fresh_position_thesis_close_recommendation(root, files, side="LONG")
                decision = _close_decision(trade_ids=["555"])
                decision["evidence_refs"].extend(
                    [
                        "position:thesis:555",
                        "self_improvement:finding:PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED",
                        "timing:audit",
                    ]
                )
                brain = _brain(root, files, decision)

                summary = brain.run(snapshot_path=files["snapshot"])

                self.assertEqual(summary.status, "REJECTED")
                payload = json.loads((root / "gpt_decision.json").read_text())
                codes = {issue["code"] for issue in payload["verification_issues"]}
                self.assertIn("CLOSE_PROFITABILITY_ACCEPTANCE_P0_REQUIRED", codes)
                self.assertIn(
                    "LOSS_CLOSE_GATE_EVIDENCE_MISSING",
                    next(
                        issue["message"]
                        for issue in payload["verification_issues"]
                        if issue["code"] == "CLOSE_PROFITABILITY_ACCEPTANCE_P0_REQUIRED"
                    ),
                )
        finally:
            if prior is None:
                _os.environ.pop("QR_OPERATOR_CLOSE_OVERRIDE", None)
            else:
                _os.environ["QR_OPERATOR_CLOSE_OVERRIDE"] = prior

    def test_soft_loss_close_under_profitability_acceptance_p0_requires_hard_gate(self) -> None:
        # The acceptance P0 is stronger than an acknowledgement requirement:
        # while recent gateway loss closes are still inside the red window, a
        # soft REVIEW_CLOSE plus operator token can keep repeating the leak.
        prior = _os.environ.get("QR_OPERATOR_CLOSE_OVERRIDE")
        _os.environ["QR_OPERATOR_CLOSE_OVERRIDE"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                files = _close_fixtures(root, position_side="LONG", m15_dir="UP", h4_dir="UP")
                files["self_improvement_audit"].write_text(json.dumps(_self_improvement_profitability_p0()))
                files["profitability_acceptance"].write_text(json.dumps(_profitability_acceptance_close_leak_p0()))
                _write_fresh_position_thesis_close_recommendation(root, files, side="LONG")
                decision = _close_decision(trade_ids=["555"])
                decision["evidence_refs"].extend(
                    [
                        "position:thesis:555",
                        "self_improvement:finding:PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED",
                        "profitability:acceptance",
                    ]
                )
                brain = _brain(root, files, decision)

                summary = brain.run(snapshot_path=files["snapshot"])

                self.assertEqual(summary.status, "REJECTED")
                payload = json.loads((root / "gpt_decision.json").read_text())
                codes = {issue["code"] for issue in payload["verification_issues"]}
                self.assertIn("CLOSE_PROFITABILITY_ACCEPTANCE_HARD_GATE_REQUIRED", codes)
                self.assertNotIn("CLOSE_PROFITABILITY_ACCEPTANCE_P0_REQUIRED", codes)
                self.assertNotIn("CLOSE_OPERATOR_AUTH_REQUIRED", codes)
        finally:
            if prior is None:
                _os.environ.pop("QR_OPERATOR_CLOSE_OVERRIDE", None)
            else:
                _os.environ["QR_OPERATOR_CLOSE_OVERRIDE"] = prior

    def test_hard_loss_close_under_profitability_acceptance_p0_accepts_cited_repair_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(
                root,
                position_side="LONG",
                m15_dir="DOWN",
                h4_dir="DOWN",
                quote_bid=1.1747,
                quote_ask=1.1748,
            )
            matrix = json.loads(files["market_context_matrix"].read_text())
            matrix["pairs"]["EUR_USD"]["LONG"]["support_count"] = 0
            matrix["pairs"]["EUR_USD"]["LONG"]["supports"] = []
            matrix["pairs"]["EUR_USD"]["LONG"]["strongest_support"] = ""
            files["market_context_matrix"].write_text(json.dumps(matrix))
            files["self_improvement_audit"].write_text(json.dumps(_self_improvement_profitability_p0()))
            files["profitability_acceptance"].write_text(json.dumps(_profitability_acceptance_close_leak_p0()))
            decision = _close_decision(
                trade_ids=["555"],
                invalidation_price=1.1750,
                invalidation_tf="H1",
            )
            decision["evidence_refs"].extend(
                [
                    "chart:EUR_USD:M15",
                    "self_improvement:finding:PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED",
                    "profitability:acceptance",
                ]
            )
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED", msg=summary)
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertNotIn("CLOSE_PROFITABILITY_ACCEPTANCE_HARD_GATE_REQUIRED", codes)
            self.assertNotIn("CLOSE_PROFITABILITY_ACCEPTANCE_P0_REQUIRED", codes)
            self.assertNotIn("CLOSE_OPERATOR_AUTH_REQUIRED", codes)

    def test_hard_loss_close_under_profitability_p0_must_cite_self_improvement_context(self) -> None:
        # Hard Gate A can still close a broken thesis, but while the live
        # account has an active profitability P0, another underwater market
        # close must explicitly cite that P0 and explain why it is repair
        # rather than more MARKET_ORDER_TRADE_CLOSE leakage.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="UP", h4_dir="UP")
            files["self_improvement_audit"].write_text(json.dumps(_self_improvement_profitability_p0()))
            decision = _close_decision(trade_ids=["555"])
            decision["evidence_refs"].append("chart:EUR_USD:H4")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED", msg=summary)
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("CLOSE_PROFITABILITY_P0_CONTEXT_REQUIRED", codes)
            self.assertNotIn("CLOSE_OPERATOR_AUTH_REQUIRED", codes)
            self.assertNotIn("CLOSE_THESIS_STILL_VALID", codes)

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

    def test_receipt_invalidation_price_cannot_harden_entry_buffer_position_thesis_close(self) -> None:
        # Regression for 2026-06-12 NZD_CAD 472312 (-1495 JPY): the sidecar
        # correctly marked "entry thesis lacks invalidation_price / entry-buffer"
        # as soft, but the GPT receipt then copied that buffer into
        # invalidation_price=M5 and bypassed Gate B as a hard loss-cut.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(
                root,
                position_side="SHORT",
                m15_dir="DOWN",
                h4_dir="DOWN",
                quote_bid=1.17600,
                quote_ask=1.17610,
            )
            _write_fresh_position_thesis_close_recommendation(
                root,
                files,
                rationale_lines=[
                    "patterns +2.2",
                    "forward-proj +8.2 UP against SHORT",
                    "chart-tech -29.8 against SHORT",
                ],
                context_notes=[
                    (
                        "adverse technical loss: entry thesis lacks invalidation_price; "
                        "current ask 1.17610 >= entry-buffer 1.17520 "
                        "(entry 1.17500, buffer 2.0p, upl -1250.1 JPY)"
                    ),
                    (
                        "technical invalidation confirmed against SHORT: "
                        "H1 RSI=65.3; M15 MACD+; M30 ST+; M5 TREND_UP"
                    ),
                ],
            )
            decision = _close_decision(
                trade_ids=["555"],
                invalidation_price=1.17500,
                invalidation_tf="M5",
            )
            decision["evidence_refs"].append("position:thesis:555")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED", msg=summary)
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("CLOSE_OPERATOR_AUTH_REQUIRED", codes)
            self.assertNotIn("CLOSE_THESIS_STILL_VALID", codes)
            recs = payload["input_packet"]["protection_sidecars"]["position_close_recommendations"]
            self.assertEqual(recs[0]["source"], "position_thesis")
            self.assertFalse(recs[0]["gate_b_standing_authorized"])

    def test_receipt_invalidation_price_cannot_harden_soft_sidecar_when_entry_thesis_lacks_recorded_invalidation(self) -> None:
        # The live NZD_CAD 472312 close had an entry_thesis row, but that row
        # recorded invalidation_price=null. The sidecar reason that reached the
        # verifier was score/technical text, so text-only detection missed the
        # unrecorded-invalidation condition and the receipt's M5 invalidation
        # became an unattended hard loss-cut.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(
                root,
                position_side="SHORT",
                m15_dir="DOWN",
                h4_dir="DOWN",
                quote_bid=1.17600,
                quote_ask=1.17610,
            )
            _write_entry_thesis(root, invalidation_price=None)
            _write_fresh_position_thesis_close_recommendation(
                root,
                files,
                rationale_lines=[
                    "pattern score +2.2 is outweighed by forward-projection +8.2 UP against the SHORT",
                    "chart-tech -29.8 against SHORT",
                ],
                context_notes=[
                    "technical invalidation confirmed against SHORT: H1 RSI=65.3; H1 ST+; H1 cloud+; M15 MACD+; M30 ST+; M5 TREND_UP",
                ],
            )
            decision = _close_decision(
                trade_ids=["555"],
                invalidation_price=1.17500,
                invalidation_tf="M5",
            )
            decision["evidence_refs"].append("position:thesis:555")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED", msg=summary)
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("CLOSE_OPERATOR_AUTH_REQUIRED", codes)
            self.assertNotIn("CLOSE_THESIS_STILL_VALID", codes)
            ctx = payload["input_packet"]["protection_sidecars"]["entry_thesis_close_context"][0]
            self.assertTrue(ctx["recorded"])
            self.assertFalse(ctx["has_recorded_invalidation_price"])

    def test_m15_structure_cannot_harden_entry_buffer_position_thesis_close(self) -> None:
        # Same incident, second hardening path: the historical packet also had
        # an M15 close-confirmed CHOCH against the position. When the only
        # matching close sidecar says "entry thesis lacks invalidation_price /
        # entry-buffer", M15 structure is still too local to become unattended
        # standing loss-cut authorization. H4 structural breaks stay hard.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(
                root,
                position_side="SHORT",
                m15_dir="UP",
                h4_dir="DOWN",
            )
            _write_fresh_position_thesis_close_recommendation(
                root,
                files,
                context_notes=[
                    (
                        "adverse technical loss: entry thesis lacks invalidation_price; "
                        "current ask 1.17610 >= entry-buffer 1.17520"
                    ),
                    "technical invalidation confirmed against SHORT: M5 TREND_UP; M15 MACD+; H1 RSI=65.3",
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
            self.assertNotIn("CLOSE_THESIS_STILL_VALID", codes)

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
            files = _close_fixtures(root, position_side="LONG", m15_dir="UP", h4_dir="DOWN")
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

    def test_hard_position_thesis_invalidation_requires_token_when_hold_sidecars_support_recovery(self) -> None:
        # Regression for 2026-06-15 USD_CAD 472427: position_thesis carried a
        # hard invalidation-hit review, but thesis_evolution and forecast
        # persistence still supported the open LONG. That conflict should
        # become explicit Gate-B work instead of unattended market close.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="LONG", m15_dir="UP", h4_dir="UP")
            _write_fresh_position_thesis_close_recommendation(
                root,
                files,
                side="LONG",
                rationale_lines=[
                    "patterns +4.0",
                    "forward-proj +6.0",
                    "chart-tech -18.0",
                ],
                context_notes=[
                    (
                        "invalidation hit: current bid 1.39646 <= buffered invalidation "
                        "1.39678 (raw 1.39698, buffer 2.0p)"
                    ),
                    (
                        "technical invalidation confirmed against LONG: "
                        "M5 regime=IMPULSE_DOWN; M15 RSI=42.0; M30 MACD-; H1 cloud-"
                    ),
                ],
            )
            _write_fresh_evolution_and_persistence_hold_support(root, files, side="LONG")
            decision = _close_decision(trade_ids=["555"])
            decision["evidence_refs"].extend(
                ["position:thesis:555", "position:evolution:555", "position:persistence:555"]
            )
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED", msg=summary)
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("CLOSE_SAME_DIRECTION_MARKET_SUPPORT", codes)
            self.assertNotIn("CLOSE_THESIS_STILL_VALID", codes)

    def test_input_packet_marks_hold_conflicted_close_sidecar_as_nonblocking_advisory(self) -> None:
        # Regression for 2026-06-15 EUR_CHF 472445: the router correctly chose
        # the entry branch, but the model saw a position_thesis REVIEW_CLOSE
        # sidecar and kept returning CLOSE. The packet must expose the
        # hold-sidecar downgrade in machine-readable form, even if an operator
        # token/env is present for unrelated discretionary closes.
        prior = _os.environ.get("QR_OPERATOR_CLOSE_OVERRIDE")
        _os.environ["QR_OPERATOR_CLOSE_OVERRIDE"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                files = _close_fixtures(root, position_side="LONG", m15_dir="UP", h4_dir="UP")
                _write_fresh_position_thesis_close_recommendation(
                    root,
                    files,
                    side="LONG",
                    rationale_lines=[
                        "patterns +4.0",
                        "forward-proj +6.0",
                        "chart-tech -18.0",
                    ],
                    context_notes=[
                        "invalidation hit: current bid 1.17300 <= buffered invalidation 1.17350",
                        "technical invalidation confirmed against LONG: H1 MACD-; M15 ST-; M30 MACD-; M5 ST-",
                    ],
                )
                _write_fresh_evolution_and_persistence_hold_support(root, files, side="LONG")
                decision = _close_decision(trade_ids=["555"])
                decision["evidence_refs"].extend(
                    ["position:thesis:555", "position:evolution:555", "position:persistence:555"]
                )
                brain = _brain(root, files, decision)

                summary = brain.run(snapshot_path=files["snapshot"])

                self.assertEqual(summary.status, "REJECTED", msg=summary)
                payload = json.loads((root / "gpt_decision.json").read_text())
                recs = payload["input_packet"]["protection_sidecars"]["position_close_recommendations"]
                thesis_rec = next(item for item in recs if item["source"] == "position_thesis")
                self.assertTrue(thesis_rec["gate_b_standing_authorized"])
                self.assertFalse(thesis_rec["blocks_non_close_actions"])
                self.assertEqual(thesis_rec["routing_effect"], "SOFT_ADVISORY_NON_BLOCKING")
                self.assertIn("Do not choose CLOSE", thesis_rec["entry_decision_guidance"])
                self.assertIn("same-direction", thesis_rec["non_blocking_reason"])
        finally:
            if prior is None:
                _os.environ.pop("QR_OPERATOR_CLOSE_OVERRIDE", None)
            else:
                _os.environ["QR_OPERATOR_CLOSE_OVERRIDE"] = prior

    def test_close_from_nonblocking_soft_advisory_with_live_ready_lanes_points_back_to_entry(self) -> None:
        # The verifier should not only reject the close; it should identify the
        # exact contract breach so the next receipt returns to the active
        # entry branch and current LIVE_READY lanes.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="LONG", m15_dir="UP", h4_dir="UP")
            matrix = json.loads(files["market_context_matrix"].read_text())
            matrix["pairs"]["EUR_USD"]["LONG"]["supports"] = []
            files["market_context_matrix"].write_text(json.dumps(matrix))
            _write_fresh_position_thesis_close_recommendation(
                root,
                files,
                side="LONG",
                rationale_lines=[
                    "patterns +4.0",
                    "forward-proj +6.0",
                    "chart-tech -18.0",
                ],
                context_notes=[
                    "invalidation hit: current bid 1.17300 <= buffered invalidation 1.17350",
                    "technical invalidation confirmed against LONG: H1 MACD-; M15 ST-; M30 MACD-; M5 ST-",
                ],
            )
            _write_fresh_evolution_and_persistence_hold_support(root, files, side="LONG")
            decision = _close_decision(trade_ids=["555"])
            decision["evidence_refs"].extend(
                ["position:thesis:555", "position:evolution:555", "position:persistence:555"]
            )
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED", msg=summary)
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("SOFT_CLOSE_ADVISORY_DOES_NOT_PREEMPT_ENTRY", codes)
            self.assertIn("CLOSE_SAME_DIRECTION_MARKET_SUPPORT", codes)
            self.assertNotIn("CLOSE_OPERATOR_AUTH_REQUIRED", codes)
            messages = "\n".join(issue["message"] for issue in payload["verification_issues"])
            self.assertIn("TRADE/CANCEL/WAIT", messages)
            self.assertIn(LANE_ID, messages)

    def test_receipt_invalidation_price_requires_token_when_hold_sidecars_support_recovery(self) -> None:
        # Same USD_CAD shape through the receipt-level invalidation path: a
        # model can copy the recorded invalidation into the CLOSE receipt, but
        # same-direction hold sidecars still make the loss close explicit Gate-B
        # work instead of unattended execution.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(
                root,
                position_side="LONG",
                m15_dir="UP",
                h4_dir="UP",
                quote_bid=1.17600,
                quote_ask=1.17610,
            )
            _write_fresh_evolution_and_persistence_hold_support(root, files, side="LONG")
            decision = _close_decision(
                trade_ids=["555"],
                invalidation_price=1.17630,
                invalidation_tf="entry_thesis",
            )
            decision["evidence_refs"].extend(["position:evolution:555", "position:persistence:555"])
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED", msg=summary)
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("CLOSE_SAME_DIRECTION_MARKET_SUPPORT", codes)
            self.assertNotIn("CLOSE_THESIS_STILL_VALID", codes)

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

    def test_close_accepted_without_operator_token_when_guardian_structural_review_exit(self) -> None:
        # Position guardian runs more frequently than the full trader cycle; its
        # REVIEW_EXIT carry-forward must be visible to the next GPT CLOSE pass.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="DOWN", h4_dir="DOWN")
            _write_recent_position_management_close_recommendation(
                root,
                files,
                path_name="position_guardian_management.json",
            )
            decision = _close_decision(trade_ids=["555"])
            decision["evidence_refs"].append("position:guardian_management:555")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED", msg=summary)
            self.assertTrue(summary.allowed)
            payload = json.loads((root / "gpt_decision.json").read_text())
            self.assertEqual(payload["verification_issues"], [])
            recs = payload["input_packet"]["protection_sidecars"]["position_close_recommendations"]
            self.assertEqual(recs[0]["source"], "position_guardian_management")
            self.assertTrue(recs[0]["gate_b_standing_authorized"])
            self.assertIn("position:guardian_management:555", payload["input_packet"]["allowed_evidence_refs"])

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

    def test_thesis_expired_without_structural_invalidation_is_not_standing_close_authorized(self) -> None:
        # Expiry ends the original prediction horizon, but the clock alone is
        # not price/structure invalidation and must not authorize an unattended
        # loss-side market close.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="LONG", m15_dir="UP", h4_dir="UP")
            _write_fresh_thesis_evolution_close_recommendation(
                root,
                files,
                side="LONG",
                rationale=(
                    "THESIS_EXPIRED: age 7.0h exceeds declared horizon 6.0h "
                    "with neither target nor invalidation reached"
                ),
            )
            decision = _close_decision(trade_ids=["555"])
            decision["evidence_refs"].append("position:evolution:555")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED", msg=summary)
            self.assertFalse(summary.allowed)
            payload = json.loads((root / "gpt_decision.json").read_text())
            sidecar = payload["input_packet"]["protection_sidecars"]["position_close_recommendations"][0]
            self.assertFalse(sidecar["gate_b_standing_authorized"])
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertTrue(
                {"CLOSE_OPERATOR_AUTH_REQUIRED", "CLOSE_SAME_DIRECTION_MARKET_SUPPORT"} & codes
            )

    def test_thesis_evolution_forecast_flip_is_not_standing_close_authorized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="LONG", m15_dir="UP", h4_dir="UP")
            _write_fresh_thesis_evolution_close_recommendation(
                root,
                files,
                side="LONG",
                rationale=(
                    "FORECAST FLIPPED: entry UP → current DOWN (position LONG)"
                ),
            )
            decision = _close_decision(trade_ids=["555"])
            decision["evidence_refs"].append("position:evolution:555")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED", msg=summary)
            payload = json.loads((root / "gpt_decision.json").read_text())
            sidecar = payload["input_packet"]["protection_sidecars"]["position_close_recommendations"][0]
            self.assertFalse(sidecar["gate_b_standing_authorized"])

    def test_thesis_evolution_wrong_side_confirmation_requires_explicit_gate_b(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="DOWN", h4_dir="DOWN")
            _write_fresh_thesis_evolution_close_recommendation(
                root,
                files,
                side="SHORT",
                rationale=(
                    "invalidation hit: current ask 1.16310 >= buffered invalidation "
                    "1.16290 (raw 1.16270, buffer 2.0p); technical invalidation "
                    "confirmed against LONG: H1 BOS_DOWN; M15 MACD-"
                ),
            )
            decision = _close_decision(trade_ids=["555"])
            decision["evidence_refs"].append("position:evolution:555")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED", msg=summary)
            payload = json.loads((root / "gpt_decision.json").read_text())
            sidecar = payload["input_packet"]["protection_sidecars"]["position_close_recommendations"][0]
            self.assertFalse(sidecar["gate_b_standing_authorized"])
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("CLOSE_OPERATOR_AUTH_REQUIRED", codes)

    def test_thesis_evolution_wrong_price_geometry_requires_explicit_gate_b(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="DOWN", h4_dir="DOWN")
            _write_fresh_thesis_evolution_close_recommendation(
                root,
                files,
                side="SHORT",
                rationale=(
                    "invalidation hit: current bid 1.16280 <= buffered invalidation "
                    "1.16290 (raw 1.16270, buffer 2.0p); technical invalidation "
                    "confirmed against SHORT: H1 BOS_UP; M15 MACD+"
                ),
            )
            decision = _close_decision(trade_ids=["555"])
            decision["evidence_refs"].append("position:evolution:555")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED", msg=summary)
            payload = json.loads((root / "gpt_decision.json").read_text())
            sidecar = payload["input_packet"]["protection_sidecars"]["position_close_recommendations"][0]
            self.assertFalse(sidecar["gate_b_standing_authorized"])
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("CLOSE_OPERATOR_AUTH_REQUIRED", codes)

    def test_thesis_evolution_invalidation_and_technical_confirmation_remains_hard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="SHORT", m15_dir="DOWN", h4_dir="DOWN")
            _write_fresh_thesis_evolution_close_recommendation(
                root,
                files,
                rationale=(
                    "invalidation hit: current ask 1.16310 >= buffered invalidation "
                    "1.16290 (raw 1.16270, buffer 2.0p); technical invalidation "
                    "confirmed against SHORT: H1 BOS_UP; H4 BOS_UP"
                ),
            )
            decision = _close_decision(trade_ids=["555"])
            decision["evidence_refs"].append("position:evolution:555")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "ACCEPTED", msg=summary)
            payload = json.loads((root / "gpt_decision.json").read_text())
            sidecar = payload["input_packet"]["protection_sidecars"]["position_close_recommendations"][0]
            self.assertTrue(sidecar["gate_b_standing_authorized"])

    def test_thesis_evolution_forecast_decay_with_position_thesis_hold_is_soft_advisory(self) -> None:
        # Regression for 2026-06-16 NZD_CAD 472380 shape: thesis_evolution can
        # recommend CLOSE from forecast/confidence/regime decay while a fresh
        # position_thesis assessment still says HOLD. That is not unattended
        # hard invalidation evidence.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="LONG", m15_dir="UP", h4_dir="UP")
            snapshot = json.loads(files["snapshot"].read_text())
            generated_at = (
                datetime.fromisoformat(snapshot["fetched_at_utc"]) + timedelta(seconds=1)
            ).isoformat()
            _write_fresh_thesis_evolution_close_recommendation(
                root,
                files,
                side="LONG",
                rationale=(
                    "current_forecast DOWN vs entry_forecast RANGE; "
                    "current_confidence 0.42 below entry_confidence 0.68; "
                    "current_regime RANGE no longer supports recovery to target"
                ),
            )
            (root / "position_thesis_report.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": generated_at,
                        "assessments": [
                            {
                                "trade_id": "555",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "verdict": "HOLD",
                                "aggregate_score": 58.25,
                                "rationale_lines": ["position thesis still supports LONG carry"],
                                "context_notes": [],
                            }
                        ],
                    }
                )
            )
            decision = _close_decision(trade_ids=["555"])
            decision["evidence_refs"].append("position:evolution:555")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED", msg=summary)
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("CLOSE_SAME_DIRECTION_MARKET_SUPPORT", codes)
            self.assertIn("SOFT_CLOSE_ADVISORY_DOES_NOT_PREEMPT_ENTRY", codes)
            self.assertNotIn("CLOSE_OPERATOR_AUTH_REQUIRED", codes)
            self.assertNotIn("CLOSE_THESIS_STILL_VALID", codes)
            recs = payload["input_packet"]["protection_sidecars"]["position_close_recommendations"]
            self.assertEqual(recs[0]["source"], "thesis_evolution")
            self.assertFalse(recs[0]["blocks_non_close_actions"])
            self.assertEqual(recs[0]["routing_effect"], "SOFT_ADVISORY_NON_BLOCKING")
            self.assertIn("position_thesis:HOLD", recs[0]["non_blocking_reason"])

    def test_thesis_expired_with_hold_sidecars_rejects_loss_close(self) -> None:
        # Regression for 2026-06-12 USD_CAD 472336 shape: a thesis can be past
        # its forecast horizon while position_thesis / forecast_persistence
        # still support HOLD/EXTEND. That is reprice/TP work, not a hard
        # unattended loss-side CLOSE.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _close_fixtures(root, position_side="LONG", m15_dir="UP", h4_dir="UP")
            snapshot = json.loads(files["snapshot"].read_text())
            generated_at = (
                datetime.fromisoformat(snapshot["fetched_at_utc"]) + timedelta(seconds=1)
            ).isoformat()
            _write_fresh_thesis_evolution_close_recommendation(
                root,
                files,
                side="LONG",
                rationale=(
                    "THESIS_EXPIRED: age 3.1h exceeds declared horizon 3.0h "
                    "with neither target nor invalidation reached"
                ),
            )
            (root / "position_thesis_report.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": generated_at,
                        "assessments": [
                            {
                                "trade_id": "555",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "verdict": "EXTEND",
                                "aggregate_score": 62.43,
                                "rationale_lines": ["position thesis still supports LONG carry"],
                                "context_notes": [],
                            }
                        ],
                    }
                )
            )
            (root / "forecast_persistence_report.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": generated_at,
                        "verdicts": [
                            {
                                "trade_id": "555",
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "verdict": "EXTEND",
                                "reason": "recent forecasts still support the open LONG",
                            }
                        ],
                    }
                )
            )
            decision = _close_decision(trade_ids=["555"])
            decision["evidence_refs"].append("position:evolution:555")
            brain = _brain(root, files, decision)

            summary = brain.run(snapshot_path=files["snapshot"])

            self.assertEqual(summary.status, "REJECTED", msg=summary)
            payload = json.loads((root / "gpt_decision.json").read_text())
            codes = {issue["code"] for issue in payload["verification_issues"]}
            self.assertIn("CLOSE_SAME_DIRECTION_MARKET_SUPPORT", codes)
            self.assertNotIn("CLOSE_THESIS_STILL_VALID", codes)

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
