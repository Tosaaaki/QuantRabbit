from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from quant_rabbit.broker.execution import (
    _range_vehicle_candidate_receipt_from_intent,
)
from quant_rabbit.models import (
    MarketContext,
    OrderIntent,
    OrderType,
    Side,
    TradeMethod,
)
from quant_rabbit.strategy import range_vehicle_candidate_ledger as ledger
from quant_rabbit.strategy import intent_generator as intent_generator_module
from quant_rabbit.strategy.forecast_technical_context import (
    build_forecast_technical_context,
    build_forecast_technical_context_evidence,
)
from quant_rabbit.strategy.intent_generator import GeneratedIntent, IntentGenerator


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


class RangeVehicleCandidateLedgerTest(unittest.TestCase):
    @staticmethod
    def _make_live_ready(result: dict, kwargs: dict) -> None:
        result["status"] = "LIVE_READY"
        result["risk_allowed"] = True
        result["live_blocker_codes"] = []
        kwargs["order_intents_serialized"] = (
            json.dumps(
                {
                    "generated_at_utc": kwargs["generated_at_utc"],
                    "results": [result],
                },
                sort_keys=True,
            )
            + "\n"
        ).encode()

    def _fixture(self, root: Path, *, scout: bool = False):
        data = root / "data"
        data.mkdir()
        cycle_id = "cycle-1"
        technical_context = build_forecast_technical_context(
            {
                "confluence": {
                    "dominant_regime": "RANGE",
                    "price_percentile_24h": 0.5,
                    "price_percentile_7d": 0.5,
                },
                "views": [
                    {
                        "granularity": "M5",
                        "regime_reading": {
                            "state": "RANGE",
                            "atr_percentile": 50.0,
                        },
                        "indicators": {"atr_pips": 4.0},
                        "family_scores": {
                            "trend_score": 0.0,
                            "mean_rev_score": 1.0,
                            "breakout_score": 0.0,
                            "disagreement": 0.0,
                        },
                        "structure": {"structure_events": []},
                    }
                ],
            },
            pair="GBP_USD",
            current_price=1.105,
            spread_pips=0.5,
        )
        technical_evidence = build_forecast_technical_context_evidence(
            technical_context,
            pair="GBP_USD",
            current_price=1.105,
        )
        forecast = {
            "timestamp_utc": "2026-07-14T00:00:00Z",
            "cycle_id": cycle_id,
            "pair": "GBP_USD",
            "direction": "RANGE",
            "confidence": 0.8,
            "current_price": 1.105,
            "range_low_price": 1.1,
            "range_high_price": 1.11,
            "horizon_min": 120,
            "technical_context_v1": technical_context,
        }
        (data / "forecast_history.jsonl").write_text(
            json.dumps(forecast) + "\n", encoding="utf-8"
        )
        receipt_body = {
            "schema_version": "QR_FORECAST_EMISSION_RECEIPT_V1",
            "sequence": 1,
            "operation": "APPEND",
            "recorded_at_utc": "2026-07-14T00:00:01Z",
            "forecast_timestamp_utc": forecast["timestamp_utc"],
            "cycle_id": cycle_id,
            "pair": "GBP_USD",
            "forecast_row_sha256": _sha(forecast),
            "previous_receipt_sha256": None,
        }
        receipt = {**receipt_body, "receipt_sha256": _sha(receipt_body)}
        (data / "forecast_emission_receipts.jsonl").write_text(
            _canonical(receipt) + "\n", encoding="utf-8"
        )
        artifacts = {}
        for name in (
            "broker_snapshot.json",
            "pair_charts.json",
            "market_context_matrix.json",
            "daily_campaign_plan.json",
            "strategy_profile.json",
        ):
            path = data / name
            path.write_text(json.dumps({"name": name}) + "\n", encoding="utf-8")
            artifacts[name] = path

        metadata = {
            "geometry_model": "RANGE_RAIL_LIMIT",
            "forecast_direction": "RANGE",
            "forecast_cycle_id": cycle_id,
            "forecast_confidence": 0.8,
            "forecast_horizon_min": 120,
            "forecast_current_price": 1.105,
            "forecast_range_low_price": 1.1,
            "forecast_range_high_price": 1.11,
            "forecast_technical_context": technical_evidence,
            "range_support": 1.1,
            "range_resistance": 1.11,
            "range_entry_side": "support",
            "range_indicator_source": "forecast_range_box",
            "range_tp_is_inside_box": True,
            "range_sl_outside_box": True,
            "attach_take_profit_on_fill": True,
            "target_reward_risk": 2.0,
            "opportunity_mode_reward_risk": 1.96,
            "disaster_sl": 1.09,
            "m1_atr_pips": 2.0,
            "m5_atr_pips": 4.0,
            "h4_atr_pips": 24.0,
        }
        if scout:
            metadata.update(
                {
                    "predictive_scout": True,
                    "predictive_scout_expires_at_utc": "2026-07-14T01:30:00Z",
                    "broker_stop_loss_mode": "INTENT_SL",
                }
            )
        result = {
            "lane_id": "range_trader:GBP_USD:LONG:RANGE_ROTATION",
            "status": "DRY_RUN_BLOCKED",
            "risk_allowed": False,
            "live_blocker_codes": ["NEGATIVE_EXPECTANCY"],
            "intent": {
                "pair": "GBP_USD",
                "side": "LONG",
                "order_type": "LIMIT",
                "units": 1000,
                "entry": 1.1001,
                "tp": 1.1099,
                "sl": 1.0951,
                "market_context": {"method": "RANGE_ROTATION"},
                "metadata": metadata,
            },
        }
        with mock.patch.dict(
            os.environ,
            {
                "QR_DISABLE_AUTO_TP": "0",
                "QR_NEW_ENTRY_INITIAL_SL": "0",
                "QR_TRADER_DISABLE_SL_REPAIR": "1",
            },
        ):
            ledger.bind_range_vehicle_candidate_ids(
                [result], generated_at_utc="2026-07-14T00:00:02+00:00"
            )
        order_intents = data / "order_intents.json"
        order_bytes = (
            json.dumps(
                {
                    "generated_at_utc": "2026-07-14T00:00:02+00:00",
                    "results": [result],
                },
                sort_keys=True,
            )
            + "\n"
        ).encode()
        order_intents.write_bytes(order_bytes)
        kwargs = {
            "generated_at_utc": "2026-07-14T00:00:02+00:00",
            "order_intents_path": order_intents,
            "order_intents_serialized": order_bytes,
            "snapshot_path": artifacts["broker_snapshot.json"],
            "pair_charts_path": artifacts["pair_charts.json"],
            "market_context_matrix_path": artifacts["market_context_matrix.json"],
            "campaign_plan_path": artifacts["daily_campaign_plan.json"],
            "strategy_profile_path": artifacts["strategy_profile.json"],
        }
        return result, kwargs

    def test_records_hash_chained_normal_candidate_and_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result, kwargs = self._fixture(root)
            with mock.patch.dict(
                os.environ,
                {
                    "QR_DISABLE_AUTO_TP": "0",
                    "QR_NEW_ENTRY_INITIAL_SL": "0",
                    "QR_TRADER_DISABLE_SL_REPAIR": "1",
                },
            ):
                self.assertEqual(
                    ledger.record_range_vehicle_candidates([result], **kwargs), 1
                )
                self.assertEqual(
                    ledger.record_range_vehicle_candidates([result], **kwargs), 0
                )
            path = kwargs["order_intents_path"].with_name(ledger.LEDGER_FILENAME)
            validation = ledger.validate_range_vehicle_candidate_ledger(path)
            self.assertEqual(validation["status"], "VALID")
            self.assertEqual(validation["rows"], 1)
            receipt = json.loads(path.read_text(encoding="utf-8"))
            payload = receipt["payload"]
            self.assertTrue(payload["candidate_geometry_frozen"])
            self.assertTrue(payload["forecast_binding_complete"])
            self.assertTrue(payload["source_artifact_snapshot_complete"])
            self.assertFalse(payload["exact_generation_input_bytes_proved"])
            self.assertFalse(payload["exact_candidate_lineage_complete"])
            self.assertEqual(payload["vehicle"]["units"], 1000)
            self.assertEqual(payload["vehicle_shape"]["units"], 1000)
            self.assertEqual(
                payload["prebinding_intent_sha256"],
                payload["vehicle_shape"]["prebinding_intent_sha256"],
            )
            self.assertEqual(len(payload["prebinding_intent_sha256"]), 64)
            self.assertTrue(ledger._candidate_payload_integrity_valid(payload))
            for field in (
                "exact_generation_input_bytes_proved",
                "exact_candidate_lineage_complete",
            ):
                altered = copy.deepcopy(payload)
                altered[field] = True
                self.assertFalse(ledger._candidate_payload_integrity_valid(altered))
            altered = copy.deepcopy(payload)
            altered["lineage_caveat"] = "overstated exact lineage"
            self.assertFalse(ledger._candidate_payload_integrity_valid(altered))
            self.assertEqual(
                payload["forecast_lineage"]["status"],
                "BOUND_TO_EMISSION_RECEIPT",
            )
            projected = payload["gateway_contract_projection"]
            self.assertEqual(projected["time_in_force"], "GTC")
            self.assertEqual(projected["position_fill"], "DEFAULT")
            self.assertEqual(projected["take_profit_on_fill"], 1.1099)
            self.assertEqual(projected["stop_loss_on_fill"], 1.09)
            self.assertEqual(projected["stop_loss_basis"], "DISASTER_SL")
            self.assertEqual(
                projected["filled_position_lifecycle"],
                "ADAPTIVE_TRADER_GUARDIAN_AND_BROKER_BARRIERS",
            )

    def test_records_zero_unit_non_live_candidate_without_granting_permission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, kwargs = self._fixture(Path(tmp))
            result["intent"]["units"] = 0
            for key in ledger.IDENTITY_METADATA_FIELDS:
                result["intent"]["metadata"].pop(key, None)
            with mock.patch.dict(
                os.environ,
                {
                    "QR_DISABLE_AUTO_TP": "0",
                    "QR_NEW_ENTRY_INITIAL_SL": "0",
                    "QR_TRADER_DISABLE_SL_REPAIR": "1",
                },
            ):
                ledger.bind_range_vehicle_candidate_ids(
                    [result], generated_at_utc=kwargs["generated_at_utc"]
                )
                kwargs["order_intents_serialized"] = (
                    json.dumps(
                        {
                            "generated_at_utc": kwargs["generated_at_utc"],
                            "results": [result],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                ).encode()
                self.assertEqual(
                    ledger.record_range_vehicle_candidates([result], **kwargs), 1
                )

            path = kwargs["order_intents_path"].with_name(ledger.LEDGER_FILENAME)
            receipt = json.loads(path.read_text(encoding="utf-8"))
            payload = receipt["payload"]
            self.assertEqual(payload["vehicle"]["units"], 0)
            self.assertEqual(payload["candidate"]["status"], "DRY_RUN_BLOCKED")
            self.assertFalse(payload["live_permission_allowed"])
            self.assertTrue(ledger._candidate_payload_integrity_valid(payload))

    def test_zero_unit_live_ready_candidate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, kwargs = self._fixture(Path(tmp))
            result["intent"]["units"] = 0
            self._make_live_ready(result, kwargs)
            for key in ledger.IDENTITY_METADATA_FIELDS:
                result["intent"]["metadata"].pop(key, None)
            with mock.patch.dict(
                os.environ,
                {
                    "QR_DISABLE_AUTO_TP": "0",
                    "QR_NEW_ENTRY_INITIAL_SL": "0",
                    "QR_TRADER_DISABLE_SL_REPAIR": "1",
                },
            ):
                ledger.bind_range_vehicle_candidate_ids(
                    [result], generated_at_utc=kwargs["generated_at_utc"]
                )
                kwargs["order_intents_serialized"] = (
                    json.dumps(
                        {
                            "generated_at_utc": kwargs["generated_at_utc"],
                            "results": [result],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                ).encode()
                with self.assertRaisesRegex(
                    ValueError, "RANGE_VEHICLE_CANDIDATE_UNITS_INVALID"
                ):
                    ledger.record_range_vehicle_candidates([result], **kwargs)

    def test_predictive_scout_projects_gtd_and_exact_intent_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, kwargs = self._fixture(Path(tmp), scout=True)
            with mock.patch.dict(
                os.environ,
                {
                    "QR_DISABLE_AUTO_TP": "0",
                    "QR_NEW_ENTRY_INITIAL_SL": "0",
                    "QR_TRADER_DISABLE_SL_REPAIR": "1",
                },
            ):
                ledger.record_range_vehicle_candidates([result], **kwargs)
            path = kwargs["order_intents_path"].with_name(ledger.LEDGER_FILENAME)
            payload = json.loads(path.read_text(encoding="utf-8"))["payload"]
            projected = payload["gateway_contract_projection"]
            self.assertEqual(projected["time_in_force"], "GTD")
            self.assertEqual(projected["position_fill"], "OPEN_ONLY")
            self.assertEqual(projected["gtd_time_utc"], "2026-07-14T01:30:00Z")
            self.assertEqual(projected["stop_loss_on_fill"], 1.0951)
            self.assertEqual(projected["stop_loss_basis"], "INTENT_SL")
            self.assertEqual(
                projected["filled_position_lifecycle"], "STATIC_BROKER_BARRIERS"
            )

    def test_projection_matches_gateway_default_take_profit_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, _kwargs = self._fixture(Path(tmp))
            metadata = result["intent"]["metadata"]
            metadata.pop("attach_take_profit_on_fill")
            for key in (
                "range_vehicle_candidate_id",
                "range_vehicle_shape_sha256",
                "range_vehicle_gateway_projection_sha256",
                "range_vehicle_candidate_generated_at_utc",
                "range_vehicle_candidate_live_permission",
            ):
                metadata.pop(key, None)

            with mock.patch.dict(
                os.environ,
                {
                    "QR_DISABLE_AUTO_TP": "0",
                    "QR_NEW_ENTRY_INITIAL_SL": "0",
                    "QR_TRADER_DISABLE_SL_REPAIR": "1",
                },
            ):
                ledger.bind_range_vehicle_candidate_ids(
                    [result], generated_at_utc="2026-07-14T00:00:03+00:00"
                )
                projection = ledger._project_gateway_contract(
                    result["intent"], metadata
                )

            self.assertEqual(projection["take_profit_on_fill"], 1.1099)
            self.assertEqual(projection["take_profit_basis"], "INTENT_TP")

    def test_runtime_contract_change_changes_bound_shape_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, _kwargs = self._fixture(Path(tmp))
            disaster_identity = result["intent"]["metadata"][
                "range_vehicle_candidate_id"
            ]
            intent_stop_result = copy.deepcopy(result)
            for key in (
                "range_vehicle_candidate_id",
                "range_vehicle_shape_sha256",
                "range_vehicle_gateway_projection_sha256",
                "range_vehicle_candidate_generated_at_utc",
                "range_vehicle_candidate_live_permission",
            ):
                intent_stop_result["intent"]["metadata"].pop(key, None)
            with mock.patch.dict(
                os.environ,
                {
                    "QR_DISABLE_AUTO_TP": "0",
                    "QR_NEW_ENTRY_INITIAL_SL": "1",
                    "QR_TRADER_DISABLE_SL_REPAIR": "1",
                },
            ):
                ledger.bind_range_vehicle_candidate_ids(
                    [intent_stop_result],
                    generated_at_utc="2026-07-14T00:00:03+00:00",
                )
            intent_stop_identity = intent_stop_result["intent"]["metadata"][
                "range_vehicle_candidate_id"
            ]
            self.assertNotEqual(disaster_identity, intent_stop_identity)
            self.assertEqual(
                intent_stop_result["intent"]["metadata"][
                    "range_vehicle_candidate_live_permission"
                ],
                False,
            )

    def test_tampered_existing_receipt_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, kwargs = self._fixture(Path(tmp))
            with mock.patch.dict(
                os.environ,
                {
                    "QR_DISABLE_AUTO_TP": "0",
                    "QR_NEW_ENTRY_INITIAL_SL": "0",
                    "QR_TRADER_DISABLE_SL_REPAIR": "1",
                },
            ):
                ledger.record_range_vehicle_candidates([result], **kwargs)
            path = kwargs["order_intents_path"].with_name(ledger.LEDGER_FILENAME)
            receipt = json.loads(path.read_text(encoding="utf-8"))
            receipt["payload"]["vehicle"]["entry"] = 1.1002
            path.write_text(_canonical(receipt) + "\n", encoding="utf-8")
            with mock.patch.dict(
                os.environ,
                {
                    "QR_DISABLE_AUTO_TP": "0",
                    "QR_NEW_ENTRY_INITIAL_SL": "0",
                    "QR_TRADER_DISABLE_SL_REPAIR": "1",
                },
            ):
                with self.assertRaisesRegex(
                    ValueError, "RANGE_VEHICLE_CANDIDATE_LEDGER_INVALID"
                ):
                    ledger.record_range_vehicle_candidates([result], **kwargs)

    def test_same_candidate_id_with_different_payload_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, kwargs = self._fixture(Path(tmp))
            with mock.patch.dict(
                os.environ,
                {
                    "QR_DISABLE_AUTO_TP": "0",
                    "QR_NEW_ENTRY_INITIAL_SL": "0",
                    "QR_TRADER_DISABLE_SL_REPAIR": "1",
                },
            ):
                ledger.record_range_vehicle_candidates([result], **kwargs)
                changed = copy.deepcopy(result)
                changed["status"] = "DRY_RUN_PASSED"
                changed["live_blocker_codes"] = []
                changed_kwargs = dict(kwargs)
                changed_kwargs["order_intents_serialized"] = (
                    json.dumps(
                        {
                            "generated_at_utc": kwargs["generated_at_utc"],
                            "results": [changed],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                ).encode()
                with self.assertRaisesRegex(
                    ValueError,
                    "RANGE_VEHICLE_CANDIDATE_IDENTITY_COLLISION",
                ):
                    ledger.record_range_vehicle_candidates(
                        [changed], **changed_kwargs
                    )

    def test_non_live_forecast_metadata_mismatch_is_unbound_shadow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, kwargs = self._fixture(Path(tmp))
            result["intent"]["metadata"]["forecast_confidence"] = 0.7
            for key in ledger.IDENTITY_METADATA_FIELDS:
                result["intent"]["metadata"].pop(key, None)
            with mock.patch.dict(
                os.environ,
                {
                    "QR_DISABLE_AUTO_TP": "0",
                    "QR_NEW_ENTRY_INITIAL_SL": "0",
                    "QR_TRADER_DISABLE_SL_REPAIR": "1",
                },
            ):
                ledger.bind_range_vehicle_candidate_ids(
                    [result], generated_at_utc=kwargs["generated_at_utc"]
                )
                mismatch_kwargs = dict(kwargs)
                mismatch_kwargs["order_intents_serialized"] = (
                    json.dumps(
                        {
                            "generated_at_utc": kwargs["generated_at_utc"],
                            "results": [result],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                ).encode()
                self.assertEqual(
                    ledger.record_range_vehicle_candidates(
                        [result], **mismatch_kwargs
                    ),
                    1,
                )
            path = kwargs["order_intents_path"].with_name(ledger.LEDGER_FILENAME)
            payload = json.loads(path.read_text(encoding="utf-8"))["payload"]
            self.assertFalse(payload["forecast_binding_complete"])
            self.assertEqual(
                payload["candidate_contract_status"], "UNBOUND_NON_LIVE_SHADOW"
            )
            self.assertEqual(
                payload["forecast_lineage"]["status"],
                "FORECAST_METADATA_MISMATCH",
            )
            altered = copy.deepcopy(payload)
            altered["lineage_caveat"] = ledger.BOUND_LINEAGE_CAVEAT
            self.assertFalse(ledger._candidate_payload_integrity_valid(altered))

    def test_live_ready_forecast_metadata_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, kwargs = self._fixture(Path(tmp))
            result["intent"]["metadata"]["forecast_confidence"] = 0.7
            for key in ledger.IDENTITY_METADATA_FIELDS:
                result["intent"]["metadata"].pop(key, None)
            with mock.patch.dict(
                os.environ,
                {
                    "QR_DISABLE_AUTO_TP": "0",
                    "QR_NEW_ENTRY_INITIAL_SL": "0",
                    "QR_TRADER_DISABLE_SL_REPAIR": "1",
                },
            ):
                ledger.bind_range_vehicle_candidate_ids(
                    [result], generated_at_utc=kwargs["generated_at_utc"]
                )
                self._make_live_ready(result, kwargs)
                with self.assertRaisesRegex(
                    ValueError,
                    "RANGE_VEHICLE_FORECAST_BINDING_INCOMPLETE:FORECAST_METADATA_MISMATCH",
                ):
                    ledger.record_range_vehicle_candidates([result], **kwargs)

            path = kwargs["order_intents_path"].with_name(ledger.LEDGER_FILENAME)
            self.assertFalse(path.exists())

    def test_missing_forecast_receipt_fails_closed_without_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, kwargs = self._fixture(Path(tmp))
            self._make_live_ready(result, kwargs)
            kwargs["order_intents_path"].with_name(
                "forecast_emission_receipts.jsonl"
            ).unlink()

            with self.assertRaisesRegex(
                ValueError,
                "RANGE_VEHICLE_FORECAST_BINDING_INCOMPLETE:MISSING_FORECAST_LINEAGE_ARTIFACT",
            ):
                ledger.record_range_vehicle_candidates([result], **kwargs)

            path = kwargs["order_intents_path"].with_name(ledger.LEDGER_FILENAME)
            self.assertFalse(path.exists())

    def test_invalid_receipt_operation_is_non_live_unbound_shadow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, kwargs = self._fixture(Path(tmp))
            receipt_path = kwargs["order_intents_path"].with_name(
                "forecast_emission_receipts.jsonl"
            )
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["operation"] = "REPLACE"
            body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
            receipt["receipt_sha256"] = _sha(body)
            receipt_path.write_text(_canonical(receipt) + "\n", encoding="utf-8")

            with mock.patch.dict(
                os.environ,
                {
                    "QR_DISABLE_AUTO_TP": "0",
                    "QR_NEW_ENTRY_INITIAL_SL": "0",
                    "QR_TRADER_DISABLE_SL_REPAIR": "1",
                },
            ):
                self.assertEqual(
                    ledger.record_range_vehicle_candidates([result], **kwargs), 1
                )

            path = kwargs["order_intents_path"].with_name(ledger.LEDGER_FILENAME)
            payload = json.loads(path.read_text(encoding="utf-8"))["payload"]
            self.assertEqual(
                payload["forecast_lineage"]["status"],
                "FORECAST_RECEIPT_CHAIN_INVALID",
            )
            self.assertEqual(
                payload["candidate_contract_status"], "UNBOUND_NON_LIVE_SHADOW"
            )
            self.assertFalse(payload["forecast_binding_complete"])

    def test_invalid_receipt_operation_fails_closed_for_live_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, kwargs = self._fixture(Path(tmp))
            self._make_live_ready(result, kwargs)
            receipt_path = kwargs["order_intents_path"].with_name(
                "forecast_emission_receipts.jsonl"
            )
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["operation"] = "REPLACE"
            body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
            receipt["receipt_sha256"] = _sha(body)
            receipt_path.write_text(_canonical(receipt) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "RANGE_VEHICLE_FORECAST_BINDING_INCOMPLETE:FORECAST_RECEIPT_CHAIN_INVALID",
            ):
                ledger.record_range_vehicle_candidates([result], **kwargs)

            path = kwargs["order_intents_path"].with_name(ledger.LEDGER_FILENAME)
            self.assertFalse(path.exists())

    def test_duplicate_forecast_row_fails_closed_without_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, kwargs = self._fixture(Path(tmp))
            self._make_live_ready(result, kwargs)
            history_path = kwargs["order_intents_path"].with_name(
                "forecast_history.jsonl"
            )
            original = history_path.read_text(encoding="utf-8")
            history_path.write_text(original + original, encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "RANGE_VEHICLE_FORECAST_BINDING_INCOMPLETE:FORECAST_ROW_NOT_UNIQUE",
            ):
                ledger.record_range_vehicle_candidates([result], **kwargs)

            path = kwargs["order_intents_path"].with_name(ledger.LEDGER_FILENAME)
            self.assertFalse(path.exists())

    def test_older_matching_receipt_cannot_override_latest_replace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, kwargs = self._fixture(Path(tmp))
            self._make_live_ready(result, kwargs)
            receipt_path = kwargs["order_intents_path"].with_name(
                "forecast_emission_receipts.jsonl"
            )
            first = json.loads(receipt_path.read_text(encoding="utf-8"))
            old_row = json.loads(
                kwargs["order_intents_path"]
                .with_name("forecast_history.jsonl")
                .read_text(encoding="utf-8")
            )
            replacement = {**old_row, "confidence": 0.7}
            second_body = {
                "schema_version": "QR_FORECAST_EMISSION_RECEIPT_V1",
                "sequence": 2,
                "operation": "REPLACE",
                "recorded_at_utc": "2026-07-14T00:00:01.500000Z",
                "forecast_timestamp_utc": replacement["timestamp_utc"],
                "cycle_id": replacement["cycle_id"],
                "pair": replacement["pair"],
                "forecast_row_sha256": _sha(replacement),
                "previous_receipt_sha256": first["receipt_sha256"],
            }
            second = {**second_body, "receipt_sha256": _sha(second_body)}
            receipt_path.write_text(
                _canonical(first) + "\n" + _canonical(second) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "RANGE_VEHICLE_FORECAST_BINDING_INCOMPLETE:FORECAST_LATEST_EMISSION_RECEIPT_ROW_MISMATCH",
            ):
                ledger.record_range_vehicle_candidates([result], **kwargs)

            path = kwargs["order_intents_path"].with_name(ledger.LEDGER_FILENAME)
            self.assertFalse(path.exists())

    def test_latest_replace_receipt_binds_current_replacement_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, kwargs = self._fixture(Path(tmp))
            history_path = kwargs["order_intents_path"].with_name(
                "forecast_history.jsonl"
            )
            current = json.loads(history_path.read_text(encoding="utf-8"))
            replacement = {**current, "confidence": 0.7}
            history_path.write_text(
                json.dumps(replacement) + "\n", encoding="utf-8"
            )
            receipt_path = kwargs["order_intents_path"].with_name(
                "forecast_emission_receipts.jsonl"
            )
            first = json.loads(receipt_path.read_text(encoding="utf-8"))
            second_body = {
                "schema_version": "QR_FORECAST_EMISSION_RECEIPT_V1",
                "sequence": 2,
                "operation": "REPLACE",
                "recorded_at_utc": "2026-07-14T00:00:01.500000Z",
                "forecast_timestamp_utc": replacement["timestamp_utc"],
                "cycle_id": replacement["cycle_id"],
                "pair": replacement["pair"],
                "forecast_row_sha256": _sha(replacement),
                "previous_receipt_sha256": first["receipt_sha256"],
            }
            second = {**second_body, "receipt_sha256": _sha(second_body)}
            receipt_path.write_text(
                _canonical(first) + "\n" + _canonical(second) + "\n",
                encoding="utf-8",
            )
            result["intent"]["metadata"]["forecast_confidence"] = 0.7
            for key in ledger.IDENTITY_METADATA_FIELDS:
                result["intent"]["metadata"].pop(key, None)
            with mock.patch.dict(
                os.environ,
                {
                    "QR_DISABLE_AUTO_TP": "0",
                    "QR_NEW_ENTRY_INITIAL_SL": "0",
                    "QR_TRADER_DISABLE_SL_REPAIR": "1",
                },
            ):
                ledger.bind_range_vehicle_candidate_ids(
                    [result], generated_at_utc=kwargs["generated_at_utc"]
                )
                kwargs["order_intents_serialized"] = (
                    json.dumps(
                        {
                            "generated_at_utc": kwargs["generated_at_utc"],
                            "results": [result],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                ).encode()
                self.assertEqual(
                    ledger.record_range_vehicle_candidates([result], **kwargs), 1
                )

            path = kwargs["order_intents_path"].with_name(ledger.LEDGER_FILENAME)
            payload = json.loads(path.read_text(encoding="utf-8"))["payload"]
            self.assertEqual(
                payload["forecast_lineage"]["forecast_emission_receipt_sequence"],
                2,
            )

    def test_multiple_candidates_build_forecast_index_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, kwargs = self._fixture(Path(tmp))
            second = copy.deepcopy(result)
            second["lane_id"] = "range_trader:GBP_USD:LONG:RANGE_ROTATION:second"
            for key in ledger.IDENTITY_METADATA_FIELDS:
                second["intent"]["metadata"].pop(key, None)
            with mock.patch.dict(
                os.environ,
                {
                    "QR_DISABLE_AUTO_TP": "0",
                    "QR_NEW_ENTRY_INITIAL_SL": "0",
                    "QR_TRADER_DISABLE_SL_REPAIR": "1",
                },
            ):
                ledger.bind_range_vehicle_candidate_ids(
                    [second], generated_at_utc=kwargs["generated_at_utc"]
                )
                with mock.patch.object(
                    ledger,
                    "_build_forecast_lineage_index",
                    wraps=ledger._build_forecast_lineage_index,
                ) as build_index:
                    multi_kwargs = dict(kwargs)
                    multi_kwargs["order_intents_serialized"] = (
                        json.dumps(
                            {
                                "generated_at_utc": kwargs["generated_at_utc"],
                                "results": [result, second],
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    ).encode()
                    appended = ledger.record_range_vehicle_candidates(
                        [result, second], **multi_kwargs
                    )

            self.assertEqual(appended, 2)
            build_index.assert_called_once()

    def test_invalid_claimed_range_geometry_is_not_silently_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, kwargs = self._fixture(Path(tmp))
            result["intent"]["tp"] = 1.12
            kwargs["order_intents_serialized"] = (
                json.dumps(
                    {
                        "generated_at_utc": kwargs["generated_at_utc"],
                        "results": [result],
                    },
                    sort_keys=True,
                )
                + "\n"
            ).encode()
            with self.assertRaisesRegex(
                ValueError, "RANGE_VEHICLE_CANDIDATE_GEOMETRY_INVALID"
            ):
                ledger.record_range_vehicle_candidates([result], **kwargs)

    def test_new_payload_is_validated_before_first_append(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, kwargs = self._fixture(Path(tmp))
            result["intent"]["metadata"]["disaster_sl"] = "bad"
            for key in ledger.IDENTITY_METADATA_FIELDS:
                result["intent"]["metadata"].pop(key, None)
            with mock.patch.dict(
                os.environ,
                {
                    "QR_DISABLE_AUTO_TP": "0",
                    "QR_NEW_ENTRY_INITIAL_SL": "0",
                    "QR_TRADER_DISABLE_SL_REPAIR": "1",
                },
            ):
                ledger.bind_range_vehicle_candidate_ids(
                    [result], generated_at_utc=kwargs["generated_at_utc"]
                )
                kwargs["order_intents_serialized"] = (
                    json.dumps(
                        {
                            "generated_at_utc": kwargs["generated_at_utc"],
                            "results": [result],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                ).encode()
                with self.assertRaisesRegex(
                    ValueError, "RANGE_VEHICLE_CANDIDATE_PAYLOAD_INVALID"
                ):
                    ledger.record_range_vehicle_candidates([result], **kwargs)

            path = kwargs["order_intents_path"].with_name(ledger.LEDGER_FILENAME)
            self.assertFalse(path.exists())

    def test_gateway_receipt_binds_final_request_without_granting_permission(self) -> None:
        intent = OrderIntent(
            pair="GBP_USD",
            side=Side.LONG,
            order_type=OrderType.LIMIT,
            units=1000,
            entry=1.1001,
            tp=1.1099,
            sl=1.0951,
            thesis="range",
            market_context=MarketContext(
                regime="RANGE",
                narrative="range",
                chart_story="range",
                method=TradeMethod.RANGE_ROTATION,
                invalidation="outside rail",
            ),
            metadata={
                "range_vehicle_candidate_id": "a" * 64,
                "range_vehicle_shape_sha256": "b" * 64,
                "range_vehicle_gateway_projection_sha256": "c" * 64,
            },
        )
        order_request = {
            "instrument": "GBP_USD",
            "type": "LIMIT",
            "units": "1000",
            "price": "1.10010",
            "timeInForce": "GTC",
            "positionFill": "DEFAULT",
            "takeProfitOnFill": {"price": "1.10990"},
            "stopLossOnFill": {"price": "1.09000"},
        }
        receipt = _range_vehicle_candidate_receipt_from_intent(
            intent, order_request
        )
        self.assertIsNotNone(receipt)
        assert receipt is not None
        self.assertEqual(
            receipt["status"],
            "CANDIDATE_IDENTITY_CARRIED_WITH_FINAL_GATEWAY_ORDER_REQUEST",
        )
        self.assertEqual(receipt["candidate_id"], "a" * 64)
        self.assertEqual(
            receipt["final_order_contract"]["stop_loss_on_fill"], "1.09000"
        )
        self.assertFalse(receipt["broker_acceptance_proved"])
        self.assertFalse(receipt["candidate_ledger_row_proved"])
        self.assertFalse(receipt["live_permission_allowed"])

    def test_ledger_failure_keeps_previous_order_intents_packet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "data" / "order_intents.json"
            output.parent.mkdir()
            output.write_text("previous-packet\n", encoding="utf-8")
            generator = IntentGenerator(
                campaign_plan=root / "data" / "daily_campaign_plan.json",
                strategy_profile=root / "data" / "strategy_profile.json",
                output_path=output,
                report_path=root / "docs" / "order_intents.md",
                pair_charts_path=root / "data" / "pair_charts.json",
                levels_path=root / "data" / "levels.json",
                market_context_matrix_path=root
                / "data"
                / "market_context_matrix.json",
                data_root=root / "data",
            )
            generated = GeneratedIntent(
                lane_id="none",
                status="NEEDS_BROKER_SNAPSHOT",
                intent=None,
                risk_metrics=None,
                risk_allowed=None,
                risk_issues=(),
                strategy_issues=(),
                live_strategy_issues=(),
                live_blockers=(),
                live_blocker_codes=(),
                note="test",
            )
            with mock.patch.object(
                intent_generator_module,
                "record_range_vehicle_candidates",
                side_effect=ValueError("ledger invalid"),
            ):
                with self.assertRaisesRegex(ValueError, "ledger invalid"):
                    generator._write_output(
                        [generated],
                        "2026-07-14T00:00:02+00:00",
                        None,
                    )

            self.assertEqual(output.read_text(encoding="utf-8"), "previous-packet\n")
            self.assertEqual(list(output.parent.glob(".order_intents.json.*.tmp")), [])

    def test_non_live_lineage_chain_defect_does_not_stale_whole_packet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result, kwargs = self._fixture(root)
            output = kwargs["order_intents_path"]
            output.write_text("previous-packet\n", encoding="utf-8")
            generator = IntentGenerator(
                campaign_plan=root / "data" / "daily_campaign_plan.json",
                strategy_profile=root / "data" / "strategy_profile.json",
                output_path=output,
                report_path=root / "docs" / "order_intents.md",
                pair_charts_path=root / "data" / "pair_charts.json",
                levels_path=root / "data" / "levels.json",
                market_context_matrix_path=root
                / "data"
                / "market_context_matrix.json",
                data_root=root / "data",
            )
            range_candidate = GeneratedIntent(
                lane_id=result["lane_id"],
                status=result["status"],
                intent=result["intent"],
                risk_metrics=None,
                risk_allowed=False,
                risk_issues=(),
                strategy_issues=(),
                live_strategy_issues=(),
                live_blockers=("negative expectancy",),
                live_blocker_codes=tuple(result["live_blocker_codes"]),
                note="non-live range diagnostic",
            )
            other_pair_surface = GeneratedIntent(
                lane_id="trend_trader:EUR_USD:LONG:TREND_FOLLOW",
                status="NO_STRATEGY",
                intent=None,
                risk_metrics=None,
                risk_allowed=None,
                risk_issues=(),
                strategy_issues=(),
                live_strategy_issues=(),
                live_blockers=(),
                live_blocker_codes=(),
                note="independent opportunity surface remains current",
            )
            with mock.patch.object(
                ledger,
                "_build_forecast_lineage_index",
                side_effect=ValueError(
                    "RANGE_VEHICLE_FORECAST_RECEIPT_CHAIN_INVALID"
                ),
            ):
                generator._write_output(
                    [range_candidate, other_pair_surface],
                    kwargs["generated_at_utc"],
                    kwargs["snapshot_path"],
                )

            packet = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(packet["results"]), 2)
            self.assertNotEqual(output.read_text(encoding="utf-8"), "previous-packet\n")
            path = output.with_name(ledger.LEDGER_FILENAME)
            payload = json.loads(path.read_text(encoding="utf-8"))["payload"]
            self.assertEqual(
                payload["forecast_lineage"]["status"],
                "FORECAST_RECEIPT_CHAIN_INVALID",
            )
            self.assertEqual(
                payload["candidate_contract_status"], "UNBOUND_NON_LIVE_SHADOW"
            )

    def test_output_directory_is_fsynced_before_packet_replace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "data" / "order_intents.json"
            output.parent.mkdir()
            generator = IntentGenerator(
                campaign_plan=root / "data" / "daily_campaign_plan.json",
                strategy_profile=root / "data" / "strategy_profile.json",
                output_path=output,
                report_path=root / "docs" / "order_intents.md",
                pair_charts_path=root / "data" / "pair_charts.json",
                levels_path=root / "data" / "levels.json",
                market_context_matrix_path=root
                / "data"
                / "market_context_matrix.json",
                data_root=root / "data",
            )
            generated = GeneratedIntent(
                lane_id="none",
                status="NEEDS_BROKER_SNAPSHOT",
                intent=None,
                risk_metrics=None,
                risk_allowed=None,
                risk_issues=(),
                strategy_issues=(),
                live_strategy_issues=(),
                live_blockers=(),
                live_blocker_codes=(),
                note="test",
            )
            events: list[str] = []
            original_fsync = os.fsync
            original_replace = os.replace

            def observed_fsync(fd: int) -> None:
                if stat.S_ISDIR(os.fstat(fd).st_mode):
                    events.append("directory_fsync")
                original_fsync(fd)

            def observed_replace(source: object, destination: object) -> None:
                events.append("replace")
                original_replace(source, destination)

            with mock.patch.object(os, "fsync", side_effect=observed_fsync), mock.patch.object(
                os, "replace", side_effect=observed_replace
            ):
                generator._write_output(
                    [generated], "2026-07-14T00:00:02+00:00", None
                )

            replace_index = events.index("replace")
            self.assertIn("directory_fsync", events[:replace_index])
            self.assertIn("directory_fsync", events[replace_index + 1 :])


if __name__ == "__main__":
    unittest.main()
