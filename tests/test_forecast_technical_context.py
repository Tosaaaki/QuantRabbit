from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from quant_rabbit.strategy.forecast_persistence_tracker import record_forecast
from quant_rabbit.strategy.regime_family_weighting import (
    verify_regime_family_weighting_receipt,
)
from quant_rabbit.strategy.tf_weights import dynamic_tf_weights_from_policy_inputs
from quant_rabbit.strategy.forecast_technical_context import (
    CONFIDENCE_SEMANTICS,
    MAX_EVIDENCE_BYTES,
    MAX_CONTEXT_NUMERIC_ABS,
    MAX_FAMILY_SCORE_ABS,
    MAX_REASON_CHARS,
    build_forecast_technical_context,
    build_forecast_technical_context_evidence,
    normalize_forecast_technical_context_evidence,
    technical_context_sha256,
    unknown_forecast_technical_context_evidence,
    verify_forecast_technical_context,
    verify_forecast_technical_context_evidence,
)


def _rehash_evidence(value: dict) -> dict:
    material = {key: item for key, item in value.items() if key != "evidence_sha256"}
    encoded = json.dumps(
        material,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    value["evidence_sha256"] = hashlib.sha256(encoded).hexdigest()
    return value


def _canonical_sha(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _chart() -> dict:
    return {
        "confluence": {
            "dominant_regime": "TREND_UP",
            "atr_percentile_24h": 0.82,
            "price_percentile_24h": 0.88,
            "price_percentile_7d": 0.42,
        },
        "views": [
            {
                "granularity": "M5",
                "regime_reading": {"state": "TREND_STRONG", "atr_percentile": 80.0},
                "indicators": {"atr_pips": 2.0},
                "structure": {
                    "structure_events": [
                        {"kind": "CHOCH_DOWN", "index": 8, "close_confirmed": True},
                        {"kind": "BOS_UP", "index": 10, "close_confirmed": True},
                    ]
                },
            },
            {
                "granularity": "M15",
                "regime_reading": {"state": "TREND_WEAK", "atr_percentile": 60.0},
                "indicators": {"atr_pips": 5.0},
                "structure": {
                    "structure_events": [
                        {
                            "kind": "CHOCH_DOWN",
                            "index": 15,
                            "close_confirmed": True,
                            "timestamp": "2026-07-13T01:00:00Z",
                        }
                    ]
                },
            },
            {
                "granularity": "H1",
                "regime_reading": {"state": "RANGE", "atr_percentile": 20.0},
                "indicators": {"atr_pips": 14.0},
                "structure": {
                    "structure_events": [
                        {"kind": "BOS_UP", "index": 20, "close_confirmed": False},
                        {"kind": "BOS_DOWN", "index": 18, "close_confirmed": True},
                    ]
                },
            },
        ],
    }


class ForecastTechnicalContextTest(unittest.TestCase):
    def test_rehashed_valid_situation_cannot_override_parent_raw_policy_evidence(self) -> None:
        context = build_forecast_technical_context(
            _chart(),
            pair="EUR_USD",
            current_price=1.1,
            spread_pips=0.5,
        )
        forged = copy.deepcopy(context)
        receipt = forged["regime_family_weighting"]
        receipt["policy_inputs"]["situation"] = "ASIA_RANGE"
        receipt["source_identity"]["situation"] = "ASIA_RANGE"
        weights = dynamic_tf_weights_from_policy_inputs(
            receipt["policy_inputs"]
        )
        receipt["weights"] = {
            timeframe: round(weight, 12)
            for timeframe, weight in weights.items()
        }
        for timeframe, row in receipt["by_timeframe"].items():
            weight = round(weights[timeframe], 12)
            row["tf_weight"] = weight
            score = row["selected_score"]
            directional_score = (
                float(score)
                if row["selected_direction"] in {"UP", "DOWN"}
                and score is not None
                else 0.0
            )
            row["weighted_directional_score"] = round(
                weight * directional_score,
                12,
            )
        rows = list(receipt["by_timeframe"].values())
        aggregate_score = round(
            sum(float(row["weighted_directional_score"]) for row in rows),
            12,
        )
        selected_coverage = sum(
            float(row["tf_weight"])
            for row in rows
            if row["selected_family"] != "NONE"
            and row["selected_score"] is not None
        )
        receipt["aggregate"] = {
            "direction": (
                "UP"
                if aggregate_score > 1e-12
                else "DOWN"
                if aggregate_score < -1e-12
                else "EITHER"
                if selected_coverage > 0.0
                else "NONE"
            ),
            "weighted_directional_score": aggregate_score,
            "directional_coverage_weight": round(
                min(
                    1.0,
                    sum(
                        float(row["tf_weight"])
                        for row in rows
                        if row["selected_direction"] in {"UP", "DOWN"}
                    ),
                ),
                12,
            ),
            "selected_family_coverage_weight": round(
                min(1.0, selected_coverage),
                12,
            ),
        }
        receipt["source_identity_sha256"] = _canonical_sha(
            receipt["source_identity"]
        )
        receipt["receipt_sha256"] = _canonical_sha(
            {
                key: item
                for key, item in receipt.items()
                if key != "receipt_sha256"
            }
        )
        self.assertEqual(
            verify_regime_family_weighting_receipt(receipt, pair="EUR_USD"),
            (True, None),
        )
        forged["context_sha256"] = technical_context_sha256(forged)
        self.assertEqual(
            verify_forecast_technical_context(forged, pair="EUR_USD"),
            (False, "REGIME_FAMILY_WEIGHTING_CONTEXT_POLICY_MISMATCH"),
        )

    def test_all_seven_weighting_rows_are_bound_to_parent_context(self) -> None:
        chart = copy.deepcopy(_chart())
        for timeframe in ("D", "H4", "M30", "M1"):
            chart["views"].append(
                {
                    "granularity": timeframe,
                    "regime_reading": {
                        "state": "TREND_WEAK",
                        "atr_percentile": 50.0,
                    },
                    "indicators": {"atr_pips": 8.0},
                    "structure": {"structure_events": []},
                }
            )
        for view in chart["views"]:
            view["family_scores"] = {
                "trend_score": -1.0,
                "mean_rev_score": -1.0,
                "breakout_score": -1.0,
                "disagreement": 0.0,
            }
        context = build_forecast_technical_context(
            chart,
            pair="EUR_USD",
            current_price=1.1,
            spread_pips=0.5,
        )
        receipt = context["regime_family_weighting"]
        self.assertEqual(receipt["aggregate"]["direction"], "DOWN")

        forged = copy.deepcopy(context)
        forged_receipt = forged["regime_family_weighting"]
        for timeframe in ("D", "M30", "M1"):
            row = forged_receipt["by_timeframe"][timeframe]
            row["selected_score"] = 100.0
            row["selected_direction"] = "UP"
            row["direction_basis"] = "TREND_SCORE_SIGN"
            row["weighted_directional_score"] = round(
                float(row["tf_weight"]) * 100.0,
                12,
            )
        rows = list(forged_receipt["by_timeframe"].values())
        aggregate_score = round(
            sum(float(row["weighted_directional_score"]) for row in rows),
            12,
        )
        forged_receipt["aggregate"] = {
            "direction": "UP",
            "weighted_directional_score": aggregate_score,
            "directional_coverage_weight": round(
                sum(
                    float(row["tf_weight"])
                    for row in rows
                    if row["selected_direction"] in {"UP", "DOWN"}
                ),
                12,
            ),
            "selected_family_coverage_weight": round(
                sum(
                    float(row["tf_weight"])
                    for row in rows
                    if row["selected_family"] != "NONE"
                    and row["selected_score"] is not None
                ),
                12,
            ),
        }
        forged_receipt["receipt_sha256"] = _canonical_sha(
            {
                key: item
                for key, item in forged_receipt.items()
                if key != "receipt_sha256"
            }
        )
        self.assertEqual(
            verify_regime_family_weighting_receipt(
                forged_receipt,
                pair="EUR_USD",
            ),
            (True, None),
        )
        forged["context_sha256"] = technical_context_sha256(forged)
        self.assertEqual(
            verify_forecast_technical_context(forged, pair="EUR_USD"),
            (False, "REGIME_FAMILY_WEIGHTING_CONTEXT_ROW_MISMATCH"),
        )

    def test_phase_aware_regime_is_identical_in_context_and_weighting_receipt(self) -> None:
        chart = copy.deepcopy(_chart())
        for view in chart["views"]:
            view["family_scores"] = {
                "trend_score": 0.8,
                "mean_rev_score": -0.6,
                "breakout_score": 0.2,
                "disagreement": 0.1,
            }
            if view["granularity"] == "M15":
                view["market_state"] = {
                    "evidence_complete": True,
                    "phase": "PRE_RANGE",
                    "readiness": "TRIGGERED",
                    "confidence": 0.75,
                }

        context = build_forecast_technical_context(
            chart,
            pair="EUR_USD",
            current_price=1.1,
            spread_pips=0.5,
        )

        self.assertEqual(context["regime"]["by_timeframe"]["M15"], "RANGE")
        self.assertEqual(context["regime"]["primary"], "RANGE")
        self.assertEqual(
            context["regime_family_weighting"]["source_identity"]["primary_regime"],
            "RANGE",
        )
        self.assertEqual(
            context["regime_family_weighting"]["by_timeframe"]["M15"]["regime_state"],
            "RANGE",
        )
        self.assertEqual(
            verify_forecast_technical_context(context, pair="EUR_USD"),
            (True, None),
        )

    def test_regime_atr_percentile_is_not_reinterpreted_as_fraction(self) -> None:
        chart = copy.deepcopy(_chart())
        chart["views"][1]["regime_reading"]["atr_percentile"] = 0.5
        context = build_forecast_technical_context(
            chart,
            pair="EUR_USD",
            current_price=1.1,
            spread_pips=0.5,
        )

        self.assertEqual(
            context["volatility"]["atr_percentile_by_timeframe"]["M15"],
            0.5,
        )
        self.assertEqual(context["volatility"]["primary_atr_band"], "LOW")
        self.assertEqual(verify_forecast_technical_context(context), (True, None))

    def test_evidence_handoff_is_content_addressed_and_tamper_closed(self) -> None:
        context = build_forecast_technical_context(
            _chart(),
            pair="EUR_USD",
            current_price=1.1,
            spread_pips=0.5,
        )
        evidence = build_forecast_technical_context_evidence(
            context,
            pair="EUR_USD",
            current_price=1.1,
        )

        self.assertEqual(evidence["status"], "VALID")
        self.assertEqual(evidence["confidence_semantics"], CONFIDENCE_SEMANTICS)
        self.assertEqual(evidence["technical_context_v1"], context)
        self.assertEqual(evidence["context_sha256"], context["context_sha256"])
        self.assertFalse(evidence["proof_eligible"])
        self.assertFalse(evidence["live_permission"])
        self.assertEqual(
            verify_forecast_technical_context_evidence(
                evidence,
                pair="EUR_USD",
                current_price=1.1,
            ),
            (True, None),
        )
        self.assertEqual(
            verify_forecast_technical_context_evidence(
                evidence,
                pair="EUR_USD",
                current_price=None,
            ),
            (False, "TECHNICAL_CONTEXT_EVIDENCE_PRICE_MISSING"),
        )

        tampered = copy.deepcopy(evidence)
        tampered["technical_context_v1"]["location"]["range_location_24h"] = "LOWER"
        normalized = normalize_forecast_technical_context_evidence(
            tampered,
            pair="EUR_USD",
            current_price=1.1,
        )
        self.assertEqual(normalized["status"], "UNKNOWN")
        self.assertEqual(
            normalized["reason"],
            "TECHNICAL_CONTEXT_EVIDENCE_HASH_MISMATCH",
        )
        self.assertIsNone(normalized["technical_context_v1"])
        self.assertIsNone(normalized["context_sha256"])
        self.assertFalse(normalized["live_permission"])

        missing = build_forecast_technical_context_evidence(
            None,
            pair="EUR_USD",
            current_price=1.1,
        )
        self.assertEqual(missing["status"], "UNKNOWN")
        self.assertEqual(missing["reason"], "TECHNICAL_CONTEXT_MISSING")
        self.assertIsNone(missing["technical_context_v1"])
        self.assertFalse(missing["live_permission"])

    def test_unknown_evidence_is_canonical_and_deterministically_bounded(self) -> None:
        normal = unknown_forecast_technical_context_evidence(
            "TECHNICAL_CONTEXT_MISSING"
        )
        self.assertEqual(
            verify_forecast_technical_context_evidence(
                normal,
                pair="EUR_USD",
                current_price=1.1,
            ),
            (True, None),
        )
        self.assertLessEqual(
            len(
                json.dumps(
                    normal,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ),
            MAX_EVIDENCE_BYTES,
        )

        oversized_reason = "R" * (MAX_REASON_CHARS + 50)
        bounded = unknown_forecast_technical_context_evidence(oversized_reason)
        self.assertEqual(bounded["reason"], "R" * MAX_REASON_CHARS)
        self.assertEqual(
            verify_forecast_technical_context_evidence(
                bounded,
                pair="EUR_USD",
                current_price=1.1,
            ),
            (True, None),
        )
        fallback = unknown_forecast_technical_context_evidence(
            {"untrusted": "reason"}  # type: ignore[arg-type]
        )
        self.assertEqual(fallback["reason"], "TECHNICAL_CONTEXT_UNKNOWN")

    def test_rehashed_oversized_or_noncanonical_unknown_fails_closed(self) -> None:
        base = unknown_forecast_technical_context_evidence("SAFE_REASON")
        malicious_reasons = (
            "DO_NOT_FORWARD" * MAX_EVIDENCE_BYTES,
            {"echo": "DO_NOT_FORWARD" * MAX_EVIDENCE_BYTES},
            ["DO_NOT_FORWARD" * MAX_EVIDENCE_BYTES],
        )
        for reason in malicious_reasons:
            with self.subTest(reason_type=type(reason).__name__):
                malicious = copy.deepcopy(base)
                malicious["reason"] = reason
                _rehash_evidence(malicious)
                valid, error = verify_forecast_technical_context_evidence(
                    malicious,
                    pair="EUR_USD",
                    current_price=1.1,
                )
                self.assertFalse(valid)
                self.assertEqual(error, "TECHNICAL_CONTEXT_EVIDENCE_TOO_LARGE")
                normalized = normalize_forecast_technical_context_evidence(
                    malicious,
                    pair="EUR_USD",
                    current_price=1.1,
                )
                self.assertEqual(normalized["status"], "UNKNOWN")
                self.assertEqual(
                    normalized["reason"],
                    "TECHNICAL_CONTEXT_EVIDENCE_TOO_LARGE",
                )
                self.assertNotIn("DO_NOT_FORWARD", json.dumps(normalized))

        for reason in ({"echo": "small"}, ["small"]):
            with self.subTest(non_string_type=type(reason).__name__):
                malicious = copy.deepcopy(base)
                malicious["reason"] = reason
                _rehash_evidence(malicious)
                self.assertEqual(
                    verify_forecast_technical_context_evidence(
                        malicious,
                        pair="EUR_USD",
                        current_price=1.1,
                    ),
                    (False, "TECHNICAL_CONTEXT_EVIDENCE_UNKNOWN_INVALID"),
                )

        for status in ("unknown", " UNKNOWN "):
            with self.subTest(status=status):
                noncanonical = copy.deepcopy(base)
                noncanonical["status"] = status
                _rehash_evidence(noncanonical)
                self.assertEqual(
                    verify_forecast_technical_context_evidence(
                        noncanonical,
                        pair="EUR_USD",
                        current_price=1.1,
                    ),
                    (False, "TECHNICAL_CONTEXT_EVIDENCE_STATUS_INVALID"),
                )

    def test_builds_normalized_content_addressed_context(self) -> None:
        context = build_forecast_technical_context(
            _chart(),
            pair="EUR_USD",
            current_price=1.123456789,
            spread_pips=0.8,
        )

        self.assertEqual(context["schema_version"], "technical_context_v1")
        self.assertEqual(context["regime"]["primary"], "TREND_WEAK")
        self.assertEqual(context["volatility"]["primary_atr_band"], "NORMAL")
        self.assertEqual(context["execution"]["spread_band"], "NORMAL")
        self.assertEqual(context["location"]["range_location_24h"], "UPPER")
        self.assertEqual(context["structure"]["primary_timeframe"], "H1")
        self.assertEqual(context["structure"]["primary_direction"], "DOWN")
        self.assertEqual(context["identity"]["pair"], "EUR_USD")
        self.assertIn("trend_score", context["families"]["by_timeframe"]["M15"])
        self.assertTrue(context["completeness"]["complete"])
        self.assertEqual(context["context_sha256"], technical_context_sha256(context))
        self.assertEqual(verify_forecast_technical_context(context), (True, None))
        self.assertEqual(
            verify_forecast_technical_context(
                context,
                pair="EUR_USD",
                current_price=1.12346,
            ),
            (True, None),
        )
        self.assertEqual(
            verify_forecast_technical_context(
                context,
                pair="EUR_USD",
                current_price=1.12360,
            ),
            (False, "TECHNICAL_CONTEXT_PRICE_MISMATCH"),
        )

        other_pair = build_forecast_technical_context(
            _chart(),
            pair="GBP_USD",
            current_price=1.123456789,
            spread_pips=0.8,
        )
        self.assertNotEqual(context["context_sha256"], other_pair["context_sha256"])
        self.assertEqual(
            verify_forecast_technical_context(context, pair="GBP_USD"),
            (False, "TECHNICAL_CONTEXT_PAIR_MISMATCH"),
        )

    def test_hash_detects_point_in_time_context_tampering(self) -> None:
        context = build_forecast_technical_context(
            _chart(),
            pair="EUR_USD",
            current_price=1.1,
            spread_pips=0.5,
        )
        tampered = copy.deepcopy(context)
        tampered["location"]["range_location_24h"] = "LOWER"

        self.assertEqual(
            verify_forecast_technical_context(tampered),
            (False, "TECHNICAL_CONTEXT_HASH_MISMATCH"),
        )

        expanded = copy.deepcopy(context)
        expanded["raw_candles"] = [1, 2, 3]
        expanded["context_sha256"] = technical_context_sha256(expanded)
        self.assertEqual(
            verify_forecast_technical_context(expanded),
            (False, "TECHNICAL_CONTEXT_SCHEMA_INVALID"),
        )

    def test_rehash_cannot_make_malformed_or_self_declared_complete_context_valid(self) -> None:
        malformed = {
            "schema_version": "technical_context_v1",
            "identity": {"pair": "EUR_USD"},
            "regime": {},
            "volatility": {},
            "execution": {},
            "location": {"current_price": 1.1},
            "structure": {},
            "families": {},
            "completeness": {"complete": True},
            "context_sha256": "",
        }
        malformed["context_sha256"] = technical_context_sha256(malformed)

        self.assertEqual(
            verify_forecast_technical_context(
                malformed,
                pair="EUR_USD",
                current_price=1.1,
            ),
            (False, "TECHNICAL_CONTEXT_SCHEMA_INVALID"),
        )
        evidence = build_forecast_technical_context_evidence(
            malformed,
            pair="EUR_USD",
            current_price=1.1,
        )
        self.assertEqual(evidence["status"], "UNKNOWN")
        self.assertEqual(evidence["reason"], "TECHNICAL_CONTEXT_SCHEMA_INVALID")
        self.assertIsNone(evidence["technical_context_v1"])

        self_declared_complete = build_forecast_technical_context(
            _chart(),
            pair="EUR_USD",
            current_price=1.1,
            spread_pips=0.5,
        )
        self_declared_complete["regime"]["dominant"] = "UNKNOWN"
        self_declared_complete["context_sha256"] = technical_context_sha256(
            self_declared_complete
        )
        self.assertEqual(
            verify_forecast_technical_context(self_declared_complete),
            (False, "TECHNICAL_CONTEXT_COMPLETENESS_INVALID"),
        )

    def test_rehash_cannot_make_inconsistent_derived_enums_valid(self) -> None:
        context = build_forecast_technical_context(
            _chart(),
            pair="EUR_USD",
            current_price=1.1,
            spread_pips=0.5,
        )
        substitutions = (
            (("regime", "primary"), "RANGE"),
            (("volatility", "primary_atr_band"), "HIGH"),
            (("volatility", "atr_band_by_timeframe", "M15"), "HIGH"),
            (("execution", "spread_band"), "NORMAL"),
            (("location", "range_location_24h"), "LOWER"),
            (("location", "range_location_7d"), "UPPER"),
        )

        for path, replacement in substitutions:
            with self.subTest(path=".".join(path)):
                tampered = copy.deepcopy(context)
                target = tampered
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = replacement
                tampered["context_sha256"] = technical_context_sha256(tampered)

                self.assertEqual(
                    verify_forecast_technical_context(tampered),
                    (False, "TECHNICAL_CONTEXT_SCHEMA_INVALID"),
                )

    def test_rehash_cannot_break_numeric_derivation_links(self) -> None:
        context = build_forecast_technical_context(
            _chart(),
            pair="EUR_USD",
            current_price=1.1,
            spread_pips=0.5,
        )

        primary_mismatch = copy.deepcopy(context)
        primary_mismatch["volatility"]["primary_atr_percentile"] = 55.0
        primary_mismatch["context_sha256"] = technical_context_sha256(
            primary_mismatch
        )

        m5_map_mismatch = copy.deepcopy(context)
        m5_map_mismatch["execution"]["m5_atr_pips"] = 4.0
        m5_map_mismatch["execution"]["spread_to_m5_atr"] = 0.125
        m5_map_mismatch["context_sha256"] = technical_context_sha256(
            m5_map_mismatch
        )

        spread_ratio_mismatch = copy.deepcopy(context)
        spread_ratio_mismatch["execution"]["spread_to_m5_atr"] = 0.5
        spread_ratio_mismatch["execution"]["spread_band"] = "NORMAL"
        spread_ratio_mismatch["context_sha256"] = technical_context_sha256(
            spread_ratio_mismatch
        )

        for label, tampered in (
            ("primary_atr_percentile", primary_mismatch),
            ("m5_atr_map", m5_map_mismatch),
            ("spread_ratio", spread_ratio_mismatch),
        ):
            with self.subTest(label=label):
                self.assertEqual(
                    verify_forecast_technical_context(tampered),
                    (False, "TECHNICAL_CONTEXT_SCHEMA_INVALID"),
                )

    def test_builder_derives_bands_from_rounded_stored_values(self) -> None:
        chart = copy.deepcopy(_chart())
        chart["views"][1]["regime_reading"]["atr_percentile"] = 25.00001
        chart["confluence"]["price_percentile_24h"] = 0.3333334
        chart["confluence"]["price_percentile_7d"] = 0.6666666

        context = build_forecast_technical_context(
            chart,
            pair="EUR_USD",
            current_price=1.1,
            spread_pips=0.5000004,
        )

        self.assertEqual(
            context["volatility"]["atr_percentile_by_timeframe"]["M15"],
            25.0,
        )
        self.assertEqual(
            context["volatility"]["atr_band_by_timeframe"]["M15"],
            "LOW",
        )
        self.assertEqual(context["volatility"]["primary_atr_band"], "LOW")
        self.assertEqual(context["execution"]["spread_pips"], 0.5)
        self.assertEqual(context["execution"]["spread_to_m5_atr"], 0.25)
        self.assertEqual(context["execution"]["spread_band"], "TIGHT")
        self.assertEqual(context["location"]["price_percentile_24h"], 0.333333)
        self.assertEqual(context["location"]["range_location_24h"], "LOWER")
        self.assertEqual(context["location"]["price_percentile_7d"], 0.666667)
        self.assertEqual(context["location"]["range_location_7d"], "UPPER")
        self.assertEqual(verify_forecast_technical_context(context), (True, None))

    def test_confluence_primary_atr_fallback_is_stored_and_revalidated(self) -> None:
        chart = copy.deepcopy(_chart())
        chart["views"][0]["regime_reading"].pop("atr_percentile")
        chart["views"][1]["regime_reading"].pop("atr_percentile")
        context = build_forecast_technical_context(
            chart,
            pair="EUR_USD",
            current_price=1.1,
            spread_pips=0.5,
        )

        self.assertEqual(
            context["volatility"]["confluence_atr_percentile_24h"],
            82.0,
        )
        self.assertEqual(context["volatility"]["primary_atr_percentile"], 82.0)
        self.assertEqual(context["volatility"]["primary_atr_band"], "HIGH")
        self.assertTrue(context["completeness"]["complete"])
        self.assertEqual(verify_forecast_technical_context(context), (True, None))

        tampered = copy.deepcopy(context)
        tampered["volatility"]["primary_atr_percentile"] = 10.0
        tampered["volatility"]["primary_atr_band"] = "LOW"
        tampered["context_sha256"] = technical_context_sha256(tampered)

        self.assertEqual(
            verify_forecast_technical_context(tampered),
            (False, "TECHNICAL_CONTEXT_SCHEMA_INVALID"),
        )
        evidence = build_forecast_technical_context_evidence(
            tampered,
            pair="EUR_USD",
            current_price=1.1,
        )
        self.assertEqual(evidence["status"], "UNKNOWN")
        self.assertEqual(evidence["reason"], "TECHNICAL_CONTEXT_SCHEMA_INVALID")

    def test_legacy_and_current_volatility_shapes_are_exactly_validated(self) -> None:
        current = build_forecast_technical_context(
            _chart(),
            pair="EUR_USD",
            current_price=1.1,
            spread_pips=0.5,
        )
        legacy_recomputable = copy.deepcopy(current)
        legacy_recomputable["volatility"].pop(
            "confluence_atr_percentile_24h"
        )
        legacy_recomputable["context_sha256"] = technical_context_sha256(
            legacy_recomputable
        )
        self.assertEqual(
            verify_forecast_technical_context(legacy_recomputable),
            (True, None),
        )

        fallback_chart = copy.deepcopy(_chart())
        fallback_chart["views"][0]["regime_reading"].pop("atr_percentile")
        fallback_chart["views"][1]["regime_reading"].pop("atr_percentile")
        legacy_unverifiable = build_forecast_technical_context(
            fallback_chart,
            pair="EUR_USD",
            current_price=1.1,
            spread_pips=0.5,
        )
        legacy_unverifiable["volatility"].pop(
            "confluence_atr_percentile_24h"
        )
        legacy_unverifiable["context_sha256"] = technical_context_sha256(
            legacy_unverifiable
        )
        self.assertEqual(
            verify_forecast_technical_context(legacy_unverifiable),
            (False, "TECHNICAL_CONTEXT_SCHEMA_INVALID"),
        )

        no_fallback_chart = copy.deepcopy(fallback_chart)
        no_fallback_chart["confluence"].pop("atr_percentile_24h")
        legacy_unknown = build_forecast_technical_context(
            no_fallback_chart,
            pair="EUR_USD",
            current_price=1.1,
            spread_pips=0.5,
        )
        legacy_unknown["volatility"].pop("confluence_atr_percentile_24h")
        legacy_unknown["context_sha256"] = technical_context_sha256(
            legacy_unknown
        )
        self.assertIsNone(
            legacy_unknown["volatility"]["primary_atr_percentile"]
        )
        self.assertEqual(
            legacy_unknown["volatility"]["primary_atr_band"],
            "UNKNOWN",
        )
        self.assertEqual(
            verify_forecast_technical_context(legacy_unknown),
            (True, None),
        )

        extra_field = copy.deepcopy(current)
        extra_field["volatility"]["unrecognized_fallback"] = 82.0
        extra_field["context_sha256"] = technical_context_sha256(extra_field)
        self.assertEqual(
            verify_forecast_technical_context(extra_field),
            (False, "TECHNICAL_CONTEXT_SCHEMA_INVALID"),
        )

    def test_extreme_rehashed_numbers_fail_closed_without_overflow(self) -> None:
        context = build_forecast_technical_context(
            _chart(),
            pair="EUR_USD",
            current_price=1.1,
            spread_pips=0.5,
        )
        mutations = (
            (("families", "by_timeframe", "M5", "trend_score"), 10**400),
            (("families", "by_timeframe", "H1", "disagreement"), 1e308),
            (("location", "current_price"), 1e308),
            (("volatility", "atr_pips_by_timeframe", "H1"), 1e308),
            (("execution", "spread_pips"), 1e308),
            (("structure", "by_timeframe", "H1", "index"), 1e308),
        )

        for path, replacement in mutations:
            with self.subTest(path=".".join(path)):
                tampered = copy.deepcopy(context)
                target = tampered
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = replacement
                tampered["context_sha256"] = technical_context_sha256(tampered)

                self.assertEqual(
                    verify_forecast_technical_context(tampered),
                    (False, "TECHNICAL_CONTEXT_SCHEMA_INVALID"),
                )
                evidence = build_forecast_technical_context_evidence(
                    tampered,
                    pair="EUR_USD",
                    current_price=1.1,
                )
                self.assertEqual(evidence["status"], "UNKNOWN")
                self.assertEqual(
                    evidence["reason"],
                    "TECHNICAL_CONTEXT_SCHEMA_INVALID",
                )

    def test_builder_bounds_extreme_raw_numbers_and_keeps_named_limits(self) -> None:
        extreme_chart = copy.deepcopy(_chart())
        m5 = extreme_chart["views"][0]
        m5["indicators"]["atr_pips"] = 1e308
        m5["family_scores"] = {
            "trend_score": 10**400,
            "mean_rev_score": -1e308,
            "breakout_score": 1e308,
            "disagreement": 10**400,
        }
        m5["structure"]["structure_events"] = [
            {
                "kind": "BOS_UP",
                "index": 10**400,
                "close_confirmed": True,
            }
        ]

        bounded = build_forecast_technical_context(
            extreme_chart,
            pair="EUR_USD",
            current_price=1e308,
            spread_pips=10**400,
        )
        self.assertIsNone(bounded["location"]["current_price"])
        self.assertIsNone(bounded["execution"]["spread_pips"])
        self.assertIsNone(bounded["execution"]["m5_atr_pips"])
        self.assertIsNone(
            bounded["volatility"]["atr_pips_by_timeframe"]["M5"]
        )
        self.assertIsNone(
            bounded["structure"]["by_timeframe"]["M5"]["index"]
        )
        self.assertTrue(
            all(
                value is None
                for value in bounded["families"]["by_timeframe"]["M5"].values()
            )
        )
        self.assertEqual(verify_forecast_technical_context(bounded), (True, None))
        evidence = build_forecast_technical_context_evidence(
            bounded,
            pair="EUR_USD",
            current_price=1e308,
        )
        self.assertEqual(evidence["status"], "UNKNOWN")
        self.assertEqual(evidence["reason"], "TECHNICAL_CONTEXT_PRICE_MISSING")

        boundary_chart = copy.deepcopy(_chart())
        boundary_chart["views"][0]["family_scores"] = {
            "trend_score": MAX_FAMILY_SCORE_ABS,
            "mean_rev_score": -MAX_FAMILY_SCORE_ABS,
            "breakout_score": MAX_FAMILY_SCORE_ABS,
            "disagreement": -MAX_FAMILY_SCORE_ABS,
        }
        boundary_chart["views"][2]["structure"]["structure_events"][1][
            "index"
        ] = MAX_CONTEXT_NUMERIC_ABS
        boundary = build_forecast_technical_context(
            boundary_chart,
            pair="EUR_USD",
            current_price=MAX_CONTEXT_NUMERIC_ABS,
            spread_pips=0.5,
        )
        self.assertEqual(
            boundary["families"]["by_timeframe"]["M5"]["trend_score"],
            MAX_FAMILY_SCORE_ABS,
        )
        self.assertEqual(
            boundary["families"]["by_timeframe"]["M5"][
                "mean_reversion_score"
            ],
            -MAX_FAMILY_SCORE_ABS,
        )
        self.assertEqual(
            boundary["structure"]["by_timeframe"]["H1"]["index"],
            MAX_CONTEXT_NUMERIC_ABS,
        )
        self.assertEqual(
            verify_forecast_technical_context(
                boundary,
                pair="EUR_USD",
                current_price=MAX_CONTEXT_NUMERIC_ABS,
            ),
            (True, None),
        )

    def test_record_forecast_persists_exact_context_and_rejects_bad_hash(self) -> None:
        context = build_forecast_technical_context(
            _chart(),
            pair="EUR_USD",
            current_price=1.1,
            spread_pips=0.5,
        )
        forecast = SimpleNamespace(
            pair="EUR_USD",
            direction="UP",
            confidence=0.6,
            current_price=1.1,
            invalidation_price=1.09,
            target_price=1.12,
            horizon_min=60,
            drivers_for=(),
            drivers_against=(),
            rationale_summary="test",
            technical_context_v1=context,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertTrue(record_forecast(forecast, data_root=root, cycle_id="cycle-1"))
            row = json.loads((root / "forecast_history.jsonl").read_text())
            self.assertEqual(row["technical_context_v1"], context)
            receipt = json.loads((root / "forecast_emission_receipts.jsonl").read_text())
            self.assertEqual(receipt["schema_version"], "QR_FORECAST_EMISSION_RECEIPT_V1")
            self.assertEqual(receipt["sequence"], 1)
            self.assertEqual(receipt["operation"], "APPEND")
            self.assertEqual(receipt["pair"], "EUR_USD")
            self.assertEqual(receipt["cycle_id"], "cycle-1")

            bad_forecast = copy.copy(forecast)
            bad_context = copy.deepcopy(context)
            bad_context["execution"]["spread_band"] = "WIDE"
            bad_forecast.technical_context_v1 = bad_context
            with self.assertRaisesRegex(ValueError, "TECHNICAL_CONTEXT_HASH_MISMATCH"):
                record_forecast(bad_forecast, data_root=root, cycle_id="cycle-2")

    def test_legacy_context_remains_display_valid_without_fresh_receipt(self) -> None:
        context = build_forecast_technical_context(
            _chart(),
            pair="EUR_USD",
            current_price=1.1,
            spread_pips=0.5,
        )
        context.pop("regime_family_weighting")
        context["context_sha256"] = technical_context_sha256(context)

        self.assertEqual(
            verify_forecast_technical_context(context, pair="EUR_USD"),
            (True, None),
        )
        evidence = build_forecast_technical_context_evidence(
            context,
            pair="EUR_USD",
            current_price=1.1,
        )
        self.assertEqual(evidence["status"], "VALID")
        self.assertNotIn(
            "regime_family_weighting",
            evidence["technical_context_v1"],
        )


if __name__ == "__main__":
    unittest.main()
