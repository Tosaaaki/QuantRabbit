"""Causal bid/ask forecast-orientation learning.

The deterministic forecaster emits one technical direction.  This module
learns whether that direction should be kept or inverted from chronological
OANDA bid/ask truth.  It never decides whether to suppress a forecast: every
prediction receives a direction and a continuous rank score.  A model becomes
an ordinary forecast correction only after a later holdout beats both the raw
direction and the always-invert baseline after spread; otherwise it remains a
forward-learning ranker for the bounded SCOUT route.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


FORECAST_LEARNING_MODEL_SCHEMA = "QR_FORECAST_ORIENTATION_NB_V1"
FORECAST_LEARNING_MIN_REPLAY_SAMPLES = 30
FORECAST_LEARNING_FEATURES = (
    "pair",
    "direction",
    "confidence_bucket",
    "raw_confidence_bucket",
    "score_margin_bucket",
    "range_competition",
    "utc_session_bucket",
    "primary_driver_family",
    "primary_against_driver_family",
    "technical_regime",
    "technical_atr_band",
    "technical_spread_band",
    "technical_range_location_24h",
    "technical_structure_alignment",
    "technical_situation",
    "technical_selected_method",
    "technical_family_direction_alignment",
)
FORECAST_LEARNING_TECHNICAL_FEATURES = tuple(
    name for name in FORECAST_LEARNING_FEATURES if name.startswith("technical_")
)
FORECAST_LEARNING_EXECUTION_GEOMETRY_SCHEMA = (
    "QR_FORECAST_LEARNING_EXECUTION_GEOMETRY_V1"
)
FORECAST_LEARNING_EXECUTION_GEOMETRY_BASIS = (
    "ORIENTATION_RANK_TO_PASSIVE_LIMIT_TECHNICAL_VEHICLE"
)
FORECAST_LEARNING_EXECUTION_DESK_BY_METHOD = {
    "BREAKOUT_FAILURE": "failure_trader",
    "RANGE_ROTATION": "range_trader",
    "TREND_CONTINUATION": "trend_trader",
}


def forecast_learning_selected_method(receipt: Mapping[str, Any] | None) -> str | None:
    """Return the point-in-time technical method bound into a learning decision.

    A rank direction without an executable technical family is not silently
    relabelled as a failed-break trade.  The caller may rank it for telemetry,
    but it cannot mint a forward vehicle until the current technical context
    names one of the supported entry families.
    """

    if not isinstance(receipt, Mapping):
        return None
    features = receipt.get("features")
    if not isinstance(features, Mapping):
        return None
    method = str(features.get("technical_selected_method") or "").strip().upper()
    return method if method in FORECAST_LEARNING_EXECUTION_DESK_BY_METHOD else None


def build_forecast_learning_execution_geometry(
    *,
    pair: str,
    side: str,
    method: str,
    entry: float,
    take_profit: float,
    stop_loss: float,
    source_decision_sha256: str,
    forecast_current_price: float | None,
    forecast_target_price: float | None,
    forecast_invalidation_price: float | None,
) -> dict[str, Any]:
    """Bind an orientation prediction to its exact passive LIMIT vehicle.

    The directional learner ranks orientation only.  Original point-forecast
    prices stay in this receipt for audit, while the executable target and
    invalidation are the current technical TP/SL around the actual LIMIT entry.
    This prevents a current-price forecast from being misread as geometry for a
    later passive fill.
    """

    body: dict[str, Any] = {
        "schema": FORECAST_LEARNING_EXECUTION_GEOMETRY_SCHEMA,
        "basis": FORECAST_LEARNING_EXECUTION_GEOMETRY_BASIS,
        "pair": str(pair).strip().upper(),
        "side": str(side).strip().upper(),
        "method": str(method).strip().upper(),
        "source_decision_sha256": str(source_decision_sha256 or ""),
        "forecast_origin_current_price": _finite_float(forecast_current_price),
        "forecast_origin_target_price": _finite_float(forecast_target_price),
        "forecast_origin_invalidation_price": _finite_float(
            forecast_invalidation_price
        ),
        "execution_entry_price": _finite_float(entry),
        "execution_target_price": _finite_float(take_profit),
        "execution_invalidation_price": _finite_float(stop_loss),
    }
    return {**body, "binding_sha256": _digest(body)}


def validate_forecast_learning_execution_geometry(
    payload: Mapping[str, Any] | None,
    *,
    pair: str,
    side: str,
    method: str,
    entry: float | None,
    take_profit: float | None,
    stop_loss: float | None,
    source_decision_sha256: str,
) -> bool:
    """Authenticate a learning-SCOUT execution receipt against the intent."""

    if not isinstance(payload, Mapping):
        return False
    binding = str(payload.get("binding_sha256") or "")
    body = {key: value for key, value in payload.items() if key != "binding_sha256"}
    if not binding or binding != _digest(body):
        return False
    if (
        str(payload.get("schema") or "")
        != FORECAST_LEARNING_EXECUTION_GEOMETRY_SCHEMA
        or str(payload.get("basis") or "")
        != FORECAST_LEARNING_EXECUTION_GEOMETRY_BASIS
        or str(payload.get("pair") or "").upper() != str(pair or "").upper()
        or str(payload.get("side") or "").upper() != str(side or "").upper()
        or str(payload.get("method") or "").upper() != str(method or "").upper()
        or str(payload.get("source_decision_sha256") or "")
        != str(source_decision_sha256 or "")
    ):
        return False
    expected = (entry, take_profit, stop_loss)
    actual = (
        payload.get("execution_entry_price"),
        payload.get("execution_target_price"),
        payload.get("execution_invalidation_price"),
    )
    for actual_value, expected_value in zip(actual, expected):
        parsed_actual = _finite_float(actual_value)
        parsed_expected = _finite_float(expected_value)
        if (
            parsed_actual is None
            or parsed_expected is None
            or not math.isclose(
                parsed_actual,
                parsed_expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            return False
    return True


def train_forecast_orientation_model(
    direct_rows: Sequence[Mapping[str, Any]],
    inverse_rows: Sequence[Mapping[str, Any]],
    *,
    train_fraction: float = 0.60,
    source: str = "",
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train on the older chronological partition and audit on the newer one."""

    inverse_by_key = {
        _row_key(row): row
        for row in inverse_rows
        if _row_key(row) is not None
    }
    examples: list[dict[str, Any]] = []
    for row in direct_rows:
        key = _row_key(row)
        inverse = inverse_by_key.get(key)
        if key is None or inverse is None:
            continue
        direct_pips = _finite_float(row.get("final_pips"))
        inverse_pips = _finite_float(inverse.get("final_pips"))
        timestamp = _parse_utc(row.get("timestamp_utc"))
        horizon_min = _positive_float(row.get("horizon_min"))
        if (
            direct_pips is None
            or inverse_pips is None
            or timestamp is None
            or horizon_min is None
        ):
            continue
        examples.append(
            {
                "timestamp": timestamp,
                "horizon_min": horizon_min,
                "features": scored_row_feature_values(row),
                "direct_better": direct_pips >= inverse_pips,
                "direct_pips": direct_pips,
                "inverse_pips": inverse_pips,
            }
        )
    examples.sort(key=lambda item: item["timestamp"])
    if len(examples) < FORECAST_LEARNING_MIN_REPLAY_SAMPLES * 2:
        return _unavailable_model(
            status="INSUFFICIENT_REPLAY_SAMPLES",
            source=source,
            examples=len(examples),
        )

    split_at, training_split_basis = _chronological_split_index(
        examples,
        train_fraction=train_fraction,
    )
    validation_start = examples[split_at]["timestamp"]
    training = [
        item
        for item in examples[:split_at]
        if item["timestamp"].timestamp() + item["horizon_min"] * 60.0
        <= validation_start.timestamp()
    ]
    validation = [
        item for item in examples if item["timestamp"] >= validation_start
    ]
    if (
        len(training) < FORECAST_LEARNING_MIN_REPLAY_SAMPLES
        or len(validation) < FORECAST_LEARNING_MIN_REPLAY_SAMPLES
    ):
        return _unavailable_model(
            status="INSUFFICIENT_PURGED_SPLIT",
            source=source,
            examples=len(examples),
            training=len(training),
            validation=len(validation),
        )

    # Use an inner chronological tail only to shrink correlated Naive-Bayes
    # evidence.  The outer validation remains untouched until the model and
    # its calibration scale are frozen.
    calibration_n = max(
        FORECAST_LEARNING_MIN_REPLAY_SAMPLES,
        len(training) // 5,
    )
    discovery = training[:-calibration_n]
    calibration = training[-calibration_n:]
    if len(discovery) < FORECAST_LEARNING_MIN_REPLAY_SAMPLES:
        discovery = training
        calibration = []
    discovery_model = _fit_naive_bayes(discovery)
    scale = _calibration_scale(discovery_model, calibration)
    learned = _fit_naive_bayes(training)
    learned["logit_scale"] = scale
    validation_metrics = _evaluate(learned, validation)
    training_metrics = _evaluate(learned, training)
    training_feature_coverage = _feature_coverage(training)
    validation_feature_coverage = _feature_coverage(validation)
    technical_feature_training_status = (
        "AVAILABLE"
        if any(
            training_feature_coverage[name] > 0.0
            for name in FORECAST_LEARNING_TECHNICAL_FEATURES
        )
        else "MISSING_IN_LEGACY_FORECAST_HISTORY"
    )
    enabled = bool(
        validation_metrics["n"] >= FORECAST_LEARNING_MIN_REPLAY_SAMPLES
        and validation_metrics["selected_avg_final_pips"] > 0.0
        and validation_metrics["selected_hit_rate"] > 0.5
        and validation_metrics["selected_avg_final_pips"]
        > validation_metrics["direct_avg_final_pips"]
        and validation_metrics["selected_avg_final_pips"]
        > validation_metrics["inverse_avg_final_pips"]
    )
    status = "ENABLED" if enabled else "RANK_ONLY"
    body: dict[str, Any] = {
        "schema": FORECAST_LEARNING_MODEL_SCHEMA,
        "status": status,
        "ordinary_forecast_correction_enabled": enabled,
        "forward_learning_rank_enabled": True,
        "source": source,
        "training_provenance": dict(provenance or {}),
        "selection": (
            "predict DIRECT when posterior>=0.5, otherwise INVERSE; never NO_TRADE"
        ),
        "feature_names": list(FORECAST_LEARNING_FEATURES),
        "minimum_replay_samples": FORECAST_LEARNING_MIN_REPLAY_SAMPLES,
        "training_split_basis": training_split_basis,
        "training_from_utc": training[0]["timestamp"].isoformat(),
        "training_to_utc_exclusive": validation_start.isoformat(),
        "validation_from_utc": validation_start.isoformat(),
        "validation_to_utc": validation[-1]["timestamp"].isoformat(),
        "training_metrics": training_metrics,
        "validation_metrics": validation_metrics,
        "training_feature_coverage": training_feature_coverage,
        "validation_feature_coverage": validation_feature_coverage,
        "technical_feature_training_status": technical_feature_training_status,
        "model": learned,
    }
    return {**body, "model_sha256": _digest(body)}


def _chronological_split_index(
    examples: Sequence[Mapping[str, Any]],
    *,
    train_fraction: float,
) -> tuple[int, str]:
    """Choose an untouched chronological holdout that can learn new context.

    Legacy forecasts legitimately lack point-in-time technical context and may
    never be backfilled from later charts. Once enough newly emitted rows carry
    frozen context, a fixed fraction of the entire multi-month ledger can keep
    every technical row on the validation side forever. Anchor the split to the
    newer technical cohort instead: older rows remain training evidence, the
    earlier chronological portion of the real context cohort warms the model,
    and its later portion remains untouched holdout evidence.
    """

    default_split = min(
        max(
            int(len(examples) * train_fraction),
            FORECAST_LEARNING_MIN_REPLAY_SAMPLES,
        ),
        len(examples) - FORECAST_LEARNING_MIN_REPLAY_SAMPLES,
    )
    technical_indexes = [
        index
        for index, item in enumerate(examples)
        if _example_has_technical_context(item)
    ]
    if len(technical_indexes) < FORECAST_LEARNING_MIN_REPLAY_SAMPLES * 2:
        return default_split, "FULL_HISTORY_CHRONOLOGICAL"
    technical_train_count = min(
        max(
            int(len(technical_indexes) * train_fraction),
            FORECAST_LEARNING_MIN_REPLAY_SAMPLES,
        ),
        len(technical_indexes) - FORECAST_LEARNING_MIN_REPLAY_SAMPLES,
    )
    split_at = technical_indexes[technical_train_count]
    split_at = min(
        max(split_at, FORECAST_LEARNING_MIN_REPLAY_SAMPLES),
        len(examples) - FORECAST_LEARNING_MIN_REPLAY_SAMPLES,
    )
    return split_at, "TECHNICAL_CONTEXT_CHRONOLOGICAL_WARM_START"


def _example_has_technical_context(item: Mapping[str, Any]) -> bool:
    features = item.get("features")
    return bool(
        isinstance(features, Mapping)
        and any(
            _label(features.get(name)) != "MISSING"
            for name in FORECAST_LEARNING_TECHNICAL_FEATURES
        )
    )


def scored_row_feature_values(row: Mapping[str, Any]) -> dict[str, str]:
    return {
        name: _label(row.get(name))
        for name in FORECAST_LEARNING_FEATURES
    }


def forecast_feature_values(
    *,
    pair: str,
    direction: str,
    confidence: float | None,
    raw_confidence: float | None,
    up_score: float | None,
    down_score: float | None,
    range_score: float | None,
    technical_context: Mapping[str, Any] | None,
    timestamp_utc: datetime,
    drivers_for: Sequence[str] = (),
    drivers_against: Sequence[str] = (),
) -> dict[str, str]:
    normalized_direction = _label(direction)
    context = technical_context if isinstance(technical_context, Mapping) else {}
    receipt = context.get("regime_family_weighting")
    receipt = receipt if isinstance(receipt, Mapping) else {}
    identity = receipt.get("source_identity")
    identity = identity if isinstance(identity, Mapping) else {}
    aggregate = receipt.get("aggregate")
    aggregate = aggregate if isinstance(aggregate, Mapping) else {}
    family_direction = _label(aggregate.get("direction"))
    if family_direction in {"UP", "DOWN"}:
        family_alignment = (
            "ALIGNED" if family_direction == normalized_direction else "CONTRADICTED"
        )
    else:
        family_alignment = "NON_DIRECTIONAL" if family_direction != "MISSING" else "MISSING"
    structure_direction = _nested_label(context, "structure", "primary_direction")
    structure_alignment = (
        "ALIGNED"
        if structure_direction == normalized_direction
        else "OPPOSED"
        if structure_direction in {"UP", "DOWN"}
        else "MISSING"
    )
    margin = _score_margin(
        normalized_direction,
        up_score,
        down_score,
        range_score,
    )
    values = {
        "pair": _label(pair),
        "direction": normalized_direction,
        "confidence_bucket": _confidence_bucket(confidence),
        "raw_confidence_bucket": _confidence_bucket(raw_confidence),
        "score_margin_bucket": _score_margin_bucket(margin),
        "range_competition": _range_competition(up_score, down_score, range_score),
        "utc_session_bucket": _utc_session_bucket(timestamp_utc),
        "primary_driver_family": _driver_family(drivers_for[0] if drivers_for else ""),
        "primary_against_driver_family": _driver_family(
            drivers_against[0] if drivers_against else ""
        ),
        "technical_regime": _nested_label(context, "regime", "primary"),
        "technical_atr_band": _nested_label(
            context, "volatility", "primary_atr_band"
        ),
        "technical_spread_band": _nested_label(context, "execution", "spread_band"),
        "technical_range_location_24h": _nested_label(
            context, "location", "range_location_24h"
        ),
        "technical_structure_alignment": structure_alignment,
        "technical_situation": _label(identity.get("situation")),
        "technical_selected_method": _label(identity.get("selected_method") or "NONE"),
        "technical_family_direction_alignment": family_alignment,
    }
    return {name: values[name] for name in FORECAST_LEARNING_FEATURES}


def forecast_orientation_decision(
    model_payload: Mapping[str, Any],
    *,
    original_direction: str,
    features: Mapping[str, Any],
) -> dict[str, Any]:
    direction = _label(original_direction)
    if direction not in {"UP", "DOWN"} or not verify_forecast_orientation_model(
        model_payload
    ):
        return {
            "direction": direction,
            "orientation": "DIRECT",
            "direct_probability": 0.5,
            "selected_probability": 0.5,
            "ordinary_correction_applied": False,
            "model_status": "UNAVAILABLE",
        }
    probability = _predict_direct_probability(
        model_payload.get("model") or {},
        {name: _label(features.get(name)) for name in FORECAST_LEARNING_FEATURES},
    )
    orientation = "DIRECT" if probability >= 0.5 else "INVERSE"
    selected_direction = direction if orientation == "DIRECT" else _opposite(direction)
    enabled = model_payload.get("ordinary_forecast_correction_enabled") is True
    applied = enabled and orientation == "INVERSE"
    return {
        "direction": selected_direction if enabled else direction,
        "rank_direction": selected_direction,
        "orientation": orientation,
        "direct_probability": round(probability, 6),
        "selected_probability": round(max(probability, 1.0 - probability), 6),
        "ordinary_correction_applied": applied,
        "model_status": str(model_payload.get("status") or "UNAVAILABLE"),
        "model_sha256": str(model_payload.get("model_sha256") or ""),
    }


def load_forecast_orientation_model(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) and verify_forecast_orientation_model(payload) else {}


def verify_forecast_orientation_model(payload: Mapping[str, Any]) -> bool:
    if payload.get("schema") != FORECAST_LEARNING_MODEL_SCHEMA:
        return False
    digest = payload.get("model_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        return False
    body = {key: value for key, value in payload.items() if key != "model_sha256"}
    if _digest(body) != digest:
        return False
    if payload.get("status") not in {"ENABLED", "RANK_ONLY"}:
        return False
    if payload.get("forward_learning_rank_enabled") is not True:
        return False
    if payload.get("ordinary_forecast_correction_enabled") is not (
        payload.get("status") == "ENABLED"
    ):
        return False
    model = payload.get("model")
    if not isinstance(model, Mapping):
        return False
    if _positive_int(model.get("direct_count")) is None or _positive_int(
        model.get("inverse_count")
    ) is None:
        return False
    if _finite_float(model.get("intercept")) is None or _finite_float(
        model.get("logit_scale")
    ) is None:
        return False
    weights = model.get("weights")
    return bool(
        isinstance(weights, Mapping)
        and all(
            isinstance(key, str) and _finite_float(value) is not None
            for key, value in weights.items()
        )
    )


def _fit_naive_bayes(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    direct_n = sum(bool(item["direct_better"]) for item in examples)
    inverse_n = len(examples) - direct_n
    values_by_feature: dict[str, set[str]] = {
        name: set() for name in FORECAST_LEARNING_FEATURES
    }
    counts: dict[tuple[str, str, bool], int] = {}
    for item in examples:
        label = bool(item["direct_better"])
        features = item["features"]
        for name in FORECAST_LEARNING_FEATURES:
            value = _label(features.get(name))
            values_by_feature[name].add(value)
            key = (name, value, label)
            counts[key] = counts.get(key, 0) + 1
    weights: dict[str, float] = {}
    for name in FORECAST_LEARNING_FEATURES:
        cardinality = max(1, len(values_by_feature[name]))
        for value in sorted(values_by_feature[name]):
            direct_probability = (
                counts.get((name, value, True), 0) + 1.0
            ) / (direct_n + cardinality)
            inverse_probability = (
                counts.get((name, value, False), 0) + 1.0
            ) / (inverse_n + cardinality)
            weights[f"{name}={value}"] = round(
                math.log(direct_probability / inverse_probability),
                12,
            )
    intercept = math.log((direct_n + 1.0) / (inverse_n + 1.0))
    return {
        "direct_count": direct_n,
        "inverse_count": inverse_n,
        "intercept": round(intercept, 12),
        "logit_scale": 1.0,
        "weights": weights,
    }


def _calibration_scale(
    model: Mapping[str, Any],
    examples: Sequence[Mapping[str, Any]],
) -> float:
    if not examples:
        return 1.0
    logits = [
        _raw_logit(model, item["features"])
        for item in examples
    ]
    labels = [1.0 if item["direct_better"] else 0.0 for item in examples]
    scale = 1.0
    for _ in range(24):
        gradient = 0.0
        hessian = 0.0
        for logit, label in zip(logits, labels):
            probability = _sigmoid(scale * logit)
            gradient += (probability - label) * logit
            hessian += probability * (1.0 - probability) * logit * logit
        if hessian <= 1e-12:
            break
        updated = min(1.0, max(0.0, scale - gradient / hessian))
        if abs(updated - scale) <= 1e-9:
            scale = updated
            break
        scale = updated
    return round(scale, 12)


def _evaluate(
    model: Mapping[str, Any],
    examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selected: list[float] = []
    direct: list[float] = []
    inverse: list[float] = []
    correct_orientation = 0
    inverted = 0
    for item in examples:
        probability = _predict_direct_probability(model, item["features"])
        choose_direct = probability >= 0.5
        selected.append(
            float(item["direct_pips"] if choose_direct else item["inverse_pips"])
        )
        direct.append(float(item["direct_pips"]))
        inverse.append(float(item["inverse_pips"]))
        correct_orientation += choose_direct == bool(item["direct_better"])
        inverted += not choose_direct
    return {
        "n": len(examples),
        "direct_avg_final_pips": _mean(direct),
        "direct_hit_rate": _rate(value > 0.0 for value in direct),
        "inverse_avg_final_pips": _mean(inverse),
        "inverse_hit_rate": _rate(value > 0.0 for value in inverse),
        "selected_avg_final_pips": _mean(selected),
        "selected_total_final_pips": round(sum(selected), 6),
        "selected_hit_rate": _rate(value > 0.0 for value in selected),
        "orientation_accuracy": round(correct_orientation / len(examples), 6),
        "inversion_rate": round(inverted / len(examples), 6),
    }


def _feature_coverage(
    examples: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    denominator = len(examples)
    return {
        name: round(
            sum(
                _label(item.get("features", {}).get(name)) != "MISSING"
                for item in examples
            )
            / denominator,
            6,
        )
        if denominator
        else 0.0
        for name in FORECAST_LEARNING_FEATURES
    }


def _predict_direct_probability(
    model: Mapping[str, Any],
    features: Mapping[str, Any],
) -> float:
    scale = _finite_float(model.get("logit_scale"))
    return _sigmoid((scale if scale is not None else 1.0) * _raw_logit(model, features))


def _raw_logit(model: Mapping[str, Any], features: Mapping[str, Any]) -> float:
    intercept = _finite_float(model.get("intercept")) or 0.0
    weights = model.get("weights")
    if not isinstance(weights, Mapping):
        return intercept
    total = intercept
    for name in FORECAST_LEARNING_FEATURES:
        weight = _finite_float(weights.get(f"{name}={_label(features.get(name))}"))
        if weight is not None:
            total += weight
    return total


def _unavailable_model(status: str, source: str, **counts: int) -> dict[str, Any]:
    return {
        "schema": FORECAST_LEARNING_MODEL_SCHEMA,
        "status": status,
        "ordinary_forecast_correction_enabled": False,
        "forward_learning_rank_enabled": False,
        "source": source,
        **counts,
    }


def _row_key(row: Mapping[str, Any]) -> tuple[Any, ...] | None:
    index = row.get("source_index")
    timestamp = str(row.get("timestamp_utc") or "")
    pair = _label(row.get("pair"))
    if isinstance(index, bool) or not isinstance(index, int) or not timestamp or pair == "MISSING":
        return None
    return index, timestamp, pair


def _score_margin(
    direction: str,
    up_score: float | None,
    down_score: float | None,
    range_score: float | None,
) -> float | None:
    values = tuple(_finite_float(value) for value in (up_score, down_score, range_score))
    if any(value is None for value in values):
        return None
    up, down, range_value = values
    assert up is not None and down is not None and range_value is not None
    if direction == "UP":
        return up - max(down, range_value)
    if direction == "DOWN":
        return down - max(up, range_value)
    return None


def _confidence_bucket(value: float | None) -> str:
    parsed = _finite_float(value)
    if parsed is None:
        return "MISSING"
    if parsed < 0.50:
        return "<0.50"
    if parsed < 0.65:
        return "0.50-0.65"
    if parsed < 0.75:
        return "0.65-0.75"
    if parsed < 0.90:
        return "0.75-0.90"
    return ">=0.90"


def _score_margin_bucket(value: float | None) -> str:
    parsed = _finite_float(value)
    if parsed is None:
        return "MISSING"
    if parsed < 0.0:
        return "<0"
    if parsed < 5.0:
        return "0-5"
    if parsed < 10.0:
        return "5-10"
    if parsed < 20.0:
        return "10-20"
    return ">=20"


def _range_competition(
    up_score: float | None,
    down_score: float | None,
    range_score: float | None,
) -> str:
    values = tuple(_finite_float(value) for value in (up_score, down_score, range_score))
    if any(value is None for value in values):
        return "MISSING"
    up, down, range_value = values
    assert up is not None and down is not None and range_value is not None
    directional_leader = max(up, down)
    directional_margin = abs(up - down)
    if range_value >= directional_leader:
        return "RANGE_LEADS_OR_TIES"
    if range_value >= directional_margin:
        return "RANGE_COMPETES_WITH_DIRECTIONAL_MARGIN"
    return "DIRECTIONAL_MARGIN_DOMINATES"


def _utc_session_bucket(value: datetime) -> str:
    hour = value.astimezone(timezone.utc).hour
    if hour < 8:
        return "UTC_00_08"
    if hour < 13:
        return "UTC_08_13"
    if hour < 17:
        return "UTC_13_17"
    if hour < 22:
        return "UTC_17_22"
    return "UTC_22_24"


def _driver_family(value: str) -> str:
    lower = str(value or "").lower()
    if "wick-only" in lower and "trap fade" in lower:
        return "WICK_TRAP_FADE"
    if "reversal-from-extreme" in lower:
        return "EXTREME_REVERSAL"
    if "range breakout pending" in lower:
        return "RANGE_BREAKOUT_PENDING"
    if "range breakout confirmed" in lower:
        return "RANGE_BREAKOUT_CONFIRMED"
    if "divergence" in lower:
        return "DIVERGENCE"
    if ("equal-high" in lower or "equal-low" in lower) and "fade" in lower:
        return "LIQUIDITY_SWEEP_FADE"
    if "hvn" in lower or "price magnet" in lower:
        return "HVN_MAGNET"
    if "aroon" in lower or "momentum" in lower:
        return "MOMENTUM"
    if any(
        marker in lower
        for marker in (
            "inside bar",
            "morning star",
            "evening star",
            "shooting star",
            "white soldiers",
            "black crows",
            "engulf",
            "doji",
        )
    ):
        return "CANDLE_PATTERN"
    if "market location" in lower:
        return "MARKET_LOCATION"
    return "OTHER" if lower else "MISSING"


def _nested_label(value: Mapping[str, Any], *path: str) -> str:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping):
            return "MISSING"
        current = current.get(key)
    return _label(current)


def _label(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text or "MISSING"


def _opposite(direction: str) -> str:
    return "DOWN" if direction == "UP" else "UP"


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -700.0))
    return z / (1.0 + z)


def _parse_utc(value: Any) -> datetime | None:
    text = str(value or "")
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _positive_float(value: Any) -> float | None:
    parsed = _finite_float(value)
    return parsed if parsed is not None and parsed > 0.0 else None


def _positive_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _rate(values: Sequence[bool] | Any) -> float:
    material = list(values)
    return round(sum(bool(value) for value in material) / len(material), 6) if material else 0.0


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
