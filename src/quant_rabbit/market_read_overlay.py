from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from quant_rabbit.capture_economics import (
    EXACT_VEHICLE_ALLOCATION_SURFACE_CONTRACT,
    EXACT_VEHICLE_NET_EDGE_MIN_TRADES,
    EXECUTION_COST_FLOOR_CONTRACT,
    evaluate_exact_vehicle_net_edge,
    execution_cost_floor_from_surface,
    exact_vehicle_metrics_from_surface,
    read_exact_vehicle_allocation_surface,
)
from quant_rabbit.forecast_precision import hit_rate_wilson_lower
from quant_rabbit.guardian_margin_contract import (
    P1_MARGIN_WARNING_CONTRACT as GUARDIAN_P1_MARGIN_WARNING_CONTRACT,
    exact_p1_margin_warning_source_event,
)
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS, instrument_pip_factor
from quant_rabbit.market_read_contract import (
    market_read_contract_payload,
    market_read_missing_fields,
)
from quant_rabbit.predictive_scout import (
    predictive_scout_geometry_claimed,
    predictive_scout_metadata_supported,
)
from quant_rabbit.risk import (
    FORECAST_LIVE_PRECISION_MIN_SAMPLES,
    FORECAST_LIVE_PRECISION_MIN_WILSON_LOWER,
    RiskPolicy,
)
from quant_rabbit.strategy.forecast_technical_context import (
    CONFIDENCE_SEMANTICS,
    build_forecast_technical_context,
    forecast_pair_chart_row_sha256,
    forecast_technical_context_is_current_for_allocation,
    normalize_forecast_technical_context_evidence,
    unknown_forecast_technical_context_evidence,
)
from quant_rabbit.strategy.failed_break_evidence import failed_break_direction
from quant_rabbit.strategy.m15_recovery_contract import (
    FORECAST_CONTRACT as M15_RECOVERY_FORECAST_CONTRACT,
    LANE_CONTRACT as M15_RECOVERY_LANE_CONTRACT,
    RECOVERY_ORDER_TYPE as M15_RECOVERY_ORDER_TYPE,
    recovery_claimed as _m15_recovery_claimed,
    validate_forecast_binding as validate_m15_recovery_forecast_binding,
    validate_lane_binding as validate_m15_recovery_lane_binding,
)
from quant_rabbit.strategy.regime_family_weighting import (
    forecast_direction_consistency,
    verify_regime_family_weighting_receipt,
)


MARKET_READ_OVERLAY_SCHEMA_VERSION = 2
CODEX_MARKET_READ_AUTHOR = "CODEX_MARKET_READ"
DETERMINISTIC_BASELINE_AUTHOR = "DETERMINISTIC_BASELINE"
DEFAULT_OVERLAY_MAX_AGE_SECONDS = 15 * 60
CAPITAL_ALLOCATION_SIZE_MULTIPLES = (0.5, 0.75, 1.0)
CAPITAL_ALLOCATION_SIZE_RATIOS = {
    0.5: (1, 2),
    0.75: (3, 4),
    1.0: (1, 1),
}
CAPITAL_ALLOCATION_MIN_EXACT_TP_TRADES = 5
CAPITAL_ALLOCATION_MIN_EXACT_NET_TRADES = EXACT_VEHICLE_NET_EDGE_MIN_TRADES
# This sizing proof deliberately shares the live forecast precision sample
# floor.  Eligibility gates and the AI allocation board must never disagree on
# whether a calibration bucket is mature enough to authorize broker risk.
CAPITAL_ALLOCATION_FORECAST_MIN_SAMPLES = FORECAST_LIVE_PRECISION_MIN_SAMPLES
CAPITAL_ALLOCATION_KELLY_FRACTION = 0.25
CAPITAL_ALLOCATION_NUMERIC_CEILING_CONTRACT = (
    "LANE_ECONOMIC_WILSON_QUARTER_KELLY_V2"
)
M15_RECOVERY_MICRO_CONTRACT = "QR_M15_RECOVERY_MICRO_V1"
M15_RECOVERY_MICRO_MODE = "M15_RECOVERY_MICRO"
M15_RECOVERY_MICRO_MAX_UNITS = 999
M15_RECOVERY_CONTEXT_TIMEFRAMES = ("M15", "M30", "H1", "H4", "D")
M15_RECOVERY_ALLOCATION_MATERIAL_CONTRACT = (
    "QR_M15_RECOVERY_ALLOCATION_MATERIAL_V1"
)
M15_RECOVERY_ALLOCATION_CONTEXT_CONTRACT = (
    "QR_M15_RECOVERY_ALLOCATION_CONTEXT_V1"
)
M15_RECOVERY_ALLOCATION_CONTEXT_SCOPE = "SELECTED_LANE_M15_RECOVERY"
M15_RECOVERY_ALLOCATION_STATUS = "M15_RECOVERY_VERIFIED"
M15_RECOVERY_RISK_MARKER = "M15_RECOVERY_RISK_REVALIDATED"
M15_RECOVERY_EDGE_COLLECTION_BASIS = "M15_RECOVERY_EDGE_COLLECTION"
M15_RECOVERY_PROOF_COLLECTION_MODE = "TP_PROOF_COLLECTION_HARVEST"
M15_RECOVERY_TP_PROVEN_MODE = "TP_PROVEN_HARVEST"
M15_RECOVERY_PROOF_COLLECTION_MIN_TRADES = 5
M15_RECOVERY_TP_PROVEN_MIN_TRADES = 20

OVERLAY_ALLOWED_FIELDS = frozenset(
    {
        "schema_version",
        "author_kind",
        "model",
        "reasoning_effort",
        "authored_at_utc",
        "baseline_sha256",
        "evidence_packet_sha256",
        "baseline_disposition",
        "market_read_first",
        "market_read_review",
        "market_read_counterargument",
        "market_read_change_summary",
        "market_read_veto_reason",
        "capital_allocation",
    }
)
OVERLAY_REQUIRED_FIELDS = OVERLAY_ALLOWED_FIELDS
REVIEW_ALLOWED_FIELDS = frozenset(
    {
        "prior_prediction_ids",
        "what_failed",
        "adjustment",
        "no_change_reason",
    }
)
REVIEW_REQUIRED_FIELDS = REVIEW_ALLOWED_FIELDS
CAPITAL_ALLOCATION_ALLOWED_FIELDS = frozenset(
    {
        "decision",
        "lane_id",
        "size_multiple",
        "selected_units",
        "allocation_board_sha256",
        "rationale",
    }
)
CAPITAL_ALLOCATION_REQUIRED_FIELDS = CAPITAL_ALLOCATION_ALLOWED_FIELDS

MUTABLE_MARKET_READ_FIELDS = frozenset(
    {
        "market_read_first",
        "market_read_review",
        "market_read_counterargument",
        "market_read_change_summary",
        "market_read_disposition",
        "market_read_veto_reason",
        "market_read_vetoed_lane_ids",
        "decision_provenance",
    }
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
# Do not interpret the separator in ``1.0840-1.0860`` as a negative sign.
NUMBER_RE = re.compile(r"(?<![\d.])[-+]?\d+(?:\.\d+)?")
WATCHDOG_MATERIAL_CONTRACT = "QR_TRADER_WATCHDOG_HEALTH_STATE_V2"
GUARDIAN_ACTION_RECEIPT_MATERIAL_CONTRACT = (
    "QR_GUARDIAN_ACTION_RECEIPT_SCOPE_V1"
)
MAX_GUARDIAN_ACTION_RECEIPT_BYTES = 1_000_000
GUARDIAN_HIGH_URGENCY_ACTIONS = frozenset(
    {"REDUCE", "HARVEST", "CANCEL_PENDING"}
)
GUARDIAN_TECHNICAL_OBSERVATION_EVENT_TYPES = frozenset(
    {"TECHNICAL_STATE_CHANGE", "TECHNICAL_INPUT_STALE"}
)
GUARDIAN_GLOBAL_SAFETY_EVENT_TYPES = frozenset(
    {
        "MARGIN_PRESSURE",
        "UNKNOWN_ORDER",
        "CONTRACT_EMERGENCY_TRIGGER",
        "WAKE_PARSE_FAILURE",
        "UNEXPECTED_PROTECTION_MISSING",
    }
)
GUARDIAN_KNOWN_EVENT_TYPES = frozenset(
    {
        "FAILED_ACCEPTANCE",
        "ACCEPTANCE_BREAK",
        "SESSION_EXPANSION",
        "RANGE_RAIL_TOUCH",
        "THEME_CONFIRMATION",
        "SQUEEZE_RELEASE",
        "HARVEST_ZONE",
        "THESIS_INVALIDATION",
        "MARGIN_PRESSURE",
        "UNKNOWN_ORDER",
        "STALE_PENDING",
        "BROKER_SNAPSHOT_STALE",
        "SPREAD_ANOMALY",
        "UNEXPECTED_PROTECTION_MISSING",
        "CONTRACT_HARVEST_TRIGGER",
        "CONTRACT_ADD_TRIGGER",
        "CONTRACT_NO_ADD_TRIGGER",
        "CONTRACT_WOUNDED_TRIGGER",
        "CONTRACT_INVALIDATION_TRIGGER",
        "CONTRACT_EMERGENCY_TRIGGER",
        "CONTRACT_STALE",
        "WAKE_PARSE_FAILURE",
        "EVENT_SUPPRESSED_WITH_PRICE_CHANGE",
        "TRIGGER_CONTRACT_EMPTY_FOR_ACTIVE_PAIR",
        "WATCH_ONLY_NO_TRIGGER_CONTRACT",
        "TECHNICAL_STATE_CHANGE",
        "TECHNICAL_INPUT_STALE",
    }
)
WATCHDOG_VOLATILE_MESSAGE_CODES = frozenset({"QR_TRADER_RUN_STALE"})
FORECAST_REPLAY_SCORECARD_CONTRACT = "QR_FORECAST_REPLAY_SCORECARD_V3"
PROJECTION_CALIBRATION_EVIDENCE_CONTRACT = (
    "QR_INDEPENDENT_PROJECTION_CALIBRATION_EVIDENCE_V1"
)
MAX_PROJECTION_CALIBRATION_ROWS = 4
FORECAST_REPLAY_METRIC_FIELDS = (
    "n",
    "hit_rate",
    "hit_wilson95_lower",
    "hit_wilson95_upper",
    "avg_final_pips",
    "median_final_pips",
    "avg_mfe_pips",
    "avg_mae_pips",
    "target_touch_rate",
    "invalidation_touch_rate",
    "target_before_invalidation_rate",
    "avg_realized_r",
    "total_realized_r",
)


class MarketReadOverlayError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class MarketReadBaselineSummary:
    baseline_path: Path
    packet_path: Path
    baseline_sha256: str
    evidence_packet_sha256: str
    source_count: int


@dataclass(frozen=True)
class MarketReadApplySummary:
    output_path: Path
    baseline_sha256: str
    evidence_packet_sha256: str
    overlay_sha256: str
    action: str
    selected_lane_ids: tuple[str, ...]


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def market_read_sha256(market_read: Any) -> str:
    return canonical_json_sha256(market_read if isinstance(market_read, dict) else {})


def projection_calibration_evidence(
    path: Path | None,
    *,
    scopes: list[Mapping[str, Any]],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a bounded scorecard for current directional forecasts only."""

    # The enclosing GPT/market-read receipt owns wall-clock freshness. Validate
    # its clock here, but never infer selected-outcome recency from global file
    # mtime (an unrelated append could otherwise promote an old scope).
    current = _utc_now(now)
    normalized_scopes = _normalize_projection_calibration_scopes(scopes)
    unchecked_parse_integrity = _projection_parse_integrity_evidence(
        status="NOT_CHECKED"
    )
    base: dict[str, Any] = {
        "contract": PROJECTION_CALIBRATION_EVIDENCE_CONTRACT,
        "evidence_ref": "projection:calibration",
        "statistics_source": "quant_rabbit.strategy.projection_ledger.compute_hit_rates",
        "freshness_basis": (
            "CURRENT_PACKET_POINT_IN_TIME_RECOMPUTATION_OF_STABLE_LEDGER"
        ),
        "selected_scope_outcome_recency_proven": False,
        "selection_contract": {
            "scope_source": "CURRENT_SELECTED_OR_FORCED_DIRECTIONAL_FORECAST_ONLY",
            "canonical_signal_only": True,
            "selection_uses_outcome": False,
            "independent_non_overlap_trials_required": True,
            "future_emissions_excluded_before_trial_selection": True,
            "resolved_at_must_be_observable_by_packet_clock": True,
            "full_resolution_window_must_be_observable_by_packet_clock": True,
            "raw_ledger_rows_exposed": False,
            "max_rows": MAX_PROJECTION_CALIBRATION_ROWS,
        },
        "thresholds": {
            "precision_floor_min_samples": FORECAST_LIVE_PRECISION_MIN_SAMPLES,
            "precision_floor_min_wilson95_lower": (
                FORECAST_LIVE_PRECISION_MIN_WILSON_LOWER
            ),
        },
        "usage_policy": {
            "read_only": True,
            "live_permission": False,
            "may_grant_gateway_permission": False,
            "may_upgrade_non_trade_baseline_to_trade": False,
            "may_veto_dampen_or_rank_existing_live_ready_only": True,
            "insufficient_sample_edge_claim_forbidden": True,
            "fallback_bucket_must_not_be_relabelled_as_pair_regime_edge": True,
            "metadata_hit_rate_claims_are_not_this_evidence": True,
        },
        "requested_scopes": normalized_scopes,
        "omitted_scope_count": max(0, len(scopes) - len(normalized_scopes)),
        "parse_integrity": unchecked_parse_integrity,
        "rows": [],
    }
    if path is None:
        return _address_projection_calibration_evidence(
            {
                **base,
                "status": "MISSING",
                "freshness_status": "MISSING",
                "parse_integrity": _projection_parse_integrity_evidence(
                    status="MISSING"
                ),
            }
        )
    try:
        before = path.stat()
    except OSError:
        return _address_projection_calibration_evidence(
            {
                **base,
                "status": "MISSING",
                "freshness_status": "MISSING",
                "parse_integrity": _projection_parse_integrity_evidence(
                    status="MISSING"
                ),
            }
        )

    if not normalized_scopes:
        return _address_projection_calibration_evidence(
            {
                **base,
                "status": "NO_SELECTED_SCOPE",
                "freshness_status": "NOT_COMPUTED",
            }
        )

    try:
        from quant_rabbit.strategy.projection_ledger import (
            CONFIDENCE_MIN_SAMPLES,
            GLOBAL_CONFIDENCE_MIN_SAMPLES,
            LEDGER_FILENAME,
            compute_hit_rates,
        )

        if path.name != LEDGER_FILENAME:
            raise ValueError(
                f"projection calibration source must be named {LEDGER_FILENAME}"
            )
        hit_rates_result = compute_hit_rates(
            path.parent,
            as_of=current,
            include_parse_integrity=True,
        )
        if not isinstance(hit_rates_result, tuple):
            raise TypeError("projection calibration diagnostics are unavailable")
        hit_rates, raw_parse_integrity = hit_rates_result
        after = path.stat()
    except (OSError, ValueError, TypeError) as exc:
        return _address_projection_calibration_evidence(
            {
                **base,
                "status": "UNAVAILABLE",
                "freshness_status": "FAILED",
                "parse_integrity": _projection_parse_integrity_evidence(
                    status="UNAVAILABLE"
                ),
                "error": f"{type(exc).__name__}: {str(exc)[:160]}",
            }
        )
    parse_integrity = _projection_parse_integrity_evidence(
        status=str(raw_parse_integrity.get("status") or "UNAVAILABLE"),
        malformed_json_rows=raw_parse_integrity.get("malformed_json_rows"),
        non_object_rows=raw_parse_integrity.get("non_object_rows"),
        unloadable_object_rows=raw_parse_integrity.get("unloadable_object_rows"),
    )
    snapshot_token = raw_parse_integrity.get("_source_snapshot_token")
    source_changed = (
        _projection_source_stat_token(before) != snapshot_token
        or _projection_source_stat_token(after) != snapshot_token
        or parse_integrity["status"] == "SOURCE_CHANGED_DURING_READ"
    )
    if source_changed:
        return _address_projection_calibration_evidence(
            {
                **base,
                "status": "SOURCE_CHANGED_DURING_READ",
                "freshness_status": "UNKNOWN",
                "parse_integrity": {
                    **parse_integrity,
                    "status": "SOURCE_CHANGED_DURING_READ",
                },
            }
        )
    if parse_integrity["status"] != "VALID":
        evidence_status = (
            "MALFORMED"
            if parse_integrity["status"] == "INVALID"
            else "UNAVAILABLE"
        )
        return _address_projection_calibration_evidence(
            {
                **base,
                "status": evidence_status,
                "freshness_status": "FAILED",
                "parse_integrity": parse_integrity,
            }
        )

    rows: list[dict[str, Any]] = []
    for scope in normalized_scopes:
        pair = str(scope["pair"])
        direction = str(scope["direction"])
        regime = scope.get("regime")
        rows.append(
            _projection_calibration_row(
                hit_rates=hit_rates,
                pair=pair,
                direction=direction,
                regime=str(regime) if regime is not None else None,
                pair_min_samples=int(CONFIDENCE_MIN_SAMPLES),
                global_min_samples=int(GLOBAL_CONFIDENCE_MIN_SAMPLES),
            )
        )
    return _address_projection_calibration_evidence(
        {
            **base,
            "status": "VALID",
            "freshness_status": "CURRENT_RECOMPUTATION",
            "parse_integrity": parse_integrity,
            "rows": rows,
        }
    )


def _address_projection_calibration_evidence(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    material = dict(payload)
    material.pop("evidence_sha256", None)
    return {**material, "evidence_sha256": canonical_json_sha256(material)}


def _projection_parse_integrity_evidence(
    *,
    status: str,
    malformed_json_rows: Any = 0,
    non_object_rows: Any = 0,
    unloadable_object_rows: Any = 0,
) -> dict[str, Any]:
    """Return only semantic parse failures, never valid-row/file identities."""

    def count(value: Any) -> int:
        return (
            value
            if isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
            else 0
        )

    malformed = count(malformed_json_rows)
    non_object = count(non_object_rows)
    unloadable = count(unloadable_object_rows)
    return {
        "status": str(status or "UNAVAILABLE"),
        "invalid_nonblank_rows": malformed + non_object + unloadable,
        "malformed_json_rows": malformed,
        "non_object_rows": non_object,
        "unloadable_object_rows": unloadable,
    }


def _projection_source_stat_token(
    stat: os.stat_result,
) -> tuple[int, int, int, int, int]:
    """Match the projection parser's private open-file coherence token."""

    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _normalize_projection_calibration_scopes(
    scopes: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str | None]] = set()
    for raw_scope in scopes:
        if not isinstance(raw_scope, Mapping):
            continue
        pair = str(raw_scope.get("pair") or "").strip().upper()
        direction = _forecast_direction_label(raw_scope.get("direction"))
        if pair not in DEFAULT_TRADER_PAIRS or direction not in {"UP", "DOWN", "RANGE"}:
            continue
        regime = _projection_calibration_regime(raw_scope.get("regime"))
        key = (pair, direction, regime)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            {
                "pair": pair,
                "direction": direction,
                "regime": regime,
                "calibration_name": f"directional_forecast_{direction.lower()}",
            }
        )
        if len(normalized) >= MAX_PROJECTION_CALIBRATION_ROWS:
            break
    return normalized


def _projection_calibration_regime(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    if not text or text == "UNKNOWN":
        return None
    if "RANGE" in text:
        return "RANGE"
    if "TREND" in text:
        return "TREND"
    if text in {"UNCLEAR", "TRANSITION", "BREAKOUT_PENDING"}:
        return "UNCLEAR"
    return text[:20]


def _projection_calibration_row(
    *,
    hit_rates: Mapping[str, Any],
    pair: str,
    direction: str,
    regime: str | None,
    pair_min_samples: int,
    global_min_samples: int,
) -> dict[str, Any]:
    calibration_name = f"directional_forecast_{direction.lower()}"
    by_bucket = hit_rates.get(calibration_name)
    by_bucket = by_bucket if isinstance(by_bucket, Mapping) else {}
    specific_key = f"{pair}:{regime}" if regime is not None else f"{pair}:_all_regimes"
    keys = [specific_key, f"{pair}:_all_regimes"]
    if regime is not None:
        keys.append(f"_all_pairs:{regime}")
    keys.append("_all_pairs:_all_regimes")
    keys = list(dict.fromkeys(keys))
    selected: tuple[str, Mapping[str, Any]] | None = None
    for key in keys:
        candidate = by_bucket.get(key)
        if not isinstance(candidate, Mapping):
            continue
        minimum = global_min_samples if key.startswith("_all_pairs") else pair_min_samples
        effective_samples = _projection_effective_samples(candidate)
        if effective_samples >= minimum:
            selected = (key, candidate)
            break
    selected_payload = (
        _projection_calibration_bucket_payload(
            selected[0],
            selected[1],
            pair_min_samples=pair_min_samples,
            global_min_samples=global_min_samples,
        )
        if selected is not None
        else None
    )
    specific_payload = (
        _projection_calibration_bucket_payload(
            specific_key,
            by_bucket[specific_key],
            pair_min_samples=pair_min_samples,
            global_min_samples=global_min_samples,
        )
        if isinstance(by_bucket.get(specific_key), Mapping)
        else None
    )
    precision_floor_eligible = bool(
        selected_payload is not None
        and selected_payload["samples"] >= FORECAST_LIVE_PRECISION_MIN_SAMPLES
        and selected_payload["economic_samples"]
        >= FORECAST_LIVE_PRECISION_MIN_SAMPLES
        and selected_payload["hit_rate_wilson95_lower"] is not None
        and selected_payload["hit_rate_wilson95_lower"]
        >= FORECAST_LIVE_PRECISION_MIN_WILSON_LOWER
        and selected_payload["economic_hit_rate_wilson95_lower"] is not None
        and selected_payload["economic_hit_rate_wilson95_lower"]
        >= FORECAST_LIVE_PRECISION_MIN_WILSON_LOWER
    )
    exact_pair_regime_precision_context = bool(
        regime is not None
        and precision_floor_eligible
        and selected_payload is not None
        and selected_payload["bucket"] == specific_key
    )
    if selected_payload is None:
        edge_status = "INSUFFICIENT_SAMPLES_NO_EDGE"
    elif not precision_floor_eligible:
        edge_status = "PRECISION_FLOOR_FAILED_NO_EDGE"
    elif regime is None and selected_payload["bucket"] == specific_key:
        edge_status = "PAIR_ALL_REGIMES_CONTEXT_ONLY_NO_PAIR_REGIME_EDGE"
    elif not exact_pair_regime_precision_context:
        edge_status = "FALLBACK_CONTEXT_ONLY_NO_PAIR_REGIME_EDGE"
    else:
        edge_status = "HISTORICAL_PAIR_REGIME_PRECISION_CONTEXT_ONLY_NO_EDGE"
    evidence_ref = (
        f"projection:calibration:{pair}:{regime or '_all_regimes'}:"
        f"{calibration_name}"
    )
    return {
        "evidence_ref": evidence_ref,
        "pair": pair,
        "direction": direction,
        "regime": regime,
        "calibration_name": calibration_name,
        "specific_bucket_key": specific_key,
        "specific_scope": "PAIR_REGIME" if regime is not None else "PAIR_ALL_REGIMES",
        "specific_bucket": specific_payload,
        "selected_bucket": selected_payload,
        "fallback_used": bool(
            selected_payload is not None
            and selected_payload["bucket"] != specific_key
        ),
        "edge_status": edge_status,
        "precision_floor_eligible": precision_floor_eligible,
        "exact_pair_regime_precision_context": (
            exact_pair_regime_precision_context
        ),
        "exact_pair_regime_edge": False,
        "insufficient_sample_edge_claim_forbidden": True,
        "decision_effect": "VETO_DAMPEN_OR_RANK_EXISTING_LIVE_READY_ONLY",
        "live_permission": False,
    }


def _projection_effective_samples(bucket: Mapping[str, Any]) -> int:
    value = bucket.get("calibration_samples", bucket.get("samples", 0))
    parsed = _optional_int(value)
    return max(0, parsed or 0)


def _projection_calibration_bucket_payload(
    key: str,
    bucket: Mapping[str, Any],
    *,
    pair_min_samples: int,
    global_min_samples: int,
) -> dict[str, Any]:
    samples = max(0, _optional_int(bucket.get("samples")) or 0)
    calibration_samples = max(
        0,
        _optional_int(bucket.get("calibration_samples")) or samples,
    )
    economic_samples = max(
        0,
        _optional_int(bucket.get("economic_samples")) or calibration_samples,
    )
    hit_rate = _optional_float(bucket.get("hit_rate"))
    economic_hit_rate = _optional_float(bucket.get("economic_hit_rate"))
    timeout_rate = _optional_float(
        bucket.get("timeout_rate", bucket.get("target_timeout_rate"))
    )
    timeout_count = _optional_int(
        bucket.get("timeout_count", bucket.get("target_timeout_count"))
    )
    invalidation_rate = _optional_float(bucket.get("invalidation_first_rate"))
    invalidation_count = _optional_int(bucket.get("invalidation_first_count"))
    minimum = global_min_samples if key.startswith("_all_pairs") else pair_min_samples
    return {
        "bucket": key,
        "scope": _projection_bucket_scope(key),
        "samples": samples,
        "calibration_samples": calibration_samples,
        "hit_rate": hit_rate,
        "hit_rate_wilson95_lower": hit_rate_wilson_lower(hit_rate, samples),
        "economic_samples": economic_samples,
        "economic_hit_rate": economic_hit_rate,
        "economic_hit_rate_wilson95_lower": hit_rate_wilson_lower(
            economic_hit_rate,
            economic_samples,
        ),
        "timeout_count": max(0, timeout_count or 0),
        "timeout_rate": timeout_rate,
        "invalidation_first_count": max(0, invalidation_count or 0),
        "invalidation_first_rate": invalidation_rate,
        "confidence_min_samples": minimum,
        "confidence_sample_sufficient": calibration_samples >= minimum,
    }


def _projection_bucket_scope(key: str) -> str:
    if key == "_all_pairs:_all_regimes" or key == "_all_pairs":
        return "ALL_PAIRS_ALL_REGIMES"
    if key.startswith("_all_pairs:"):
        return "ALL_PAIRS_REGIME"
    if key.endswith(":_all_regimes") or ":" not in key:
        return "PAIR_ALL_REGIMES"
    return "PAIR_REGIME"


def baseline_core_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    core = dict(payload)
    for key in MUTABLE_MARKET_READ_FIELDS - {"market_read_first"}:
        core.pop(key, None)
    return core


def execution_envelope_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in payload.items()
        if key not in MUTABLE_MARKET_READ_FIELDS
    }


def _immutable_risk_envelope(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fields an AI overlay cannot change.

    The merge tool itself may downgrade TRADE to WAIT/REQUEST_EVIDENCE and
    clear the selected lane so the non-trade receipt cannot reach a gateway. It
    may also add a separately validated capital-allocation authorization that
    can only reduce the deterministic lane's units. Everything else, including
    units embedded in cited intents, risk notes, cancel/close ids, and evidence
    refs, remains byte-equivalent to baseline.
    """

    return {
        key: deepcopy(value)
        for key, value in payload.items()
        if key
        not in {
            "action",
            "selected_lane_id",
            "selected_lane_ids",
            "capital_allocation",
        }
    }


def prepare_market_read_baseline(
    *,
    baseline_path: Path,
    packet_path: Path,
    evidence_sources: Mapping[str, Path],
    now: datetime | None = None,
) -> MarketReadBaselineSummary:
    baseline = _load_json_object(baseline_path, label="baseline receipt")
    prepared_at = _utc_now(now)
    baseline_sha = canonical_json_sha256(baseline_core_payload(baseline))
    packet = _build_evidence_packet(
        baseline=baseline,
        baseline_sha256=baseline_sha,
        evidence_sources=evidence_sources,
        prepared_at=prepared_at,
        use_runtime_guardian_clock=now is None,
    )
    packet_sha = str(packet["evidence_packet_sha256"])
    stamped = dict(baseline)
    stamped["decision_provenance"] = {
        "schema_version": MARKET_READ_OVERLAY_SCHEMA_VERSION,
        "author_kind": DETERMINISTIC_BASELINE_AUTHOR,
        "generated_by": "trader-draft-decision",
        "prepared_at_utc": prepared_at.isoformat(),
        "baseline_sha256": baseline_sha,
        "evidence_packet_sha256": packet_sha,
        "market_read_sha256": market_read_sha256(stamped.get("market_read_first")),
        "execution_envelope_sha256": canonical_json_sha256(
            execution_envelope_payload(stamped)
        ),
        "execution_fields_preserved": True,
        "live_permission_granted": False,
    }
    _atomic_write_json(packet_path, packet)
    _atomic_write_json(baseline_path, stamped)
    return MarketReadBaselineSummary(
        baseline_path=baseline_path,
        packet_path=packet_path,
        baseline_sha256=baseline_sha,
        evidence_packet_sha256=packet_sha,
        source_count=len(evidence_sources),
    )


def apply_codex_market_read_overlay(
    *,
    baseline_path: Path,
    packet_path: Path,
    overlay_path: Path,
    output_path: Path,
    evidence_sources: Mapping[str, Path],
    now: datetime | None = None,
    max_overlay_age_seconds: int = DEFAULT_OVERLAY_MAX_AGE_SECONDS,
) -> MarketReadApplySummary:
    current_time = _utc_now(now)
    baseline = _load_json_object(baseline_path, label="baseline receipt")
    packet = _load_json_object(packet_path, label="market-read evidence packet")
    overlay = _load_json_object(overlay_path, label="Codex market-read overlay")

    _validate_exact_keys(
        overlay,
        allowed=OVERLAY_ALLOWED_FIELDS,
        required=OVERLAY_REQUIRED_FIELDS,
        code="MARKET_READ_OVERLAY_SCHEMA_INVALID",
        label="overlay",
    )
    if overlay.get("schema_version") != MARKET_READ_OVERLAY_SCHEMA_VERSION:
        raise MarketReadOverlayError(
            "MARKET_READ_OVERLAY_SCHEMA_INVALID",
            f"schema_version must be {MARKET_READ_OVERLAY_SCHEMA_VERSION}",
        )
    if overlay.get("author_kind") != CODEX_MARKET_READ_AUTHOR:
        raise MarketReadOverlayError(
            "MARKET_READ_OVERLAY_AUTHOR_INVALID",
            f"author_kind must be {CODEX_MARKET_READ_AUTHOR}",
        )
    if str(overlay.get("model") or "").strip() != "gpt-5.5":
        raise MarketReadOverlayError(
            "MARKET_READ_OVERLAY_MODEL_INVALID",
            "model must be gpt-5.5",
        )
    if str(overlay.get("reasoning_effort") or "").strip().lower() != "high":
        raise MarketReadOverlayError(
            "MARKET_READ_OVERLAY_REASONING_INVALID",
            "reasoning_effort must be high",
        )
    authored_at = _parse_utc(
        overlay.get("authored_at_utc"),
        code="MARKET_READ_OVERLAY_TIMESTAMP_INVALID",
    )
    age_seconds = (current_time - authored_at).total_seconds()
    if age_seconds < -60 or age_seconds > max_overlay_age_seconds:
        raise MarketReadOverlayError(
            "MARKET_READ_OVERLAY_STALE",
            f"overlay age {age_seconds:.1f}s is outside -60..{max_overlay_age_seconds}s",
        )

    baseline_sha = canonical_json_sha256(baseline_core_payload(baseline))
    _require_sha_match(
        actual=baseline_sha,
        claimed=overlay.get("baseline_sha256"),
        code="MARKET_READ_BASELINE_SHA_STALE",
        label="baseline",
    )
    provenance = baseline.get("decision_provenance")
    if not isinstance(provenance, dict) or provenance.get("author_kind") != DETERMINISTIC_BASELINE_AUTHOR:
        raise MarketReadOverlayError(
            "MARKET_READ_BASELINE_PROVENANCE_INVALID",
            "baseline must be prepared by the deterministic baseline command",
        )
    _require_sha_match(
        actual=baseline_sha,
        claimed=provenance.get("baseline_sha256"),
        code="MARKET_READ_BASELINE_SHA_STALE",
        label="stamped baseline",
    )
    _require_sha_match(
        actual=canonical_json_sha256(execution_envelope_payload(baseline)),
        claimed=provenance.get("execution_envelope_sha256"),
        code="MARKET_READ_BASELINE_EXECUTION_SHA_STALE",
        label="baseline execution envelope",
    )
    if (
        provenance.get("execution_fields_preserved") is not True
        or provenance.get("live_permission_granted") is not False
    ):
        raise MarketReadOverlayError(
            "MARKET_READ_BASELINE_PROVENANCE_INVALID",
            "baseline provenance must preserve execution fields and grant no live permission",
        )

    rebuilt = _build_evidence_packet(
        baseline=baseline,
        baseline_sha256=baseline_sha,
        evidence_sources=evidence_sources,
        prepared_at=current_time,
        use_runtime_guardian_clock=now is None,
    )
    stored_packet_sha = str(packet.get("evidence_packet_sha256") or "")
    rebuilt_packet_sha = str(rebuilt["evidence_packet_sha256"])
    if rebuilt_packet_sha != stored_packet_sha:
        changed_material = _changed_evidence_packet_material(packet, rebuilt)
        raise MarketReadOverlayError(
            "MARKET_READ_EVIDENCE_PACKET_STALE",
            "stored evidence packet no longer matches current evidence; "
            f"changed_material={','.join(changed_material) or 'UNKNOWN'}; "
            f"expected={rebuilt_packet_sha}; claimed={stored_packet_sha}",
        )
    _require_sha_match(
        actual=rebuilt_packet_sha,
        claimed=overlay.get("evidence_packet_sha256"),
        code="MARKET_READ_EVIDENCE_PACKET_STALE",
        label="overlay evidence packet",
    )
    _require_sha_match(
        actual=rebuilt_packet_sha,
        claimed=provenance.get("evidence_packet_sha256"),
        code="MARKET_READ_EVIDENCE_PACKET_STALE",
        label="baseline provenance evidence packet",
    )
    stored_board = packet.get("capital_allocation_board")
    if not isinstance(stored_board, Mapping):
        raise MarketReadOverlayError(
            "MARKET_READ_CAPITAL_ALLOCATION_BOARD_INVALID",
            "stored evidence packet lacks a capital-allocation board object",
        )
    stored_board_sha = canonical_json_sha256(dict(stored_board))
    _require_sha_match(
        actual=stored_board_sha,
        claimed=packet.get("capital_allocation_board_sha256"),
        code="MARKET_READ_CAPITAL_ALLOCATION_BOARD_STALE",
        label="stored capital-allocation board body",
    )
    _require_sha_match(
        actual=rebuilt.get("capital_allocation_board_sha256"),
        claimed=stored_board_sha,
        code="MARKET_READ_CAPITAL_ALLOCATION_BOARD_STALE",
        label="rebuilt capital-allocation board",
    )
    stored_projection_calibration = packet.get(
        "projection_calibration_evidence"
    )
    if not isinstance(stored_projection_calibration, Mapping):
        raise MarketReadOverlayError(
            "MARKET_READ_PROJECTION_CALIBRATION_INVALID",
            "stored packet lacks bounded projection calibration evidence",
        )
    stored_projection_material = dict(stored_projection_calibration)
    stored_projection_sha = str(
        stored_projection_material.pop("evidence_sha256", "") or ""
    )
    _require_sha_match(
        actual=canonical_json_sha256(stored_projection_material),
        claimed=stored_projection_sha,
        code="MARKET_READ_PROJECTION_CALIBRATION_INVALID",
        label="stored projection calibration evidence",
    )
    _require_sha_match(
        actual=str(rebuilt.get("projection_calibration_evidence_sha256") or ""),
        claimed=stored_projection_sha,
        code="MARKET_READ_PROJECTION_CALIBRATION_STALE",
        label="rebuilt projection calibration evidence",
    )
    stored_material = {
        "schema_version": packet.get("schema_version"),
        "baseline_sha256": packet.get("baseline_sha256"),
        "sources": packet.get("sources"),
        "capital_allocation_board_sha256": packet.get(
            "capital_allocation_board_sha256"
        ),
        "projection_calibration_evidence_sha256": packet.get(
            "projection_calibration_evidence_sha256"
        ),
    }
    _require_sha_match(
        actual=canonical_json_sha256(stored_material),
        claimed=stored_packet_sha,
        code="MARKET_READ_EVIDENCE_PACKET_STALE",
        label="stored evidence packet material",
    )
    stored_packet_body = dict(packet)
    rebuilt_packet_body = dict(rebuilt)
    # Publication time is intentionally refreshed during reconstruction; all
    # evidence shown to the model must otherwise be byte-equivalent in its
    # canonical JSON form, not merely accompanied by an unchanged digest claim.
    stored_packet_body.pop("generated_at_utc", None)
    rebuilt_packet_body.pop("generated_at_utc", None)
    # Only sources with an explicit semantic-material contract may ignore their
    # raw descriptor churn. Every other source descriptor remains body-bound so
    # packet metadata cannot be forged after the model reviewed it.
    for body in (stored_packet_body, rebuilt_packet_body):
        raw_metadata = body.get("source_metadata")
        if not isinstance(raw_metadata, Mapping):
            continue
        stable_metadata = dict(raw_metadata)
        stable_metadata.pop("qr_trader_run_watchdog", None)
        stable_metadata.pop("guardian_action_receipt", None)
        stable_metadata.pop("market_read_predictions", None)
        stable_metadata.pop("projection_ledger", None)
        body["source_metadata"] = stable_metadata
    if canonical_json_sha256(stored_packet_body) != canonical_json_sha256(
        rebuilt_packet_body
    ):
        raise MarketReadOverlayError(
            "MARKET_READ_EVIDENCE_PACKET_BODY_STALE",
            "stored evidence packet body no longer matches current evidence reconstruction",
        )

    review = overlay.get("market_read_review")
    if not isinstance(review, dict):
        raise MarketReadOverlayError(
            "MARKET_READ_REVIEW_INVALID",
            "market_read_review must be an object",
        )
    _validate_exact_keys(
        review,
        allowed=REVIEW_ALLOWED_FIELDS,
        required=REVIEW_REQUIRED_FIELDS,
        code="MARKET_READ_REVIEW_INVALID",
        label="market_read_review",
    )
    prior_ids = review.get("prior_prediction_ids")
    if not isinstance(prior_ids, list) or any(not isinstance(item, str) or not item for item in prior_ids):
        raise MarketReadOverlayError(
            "MARKET_READ_REVIEW_INVALID",
            "prior_prediction_ids must be a list of non-empty strings",
        )
    latest_prior_id = _latest_resolved_prediction_id(rebuilt)
    if latest_prior_id and latest_prior_id not in prior_ids:
        raise MarketReadOverlayError(
            "MARKET_READ_PRIOR_PREDICTION_NOT_REVIEWED",
            f"latest resolved prediction {latest_prior_id} was not cited",
        )
    what_failed = _required_text(review, "what_failed", "MARKET_READ_REVIEW_INVALID")
    adjustment = _required_text(review, "adjustment", "MARKET_READ_REVIEW_INVALID")
    no_change_reason = str(review.get("no_change_reason") or "").strip()
    if adjustment.upper() in {"NO_CHANGE", "UNCHANGED"} and not no_change_reason:
        raise MarketReadOverlayError(
            "MARKET_READ_REVIEW_INVALID",
            "no_change_reason is required when adjustment is NO_CHANGE",
        )
    if not latest_prior_id and what_failed.upper() not in {
        "NO_RESOLVED_PRIOR",
        "NO_RESOLVED_PRIOR_PREDICTION",
    }:
        raise MarketReadOverlayError(
            "MARKET_READ_REVIEW_INVALID",
            "what_failed must state NO_RESOLVED_PRIOR when no resolved prior prediction exists",
        )

    market_read = overlay.get("market_read_first")
    if not isinstance(market_read, dict):
        raise MarketReadOverlayError(
            "MARKET_READ_OVERLAY_SCHEMA_INVALID",
            "market_read_first must be an object",
        )
    semantic_issues = market_read_missing_fields(market_read)
    if semantic_issues:
        raise MarketReadOverlayError(
            "MARKET_READ_OVERLAY_SCHEMA_INVALID",
            "market_read_first is missing or has noncanonical fields: "
            + ", ".join(semantic_issues),
        )
    geometry_issues = validate_market_read_numeric_geometry(
        market_read,
        quote_basis_by_pair=rebuilt.get("quote_basis_by_pair") or {},
    )
    if geometry_issues:
        code, message = geometry_issues[0]
        raise MarketReadOverlayError(code, message)

    counterargument = _required_text(
        overlay,
        "market_read_counterargument",
        "MARKET_READ_COUNTERARGUMENT_MISSING",
    )
    change_summary = _required_text(
        overlay,
        "market_read_change_summary",
        "MARKET_READ_CHANGE_SUMMARY_MISSING",
    )

    disposition = str(overlay.get("baseline_disposition") or "").strip().upper()
    if disposition not in {"ACCEPT_BASELINE", "VETO_WAIT", "VETO_REQUEST_EVIDENCE"}:
        raise MarketReadOverlayError(
            "MARKET_READ_DISPOSITION_INVALID",
            "baseline_disposition must be ACCEPT_BASELINE, VETO_WAIT, or VETO_REQUEST_EVIDENCE",
        )
    veto_reason = str(overlay.get("market_read_veto_reason") or "").strip()
    baseline_action = str(baseline.get("action") or "").strip().upper()
    if baseline_action != "TRADE" and disposition != "ACCEPT_BASELINE":
        raise MarketReadOverlayError(
            "MARKET_READ_NONTRADE_UPGRADE_FORBIDDEN",
            f"a {baseline_action or 'missing'} baseline may only use ACCEPT_BASELINE",
        )
    if baseline_action == "CLOSE":
        raw_close_trade_ids = baseline.get("close_trade_ids")
        if (
            not isinstance(raw_close_trade_ids, list)
            or len(raw_close_trade_ids) != 1
            or not isinstance(raw_close_trade_ids[0], str)
            or not raw_close_trade_ids[0]
        ):
            raise MarketReadOverlayError(
                "MARKET_READ_BASELINE_SINGLE_CLOSE_REQUIRED",
                "an accepted CLOSE baseline must bind exactly one non-empty close_trade_ids item",
            )
        raw_selected_lane_ids = baseline.get("selected_lane_ids")
        raw_cancel_order_ids = baseline.get("cancel_order_ids")
        if (
            baseline.get("selected_lane_id") is not None
            or raw_selected_lane_ids != []
            or raw_cancel_order_ids != []
        ):
            raise MarketReadOverlayError(
                "MARKET_READ_BASELINE_CLOSE_SCOPE_INVALID",
                "a CLOSE baseline must be close-only with no selected lane or pending-order cancellation",
            )
    if disposition.startswith("VETO_") and not veto_reason:
        raise MarketReadOverlayError(
            "MARKET_READ_VETO_REASON_MISSING",
            "market_read_veto_reason is required for an AI veto",
        )
    if disposition == "ACCEPT_BASELINE" and veto_reason:
        raise MarketReadOverlayError(
            "MARKET_READ_VETO_REASON_INVALID",
            "market_read_veto_reason must be empty when the baseline is accepted",
        )
    if baseline_action == "TRADE" and disposition == "ACCEPT_BASELINE":
        _validate_trade_source_freshness(
            baseline=baseline,
            market_read=market_read,
            evidence_sources=evidence_sources,
            now=current_time,
        )

    before_envelope = execution_envelope_payload(baseline)
    merged = deepcopy(baseline)
    merged["market_read_first"] = deepcopy(market_read)
    merged["market_read_review"] = deepcopy(review)
    merged["market_read_counterargument"] = counterargument
    merged["market_read_change_summary"] = change_summary
    merged["market_read_disposition"] = disposition
    merged["market_read_veto_reason"] = veto_reason
    raw_lane_ids = baseline.get("selected_lane_ids")
    if raw_lane_ids is not None and (
        not isinstance(raw_lane_ids, list)
        or any(not isinstance(item, str) or not item for item in raw_lane_ids)
    ):
        raise MarketReadOverlayError(
            "MARKET_READ_BASELINE_PROVENANCE_INVALID",
            "baseline selected_lane_ids must be a list of non-empty strings",
        )
    baseline_selected_lane_ids = list(raw_lane_ids or [])
    primary_lane_id = baseline.get("selected_lane_id")
    if primary_lane_id is not None and not isinstance(primary_lane_id, str):
        raise MarketReadOverlayError(
            "MARKET_READ_BASELINE_PROVENANCE_INVALID",
            "baseline selected_lane_id must be a string or null",
        )
    if baseline_action == "TRADE" and disposition == "ACCEPT_BASELINE" and (
        len(baseline_selected_lane_ids) != 1
        or not primary_lane_id
        or baseline_selected_lane_ids[0] != primary_lane_id
    ):
        raise MarketReadOverlayError(
            "MARKET_READ_BASELINE_SINGLE_LANE_REQUIRED",
            "an accepted TRADE baseline must name exactly one selected_lane_ids item equal to selected_lane_id",
        )
    if not baseline_selected_lane_ids and primary_lane_id:
        baseline_selected_lane_ids = [primary_lane_id]
    if primary_lane_id and primary_lane_id not in baseline_selected_lane_ids:
        raise MarketReadOverlayError(
            "MARKET_READ_BASELINE_PROVENANCE_INVALID",
            "baseline selected_lane_id must appear in selected_lane_ids",
        )
    capital_allocation = _validated_capital_allocation(
        overlay.get("capital_allocation"),
        allocation_board=rebuilt.get("capital_allocation_board"),
        allocation_board_sha256=rebuilt.get("capital_allocation_board_sha256"),
        baseline_action=baseline_action,
        disposition=disposition,
        primary_lane_id=primary_lane_id,
    )
    merged["capital_allocation"] = deepcopy(capital_allocation)
    merged["market_read_vetoed_lane_ids"] = (
        baseline_selected_lane_ids if disposition.startswith("VETO_") else []
    )
    if disposition == "VETO_WAIT":
        merged["action"] = "WAIT"
        merged["selected_lane_id"] = None
        merged["selected_lane_ids"] = []
    elif disposition == "VETO_REQUEST_EVIDENCE":
        merged["action"] = "REQUEST_EVIDENCE"
        merged["selected_lane_id"] = None
        merged["selected_lane_ids"] = []
    after_envelope = execution_envelope_payload(merged)
    baseline_immutable = _immutable_risk_envelope(before_envelope)
    merged_immutable = _immutable_risk_envelope(after_envelope)
    if baseline_immutable != merged_immutable:
        raise MarketReadOverlayError(
            "MARKET_READ_EXECUTION_ENVELOPE_CHANGED",
            "overlay changed units/risk/orders/permission or selected a different lane",
        )

    overlay_sha = canonical_json_sha256(overlay)
    baseline_envelope_sha = canonical_json_sha256(before_envelope)
    final_envelope_sha = canonical_json_sha256(after_envelope)
    rebuilt_selected_lane = (
        rebuilt.get("capital_allocation_board", {}).get("selected_lane")
        if isinstance(rebuilt.get("capital_allocation_board"), Mapping)
        else None
    )
    rebuilt_numeric_ceiling = (
        rebuilt_selected_lane.get("numeric_ceiling")
        if isinstance(rebuilt_selected_lane, Mapping)
        and isinstance(rebuilt_selected_lane.get("numeric_ceiling"), Mapping)
        else {}
    )
    rebuilt_cost_floor = (
        rebuilt_numeric_ceiling.get("execution_cost_floor")
        if isinstance(rebuilt_numeric_ceiling.get("execution_cost_floor"), Mapping)
        else {}
    )
    execution_cost_floor_sha256 = str(
        rebuilt_cost_floor.get("proof_sha256") or ""
    ) or None
    capital_allocation_edge_basis = (
        str(rebuilt_selected_lane.get("edge_basis") or "") or None
        if isinstance(rebuilt_selected_lane, Mapping)
        else None
    )
    guardian_action_receipt_material = rebuilt.get(
        "guardian_action_receipt_material"
    )
    guardian_action_receipt_baseline_pairs = (
        list(guardian_action_receipt_material.get("baseline_pairs") or [])
        if isinstance(guardian_action_receipt_material, Mapping)
        else []
    )
    guardian_action_receipt_scope_state_sha256 = canonical_json_sha256(
        guardian_action_receipt_material
    )
    merged["decision_provenance"] = {
        "schema_version": MARKET_READ_OVERLAY_SCHEMA_VERSION,
        "author_kind": CODEX_MARKET_READ_AUTHOR,
        "model": "gpt-5.5",
        "reasoning_effort": "high",
        "authored_at_utc": authored_at.isoformat(),
        "applied_at_utc": current_time.isoformat(),
        "baseline_sha256": baseline_sha,
        "evidence_packet_sha256": rebuilt_packet_sha,
        "overlay_sha256": overlay_sha,
        "market_read_sha256": market_read_sha256(market_read),
        "execution_envelope_sha256": final_envelope_sha,
        "baseline_execution_envelope_sha256": baseline_envelope_sha,
        "final_execution_envelope_sha256": final_envelope_sha,
        "baseline_action": baseline_action,
        "final_action": str(merged.get("action") or "").upper(),
        "baseline_selected_lane_ids": baseline_selected_lane_ids,
        "baseline_disposition": disposition,
        "action_downgrade_only": disposition.startswith("VETO_"),
        "capital_allocation_sha256": canonical_json_sha256(capital_allocation),
        "capital_allocation_board_sha256": str(
            capital_allocation["allocation_board_sha256"]
        ),
        "capital_allocation_edge_basis": capital_allocation_edge_basis,
        "execution_cost_floor_sha256": execution_cost_floor_sha256,
        "guardian_action_receipt_material_contract": (
            GUARDIAN_ACTION_RECEIPT_MATERIAL_CONTRACT
        ),
        "guardian_action_receipt_baseline_pairs": (
            guardian_action_receipt_baseline_pairs
        ),
        "guardian_action_receipt_scope_state_sha256": (
            guardian_action_receipt_scope_state_sha256
        ),
        "authorized_size_multiple": float(capital_allocation["size_multiple"]),
        "authorized_units": int(capital_allocation["selected_units"]),
        "execution_fields_preserved": True,
        "risk_envelope_not_expanded": True,
        "live_permission_granted": False,
    }
    _atomic_write_json(output_path, merged)
    lane_ids = merged.get("selected_lane_ids")
    if not isinstance(lane_ids, list):
        single = merged.get("selected_lane_id")
        lane_ids = [single] if isinstance(single, str) and single else []
    return MarketReadApplySummary(
        output_path=output_path,
        baseline_sha256=baseline_sha,
        evidence_packet_sha256=rebuilt_packet_sha,
        overlay_sha256=overlay_sha,
        action=str(merged.get("action") or ""),
        selected_lane_ids=tuple(str(item) for item in lane_ids if item),
    )


def _validated_capital_allocation(
    value: Any,
    *,
    allocation_board: Any,
    allocation_board_sha256: Any,
    baseline_action: str,
    disposition: str,
    primary_lane_id: str | None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MarketReadOverlayError(
            "MARKET_READ_CAPITAL_ALLOCATION_INVALID",
            "capital_allocation must be an object",
        )
    allocation = dict(value)
    _validate_exact_keys(
        allocation,
        allowed=CAPITAL_ALLOCATION_ALLOWED_FIELDS,
        required=CAPITAL_ALLOCATION_REQUIRED_FIELDS,
        code="MARKET_READ_CAPITAL_ALLOCATION_INVALID",
        label="capital_allocation",
    )
    raw_rationale = allocation.get("rationale")
    if not isinstance(raw_rationale, str) or not raw_rationale.strip():
        raise MarketReadOverlayError(
            "MARKET_READ_CAPITAL_ALLOCATION_INVALID",
            "capital_allocation.rationale must be a non-empty string",
        )
    board_sha = str(allocation_board_sha256 or "")
    if not SHA256_RE.fullmatch(board_sha):
        raise MarketReadOverlayError(
            "MARKET_READ_CAPITAL_ALLOCATION_BOARD_INVALID",
            "evidence packet lacks a valid capital-allocation board digest",
        )
    _require_sha_match(
        actual=board_sha,
        claimed=allocation.get("allocation_board_sha256"),
        code="MARKET_READ_CAPITAL_ALLOCATION_BOARD_STALE",
        label="capital-allocation board",
    )
    if not isinstance(allocation_board, Mapping):
        raise MarketReadOverlayError(
            "MARKET_READ_CAPITAL_ALLOCATION_BOARD_INVALID",
            "capital-allocation board must be an object",
        )
    _require_sha_match(
        actual=canonical_json_sha256(dict(allocation_board)),
        claimed=board_sha,
        code="MARKET_READ_CAPITAL_ALLOCATION_BOARD_STALE",
        label="capital-allocation board body",
    )

    raw_allocation_decision = allocation.get("decision")
    if not isinstance(raw_allocation_decision, str):
        raise MarketReadOverlayError(
            "MARKET_READ_CAPITAL_ALLOCATION_INVALID",
            "capital_allocation.decision must be a string",
        )
    allocation_decision = raw_allocation_decision.strip().upper()
    trade_allocation_required = (
        baseline_action == "TRADE" and disposition == "ACCEPT_BASELINE"
    )
    raw_multiple = allocation.get("size_multiple")
    raw_units = allocation.get("selected_units")
    if isinstance(raw_multiple, bool) or not isinstance(raw_multiple, (int, float)):
        raise MarketReadOverlayError(
            "MARKET_READ_CAPITAL_ALLOCATION_INVALID",
            "capital_allocation.size_multiple must be numeric",
        )
    size_multiple = float(raw_multiple)
    if not math.isfinite(size_multiple):
        raise MarketReadOverlayError(
            "MARKET_READ_CAPITAL_ALLOCATION_INVALID",
            "capital_allocation.size_multiple must be finite",
        )
    if isinstance(raw_units, bool) or not isinstance(raw_units, int):
        raise MarketReadOverlayError(
            "MARKET_READ_CAPITAL_ALLOCATION_INVALID",
            "capital_allocation.selected_units must be an integer",
        )

    if not trade_allocation_required:
        if (
            allocation_decision != "NO_TRADE"
            or allocation.get("lane_id") is not None
            or size_multiple != 0.0
            or raw_units != 0
        ):
            raise MarketReadOverlayError(
                "MARKET_READ_CAPITAL_ALLOCATION_NONTRADE_REQUIRED",
                "non-trade, CLOSE, and veto receipts require NO_TRADE with null lane, 0 multiple, and 0 units",
            )
        allocation["decision"] = "NO_TRADE"
        allocation["size_multiple"] = 0.0
        return allocation

    if allocation_decision != "ALLOCATE":
        raise MarketReadOverlayError(
            "MARKET_READ_CAPITAL_ALLOCATION_REQUIRED",
            "an accepted TRADE baseline requires an ALLOCATE capital decision",
        )
    raw_lane_id = allocation.get("lane_id")
    if not isinstance(raw_lane_id, str) or not raw_lane_id.strip():
        raise MarketReadOverlayError(
            "MARKET_READ_CAPITAL_ALLOCATION_LANE_MISMATCH",
            "capital allocation lane_id must be a non-empty string",
        )
    lane_id = raw_lane_id.strip()
    if not primary_lane_id or lane_id != primary_lane_id:
        raise MarketReadOverlayError(
            "MARKET_READ_CAPITAL_ALLOCATION_LANE_MISMATCH",
            "capital allocation must bind the deterministic baseline selected lane",
        )
    board = allocation_board if isinstance(allocation_board, Mapping) else {}
    selected_lane = (
        board.get("selected_lane")
        if isinstance(board.get("selected_lane"), Mapping)
        else None
    )
    if (
        not isinstance(selected_lane, Mapping)
        or str(selected_lane.get("lane_id") or "") != lane_id
    ):
        raise MarketReadOverlayError(
            "MARKET_READ_CAPITAL_ALLOCATION_LANE_MISSING",
            "selected baseline lane is absent from the content-addressed allocation board",
        )
    forecast_context_error = _selected_lane_forecast_context_binding_error(
        board=board,
        selected_lane=selected_lane,
        lane_id=lane_id,
    )
    if forecast_context_error is not None:
        raise MarketReadOverlayError(
            "MARKET_READ_CAPITAL_ALLOCATION_FORECAST_CONTEXT_INVALID",
            forecast_context_error,
        )
    if selected_lane.get("allocation_eligible") is not True:
        raise MarketReadOverlayError(
            "MARKET_READ_CAPITAL_ALLOCATION_EDGE_NOT_PROVEN",
            "selected lane is not LIVE_READY with current risk permission and either "
            "positive edge proof or a separately labeled M15 recovery evidence-collection proof; "
            "use a non-trade disposition",
        )
    allowed_multiples = selected_lane.get("allowed_size_multiples")
    allowed = [
        float(item)
        for item in allowed_multiples or []
        if isinstance(item, (int, float))
        and not isinstance(item, bool)
        and math.isfinite(float(item))
        and float(item) in CAPITAL_ALLOCATION_SIZE_RATIOS
    ]
    if size_multiple not in CAPITAL_ALLOCATION_SIZE_RATIOS or size_multiple not in allowed:
        raise MarketReadOverlayError(
            "MARKET_READ_CAPITAL_ALLOCATION_MULTIPLE_INVALID",
            f"size_multiple {size_multiple:g} is not allowed for lane {lane_id}: {allowed}",
        )
    base_units = selected_lane.get("base_units")
    if isinstance(base_units, bool) or not isinstance(base_units, int) or base_units <= 0:
        raise MarketReadOverlayError(
            "MARKET_READ_CAPITAL_ALLOCATION_UNITS_INVALID",
            "allocation board base_units must be a positive integer",
        )
    numerator, denominator = CAPITAL_ALLOCATION_SIZE_RATIOS[size_multiple]
    expected_units = base_units * numerator // denominator
    if expected_units <= 0 or raw_units != expected_units:
        raise MarketReadOverlayError(
            "MARKET_READ_CAPITAL_ALLOCATION_UNITS_MISMATCH",
            f"selected_units must equal floor({base_units} * {size_multiple:g}) = {expected_units}",
        )
    allocation["decision"] = "ALLOCATE"
    allocation["lane_id"] = lane_id
    allocation["size_multiple"] = size_multiple
    return allocation


def _selected_lane_range_tp_breakout_failure_proven(
    selected_lane: Mapping[str, Any],
) -> bool:
    """Re-derive the narrow RANGE failed-break exception from the board.

    The board is content addressed, but a naked boolean inside it is still a
    claim.  Re-bind the executable LIMIT vehicle, exact-TP cohort, current
    ledger surface, cost-aware numeric proof, and range rails before allowing
    a non-directional RANGE forecast to carry the failed-break side.
    """

    forecast = (
        selected_lane.get("forecast")
        if isinstance(selected_lane.get("forecast"), Mapping)
        else {}
    )
    capture = (
        selected_lane.get("capture")
        if isinstance(selected_lane.get("capture"), Mapping)
        else {}
    )
    numeric = (
        selected_lane.get("numeric_ceiling")
        if isinstance(selected_lane.get("numeric_ceiling"), Mapping)
        else {}
    )
    inputs = (
        numeric.get("inputs")
        if isinstance(numeric.get("inputs"), Mapping)
        else {}
    )
    geometry = (
        numeric.get("geometry")
        if isinstance(numeric.get("geometry"), Mapping)
        else {}
    )
    forecast_basis = forecast.get("range_trade_economic_basis")
    numeric_basis = inputs.get("range_trade_economic_basis")
    ledger_sha = str(capture.get("execution_ledger_surface_sha256") or "")
    max_multiple = _strict_positive_number(numeric.get("max_multiple"))
    return bool(
        str(selected_lane.get("method") or "").strip().upper()
        == "BREAKOUT_FAILURE"
        and _normalized_order_vehicle(selected_lane.get("order_type")) == "LIMIT"
        and str(forecast.get("direction") or "").strip().upper() == "RANGE"
        and str(forecast.get("calibration_name") or "").strip().lower()
        == "directional_forecast_range"
        and selected_lane.get("edge_basis") == "EXACT_VEHICLE_TAKE_PROFIT"
        and capture.get("exact_vehicle_edge_proven") is True
        and SHA256_RE.fullmatch(ledger_sha)
        and isinstance(forecast_basis, Mapping)
        and isinstance(numeric_basis, Mapping)
        and dict(forecast_basis) == dict(numeric_basis)
        and str(forecast_basis.get("basis") or "").strip().upper()
        == "EXACT_TP_PROVEN_HARVEST"
        and str(forecast_basis.get("execution_ledger_surface_sha256") or "")
        == ledger_sha
        and inputs.get("range_trade_economic_basis_valid") is True
        and numeric.get("reason")
        == "TP_PROVEN_RANGE_NONMARKET_PREBOUNDED_CONTRACT"
        and numeric.get("execution_cost_floor_valid") is True
        and geometry.get("range_rails_bound") is True
        and geometry.get("forecast_outcome_conservatively_contains_order") is True
        and geometry.get("range_risk_cap_passed") is True
        and geometry.get("passed") is True
        and max_multiple is not None
        and math.isclose(
            max_multiple,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )


def _selected_lane_forecast_context_binding_error(
    *,
    board: Mapping[str, Any],
    selected_lane: Mapping[str, Any],
    lane_id: str,
) -> str | None:
    """Re-prove the selected context binding before accepting GPT allocation."""

    match_count = board.get("selected_lane_match_count")
    if (
        isinstance(match_count, bool)
        or not isinstance(match_count, int)
        or match_count != 1
        or board.get("selected_lane_unique") is not True
        or str(board.get("selected_lane_id") or "") != lane_id
    ):
        return "capital-allocation board does not bind exactly one selected lane"
    if (
        board.get("forecast_context_scope")
        == M15_RECOVERY_ALLOCATION_CONTEXT_SCOPE
    ):
        return _selected_lane_m15_recovery_context_binding_error(
            board=board,
            selected_lane=selected_lane,
            lane_id=lane_id,
        )
    if board.get("forecast_context_scope") != "SELECTED_LANE":
        return "selected trade allocation requires SELECTED_LANE forecast context"
    scoped = board.get("forecast_context")
    if not isinstance(scoped, Mapping):
        return "selected trade allocation lacks a forecast-context receipt"
    if set(scoped) != {
        "pair",
        "direction",
        "source_lane_id",
        "source_generated_at_utc",
        "candidate_count",
        "valid_candidate_count",
        "invalid_candidate_count",
        "distinct_context_sha256_count",
        "technical_context",
    }:
        return "selected forecast-context receipt has an unexpected schema"
    if (
        _strict_nonnegative_int(scoped.get("candidate_count")) != 1
        or _strict_nonnegative_int(scoped.get("valid_candidate_count")) != 1
        or _strict_nonnegative_int(scoped.get("invalid_candidate_count")) != 0
        or _strict_nonnegative_int(
            scoped.get("distinct_context_sha256_count")
        )
        != 1
    ):
        return "selected forecast-context receipt does not prove one valid candidate"
    if str(scoped.get("source_lane_id") or "") != lane_id:
        return "selected forecast context is bound to a different source lane"
    if scoped.get("source_generated_at_utc") != board.get(
        "order_intents_generated_at_utc"
    ):
        return "selected forecast context is bound to a different intent snapshot"

    pair = str(selected_lane.get("pair") or "").strip().upper()
    forecast = (
        selected_lane.get("forecast")
        if isinstance(selected_lane.get("forecast"), Mapping)
        else None
    )
    if not pair or pair not in DEFAULT_TRADER_PAIRS or not isinstance(forecast, Mapping):
        return "selected lane lacks a supported pair or forecast receipt"
    forecast_binding = _directional_forecast_trade_binding(
        side=selected_lane.get("side"),
        forecast_direction=forecast.get("direction"),
        calibration_name=forecast.get("calibration_name"),
        method=selected_lane.get("method"),
    )
    if not forecast_binding["direction_side_aligned"]:
        return "selected forecast direction contradicts the executable order side"
    if not forecast_binding["calibration_identity_valid"]:
        return "selected forecast calibration identity contradicts its direction"
    if str(scoped.get("pair") or "") != pair:
        return "selected forecast-context pair does not match the lane"
    if scoped.get("direction") != forecast_binding["forecast_direction"]:
        return "selected forecast-context direction does not match the lane"

    current_price = forecast.get("current_price")
    lane_evidence = normalize_forecast_technical_context_evidence(
        forecast.get("technical_context"),
        pair=pair,
        current_price=current_price,
    )
    scoped_evidence = normalize_forecast_technical_context_evidence(
        scoped.get("technical_context"),
        pair=pair,
        current_price=current_price,
    )
    if (
        lane_evidence.get("status") != "VALID"
        or scoped_evidence.get("status") != "VALID"
    ):
        return "selected lane forecast context is missing, invalid, or unbound"
    lane_evidence_sha = str(lane_evidence.get("evidence_sha256") or "")
    scoped_evidence_sha = str(scoped_evidence.get("evidence_sha256") or "")
    lane_context_sha = str(lane_evidence.get("context_sha256") or "")
    scoped_context_sha = str(scoped_evidence.get("context_sha256") or "")
    if (
        not SHA256_RE.fullmatch(lane_evidence_sha)
        or lane_evidence_sha != scoped_evidence_sha
        or not SHA256_RE.fullmatch(lane_context_sha)
        or lane_context_sha != scoped_context_sha
    ):
        return "selected forecast-context digest does not match its source lane"
    regime_family_binding = _regime_family_allocation_binding(
        technical_context=lane_evidence,
        pair=pair,
        side=selected_lane.get("side"),
        method=selected_lane.get("method"),
        forecast_direction=forecast.get("direction"),
        explicit_receipt_sha256=forecast.get(
            "regime_family_weighting_sha256"
        ),
        explicit_selected_method=forecast.get(
            "regime_family_selected_method"
        ),
        explicit_family_direction=forecast.get("regime_family_direction"),
        range_tp_proven_breakout_failure=(
            _selected_lane_range_tp_breakout_failure_proven(selected_lane)
        ),
        canonical_policy_sources=board.get("dynamic_tf_policy_sources"),
        canonical_forecast_context_sha256=board.get(
            "canonical_forecast_context_sha256"
        ),
    )
    if not regime_family_binding["current_context_bound"]:
        return "selected lane forecast context is legacy/intermediate and cannot allocate"
    if not regime_family_binding["canonical_policy_sources_bound"]:
        return "selected lane dynamic-TF policy sources do not match the evidence packet"
    if not regime_family_binding["canonical_forecast_context_bound"]:
        return "selected lane technical context does not match canonical chart/source/quote rebuild"
    if not regime_family_binding["receipt_valid"]:
        return "selected lane lacks a valid regime-family weighting receipt"
    if not regime_family_binding["sha_bound"]:
        return "selected lane regime-family receipt SHA is unbound"
    if (
        not regime_family_binding["selected_method_bound"]
        or not regime_family_binding["executable_method_bound"]
    ):
        return "selected lane method contradicts the regime-family receipt"
    if not regime_family_binding["family_direction_bound"]:
        return "selected lane family direction is unbound"
    if not regime_family_binding["direction_consistent"]:
        return "selected lane forecast direction contradicts family evidence"
    if not regime_family_binding["actionable_direction_bound"]:
        return "selected lane family direction is non-actionable or has no coverage"
    scoped_body = scoped_evidence.get("technical_context_v1")
    scoped_receipt = (
        scoped_body.get("regime_family_weighting")
        if isinstance(scoped_body, Mapping)
        and isinstance(scoped_body.get("regime_family_weighting"), Mapping)
        else {}
    )
    if scoped_receipt.get("receipt_sha256") != regime_family_binding[
        "receipt_sha256"
    ]:
        return "selected regime-family receipt does not match scoped context"
    return None


def _selected_lane_m15_recovery_context_binding_error(
    *,
    board: Mapping[str, Any],
    selected_lane: Mapping[str, Any],
    lane_id: str,
) -> str | None:
    """Rebuild the recovery chain instead of trusting its scoped label."""

    scoped = board.get("forecast_context")
    expected_scoped_keys = {
        "contract",
        "status",
        "pair",
        "direction",
        "side",
        "method",
        "order_type",
        "source_lane_id",
        "source_generated_at_utc",
        "forecast_cycle_id",
        "source_receipt_sha256",
        "forecast_binding_sha256",
        "lane_binding_sha256",
        "context_timeframes",
        "positive_rotation_mode",
        "edge_basis",
        "recovery_material_sha256",
        "recovery_context_sha256",
    }
    if not isinstance(scoped, Mapping) or set(scoped) != expected_scoped_keys:
        return "selected M15 recovery context has an unexpected schema"
    if not _content_addressed_mapping_valid(
        scoped,
        digest_key="recovery_context_sha256",
    ):
        return "selected M15 recovery context digest is invalid"
    if (
        scoped.get("contract") != M15_RECOVERY_ALLOCATION_CONTEXT_CONTRACT
        or scoped.get("status") != M15_RECOVERY_ALLOCATION_STATUS
        or scoped.get("source_lane_id") != lane_id
        or scoped.get("source_generated_at_utc")
        != board.get("order_intents_generated_at_utc")
    ):
        return "selected M15 recovery context source binding is invalid"

    recovery = selected_lane.get("m15_recovery")
    if not isinstance(recovery, Mapping):
        return "selected lane lacks its M15 recovery validation material"
    material = recovery.get("material")
    if (
        recovery.get("claimed") is not True
        or recovery.get("valid") is not True
        or recovery.get("status") != M15_RECOVERY_ALLOCATION_STATUS
        or not isinstance(material, Mapping)
        or not _content_addressed_mapping_valid(
            material,
            digest_key="material_sha256",
        )
        or recovery.get("material_sha256") != material.get("material_sha256")
    ):
        return "selected lane M15 recovery material is missing or invalid"

    metadata = material.get("metadata")
    risk_issues = material.get("risk_issues")
    capture = selected_lane.get("capture")
    forecast = selected_lane.get("forecast")
    base_units = selected_lane.get("base_units")
    if (
        not isinstance(metadata, Mapping)
        or not isinstance(risk_issues, list)
        or not isinstance(capture, Mapping)
        or not isinstance(forecast, Mapping)
        or isinstance(base_units, bool)
        or not isinstance(base_units, int)
        or not 1 <= base_units <= M15_RECOVERY_MICRO_MAX_UNITS
        or capture.get("exact_vehicle_edge_proven") is not True
        or not SHA256_RE.fullmatch(
            str(capture.get("execution_ledger_surface_sha256") or "")
        )
        or capture.get("take_profit_trades")
        != metadata.get("capture_take_profit_trades")
        or capture.get("take_profit_wins")
        != metadata.get("capture_take_profit_wins")
        or capture.get("take_profit_losses")
        != metadata.get("capture_take_profit_losses")
        or capture.get("take_profit_expectancy_jpy")
        != metadata.get("capture_take_profit_expectancy_jpy")
    ):
        return "selected lane M15 recovery exact-TP receipt is invalid"

    rebuilt = _m15_recovery_allocation_contract(
        result={"risk_issues": risk_issues},
        intent={
            "pair": selected_lane.get("pair"),
            "side": selected_lane.get("side"),
            "order_type": selected_lane.get("order_type"),
            "units": base_units,
            "entry": selected_lane.get("entry"),
            "tp": selected_lane.get("tp"),
            "sl": selected_lane.get("sl"),
            "metadata": metadata,
        },
        metadata=metadata,
        method=str(selected_lane.get("method") or "").strip().upper(),
        base_units=base_units,
        positive_tp_edge=True,
    )
    if rebuilt.get("valid") is not True or rebuilt.get("material") != dict(material):
        return "selected lane M15 recovery chain cannot be rebuilt exactly"

    mode = rebuilt.get("mode")
    edge_basis = rebuilt.get("edge_basis")
    if mode == M15_RECOVERY_PROOF_COLLECTION_MODE:
        if (
            edge_basis != M15_RECOVERY_EDGE_COLLECTION_BASIS
            or selected_lane.get("edge_basis")
            != M15_RECOVERY_EDGE_COLLECTION_BASIS
            or selected_lane.get("edge_collection_proven") is not True
            or selected_lane.get("positive_edge_proven") is not False
        ):
            return "M15 recovery proof collection was mislabeled as proven edge"
    elif mode == M15_RECOVERY_TP_PROVEN_MODE:
        if (
            edge_basis != "EXACT_VEHICLE_TAKE_PROFIT"
            or selected_lane.get("edge_basis")
            != "EXACT_VEHICLE_TAKE_PROFIT"
            or selected_lane.get("positive_edge_proven") is not True
            or selected_lane.get("edge_collection_proven") is not False
        ):
            return "M15 recovery TP-proven edge basis is invalid"
    else:
        return "selected M15 recovery proof mode is not executable"

    expected = {
        "pair": str(selected_lane.get("pair") or "").strip().upper(),
        "direction": forecast.get("direction"),
        "side": str(selected_lane.get("side") or "").strip().upper(),
        "method": str(selected_lane.get("method") or "").strip().upper(),
        "order_type": _m15_recovery_contract_order_type(
            selected_lane.get("order_type")
        ),
        "forecast_cycle_id": rebuilt.get("forecast_cycle_id"),
        "source_receipt_sha256": rebuilt.get("source_receipt_sha256"),
        "forecast_binding_sha256": rebuilt.get("forecast_binding_sha256"),
        "lane_binding_sha256": rebuilt.get("lane_binding_sha256"),
        "context_timeframes": list(M15_RECOVERY_CONTEXT_TIMEFRAMES),
        "positive_rotation_mode": mode,
        "edge_basis": edge_basis,
        "recovery_material_sha256": rebuilt.get("material_sha256"),
    }
    if any(scoped.get(key) != value for key, value in expected.items()):
        return "selected M15 recovery context contradicts its source lane"
    return None


def revalidate_codex_market_read_artifacts(
    *,
    final_payload: Mapping[str, Any],
    baseline_path: Path,
    packet_path: Path,
    overlay_path: Path,
    evidence_sources: Mapping[str, Path],
    max_overlay_age_seconds: int = DEFAULT_OVERLAY_MAX_AGE_SECONDS,
) -> list[tuple[str, str]]:
    """Rebuild an execution-bearing receipt from its actual handoff artifacts.

    Provenance fields inside the final JSON are claims, not proof.  The
    verifier therefore re-runs the same content-addressed merge against the
    current baseline, stored packet, overlay, and every named evidence source,
    then requires the rebuilt final object to be identical. TRADE and CLOSE
    carry execution authority; WAIT and REQUEST_EVIDENCE carry veto authority
    over campaign pressure. All four therefore require the same artifact proof.
    """

    action = str(final_payload.get("action") or "").strip().upper()
    if action not in {"TRADE", "CLOSE", "WAIT", "REQUEST_EVIDENCE"}:
        return []
    provenance = final_payload.get("decision_provenance")
    if not isinstance(provenance, Mapping):
        return [
            (
                "AI_MARKET_READ_ARTIFACT_PROVENANCE_MISSING",
                f"{action} cannot bind market-read artifacts without decision_provenance",
            )
        ]
    try:
        applied_at = _parse_utc(
            provenance.get("applied_at_utc"),
            code="AI_MARKET_READ_ARTIFACT_PROVENANCE_INVALID",
        )
    except MarketReadOverlayError as exc:
        return [(exc.code, exc.message)]

    try:
        with tempfile.TemporaryDirectory(prefix="qr-market-read-revalidate-") as tmp:
            rebuilt_path = Path(tmp) / "rebuilt_final.json"
            apply_codex_market_read_overlay(
                baseline_path=baseline_path,
                packet_path=packet_path,
                overlay_path=overlay_path,
                output_path=rebuilt_path,
                evidence_sources=evidence_sources,
                # Reproduce the original atomic publication timestamp.  The
                # ordinary provenance validator separately checks that this
                # publication is still fresh at verification time.
                now=applied_at,
                max_overlay_age_seconds=max_overlay_age_seconds,
            )
            rebuilt = _load_json_object(rebuilt_path, label="rebuilt market-read decision")
    except MarketReadOverlayError as exc:
        return [(exc.code, exc.message)]
    except (OSError, ValueError) as exc:
        return [
            (
                "AI_MARKET_READ_ARTIFACT_REVALIDATION_FAILED",
                f"could not rebuild the final {action} receipt from market-read artifacts: {exc}",
            )
        ]

    rebuilt_sha = canonical_json_sha256(rebuilt)
    final_sha = canonical_json_sha256(dict(final_payload))
    if rebuilt_sha != final_sha:
        return [
            (
                "AI_MARKET_READ_ARTIFACT_FINAL_MISMATCH",
                f"final {action} receipt does not exactly match the current baseline/evidence/overlay merge "
                f"(rebuilt={rebuilt_sha}, final={final_sha})",
            )
        ]
    return []


def validate_codex_market_read_provenance(
    *,
    action: str,
    market_read: Mapping[str, Any],
    provenance: Mapping[str, Any] | None,
    review: Mapping[str, Any] | None,
    counterargument: str,
    change_summary: str,
    disposition: str = "",
    veto_reason: str = "",
    vetoed_lane_ids: tuple[str, ...] = (),
    capital_allocation: Mapping[str, Any] | None = None,
    execution_envelope_sha256: str | None = None,
    now: datetime | None = None,
    max_age_seconds: int = DEFAULT_OVERLAY_MAX_AGE_SECONDS,
) -> list[tuple[str, str]]:
    claimed = dict(provenance) if isinstance(provenance, Mapping) else {}
    if action.upper() == "TRADE" and claimed.get("author_kind") != CODEX_MARKET_READ_AUTHOR:
        return [
            (
                "AI_MARKET_READ_REQUIRED",
                "TRADE requires a fresh CODEX_MARKET_READ overlay; deterministic fallback remains non-entry only.",
            )
        ]
    if claimed.get("author_kind") != CODEX_MARKET_READ_AUTHOR:
        return []

    issues: list[tuple[str, str]] = []
    if claimed.get("schema_version") != MARKET_READ_OVERLAY_SCHEMA_VERSION:
        issues.append(
            (
                "AI_MARKET_READ_PROVENANCE_INVALID",
                f"CODEX_MARKET_READ provenance schema_version must be {MARKET_READ_OVERLAY_SCHEMA_VERSION}",
            )
        )
    for key in (
        "baseline_sha256",
        "evidence_packet_sha256",
        "overlay_sha256",
        "market_read_sha256",
        "execution_envelope_sha256",
        "baseline_execution_envelope_sha256",
        "final_execution_envelope_sha256",
        "capital_allocation_sha256",
        "capital_allocation_board_sha256",
        "guardian_action_receipt_scope_state_sha256",
    ):
        if not SHA256_RE.fullmatch(str(claimed.get(key) or "")):
            issues.append(("AI_MARKET_READ_PROVENANCE_INVALID", f"{key} must be a sha256 digest"))
    if claimed.get("guardian_action_receipt_material_contract") != (
        GUARDIAN_ACTION_RECEIPT_MATERIAL_CONTRACT
    ):
        issues.append(
            (
                "AI_MARKET_READ_PROVENANCE_INVALID",
                "guardian_action_receipt_material_contract must be "
                f"{GUARDIAN_ACTION_RECEIPT_MATERIAL_CONTRACT}",
            )
        )
    raw_guardian_pairs = claimed.get("guardian_action_receipt_baseline_pairs")
    raw_baseline_lane_ids = claimed.get("baseline_selected_lane_ids")
    expected_guardian_pairs = list(
        _guardian_receipt_baseline_pairs(
            {
                "selected_lane_ids": (
                    raw_baseline_lane_ids
                    if isinstance(raw_baseline_lane_ids, list)
                    else []
                )
            }
        )
    )
    guardian_pairs_valid = bool(
        isinstance(raw_guardian_pairs, list)
        and all(isinstance(pair, str) for pair in raw_guardian_pairs)
        and raw_guardian_pairs
        == sorted(set(raw_guardian_pairs))
        and all(pair in DEFAULT_TRADER_PAIRS for pair in raw_guardian_pairs)
        and set(expected_guardian_pairs).issubset(set(raw_guardian_pairs))
    )
    if not guardian_pairs_valid:
        issues.append(
            (
                "AI_MARKET_READ_PROVENANCE_INVALID",
                "guardian_action_receipt_baseline_pairs must be the sorted unique "
                "reviewed pairs and include every execution pair derived from "
                "baseline_selected_lane_ids",
            )
        )
    claimed_cost_floor_sha = claimed.get("execution_cost_floor_sha256")
    if action.upper() == "TRADE" and not SHA256_RE.fullmatch(
        str(claimed_cost_floor_sha or "")
    ):
        issues.append(
            (
                "AI_EXECUTION_COST_FLOOR_PROVENANCE_INVALID",
                "schema-v2 TRADE requires the exact ledger-derived execution-cost floor digest",
            )
        )
    claimed_edge_basis = str(
        claimed.get("capital_allocation_edge_basis") or ""
    )
    if action.upper() == "TRADE" and claimed_edge_basis not in {
        "EXACT_VEHICLE_ALL_EXIT_NET",
        "EXACT_VEHICLE_TAKE_PROFIT",
        M15_RECOVERY_EDGE_COLLECTION_BASIS,
        "PREDICTIVE_SCOUT_FORWARD_EVIDENCE",
        "HEDGE_RISK_REDUCTION",
    }:
        issues.append(
            (
                "AI_CAPITAL_ALLOCATION_EDGE_BASIS_INVALID",
                "schema-v2 TRADE requires the exact board-selected edge basis",
            )
        )
    if claimed.get("model") != "gpt-5.5" or str(claimed.get("reasoning_effort") or "").lower() != "high":
        issues.append(
            (
                "AI_MARKET_READ_PROVENANCE_INVALID",
                "CODEX_MARKET_READ provenance must name gpt-5.5 with high reasoning effort",
            )
        )
    allocation = (
        dict(capital_allocation)
        if isinstance(capital_allocation, Mapping)
        else {}
    )
    if not allocation:
        issues.append(
            (
                "AI_CAPITAL_ALLOCATION_MISSING",
                "CODEX_MARKET_READ provenance requires a bounded capital_allocation receipt",
            )
        )
    else:
        claimed_allocation_sha = str(claimed.get("capital_allocation_sha256") or "")
        if claimed_allocation_sha != canonical_json_sha256(allocation):
            issues.append(
                (
                    "AI_CAPITAL_ALLOCATION_DIGEST_MISMATCH",
                    "capital_allocation no longer matches the applied overlay digest",
                )
            )
        if str(allocation.get("allocation_board_sha256") or "") != str(
            claimed.get("capital_allocation_board_sha256") or ""
        ):
            issues.append(
                (
                    "AI_CAPITAL_ALLOCATION_BOARD_MISMATCH",
                    "capital_allocation board digest does not match provenance",
                )
            )
        allocation_decision = str(allocation.get("decision") or "").upper()
        allocation_lane = str(allocation.get("lane_id") or "") or None
        allocation_multiple = _optional_float(allocation.get("size_multiple"))
        raw_allocation_units = allocation.get("selected_units")
        allocation_units = (
            raw_allocation_units
            if isinstance(raw_allocation_units, int)
            and not isinstance(raw_allocation_units, bool)
            else None
        )
        if action.upper() == "TRADE":
            baseline_lane_ids = claimed.get("baseline_selected_lane_ids")
            expected_lane = (
                str(baseline_lane_ids[0])
                if isinstance(baseline_lane_ids, list) and len(baseline_lane_ids) == 1
                else None
            )
            if (
                allocation_decision != "ALLOCATE"
                or not expected_lane
                or allocation_lane != expected_lane
                or allocation_multiple not in CAPITAL_ALLOCATION_SIZE_MULTIPLES
                or allocation_units is None
                or allocation_units <= 0
            ):
                issues.append(
                    (
                        "AI_CAPITAL_ALLOCATION_INVALID",
                        "TRADE requires a positive bounded allocation for the one deterministic baseline lane",
                    )
                )
        elif (
            allocation_decision != "NO_TRADE"
            or allocation_lane is not None
            or allocation_multiple != 0.0
            or allocation_units != 0
        ):
            issues.append(
                (
                    "AI_CAPITAL_ALLOCATION_NONTRADE_INVALID",
                    "non-TRADE receipts require NO_TRADE with null lane, 0 multiple, and 0 units",
                )
            )
        claimed_size_multiple = _optional_float(
            claimed.get("authorized_size_multiple")
        )
        if (
            allocation_multiple is None
            or claimed_size_multiple is None
            or not math.isclose(
                claimed_size_multiple,
                allocation_multiple,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            issues.append(
                (
                    "AI_CAPITAL_ALLOCATION_PROVENANCE_MISMATCH",
                    "authorized_size_multiple does not match capital_allocation",
                )
            )
        claimed_units = claimed.get("authorized_units")
        if (
            allocation_units is None
            or not isinstance(claimed_units, int)
            or isinstance(claimed_units, bool)
            or claimed_units != allocation_units
        ):
            issues.append(
                (
                    "AI_CAPITAL_ALLOCATION_PROVENANCE_MISMATCH",
                    "authorized_units does not match capital_allocation",
                )
            )
    if claimed.get("execution_fields_preserved") is not True:
        issues.append(
            (
                "AI_MARKET_READ_EXECUTION_ENVELOPE_UNPROVEN",
                "CODEX_MARKET_READ provenance must prove the execution envelope was preserved",
            )
        )
    if claimed.get("risk_envelope_not_expanded") is not True:
        issues.append(
            (
                "AI_MARKET_READ_EXECUTION_ENVELOPE_UNPROVEN",
                "CODEX_MARKET_READ provenance must prove the risk envelope was not expanded",
            )
        )
    if claimed.get("live_permission_granted") is not False:
        issues.append(
            (
                "AI_MARKET_READ_PERMISSION_CLAIM_INVALID",
                "market-read provenance cannot grant live permission",
            )
        )
    claimed_market_read_sha = str(claimed.get("market_read_sha256") or "")
    if claimed_market_read_sha and claimed_market_read_sha != market_read_sha256(dict(market_read)):
        issues.append(
            (
                "AI_MARKET_READ_DIGEST_MISMATCH",
                "market_read_first no longer matches the applied overlay digest",
            )
        )
    claimed_execution_sha = str(claimed.get("execution_envelope_sha256") or "")
    if execution_envelope_sha256 and claimed_execution_sha != execution_envelope_sha256:
        issues.append(
            (
                "AI_MARKET_READ_EXECUTION_DIGEST_MISMATCH",
                "the final decision execution envelope no longer matches the applied overlay digest",
            )
        )
    if str(claimed.get("final_execution_envelope_sha256") or "") != claimed_execution_sha:
        issues.append(
            (
                "AI_MARKET_READ_EXECUTION_DIGEST_MISMATCH",
                "final_execution_envelope_sha256 must equal execution_envelope_sha256",
            )
        )
    try:
        applied_at = _parse_utc(
            claimed.get("applied_at_utc"),
            code="AI_MARKET_READ_PROVENANCE_INVALID",
        )
    except MarketReadOverlayError as exc:
        issues.append((exc.code, exc.message))
    else:
        current = _utc_now(now)
        age_seconds = (current - applied_at).total_seconds()
        if age_seconds < -60 or age_seconds > max_age_seconds:
            issues.append(
                (
                    "AI_MARKET_READ_STALE",
                    f"CODEX_MARKET_READ provenance age {age_seconds:.1f}s exceeds the live decision window",
                )
            )
    if not isinstance(review, Mapping):
        issues.append(("AI_MARKET_READ_REVIEW_MISSING", "market_read_review is required"))
    if not str(counterargument or "").strip():
        issues.append(("AI_MARKET_READ_COUNTERARGUMENT_MISSING", "market_read_counterargument is required"))
    if not str(change_summary or "").strip():
        issues.append(("AI_MARKET_READ_CHANGE_SUMMARY_MISSING", "market_read_change_summary is required"))
    claimed_disposition = str(claimed.get("baseline_disposition") or "").upper()
    receipt_disposition = str(disposition or "").upper()
    baseline_action = str(claimed.get("baseline_action") or "").upper()
    final_action = str(claimed.get("final_action") or "").upper()
    allowed_transition = (
        claimed_disposition == "ACCEPT_BASELINE"
        and baseline_action == final_action == action.upper()
        and claimed.get("action_downgrade_only") is False
    ) or (
        baseline_action == "TRADE"
        and final_action == action.upper()
        and (
            (claimed_disposition == "VETO_WAIT" and final_action == "WAIT")
            or (
                claimed_disposition == "VETO_REQUEST_EVIDENCE"
                and final_action == "REQUEST_EVIDENCE"
            )
        )
        and claimed.get("action_downgrade_only") is True
    )
    if not allowed_transition:
        issues.append(
            (
                "AI_MARKET_READ_DISPOSITION_INVALID",
                "provenance does not prove an unchanged baseline or TRADE-to-non-TRADE veto",
            )
        )
    if receipt_disposition != claimed_disposition:
        issues.append(
            (
                "AI_MARKET_READ_DISPOSITION_INVALID",
                "market_read_disposition does not match provenance baseline_disposition",
            )
        )
    baseline_lane_ids = claimed.get("baseline_selected_lane_ids")
    if not isinstance(baseline_lane_ids, list) or any(
        not isinstance(item, str) or not item for item in baseline_lane_ids
    ):
        issues.append(
            (
                "AI_MARKET_READ_PROVENANCE_INVALID",
                "baseline_selected_lane_ids must be a list of non-empty strings",
            )
        )
        baseline_lane_ids = []
    if claimed_disposition.startswith("VETO_"):
        if not baseline_lane_ids:
            issues.append(
                (
                    "AI_MARKET_READ_VETO_LANES_INVALID",
                    "a baseline TRADE veto must preserve at least one deterministic baseline lane id",
                )
            )
        if not str(veto_reason or "").strip():
            issues.append(("AI_MARKET_READ_VETO_REASON_MISSING", "an AI veto requires market_read_veto_reason"))
        if list(vetoed_lane_ids) != list(baseline_lane_ids):
            issues.append(
                (
                    "AI_MARKET_READ_VETO_LANES_INVALID",
                    "market_read_vetoed_lane_ids must preserve the deterministic baseline lane ids",
                )
            )
    elif str(veto_reason or "").strip() or vetoed_lane_ids:
        issues.append(
            (
                "AI_MARKET_READ_VETO_FIELDS_INVALID",
                "accepted baselines cannot carry veto reason or vetoed lane ids",
            )
        )
    return issues


def validate_market_read_numeric_geometry(
    market_read: Mapping[str, Any],
    *,
    quote_basis_by_pair: Mapping[str, Any],
) -> list[tuple[str, str]]:
    issues: list[tuple[str, str]] = []
    for section_name in ("next_30m_prediction", "next_2h_prediction"):
        section = market_read.get(section_name)
        if not isinstance(section, Mapping):
            issues.append(("AI_MARKET_READ_GEOMETRY_INCOMPLETE", f"{section_name} must be an object"))
            continue
        pair = str(section.get("pair") or "").strip()
        direction = _normalized_direction(section.get("direction"))
        if not pair or direction is None:
            issues.append(
                (
                    "AI_MARKET_READ_GEOMETRY_INCOMPLETE",
                    f"{section_name} requires pair and LONG/SHORT/RANGE direction",
                )
            )
            continue
        basis = _basis_price(quote_basis_by_pair.get(pair))
        targets = _plausible_prices(_numbers(section.get("target_zone")), basis)
        invalidations = _plausible_prices(_numbers(section.get("invalidation")), basis)
        if direction == "RANGE":
            if len(targets) < 2 or len(invalidations) < 2:
                issues.append(
                    (
                        "AI_MARKET_READ_RANGE_GEOMETRY_INCOMPLETE",
                        f"{section_name} RANGE requires numeric lower/upper target rails "
                        "and lower/upper invalidation rails",
                    )
                )
                continue
            if basis is not None and not (min(targets) < basis < max(targets)):
                issues.append(
                    (
                        "AI_MARKET_READ_RANGE_GEOMETRY_CONFLICT",
                        f"{section_name} RANGE target rails do not bracket current {pair} price {basis}",
                    )
                )
            if basis is not None and not (min(invalidations) < basis < max(invalidations)):
                issues.append(
                    (
                        "AI_MARKET_READ_RANGE_GEOMETRY_CONFLICT",
                        f"{section_name} RANGE invalidation rails do not bracket current {pair} price {basis}",
                    )
                )
            if min(invalidations) >= min(targets) or max(invalidations) <= max(targets):
                issues.append(
                    (
                        "AI_MARKET_READ_RANGE_GEOMETRY_CONFLICT",
                        f"{section_name} RANGE invalidation rails must sit outside the target rails",
                    )
                )
            continue
        if not targets or not invalidations:
            issues.append(
                (
                    "AI_MARKET_READ_GEOMETRY_INCOMPLETE",
                    f"{section_name} {direction} requires numeric target and invalidation",
                )
            )
            continue
        if basis is None:
            issues.append(
                (
                    "AI_MARKET_READ_BASIS_MISSING",
                    f"{section_name} has no current broker quote basis for {pair}",
                )
            )
            continue
        # Every declared rail must agree with the directional thesis.  Using
        # only max(targets)/min(invalidations) allowed a straddling list such
        # as LONG target ``1.09, 1.11`` around a 1.10 quote to pass and later
        # be counted as a target touch on the wrong-side rail.
        if direction == "LONG":
            conflict = min(targets) <= basis or max(invalidations) >= basis
        else:
            conflict = max(targets) >= basis or min(invalidations) <= basis
        if conflict:
            issues.append(
                (
                    "AI_MARKET_READ_GEOMETRY_CONFLICT",
                    f"{section_name} {direction} target/invalidation conflicts with current {pair} price {basis}",
                )
            )

    forced = market_read.get("best_trade_if_forced")
    if not isinstance(forced, Mapping):
        issues.append(("AI_MARKET_READ_FORCED_GEOMETRY_INCOMPLETE", "best_trade_if_forced must be an object"))
        return issues
    forced_pair = str(forced.get("pair") or "").strip()
    forced_direction = _normalized_direction(forced.get("direction"))
    forced_basis = _basis_price(quote_basis_by_pair.get(forced_pair))
    entry_candidates = _plausible_prices(_numbers(forced.get("entry")), forced_basis)
    entry = entry_candidates[0] if entry_candidates else None
    tp = _plausible_prices(_numbers(forced.get("tp")), entry)
    sl = _plausible_prices(_numbers(forced.get("sl")), entry)
    if not forced_pair or forced_direction not in {"LONG", "SHORT"} or entry is None or not tp or not sl:
        issues.append(
            (
                "AI_MARKET_READ_FORCED_GEOMETRY_INCOMPLETE",
                "best_trade_if_forced requires pair, LONG/SHORT, and numeric entry/TP/SL",
            )
        )
    elif forced_basis is None:
        issues.append(
            (
                "AI_MARKET_READ_BASIS_MISSING",
                f"best_trade_if_forced has no current broker quote basis for {forced_pair}",
            )
        )
    elif forced_direction == "LONG" and (min(tp) <= entry or max(sl) >= entry):
        issues.append(("AI_MARKET_READ_FORCED_GEOMETRY_CONFLICT", "forced LONG TP/SL geometry is inverted"))
    elif forced_direction == "SHORT" and (max(tp) >= entry or min(sl) <= entry):
        issues.append(("AI_MARKET_READ_FORCED_GEOMETRY_CONFLICT", "forced SHORT TP/SL geometry is inverted"))
    return issues


def quote_basis_by_pair_from_broker_payload(payload: Mapping[str, Any]) -> dict[str, float]:
    quotes = payload.get("quotes") if isinstance(payload, Mapping) else None
    if not isinstance(quotes, Mapping):
        return {}
    result: dict[str, float] = {}
    for pair, raw in quotes.items():
        if not isinstance(raw, Mapping):
            continue
        bid = _optional_float(raw.get("bid"))
        ask = _optional_float(raw.get("ask"))
        if bid is not None and ask is not None:
            result[str(pair)] = (bid + ask) / 2.0
    return result


def _validate_trade_source_freshness(
    *,
    baseline: Mapping[str, Any],
    market_read: Mapping[str, Any],
    evidence_sources: Mapping[str, Path],
    now: datetime,
) -> None:
    """Reject a final TRADE whose cited broker truth is already stale."""

    # AI evidence may be older than the final POST quote without weakening the
    # gateway.  Reuse the existing five-minute read-only guardian snapshot
    # window here; RiskEngine still enforces its separate 20-second quote
    # contract immediately before any broker send.
    from quant_rabbit.guardian_events import DEFAULT_ROUTER_SNAPSHOT_MAX_AGE_SECONDS

    max_age_seconds = float(DEFAULT_ROUTER_SNAPSHOT_MAX_AGE_SECONDS)

    def require_fresh(value: Any, *, label: str) -> None:
        try:
            observed_at = _parse_utc(value, code="MARKET_READ_SOURCE_STALE")
        except MarketReadOverlayError as exc:
            raise MarketReadOverlayError(
                "MARKET_READ_SOURCE_STALE",
                f"{label} timestamp is missing or invalid",
            ) from exc
        age_seconds = (now - observed_at).total_seconds()
        if age_seconds < -60 or age_seconds > max_age_seconds:
            raise MarketReadOverlayError(
                "MARKET_READ_SOURCE_STALE",
                f"{label} age {age_seconds:.1f}s is outside -60..{max_age_seconds:.1f}s",
            )

    require_fresh(baseline.get("generated_at_utc"), label="deterministic baseline")
    broker_path = evidence_sources.get("broker_snapshot")
    if broker_path is None:
        raise MarketReadOverlayError(
            "MARKET_READ_SOURCE_STALE",
            "broker_snapshot evidence source is required for a TRADE",
        )
    broker = _load_json_object(_normalize_source_path(broker_path), label="broker snapshot")
    require_fresh(broker.get("fetched_at_utc"), label="broker snapshot")
    quotes = broker.get("quotes")
    if not isinstance(quotes, Mapping):
        raise MarketReadOverlayError(
            "MARKET_READ_SOURCE_STALE",
            "broker snapshot quotes are missing for a TRADE",
        )
    pairs: set[str] = set()
    naked = market_read.get("naked_read")
    if isinstance(naked, Mapping):
        pair = str(naked.get("cleanest_pair_expression") or "").strip()
        if pair:
            pairs.add(pair)
    for key in ("next_30m_prediction", "next_2h_prediction", "best_trade_if_forced"):
        section = market_read.get(key)
        if not isinstance(section, Mapping):
            continue
        pair = str(section.get("pair") or "").strip()
        if pair:
            pairs.add(pair)
    if not pairs:
        raise MarketReadOverlayError(
            "MARKET_READ_SOURCE_STALE",
            "market read has no pair whose quote freshness can be verified",
        )
    for pair in sorted(pairs):
        quote = quotes.get(pair)
        if not isinstance(quote, Mapping):
            raise MarketReadOverlayError(
                "MARKET_READ_SOURCE_STALE",
                f"broker quote is missing for market-read pair {pair}",
            )
        require_fresh(quote.get("timestamp_utc"), label=f"{pair} quote")


def _build_evidence_packet(
    *,
    baseline: Mapping[str, Any],
    baseline_sha256: str,
    evidence_sources: Mapping[str, Path],
    prepared_at: datetime,
    use_runtime_guardian_clock: bool = False,
) -> dict[str, Any]:
    if "guardian_action_receipt" not in evidence_sources:
        raise MarketReadOverlayError(
            "MARKET_READ_EVIDENCE_SOURCE_CONTRACT_INVALID",
            "guardian_action_receipt must be an explicitly named evidence source",
        )
    guardian_source_path = evidence_sources.get("guardian_action_receipt")
    if not isinstance(guardian_source_path, (Path, str)) or not str(
        guardian_source_path
    ).strip():
        raise MarketReadOverlayError(
            "MARKET_READ_EVIDENCE_SOURCE_CONTRACT_INVALID",
            "guardian_action_receipt evidence path must be a non-empty path",
        )
    normalized_sources = {
        name: _normalize_source_path(path) for name, path in evidence_sources.items()
    }
    sources: dict[str, dict[str, Any]] = {}
    for name in sorted(normalized_sources):
        path = normalized_sources[name]
        if name == "execution_ledger":
            # Never hash SQLite's main file as allocation truth: committed WAL
            # rows may not be checkpointed into it yet. The semantic snapshot
            # below is the authoritative content-addressed source.
            sources[name] = {
                "path": str(path),
                "exists": path.exists() and path.is_file(),
                "sha256": None,
                "size_bytes": None,
                "generated_at_utc": None,
            }
        elif name in {"projection_ledger", "guardian_action_receipt"}:
            # The 100MB+ append ledger is not a prompt artifact.  Bind the
            # selected semantic calibration evidence below instead of reading
            # and hashing every raw row a second time.  The frequently rotated
            # Guardian receipt is likewise bound through its scoped semantic
            # payload, not through volatile publication bytes.
            sources[name] = _source_descriptor_without_hash(path)
        else:
            sources[name] = _source_descriptor(path)
    execution_ledger_path = normalized_sources.get("execution_ledger")
    full_execution_ledger_surface = (
        read_exact_vehicle_allocation_surface(execution_ledger_path)
        if execution_ledger_path is not None
        else _missing_execution_ledger_allocation_surface()
    )
    predictions_path = normalized_sources.get("market_read_predictions")
    recent_predictions = (
        _recent_resolved_predictions(Path(predictions_path))
        if predictions_path is not None
        else []
    )
    replay_path = normalized_sources.get("bidask_replay_validation")
    forecast_replay_scorecard = _forecast_replay_scorecard(
        Path(replay_path) if replay_path is not None else None,
        baseline=baseline,
    )
    order_intents_path = normalized_sources.get("order_intents")
    order_intents = (
        _load_optional_json_object(Path(order_intents_path))
        if order_intents_path is not None
        else {}
    )
    execution_ledger_surface = _selected_execution_ledger_allocation_surface(
        full_execution_ledger_surface,
        baseline=baseline,
        order_intents=order_intents,
    )
    material_sources: dict[str, dict[str, Any]] = {}
    guardian_action_receipt_material: dict[str, Any] | None = None
    for name, item in sources.items():
        if name == "market_read_predictions":
            # GPTTraderBrain appends the just-verified prediction after the
            # artifact merge. That new row is intentionally unresolved and
            # therefore is not evidence the Codex read could have reviewed.
            # Bind only the exact resolved projection exposed below; otherwise
            # the verifier invalidates its own accepted handoff before the
            # cycle can consume it. A prediction becoming resolved (or a
            # resolved row changing) still changes this digest and requires a
            # fresh overlay.
            material_sources[name] = {
                "path": item["path"],
                "resolved_predictions_sha256": canonical_json_sha256(recent_predictions),
                "resolved_prediction_count": len(recent_predictions),
            }
            continue
        if name in {"qr_trader_run_watchdog", "guardian_action_receipt"}:
            # Both are intentionally sampled after every slower allocation and
            # calibration surface below, immediately before packet material is
            # finalized.
            continue
        if name == "execution_ledger":
            material_sources[name] = {
                "path": item["path"],
                "exists": item["exists"],
                "material_contract": EXACT_VEHICLE_ALLOCATION_SURFACE_CONTRACT,
                "parse_status": execution_ledger_surface.get("parse_status"),
                "coverage_start_utc": execution_ledger_surface.get(
                    "coverage_start_utc"
                ),
                "selected_scope_key": execution_ledger_surface.get(
                    "selected_scope_key"
                ),
                "allocation_surface_sha256": execution_ledger_surface.get(
                    "allocation_surface_sha256"
                ),
            }
            continue
        if name == "projection_ledger":
            # Filled after the selected pair/signal scope is known.
            continue
        material_sources[name] = {
            "path": item["path"],
            "exists": item["exists"],
            "sha256": item.get("sha256"),
            "size_bytes": item.get("size_bytes"),
        }
    profitability_path = normalized_sources.get("profitability_acceptance")
    profitability = (
        _load_optional_json_object(Path(profitability_path))
        if profitability_path is not None
        else {}
    )
    broker_path = normalized_sources.get("broker_snapshot")
    broker = (
        _load_optional_json_object(Path(broker_path))
        if broker_path is not None
        else {}
    )
    capital_allocation_board = _build_capital_allocation_board(
        baseline=baseline,
        order_intents=order_intents,
        profitability_acceptance=profitability,
        execution_ledger_surface=execution_ledger_surface,
        broker_snapshot=broker,
        dynamic_tf_policy_sources=_dynamic_tf_packet_source_descriptors(
            sources
        ),
        pair_charts_path=normalized_sources.get("pair_charts"),
        calendar_path=normalized_sources.get("calendar"),
        strategy_profile_path=normalized_sources.get("strategy_profile"),
        as_of=prepared_at,
    )
    allocation_board_sha = canonical_json_sha256(capital_allocation_board)
    projection_calibration = projection_calibration_evidence(
        normalized_sources.get("projection_ledger"),
        scopes=_projection_calibration_scopes_for_market_read(
            baseline=baseline,
            capital_allocation_board=capital_allocation_board,
        ),
        now=prepared_at,
    )
    projection_source = sources.get("projection_ledger")
    if isinstance(projection_source, Mapping):
        material_sources["projection_ledger"] = {
            "path": projection_source.get("path"),
            "exists": projection_source.get("exists"),
            "material_contract": PROJECTION_CALIBRATION_EVIDENCE_CONTRACT,
            "calibration_evidence_sha256": projection_calibration.get(
                "evidence_sha256"
            ),
        }

    # These two files rotate while the rest of a production packet is being
    # hashed and its allocation/calibration surfaces are rebuilt.  Re-read
    # their semantic state last so a 20-30 second reconstruction does not
    # publish the early view of a receipt/watchdog that changed mid-build.
    guardian_path = normalized_sources.get("guardian_action_receipt")
    if guardian_path is not None:
        # ``prepared_at`` is captured before the allocation/calibration work.
        # In production the receipt may rotate while that work runs, so judge
        # its clocks against the instant its final bytes are read.  Explicit
        # test/replay clocks remain deterministic and keep using prepared_at.
        guardian_as_of = None if use_runtime_guardian_clock else prepared_at
        guardian_action_receipt_material = (
            guardian_action_receipt_scope_material(
                Path(guardian_path),
                baseline=baseline,
                as_of=guardian_as_of,
            )
        )
        guardian_source = sources.get("guardian_action_receipt")
        guardian_source = (
            guardian_source if isinstance(guardian_source, Mapping) else {}
        )
        material_sources["guardian_action_receipt"] = {
            "path": guardian_source.get("path") or str(guardian_path),
            "exists": Path(guardian_path).is_file(),
            "material_contract": GUARDIAN_ACTION_RECEIPT_MATERIAL_CONTRACT,
            "scope_state_sha256": canonical_json_sha256(
                guardian_action_receipt_material
            ),
        }
    watchdog_path = normalized_sources.get("qr_trader_run_watchdog")
    if watchdog_path is not None:
        watchdog_material = _watchdog_material_payload(Path(watchdog_path))
        watchdog_source = sources.get("qr_trader_run_watchdog")
        watchdog_source = (
            watchdog_source if isinstance(watchdog_source, Mapping) else {}
        )
        material_sources["qr_trader_run_watchdog"] = {
            "path": watchdog_source.get("path") or str(watchdog_path),
            "exists": Path(watchdog_path).is_file(),
            "material_contract": WATCHDOG_MATERIAL_CONTRACT,
            "safety_state_sha256": canonical_json_sha256(watchdog_material),
        }
    material = {
        "schema_version": MARKET_READ_OVERLAY_SCHEMA_VERSION,
        "baseline_sha256": baseline_sha256,
        "sources": material_sources,
        "capital_allocation_board_sha256": allocation_board_sha,
        "projection_calibration_evidence_sha256": (
            projection_calibration.get("evidence_sha256")
        ),
    }
    packet_sha = canonical_json_sha256(material)
    overrides_path = normalized_sources.get("trader_overrides")
    overrides = _load_optional_json_object(Path(overrides_path)) if overrides_path is not None else {}
    return {
        **material,
        "generated_at_utc": prepared_at.isoformat(),
        "evidence_packet_sha256": packet_sha,
        "baseline": {
            "action": baseline.get("action"),
            "selected_lane_id": baseline.get("selected_lane_id"),
            "selected_lane_ids": list(baseline.get("selected_lane_ids") or []),
            "generated_at_utc": baseline.get("generated_at_utc"),
            "market_read_first": baseline.get("market_read_first"),
        },
        "source_paths": {name: str(path) for name, path in sorted(normalized_sources.items())},
        "source_metadata": sources,
        "guardian_action_receipt_material": (
            guardian_action_receipt_material
            if guardian_action_receipt_material is not None
            else {"parse_status": "SOURCE_NOT_CONFIGURED"}
        ),
        "quote_basis_by_pair": quote_basis_by_pair_from_broker_payload(broker),
        "prior_market_read_review": (
            overrides.get("market_read_review")
            if isinstance(overrides.get("market_read_review"), dict)
            else {}
        ),
        "recent_resolved_predictions": recent_predictions,
        "forecast_replay_scorecard": forecast_replay_scorecard,
        "projection_calibration_evidence": projection_calibration,
        "execution_ledger_allocation_surface": execution_ledger_surface,
        "capital_allocation_board": capital_allocation_board,
        "capital_allocation_board_sha256": allocation_board_sha,
        "forecast_context_scope": capital_allocation_board.get(
            "forecast_context_scope"
        ),
        "forecast_context": deepcopy(
            capital_allocation_board.get("forecast_context")
        ),
        "contract": {
            "overlay_author": CODEX_MARKET_READ_AUTHOR,
            "allowed_overlay_fields": sorted(OVERLAY_ALLOWED_FIELDS),
            "execution_fields_are_immutable_except_bounded_size_reduction": True,
            "capital_allocation_required": True,
            "capital_allocation_cannot_exceed_base_units": True,
            "overlay_grants_live_permission": False,
            "directional_and_range_geometry_must_be_numeric": True,
            "forecast_replay_scorecard_is_read_only": True,
            "forecast_replay_scorecard_grants_live_permission": False,
            "projection_calibration_is_read_only": True,
            "projection_calibration_grants_live_permission": False,
            "projection_calibration_may_upgrade_non_trade_to_trade": False,
            "projection_calibration_insufficient_sample_edge_forbidden": True,
            "market_read_first": market_read_contract_payload(),
        },
    }


def _projection_calibration_scopes_for_market_read(
    *,
    baseline: Mapping[str, Any],
    capital_allocation_board: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    scope_kind = str(
        capital_allocation_board.get("forecast_context_scope") or ""
    ).strip().upper()
    baseline_action = str(baseline.get("action") or "").strip().upper()
    if baseline_action == "TRADE" and scope_kind != "SELECTED_LANE":
        return []
    context = capital_allocation_board.get("forecast_context")
    context = context if isinstance(context, Mapping) else {}
    pair = str(context.get("pair") or "").strip().upper()
    direction = _forecast_direction_label(context.get("direction"))
    technical_context = context.get("technical_context")
    if scope_kind == "SELECTED_LANE":
        lane = capital_allocation_board.get("selected_lane")
        lane = lane if isinstance(lane, Mapping) else {}
        forecast = lane.get("forecast")
        forecast = forecast if isinstance(forecast, Mapping) else {}
        lane_pair = str(lane.get("pair") or "").strip().upper()
        lane_direction = _forecast_direction_label(forecast.get("direction"))
        expected = (
            f"directional_forecast_{lane_direction.lower()}"
            if lane_direction in {"UP", "DOWN", "RANGE"}
            else None
        )
        calibration = str(forecast.get("calibration_name") or "").strip().lower()
        technical = technical_context if isinstance(technical_context, Mapping) else {}
        if (
            lane_pair not in DEFAULT_TRADER_PAIRS
            or pair != lane_pair
            or direction != lane_direction
            or calibration != expected
            or technical.get("status") != "VALID"
        ):
            return []
    elif scope_kind != "FORCED_PREDICTION":
        pair = ""
        direction = None
        technical_context = None
    if (
        pair not in DEFAULT_TRADER_PAIRS
        or direction not in {"UP", "DOWN", "RANGE"}
    ):
        if baseline_action == "TRADE":
            return []
        market_read = baseline.get("market_read_first")
        market_read = market_read if isinstance(market_read, Mapping) else {}
        forced = market_read.get("best_trade_if_forced")
        forced = forced if isinstance(forced, Mapping) else {}
        pair = str(forced.get("pair") or "").strip().upper()
        direction = _forecast_direction_label(forced.get("direction"))
        technical_context = None
    if pair not in DEFAULT_TRADER_PAIRS or direction not in {"UP", "DOWN", "RANGE"}:
        return []
    return [
        {
            "pair": pair,
            "direction": direction,
            "regime": _projection_regime_from_technical_context(
                technical_context
            ),
        }
    ]


def projection_calibration_scopes_from_lanes(
    lanes: list[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Extract canonical directional selectors from current GPT lane packets."""

    scopes: list[Mapping[str, Any]] = []
    seen: set[tuple[str, str, str | None]] = set()
    for lane in lanes:
        if (
            not isinstance(lane, Mapping)
            or str(lane.get("status") or "").strip().upper() != "LIVE_READY"
        ):
            continue
        forecast = lane.get("forecast")
        forecast = forecast if isinstance(forecast, Mapping) else {}
        direction = _forecast_direction_label(forecast.get("forecast_direction"))
        pair = str(lane.get("pair") or "").strip().upper()
        expected_calibration = (
            f"directional_forecast_{direction.lower()}" if direction else None
        )
        calibration_name = str(
            forecast.get("forecast_directional_calibration_name") or ""
        ).strip().lower()
        if (
            pair not in DEFAULT_TRADER_PAIRS
            or direction not in {"UP", "DOWN", "RANGE"}
            or calibration_name != expected_calibration
        ):
            continue
        regime = _projection_regime_from_technical_context(
            forecast.get("technical_context")
        )
        key = (pair, direction, regime)
        if key in seen:
            continue
        seen.add(key)
        scopes.append({"pair": pair, "direction": direction, "regime": regime})
        if len(scopes) >= 4:
            break
    return scopes


def _projection_regime_from_technical_context(value: Any) -> str | None:
    envelope = value if isinstance(value, Mapping) else {}
    body = envelope.get("technical_context_v1")
    body = body if isinstance(body, Mapping) else {}
    regime = body.get("regime")
    regime = regime if isinstance(regime, Mapping) else {}
    return _projection_calibration_regime(
        regime.get("primary") or regime.get("dominant")
    )


def _forecast_replay_scorecard(
    path: Path | None,
    *,
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    """Expose bounded replay diagnostics directly to the GPT market read.

    The full replay artifact can be multi-megabyte. The packet binds its file
    hash as normal evidence, but gives GPT only the numeric slices needed to
    challenge a current directional thesis. This scorecard is diagnostic and
    cannot authorize a lane or increase units.
    """

    selected_pair, selected_direction = _baseline_replay_scope(baseline)
    base = {
        "contract": FORECAST_REPLAY_SCORECARD_CONTRACT,
        "source_path": str(path) if path is not None else None,
        "selected_pair": selected_pair,
        "selected_direction": selected_direction,
        "read_only": True,
        "live_permission": False,
    }
    if path is None or not path.exists() or not path.is_file():
        return {**base, "status": "MISSING"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {**base, "status": "MALFORMED"}
    if not isinstance(payload, Mapping):
        return {**base, "status": "MALFORMED"}

    segments = payload.get("segments")
    segments = segments if isinstance(segments, Mapping) else {}
    raw_pair_direction_rows = segments.get("by_pair_direction")
    raw_pair_direction_count = (
        len(raw_pair_direction_rows)
        if isinstance(raw_pair_direction_rows, list)
        else 0
    )
    pair_direction_rows = _forecast_replay_rows(
        raw_pair_direction_rows,
        dimension_fields=("pair", "direction"),
        limit=64,
    )
    confidence_rows = _forecast_replay_rows(
        segments.get("by_confidence"),
        dimension_fields=("confidence_bucket",),
        limit=16,
    )
    evaluated_rows = _nonnegative_int(payload.get("evaluated_rows"))
    confidence_rows_accounted = sum(
        int(row["n"])
        for row in confidence_rows
        if isinstance(row.get("n"), int)
    )
    confidence_rows_unreported = (
        max(0, evaluated_rows - confidence_rows_accounted)
        if evaluated_rows is not None
        else None
    )
    pair_filter = sorted(
        {
            str(pair or "").strip().upper()
            for pair in (payload.get("pair_filter") or [])
            if str(pair or "").strip()
        }
    )
    selected_pair_direction = next(
        (
            row
            for row in pair_direction_rows
            if row.get("pair") == selected_pair
            and row.get("direction") == selected_direction
        ),
        None,
    )
    return {
        **base,
        "status": "VALID",
        "proof_eligible": False,
        "proof_status": (
            "UNVERIFIED_LEGACY"
            if not isinstance(payload.get("selection_contract"), Mapping)
            else "DIAGNOSTIC_ONLY"
        ),
        "proof_blockers": [
            "REPLAY_SCORECARD_DOES_NOT_GRANT_LIVE_PERMISSION",
            *(
                ["MISSING_SELECTION_CONTRACT"]
                if not isinstance(payload.get("selection_contract"), Mapping)
                else []
            ),
        ],
        "generated_at_utc": payload.get("generated_at_utc"),
        "source": payload.get("source"),
        "truth_source": payload.get("truth_source"),
        "granularity": payload.get("granularity"),
        "scope": {
            "pair_filter": pair_filter,
            "history_pairs": _nonnegative_int(payload.get("history_pairs")),
            "evaluated_rows": evaluated_rows,
            "forecast_time_from_utc": payload.get("forecast_time_from_utc"),
            "forecast_time_to_utc": payload.get("forecast_time_to_utc"),
            "independent_non_overlap": payload.get("independent_non_overlap"),
            "independent_selected_rows": _nonnegative_int(
                payload.get("independent_selected_rows")
            ),
            "pair_direction_rows": raw_pair_direction_count,
            "pair_direction_rows_included": len(pair_direction_rows),
            "pair_direction_rows_truncated": raw_pair_direction_count
            > len(pair_direction_rows),
            "confidence_segment_rows_accounted": confidence_rows_accounted,
            "confidence_segment_rows_unreported": confidence_rows_unreported,
            "confidence_segment_complete": confidence_rows_unreported == 0,
            "technical_context_missing_rows": _nonnegative_int(
                payload.get("technical_context_missing_rows")
            ),
            "technical_context_invalid_rows": _nonnegative_int(
                payload.get("technical_context_invalid_rows")
            ),
            "technical_context_incomplete_rows": _nonnegative_int(
                payload.get("technical_context_incomplete_rows")
            ),
        },
        "selection_contract": _forecast_replay_selection_contract(
            payload.get("selection_contract")
        ),
        "experiment": _forecast_replay_experiment(payload.get("experiment")),
        "global": _forecast_replay_metrics(payload.get("summary")),
        "selected_pair_direction": selected_pair_direction,
        "selected_coverage_status": (
            "COVERED"
            if selected_pair_direction is not None
            else (
                "NOT_COVERED"
                if selected_pair is not None
                and pair_filter
                and selected_pair not in pair_filter
                else "NO_MATCH"
            )
        ),
        "by_pair_direction": pair_direction_rows,
        "by_confidence": confidence_rows,
        "by_horizon": _forecast_replay_rows(
            segments.get("by_horizon"),
            dimension_fields=("horizon_bucket",),
            limit=16,
        ),
        "by_primary_driver_family": _forecast_replay_rows(
            segments.get("by_primary_driver_family"),
            dimension_fields=("primary_driver_family",),
            limit=16,
        ),
        "by_driver_family_presence": _forecast_replay_rows(
            segments.get("by_driver_family_presence"),
            dimension_fields=("driver_family",),
            limit=32,
        ),
        "by_primary_driver_family_direction": _forecast_replay_rows(
            segments.get("by_primary_driver_family_direction"),
            dimension_fields=("primary_driver_family", "direction"),
            limit=32,
        ),
        "by_raw_confidence": _forecast_replay_rows(
            segments.get("by_raw_confidence"),
            dimension_fields=("raw_confidence_bucket",),
            limit=16,
        ),
        "by_score_margin": _forecast_replay_rows(
            segments.get("by_score_margin"),
            dimension_fields=("score_margin_bucket",),
            limit=16,
        ),
        "by_range_competition": _forecast_replay_rows(
            segments.get("by_range_competition"),
            dimension_fields=("range_competition",),
            limit=16,
        ),
        "by_against_driver_family_presence": _forecast_replay_rows(
            segments.get("by_against_driver_family_presence"),
            dimension_fields=("against_driver_family",),
            limit=32,
        ),
        "by_session": _forecast_replay_rows(
            segments.get("by_session"),
            dimension_fields=("utc_session_bucket",),
            limit=16,
        ),
        "by_technical_context_completeness": _forecast_replay_rows(
            segments.get("by_technical_context_completeness"),
            dimension_fields=("technical_context_complete",),
            limit=4,
        ),
        "by_technical_regime": _forecast_replay_rows(
            segments.get("by_technical_regime"),
            dimension_fields=("technical_regime",),
            limit=16,
        ),
        "by_technical_atr_band": _forecast_replay_rows(
            segments.get("by_technical_atr_band"),
            dimension_fields=("technical_atr_band",),
            limit=8,
        ),
        "by_technical_spread_band": _forecast_replay_rows(
            segments.get("by_technical_spread_band"),
            dimension_fields=("technical_spread_band",),
            limit=8,
        ),
        "by_technical_range_location_24h": _forecast_replay_rows(
            segments.get("by_technical_range_location_24h"),
            dimension_fields=("technical_range_location_24h",),
            limit=8,
        ),
        "by_technical_structure_alignment": _forecast_replay_rows(
            segments.get("by_technical_structure_alignment"),
            dimension_fields=("technical_structure_alignment",),
            limit=8,
        ),
        "exit_policy_validation": _forecast_replay_exit_policy_validation(
            payload.get("train_validation_exit_selection")
        ),
    }


def _baseline_replay_scope(baseline: Mapping[str, Any]) -> tuple[str | None, str | None]:
    market_read = baseline.get("market_read_first")
    market_read = market_read if isinstance(market_read, Mapping) else {}
    forced = market_read.get("best_trade_if_forced")
    forced = forced if isinstance(forced, Mapping) else {}
    pair = str(forced.get("pair") or "").strip().upper() or None
    direction_text = str(forced.get("direction") or "").strip().upper()
    direction = {
        "LONG": "UP",
        "SHORT": "DOWN",
        "UP": "UP",
        "DOWN": "DOWN",
    }.get(direction_text)
    return pair, direction


def _forecast_replay_rows(
    value: Any,
    *,
    dimension_fields: tuple[str, ...],
    limit: int,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in value[:limit]:
        if not isinstance(item, Mapping):
            continue
        row = {}
        for field in dimension_fields:
            dimension = item.get(field)
            row[field] = (
                dimension
                if isinstance(dimension, bool)
                else str(dimension or "").strip().upper()
            )
        row.update(_forecast_replay_metrics(item))
        rows.append(row)
    return rows


def _forecast_replay_metrics(value: Any) -> dict[str, float | int | None]:
    item = value if isinstance(value, Mapping) else {}
    metrics: dict[str, float | int | None] = {}
    for field in FORECAST_REPLAY_METRIC_FIELDS:
        parsed = _optional_float(item.get(field))
        if field == "n" and parsed is not None and parsed >= 0 and parsed.is_integer():
            metrics[field] = int(parsed)
        else:
            metrics[field] = parsed
    if metrics.get("hit_wilson95_lower") is None:
        metrics["hit_wilson95_lower"] = hit_rate_wilson_lower(
            metrics.get("hit_rate"),
            metrics.get("n") if isinstance(metrics.get("n"), int) else None,
        )
    return metrics


def _nonnegative_int(value: Any) -> int | None:
    parsed = _optional_float(value)
    if parsed is None or parsed < 0 or not parsed.is_integer():
        return None
    return int(parsed)


def _forecast_replay_selection_contract(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return {
        key: deepcopy(value.get(key))
        for key in (
            "audit_mode",
            "proof_eligible",
            "proof_blockers",
            "forecast_from_utc_inclusive",
            "forecast_to_utc_exclusive",
            "forecast_window_start_utc",
            "forecast_window_end_utc",
            "independent_non_overlap_per_pair",
            "independence_limit",
            "confidence_field",
            "min_confidence",
        )
        if key in value
    }


def _forecast_replay_experiment(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    semantics = value.get("semantics")
    return {
        "schema_version": value.get("schema_version"),
        "status": value.get("status"),
        "audit_mode": value.get("audit_mode"),
        "evaluator_sha256": value.get("evaluator_sha256"),
        "experiment_id": value.get("experiment_id"),
        "semantics_version": (
            semantics.get("version") if isinstance(semantics, Mapping) else None
        ),
    }


def _forecast_replay_exit_policy_validation(value: Any) -> dict[str, Any] | None:
    """Expose train-selected exit geometry and its untouched validation result."""

    if not isinstance(value, Mapping):
        return None

    def result(item: Any) -> dict[str, float | int | None] | None:
        if not isinstance(item, Mapping):
            return None
        parsed: dict[str, float | int | None] = {}
        for field in (
            "n",
            "take_profit_pips",
            "stop_loss_pips",
            "avg_realized_pips",
            "profit_factor",
            "win_rate",
            "tp_rate",
            "sl_rate",
            "timeout_rate",
        ):
            number = _optional_float(item.get(field))
            if field == "n" and number is not None and number >= 0 and number.is_integer():
                parsed[field] = int(number)
            else:
                parsed[field] = number
        return parsed

    return {
        "status": value.get("status"),
        "train_n": _nonnegative_int(value.get("train_n")),
        "validation_n": _nonnegative_int(value.get("validation_n")),
        "validation_start_utc": value.get("validation_start_utc"),
        "selected_by_train": result(value.get("selected_by_train")),
        "validation": result(value.get("validation")),
    }


def _rebuild_canonical_forecast_context(
    result: Mapping[str, Any] | None,
    *,
    pair_charts_path: Path | None,
    calendar_path: Path | None,
    strategy_profile_path: Path | None,
    broker_snapshot: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Rebuild context only from packet-authoritative chart/source/quote facts."""

    if (
        not isinstance(result, Mapping)
        or pair_charts_path is None
        or calendar_path is None
        or strategy_profile_path is None
    ):
        return None
    intent = result.get("intent")
    if not isinstance(intent, Mapping):
        return None
    pair = str(intent.get("pair") or "").strip().upper()
    if pair not in DEFAULT_TRADER_PAIRS:
        return None
    try:
        payload = json.loads(pair_charts_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    charts = payload.get("charts") if isinstance(payload, Mapping) else None
    matches = [
        chart
        for chart in charts if isinstance(chart, Mapping)
        and str(chart.get("pair") or "").strip().upper() == pair
    ] if isinstance(charts, list) else []
    if len(matches) != 1:
        return None
    canonical_chart = dict(matches[0])
    if isinstance(payload, Mapping):
        packet_generated_at = payload.get("generated_at_utc")
        if packet_generated_at is not None:
            canonical_chart.setdefault(
                "generated_at_utc",
                packet_generated_at,
            )
    quotes = broker_snapshot.get("quotes")
    quote = quotes.get(pair) if isinstance(quotes, Mapping) else None
    if not isinstance(quote, Mapping):
        return None
    bid = _strict_positive_number(quote.get("bid"))
    ask = _strict_positive_number(quote.get("ask"))
    if bid is None or ask is None or ask <= bid:
        return None
    try:
        evaluated_at = _parse_utc(
            broker_snapshot.get("fetched_at_utc"),
            code="DYNAMIC_TF_POLICY_CANONICAL_CLOCK_INVALID",
        )
        spread_pips = (ask - bid) * float(instrument_pip_factor(pair))
        return build_forecast_technical_context(
            canonical_chart,
            pair=pair,
            current_price=(bid + ask) / 2.0,
            spread_pips=spread_pips,
            calendar_path=calendar_path,
            strategy_profile_path=strategy_profile_path,
            now_utc=evaluated_at,
        )
    except (
        MarketReadOverlayError,
        OSError,
        TypeError,
        ValueError,
        OverflowError,
    ):
        return None


def _build_capital_allocation_board(
    *,
    baseline: Mapping[str, Any],
    order_intents: Mapping[str, Any],
    profitability_acceptance: Mapping[str, Any],
    execution_ledger_surface: Mapping[str, Any],
    broker_snapshot: Mapping[str, Any],
    dynamic_tf_policy_sources: Mapping[str, Any],
    pair_charts_path: Path | None,
    calendar_path: Path | None,
    strategy_profile_path: Path | None,
    as_of: datetime,
) -> dict[str, Any]:
    baseline_action = str(baseline.get("action") or "").strip().upper()
    selected_lane_id = str(baseline.get("selected_lane_id") or "").strip() or None
    selected_matches = _order_intent_lane_matches(order_intents, selected_lane_id)
    selected_result: Mapping[str, Any] | None = (
        selected_matches[0] if len(selected_matches) == 1 else None
    )
    selected_lane_match_count = len(selected_matches)

    metrics = (
        profitability_acceptance.get("metrics")
        if isinstance(profitability_acceptance.get("metrics"), Mapping)
        else {}
    )
    capture = (
        metrics.get("capture_economics")
        if isinstance(metrics.get("capture_economics"), Mapping)
        else {}
    )
    month_scale_finding = next(
        (
            item
            for item in profitability_acceptance.get("findings", []) or []
            if isinstance(item, Mapping)
            and str(item.get("code") or "")
            == "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE"
        ),
        {},
    )
    month_scale_evidence = (
        month_scale_finding.get("evidence")
        if isinstance(month_scale_finding.get("evidence"), Mapping)
        else {}
    )
    global_profitability = {
        "status": profitability_acceptance.get("status"),
        "blocker_codes": [
            str(item.get("code"))
            for item in profitability_acceptance.get("findings", []) or []
            if isinstance(item, Mapping) and item.get("code")
        ],
        "capture_economics": {
            "status": capture.get("status"),
            "overall": _allocation_metric_fields(capture.get("overall")),
            "market_close": _allocation_metric_fields(capture.get("market_close")),
            "take_profit": _allocation_metric_fields(capture.get("take_profit")),
        },
        "month_scale_replay": {
            "status": (
                "NEGATIVE"
                if month_scale_finding
                else "NO_ACTIVE_NEGATIVE_BLOCKER"
            ),
            "active_counterfactual_profit_capture_pl_jpy": _optional_float(
                month_scale_evidence.get("active_counterfactual_profit_capture_pl_jpy")
            ),
            "counterfactual_profit_capture_delta_jpy": _optional_float(
                month_scale_evidence.get("counterfactual_profit_capture_delta_jpy")
            ),
            "repair_replay_counterfactual_pl_jpy": _optional_float(
                month_scale_evidence.get("repair_replay_counterfactual_pl_jpy")
            ),
        },
    }

    current_exact_net = exact_vehicle_metrics_from_surface(
        execution_ledger_surface,
        field="exact_vehicle_net",
    )
    current_exact_tp = exact_vehicle_metrics_from_surface(
        execution_ledger_surface,
        field="exact_vehicle_take_profit",
    )
    canonical_forecast_context = _rebuild_canonical_forecast_context(
        selected_result,
        pair_charts_path=pair_charts_path,
        calendar_path=calendar_path,
        strategy_profile_path=strategy_profile_path,
        broker_snapshot=broker_snapshot,
    )
    canonical_forecast_context_sha256 = (
        str(canonical_forecast_context.get("context_sha256") or "")
        if isinstance(canonical_forecast_context, Mapping)
        else None
    )
    lane = _capital_allocation_lane(
        selected_result,
        current_exact_vehicle_net_metrics=current_exact_net,
        current_exact_vehicle_tp_metrics=current_exact_tp,
        execution_ledger_surface_sha256=str(
            execution_ledger_surface.get("allocation_surface_sha256") or ""
        ),
        account_nav_jpy=_broker_snapshot_nav_jpy(broker_snapshot),
        broker_snapshot=broker_snapshot,
        dynamic_tf_policy_sources=dynamic_tf_policy_sources,
        canonical_forecast_context_sha256=(
            canonical_forecast_context_sha256
        ),
        execution_cost_floor=(
            execution_cost_floor_from_surface(
                execution_ledger_surface,
                exact_key=_allocation_lane_exact_key(selected_result),
                as_of=as_of,
            )
            if _allocation_lane_exact_key(selected_result) is not None
            else None
        ),
    )
    forecast_context_scope, forecast_context = _allocation_board_forecast_context(
        baseline=baseline,
        order_intents=order_intents,
        selected_result=selected_result,
        selected_lane=lane,
        selected_lane_id=selected_lane_id,
        selected_lane_match_count=selected_lane_match_count,
    )
    return {
        "schema_version": 2,
        "baseline_action": baseline_action,
        "selected_lane_id": selected_lane_id,
        "selected_lane_match_count": selected_lane_match_count,
        "selected_lane_unique": selected_lane_match_count == 1,
        "order_intents_generated_at_utc": order_intents.get("generated_at_utc"),
        "selected_lane": lane,
        "dynamic_tf_policy_sources": deepcopy(
            dict(dynamic_tf_policy_sources)
        ),
        "canonical_forecast_context_sha256": (
            canonical_forecast_context_sha256
        ),
        "forecast_context_scope": forecast_context_scope,
        "forecast_context": forecast_context,
        "execution_ledger": {
            "parse_status": execution_ledger_surface.get("parse_status"),
            "coverage_start_utc": execution_ledger_surface.get(
                "coverage_start_utc"
            ),
            "selected_scope_key": execution_ledger_surface.get(
                "selected_scope_key"
            ),
            "allocation_surface_sha256": execution_ledger_surface.get(
                "allocation_surface_sha256"
            ),
        },
        "global_profitability": global_profitability,
        "allocation_rule": (
            "ALLOCATE only the deterministic selected lane, never exceed base_units, "
            "and only when allocation_eligible is true with either positive_edge_proven "
            "or the separately labeled M15 recovery edge_collection_proven route. "
            "M15_RECOVERY_EDGE_COLLECTION is evidence collection, not positive edge proof; "
            "use NO_TRADE when neither proof route passes or the numeric thesis is contradicted."
        ),
    }


def _allocation_board_forecast_context(
    *,
    baseline: Mapping[str, Any],
    order_intents: Mapping[str, Any],
    selected_result: Mapping[str, Any] | None,
    selected_lane: Mapping[str, Any] | None,
    selected_lane_id: str | None,
    selected_lane_match_count: int,
) -> tuple[str, dict[str, Any]]:
    """Select one pair/direction-bound point-in-time forecast context.

    All candidates belong to the same generated order-intent snapshot.  A
    forced-prediction lookup accepts duplicate lanes only when their verified
    context SHA is identical; competing SHAs are UNKNOWN instead of being
    resolved by append order or score.
    """

    generated_at = order_intents.get("generated_at_utc")
    baseline_action = str(baseline.get("action") or "").strip().upper()
    if (
        baseline_action == "TRADE"
        and selected_lane_id
        and selected_lane_match_count != 1
    ):
        reason = (
            "FORECAST_TECHNICAL_CONTEXT_SELECTED_LANE_NOT_FOUND"
            if selected_lane_match_count == 0
            else "FORECAST_TECHNICAL_CONTEXT_SELECTED_LANE_DUPLICATE"
        )
        return "SELECTED_LANE_INVALID", {
            "pair": None,
            "direction": None,
            "source_lane_id": selected_lane_id,
            "source_generated_at_utc": generated_at,
            "candidate_count": selected_lane_match_count,
            "valid_candidate_count": 0,
            "invalid_candidate_count": selected_lane_match_count,
            "distinct_context_sha256_count": 0,
            "technical_context": unknown_forecast_technical_context_evidence(reason),
        }
    if isinstance(selected_result, Mapping):
        intent = (
            selected_result.get("intent")
            if isinstance(selected_result.get("intent"), Mapping)
            else {}
        )
        metadata = (
            intent.get("metadata")
            if isinstance(intent.get("metadata"), Mapping)
            else {}
        )
        recovery = (
            selected_lane.get("m15_recovery")
            if isinstance(selected_lane, Mapping)
            and isinstance(selected_lane.get("m15_recovery"), Mapping)
            else {}
        )
        recovery_material = (
            recovery.get("material")
            if isinstance(recovery.get("material"), Mapping)
            else None
        )
        if (
            recovery.get("claimed") is True
            and recovery.get("valid") is True
            and recovery.get("status") == M15_RECOVERY_ALLOCATION_STATUS
            and isinstance(recovery_material, Mapping)
            and _content_addressed_mapping_valid(
                recovery_material,
                digest_key="material_sha256",
            )
            and recovery.get("material_sha256")
            == recovery_material.get("material_sha256")
        ):
            scoped_body = {
                "contract": M15_RECOVERY_ALLOCATION_CONTEXT_CONTRACT,
                "status": M15_RECOVERY_ALLOCATION_STATUS,
                "pair": selected_lane.get("pair"),
                "direction": (
                    (selected_lane.get("forecast") or {}).get("direction")
                    if isinstance(selected_lane.get("forecast"), Mapping)
                    else None
                ),
                "side": selected_lane.get("side"),
                "method": selected_lane.get("method"),
                "order_type": _m15_recovery_contract_order_type(
                    selected_lane.get("order_type")
                ),
                "source_lane_id": str(selected_result.get("lane_id") or "")
                or None,
                "source_generated_at_utc": generated_at,
                "forecast_cycle_id": recovery.get("forecast_cycle_id"),
                "source_receipt_sha256": recovery.get(
                    "source_receipt_sha256"
                ),
                "forecast_binding_sha256": recovery.get(
                    "forecast_binding_sha256"
                ),
                "lane_binding_sha256": recovery.get("lane_binding_sha256"),
                "context_timeframes": list(M15_RECOVERY_CONTEXT_TIMEFRAMES),
                "positive_rotation_mode": recovery.get("mode"),
                "edge_basis": selected_lane.get("edge_basis"),
                "recovery_material_sha256": recovery.get("material_sha256"),
            }
            return M15_RECOVERY_ALLOCATION_CONTEXT_SCOPE, {
                **scoped_body,
                "recovery_context_sha256": canonical_json_sha256(scoped_body),
            }
        pair = str(intent.get("pair") or "").strip().upper() or None
        forecast_binding = _directional_forecast_trade_binding(
            side=intent.get("side"),
            forecast_direction=metadata.get("forecast_direction"),
            calibration_name=metadata.get(
                "forecast_directional_calibration_name"
            ),
            method=(
                intent.get("market_context", {}).get("method")
                if isinstance(intent.get("market_context"), Mapping)
                else None
            ),
        )
        side_direction = forecast_binding["side_direction"]
        forecast_direction = forecast_binding["forecast_direction"]
        lane_forecast = (
            selected_lane.get("forecast")
            if isinstance(selected_lane, Mapping)
            and isinstance(selected_lane.get("forecast"), Mapping)
            else {}
        )
        evidence = normalize_forecast_technical_context_evidence(
            lane_forecast.get("technical_context"),
            pair=pair,
            current_price=metadata.get("forecast_current_price"),
        )
        if not forecast_binding["direction_side_aligned"]:
            evidence = unknown_forecast_technical_context_evidence(
                "FORECAST_TECHNICAL_CONTEXT_DIRECTION_MISMATCH"
            )
        elif not forecast_binding["calibration_identity_valid"]:
            evidence = unknown_forecast_technical_context_evidence(
                "FORECAST_TECHNICAL_CONTEXT_CALIBRATION_MISMATCH"
            )
        return "SELECTED_LANE", {
            "pair": pair,
            "direction": forecast_direction or side_direction,
            "source_lane_id": str(selected_result.get("lane_id") or "") or None,
            "source_generated_at_utc": generated_at,
            "candidate_count": 1,
            "valid_candidate_count": (
                1 if evidence.get("status") == "VALID" else 0
            ),
            "invalid_candidate_count": (
                0 if evidence.get("status") == "VALID" else 1
            ),
            "distinct_context_sha256_count": (
                1 if evidence.get("status") == "VALID" else 0
            ),
            "technical_context": evidence,
        }

    market_read = (
        baseline.get("market_read_first")
        if isinstance(baseline.get("market_read_first"), Mapping)
        else {}
    )
    forced = (
        market_read.get("best_trade_if_forced")
        if isinstance(market_read.get("best_trade_if_forced"), Mapping)
        else {}
    )
    pair = str(forced.get("pair") or "").strip().upper() or None
    direction = _forecast_direction_label(forced.get("direction"))
    if pair is None or direction is None:
        return "MISSING", {
            "pair": pair,
            "direction": direction,
            "source_lane_id": None,
            "source_generated_at_utc": generated_at,
            "candidate_count": 0,
            "valid_candidate_count": 0,
            "invalid_candidate_count": 0,
            "distinct_context_sha256_count": 0,
            "technical_context": unknown_forecast_technical_context_evidence(
                "FORECAST_TECHNICAL_CONTEXT_SCOPE_MISSING"
            ),
        }

    matching_count = 0
    valid_candidates: list[tuple[str, dict[str, Any]]] = []
    invalid_count = 0
    results = order_intents.get("results")
    for result in results if isinstance(results, list) else []:
        if not isinstance(result, Mapping):
            continue
        intent = (
            result.get("intent")
            if isinstance(result.get("intent"), Mapping)
            else {}
        )
        metadata = (
            intent.get("metadata")
            if isinstance(intent.get("metadata"), Mapping)
            else {}
        )
        if str(intent.get("pair") or "").strip().upper() != pair:
            continue
        if _forecast_direction_label(metadata.get("forecast_direction")) != direction:
            continue
        matching_count += 1
        if direction in {"UP", "DOWN"} and _forecast_direction_label(
            intent.get("side")
        ) != direction:
            invalid_count += 1
            continue
        evidence = normalize_forecast_technical_context_evidence(
            metadata.get("forecast_technical_context"),
            pair=pair,
            current_price=metadata.get("forecast_current_price"),
        )
        if evidence.get("status") != "VALID":
            invalid_count += 1
            continue
        valid_candidates.append((str(result.get("lane_id") or ""), evidence))

    distinct_shas = {
        str(evidence.get("context_sha256") or "")
        for _lane_id, evidence in valid_candidates
        if evidence.get("context_sha256")
    }
    if invalid_count > 0 or len(distinct_shas) != 1:
        reason = (
            "FORECAST_TECHNICAL_CONTEXT_CANDIDATE_INVALID"
            if invalid_count > 0
            else (
                "FORECAST_TECHNICAL_CONTEXT_NOT_FOUND"
                if not distinct_shas
                else "FORECAST_TECHNICAL_CONTEXT_AMBIGUOUS"
            )
        )
        return "FORCED_PREDICTION", {
            "pair": pair,
            "direction": direction,
            "source_lane_id": None,
            "source_generated_at_utc": generated_at,
            "candidate_count": matching_count,
            "valid_candidate_count": len(valid_candidates),
            "invalid_candidate_count": invalid_count,
            "distinct_context_sha256_count": len(distinct_shas),
            "technical_context": unknown_forecast_technical_context_evidence(reason),
        }

    valid_candidates.sort(key=lambda item: (item[0], item[1]["evidence_sha256"]))
    source_lane_id, evidence = valid_candidates[0]
    return "FORCED_PREDICTION", {
        "pair": pair,
        "direction": direction,
        "source_lane_id": source_lane_id or None,
        "source_generated_at_utc": generated_at,
        "candidate_count": matching_count,
        "valid_candidate_count": len(valid_candidates),
        "invalid_candidate_count": 0,
        "distinct_context_sha256_count": 1,
        "technical_context": evidence,
    }


def _forecast_direction_label(value: Any) -> str | None:
    normalized = _normalized_direction(value)
    if normalized == "LONG":
        return "UP"
    if normalized == "SHORT":
        return "DOWN"
    return normalized


def _normalized_order_vehicle(value: Any) -> str | None:
    order_type = str(value or "").strip().upper()
    vehicle = {
        "STOP_ENTRY": "STOP",
        "STOP-ENTRY": "STOP",
        "STOP_ORDER": "STOP",
        "LIMIT_ORDER": "LIMIT",
        "MARKET_ORDER": "MARKET",
    }.get(order_type, order_type)
    return vehicle if vehicle in {"LIMIT", "MARKET", "STOP"} else None


def _directional_forecast_trade_binding(
    *,
    side: Any,
    forecast_direction: Any,
    calibration_name: Any,
    method: Any,
) -> dict[str, Any]:
    """Return the lane-aware forecast/calibration binding for live sizing.

    Context integrity alone cannot authorize a lane: the verified body must be
    attached to the same directional thesis the order will execute.  Keep this
    one binding shared by board construction, numeric sizing, and the final
    capital-allocation verifier so pre-bounded HEDGE/SCOUT paths cannot bypass
    it at a different layer. RANGE is not coerced into UP/DOWN: only the two
    range-compatible methods can carry a side, and their family/proof direction
    is checked independently below.
    """

    side_text = str(side or "").strip().upper()
    direction_text = str(forecast_direction or "").strip().upper()
    method_text = str(method or "").strip().upper()
    side_direction = (
        "UP"
        if side_text == "LONG"
        else "DOWN"
        if side_text == "SHORT"
        else None
    )
    normalized_direction = (
        direction_text if direction_text in {"UP", "DOWN", "RANGE"} else None
    )
    expected_calibration_name = (
        f"directional_forecast_{normalized_direction.lower()}"
        if normalized_direction is not None
        else None
    )
    normalized_calibration_name = str(calibration_name or "").strip().lower()
    range_method_bound = bool(
        normalized_direction == "RANGE"
        and method_text in {"RANGE_ROTATION", "BREAKOUT_FAILURE"}
    )
    return {
        "side_direction": side_direction,
        "forecast_direction": normalized_direction,
        "method": method_text or None,
        "range_method_bound": range_method_bound,
        "direction_side_aligned": bool(
            side_direction is not None
            and (
                normalized_direction == side_direction
                or range_method_bound
            )
        ),
        "calibration_name": normalized_calibration_name or None,
        "expected_calibration_name": expected_calibration_name,
        "calibration_identity_valid": bool(
            expected_calibration_name is not None
            and normalized_calibration_name == expected_calibration_name
        ),
    }


def _dynamic_tf_policy_source_matches_packet(
    policy_source: object,
    packet_source: object,
) -> bool:
    if not isinstance(policy_source, Mapping) or not isinstance(
        packet_source, Mapping
    ):
        return False
    if set(packet_source) != {"path", "exists", "sha256", "size_bytes"}:
        return False
    policy_path = policy_source.get("path")
    packet_path = packet_source.get("path")
    if not isinstance(policy_path, str) or not isinstance(packet_path, str):
        return False
    try:
        normalized_policy_path = str(_normalize_source_path(policy_path))
        normalized_packet_path = str(_normalize_source_path(packet_path))
    except (OSError, RuntimeError, ValueError):
        return False
    if normalized_policy_path != normalized_packet_path:
        return False
    status = policy_source.get("status")
    exists = packet_source.get("exists")
    if not isinstance(exists, bool):
        return False
    if status == "MISSING":
        return bool(
            not exists
            and policy_source.get("sha256") is None
            and policy_source.get("byte_length") is None
            and packet_source.get("sha256") is None
            and packet_source.get("size_bytes") is None
        )
    if status == "NOT_APPLICABLE":
        return bool(
            policy_source.get("sha256") is None
            and policy_source.get("byte_length") is None
        )
    if status not in {
        "OK",
        "INVALID_JSON",
        "INVALID_ROOT",
        "LIMIT_EXCEEDED",
    } or not exists:
        return False
    policy_size = policy_source.get("byte_length")
    packet_size = packet_source.get("size_bytes")
    if (
        isinstance(policy_size, bool)
        or not isinstance(policy_size, int)
        or packet_size != policy_size
    ):
        return False
    policy_sha = policy_source.get("sha256")
    if policy_sha is not None and packet_source.get("sha256") != policy_sha:
        return False
    return True


def _dynamic_tf_policy_sources_bound(
    body: Mapping[str, Any],
    canonical_policy_sources: Mapping[str, Any] | None,
) -> bool:
    parent_source = body.get("dynamic_tf_policy_source_context")
    if not isinstance(parent_source, Mapping) or not isinstance(
        canonical_policy_sources, Mapping
    ):
        return False
    return bool(
        _dynamic_tf_policy_source_matches_packet(
            parent_source.get("news_source"),
            canonical_policy_sources.get("calendar"),
        )
        and _dynamic_tf_policy_source_matches_packet(
            parent_source.get("strategy_profile_source"),
            canonical_policy_sources.get("strategy_profile"),
        )
    )


def _regime_family_allocation_binding(
    *,
    technical_context: Mapping[str, Any],
    pair: Any,
    side: Any,
    method: Any,
    forecast_direction: Any,
    explicit_receipt_sha256: Any,
    explicit_selected_method: Any,
    explicit_family_direction: Any,
    range_tp_proven_breakout_failure: bool = False,
    canonical_policy_sources: Mapping[str, Any] | None = None,
    canonical_forecast_context_sha256: Any = None,
) -> dict[str, Any]:
    """Re-prove receipt/context/intent identity before fresh allocation."""

    body = (
        technical_context.get("technical_context_v1")
        if isinstance(technical_context.get("technical_context_v1"), Mapping)
        else {}
    )
    receipt = (
        body.get("regime_family_weighting")
        if isinstance(body.get("regime_family_weighting"), Mapping)
        else None
    )
    current_context_bound = forecast_technical_context_is_current_for_allocation(
        body
    )
    canonical_policy_sources_bound = _dynamic_tf_policy_sources_bound(
        body,
        canonical_policy_sources,
    )
    expected_context_sha = str(canonical_forecast_context_sha256 or "")
    canonical_forecast_context_bound = bool(
        SHA256_RE.fullmatch(expected_context_sha)
        and str(body.get("context_sha256") or "") == expected_context_sha
    )
    valid, receipt_error = verify_regime_family_weighting_receipt(
        receipt,
        pair=str(pair or "").strip().upper() or None,
    )
    source = (
        receipt.get("source_identity")
        if valid and isinstance(receipt, Mapping)
        and isinstance(receipt.get("source_identity"), Mapping)
        else {}
    )
    aggregate = (
        receipt.get("aggregate")
        if valid and isinstance(receipt, Mapping)
        and isinstance(receipt.get("aggregate"), Mapping)
        else {}
    )
    receipt_sha = str(receipt.get("receipt_sha256") or "") if valid and isinstance(receipt, Mapping) else ""
    selected_method = source.get("selected_method")
    family_direction = str(aggregate.get("direction") or "") or None
    directional_coverage = _strict_positive_number(
        aggregate.get("directional_coverage_weight")
    )
    explicit_sha = str(explicit_receipt_sha256 or "")
    method_text = str(method or "").strip().upper()
    explicit_method = (
        str(explicit_selected_method).strip().upper()
        if explicit_selected_method is not None
        else None
    )
    direction_consistent, direction_reason = forecast_direction_consistency(
        receipt,
        forecast_direction=forecast_direction,
    )
    normalized_forecast_direction = str(forecast_direction or "").strip().upper()
    side_text = str(side or "").strip().upper()
    side_direction = (
        "UP"
        if side_text == "LONG"
        else "DOWN"
        if side_text == "SHORT"
        else None
    )
    failed_break = (
        body.get("m5_failed_break_evidence")
        if isinstance(body.get("m5_failed_break_evidence"), Mapping)
        else None
    )
    failed_break_side = failed_break_direction(failed_break)
    failed_break_forecast_direction = (
        "UP"
        if failed_break_side == "LONG"
        else "DOWN"
        if failed_break_side == "SHORT"
        else None
    )
    breakout_failure_method = method_text == "BREAKOUT_FAILURE"
    failed_break_direction_bound = bool(
        failed_break_forecast_direction is not None
        and failed_break_forecast_direction == side_direction
    )
    opposite_side_direction = (
        "DOWN"
        if side_direction == "UP"
        else "UP"
        if side_direction == "DOWN"
        else None
    )
    range_rotation_method = method_text == "RANGE_ROTATION"
    forecast_side_bound = bool(
        side_direction is not None
        and (
            normalized_forecast_direction == side_direction
            or (
                range_rotation_method
                and normalized_forecast_direction == "RANGE"
            )
            or (
                breakout_failure_method
                and normalized_forecast_direction == "RANGE"
                and range_tp_proven_breakout_failure is True
            )
        )
    )
    actionable_direction_bound = bool(
        (
            breakout_failure_method
            and forecast_side_bound
            and failed_break_direction_bound
            and family_direction != opposite_side_direction
        )
        or (
            not breakout_failure_method
            and forecast_side_bound
            and family_direction in {"UP", "DOWN"}
            and family_direction == side_direction
            and directional_coverage is not None
        )
    )
    sha_bound = bool(
        valid
        and SHA256_RE.fullmatch(receipt_sha)
        and explicit_sha == receipt_sha
    )
    selected_method_bound = explicit_method == selected_method
    family_direction_bound = explicit_family_direction == family_direction
    # NONE is deliberately not an executable method. TRANSITION/UNKNOWN and
    # BREAKOUT_PENDING therefore cannot inherit permission from a lane method.
    executable_method_bound = bool(
        selected_method
        and method_text
        and method_text == selected_method
    )
    passed = bool(
        technical_context.get("status") == "VALID"
        and current_context_bound
        and canonical_policy_sources_bound
        and canonical_forecast_context_bound
        and valid
        and sha_bound
        and selected_method_bound
        and family_direction_bound
        and executable_method_bound
        and direction_consistent
        and forecast_side_bound
        and actionable_direction_bound
    )
    return {
        "passed": passed,
        "receipt_valid": valid,
        "current_context_bound": current_context_bound,
        "canonical_policy_sources_bound": canonical_policy_sources_bound,
        "canonical_forecast_context_bound": canonical_forecast_context_bound,
        "receipt_error": receipt_error,
        "receipt_sha256": receipt_sha or None,
        "sha_bound": sha_bound,
        "selected_method": selected_method,
        "selected_method_bound": selected_method_bound,
        "executable_method_bound": executable_method_bound,
        "family_direction": family_direction,
        "family_direction_bound": family_direction_bound,
        "direction_consistent": direction_consistent,
        "direction_reason": direction_reason,
        "directional_coverage_weight": directional_coverage,
        "side_direction": side_direction,
        "forecast_side_bound": forecast_side_bound,
        "range_tp_proven_breakout_failure": bool(
            range_tp_proven_breakout_failure
        ),
        "failed_break_side": failed_break_side,
        "failed_break_direction_bound": failed_break_direction_bound,
        "actionable_direction_bound": actionable_direction_bound,
    }


def _order_intent_lane_matches(
    order_intents: Mapping[str, Any],
    lane_id: str | None,
) -> list[Mapping[str, Any]]:
    """Return every exact lane-id match so allocation cannot pick by order."""

    if not lane_id:
        return []
    results = order_intents.get("results")
    if not isinstance(results, list):
        return []
    return [
        item
        for item in results
        if isinstance(item, Mapping)
        and str(item.get("lane_id") or "").strip() == lane_id
    ]


def _selected_execution_ledger_allocation_surface(
    surface: Mapping[str, Any],
    *,
    baseline: Mapping[str, Any],
    order_intents: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the WAL-safe ledger snapshot onto the deterministic lane only.

    An unrelated edge-only row does not invalidate the selected exact lane,
    but the projected surface also carries the global execution-cost cohort:
    a row that changes entry/exit/financing calibration is material regardless
    of pair. Any selected close/reduction is material as well.
    """

    selected_lane_id = str(baseline.get("selected_lane_id") or "").strip()
    selected_matches = _order_intent_lane_matches(
        order_intents,
        selected_lane_id or None,
    )
    selected_result: Mapping[str, Any] | None = (
        selected_matches[0] if len(selected_matches) == 1 else None
    )
    exact_key = _allocation_lane_exact_key(selected_result)

    def selected_rows(field: str) -> list[dict[str, Any]]:
        rows = surface.get(field)
        if exact_key is None or not isinstance(rows, list):
            return []
        return [
            dict(row)
            for row in rows
            if isinstance(row, Mapping)
            and tuple(
                str(row.get(name) or "").strip().upper()
                for name in ("pair", "side", "method", "vehicle")
            )
            == exact_key
        ]

    material = {
        "contract": EXACT_VEHICLE_ALLOCATION_SURFACE_CONTRACT,
        "parse_status": surface.get("parse_status"),
        "coverage_start_utc": surface.get("coverage_start_utc"),
        "selected_scope_key": "|".join(exact_key) if exact_key else None,
        "exact_vehicle_net": selected_rows("exact_vehicle_net"),
        "exact_vehicle_take_profit": selected_rows(
            "exact_vehicle_take_profit"
        ),
        "execution_cost": deepcopy(surface.get("execution_cost") or {}),
    }
    return {
        **material,
        "allocation_surface_sha256": canonical_json_sha256(material),
    }


def _missing_execution_ledger_allocation_surface() -> dict[str, Any]:
    material = {
        "contract": EXACT_VEHICLE_ALLOCATION_SURFACE_CONTRACT,
        "parse_status": "MISSING",
        "coverage_start_utc": None,
        "latest_realized_event": None,
        "last_oanda_transaction_id": None,
        "exact_vehicle_net": [],
        "exact_vehicle_take_profit": [],
        "execution_cost": {},
    }
    return {
        **material,
        "allocation_surface_sha256": canonical_json_sha256(material),
    }


def _allocation_lane_exact_key(
    selected_result: Mapping[str, Any] | None,
) -> tuple[str, str, str, str] | None:
    if not isinstance(selected_result, Mapping):
        return None
    intent = (
        selected_result.get("intent")
        if isinstance(selected_result.get("intent"), Mapping)
        else {}
    )
    metadata = (
        intent.get("metadata")
        if isinstance(intent.get("metadata"), Mapping)
        else {}
    )
    market_context = (
        intent.get("market_context")
        if isinstance(intent.get("market_context"), Mapping)
        else {}
    )
    pair = str(intent.get("pair") or "").strip().upper()
    side = str(intent.get("side") or "").strip().upper()
    # The typed intent market context is the method authority consumed by the
    # gateway.  Optional metadata may repeat that value for provenance, but it
    # must never redirect the allocation lookup to a different ledger cohort.
    method = str(market_context.get("method") or "").strip().upper()
    metadata_method = str(metadata.get("method") or "").strip().upper()
    method_scope_consistent = bool(
        method and (not metadata_method or metadata_method == method)
    )
    order_type = str(intent.get("order_type") or "").strip().upper()
    vehicle = {
        "STOP_ENTRY": "STOP",
        "STOP-ENTRY": "STOP",
        "STOP_ORDER": "STOP",
        "LIMIT_ORDER": "LIMIT",
        "MARKET_ORDER": "MARKET",
    }.get(order_type, order_type)
    if (
        pair
        and side in {"LONG", "SHORT"}
        and method_scope_consistent
        and vehicle in {"LIMIT", "MARKET", "STOP"}
    ):
        return pair, side, method, vehicle
    return None

def _allocation_metric_fields(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, Mapping) else {}
    return {
        "expectancy_jpy_per_trade": _optional_float(
            source.get("expectancy_jpy_per_trade")
        ),
        "net_jpy": _optional_float(source.get("net_jpy")),
        "trades": _optional_int(source.get("trades")),
        "win_rate": _optional_float(source.get("win_rate")),
        "payoff_ratio": _optional_float(source.get("payoff_ratio")),
        "profit_factor": _optional_float(source.get("profit_factor")),
    }


def _exact_vehicle_net_edge_evidence(
    *,
    intent: Mapping[str, Any],
    metadata: Mapping[str, Any],
    method: Any,
    current_metrics: Mapping[tuple[str, str, str, str], Mapping[str, Any]] | None,
    execution_ledger_surface_sha256: str,
) -> dict[str, Any]:
    """Bind intent claims to current all-exit ledger truth and prove its edge."""

    pair = str(intent.get("pair") or "").strip().upper()
    side = str(intent.get("side") or "").strip().upper()
    method_name = str(method or "").strip().upper()
    order_type = str(intent.get("order_type") or "").strip().upper()
    vehicle = {
        "STOP_ENTRY": "STOP",
        "STOP-ENTRY": "STOP",
        "STOP_ORDER": "STOP",
        "LIMIT_ORDER": "LIMIT",
        "MARKET_ORDER": "MARKET",
    }.get(order_type, order_type)
    expected_scope_key = f"{pair}|{side}|{method_name}|{vehicle}|ALL_AUDITED_EXITS"
    exact_key = (pair, side, method_name, vehicle)
    claimed = {
        "trades": metadata.get("capture_exact_vehicle_net_trades"),
        "wins": metadata.get("capture_exact_vehicle_net_wins"),
        "losses": metadata.get("capture_exact_vehicle_net_losses"),
        "net_jpy": metadata.get("capture_exact_vehicle_net_jpy"),
        "expectancy_jpy_per_trade": metadata.get(
            "capture_exact_vehicle_net_expectancy_jpy"
        ),
        "avg_win_jpy": metadata.get("capture_exact_vehicle_net_avg_win_jpy"),
        "avg_loss_jpy": metadata.get("capture_exact_vehicle_net_avg_loss_jpy"),
        "unresolved_realized_trades": metadata.get(
            "capture_exact_vehicle_net_unresolved_realized_trades"
        ),
        "unresolved_realized_net_jpy": metadata.get(
            "capture_exact_vehicle_net_unresolved_realized_net_jpy"
        ),
    }
    claimed_evaluation = evaluate_exact_vehicle_net_edge(claimed)
    current_row = (
        current_metrics.get(exact_key)
        if isinstance(current_metrics, Mapping)
        else None
    )
    current_evaluation = evaluate_exact_vehicle_net_edge(current_row)
    scope_binding_valid = bool(
        pair
        and side in {"LONG", "SHORT"}
        and method_name
        and vehicle in {"LIMIT", "STOP", "MARKET"}
        and str(metadata.get("capture_exact_vehicle_net_scope") or "").strip().upper()
        == "PAIR_SIDE_METHOD_VEHICLE"
        and str(metadata.get("capture_exact_vehicle_net_scope_key") or "").strip().upper()
        == expected_scope_key
        and str(metadata.get("capture_exact_vehicle_net_vehicle") or "").strip().upper()
        == vehicle
        and str(metadata.get("capture_exact_vehicle_net_metrics_source") or "").strip()
        == "data/execution_ledger.db:exact_vehicle_net"
        and str(metadata.get("capture_exact_vehicle_net_exit_scope") or "").strip().upper()
        == "ALL_AUDITED_EXITS"
    )
    metrics_binding_valid = bool(
        isinstance(current_row, Mapping)
        and _exact_vehicle_metric_claim_matches_current(
            claimed=claimed,
            current=current_row,
        )
        and str(current_row.get("source_scope") or "").strip().upper()
        == "PAIR_SIDE_METHOD_VEHICLE"
        and str(current_row.get("exit_scope") or "").strip().upper()
        == "ALL_AUDITED_EXITS"
        and str(
            metadata.get("capture_exact_vehicle_net_unresolved_trade_ids_sha256")
            or ""
        )
        == str(current_row.get("unresolved_trade_ids_sha256") or "")
    )
    binding_valid = bool(
        scope_binding_valid
        and metrics_binding_valid
        and SHA256_RE.fullmatch(execution_ledger_surface_sha256)
    )
    proven = bool(binding_valid and current_evaluation["proven"])
    evidence_present = bool(
        claimed_evaluation["evidence_present"]
        or current_evaluation["evidence_present"]
    )
    blocks_tp_exception = bool(
        current_evaluation["blocks_tp_exception"]
        or (evidence_present and not binding_valid)
    )
    return {
        **current_evaluation,
        "proven": proven,
        "blocks_tp_exception": blocks_tp_exception,
        "scope_binding_valid": scope_binding_valid,
        "metrics_binding_valid": metrics_binding_valid,
        "binding_valid": binding_valid,
        "claimed_arithmetic_consistent": claimed_evaluation[
            "arithmetic_consistent"
        ],
        "execution_ledger_surface_sha256": execution_ledger_surface_sha256,
    }


def _exact_vehicle_metric_claim_matches_current(
    *,
    claimed: Mapping[str, Any],
    current: Mapping[str, Any],
    include_unresolved: bool = True,
) -> bool:
    integer_fields = ("trades", "wins", "losses") + (
        ("unresolved_realized_trades",) if include_unresolved else ()
    )
    numeric_fields = (
        "net_jpy",
        "expectancy_jpy_per_trade",
        "avg_win_jpy",
        "avg_loss_jpy",
    ) + (("unresolved_realized_net_jpy",) if include_unresolved else ())
    for field in integer_fields:
        left = claimed.get(field)
        right = current.get(field)
        if (
            isinstance(left, bool)
            or isinstance(right, bool)
            or not isinstance(left, int)
            or not isinstance(right, int)
            or left != right
        ):
            return False
    for field in numeric_fields:
        left = claimed.get(field)
        right = current.get(field)
        if (
            isinstance(left, bool)
            or isinstance(right, bool)
            or not isinstance(left, (int, float))
            or not isinstance(right, (int, float))
        ):
            return False
        left_float = float(left)
        right_float = float(right)
        if (
            not math.isfinite(left_float)
            or not math.isfinite(right_float)
            or not math.isclose(left_float, right_float, rel_tol=1e-9, abs_tol=0.0001)
        ):
            return False
    return True


def _exact_vehicle_take_profit_edge_proven(
    *,
    intent: Mapping[str, Any],
    metadata: Mapping[str, Any],
    method: Any,
    tp_expectancy: float | None,
    tp_trades: int,
    tp_losses: int | None,
    current_metrics: Mapping[tuple[str, str, str, str], Mapping[str, Any]] | None,
    execution_ledger_surface_sha256: str,
) -> bool:
    pair = str(intent.get("pair") or "").strip().upper()
    side = str(intent.get("side") or "").strip().upper()
    method_name = str(method or "").strip().upper()
    order_type = str(intent.get("order_type") or "").strip().upper()
    vehicle = {
        "STOP_ENTRY": "STOP",
        "STOP-ENTRY": "STOP",
        "STOP_ORDER": "STOP",
        "LIMIT_ORDER": "LIMIT",
        "MARKET_ORDER": "MARKET",
    }.get(order_type, order_type)
    expected_scope_key = (
        f"{pair}|{side}|{method_name}|{vehicle}|TAKE_PROFIT_ORDER"
    )
    tp_avg_win = _optional_float(metadata.get("capture_take_profit_avg_win_jpy"))
    current_row = (
        current_metrics.get((pair, side, method_name, vehicle))
        if isinstance(current_metrics, Mapping)
        else None
    )
    claimed = {
        "trades": metadata.get("capture_take_profit_trades"),
        "wins": metadata.get("capture_take_profit_wins"),
        "losses": metadata.get("capture_take_profit_losses"),
        "net_jpy": metadata.get("capture_take_profit_net_jpy"),
        "expectancy_jpy_per_trade": metadata.get(
            "capture_take_profit_expectancy_jpy"
        ),
        "avg_win_jpy": metadata.get("capture_take_profit_avg_win_jpy"),
        "avg_loss_jpy": metadata.get("capture_take_profit_avg_loss_jpy"),
    }
    current_evaluation = evaluate_exact_vehicle_net_edge(current_row)
    return bool(
        pair
        and side in {"LONG", "SHORT"}
        and method_name
        # This does not relax profile/risk proof gates: the result must already
        # be LIVE_READY. It only proves that the allocation evidence matches
        # the current attached-TP execution vehicle, including MARKET when the
        # ledger scope is explicitly MARKET rather than a pooled statistic.
        and vehicle in {"LIMIT", "STOP", "MARKET"}
        and metadata.get("attach_take_profit_on_fill") is True
        and str(metadata.get("tp_execution_mode") or "").strip().upper()
        == "ATTACHED_TECHNICAL_TP"
        and metadata.get("capture_take_profit_exact_vehicle_required") is True
        and str(metadata.get("capture_take_profit_scope") or "").strip().upper()
        == "PAIR_SIDE_METHOD_VEHICLE"
        and str(metadata.get("capture_take_profit_vehicle") or "").strip().upper()
        == vehicle
        and str(metadata.get("capture_take_profit_scope_key") or "").strip().upper()
        == expected_scope_key
        and str(metadata.get("capture_take_profit_metrics_source") or "").strip()
        == "data/execution_ledger.db:exact_vehicle_take_profit"
        and SHA256_RE.fullmatch(execution_ledger_surface_sha256)
        and isinstance(current_row, Mapping)
        and str(current_row.get("source_scope") or "").strip().upper()
        == "PAIR_SIDE_METHOD_VEHICLE"
        and str(current_row.get("exit_scope") or "").strip().upper()
        == "PURE_TAKE_PROFIT_LIFECYCLE"
        and _exact_vehicle_metric_claim_matches_current(
            claimed=claimed,
            current=current_row,
            include_unresolved=False,
        )
        and current_evaluation["arithmetic_consistent"] is True
        and tp_trades >= CAPITAL_ALLOCATION_MIN_EXACT_TP_TRADES
        and tp_losses == 0
        and tp_expectancy is not None
        and tp_expectancy > 0.0
        and tp_avg_win is not None
        and tp_avg_win > 0.0
    )


_M15_RECOVERY_OUTER_METADATA_KEYS = (
    "m15_recovery_micro_contract",
    "m15_recovery_micro_mode",
    "m15_recovery_micro_receipt",
    "m15_recovery_micro_receipt_sha256",
    "m15_recovery_micro_units",
    "m15_recovery_micro_max_units",
    "m15_recovery_micro_full_size_allowed",
    "m15_recovery_micro_live_permission",
    "m15_recovery_micro_requires_risk_gateway_revalidation",
    "m15_recovery_micro_manual_position_mutation_allowed",
    "m15_recovery_micro_shape_eligible",
    "m15_recovery_micro_positive_rotation_mode",
    "m15_recovery_micro_proof_contract",
    "m15_recovery_micro_risk_revalidated",
    "m15_recovery_micro_gateway_revalidated",
    "forecast_m15_recovery_receipt",
    "forecast_m15_recovery_mode",
    "forecast_m15_recovery_live_permission",
    "forecast_m15_recovery_evidence",
    "forecast_m15_recovery_binding",
    "forecast_m15_recovery_binding_sha256",
    "forecast_cycle_id",
    "forecast_direction",
    "forecast_confidence",
    "forecast_raw_confidence",
    "forecast_calibration_multiplier",
    "forecast_current_price",
    "forecast_target_price",
    "forecast_invalidation_price",
    "forecast_horizon_min",
    "forecast_component_scores",
    "m15_recovery_context_contract",
    "m15_recovery_context_timeframes",
    "geometry_model",
    "geometry_atr_source_timeframe",
    "geometry_atr_pips",
    "geometry_source_recovery_receipt_sha256",
    "geometry_forecast_binding_sha256",
    "geometry_forecast_target_price",
    "geometry_forecast_invalidation_price",
    "geometry_tp_within_forecast_target",
    "geometry_sl_at_or_beyond_forecast_invalidation",
    "geometry_generic_overwrite_forbidden",
    "m15_recovery_lane_contract",
    "m15_recovery_lane_binding",
    "m15_recovery_lane_binding_sha256",
)


def _content_addressed_mapping_valid(
    value: Any,
    *,
    digest_key: str,
) -> bool:
    if not isinstance(value, Mapping):
        return False
    body = dict(value)
    digest = body.pop(digest_key, None)
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        return False
    try:
        return canonical_json_sha256(body) == digest
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError):
        return False


def _m15_recovery_contract_order_type(value: Any) -> str:
    normalized = str(value or "").strip().upper()
    if normalized in {"STOP", "STOP_ENTRY", "STOP-ENTRY", "STOP_ORDER"}:
        return "STOP-ENTRY"
    if normalized in {"LIMIT", "LIMIT_ORDER"}:
        return "LIMIT"
    return normalized


def _m15_recovery_rotation_proof_error(
    metadata: Mapping[str, Any],
    *,
    positive_tp_edge: bool,
) -> str | None:
    """Re-derive the exact TP proof mode without upgrading thin evidence."""

    mode = str(metadata.get("positive_rotation_mode") or "").strip().upper()
    proof_contract = metadata.get("m15_recovery_micro_proof_contract")
    trades = _strict_nonnegative_int(metadata.get("capture_take_profit_trades"))
    wins = _strict_nonnegative_int(metadata.get("capture_take_profit_wins"))
    losses = _strict_nonnegative_int(metadata.get("capture_take_profit_losses"))
    mirrored_trades = _strict_nonnegative_int(
        metadata.get("positive_rotation_tp_trades")
    )
    mirrored_wins = _strict_nonnegative_int(
        metadata.get("positive_rotation_tp_wins")
    )
    average_win = _strict_positive_number(
        metadata.get("capture_take_profit_avg_win_jpy")
    )
    loss_candidates = (
        _strict_positive_number(
            metadata.get("capture_take_profit_avg_loss_jpy")
        ),
        _strict_positive_number(metadata.get("capture_avg_loss_jpy")),
    )
    loss_proxy = max(
        (value for value in loss_candidates if value is not None),
        default=None,
    )
    stored_loss_proxy = _strict_positive_number(
        metadata.get("positive_rotation_loss_proxy_jpy")
    )
    stored_lower = _strict_probability(
        metadata.get("positive_rotation_tp_win_rate_lower")
    )
    stored_pessimistic = _strict_positive_number(
        metadata.get("positive_rotation_pessimistic_expectancy_jpy")
    )
    confidence_z = _strict_positive_number(
        metadata.get("positive_rotation_confidence_z")
    )
    market_close_expectancy = _optional_float(
        metadata.get("capture_market_close_expectancy_jpy")
    )
    if (
        not positive_tp_edge
        or mode
        not in {
            M15_RECOVERY_PROOF_COLLECTION_MODE,
            M15_RECOVERY_TP_PROVEN_MODE,
        }
        or not isinstance(proof_contract, Mapping)
        or proof_contract.get("positive_rotation_mode") != mode
        or proof_contract.get("reachable") is not True
        or bool(proof_contract.get("failed_checks") or [])
        or metadata.get("loss_asymmetry_guard_active") is not True
        or str(metadata.get("capture_economics_status") or "").strip().upper()
        != "NEGATIVE_EXPECTANCY"
        or str(metadata.get("position_intent") or "NEW").strip().upper()
        == "HEDGE"
        or metadata.get("attach_take_profit_on_fill") is not True
        or str(metadata.get("tp_execution_mode") or "").strip().upper()
        != "ATTACHED_TECHNICAL_TP"
        or str(metadata.get("tp_target_intent") or "").strip().upper()
        != "HARVEST"
        or str(metadata.get("opportunity_mode") or "").strip().upper()
        != "HARVEST"
        or metadata.get("capture_take_profit_exact_vehicle_required") is not True
        or str(metadata.get("capture_take_profit_scope") or "").strip().upper()
        != "PAIR_SIDE_METHOD_VEHICLE"
        or trades is None
        or wins is None
        or losses != 0
        or wins != trades
        or mirrored_trades != trades
        or mirrored_wins != wins
        or average_win is None
        or loss_proxy is None
        or stored_loss_proxy is None
        or stored_lower is None
        or stored_pessimistic is None
        or confidence_z is None
        or not math.isclose(confidence_z, 1.96, rel_tol=0.0, abs_tol=1e-12)
        or str(metadata.get("positive_rotation_confidence_method") or "")
        != "WILSON_LOWER_BOUND_STRESS_EXPECTANCY"
        or market_close_expectancy is None
        or market_close_expectancy >= 0.0
    ):
        return "M15_RECOVERY_EXACT_TP_PROOF_INVALID"

    expected_lower = hit_rate_wilson_lower(wins / trades, trades)
    if expected_lower is None:
        return "M15_RECOVERY_EXACT_TP_PROOF_INVALID"
    expected_pessimistic = (
        expected_lower * average_win - (1.0 - expected_lower) * loss_proxy
    )
    if (
        expected_pessimistic <= 0.0
        or not math.isclose(
            stored_loss_proxy,
            round(loss_proxy, 4),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            stored_lower,
            round(expected_lower, 6),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            stored_pessimistic,
            round(expected_pessimistic, 4),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        return "M15_RECOVERY_EXACT_TP_PROOF_ARITHMETIC_INVALID"

    guard_mode = str(
        metadata.get("loss_asymmetry_guard_mode") or ""
    ).strip().upper()
    if mode == M15_RECOVERY_PROOF_COLLECTION_MODE:
        if (
            not M15_RECOVERY_PROOF_COLLECTION_MIN_TRADES
            <= trades
            < M15_RECOVERY_TP_PROVEN_MIN_TRADES
            or metadata.get("positive_rotation_proof_collection_ready")
            is not True
            or str(
                metadata.get("positive_rotation_proof_collection_mode") or ""
            ).strip().upper()
            != M15_RECOVERY_PROOF_COLLECTION_MODE
            or metadata.get("positive_rotation_proof_collection_min_trades")
            != M15_RECOVERY_PROOF_COLLECTION_MIN_TRADES
            or metadata.get("positive_rotation_proof_collection_target_trades")
            != M15_RECOVERY_TP_PROVEN_MIN_TRADES
            or metadata.get("positive_rotation_proof_collection_gap_trades")
            != M15_RECOVERY_TP_PROVEN_MIN_TRADES - trades
            or metadata.get("positive_rotation_live_ready") not in {None, False}
            or guard_mode not in {"CAP_AVG_WIN", "TP_PROOF_COLLECTION_MIN_LOT"}
            or metadata.get("loss_asymmetry_guard_relaxed") is not False
        ):
            return "M15_RECOVERY_PROOF_COLLECTION_CONTRACT_INVALID"
        return None

    if (
        trades < M15_RECOVERY_TP_PROVEN_MIN_TRADES
        or metadata.get("positive_rotation_live_ready") is not True
        or guard_mode != "TP_PROVEN_RELAXED"
        or metadata.get("loss_asymmetry_guard_relaxed") is not True
    ):
        return "M15_RECOVERY_TP_PROVEN_CONTRACT_INVALID"
    return None


def _m15_recovery_allocation_contract(
    *,
    result: Mapping[str, Any],
    intent: Mapping[str, Any],
    metadata: Mapping[str, Any],
    method: str,
    base_units: int,
    positive_tp_edge: bool,
) -> dict[str, Any]:
    """Validate and freeze the recovery-only allocation evidence chain."""

    claimed = _m15_recovery_claimed(metadata)
    if not claimed:
        return {
            "claimed": False,
            "valid": False,
            "status": "NOT_CLAIMED",
            "error_codes": [],
            "mode": None,
            "edge_basis": None,
            "context_timeframes": None,
            "material": None,
            "material_sha256": None,
        }

    errors: list[str] = []

    def fail(code: str) -> None:
        if code not in errors:
            errors.append(code)

    pair = str(intent.get("pair") or "").strip().upper()
    side = str(intent.get("side") or "").strip().upper()
    raw_order_type = str(intent.get("order_type") or "").strip().upper()
    order_type = _m15_recovery_contract_order_type(raw_order_type)
    entry = _optional_float(intent.get("entry"))
    tp = _optional_float(intent.get("tp"))
    sl = _optional_float(intent.get("sl"))
    receipt = metadata.get("m15_recovery_micro_receipt")
    forecast_binding = metadata.get("forecast_m15_recovery_binding")
    lane_binding = metadata.get("m15_recovery_lane_binding")
    mode = str(metadata.get("positive_rotation_mode") or "").strip().upper()

    if not _content_addressed_mapping_valid(receipt, digest_key="receipt_sha256"):
        fail("M15_RECOVERY_SOURCE_RECEIPT_DIGEST_INVALID")
    if isinstance(receipt, Mapping):
        sizing = receipt.get("sizing")
        geometry_source = receipt.get("geometry_source")
        fast_receipts = receipt.get("fast_timeframe_receipts")
        spread = _strict_positive_number(receipt.get("current_spread_pips"))
        spread_cap = _strict_positive_number(receipt.get("spread_cap_pips"))
        if (
            receipt.get("contract") != M15_RECOVERY_MICRO_CONTRACT
            or receipt.get("mode") != M15_RECOVERY_MICRO_MODE
            or receipt.get("status") != "ELIGIBLE_FOR_MICRO_REVALIDATION"
            or receipt.get("pair") != pair
            or receipt.get("live_permission") is not False
            or receipt.get("requires_risk_gateway_revalidation") is not True
            or receipt.get("manual_position_mutation_allowed") is not False
            or not isinstance(sizing, Mapping)
            or sizing.get("max_units") != M15_RECOVERY_MICRO_MAX_UNITS
            or sizing.get("full_size_allowed") is not False
            or sizing.get("minimum_units_override") is not False
            or not isinstance(geometry_source, Mapping)
            or geometry_source.get("timeframe") != "M15"
            or not isinstance(fast_receipts, Mapping)
            or set(fast_receipts) != {"M1", "M5"}
            or any(
                not isinstance(item, Mapping)
                or item.get("recovery_state") not in {"RECOVERING", "RECOVERED"}
                for item in fast_receipts.values()
            )
            or not any(
                isinstance(item, Mapping)
                and item.get("recovery_state") == "RECOVERING"
                for item in fast_receipts.values()
            )
            or spread is None
            or spread_cap is None
            or spread > spread_cap
        ):
            fail("M15_RECOVERY_SOURCE_RECEIPT_CONTRACT_INVALID")
    else:
        fail("M15_RECOVERY_SOURCE_RECEIPT_MISSING")

    receipt_sha = receipt.get("receipt_sha256") if isinstance(receipt, Mapping) else None
    if (
        metadata.get("m15_recovery_micro_contract")
        != M15_RECOVERY_MICRO_CONTRACT
        or metadata.get("m15_recovery_micro_mode") != M15_RECOVERY_MICRO_MODE
        or metadata.get("m15_recovery_micro_receipt_sha256") != receipt_sha
        or metadata.get("forecast_m15_recovery_receipt") != receipt
        or metadata.get("m15_recovery_micro_live_permission") is not False
        or metadata.get(
            "m15_recovery_micro_requires_risk_gateway_revalidation"
        )
        is not True
        or metadata.get(
            "m15_recovery_micro_manual_position_mutation_allowed"
        )
        is not False
        or metadata.get("m15_recovery_micro_full_size_allowed") is not False
        or metadata.get("m15_recovery_micro_max_units")
        != M15_RECOVERY_MICRO_MAX_UNITS
        or metadata.get("m15_recovery_micro_units") != base_units
        or metadata.get("m15_recovery_micro_shape_eligible") is not True
        or metadata.get("m15_recovery_micro_positive_rotation_mode") != mode
        or metadata.get("forecast_m15_recovery_mode")
        != M15_RECOVERY_MICRO_MODE
        or metadata.get("forecast_m15_recovery_live_permission") is not False
        or metadata.get("m15_recovery_micro_gateway_revalidated") is not False
    ):
        fail("M15_RECOVERY_OUTER_RECEIPT_BINDING_INVALID")

    expected_direction = "UP" if side == "LONG" else "DOWN" if side == "SHORT" else None
    if (
        pair not in DEFAULT_TRADER_PAIRS
        or expected_direction is None
        or metadata.get("forecast_direction") != expected_direction
        or method != "BREAKOUT_FAILURE"
        or order_type != M15_RECOVERY_ORDER_TYPE
        or raw_order_type == "MARKET"
        or base_units <= 0
        or base_units > M15_RECOVERY_MICRO_MAX_UNITS
        or str(metadata.get("position_intent") or "NEW").strip().upper()
        == "HEDGE"
        or metadata.get("m15_recovery_context_contract")
        != M15_RECOVERY_FORECAST_CONTRACT
        or metadata.get("m15_recovery_context_timeframes")
        != list(M15_RECOVERY_CONTEXT_TIMEFRAMES)
    ):
        fail("M15_RECOVERY_EXECUTABLE_SHAPE_INVALID")

    risk_issues = [
        dict(issue)
        for issue in result.get("risk_issues", []) or []
        if isinstance(issue, Mapping)
        and str(issue.get("code") or "").startswith("M15_RECOVERY_")
    ]
    risk_marker_present = any(
        issue.get("code") == M15_RECOVERY_RISK_MARKER
        and str(issue.get("severity") or "").strip().upper() == "WARN"
        for issue in risk_issues
    )
    risk_recovery_block = any(
        str(issue.get("severity") or "").strip().upper() == "BLOCK"
        for issue in risk_issues
    )
    if (
        metadata.get("m15_recovery_micro_risk_revalidated") is not True
        or not risk_marker_present
        or risk_recovery_block
    ):
        fail("M15_RECOVERY_RISK_MARKER_INVALID")

    if not isinstance(forecast_binding, Mapping):
        fail("M15_RECOVERY_FORECAST_BINDING_MISSING")
    else:
        try:
            forecast_valid, forecast_error = (
                validate_m15_recovery_forecast_binding(
                    forecast_binding,
                    recovery_receipt=(receipt if isinstance(receipt, Mapping) else {}),
                    metadata=metadata,
                )
            )
        except (TypeError, ValueError, OverflowError):
            forecast_valid, forecast_error = False, (
                "M15_RECOVERY_FORECAST_BINDING_VALIDATOR_FAILED"
            )
        history = forecast_binding.get("forecast_history_identity")
        if (
            not forecast_valid
            or forecast_binding.get("contract")
            != M15_RECOVERY_FORECAST_CONTRACT
            or metadata.get("forecast_m15_recovery_binding_sha256")
            != forecast_binding.get("binding_sha256")
            or forecast_binding.get("pair") != pair
            or forecast_binding.get("final_direction") != expected_direction
            or not isinstance(history, Mapping)
            or history.get("cycle_id") != metadata.get("forecast_cycle_id")
            or history.get("pair") != pair
            or history.get("direction") != expected_direction
        ):
            fail(
                str(forecast_error or "M15_RECOVERY_FORECAST_BINDING_INVALID")
            )

    if not isinstance(lane_binding, Mapping):
        fail("M15_RECOVERY_LANE_BINDING_MISSING")
    else:
        try:
            lane_valid, lane_error = validate_m15_recovery_lane_binding(
                lane_binding,
                forecast_binding=(
                    forecast_binding if isinstance(forecast_binding, Mapping) else {}
                ),
                pair=pair,
                side=side,
                method=method,
                order_type=order_type,
                entry=entry,
                tp=tp,
                sl=sl,
                current_units=base_units,
                metadata=metadata,
            )
        except (TypeError, ValueError, OverflowError):
            lane_valid, lane_error = False, (
                "M15_RECOVERY_LANE_BINDING_VALIDATOR_FAILED"
            )
        if (
            not lane_valid
            or lane_binding.get("contract") != M15_RECOVERY_LANE_CONTRACT
            or metadata.get("m15_recovery_lane_contract")
            != M15_RECOVERY_LANE_CONTRACT
            or metadata.get("m15_recovery_lane_binding_sha256")
            != lane_binding.get("binding_sha256")
            or lane_binding.get("manual_position_mutation_allowed") is not False
            or lane_binding.get("live_permission") is not False
        ):
            fail(str(lane_error or "M15_RECOVERY_LANE_BINDING_INVALID"))

    proof_error = _m15_recovery_rotation_proof_error(
        metadata,
        positive_tp_edge=positive_tp_edge,
    )
    if proof_error is not None:
        fail(proof_error)

    valid = not errors
    edge_basis = (
        M15_RECOVERY_EDGE_COLLECTION_BASIS
        if valid and mode == M15_RECOVERY_PROOF_COLLECTION_MODE
        else "EXACT_VEHICLE_TAKE_PROFIT"
        if valid and mode == M15_RECOVERY_TP_PROVEN_MODE
        else None
    )
    material: dict[str, Any] | None = None
    material_sha256: str | None = None
    if valid:
        lane_proof = lane_binding.get("proof_source_receipt")
        proof_metadata_keys = (
            set(lane_proof) - {"proof_source_sha256"}
            if isinstance(lane_proof, Mapping)
            else set()
        )
        material_metadata = {
            key: deepcopy(metadata.get(key))
            for key in sorted(
                set(_M15_RECOVERY_OUTER_METADATA_KEYS) | proof_metadata_keys
            )
        }
        material_body = {
            "contract": M15_RECOVERY_ALLOCATION_MATERIAL_CONTRACT,
            "pair": pair,
            "side": side,
            "method": method,
            "order_type": order_type,
            "entry": entry,
            "tp": tp,
            "sl": sl,
            "base_units": base_units,
            "positive_rotation_mode": mode,
            "edge_basis": edge_basis,
            "context_timeframes": list(M15_RECOVERY_CONTEXT_TIMEFRAMES),
            "source_receipt": deepcopy(dict(receipt)),
            "forecast_binding": deepcopy(dict(forecast_binding)),
            "lane_binding": deepcopy(dict(lane_binding)),
            "metadata": material_metadata,
            "risk_issues": risk_issues,
        }
        material_sha256 = canonical_json_sha256(material_body)
        material = {**material_body, "material_sha256": material_sha256}

    return {
        "claimed": True,
        "valid": valid,
        "status": (
            M15_RECOVERY_ALLOCATION_STATUS
            if valid
            else "M15_RECOVERY_REJECTED"
        ),
        "error_codes": errors,
        "mode": mode or None,
        "edge_basis": edge_basis,
        "context_timeframes": list(M15_RECOVERY_CONTEXT_TIMEFRAMES),
        "source_receipt_sha256": receipt_sha,
        "forecast_cycle_id": metadata.get("forecast_cycle_id"),
        "forecast_binding_sha256": (
            forecast_binding.get("binding_sha256")
            if isinstance(forecast_binding, Mapping)
            else None
        ),
        "lane_binding_sha256": (
            lane_binding.get("binding_sha256")
            if isinstance(lane_binding, Mapping)
            else None
        ),
        "material": material,
        "material_sha256": material_sha256,
    }


def _tp_proven_harvest_range_economic_basis(
    *,
    intent: Mapping[str, Any],
    metadata: Mapping[str, Any],
    method: str,
    positive_tp_edge: bool,
    execution_ledger_surface_sha256: str,
) -> dict[str, Any] | None:
    """Return a strict passive TP HARVEST cohort receipt for RANGE.

    The caller has already rebound the exact TP cohort to the current semantic
    ledger surface.  This additionally rechecks the stronger live rotation
    contract and its Wilson-stressed expectancy instead of treating a pooled TP
    label as a trade probability.
    """

    if not positive_tp_edge or method not in {"RANGE_ROTATION", "BREAKOUT_FAILURE"}:
        return None
    if str(metadata.get("forecast_direction") or "").strip().upper() != "RANGE":
        return None
    vehicle = _normalized_order_vehicle(intent.get("order_type"))
    # This is a passive rail/fade exception.  STOP is a momentum trigger and
    # has no canonical RANGE TP_PROVEN geometry in the current contract.
    if vehicle != "LIMIT":
        return None
    if str(metadata.get("position_intent") or "NEW").strip().upper() == "HEDGE":
        return None
    if metadata.get("attach_take_profit_on_fill") is not True:
        return None
    if str(metadata.get("tp_execution_mode") or "").strip().upper() != "ATTACHED_TECHNICAL_TP":
        return None
    if str(metadata.get("tp_target_intent") or "").strip().upper() != "HARVEST":
        return None
    if str(metadata.get("opportunity_mode") or "").strip().upper() != "HARVEST":
        return None
    if str(metadata.get("positive_rotation_mode") or "").strip().upper() != "TP_PROVEN_HARVEST":
        return None
    if metadata.get("positive_rotation_live_ready") is not True:
        return None
    if str(metadata.get("loss_asymmetry_guard_mode") or "").strip().upper() != "TP_PROVEN_RELAXED":
        return None

    trades = _strict_nonnegative_int(metadata.get("capture_take_profit_trades"))
    wins = _strict_nonnegative_int(metadata.get("capture_take_profit_wins"))
    losses = _strict_nonnegative_int(metadata.get("capture_take_profit_losses"))
    mirrored_trades = _strict_nonnegative_int(
        metadata.get("positive_rotation_tp_trades")
    )
    mirrored_wins = _strict_nonnegative_int(
        metadata.get("positive_rotation_tp_wins")
    )
    avg_win = _strict_positive_number(
        metadata.get("capture_take_profit_avg_win_jpy")
    )
    loss_proxy = _strict_positive_number(
        metadata.get("positive_rotation_loss_proxy_jpy")
    )
    stored_wilson = _strict_probability(
        metadata.get("positive_rotation_tp_win_rate_lower")
    )
    stored_pessimistic = _strict_positive_number(
        metadata.get("positive_rotation_pessimistic_expectancy_jpy")
    )
    if (
        trades is None
        or wins is None
        or losses is None
        or trades < 20
        or wins > trades
        or losses != 0
        or mirrored_trades != trades
        or mirrored_wins != wins
        or avg_win is None
        or loss_proxy is None
        or stored_wilson is None
        or stored_pessimistic is None
    ):
        return None
    hit_rate = wins / trades
    wilson_lower = hit_rate_wilson_lower(hit_rate, trades)
    if wilson_lower is None:
        return None
    pessimistic = wilson_lower * avg_win - (1.0 - wilson_lower) * loss_proxy
    if (
        pessimistic <= 0.0
        or not math.isclose(
            stored_wilson,
            round(wilson_lower, 6),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            stored_pessimistic,
            round(pessimistic, 4),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        return None
    return {
        "basis": "EXACT_TP_PROVEN_HARVEST",
        # These are conditional outcomes inside the exact TP exit cohort, not
        # an unconditional probability that a fresh RANGE trade will win.
        "conditional_tp_exit_win_rate": hit_rate,
        "tp_exit_samples": trades,
        "minimum_tp_exit_samples": 20,
        "conditional_tp_exit_wilson95_lower": wilson_lower,
        "stressed_harvest_expectancy_jpy": pessimistic,
        "execution_ledger_surface_sha256": execution_ledger_surface_sha256,
    }


def _range_lane_economic_basis(
    *,
    intent: Mapping[str, Any],
    metadata: Mapping[str, Any],
    method: str,
    exact_vehicle_net_edge: Mapping[str, Any],
    positive_tp_edge: bool,
    execution_ledger_surface_sha256: str,
) -> dict[str, Any] | None:
    """Select a RANGE trade-outcome basis without reusing box-hold precision."""

    if str(metadata.get("forecast_direction") or "").strip().upper() != "RANGE":
        return None
    # Ordinary schema-v2 GPT numeric allocation is MARKET-only.  A positive
    # all-exit LIMIT/STOP row is valuable evidence, but it cannot silently widen
    # that transport contract.  The sole non-market RANGE path is the existing
    # exact TP_PROVEN_HARVEST repair exception below.
    _ = exact_vehicle_net_edge
    return _tp_proven_harvest_range_economic_basis(
        intent=intent,
        metadata=metadata,
        method=method,
        positive_tp_edge=positive_tp_edge,
        execution_ledger_surface_sha256=execution_ledger_surface_sha256,
    )


def _broker_snapshot_nav_jpy(broker_snapshot: Mapping[str, Any]) -> float | None:
    account = (
        broker_snapshot.get("account")
        if isinstance(broker_snapshot.get("account"), Mapping)
        else None
    )
    if not isinstance(account, Mapping):
        return None
    return _strict_positive_number(account.get("nav_jpy"))


def _broker_snapshot_bid_ask(
    broker_snapshot: Mapping[str, Any],
    *,
    pair: str,
) -> tuple[float | None, float | None]:
    quotes = (
        broker_snapshot.get("quotes")
        if isinstance(broker_snapshot.get("quotes"), Mapping)
        else None
    )
    quote = quotes.get(pair) if isinstance(quotes, Mapping) else None
    if not isinstance(quote, Mapping):
        return None, None
    return (
        _strict_positive_number(quote.get("bid")),
        _strict_positive_number(quote.get("ask")),
    )


def _broker_snapshot_quote_to_jpy(
    broker_snapshot: Mapping[str, Any],
    *,
    pair: str,
) -> float | None:
    if pair not in DEFAULT_TRADER_PAIRS:
        return None
    quote_currency = pair.split("_", 1)[1]
    if quote_currency == "JPY":
        return 1.0
    conversions = (
        broker_snapshot.get("home_conversions")
        if isinstance(broker_snapshot.get("home_conversions"), Mapping)
        else None
    )
    if not isinstance(conversions, Mapping):
        return None
    return _strict_positive_number(conversions.get(quote_currency))


def _capital_allocation_numeric_ceiling(
    *,
    intent: Mapping[str, Any],
    metadata: Mapping[str, Any],
    risk_metrics: Mapping[str, Any],
    account_nav_jpy: float | None,
    broker_bid: float | None,
    broker_ask: float | None,
    broker_quote_to_jpy: float | None,
    predictive_scout: bool,
    hedge: bool,
    range_economic_basis: Mapping[str, Any] | None = None,
    forecast_current_binding_mode: str = "EXACT_SNAPSHOT_MID",
    execution_cost_floor: Mapping[str, Any] | None = None,
    market_entry_slippage_embedded: bool = False,
    m15_recovery_prebounded: bool = False,
) -> tuple[dict[str, Any], float]:
    """Return content-addressed numeric sizing evidence and its maximum multiple."""

    pair = str(intent.get("pair") or "").strip().upper()
    side = str(intent.get("side") or "").strip().upper()
    order_type = str(intent.get("order_type") or "").strip().upper()
    order_vehicle = _normalized_order_vehicle(order_type)
    intent_market_context = (
        intent.get("market_context")
        if isinstance(intent.get("market_context"), Mapping)
        else {}
    )
    cost_method = str(
        intent_market_context.get("method") or "UNKNOWN"
    ).strip().upper()
    raw_units = intent.get("units")
    units = (
        abs(raw_units)
        if isinstance(raw_units, int)
        and not isinstance(raw_units, bool)
        and raw_units != 0
        else None
    )
    forecast_binding = _directional_forecast_trade_binding(
        side=side,
        forecast_direction=metadata.get("forecast_direction"),
        calibration_name=metadata.get("forecast_directional_calibration_name"),
        method=cost_method,
    )
    forecast_direction = str(forecast_binding["forecast_direction"] or "")
    calibration_name = str(forecast_binding["calibration_name"] or "")
    entry_price = _strict_positive_number(risk_metrics.get("entry_price"))
    intent_entry = _strict_positive_number(intent.get("entry"))
    bid = _strict_positive_number(broker_bid)
    ask = _strict_positive_number(broker_ask)
    broker_quote_valid = bool(bid is not None and ask is not None and ask > bid)
    broker_entry = (
        ask
        if broker_quote_valid and side == "LONG"
        else bid
        if broker_quote_valid and side == "SHORT"
        else None
    )
    broker_mid = (
        (bid + ask) / 2.0
        if bid is not None and ask is not None and ask > bid
        else None
    )
    half_spread_price = (
        (ask - bid) / 2.0
        if bid is not None and ask is not None and ask > bid
        else None
    )
    pip_factor = (
        instrument_pip_factor(pair)
        if pair in DEFAULT_TRADER_PAIRS
        else None
    )
    quote_to_jpy = _strict_positive_number(broker_quote_to_jpy)
    expected_jpy_per_pip = (
        units / float(pip_factor) * quote_to_jpy
        if units is not None
        and pip_factor is not None
        and quote_to_jpy is not None
        else None
    )
    order_tp = _strict_positive_number(intent.get("tp"))
    order_sl = _strict_positive_number(intent.get("sl"))
    forecast_current = _strict_positive_number(metadata.get("forecast_current_price"))
    forecast_target = _strict_positive_number(metadata.get("forecast_target_price"))
    forecast_invalidation = _strict_positive_number(
        metadata.get("forecast_invalidation_price")
    )
    forecast_range_low = _strict_positive_number(
        metadata.get("forecast_range_low_price")
    )
    forecast_range_high = _strict_positive_number(
        metadata.get("forecast_range_high_price")
    )
    range_support = _strict_positive_number(metadata.get("range_support"))
    range_resistance = _strict_positive_number(metadata.get("range_resistance"))
    range_entry_side = str(metadata.get("range_entry_side") or "").strip().upper()
    forecast_economic_hit_rate = _strict_probability(
        metadata.get("forecast_directional_economic_hit_rate")
    )
    forecast_economic_samples = _strict_nonnegative_int(
        metadata.get("forecast_directional_economic_samples")
    )
    headline_hit_rate = _strict_probability(
        metadata.get("forecast_directional_hit_rate")
    )
    headline_samples = _strict_nonnegative_int(
        metadata.get("forecast_directional_samples")
    )
    timeout_rate = _strict_probability(
        metadata.get("forecast_directional_timeout_rate")
    )
    risk_jpy = _strict_positive_number(risk_metrics.get("risk_jpy"))
    reward_jpy = _strict_positive_number(risk_metrics.get("reward_jpy"))
    loss_pips = _strict_positive_number(risk_metrics.get("loss_pips"))
    reward_pips = _strict_positive_number(risk_metrics.get("reward_pips"))
    jpy_per_pip = _strict_positive_number(risk_metrics.get("jpy_per_pip"))
    reported_reward_risk = _strict_positive_number(risk_metrics.get("reward_risk"))
    spread_pips = _strict_nonnegative_number(risk_metrics.get("spread_pips"))
    nav_jpy = _strict_positive_number(account_nav_jpy)
    intent_max_loss_jpy = _strict_positive_number(metadata.get("max_loss_jpy"))
    cost_floor = (
        dict(execution_cost_floor)
        if isinstance(execution_cost_floor, Mapping)
        else {}
    )
    cost_floor_sha256 = str(cost_floor.get("proof_sha256") or "")
    cost_floor_material = dict(cost_floor)
    cost_floor_material.pop("proof_sha256", None)
    cost_floor_digest_valid = bool(
        SHA256_RE.fullmatch(cost_floor_sha256)
        and canonical_json_sha256(cost_floor_material)
        == cost_floor_sha256
    )
    market_entry_adverse_p95_pips = _strict_nonnegative_number(
        cost_floor.get("market_entry_adverse_p95_pips")
    )
    audited_exit_adverse_p95_pips = _strict_nonnegative_number(
        cost_floor.get("audited_protected_exit_adverse_p95_pips")
    )
    financing_stress_jpy_per_unit = _strict_nonnegative_number(
        cost_floor.get("financing_adverse_stress_jpy_per_unit")
    )
    metadata_method = str(metadata.get("method") or "").strip().upper()
    method_scope_consistent = bool(
        cost_method != "UNKNOWN"
        and (not metadata_method or metadata_method == cost_method)
    )
    expected_cost_scope_key = (
        "|".join((pair, side, cost_method, order_vehicle))
        if order_vehicle is not None
        else ""
    )
    cost_floor_valid = bool(
        cost_floor.get("status") == "PASSED"
        and cost_floor.get("contract") == EXECUTION_COST_FLOOR_CONTRACT
        and cost_floor_digest_valid
        and market_entry_adverse_p95_pips is not None
        and audited_exit_adverse_p95_pips is not None
        and financing_stress_jpy_per_unit is not None
        and cost_floor.get("spread_double_count_forbidden") is True
        and method_scope_consistent
        and bool(expected_cost_scope_key)
        and str(cost_floor.get("scope_key") or "").upper()
        == expected_cost_scope_key
    )

    normalized_range_basis = (
        dict(range_economic_basis)
        if isinstance(range_economic_basis, Mapping)
        else {}
    )
    range_basis_hit_rate = _strict_probability(
        normalized_range_basis.get("conditional_tp_exit_win_rate")
    )
    range_basis_samples = _strict_nonnegative_int(
        normalized_range_basis.get("tp_exit_samples")
    )
    range_basis_minimum_samples = _strict_nonnegative_int(
        normalized_range_basis.get("minimum_tp_exit_samples")
    )
    range_basis_wilson = _strict_probability(
        normalized_range_basis.get("conditional_tp_exit_wilson95_lower")
    )
    range_basis_stressed_expectancy = _strict_positive_number(
        normalized_range_basis.get("stressed_harvest_expectancy_jpy")
    )
    range_basis_name = str(normalized_range_basis.get("basis") or "").strip().upper()
    range_basis_digest = str(
        normalized_range_basis.get("execution_ledger_surface_sha256") or ""
    )
    range_lane = bool(
        forecast_binding["forecast_direction"] == "RANGE"
        and forecast_binding["range_method_bound"]
    )
    recomputed_range_wilson = (
        hit_rate_wilson_lower(range_basis_hit_rate, range_basis_samples)
        if range_basis_hit_rate is not None
        and range_basis_samples is not None
        and range_basis_samples > 0
        else None
    )
    range_economic_basis_valid = bool(
        range_lane
        and range_basis_name
        == "EXACT_TP_PROVEN_HARVEST"
        and range_basis_hit_rate is not None
        and range_basis_samples is not None
        and range_basis_minimum_samples is not None
        and range_basis_samples >= range_basis_minimum_samples
        and range_basis_stressed_expectancy is not None
        and SHA256_RE.fullmatch(range_basis_digest)
        and recomputed_range_wilson is not None
        and range_basis_wilson is not None
        and math.isclose(
            range_basis_wilson,
            recomputed_range_wilson,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    # RANGE precision is box integrity, while this receipt is conditional on a
    # TP exit.  Neither is an unconditional side win probability, so the
    # generic EV/Kelly inputs remain null on the prebounded exception.
    economic_hit_rate = None if range_lane else forecast_economic_hit_rate
    economic_samples = None if range_lane else forecast_economic_samples
    minimum_economic_samples = (
        None if range_lane else CAPITAL_ALLOCATION_FORECAST_MIN_SAMPLES
    )
    probability_basis = (
        "PREBOUNDED_CONTRACT_NO_UNCONDITIONAL_TRADE_PROBABILITY"
        if range_lane
        else "DIRECTIONAL_ECONOMIC_HIT_RATE"
    )

    direction_side_aligned = bool(forecast_binding["direction_side_aligned"])
    expected_calibration_name = forecast_binding["expected_calibration_name"]
    calibration_identity_valid = bool(
        forecast_binding["calibration_identity_valid"]
    )
    mid_tolerance = (
        0.5 / (float(pip_factor) * 10.0) + 1e-12
        if pip_factor is not None
        else 0.0
    )
    forecast_current_matches_broker_mid = bool(
        forecast_current is not None
        and broker_mid is not None
        and math.isclose(
            forecast_current,
            broker_mid,
            rel_tol=0.0,
            abs_tol=mid_tolerance,
        )
    )
    normalized_current_binding_mode = str(
        forecast_current_binding_mode or ""
    ).strip().upper()
    signed_fresh_mid_drift_pips = (
        (
            broker_mid - forecast_current
            if side == "LONG"
            else forecast_current - broker_mid
        )
        * pip_factor
        if broker_mid is not None
        and forecast_current is not None
        and pip_factor is not None
        and side in {"LONG", "SHORT"}
        else None
    )
    directional_fresh_market_drift_passed = bool(
        signed_fresh_mid_drift_pips is not None
        and signed_fresh_mid_drift_pips >= -1e-9
    )
    range_fresh_quote_inside_rails = bool(
        range_lane
        and forecast_range_low is not None
        and forecast_range_high is not None
        and bid is not None
        and ask is not None
        and forecast_range_low < bid < ask < forecast_range_high
    )
    range_fresh_limit_remains_passive = bool(
        range_lane
        and order_vehicle == "LIMIT"
        and intent_entry is not None
        and forecast_current is not None
        and (
            side == "LONG"
            and ask is not None
            and intent_entry < ask
            and intent_entry < forecast_current
            or side == "SHORT"
            and bid is not None
            and intent_entry > bid
            and intent_entry > forecast_current
        )
    )
    range_fresh_passive_rail_passed = bool(
        range_fresh_quote_inside_rails
        and range_fresh_limit_remains_passive
    )
    forecast_current_binding_passed = bool(
        forecast_current_matches_broker_mid
        if normalized_current_binding_mode == "EXACT_SNAPSHOT_MID"
        else directional_fresh_market_drift_passed
        if normalized_current_binding_mode == "DIRECTIONAL_FRESH_MARKET_DRIFT"
        else range_fresh_passive_rail_passed
        if normalized_current_binding_mode == "RANGE_FRESH_PASSIVE_RAIL"
        else False
    )
    expected_spread_pips = (
        (ask - bid) * pip_factor
        if broker_quote_valid and pip_factor is not None
        else None
    )
    spread_matches_broker = bool(
        spread_pips is not None
        and expected_spread_pips is not None
        and math.isclose(
            spread_pips,
            expected_spread_pips,
            rel_tol=1e-9,
            abs_tol=1e-6,
        )
    )
    if order_type in {"STOP", "STOP_ENTRY", "STOP-ENTRY", "LIMIT"}:
        entry_binding_basis = "PENDING_INTENT_ENTRY_EQUALS_RISK_ENTRY"
        entry_binding_passed = bool(
            intent_entry is not None
            and entry_price is not None
            and math.isclose(
                intent_entry,
                entry_price,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
    elif order_type == "MARKET":
        entry_binding_basis = "MARKET_RISK_ENTRY_EQUALS_FRESH_BROKER_ASK_OR_BID"
        entry_binding_passed = bool(
            entry_price is not None
            and broker_entry is not None
            and math.isclose(
                entry_price,
                broker_entry,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
    else:
        entry_binding_basis = "SUPPORTED_ORDER_TYPE_REQUIRED"
        entry_binding_passed = False
    price_reward_risk: float | None = None
    executable_forecast_target: float | None = None
    executable_forecast_invalidation: float | None = None
    range_rails_bound = False
    range_expected_rail_entry: float | None = None
    range_entry_exact_rail_bound = False
    range_entry_within_rail_zone = False
    if range_lane:
        range_support_bound = bool(
            forecast_range_low is not None
            and range_support is not None
            and math.isclose(
                forecast_range_low,
                range_support,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        range_resistance_bound = bool(
            forecast_range_high is not None
            and range_resistance is not None
            and math.isclose(
                forecast_range_high,
                range_resistance,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        range_model_bound = bool(
            str(metadata.get("geometry_model") or "").strip().upper()
            == "RANGE_RAIL_LIMIT"
        )
        range_flags_bound = bool(
            metadata.get("range_tp_is_inside_box") is True
            and metadata.get("range_sl_outside_box") is True
        )
        range_entry_side_bound = bool(
            side == "LONG" and range_entry_side == "SUPPORT"
            or side == "SHORT" and range_entry_side == "RESISTANCE"
        )
        box_width = (
            forecast_range_high - forecast_range_low
            if forecast_range_low is not None
            and forecast_range_high is not None
            else None
        )
        rail_entry_buffer = (
            max(
                (1.0 / pip_factor) / 10.0,
                min(
                    spread_pips * 0.5 * (1.0 / pip_factor),
                    box_width * 0.25,
                ),
            )
            if pip_factor is not None
            and spread_pips is not None
            and box_width is not None
            and box_width > 2.0 * ((1.0 / pip_factor) / 10.0)
            else None
        )
        price_precision = (
            3 if pip_factor == 100 else 5 if pip_factor == 10000 else None
        )
        range_expected_rail_entry = (
            round(
                forecast_range_low + rail_entry_buffer
                if side == "LONG"
                else forecast_range_high - rail_entry_buffer,
                price_precision,
            )
            if rail_entry_buffer is not None
            and forecast_range_low is not None
            and forecast_range_high is not None
            and price_precision is not None
            and side in {"LONG", "SHORT"}
            else None
        )
        range_entry_exact_rail_bound = bool(
            range_expected_rail_entry is not None
            and entry_price is not None
            and math.isclose(
                entry_price,
                range_expected_rail_entry,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        range_entry_distance_from_rail = (
            entry_price - forecast_range_low
            if side == "LONG"
            and entry_price is not None
            and forecast_range_low is not None
            else forecast_range_high - entry_price
            if side == "SHORT"
            and entry_price is not None
            and forecast_range_high is not None
            else None
        )
        range_entry_within_rail_zone = bool(
            range_entry_distance_from_rail is not None
            and box_width is not None
            and range_entry_distance_from_rail > 0.0
            and range_entry_distance_from_rail
            <= box_width * 0.25 + 1e-12
        )
        range_entry_geometry_bound = bool(
            range_entry_exact_rail_bound
            if normalized_current_binding_mode == "EXACT_SNAPSHOT_MID"
            else range_entry_within_rail_zone
            if normalized_current_binding_mode == "RANGE_FRESH_PASSIVE_RAIL"
            else False
        )
        range_rails_bound = bool(
            forecast_range_low is not None
            and forecast_range_high is not None
            and forecast_range_low < forecast_range_high
            and forecast_current is not None
            and forecast_range_low < forecast_current < forecast_range_high
            and range_fresh_quote_inside_rails
            and range_support_bound
            and range_resistance_bound
            and range_model_bound
            and range_flags_bound
            and range_entry_side_bound
            and range_entry_geometry_bound
            and range_fresh_limit_remains_passive
            and order_vehicle == "LIMIT"
        )
        if side == "LONG":
            required_relation = (
                "order_sl < range_low-half_spread <= entry < order_tp "
                "<= range_high-half_spread"
            )
            executable_forecast_target = (
                forecast_range_high - half_spread_price
                if forecast_range_high is not None
                and half_spread_price is not None
                else None
            )
            executable_forecast_invalidation = (
                forecast_range_low - half_spread_price
                if forecast_range_low is not None
                and half_spread_price is not None
                else None
            )
            geometry_contains_order = bool(
                range_rails_bound
                and None
                not in (
                    executable_forecast_invalidation,
                    entry_price,
                    order_tp,
                    executable_forecast_target,
                    order_sl,
                )
                and order_sl < executable_forecast_invalidation
                <= entry_price
                < order_tp
                <= executable_forecast_target
            )
            if geometry_contains_order:
                price_reward_risk = (order_tp - entry_price) / (
                    entry_price - order_sl
                )
        elif side == "SHORT":
            required_relation = (
                "order_sl > range_high+half_spread >= entry > order_tp "
                ">= range_low+half_spread"
            )
            executable_forecast_target = (
                forecast_range_low + half_spread_price
                if forecast_range_low is not None
                and half_spread_price is not None
                else None
            )
            executable_forecast_invalidation = (
                forecast_range_high + half_spread_price
                if forecast_range_high is not None
                and half_spread_price is not None
                else None
            )
            geometry_contains_order = bool(
                range_rails_bound
                and None
                not in (
                    executable_forecast_target,
                    order_tp,
                    entry_price,
                    executable_forecast_invalidation,
                    order_sl,
                )
                and order_sl > executable_forecast_invalidation
                >= entry_price
                > order_tp
                >= executable_forecast_target
            )
            if geometry_contains_order:
                price_reward_risk = (entry_price - order_tp) / (
                    order_sl - entry_price
                )
        else:
            required_relation = "LONG_OR_SHORT_RANGE_RAIL_REQUIRED"
            geometry_contains_order = False
    elif side == "LONG":
        required_relation = (
            "order_sl <= forecast_invalidation-half_spread < forecast_current "
            "<= entry < order_tp <= forecast_target-half_spread"
        )
        executable_forecast_target = (
            forecast_target - half_spread_price
            if forecast_target is not None and half_spread_price is not None
            else None
        )
        executable_forecast_invalidation = (
            forecast_invalidation - half_spread_price
            if forecast_invalidation is not None and half_spread_price is not None
            else None
        )
        geometry_contains_order = bool(
            None not in (
                executable_forecast_invalidation,
                forecast_current,
                entry_price,
                order_tp,
                executable_forecast_target,
                order_sl,
            )
            and order_sl
            <= executable_forecast_invalidation
            < forecast_current
            <= entry_price
            < order_tp
            <= executable_forecast_target
        )
        if geometry_contains_order:
            price_reward_risk = (order_tp - entry_price) / (entry_price - order_sl)
    elif side == "SHORT":
        required_relation = (
            "order_sl >= forecast_invalidation+half_spread > forecast_current "
            ">= entry > order_tp >= forecast_target+half_spread"
        )
        executable_forecast_target = (
            forecast_target + half_spread_price
            if forecast_target is not None and half_spread_price is not None
            else None
        )
        executable_forecast_invalidation = (
            forecast_invalidation + half_spread_price
            if forecast_invalidation is not None and half_spread_price is not None
            else None
        )
        geometry_contains_order = bool(
            None not in (
                executable_forecast_target,
                order_tp,
                entry_price,
                forecast_current,
                executable_forecast_invalidation,
                order_sl,
            )
            and order_sl
            >= executable_forecast_invalidation
            > forecast_current
            >= entry_price
            > order_tp
            >= executable_forecast_target
        )
        if geometry_contains_order:
            price_reward_risk = (entry_price - order_tp) / (order_sl - entry_price)
    else:
        required_relation = "LONG_OR_SHORT_DIRECTIONAL_FORECAST_REQUIRED"
        geometry_contains_order = False

    p_lower = (
        hit_rate_wilson_lower(economic_hit_rate, economic_samples)
        if not range_lane
        and economic_hit_rate is not None
        and economic_samples is not None
        and economic_samples > 0
        else None
    )
    expected_loss_pips: float | None = None
    expected_reward_pips: float | None = None
    if (
        pip_factor is not None
        and entry_price is not None
        and order_sl is not None
        and order_tp is not None
    ):
        if side == "LONG":
            expected_loss_pips = (entry_price - order_sl) * pip_factor
            expected_reward_pips = (order_tp - entry_price) * pip_factor
        elif side == "SHORT":
            expected_loss_pips = (order_sl - entry_price) * pip_factor
            expected_reward_pips = (entry_price - order_tp) * pip_factor
    derived_reward_risk = (
        reward_jpy / risk_jpy
        if reward_jpy is not None and risk_jpy is not None
        else None
    )
    reward_risk_consistent = bool(
        derived_reward_risk is not None
        and reported_reward_risk is not None
        and price_reward_risk is not None
        and math.isclose(
            reported_reward_risk,
            derived_reward_risk,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        and math.isclose(
            price_reward_risk,
            derived_reward_risk,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    )
    risk_metrics_consistent = bool(
        reward_risk_consistent
        and loss_pips is not None
        and reward_pips is not None
        and jpy_per_pip is not None
        and expected_jpy_per_pip is not None
        and expected_loss_pips is not None
        and expected_reward_pips is not None
        and math.isclose(
            loss_pips,
            expected_loss_pips,
            rel_tol=1e-9,
            abs_tol=1e-6,
        )
        and math.isclose(
            reward_pips,
            expected_reward_pips,
            rel_tol=1e-9,
            abs_tol=1e-6,
        )
        and math.isclose(
            jpy_per_pip,
            expected_jpy_per_pip,
            rel_tol=1e-9,
            abs_tol=1e-6,
        )
        and math.isclose(
            risk_jpy,
            loss_pips * jpy_per_pip,
            rel_tol=1e-9,
            abs_tol=1e-6,
        )
        and math.isclose(
            reward_jpy,
            reward_pips * jpy_per_pip,
            rel_tol=1e-9,
            abs_tol=1e-6,
        )
    )
    entry_slippage_stress_pips = (
        0.0
        if market_entry_slippage_embedded or order_vehicle == "LIMIT"
        else market_entry_adverse_p95_pips
        if cost_floor_valid
        else None
    )
    # The normal bid/ask spread is already present in executable entry/rail
    # geometry.  This is a separate adverse-exit stress: use at least the fresh
    # spread-sized shock, and never less than the audited broker-protection p95.
    exit_execution_stress_pips = (
        max(float(spread_pips), float(audited_exit_adverse_p95_pips))
        if cost_floor_valid
        and spread_pips is not None
        and audited_exit_adverse_p95_pips is not None
        else None
    )
    entry_slippage_stress_jpy = (
        entry_slippage_stress_pips * jpy_per_pip
        if entry_slippage_stress_pips is not None
        and jpy_per_pip is not None
        else None
    )
    exit_execution_stress_jpy = (
        exit_execution_stress_pips * jpy_per_pip
        if exit_execution_stress_pips is not None
        and jpy_per_pip is not None
        else None
    )
    financing_stress_jpy = (
        financing_stress_jpy_per_unit * units
        if financing_stress_jpy_per_unit is not None and units is not None
        else None
    )
    outcome_cost_jpy = (
        exit_execution_stress_jpy + financing_stress_jpy
        if exit_execution_stress_jpy is not None
        and financing_stress_jpy is not None
        else None
    )
    additional_cost_jpy = (
        entry_slippage_stress_jpy + outcome_cost_jpy
        if entry_slippage_stress_jpy is not None
        and outcome_cost_jpy is not None
        else None
    )
    net_reward_jpy = (
        reward_jpy - additional_cost_jpy
        if reward_jpy is not None and additional_cost_jpy is not None
        else None
    )
    net_risk_jpy = (
        risk_jpy + additional_cost_jpy
        if risk_jpy is not None and additional_cost_jpy is not None
        else None
    )
    net_reward_risk = (
        net_reward_jpy / net_risk_jpy
        if net_reward_jpy is not None
        and net_reward_jpy > 0.0
        and net_risk_jpy is not None
        and net_risk_jpy > 0.0
        else None
    )
    ev_lower_jpy = (
        p_lower * reward_jpy
        - (1.0 - p_lower) * risk_jpy
        - additional_cost_jpy
        if p_lower is not None
        and reward_jpy is not None
        and risk_jpy is not None
        and additional_cost_jpy is not None
        else None
    )
    full_kelly_risk_fraction = (
        max(0.0, p_lower - (1.0 - p_lower) / net_reward_risk)
        if p_lower is not None
        and net_reward_risk is not None
        and net_reward_risk > 0.0
        else None
    )
    quarter_kelly_risk_nav_pct = (
        CAPITAL_ALLOCATION_KELLY_FRACTION * full_kelly_risk_fraction * 100.0
        if full_kelly_risk_fraction is not None
        else None
    )
    base_risk_nav_pct = (
        net_risk_jpy / nav_jpy * 100.0
        if net_risk_jpy is not None and nav_jpy is not None
        else None
    )
    risk_budget_jpy = (
        nav_jpy * quarter_kelly_risk_nav_pct / 100.0
        if nav_jpy is not None and quarter_kelly_risk_nav_pct is not None
        else None
    )
    ev_strictly_positive = bool(
        ev_lower_jpy is not None
        and ev_lower_jpy > 0.0
        and not math.isclose(
            ev_lower_jpy,
            0.0,
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
    )
    range_tp_prebounded = bool(
        range_lane
        and range_economic_basis_valid
        and order_vehicle == "LIMIT"
    )
    range_risk_cap_passed = bool(
        range_tp_prebounded
        and net_risk_jpy is not None
        and net_reward_jpy is not None
        and net_reward_jpy > 0.0
        and intent_max_loss_jpy is not None
        and net_risk_jpy <= intent_max_loss_jpy + 1e-9
        and nav_jpy is not None
        and net_risk_jpy / nav_jpy * 100.0
        <= 1.0 + 1e-12
    )

    bypass_reason = (
        "M15_RECOVERY_MICRO_PREBOUNDED_CONTRACT"
        if m15_recovery_prebounded
        else "PREDICTIVE_SCOUT_PREBOUNDED_CONTRACT"
        if predictive_scout
        else "HEDGE_RISK_REDUCTION_PREBOUNDED_CONTRACT"
        if hedge
        else None
    )
    if not method_scope_consistent:
        reason = "METHOD_SCOPE_MISMATCH"
        max_multiple = 0.0
    elif m15_recovery_prebounded:
        # Recovery direction, vehicle, TP proof and 999-unit cap are proved by
        # the separate M15 content-addressed chain. Do not coerce it into the
        # ordinary calibration/market-probability contract here.
        reason = "M15_RECOVERY_MICRO_PREBOUNDED_CONTRACT"
        max_multiple = 1.0
    elif not direction_side_aligned:
        reason = "FORECAST_DIRECTION_SIDE_MISMATCH"
        max_multiple = 0.0
    elif not calibration_identity_valid:
        reason = "FORECAST_CALIBRATION_IDENTITY_MISMATCH"
        max_multiple = 0.0
    elif bypass_reason is not None:
        reason = bypass_reason
        max_multiple = 1.0
    elif not entry_binding_passed:
        reason = "ORDER_ENTRY_RISK_METRICS_BINDING_INVALID"
        max_multiple = 0.0
    elif range_lane and not range_economic_basis_valid:
        reason = "RANGE_TRADE_ECONOMIC_BASIS_MISSING_INVALID_OR_UNBOUND"
        max_multiple = 0.0
    elif order_vehicle != "MARKET" and not range_lane:
        reason = "FORECAST_ECONOMIC_PROBABILITY_ENTRY_VEHICLE_UNBOUND"
        max_multiple = 0.0
    elif not cost_floor_valid:
        reason = "NET_EXECUTION_COST_FLOOR_MISSING_INVALID_OR_STALE"
        max_multiple = 0.0
    elif not forecast_current_binding_passed:
        reason = (
            "FORECAST_CURRENT_BROKER_MID_MISMATCH"
            if normalized_current_binding_mode == "EXACT_SNAPSHOT_MID"
            else "FORECAST_CURRENT_FRESH_DIRECTIONAL_DRIFT_INVALID"
            if normalized_current_binding_mode
            == "DIRECTIONAL_FRESH_MARKET_DRIFT"
            else "FORECAST_RANGE_FRESH_PASSIVE_RAIL_INVALID"
            if normalized_current_binding_mode
            == "RANGE_FRESH_PASSIVE_RAIL"
            else "FORECAST_CURRENT_BINDING_MODE_INVALID"
        )
        max_multiple = 0.0
    elif not spread_matches_broker:
        reason = "BROKER_SPREAD_RISK_METRICS_MISMATCH"
        max_multiple = 0.0
    elif not geometry_contains_order:
        reason = "FORECAST_RAIL_DOES_NOT_CONSERVATIVELY_CONTAIN_ORDER"
        max_multiple = 0.0
    elif risk_jpy is None or reward_jpy is None or not risk_metrics_consistent:
        reason = "RISK_REWARD_JPY_GEOMETRY_INCONSISTENT"
        max_multiple = 0.0
    elif range_tp_prebounded and not range_risk_cap_passed:
        reason = "TP_PROVEN_RANGE_PREBOUNDED_RISK_CAP_INVALID"
        max_multiple = 0.0
    elif range_tp_prebounded:
        reason = "TP_PROVEN_RANGE_NONMARKET_PREBOUNDED_CONTRACT"
        max_multiple = 1.0
    elif economic_hit_rate is None:
        reason = "ECONOMIC_HIT_RATE_MISSING_OR_INVALID"
        max_multiple = 0.0
    elif (
        economic_samples is None
        or economic_samples < minimum_economic_samples
    ):
        reason = "ECONOMIC_SAMPLE_FLOOR_NOT_MET"
        max_multiple = 0.0
    elif p_lower is None:
        reason = "ECONOMIC_WILSON_LOWER_UNAVAILABLE"
        max_multiple = 0.0
    elif (
        additional_cost_jpy is None
        or net_reward_jpy is None
        or net_reward_jpy <= 0.0
        or net_risk_jpy is None
        or net_risk_jpy <= 0.0
    ):
        reason = "NET_EXECUTION_COST_EXHAUSTS_REWARD"
        max_multiple = 0.0
    elif nav_jpy is None or base_risk_nav_pct is None or base_risk_nav_pct <= 0.0:
        reason = "BROKER_NAV_MISSING_OR_INVALID"
        max_multiple = 0.0
    elif not ev_strictly_positive:
        reason = "CONSERVATIVE_EV_NOT_POSITIVE"
        max_multiple = 0.0
    elif (
        full_kelly_risk_fraction is None
        or quarter_kelly_risk_nav_pct is None
        or quarter_kelly_risk_nav_pct <= 0.0
    ):
        reason = "QUARTER_KELLY_RISK_NOT_POSITIVE"
        max_multiple = 0.0
    else:
        # Authorize by NAV percentage, not an absolute-JPY threshold.  The JPY
        # budget below is explanatory truth for this exact broker snapshot.
        max_multiple = min(1.0, quarter_kelly_risk_nav_pct / base_risk_nav_pct)
        reason = (
            "POSITIVE_EV_QUARTER_KELLY_CEILING_PROVEN"
            if max_multiple + 1e-12 >= min(CAPITAL_ALLOCATION_SIZE_MULTIPLES)
            else "QUARTER_KELLY_CAP_BELOW_MINIMUM_MULTIPLE"
        )

    range_tp_contract_authorized = (
        reason == "TP_PROVEN_RANGE_NONMARKET_PREBOUNDED_CONTRACT"
    )
    evidence = {
        "contract": CAPITAL_ALLOCATION_NUMERIC_CEILING_CONTRACT,
        "applies_to_ordinary_new_entry": bool(
            bypass_reason is None and not range_tp_contract_authorized
        ),
        "inputs": {
            "side": side or None,
            "order_type": order_type or None,
            "units": units,
            "m15_recovery_prebounded": bool(m15_recovery_prebounded),
            "authoritative_market_context_method": (
                None if cost_method == "UNKNOWN" else cost_method
            ),
            "metadata_method": metadata_method or None,
            "method_scope_consistent": method_scope_consistent,
            "forecast_direction": forecast_direction or None,
            "calibration_name": calibration_name or None,
            "expected_calibration_name": expected_calibration_name,
            "entry_price": _rounded_evidence_number(entry_price),
            "intent_entry": _rounded_evidence_number(intent_entry),
            "broker_executable_entry": _rounded_evidence_number(broker_entry),
            "broker_bid": _rounded_evidence_number(bid),
            "broker_ask": _rounded_evidence_number(ask),
            "broker_mid": _rounded_evidence_number(broker_mid),
            "half_spread_price": _rounded_evidence_number(half_spread_price),
            "order_tp": _rounded_evidence_number(order_tp),
            "order_sl": _rounded_evidence_number(order_sl),
            "forecast_current_price_context": _rounded_evidence_number(
                forecast_current
            ),
            "forecast_current_binding_mode": (
                normalized_current_binding_mode or None
            ),
            "signed_fresh_mid_drift_pips": _rounded_evidence_number(
                signed_fresh_mid_drift_pips
            ),
            "forecast_target_price": _rounded_evidence_number(forecast_target),
            "forecast_invalidation_price": _rounded_evidence_number(
                forecast_invalidation
            ),
            "forecast_range_low_price": _rounded_evidence_number(
                forecast_range_low
            ),
            "forecast_range_high_price": _rounded_evidence_number(
                forecast_range_high
            ),
            "forecast_directional_economic_hit_rate_context_only": (
                _rounded_evidence_number(forecast_economic_hit_rate)
            ),
            "forecast_directional_economic_samples_context_only": (
                forecast_economic_samples
            ),
            "economic_hit_rate": _rounded_evidence_number(economic_hit_rate),
            "economic_samples": economic_samples,
            "minimum_economic_samples": minimum_economic_samples,
            "range_trade_economic_basis": (
                normalized_range_basis if range_lane else None
            ),
            "range_trade_economic_basis_valid": range_economic_basis_valid,
            "headline_hit_rate_context_only": _rounded_evidence_number(
                headline_hit_rate
            ),
            "headline_samples_context_only": headline_samples,
            "timeout_rate_context": _rounded_evidence_number(timeout_rate),
            "account_nav_jpy_snapshot": _rounded_evidence_number(nav_jpy),
            "spread_pips": _rounded_evidence_number(spread_pips),
            "broker_spread_pips": _rounded_evidence_number(
                expected_spread_pips
            ),
            "broker_quote_to_jpy": _rounded_evidence_number(quote_to_jpy),
        },
        "geometry": {
            "required_relation": required_relation,
            "direction_side_aligned": direction_side_aligned,
            "calibration_identity_valid": calibration_identity_valid,
            "entry_binding_basis": entry_binding_basis,
            "entry_binding_passed": entry_binding_passed,
            "forecast_current_matches_broker_mid": (
                forecast_current_matches_broker_mid
            ),
            "directional_fresh_market_drift_passed": (
                directional_fresh_market_drift_passed
            ),
            "range_fresh_quote_inside_rails": range_fresh_quote_inside_rails,
            "range_fresh_limit_remains_passive": (
                range_fresh_limit_remains_passive
            ),
            "range_fresh_passive_rail_passed": (
                range_fresh_passive_rail_passed
            ),
            "forecast_current_binding_passed": (
                forecast_current_binding_passed
            ),
            "spread_matches_broker": spread_matches_broker,
            "range_rails_bound": range_rails_bound,
            "range_support": _rounded_evidence_number(range_support),
            "range_resistance": _rounded_evidence_number(range_resistance),
            "range_entry_side": range_entry_side or None,
            "range_expected_rail_entry": _rounded_evidence_number(
                range_expected_rail_entry
            ),
            "range_entry_exact_rail_bound": range_entry_exact_rail_bound,
            "range_entry_within_rail_zone": range_entry_within_rail_zone,
            "executable_forecast_target": _rounded_evidence_number(
                executable_forecast_target
            ),
            "executable_forecast_invalidation": _rounded_evidence_number(
                executable_forecast_invalidation
            ),
            "forecast_outcome_conservatively_contains_order": geometry_contains_order,
            "price_reward_risk": _rounded_evidence_number(price_reward_risk),
            "risk_metrics_reward_risk": _rounded_evidence_number(
                reported_reward_risk
            ),
            "reward_risk_consistent": reward_risk_consistent,
            "expected_loss_pips": _rounded_evidence_number(expected_loss_pips),
            "risk_metrics_loss_pips": _rounded_evidence_number(loss_pips),
            "expected_reward_pips": _rounded_evidence_number(
                expected_reward_pips
            ),
            "risk_metrics_reward_pips": _rounded_evidence_number(reward_pips),
            "risk_metrics_jpy_per_pip": _rounded_evidence_number(jpy_per_pip),
            "expected_jpy_per_pip": _rounded_evidence_number(
                expected_jpy_per_pip
            ),
            "risk_metrics_consistent": risk_metrics_consistent,
            "intent_max_loss_jpy": _rounded_evidence_number(
                intent_max_loss_jpy
            ),
            "range_tp_prebounded": range_tp_prebounded,
            "range_risk_cap_passed": range_risk_cap_passed,
            "passed": (
                direction_side_aligned
                and calibration_identity_valid
                and entry_binding_passed
                and forecast_current_binding_passed
                and spread_matches_broker
                and geometry_contains_order
                and risk_metrics_consistent
                and (not range_tp_prebounded or range_risk_cap_passed)
            ),
        },
        "probability": {
            "basis": probability_basis,
            "headline_hit_rate_may_not_substitute": True,
            "range_box_hold_rate_may_not_substitute_for_trade_ev": True,
            "shared_live_wilson_floor_rechecked": False,
            "shared_live_wilson_floor_basis": (
                "LIVE_READY_ALREADY_GATES_PRECISION;THIS_LAYER_SIZES_BY_EV_AND_KELLY"
            ),
            "economic_wilson95_lower": _rounded_evidence_number(p_lower),
            "range_lane_evidence_wilson95_lower": (
                _rounded_evidence_number(recomputed_range_wilson)
                if range_lane
                else None
            ),
        },
        "ev_lower": {
            "formula": (
                None
                if range_tp_contract_authorized
                else "p_lower*reward_jpy-(1-p_lower)*risk_jpy-additional_cost_jpy"
            ),
            "risk_jpy_snapshot": _rounded_evidence_number(risk_jpy),
            "reward_jpy_snapshot": _rounded_evidence_number(reward_jpy),
            "gross_reward_risk": _rounded_evidence_number(derived_reward_risk),
            "net_reward_jpy_snapshot": _rounded_evidence_number(net_reward_jpy),
            "net_risk_jpy_snapshot": _rounded_evidence_number(net_risk_jpy),
            "net_reward_risk": _rounded_evidence_number(net_reward_risk),
            "market_entry_slippage_embedded": bool(
                market_entry_slippage_embedded
            ),
            "market_entry_adverse_p95_pips": _rounded_evidence_number(
                market_entry_adverse_p95_pips
            ),
            "entry_slippage_stress_pips": _rounded_evidence_number(
                entry_slippage_stress_pips
            ),
            "entry_slippage_stress_jpy": _rounded_evidence_number(
                entry_slippage_stress_jpy
            ),
            "audited_protected_exit_adverse_p95_pips": (
                _rounded_evidence_number(audited_exit_adverse_p95_pips)
            ),
            "fresh_spread_sized_exit_stress_pips": _rounded_evidence_number(
                spread_pips
            ),
            "exit_execution_stress_pips": _rounded_evidence_number(
                exit_execution_stress_pips
            ),
            "exit_execution_stress_jpy": _rounded_evidence_number(
                exit_execution_stress_jpy
            ),
            "financing_adverse_stress_jpy_per_unit": (
                _rounded_evidence_number(financing_stress_jpy_per_unit)
            ),
            "financing_stress_jpy": _rounded_evidence_number(
                financing_stress_jpy
            ),
            "outcome_cost_jpy": _rounded_evidence_number(outcome_cost_jpy),
            "additional_cost_jpy": _rounded_evidence_number(
                additional_cost_jpy
            ),
            "cost_basis": (
                "RISK_METRICS_EXECUTABLE_BID_ASK_GEOMETRY_INCLUDES_SPREAD;"
                + (
                    "WORST_FILL_PRICE_EMBEDS_ENTRY_ADVERSE_MOVE;"
                    "ENTRY_P95_NOT_DOUBLE_COUNTED;"
                    if market_entry_slippage_embedded
                    else "PASSIVE_LIMIT_CANNOT_FILL_WORSE_THAN_LIMIT_PRICE;"
                    if order_vehicle == "LIMIT"
                    else "ENTRY_P95_ADDED_AS_SEPARATE_STRESS;"
                )
                +
                "EXTRA_EXIT_STRESS_MAX_FRESH_SPREAD_AND_AUDITED_PROTECTION_P95;"
                "FINANCING_MAX_GLOBAL_AND_EXACT_WILSON95_UPPER_STRESS;"
                "CREDITS_DO_NOT_OFFSET"
            ),
            "value_jpy_snapshot": _rounded_evidence_number(ev_lower_jpy),
            "positive": ev_strictly_positive,
        },
        "kelly": {
            "formula": (
                None
                if range_tp_contract_authorized
                else "p_lower-(1-p_lower)/(net_reward_jpy/net_risk_jpy)"
            ),
            "fractional_kelly_multiplier": CAPITAL_ALLOCATION_KELLY_FRACTION,
            "full_kelly_risk_fraction": _rounded_evidence_number(
                full_kelly_risk_fraction
            ),
            "base_risk_nav_pct": _rounded_evidence_number(base_risk_nav_pct),
            "quarter_kelly_risk_nav_pct": _rounded_evidence_number(
                quarter_kelly_risk_nav_pct
            ),
            "decision_basis": (
                "PREBOUNDED_M15_RECOVERY_BASE_UNITS"
                if m15_recovery_prebounded
                else "PREBOUNDED_EXACT_TP_HARVEST_BASE_UNITS"
                if range_tp_contract_authorized
                else "NAV_PERCENT_RATIO"
            ),
            "risk_budget_jpy_snapshot_explanation": _rounded_evidence_number(
                risk_budget_jpy
            ),
        },
        "execution_cost_floor": cost_floor,
        "execution_cost_floor_digest_valid": cost_floor_digest_valid,
        "execution_cost_floor_valid": cost_floor_valid,
        "max_multiple": _rounded_evidence_number(max_multiple),
        "reason": reason,
    }
    return evidence, max_multiple


def _strict_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _strict_probability(value: Any) -> float | None:
    parsed = _strict_nonnegative_number(value)
    if parsed is None or parsed > 1.0:
        return None
    return parsed


def _strict_nonnegative_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        return None
    return parsed


def _strict_positive_number(value: Any) -> float | None:
    parsed = _strict_nonnegative_number(value)
    return parsed if parsed is not None and parsed > 0.0 else None


def _rounded_evidence_number(value: float | None) -> float | None:
    return round(float(value), 12) if value is not None else None


def _capital_allocation_lane(
    result: Mapping[str, Any] | None,
    *,
    current_exact_vehicle_net_metrics: Mapping[
        tuple[str, str, str, str], Mapping[str, Any]
    ]
    | None = None,
    current_exact_vehicle_tp_metrics: Mapping[
        tuple[str, str, str, str], Mapping[str, Any]
    ]
    | None = None,
    execution_ledger_surface_sha256: str = "",
    account_nav_jpy: float | None = None,
    broker_snapshot: Mapping[str, Any] | None = None,
    dynamic_tf_policy_sources: Mapping[str, Any] | None = None,
    canonical_forecast_context_sha256: str | None = None,
    execution_cost_floor: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not isinstance(result, Mapping):
        return None
    intent = result.get("intent") if isinstance(result.get("intent"), Mapping) else {}
    metadata = intent.get("metadata") if isinstance(intent.get("metadata"), Mapping) else {}
    market_context = (
        intent.get("market_context")
        if isinstance(intent.get("market_context"), Mapping)
        else {}
    )
    risk_metrics = (
        result.get("risk_metrics")
        if isinstance(result.get("risk_metrics"), Mapping)
        else {}
    )
    raw_base_units = intent.get("units")
    base_units = (
        abs(raw_base_units)
        if isinstance(raw_base_units, int) and not isinstance(raw_base_units, bool)
        else 0
    )
    method = str(market_context.get("method") or "").strip().upper()
    metadata_method = str(metadata.get("method") or "").strip().upper()
    method_scope_consistent = bool(
        method and (not metadata_method or metadata_method == method)
    )
    predictive_scout_claimed = predictive_scout_geometry_claimed(
        dict(metadata),
        pair=str(intent.get("pair") or ""),
        side=str(intent.get("side") or ""),
        order_type=str(intent.get("order_type") or ""),
        method=method if method else None,
    )
    predictive_scout = bool(
        predictive_scout_claimed
        and predictive_scout_metadata_supported(dict(metadata))
        and str(intent.get("order_type") or "").strip().upper() == "LIMIT"
        and method_scope_consistent
        and method == "BREAKOUT_FAILURE"
        and str(metadata.get("desk") or "").strip().lower() == "failure_trader"
        and str(metadata.get("campaign_role") or "").strip().upper()
        == "BIDASK_REPLAY_CONTRARIAN_SCOUT"
        and metadata.get("attach_take_profit_on_fill") is True
        and str(metadata.get("tp_execution_mode") or "").strip().upper()
        == "ATTACHED_TECHNICAL_TP"
    )
    hedge = str(metadata.get("position_intent") or "").strip().upper() == "HEDGE"
    tp_expectancy = _optional_float(
        metadata.get("capture_take_profit_expectancy_jpy")
    )
    raw_tp_trades = metadata.get("capture_take_profit_trades")
    tp_trades = (
        raw_tp_trades
        if isinstance(raw_tp_trades, int) and not isinstance(raw_tp_trades, bool)
        else 0
    )
    raw_tp_losses = metadata.get("capture_take_profit_losses")
    tp_losses = (
        raw_tp_losses
        if isinstance(raw_tp_losses, int) and not isinstance(raw_tp_losses, bool)
        else None
    )
    capture_status = str(
        metadata.get("capture_economics_status") or ""
    ).strip().upper()
    overall_expectancy = _optional_float(metadata.get("capture_expectancy_jpy"))
    if overall_expectancy is None and capture_status and capture_status != "NEGATIVE_EXPECTANCY":
        overall_expectancy = 0.0
    exact_vehicle_tp_edge = _exact_vehicle_take_profit_edge_proven(
        intent=intent,
        metadata=metadata,
        method=method,
        tp_expectancy=tp_expectancy,
        tp_trades=tp_trades,
        tp_losses=tp_losses,
        current_metrics=current_exact_vehicle_tp_metrics,
        execution_ledger_surface_sha256=execution_ledger_surface_sha256,
    )
    exact_vehicle_net_edge = _exact_vehicle_net_edge_evidence(
        intent=intent,
        metadata=metadata,
        method=method,
        current_metrics=current_exact_vehicle_net_metrics,
        execution_ledger_surface_sha256=execution_ledger_surface_sha256,
    )
    positive_tp_edge = bool(
        exact_vehicle_tp_edge
        and tp_expectancy is not None
        and tp_expectancy > 0.0
        and tp_trades >= CAPITAL_ALLOCATION_MIN_EXACT_TP_TRADES
        and tp_losses == 0
        and exact_vehicle_net_edge["blocks_tp_exception"] is not True
    )
    range_economic_basis = _range_lane_economic_basis(
        intent=intent,
        metadata=metadata,
        method=method,
        exact_vehicle_net_edge=exact_vehicle_net_edge,
        positive_tp_edge=positive_tp_edge,
        execution_ledger_surface_sha256=execution_ledger_surface_sha256,
    )
    range_tp_proven_breakout_failure = bool(
        method == "BREAKOUT_FAILURE"
        and isinstance(range_economic_basis, Mapping)
    )
    technical_context = normalize_forecast_technical_context_evidence(
        metadata.get("forecast_technical_context"),
        pair=str(intent.get("pair") or "").strip().upper() or None,
        current_price=metadata.get("forecast_current_price"),
    )
    forecast_binding = _directional_forecast_trade_binding(
        side=intent.get("side"),
        forecast_direction=metadata.get("forecast_direction"),
        calibration_name=metadata.get("forecast_directional_calibration_name"),
        method=method,
    )
    technical_context_envelope_valid = technical_context.get("status") == "VALID"
    regime_family_binding = _regime_family_allocation_binding(
        technical_context=technical_context,
        pair=intent.get("pair"),
        side=intent.get("side"),
        method=method,
        forecast_direction=metadata.get("forecast_direction"),
        explicit_receipt_sha256=metadata.get(
            "forecast_regime_family_weighting_sha256"
        ),
        explicit_selected_method=metadata.get(
            "forecast_regime_family_selected_method"
        ),
        explicit_family_direction=metadata.get(
            "forecast_regime_family_direction"
        ),
        range_tp_proven_breakout_failure=(
            range_tp_proven_breakout_failure
        ),
        canonical_policy_sources=dynamic_tf_policy_sources,
        canonical_forecast_context_sha256=(
            canonical_forecast_context_sha256
        ),
    )
    technical_context_valid = bool(
        technical_context_envelope_valid
        and forecast_binding["direction_side_aligned"]
        and forecast_binding["calibration_identity_valid"]
        and regime_family_binding["passed"]
    )
    m15_recovery = _m15_recovery_allocation_contract(
        result=result,
        intent=intent,
        metadata=metadata,
        method=method,
        base_units=base_units,
        positive_tp_edge=positive_tp_edge,
    )
    m15_recovery_claimed = m15_recovery["claimed"] is True
    m15_recovery_valid = m15_recovery["valid"] is True
    allocation_context_valid = (
        m15_recovery_valid
        if m15_recovery_claimed
        else technical_context_valid
    )
    broker_bid, broker_ask = _broker_snapshot_bid_ask(
        broker_snapshot or {},
        pair=str(intent.get("pair") or "").strip().upper(),
    )
    numeric_ceiling, numeric_max_multiple = _capital_allocation_numeric_ceiling(
        intent=intent,
        metadata=metadata,
        risk_metrics=risk_metrics,
        account_nav_jpy=account_nav_jpy,
        broker_bid=broker_bid,
        broker_ask=broker_ask,
        broker_quote_to_jpy=_broker_snapshot_quote_to_jpy(
            broker_snapshot or {},
            pair=str(intent.get("pair") or "").strip().upper(),
        ),
        predictive_scout=predictive_scout,
        hedge=hedge,
        range_economic_basis=range_economic_basis,
        execution_cost_floor=execution_cost_floor,
        m15_recovery_prebounded=m15_recovery_valid,
    )
    candidate_size_multiples = [
        multiple
        for multiple in (
            [1.0]
            if predictive_scout or hedge or m15_recovery_valid
            else list(CAPITAL_ALLOCATION_SIZE_MULTIPLES)
        )
        if multiple <= numeric_max_multiple + 1e-12
    ]
    allowed_size_multiples = [
        multiple
        for multiple in candidate_size_multiples
        if base_units * CAPITAL_ALLOCATION_SIZE_RATIOS[multiple][0]
        // CAPITAL_ALLOCATION_SIZE_RATIOS[multiple][1]
        > 0
        and allocation_context_valid
    ]
    explicit_negative_edge = (
        capture_status == "NEGATIVE_EXPECTANCY"
        or (overall_expectancy is not None and overall_expectancy < 0.0)
    )
    range_nonmarket = bool(
        str(metadata.get("forecast_direction") or "").strip().upper()
        == "RANGE"
        and _normalized_order_vehicle(intent.get("order_type"))
        in {"LIMIT", "STOP"}
    )
    if m15_recovery_claimed:
        positive_edge_proven = bool(
            m15_recovery_valid
            and method_scope_consistent
            and positive_tp_edge
            and m15_recovery["mode"] == M15_RECOVERY_TP_PROVEN_MODE
        )
        edge_collection_proven = bool(
            m15_recovery_valid
            and method_scope_consistent
            and positive_tp_edge
            and m15_recovery["mode"]
            == M15_RECOVERY_PROOF_COLLECTION_MODE
        )
    else:
        positive_edge_proven = bool(
            technical_context_valid
            and method_scope_consistent
            and (
                predictive_scout
                or (
                    positive_tp_edge
                    and (
                        not range_nonmarket
                        or isinstance(range_economic_basis, Mapping)
                    )
                )
                or (
                    exact_vehicle_net_edge["proven"]
                    and not range_nonmarket
                )
                or hedge
            )
        )
        edge_collection_proven = False
    live_blocker_codes = [
        str(code)
        for code in result.get("live_blocker_codes", []) or []
        if str(code)
    ]
    if m15_recovery_claimed:
        live_blocker_codes.extend(m15_recovery["error_codes"])
    else:
        if not technical_context_envelope_valid:
            live_blocker_codes.append(
                "FORECAST_TECHNICAL_CONTEXT_UNKNOWN_FOR_ALLOCATION"
            )
        if not regime_family_binding["current_context_bound"]:
            live_blocker_codes.append(
                "FORECAST_TECHNICAL_CONTEXT_NOT_CURRENT_FOR_ALLOCATION"
            )
        if not regime_family_binding["canonical_policy_sources_bound"]:
            live_blocker_codes.append(
                "DYNAMIC_TF_POLICY_CANONICAL_SOURCE_MISMATCH"
            )
        if not regime_family_binding["canonical_forecast_context_bound"]:
            live_blocker_codes.append(
                "FORECAST_TECHNICAL_CONTEXT_CANONICAL_REBUILD_MISMATCH"
            )
        if not forecast_binding["direction_side_aligned"]:
            live_blocker_codes.append("FORECAST_DIRECTION_SIDE_MISMATCH")
        if not forecast_binding["calibration_identity_valid"]:
            live_blocker_codes.append("FORECAST_CALIBRATION_IDENTITY_MISMATCH")
        if not regime_family_binding["receipt_valid"]:
            live_blocker_codes.append(
                "REGIME_FAMILY_WEIGHTING_MISSING_OR_INVALID"
            )
        if not regime_family_binding["sha_bound"]:
            live_blocker_codes.append("REGIME_FAMILY_WEIGHTING_SHA_UNBOUND")
        if (
            not regime_family_binding["selected_method_bound"]
            or not regime_family_binding["executable_method_bound"]
        ):
            live_blocker_codes.append("REGIME_FAMILY_WEIGHTING_METHOD_MISMATCH")
        if not regime_family_binding["family_direction_bound"]:
            live_blocker_codes.append(
                "REGIME_FAMILY_WEIGHTING_DIRECTION_UNBOUND"
            )
        if not regime_family_binding["direction_consistent"]:
            live_blocker_codes.append(
                "REGIME_FAMILY_WEIGHTING_FORECAST_CONTRADICTION"
            )
        if not regime_family_binding["actionable_direction_bound"]:
            live_blocker_codes.append(
                "REGIME_FAMILY_WEIGHTING_DIRECTION_NOT_ACTIONABLE"
            )
    live_blocker_codes = list(dict.fromkeys(live_blocker_codes))
    allocation_eligible = (
        str(result.get("status") or "").strip().upper() == "LIVE_READY"
        and result.get("risk_allowed") is True
        and not live_blocker_codes
        and method_scope_consistent
        and base_units > 0
        and (positive_edge_proven or edge_collection_proven)
        and bool(allowed_size_multiples)
    )
    standard_edge_basis = (
        "FORECAST_TECHNICAL_CONTEXT_UNKNOWN"
        if not technical_context_envelope_valid
        else "FORECAST_TECHNICAL_CONTEXT_NOT_CURRENT"
        if not regime_family_binding["current_context_bound"]
        else "DYNAMIC_TF_POLICY_CANONICAL_SOURCE_MISMATCH"
        if not regime_family_binding["canonical_policy_sources_bound"]
        else "FORECAST_TECHNICAL_CONTEXT_CANONICAL_REBUILD_MISMATCH"
        if not regime_family_binding["canonical_forecast_context_bound"]
        else "FORECAST_DIRECTION_SIDE_MISMATCH"
        if not forecast_binding["direction_side_aligned"]
        else "FORECAST_CALIBRATION_IDENTITY_MISMATCH"
        if not forecast_binding["calibration_identity_valid"]
        else "REGIME_FAMILY_WEIGHTING_INVALID"
        if not regime_family_binding["receipt_valid"]
        else "REGIME_FAMILY_WEIGHTING_SHA_UNBOUND"
        if not regime_family_binding["sha_bound"]
        else "REGIME_FAMILY_WEIGHTING_METHOD_MISMATCH"
        if not regime_family_binding["executable_method_bound"]
        or not regime_family_binding["selected_method_bound"]
        else "REGIME_FAMILY_WEIGHTING_DIRECTION_UNBOUND"
        if not regime_family_binding["family_direction_bound"]
        else "REGIME_FAMILY_WEIGHTING_FORECAST_CONTRADICTION"
        if not regime_family_binding["direction_consistent"]
        else "REGIME_FAMILY_WEIGHTING_DIRECTION_NOT_ACTIONABLE"
        if not regime_family_binding["actionable_direction_bound"]
        else "METHOD_SCOPE_MISMATCH"
        if not method_scope_consistent
        else "PREDICTIVE_SCOUT_FORWARD_EVIDENCE"
        if predictive_scout
        else "HEDGE_RISK_REDUCTION"
        if hedge
        else "EXACT_VEHICLE_TAKE_PROFIT"
        if positive_tp_edge and range_nonmarket
        else "EXACT_VEHICLE_ALL_EXIT_NET"
        if exact_vehicle_net_edge["proven"]
        else "EXACT_VEHICLE_TAKE_PROFIT"
        if positive_tp_edge
        else "EXACT_VEHICLE_ALL_EXIT_CONTRADICTS_TP"
        if exact_vehicle_tp_edge
        and exact_vehicle_net_edge["blocks_tp_exception"] is True
        else "INVALID_PREDICTIVE_SCOUT_CLAIM"
        if predictive_scout_claimed
        else "UNKNOWN_OR_NON_EXACT_EDGE"
        if not explicit_negative_edge
        else "NO_POSITIVE_EDGE_PROOF"
    )
    edge_basis = (
        m15_recovery["edge_basis"]
        if m15_recovery_valid
        else "M15_RECOVERY_ALLOCATION_INVALID"
        if m15_recovery_claimed
        else standard_edge_basis
    )
    return {
        "lane_id": str(result.get("lane_id") or ""),
        "intent_result_sha256": canonical_json_sha256(dict(result)),
        "status": result.get("status"),
        "risk_allowed": result.get("risk_allowed") is True,
        "live_blocker_codes": live_blocker_codes,
        "allocation_eligible": allocation_eligible,
        "positive_edge_proven": positive_edge_proven,
        "edge_collection_proven": edge_collection_proven,
        "edge_basis": edge_basis,
        "pair": intent.get("pair"),
        "side": intent.get("side"),
        "method": method,
        "metadata_method": metadata_method or None,
        "method_scope_consistent": method_scope_consistent,
        "order_type": intent.get("order_type"),
        "base_units": base_units,
        "entry": _optional_float(intent.get("entry")),
        "tp": _optional_float(intent.get("tp")),
        "sl": _optional_float(intent.get("sl")),
        "allowed_size_multiples": allowed_size_multiples,
        "numeric_ceiling": numeric_ceiling,
        "m15_recovery": m15_recovery,
        "predictive_scout": predictive_scout,
        "predictive_scout_claimed": predictive_scout_claimed,
        "hedge": hedge,
        "risk": {
            "entry_price": _optional_float(risk_metrics.get("entry_price")),
            "loss_pips": _optional_float(risk_metrics.get("loss_pips")),
            "reward_pips": _optional_float(risk_metrics.get("reward_pips")),
            "risk_jpy": _optional_float(risk_metrics.get("risk_jpy")),
            "reward_jpy": _optional_float(risk_metrics.get("reward_jpy")),
            "max_loss_jpy": _optional_float(metadata.get("max_loss_jpy")),
            "reward_risk": _optional_float(risk_metrics.get("reward_risk")),
            "spread_pips": _optional_float(risk_metrics.get("spread_pips")),
            "estimated_margin_jpy": _optional_float(
                risk_metrics.get("estimated_margin_jpy")
                if risk_metrics.get("estimated_margin_jpy") is not None
                else metadata.get("estimated_margin_jpy")
            ),
        },
        "forecast": {
            "direction": metadata.get("forecast_direction"),
            "confidence": _optional_float(metadata.get("forecast_confidence")),
            "confidence_semantics": CONFIDENCE_SEMANTICS,
            "raw_confidence": _optional_float(metadata.get("forecast_raw_confidence")),
            "calibration_name": metadata.get("forecast_directional_calibration_name"),
            "calibration_multiplier": _optional_float(
                metadata.get("forecast_calibration_multiplier")
            ),
            "economic_hit_rate": _optional_float(
                metadata.get("forecast_directional_economic_hit_rate")
            ),
            "economic_samples": _optional_int(
                metadata.get("forecast_directional_economic_samples")
            ),
            "hit_rate": _optional_float(metadata.get("forecast_directional_hit_rate")),
            "samples": _optional_int(metadata.get("forecast_directional_samples")),
            "timeout_rate": _optional_float(
                metadata.get("forecast_directional_timeout_rate")
            ),
            "current_price": _optional_float(metadata.get("forecast_current_price")),
            "target_price": _optional_float(metadata.get("forecast_target_price")),
            "invalidation_price": _optional_float(
                metadata.get("forecast_invalidation_price")
            ),
            "range_low_price": _optional_float(
                metadata.get("forecast_range_low_price")
            ),
            "range_high_price": _optional_float(
                metadata.get("forecast_range_high_price")
            ),
            "range_trade_economic_basis": (
                dict(range_economic_basis)
                if isinstance(range_economic_basis, Mapping)
                else None
            ),
            "range_tp_proven_breakout_failure": (
                range_tp_proven_breakout_failure
            ),
            "technical_context": technical_context,
            "regime_family_weighting_sha256": regime_family_binding[
                "receipt_sha256"
            ],
            "regime_family_selected_method": regime_family_binding[
                "selected_method"
            ],
            "regime_family_direction": regime_family_binding[
                "family_direction"
            ],
            "regime_family_binding": regime_family_binding,
        },
        "capture": {
            "status": metadata.get("capture_economics_status"),
            "overall_expectancy_jpy": overall_expectancy,
            "take_profit_expectancy_jpy": tp_expectancy,
            "take_profit_trades": tp_trades,
            "take_profit_wins": _optional_int(metadata.get("capture_take_profit_wins")),
            "take_profit_losses": tp_losses,
            "exact_vehicle_edge_proven": exact_vehicle_tp_edge,
            "minimum_exact_vehicle_trades": CAPITAL_ALLOCATION_MIN_EXACT_TP_TRADES,
            "exact_vehicle_all_exit": exact_vehicle_net_edge,
            "execution_ledger_surface_sha256": execution_ledger_surface_sha256,
            "minimum_exact_vehicle_all_exit_trades": (
                CAPITAL_ALLOCATION_MIN_EXACT_NET_TRADES
            ),
            "market_close_expectancy_jpy": _optional_float(
                metadata.get("capture_market_close_expectancy_jpy")
            ),
            "avg_win_jpy": _optional_float(metadata.get("capture_avg_win_jpy")),
            "avg_loss_jpy": _optional_float(metadata.get("capture_avg_loss_jpy")),
        },
    }


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed


def _source_descriptor(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {
            "path": str(path),
            "exists": False,
            "sha256": None,
            "size_bytes": None,
            "generated_at_utc": None,
        }
    raw = path.read_bytes()
    generated_at = None
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = None
        if isinstance(payload, dict):
            generated_at = payload.get("generated_at_utc") or payload.get("fetched_at_utc")
    return {
        "path": str(path),
        "exists": True,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "generated_at_utc": generated_at,
    }


def _source_descriptor_without_hash(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError:
        return {
            "path": str(path),
            "exists": False,
            "sha256": None,
            "size_bytes": None,
            "generated_at_utc": None,
        }
    return {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": None,
        "size_bytes": int(stat.st_size) if path.is_file() else None,
        "generated_at_utc": None,
    }


def _dynamic_tf_packet_source_descriptors(
    sources: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze the packet-authoritative files used by dynamic TF policy."""

    result: dict[str, Any] = {}
    for name in ("calendar", "strategy_profile"):
        source = sources.get(name)
        if not isinstance(source, Mapping):
            result[name] = None
            continue
        result[name] = {
            "path": source.get("path"),
            "exists": source.get("exists"),
            "sha256": source.get("sha256"),
            "size_bytes": source.get("size_bytes"),
        }
    return result


def _normalize_source_path(path: Path | str) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _exact_guardian_p1_margin_warning(
    *,
    selected_event: Mapping[str, Any],
    event_mirror: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> bool:
    """Bind the shared source-event contract to the mirrored receipt shape."""

    if (
        str(selected_event.get("pair") or "").strip().upper()
        != "PORTFOLIO"
        or str(event_mirror.get("pair") or "").strip().upper()
        != "PORTFOLIO"
        or str(receipt.get("pair") or "").strip().upper()
        != "PORTFOLIO"
        or str(selected_event.get("thesis_state") or "").strip().upper()
        != "WOUNDED"
        or str(event_mirror.get("thesis_state") or "").strip().upper()
        != "WOUNDED"
        or str(receipt.get("thesis_state") or "").strip().upper()
        != "WOUNDED"
    ):
        return False
    selected_details = selected_event.get("details")
    mirror_details = event_mirror.get("details")
    if not isinstance(selected_details, Mapping) or not isinstance(
        mirror_details,
        Mapping,
    ):
        return False
    if dict(selected_details) != dict(mirror_details):
        return False
    expected_cap = RiskPolicy().max_margin_utilization_pct
    selected_valid = exact_p1_margin_warning_source_event(
        event_type=selected_event.get("event_type"),
        event_severity=selected_event.get("severity"),
        action=selected_event.get("action_hint"),
        details=selected_details,
        expected_max_margin_utilization_pct=expected_cap,
    )
    mirror_valid = exact_p1_margin_warning_source_event(
        event_type=event_mirror.get("event_type"),
        event_severity=event_mirror.get("severity"),
        action=event_mirror.get("action_hint"),
        details=mirror_details,
        expected_max_margin_utilization_pct=expected_cap,
    )
    receipt_action_valid = str(receipt.get("action") or "").strip().upper() in {
        "HOLD",
        "NO_ACTION",
    }
    return selected_valid and mirror_valid and receipt_action_valid


def guardian_action_receipt_scope_material(
    path: Path | None,
    *,
    baseline: Mapping[str, Any] | None = None,
    baseline_pairs: Iterable[str] | None = None,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    """Bind only the Guardian receipt meaning that can affect this read.

    Guardian publishes one canonical receipt for the whole portfolio.  Routine
    HOLD/NO_ACTION rotation for another pair must not discard an in-flight read,
    while the selected pair, portfolio-risk actions, and broken execution
    boundaries remain fail-closed material.
    """

    if baseline is not None and baseline_pairs is not None:
        raise ValueError("baseline and baseline_pairs are mutually exclusive")
    if baseline is None and baseline_pairs is None:
        raise ValueError("baseline or baseline_pairs is required")
    if baseline_pairs is None:
        scoped_pairs = _guardian_receipt_baseline_pairs(baseline or {})
    else:
        scoped_pairs = tuple(
            sorted(
                {
                    str(pair).strip().upper()
                    for pair in baseline_pairs
                    if isinstance(pair, str)
                    and str(pair).strip().upper() in DEFAULT_TRADER_PAIRS
                }
            )
        )
    common = {
        "baseline_pairs": list(scoped_pairs),
    }
    if path is None or not path.exists() or not path.is_file():
        return {"parse_status": "MISSING", **common}
    try:
        size_bytes = int(path.stat().st_size)
    except OSError:
        return {"parse_status": "UNREADABLE", **common}
    if size_bytes > MAX_GUARDIAN_ACTION_RECEIPT_BYTES:
        return {
            "parse_status": "OVERSIZED",
            "size_bytes": size_bytes,
            "max_size_bytes": MAX_GUARDIAN_ACTION_RECEIPT_BYTES,
            **common,
        }
    raw_bytes: bytes | None = None
    try:
        with path.open("rb") as handle:
            raw_bytes = handle.read(MAX_GUARDIAN_ACTION_RECEIPT_BYTES + 1)
        if len(raw_bytes) > MAX_GUARDIAN_ACTION_RECEIPT_BYTES:
            return {
                "parse_status": "OVERSIZED",
                "size_bytes": max(size_bytes, len(raw_bytes)),
                "max_size_bytes": MAX_GUARDIAN_ACTION_RECEIPT_BYTES,
                **common,
            }
        raw = json.loads(
            raw_bytes.decode("utf-8"),
            parse_constant=_reject_nonfinite_json_constant,
            parse_float=_parse_finite_guardian_json_float,
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
    ):
        raw_sha = (
            hashlib.sha256(raw_bytes).hexdigest()
            if raw_bytes is not None
            else None
        )
        return {
            "parse_status": "INVALID",
            "raw_sha256": raw_sha,
            **common,
        }
    if not isinstance(raw, Mapping):
        return {
            "parse_status": "INVALID",
            "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            **common,
        }

    # Strict decoding alone does not reject escaped lone surrogates, and the
    # stdlib may otherwise materialize an overflow exponent as a non-finite
    # float.  Prove the complete bounded object is canonical-JSON encodable
    # before selecting a semantic subset from it.
    try:
        canonical_json_sha256(raw)
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError):
        return {
            "parse_status": "INVALID",
            "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            **common,
        }

    event_mirror = raw.get("event")
    event_mirror = event_mirror if isinstance(event_mirror, Mapping) else {}
    raw_selected_event = raw.get("selected_event")
    selected_event = (
        raw_selected_event if isinstance(raw_selected_event, Mapping) else {}
    )
    receipt = raw.get("receipt")
    receipt = receipt if isinstance(receipt, Mapping) else {}
    event_pair = str(
        selected_event.get("pair") or receipt.get("pair") or ""
    ).strip().upper()
    event_type = str(selected_event.get("event_type") or "").strip().upper()
    severity = str(selected_event.get("severity") or "").strip().upper()
    action_hint = str(selected_event.get("action_hint") or "").strip().upper()
    action = str(receipt.get("action") or "").strip().upper()
    p1_margin_warning_candidate = bool(
        str(raw.get("receipt_status") or raw.get("status") or "")
        .strip()
        .upper()
        == "ACCEPTED"
        and str(raw.get("receipt_lifecycle") or "").strip().upper()
        == "ACTIVE"
        and raw.get("consumed_by_trader") is False
        and _exact_guardian_p1_margin_warning(
            selected_event=selected_event,
            event_mirror=event_mirror,
            receipt=receipt,
        )
    )
    global_reasons: list[str] = []
    producer_source = str(raw.get("source") or "").strip()
    producer_model = str(raw.get("model") or "").strip()
    if producer_source != "guardian_wake_dispatcher":
        global_reasons.append(
            f"PRODUCER_SOURCE_INVALID:{producer_source or 'MISSING'}"
        )
    if producer_model != "gpt-5.5":
        global_reasons.append(
            f"PRODUCER_MODEL_INVALID:{producer_model or 'MISSING'}"
        )
    if not event_mirror:
        global_reasons.append("EVENT_MIRROR_MISSING")
    if "selected_event" not in raw:
        global_reasons.append("SELECTED_EVENT_MISSING")
    elif not isinstance(raw_selected_event, Mapping):
        global_reasons.append("SELECTED_EVENT_INVALID")
    if not receipt:
        global_reasons.append("RECEIPT_BODY_MISSING")
    for field in (
        "action",
        "event_id",
        "new_information",
        "pair",
        "side",
        "thesis_state",
        "reason",
        "harvest_trigger",
        "margin_state",
        "gateway_required",
        "no_direct_oanda",
    ):
        if field not in receipt:
            global_reasons.append(f"RECEIPT_FIELD_MISSING:{field}")
    if (
        "invalidation" not in receipt
        and "invalidation_evidence" not in receipt
    ):
        global_reasons.append("RECEIPT_FIELD_MISSING:invalidation")
    if (
        "ownership" not in receipt
        and "manual_system_ownership" not in receipt
    ):
        global_reasons.append("RECEIPT_FIELD_MISSING:ownership")
    if "new_information" in receipt and not isinstance(
        receipt.get("new_information"), bool
    ):
        global_reasons.append("RECEIPT_FIELD_INVALID:new_information")
    receipt_side = str(receipt.get("side") or "").strip().upper()
    if receipt_side not in {"LONG", "SHORT", "NONE", "N/A"}:
        global_reasons.append(
            f"RECEIPT_FIELD_INVALID:side={receipt_side or 'MISSING'}"
        )
    receipt_ownership = str(
        receipt.get("ownership")
        or receipt.get("manual_system_ownership")
        or ""
    ).strip().upper()
    if receipt_ownership not in {"SYSTEM", "OPERATOR_MANUAL", "UNKNOWN"}:
        global_reasons.append(
            "RECEIPT_FIELD_INVALID:ownership="
            f"{receipt_ownership or 'MISSING'}"
        )
    for field, expected in (
        ("status", "ACCEPTED"),
        ("receipt_status", "ACCEPTED"),
        ("dispatcher_status", "RECEIPT_WRITTEN"),
    ):
        actual = str(raw.get(field) or "").strip().upper()
        if actual != expected:
            global_reasons.append(f"RECEIPT_STATE_INVALID:{field}={actual or 'MISSING'}")
    lifecycle = str(raw.get("receipt_lifecycle") or "").strip().upper()
    if lifecycle not in {"ACTIVE", "CONSUMED"}:
        global_reasons.append(
            f"RECEIPT_STATE_INVALID:receipt_lifecycle={lifecycle or 'MISSING'}"
        )
    consumed_by_trader = raw.get("consumed_by_trader")
    if (
        not isinstance(consumed_by_trader, bool)
        or (lifecycle == "ACTIVE" and consumed_by_trader is not False)
        or (lifecycle == "CONSUMED" and consumed_by_trader is not True)
    ):
        global_reasons.append("RECEIPT_STATE_INVALID:consumed_by_trader")

    # Read and validate the complete bounded receipt before sampling the
    # production clock.  A receipt rotating during the slower packet build is
    # therefore judged against the instant immediately after these exact bytes
    # became the semantic input, never against the earlier build-start clock.
    current_time = _utc_now(as_of)
    generated_at = _strict_guardian_utc(raw.get("generated_at_utc"))
    expires_at = _strict_guardian_utc(raw.get("expires_at_utc"))
    if raw.get("generated_at_utc") in (None, ""):
        generated_state = "MISSING"
    elif generated_at is None:
        generated_state = "MALFORMED"
    elif generated_at > current_time + timedelta(seconds=5):
        generated_state = "FUTURE"
    else:
        generated_state = "VALID"
    if raw.get("expires_at_utc") in (None, ""):
        expiry_state = "MISSING"
    elif expires_at is None:
        expiry_state = "MALFORMED"
    elif generated_at is not None and expires_at <= generated_at:
        expiry_state = "ORDER_INVALID"
    elif expires_at > current_time:
        expiry_state = "FRESH"
    else:
        expiry_state = "EXPIRED"
    if lifecycle == "CONSUMED" and expiry_state in {"FRESH", "EXPIRED"}:
        expiry_state = "NOT_APPLICABLE_CONSUMED"
    time_state = {
        "generated_at_utc": generated_state,
        "expires_at_utc": expiry_state,
    }
    if generated_state != "VALID":
        global_reasons.append(f"GENERATED_AT_{generated_state}")
    if expiry_state in {"MISSING", "MALFORMED", "ORDER_INVALID"}:
        global_reasons.append(f"EXPIRES_AT_{expiry_state}")
    elif lifecycle == "ACTIVE" and expiry_state != "FRESH":
        global_reasons.append(f"ACTIVE_RECEIPT_{expiry_state}")
    if event_type not in GUARDIAN_KNOWN_EVENT_TYPES:
        global_reasons.append(f"EVENT_TYPE_UNKNOWN:{event_type or 'MISSING'}")
    if severity not in {"P0", "P1", "P2"}:
        global_reasons.append(f"SEVERITY_UNKNOWN:{severity or 'MISSING'}")
    if action_hint not in {
        "TRADE",
        "ADD",
        "HOLD",
        "HARVEST",
        "REDUCE",
        "CANCEL_PENDING",
        "NO_ACTION",
    }:
        global_reasons.append(f"ACTION_HINT_UNKNOWN:{action_hint or 'MISSING'}")
    if action not in {
        "TRADE",
        "ADD",
        "HOLD",
        "HARVEST",
        "REDUCE",
        "CANCEL_PENDING",
        "NO_ACTION",
    }:
        global_reasons.append(f"RECEIPT_ACTION_UNKNOWN:{action or 'MISSING'}")
    if event_type in GUARDIAN_TECHNICAL_OBSERVATION_EVENT_TYPES:
        if action not in {"HOLD", "NO_ACTION"}:
            global_reasons.append(
                f"TECHNICAL_OBSERVATION_ACTION_FORBIDDEN:{action or 'MISSING'}"
            )
        if action_hint not in {"HOLD", "NO_ACTION"}:
            global_reasons.append(
                "TECHNICAL_OBSERVATION_ACTION_HINT_FORBIDDEN:"
                f"{action_hint or 'MISSING'}"
            )
    selected_direction_raw = str(
        selected_event.get("direction") or ""
    ).strip().upper()
    mirror_direction_raw = str(
        event_mirror.get("direction") or ""
    ).strip().upper()
    selected_direction = _guardian_direction(selected_direction_raw)
    mirror_direction = _guardian_direction(mirror_direction_raw)
    receipt_direction = _guardian_direction(
        receipt.get("side") or receipt.get("direction")
    )
    if selected_direction != mirror_direction:
        global_reasons.append("EVENT_MIRROR_MISMATCH:direction")
    if selected_direction_raw and selected_direction is None:
        global_reasons.append(
            f"EVENT_DIRECTION_INVALID:selected_event={selected_direction_raw}"
        )
    if mirror_direction_raw and mirror_direction is None:
        global_reasons.append(
            f"EVENT_DIRECTION_INVALID:event={mirror_direction_raw}"
        )
    if selected_direction is not None:
        if receipt_direction != selected_direction:
            global_reasons.append("RECEIPT_IDENTITY_BROKEN:direction")
    elif receipt_side not in {"NONE", "N/A"}:
        global_reasons.append("RECEIPT_IDENTITY_BROKEN:direction")
    if action in {"TRADE", "ADD"}:
        if action_hint not in {"TRADE", "ADD"}:
            global_reasons.append("ENTRY_ACTION_HINT_CONTRACT_BROKEN")
        if selected_direction_raw not in {"LONG", "SHORT"}:
            global_reasons.append(
                "ENTRY_DIRECTION_REQUIRED:selected_event"
            )
        if mirror_direction_raw not in {"LONG", "SHORT"}:
            global_reasons.append("ENTRY_DIRECTION_REQUIRED:event")
        if receipt_side not in {"LONG", "SHORT"}:
            global_reasons.append("ENTRY_DIRECTION_REQUIRED:receipt")
        if not (
            selected_direction_raw
            == mirror_direction_raw
            == receipt_side
            and receipt_side in {"LONG", "SHORT"}
        ):
            global_reasons.append("ENTRY_DIRECTION_CONTRACT_BROKEN")
        if receipt.get("new_information") is not True:
            global_reasons.append("ENTRY_NEW_INFORMATION_CONTRACT_BROKEN")
    identity_fields = (
        "event_id",
        "dedupe_key",
        "pair",
        "event_type",
        "severity",
        "action_hint",
    )
    for field in identity_fields:
        selected_value = selected_event.get(field)
        mirror_value = event_mirror.get(field)
        selected_text = str(selected_value or "").strip().upper()
        mirror_text = str(mirror_value or "").strip().upper()
        if not selected_text or not mirror_text:
            global_reasons.append(f"EVENT_IDENTITY_MISSING:{field}")
        elif selected_text != mirror_text:
            global_reasons.append(f"EVENT_MIRROR_MISMATCH:{field}")
    selected_event_id = str(selected_event.get("event_id") or "").strip()
    selected_dedupe_key = str(selected_event.get("dedupe_key") or "").strip()
    for field, expected in (
        ("selected_event_id", selected_event_id),
        ("selected_event_dedupe_key", selected_dedupe_key),
    ):
        claimed = str(raw.get(field) or "").strip()
        if expected and claimed != expected:
            global_reasons.append(f"IDENTITY_CONTRACT_BROKEN:{field}")
    for field, expected in (
        ("event_id", selected_event_id),
        ("dedupe_key", selected_dedupe_key),
    ):
        claimed = str(receipt.get(field) or "").strip()
        if expected and claimed != expected:
            global_reasons.append(f"RECEIPT_IDENTITY_BROKEN:{field}")
    receipt_pair = str(receipt.get("pair") or "").strip().upper()
    if event_pair and receipt_pair != event_pair:
        global_reasons.append("RECEIPT_IDENTITY_BROKEN:pair")
    if receipt.get("gateway_required") is not True:
        global_reasons.append("RECEIPT_GATEWAY_REQUIRED_CONTRACT_BROKEN")
    if receipt.get("no_direct_oanda") is not True:
        global_reasons.append("RECEIPT_NO_DIRECT_OANDA_CONTRACT_BROKEN")
    if (
        event_pair not in DEFAULT_TRADER_PAIRS
        and not p1_margin_warning_candidate
    ):
        global_reasons.append("PAIR_SCOPE_UNKNOWN")
    if severity == "P0":
        global_reasons.append("P0_EVENT")
    if (
        event_type in GUARDIAN_GLOBAL_SAFETY_EVENT_TYPES
        and not p1_margin_warning_candidate
    ):
        global_reasons.append(f"GLOBAL_EVENT:{event_type}")
    if action in GUARDIAN_HIGH_URGENCY_ACTIONS:
        global_reasons.append(f"HIGH_URGENCY_ACTION:{action}")
    if action_hint in GUARDIAN_HIGH_URGENCY_ACTIONS:
        global_reasons.append(f"HIGH_URGENCY_HINT:{action_hint}")
    for source_name, source in (
        ("selected_event", selected_event),
        ("event", event_mirror),
        ("receipt", receipt),
    ):
        source_thesis_state = str(
            source.get("thesis_state") or ""
        ).strip().upper()
        if source_thesis_state not in {
            "ALIVE",
            "WOUNDED",
            "INVALIDATED",
            "EMERGENCY",
            "UNKNOWN",
        }:
            global_reasons.append(
                f"THESIS_STATE_INVALID:{source_name}="
                f"{source_thesis_state or 'MISSING'}"
            )
        if source_thesis_state in {"INVALIDATED", "EMERGENCY"}:
            global_reasons.append(
                f"THESIS_STATE:{source_name}={source_thesis_state}"
            )
    selected_thesis_state = str(
        selected_event.get("thesis_state") or ""
    ).strip().upper()
    mirror_thesis_state = str(
        event_mirror.get("thesis_state") or ""
    ).strip().upper()
    if selected_thesis_state != mirror_thesis_state:
        global_reasons.append("EVENT_MIRROR_MISMATCH:thesis_state")
    if raw.get("gateway_required") is not True:
        global_reasons.append("GATEWAY_REQUIRED_CONTRACT_BROKEN")
    if raw.get("no_direct_oanda") is not True:
        global_reasons.append("NO_DIRECT_OANDA_CONTRACT_BROKEN")
    details = selected_event.get("details")
    details = details if isinstance(details, Mapping) else {}
    for field, expected in (
        ("live_permission_allowed", False),
        ("no_direct_oanda", True),
        ("preserve_blockers", True),
    ):
        if field in details and details.get(field) is not expected:
            global_reasons.append(f"EVENT_BOUNDARY_BROKEN:{field}")
    tuning_review = receipt.get("bot_tuning_review")
    tuning_review = tuning_review if isinstance(tuning_review, Mapping) else {}
    for field, expected in (
        ("live_permission_allowed", False),
        ("no_direct_oanda", True),
        ("preserve_blockers", True),
    ):
        if field in tuning_review and tuning_review.get(field) is not expected:
            global_reasons.append(f"TUNING_BOUNDARY_BROKEN:{field}")
    for source_name, source in (("root", raw), ("receipt", receipt)):
        if source.get("normal_routing_allowed") is False:
            global_reasons.append(f"ROUTING_PROHIBITED:{source_name}")
    raw_issues = raw.get("issues")
    if not isinstance(raw_issues, list):
        global_reasons.append("ISSUES_CONTRACT_INVALID")
        raw_issues = []
    if raw_issues:
        global_reasons.append("RECEIPT_ISSUES_PRESENT")
    for issue in raw_issues:
        if not isinstance(issue, Mapping):
            continue
        if issue.get("normal_routing_allowed") is False:
            global_reasons.append("ROUTING_PROHIBITED:issue")
        if str(issue.get("severity") or "").strip().upper() == "P0":
            global_reasons.append("P0_RECEIPT_ISSUE")
    execution_boundary = raw.get("execution_boundary")
    execution_boundary = (
        execution_boundary if isinstance(execution_boundary, Mapping) else {}
    )
    for field in (
        "gpt_wake_never_calls_oanda_directly",
        "guardian_never_trades",
        "only_live_order_gateway_may_send_cancel_close",
    ):
        if execution_boundary.get(field) is not True:
            global_reasons.append(f"EXECUTION_BOUNDARY_BROKEN:{field}")

    global_reasons = sorted(set(global_reasons))
    relevant_pair = event_pair in scoped_pairs or p1_margin_warning_candidate
    routine_other_pair = (
        not relevant_pair
        and not global_reasons
        and action in {"HOLD", "NO_ACTION"}
    )
    if routine_other_pair:
        return {
            "parse_status": "VALID",
            "scope": "IRRELEVANT_PAIR_ROUTINE",
            "global_safety": False,
            **common,
        }
    if not relevant_pair and not global_reasons:
        global_reasons.append(f"OTHER_PAIR_NON_ROUTINE_ACTION:{action or 'UNKNOWN'}")

    try:
        selected_event_material = _guardian_receipt_semantic_copy(
            selected_event
        )
        event_mirror_material = _guardian_receipt_semantic_copy(event_mirror)
        receipt_material = _guardian_receipt_semantic_copy(receipt)
        execution_boundary_material = _guardian_receipt_semantic_copy(
            execution_boundary
        )
        issues_material = _guardian_receipt_semantic_copy(raw_issues)
    except RecursionError:
        return {
            "parse_status": "INVALID",
            "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "reason": "SEMANTIC_DEPTH_EXCEEDED",
            **common,
        }

    return {
        "parse_status": "VALID",
        "scope": (
            "GLOBAL_SAFETY"
            if global_reasons
            else "GLOBAL_MARGIN_OBSERVATION"
            if p1_margin_warning_candidate
            else "SELECTED_PAIR"
        ),
        "global_safety": bool(global_reasons),
        "global_reasons": global_reasons,
        "p1_margin_warning_observed": bool(
            p1_margin_warning_candidate and not global_reasons
        ),
        "margin_contract": (
            GUARDIAN_P1_MARGIN_WARNING_CONTRACT
            if p1_margin_warning_candidate and not global_reasons
            else None
        ),
        "event_pair_relevant": relevant_pair,
        "time_state": time_state,
        **common,
        "receipt_state": _mapping_fields(
            raw,
            (
                "status",
                "source",
                "model",
                "receipt_status",
                "receipt_lifecycle",
                "consumed_by_trader",
                "dispatcher_status",
                "gateway_required",
                "no_direct_oanda",
                "selected_event_id",
                "selected_event_dedupe_key",
                "superseded_by_event_id",
            ),
        ),
        "selected_event": selected_event_material,
        "event": event_mirror_material,
        "receipt": receipt_material,
        "execution_boundary": execution_boundary_material,
        "issues": issues_material,
    }


def _guardian_receipt_baseline_pairs(
    baseline: Mapping[str, Any],
) -> tuple[str, ...]:
    pairs: set[str] = set()
    lane_ids: list[Any] = []
    selected_lane_id = baseline.get("selected_lane_id")
    if selected_lane_id is not None:
        lane_ids.append(selected_lane_id)
    selected_lane_ids = baseline.get("selected_lane_ids")
    if isinstance(selected_lane_ids, list):
        lane_ids.extend(selected_lane_ids)
    for lane_id in lane_ids:
        if not isinstance(lane_id, str):
            continue
        pairs.update(
            token.strip().upper()
            for token in lane_id.split(":")
            if token.strip().upper() in DEFAULT_TRADER_PAIRS
        )

    market_read = baseline.get("market_read_first")
    market_read = market_read if isinstance(market_read, Mapping) else {}
    naked = market_read.get("naked_read")
    if isinstance(naked, Mapping):
        pair = str(naked.get("cleanest_pair_expression") or "").strip().upper()
        if pair in DEFAULT_TRADER_PAIRS:
            pairs.add(pair)
    for section_name in (
        "next_30m_prediction",
        "next_2h_prediction",
        "best_trade_if_forced",
    ):
        section = market_read.get(section_name)
        if not isinstance(section, Mapping):
            continue
        pair = str(section.get("pair") or "").strip().upper()
        if pair in DEFAULT_TRADER_PAIRS:
            pairs.add(pair)
    return tuple(sorted(pairs))


def _reject_nonfinite_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _parse_finite_guardian_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON float is forbidden: {value}")
    return parsed


def _strict_guardian_utc(value: Any) -> datetime | None:
    """Parse an explicit timezone-aware UTC producer timestamp."""

    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(None):
        return None
    return parsed.astimezone(timezone.utc)


def _guardian_direction(value: Any) -> str | None:
    normalized = str(value or "").strip().upper()
    if normalized == "LONG":
        return "LONG"
    if normalized == "SHORT":
        return "SHORT"
    return None


def _guardian_receipt_semantic_copy(value: Any) -> Any:
    """Remove publication clocks while preserving technical/risk semantics."""

    volatile_fields = {
        "detected_at_utc",
        "expires_at_utc",
        "generated_at_utc",
        "receipt_id",
        "superseded_previous_receipt",
    }
    if isinstance(value, Mapping):
        return {
            str(key): _guardian_receipt_semantic_copy(field_value)
            for key, field_value in value.items()
            if str(key) not in volatile_fields
        }
    if isinstance(value, list):
        return [_guardian_receipt_semantic_copy(item) for item in value]
    return deepcopy(value)


def _watchdog_material_payload(path: Path | None) -> dict[str, Any]:
    """Return watchdog health, without duplicating canonical receipt identity.

    Guardian action meaning is bound directly by
    ``QR_GUARDIAN_ACTION_RECEIPT_SCOPE_V1``.  The watchdog may observe that
    receipt several minutes later; copying event identity/action here made an
    otherwise unchanged market read race two independently published views of
    the same fact.
    """

    if path is None or not path.exists() or not path.is_file():
        return {"parse_status": "MISSING"}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {"parse_status": "INVALID"}
    if not isinstance(raw, Mapping):
        return {"parse_status": "INVALID"}

    automation = raw.get("automation_config")
    automation = automation if isinstance(automation, Mapping) else {}
    automation_weekend = automation.get("weekend_pause")
    automation_weekend = automation_weekend if isinstance(automation_weekend, Mapping) else {}
    weekend = raw.get("weekend_pause")
    weekend = weekend if isinstance(weekend, Mapping) else {}
    guardian = raw.get("guardian_receipt")
    guardian = guardian if isinstance(guardian, Mapping) else {}

    guardian_material: dict[str, Any] = {}
    if "issues" in guardian:
        guardian_material["issues"] = _watchdog_guardian_health_issues(
            guardian.get("issues")
        )

    health = _mapping_fields(
        raw,
        (
            "status",
            "runtime_status",
            "issue_status",
            "overall_status",
            "severity",
            "missed_expected_window",
            "suspected_cause",
            "recommended_operator_action",
            "expected_cadence_minutes",
            "grace_minutes",
            "threshold_minutes",
            "broker_writes_enabled",
            "codex_exec_enabled",
            "no_live_side_effects",
        ),
    )
    if "issues" in raw:
        health["issues"] = _watchdog_material_issues(raw.get("issues"))

    automation_material = _mapping_fields(
        automation,
        (
            "exists",
            "model",
            "reasoning_effort",
            "cadence_minutes",
            "cwd",
            "cwds",
            "rrule",
            "status",
        ),
    )
    if "issues" in automation:
        automation_material["issues"] = _watchdog_material_issues(
            automation.get("issues")
        )

    return {
        "parse_status": "VALID",
        "health": health,
        "automation": {
            **automation_material,
            "weekend_pause": _mapping_fields(
                automation_weekend,
                (
                    "active",
                    "automation_status",
                    "exists",
                    "in_weekend_pause_window",
                    "mode",
                    "qr_trader_managed",
                    "reason",
                ),
            ),
        },
        "weekend_pause": _mapping_fields(
            weekend,
            (
                "active",
                "automation_status",
                "exists",
                "in_weekend_pause_window",
                "mode",
                "qr_trader_managed",
                "reason",
            ),
        ),
        "guardian_receipt": guardian_material,
        "execution_boundary": deepcopy(raw.get("execution_boundary")),
        "environment": deepcopy(raw.get("environment")),
    }


def _watchdog_material_issues(value: Any) -> Any:
    """Remove observation prose while retaining every structured safety fact.

    The watchdog rewrites human-readable issue messages with running values
    such as ``1861.4 minutes old`` on each observation.  Code, severity, and
    structured receipt/config fields are the stable safety contract; message
    prose is a rendered diagnostic and must not invalidate an in-flight GPT
    review when those facts are unchanged.
    """

    if not isinstance(value, list):
        return deepcopy(value)
    material: list[Any] = []
    for item in value:
        if not isinstance(item, Mapping):
            material.append(deepcopy(item))
            continue
        issue_code = str(item.get("code") or "").strip().upper()
        material.append(
            {
                key: deepcopy(field_value)
                for key, field_value in item.items()
                if key != "message" or issue_code not in WATCHDOG_VOLATILE_MESSAGE_CODES
            }
        )
    return material


def _watchdog_guardian_health_issues(value: Any) -> Any:
    """Keep routing health while dropping the duplicated receipt identity."""

    material = _watchdog_material_issues(value)
    if not isinstance(material, list):
        return material
    identity_fields = {
        "message",
        "receipt_identity",
        "receipt_event_id",
        "receipt_action",
        "receipt_lifecycle",
        "receipt_sources",
        "receipt_source_paths",
    }
    result: list[Any] = []
    for item in material:
        if not isinstance(item, Mapping):
            result.append(deepcopy(item))
            continue
        if not any(field in item for field in identity_fields - {"message"}):
            result.append(deepcopy(item))
            continue
        result.append(
            {
                key: deepcopy(field_value)
                for key, field_value in item.items()
                if key not in identity_fields
            }
        )
    return result


def _mapping_fields(source: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: deepcopy(source.get(key)) for key in keys if key in source}


def _recent_resolved_predictions(path: Path, *, limit: int = 8) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                if item.get("schema_version") == 2:
                    if item.get("score_eligible") is not True or item.get("source_snapshot_conflict") is True:
                        continue
                    horizon_results = (
                        item.get("horizon_results")
                        if isinstance(item.get("horizon_results"), dict)
                        else {}
                    )
                    resolved: dict[str, dict[str, Any]] = {}
                    for horizon in ("30m", "2h"):
                        result = horizon_results.get(horizon)
                        if not isinstance(result, dict):
                            continue
                        if result.get("resolution_status") != "RESOLVED_MID_CANDLE_DIAGNOSTIC":
                            continue
                        resolved[horizon] = {
                            key: result.get(key)
                            for key in (
                                "resolution_status",
                                "direction_status",
                                "target_completion_status",
                                "invalidation_status",
                                "first_touch_status",
                                "full_read_status",
                            )
                        }
                    if not resolved:
                        continue
                    rows.append(
                        {
                            "schema_version": 2,
                            "prediction_id": item.get("prediction_id"),
                            "generated_at_utc": item.get("generated_at_utc"),
                            "pair": item.get("pair"),
                            "direction": item.get("direction"),
                            "action": item.get("action"),
                            "verdict": item.get("verdict"),
                            "resolved_horizons": resolved,
                        }
                    )
                    continue

                verdict = str(item.get("verdict") or "PENDING").upper()
                if verdict in {"", "PENDING", "UNRESOLVED"} or item.get("score_eligible") is False:
                    continue
                rows.append(
                    {
                        "schema_version": 1,
                        "prediction_id": item.get("prediction_id"),
                        "generated_at_utc": item.get("generated_at_utc"),
                        "pair": item.get("pair"),
                        "direction": item.get("direction"),
                        "action": item.get("action"),
                        "verdict": verdict,
                        "thirty_minute_verdict": item.get("thirty_minute_verdict"),
                        "two_hour_verdict": item.get("two_hour_verdict"),
                    }
                )
    except OSError:
        return []
    return rows[-limit:]


def _latest_resolved_prediction_id(packet: Mapping[str, Any]) -> str | None:
    rows = packet.get("recent_resolved_predictions")
    if not isinstance(rows, list):
        return None
    for item in reversed(rows):
        if isinstance(item, dict) and item.get("prediction_id"):
            return str(item["prediction_id"])
    return None


def _validate_exact_keys(
    payload: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    required: frozenset[str],
    code: str,
    label: str,
) -> None:
    keys = set(payload)
    unknown = sorted(keys - allowed)
    missing = sorted(required - keys)
    if unknown or missing:
        parts: list[str] = []
        if unknown:
            parts.append("unknown=" + ",".join(unknown))
        if missing:
            parts.append("missing=" + ",".join(missing))
        raise MarketReadOverlayError(code, f"{label} keys invalid: {'; '.join(parts)}")


def _required_text(payload: Mapping[str, Any], key: str, code: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise MarketReadOverlayError(code, f"{key} must be a non-empty string")
    return value.strip()


def _changed_evidence_packet_material(
    stored: Mapping[str, Any],
    rebuilt: Mapping[str, Any],
) -> list[str]:
    """Name the exact material surface that invalidated an in-flight read."""

    changed: list[str] = []
    stored_sources = (
        stored.get("sources") if isinstance(stored.get("sources"), Mapping) else {}
    )
    rebuilt_sources = (
        rebuilt.get("sources") if isinstance(rebuilt.get("sources"), Mapping) else {}
    )
    for name in sorted(set(stored_sources) | set(rebuilt_sources)):
        if canonical_json_sha256(stored_sources.get(name)) != canonical_json_sha256(
            rebuilt_sources.get(name)
        ):
            changed.append(f"source:{name}")
    for field in (
        "schema_version",
        "baseline_sha256",
        "capital_allocation_board_sha256",
        "projection_calibration_evidence_sha256",
    ):
        if stored.get(field) != rebuilt.get(field):
            changed.append(field)
    return changed


def _require_sha_match(*, actual: str, claimed: Any, code: str, label: str) -> None:
    claimed_text = str(claimed or "")
    if not SHA256_RE.fullmatch(claimed_text) or claimed_text != actual:
        raise MarketReadOverlayError(
            code,
            f"{label} sha mismatch: expected={actual} claimed={claimed_text or 'missing'}",
        )


def _normalized_direction(value: Any) -> str | None:
    raw = str(value or "").strip().upper()
    if raw in {"LONG", "UP", "BUY", "BULL", "BULLISH"}:
        return "LONG"
    if raw in {"SHORT", "DOWN", "SELL", "BEAR", "BEARISH"}:
        return "SHORT"
    if raw == "RANGE":
        return "RANGE"
    return None


def _numbers(value: Any) -> list[float]:
    values: list[float] = []
    for match in NUMBER_RE.finditer(str(value or "")):
        try:
            values.append(float(match.group(0)))
        except ValueError:
            continue
    return values


def _plausible_prices(values: list[float], basis: float | None) -> list[float]:
    if basis is None:
        return values
    tolerance = max(abs(basis) * 0.20, 0.20)
    return [value for value in values if value > 0 and abs(value - basis) <= tolerance]


def _basis_price(value: Any) -> float | None:
    if isinstance(value, Mapping):
        for key in ("mid", "price", "basis"):
            parsed = _optional_float(value.get(key))
            if parsed is not None:
                return parsed
        bid = _optional_float(value.get("bid"))
        ask = _optional_float(value.get("ask"))
        if bid is not None and ask is not None:
            return (bid + ask) / 2.0
    return _optional_float(value)


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value) if value is not None else None
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed is not None and math.isfinite(parsed) else None


def _parse_utc(value: Any, *, code: str) -> datetime:
    if not value:
        raise MarketReadOverlayError(code, "timestamp is missing")
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise MarketReadOverlayError(code, f"invalid timestamp: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _utc_now(now: datetime | None) -> datetime:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc)


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise MarketReadOverlayError("MARKET_READ_ARTIFACT_MISSING", f"{label} missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise MarketReadOverlayError("MARKET_READ_ARTIFACT_UNREADABLE", f"{label} unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise MarketReadOverlayError("MARKET_READ_ARTIFACT_INVALID", f"{label} must be a JSON object: {path}")
    return payload


def _load_optional_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
