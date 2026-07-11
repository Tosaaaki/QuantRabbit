from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any


EVALUATOR_NAME = "guardian_tuning_threshold_evaluator_v1"
SOURCE_SCHEMA_VERSION = 5
COHORT_GENERATOR_NAME = "guardian_tuning_cohort_builder_v3"
MAX_SAMPLE_COUNT = 5_000
MIN_FORWARD_SAMPLE_COUNT = 20
MIN_CANDIDATE_TRADE_RATIO = 0.80
FIXED_ACCEPTANCE_THRESHOLD = 0.0
PRIMARY_METRIC = "net_jpy_per_1000_units_per_opportunity"
OBJECTIVE = "MAXIMIZE"
METRIC_NAMES = (
    "expectancy_net_jpy_per_1000_units",
    "hit_rate",
    "net_jpy_per_1000_units",
    "net_jpy_per_1000_units_per_opportunity",
    "trade_count",
    "trade_rate",
)

# Version 1 has one canonical, entry-time signal source today:
# ``forecast_history.jsonl`` joined through ``entry_thesis_ledger.jsonl``.
# Other score floors need their own pre-entry append-only evidence before they
# can be evaluated without hand-authored/post-outcome signal values.
SUPPORTED_THRESHOLD_PARAMETERS = frozenset(
    {
        "forecast_confidence_floor",
    }
)

_SOURCE_FIELDS = {
    "schema_version",
    "cohort_id",
    "source_watermark",
    "selection_cutoff_utc",
    "pair",
    "bot_family",
    "lane_id",
    "parameter",
    "validation_contract",
    "provenance",
    "samples",
}
_SAMPLE_FIELDS = {
    "sample_id",
    "pair",
    "bot_family",
    "lane_id",
    "trade_id",
    "order_id",
    "entry_at_utc",
    "closed_at_utc",
    "signal_observed_at_utc",
    "signal_record_sha256",
    "signal_value",
    "realized_net_jpy",
    "entry_units",
    "net_jpy_per_1000_units",
}
_PROVENANCE_FIELDS = {
    "generator",
    "execution_ledger_coverage_start_utc",
    "last_oanda_transaction_id",
    "post_cost_financing_included",
}
_VALIDATION_CONTRACT_FIELDS = {
    "mode",
    "review_digest_sha256",
    "review_completed_at_utc",
    "minimum_sample_count",
}
_SOURCE_WATERMARK_FIELDS = {
    "selection_cutoff_utc",
    "last_oanda_transaction_id",
    "ledger_rowid_watermark",
    "ledger_prefix_sha256",
    "canonical_outcome_set_sha256",
    "entry_thesis_prefix_bytes",
    "entry_thesis_prefix_sha256",
    "forecast_history_prefix_bytes",
    "forecast_history_prefix_sha256",
}

# OANDA RFC3339 timestamps carry nanoseconds while Python ``datetime`` stores
# microseconds.  Normalize before ``fromisoformat`` so Python 3.9 and 3.12 parse
# and truncate the same instant instead of accepting different input surfaces.
_SUBMICROSECOND_RFC3339_RE = re.compile(
    r"^(?P<prefix>.+T\d{2}:\d{2}:\d{2})\.(?P<fraction>\d{7,})(?P<suffix>Z|[+-]\d{2}:\d{2})$"
)


def source_identity(payload: dict[str, Any]) -> tuple[str, dict[str, Any], int, str]:
    identity = validate_source(payload)
    return (
        str(identity["cohort_id"]),
        dict(identity["source_watermark"]),
        int(identity["sample_count"]),
        str(identity["parameter"]),
    )


def validate_source(payload: dict[str, Any]) -> dict[str, Any]:
    if set(payload) != _SOURCE_FIELDS:
        raise ValueError("source data contains unsupported top-level fields")
    if payload.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise ValueError(
            f"source data must use schema_version {SOURCE_SCHEMA_VERSION}"
        )
    cohort_id = str(payload.get("cohort_id") or "").strip()
    watermark = payload.get("source_watermark")
    cutoff = _utc(payload.get("selection_cutoff_utc"))
    pair = str(payload.get("pair") or "").strip().upper()
    bot_family = str(payload.get("bot_family") or "").strip().lower()
    lane_id = str(payload.get("lane_id") or "").strip()
    parameter = str(payload.get("parameter") or "").strip()
    provenance = payload.get("provenance")
    validation_contract = payload.get("validation_contract")
    samples = payload.get("samples")
    if (
        not cohort_id
        or not isinstance(watermark, dict)
        or set(watermark) != _SOURCE_WATERMARK_FIELDS
    ):
        raise ValueError("source data needs cohort_id and a non-empty source_watermark")
    if cutoff is None:
        raise ValueError("selection_cutoff_utc must be an aware UTC timestamp")
    if _utc(watermark.get("selection_cutoff_utc")) != cutoff:
        raise ValueError("source watermark cutoff must equal selection_cutoff_utc")
    if not str(watermark.get("last_oanda_transaction_id") or "").isdigit():
        raise ValueError("source transaction watermark is invalid")
    for key in (
        "ledger_rowid_watermark",
        "entry_thesis_prefix_bytes",
        "forecast_history_prefix_bytes",
    ):
        value = watermark.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"source watermark {key} must be a positive integer")
    for key in (
        "ledger_prefix_sha256",
        "canonical_outcome_set_sha256",
        "entry_thesis_prefix_sha256",
        "forecast_history_prefix_sha256",
    ):
        digest = str(watermark.get(key) or "")
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError(f"source watermark {key} is invalid")
    if not pair or not bot_family or not lane_id:
        raise ValueError("source data must bind pair, bot_family, and lane_id")
    if parameter not in SUPPORTED_THRESHOLD_PARAMETERS:
        raise ValueError("source data parameter is not supported by evaluator v1")
    if (
        not isinstance(validation_contract, dict)
        or set(validation_contract) != _VALIDATION_CONTRACT_FIELDS
        or validation_contract.get("mode") != "FORWARD_POST_REVIEW"
        or validation_contract.get("minimum_sample_count") != MIN_FORWARD_SAMPLE_COUNT
    ):
        raise ValueError("source validation contract must be fixed forward-post-review evidence")
    review_digest = str(validation_contract.get("review_digest_sha256") or "")
    if len(review_digest) != 64 or any(ch not in "0123456789abcdef" for ch in review_digest):
        raise ValueError("source review digest is invalid")
    review_completed_at = _utc(validation_contract.get("review_completed_at_utc"))
    if review_completed_at is None:
        raise ValueError("source review completion timestamp is invalid")
    if not isinstance(provenance, dict) or set(provenance) != _PROVENANCE_FIELDS:
        raise ValueError("source provenance is incomplete")
    if (
        provenance.get("generator") != COHORT_GENERATOR_NAME
        or provenance.get("post_cost_financing_included") is not True
        or _utc(provenance.get("execution_ledger_coverage_start_utc")) is None
        or not str(provenance.get("last_oanda_transaction_id") or "").isdigit()
    ):
        raise ValueError("source provenance is not canonical post-cost ledger evidence")
    if not isinstance(samples, list) or len(samples) != MIN_FORWARD_SAMPLE_COUNT:
        raise ValueError(
            f"source data needs exactly the first {MIN_FORWARD_SAMPLE_COUNT} forward entries"
        )
    if len(samples) > MAX_SAMPLE_COUNT:
        raise ValueError("source data exceeds the bounded sample limit")

    sample_ids: set[str] = set()
    trade_ids: set[str] = set()
    for sample in samples:
        if not isinstance(sample, dict) or set(sample) != _SAMPLE_FIELDS:
            raise ValueError("each sample has an unsupported field set")
        sample_id = str(sample.get("sample_id") or "").strip()
        trade_id = str(sample.get("trade_id") or "").strip()
        order_id = str(sample.get("order_id") or "").strip()
        entry_at = _utc(sample.get("entry_at_utc"))
        closed_at = _utc(sample.get("closed_at_utc"))
        signal_observed_at = _utc(sample.get("signal_observed_at_utc"))
        signal_record_sha256 = str(sample.get("signal_record_sha256") or "")
        if not sample_id or sample_id in sample_ids:
            raise ValueError("sample_id must be non-empty and unique")
        if not trade_id or trade_id in trade_ids or not order_id:
            raise ValueError("trade_id must be unique and order_id must be present")
        if (
            entry_at is None
            or closed_at is None
            or signal_observed_at is None
            or entry_at <= review_completed_at
            or signal_observed_at >= entry_at
            or closed_at < entry_at
            or closed_at > cutoff
        ):
            raise ValueError("sample close must be valid and no later than the cutoff")
        if len(signal_record_sha256) != 64 or any(
            ch not in "0123456789abcdef" for ch in signal_record_sha256
        ):
            raise ValueError("sample signal record digest is invalid")
        if (
            str(sample.get("pair") or "").strip().upper() != pair
            or str(sample.get("bot_family") or "").strip().lower() != bot_family
            or str(sample.get("lane_id") or "").strip() != lane_id
        ):
            raise ValueError("sample pair/family/lane does not match cohort identity")
        signal = _finite(sample.get("signal_value"), field="signal_value")
        realized = _finite(sample.get("realized_net_jpy"), field="realized_net_jpy")
        units = _finite(sample.get("entry_units"), field="entry_units")
        normalized = _finite(
            sample.get("net_jpy_per_1000_units"),
            field="net_jpy_per_1000_units",
        )
        if units == 0.0:
            raise ValueError("entry_units must be non-zero")
        expected_normalized = realized / abs(units) * 1000.0
        if not math.isclose(normalized, expected_normalized, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("normalized post-cost outcome does not match realized JPY and units")
        # Explicit conversions above are also the finite-value validation.
        _ = signal
        sample_ids.add(sample_id)
        trade_ids.add(trade_id)
    return {
        "cohort_id": cohort_id,
        "source_watermark": watermark,
        "selection_cutoff_utc": cutoff.isoformat(),
        "pair": pair,
        "bot_family": bot_family,
        "lane_id": lane_id,
        "parameter": parameter,
        "sample_count": len(samples),
        "validation_contract": dict(validation_contract),
    }


def evaluate_threshold_cohort(
    payload: dict[str, Any],
    *,
    parameter: str,
    current_value: float,
    candidate_value: float,
) -> dict[str, Any]:
    identity = validate_source(payload)
    if parameter != identity["parameter"] or parameter not in SUPPORTED_THRESHOLD_PARAMETERS:
        raise ValueError("prepared parameter does not match the frozen source cohort")
    current = _finite(current_value, field="current_value")
    candidate = _finite(candidate_value, field="candidate_value")
    if not (0.0 <= current < candidate <= 1.0):
        raise ValueError(
            "evaluator v1 supports confidence-floor tightening within [0, 1] only"
        )

    samples = payload["samples"]
    # The production baseline is every trade that actually executed. Existing
    # audited support paths can admit a trade below the nominal base floor, so
    # filtering baseline samples by ``current`` would erase real outcomes and
    # could manufacture an improvement. Only the candidate is counterfactual.
    baseline = _metrics_for_floor(samples, floor=None)
    candidate_metrics = _metrics_for_floor(samples, floor=candidate)
    minimum_candidate_trades = (
        1
        if baseline["trade_count"] == 0
        else max(1, math.ceil(baseline["trade_count"] * MIN_CANDIDATE_TRADE_RATIO))
    )
    return {
        "schema_version": 1,
        "status": "EVALUATION_COMPLETED",
        "evaluator": EVALUATOR_NAME,
        **identity,
        "current_value": current,
        "candidate_value": candidate,
        "metric_names": list(METRIC_NAMES),
        "baseline_metrics": baseline,
        "candidate_metrics": candidate_metrics,
        "acceptance_constraints": {
            "minimum_forward_sample_count": MIN_FORWARD_SAMPLE_COUNT,
            "forward_sample_count_sufficient": len(samples) >= MIN_FORWARD_SAMPLE_COUNT,
            "minimum_candidate_trade_ratio": MIN_CANDIDATE_TRADE_RATIO,
            "minimum_candidate_trade_count": minimum_candidate_trades,
            "candidate_trade_count_sufficient": (
                candidate_metrics["trade_count"] >= minimum_candidate_trades
            ),
            "candidate_positive_expectancy": (
                candidate_metrics[PRIMARY_METRIC] > 0.0
            ),
        },
    }


def derive_result(
    evaluation: dict[str, Any],
    *,
    primary_metric: str,
    objective: str,
    acceptance_threshold: float,
) -> tuple[str, float]:
    if primary_metric != PRIMARY_METRIC or objective != OBJECTIVE:
        raise ValueError("evaluator v1 requires its fixed primary metric and objective")
    threshold = _finite(acceptance_threshold, field="acceptance_threshold")
    if threshold != FIXED_ACCEPTANCE_THRESHOLD:
        raise ValueError("evaluator v1 uses its fixed zero threshold with strict > comparison")
    baseline = _finite(
        evaluation["baseline_metrics"][PRIMARY_METRIC],
        field="baseline primary metric",
    )
    candidate = _finite(
        evaluation["candidate_metrics"][PRIMARY_METRIC],
        field="candidate primary metric",
    )
    improvement = candidate - baseline
    if not math.isfinite(improvement):
        raise ValueError("metric improvement must be finite")
    constraints = evaluation.get("acceptance_constraints")
    frequency_ok = bool(
        isinstance(constraints, dict)
        and constraints.get("candidate_trade_count_sufficient") is True
    )
    forward_ok = bool(
        isinstance(constraints, dict)
        and constraints.get("forward_sample_count_sufficient") is True
    )
    positive_ok = bool(
        isinstance(constraints, dict)
        and constraints.get("candidate_positive_expectancy") is True
    )
    result = (
        "ACCEPTED_IMPROVEMENT"
        if forward_ok and frequency_ok and positive_ok and improvement > threshold
        else "REJECTED_NO_IMPROVEMENT"
    )
    return result, improvement


def source_semantic_digest(payload: dict[str, Any]) -> str:
    """Stable no-repeat identity, excluding labels, refs, and JSON formatting."""

    import hashlib
    import json

    identity = validate_source(payload)
    samples = [
        {
            key: sample[key]
            for key in (
                "trade_id",
                "order_id",
                "entry_at_utc",
                "closed_at_utc",
                "signal_observed_at_utc",
                "signal_value",
                "realized_net_jpy",
                "entry_units",
                "net_jpy_per_1000_units",
            )
        }
        for sample in sorted(payload["samples"], key=lambda item: str(item["trade_id"]))
    ]
    material = {
        "pair": identity["pair"],
        "bot_family": identity["bot_family"],
        "lane_id": identity["lane_id"],
        "parameter": identity["parameter"],
        # Review labels/timestamps prove forward ordering but must not create a
        # new no-repeat identity for the same candidate and exact samples.
        "validation_mode": identity["validation_contract"]["mode"],
        "minimum_sample_count": identity["validation_contract"][
            "minimum_sample_count"
        ],
        "samples": samples,
    }
    return hashlib.sha256(
        json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def evaluate_precommitted_threshold_cohort(
    payload: dict[str, Any],
    *,
    parameter: str,
    current_value: float,
    candidate_value: float,
    primary_metric: str,
    objective: str,
    acceptance_threshold: float,
) -> dict[str, Any]:
    """Evaluate one precommitted contract with trusted declarative code.

    Content-addressed evaluator files are provenance only.  Runtime callers use
    this imported implementation so data-directory Python is never imported or
    executed during either the initial run or a later evidence revalidation.
    """

    evaluation = evaluate_threshold_cohort(
        payload,
        parameter=parameter,
        current_value=current_value,
        candidate_value=candidate_value,
    )
    threshold = _finite(acceptance_threshold, field="acceptance_threshold")
    result, improvement = derive_result(
        evaluation,
        primary_metric=primary_metric,
        objective=objective,
        acceptance_threshold=threshold,
    )
    return {
        **evaluation,
        "primary_metric": primary_metric,
        "objective": objective,
        "acceptance_threshold": threshold,
        "improvement": improvement,
        "derived_result": result,
    }


def _metrics_for_floor(
    samples: list[dict[str, Any]],
    *,
    floor: float | None,
) -> dict[str, Any]:
    selected = [
        _finite(sample["net_jpy_per_1000_units"], field="normalized outcome")
        for sample in samples
        if floor is None
        or _finite(sample["signal_value"], field="signal value") >= floor
    ]
    trade_count = len(selected)
    opportunities = len(samples)
    try:
        net_normalized = math.fsum(selected)
    except OverflowError:
        raise ValueError("normalized outcome sum overflowed") from None
    if not math.isfinite(net_normalized):
        raise ValueError("normalized outcome sum must be finite")
    wins = sum(1 for value in selected if value > 0.0)
    return {
        "expectancy_net_jpy_per_1000_units": (
            net_normalized / trade_count if trade_count else 0.0
        ),
        "hit_rate": wins / trade_count if trade_count else 0.0,
        "net_jpy_per_1000_units": net_normalized,
        "net_jpy_per_1000_units_per_opportunity": net_normalized / opportunities,
        "trade_count": trade_count,
        "trade_rate": trade_count / opportunities,
    }


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    try:
        parsed = float(value)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f"{field} must be a finite float") from None
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def _utc(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    match = _SUBMICROSECOND_RFC3339_RE.fullmatch(text)
    if match is not None:
        text = (
            f"{match.group('prefix')}.{match.group('fraction')[:6]}"
            f"{match.group('suffix')}"
        )
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)
