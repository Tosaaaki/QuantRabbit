"""Market-derived truth and fixed-denominator scoring for AI forward V2.

The model is tested only on a preregistered 24-hour direction/size decision.
Targets, invalidations, and prose confidence remain diagnostic fields; they do
not silently change the economic scorer.  Truth is acquired only after every
response terminal for the day is immutable.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

from quant_rabbit.dojo_ai_discretion import (
    DIAGNOSTIC_TIER,
    canonical_sha256,
    score_sealed_response,
    seal_answer_key,
    validate_score_receipt,
)
from quant_rabbit.dojo_ai_forward import (
    EXPECTED_CELL_COUNT,
    EXPECTED_DAY_COUNT,
    M5_MAX_CONTIGUOUS_GAP,
    M5_MINIMUM_COVERAGE,
    OFFICIAL_OANDA_BASE_URL,
    PAIR,
    PHASE_ID,
    SOURCE_GRANULARITY,
    SOURCE_PRICE,
    TRUTH_FINANCING_PIPS_PER_DAY,
    TRUTH_NOTIONAL_MULTIPLE,
    TRUTH_SEAL_DEADLINE_DELAY,
    TRUTH_SLIPPAGE_PIPS_PER_FILL,
    validate_precommit,
)
from quant_rabbit.dojo_market_calendar import expected_oanda_fx_slots
from quant_rabbit.dojo_prompt_phase import LOCKED_VARIANT_PROMPT_SHA256


TRUTH_REQUEST_CONTRACT = "QR_DOJO_AI_MARKET_TRUTH_REQUEST_V1"
TRUTH_CAPTURE_CONTRACT = "QR_DOJO_AI_MARKET_TRUTH_CAPTURE_V1"
TRUTH_BUNDLE_CONTRACT = "QR_DOJO_AI_MARKET_TRUTH_BUNDLE_V1"
DAY_SCORE_CONTRACT = "QR_DOJO_AI_MARKET_TRUTH_DAY_SCORE_V1"
PHASE_SCORE_CONTRACT = "QR_DOJO_AI_FORWARD_PHASE_SCORE_V2"
TRUTH_COVERAGE_POLICY = "OANDA_M5_EXACT_BOUNDARY_FIXED_HORIZON_TRUTH_V1"
_VARIANTS = tuple(LOCKED_VARIANT_PROMPT_SHA256)
_TIME = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(0+))?Z")
_DECIMAL = re.compile(r"(?:0|[1-9][0-9]*)\.[0-9]+")


class DojoAITruthError(ValueError):
    """Market truth, score, or phase aggregation is not causally valid."""


def build_truth_request(
    precommit: Mapping[str, Any],
    day_seal: Mapping[str, Any],
    cell_terminals: Sequence[Mapping[str, Any]],
    *,
    now_utc: datetime,
) -> dict[str, Any]:
    """Freeze one post-response OANDA request before network access."""

    lock = validate_precommit(precommit)
    day = _validated_day(day_seal, lock)
    terminals = _validated_terminal_set(day, cell_terminals)
    now = _utc(now_utc, "now_utc")
    not_before = _parse_utc(day["schedule"]["truth_not_before_utc"], "truth_not_before")
    deadline = not_before + TRUTH_SEAL_DEADLINE_DELAY
    if now < not_before or now > deadline:
        raise DojoAITruthError("truth request is outside its fixed acquisition window")
    if any(_parse_utc(row["sealed_at_utc"], "terminal sealed_at") > now for row in terminals):
        raise DojoAITruthError("truth request predates a response terminal")
    cutoff = _parse_utc(day["schedule"]["decision_cutoff_utc"], "decision cutoff")
    horizon = _parse_utc(day["schedule"]["truth_not_before_utc"], "truth maturity")
    horizon -= timedelta(seconds=lock["market_policy"]["truth_maturity_delay_seconds"])
    body = {
        "contract": TRUTH_REQUEST_CONTRACT,
        "schema_version": 1,
        "state": "TRUTH_REQUESTED_AFTER_RESPONSES",
        "experiment_id": lock["experiment_id"],
        "phase_id": PHASE_ID,
        "ordinal": day["ordinal"],
        "blind_day_rank": day["schedule"]["blind_day_rank"],
        "blind_day_id": day["schedule"]["blind_day_id"],
        "requested_at_utc": _iso(now),
        "truth_seal_deadline_utc": _iso(deadline),
        "precommit_sha256": lock["precommit_sha256"],
        "day_seal_sha256": day["day_seal_sha256"],
        "cell_terminal_sha256s": [row["cell_terminal_sha256"] for row in terminals],
        "method": "GET",
        "base_url": OFFICIAL_OANDA_BASE_URL,
        "path": f"/v3/instruments/{PAIR}/candles",
        "query": {
            "from": _iso(cutoff),
            "granularity": SOURCE_GRANULARITY,
            "includeFirst": "true",
            "price": SOURCE_PRICE,
            "to": _iso(horizon),
        },
        "response_and_answer_material_present": False,
        "authority": _authority(),
    }
    return _seal(body, "truth_request_sha256")


def validate_truth_request(
    value: Mapping[str, Any],
    precommit: Mapping[str, Any],
    day_seal: Mapping[str, Any],
    cell_terminals: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    request = _mapping(value, "truth request")
    _validate_seal(request, "truth_request_sha256")
    expected = build_truth_request(
        precommit,
        day_seal,
        cell_terminals,
        now_utc=_parse_utc(request.get("requested_at_utc"), "requested_at_utc"),
    )
    if request != expected:
        raise DojoAITruthError("truth request bytes or parents drifted")
    return _snapshot(request)


def build_truth_capture(
    request: Mapping[str, Any],
    response: Any,
    *,
    acquired_at_utc: datetime,
) -> dict[str, Any]:
    """Capture the first finite JSON response before semantic validation."""

    request_value = _mapping(request, "truth request")
    _validate_seal(request_value, "truth_request_sha256")
    acquired = _utc(acquired_at_utc, "acquired_at_utc")
    requested = _parse_utc(request_value["requested_at_utc"], "requested_at_utc")
    deadline = _parse_utc(
        request_value["truth_seal_deadline_utc"], "truth_seal_deadline_utc"
    )
    if acquired < requested or acquired > deadline:
        raise DojoAITruthError("truth capture ordering or deadline is invalid")
    response_value = _snapshot(response)
    body = {
        "contract": TRUTH_CAPTURE_CONTRACT,
        "schema_version": 1,
        "state": "FIRST_TRUTH_RESPONSE_CAPTURED",
        "ordinal": request_value["ordinal"],
        "acquired_at_utc": _iso(acquired),
        "truth_request": request_value,
        "truth_request_sha256": request_value["truth_request_sha256"],
        "response": response_value,
        "response_sha256": canonical_sha256(response_value),
        "credentials_persisted": False,
        "response_reselection_allowed": False,
        "authority": _authority(),
    }
    return _seal(body, "truth_capture_sha256")


def validate_truth_capture(
    value: Mapping[str, Any], request: Mapping[str, Any]
) -> dict[str, Any]:
    capture = _mapping(value, "truth capture")
    _validate_seal(capture, "truth_capture_sha256")
    expected = build_truth_capture(
        request,
        capture.get("response"),
        acquired_at_utc=_parse_utc(capture.get("acquired_at_utc"), "acquired_at_utc"),
    )
    if capture != expected:
        raise DojoAITruthError("truth capture bytes or request parent drifted")
    return _snapshot(capture)


def build_truth_bundle(
    precommit: Mapping[str, Any],
    day_seal: Mapping[str, Any],
    cell_terminals: Sequence[Mapping[str, Any]],
    truth_capture: Mapping[str, Any],
    *,
    sealed_at_utc: datetime,
) -> dict[str, Any]:
    """Derive fixed-horizon executable returns and packet-bound answer keys."""

    lock = validate_precommit(precommit)
    day = _validated_day(day_seal, lock)
    terminals = _validated_terminal_set(day, cell_terminals)
    request = validate_truth_request(
        truth_capture.get("truth_request", {}), lock, day, terminals
    )
    capture = validate_truth_capture(truth_capture, request)
    sealed = _utc(sealed_at_utc, "sealed_at_utc")
    acquired = _parse_utc(capture["acquired_at_utc"], "acquired_at_utc")
    if sealed < acquired:
        raise DojoAITruthError("truth bundle predates its first response capture")
    candles, coverage = _normalize_truth_response(
        capture["response"],
        start=_parse_utc(request["query"]["from"], "truth query from"),
        end=_parse_utc(request["query"]["to"], "truth query to"),
    )
    returns, execution = _fixed_horizon_returns(candles)
    answer_keys: dict[str, dict[str, Any]] = {}
    for terminal in terminals:
        if terminal["state"] != "RESPONSE_SEALED":
            continue
        receipt = terminal["response_receipt"]
        key = seal_answer_key(
            trial_id=receipt["trial_id"],
            packet_sha256=receipt["packet_sha256"],
            returns=returns,
            sealed_at_utc=sealed,
        )
        answer_keys[terminal["cell_id"]] = key
    body = {
        "contract": TRUTH_BUNDLE_CONTRACT,
        "schema_version": 1,
        "validity_status": "VALID",
        "state": "MARKET_TRUTH_AND_KEYS_SEALED",
        "experiment_id": lock["experiment_id"],
        "phase_id": PHASE_ID,
        "ordinal": day["ordinal"],
        "blind_day_rank": day["schedule"]["blind_day_rank"],
        "blind_day_id": day["schedule"]["blind_day_id"],
        "sealed_at_utc": _iso(sealed),
        "precommit_sha256": lock["precommit_sha256"],
        "day_seal_sha256": day["day_seal_sha256"],
        "cell_terminal_sha256s": [row["cell_terminal_sha256"] for row in terminals],
        "truth_request_sha256": request["truth_request_sha256"],
        "truth_capture_sha256": capture["truth_capture_sha256"],
        "truth_response_sha256": capture["response_sha256"],
        "truth_window_start_utc": request["query"]["from"],
        "truth_window_end_utc": request["query"]["to"],
        "truth_semantics": "FIXED_24H_DIRECTION_AND_SIZE_ONLY",
        "coverage": coverage,
        "execution_evidence": execution,
        "returns": returns,
        "answer_keys": dict(sorted(answer_keys.items())),
        "target_invalidation_confidence_used_for_return": False,
        "evidence_tier": DIAGNOSTIC_TIER,
        "authority": _authority(),
    }
    return _seal(body, "truth_bundle_sha256")


def validate_truth_bundle_with_capture(
    value: Mapping[str, Any],
    precommit: Mapping[str, Any],
    day_seal: Mapping[str, Any],
    cell_terminals: Sequence[Mapping[str, Any]],
    truth_capture: Mapping[str, Any],
) -> dict[str, Any]:
    bundle = _mapping(value, "truth bundle")
    _validate_seal(bundle, "truth_bundle_sha256")
    expected = build_truth_bundle(
        precommit,
        day_seal,
        cell_terminals,
        truth_capture,
        sealed_at_utc=_parse_utc(bundle.get("sealed_at_utc"), "sealed_at_utc"),
    )
    if bundle != expected:
        raise DojoAITruthError("truth bundle is not market-derived from its capture")
    return _snapshot(bundle)


def build_day_score(
    precommit: Mapping[str, Any],
    day_seal: Mapping[str, Any],
    cell_terminals: Sequence[Mapping[str, Any]],
    truth_capture: Mapping[str, Any],
    truth_bundle: Mapping[str, Any],
    *,
    scored_at_utc: datetime,
) -> dict[str, Any]:
    """Score all three fixed cells; deadline failures remain zero-return failures."""

    lock = validate_precommit(precommit)
    day = _validated_day(day_seal, lock)
    terminals = _validated_terminal_set(day, cell_terminals)
    bundle = validate_truth_bundle_with_capture(
        truth_bundle, lock, day, terminals, truth_capture
    )
    scored_at = _utc(scored_at_utc, "scored_at_utc")
    if scored_at < _parse_utc(bundle["sealed_at_utc"], "truth sealed_at"):
        raise DojoAITruthError("day score predates truth sealing")
    rows: list[dict[str, Any]] = []
    for terminal in terminals:
        cell = _find_cell(day, terminal["cell_id"])
        if terminal["state"] == "MISSING_RESPONSE_DEADLINE":
            rows.append(_failure_row(day, cell, terminal, "MISSING_RESPONSE_DEADLINE"))
            continue
        answer_key = bundle["answer_keys"].get(terminal["cell_id"])
        if answer_key is None:
            raise DojoAITruthError("response-sealed cell lacks an answer key")
        score = score_sealed_response(
            terminal["response_receipt"],
            packet=cell["packet"],
            prompt_lock=lock["prompt_locks"][cell["variant_id"]],
            model_manifest=cell["model_manifest"],
            capability_manifest=cell["capability_manifest"],
            scorer_lock=lock["scorer_lock"],
            answer_key_loader=lambda key=answer_key: key,
            opened_at_utc=scored_at,
        )
        score = validate_score_receipt(score)
        if score["validity_status"] != "VALID":
            raise DojoAITruthError("invalidated score cannot become positive evidence")
        rows.append(
            {
                "ordinal": day["ordinal"],
                "blind_day_rank": day["schedule"]["blind_day_rank"],
                "blind_day_id": day["schedule"]["blind_day_id"],
                "variant_id": cell["variant_id"],
                "cell_id": terminal["cell_id"],
                "terminal_state": terminal["state"],
                "terminal_artifact_sha256": terminal["cell_terminal_sha256"],
                "status": "VALID_RESPONSE",
                "response_failure": False,
                "economic_fallback": None,
                "return_key": score["return_key"],
                "net_return": score["net_return"],
                "net_log_growth": score["net_log_growth"],
                "score_receipt": score,
                "score_receipt_sha256": score["score_receipt_sha256"],
                "declared_model_lineage": score["declared_model_lineage"],
            }
        )
    rows.sort(key=lambda row: row["variant_id"])
    body = {
        "contract": DAY_SCORE_CONTRACT,
        "schema_version": 1,
        "validity_status": "VALID",
        "state": "DAY_SCORED",
        "experiment_id": lock["experiment_id"],
        "phase_id": PHASE_ID,
        "ordinal": day["ordinal"],
        "scored_at_utc": _iso(scored_at),
        "precommit_sha256": lock["precommit_sha256"],
        "day_seal_sha256": day["day_seal_sha256"],
        "truth_bundle_sha256": bundle["truth_bundle_sha256"],
        "cell_results": rows,
        "evidence_tier": DIAGNOSTIC_TIER,
        "authority": _authority(),
    }
    return _seal(body, "day_score_sha256")


def validate_day_score(
    value: Mapping[str, Any],
    precommit: Mapping[str, Any],
    day_seal: Mapping[str, Any],
    cell_terminals: Sequence[Mapping[str, Any]],
    truth_capture: Mapping[str, Any],
    truth_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    score = _mapping(value, "truth day score")
    _validate_seal(score, "day_score_sha256")
    expected = build_day_score(
        precommit,
        day_seal,
        cell_terminals,
        truth_capture,
        truth_bundle,
        scored_at_utc=_parse_utc(score.get("scored_at_utc"), "scored_at_utc"),
    )
    if score != expected:
        raise DojoAITruthError("truth day score is not recomputable")
    return _snapshot(score)


def build_phase_score(
    precommit: Mapping[str, Any],
    phase_index: Mapping[str, Any],
    day_seals: Sequence[Mapping[str, Any]],
    day_scores: Sequence[Mapping[str, Any]],
    *,
    sealed_at_utc: datetime,
) -> dict[str, Any]:
    """Aggregate the immutable 90-cell denominator without prompt selection."""

    lock = validate_precommit(precommit)
    index = _mapping(phase_index, "phase index")
    _validate_external_seal(index, "phase_index_sha256")
    if (
        index.get("precommit_sha256") != lock["precommit_sha256"]
        or index.get("allocated_cell_count") != EXPECTED_CELL_COUNT
        or len(index.get("cell_index", [])) != EXPECTED_CELL_COUNT
    ):
        raise DojoAITruthError("phase index parent or denominator drifted")
    if len(day_seals) != EXPECTED_DAY_COUNT:
        raise DojoAITruthError("phase score requires exactly 30 day seals")
    day_by_ordinal = {day["ordinal"]: day for day in day_seals}
    validated_scores = [
        _validate_phase_day_score_shape(score, day_by_ordinal, lock)
        for score in day_scores
    ]
    score_by_ordinal = {score["ordinal"]: score for score in validated_scores}
    request_day_ordinals = {
        day["ordinal"] for day in day_seals if day["state"] == "REQUESTS_SEALED"
    }
    if set(score_by_ordinal) != request_day_ordinals:
        raise DojoAITruthError("phase score day-score set is incomplete or extra")
    row_by_cell: dict[str, dict[str, Any]] = {}
    for score in validated_scores:
        for row in score["cell_results"]:
            if row["cell_id"] in row_by_cell:
                raise DojoAITruthError("phase score reuses a cell result")
            row_by_cell[row["cell_id"]] = _snapshot(row)
    terminal_rows: list[dict[str, Any]] = []
    for indexed in index["cell_index"]:
        cell_id = indexed["cell_id"]
        if indexed["terminal_state"] == "MISSING_SOURCE_DEADLINE":
            terminal_rows.append(
                {
                    **{key: indexed[key] for key in ("ordinal", "blind_day_rank", "blind_day_id", "variant_id", "cell_id")},
                    "status": "MISSING_SOURCE_DEADLINE",
                    "response_failure": True,
                    "economic_fallback": "SYNTHETIC_FLAT_ZERO_RETURN",
                    "return_key": None,
                    "net_return": 0.0,
                    "net_log_growth": 0.0,
                    "score_receipt_sha256": None,
                    "declared_model_lineage": None,
                }
            )
            continue
        row = row_by_cell.pop(cell_id, None)
        identity_keys = (
            "ordinal",
            "blind_day_rank",
            "blind_day_id",
            "variant_id",
            "cell_id",
            "terminal_state",
            "terminal_artifact_sha256",
        )
        if row is None or any(row.get(key) != indexed.get(key) for key in identity_keys):
            raise DojoAITruthError("phase index and day score cell binding drifted")
        terminal_rows.append(
            {
                key: row[key]
                for key in (
                    "ordinal",
                    "blind_day_rank",
                    "blind_day_id",
                    "variant_id",
                    "cell_id",
                    "status",
                    "response_failure",
                    "economic_fallback",
                    "return_key",
                    "net_return",
                    "net_log_growth",
                    "score_receipt_sha256",
                    "declared_model_lineage",
                )
            }
        )
    if row_by_cell or len(terminal_rows) != EXPECTED_CELL_COUNT:
        raise DojoAITruthError("phase score terminal denominator is not exact")
    terminal_rows.sort(key=lambda row: (row["blind_day_rank"], row["variant_id"]))
    variants = {
        variant: _variant_summary(terminal_rows, variant, lock)
        for variant in _VARIANTS
    }
    by_key = {(row["blind_day_rank"], row["variant_id"]): row for row in terminal_rows}
    paired = []
    for rank in range(1, EXPECTED_DAY_COUNT + 1):
        a = by_key[(rank, "A_FABLE_MINIMAL")]
        b = by_key[(rank, "B_CALIBRATED_ABSTENTION")]
        c = by_key[(rank, "C_STRUCTURAL_REGIME")]
        complete = not any(row["response_failure"] for row in (a, b, c))
        paired.append(
            {
                "blind_day_rank": rank,
                "blind_day_id": a["blind_day_id"],
                "all_three_valid": complete,
                "b_minus_a_log_growth": b["net_log_growth"] - a["net_log_growth"],
                "c_minus_a_log_growth": c["net_log_growth"] - a["net_log_growth"],
                "inference_eligible": complete,
            }
        )
    failure_count = sum(row["response_failure"] for row in terminal_rows)
    best_monthly = max(summary["calendar_30d_multiple"] for summary in variants.values())
    body = {
        "contract": PHASE_SCORE_CONTRACT,
        "schema_version": 2,
        "validity_status": "VALID",
        "state": "PHASE_SCORED_DIAGNOSTIC",
        "experiment_id": lock["experiment_id"],
        "phase_id": PHASE_ID,
        "sealed_at_utc": _iso(_utc(sealed_at_utc, "sealed_at_utc")),
        "precommit_sha256": lock["precommit_sha256"],
        "phase_index_sha256": index["phase_index_sha256"],
        "day_score_sha256s": [score["day_score_sha256"] for score in sorted(validated_scores, key=lambda row: row["ordinal"])],
        "allocated_cell_count": EXPECTED_CELL_COUNT,
        "valid_response_cell_count": EXPECTED_CELL_COUNT - failure_count,
        "response_failure_cell_count": failure_count,
        "all_allocated_cells_terminal": True,
        "missing_failures_count_in_denominator": True,
        "truth_semantics": "FIXED_24H_DIRECTION_AND_SIZE_ONLY",
        "variant_summaries": variants,
        "paired_contrasts": {
            "complete_day_count": sum(row["inference_eligible"] for row in paired),
            "allocated_day_count": EXPECTED_DAY_COUNT,
            "b_minus_a_total_log_growth_all_days": sum(row["b_minus_a_log_growth"] for row in paired),
            "c_minus_a_total_log_growth_all_days": sum(row["c_minus_a_log_growth"] for row in paired),
            "confirmatory_inference_allowed": False,
            "selection_status": "NO_SELECTION_DIAGNOSTIC_ONLY",
        },
        "cell_results": terminal_rows,
        "paired_day_results": paired,
        "best_calendar_30d_multiple": best_monthly,
        "goal_status": (
            "3X_DIAGNOSTIC_THRESHOLD_OBSERVED_NOT_PROVEN"
            if best_monthly >= 3.0
            else "3X_NOT_REACHABLE"
        ),
        "prompt_selection_allowed": False,
        "positive_superiority_claim_allowed": False,
        "effective_independent_n": 0,
        "evidence_tier": DIAGNOSTIC_TIER,
        "authority": _authority(),
    }
    return _seal(body, "phase_score_sha256")


def validate_phase_score(
    value: Mapping[str, Any],
    precommit: Mapping[str, Any],
    phase_index: Mapping[str, Any],
    day_seals: Sequence[Mapping[str, Any]],
    day_scores: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    score = _mapping(value, "AI phase score")
    _validate_seal(score, "phase_score_sha256")
    expected = build_phase_score(
        precommit,
        phase_index,
        day_seals,
        day_scores,
        sealed_at_utc=_parse_utc(score.get("sealed_at_utc"), "sealed_at_utc"),
    )
    if score != expected:
        raise DojoAITruthError("AI phase score is not derived from exact day scores")
    return _snapshot(score)


def _normalize_truth_response(
    value: Any, *, start: datetime, end: datetime
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    response = _mapping(value, "OANDA truth response")
    if set(response) != {"instrument", "granularity", "candles"}:
        raise DojoAITruthError("OANDA truth response schema is not exact")
    if response["instrument"] != PAIR or response["granularity"] != SOURCE_GRANULARITY:
        raise DojoAITruthError("OANDA truth response identity drifted")
    raw_candles = response["candles"]
    if isinstance(raw_candles, (str, bytes)) or not isinstance(raw_candles, Sequence):
        raise DojoAITruthError("OANDA truth candles must be a sequence")
    expected = expected_oanda_fx_slots(start, end, step=timedelta(minutes=5))
    expected_set = set(expected)
    candles: list[dict[str, Any]] = []
    actual: list[datetime] = []
    for index, raw in enumerate(raw_candles):
        candle = _mapping(raw, f"truth candle {index}")
        if set(candle) != {"complete", "volume", "time", "bid", "ask"}:
            raise DojoAITruthError("truth candle schema is not exact BA")
        if candle["complete"] is not True:
            raise DojoAITruthError("truth candle is incomplete")
        opened = _parse_oanda_time(candle["time"])
        if opened not in expected_set or (actual and opened <= actual[-1]):
            raise DojoAITruthError("truth candle is off-grid, duplicated, or unsorted")
        bid = _ohlc(candle["bid"], "bid")
        ask = _ohlc(candle["ask"], "ask")
        if any(Decimal(ask[key]) < Decimal(bid[key]) for key in ("o", "h", "l", "c")):
            raise DojoAITruthError("truth ask is below bid")
        if isinstance(candle["volume"], bool) or not isinstance(candle["volume"], int) or candle["volume"] < 0:
            raise DojoAITruthError("truth candle volume is invalid")
        actual.append(opened)
        candles.append({"time": _iso(opened), "bid": bid, "ask": ask, "volume": candle["volume"]})
    required = (len(expected) * M5_MINIMUM_COVERAGE[0] + M5_MINIMUM_COVERAGE[1] - 1) // M5_MINIMUM_COVERAGE[1]
    if len(candles) < required:
        raise DojoAITruthError("truth response is below fixed coverage")
    if not actual or actual[0] != expected[0] or actual[-1] != expected[-1]:
        raise DojoAITruthError("truth response lacks exact entry or exit boundary")
    max_gap = max((right - left for left, right in zip(actual, actual[1:])), default=timedelta(0))
    if max_gap > M5_MAX_CONTIGUOUS_GAP:
        raise DojoAITruthError("truth response exceeds fixed gap ceiling")
    actual_set = set(actual)
    missing = [_iso(slot) for slot in expected if slot not in actual_set]
    coverage = {
        "coverage_policy": TRUTH_COVERAGE_POLICY,
        "expected_open_slot_count": len(expected),
        "returned_slot_count": len(actual),
        "required_returned_slot_count": required,
        "missing_slot_count": len(missing),
        "missing_slots_sha256": canonical_sha256(missing),
        "exact_entry_boundary_present": True,
        "exact_exit_boundary_present": True,
        "max_observed_gap_seconds": int(max_gap.total_seconds()),
        "coverage_passed": True,
    }
    return candles, coverage


def _fixed_horizon_returns(
    candles: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, float], dict[str, Any]]:
    first = candles[0]
    last = candles[-1]
    pip = Decimal("0.01")
    slip = pip * TRUTH_SLIPPAGE_PIPS_PER_FILL
    financing = pip * TRUTH_FINANCING_PIPS_PER_DAY
    long_entry = Decimal(first["ask"]["o"]) + slip
    long_exit = Decimal(last["bid"]["c"]) - slip - financing
    short_entry = Decimal(first["bid"]["o"]) - slip
    short_exit = Decimal(last["ask"]["c"]) + slip + financing
    long_full = ((long_exit - long_entry) / long_entry) * TRUTH_NOTIONAL_MULTIPLE
    short_full = ((short_entry - short_exit) / short_entry) * TRUTH_NOTIONAL_MULTIPLE
    returns = {
        "FLAT": 0.0,
        "LONG_HALF": _float12(long_full / 2),
        "LONG_FULL": _float12(long_full),
        "SHORT_HALF": _float12(short_full / 2),
        "SHORT_FULL": _float12(short_full),
    }
    if any(value <= -1 for value in returns.values()):
        raise DojoAITruthError("derived return destroys the account multiplier")
    evidence = {
        "entry_candle_open_utc": first["time"],
        "exit_candle_open_utc": last["time"],
        "entry_executable_bid": first["bid"]["o"],
        "entry_executable_ask": first["ask"]["o"],
        "exit_executable_bid": last["bid"]["c"],
        "exit_executable_ask": last["ask"]["c"],
        "slippage_pips_per_fill": float(TRUTH_SLIPPAGE_PIPS_PER_FILL),
        "financing_pips_per_day": float(TRUTH_FINANCING_PIPS_PER_DAY),
        "notional_multiple": float(TRUTH_NOTIONAL_MULTIPLE),
        "spread_included_at_entry_and_exit": True,
    }
    return returns, evidence


def _variant_summary(
    rows: Sequence[Mapping[str, Any]], variant: str, precommit: Mapping[str, Any]
) -> dict[str, Any]:
    selected = [row for row in rows if row["variant_id"] == variant]
    total_log = sum(float(row["net_log_growth"]) for row in selected)
    failures = sum(bool(row["response_failure"]) for row in selected)
    first = _parse_utc(precommit["schedule"][0]["decision_cutoff_utc"], "first cutoff")
    last = _parse_utc(precommit["schedule"][-1]["truth_not_before_utc"], "last truth")
    calendar_days = (last - first).total_seconds() / 86_400
    return {
        "allocated_cell_count": len(selected),
        "valid_response_cell_count": len(selected) - failures,
        "response_failure_cell_count": failures,
        "genuine_flat_count": sum(row["status"] == "VALID_RESPONSE" and row["return_key"] == "FLAT" for row in selected),
        "synthetic_flat_failure_count": failures,
        "total_log_growth": total_log,
        "compounded_net_return": math.expm1(total_log),
        "calendar_days": calendar_days,
        "calendar_30d_multiple": math.exp(total_log * 30.0 / calendar_days),
        "diagnostic_only": True,
    }


def _failure_row(
    day: Mapping[str, Any],
    cell: Mapping[str, Any],
    terminal: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        "ordinal": day["ordinal"],
        "blind_day_rank": day["schedule"]["blind_day_rank"],
        "blind_day_id": day["schedule"]["blind_day_id"],
        "variant_id": cell["variant_id"],
        "cell_id": terminal["cell_id"],
        "terminal_state": terminal["state"],
        "terminal_artifact_sha256": terminal["cell_terminal_sha256"],
        "status": reason,
        "response_failure": True,
        "economic_fallback": "SYNTHETIC_FLAT_ZERO_RETURN",
        "return_key": None,
        "net_return": 0.0,
        "net_log_growth": 0.0,
        "score_receipt": None,
        "score_receipt_sha256": None,
        "declared_model_lineage": None,
    }


def _validate_phase_day_score_shape(
    value: Mapping[str, Any],
    days: Mapping[int, Mapping[str, Any]],
    precommit: Mapping[str, Any],
) -> dict[str, Any]:
    score = _mapping(value, "phase day score")
    _validate_seal(score, "day_score_sha256")
    ordinal = score.get("ordinal")
    day = days.get(ordinal) if isinstance(ordinal, int) and not isinstance(ordinal, bool) else None
    if (
        day is None
        or day.get("state") != "REQUESTS_SEALED"
        or score.get("contract") != DAY_SCORE_CONTRACT
        or score.get("validity_status") != "VALID"
        or score.get("precommit_sha256") != precommit["precommit_sha256"]
        or score.get("day_seal_sha256") != day["day_seal_sha256"]
        or score.get("authority") != _authority()
    ):
        raise DojoAITruthError("phase day score identity or parent drifted")
    rows = score.get("cell_results")
    if not isinstance(rows, list) or len(rows) != 3:
        raise DojoAITruthError("phase day score must contain exactly three cells")
    cells = {cell["assignment"]["cell_id"]: cell for cell in day["cells"]}
    seen: set[str] = set()
    for row in rows:
        cell_id = row.get("cell_id")
        cell = cells.get(cell_id)
        if cell is None or cell_id in seen:
            raise DojoAITruthError("phase day score cell is unknown or duplicated")
        seen.add(cell_id)
        if (
            row.get("ordinal") != ordinal
            or row.get("blind_day_rank") != day["schedule"]["blind_day_rank"]
            or row.get("blind_day_id") != day["schedule"]["blind_day_id"]
            or row.get("variant_id") != cell["variant_id"]
        ):
            raise DojoAITruthError("phase day score cell assignment drifted")
        if row.get("response_failure") is True:
            if (
                row.get("net_return") != 0.0
                or row.get("net_log_growth") != 0.0
                or row.get("score_receipt") is not None
                or row.get("score_receipt_sha256") is not None
            ):
                raise DojoAITruthError("failed phase cell is not canonical zero-return")
            continue
        receipt = row.get("score_receipt")
        if not isinstance(receipt, Mapping):
            raise DojoAITruthError("valid phase cell lacks a score receipt")
        validated = validate_score_receipt(receipt)
        if (
            validated["validity_status"] != "VALID"
            or validated["packet_sha256"] != cell["packet"]["packet_sha256"]
            or validated["prompt_sha256"] != cell["assignment"]["prompt_sha256"]
            or validated["model_sha256"] != cell["assignment"]["model_sha256"]
            or row.get("return_key") != validated["return_key"]
            or row.get("net_return") != validated["net_return"]
            or row.get("net_log_growth") != validated["net_log_growth"]
            or row.get("score_receipt_sha256")
            != validated["score_receipt_sha256"]
        ):
            raise DojoAITruthError("phase cell result differs from its score receipt")
    if seen != set(cells):
        raise DojoAITruthError("phase day score cell set is incomplete")
    return score


def _validated_day(day_seal: Mapping[str, Any], precommit: Mapping[str, Any]) -> dict[str, Any]:
    day = _mapping(day_seal, "AI day seal")
    _validate_external_seal(day, "day_seal_sha256")
    if day.get("precommit_sha256") != precommit["precommit_sha256"] or day.get("state") != "REQUESTS_SEALED":
        raise DojoAITruthError("truth requires one request-sealed day")
    return _snapshot(day)


def _validated_terminal_set(
    day: Mapping[str, Any], terminals: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    if isinstance(terminals, (str, bytes)) or not isinstance(terminals, Sequence):
        raise DojoAITruthError("truth terminals must be a sequence")
    expected = {cell["cell_id"] for cell in day["schedule"]["cells"]}
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in terminals:
        terminal = _mapping(raw, "cell terminal")
        _validate_external_seal(terminal, "cell_terminal_sha256")
        cell_id = terminal.get("cell_id")
        if cell_id not in expected or cell_id in seen:
            raise DojoAITruthError("truth terminal is unknown or duplicated")
        if terminal.get("day_seal_sha256") != day["day_seal_sha256"] or terminal.get("state") not in {"RESPONSE_SEALED", "MISSING_RESPONSE_DEADLINE"}:
            raise DojoAITruthError("truth terminal parent or state drifted")
        if terminal.get("answer_key_opened") is not False:
            raise DojoAITruthError("truth terminal claims premature answer-key access")
        seen.add(cell_id)
        result.append(_snapshot(terminal))
    if seen != expected:
        raise DojoAITruthError("truth requires all three response terminals")
    result.sort(key=lambda row: row["variant_id"])
    return result


def _find_cell(day: Mapping[str, Any], cell_id: str) -> dict[str, Any]:
    matches = [cell for cell in day["cells"] if cell["assignment"]["cell_id"] == cell_id]
    if len(matches) != 1:
        raise DojoAITruthError("day cell material is not unique")
    return _snapshot(matches[0])


def _ohlc(value: Any, label: str) -> dict[str, str]:
    source = _mapping(value, f"truth {label}")
    if set(source) != {"o", "h", "l", "c"}:
        raise DojoAITruthError("truth OHLC schema is not exact")
    parsed: dict[str, Decimal] = {}
    result: dict[str, str] = {}
    for key, raw in source.items():
        if not isinstance(raw, str) or _DECIMAL.fullmatch(raw) is None:
            raise DojoAITruthError("truth OHLC must be plain decimal strings")
        try:
            number = Decimal(raw)
        except InvalidOperation as exc:
            raise DojoAITruthError("truth OHLC decimal is invalid") from exc
        if not number.is_finite() or number <= 0:
            raise DojoAITruthError("truth OHLC must be finite and positive")
        parsed[key] = number
        result[key] = raw
    if parsed["h"] < max(parsed["o"], parsed["l"], parsed["c"]) or parsed["l"] > min(parsed["o"], parsed["h"], parsed["c"]):
        raise DojoAITruthError("truth OHLC geometry is invalid")
    return result


def _parse_oanda_time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise DojoAITruthError("truth candle time must be text")
    match = _TIME.fullmatch(value)
    if match is None:
        raise DojoAITruthError("truth candle time format is invalid")
    return datetime.fromisoformat(match.group(1) + "+00:00").astimezone(timezone.utc)


def _float12(value: Decimal) -> float:
    result = float(value.quantize(Decimal("0.000000000001")))
    if not math.isfinite(result):
        raise DojoAITruthError("derived return is non-finite")
    return result


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DojoAITruthError(f"{label} must be an object")
    return _snapshot(value)


def _snapshot(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise DojoAITruthError("artifact must be finite JSON") from exc


def _seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    body = _snapshot(value)
    return {**body, field: canonical_sha256(body)}


def _validate_seal(value: Mapping[str, Any], field: str) -> None:
    claimed = value.get(field)
    if not isinstance(claimed, str) or claimed != canonical_sha256({key: item for key, item in value.items() if key != field}):
        raise DojoAITruthError(f"{field} digest mismatch")


def _validate_external_seal(value: Mapping[str, Any], field: str) -> None:
    _validate_seal(value, field)


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise DojoAITruthError(f"{label} must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DojoAITruthError(f"{label} is invalid") from exc
    return _utc(parsed, label)


def _utc(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise DojoAITruthError(f"{label} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return _utc(value, "timestamp").isoformat().replace("+00:00", "Z")


def _authority() -> dict[str, Any]:
    return {
        "read_only": True,
        "ai_order_authority": "NONE",
        "broker_mutation_allowed": False,
        "live_permission": False,
        "positive_result_grants_live_permission": False,
    }
