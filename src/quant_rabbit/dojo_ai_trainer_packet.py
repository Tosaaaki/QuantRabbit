"""Compact, fail-closed evidence packet for the DOJO AI bot trainer.

The packet is a reducer, not a scorer.  It accepts only a terminal worn-TRAIN
run whose evaluation has already been bound to the authoritative local
candidate-lineage registry.  It exposes every candidate and every fixed-grid
cell, while deliberately omitting session paths and raw-ledger references.

The local lineage has no external monotonic witness.  Consequently this packet
is research input only and can never grant proof, promotion, live-order, or
broker-mutation authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Final

from quant_rabbit.dojo_ai_tuning_state import (
    MAX_ATTEMPTS,
    MAX_PROPOSAL_SLOTS,
    status_artifact as tuning_status_artifact,
    verify_tuning_state,
)
from quant_rabbit.dojo_bot_catalog import catalog_manifest
from quant_rabbit.dojo_bot_trainer import (
    CELL_CONTRACT,
    EVALUATION_CONTRACT,
    REQUIRED_COST_ARMS,
    REQUIRED_INTRABAR_PATHS,
    _score_candidate,
    verify_sealed_study,
)
from quant_rabbit.dojo_candidate_lineage_registry import verify_registry

if TYPE_CHECKING:
    from quant_rabbit.dojo_drive_remote_evidence import _VerifiedDriveTrainerReadback


PACKET_CONTRACT: Final = "QR_DOJO_AI_TRAINER_PACKET_V1"
RUN_CONTRACT: Final = "QR_DOJO_BOT_TRAINER_RUN_V1"
SCHEMA_VERSION: Final = 1
MAX_BOUND_ARTIFACT_BYTES: Final = 64 * 1024 * 1024
TRAINER_READBACK_KINDS: Final = (
    "RUN",
    "EVALUATION",
    "CELLS",
    "SEALED_STUDY",
    "TERMINAL_HANDOFF",
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}\Z")
_DRIVE_ID = re.compile(r"[A-Za-z0-9_-]{8,256}\Z")

_AUTHORITY = {
    "proof_eligible": False,
    "promotion_eligible": False,
    "live_permission": False,
    "order_authority": "NONE",
    "broker_mutation_allowed": False,
}
_ALLOWED_MUTATIONS: Final = (
    "candidate.family",
    "candidate.hypothesis",
    "candidate.config.atr_floor_pips",
    "candidate.config.be_offset_pips",
    "candidate.config.be_trigger_atr",
    "candidate.config.ceiling_min",
    "candidate.config.eff_max",
    "candidate.config.exit_policy",
    "candidate.config.fade_atr",
    "candidate.config.global_max_concurrent",
    "candidate.config.max_concurrent_per_pair",
    "candidate.config.pairs",
    "candidate.config.per_pos_lev",
    "candidate.config.pull_atr",
    "candidate.config.session_buffer_atr",
    "candidate.config.session_sl_range",
    "candidate.config.session_tp_range",
    "candidate.config.signal",
    "candidate.config.sl_pips",
    "candidate.config.tp_atr",
    "candidate.config.tp_pips",
    "candidate.config.trail_distance_atr",
    "candidate.config.trail_trigger_atr",
    "candidate.config.weekend_gap_atr",
    "candidate.config.weekend_sl_gap",
    "candidate.config.weekend_spread_fraction",
    "candidate.config.weekend_wait_bars",
)
_FORBIDDEN_MUTATIONS: Final = (
    "artifact_or_session_path",
    "broker_state_or_order",
    "candidate.risk_increase",
    "cell_or_evaluation_metric",
    "code_import_command_or_plugin",
    "cost_arm_or_cost_parameters",
    "corpus_window_or_source_digest",
    "fixed_denominator_or_intrabar_paths",
    "holdout_or_forward_evidence",
    "lineage_event_or_result_binding",
    "live_permission_or_order_authority",
    "raw_ledger_or_archive_payload",
    "risk_or_evaluation_threshold",
    "search_budget_or_attempt_ordinal",
)
_EVALUATION_OBLIGATIONS: Final = (
    "ACCOUNT_FOR_EVERY_CANDIDATE_CELL_AND_FAILED_COORDINATE",
    "COMPARE_BASE_VS_STRESS_AND_OHLC_VS_OLHC",
    "TREAT_UNKNOWN_METRICS_AS_UNKNOWN_NOT_ZERO",
    "EXPLAIN_LOPO_DRAWDOWN_MARGIN_COST_AND_CAPITAL_PRODUCTIVITY",
    "PROPOSE_ONLY_CATALOG_VALID_MUTATIONS_WITHIN_THE_FIXED_ENVELOPE",
    "DO_NOT_INCREASE_RISK_OR_LEVERAGE_TO_CLOSE_THE_3X_TARGET_GAP",
    "DO_NOT_READ_HOLDOUT_FORWARD_LIVE_PATH_OR_RAW_LEDGER_INPUTS",
)
_LIMITATIONS: Final = (
    "WORN_HISTORICAL_TRAIN_DIAGNOSTIC_ONLY",
    "LOCAL_LINEAGE_HAS_NO_EXTERNAL_MONOTONIC_WITNESS",
    "RAW_LEDGERS_ARE_NOT_READ_OR_INCLUDED",
    "CELL_METRICS_ARE_RECONCILED_TO_BOUND_EVALUATION_NOT_RAW_LEDGER_REPLAYED",
    "DRIVE_EXACT_DOWNLOADED_TERMINAL_ARTIFACT_BYTES_VERIFIED",
    "DRIVE_CUSTODY_DEPENDS_ON_AUTHENTICATED_CONNECTOR_CALL_BOUNDARY",
    "PACKET_CANNOT_AUTHOR_OR_EXECUTE_TRADES",
)

_RUN_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "study_sha256",
        "status",
        "corpus",
        "fixed_denominator",
        "coordinates",
        "cells_path",
        "evaluation_path",
        "evaluation_sha256",
        "classification",
        *_AUTHORITY,
        "run_sha256",
    }
)
_RUN_DENOMINATOR_KEYS = frozenset(
    {
        "expected_cell_count",
        "observed_cell_count",
        "failed_cell_count",
        "dropped_cell_count",
        "coordinate_receipts_complete",
        "execution_success_complete",
    }
)
_CELL_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "study_sha256",
        "candidate_id",
        "proposal_sha256",
        "intrabar",
        "cost_arm",
        "metrics",
        "ledger_evidence",
        "execution_status",
        "failure_code",
        "cell_sha256",
    }
)
_CELL_METRIC_KEYS = frozenset(
    {
        "terminal_net_jpy",
        "terminal_flat",
        "margin_closeouts",
        "realized_max_drawdown_fraction",
        "mtm_complete",
        "mtm_max_drawdown_fraction",
        "peak_margin_usage_fraction",
        "fill_count",
        "margin_reject_count",
        "capital_lock_margin_jpy_hours",
        "pair_pnl_jpy",
        "leave_one_pair_out_net_jpy",
        "lopo_replay_complete",
    }
)
_COORDINATE_KEYS = frozenset(
    {
        "candidate_id",
        "intrabar",
        "cost_arm",
        "status",
        "main_session_dir",
        "main_error",
        "lopo_replay_complete",
        "lopo",
        "cell_sha256",
    }
)
_DRIVE_REF_KEYS = frozenset(
    {
        "artifact_kind",
        "drive_file_id",
        "drive_parent_id",
        "content_sha256",
        "content_size_bytes",
        "version",
        "head_revision_id",
        "readback_sha256",
        "remote_verified",
    }
)
_PACKET_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "classification",
        "source_bindings",
        "fixed_environment",
        "search_budget",
        "mutation_policy",
        "evaluation_obligations",
        "current_run",
        "candidates",
        "cells",
        "base_stress_comparisons",
        "ohlc_olhc_comparisons",
        "failed_coordinates",
        "previous_attempts",
        "drive_evidence_refs",
        "limitations",
        *_AUTHORITY,
        "packet_sha256",
    }
)


class DojoAITrainerPacketError(ValueError):
    """The requested trainer packet is incomplete, unbound, or unsafe."""


def canonical_packet_bytes(value: Any) -> bytes:
    """Return finite, deterministic canonical JSON bytes."""

    _validate_json(value, "value")
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DojoAITrainerPacketError("value is not canonical JSON") from exc


def canonical_packet_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_packet_bytes(value)).hexdigest()


def validate_terminal_result_bundle(
    *,
    run: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    sealed_study: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Validate one complete terminal trainer bundle without lineage side effects.

    This is the public hand-off boundary used before a ``RESULT_BOUND`` event is
    appended.  In particular, an evaluation cannot be admitted by itself: the
    terminal run receipt, the complete fixed-denominator cell array, and the
    exact sealed study are all mandatory.  The evaluation is rebuilt from the
    normalized cells and must match byte-for-byte at the JSON value level.

    The function is deliberately pure.  It reads no paths, writes no lineage,
    and grants no proof, promotion, model, runner, broker, or live authority.
    """

    try:
        verified_study = verify_sealed_study(
            sealed_study, sealed_study["source_digests"]
        )
    except (KeyError, ValueError) as exc:
        raise DojoAITrainerPacketError("terminal sealed study is invalid") from exc
    normalized_run, normalized_cells = _validate_terminal_inputs(
        run=run,
        evaluation=evaluation,
        cells=cells,
        sealed_study=verified_study,
    )
    normalized_evaluation = _json_copy(evaluation)
    rebuilt_evaluation = _rebuild_evaluation_from_cells(
        verified_study, normalized_cells
    )
    if rebuilt_evaluation != normalized_evaluation:
        raise DojoAITrainerPacketError(
            "cells do not reconstruct the exact lineage-bound evaluation"
        )
    return normalized_run, normalized_evaluation, normalized_cells


def build_trainer_packet(
    *,
    run: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    lineage_events_dir: Path,
    artifact_root: Path,
    tuning_state: Mapping[str, Any],
    verified_drive_readback: "_VerifiedDriveTrainerReadback",
) -> dict[str, Any]:
    """Reduce one exact terminal TRAIN result into bounded AI input.

    ``ledger_evidence`` fields are authenticated by their enclosing cell seal
    and then ignored.  This function never opens a session or ledger path.
    """

    lineage = verify_registry(
        Path(lineage_events_dir), artifact_root=Path(artifact_root)
    )
    if not lineage.results or len(lineage.studies) != len(lineage.results):
        raise DojoAITrainerPacketError(
            "lineage does not have one terminal result per study"
        )

    verified_state = verify_tuning_state(tuning_state)
    tuning_status = tuning_status_artifact(verified_state)
    if tuning_status["phase"] not in {"READY_FOR_MODEL", "EXHAUSTED"}:
        raise DojoAITrainerPacketError(
            "tuning status is not at a terminal result boundary"
        )
    _verify_authority(verified_state, "tuning state")
    if (
        verified_state["registry_id"] != lineage.registry_id
        or verified_state["lineage_prefix"] != lineage.lineage_prefix
    ):
        raise DojoAITrainerPacketError("tuning state belongs to another lineage")

    loaded_studies: list[dict[str, Any]] = []
    loaded_evaluations: list[dict[str, Any]] = []
    for study_binding, result_binding in zip(
        lineage.studies, lineage.results, strict=True
    ):
        sealed = _load_bound_json(
            Path(artifact_root),
            study_binding["study_artifact_relpath"],
            expected_sha256=study_binding["study_artifact_sha256"],
            expected_size=study_binding["study_artifact_size_bytes"],
        )
        try:
            sealed = verify_sealed_study(sealed, sealed["source_digests"])
        except (KeyError, ValueError) as exc:
            raise DojoAITrainerPacketError("lineage-bound study is invalid") from exc
        bound_evaluation = _load_bound_json(
            Path(artifact_root),
            result_binding["evaluation_artifact_relpath"],
            expected_sha256=result_binding["evaluation_artifact_sha256"],
            expected_size=result_binding["evaluation_artifact_size_bytes"],
        )
        loaded_studies.append(sealed)
        loaded_evaluations.append(bound_evaluation)

    current_study = loaded_studies[-1]
    current_evaluation = loaded_evaluations[-1]
    if _json_copy(evaluation) != current_evaluation:
        raise DojoAITrainerPacketError(
            "evaluation is not the exact latest bound result"
        )
    latest_result = lineage.results[-1]
    latest_result_event = next(
        event
        for event in reversed(lineage.events)
        if event["event_type"] == "RESULT_BOUND"
    )
    expected_result_binding = {
        "registry_id": lineage.registry_id,
        "lineage_prefix": lineage.lineage_prefix,
        "attempt_ordinal": latest_result["attempt_ordinal"],
        "study_sha256": latest_result["study_sha256"],
        "evaluation_sha256": latest_result["evaluation_sha256"],
        "evaluation_artifact_sha256": latest_result["evaluation_artifact_sha256"],
        "evaluation_artifact_size_bytes": latest_result[
            "evaluation_artifact_size_bytes"
        ],
        "result_event_sha256": latest_result_event["event_sha256"],
        "result_event_sequence": latest_result_event["sequence"],
        "lineage_tip_sha256": lineage.latest_event_sha256,
    }
    if verified_state["last_terminal_result_binding"] != expected_result_binding:
        raise DojoAITrainerPacketError(
            "tuning state does not bind the latest exact result"
        )

    normalized_run, current_evaluation, normalized_cells = (
        validate_terminal_result_bundle(
            run=run,
            evaluation=current_evaluation,
            cells=cells,
            sealed_study=current_study,
        )
    )
    from quant_rabbit.dojo_drive_remote_evidence import (
        DojoDriveRemoteEvidenceError,
        _trainer_packet_drive_evidence_refs,
    )

    try:
        normalized_drive_refs = _normalize_drive_refs(
            _trainer_packet_drive_evidence_refs(
                verified_drive_readback,
                expected_run_sha256=normalized_run["run_sha256"],
                expected_study_sha256=current_study["study_sha256"],
                expected_evaluation_sha256=current_evaluation["evaluation_sha256"],
                expected_lineage_tip_sha256=lineage.latest_event_sha256,
                expected_cell_count=len(normalized_cells),
            )
        )
    except DojoDriveRemoteEvidenceError as exc:
        raise DojoAITrainerPacketError(
            f"typed Drive readback is invalid: {exc}"
        ) from exc
    candidate_rows = _current_candidate_rows(
        current_study, current_evaluation, normalized_cells
    )
    reduced_cells = [_reduce_cell(cell) for cell in normalized_cells]
    failed_coordinates = sorted(
        {
            failed
            for candidate in current_evaluation["candidate_evaluations"]
            for failed in candidate["failed_coordinates"]
        }
    )
    history = [
        _reduce_previous_attempt(study, result, bound_evaluation)
        for study, result, bound_evaluation in zip(
            loaded_studies[:-1],
            lineage.results[:-1],
            loaded_evaluations[:-1],
            strict=True,
        )
    ]

    attempts_consumed = tuning_status["attempts_consumed"]
    proposal_slots_consumed = tuning_status["proposal_slots_consumed"]
    if attempts_consumed != len(lineage.studies):
        raise DojoAITrainerPacketError(
            "tuning attempt budget diverges from terminal lineage"
        )
    lineage_candidate_count = sum(item["candidate_count"] for item in lineage.studies)
    if proposal_slots_consumed < lineage_candidate_count:
        raise DojoAITrainerPacketError(
            "tuning proposal budget omits lineage candidates"
        )

    study = current_study["study"]
    body = {
        "contract": PACKET_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "classification": "WORN_HISTORICAL_TRAIN_DIAGNOSTIC_ONLY",
        "source_bindings": {
            "registry_id": lineage.registry_id,
            "lineage_prefix": lineage.lineage_prefix,
            "attempt_ordinal": latest_result["attempt_ordinal"],
            "study_sha256": current_study["study_sha256"],
            "run_sha256": normalized_run["run_sha256"],
            "evaluation_sha256": current_evaluation["evaluation_sha256"],
            "evaluation_artifact_sha256": latest_result["evaluation_artifact_sha256"],
            "lineage_result_event_sha256": latest_result_event["event_sha256"],
            "lineage_tip_sha256": lineage.latest_event_sha256,
            "tuning_state_sha256": verified_state["state_sha256"],
            "fixed_envelope_sha256": tuning_status["fixed_envelope_sha256"],
            "external_witness_status": lineage.external_witness_status,
            "exact_result_binding_verified": True,
        },
        "fixed_environment": {
            "window_role": study["window_role"],
            "window": _json_copy(study["window"]),
            "initial_balance_jpy": study["initial_balance_jpy"],
            "trade_pairs": list(study["trade_pairs"]),
            "feed_pairs": list(study["feed_pairs"]),
            "intrabar_paths": list(REQUIRED_INTRABAR_PATHS),
            "cost_arms": _json_copy(study["cost_arms"]),
            "thresholds": _json_copy(study["thresholds"]),
            "catalog": catalog_manifest(),
        },
        "search_budget": {
            "phase": tuning_status["phase"],
            "attempts_consumed": attempts_consumed,
            "attempts_remaining": MAX_ATTEMPTS - attempts_consumed,
            "max_attempts": MAX_ATTEMPTS,
            "proposal_slots_consumed": proposal_slots_consumed,
            "proposal_slots_remaining": MAX_PROPOSAL_SLOTS - proposal_slots_consumed,
            "max_proposal_slots": MAX_PROPOSAL_SLOTS,
            "invalid_proposal_count": tuning_status["invalid_proposal_count"],
            "duplicate_proposal_count": tuning_status["duplicate_proposal_count"],
        },
        "mutation_policy": {
            "allowed": list(_ALLOWED_MUTATIONS),
            "forbidden": list(_FORBIDDEN_MUTATIONS),
            "risk_increase_required_value": False,
            "catalog_validation_required": True,
            "fixed_envelope_mutation_allowed": False,
        },
        "evaluation_obligations": list(_EVALUATION_OBLIGATIONS),
        "current_run": {
            "status": normalized_run["status"],
            "fixed_denominator": _json_copy(normalized_run["fixed_denominator"]),
            "candidate_ids": [row["candidate_id"] for row in candidate_rows],
            "diagnostic_ranking": list(current_evaluation["diagnostic_ranking"]),
            "rank_eligible_candidate_ids": list(
                current_evaluation["rank_eligible_candidate_ids"]
            ),
            "unranked_candidate_ids": list(
                current_evaluation["unranked_candidate_ids"]
            ),
        },
        "candidates": candidate_rows,
        "cells": reduced_cells,
        "base_stress_comparisons": _base_stress_comparisons(normalized_cells),
        "ohlc_olhc_comparisons": _intrabar_comparisons(normalized_cells),
        "failed_coordinates": failed_coordinates,
        "previous_attempts": history,
        "drive_evidence_refs": normalized_drive_refs,
        "limitations": list(_LIMITATIONS),
        **_AUTHORITY,
    }
    _reject_reference_surface(body)
    packet = {**body, "packet_sha256": canonical_packet_sha256(body)}
    return verify_trainer_packet(packet)


def verify_trainer_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the canonical seal and immutable research-only packet surface."""

    row = _exact_mapping(packet, _PACKET_KEYS, "trainer packet")
    if row["contract"] != PACKET_CONTRACT or row["schema_version"] != SCHEMA_VERSION:
        raise DojoAITrainerPacketError("trainer packet contract/version drifted")
    claimed = _sha(row["packet_sha256"], "packet_sha256")
    body = {key: value for key, value in row.items() if key != "packet_sha256"}
    if canonical_packet_sha256(body) != claimed:
        raise DojoAITrainerPacketError("trainer packet SHA-256 mismatch")
    if row["classification"] != "WORN_HISTORICAL_TRAIN_DIAGNOSTIC_ONLY":
        raise DojoAITrainerPacketError("trainer packet is not worn TRAIN evidence")
    _verify_authority(row, "trainer packet")
    policy = row["mutation_policy"]
    if policy != {
        "allowed": list(_ALLOWED_MUTATIONS),
        "forbidden": list(_FORBIDDEN_MUTATIONS),
        "risk_increase_required_value": False,
        "catalog_validation_required": True,
        "fixed_envelope_mutation_allowed": False,
    }:
        raise DojoAITrainerPacketError("trainer mutation policy drifted")
    if row["evaluation_obligations"] != list(_EVALUATION_OBLIGATIONS):
        raise DojoAITrainerPacketError("trainer evaluation obligations drifted")
    if row["limitations"] != list(_LIMITATIONS):
        raise DojoAITrainerPacketError("trainer packet limitations drifted")
    refs = _normalize_drive_refs(row["drive_evidence_refs"])
    if refs != row["drive_evidence_refs"]:
        raise DojoAITrainerPacketError("Drive evidence references are not canonical")
    candidates = row["candidates"]
    cells = row["cells"]
    if not isinstance(candidates, list) or not candidates:
        raise DojoAITrainerPacketError("trainer packet has no candidates")
    if not isinstance(cells, list) or not cells:
        raise DojoAITrainerPacketError("trainer packet has no cells")
    candidate_ids = [
        item.get("candidate_id") for item in candidates if isinstance(item, Mapping)
    ]
    if len(candidate_ids) != len(candidates) or len(set(candidate_ids)) != len(
        candidate_ids
    ):
        raise DojoAITrainerPacketError(
            "trainer packet candidate denominator is invalid"
        )
    expected = row["current_run"]["fixed_denominator"]["expected_cell_count"]
    if len(cells) != expected:
        raise DojoAITrainerPacketError("trainer packet cell denominator is partial")
    expected_coordinates = {
        (candidate_id, path, arm)
        for candidate_id in candidate_ids
        for path in REQUIRED_INTRABAR_PATHS
        for arm in REQUIRED_COST_ARMS
    }
    actual_coordinates = {
        (cell.get("candidate_id"), cell.get("intrabar"), cell.get("cost_arm"))
        for cell in cells
        if isinstance(cell, Mapping)
    }
    if actual_coordinates != expected_coordinates or len(actual_coordinates) != len(
        cells
    ):
        raise DojoAITrainerPacketError("trainer packet fixed cell grid is incomplete")
    _reject_reference_surface(body)
    return _json_copy(row)


def _validate_terminal_inputs(
    *,
    run: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    sealed_study: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    normalized_run = _exact_mapping(run, _RUN_KEYS, "terminal run")
    _verify_seal(normalized_run, "run_sha256", "terminal run")
    if (
        normalized_run["contract"] != RUN_CONTRACT
        or normalized_run["schema_version"] != 1
    ):
        raise DojoAITrainerPacketError("terminal run contract/version drifted")
    if normalized_run["status"] not in {"COMPLETE", "COMPLETE_WITH_FAILED_CELLS"}:
        raise DojoAITrainerPacketError("partial run cannot become trainer input")
    if normalized_run["classification"] != "WORN_HISTORICAL_TRAIN_DIAGNOSTIC_ONLY":
        raise DojoAITrainerPacketError("holdout or forward run is forbidden")
    _verify_authority(normalized_run, "terminal run")
    if normalized_run["study_sha256"] != sealed_study["study_sha256"]:
        raise DojoAITrainerPacketError("terminal run belongs to another study")
    if normalized_run["evaluation_sha256"] != evaluation["evaluation_sha256"]:
        raise DojoAITrainerPacketError("terminal run does not bind the evaluation")
    if evaluation["contract"] != EVALUATION_CONTRACT:
        raise DojoAITrainerPacketError("unsupported trainer evaluation")
    if evaluation["classification"] != "WORN_HISTORICAL_TRAIN_DIAGNOSTIC_ONLY":
        raise DojoAITrainerPacketError("holdout or forward evaluation is forbidden")
    _verify_authority(evaluation, "evaluation")

    denominator = _exact_mapping(
        normalized_run["fixed_denominator"],
        _RUN_DENOMINATOR_KEYS,
        "run fixed denominator",
    )
    expected = _integer(denominator["expected_cell_count"], "expected_cell_count")
    observed = _integer(denominator["observed_cell_count"], "observed_cell_count")
    failed = _integer(denominator["failed_cell_count"], "failed_cell_count")
    if (
        expected <= 0
        or observed != expected
        or not 0 <= failed <= expected
        or denominator["dropped_cell_count"] != 0
        or denominator["coordinate_receipts_complete"] is not True
        or denominator["execution_success_complete"] is not (failed == 0)
        or (normalized_run["status"] == "COMPLETE") is not (failed == 0)
    ):
        raise DojoAITrainerPacketError("run fixed denominator is incomplete")
    if not isinstance(cells, Sequence) or isinstance(cells, (str, bytes, bytearray)):
        raise DojoAITrainerPacketError("cells must be a JSON array")
    if len(cells) != expected:
        raise DojoAITrainerPacketError("partial or best-only cell input is forbidden")

    study = sealed_study["study"]
    candidates = {item["candidate_id"]: item for item in study["candidates"]}
    expected_grid = {
        (candidate_id, path, arm)
        for candidate_id in candidates
        for path in REQUIRED_INTRABAR_PATHS
        for arm in REQUIRED_COST_ARMS
    }
    if expected != len(expected_grid):
        raise DojoAITrainerPacketError(
            "run denominator differs from sealed candidate grid"
        )
    normalized_cells = [
        _normalize_cell(cell, sealed_study=sealed_study) for cell in cells
    ]
    cell_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    for cell in normalized_cells:
        identity = (cell["candidate_id"], cell["intrabar"], cell["cost_arm"])
        if identity in cell_map:
            raise DojoAITrainerPacketError("duplicate trainer cell coordinate")
        cell_map[identity] = cell
    if set(cell_map) != expected_grid:
        raise DojoAITrainerPacketError("trainer cell grid is incomplete")

    evaluation_ids = [
        row["candidate_id"] for row in evaluation["candidate_evaluations"]
    ]
    if set(evaluation_ids) != set(candidates) or len(evaluation_ids) != len(candidates):
        raise DojoAITrainerPacketError(
            "partial or best-only candidate evaluation is forbidden"
        )
    expected_failed = {
        (candidate_id, *coordinate.split(":", 2))
        for candidate_id, candidate in (
            (row["candidate_id"], row) for row in evaluation["candidate_evaluations"]
        )
        for coordinate in candidate["failed_coordinates"]
    }
    actual_failed = {
        (
            cell["candidate_id"],
            cell["intrabar"],
            cell["cost_arm"],
            cell["failure_code"],
        )
        for cell in normalized_cells
        if cell["execution_status"] == "FAILED"
    }
    if expected_failed != actual_failed or len(actual_failed) != failed:
        raise DojoAITrainerPacketError(
            "failed coordinates diverge from bound evaluation"
        )

    coordinates = normalized_run["coordinates"]
    if not isinstance(coordinates, list) or len(coordinates) != expected:
        raise DojoAITrainerPacketError("run coordinate receipts are incomplete")
    seen: set[tuple[str, str, str]] = set()
    for index, raw_coordinate in enumerate(coordinates):
        coordinate = _exact_mapping(
            raw_coordinate, _COORDINATE_KEYS, f"run coordinate {index}"
        )
        identity = (
            _identifier(coordinate["candidate_id"], "coordinate candidate_id"),
            coordinate["intrabar"],
            coordinate["cost_arm"],
        )
        if identity not in cell_map or identity in seen:
            raise DojoAITrainerPacketError(
                "run coordinate grid is duplicated or foreign"
            )
        seen.add(identity)
        cell = cell_map[identity]
        if coordinate["cell_sha256"] != cell["cell_sha256"]:
            raise DojoAITrainerPacketError("run coordinate does not bind its cell")
        status = coordinate["status"]
        if status not in {
            "COMPLETE",
            "MAIN_REPLAY_FAILED_SENTINEL",
            "LOPO_INCOMPLETE_NO_ADDITIVE_SUBSTITUTE",
        }:
            raise DojoAITrainerPacketError("run coordinate status is unsupported")
        lopo = coordinate["lopo"]
        if not isinstance(lopo, list):
            raise DojoAITrainerPacketError("run coordinate LOPO must be an array")
        if status == "COMPLETE":
            if (
                cell["execution_status"] != "SUCCESS"
                or coordinate["lopo_replay_complete"] is not True
                or cell["metrics"]["lopo_replay_complete"] is not True
            ):
                raise DojoAITrainerPacketError("complete coordinate is inconsistent")
        elif cell["execution_status"] != "FAILED":
            raise DojoAITrainerPacketError("failed coordinate lacks failed cell")
        if status == "MAIN_REPLAY_FAILED_SENTINEL":
            if (
                lopo
                or coordinate["lopo_replay_complete"] is not False
                or cell["failure_code"] != "MAIN_REPLAY_FAILED"
            ):
                raise DojoAITrainerPacketError(
                    "main-failure coordinate is inconsistent"
                )
            continue
        trade_pairs = list(study["trade_pairs"])
        if len(lopo) != len(trade_pairs):
            raise DojoAITrainerPacketError(
                "run coordinate LOPO denominator is incomplete"
            )
        held_out: set[str] = set()
        lopo_failed = False
        for lopo_index, raw_lopo in enumerate(lopo):
            if not isinstance(raw_lopo, Mapping):
                raise DojoAITrainerPacketError(
                    f"run coordinate LOPO {lopo_index} must be an object"
                )
            pair = raw_lopo.get("held_out_pair")
            lopo_status = raw_lopo.get("status")
            if (
                pair not in trade_pairs
                or pair in held_out
                or lopo_status
                not in {
                    "VALID_COUNTERFACTUAL_REPLAY",
                    "FAILED_NO_ADDITIVE_SUBSTITUTE",
                }
            ):
                raise DojoAITrainerPacketError(
                    "run coordinate LOPO identity is invalid"
                )
            held_out.add(pair)
            if lopo_status == "VALID_COUNTERFACTUAL_REPLAY":
                terminal_net = _number(
                    raw_lopo.get("terminal_net_jpy"), "LOPO terminal net"
                )
                if status == "COMPLETE" and not math.isclose(
                    terminal_net,
                    cell["metrics"]["leave_one_pair_out_net_jpy"][pair],
                    rel_tol=0,
                    abs_tol=1e-8,
                ):
                    raise DojoAITrainerPacketError(
                        "LOPO receipt diverges from sealed cell metric"
                    )
            else:
                lopo_failed = True
                if raw_lopo.get("terminal_net_jpy") is not None:
                    raise DojoAITrainerPacketError(
                        "failed LOPO receipt claims a terminal result"
                    )
        if held_out != set(trade_pairs):
            raise DojoAITrainerPacketError("run coordinate LOPO pair set is incomplete")
        if status == "COMPLETE" and lopo_failed:
            raise DojoAITrainerPacketError("complete coordinate contains failed LOPO")
        if status == "LOPO_INCOMPLETE_NO_ADDITIVE_SUBSTITUTE" and (
            not lopo_failed
            or coordinate["lopo_replay_complete"] is not False
            or cell["failure_code"] != "COUNTERFACTUAL_LOPO_INCOMPLETE"
            or cell["metrics"]["lopo_replay_complete"] is not False
        ):
            raise DojoAITrainerPacketError("LOPO-failure coordinate is inconsistent")
    if seen != expected_grid:
        raise DojoAITrainerPacketError("run coordinate grid is incomplete")
    return _json_copy(normalized_run), sorted(
        normalized_cells,
        key=lambda item: (item["candidate_id"], item["intrabar"], item["cost_arm"]),
    )


def _normalize_cell(
    value: Mapping[str, Any], *, sealed_study: Mapping[str, Any]
) -> dict[str, Any]:
    row = _exact_mapping(value, _CELL_KEYS, "trainer cell")
    _verify_seal(row, "cell_sha256", "trainer cell")
    if row["contract"] != CELL_CONTRACT or row["schema_version"] != 1:
        raise DojoAITrainerPacketError("trainer cell contract/version drifted")
    if row["study_sha256"] != sealed_study["study_sha256"]:
        raise DojoAITrainerPacketError("trainer cell belongs to another study")
    candidates = {
        item["candidate_id"]: item for item in sealed_study["study"]["candidates"]
    }
    candidate_id = _identifier(row["candidate_id"], "cell candidate_id")
    candidate = candidates.get(candidate_id)
    if candidate is None or row["proposal_sha256"] != candidate["proposal_sha256"]:
        raise DojoAITrainerPacketError("trainer cell candidate binding drifted")
    if (
        row["intrabar"] not in REQUIRED_INTRABAR_PATHS
        or row["cost_arm"] not in REQUIRED_COST_ARMS
    ):
        raise DojoAITrainerPacketError("trainer cell is outside the fixed grid")
    status = row["execution_status"]
    if status == "SUCCESS":
        if row["failure_code"] is not None:
            raise DojoAITrainerPacketError("successful trainer cell has a failure code")
    elif status == "FAILED":
        _identifier(row["failure_code"], "cell failure_code")
    else:
        raise DojoAITrainerPacketError("trainer cell execution status is unsupported")
    if not isinstance(row["ledger_evidence"], Mapping):
        raise DojoAITrainerPacketError("cell ledger evidence envelope is malformed")

    metrics = _exact_mapping(row["metrics"], _CELL_METRIC_KEYS, "cell metrics")
    for field in ("terminal_flat", "mtm_complete", "lopo_replay_complete"):
        if not isinstance(metrics[field], bool):
            raise DojoAITrainerPacketError(f"cell metric {field} must be boolean")
    numeric = {
        "terminal_net_jpy": _number(metrics["terminal_net_jpy"], "terminal_net_jpy"),
        "realized_max_drawdown_fraction": _nonnegative_number(
            metrics["realized_max_drawdown_fraction"], "realized drawdown"
        ),
        "peak_margin_usage_fraction": _nonnegative_number(
            metrics["peak_margin_usage_fraction"], "peak margin"
        ),
        "capital_lock_margin_jpy_hours": _nonnegative_number(
            metrics["capital_lock_margin_jpy_hours"], "capital lock"
        ),
    }
    for field in ("margin_closeouts", "fill_count", "margin_reject_count"):
        numeric[field] = _integer(metrics[field], field)
        if numeric[field] < 0:
            raise DojoAITrainerPacketError(f"cell metric {field} cannot be negative")
    mtm = metrics["mtm_max_drawdown_fraction"]
    if metrics["mtm_complete"]:
        mtm = _nonnegative_number(mtm, "MTM drawdown")
    elif mtm is not None:
        raise DojoAITrainerPacketError("incomplete MTM must use null drawdown")
    pairs = list(sealed_study["study"]["trade_pairs"])
    pair_pnl = _pair_values(metrics["pair_pnl_jpy"], pairs, "pair PnL")
    lopo = _pair_values(metrics["leave_one_pair_out_net_jpy"], pairs, "LOPO")
    tolerance = max(0.05, numeric["fill_count"] * 0.01)
    if not math.isclose(
        sum(pair_pnl.values()),
        numeric["terminal_net_jpy"],
        rel_tol=0,
        abs_tol=tolerance,
    ):
        raise DojoAITrainerPacketError("cell pair PnL does not reconcile terminal net")
    normalized_metrics = {
        "terminal_net_jpy": numeric["terminal_net_jpy"],
        "terminal_flat": metrics["terminal_flat"],
        "margin_closeouts": numeric["margin_closeouts"],
        "realized_max_drawdown_fraction": numeric["realized_max_drawdown_fraction"],
        "mtm_complete": metrics["mtm_complete"],
        "mtm_max_drawdown_fraction": mtm,
        "peak_margin_usage_fraction": numeric["peak_margin_usage_fraction"],
        "fill_count": numeric["fill_count"],
        "margin_reject_count": numeric["margin_reject_count"],
        "capital_lock_margin_jpy_hours": numeric["capital_lock_margin_jpy_hours"],
        "pair_pnl_jpy": pair_pnl,
        "leave_one_pair_out_net_jpy": lopo,
        "lopo_replay_complete": metrics["lopo_replay_complete"],
    }
    normalized = {**_json_copy(row), "metrics": normalized_metrics}
    return normalized


def _rebuild_evaluation_from_cells(
    sealed_study: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Mirror the V1 trainer reducer without opening any ledger artifact.

    ``_score_candidate`` is the deterministic reducer used by
    :func:`dojo_bot_trainer.evaluate_training` after that function has verified
    ledgers.  The packet intentionally reuses it so scoring/gate drift cannot
    create a second interpretation of the same normalized cell metrics.  A V2
    evaluation contract must replace this binding rather than silently changing
    either side.
    """

    study = sealed_study["study"]
    by_coordinate = {
        (cell["candidate_id"], cell["intrabar"], cell["cost_arm"]): cell
        for cell in cells
    }
    evaluations = [
        _score_candidate(
            candidate["candidate_id"],
            [
                by_coordinate[(candidate["candidate_id"], path, arm)]
                for path in REQUIRED_INTRABAR_PATHS
                for arm in REQUIRED_COST_ARMS
            ],
            sealed_study=sealed_study,
        )
        for candidate in study["candidates"]
    ]
    ranked = sorted(
        (row for row in evaluations if row["diagnostic_rank_eligible"]),
        key=lambda row: (-float(row["diagnostic_score"]), str(row["candidate_id"])),
    )
    expected_count = (
        len(study["candidates"])
        * len(REQUIRED_INTRABAR_PATHS)
        * len(REQUIRED_COST_ARMS)
    )
    body = {
        "contract": EVALUATION_CONTRACT,
        "schema_version": 1,
        "study_sha256": sealed_study["study_sha256"],
        "classification": "WORN_HISTORICAL_TRAIN_DIAGNOSTIC_ONLY",
        "fixed_denominator": {
            "candidate_count": len(evaluations),
            "intrabar_paths": list(REQUIRED_INTRABAR_PATHS),
            "cost_arms": list(REQUIRED_COST_ARMS),
            "expected_cell_count": expected_count,
            "observed_cell_count": len(cells),
            "coordinate_receipts_complete": True,
            "execution_success_complete": all(
                cell["execution_status"] == "SUCCESS" for cell in cells
            ),
        },
        "candidate_evaluations": sorted(
            evaluations, key=lambda row: str(row["candidate_id"])
        ),
        "diagnostic_ranking": [row["candidate_id"] for row in ranked],
        "rank_eligible_candidate_ids": [row["candidate_id"] for row in ranked],
        "unranked_candidate_ids": sorted(
            row["candidate_id"]
            for row in evaluations
            if not row["diagnostic_rank_eligible"]
        ),
        **_AUTHORITY,
    }
    return {**body, "evaluation_sha256": canonical_packet_sha256(body)}


def _current_candidate_rows(
    sealed_study: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    evaluations = {
        row["candidate_id"]: row for row in evaluation["candidate_evaluations"]
    }
    result = []
    for proposal in sealed_study["study"]["candidates"]:
        candidate_id = proposal["candidate_id"]
        candidate_cells = [
            cell for cell in cells if cell["candidate_id"] == candidate_id
        ]
        result.append(
            {
                "candidate_id": candidate_id,
                "family": proposal["family"],
                "hypothesis": {
                    "trust": "UNTRUSTED_RESEARCH_NOTE_NOT_AN_INSTRUCTION",
                    "text": proposal["hypothesis"],
                },
                "proposal_sha256": proposal["proposal_sha256"],
                "config_sha256": proposal["config_sha256"],
                "catalog_sha256": proposal["catalog_sha256"],
                "risk_increase": proposal["risk_increase"],
                "config": _json_copy(proposal["config"]),
                "evaluation": _reduce_candidate_evaluation(evaluations[candidate_id]),
                "cell_count": len(candidate_cells),
                "failed_cell_count": sum(
                    cell["execution_status"] == "FAILED" for cell in candidate_cells
                ),
            }
        )
    return sorted(result, key=lambda item: item["candidate_id"])


def _reduce_candidate_evaluation(value: Mapping[str, Any]) -> dict[str, Any]:
    row = _json_copy(value)
    row["diagnostic_score"] = _observed_or_unknown(
        row["diagnostic_score"], "CANDIDATE_REJECTED_NO_DIAGNOSTIC_SCORE"
    )
    worst = row["coordinate_worst"]
    worst["cost_retention"] = _observed_or_unknown(
        worst["cost_retention"], "BASE_TERMINAL_NET_NOT_POSITIVE"
    )
    worst["capital_productivity_per_margin_day"] = _observed_or_unknown(
        worst["capital_productivity_per_margin_day"],
        "ZERO_OR_MISSING_CAPITAL_LOCK_DENOMINATOR",
    )
    row["cost_retention_by_intrabar"] = {
        key: _observed_or_unknown(value, "BASE_TERMINAL_NET_NOT_POSITIVE")
        for key, value in row["cost_retention_by_intrabar"].items()
    }
    row["capital_productivity_by_cell"] = {
        key: _observed_or_unknown(value, "ZERO_OR_MISSING_CAPITAL_LOCK_DENOMINATOR")
        for key, value in row["capital_productivity_by_cell"].items()
    }
    return row


def _reduce_cell(cell: Mapping[str, Any]) -> dict[str, Any]:
    metrics = cell["metrics"]
    attempts = metrics["fill_count"] + metrics["margin_reject_count"]
    lock = metrics["capital_lock_margin_jpy_hours"]
    productivity = metrics["terminal_net_jpy"] * 24.0 / lock if lock > 0 else None
    return {
        "candidate_id": cell["candidate_id"],
        "proposal_sha256": cell["proposal_sha256"],
        "intrabar": cell["intrabar"],
        "cost_arm": cell["cost_arm"],
        "execution_status": cell["execution_status"],
        "failure_code": cell["failure_code"],
        "cell_sha256": cell["cell_sha256"],
        "terminal_net_jpy": metrics["terminal_net_jpy"],
        "terminal_flat": metrics["terminal_flat"],
        "margin_closeouts": metrics["margin_closeouts"],
        "realized_max_drawdown_fraction": metrics["realized_max_drawdown_fraction"],
        "mtm_max_drawdown_fraction": _observed_or_unknown(
            metrics["mtm_max_drawdown_fraction"], "CONTINUOUS_MTM_EVIDENCE_INCOMPLETE"
        ),
        "peak_margin_usage_fraction": metrics["peak_margin_usage_fraction"],
        "fill_count": metrics["fill_count"],
        "margin_reject_count": metrics["margin_reject_count"],
        "margin_reject_rate": (
            metrics["margin_reject_count"] / attempts if attempts else 1.0
        ),
        "capital_lock_margin_jpy_hours": lock,
        "capital_productivity_per_margin_day": _observed_or_unknown(
            productivity, "ZERO_CAPITAL_LOCK_DENOMINATOR"
        ),
        "pair_pnl_jpy": _json_copy(metrics["pair_pnl_jpy"]),
        "leave_one_pair_out_net_jpy": _json_copy(metrics["leave_one_pair_out_net_jpy"]),
        "lopo_replay_complete": metrics["lopo_replay_complete"],
    }


def _base_stress_comparisons(
    cells: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {
        (cell["candidate_id"], cell["intrabar"], cell["cost_arm"]): cell
        for cell in cells
    }
    rows = []
    candidate_ids = sorted({cell["candidate_id"] for cell in cells})
    for candidate_id in candidate_ids:
        for intrabar in REQUIRED_INTRABAR_PATHS:
            base = by_key[(candidate_id, intrabar, "BASE")]["metrics"][
                "terminal_net_jpy"
            ]
            stress = by_key[(candidate_id, intrabar, "STRESS")]["metrics"][
                "terminal_net_jpy"
            ]
            retention = stress / base if base > 0 else None
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "intrabar": intrabar,
                    "base_terminal_net_jpy": base,
                    "stress_terminal_net_jpy": stress,
                    "stress_minus_base_jpy": stress - base,
                    "cost_retention": _observed_or_unknown(
                        retention, "BASE_TERMINAL_NET_NOT_POSITIVE"
                    ),
                }
            )
    return rows


def _intrabar_comparisons(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (cell["candidate_id"], cell["intrabar"], cell["cost_arm"]): cell
        for cell in cells
    }
    rows = []
    candidate_ids = sorted({cell["candidate_id"] for cell in cells})
    for candidate_id in candidate_ids:
        for cost_arm in REQUIRED_COST_ARMS:
            ohlc = by_key[(candidate_id, "OHLC", cost_arm)]["metrics"][
                "terminal_net_jpy"
            ]
            olhc = by_key[(candidate_id, "OLHC", cost_arm)]["metrics"][
                "terminal_net_jpy"
            ]
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "cost_arm": cost_arm,
                    "ohlc_terminal_net_jpy": ohlc,
                    "olhc_terminal_net_jpy": olhc,
                    "olhc_minus_ohlc_jpy": olhc - ohlc,
                }
            )
    return rows


def _reduce_previous_attempt(
    sealed_study: Mapping[str, Any],
    result: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    evaluations = {
        row["candidate_id"]: row for row in evaluation["candidate_evaluations"]
    }
    return {
        "attempt_ordinal": result["attempt_ordinal"],
        "study_sha256": result["study_sha256"],
        "evaluation_sha256": result["evaluation_sha256"],
        "candidate_count": len(sealed_study["study"]["candidates"]),
        "candidates": [
            {
                "candidate_id": proposal["candidate_id"],
                "family": proposal["family"],
                "proposal_sha256": proposal["proposal_sha256"],
                "config_sha256": proposal["config_sha256"],
                "config": _json_copy(proposal["config"]),
                "evaluation": _reduce_candidate_evaluation(
                    evaluations[proposal["candidate_id"]]
                ),
            }
            for proposal in sealed_study["study"]["candidates"]
        ],
    }


def _normalize_drive_refs(value: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise DojoAITrainerPacketError("Drive evidence references must be an array")
    if len(value) != len(TRAINER_READBACK_KINDS):
        raise DojoAITrainerPacketError(
            "Drive readback must contain the exact five terminal artifacts"
        )
    result = []
    kinds: set[str] = set()
    drive_ids: set[str] = set()
    parents: set[str] = set()
    readback_shas: set[str] = set()
    for index, item in enumerate(value):
        row = _exact_mapping(item, _DRIVE_REF_KEYS, f"Drive evidence ref {index}")
        kind = row["artifact_kind"]
        if kind not in TRAINER_READBACK_KINDS or kind in kinds:
            raise DojoAITrainerPacketError(
                "Drive readback artifact kind is duplicate or unsupported"
            )
        kinds.add(kind)
        drive_file_id = row["drive_file_id"]
        parent_id = row["drive_parent_id"]
        head_revision = row["head_revision_id"]
        if (
            not isinstance(drive_file_id, str)
            or not _DRIVE_ID.fullmatch(drive_file_id)
            or drive_file_id in drive_ids
        ):
            raise DojoAITrainerPacketError("Drive evidence file id is invalid")
        if not isinstance(parent_id, str) or not _DRIVE_ID.fullmatch(parent_id):
            raise DojoAITrainerPacketError("Drive evidence parent id is invalid")
        if not isinstance(head_revision, str) or not _DRIVE_ID.fullmatch(head_revision):
            raise DojoAITrainerPacketError("Drive head revision id is invalid")
        drive_ids.add(drive_file_id)
        parents.add(parent_id)
        digest = _sha(row["content_sha256"], "Drive content SHA-256")
        size = _integer(row["content_size_bytes"], "Drive content size")
        if size <= 0:
            raise DojoAITrainerPacketError("Drive evidence must be non-empty")
        version = row["version"]
        if (
            not isinstance(version, str)
            or not version.isdecimal()
            or version.startswith("0")
        ):
            raise DojoAITrainerPacketError("Drive version must be a positive decimal")
        if row["remote_verified"] is not True:
            raise DojoAITrainerPacketError(
                "Drive evidence reference is not remotely verified"
            )
        readback_sha = _sha(row["readback_sha256"], "Drive readback SHA-256")
        readback_shas.add(readback_sha)
        result.append(
            {
                "artifact_kind": kind,
                "drive_file_id": drive_file_id,
                "drive_parent_id": parent_id,
                "content_sha256": digest,
                "content_size_bytes": size,
                "version": version,
                "head_revision_id": head_revision,
                "readback_sha256": readback_sha,
                "remote_verified": True,
            }
        )
    if kinds != set(TRAINER_READBACK_KINDS):
        raise DojoAITrainerPacketError(
            "Drive readback omits a required terminal artifact"
        )
    if len(parents) != 1 or len(readback_shas) != 1:
        raise DojoAITrainerPacketError(
            "Drive readback references do not share one parent/bundle seal"
        )
    return sorted(
        result,
        key=lambda row: (
            row["artifact_kind"],
            row["drive_file_id"],
        ),
    )


def _observed_or_unknown(value: Any, reason: str) -> dict[str, Any]:
    if value is None:
        return {"status": "UNKNOWN", "value": None, "reason": reason}
    return {
        "status": "OBSERVED",
        "value": _number(value, "observed metric"),
        "reason": None,
    }


def _load_bound_json(
    root: Path,
    relative: Any,
    *,
    expected_sha256: Any,
    expected_size: Any,
) -> dict[str, Any]:
    digest = _sha(expected_sha256, "bound artifact SHA-256")
    size = _integer(expected_size, "bound artifact size")
    if size <= 0 or size > MAX_BOUND_ARTIFACT_BYTES:
        raise DojoAITrainerPacketError("bound artifact size is outside the limit")
    pure = PurePosixPath(str(relative))
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise DojoAITrainerPacketError("bound artifact path is unsafe")
    current = root.resolve(strict=True)
    try:
        for part in pure.parts[:-1]:
            current = current / part
            info = current.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise DojoAITrainerPacketError("bound artifact path is not a real tree")
        target = current / pure.name
        info = target.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise DojoAITrainerPacketError("bound artifact is not a regular file")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(target, flags)
        with os.fdopen(descriptor, "rb") as handle:
            raw = handle.read(MAX_BOUND_ARTIFACT_BYTES + 1)
    except DojoAITrainerPacketError:
        raise
    except (OSError, RuntimeError) as exc:
        raise DojoAITrainerPacketError("bound artifact cannot be read safely") from exc
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
        raise DojoAITrainerPacketError("bound artifact bytes changed")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DojoAITrainerPacketError("bound artifact is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise DojoAITrainerPacketError("bound artifact must be a JSON object")
    return _json_copy(value)


def _reject_reference_surface(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = key.lower()
            if (
                "ledger" in lowered
                or lowered in {"path", "relpath", "url"}
                or lowered.endswith(("_path", "_relpath", "_url"))
            ):
                raise DojoAITrainerPacketError(
                    "trainer packet contains a path or ledger reference"
                )
            _reject_reference_surface(item)
    elif isinstance(value, list):
        for item in value:
            _reject_reference_surface(item)


def _verify_authority(value: Mapping[str, Any], label: str) -> None:
    for key, expected in _AUTHORITY.items():
        if value.get(key) != expected:
            raise DojoAITrainerPacketError(f"{label} exceeds research authority")


def _verify_seal(value: Mapping[str, Any], field: str, label: str) -> None:
    claimed = _sha(value.get(field), field)
    body = {key: item for key, item in value.items() if key != field}
    if canonical_packet_sha256(body) != claimed:
        raise DojoAITrainerPacketError(f"{label} seal mismatch")


def _pair_values(value: Any, pairs: list[str], label: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(pairs):
        raise DojoAITrainerPacketError(f"{label} pair denominator is incomplete")
    return {pair: _number(value[pair], f"{label}.{pair}") for pair in pairs}


def _exact_mapping(
    value: Any, keys: set[str] | frozenset[str], label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DojoAITrainerPacketError(f"{label} must be a JSON object")
    if set(value) != set(keys):
        raise DojoAITrainerPacketError(f"{label} schema mismatch")
    return _json_copy(value)


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise DojoAITrainerPacketError(f"{label} is invalid")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise DojoAITrainerPacketError(f"{label} is invalid")
    return value


def _integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise DojoAITrainerPacketError(f"{label} must be an integer")
    return value


def _number(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise DojoAITrainerPacketError(f"{label} must be finite")
    return float(value)


def _nonnegative_number(value: Any, label: str) -> float:
    number = _number(value, label)
    if number < 0:
        raise DojoAITrainerPacketError(f"{label} cannot be negative")
    return number


def _validate_json(value: Any, label: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DojoAITrainerPacketError(f"{label} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise DojoAITrainerPacketError(f"{label} contains a non-string key")
            _validate_json(item, f"{label}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _validate_json(item, f"{label}[{index}]")
        return
    raise DojoAITrainerPacketError(f"{label} contains a non-JSON value")


def _json_copy(value: Any) -> Any:
    _validate_json(value, "value")
    return json.loads(canonical_packet_bytes(value).decode("utf-8"))


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DojoAITrainerPacketError(
                "bound artifact contains duplicate JSON keys"
            )
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise DojoAITrainerPacketError(f"non-finite JSON constant is forbidden: {value}")


__all__ = [
    "DojoAITrainerPacketError",
    "PACKET_CONTRACT",
    "SCHEMA_VERSION",
    "build_trainer_packet",
    "canonical_packet_bytes",
    "canonical_packet_sha256",
    "validate_terminal_result_bundle",
    "verify_trainer_packet",
]
